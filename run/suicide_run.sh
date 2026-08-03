#!/usr/bin/env bash
# ============================================================
# suicide_run.sh — mindcast_kote 실행 스크립트
#
# Usage:
#   ./suicide_run.sh [-g <GPU>] <command> [options]
#
#   ./suicide_run.sh    preprocess            # 01~04 전처리 (CPU or 자동 GPU)
#   ./suicide_run.sh -g 2 preprocess          # GPU 2 지정
#   ./suicide_run.sh -g 2 ablation            # 05 Ablation A~F (일별 예측기)
#   ./suicide_run.sh -g 2 ablation point      # 05 point-head 버전
#   ./suicide_run.sh -g 2 parallel [nb|point] # 05 병렬 실행
#   ./suicide_run.sh    main                  # 11 메인 모델 확정
#   ./suicide_run.sh    monthly               # 13 월별 자살자수 예측
#   ./suicide_run.sh    all                   # 전처리 → ablation → main → monthly
# ============================================================

set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/../pipeline"
PYTHON="${PYTHON:-python3}"
GPU=""

# ── 색상 출력 ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${NC}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
die()  { echo -e "${RED}[ERROR] $*${NC}" >&2; exit 1; }

# ── 사용법 출력 ──────────────────────────────────────────────
usage() {
    echo "Usage: ./suicide_run.sh [-g <GPU>] <command> [options]"
    echo ""
    echo "Options:"
    echo "  -g <N>    GPU 번호 (선택, 예: -g 0, -g 2 / 미지정 시 PyTorch 자동 탐지)"
    echo "  -h        도움말"
    echo ""
    echo "Commands:"
    echo "  preprocess              01~04 전처리 (HuggingFace → parquet → KOTE → 토픽)"
    echo "  ablation [mode]         일별 Ablation A~F  (mode: nb|point, 기본 nb)"
    echo "  parallel [mode]         일별 Ablation 병렬 실행"
    echo "  main                    메인 모델 확정·저장 (Setting A)"
    echo "  monthly                 월별 자살자수 × 7 조합 RidgeCV"
    echo "  main-infer <start> [end]  일별 예측 (main_model.pt, YYYY-MM-DD)"
    echo "  monthly-infer <month>   월별 자살자수 예측 (미래 월, YYYY-MM)"
    echo "  all                     전처리 → ablation → main → monthly"
    echo ""
    echo "Examples:"
    echo "  ./suicide_run.sh -g 2 preprocess"
    echo "  ./suicide_run.sh -g 2 ablation"
    echo "  ./suicide_run.sh -g 2 ablation point"
    echo "  ./suicide_run.sh -g 2 parallel"
    echo "  ./suicide_run.sh -g 2 main"
    echo "  ./suicide_run.sh -g 2 monthly"
    echo "  ./suicide_run.sh    main-infer 2024-01-01"
    echo "  ./suicide_run.sh    main-infer 2024-01-01 2024-01-31"
    echo "  ./suicide_run.sh    monthly-infer 2024-03"
    echo "  ./suicide_run.sh -g 2 all"
}

# ── 인자 파싱 ────────────────────────────────────────────────
while getopts ":g:h" opt; do
    case $opt in
        g) GPU="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) die "-$OPTARG 옵션에 값이 필요합니다 (예: -g 2)" ;;
        \?) die "알 수 없는 옵션: -$OPTARG" ;;
    esac
done
shift $((OPTIND - 1))

# ── HF_TOKEN 자동 로드 (파일에 있으면 환경변수로) ────────────────
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# ── GPU 설정 (지정 시에만) ────────────────────────────────────
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    log "GPU: $GPU (CUDA_VISIBLE_DEVICES=$GPU)"
else
    log "GPU: 미지정 (PyTorch 자동 탐지)"
fi

# ── 헬퍼: GPU 인자 조립 ──────────────────────────────────────
gpu_args() {
    [[ -n "$GPU" ]] && echo "--gpu $GPU" || echo ""
}

# ── 헬퍼: python 실행 ─────────────────────────────────────────
run_py() {
    local script="$1"; shift
    log "실행: $script $*"
    "$PYTHON" "$SRC/$script" "$@"
    ok "$script 완료"
}

# ── pipeline 함수 ─────────────────────────────────────────────

do_preprocess() {
    log "=== [전처리] 01~04 시작 ==="
    # shellcheck disable=SC2046
    run_py suicide_pipeline.py --run preprocess $(gpu_args)
    ok "=== 전처리 완료 ==="
}

do_ablation() {
    local mode="${1:-nb}"
    log "=== [일별 예측기] Ablation suite (mode=$mode) ==="
    run_py suicide_pipeline.py --run ablation --mode "$mode"
    ok "=== Ablation 완료 ==="
}

do_parallel() {
    local mode="${1:-nb}"
    local n_exp

    log "=== [일별 예측기] 병렬 Ablation (mode=$mode) ==="
    n_exp=$("$PYTHON" "$SRC/suicide_pipeline.py" --run parallel --list --mode "$mode")
    log "실험 수: $n_exp"

    for i in $(seq 0 $((n_exp - 1))); do
        log "  worker $i 시작..."
        # shellcheck disable=SC2046
        "$PYTHON" "$SRC/suicide_pipeline.py" \
            --run parallel --idx "$i" --mode "$mode" $(gpu_args) &
    done

    log "모든 worker 종료 대기 중..."
    wait
    ok "모든 worker 완료"

    log "결과 집계..."
    run_py suicide_pipeline.py --run parallel --aggregate --mode "$mode"
    ok "=== 병렬 Ablation 완료 ==="
}

do_main() {
    log "=== [일별 예측기] 메인 모델 확정 ==="
    run_py suicide_pipeline.py --run main
    ok "=== 메인 모델 저장 완료 ==="
}

do_monthly() {
    log "=== [월별 예측기] 자살자수 × 7 조합 ==="
    run_py suicide_pipeline.py --run monthly
    ok "=== 월별 예측 완료 ==="
}

do_monthly_infer() {
    local month="${1:-}"
    [[ -z "$month" ]] && die "월을 지정하세요 (예: ./suicide_run.sh monthly-infer 2024-03)"
    log "=== [월별 inference] $month ==="
    run_py suicide_pipeline.py --run monthly-infer --start "$month"
    ok "=== 월별 inference 완료 ==="
}

do_infer() {
    local start="${1:-}"; local end="${2:-}"
    [[ -z "$start" ]] && die "날짜를 지정하세요 (예: ./suicide_run.sh infer 2024-01-01)"
    local args="--start $start"
    [[ -n "$end" ]] && args="$args --end $end"
    log "=== [일별 예측] $start ${end:+~ $end} ==="
    # shellcheck disable=SC2086
    run_py suicide_pipeline.py --run infer $args
    ok "=== 예측 완료 ==="
}

do_all() {
    log "=== 전체 파이프라인 시작 ==="
    do_preprocess
    do_ablation nb
    do_main
    do_monthly
    ok "=== 전체 파이프라인 완료 ==="
}

# ── 진입점 ──────────────────────────────────────────────────
cd "$SCRIPT_DIR"

CMD="${1:-}"
shift || true

case "$CMD" in
    preprocess)   do_preprocess ;;
    ablation)     do_ablation  "${1:-nb}" ;;
    parallel)     do_parallel  "${1:-nb}" ;;
    main)         do_main ;;
    monthly)      do_monthly ;;
    main-infer)      do_infer "${1:-}" "${2:-}" ;;
    monthly-infer)   do_monthly_infer "${1:-}" ;;
    all)             do_all ;;
    ""|--help|-h) usage ;;
    *) die "알 수 없는 명령: $CMD\n\n$(usage)" ;;
esac
