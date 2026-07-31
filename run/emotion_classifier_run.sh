#!/usr/bin/env bash
# ============================================================
# emotion_classifier_run.sh — 감정 분류기 실행 스크립트
#
# Usage:
#   ./emotion_classifier_run.sh [-g <GPU>] <command> [옵션]
#   (GPU 지정: -g 0  /  미지정 시 PyTorch 자동 탐지)
#
# ── 모드별 전체 실행 ──────────────────────────────────────────
#   ./emotion_classifier_run.sh all [full]              # A+B 학습 전체
#   ./emotion_classifier_run.sh all stagea              # Stage A만 학습
#   ./emotion_classifier_run.sh all stageb              # Stage B만 학습 (기존 A 재사용)
#   ./emotion_classifier_run.sh all offline in.json out.csv  # 추론 전체
#
# ── 개별 단계 (online: 학습) ─────────────────────────────────
#   ./emotion_classifier_run.sh prep-a                  # Stage A 정규화 (중간 산출물 저장)
#   ./emotion_classifier_run.sh train-a                 # Stage A 학습
#   ./emotion_classifier_run.sh prep-b                  # Stage B 정규화 (중간 산출물 저장, EC_STAGEB_INPUT 필수)
#   ./emotion_classifier_run.sh train-b                 # Stage B 학습 (EC_STAGEB_INPUT 필수)
#   ./emotion_classifier_run.sh export                  # 번들 내보내기 (Stage B 없으면 base-only)
#
#   [보조] teacher: 교사 확률 파일 생성. 기본 시작점은 이미 생성된 EC_TEACHER_OUT.
#   ./emotion_classifier_run.sh teacher
#
# ── 개별 단계 (offline: 추론) ────────────────────────────────
#   ./emotion_classifier_run.sh predict in.json out.csv      # 감정 추론
#   ./emotion_classifier_run.sh attach-major in.csv out.csv  # 대분류 컬럼 추가
# ============================================================

set -euo pipefail

# ── .env 자동 로드 ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# ── 설정 ─────────────────────────────────────────────────────
PIPELINE="$SCRIPT_DIR/../pipeline/emotion_classifier_pipeline.py"
PYTHON="${PYTHON:-python3}"
GPU=""

# 기본 경로 (환경변수로 덮어쓸 수 있음)
DATA_DIR="${EC_DATA_DIR:-$SCRIPT_DIR/../data/emotion_classifier}"
MODEL_DIR="${EC_MODEL_DIR:-$SCRIPT_DIR/../models/emotion_classifier}"
TEACHER_OUT="${EC_TEACHER_OUT:-$DATA_DIR/teacher_targets.parquet}"
STAGEA_NORMALIZED="${EC_STAGEA_NORMALIZED:-$DATA_DIR/stagea_normalized.parquet}"
STAGEA_DIR="${EC_STAGEA_DIR:-$MODEL_DIR/stage_a}"
STAGEB_INPUT="${EC_STAGEB_INPUT:-}"
STAGEB_NORMALIZED="${EC_STAGEB_NORMALIZED:-$DATA_DIR/stageb_normalized.parquet}"
STAGEB_DIR="${EC_STAGEB_DIR:-$MODEL_DIR/stage_b}"
BUNDLE_OUT="${EC_BUNDLE_OUT:-$MODEL_DIR/offline_bundle.pt}"

# ── 색상 출력 ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[0;33m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${NC}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
info() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ℹ $*${NC}"; }
die()  { echo -e "${RED}[ERROR] $*${NC}" >&2; exit 1; }

# ── 사용법 ───────────────────────────────────────────────────
usage() {
    echo "Usage: ./emotion_classifier_run.sh [-g <GPU>] <command> [옵션]"
    echo ""
    echo "Options:"
    echo "  -g <N>    GPU 번호 (teacher/train-a/train-b 에서 사용, 예: -g 0)"
    echo "  -h        도움말"
    echo ""
    echo "── 전체 실행 ──────────────────────────────────────────"
    echo "  all [full]              A+B 학습: prep-a → train-a → prep-b → train-b → export"
    echo "  all stagea              Stage A만: prep-a → train-a → export (base-only 번들)"
    echo "  all stageb              Stage B만: prep-b → train-b → export (기존 Stage A 재사용)"
    echo "  all offline <in> <out>  추론 파이프라인: predict → attach-major"
    echo ""
    echo "── 개별 단계 (online: 학습) ───────────────────────────"
    echo "  prep-a       Stage A 정규화 ← \$EC_TEACHER_OUT → \$EC_STAGEA_NORMALIZED"
    echo "  train-a      Stage A 학습 ← \$EC_STAGEA_NORMALIZED → \$EC_STAGEA_DIR"
    echo "  prep-b       Stage B 정규화 (\$EC_STAGEB_INPUT 필수) → \$EC_STAGEB_NORMALIZED"
    echo "  train-b      Stage B 학습 ← \$EC_STAGEB_NORMALIZED → \$EC_STAGEB_DIR"
    echo "  export       번들 내보내기 (Stage B 있으면 A+B, 없으면 base-only)"
    echo ""
    echo "  [보조] teacher  교사 확률 생성 → \$EC_TEACHER_OUT"
    echo "                  기본 시작점은 이미 생성된 EC_TEACHER_OUT 파일."
    echo ""
    echo "── 개별 단계 (offline: 추론) ──────────────────────────"
    echo "  predict  <in> <out>       감정 추론 (json/csv/parquet → csv)"
    echo "  attach-major <in> <out>   소분류 → 대분류 컬럼 추가"
    echo ""
    echo "경로 환경변수 (미지정 시 기본값 사용):"
    echo "  EC_TEACHER_OUT       교사 확률 parquet          (기본: \$EC_DATA_DIR/teacher_targets.parquet)"
    echo "  EC_STAGEA_NORMALIZED Stage A 정규화 중간 산출물  (기본: \$EC_DATA_DIR/stagea_normalized.parquet)"
    echo "  EC_STAGEA_DIR        Stage A 출력 디렉토리       (기본: \$EC_MODEL_DIR/stage_a)"
    echo "  EC_STAGEB_INPUT      Stage B gold 데이터         (sarcasm_label 포함, prep-b/train-b 필수)"
    echo "  EC_STAGEB_NORMALIZED Stage B 정규화 중간 산출물  (기본: \$EC_DATA_DIR/stageb_normalized.parquet)"
    echo "  EC_STAGEB_DIR        Stage B 출력 디렉토리       (기본: \$EC_MODEL_DIR/stage_b)"
    echo "  EC_BUNDLE_OUT        번들 출력 경로              (기본: \$EC_MODEL_DIR/offline_bundle.pt)"
    echo "  EC_DATA_DIR          데이터 디렉토리             (기본: data/emotion_classifier)"
    echo "  EC_MODEL_DIR         모델 디렉토리              (기본: models/emotion_classifier)"
    echo "  EC_LABEL_MAP         label_map.json 경로        (미지정 시 패키지 기본 KOTE 44라벨 사용)"
    echo "  HF_TOKEN             HuggingFace 접근 토큰"
    echo ""
    echo "Examples:"
    echo "  ./emotion_classifier_run.sh all stagea"
    echo "  EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx ./emotion_classifier_run.sh all full"
    echo "  ./emotion_classifier_run.sh prep-a && ./emotion_classifier_run.sh train-a"
    echo "  ./emotion_classifier_run.sh predict data/comments.json data/results.csv"
}

# ── 인자 파싱 ────────────────────────────────────────────────
while getopts ":g:h" opt; do
    case $opt in
        g) GPU="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) die "-$OPTARG 옵션에 값이 필요합니다 (예: -g 0)" ;;
        \?) die "알 수 없는 옵션: -$OPTARG" ;;
    esac
done
shift $((OPTIND - 1))

# ── HF_TOKEN 자동 로드 ───────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# ── GPU 설정 ─────────────────────────────────────────────────
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    log "GPU: $GPU (CUDA_VISIBLE_DEVICES=$GPU)"
else
    log "GPU: 미지정 (PyTorch 자동 탐지)"
fi

# ── 헬퍼 ─────────────────────────────────────────────────────
hf_token_args() {
    [[ -n "${HF_TOKEN:-}" ]] && echo "--hf-token $HF_TOKEN" || echo ""
}

run_py() {
    local desc="$1"; shift
    log "실행: $desc"
    "$PYTHON" "$PIPELINE" "$@"
    ok "$desc 완료"
}

# ════════════════════════════════════════════════════════════
# Online (학습) 단계
# ════════════════════════════════════════════════════════════

do_teacher() {
    # 보조 스크립트: 교사 확률 파일이 없을 때 사용.
    # 기본 학습 시작점은 이미 생성된 EC_TEACHER_OUT 파일.
    log "=== [보조] KOTE 교사 확률 생성 ==="
    mkdir -p "$(dirname "$TEACHER_OUT")"
    # shellcheck disable=SC2046
    run_py "teacher" teacher \
        --output "$TEACHER_OUT" \
        $(hf_token_args)
    ok "=== teacher 완료 → $TEACHER_OUT ==="
}

do_prep_a() {
    log "=== Stage A 정규화 ==="
    [[ ! -f "$TEACHER_OUT" ]] && die "Stage A 입력 파일이 없습니다: $TEACHER_OUT\n먼저 'teacher'를 실행하거나 EC_TEACHER_OUT을 설정하세요."
    mkdir -p "$(dirname "$STAGEA_NORMALIZED")"
    run_py "prep-a" prep-a \
        --input  "$TEACHER_OUT" \
        --output "$STAGEA_NORMALIZED"
    ok "=== prep-a 완료 → $STAGEA_NORMALIZED ==="
}

do_train_a() {
    log "=== Stage A 학습 (지식 증류) ==="
    [[ ! -f "$STAGEA_NORMALIZED" ]] && die "정규화된 Stage A 데이터가 없습니다: $STAGEA_NORMALIZED\n먼저 'prep-a'를 실행하세요."
    mkdir -p "$STAGEA_DIR"
    run_py "train-a" train-a \
        --input      "$STAGEA_NORMALIZED" \
        --output-dir "$STAGEA_DIR"
    ok "=== train-a 완료 → $STAGEA_DIR/student_comment_distill.pt ==="
}

do_prep_b() {
    log "=== Stage B 정규화 ==="
    [[ -z "$STAGEB_INPUT" ]] && die "EC_STAGEB_INPUT 환경변수를 설정하세요\n  export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx"
    mkdir -p "$(dirname "$STAGEB_NORMALIZED")"
    run_py "prep-b" prep-b \
        --input  "$STAGEB_INPUT" \
        --output "$STAGEB_NORMALIZED"
    ok "=== prep-b 완료 → $STAGEB_NORMALIZED ==="
}

do_train_b() {
    log "=== Stage B 학습 (풍자 감정 어댑터) ==="
    [[ -z "$STAGEB_INPUT" && ! -f "$STAGEB_NORMALIZED" ]] && die "EC_STAGEB_INPUT 환경변수를 설정하세요\n  export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx"
    [[ ! -f "$STAGEA_DIR/student_comment_distill.pt" ]] && die "Stage A checkpoint가 없습니다: $STAGEA_DIR/student_comment_distill.pt\n먼저 'train-a'를 실행하세요."
    mkdir -p "$STAGEB_DIR"
    local label_map_args=()
    [[ -n "${EC_LABEL_MAP:-}" ]] && label_map_args=(--label-map "$EC_LABEL_MAP")
    local input_arg="$STAGEB_NORMALIZED"
    [[ ! -f "$STAGEB_NORMALIZED" && -n "$STAGEB_INPUT" ]] && input_arg="$STAGEB_INPUT"
    run_py "train-b" train-b \
        --input              "$input_arg" \
        --stagea-checkpoint  "$STAGEA_DIR/student_comment_distill.pt" \
        "${label_map_args[@]}" \
        --output-dir         "$STAGEB_DIR"
    ok "=== train-b 완료 → $STAGEB_DIR/stageB_adapter_checkpoint.pt ==="
}

do_export() {
    log "=== 오프라인 번들 내보내기 ==="
    [[ ! -f "$STAGEA_DIR/student_comment_distill.pt" ]] && die "Stage A checkpoint가 없습니다: $STAGEA_DIR/student_comment_distill.pt"
    mkdir -p "$(dirname "$BUNDLE_OUT")"
    local stageb_args=()
    local stageb_ckpt="$STAGEB_DIR/stageB_adapter_checkpoint.pt"
    [[ -f "$stageb_ckpt" ]] && stageb_args=(--stageb-checkpoint "$stageb_ckpt")
    run_py "export" export \
        --base-checkpoint "$STAGEA_DIR/student_comment_distill.pt" \
        "${stageb_args[@]}" \
        --output "$BUNDLE_OUT"
    local variant
    variant=$([[ ${#stageb_args[@]} -gt 0 ]] && echo "stagea_stageb" || echo "stagea_only")
    ok "=== export 완료 → $BUNDLE_OUT  (variant: $variant) ==="
}

# ════════════════════════════════════════════════════════════
# Offline (추론) 단계
# ════════════════════════════════════════════════════════════

do_predict() {
    [[ ! -f "$BUNDLE_OUT" ]] && die "번들 파일이 없습니다: $BUNDLE_OUT\n먼저 'export'를 실행하세요."
    # positional: predict <in> <out>
    # named args: predict --hf-source <src> --output <out>  또는  predict --input <in> --output <out>
    if [[ $# -ge 1 && "${1:-}" != --* ]]; then
        local input="${1:-}"
        local output="${2:-}"
        [[ -z "$input" ]] && die "입력 파일 경로를 지정하세요  (예: data/comments.json)"
        local output_args=()
        [[ -n "$output" ]] && output_args=(--output "$output")
        log "=== 감정 추론  $input ==="
        run_py "predict" predict \
            --bundle "$BUNDLE_OUT" \
            --input  "$input" \
            "${output_args[@]}"
    else
        log "=== 감정 추론 ==="
        run_py "predict" predict --bundle "$BUNDLE_OUT" "$@"
    fi
    ok "=== predict 완료 ==="
}

do_attach_major() {
    local input="${1:-}"
    local output="${2:-}"
    [[ -z "$input"  ]] && die "입력 파일 경로를 지정하세요  (예: data/results.csv)"
    [[ -z "$output" ]] && die "출력 파일 경로를 지정하세요  (예: data/results_major.csv)"
    log "=== 대분류 컬럼 추가  $input → $output ==="
    run_py "attach-major" attach-major \
        --input  "$input" \
        --output "$output"
    ok "=== attach-major 완료 → $output ==="
}

# ════════════════════════════════════════════════════════════
# all: 모드 선택
# ════════════════════════════════════════════════════════════

do_all_full() {
    [[ -z "$STAGEB_INPUT" ]] && die "EC_STAGEB_INPUT 환경변수를 설정하세요\n  export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx"
    info "모드: full (A+B 학습 파이프라인)"
    log "=== [1/5] Stage A 정규화 ==="
    do_prep_a
    log "=== [2/5] Stage A 학습 ==="
    do_train_a
    log "=== [3/5] Stage B 정규화 ==="
    do_prep_b
    log "=== [4/5] Stage B 학습 ==="
    do_train_b
    log "=== [5/5] 번들 내보내기 ==="
    do_export
    ok "=== 학습 파이프라인 완료 (full) → $BUNDLE_OUT ==="
}

do_all_stagea() {
    info "모드: stagea (Stage A만 학습)"
    log "=== [1/3] Stage A 정규화 ==="
    do_prep_a
    log "=== [2/3] Stage A 학습 ==="
    do_train_a
    log "=== [3/3] 번들 내보내기 (base-only) ==="
    do_export
    ok "=== 학습 파이프라인 완료 (stagea) → $BUNDLE_OUT ==="
}

do_all_stageb() {
    [[ -z "$STAGEB_INPUT" ]] && die "EC_STAGEB_INPUT 환경변수를 설정하세요\n  export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx"
    info "모드: stageb (기존 Stage A 재사용, Stage B 학습)"
    log "=== [1/3] Stage B 정규화 ==="
    do_prep_b
    log "=== [2/3] Stage B 학습 ==="
    do_train_b
    log "=== [3/3] 번들 내보내기 ==="
    do_export
    ok "=== 학습 파이프라인 완료 (stageb) → $BUNDLE_OUT ==="
}

do_all_offline() {
    local input="${1:-}"
    local output="${2:-}"
    info "모드: offline (추론 파이프라인)"
    log "=== 추론 파이프라인 시작: predict → attach-major ==="
    local major_out="${output%.csv}_major.csv"
    do_predict      "$input" "$output"
    do_attach_major "$output" "$major_out"
    ok "=== 추론 파이프라인 완료 → $major_out ==="
}

# ── 진입점 ───────────────────────────────────────────────────
cd "$SCRIPT_DIR"

CMD="${1:-}"
shift || true

case "$CMD" in
    # ── 전체 실행 ──────────────────────────────────────────
    all)
        MODE="${1:-full}"
        shift || true
        case "$MODE" in
            full)    do_all_full ;;
            stagea)  do_all_stagea ;;
            stageb)  do_all_stageb ;;
            offline) do_all_offline "${1:-}" "${2:-}" ;;
            *) die "all 모드는 full / stagea / stageb / offline 중 하나여야 합니다\n  예: all full / all stagea / all offline in.json out.csv" ;;
        esac
        ;;
    # ── 개별 단계 (online) ─────────────────────────────────
    teacher)      do_teacher ;;
    prep-a)       do_prep_a ;;
    train-a)      do_train_a ;;
    prep-b)       do_prep_b ;;
    train-b)      do_train_b ;;
    export)       do_export ;;
    # ── 개별 단계 (offline) ────────────────────────────────
    predict)      do_predict      "$@" ;;
    attach-major) do_attach_major "${1:-}" "${2:-}" ;;
    # ── 기타 ──────────────────────────────────────────────
    ""|--help|-h) usage ;;
    *) die "알 수 없는 명령: '$CMD'\n\n$(usage)" ;;
esac
