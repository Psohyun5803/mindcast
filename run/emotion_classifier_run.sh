#!/usr/bin/env bash
# ============================================================
# emotion_classifier_run.sh — 감정 분류기 실행 스크립트
#
# Usage:
#   ./emotion_classifier_run.sh [-g <GPU>] <command> [옵션]
#   (GPU 지정: -g 0  /  미지정 시 PyTorch 자동 탐지)
#
# ── 모드별 전체 실행 ──────────────────────────────────────────
#   ./emotion_classifier_run.sh all online              # 학습 전체 (teacher→train-a→train-b→export)
#   ./emotion_classifier_run.sh all offline in.json out.csv  # 추론 전체 (predict→attach-major)
#   ./emotion_classifier_run.sh all                     # online과 동일 (기본값)
#
# ── 개별 단계 (online) ────────────────────────────────────────
#   ./emotion_classifier_run.sh teacher                 # KOTE 교사 확률 생성
#   ./emotion_classifier_run.sh train-a                 # Stage A 학습
#   ./emotion_classifier_run.sh train-b                 # Stage B 학습 (EC_STAGEB_INPUT 필수)
#   ./emotion_classifier_run.sh export                  # 번들 내보내기
#
#   [보조] prep-b: 기본 흐름에서는 사용하지 않는 보조 스크립트.
#   ./emotion_classifier_run.sh prep-b
#
# ── 개별 단계 (offline) ───────────────────────────────────────
#   ./emotion_classifier_run.sh predict in.json out.csv      # 감정 추론
#   ./emotion_classifier_run.sh attach-major in.csv out.csv  # 대분류 컬럼 추가
# ============================================================

set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/../pipeline/emotion_classifier_pipeline.py"
PYTHON="${PYTHON:-python3}"
GPU=""

# 기본 경로 (환경변수로 덮어쓸 수 있음)
DATA_DIR="${EC_DATA_DIR:-$SCRIPT_DIR/../data/emotion_classifier}"
MODEL_DIR="${EC_MODEL_DIR:-$SCRIPT_DIR/../models/emotion_classifier}"
TEACHER_OUT="${EC_TEACHER_OUT:-$DATA_DIR/teacher_targets.parquet}"
STAGEA_DIR="${EC_STAGEA_DIR:-$MODEL_DIR/stage_a}"
STAGEB_INPUT="${EC_STAGEB_INPUT:-}"           # Stage B gold 데이터 (sarcasm_label 포함, 필수)
STAGEB_TARGETS="${EC_STAGEB_TARGETS:-$DATA_DIR/stageb_targets.parquet}"
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
    echo "  all [online]              학습 파이프라인: teacher → train-a → prep-b → train-b → export"
    echo "  all offline <in> <out>    추론 파이프라인: predict → attach-major"
    echo ""
    echo "── 개별 단계 (online: 학습) ───────────────────────────"
    echo "  teacher      KOTE 교사 확률 생성 → \$EC_TEACHER_OUT"
    echo "  train-a      Stage A 학습 (지식 증류) → \$EC_STAGEA_DIR"
    echo "  train-b      Stage B 학습 (\$EC_STAGEB_INPUT 또는 --hf-source) → \$EC_STAGEB_DIR"
    echo "  export       오프라인 번들 내보내기 → \$EC_BUNDLE_OUT"
    echo "  prep-b       [보조] 기본 흐름에서는 사용하지 않는 보조 스크립트"
    echo ""
    echo "── 개별 단계 (offline: 추론) ──────────────────────────"
    echo "  predict  <in> <out>       감정 추론 (json/csv/parquet → csv)"
    echo "  attach-major <in> <out>   소분류 → 대분류 컬럼 추가"
    echo ""
    echo "경로 환경변수 (미지정 시 기본값 사용):"
    echo "  EC_DATA_DIR       데이터 디렉토리       (기본: data/emotion_classifier)"
    echo "  EC_MODEL_DIR      모델 디렉토리         (기본: models/emotion_classifier)"
    echo "  EC_STAGEB_INPUT   Stage B 로컬 데이터    (sarcasm_label 포함 파일, EC_HF_SOURCE와 택1)"
    echo "  EC_HF_SOURCE      Stage B HF 소스       (EC_STAGEB_INPUT와 택1, train-b/prep-b)"
    echo "  EC_LABEL_MAP      label_map.json 경로   (미지정 시 패키지 기본 KOTE 44라벨 사용)"
    echo "  EC_TEACHER_OUT    교사 확률 parquet      (기본: \$EC_DATA_DIR/teacher_targets.parquet)"
    echo "  EC_STAGEA_DIR     Stage A 출력 디렉토리  (기본: \$EC_MODEL_DIR/stage_a)"
    echo "  EC_STAGEB_TARGETS Stage B 타겟 parquet   (기본: \$EC_DATA_DIR/stageb_targets.parquet)"
    echo "  EC_STAGEB_DIR     Stage B 출력 디렉토리  (기본: \$EC_MODEL_DIR/stage_b)"
    echo "  EC_BUNDLE_OUT     번들 출력 경로         (기본: \$EC_MODEL_DIR/offline_bundle.pt)"
    echo "  HF_TOKEN          HuggingFace 접근 토큰"
    echo ""
    echo "Examples:"
    echo "  ./emotion_classifier_run.sh all online"
    echo "  ./emotion_classifier_run.sh all offline data/comments.json data/results.csv"
    echo "  ./emotion_classifier_run.sh teacher"
    echo "  ./emotion_classifier_run.sh predict data/comments.json data/results.csv"
    echo "  ./emotion_classifier_run.sh attach-major data/results.csv data/results_major.csv"
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
    log "=== [online 1/4] KOTE 교사 확률 생성 ==="
    mkdir -p "$(dirname "$TEACHER_OUT")"
    # shellcheck disable=SC2046
    run_py "teacher" teacher \
        --output "$TEACHER_OUT" \
        $(hf_token_args)
    ok "=== teacher 완료 → $TEACHER_OUT ==="
}

do_train_a() {
    log "=== [online 2/4] Stage A 학습 (지식 증류) ==="
    mkdir -p "$STAGEA_DIR"
    run_py "train-a" train-a \
        --input "$TEACHER_OUT" \
        --output-dir "$STAGEA_DIR"
    ok "=== train-a 완료 → $STAGEA_DIR/student_comment_distill.pt ==="
}

do_prep_b() {
    if [[ -z "$STAGEB_INPUT" && -z "${EC_HF_SOURCE:-}" ]]; then
        die "Stage B 입력 데이터가 필요합니다.\n" \
            "  로컬 파일:  export EC_STAGEB_INPUT=/path/to/data.xlsx\n" \
            "  HF 소스:    export EC_HF_SOURCE=<repo/path>"
    fi
    log "=== [prep-b] Stage B 타겟 준비 (보조, 기본 흐름 외) ==="
    mkdir -p "$(dirname "$STAGEB_TARGETS")"
    local input_args=()
    [[ -n "$STAGEB_INPUT" ]]     && input_args+=(--input "$STAGEB_INPUT")
    [[ -n "${EC_HF_SOURCE:-}" ]] && input_args+=(--hf-source "$EC_HF_SOURCE")
    local label_map_args=()
    [[ -n "${EC_LABEL_MAP:-}" ]] && label_map_args=(--label-map "$EC_LABEL_MAP")
    run_py "prep-b" prep-b \
        "${input_args[@]}" \
        "${label_map_args[@]}" \
        --output "$STAGEB_TARGETS"
    ok "=== prep-b 완료 → $STAGEB_TARGETS ==="
}

do_train_b() {
    if [[ -z "$STAGEB_INPUT" && -z "${EC_HF_SOURCE:-}" ]]; then
        die "Stage B 입력 데이터가 필요합니다.\n" \
            "  로컬 파일:  export EC_STAGEB_INPUT=/path/to/stageb_gold.xlsx\n" \
            "  HF 소스:    export EC_HF_SOURCE=<repo/path>"
    fi
    log "=== [online 3/4] Stage B 학습 (풍자 감정 어댑터) ==="
    mkdir -p "$STAGEB_DIR"
    local input_args=()
    [[ -n "$STAGEB_INPUT" ]]        && input_args+=(--input "$STAGEB_INPUT")
    [[ -n "${EC_HF_SOURCE:-}" ]]    && input_args+=(--hf-source "$EC_HF_SOURCE")
    local label_map_args=()
    [[ -n "${EC_LABEL_MAP:-}" ]] && label_map_args=(--label-map "$EC_LABEL_MAP")
    run_py "train-b" train-b \
        "${input_args[@]}" \
        --stagea-checkpoint "$STAGEA_DIR/student_comment_distill.pt" \
        "${label_map_args[@]}" \
        --output-dir "$STAGEB_DIR"
    ok "=== train-b 완료 → $STAGEB_DIR/stageB_adapter_checkpoint.pt ==="
}

do_export() {
    log "=== [online 4/4] 오프라인 번들 내보내기 ==="
    mkdir -p "$(dirname "$BUNDLE_OUT")"
    run_py "export" export \
        --base-checkpoint "$STAGEA_DIR/student_comment_distill.pt" \
        --stageb-checkpoint "$STAGEB_DIR/stageB_adapter_checkpoint.pt" \
        --output "$BUNDLE_OUT"
    ok "=== export 완료 → $BUNDLE_OUT ==="
}

# ════════════════════════════════════════════════════════════
# Offline (추론) 단계
# ════════════════════════════════════════════════════════════

do_predict() {
    local input="${1:-}"
    local output="${2:-}"
    [[ -z "$input"  ]] && die "입력 파일 경로를 지정하세요  (예: data/comments.json)"
    [[ -z "$output" ]] && die "출력 파일 경로를 지정하세요  (예: data/results.csv)"
    [[ ! -f "$BUNDLE_OUT" ]] && die "번들 파일이 없습니다: $BUNDLE_OUT\n먼저 'all online' 또는 'export'를 실행하세요."
    log "=== [offline 1/2] 감정 추론  $input → $output ==="
    run_py "predict" predict \
        --bundle "$BUNDLE_OUT" \
        --input  "$input" \
        --output "$output"
    ok "=== predict 완료 → $output ==="
}

do_attach_major() {
    local input="${1:-}"
    local output="${2:-}"
    [[ -z "$input"  ]] && die "입력 파일 경로를 지정하세요  (예: data/results.csv)"
    [[ -z "$output" ]] && die "출력 파일 경로를 지정하세요  (예: data/results_major.csv)"
    log "=== [offline 2/2] 대분류 컬럼 추가  $input → $output ==="
    run_py "attach-major" attach-major \
        --input  "$input" \
        --output "$output"
    ok "=== attach-major 완료 → $output ==="
}

# ════════════════════════════════════════════════════════════
# all: 모드 선택
# ════════════════════════════════════════════════════════════

do_all_online() {
    info "모드: online (학습 파이프라인)"
    log "=== 학습 파이프라인 시작: teacher → train-a → train-b → export ==="
    do_teacher
    do_train_a
    do_train_b
    do_export
    ok "=== 학습 파이프라인 완료 → $BUNDLE_OUT ==="
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
        MODE="${1:-online}"
        shift || true
        case "$MODE" in
            online)  do_all_online ;;
            offline) do_all_offline "${1:-}" "${2:-}" ;;
            *) die "all 모드는 online 또는 offline이어야 합니다 (예: all online / all offline in.json out.csv)" ;;
        esac
        ;;
    # ── 개별 단계 (online) ─────────────────────────────────
    teacher)      do_teacher ;;
    train-a)      do_train_a ;;
    prep-b)       do_prep_b ;;
    train-b)      do_train_b ;;
    export)       do_export ;;
    # ── 개별 단계 (offline) ────────────────────────────────
    predict)      do_predict      "${1:-}" "${2:-}" ;;
    attach-major) do_attach_major "${1:-}" "${2:-}" ;;
    # ── 기타 ──────────────────────────────────────────────
    ""|--help|-h) usage ;;
    *) die "알 수 없는 명령: '$CMD'\n\n$(usage)" ;;
esac
