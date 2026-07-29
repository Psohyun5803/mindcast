#!/usr/bin/env bash
# ============================================================
# emotion_classifier_run.sh — 감정 분류기 실행 스크립트
#
# Usage:
#   ./emotion_classifier_run.sh [-g <GPU>] <command> [옵션]
#
#   ./emotion_classifier_run.sh -g 0 teacher                 # Stage A 교사 확률 생성
#   ./emotion_classifier_run.sh -g 0 train-a                 # Stage A 학습 (지식 증류)
#   ./emotion_classifier_run.sh    prep-b                    # Stage B 타겟 준비
#   ./emotion_classifier_run.sh -g 0 train-b                 # Stage B 학습 (풍자 어댑터)
#   ./emotion_classifier_run.sh    export                    # 오프라인 번들 내보내기
#   ./emotion_classifier_run.sh    predict input.json out.csv # 오프라인 추론
#   ./emotion_classifier_run.sh    attach-major in.csv out.csv # 대분류 컬럼 추가
#   ./emotion_classifier_run.sh -g 0 all                     # teacher → train-a → prep-b → train-b → export
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
STAGEB_TARGETS="${EC_STAGEB_TARGETS:-$DATA_DIR/stageb_targets.parquet}"
STAGEB_DIR="${EC_STAGEB_DIR:-$MODEL_DIR/stage_b}"
BUNDLE_OUT="${EC_BUNDLE_OUT:-$MODEL_DIR/offline_bundle.pt}"

# ── 색상 출력 ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${NC}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
die()  { echo -e "${RED}[ERROR] $*${NC}" >&2; exit 1; }

# ── 사용법 ───────────────────────────────────────────────────
usage() {
    echo "Usage: ./emotion_classifier_run.sh [-g <GPU>] <command> [옵션]"
    echo ""
    echo "Options:"
    echo "  -g <N>    GPU 번호 (teacher/train-a/train-b/all 에서 사용, 예: -g 0)"
    echo "  -h        도움말"
    echo ""
    echo "Commands:"
    echo "  teacher               KOTE 교사 확률 생성 → \$EC_TEACHER_OUT"
    echo "  train-a               Stage A 학습 (지식 증류) → \$EC_STAGEA_DIR"
    echo "  prep-b                Stage B 타겟 준비 → \$EC_STAGEB_TARGETS"
    echo "  train-b               Stage B 학습 (풍자 감정 어댑터) → \$EC_STAGEB_DIR"
    echo "  export                오프라인 번들 내보내기 → \$EC_BUNDLE_OUT"
    echo "  predict  <in> <out>   오프라인 추론 (입력: json/csv/parquet, 출력: csv)"
    echo "  attach-major <in> <out>  소분류 → 대분류 컬럼 추가"
    echo "  all                   teacher → train-a → prep-b → train-b → export"
    echo ""
    echo "경로 환경변수 (미지정 시 기본값 사용):"
    echo "  EC_DATA_DIR          데이터 디렉토리       (기본: data/emotion_classifier)"
    echo "  EC_MODEL_DIR         모델 디렉토리         (기본: models/emotion_classifier)"
    echo "  EC_TEACHER_OUT       교사 확률 parquet      (기본: \$EC_DATA_DIR/teacher_targets.parquet)"
    echo "  EC_STAGEA_DIR        Stage A 출력 디렉토리  (기본: \$EC_MODEL_DIR/stage_a)"
    echo "  EC_STAGEB_TARGETS    Stage B 타겟 parquet   (기본: \$EC_DATA_DIR/stageb_targets.parquet)"
    echo "  EC_STAGEB_DIR        Stage B 출력 디렉토리  (기본: \$EC_MODEL_DIR/stage_b)"
    echo "  EC_BUNDLE_OUT        번들 출력 경로         (기본: \$EC_MODEL_DIR/offline_bundle.pt)"
    echo "  HF_TOKEN             HuggingFace 접근 토큰"
    echo ""
    echo "Examples:"
    echo "  ./emotion_classifier_run.sh -g 0 teacher"
    echo "  ./emotion_classifier_run.sh -g 0 train-a"
    echo "  ./emotion_classifier_run.sh    prep-b"
    echo "  ./emotion_classifier_run.sh -g 0 train-b"
    echo "  ./emotion_classifier_run.sh    export"
    echo "  ./emotion_classifier_run.sh    predict data/comments.json data/results.csv"
    echo "  ./emotion_classifier_run.sh    attach-major data/results.csv data/results_major.csv"
    echo "  ./emotion_classifier_run.sh -g 0 all"
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
    [[ -n "${HF_TOKEN:-}" ]] && echo "--token $HF_TOKEN" || echo ""
}

run_py() {
    local desc="$1"; shift
    log "실행: $desc"
    "$PYTHON" "$PIPELINE" "$@"
    ok "$desc 완료"
}

# ── pipeline 함수 ─────────────────────────────────────────────

do_teacher() {
    log "=== [teacher] KOTE 교사 확률 생성 ==="
    mkdir -p "$(dirname "$TEACHER_OUT")"
    # shellcheck disable=SC2046
    run_py "teacher" teacher \
        --output "$TEACHER_OUT" \
        $(hf_token_args)
    ok "=== teacher 완료 → $TEACHER_OUT ==="
}

do_train_a() {
    log "=== [train-a] Stage A 학습 (지식 증류) ==="
    mkdir -p "$STAGEA_DIR"
    run_py "train-a" train-a \
        --input "$TEACHER_OUT" \
        --output-dir "$STAGEA_DIR"
    ok "=== train-a 완료 → $STAGEA_DIR/student_comment_distill.pt ==="
}

do_prep_b() {
    local label_map="${EC_LABEL_MAP:-}"
    [[ -z "$label_map" ]] && die "EC_LABEL_MAP 환경변수를 설정하세요 (label_map.json 경로)"
    log "=== [prep-b] Stage B 타겟 준비 ==="
    mkdir -p "$(dirname "$STAGEB_TARGETS")"
    # shellcheck disable=SC2046
    run_py "prep-b" prep-b \
        --input "$TEACHER_OUT" \
        --label-map "$label_map" \
        --output "$STAGEB_TARGETS" \
        $(hf_token_args)
    ok "=== prep-b 완료 → $STAGEB_TARGETS ==="
}

do_train_b() {
    local label_map="${EC_LABEL_MAP:-}"
    [[ -z "$label_map" ]] && die "EC_LABEL_MAP 환경변수를 설정하세요 (label_map.json 경로)"
    log "=== [train-b] Stage B 학습 (풍자 감정 어댑터) ==="
    mkdir -p "$STAGEB_DIR"
    run_py "train-b" train-b \
        --input "$STAGEB_TARGETS" \
        --stagea-checkpoint "$STAGEA_DIR/student_comment_distill.pt" \
        --label-map "$label_map" \
        --output-dir "$STAGEB_DIR"
    ok "=== train-b 완료 → $STAGEB_DIR/stageB_adapter_checkpoint.pt ==="
}

do_export() {
    log "=== [export] 오프라인 번들 내보내기 ==="
    mkdir -p "$(dirname "$BUNDLE_OUT")"
    run_py "export" export \
        --base-checkpoint "$STAGEA_DIR/student_comment_distill.pt" \
        --stageb-checkpoint "$STAGEB_DIR/stageB_adapter_checkpoint.pt" \
        --output "$BUNDLE_OUT"
    ok "=== export 완료 → $BUNDLE_OUT ==="
}

do_predict() {
    local input="${1:-}"
    local output="${2:-}"
    [[ -z "$input"  ]] && die "입력 파일 경로를 지정하세요  (예: data/comments.json)"
    [[ -z "$output" ]] && die "출력 파일 경로를 지정하세요  (예: data/results.csv)"
    log "=== [predict] 오프라인 추론  $input → $output ==="
    run_py "predict" predict \
        --bundle "$BUNDLE_OUT" \
        --input "$input" \
        --output "$output"
    ok "=== predict 완료 → $output ==="
}

do_attach_major() {
    local input="${1:-}"
    local output="${2:-}"
    [[ -z "$input"  ]] && die "입력 파일 경로를 지정하세요  (예: data/results.csv)"
    [[ -z "$output" ]] && die "출력 파일 경로를 지정하세요  (예: data/results_major.csv)"
    log "=== [attach-major] 대분류 컬럼 추가  $input → $output ==="
    run_py "attach-major" attach-major \
        --input  "$input" \
        --output "$output"
    ok "=== attach-major 완료 → $output ==="
}

do_all() {
    log "=== 전체 학습 파이프라인 시작 (teacher → train-a → prep-b → train-b → export) ==="
    do_teacher
    do_train_a
    do_prep_b
    do_train_b
    do_export
    ok "=== 전체 학습 파이프라인 완료 → $BUNDLE_OUT ==="
}

# ── 진입점 ───────────────────────────────────────────────────
cd "$SCRIPT_DIR"

CMD="${1:-}"
shift || true

case "$CMD" in
    teacher)      do_teacher ;;
    train-a)      do_train_a ;;
    prep-b)       do_prep_b ;;
    train-b)      do_train_b ;;
    export)       do_export ;;
    predict)      do_predict      "${1:-}" "${2:-}" ;;
    attach-major) do_attach_major "${1:-}" "${2:-}" ;;
    all)          do_all ;;
    ""|--help|-h) usage ;;
    *) die "알 수 없는 명령: '$CMD'\n\n$(usage)" ;;
esac
