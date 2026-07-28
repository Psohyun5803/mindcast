#!/usr/bin/env bash
# ============================================================
# event_online_run.sh — mindcast-online 실행 스크립트
#
# Usage:
#   ./event_online_run.sh [-g <GPU>] <command> [옵션]
#
#   ./event_online_run.sh -g 0 prep 2025-09 hf        # 정제·임베딩 (GPU 필요)
#   ./event_online_run.sh -g 0 prep 2025-09 local      # 로컬 덤프에서 읽기
#   ./event_online_run.sh    track 2025-09              # 온라인 트래킹
#   ./event_online_run.sh    viz   2025-09              # 시각화 (png)
#   ./event_online_run.sh    export                     # 전체 월 HTML 리플레이
#   ./event_online_run.sh    export 2025-09,2025-10     # 특정 월만 HTML
#   ./event_online_run.sh    upload                     # PII 가드 + HF 업로드
#   ./event_online_run.sh    upload --dry-run           # 업로드 없이 가드만
#   ./event_online_run.sh    extract                    # 로컬 덤프 → parquet 캐시
#   ./event_online_run.sh    extract --with-comments    # video_comment 포함
#   ./event_online_run.sh -g 0 all 2025-09 hf          # prep→track→viz→export 한번에
# ============================================================

set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$SCRIPT_DIR/../pipeline/event_online_pipeline.py"
PYTHON="${PYTHON:-python3}"
GPU=""

# ── 색상 출력 ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${NC}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
die()  { echo -e "${RED}[ERROR] $*${NC}" >&2; exit 1; }

# ── 사용법 ───────────────────────────────────────────────────
usage() {
    echo "Usage: ./event_online_run.sh [-g <GPU>] <command> [옵션]"
    echo ""
    echo "Options:"
    echo "  -g <N>    GPU 번호 (prep/all 에서만 사용, 예: -g 0)"
    echo "  -h        도움말"
    echo ""
    echo "Commands:"
    echo "  prep    <month> [source]    정제·kiwi·ko-sroberta 임베딩 (GPU 권장)"
    echo "                                source: hf|local  (기본 hf)"
    echo "  track   <month>             온라인 autoregressive 트래킹"
    echo "  viz     <month>             스트리밍 타임라인 그림 (png)"
    echo "  export  [month1,month2...]  일별 리플레이 HTML (기본: config 전체 월)"
    echo "  upload  [--dry-run]         PII 가드 + HF private 데이터셋 업로드"
    echo "  extract [--with-comments]   로컬 덤프 → data/*.parquet 캐시"
    echo "  all     <month> [source]    prep → track → viz → export 한번에"
    echo ""
    echo "Examples:"
    echo "  ./event_online_run.sh -g 0 prep 2025-09 hf"
    echo "  ./event_online_run.sh track 2025-09"
    echo "  ./event_online_run.sh viz 2025-09"
    echo "  ./event_online_run.sh export 2025-09,2025-10,2025-11"
    echo "  ./event_online_run.sh upload --dry-run"
    echo "  ./event_online_run.sh extract --with-comments"
    echo "  ./event_online_run.sh -g 0 all 2025-09 hf"
    echo ""
    echo "환경변수:"
    echo "  MINDCAST_HF_REPO   HF 데이터셋 repo (기본: merrybabyxmas/mindcast-news-events)"
    echo "  MINDCAST_DUMP      로컬 sql.gz 경로 (source=local 필수)"
    echo "  MINDCAST_FONT      한글 폰트 .ttf/.ttc 경로 (viz 자동탐색)"
    echo "  HF_TOKEN           private 데이터셋 접근 토큰"
}

# ── HF_TOKEN 자동 로드 (파일에 있으면 환경변수로) ────────────────
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

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

# ── 헬퍼: GPU 인자 조립 ──────────────────────────────────────
gpu_args() {
    [[ -n "$GPU" ]] && echo "--gpu $GPU" || echo ""
}

# ── 헬퍼: pipeline 실행 ──────────────────────────────────────
run_py() {
    local desc="$1"; shift
    log "실행: $desc"
    "$PYTHON" "$PIPELINE" "$@"
    ok "$desc 완료"
}

# ── pipeline 함수 ─────────────────────────────────────────────

do_prep() {
    local month="${1:-2025-09}"
    local source="${2:-hf}"
    log "=== [prep] 정제·임베딩  month=$month  source=$source  gpu=${GPU:-auto} ==="
    # shellcheck disable=SC2046
    run_py "prep ($month, $source)" --run prep --month "$month" --source "$source" $(gpu_args)
    ok "=== prep 완료 → data/posts_${month}.parquet + data/emb_${month}.npy ==="
}

do_track() {
    local month="${1:-2025-09}"
    log "=== [track] 온라인 트래킹  month=$month ==="
    run_py "track ($month)" --run track --month "$month"
    ok "=== track 완료 → outputs/tracks_online_${month}.pkl + events_online_${month}.csv ==="
}

do_viz() {
    local month="${1:-2025-09}"
    log "=== [viz] 스트리밍 타임라인  month=$month ==="
    run_py "viz ($month)" --run viz --month "$month"
    ok "=== viz 완료 → figures/06_online_dynamics_${month}.png ==="
}

do_export() {
    local months_arg="${1:-}"
    if [[ -n "$months_arg" ]]; then
        log "=== [export] HTML 리플레이  months=$months_arg ==="
        run_py "export ($months_arg)" --run export --months "$months_arg"
    else
        log "=== [export] HTML 리플레이  (config 전체 월) ==="
        run_py "export (all months)" --run export
    fi
    ok "=== export 완료 → online_replay.html ==="
}

do_upload() {
    local dry="${1:-}"
    if [[ "$dry" == "--dry-run" ]]; then
        log "=== [upload] PII 가드 (dry-run, 업로드 없음) ==="
        run_py "upload (dry-run)" --run upload --dry-run
    else
        log "=== [upload] PII 가드 + HF 업로드 ==="
        run_py "upload" --run upload
    fi
    ok "=== upload 완료 ==="
}

do_extract() {
    local with_cmt="${1:-}"
    if [[ "$with_cmt" == "--with-comments" ]]; then
        log "=== [extract] 로컬 덤프 → parquet  (video_comment 포함) ==="
        run_py "extract (with-comments)" --run extract --with-comments
    else
        log "=== [extract] 로컬 덤프 → parquet  (video_video + video_channel) ==="
        run_py "extract" --run extract
    fi
    ok "=== extract 완료 → data/*.parquet ==="
}

do_all() {
    local month="${1:-2025-09}"
    local source="${2:-hf}"
    log "=== 전체 파이프라인  month=$month  source=$source  gpu=${GPU:-auto} ==="
    do_prep  "$month" "$source"
    do_track "$month"
    do_viz   "$month"
    do_export "$month"
    ok "=== 전체 파이프라인 완료 ($month) ==="
}

# ── 진입점 ───────────────────────────────────────────────────
cd "$SCRIPT_DIR"

CMD="${1:-}"
shift || true

case "$CMD" in
    prep)    do_prep    "${1:-2025-09}" "${2:-hf}" ;;
    track)   do_track   "${1:-2025-09}" ;;
    viz)     do_viz     "${1:-2025-09}" ;;
    export)  do_export  "${1:-}" ;;
    upload)  do_upload  "${1:-}" ;;
    extract) do_extract "${1:-}" ;;
    all)     do_all     "${1:-2025-09}" "${2:-hf}" ;;
    ""|--help|-h) usage ;;
    *) die "알 수 없는 명령: '$CMD'\n\n$(usage)" ;;
esac
