#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${ROOT_DIR}/run/_common.sh"

INPUT_PATH=""
OUTPUT_PATH=""
BUNDLE_CHECKPOINT=""
BUNDLE_DIR=""
BUNDLE_PATTERN="offline_bundle*.pt"
COMMENT_TEXT=""
TITLE_TEXT=""
DATE_TEXT=""
LOCAL_FILES_ONLY="0"
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run/01_offline_infer.sh --input INPUT_FILE [--output OUTPUT_FILE]
  ./run/01_offline_infer.sh --comment "댓글" [--output OUTPUT_FILE]

Bundle examples:
  --bundle-checkpoint /home/yein38/mindcastlib_trainer/emotion_classifier/outputs/pt/offline_bundle_stagea_stageb_from_stagea_comment_distill_xxx_with_stageb_adapter_xxx_20260729.pt
  --bundle-dir /home/yein38/mindcastlib_trainer/emotion_classifier/outputs/pt

If --output is omitted, the file is auto-saved under outputs/ with
<prediction>_<source>_<YYYYMMDD>.json naming.
Bundle checkpoints are searched in outputs/pt first, then outputs/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_PATH="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --bundle-checkpoint) BUNDLE_CHECKPOINT="$2"; shift 2 ;;
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --bundle-pattern) BUNDLE_PATTERN="$2"; shift 2 ;;
    --comment) COMMENT_TEXT="$2"; shift 2 ;;
    --title) TITLE_TEXT="$2"; shift 2 ;;
    --date) DATE_TEXT="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY="1"; shift 1 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${INPUT_PATH}" && -z "${COMMENT_TEXT}" ]]; then
  echo "Either --input or --comment is required" >&2
  usage
  exit 1
fi

SOURCE_RESOLVED="$(resolve_input_source_tag "${INPUT_PATH}" "" "${SOURCE_TAG}")"
if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="$(make_data_output_path prediction "${SOURCE_RESOLVED}" json)"
fi
print_resolved_path "Resolved source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved output" "${OUTPUT_PATH}"

if [[ -z "${BUNDLE_CHECKPOINT}" && -z "${BUNDLE_DIR}" ]]; then
  if ! has_bundle_in_default_dirs "${BUNDLE_PATTERN}"; then
    echo "No bundle checkpoint found in default search dirs:" >&2
    while read -r dir; do
      echo "  ${dir}" >&2
    done < <(bundle_search_dirs)
    echo "Provide one of the following:" >&2
    echo "  --bundle-checkpoint /absolute/path/to/offline_bundle.pt" >&2
    echo "  --bundle-dir /absolute/path/to/bundle_directory" >&2
    exit 1
  fi
fi

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/offline_infer.py"
  --output "${OUTPUT_PATH}"
  --bundle-pattern "${BUNDLE_PATTERN}"
)
if [[ -n "${INPUT_PATH}" ]]; then
  CMD+=(--input "${INPUT_PATH}")
else
  CMD+=(--comment "${COMMENT_TEXT}")
  if [[ -n "${TITLE_TEXT}" ]]; then CMD+=(--title "${TITLE_TEXT}"); fi
  if [[ -n "${DATE_TEXT}" ]]; then CMD+=(--date "${DATE_TEXT}"); fi
fi
if [[ -n "${BUNDLE_CHECKPOINT}" ]]; then CMD+=(--bundle-checkpoint "${BUNDLE_CHECKPOINT}"); fi
if [[ -n "${BUNDLE_DIR}" ]]; then CMD+=(--bundle-dir "${BUNDLE_DIR}"); fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then CMD+=(--local-files-only); fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '
'
"${CMD[@]}"
