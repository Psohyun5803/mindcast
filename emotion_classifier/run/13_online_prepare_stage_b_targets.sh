#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${ROOT_DIR}/run/_common.sh"

INPUT_PATH=""
HF_SOURCE=""
HF_REVISION="main"
HF_TOKEN=""
HF_MAX_FILES=""
LABEL_MAP=""
OUTPUT_PATH=""
DROP_POSITIVE_WITHOUT_TARGET="0"
MAX_ROWS=""
SEED="42"
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run/13_online_prepare_stage_b_targets.sh --input INPUT_FILE --label-map LABEL_MAP_JSON [--output OUTPUT_FILE]

This script usually receives the normalized output of run/10_online_normalize_data.sh.
If --output is omitted, the prepared dataset is auto-saved under outputs/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_PATH="$2"; shift 2 ;;
    --hf-source) HF_SOURCE="$2"; shift 2 ;;
    --hf-revision) HF_REVISION="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --hf-max-files) HF_MAX_FILES="$2"; shift 2 ;;
    --label-map) LABEL_MAP="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --drop-positive-without-target) DROP_POSITIVE_WITHOUT_TARGET="1"; shift 1 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${LABEL_MAP}" ]]; then
  echo "--label-map is required" >&2
  usage
  exit 1
fi
if [[ -z "${INPUT_PATH}" && -z "${HF_SOURCE}" ]]; then
  echo "Either --input or --hf-source is required" >&2
  usage
  exit 1
fi

SOURCE_RESOLVED="$(resolve_input_source_tag "${INPUT_PATH}" "${HF_SOURCE}" "${SOURCE_TAG}")"
if [[ -z "${OUTPUT_PATH}" ]]; then OUTPUT_PATH="$(make_data_output_path stageB_prepared "${SOURCE_RESOLVED}" parquet)"; fi
print_resolved_path "Resolved source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved output" "${OUTPUT_PATH}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/prepare_stage_b_targets.py"
  --label-map "${LABEL_MAP}"
  --output "${OUTPUT_PATH}"
  --seed "${SEED}"
)
if [[ -n "${INPUT_PATH}" ]]; then CMD+=(--input "${INPUT_PATH}"); fi
if [[ -n "${HF_SOURCE}" ]]; then CMD+=(--hf-source "${HF_SOURCE}" --hf-revision "${HF_REVISION}"); fi
if [[ -n "${HF_TOKEN}" ]]; then CMD+=(--hf-token "${HF_TOKEN}"); fi
if [[ -n "${HF_MAX_FILES}" ]]; then CMD+=(--hf-max-files "${HF_MAX_FILES}"); fi
if [[ -n "${MAX_ROWS}" ]]; then CMD+=(--max-rows "${MAX_ROWS}"); fi
if [[ "${DROP_POSITIVE_WITHOUT_TARGET}" == "1" ]]; then CMD+=(--drop-positive-without-target); fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '
'
"${CMD[@]}"
