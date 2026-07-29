#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/run/_common.sh"

INPUT_PATH=""
HF_SOURCE=""
HF_REVISION="main"
HF_TOKEN=""
HF_MAX_FILES=""
MAX_ROWS=""
SEED="42"
NORMALIZED_OUTPUT=""
PREDICTION_OUTPUT=""
BUNDLE_CHECKPOINT=""
BUNDLE_DIR=""
BUNDLE_PATTERN="offline_bundle*.pt"
LOCAL_FILES_ONLY="0"
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run/02_offline_pipeline.sh --input RAW_FILE [--normalized-output FILE] [--prediction-output FILE]
  ./run/02_offline_pipeline.sh --hf-source HF_SOURCE [--normalized-output FILE] [--prediction-output FILE]

HF_SOURCE examples:
  MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1
  MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020
  MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020/01/01-10/news_comments.json

If outputs are omitted:
- normalized data is auto-saved under outputs/
- prediction json is auto-saved under outputs/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_PATH="$2"; shift 2 ;;
    --hf-source) HF_SOURCE="$2"; shift 2 ;;
    --hf-revision) HF_REVISION="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --hf-max-files) HF_MAX_FILES="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --normalized-output) NORMALIZED_OUTPUT="$2"; shift 2 ;;
    --prediction-output) PREDICTION_OUTPUT="$2"; shift 2 ;;
    --bundle-checkpoint) BUNDLE_CHECKPOINT="$2"; shift 2 ;;
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --bundle-pattern) BUNDLE_PATTERN="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY="1"; shift 1 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${INPUT_PATH}" && -z "${HF_SOURCE}" ]]; then
  echo "Either --input or --hf-source is required" >&2
  usage
  exit 1
fi

SOURCE_RESOLVED="$(resolve_input_source_tag "${INPUT_PATH}" "${HF_SOURCE}" "${SOURCE_TAG}")"
if [[ -z "${NORMALIZED_OUTPUT}" ]]; then NORMALIZED_OUTPUT="$(make_data_output_path normalized "${SOURCE_RESOLVED}" parquet)"; fi
if [[ -z "${PREDICTION_OUTPUT}" ]]; then PREDICTION_OUTPUT="$(make_data_output_path prediction "${SOURCE_RESOLVED}" json)"; fi
print_resolved_path "Resolved source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved normalized output" "${NORMALIZED_OUTPUT}"
print_resolved_path "Resolved prediction output" "${PREDICTION_OUTPUT}"

NORMALIZE_CMD=("${ROOT_DIR}/run/00_offline_normalize_data.sh" --output "${NORMALIZED_OUTPUT}" --seed "${SEED}" --source-tag "${SOURCE_RESOLVED}")
if [[ -n "${INPUT_PATH}" ]]; then NORMALIZE_CMD+=(--input "${INPUT_PATH}"); fi
if [[ -n "${HF_SOURCE}" ]]; then NORMALIZE_CMD+=(--hf-source "${HF_SOURCE}" --hf-revision "${HF_REVISION}"); fi
if [[ -n "${HF_TOKEN}" ]]; then NORMALIZE_CMD+=(--hf-token "${HF_TOKEN}"); fi
if [[ -n "${HF_MAX_FILES}" ]]; then NORMALIZE_CMD+=(--hf-max-files "${HF_MAX_FILES}"); fi
if [[ -n "${MAX_ROWS}" ]]; then NORMALIZE_CMD+=(--max-rows "${MAX_ROWS}"); fi

INFER_CMD=("${ROOT_DIR}/run/01_offline_infer.sh" --input "${NORMALIZED_OUTPUT}" --output "${PREDICTION_OUTPUT}" --bundle-pattern "${BUNDLE_PATTERN}" --source-tag "${SOURCE_RESOLVED}")
if [[ -n "${BUNDLE_CHECKPOINT}" ]]; then INFER_CMD+=(--bundle-checkpoint "${BUNDLE_CHECKPOINT}"); fi
if [[ -n "${BUNDLE_DIR}" ]]; then INFER_CMD+=(--bundle-dir "${BUNDLE_DIR}"); fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then INFER_CMD+=(--local-files-only); fi

printf '[1/2] normalize
'
"${NORMALIZE_CMD[@]}"
printf '[2/2] offline inference
'
"${INFER_CMD[@]}"
