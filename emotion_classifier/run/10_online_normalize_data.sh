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
MAX_ROWS=""
SEED="42"
OUTPUT_PATH=""
SOURCE_TAG=""
KIND="stageB"

resolve_kind_prefix() {
  case "$1" in
    stageA|stagea|a) printf 'stageA_normalized' ;;
    stageB|stageb|b) printf 'stageB_normalized' ;;
    generic|online|raw) printf 'online_normalized' ;;
    *) printf '%s_normalized' "$1" ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  ./run/10_online_normalize_data.sh --input RAW_FILE [--output NORMALIZED_FILE]
  ./run/10_online_normalize_data.sh --hf-source HF_SOURCE [--output NORMALIZED_FILE]

Kinds:
  --kind stageA   precomputed teacher-target dataset for Stage A student training
  --kind stageB   Stage B raw dataset before prepare/train
  --kind generic  generic online raw dataset normalization

HF_SOURCE examples:
  MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1
  MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020
  MindCastSogang/Youtube_news_preprocessed_data/preprocessed/v1/2020/01/01-10/news_comments.json

If --output is omitted, the file is auto-saved under outputs/ with:
  stageA -> stageA_normalized_<source>_<YYYYMMDD>.parquet
  stageB -> stageB_normalized_<source>_<YYYYMMDD>.parquet
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
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
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
KIND_PREFIX="$(resolve_kind_prefix "${KIND}")"
if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="$(make_data_output_path "${KIND_PREFIX}" "${SOURCE_RESOLVED}" parquet)"
fi
print_resolved_path "Resolved kind" "${KIND}"
print_resolved_path "Resolved source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved output" "${OUTPUT_PATH}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/normalize_dataset.py"
  --output "${OUTPUT_PATH}"
  --seed "${SEED}"
)
if [[ -n "${INPUT_PATH}" ]]; then CMD+=(--input "${INPUT_PATH}"); fi
if [[ -n "${HF_SOURCE}" ]]; then CMD+=(--hf-source "${HF_SOURCE}" --hf-revision "${HF_REVISION}"); fi
if [[ -n "${HF_TOKEN}" ]]; then CMD+=(--hf-token "${HF_TOKEN}"); fi
if [[ -n "${HF_MAX_FILES}" ]]; then CMD+=(--hf-max-files "${HF_MAX_FILES}"); fi
if [[ -n "${MAX_ROWS}" ]]; then CMD+=(--max-rows "${MAX_ROWS}"); fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '
'
"${CMD[@]}"
