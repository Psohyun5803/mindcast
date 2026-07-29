#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${ROOT_DIR}/run/_common.sh"

OUTPUT_PATH=""
TEACHER_MODEL="searle-j/kote_for_easygoing_people"
YEARS=("2020" "2022" "2023" "2025" "2026")
INCLUDE_2025_VARIANT="ver1"
INCLUDE_2026_VARIANT="ver1"
REVISION="main"
TOKEN=""
MAX_FILES=""
MAX_COMMENTS_PER_FILE="300"
MAX_ROWS=""
SEED="42"
BATCH_SIZE="32"
MAX_LENGTH="192"
LOCAL_FILES_ONLY="0"
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run/11_online_prepare_stage_a_teacher_targets.sh [--output OUTPUT_PARQUET]

This is an auxiliary data-preparation utility, not the default start point of
model training. The main online training flow assumes teacher targets were
already produced in an earlier step outside this package.

If --output is omitted, the file is auto-saved under outputs/ as
stageA_teacher_targets_<source>_<YYYYMMDD>.parquet.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --years)
      shift; YEARS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do YEARS+=("$1"); shift; done ;;
    --include-2025-variant) INCLUDE_2025_VARIANT="$2"; shift 2 ;;
    --include-2026-variant) INCLUDE_2026_VARIANT="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    --max-comments-per-file) MAX_COMMENTS_PER_FILE="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY="1"; shift 1 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

SOURCE_RESOLVED="$(resolve_teacher_source_tag "${SOURCE_TAG}" "${YEARS[@]}")"
if [[ -z "${OUTPUT_PATH}" ]]; then OUTPUT_PATH="$(make_data_output_path stageA_teacher_targets "${SOURCE_RESOLVED}" parquet)"; fi
print_resolved_path "Resolved source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved output" "${OUTPUT_PATH}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/prepare_stage_a_teacher_targets.py"
  --teacher-model "${TEACHER_MODEL}"
  --output "${OUTPUT_PATH}"
  --include-2025-variant "${INCLUDE_2025_VARIANT}"
  --include-2026-variant "${INCLUDE_2026_VARIANT}"
  --revision "${REVISION}"
  --max-comments-per-file "${MAX_COMMENTS_PER_FILE}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
)
if [[ ${#YEARS[@]} -gt 0 ]]; then CMD+=(--years "${YEARS[@]}"); fi
if [[ -n "${TOKEN}" ]]; then CMD+=(--token "${TOKEN}"); fi
if [[ -n "${MAX_FILES}" ]]; then CMD+=(--max-files "${MAX_FILES}"); fi
if [[ -n "${MAX_ROWS}" ]]; then CMD+=(--max-rows "${MAX_ROWS}"); fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then CMD+=(--local-files-only); fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '
'
"${CMD[@]}"
