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
OUTPUT_DIR=""
STUDENT_MODEL="beomi/KcELECTRA-base"
BATCH_SIZE="16"
EPOCHS="8"
LR="2e-5"
MAX_LENGTH="192"
SEED="42"
VAL_SIZE="0.1"
MAX_ROWS=""
LOCAL_FILES_ONLY="0"
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run/12_online_train_stage_a.sh --input TEACHER_TARGETS.parquet [--output-dir OUTPUT_DIR]

Input should be a precomputed teacher-target dataset with `comment` and
`teacher_prob_*` columns. In the default flow, this is usually the normalized
output of run/10_online_normalize_data.sh --kind stageA.

If --output-dir is omitted, the checkpoint directory is auto-created under outputs/pt/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_PATH="$2"; shift 2 ;;
    --hf-source) HF_SOURCE="$2"; shift 2 ;;
    --hf-revision) HF_REVISION="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --hf-max-files) HF_MAX_FILES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --val-size) VAL_SIZE="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
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
if [[ -z "${OUTPUT_DIR}" ]]; then OUTPUT_DIR="$(make_pt_output_dir stageA_comment_distill "${SOURCE_RESOLVED}")"; fi
print_resolved_path "Resolved source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved output dir" "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/train_stage_a.py"
  --output-dir "${OUTPUT_DIR}"
  --student-model "${STUDENT_MODEL}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --max-length "${MAX_LENGTH}"
  --seed "${SEED}"
  --val-size "${VAL_SIZE}"
)
if [[ -n "${INPUT_PATH}" ]]; then CMD+=(--input "${INPUT_PATH}"); fi
if [[ -n "${HF_SOURCE}" ]]; then CMD+=(--hf-source "${HF_SOURCE}" --hf-revision "${HF_REVISION}"); fi
if [[ -n "${HF_TOKEN}" ]]; then CMD+=(--hf-token "${HF_TOKEN}"); fi
if [[ -n "${HF_MAX_FILES}" ]]; then CMD+=(--hf-max-files "${HF_MAX_FILES}"); fi
if [[ -n "${MAX_ROWS}" ]]; then CMD+=(--max-rows "${MAX_ROWS}"); fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then CMD+=(--local-files-only); fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '
'
"${CMD[@]}"
