#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/run/_common.sh"

MODE="full"
STAGEA_INPUT=""
STAGEA_CHECKPOINT=""
STAGEA_NORMALIZED_OUTPUT=""
STAGEA_OUTPUT_DIR=""
LABEL_MAP_PATH="${ROOT_DIR}/assets/kote_id2label.json"
STAGEB_INPUT=""
STAGEB_NORMALIZED_OUTPUT=""
STAGEB_OUTPUT_DIR=""
BUNDLE_OUTPUT=""

STUDENT_MODEL="beomi/KcELECTRA-base"
STAGEA_EPOCHS="8"
STAGEA_BATCH_SIZE="16"
STAGEA_MAX_ROWS=""
STAGEB_EPOCHS="5"
STAGEB_BATCH_SIZE="16"
STAGEB_MAX_ROWS=""
SEED="42"
LOCAL_FILES_ONLY="0"
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  Full A+B run:
    ./run/16_online_pipeline.sh       --mode full       --stagea-input data/stageA_teacher_targets.parquet       [--label-map assets/kote_id2label.json]       --stageb-input data/stageB_gold_raw.xlsx

  Stage A only:
    ./run/16_online_pipeline.sh       --mode stagea       --stagea-input data/stageA_teacher_targets.parquet       [--label-map assets/kote_id2label.json]

  Stage B only:
    ./run/16_online_pipeline.sh       --mode stageb       [--stagea-checkpoint outputs/pt/stageA_comment_distill_xxx/student_comment_distill.pt]       [--label-map assets/kote_id2label.json]       --stageb-input data/stageB_gold_raw.xlsx

Modes:
  --mode full    normalize Stage A -> train A -> normalize Stage B -> train B -> export bundle
  --mode stagea  normalize Stage A -> train A -> export bundle(base only)
  --mode stageb  normalize Stage B -> train B(using existing or latest Stage A checkpoint) -> export bundle

Data assumptions:
  1. Stage A input is already a precomputed teacher-target dataset from an earlier step.
     Minimum expected fields after normalization:
       - comment
       - teacher_prob_* columns

  2. Stage label order defaults to assets/kote_id2label.json.
     If you intentionally trained against a different label order, pass that label map explicitly.

  3. Stage B input is a gold dataset.
     Minimum expected fields after normalization:
       - comment
       - sarcasm_label
       - emotion_labels
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --stagea-input) STAGEA_INPUT="$2"; shift 2 ;;
    --stagea-checkpoint) STAGEA_CHECKPOINT="$2"; shift 2 ;;
    --stagea-normalized-output) STAGEA_NORMALIZED_OUTPUT="$2"; shift 2 ;;
    --stagea-output-dir) STAGEA_OUTPUT_DIR="$2"; shift 2 ;;
    --label-map) LABEL_MAP_PATH="$2"; shift 2 ;;
    --stageb-input) STAGEB_INPUT="$2"; shift 2 ;;
    --stageb-normalized-output) STAGEB_NORMALIZED_OUTPUT="$2"; shift 2 ;;
    --stageb-output-dir) STAGEB_OUTPUT_DIR="$2"; shift 2 ;;
    --bundle-output) BUNDLE_OUTPUT="$2"; shift 2 ;;
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --stagea-epochs) STAGEA_EPOCHS="$2"; shift 2 ;;
    --stagea-batch-size) STAGEA_BATCH_SIZE="$2"; shift 2 ;;
    --stagea-max-rows) STAGEA_MAX_ROWS="$2"; shift 2 ;;
    --stageb-epochs) STAGEB_EPOCHS="$2"; shift 2 ;;
    --stageb-batch-size) STAGEB_BATCH_SIZE="$2"; shift 2 ;;
    --stageb-max-rows) STAGEB_MAX_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY="1"; shift 1 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "${MODE}" in
  full)
    if [[ -z "${STAGEA_INPUT}" || -z "${STAGEB_INPUT}" ]]; then
      echo "full mode requires --stagea-input and --stageb-input" >&2
      usage
      exit 1
    fi
    ;;
  stagea)
    if [[ -z "${STAGEA_INPUT}" ]]; then
      echo "stagea mode requires --stagea-input" >&2
      usage
      exit 1
    fi
    ;;
  stageb)
    if [[ -z "${STAGEB_INPUT}" ]]; then
      echo "stageb mode requires --stageb-input" >&2
      usage
      exit 1
    fi
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

STAGEA_SOURCE_TAG=""
if [[ -n "${STAGEA_INPUT}" ]]; then
  STAGEA_SOURCE_TAG="$(resolve_input_source_tag "${STAGEA_INPUT}" "" "${SOURCE_TAG}")"
fi
STAGEB_SOURCE_TAG=""
if [[ -n "${STAGEB_INPUT}" ]]; then
  STAGEB_SOURCE_TAG="$(resolve_input_source_tag "${STAGEB_INPUT}" "" "${SOURCE_TAG}")"
fi

if [[ "${MODE}" != "stageb" ]]; then
  if [[ -z "${STAGEA_NORMALIZED_OUTPUT}" ]]; then STAGEA_NORMALIZED_OUTPUT="$(make_data_output_path stageA_normalized "${STAGEA_SOURCE_TAG}" parquet)"; fi
  if [[ -z "${STAGEA_OUTPUT_DIR}" ]]; then STAGEA_OUTPUT_DIR="$(make_pt_output_dir stageA_comment_distill "${STAGEA_SOURCE_TAG}")"; fi
  STAGEA_CHECKPOINT="${STAGEA_OUTPUT_DIR}/student_comment_distill.pt"
else
  STAGEA_CHECKPOINT="$(require_or_resolve_stagea_checkpoint "${STAGEA_CHECKPOINT}")"
fi
if [[ "${MODE}" != "stagea" ]]; then
  if [[ -z "${STAGEB_NORMALIZED_OUTPUT}" ]]; then STAGEB_NORMALIZED_OUTPUT="$(make_data_output_path stageB_normalized "${STAGEB_SOURCE_TAG}" parquet)"; fi
  if [[ -z "${STAGEB_OUTPUT_DIR}" ]]; then STAGEB_OUTPUT_DIR="$(make_pt_output_dir stageB_adapter "${STAGEB_SOURCE_TAG}")"; fi
  STAGEB_CHECKPOINT="${STAGEB_OUTPUT_DIR}/stageB_adapter_checkpoint.pt"
fi
BUNDLE_SOURCE_TAG="$(resolve_bundle_source_tag "${STAGEA_CHECKPOINT}" "${STAGEB_CHECKPOINT:-}" "${SOURCE_TAG}")"
if [[ -z "${BUNDLE_OUTPUT}" ]]; then
  if [[ "${MODE}" == "stagea" ]]; then
    BUNDLE_OUTPUT="$(make_pt_output_path offline_bundle_stagea_only "${BUNDLE_SOURCE_TAG}")"
  else
    BUNDLE_OUTPUT="$(make_pt_output_path offline_bundle_stagea_stageb "${BUNDLE_SOURCE_TAG}")"
  fi
fi

print_resolved_path "Resolved mode" "${MODE}"
print_resolved_path "Resolved bundle source tag" "${BUNDLE_SOURCE_TAG}"
print_resolved_path "Resolved label map" "${LABEL_MAP_PATH}"
if [[ -n "${STAGEA_SOURCE_TAG}" ]]; then print_resolved_path "Resolved Stage A source tag" "${STAGEA_SOURCE_TAG}"; fi
print_resolved_path "Resolved Stage A checkpoint" "${STAGEA_CHECKPOINT}"
if [[ -n "${STAGEB_SOURCE_TAG}" ]]; then print_resolved_path "Resolved Stage B source tag" "${STAGEB_SOURCE_TAG}"; fi
if [[ -n "${STAGEA_NORMALIZED_OUTPUT}" ]]; then print_resolved_path "Resolved Stage A normalized output" "${STAGEA_NORMALIZED_OUTPUT}"; fi
if [[ -n "${STAGEA_OUTPUT_DIR}" ]]; then print_resolved_path "Resolved Stage A output dir" "${STAGEA_OUTPUT_DIR}"; fi
if [[ -n "${STAGEB_NORMALIZED_OUTPUT}" ]]; then print_resolved_path "Resolved Stage B normalized output" "${STAGEB_NORMALIZED_OUTPUT}"; fi
if [[ -n "${STAGEB_OUTPUT_DIR}" ]]; then print_resolved_path "Resolved Stage B output dir" "${STAGEB_OUTPUT_DIR}"; fi
print_resolved_path "Resolved bundle output" "${BUNDLE_OUTPUT}"

if [[ "${MODE}" == "full" || "${MODE}" == "stagea" ]]; then
  STEP1_CMD=(
    "${ROOT_DIR}/run/10_online_normalize_data.sh"
    --kind stageA
    --input "${STAGEA_INPUT}"
    --output "${STAGEA_NORMALIZED_OUTPUT}"
    --seed "${SEED}"
    --source-tag "${STAGEA_SOURCE_TAG}"
  )

  STEP2_CMD=(
    "${ROOT_DIR}/run/12_online_train_stage_a.sh"
    --input "${STAGEA_NORMALIZED_OUTPUT}"
    --output-dir "${STAGEA_OUTPUT_DIR}"
    --student-model "${STUDENT_MODEL}"
    --batch-size "${STAGEA_BATCH_SIZE}"
    --epochs "${STAGEA_EPOCHS}"
    --seed "${SEED}"
    --source-tag "${STAGEA_SOURCE_TAG}"
  )
  if [[ -n "${STAGEA_MAX_ROWS}" ]]; then STEP2_CMD+=(--max-rows "${STAGEA_MAX_ROWS}"); fi
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then STEP2_CMD+=(--local-files-only); fi
fi

if [[ "${MODE}" == "full" || "${MODE}" == "stageb" ]]; then
  STEP3_CMD=(
    "${ROOT_DIR}/run/10_online_normalize_data.sh"
    --kind stageB
    --input "${STAGEB_INPUT}"
    --output "${STAGEB_NORMALIZED_OUTPUT}"
    --seed "${SEED}"
    --source-tag "${STAGEB_SOURCE_TAG}"
  )

  STEP4_CMD=(
    "${ROOT_DIR}/run/14_online_train_stage_b.sh"
    --input "${STAGEB_NORMALIZED_OUTPUT}"
    --stagea-checkpoint "${STAGEA_CHECKPOINT}"
    --label-map "${LABEL_MAP_PATH}"
    --student-model "${STUDENT_MODEL}"
    --output-dir "${STAGEB_OUTPUT_DIR}"
    --batch-size "${STAGEB_BATCH_SIZE}"
    --epochs "${STAGEB_EPOCHS}"
    --seed "${SEED}"
    --source-tag "${STAGEB_SOURCE_TAG}"
  )
  if [[ -n "${STAGEB_MAX_ROWS}" ]]; then STEP4_CMD+=(--max-rows "${STAGEB_MAX_ROWS}"); fi
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then STEP4_CMD+=(--local-files-only); fi
fi

STEP5_CMD=(
  "${ROOT_DIR}/run/15_online_export_offline_bundle.sh"
  --base-checkpoint "${STAGEA_CHECKPOINT}"
  --student-model "${STUDENT_MODEL}"
  --label-map "${LABEL_MAP_PATH}"
  --output "${BUNDLE_OUTPUT}"
)
if [[ "${MODE}" != "stagea" ]]; then STEP5_CMD+=(--stageb-checkpoint "${STAGEB_CHECKPOINT}"); fi
if [[ -n "${SOURCE_TAG}" ]]; then STEP5_CMD+=(--source-tag "${SOURCE_TAG}"); fi

if [[ "${MODE}" == "full" ]]; then
  printf '[1/5] normalize Stage A teacher targets
'
  "${STEP1_CMD[@]}"
  printf '[2/5] train Stage A student
'
  "${STEP2_CMD[@]}"
  printf '[3/5] normalize Stage B gold input
'
  "${STEP3_CMD[@]}"
  printf '[4/5] train Stage B
'
  "${STEP4_CMD[@]}"
  printf '[5/5] export offline bundle
'
  "${STEP5_CMD[@]}"
elif [[ "${MODE}" == "stagea" ]]; then
  printf '[1/3] normalize Stage A teacher targets
'
  "${STEP1_CMD[@]}"
  printf '[2/3] train Stage A student
'
  "${STEP2_CMD[@]}"
  printf '[3/3] export offline bundle(base only)
'
  "${STEP5_CMD[@]}"
else
  printf '[1/3] normalize Stage B gold input
'
  "${STEP3_CMD[@]}"
  printf '[2/3] train Stage B
'
  "${STEP4_CMD[@]}"
  printf '[3/3] export offline bundle
'
  "${STEP5_CMD[@]}"
fi
