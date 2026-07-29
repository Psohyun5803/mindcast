#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${ROOT_DIR}/run/_common.sh"

BASE_CHECKPOINT=""
STAGEB_CHECKPOINT=""
STUDENT_MODEL="beomi/KcELECTRA-base"
LABEL_MAP="${ROOT_DIR}/assets/kote_id2label.json"
MAJOR_MAPPING="${ROOT_DIR}/assets/mapping_ver1.json"
MAX_LENGTH="192"
OUTPUT_PATH=""
SOURCE_TAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run/15_online_export_offline_bundle.sh --base-checkpoint STAGEA_PT [--label-map LABEL_MAP_JSON] [--output OUTPUT_BUNDLE_PT] [--stageb-checkpoint STAGEB_PT]

If --label-map is omitted, assets/kote_id2label.json is used automatically.
If --output is omitted, the bundle is auto-saved under outputs/pt/.
Base-only export uses offline_bundle_stagea_only_<bundle-source>_<YYYYMMDD>.pt.
A+B export uses offline_bundle_stagea_stageb_<bundle-source>_<YYYYMMDD>.pt.
The default <bundle-source> is derived from the Stage A / Stage B checkpoint directory names.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --stageb-checkpoint) STAGEB_CHECKPOINT="$2"; shift 2 ;;
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --label-map) LABEL_MAP="$2"; shift 2 ;;
    --major-mapping) MAJOR_MAPPING="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --source-tag) SOURCE_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${BASE_CHECKPOINT}" ]]; then
  echo "--base-checkpoint is required" >&2
  usage
  exit 1
fi

SOURCE_RESOLVED="$(resolve_bundle_source_tag "${BASE_CHECKPOINT}" "${STAGEB_CHECKPOINT}" "${SOURCE_TAG}")"
if [[ -z "${OUTPUT_PATH}" ]]; then
  if [[ -n "${STAGEB_CHECKPOINT}" ]]; then
    OUTPUT_PATH="$(make_pt_output_path offline_bundle_stagea_stageb "${SOURCE_RESOLVED}")"
  else
    OUTPUT_PATH="$(make_pt_output_path offline_bundle_stagea_only "${SOURCE_RESOLVED}")"
  fi
fi
print_resolved_path "Resolved bundle source tag" "${SOURCE_RESOLVED}"
print_resolved_path "Resolved label map" "${LABEL_MAP}"
print_resolved_path "Resolved output" "${OUTPUT_PATH}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/export_offline_bundle.py"
  --base-checkpoint "${BASE_CHECKPOINT}"
  --student-model "${STUDENT_MODEL}"
  --label-map "${LABEL_MAP}"
  --major-mapping "${MAJOR_MAPPING}"
  --max-length "${MAX_LENGTH}"
  --output "${OUTPUT_PATH}"
)
if [[ -n "${STAGEB_CHECKPOINT}" ]]; then CMD+=(--stageb-checkpoint "${STAGEB_CHECKPOINT}"); fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '
'
"${CMD[@]}"
