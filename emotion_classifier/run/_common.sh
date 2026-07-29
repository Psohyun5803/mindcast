#!/usr/bin/env bash

# Shared helpers for auto-naming outputs and organizing checkpoint files.

today_stamp() {
  date +%Y%m%d
}

slugify() {
  printf '%s' "$1"     | tr '[:upper:]' '[:lower:]'     | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g'
}

basename_no_ext() {
  local base
  base="$(basename "$1")"
  printf '%s' "${base%.*}"
}

source_tag_from_local_path() {
  local path parent base
  path="${1%/}"
  parent="$(basename "$(dirname "$path")")"
  base="$(basename_no_ext "$path")"
  case "$parent" in
    outputs|output|data|pt|checkpoints|results|run|docs|src)
      slugify "$base"
      ;;
    *)
      if [[ -n "$parent" && "$parent" != "." && "$parent" != "/" ]]; then
        slugify "${parent}_${base}"
      else
        slugify "$base"
      fi
      ;;
  esac
}

source_tag_from_hf_source() {
  local src parent base last
  src="${1%/}"
  if [[ "$src" == *.json ]]; then
    parent="$(basename "$(dirname "$src")")"
    base="$(basename_no_ext "$src")"
    printf 'hf_%s' "$(slugify "${parent}_${base}")"
  else
    last="$(basename "$src")"
    printf 'hf_%s' "$(slugify "$last")"
  fi
}

source_tag_from_years() {
  local joined=""
  local year
  for year in "$@"; do
    joined+="${year}_"
  done
  joined="${joined%_}"
  printf 'hf_%s' "$(slugify "$joined")"
}

resolve_input_source_tag() {
  local input_path="${1:-}"
  local hf_source="${2:-}"
  local source_tag_override="${3:-}"
  if [[ -n "$source_tag_override" ]]; then
    slugify "$source_tag_override"
  elif [[ -n "$hf_source" ]]; then
    source_tag_from_hf_source "$hf_source"
  elif [[ -n "$input_path" ]]; then
    source_tag_from_local_path "$input_path"
  else
    printf 'direct_text'
  fi
}

resolve_teacher_source_tag() {
  local source_tag_override="${1:-}"
  shift || true
  if [[ -n "$source_tag_override" ]]; then
    slugify "$source_tag_override"
  else
    source_tag_from_years "$@"
  fi
}

make_data_output_path() {
  local kind="$1"
  local source_tag="$2"
  local ext="$3"
  mkdir -p "${ROOT_DIR}/outputs"
  printf '%s/outputs/%s_%s_%s.%s' "$ROOT_DIR" "$kind" "$source_tag" "$(today_stamp)" "$ext"
}

make_pt_output_path() {
  local kind="$1"
  local source_tag="$2"
  mkdir -p "${ROOT_DIR}/outputs/pt"
  printf '%s/outputs/pt/%s_%s_%s.pt' "$ROOT_DIR" "$kind" "$source_tag" "$(today_stamp)"
}

make_pt_output_dir() {
  local kind="$1"
  local source_tag="$2"
  mkdir -p "${ROOT_DIR}/outputs/pt"
  printf '%s/outputs/pt/%s_%s_%s' "$ROOT_DIR" "$kind" "$source_tag" "$(today_stamp)"
}

print_resolved_path() {
  local label="$1"
  local value="$2"
  printf '%s: %s
' "$label" "$value"
}

bundle_search_dirs() {
  printf '%s
' "${ROOT_DIR}/outputs/pt" "${ROOT_DIR}/outputs"
}

has_bundle_in_default_dirs() {
  local pattern="$1"
  local dir
  while read -r dir; do
    if [[ -d "$dir" ]] && compgen -G "$dir/$pattern" > /dev/null; then
      return 0
    fi
  done < <(bundle_search_dirs)
  return 1
}


latest_stagea_checkpoint() {
  local found=""
  found="$({ find "${ROOT_DIR}/outputs/pt" -maxdepth 2 -type f -path "*/student_comment_distill.pt" -printf '%T@ %p
' 2>/dev/null || true; } | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -n "$found" ]]; then
    printf '%s
' "$found"
    return 0
  fi
  return 1
}

require_or_resolve_stagea_checkpoint() {
  local explicit_path="${1:-}"
  if [[ -n "$explicit_path" ]]; then
    printf '%s
' "$explicit_path"
    return 0
  fi
  if latest_stagea_checkpoint; then
    return 0
  fi
  echo "No Stage A checkpoint was provided and no student_comment_distill.pt was found under ${ROOT_DIR}/outputs/pt" >&2
  return 1
}

checkpoint_run_tag() {
  local path parent base
  path="${1%/}"
  parent="$(basename "$(dirname "$path")")"
  base="$(basename "$path")"
  case "$base" in
    student_comment_distill.pt|stageB_adapter_checkpoint.pt)
      slugify "$parent"
      ;;
    *)
      source_tag_from_local_path "$path"
      ;;
  esac
}

resolve_bundle_source_tag() {
  local base_checkpoint="${1:-}"
  local stageb_checkpoint="${2:-}"
  local source_tag_override="${3:-}"
  local base_tag stageb_tag
  if [[ -n "$source_tag_override" ]]; then
    slugify "$source_tag_override"
    return 0
  fi
  base_tag="from_$(checkpoint_run_tag "$base_checkpoint")"
  if [[ -n "$stageb_checkpoint" ]]; then
    stageb_tag="with_$(checkpoint_run_tag "$stageb_checkpoint")"
    slugify "${base_tag}__${stageb_tag}"
  else
    slugify "$base_tag"
  fi
}
