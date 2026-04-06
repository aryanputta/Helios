#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_ROOT="$REPO_ROOT/data/raw/suitesparse"

require_tool() {
  local tool="$1"
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Missing required tool: $tool" >&2
    exit 1
  }
}

download_file() {
  local url="$1"
  local output="$2"
  local tmp_file

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/helios-suitesparse.XXXXXX")"

  if ! curl --fail --location --silent --show-error \
    --retry 3 --retry-all-errors --connect-timeout 10 \
    "$url" -o "$tmp_file"; then
    rm -f "$tmp_file"
    echo "Failed to download: $url" >&2
    exit 1
  fi

  if [[ ! -s "$tmp_file" ]]; then
    rm -f "$tmp_file"
    echo "Downloaded file is empty: $url" >&2
    exit 1
  fi

  mv "$tmp_file" "$output"
}

verify_tarball() {
  local archive="$1"
  tar -tzf "$archive" >/dev/null
}

extract_matrix_market() {
  local archive="$1"
  local target_dir="$2"
  local matrix_name="$3"

  local extract_dir
  extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/helios-suitesparse-extract.XXXXXX")"
  if ! tar -xzf "$archive" -C "$extract_dir"; then
    rm -rf "$extract_dir"
    echo "Failed to extract archive: $archive" >&2
    exit 1
  fi

  local extracted_matrix
  extracted_matrix="$(find "$extract_dir" -type f -name '*.mtx' | sort | head -n 1)"
  if [[ -z "$extracted_matrix" ]]; then
    rm -rf "$extract_dir"
    echo "No Matrix Market file found in archive: $archive" >&2
    exit 1
  fi

  mkdir -p "$target_dir"
  mv -f "$extracted_matrix" "$target_dir/$matrix_name.mtx"
  rm -rf "$extract_dir"
}

ensure_dataset() {
  local group="$1"
  local matrix="$2"
  local download_url="$3"

  local dataset_dir="$RAW_ROOT/$group/$matrix"
  local archive_path="$dataset_dir/$matrix.tar.gz"
  local matrix_path="$dataset_dir/$matrix.mtx"

  mkdir -p "$dataset_dir"

  if [[ -s "$archive_path" ]]; then
    if verify_tarball "$archive_path"; then
      echo "Reusing existing archive: $archive_path"
    else
      echo "Existing archive failed verification, redownloading: $archive_path"
      rm -f "$archive_path"
      download_file "$download_url" "$archive_path"
      verify_tarball "$archive_path"
      rm -f "$matrix_path"
    fi
  else
    echo "Downloading SuiteSparse dataset: $group/$matrix"
    download_file "$download_url" "$archive_path"
    rm -f "$matrix_path"
  fi

  if [[ ! -s "$matrix_path" ]]; then
    echo "Extracting Matrix Market file for $group/$matrix"
    extract_matrix_market "$archive_path" "$dataset_dir" "$matrix"
  else
    echo "Reusing existing extracted file: $matrix_path"
  fi

  echo "Ready: $matrix_path"
}

main() {
  require_tool curl
  require_tool tar
  require_tool find
  require_tool mv
  require_tool mktemp

  mkdir -p "$RAW_ROOT"

  # These are real SuiteSparse Matrix Collection pages with Matrix Market tarballs.
  ensure_dataset "HB" "bcsstk30" "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk30.tar.gz"
  ensure_dataset "Hamm" "add20" "https://suitesparse-collection-website.herokuapp.com/MM/Hamm/add20.tar.gz"

  echo "SuiteSparse download complete."
}

main "$@"
