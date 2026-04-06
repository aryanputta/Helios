#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_ROOT="$REPO_ROOT/data/raw/snap"

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

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/helios-snap.XXXXXX")"

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

  mv -f "$tmp_file" "$output"
}

verify_gzip() {
  local archive="$1"
  gzip -t "$archive"
}

extract_gzip() {
  local archive="$1"
  local output="$2"
  local tmp_file

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/helios-snap-extract.XXXXXX")"

  if ! gzip -dc "$archive" >"$tmp_file"; then
    rm -f "$tmp_file"
    echo "Failed to decompress archive: $archive" >&2
    exit 1
  fi

  if [[ ! -s "$tmp_file" ]]; then
    rm -f "$tmp_file"
    echo "Decompressed file is empty: $archive" >&2
    exit 1
  fi

  mv -f "$tmp_file" "$output"
}

ensure_dataset() {
  local name="$1"
  local download_url="$2"

  local dataset_dir="$RAW_ROOT/$name"
  local archive_path="$dataset_dir/$name.txt.gz"
  local edge_list_path="$dataset_dir/$name.txt"

  mkdir -p "$dataset_dir"

  if [[ -s "$archive_path" ]]; then
    if verify_gzip "$archive_path"; then
      echo "Reusing existing archive: $archive_path"
    else
      echo "Existing archive failed verification, redownloading: $archive_path"
      rm -f "$archive_path"
      download_file "$download_url" "$archive_path"
      verify_gzip "$archive_path"
      rm -f "$edge_list_path"
    fi
  else
    echo "Downloading SNAP dataset: $name"
    download_file "$download_url" "$archive_path"
    rm -f "$edge_list_path"
  fi

  if [[ ! -s "$edge_list_path" ]]; then
    echo "Extracting edge list for $name"
    extract_gzip "$archive_path" "$edge_list_path"
  else
    echo "Reusing existing extracted file: $edge_list_path"
  fi

  echo "Ready: $edge_list_path"
}

main() {
  require_tool curl
  require_tool gzip
  require_tool mv
  require_tool mktemp

  mkdir -p "$RAW_ROOT"

  # These are real SNAP graph benchmarks used for sparse graph kernels.
  ensure_dataset "facebook_combined" "https://snap.stanford.edu/data/facebook_combined.txt.gz"
  ensure_dataset "ca-GrQc" "https://snap.stanford.edu/data/ca-GrQc.txt.gz"

  echo "SNAP download complete."
}

main "$@"
