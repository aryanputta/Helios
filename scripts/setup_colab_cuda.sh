#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

echo "[colab] repo_root=$REPO_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[colab] nvidia-smi"
  nvidia-smi || true
else
  echo "[colab] nvidia-smi not found"
fi

if [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "[colab] nvcc"
  nvcc --version
else
  echo "[colab] nvcc not found in PATH"
fi

APT_RUNNER=()
if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
  APT_RUNNER=(sudo)
fi

DEBIAN_FRONTEND=noninteractive "${APT_RUNNER[@]}" apt-get update
DEBIAN_FRONTEND=noninteractive "${APT_RUNNER[@]}" apt-get install -y build-essential cmake ninja-build

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHELIOS_BUILD_GOOGLE_BENCHMARK=OFF \
  -DHELIOS_BUILD_NVBENCH=OFF

cmake --build "$BUILD_DIR" -- -j"$(nproc)"

echo "[colab] build complete: $BUILD_DIR/helios"
