#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$REPO_ROOT/scripts/setup_colab_cuda.sh"
"$REPO_ROOT/scripts/repro_suite.sh"
python3 "$REPO_ROOT/scripts/export_public_bundle.py" \
  --repo-root "$REPO_ROOT" \
  --output-dir "$REPO_ROOT/public_artifacts"

echo "[colab] proof run complete"
echo "[colab] sanitized bundle: $REPO_ROOT/public_artifacts"

