#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

RESULT_DIR="$REPO_ROOT/results/json"
VALIDATE_JSON="$RESULT_DIR/bcsstk30_validate_walkthrough.json"
SCALAR_JSON="$RESULT_DIR/bcsstk30_scalar_walkthrough.json"
THREADED_JSON="$RESULT_DIR/bcsstk30_threaded_walkthrough.json"
COMPARE_JSON="$RESULT_DIR/bcsstk30_scalar_vs_threaded_walkthrough.json"
PROOF_JSON="$RESULT_DIR/bcsstk30_spmv_walkthrough.json"
PLANNER_LOG="$RESULT_DIR/planner_observations_walkthrough.jsonl"

mkdir -p "$RESULT_DIR"
rm -f "$VALIDATE_JSON" "$SCALAR_JSON" "$THREADED_JSON" "$COMPARE_JSON" "$PROOF_JSON" "$PLANNER_LOG"

run_step() {
  local title="$1"
  shift
  echo
  echo "=== $title ==="
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

run_step "Fetch real datasets" "$REPO_ROOT/scripts/fetch_suitesparse.sh"

run_step "Configure Helios" \
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHELIOS_BUILD_GOOGLE_BENCHMARK=OFF \
    -DHELIOS_BUILD_NVBENCH=OFF

run_step "Build Helios" cmake --build "$BUILD_DIR" -- -j4

run_step "Validate sparse kernels on real SuiteSparse data" \
  "$BUILD_DIR/helios" validate sparse \
    --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
    --compare-all \
    --threads 8 \
    --json "$VALIDATE_JSON"

run_step "Benchmark scalar baseline on the same matrix" \
  "$BUILD_DIR/helios" bench sparse \
    --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
    --backend scalar \
    --threads 8 \
    --warmup 1 \
    --trials 3 \
    --json "$SCALAR_JSON"

run_step "Benchmark threaded baseline on the same matrix" \
  "$BUILD_DIR/helios" bench sparse \
    --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
    --backend threaded \
    --threads 8 \
    --warmup 1 \
    --trials 3 \
    --json "$THREADED_JSON"

run_step "Compare scalar vs threaded directly" \
  "$BUILD_DIR/helios" compare \
    --lhs "$SCALAR_JSON" \
    --rhs "$THREADED_JSON" \
    --json "$COMPARE_JSON"

run_step "Let the planner auto-pick and log whether it chose the measured winner" \
  "$BUILD_DIR/helios" bench sparse \
    --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
    --compare-baselines \
    --threads 8 \
    --warmup 2 \
    --trials 5 \
    --planner-log "$PLANNER_LOG" \
    --json "$PROOF_JSON"

echo
echo "=== Raw Proof Extracted From Fresh Outputs ==="
python3 - "$VALIDATE_JSON" "$SCALAR_JSON" "$THREADED_JSON" "$COMPARE_JSON" "$PROOF_JSON" "$PLANNER_LOG" <<'PY'
import json
import sys
from pathlib import Path

validate_path, scalar_path, threaded_path, compare_path, proof_path, planner_path = map(Path, sys.argv[1:])

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

validate = load_json(validate_path)
scalar = load_json(scalar_path)
threaded = load_json(threaded_path)
compare = load_json(compare_path)
proof = load_json(proof_path)

with planner_path.open("r", encoding="utf-8") as handle:
    planner_entries = [json.loads(line) for line in handle if line.strip()]

planner_last = planner_entries[-1] if planner_entries else {}

print(f"dataset_source_url={validate.get('dataset_source_url')}")
print(f"dataset_input_path={validate.get('dataset_input_path')}")
print(f"validation_passed={validate.get('validation_passed')}")
print(f"threaded_max_abs_diff_vs_scalar={validate.get('threaded_max_abs_diff_vs_scalar')}")
print(f"scalar_median_ms={scalar.get('scalar_median_ms', scalar.get('median_ms'))}")
print(f"threaded_median_ms={threaded.get('threaded_median_ms', threaded.get('median_ms'))}")
print(f"speedup_rhs_over_lhs={compare.get('rhs_speedup_over_lhs')}")
print(f"planner_selected_backend={proof.get('selected_backend')}")
print(f"planner_selected_won={proof.get('selected_won')}")
print(f"winning_backend={proof.get('winning_backend')}")
print(f"planner_log_selected_backend={planner_last.get('selected_backend')}")
print(f"planner_log_winning_backend={planner_last.get('winning_backend')}")
print(f"planner_log_selected_won={planner_last.get('selected_won')}")
PY

echo
echo "Walkthrough outputs:"
echo "  $VALIDATE_JSON"
echo "  $SCALAR_JSON"
echo "  $THREADED_JSON"
echo "  $COMPARE_JSON"
echo "  $PROOF_JSON"
echo "  $PLANNER_LOG"
