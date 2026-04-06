#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

mkdir -p "$REPO_ROOT/results/csv" "$REPO_ROOT/results/json" "$REPO_ROOT/results/reports"

rm -f \
  "$REPO_ROOT/results/json/planner_observations_sparse_proof.jsonl" \
  "$REPO_ROOT/results/json/planner_observations_dense_proof.jsonl" \
  "$REPO_ROOT/results/json/proof_report.json" \
  "$REPO_ROOT/results/reports/proof_report.md"

"$REPO_ROOT/scripts/fetch_suitesparse.sh"
"$REPO_ROOT/scripts/fetch_snap.sh"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -- -j4

"$BUILD_DIR/helios" profile sparse \
  --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
  --csv "$REPO_ROOT/results/csv/bcsstk30_profile_v2.csv" \
  --json "$REPO_ROOT/results/json/bcsstk30_profile_v2.json"

"$BUILD_DIR/helios" validate sparse \
  --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
  --compare-all \
  --threads 8 \
  --json "$REPO_ROOT/results/json/bcsstk30_validate_v2.json"

"$BUILD_DIR/helios" bench sparse \
  --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
  --backend scalar \
  --threads 8 \
  --warmup 1 \
  --trials 3 \
  --json "$REPO_ROOT/results/json/bcsstk30_scalar_v2.json"

"$BUILD_DIR/helios" bench sparse \
  --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
  --backend threaded \
  --threads 8 \
  --warmup 1 \
  --trials 3 \
  --json "$REPO_ROOT/results/json/bcsstk30_threaded_v2.json"

"$BUILD_DIR/helios" compare \
  --lhs "$REPO_ROOT/results/json/bcsstk30_scalar_v2.json" \
  --rhs "$REPO_ROOT/results/json/bcsstk30_threaded_v2.json" \
  --json "$REPO_ROOT/results/json/bcsstk30_scalar_vs_threaded_v2.json"

"$BUILD_DIR/helios" bench sparse \
  --matrix "$REPO_ROOT/data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx" \
  --compare-baselines \
  --threads 8 \
  --warmup 3 \
  --trials 7 \
  --planner-log "$REPO_ROOT/results/json/planner_observations_sparse_proof.jsonl" \
  --json "$REPO_ROOT/results/json/bcsstk30_spmv_proof.json"

"$BUILD_DIR/helios" bench dense \
  --m 512 \
  --n 512 \
  --k 512 \
  --compare-baselines \
  --threads 8 \
  --warmup 2 \
  --trials 5 \
  --planner-log "$REPO_ROOT/results/json/planner_observations_dense_proof.jsonl" \
  --json "$REPO_ROOT/results/json/dense_512_proof.json"

"$BUILD_DIR/helios" validate graph \
  --graph "$REPO_ROOT/data/raw/snap/facebook_combined/facebook_combined.txt" \
  --algo bfs \
  --source 0 \
  --undirected \
  --json "$REPO_ROOT/results/json/facebook_bfs_validate_v2.json"

"$BUILD_DIR/helios" report \
  --result "$REPO_ROOT/results/json/bcsstk30_spmv_proof.json" \
  --result "$REPO_ROOT/results/json/dense_512_proof.json" \
  --planner-log "$REPO_ROOT/results/json/planner_observations_sparse_proof.jsonl" \
  --planner-log "$REPO_ROOT/results/json/planner_observations_dense_proof.jsonl" \
  --md "$REPO_ROOT/results/reports/proof_report.md" \
  --json "$REPO_ROOT/results/json/proof_report.json"

echo "Helios repro suite complete."
