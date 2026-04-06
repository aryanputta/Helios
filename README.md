# Helios Compute Runtime

Helios is a real-data-first compute runtime for dense, sparse, and graph workloads. It is designed to ingest public datasets, measure workload characteristics at runtime, and route execution across CPU and GPU paths based on observed shape, density, and cost.

## What This Project Optimizes For
- real datasets from SuiteSparse and SNAP, not synthetic placeholders
- reproducible runs with raw inputs, exported outputs, and clear provenance
- backend selection that is driven by measured workload characteristics
- correctness checks against reference results before any performance claim
- benchmark reporting that includes CSV/JSON artifacts and enough context to rerun
- publish-safe sharing so local paths and hostnames do not need to leak when you show results publicly

## Runtime Areas
- `src/cli/` - command-line interface and workload dispatch
- `src/planner/` - backend selection and kernel strategy
- `src/profiler/` - workload profiling and execution metrics
- `src/io/` - dataset loading for SuiteSparse Matrix Market and SNAP edge lists
- `src/cpu/` - scalar, SIMD, and reference CPU kernels
- `src/gpu/` - CUDA kernels and GPU execution paths
- `bench/` - benchmark harnesses for NVBench and Google Benchmark
- `results/` - exported CSV, JSON, and profiler reports
- `docs/` - Colab and publishing guidance

## Data And Reproducibility
Use only real public datasets in benchmark runs. Each dataset should be traceable back to its source, format, and processing steps.

Minimum reproducibility expectations:
- record the dataset source, version or snapshot date, and input format
- keep raw inputs separate from processed artifacts
- export benchmark summaries as CSV and JSON
- preserve profiler output alongside benchmark output
- note the exact command used to run the workload or benchmark

If a result cannot be reproduced from the recorded dataset and command, it should not be treated as a final project claim.

## Privacy And Publishing
Generated artifacts in `build/`, `results/`, `data/raw/`, and `data/processed/` can contain local machine details such as absolute paths and hostnames. Those directories are now ignored by `.gitignore` and should not be pushed directly.

For a public-safe artifact bundle, run:

```bash
python3 scripts/export_public_bundle.py --repo-root . --output-dir public_artifacts
```

That bundle keeps the proof artifacts while redacting hostnames and rewriting local absolute paths to placeholders.

## What Runs Today
- real SuiteSparse Matrix Market ingestion with support for `real`, `integer`, and `pattern` coordinate data plus `general`, `symmetric`, and `skew-symmetric` storage
- real SNAP edge-list ingestion with CSR normalization and an explicit `--directed` or `--undirected` override when dataset metadata is not embedded in the file
- CPU scalar, AVX2, and threaded baselines for dense GEMM and sparse SpMV
- CLI benchmark runs with warmup, repeated trials, scalar-versus-optimized comparisons, median/p95/stddev summaries, effective GFLOP/s and bandwidth metrics, plus CSV and JSON exports
- optional CUDA backend wiring for dense GEMM and sparse CSR SpMV, with explicit capability reporting when CUDA is unavailable on the current machine
- dataset manifests under `data/processed/manifests/` and processed CSR/graph caches under `data/processed/cache/`
- `validate`, `compare`, and `report` commands for correctness checks, result-to-result analysis, and proof-summary generation
- planner observation logs in JSONL form so compare-baseline runs can record what the planner picked versus what actually won
- optional Google Benchmark and NVBench harness sources under `bench/`, built only when those dependencies are available
- vendor-baseline capability plumbing for cuBLAS and cuSPARSE through `--backend vendor`, with explicit unsupported reporting on non-CUDA hosts

## Real Dataset Workflow
Fetch the public datasets:

```bash
scripts/fetch_suitesparse.sh
scripts/fetch_snap.sh
```

Profile a real SuiteSparse matrix:

```bash
./build/helios profile sparse \
  --matrix data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx \
  --csv results/csv/bcsstk30_profile.csv \
  --json results/json/bcsstk30_profile.json
```

Benchmark a real SNAP graph:

```bash
./build/helios bench graph \
  --graph data/raw/snap/facebook_combined/facebook_combined.txt \
  --algo bfs \
  --source 0 \
  --undirected \
  --warmup 1 \
  --trials 3 \
  --csv results/csv/facebook_bfs_undirected.csv \
  --json results/json/facebook_bfs_undirected.json
```

Validate a real sparse dataset across available backends:

```bash
./build/helios validate sparse \
  --matrix data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx \
  --compare-all \
  --threads 8 \
  --json results/json/bcsstk30_validate_v2.json
```

Compare two exported result bundles:

```bash
./build/helios compare \
  --lhs results/json/bcsstk30_scalar_v2.json \
  --rhs results/json/bcsstk30_threaded_v2.json \
  --json results/json/bcsstk30_scalar_vs_threaded_v2.json
```

Write planner-observation training data while benchmarking:

```bash
./build/helios bench dense \
  --m 32 --n 32 --k 32 \
  --compare-baselines \
  --planner-log results/json/planner_observations.jsonl
```

Generate a proof summary from exported runs and planner logs:

```bash
./build/helios report \
  --result results/json/bcsstk30_spmv_proof.json \
  --result results/json/dense_512_proof.json \
  --planner-log results/json/planner_observations_sparse_proof.jsonl \
  --planner-log results/json/planner_observations_dense_proof.jsonl \
  --md results/reports/proof_report.md \
  --json results/json/proof_report.json
```

Run the end-to-end repro suite:

```bash
scripts/repro_suite.sh
```

The repro script now resets proof-log outputs before running so the generated report reflects only the fresh run that just completed.

## Colab CUDA Path
Colab is a good way to validate that Helios builds and runs CUDA code on a real NVIDIA GPU. For the fastest path:

```bash
bash scripts/run_colab_proof.sh
```

That will build Helios in a Colab-friendly way, run the current proof suite, and export a sanitized `public_artifacts/` bundle. See [COLAB.md](docs/COLAB.md) and [PUBLISHING.md](docs/PUBLISHING.md) for the full workflow.

## Current Status
The runtime now executes real sparse and graph inputs end to end, emits planner decisions and backend capability notes in the CLI, and writes reproducible CSV and JSON outputs under `results/`. Dense benchmarking still uses synthetic sanity inputs, but it now has measured scalar, AVX2, and threaded CPU baselines, small-dense auto selection that favors AVX2 instead of falling back to scalar too often, plus a wired CUDA execution hook for systems that actually have CUDA support.

On this machine specifically, CUDA, cuBLAS, and cuSPARSE remain unavailable because there is no CUDA compiler/runtime installed. Helios now reports that fact directly instead of implying that GPU baselines ran.

## Next Milestones
1. expand dense benchmarking beyond sanity inputs so optimized CPU and CUDA paths can be exercised on more realistic matrices
2. compile and validate the CUDA dense and sparse paths on a machine with an actual CUDA toolchain and device
3. tighten planner heuristics around transfer cost, irregular sparsity, and when AVX2 loses to threading
4. add dedicated NVBench and Google Benchmark harnesses on top of the current CLI timing path
5. expand provenance notes so every benchmark artifact records dataset source, local path, backend, and command line
