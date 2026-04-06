# Helios Agent Task Tracker

## Current agent assignments

### Data Agent
- [x] create dataset download entry points for SuiteSparse and SNAP
- [x] validate loader support for Matrix Market and SNAP edge lists
- [x] add processed-data caches for repeated sparse and graph runs
- [ ] document dataset provenance, snapshot dates, and processing steps in README
- [x] emit dataset manifests under `data/processed/manifests/`
- [ ] propose the first benchmark dataset set for dense, sparse, and graph runs
- [x] add a public-safe artifact export path so proof bundles can be shared without leaking local machine details

### Kernel Agent
- [x] add CUDA kernel stub `src/gpu/gpu_kernels.cu`
- [x] implement CPU scalar dense and sparse reference kernels
- [x] implement CPU reference BFS and PageRank graph kernels
- [x] implement AVX2 specialization and correctness harness
- [x] implement threaded CPU dense and sparse baselines
- [ ] validate CUDA tiled GEMM and CSR SpMV on a machine with a real CUDA toolchain
- [ ] keep baseline scalar CPU output available for benchmark comparisons
- [x] wire vendor baseline entry points so cuBLAS and cuSPARSE can be benchmarked once a CUDA host is available
- [x] add Colab-oriented build and proof scripts for future CUDA validation

### Planner Agent
- [x] define `include/helios/planner.h`
- [x] implement basic strategy selection in `src/planner/planner.cpp`
- [x] refine dense CPU backend selection so small-but-vectorizable cases prefer AVX2 over scalar
- [ ] refine heuristics for density, size, and backend cost models using accumulated planner observations
- [x] expose planner decisions in CLI output and logs
- [x] connect planner outputs to profiler and benchmark metadata
- [x] handle explicit graph direction overrides for datasets that do not encode semantics in-file
- [x] surface CPU and CUDA capability notes so unavailable GPU paths are reported honestly
- [x] record planner observations so future heuristics can be tuned from actual benchmark winners

### Benchmark Agent
- [x] add NVBench harness scaffolding under `bench/nvbench`
- [x] add Google Benchmark scaffolding under `bench/google_benchmark`
- [x] implement CSV and JSON result exporters
- [x] add warmup, repeated trials, and median/p95 reporting
- [x] add measured scalar versus AVX2 versus threaded comparisons for dense and sparse CPU runs
- [x] add result-to-result comparison tooling
- [x] add proof-report generation from exported runs and planner logs
- [x] add effective GFLOP/s, effective bandwidth, and variability metrics to benchmark outputs
- [ ] require dataset provenance in every benchmark summary
- [x] separate private raw artifacts from publish-safe exported artifacts

### Documentation Agent
- [x] create `AGENTS.md` with role definitions
- [x] capture current progress in `WORKPLAN.md`
- [x] update README with dataset and benchmarking discipline
- [x] document how to reproduce results using `scripts/`
- [x] keep README, WORKPLAN, and AGENT_TASKS synchronized with actual implementation status
- [x] add a single-command repro script for the current runtime feature set
- [x] document proof-report generation and fresh-run planner-log handling
- [x] document Colab usage and safe public publishing

## Improvement focus
- each agent should iterate on its own area and cross-review adjacent components
- each agent should propose at least one stronger approach after their first implementation
- all decisions must remain grounded in real data, real tests, and reproducible outputs
- any benchmark claim must be backed by raw CSV or JSON plus dataset provenance
