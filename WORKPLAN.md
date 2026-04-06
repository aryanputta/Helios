# Helios Compute Runtime Workplan

## Objective
Build Helios as a compute runtime that can be trusted on real datasets: load public inputs, profile them, choose an execution path, and publish reproducible benchmark evidence.

## Current Focus
- extend the newly working real-data CLI path from CPU reference execution toward optimized CPU and CUDA backends
- keep dataset semantics explicit, especially graph directionality and matrix symmetry handling
- attach every new benchmark result to reproducible CSV and JSON exports under `results/`
- keep planner output honest about what is implemented today versus what is only planned
- evolve documentation and task tracking immediately after each implemented capability lands

## Status Snapshot
- real SuiteSparse and SNAP download scripts are working and place datasets under `data/raw/...`
- the runtime now loads Matrix Market and SNAP edge-list inputs, profiles them, and emits planner decisions
- CPU scalar, AVX2, and threaded dense/sparse baselines now run through the CLI with repeated-trial summaries and correctness deltas versus scalar
- CSV and JSON result export is live for `profile` and `bench` commands
- `report` now turns exported results plus planner logs into a proof-oriented markdown/JSON summary
- repo publishing is now guarded by `.gitignore`, plus a sanitized public-artifact export path for sharing proof without leaking local machine details
- dataset manifests and processed-data caches are now created under `data/processed/`
- `validate` and `compare` commands now cover correctness and result-to-result analysis
- compare-baseline runs can now append planner-observation records for later heuristic tuning
- dense and sparse benchmark bundles now include effective GFLOP/s, effective bandwidth, and trial-variability metrics
- optional Google Benchmark and NVBench harness sources now exist under `bench/`
- CUDA dense and sparse backend hooks exist, but this machine still reports CUDA unavailable because no CUDA compiler/runtime is present
- vendor baseline plumbing is wired behind `--backend vendor`, ready for cuBLAS/cuSPARSE validation on a real CUDA host
- next step is to validate the CUDA path on appropriate hardware rather than pretending it already ran

## Near-Term Work
1. Data Agent
   - validate real dataset downloads and document provenance details
   - finish loader coverage for Matrix Market and edge list inputs
   - record recommended benchmark dataset selections
2. Kernel Agent
   - implement CPU scalar and AVX2 dense kernels
   - implement CUDA tiled GEMM, fused kernels, and sparse CSR paths
   - add correctness harnesses against reference outputs
3. Planner Agent
   - design profiling features that support backend selection
   - route dense, sparse, and graph workloads to the right execution path
   - surface planner decisions in CLI output
4. Benchmark Agent
   - implement NVBench and Google Benchmark harnesses
   - add warmup, repeated trials, and median/p95 reporting
   - export CSV and JSON summaries with provenance metadata
5. Documentation Agent
   - keep README, WORKPLAN, and AGENT_TASKS aligned with real project state
   - document dataset, benchmark, and reproducibility expectations
   - capture final results only after they are backed by raw exports and source provenance

## Milestones
### Milestone 1: Real-data ingestion
- completed for the initial dataset set
- raw SuiteSparse and SNAP inputs are downloaded and tracked under `data/raw/`
- dataset loaders accept Matrix Market and edge list formats with symmetry and graph-direction handling
- provenance documentation still needs expansion per dataset

### Milestone 2: Reference execution
- partially completed
- CPU scalar reference paths are available for dense, sparse, BFS, and PageRank workloads
- AVX2 and threaded dense/sparse CPU baselines are available for benchmark comparison
- planner and CLI output now describe which path was selected and why
- correctness coverage exists for loaders and reference graph kernels, with more backend comparison tests still needed

### Milestone 3: Benchmark evidence
- in progress
- CLI runs emit CSV and JSON artifacts today
- repeated trials plus min, median, p95, max, and mean are recorded
- result bundles now include host metadata, command lines, cache/manifest paths, and dataset provenance fields
- planner observations can now be accumulated to tune backend heuristics from measured outcomes
- repro runs now generate a clean proof report so the current evidence is easy to present
- Colab build/proof scripts now exist so CUDA validation can happen on a real hosted NVIDIA runtime
- dedicated benchmark harness integration is still pending

## Immediate Next Actions
- validate the wired CUDA dense and sparse executors on a real CUDA build host
- validate cuBLAS and cuSPARSE baseline paths on the same CUDA host
- exercise the new Colab proof path and inspect whether hosted GPU variance is acceptable for validation-only use
- start loading planner heuristics from accumulated observation logs instead of using static thresholds only
- extend sparse and dense comparison runs across more dataset sizes and matrix structures
- expand tests from the current runtime-level CLI regression coverage into more dataset and failure-mode coverage
- keep recording benchmark and profile runs only when the raw exports are present in `results/`
