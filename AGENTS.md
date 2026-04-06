# Helios Agent Roles

This file defines the initial agent roles for the Helios Compute Runtime project.
Each agent owns a separate area of the project and keeps iterating to make it stronger.

## Data Agent
Responsibilities:
- download and validate real datasets from SuiteSparse and SNAP
- implement dataset loaders for Matrix Market and edge list formats
- maintain data provenance and reproducibility notes
- propose dataset selections for sparse and graph benchmarks

## Kernel Agent
Responsibilities:
- implement CPU scalar and AVX2 dense kernels
- implement CUDA tiled GEMM and fused kernels
- implement sparse CSR and graph kernels
- add correctness harnesses comparing against reference outputs

## Planner Agent
Responsibilities:
- design workload profiling features
- build execution heuristics for CPU vs GPU vs fused kernels
- route sparse, dense, and graph workloads correctly
- tune path selection and expose planner decisions in CLI output

## Benchmark Agent
Responsibilities:
- implement reproducible benchmark harnesses using NVBench and Google Benchmark
- add warmup, repeated trials, median/p95 reporting
- export CSV and JSON result summaries
- compare against baselines such as scalar CPU and cuBLAS/cuSPARSE where applicable

## Documentation Agent
Responsibilities:
- write README sections and reproducibility guides
- document CLI usage, dataset pipelines, and profiler outputs
- capture project decisions and final results
- ensure claims are backed by real runs and data

## Coordination
- Agents should discuss improvements openly in this workspace via docs and comments.
- Each agent should add tasks, findings, and planned refinements to `WORKPLAN.md`.
- Every benchmark or performance claim must include raw CSV/JSON export and dataset provenance.
