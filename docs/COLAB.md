# Helios on Colab

Use Colab for CUDA validation and quick proof runs. It is good for proving that Helios builds and executes against a real NVIDIA GPU, but it is not the strongest place for final benchmark claims because hosted Colab GPU type and availability can vary.

## Recommended Usage
- use hosted Colab GPU to validate `cuda` and `vendor` paths
- use a controlled CUDA VM or your own NVIDIA box for final benchmark numbers you want to present as the strongest proof

## Quick Start

In a Colab notebook, use a GPU runtime and run:

```bash
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
!bash scripts/run_colab_proof.sh
```

That will:
- install build prerequisites
- print `nvidia-smi` and `nvcc --version` if available
- configure and build Helios
- fetch the real datasets
- run the current repro suite
- export a sanitized `public_artifacts/` bundle that is safer to publish

## Manual Commands

If you want a more manual flow:

```bash
!bash scripts/setup_colab_cuda.sh
!./build/helios bench sparse \
  --matrix data/raw/suitesparse/HB/bcsstk30/bcsstk30.mtx \
  --compare-baselines \
  --threads 4 \
  --warmup 2 \
  --trials 5 \
  --json results/json/bcsstk30_spmv_colab.json
```

Then export a public bundle:

```bash
!python3 scripts/export_public_bundle.py --repo-root . --output-dir public_artifacts
```

## What To Look For
- `cuda_compiled=true`
- `cuda_runtime_available=true`
- `cublas_available=true` and `cusparse_available=true` when those libraries are present
- compare-baseline runs that include `cuda` and `vendor` in the exported metrics

## Publish Safely
Do not push `build/`, `results/`, or raw `data/` outputs directly. Use the sanitized `public_artifacts/` output when you want to share result bundles publicly.

