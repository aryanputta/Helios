#include "helios/planner.h"

namespace Helios {

Planner::Planner() = default;

Planner::~Planner() = default;

Planner::Decision Planner::select_strategy(const WorkloadProfile& profile,
                                           const std::string& requested_path,
                                           bool gpu_available) const {
    if (requested_path == "cpu") {
        if (profile.is_graph) {
            return {Strategy::GraphCPU, "CPU path forced by CLI option."};
        }
        if (profile.is_sparse) {
            return {Strategy::SparseCPU, "CPU path forced by CLI option."};
        }
        return {Strategy::DenseCPU, "CPU path forced by CLI option."};
    }

    if (requested_path == "gpu") {
        if (profile.is_graph) {
            return {
                Strategy::GraphGPU,
                gpu_available ? "GPU path forced by CLI option." : "GPU path was forced, but CUDA is not available in this build/runtime."
            };
        }
        if (profile.is_sparse) {
            return {
                Strategy::SparseGPU,
                gpu_available ? "GPU path forced by CLI option." : "GPU path was forced, but CUDA is not available in this build/runtime."
            };
        }
        return {
            Strategy::DenseGPU,
            gpu_available ? "GPU path forced by CLI option." : "GPU path was forced, but CUDA is not available in this build/runtime."
        };
    }

    if (profile.is_graph) {
        return {Strategy::GraphCPU, "Graph workloads currently run through the validated CPU reference path."};
    }

    if (profile.is_sparse) {
        if (gpu_available && profile.rows >= 4096 && profile.bytes_moved >= (32ULL * 1024ULL * 1024ULL) && profile.density >= 0.001) {
            return {Strategy::SparseGPU, "Sparse workload is large enough to justify trying the CUDA CSR path."};
        }
        if (!gpu_available) {
            return {Strategy::SparseCPU, "CUDA is unavailable, so sparse workloads stay on CPU baselines in this build."};
        }
        return {Strategy::SparseCPU, "Sparse workload is too small or too irregular to justify GPU transfer overhead."};
    }

    if (gpu_available && profile.estimated_flops >= 1.0e8) {
        return {Strategy::DenseGPU, "Dense workload is large enough to justify the CUDA tiled GEMM path."};
    }
    if (!gpu_available) {
        return {Strategy::DenseCPU, "CUDA is unavailable, so dense workloads stay on CPU baselines in this build."};
    }

    return {Strategy::DenseCPU, "Dense workload is small enough that the CPU baselines are the safer current option."};
}

const char* Planner::to_string(Strategy strategy) {
    switch (strategy) {
    case Strategy::Auto:
        return "auto";
    case Strategy::DenseGPU:
        return "dense_gpu";
    case Strategy::DenseCPU:
        return "dense_cpu";
    case Strategy::SparseGPU:
        return "sparse_gpu";
    case Strategy::SparseCPU:
        return "sparse_cpu";
    case Strategy::GraphCPU:
        return "graph_cpu";
    case Strategy::GraphGPU:
        return "graph_gpu";
    }

    return "unknown";
}

} // namespace Helios
