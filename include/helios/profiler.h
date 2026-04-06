#pragma once

#include "helios/dataset_loader.h"

#include <string>

namespace Helios {

struct WorkloadProfile {
    bool is_graph = false;
    bool is_sparse = false;
    bool is_dense = false;
    size_t rows = 0;
    size_t cols = 0;
    size_t nnz = 0;
    size_t bytes_read = 0;
    size_t bytes_written = 0;
    size_t bytes_moved = 0;
    double density = 0.0;
    double arithmetic_intensity = 0.0;
    double estimated_flops = 0.0;
    double average_degree = 0.0;
    size_t max_degree = 0;
    std::string name;
};

class Profiler {
public:
    Profiler() = default;
    ~Profiler() = default;

    WorkloadProfile profile_dense(size_t rows, size_t cols, size_t depth) const;
    WorkloadProfile profile_matrix(const SparseMatrix& matrix) const;
    WorkloadProfile profile_graph(const GraphData& graph) const;
};

} // namespace Helios
