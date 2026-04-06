#include "helios/profiler.h"

#include <algorithm>

namespace Helios {

WorkloadProfile Profiler::profile_dense(size_t rows, size_t cols, size_t depth) const {
    WorkloadProfile profile;
    profile.is_dense = true;
    profile.rows = rows;
    profile.cols = cols;
    profile.nnz = rows * cols;
    profile.estimated_flops = static_cast<double>(2 * rows) * static_cast<double>(cols) * static_cast<double>(depth);
    profile.bytes_read = (rows * depth + depth * cols) * sizeof(double);
    profile.bytes_written = rows * cols * sizeof(double);
    profile.bytes_moved = profile.bytes_read + profile.bytes_written;
    profile.density = 1.0;
    profile.arithmetic_intensity =
        profile.bytes_moved == 0 ? 0.0 : profile.estimated_flops / static_cast<double>(profile.bytes_moved);
    return profile;
}

WorkloadProfile Profiler::profile_matrix(const SparseMatrix& matrix) const {
    WorkloadProfile profile;
    profile.is_sparse = true;
    profile.rows = matrix.rows;
    profile.cols = matrix.cols;
    profile.nnz = matrix.nnz;

    const double total_slots = static_cast<double>(matrix.rows) * static_cast<double>(matrix.cols);
    profile.density = total_slots == 0.0 ? 0.0 : static_cast<double>(matrix.nnz) / total_slots;
    profile.is_dense = profile.density >= 0.2;

    profile.estimated_flops = 2.0 * static_cast<double>(matrix.nnz);
    profile.bytes_read = matrix.row_ptr.size() * sizeof(size_t)
        + matrix.col_idx.size() * sizeof(size_t)
        + matrix.values.size() * sizeof(double)
        + matrix.cols * sizeof(double);
    profile.bytes_written = matrix.rows * sizeof(double);
    profile.bytes_moved = profile.bytes_read + profile.bytes_written;
    profile.arithmetic_intensity =
        profile.bytes_moved == 0 ? 0.0 : profile.estimated_flops / static_cast<double>(profile.bytes_moved);

    size_t max_degree = 0;
    for (size_t row = 0; row < matrix.rows; ++row) {
        max_degree = std::max(max_degree, matrix.row_ptr[row + 1] - matrix.row_ptr[row]);
    }
    profile.max_degree = max_degree;
    profile.average_degree = matrix.rows == 0 ? 0.0 : static_cast<double>(matrix.nnz) / static_cast<double>(matrix.rows);
    return profile;
}

WorkloadProfile Profiler::profile_graph(const GraphData& graph) const {
    WorkloadProfile profile;
    profile.is_graph = true;
    profile.is_sparse = true;
    profile.rows = graph.node_count;
    profile.cols = graph.node_count;
    profile.nnz = graph.stored_edge_count;

    const double total_slots = static_cast<double>(graph.node_count) * static_cast<double>(graph.node_count);
    profile.density = total_slots == 0.0 ? 0.0 : static_cast<double>(graph.stored_edge_count) / total_slots;
    profile.estimated_flops = static_cast<double>(graph.stored_edge_count);
    profile.bytes_read = graph.row_ptr.size() * sizeof(size_t) + graph.col_idx.size() * sizeof(size_t);
    profile.bytes_written = graph.node_count * sizeof(double);
    profile.bytes_moved = profile.bytes_read + profile.bytes_written;
    profile.arithmetic_intensity =
        profile.bytes_moved == 0 ? 0.0 : profile.estimated_flops / static_cast<double>(profile.bytes_moved);

    size_t max_degree = 0;
    for (size_t node = 0; node < graph.node_count; ++node) {
        max_degree = std::max(max_degree, graph.row_ptr[node + 1] - graph.row_ptr[node]);
    }
    profile.max_degree = max_degree;
    profile.average_degree =
        graph.node_count == 0 ? 0.0 : static_cast<double>(graph.stored_edge_count) / static_cast<double>(graph.node_count);
    return profile;
}

} // namespace Helios
