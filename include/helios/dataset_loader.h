#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Helios {

struct MatrixMarketMetadata {
    std::string field = "real";
    std::string symmetry = "general";
    bool pattern = false;
    bool symmetric = false;
    bool skew_symmetric = false;
};

struct SparseMatrix {
    size_t rows = 0;
    size_t cols = 0;
    size_t nnz = 0;
    MatrixMarketMetadata metadata;
    std::vector<size_t> row_ptr;
    std::vector<size_t> col_idx;
    std::vector<double> values;
};

struct GraphData {
    size_t node_count = 0;
    size_t edge_count = 0;
    size_t stored_edge_count = 0;
    bool directed = true;
    std::vector<std::pair<size_t, size_t>> edges;
    std::vector<size_t> row_ptr;
    std::vector<size_t> col_idx;
    std::vector<size_t> out_degree;
    std::vector<size_t> in_degree;
};

struct DatasetLoadInfo {
    bool cache_hit = false;
    std::string cache_path;
    std::string source_url;
    std::string input_path;
    std::string input_format;
    std::uint64_t input_checksum_fnv1a = 0;
    std::uint64_t input_mtime_ticks = 0;
    std::uint64_t input_size_bytes = 0;
};

class DatasetLoader {
public:
    DatasetLoader() = default;
    ~DatasetLoader() = default;

    bool load_matrix_market(const std::string& path, SparseMatrix& matrix, DatasetLoadInfo* info = nullptr) const;
    bool load_snap_edge_list(const std::string& path, GraphData& graph, DatasetLoadInfo* info = nullptr) const;
    void rebuild_graph_storage(GraphData& graph, bool directed) const;
};

} // namespace Helios
