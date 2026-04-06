#include "helios/dataset_loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace Helios {

namespace {

constexpr std::uint64_t kMatrixCacheMagic = 0x48454C494F534D31ULL; // HELIOSM1
constexpr std::uint64_t kGraphCacheMagic = 0x48454C494F534731ULL;  // HELIOSG1

std::string to_lower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

bool contains_case_insensitive(const std::string& text, const std::string& needle) {
    return to_lower(text).find(to_lower(needle)) != std::string::npos;
}

std::uint64_t fnv1a_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return 0;
    }

    std::uint64_t hash = 1469598103934665603ULL;
    char buffer[8192];
    while (input.good()) {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(buffer[index]);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::uint64_t mtime_ticks(const std::string& path) {
    std::error_code error;
    const auto timestamp = std::filesystem::last_write_time(path, error);
    if (error) {
        return 0;
    }
    return static_cast<std::uint64_t>(timestamp.time_since_epoch().count());
}

std::uint64_t file_size_bytes(const std::string& path) {
    std::error_code error;
    return error ? 0 : static_cast<std::uint64_t>(std::filesystem::file_size(path, error));
}

std::filesystem::path derive_processed_root(const std::filesystem::path& input_path) {
    std::filesystem::path absolute_path = std::filesystem::absolute(input_path);
    std::vector<std::filesystem::path> parts;
    for (const auto& part : absolute_path) {
        parts.push_back(part);
    }

    for (size_t index = 0; index + 1 < parts.size(); ++index) {
        if (parts[index] == "data" && parts[index + 1] == "raw") {
            std::filesystem::path root;
            for (size_t root_index = 0; root_index < index; ++root_index) {
                root /= parts[root_index];
            }
            if (root.empty()) {
                root = "/";
            }
            return root / "data" / "processed";
        }
    }

    return absolute_path.parent_path();
}

std::filesystem::path derive_relative_dataset_path(const std::filesystem::path& input_path) {
    std::filesystem::path absolute_path = std::filesystem::absolute(input_path);
    std::vector<std::filesystem::path> parts;
    for (const auto& part : absolute_path) {
        parts.push_back(part);
    }

    for (size_t index = 0; index + 1 < parts.size(); ++index) {
        if (parts[index] == "data" && parts[index + 1] == "raw") {
            std::filesystem::path relative;
            for (size_t rel_index = index + 2; rel_index < parts.size(); ++rel_index) {
                relative /= parts[rel_index];
            }
            return relative;
        }
    }

    return absolute_path.filename();
}

std::filesystem::path derive_cache_path(const std::string& path, const std::string& suffix) {
    const std::filesystem::path input_path(path);
    std::filesystem::path relative = derive_relative_dataset_path(input_path);
    relative += suffix;
    return derive_processed_root(input_path) / "cache" / relative;
}

std::string infer_source_url(const std::string& path) {
    const std::string lower = to_lower(path);
    if (lower.find("data/raw/suitesparse/hb/bcsstk30/bcsstk30.mtx") != std::string::npos) {
        return "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk30.tar.gz";
    }
    if (lower.find("data/raw/suitesparse/hamm/add20/add20.mtx") != std::string::npos) {
        return "https://suitesparse-collection-website.herokuapp.com/MM/Hamm/add20.tar.gz";
    }
    if (lower.find("data/raw/snap/facebook_combined/facebook_combined.txt") != std::string::npos) {
        return "https://snap.stanford.edu/data/facebook_combined.txt.gz";
    }
    if (lower.find("data/raw/snap/ca-grqc/ca-grqc.txt") != std::string::npos) {
        return "https://snap.stanford.edu/data/ca-GrQc.txt.gz";
    }
    return {};
}

void populate_load_info(const std::string& path,
                        const std::string& format,
                        const std::string& cache_path,
                        DatasetLoadInfo* info) {
    if (info == nullptr) {
        return;
    }
    info->input_path = path;
    info->input_format = format;
    info->cache_path = cache_path;
    info->source_url = infer_source_url(path);
    info->input_size_bytes = file_size_bytes(path);
    info->input_mtime_ticks = mtime_ticks(path);
    info->input_checksum_fnv1a = fnv1a_file(path);
}

template <typename T>
bool write_scalar(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(output);
}

template <typename T>
bool read_scalar(std::ifstream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

template <typename T>
bool write_vector(std::ofstream& output, const std::vector<T>& values) {
    const std::uint64_t count = values.size();
    if (!write_scalar(output, count)) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    output.write(reinterpret_cast<const char*>(values.data()), sizeof(T) * values.size());
    return static_cast<bool>(output);
}

template <typename T>
bool read_vector(std::ifstream& input, std::vector<T>& values) {
    std::uint64_t count = 0;
    if (!read_scalar(input, count)) {
        return false;
    }
    values.resize(static_cast<size_t>(count));
    if (count == 0) {
        return true;
    }
    input.read(reinterpret_cast<char*>(values.data()), sizeof(T) * values.size());
    return static_cast<bool>(input);
}

bool write_matrix_cache(const std::string& cache_path,
                        const DatasetLoadInfo& info,
                        const SparseMatrix& matrix) {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path(), error);
    if (error) {
        return false;
    }

    std::ofstream output(cache_path, std::ios::binary);
    if (!output) {
        return false;
    }

    const std::uint64_t field_size = matrix.metadata.field.size();
    const std::uint64_t symmetry_size = matrix.metadata.symmetry.size();
    return write_scalar(output, kMatrixCacheMagic)
        && write_scalar(output, info.input_size_bytes)
        && write_scalar(output, info.input_mtime_ticks)
        && write_scalar(output, info.input_checksum_fnv1a)
        && write_scalar(output, matrix.rows)
        && write_scalar(output, matrix.cols)
        && write_scalar(output, matrix.nnz)
        && write_scalar(output, matrix.metadata.pattern)
        && write_scalar(output, matrix.metadata.symmetric)
        && write_scalar(output, matrix.metadata.skew_symmetric)
        && write_scalar(output, field_size)
        && static_cast<bool>(output.write(matrix.metadata.field.data(), static_cast<std::streamsize>(field_size)))
        && write_scalar(output, symmetry_size)
        && static_cast<bool>(output.write(matrix.metadata.symmetry.data(), static_cast<std::streamsize>(symmetry_size)))
        && write_vector(output, matrix.row_ptr)
        && write_vector(output, matrix.col_idx)
        && write_vector(output, matrix.values);
}

bool read_matrix_cache(const std::string& cache_path,
                       const DatasetLoadInfo& info,
                       SparseMatrix& matrix) {
    std::ifstream input(cache_path, std::ios::binary);
    if (!input) {
        return false;
    }

    std::uint64_t magic = 0;
    std::uint64_t input_size = 0;
    std::uint64_t input_mtime = 0;
    std::uint64_t checksum = 0;
    std::uint64_t field_size = 0;
    std::uint64_t symmetry_size = 0;
    if (!read_scalar(input, magic) || magic != kMatrixCacheMagic
        || !read_scalar(input, input_size) || input_size != info.input_size_bytes
        || !read_scalar(input, input_mtime) || input_mtime != info.input_mtime_ticks
        || !read_scalar(input, checksum) || checksum != info.input_checksum_fnv1a
        || !read_scalar(input, matrix.rows)
        || !read_scalar(input, matrix.cols)
        || !read_scalar(input, matrix.nnz)
        || !read_scalar(input, matrix.metadata.pattern)
        || !read_scalar(input, matrix.metadata.symmetric)
        || !read_scalar(input, matrix.metadata.skew_symmetric)
        || !read_scalar(input, field_size)) {
        return false;
    }

    matrix.metadata.field.assign(static_cast<size_t>(field_size), '\0');
    input.read(matrix.metadata.field.data(), static_cast<std::streamsize>(field_size));
    if (!read_scalar(input, symmetry_size)) {
        return false;
    }
    matrix.metadata.symmetry.assign(static_cast<size_t>(symmetry_size), '\0');
    input.read(matrix.metadata.symmetry.data(), static_cast<std::streamsize>(symmetry_size));

    return static_cast<bool>(input)
        && read_vector(input, matrix.row_ptr)
        && read_vector(input, matrix.col_idx)
        && read_vector(input, matrix.values);
}

bool write_graph_cache(const std::string& cache_path,
                       const DatasetLoadInfo& info,
                       const GraphData& graph) {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path(), error);
    if (error) {
        return false;
    }

    std::ofstream output(cache_path, std::ios::binary);
    if (!output) {
        return false;
    }

    const std::uint64_t edge_pairs = graph.edges.size();
    if (!write_scalar(output, kGraphCacheMagic)
        || !write_scalar(output, info.input_size_bytes)
        || !write_scalar(output, info.input_mtime_ticks)
        || !write_scalar(output, info.input_checksum_fnv1a)
        || !write_scalar(output, graph.node_count)
        || !write_scalar(output, graph.edge_count)
        || !write_scalar(output, graph.stored_edge_count)
        || !write_scalar(output, graph.directed)
        || !write_scalar(output, edge_pairs)) {
        return false;
    }

    for (const auto& edge : graph.edges) {
        if (!write_scalar(output, edge.first) || !write_scalar(output, edge.second)) {
            return false;
        }
    }

    return write_vector(output, graph.row_ptr)
        && write_vector(output, graph.col_idx)
        && write_vector(output, graph.out_degree)
        && write_vector(output, graph.in_degree);
}

bool read_graph_cache(const std::string& cache_path,
                      const DatasetLoadInfo& info,
                      GraphData& graph) {
    std::ifstream input(cache_path, std::ios::binary);
    if (!input) {
        return false;
    }

    std::uint64_t magic = 0;
    std::uint64_t input_size = 0;
    std::uint64_t input_mtime = 0;
    std::uint64_t checksum = 0;
    std::uint64_t edge_pairs = 0;
    if (!read_scalar(input, magic) || magic != kGraphCacheMagic
        || !read_scalar(input, input_size) || input_size != info.input_size_bytes
        || !read_scalar(input, input_mtime) || input_mtime != info.input_mtime_ticks
        || !read_scalar(input, checksum) || checksum != info.input_checksum_fnv1a
        || !read_scalar(input, graph.node_count)
        || !read_scalar(input, graph.edge_count)
        || !read_scalar(input, graph.stored_edge_count)
        || !read_scalar(input, graph.directed)
        || !read_scalar(input, edge_pairs)) {
        return false;
    }

    graph.edges.resize(static_cast<size_t>(edge_pairs));
    for (auto& edge : graph.edges) {
        if (!read_scalar(input, edge.first) || !read_scalar(input, edge.second)) {
            return false;
        }
    }

    return read_vector(input, graph.row_ptr)
        && read_vector(input, graph.col_idx)
        && read_vector(input, graph.out_degree)
        && read_vector(input, graph.in_degree);
}

bool parse_matrix_market_text(std::ifstream& input, SparseMatrix& matrix) {
    std::string line;
    if (!std::getline(input, line)) {
        return false;
    }

    std::istringstream banner_stream(line);
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    if (!(banner_stream >> banner >> object >> format >> field >> symmetry)) {
        return false;
    }

    banner = to_lower(banner);
    object = to_lower(object);
    format = to_lower(format);
    field = to_lower(field);
    symmetry = to_lower(symmetry);

    if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate") {
        return false;
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        return false;
    }
    if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric") {
        return false;
    }

    matrix.metadata.field = field;
    matrix.metadata.symmetry = symmetry;
    matrix.metadata.pattern = (field == "pattern");
    matrix.metadata.symmetric = (symmetry == "symmetric");
    matrix.metadata.skew_symmetric = (symmetry == "skew-symmetric");

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }

        std::istringstream dimensions(line);
        size_t rows = 0;
        size_t cols = 0;
        size_t input_nnz = 0;
        if (!(dimensions >> rows >> cols >> input_nnz)) {
            return false;
        }

        matrix.rows = rows;
        matrix.cols = cols;

        std::vector<std::tuple<size_t, size_t, double>> entries;
        entries.reserve(input_nnz * (matrix.metadata.symmetric || matrix.metadata.skew_symmetric ? 2 : 1));

        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '%') {
                continue;
            }

            std::istringstream entry_stream(line);
            size_t row = 0;
            size_t col = 0;
            double value = 1.0;
            if (!(entry_stream >> row >> col)) {
                return false;
            }
            if (!matrix.metadata.pattern && !(entry_stream >> value)) {
                return false;
            }
            if (row == 0 || col == 0 || row > rows || col > cols) {
                return false;
            }

            const size_t row_index = row - 1;
            const size_t col_index = col - 1;
            entries.emplace_back(row_index, col_index, value);

            if ((matrix.metadata.symmetric || matrix.metadata.skew_symmetric) && row_index != col_index) {
                const double mirrored_value = matrix.metadata.skew_symmetric ? -value : value;
                entries.emplace_back(col_index, row_index, mirrored_value);
            }
        }

        std::sort(
            entries.begin(),
            entries.end(),
            [](const auto& lhs, const auto& rhs) {
                if (std::get<0>(lhs) != std::get<0>(rhs)) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                }
                return std::get<1>(lhs) < std::get<1>(rhs);
            });

        matrix.nnz = entries.size();
        matrix.row_ptr.assign(rows + 1, 0);
        matrix.col_idx.resize(entries.size());
        matrix.values.resize(entries.size());

        for (const auto& entry : entries) {
            ++matrix.row_ptr[std::get<0>(entry) + 1];
        }

        for (size_t row_index = 0; row_index < rows; ++row_index) {
            matrix.row_ptr[row_index + 1] += matrix.row_ptr[row_index];
        }

        for (size_t index = 0; index < entries.size(); ++index) {
            matrix.col_idx[index] = std::get<1>(entries[index]);
            matrix.values[index] = std::get<2>(entries[index]);
        }
        return true;
    }

    return false;
}

bool parse_snap_text(std::ifstream& input, GraphData& graph) {
    std::string line;
    size_t max_node_id = 0;
    bool saw_edge = false;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            if (contains_case_insensitive(line, "undirected")) {
                graph.directed = false;
            } else if (contains_case_insensitive(line, "directed")) {
                graph.directed = true;
            }
            continue;
        }

        std::istringstream edge_stream(line);
        size_t source = 0;
        size_t destination = 0;
        if (!(edge_stream >> source >> destination)) {
            return false;
        }

        graph.edges.emplace_back(source, destination);
        max_node_id = std::max(max_node_id, std::max(source, destination));
        saw_edge = true;
    }

    if (!saw_edge) {
        return false;
    }

    graph.edge_count = graph.edges.size();
    graph.node_count = max_node_id + 1;
    return true;
}

} // namespace

bool DatasetLoader::load_matrix_market(const std::string& path, SparseMatrix& matrix, DatasetLoadInfo* info) const {
    matrix = {};
    const std::string cache_path = derive_cache_path(path, ".csr.bin").string();
    populate_load_info(path, "matrix_market", cache_path, info);

    if (info != nullptr && read_matrix_cache(cache_path, *info, matrix)) {
        info->cache_hit = true;
        return true;
    }

    std::ifstream input(path);
    if (!input) {
        return false;
    }
    if (!parse_matrix_market_text(input, matrix)) {
        return false;
    }

    if (info != nullptr) {
        info->cache_hit = false;
        write_matrix_cache(cache_path, *info, matrix);
    }
    return true;
}

bool DatasetLoader::load_snap_edge_list(const std::string& path, GraphData& graph, DatasetLoadInfo* info) const {
    graph = {};
    const std::string cache_path = derive_cache_path(path, ".graph.bin").string();
    populate_load_info(path, "snap_edge_list", cache_path, info);

    if (info != nullptr && read_graph_cache(cache_path, *info, graph)) {
        info->cache_hit = true;
        return true;
    }

    std::ifstream input(path);
    if (!input) {
        return false;
    }
    if (!parse_snap_text(input, graph)) {
        return false;
    }

    rebuild_graph_storage(graph, graph.directed);

    if (info != nullptr) {
        info->cache_hit = false;
        write_graph_cache(cache_path, *info, graph);
    }
    return true;
}

void DatasetLoader::rebuild_graph_storage(GraphData& graph, bool directed) const {
    graph.directed = directed;
    graph.out_degree.assign(graph.node_count, 0);
    graph.in_degree.assign(graph.node_count, 0);

    std::vector<std::pair<size_t, size_t>> expanded_edges;
    expanded_edges.reserve(graph.directed ? graph.edges.size() : graph.edges.size() * 2);
    for (const auto& edge : graph.edges) {
        expanded_edges.push_back(edge);
        ++graph.out_degree[edge.first];
        ++graph.in_degree[edge.second];

        if (!graph.directed && edge.first != edge.second) {
            expanded_edges.emplace_back(edge.second, edge.first);
            ++graph.out_degree[edge.second];
            ++graph.in_degree[edge.first];
        }
    }

    std::sort(
        expanded_edges.begin(),
        expanded_edges.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

    graph.stored_edge_count = expanded_edges.size();
    graph.row_ptr.assign(graph.node_count + 1, 0);
    graph.col_idx.resize(expanded_edges.size());

    for (const auto& edge : expanded_edges) {
        ++graph.row_ptr[edge.first + 1];
    }

    for (size_t node = 0; node < graph.node_count; ++node) {
        graph.row_ptr[node + 1] += graph.row_ptr[node];
    }

    for (size_t index = 0; index < expanded_edges.size(); ++index) {
        graph.col_idx[index] = expanded_edges[index].second;
    }
}

} // namespace Helios
