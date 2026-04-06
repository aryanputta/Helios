#include "helios/cpu_kernels.h"
#include "helios/dataset_loader.h"
#include "helios/gpu_kernels.h"
#include "helios/profiler.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {

std::filesystem::path write_temp_file(const std::string& name, const std::string& contents) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream output(path);
    assert(output.good());
    output << contents;
    output.close();
    return path;
}

void assert_vectors_close(const std::vector<double>& lhs,
                          const std::vector<double>& rhs,
                          double tolerance = 1.0e-9) {
    assert(lhs.size() == rhs.size());
    for (size_t index = 0; index < lhs.size(); ++index) {
        assert(std::abs(lhs[index] - rhs[index]) <= tolerance);
    }
}

void test_general_matrix_market() {
    const auto path = write_temp_file(
        "helios_test_general.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "% comment\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "2 3 2.0\n"
        "3 1 3.0\n"
        "3 3 4.0\n");

    Helios::DatasetLoader loader;
    Helios::SparseMatrix matrix;
    Helios::DatasetLoadInfo load_info;
    const bool loaded = loader.load_matrix_market(path.string(), matrix, &load_info);
    assert(loaded);
    assert(!load_info.input_format.empty());
    assert(!load_info.cache_path.empty());
    assert(matrix.rows == 3);
    assert(matrix.cols == 3);
    assert(matrix.nnz == 4);
    assert(matrix.metadata.field == "real");
    assert(matrix.metadata.symmetry == "general");
    assert(matrix.row_ptr == std::vector<size_t>({0, 1, 2, 4}));
    assert(matrix.col_idx == std::vector<size_t>({0, 2, 0, 2}));
    assert(matrix.values == std::vector<double>({1.0, 2.0, 3.0, 4.0}));

    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y;
    Helios::cpu::sparse_matvec_reference(matrix, x, y);
    assert(y == std::vector<double>({1.0, 6.0, 15.0}));

    const Helios::Profiler profiler;
    const Helios::WorkloadProfile profile = profiler.profile_matrix(matrix);
    assert(profile.is_sparse);
    assert(profile.max_degree == 2);

    Helios::SparseMatrix cached_matrix;
    Helios::DatasetLoadInfo cached_info;
    const bool cached_loaded = loader.load_matrix_market(path.string(), cached_matrix, &cached_info);
    assert(cached_loaded);
    assert(cached_info.cache_hit);
    assert(cached_matrix.row_ptr == matrix.row_ptr);
    assert(cached_matrix.col_idx == matrix.col_idx);
    assert(cached_matrix.values == matrix.values);

    std::filesystem::remove(path);
}

void test_symmetric_pattern_matrix_market() {
    const auto path = write_temp_file(
        "helios_test_symmetric_pattern.mtx",
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "3 3 2\n"
        "1 2\n"
        "3 3\n");

    Helios::DatasetLoader loader;
    Helios::SparseMatrix matrix;
    const bool loaded = loader.load_matrix_market(path.string(), matrix);
    assert(loaded);
    assert(matrix.nnz == 3);
    assert(matrix.metadata.pattern);
    assert(matrix.metadata.symmetric);
    assert(matrix.row_ptr == std::vector<size_t>({0, 1, 2, 3}));
    assert(matrix.col_idx == std::vector<size_t>({1, 0, 2}));
    assert(matrix.values == std::vector<double>({1.0, 1.0, 1.0}));

    std::filesystem::remove(path);
}

void test_snap_loader_and_graph_kernels() {
    const auto path = write_temp_file(
        "helios_test_graph.txt",
        "# Undirected graph\n"
        "# Nodes: 3 Edges: 2\n"
        "0 1\n"
        "1 2\n");

    Helios::DatasetLoader loader;
    Helios::GraphData graph;
    Helios::DatasetLoadInfo load_info;
    const bool loaded = loader.load_snap_edge_list(path.string(), graph, &load_info);
    assert(loaded);
    assert(!load_info.cache_path.empty());
    assert(!graph.directed);
    assert(graph.node_count == 3);
    assert(graph.edge_count == 2);
    assert(graph.stored_edge_count == 4);
    assert(graph.row_ptr == std::vector<size_t>({0, 1, 3, 4}));
    assert(graph.col_idx == std::vector<size_t>({1, 0, 2, 1}));

    std::vector<int> distances;
    Helios::cpu::bfs_reference(graph, 0, distances);
    assert(distances == std::vector<int>({0, 1, 2}));

    std::vector<double> ranks;
    Helios::cpu::pagerank_reference(graph, 10, 0.85, ranks);
    const double rank_sum = std::accumulate(ranks.begin(), ranks.end(), 0.0);
    assert(rank_sum > 0.99 && rank_sum < 1.01);

    const Helios::Profiler profiler;
    const Helios::WorkloadProfile profile = profiler.profile_graph(graph);
    assert(profile.is_graph);
    assert(profile.average_degree > 1.0);

    Helios::GraphData cached_graph;
    Helios::DatasetLoadInfo cached_info;
    const bool cached_loaded = loader.load_snap_edge_list(path.string(), cached_graph, &cached_info);
    assert(cached_loaded);
    assert(cached_info.cache_hit);
    assert(cached_graph.row_ptr == graph.row_ptr);
    assert(cached_graph.col_idx == graph.col_idx);

    std::filesystem::remove(path);
}

void test_graph_direction_override() {
    const auto path = write_temp_file(
        "helios_test_graph_override.txt",
        "0 1\n"
        "1 2\n");

    Helios::DatasetLoader loader;
    Helios::GraphData graph;
    const bool loaded = loader.load_snap_edge_list(path.string(), graph);
    assert(loaded);
    assert(graph.directed);
    assert(graph.stored_edge_count == 2);

    loader.rebuild_graph_storage(graph, false);
    assert(!graph.directed);
    assert(graph.stored_edge_count == 4);
    assert(graph.row_ptr == std::vector<size_t>({0, 1, 3, 4}));

    std::filesystem::remove(path);
}

void test_dense_cpu_backends() {
    std::vector<double> a = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    std::vector<double> b = {
        2.0, 0.0, 1.0, 3.0,
        1.0, 2.0, 0.0, 1.0,
        4.0, 1.0, 2.0, 0.0,
        0.0, 3.0, 1.0, 2.0
    };

    std::vector<double> scalar;
    Helios::cpu::dense_matmul_reference(a, b, scalar, 3, 4, 4);

    std::vector<double> threaded;
    const bool threaded_ok = Helios::cpu::dense_matmul(a, b, threaded, 3, 4, 4, Helios::cpu::Backend::Threaded, 2);
    assert(threaded_ok);
    assert_vectors_close(scalar, threaded);

    if (Helios::cpu::backend_available(Helios::cpu::Backend::Avx2)) {
        std::vector<double> avx2;
        const bool avx2_ok = Helios::cpu::dense_matmul(a, b, avx2, 3, 4, 4, Helios::cpu::Backend::Avx2);
        assert(avx2_ok);
        assert_vectors_close(scalar, avx2);
    }
}

void test_sparse_cpu_backends() {
    Helios::SparseMatrix matrix;
    matrix.rows = 4;
    matrix.cols = 4;
    matrix.nnz = 8;
    matrix.row_ptr = {0, 2, 4, 6, 8};
    matrix.col_idx = {0, 3, 1, 2, 0, 2, 1, 3};
    matrix.values = {1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 5.0, 6.0};

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> scalar;
    Helios::cpu::sparse_matvec_reference(matrix, x, scalar);

    std::vector<double> threaded;
    const bool threaded_ok = Helios::cpu::sparse_matvec(matrix, x, threaded, Helios::cpu::Backend::Threaded, 2);
    assert(threaded_ok);
    assert_vectors_close(scalar, threaded);

    if (Helios::cpu::backend_available(Helios::cpu::Backend::Avx2)) {
        std::vector<double> avx2;
        const bool avx2_ok = Helios::cpu::sparse_matvec(matrix, x, avx2, Helios::cpu::Backend::Avx2);
        assert(avx2_ok);
        assert_vectors_close(scalar, avx2);
    }
}

void test_gpu_capability_stub_or_runtime() {
    const Helios::gpu::CapabilityInfo gpu_caps = Helios::gpu::capabilities();
    if (!gpu_caps.compiled_with_cuda) {
        assert(!gpu_caps.runtime_available);
        assert(!gpu_caps.reason.empty());
    }
}

} // namespace

int main() {
    test_general_matrix_market();
    test_symmetric_pattern_matrix_market();
    test_snap_loader_and_graph_kernels();
    test_graph_direction_override();
    test_dense_cpu_backends();
    test_sparse_cpu_backends();
    test_gpu_capability_stub_or_runtime();
    std::cout << "Dataset, profiler, CPU backend, and graph kernel tests passed.\n";
    return 0;
}
