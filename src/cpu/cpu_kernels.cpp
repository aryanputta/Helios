#include "helios/cpu_kernels.h"

#include <algorithm>
#include <queue>
#include <thread>

namespace Helios {
namespace cpu {

namespace detail {

#if !defined(HELIOS_HAVE_X86_64)
void dense_matmul_avx2_impl(const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            size_t M,
                            size_t N,
                            size_t K) {
    dense_matmul_reference(A, B, C, M, N, K);
}

void sparse_matvec_avx2_impl(const SparseMatrix& matrix,
                             const std::vector<double>& x,
                             std::vector<double>& y) {
    sparse_matvec_reference(matrix, x, y);
}
#else
void dense_matmul_avx2_impl(const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            size_t M,
                            size_t N,
                            size_t K);

void sparse_matvec_avx2_impl(const SparseMatrix& matrix,
                             const std::vector<double>& x,
                             std::vector<double>& y);
#endif

} // namespace detail

namespace {

size_t normalized_thread_count(size_t requested_threads) {
    if (requested_threads > 0) {
        return requested_threads;
    }

    const size_t detected_threads = static_cast<size_t>(std::thread::hardware_concurrency());
    return detected_threads == 0 ? 1 : detected_threads;
}

void dense_matmul_rows_scalar(const std::vector<double>& A,
                              const std::vector<double>& B,
                              std::vector<double>& C,
                              size_t M,
                              size_t N,
                              size_t K,
                              size_t row_begin,
                              size_t row_end) {
    (void)M;
    for (size_t row = row_begin; row < row_end; ++row) {
        for (size_t inner = 0; inner < K; ++inner) {
            const double a_value = A[row * K + inner];
            for (size_t col = 0; col < N; ++col) {
                C[row * N + col] += a_value * B[inner * N + col];
            }
        }
    }
}

void sparse_matvec_rows_scalar(const SparseMatrix& matrix,
                               const std::vector<double>& x,
                               std::vector<double>& y,
                               size_t row_begin,
                               size_t row_end) {
    for (size_t row = row_begin; row < row_end; ++row) {
        const size_t row_start = matrix.row_ptr[row];
        const size_t row_limit = matrix.row_ptr[row + 1];
        double sum = 0.0;
        for (size_t index = row_start; index < row_limit; ++index) {
            sum += matrix.values[index] * x[matrix.col_idx[index]];
        }
        y[row] = sum;
    }
}

bool runtime_avx2_support() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
#else
    return false;
#endif
}

} // namespace

CapabilityInfo capabilities() {
    CapabilityInfo info;
    info.avx2_available = runtime_avx2_support();
    info.hardware_threads = normalized_thread_count(0);
    return info;
}

bool backend_available(Backend backend) {
    switch (backend) {
    case Backend::Scalar:
        return true;
    case Backend::Avx2:
        return capabilities().avx2_available;
    case Backend::Threaded:
        return capabilities().hardware_threads > 1;
    }
    return false;
}

const char* to_string(Backend backend) {
    switch (backend) {
    case Backend::Scalar:
        return "scalar";
    case Backend::Avx2:
        return "avx2";
    case Backend::Threaded:
        return "threaded";
    }
    return "unknown";
}

void dense_matmul_reference(const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            size_t M,
                            size_t N,
                            size_t K) {
    C.assign(M * N, 0.0);
    dense_matmul_rows_scalar(A, B, C, M, N, K, 0, M);
}

void sparse_matvec_reference(const SparseMatrix& matrix,
                             const std::vector<double>& x,
                             std::vector<double>& y) {
    y.assign(matrix.rows, 0.0);
    sparse_matvec_rows_scalar(matrix, x, y, 0, matrix.rows);
}

bool dense_matmul(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  size_t M,
                  size_t N,
                  size_t K,
                  Backend backend,
                  size_t thread_count) {
    switch (backend) {
    case Backend::Scalar:
        dense_matmul_reference(A, B, C, M, N, K);
        return true;

    case Backend::Avx2:
        if (!backend_available(Backend::Avx2)) {
            return false;
        }
        C.assign(M * N, 0.0);
        detail::dense_matmul_avx2_impl(A, B, C, M, N, K);
        return true;

    case Backend::Threaded: {
        C.assign(M * N, 0.0);
        const size_t threads = std::min(normalized_thread_count(thread_count), std::max<size_t>(1, M));
        if (threads <= 1) {
            dense_matmul_rows_scalar(A, B, C, M, N, K, 0, M);
            return true;
        }

        std::vector<std::thread> workers;
        workers.reserve(threads);

        size_t row_begin = 0;
        const size_t rows_per_thread = M / threads;
        const size_t row_remainder = M % threads;
        for (size_t worker = 0; worker < threads; ++worker) {
            const size_t chunk = rows_per_thread + (worker < row_remainder ? 1 : 0);
            const size_t row_end = row_begin + chunk;
            workers.emplace_back(
                [&A, &B, &C, M, N, K, row_begin, row_end]() {
                    dense_matmul_rows_scalar(A, B, C, M, N, K, row_begin, row_end);
                });
            row_begin = row_end;
        }

        for (auto& worker : workers) {
            worker.join();
        }
        return true;
    }
    }

    return false;
}

bool sparse_matvec(const SparseMatrix& matrix,
                   const std::vector<double>& x,
                   std::vector<double>& y,
                   Backend backend,
                   size_t thread_count) {
    switch (backend) {
    case Backend::Scalar:
        sparse_matvec_reference(matrix, x, y);
        return true;

    case Backend::Avx2:
        if (!backend_available(Backend::Avx2)) {
            return false;
        }
        y.assign(matrix.rows, 0.0);
        detail::sparse_matvec_avx2_impl(matrix, x, y);
        return true;

    case Backend::Threaded: {
        y.assign(matrix.rows, 0.0);
        const size_t threads = std::min(normalized_thread_count(thread_count), std::max<size_t>(1, matrix.rows));
        if (threads <= 1) {
            sparse_matvec_rows_scalar(matrix, x, y, 0, matrix.rows);
            return true;
        }

        std::vector<std::thread> workers;
        workers.reserve(threads);

        size_t row_begin = 0;
        const size_t rows_per_thread = matrix.rows / threads;
        const size_t row_remainder = matrix.rows % threads;
        for (size_t worker = 0; worker < threads; ++worker) {
            const size_t chunk = rows_per_thread + (worker < row_remainder ? 1 : 0);
            const size_t row_end = row_begin + chunk;
            workers.emplace_back(
                [&matrix, &x, &y, row_begin, row_end]() {
                    sparse_matvec_rows_scalar(matrix, x, y, row_begin, row_end);
                });
            row_begin = row_end;
        }

        for (auto& worker : workers) {
            worker.join();
        }
        return true;
    }
    }

    return false;
}

void bfs_reference(const GraphData& graph,
                   size_t source,
                   std::vector<int>& distances) {
    distances.assign(graph.node_count, -1);
    if (source >= graph.node_count) {
        return;
    }

    std::queue<size_t> frontier;
    distances[source] = 0;
    frontier.push(source);

    while (!frontier.empty()) {
        const size_t node = frontier.front();
        frontier.pop();

        for (size_t edge = graph.row_ptr[node]; edge < graph.row_ptr[node + 1]; ++edge) {
            const size_t neighbor = graph.col_idx[edge];
            if (distances[neighbor] != -1) {
                continue;
            }
            distances[neighbor] = distances[node] + 1;
            frontier.push(neighbor);
        }
    }
}

void pagerank_reference(const GraphData& graph,
                        size_t iterations,
                        double damping_factor,
                        std::vector<double>& ranks) {
    if (graph.node_count == 0) {
        ranks.clear();
        return;
    }

    const double base_rank = 1.0 / static_cast<double>(graph.node_count);
    ranks.assign(graph.node_count, base_rank);
    std::vector<double> next_ranks(graph.node_count, 0.0);

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::fill(next_ranks.begin(), next_ranks.end(), (1.0 - damping_factor) / static_cast<double>(graph.node_count));

        double dangling_mass = 0.0;
        for (size_t node = 0; node < graph.node_count; ++node) {
            if (graph.out_degree[node] == 0) {
                dangling_mass += ranks[node];
                continue;
            }

            const double contribution = damping_factor * ranks[node] / static_cast<double>(graph.out_degree[node]);
            for (size_t edge = graph.row_ptr[node]; edge < graph.row_ptr[node + 1]; ++edge) {
                next_ranks[graph.col_idx[edge]] += contribution;
            }
        }

        const double dangling_contribution =
            damping_factor * dangling_mass / static_cast<double>(graph.node_count);
        for (double& rank : next_ranks) {
            rank += dangling_contribution;
        }

        ranks.swap(next_ranks);
    }
}

} // namespace cpu
} // namespace Helios
