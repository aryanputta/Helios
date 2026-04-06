#pragma once

#include "helios/dataset_loader.h"

#include <cstddef>
#include <vector>

namespace Helios {

namespace cpu {

enum class Backend {
    Scalar,
    Avx2,
    Threaded,
};

struct CapabilityInfo {
    bool avx2_available = false;
    size_t hardware_threads = 1;
};

CapabilityInfo capabilities();
bool backend_available(Backend backend);
const char* to_string(Backend backend);

void dense_matmul_reference(const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            size_t M,
                            size_t N,
                            size_t K);

void sparse_matvec_reference(const SparseMatrix& matrix,
                             const std::vector<double>& x,
                             std::vector<double>& y);

bool dense_matmul(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  size_t M,
                  size_t N,
                  size_t K,
                  Backend backend,
                  size_t thread_count = 0);

bool sparse_matvec(const SparseMatrix& matrix,
                   const std::vector<double>& x,
                   std::vector<double>& y,
                   Backend backend,
                   size_t thread_count = 0);

void bfs_reference(const GraphData& graph,
                   size_t source,
                   std::vector<int>& distances);

void pagerank_reference(const GraphData& graph,
                        size_t iterations,
                        double damping_factor,
                        std::vector<double>& ranks);

} // namespace cpu

} // namespace Helios
