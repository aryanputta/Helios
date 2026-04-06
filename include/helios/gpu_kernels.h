#pragma once

#include "helios/dataset_loader.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Helios {

namespace gpu {

struct CapabilityInfo {
    bool compiled_with_cuda = false;
    bool runtime_available = false;
    bool cublas_available = false;
    bool cusparse_available = false;
    int device_count = 0;
    std::string reason;
};

CapabilityInfo capabilities();

bool dense_matmul(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  size_t M,
                  size_t N,
                  size_t K,
                  std::string* error = nullptr);

bool sparse_matvec(const SparseMatrix& matrix,
                   const std::vector<double>& x,
                   std::vector<double>& y,
                   std::string* error = nullptr);

bool dense_matmul_vendor(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         size_t M,
                         size_t N,
                         size_t K,
                         std::string* error = nullptr);

bool sparse_matvec_vendor(const SparseMatrix& matrix,
                          const std::vector<double>& x,
                          std::vector<double>& y,
                          std::string* error = nullptr);

} // namespace gpu

} // namespace Helios
