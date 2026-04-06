#include "helios/gpu_kernels.h"

namespace Helios {
namespace gpu {

CapabilityInfo capabilities() {
    CapabilityInfo info;
    info.compiled_with_cuda = false;
    info.runtime_available = false;
    info.cublas_available = false;
    info.cusparse_available = false;
    info.device_count = 0;
    info.reason = "Helios was built without a CUDA compiler on this machine.";
    return info;
}

bool dense_matmul(const std::vector<double>&,
                  const std::vector<double>&,
                  std::vector<double>&,
                  size_t,
                  size_t,
                  size_t,
                  std::string* error) {
    if (error != nullptr) {
        *error = capabilities().reason;
    }
    return false;
}

bool sparse_matvec(const SparseMatrix&,
                   const std::vector<double>&,
                   std::vector<double>&,
                   std::string* error) {
    if (error != nullptr) {
        *error = capabilities().reason;
    }
    return false;
}

bool dense_matmul_vendor(const std::vector<double>&,
                         const std::vector<double>&,
                         std::vector<double>&,
                         size_t,
                         size_t,
                         size_t,
                         std::string* error) {
    if (error != nullptr) {
        *error = "Vendor dense baseline unavailable because CUDA/cuBLAS support is not built on this machine.";
    }
    return false;
}

bool sparse_matvec_vendor(const SparseMatrix&,
                          const std::vector<double>&,
                          std::vector<double>&,
                          std::string* error) {
    if (error != nullptr) {
        *error = "Vendor sparse baseline unavailable because CUDA/cuSPARSE support is not built on this machine.";
    }
    return false;
}

} // namespace gpu
} // namespace Helios
