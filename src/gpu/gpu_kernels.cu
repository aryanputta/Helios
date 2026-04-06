#include "helios/gpu_kernels.h"

#include <cuda_runtime.h>

#if defined(HELIOS_HAVE_CUBLAS)
#include <cublas_v2.h>
#endif

#if defined(HELIOS_HAVE_CUSPARSE)
#include <cusparse.h>
#endif

#include <cstdint>
#include <sstream>

namespace Helios {
namespace gpu {

namespace {

constexpr int kTileSize = 16;

__global__ void dense_matmul_tiled_kernel(const double* A,
                                          const double* B,
                                          double* C,
                                          size_t M,
                                          size_t N,
                                          size_t K) {
    __shared__ double tile_a[kTileSize][kTileSize];
    __shared__ double tile_b[kTileSize][kTileSize];

    const size_t row = static_cast<size_t>(blockIdx.y) * kTileSize + static_cast<size_t>(threadIdx.y);
    const size_t col = static_cast<size_t>(blockIdx.x) * kTileSize + static_cast<size_t>(threadIdx.x);

    double sum = 0.0;
    for (size_t tile = 0; tile < (K + kTileSize - 1) / kTileSize; ++tile) {
        const size_t tiled_col = tile * kTileSize + static_cast<size_t>(threadIdx.x);
        const size_t tiled_row = tile * kTileSize + static_cast<size_t>(threadIdx.y);

        tile_a[threadIdx.y][threadIdx.x] =
            (row < M && tiled_col < K) ? A[row * K + tiled_col] : 0.0;
        tile_b[threadIdx.y][threadIdx.x] =
            (tiled_row < K && col < N) ? B[tiled_row * N + col] : 0.0;
        __syncthreads();

        for (int inner = 0; inner < kTileSize; ++inner) {
            sum += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void sparse_matvec_csr_kernel(const size_t* row_ptr,
                                         const size_t* col_idx,
                                         const double* values,
                                         const double* x,
                                         double* y,
                                         size_t rows) {
    const size_t row = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (size_t index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
        sum += values[index] * x[col_idx[index]];
    }
    y[row] = sum;
}

bool check_cuda(cudaError_t status, std::string* error, const char* step) {
    if (status == cudaSuccess) {
        return true;
    }
    if (error != nullptr) {
        std::ostringstream stream;
        stream << step << ": " << cudaGetErrorString(status);
        *error = stream.str();
    }
    return false;
}

template <typename T>
bool allocate_device_array(T** pointer, size_t count, std::string* error, const char* step) {
    return check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)), error, step);
}

#if defined(HELIOS_HAVE_CUBLAS)
bool check_cublas(cublasStatus_t status, std::string* error, const char* step) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    if (error != nullptr) {
        std::ostringstream stream;
        stream << step << ": cuBLAS status=" << static_cast<int>(status);
        *error = stream.str();
    }
    return false;
}
#endif

#if defined(HELIOS_HAVE_CUSPARSE)
bool check_cusparse(cusparseStatus_t status, std::string* error, const char* step) {
    if (status == CUSPARSE_STATUS_SUCCESS) {
        return true;
    }
    if (error != nullptr) {
        std::ostringstream stream;
        stream << step << ": cuSPARSE status=" << static_cast<int>(status);
        *error = stream.str();
    }
    return false;
}
#endif

} // namespace

CapabilityInfo capabilities() {
    CapabilityInfo info;
    info.compiled_with_cuda = true;
    info.cublas_available =
#if defined(HELIOS_HAVE_CUBLAS)
        true;
#else
        false;
#endif
    info.cusparse_available =
#if defined(HELIOS_HAVE_CUSPARSE)
        true;
#else
        false;
#endif

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        info.runtime_available = false;
        info.reason = cudaGetErrorString(status);
        return info;
    }

    info.device_count = device_count;
    info.runtime_available = device_count > 0;
    info.reason = device_count > 0 ? "CUDA runtime available." : "CUDA compiled in, but no CUDA device was detected.";
    return info;
}

bool dense_matmul(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  size_t M,
                  size_t N,
                  size_t K,
                  std::string* error) {
    const CapabilityInfo info = capabilities();
    if (!info.runtime_available) {
        if (error != nullptr) {
            *error = info.reason;
        }
        return false;
    }

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;
    C.assign(M * N, 0.0);

    if (!allocate_device_array(&d_a, A.size(), error, "cudaMalloc(A)")
        || !allocate_device_array(&d_b, B.size(), error, "cudaMalloc(B)")
        || !allocate_device_array(&d_c, C.size(), error, "cudaMalloc(C)")) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return false;
    }

    bool success =
        check_cuda(cudaMemcpy(d_a, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(A)")
        && check_cuda(cudaMemcpy(d_b, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(B)");

    if (success) {
        dim3 block(kTileSize, kTileSize);
        dim3 grid(
            static_cast<unsigned int>((N + kTileSize - 1) / kTileSize),
            static_cast<unsigned int>((M + kTileSize - 1) / kTileSize));
        dense_matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        success = check_cuda(cudaGetLastError(), error, "dense_matmul_tiled_kernel launch")
            && check_cuda(cudaDeviceSynchronize(), error, "dense_matmul_tiled_kernel synchronize")
            && check_cuda(cudaMemcpy(C.data(), d_c, C.size() * sizeof(double), cudaMemcpyDeviceToHost), error, "cudaMemcpy(C)");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return success;
}

bool sparse_matvec(const SparseMatrix& matrix,
                   const std::vector<double>& x,
                   std::vector<double>& y,
                   std::string* error) {
    const CapabilityInfo info = capabilities();
    if (!info.runtime_available) {
        if (error != nullptr) {
            *error = info.reason;
        }
        return false;
    }

    size_t* d_row_ptr = nullptr;
    size_t* d_col_idx = nullptr;
    double* d_values = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;
    y.assign(matrix.rows, 0.0);

    if (!allocate_device_array(&d_row_ptr, matrix.row_ptr.size(), error, "cudaMalloc(row_ptr)")
        || !allocate_device_array(&d_col_idx, matrix.col_idx.size(), error, "cudaMalloc(col_idx)")
        || !allocate_device_array(&d_values, matrix.values.size(), error, "cudaMalloc(values)")
        || !allocate_device_array(&d_x, x.size(), error, "cudaMalloc(x)")
        || !allocate_device_array(&d_y, y.size(), error, "cudaMalloc(y)")) {
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        return false;
    }

    bool success =
        check_cuda(cudaMemcpy(d_row_ptr, matrix.row_ptr.data(), matrix.row_ptr.size() * sizeof(size_t), cudaMemcpyHostToDevice), error, "cudaMemcpy(row_ptr)")
        && check_cuda(cudaMemcpy(d_col_idx, matrix.col_idx.data(), matrix.col_idx.size() * sizeof(size_t), cudaMemcpyHostToDevice), error, "cudaMemcpy(col_idx)")
        && check_cuda(cudaMemcpy(d_values, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(values)")
        && check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(x)");

    if (success) {
        const int block_size = 256;
        const int grid_size = static_cast<int>((matrix.rows + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size));
        sparse_matvec_csr_kernel<<<grid_size, block_size>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, matrix.rows);
        success = check_cuda(cudaGetLastError(), error, "sparse_matvec_csr_kernel launch")
            && check_cuda(cudaDeviceSynchronize(), error, "sparse_matvec_csr_kernel synchronize")
            && check_cuda(cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost), error, "cudaMemcpy(y)");
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    return success;
}

bool dense_matmul_vendor(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         size_t M,
                         size_t N,
                         size_t K,
                         std::string* error) {
#if !defined(HELIOS_HAVE_CUBLAS)
    if (error != nullptr) {
        *error = "cuBLAS support is not linked into this Helios build.";
    }
    return false;
#else
    const CapabilityInfo info = capabilities();
    if (!info.runtime_available) {
        if (error != nullptr) {
            *error = info.reason;
        }
        return false;
    }

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;
    cublasHandle_t handle = nullptr;
    C.assign(M * N, 0.0);

    bool success =
        allocate_device_array(&d_a, A.size(), error, "cudaMalloc(A)")
        && allocate_device_array(&d_b, B.size(), error, "cudaMalloc(B)")
        && allocate_device_array(&d_c, C.size(), error, "cudaMalloc(C)")
        && check_cublas(cublasCreate(&handle), error, "cublasCreate");

    if (success) {
        success =
            check_cuda(cudaMemcpy(d_a, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(A)")
            && check_cuda(cudaMemcpy(d_b, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(B)");
    }

    if (success) {
        const double alpha = 1.0;
        const double beta = 0.0;
        success =
            check_cublas(
                cublasDgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    static_cast<int>(N),
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    d_b,
                    static_cast<int>(N),
                    d_a,
                    static_cast<int>(K),
                    &beta,
                    d_c,
                    static_cast<int>(N)),
                error,
                "cublasDgemm")
            && check_cuda(cudaMemcpy(C.data(), d_c, C.size() * sizeof(double), cudaMemcpyDeviceToHost), error, "cudaMemcpy(C)");
    }

    if (handle != nullptr) {
        cublasDestroy(handle);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return success;
#endif
}

bool sparse_matvec_vendor(const SparseMatrix& matrix,
                          const std::vector<double>& x,
                          std::vector<double>& y,
                          std::string* error) {
#if !defined(HELIOS_HAVE_CUSPARSE)
    if (error != nullptr) {
        *error = "cuSPARSE support is not linked into this Helios build.";
    }
    return false;
#else
    const CapabilityInfo info = capabilities();
    if (!info.runtime_available) {
        if (error != nullptr) {
            *error = info.reason;
        }
        return false;
    }

    size_t* d_row_ptr = nullptr;
    size_t* d_col_idx = nullptr;
    double* d_values = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;
    void* d_buffer = nullptr;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t spmat = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    y.assign(matrix.rows, 0.0);

    bool success =
        allocate_device_array(&d_row_ptr, matrix.row_ptr.size(), error, "cudaMalloc(row_ptr)")
        && allocate_device_array(&d_col_idx, matrix.col_idx.size(), error, "cudaMalloc(col_idx)")
        && allocate_device_array(&d_values, matrix.values.size(), error, "cudaMalloc(values)")
        && allocate_device_array(&d_x, x.size(), error, "cudaMalloc(x)")
        && allocate_device_array(&d_y, y.size(), error, "cudaMalloc(y)")
        && check_cusparse(cusparseCreate(&handle), error, "cusparseCreate");

    if (success) {
        success =
            check_cuda(cudaMemcpy(d_row_ptr, matrix.row_ptr.data(), matrix.row_ptr.size() * sizeof(size_t), cudaMemcpyHostToDevice), error, "cudaMemcpy(row_ptr)")
            && check_cuda(cudaMemcpy(d_col_idx, matrix.col_idx.data(), matrix.col_idx.size() * sizeof(size_t), cudaMemcpyHostToDevice), error, "cudaMemcpy(col_idx)")
            && check_cuda(cudaMemcpy(d_values, matrix.values.data(), matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(values)")
            && check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice), error, "cudaMemcpy(x)");
    }

    if (success) {
        success =
            check_cusparse(
                cusparseCreateCsr(
                    &spmat,
                    static_cast<std::int64_t>(matrix.rows),
                    static_cast<std::int64_t>(matrix.cols),
                    static_cast<std::int64_t>(matrix.nnz),
                    d_row_ptr,
                    d_col_idx,
                    d_values,
                    CUSPARSE_INDEX_64I,
                    CUSPARSE_INDEX_64I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F),
                error,
                "cusparseCreateCsr")
            && check_cusparse(
                cusparseCreateDnVec(&vec_x, static_cast<std::int64_t>(x.size()), d_x, CUDA_R_64F),
                error,
                "cusparseCreateDnVec(x)")
            && check_cusparse(
                cusparseCreateDnVec(&vec_y, static_cast<std::int64_t>(y.size()), d_y, CUDA_R_64F),
                error,
                "cusparseCreateDnVec(y)");
    }

    if (success) {
        const double alpha = 1.0;
        const double beta = 0.0;
        size_t buffer_size = 0;
        success =
            check_cusparse(
                cusparseSpMV_bufferSize(
                    handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    spmat,
                    vec_x,
                    &beta,
                    vec_y,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT,
                    &buffer_size),
                error,
                "cusparseSpMV_bufferSize")
            && check_cuda(cudaMalloc(&d_buffer, buffer_size), error, "cudaMalloc(spmv_buffer)")
            && check_cusparse(
                cusparseSpMV(
                    handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    spmat,
                    vec_x,
                    &beta,
                    vec_y,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT,
                    d_buffer),
                error,
                "cusparseSpMV")
            && check_cuda(cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost), error, "cudaMemcpy(y)");
    }

    if (vec_y != nullptr) {
        cusparseDestroyDnVec(vec_y);
    }
    if (vec_x != nullptr) {
        cusparseDestroyDnVec(vec_x);
    }
    if (spmat != nullptr) {
        cusparseDestroySpMat(spmat);
    }
    if (handle != nullptr) {
        cusparseDestroy(handle);
    }
    cudaFree(d_buffer);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    return success;
#endif
}

} // namespace gpu
} // namespace Helios
