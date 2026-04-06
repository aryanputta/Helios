#include "helios/cpu_kernels.h"

#include <immintrin.h>

namespace Helios {
namespace cpu {
namespace detail {

void dense_matmul_avx2_impl(const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            size_t M,
                            size_t N,
                            size_t K) {
    for (size_t row = 0; row < M; ++row) {
        double* c_row = C.data() + row * N;
        for (size_t inner = 0; inner < K; ++inner) {
            const __m256d a_broadcast = _mm256_set1_pd(A[row * K + inner]);
            const double* b_row = B.data() + inner * N;

            size_t col = 0;
            for (; col + 4 <= N; col += 4) {
                const __m256d b_values = _mm256_loadu_pd(b_row + col);
                const __m256d c_values = _mm256_loadu_pd(c_row + col);
                const __m256d updated = _mm256_fmadd_pd(a_broadcast, b_values, c_values);
                _mm256_storeu_pd(c_row + col, updated);
            }

            for (; col < N; ++col) {
                c_row[col] += A[row * K + inner] * b_row[col];
            }
        }
    }
}

void sparse_matvec_avx2_impl(const SparseMatrix& matrix,
                             const std::vector<double>& x,
                             std::vector<double>& y) {
    for (size_t row = 0; row < matrix.rows; ++row) {
        const size_t row_start = matrix.row_ptr[row];
        const size_t row_end = matrix.row_ptr[row + 1];

        __m256d vector_sum = _mm256_setzero_pd();
        size_t index = row_start;
        for (; index + 4 <= row_end; index += 4) {
            const __m256d matrix_values = _mm256_loadu_pd(matrix.values.data() + index);
            const __m256i gather_indices = _mm256_set_epi64x(
                static_cast<long long>(matrix.col_idx[index + 3]),
                static_cast<long long>(matrix.col_idx[index + 2]),
                static_cast<long long>(matrix.col_idx[index + 1]),
                static_cast<long long>(matrix.col_idx[index + 0]));
            const __m256d gathered = _mm256_i64gather_pd(x.data(), gather_indices, static_cast<int>(sizeof(double)));
            vector_sum = _mm256_fmadd_pd(matrix_values, gathered, vector_sum);
        }

        alignas(32) double partials[4];
        _mm256_store_pd(partials, vector_sum);
        double sum = partials[0] + partials[1] + partials[2] + partials[3];

        for (; index < row_end; ++index) {
            sum += matrix.values[index] * x[matrix.col_idx[index]];
        }
        y[row] = sum;
    }
}

} // namespace detail
} // namespace cpu
} // namespace Helios
