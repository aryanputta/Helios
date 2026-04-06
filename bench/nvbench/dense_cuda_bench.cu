#include <nvbench/nvbench.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cstdint>

namespace {

template <int Tile>
__global__ void dense_bench_kernel(const double* a,
                                   const double* b,
                                   double* c,
                                   std::int64_t m,
                                   std::int64_t n,
                                   std::int64_t k) {
    __shared__ double tile_a[Tile][Tile];
    __shared__ double tile_b[Tile][Tile];

    const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * Tile + threadIdx.y;
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * Tile + threadIdx.x;

    double sum = 0.0;
    for (std::int64_t tile = 0; tile < (k + Tile - 1) / Tile; ++tile) {
        const std::int64_t tiled_col = tile * Tile + threadIdx.x;
        const std::int64_t tiled_row = tile * Tile + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && tiled_col < k) ? a[row * k + tiled_col] : 0.0;
        tile_b[threadIdx.y][threadIdx.x] =
            (tiled_row < k && col < n) ? b[tiled_row * n + col] : 0.0;
        __syncthreads();

        for (int inner = 0; inner < Tile; ++inner) {
            sum += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

void dense_cuda_bench(nvbench::state& state) {
    const auto m = state.get_int64("M");
    const auto n = state.get_int64("N");
    const auto k = state.get_int64("K");

    state.add_global_memory_reads<double>(static_cast<std::size_t>(m * k + k * n));
    state.add_global_memory_writes<double>(static_cast<std::size_t>(m * n));
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();

    thrust::device_vector<double> a(static_cast<std::size_t>(m * k), 1.0);
    thrust::device_vector<double> b(static_cast<std::size_t>(k * n), 1.0);
    thrust::device_vector<double> c(static_cast<std::size_t>(m * n), 0.0);

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((m + block.y - 1) / block.y));

    state.exec([&](nvbench::launch& launch) {
        dense_bench_kernel<16><<<grid, block, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(a.data()),
            thrust::raw_pointer_cast(b.data()),
            thrust::raw_pointer_cast(c.data()),
            m,
            n,
            k);
    });
}

} // namespace

NVBENCH_BENCH(dense_cuda_bench)
    .add_int64_axis("M", {256, 512, 1024})
    .add_int64_axis("N", {256, 512, 1024})
    .add_int64_axis("K", {256, 512, 1024});
