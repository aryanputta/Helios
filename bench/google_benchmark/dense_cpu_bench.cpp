#include "helios/cpu_kernels.h"

#include <benchmark/benchmark.h>

#include <thread>
#include <vector>

namespace {

void run_dense(benchmark::State& state, Helios::cpu::Backend backend) {
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    std::vector<double> a(m * k, 1.0);
    std::vector<double> b(k * n, 1.0);
    std::vector<double> c;

    for (auto _ : state) {
        Helios::cpu::dense_matmul(a, b, c, m, n, k, backend, std::thread::hardware_concurrency());
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(2 * m * n * k));
}

void dense_scalar(benchmark::State& state) {
    run_dense(state, Helios::cpu::Backend::Scalar);
}

void dense_avx2(benchmark::State& state) {
    if (!Helios::cpu::backend_available(Helios::cpu::Backend::Avx2)) {
        state.SkipWithError("AVX2 backend unavailable on this machine.");
        return;
    }
    run_dense(state, Helios::cpu::Backend::Avx2);
}

void dense_threaded(benchmark::State& state) {
    if (!Helios::cpu::backend_available(Helios::cpu::Backend::Threaded)) {
        state.SkipWithError("Threaded backend unavailable on this machine.");
        return;
    }
    run_dense(state, Helios::cpu::Backend::Threaded);
}

} // namespace

BENCHMARK(dense_scalar)->Args({256, 256, 256})->Args({512, 512, 512});
BENCHMARK(dense_avx2)->Args({256, 256, 256})->Args({512, 512, 512});
BENCHMARK(dense_threaded)->Args({256, 256, 256})->Args({512, 512, 512});

BENCHMARK_MAIN();
