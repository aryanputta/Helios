#include "helios/cpu_kernels.h"
#include "helios/dataset_loader.h"

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

const Helios::SparseMatrix* load_matrix(benchmark::State& state) {
    static Helios::SparseMatrix matrix;
    static bool initialized = false;
    static bool loaded = false;

    if (!initialized) {
        initialized = true;
        const char* env_path = std::getenv("HELIOS_SPARSE_MATRIX");
        if (env_path != nullptr && env_path[0] != '\0') {
            Helios::DatasetLoader loader;
            loaded = loader.load_matrix_market(env_path, matrix, nullptr);
        }
    }

    if (!loaded) {
        state.SkipWithError("Set HELIOS_SPARSE_MATRIX to a real Matrix Market file before running this harness.");
        return nullptr;
    }
    return &matrix;
}

void run_sparse(benchmark::State& state, Helios::cpu::Backend backend) {
    const Helios::SparseMatrix* matrix = load_matrix(state);
    if (matrix == nullptr) {
        return;
    }

    const std::vector<double> x(matrix->cols, 1.0);
    std::vector<double> y;

    for (auto _ : state) {
        Helios::cpu::sparse_matvec(*matrix, x, y, backend, std::thread::hardware_concurrency());
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(matrix->nnz));
}

void sparse_scalar(benchmark::State& state) {
    run_sparse(state, Helios::cpu::Backend::Scalar);
}

void sparse_avx2(benchmark::State& state) {
    if (!Helios::cpu::backend_available(Helios::cpu::Backend::Avx2)) {
        state.SkipWithError("AVX2 backend unavailable on this machine.");
        return;
    }
    run_sparse(state, Helios::cpu::Backend::Avx2);
}

void sparse_threaded(benchmark::State& state) {
    if (!Helios::cpu::backend_available(Helios::cpu::Backend::Threaded)) {
        state.SkipWithError("Threaded backend unavailable on this machine.");
        return;
    }
    run_sparse(state, Helios::cpu::Backend::Threaded);
}

} // namespace

BENCHMARK(sparse_scalar);
BENCHMARK(sparse_avx2);
BENCHMARK(sparse_threaded);

BENCHMARK_MAIN();
