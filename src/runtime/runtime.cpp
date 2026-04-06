#include "helios/runtime.h"

#include "helios/cpu_kernels.h"
#include "helios/dataset_loader.h"
#include "helios/gpu_kernels.h"
#include "helios/metrics.h"
#include "helios/planner.h"
#include "helios/profiler.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace Helios {

namespace {

enum class ExecutionBackend {
    Auto,
    Scalar,
    Avx2,
    Threaded,
    Cuda,
    Vendor,
};

struct BackendRun {
    std::string name;
    bool executed = false;
    bool supported = false;
    bool valid = false;
    std::string note;
    std::vector<double> samples_ms;
    Metrics::BenchmarkSummary summary;
    std::vector<double> output;
    double checksum = 0.0;
    size_t threads_used = 0;
};

std::string get_option(const std::vector<std::string>& args,
                       const std::string& name,
                       const std::string& fallback = {}) {
    for (size_t index = 0; index + 1 < args.size(); ++index) {
        if (args[index] == name) {
            return args[index + 1];
        }
    }
    return fallback;
}

std::vector<std::string> get_all_options(const std::vector<std::string>& args,
                                         const std::string& name) {
    std::vector<std::string> values;
    for (size_t index = 0; index + 1 < args.size(); ++index) {
        if (args[index] == name) {
            values.push_back(args[index + 1]);
        }
    }
    return values;
}

bool has_option(const std::vector<std::string>& args, const std::string& name) {
    return std::find(args.begin(), args.end(), name) != args.end();
}

bool parse_size_value(const std::string& text, size_t& value) {
    if (text.empty()) {
        return false;
    }

    size_t consumed = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (...) {
        return false;
    }
    return consumed == text.size();
}

size_t get_size_option(const std::vector<std::string>& args,
                       const std::string& name,
                       size_t fallback,
                       bool* ok = nullptr) {
    const std::string raw_value = get_option(args, name);
    if (raw_value.empty()) {
        if (ok != nullptr) {
            *ok = true;
        }
        return fallback;
    }

    size_t parsed_value = 0;
    const bool parse_ok = parse_size_value(raw_value, parsed_value);
    if (ok != nullptr) {
        *ok = parse_ok;
    }
    return parse_ok ? parsed_value : fallback;
}

bool parse_double_value(const std::string& text, double& value) {
    if (text.empty()) {
        return false;
    }

    size_t consumed = 0;
    try {
        value = std::stod(text, &consumed);
    } catch (...) {
        return false;
    }
    return consumed == text.size();
}

double get_double_option(const std::vector<std::string>& args,
                         const std::string& name,
                         double fallback,
                         bool* ok = nullptr) {
    const std::string raw_value = get_option(args, name);
    if (raw_value.empty()) {
        if (ok != nullptr) {
            *ok = true;
        }
        return fallback;
    }

    double parsed_value = 0.0;
    const bool parse_ok = parse_double_value(raw_value, parsed_value);
    if (ok != nullptr) {
        *ok = parse_ok;
    }
    return parse_ok ? parsed_value : fallback;
}

bool parse_execution_backend(const std::string& text, ExecutionBackend& backend) {
    if (text.empty() || text == "auto") {
        backend = ExecutionBackend::Auto;
        return true;
    }
    if (text == "scalar") {
        backend = ExecutionBackend::Scalar;
        return true;
    }
    if (text == "avx2") {
        backend = ExecutionBackend::Avx2;
        return true;
    }
    if (text == "threaded") {
        backend = ExecutionBackend::Threaded;
        return true;
    }
    if (text == "cuda") {
        backend = ExecutionBackend::Cuda;
        return true;
    }
    if (text == "vendor") {
        backend = ExecutionBackend::Vendor;
        return true;
    }
    return false;
}

const char* to_string(ExecutionBackend backend) {
    switch (backend) {
    case ExecutionBackend::Auto:
        return "auto";
    case ExecutionBackend::Scalar:
        return "scalar";
    case ExecutionBackend::Avx2:
        return "avx2";
    case ExecutionBackend::Threaded:
        return "threaded";
    case ExecutionBackend::Cuda:
        return "cuda";
    case ExecutionBackend::Vendor:
        return "vendor";
    }
    return "unknown";
}

cpu::Backend to_cpu_backend(ExecutionBackend backend) {
    switch (backend) {
    case ExecutionBackend::Scalar:
        return cpu::Backend::Scalar;
    case ExecutionBackend::Avx2:
        return cpu::Backend::Avx2;
    case ExecutionBackend::Threaded:
        return cpu::Backend::Threaded;
    case ExecutionBackend::Auto:
    case ExecutionBackend::Cuda:
    case ExecutionBackend::Vendor:
        return cpu::Backend::Scalar;
    }
    return cpu::Backend::Scalar;
}

std::vector<double> benchmark_kernel(size_t warmup,
                                     size_t trials,
                                     const std::function<void()>& runner) {
    for (size_t iteration = 0; iteration < warmup; ++iteration) {
        runner();
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(trials);
    for (size_t iteration = 0; iteration < trials; ++iteration) {
        const double start = Metrics::now();
        runner();
        samples_ms.push_back((Metrics::now() - start) * 1000.0);
    }
    return samples_ms;
}

std::string format_double(double value, int precision = 6) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string format_hex_u64(std::uint64_t value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << value;
    return stream.str();
}

std::string sanitize_metric_token(std::string text) {
    for (char& character : text) {
        const bool alpha_numeric =
            (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9');
        if (!alpha_numeric) {
            character = '_';
        } else if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    return text;
}

double summary_seconds(const Metrics::BenchmarkSummary& summary) {
    return summary.median_ms / 1000.0;
}

double effective_gflops(const WorkloadProfile& profile,
                        const Metrics::BenchmarkSummary& summary) {
    const double seconds = summary_seconds(summary);
    if (seconds <= 0.0) {
        return 0.0;
    }
    return profile.estimated_flops / seconds / 1.0e9;
}

double effective_bandwidth_gbps(const WorkloadProfile& profile,
                                const Metrics::BenchmarkSummary& summary) {
    const double seconds = summary_seconds(summary);
    if (seconds <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(profile.bytes_moved) / seconds / 1.0e9;
}

bool write_text_file(const std::string& path, const std::string& contents) {
    const std::filesystem::path filesystem_path(path);
    if (filesystem_path.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(filesystem_path.parent_path(), error);
        if (error) {
            return false;
        }
    }

    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << contents;
    return true;
}

size_t count_reached_nodes(const std::vector<int>& distances) {
    size_t reached = 0;
    for (const int distance : distances) {
        if (distance >= 0) {
            ++reached;
        }
    }
    return reached;
}

double vector_checksum(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double diff = 0.0;
    for (size_t index = 0; index < lhs.size(); ++index) {
        diff = std::max(diff, std::abs(lhs[index] - rhs[index]));
    }
    return diff;
}

void append_entries(std::vector<Metrics::MetricEntry>& target,
                    const std::vector<Metrics::MetricEntry>& extra) {
    target.insert(target.end(), extra.begin(), extra.end());
}

ExecutionBackend choose_dense_cpu_backend(const WorkloadProfile& profile,
                                          const cpu::CapabilityInfo& cpu_caps) {
    if (cpu_caps.hardware_threads > 1 && profile.estimated_flops >= 2.0e7) {
        return ExecutionBackend::Threaded;
    }
    if (cpu_caps.avx2_available
        && profile.cols >= 8
        && profile.estimated_flops >= 3.0e4) {
        return ExecutionBackend::Avx2;
    }
    return ExecutionBackend::Scalar;
}

ExecutionBackend choose_sparse_cpu_backend(const WorkloadProfile& profile,
                                           const cpu::CapabilityInfo& cpu_caps) {
    if (cpu_caps.hardware_threads > 1 && profile.rows >= 4096) {
        return ExecutionBackend::Threaded;
    }
    if (cpu_caps.avx2_available && profile.average_degree >= 8.0) {
        return ExecutionBackend::Avx2;
    }
    return ExecutionBackend::Scalar;
}

ExecutionBackend resolve_dense_backend(ExecutionBackend requested_backend,
                                       const Planner::Decision& decision,
                                       const WorkloadProfile& profile,
                                       const cpu::CapabilityInfo& cpu_caps,
                                       const gpu::CapabilityInfo& gpu_caps) {
    if (requested_backend != ExecutionBackend::Auto) {
        return requested_backend;
    }
    if (decision.strategy == Planner::Strategy::DenseGPU && gpu_caps.runtime_available) {
        return ExecutionBackend::Cuda;
    }
    return choose_dense_cpu_backend(profile, cpu_caps);
}

ExecutionBackend resolve_sparse_backend(ExecutionBackend requested_backend,
                                        const Planner::Decision& decision,
                                        const WorkloadProfile& profile,
                                        const cpu::CapabilityInfo& cpu_caps,
                                        const gpu::CapabilityInfo& gpu_caps) {
    if (requested_backend != ExecutionBackend::Auto) {
        return requested_backend;
    }
    if (decision.strategy == Planner::Strategy::SparseGPU && gpu_caps.runtime_available) {
        return ExecutionBackend::Cuda;
    }
    return choose_sparse_cpu_backend(profile, cpu_caps);
}

std::string build_command_line(const std::string& command,
                               const std::string& workload,
                               const std::vector<std::string>& args) {
    std::ostringstream stream;
    stream << "helios " << command;
    if (!workload.empty()) {
        stream << " " << workload;
    }
    for (const auto& arg : args) {
        stream << " " << arg;
    }
    return stream.str();
}

std::vector<Metrics::MetricEntry> make_common_entries(const std::string& command,
                                                      const std::string& workload,
                                                      const std::string& result_kind,
                                                      const std::vector<std::string>& args,
                                                      const WorkloadProfile& profile,
                                                      const Planner::Decision& decision,
                                                      const cpu::CapabilityInfo& cpu_caps,
                                                      const gpu::CapabilityInfo& gpu_caps) {
    std::vector<Metrics::MetricEntry> entries = Metrics::collect_host_metadata();
    append_entries(entries, {
        {"result_kind", result_kind},
        {"command_line", build_command_line(command, workload, args)},
        {"command", command},
        {"workload", workload},
        {"planner_strategy", Planner::to_string(decision.strategy)},
        {"planner_reason", decision.reason},
        {"rows", std::to_string(profile.rows)},
        {"cols", std::to_string(profile.cols)},
        {"nnz", std::to_string(profile.nnz)},
        {"density", format_double(profile.density)},
        {"arithmetic_intensity", format_double(profile.arithmetic_intensity)},
        {"estimated_flops", format_double(profile.estimated_flops)},
        {"bytes_moved", std::to_string(profile.bytes_moved)},
        {"average_degree", format_double(profile.average_degree)},
        {"max_degree", std::to_string(profile.max_degree)},
        {"cpu_avx2_available", cpu_caps.avx2_available ? "true" : "false"},
        {"cpu_hardware_threads", std::to_string(cpu_caps.hardware_threads)},
        {"cuda_compiled", gpu_caps.compiled_with_cuda ? "true" : "false"},
        {"cuda_runtime_available", gpu_caps.runtime_available ? "true" : "false"},
        {"cublas_available", gpu_caps.cublas_available ? "true" : "false"},
        {"cusparse_available", gpu_caps.cusparse_available ? "true" : "false"},
        {"cuda_device_count", std::to_string(gpu_caps.device_count)},
        {"cuda_reason", gpu_caps.reason},
    });
    return entries;
}

void append_dataset_load_info(std::vector<Metrics::MetricEntry>& metrics,
                              const DatasetLoadInfo& info) {
    append_entries(metrics, {
        {"dataset_input_path", info.input_path},
        {"dataset_input_format", info.input_format},
        {"dataset_source_url", info.source_url},
        {"dataset_cache_hit", info.cache_hit ? "true" : "false"},
        {"dataset_cache_path", info.cache_path},
        {"dataset_input_size_bytes", std::to_string(info.input_size_bytes)},
        {"dataset_input_mtime_ticks", std::to_string(info.input_mtime_ticks)},
        {"dataset_input_checksum_fnv1a", format_hex_u64(info.input_checksum_fnv1a)},
    });
}

std::filesystem::path default_manifest_path(const DatasetLoadInfo& info) {
    if (info.cache_path.empty()) {
        return {};
    }

    std::filesystem::path cache_path(info.cache_path);
    std::filesystem::path output;
    bool replaced = false;
    for (const auto& part : cache_path) {
        if (part == "cache") {
            output /= "manifests";
            replaced = true;
        } else {
            output /= part;
        }
    }
    if (!replaced) {
        return {};
    }

    std::string filename = output.filename().string();
    if (filename.size() >= 8 && filename.substr(filename.size() - 8) == ".csr.bin") {
        filename = filename.substr(0, filename.size() - 8) + ".json";
    } else if (filename.size() >= 10 && filename.substr(filename.size() - 10) == ".graph.bin") {
        filename = filename.substr(0, filename.size() - 10) + ".json";
    } else {
        filename += ".json";
    }
    output.replace_filename(filename);
    return output;
}

bool maybe_write_manifest(const std::vector<std::string>& args,
                          const DatasetLoadInfo& info,
                          const std::vector<Metrics::MetricEntry>& dataset_entries,
                          std::string& manifest_path) {
    std::string requested_path = get_option(args, "--manifest");
    std::filesystem::path output_path =
        requested_path.empty() ? default_manifest_path(info) : std::filesystem::path(requested_path);
    if (output_path.empty()) {
        return false;
    }

    std::vector<Metrics::MetricEntry> manifest_entries = Metrics::collect_host_metadata();
    manifest_entries[0].value = "helios_dataset_manifest_v1";
    append_entries(manifest_entries, dataset_entries);
    manifest_path = output_path.string();
    return Metrics::write_json(manifest_path, manifest_entries);
}

void emit_outputs(const std::vector<std::string>& args,
                  const std::vector<Metrics::MetricEntry>& entries) {
    const std::string csv_path = get_option(args, "--csv", get_option(args, "--output"));
    if (!csv_path.empty()) {
        if (Metrics::write_csv(csv_path, entries)) {
            std::cout << "[output] csv=" << csv_path << "\n";
        } else {
            std::cerr << "Warning: failed to write CSV metrics to " << csv_path << "\n";
        }
    }

    const std::string json_path = get_option(args, "--json");
    if (!json_path.empty()) {
        if (Metrics::write_json(json_path, entries)) {
            std::cout << "[output] json=" << json_path << "\n";
        } else {
            std::cerr << "Warning: failed to write JSON metrics to " << json_path << "\n";
        }
    }
}

void print_profile_summary(const WorkloadProfile& profile) {
    std::cout << "[profile] density=" << format_double(profile.density)
              << " arithmetic_intensity=" << format_double(profile.arithmetic_intensity)
              << " bytes_moved=" << profile.bytes_moved
              << " average_degree=" << format_double(profile.average_degree)
              << " max_degree=" << profile.max_degree << "\n";
}

void print_capability_summary(const cpu::CapabilityInfo& cpu_caps,
                              const gpu::CapabilityInfo& gpu_caps) {
    std::cout << "[capability] cpu_avx2=" << (cpu_caps.avx2_available ? "true" : "false")
              << " cpu_threads=" << cpu_caps.hardware_threads
              << " cuda_compiled=" << (gpu_caps.compiled_with_cuda ? "true" : "false")
              << " cuda_runtime=" << (gpu_caps.runtime_available ? "true" : "false")
              << " cublas=" << (gpu_caps.cublas_available ? "true" : "false")
              << " cusparse=" << (gpu_caps.cusparse_available ? "true" : "false");
    if (!gpu_caps.reason.empty()) {
        std::cout << " cuda_note=\"" << gpu_caps.reason << "\"";
    }
    std::cout << "\n";
}

bool validate_backend_request(const std::string& requested_path,
                              ExecutionBackend requested_backend,
                              std::string& error) {
    if (requested_path == "cpu"
        && (requested_backend == ExecutionBackend::Cuda || requested_backend == ExecutionBackend::Vendor)) {
        error = "--path cpu cannot be combined with --backend cuda or --backend vendor.";
        return false;
    }
    if (requested_path == "gpu"
        && requested_backend != ExecutionBackend::Auto
        && requested_backend != ExecutionBackend::Cuda
        && requested_backend != ExecutionBackend::Vendor) {
        error = "--path gpu can only be combined with --backend auto, --backend cuda, or --backend vendor.";
        return false;
    }
    return true;
}

BackendRun run_dense_backend(ExecutionBackend backend,
                             const std::vector<double>& A,
                             const std::vector<double>& B,
                             size_t rows,
                             size_t cols,
                             size_t depth,
                             size_t warmup,
                             size_t trials,
                             size_t thread_count) {
    BackendRun run;
    run.name = to_string(backend);
    run.supported = true;
    const gpu::CapabilityInfo gpu_caps = gpu::capabilities();

    auto runner = [&run, &A, &B, rows, cols, depth, thread_count, backend]() {
        if (backend == ExecutionBackend::Cuda) {
            std::string error;
            if (!gpu::dense_matmul(A, B, run.output, rows, cols, depth, &error)) {
                run.supported = false;
                run.note = error;
            }
            return;
        }
        if (backend == ExecutionBackend::Vendor) {
            std::string error;
            if (!gpu::dense_matmul_vendor(A, B, run.output, rows, cols, depth, &error)) {
                run.supported = false;
                run.note = error;
            }
            return;
        }

        if (!cpu::dense_matmul(A, B, run.output, rows, cols, depth, to_cpu_backend(backend), thread_count)) {
            run.supported = false;
            run.note = "Requested CPU backend is not available on this machine.";
        }
    };

    if (backend != ExecutionBackend::Cuda
        && backend != ExecutionBackend::Vendor
        && !cpu::backend_available(to_cpu_backend(backend))) {
        run.supported = false;
        run.note = "Requested CPU backend is not available on this machine.";
        return run;
    }

    if (backend == ExecutionBackend::Cuda && !gpu_caps.runtime_available) {
        run.supported = false;
        run.note = gpu_caps.reason;
        return run;
    }

    if (backend == ExecutionBackend::Vendor && (!gpu_caps.runtime_available || !gpu_caps.cublas_available)) {
        run.supported = false;
        run.note = gpu_caps.runtime_available
            ? "cuBLAS support is not linked into this Helios build."
            : gpu_caps.reason;
        return run;
    }

    run.samples_ms = benchmark_kernel(warmup, trials, runner);
    if (!run.supported) {
        return run;
    }
    run.summary = Metrics::summarize(run.samples_ms);
    run.checksum = vector_checksum(run.output);
    run.executed = true;
    run.valid = true;
    run.threads_used = backend == ExecutionBackend::Threaded ? std::max<size_t>(1, thread_count) : 1;
    return run;
}

BackendRun run_sparse_backend(ExecutionBackend backend,
                              const SparseMatrix& matrix,
                              const std::vector<double>& x,
                              size_t warmup,
                              size_t trials,
                              size_t thread_count) {
    BackendRun run;
    run.name = to_string(backend);
    run.supported = true;
    const gpu::CapabilityInfo gpu_caps = gpu::capabilities();

    auto runner = [&run, &matrix, &x, thread_count, backend]() {
        if (backend == ExecutionBackend::Cuda) {
            std::string error;
            if (!gpu::sparse_matvec(matrix, x, run.output, &error)) {
                run.supported = false;
                run.note = error;
            }
            return;
        }
        if (backend == ExecutionBackend::Vendor) {
            std::string error;
            if (!gpu::sparse_matvec_vendor(matrix, x, run.output, &error)) {
                run.supported = false;
                run.note = error;
            }
            return;
        }

        if (!cpu::sparse_matvec(matrix, x, run.output, to_cpu_backend(backend), thread_count)) {
            run.supported = false;
            run.note = "Requested CPU backend is not available on this machine.";
        }
    };

    if (backend != ExecutionBackend::Cuda
        && backend != ExecutionBackend::Vendor
        && !cpu::backend_available(to_cpu_backend(backend))) {
        run.supported = false;
        run.note = "Requested CPU backend is not available on this machine.";
        return run;
    }

    if (backend == ExecutionBackend::Cuda && !gpu_caps.runtime_available) {
        run.supported = false;
        run.note = gpu_caps.reason;
        return run;
    }

    if (backend == ExecutionBackend::Vendor && (!gpu_caps.runtime_available || !gpu_caps.cusparse_available)) {
        run.supported = false;
        run.note = gpu_caps.runtime_available
            ? "cuSPARSE support is not linked into this Helios build."
            : gpu_caps.reason;
        return run;
    }

    run.samples_ms = benchmark_kernel(warmup, trials, runner);
    if (!run.supported) {
        return run;
    }
    run.summary = Metrics::summarize(run.samples_ms);
    run.checksum = vector_checksum(run.output);
    run.executed = true;
    run.valid = true;
    run.threads_used = backend == ExecutionBackend::Threaded ? std::max<size_t>(1, thread_count) : 1;
    return run;
}

void append_backend_metrics(std::vector<Metrics::MetricEntry>& metrics,
                            const BackendRun& run,
                            const BackendRun* scalar_reference,
                            const std::string& workload,
                            const WorkloadProfile& profile) {
    const std::string prefix = run.name + "_";
    metrics.push_back({prefix + "executed", run.executed ? "true" : "false"});
    metrics.push_back({prefix + "supported", run.supported ? "true" : "false"});
    if (!run.note.empty()) {
        metrics.push_back({prefix + "note", run.note});
    }
    if (!run.executed) {
        return;
    }

    metrics.push_back({prefix + "min_ms", format_double(run.summary.min_ms)});
    metrics.push_back({prefix + "median_ms", format_double(run.summary.median_ms)});
    metrics.push_back({prefix + "p95_ms", format_double(run.summary.p95_ms)});
    metrics.push_back({prefix + "max_ms", format_double(run.summary.max_ms)});
    metrics.push_back({prefix + "mean_ms", format_double(run.summary.mean_ms)});
    metrics.push_back({prefix + "stddev_ms", format_double(run.summary.stddev_ms)});
    metrics.push_back({prefix + "cv_pct", format_double(run.summary.cv_pct)});
    metrics.push_back({prefix + "checksum", format_double(run.checksum)});
    metrics.push_back({prefix + "threads_used", std::to_string(run.threads_used)});
    if (workload == "dense" || workload == "sparse") {
        metrics.push_back({prefix + "effective_gflops", format_double(effective_gflops(profile, run.summary))});
        metrics.push_back({prefix + "effective_bandwidth_gbps", format_double(effective_bandwidth_gbps(profile, run.summary))});
    }

    if (scalar_reference != nullptr && scalar_reference->executed && run.name != "scalar") {
        metrics.push_back({prefix + "speedup_vs_scalar", format_double(scalar_reference->summary.median_ms / run.summary.median_ms)});
        metrics.push_back({prefix + "max_abs_diff_vs_scalar", format_double(max_abs_diff(run.output, scalar_reference->output))});
    }
}

void print_backend_run(const BackendRun& run,
                       const BackendRun* scalar_reference,
                       const std::string& workload,
                       const WorkloadProfile& profile) {
    if (!run.supported) {
        std::cout << "[baseline] backend=" << run.name << " skipped=\"" << run.note << "\"\n";
        return;
    }
    if (!run.executed) {
        return;
    }

    std::cout << "[baseline] backend=" << run.name
              << " median_ms=" << format_double(run.summary.median_ms)
              << " p95_ms=" << format_double(run.summary.p95_ms)
              << " cv_pct=" << format_double(run.summary.cv_pct)
              << " checksum=" << format_double(run.checksum);
    if (workload == "dense" || workload == "sparse") {
        std::cout << " effective_gflops=" << format_double(effective_gflops(profile, run.summary))
                  << " effective_bandwidth_gbps=" << format_double(effective_bandwidth_gbps(profile, run.summary));
    }
    if (run.threads_used > 1) {
        std::cout << " threads=" << run.threads_used;
    }
    if (scalar_reference != nullptr && scalar_reference->executed && run.name != "scalar") {
        std::cout << " speedup_vs_scalar=" << format_double(scalar_reference->summary.median_ms / run.summary.median_ms)
                  << " max_abs_diff_vs_scalar=" << format_double(max_abs_diff(run.output, scalar_reference->output));
    }
    std::cout << "\n";
}

void print_validation_run(const BackendRun& run,
                          const BackendRun* scalar_reference,
                          double tolerance) {
    if (!run.supported) {
        std::cout << "[validation] backend=" << run.name << " skipped=\"" << run.note << "\"\n";
        return;
    }
    if (!run.executed) {
        return;
    }

    const double diff =
        (scalar_reference != nullptr && scalar_reference->executed && run.name != "scalar")
            ? max_abs_diff(run.output, scalar_reference->output)
            : 0.0;
    const bool pass = run.name == "scalar" || diff <= tolerance;
    std::cout << "[validation] backend=" << run.name
              << " status=" << (pass ? "pass" : "fail")
              << " checksum=" << format_double(run.checksum);
    if (run.name != "scalar") {
        std::cout << " max_abs_diff_vs_scalar=" << format_double(diff)
                  << " tolerance=" << format_double(tolerance, 12);
    }
    std::cout << "\n";
}

std::vector<ExecutionBackend> build_backend_plan(ExecutionBackend selected_backend,
                                                 bool compare_all,
                                                 const cpu::CapabilityInfo& cpu_caps,
                                                 bool cuda_available,
                                                 bool vendor_available) {
    std::vector<ExecutionBackend> plan;
    auto append_unique = [&plan](ExecutionBackend backend) {
        if (std::find(plan.begin(), plan.end(), backend) == plan.end()) {
            plan.push_back(backend);
        }
    };

    if (compare_all) {
        append_unique(ExecutionBackend::Scalar);
        if (cpu_caps.avx2_available) {
            append_unique(ExecutionBackend::Avx2);
        }
        if (cpu_caps.hardware_threads > 1) {
            append_unique(ExecutionBackend::Threaded);
        }
        if (cuda_available) {
            append_unique(ExecutionBackend::Cuda);
        }
        if (vendor_available) {
            append_unique(ExecutionBackend::Vendor);
        }
    }
    append_unique(selected_backend);
    return plan;
}

std::string default_planner_log_path() {
    return "results/json/planner_observations.jsonl";
}

const BackendRun* find_winning_run(const std::vector<BackendRun>& runs) {
    const BackendRun* winning_run = nullptr;
    for (const auto& run : runs) {
        if (!run.executed || !run.supported) {
            continue;
        }
        if (winning_run == nullptr || run.summary.median_ms < winning_run->summary.median_ms) {
            winning_run = &run;
        }
    }
    return winning_run;
}

const BackendRun* find_run_by_name(const std::vector<BackendRun>& runs,
                                   const std::string& name) {
    for (const auto& run : runs) {
        if (run.name == name) {
            return &run;
        }
    }
    return nullptr;
}

void append_selection_metrics(std::vector<Metrics::MetricEntry>& metrics,
                              const std::vector<BackendRun>& runs,
                              const std::string& selected_backend) {
    const BackendRun* winning_run = find_winning_run(runs);
    const BackendRun* selected_run = find_run_by_name(runs, selected_backend);
    if (winning_run == nullptr || selected_run == nullptr || !selected_run->executed) {
        return;
    }

    metrics.push_back({"winning_backend", winning_run->name});
    metrics.push_back({"winning_backend_median_ms", format_double(winning_run->summary.median_ms)});
    metrics.push_back({"selected_won", selected_backend == winning_run->name ? "true" : "false"});
    metrics.push_back({"selected_backend_regret_vs_winner", format_double(selected_run->summary.median_ms / winning_run->summary.median_ms)});
}

void maybe_log_planner_observation(const std::vector<std::string>& args,
                                   const std::string& command,
                                   const std::string& workload,
                                   const DatasetLoadInfo* load_info,
                                   const WorkloadProfile& profile,
                                   const Planner::Decision& decision,
                                   const std::string& selected_backend,
                                   const std::vector<BackendRun>& runs,
                                   bool compare_all) {
    const std::string explicit_log_path = get_option(args, "--planner-log");
    if (explicit_log_path.empty() && !compare_all) {
        return;
    }

    const std::string planner_log_path =
        explicit_log_path.empty() ? default_planner_log_path() : explicit_log_path;

    const BackendRun* winning_run = find_winning_run(runs);
    if (winning_run == nullptr) {
        return;
    }

    std::vector<Metrics::MetricEntry> observation = Metrics::collect_host_metadata();
    observation[0].value = "helios_planner_observation_v1";
    append_entries(observation, {
        {"command_line", build_command_line(command, workload, args)},
        {"command", command},
        {"workload", workload},
        {"planner_strategy", Planner::to_string(decision.strategy)},
        {"planner_reason", decision.reason},
        {"selected_backend", selected_backend},
        {"winning_backend", winning_run->name},
        {"selected_won", selected_backend == winning_run->name ? "true" : "false"},
        {"winning_backend_median_ms", format_double(winning_run->summary.median_ms)},
        {"rows", std::to_string(profile.rows)},
        {"cols", std::to_string(profile.cols)},
        {"nnz", std::to_string(profile.nnz)},
        {"density", format_double(profile.density)},
        {"arithmetic_intensity", format_double(profile.arithmetic_intensity)},
        {"estimated_flops", format_double(profile.estimated_flops)},
        {"bytes_moved", std::to_string(profile.bytes_moved)},
    });

    if (load_info != nullptr) {
        append_entries(observation, {
            {"dataset_input_path", load_info->input_path},
            {"dataset_cache_path", load_info->cache_path},
            {"dataset_input_checksum_fnv1a", format_hex_u64(load_info->input_checksum_fnv1a)},
        });
    }

    for (const auto& run : runs) {
        if (!run.executed || !run.supported) {
            continue;
        }
        observation.push_back({run.name + "_median_ms", format_double(run.summary.median_ms)});
    }

    if (Metrics::append_jsonl(planner_log_path, observation)) {
        std::cout << "[planner-log] path=" << planner_log_path
                  << " winning_backend=" << winning_run->name << "\n";
    } else {
        std::cerr << "Warning: failed to append planner observation to " << planner_log_path << "\n";
    }
}

bool parse_metric_double(const std::vector<Metrics::MetricEntry>& metrics,
                         const std::string& key,
                         double& value) {
    return parse_double_value(Metrics::find_value(metrics, key), value);
}

std::string primary_median_key(const std::vector<Metrics::MetricEntry>& metrics) {
    const std::string selected_backend = Metrics::find_value(metrics, "selected_backend");
    if (!selected_backend.empty()) {
        const std::string candidate = selected_backend + "_median_ms";
        if (!Metrics::find_value(metrics, candidate).empty()) {
            return candidate;
        }
    }
    if (!Metrics::find_value(metrics, "median_ms").empty()) {
        return "median_ms";
    }
    return {};
}

struct ReportRecord {
    std::string path;
    std::string workload;
    std::string dataset;
    std::string selected_backend;
    std::string winning_backend;
    std::string planner_strategy;
    std::string planner_reason;
    std::string cuda_reason;
    double median_ms = 0.0;
    double selected_speedup_vs_scalar = 0.0;
    double best_speedup_vs_scalar = 0.0;
    double selected_effective_gflops = 0.0;
    double selected_effective_bandwidth_gbps = 0.0;
    double avx2_speedup_vs_scalar = 0.0;
    double threaded_speedup_vs_scalar = 0.0;
    double selected_regret_vs_winner = 0.0;
    bool selected_won = false;
    bool has_selected_speedup = false;
    bool has_best_speedup = false;
    bool has_selected_effective_gflops = false;
    bool has_selected_effective_bandwidth = false;
    bool has_avx2_speedup = false;
    bool has_threaded_speedup = false;
    bool has_selected_regret = false;
    bool has_selected_won = false;
    bool cuda_compiled = false;
    bool cuda_runtime_available = false;
};

bool parse_metric_bool(const std::vector<Metrics::MetricEntry>& metrics,
                       const std::string& key,
                       bool& value) {
    const std::string raw = Metrics::find_value(metrics, key);
    if (raw.empty()) {
        return false;
    }
    value = raw == "true" || raw == "1" || raw == "yes";
    return true;
}

std::string dataset_label(const std::vector<Metrics::MetricEntry>& metrics) {
    const std::string dataset_path = Metrics::find_value(metrics, "dataset_input_path");
    if (!dataset_path.empty()) {
        return dataset_path;
    }

    const std::string workload = Metrics::find_value(metrics, "workload");
    if (workload == "dense") {
        const std::string rows = Metrics::find_value(metrics, "rows");
        const std::string cols = Metrics::find_value(metrics, "cols");
        const std::string flops = Metrics::find_value(metrics, "estimated_flops");
        if (!rows.empty() && !cols.empty()) {
            return "synthetic_dense_sanity_" + rows + "x" + cols + "_flops_" + sanitize_metric_token(flops);
        }
        return "synthetic_dense_sanity";
    }
    return {};
}

bool load_report_record(const std::string& path, ReportRecord& record) {
    std::vector<Metrics::MetricEntry> metrics;
    if (!Metrics::read_metrics(path, metrics)) {
        return false;
    }

    const std::string median_key = primary_median_key(metrics);
    if (median_key.empty() || !parse_metric_double(metrics, median_key, record.median_ms)) {
        return false;
    }

    record.path = path;
    record.workload = Metrics::find_value(metrics, "workload");
    record.dataset = dataset_label(metrics);
    record.selected_backend = Metrics::find_value(metrics, "selected_backend");
    if (record.selected_backend.empty() && record.workload == "graph") {
        record.selected_backend = "scalar";
    }
    record.winning_backend = Metrics::find_value(metrics, "winning_backend");
    record.planner_strategy = Metrics::find_value(metrics, "planner_strategy");
    record.planner_reason = Metrics::find_value(metrics, "planner_reason");
    record.cuda_reason = Metrics::find_value(metrics, "cuda_reason");
    parse_metric_bool(metrics, "cuda_compiled", record.cuda_compiled);
    parse_metric_bool(metrics, "cuda_runtime_available", record.cuda_runtime_available);
    record.has_selected_won = parse_metric_bool(metrics, "selected_won", record.selected_won);
    record.has_selected_regret =
        parse_metric_double(metrics, "selected_backend_regret_vs_winner", record.selected_regret_vs_winner);

    const std::string selected_prefix =
        record.selected_backend.empty() ? std::string() : record.selected_backend + "_";
    if (!selected_prefix.empty()) {
        record.has_selected_speedup =
            parse_metric_double(metrics, selected_prefix + "speedup_vs_scalar", record.selected_speedup_vs_scalar);
        record.has_selected_effective_gflops =
            parse_metric_double(metrics, selected_prefix + "effective_gflops", record.selected_effective_gflops);
        record.has_selected_effective_bandwidth =
            parse_metric_double(metrics, selected_prefix + "effective_bandwidth_gbps", record.selected_effective_bandwidth_gbps);
    }

    record.has_avx2_speedup = parse_metric_double(metrics, "avx2_speedup_vs_scalar", record.avx2_speedup_vs_scalar);
    record.has_threaded_speedup =
        parse_metric_double(metrics, "threaded_speedup_vs_scalar", record.threaded_speedup_vs_scalar);

    static const std::vector<std::string> speedup_backends = {"scalar", "avx2", "threaded", "cuda", "vendor"};
    for (const auto& backend : speedup_backends) {
        double speedup = 0.0;
        if (!parse_metric_double(metrics, backend + "_speedup_vs_scalar", speedup)) {
            continue;
        }
        if (!record.has_best_speedup || speedup > record.best_speedup_vs_scalar) {
            record.best_speedup_vs_scalar = speedup;
            record.has_best_speedup = true;
        }
    }
    return true;
}

int handle_report(const std::vector<std::string>& args) {
    std::vector<std::string> result_paths = get_all_options(args, "--result");
    std::vector<std::string> planner_log_paths = get_all_options(args, "--planner-log");
    const std::string results_dir = get_option(args, "--results-dir");
    if (result_paths.empty() && planner_log_paths.empty() && !results_dir.empty()) {
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator iterator(results_dir, error), end; iterator != end; iterator.increment(error)) {
            if (error) {
                break;
            }
            if (!iterator->is_regular_file()) {
                continue;
            }

            const auto extension = iterator->path().extension().string();
            if (extension == ".json") {
                result_paths.push_back(iterator->path().string());
            } else if (extension == ".jsonl") {
                planner_log_paths.push_back(iterator->path().string());
            }
        }
        std::sort(result_paths.begin(), result_paths.end());
        std::sort(planner_log_paths.begin(), planner_log_paths.end());
    }

    if (result_paths.empty() && planner_log_paths.empty()) {
        std::cerr << "Error: report requires at least one --result <result.csv|json>, --planner-log <observations.jsonl>, or --results-dir <directory>.\n";
        return 1;
    }

    std::vector<ReportRecord> records;
    records.reserve(result_paths.size());
    for (const auto& path : result_paths) {
        ReportRecord record;
        if (!load_report_record(path, record)) {
            if (results_dir.empty()) {
                std::cerr << "Error: failed to load report input from " << path << "\n";
                return 1;
            }
            continue;
        }
        records.push_back(std::move(record));
    }

    size_t planner_observation_count = 0;
    size_t planner_selected_won_count = 0;
    std::map<std::string, size_t> winner_counts;
    size_t real_data_result_count = 0;
    bool any_cuda_available = false;
    bool have_best_overall = false;
    bool have_best_real_data = false;
    bool have_worst_regret = false;
    ReportRecord best_overall_record;
    ReportRecord best_real_data_record;
    ReportRecord worst_regret_record;

    for (const auto& record : records) {
        any_cuda_available = any_cuda_available || record.cuda_runtime_available;
        const bool is_real_data = !record.dataset.empty() && record.dataset != "synthetic_dense_sanity";
        if (is_real_data) {
            ++real_data_result_count;
        }

        if (record.has_best_speedup && (!have_best_overall || record.best_speedup_vs_scalar > best_overall_record.best_speedup_vs_scalar)) {
            best_overall_record = record;
            have_best_overall = true;
        }
        if (is_real_data
            && record.has_best_speedup
            && (!have_best_real_data || record.best_speedup_vs_scalar > best_real_data_record.best_speedup_vs_scalar)) {
            best_real_data_record = record;
            have_best_real_data = true;
        }
        if (record.has_selected_regret
            && record.selected_regret_vs_winner > 1.0
            && (!have_worst_regret || record.selected_regret_vs_winner > worst_regret_record.selected_regret_vs_winner)) {
            worst_regret_record = record;
            have_worst_regret = true;
        }
    }

    for (const auto& path : planner_log_paths) {
        std::vector<std::vector<Metrics::MetricEntry>> observations;
        if (!Metrics::read_jsonl(path, observations)) {
            std::cerr << "Error: failed to read planner observations from " << path << "\n";
            return 1;
        }

        planner_observation_count += observations.size();
        for (const auto& observation : observations) {
            bool selected_won = false;
            parse_metric_bool(observation, "selected_won", selected_won);
            if (selected_won) {
                ++planner_selected_won_count;
            }

            const std::string workload = Metrics::find_value(observation, "workload");
            const std::string winning_backend = Metrics::find_value(observation, "winning_backend");
            if (!workload.empty() && !winning_backend.empty()) {
                winner_counts[sanitize_metric_token(workload) + "_" + sanitize_metric_token(winning_backend)] += 1;
            }

            if (!selected_won) {
                const std::string selected_backend = Metrics::find_value(observation, "selected_backend");
                double selected_median_ms = 0.0;
                double winning_median_ms = 0.0;
                if (parse_metric_double(observation, selected_backend + "_median_ms", selected_median_ms)
                    && parse_metric_double(observation, winning_backend + "_median_ms", winning_median_ms)
                    && winning_median_ms > 0.0) {
                    ReportRecord mismatch;
                    mismatch.path = path;
                    mismatch.workload = workload;
                    mismatch.dataset = Metrics::find_value(observation, "dataset_input_path");
                    mismatch.selected_backend = selected_backend;
                    mismatch.winning_backend = winning_backend;
                    mismatch.selected_regret_vs_winner = selected_median_ms / winning_median_ms;
                    mismatch.has_selected_regret = true;
                    if (!have_worst_regret || mismatch.selected_regret_vs_winner > worst_regret_record.selected_regret_vs_winner) {
                        worst_regret_record = std::move(mismatch);
                        have_worst_regret = true;
                    }
                }
            }
        }
    }

    const double planner_accuracy = planner_observation_count == 0
        ? 0.0
        : static_cast<double>(planner_selected_won_count) / static_cast<double>(planner_observation_count);

    std::cout << "[report] results=" << records.size()
              << " planner_logs=" << planner_log_paths.size()
              << " planner_observations=" << planner_observation_count << "\n";
    for (const auto& record : records) {
        std::cout << "[report] workload=" << record.workload
                  << " dataset=" << record.dataset
                  << " backend=" << record.selected_backend
                  << " median_ms=" << format_double(record.median_ms);
        if (record.has_selected_speedup) {
            std::cout << " speedup_vs_scalar=" << format_double(record.selected_speedup_vs_scalar);
        }
        if (record.has_selected_effective_gflops) {
            std::cout << " effective_gflops=" << format_double(record.selected_effective_gflops);
        }
        if (record.has_selected_effective_bandwidth) {
            std::cout << " effective_bandwidth_gbps=" << format_double(record.selected_effective_bandwidth_gbps);
        }
        if (record.has_avx2_speedup && record.avx2_speedup_vs_scalar < 1.0) {
            std::cout << " avx2_regressed_vs_scalar=true";
        }
        std::cout << "\n";
    }

    if (planner_observation_count > 0) {
        std::cout << "[report] planner_accuracy=" << format_double(planner_accuracy)
                  << " selected_won=" << planner_selected_won_count
                  << " observations=" << planner_observation_count << "\n";
    }
    if (have_best_overall) {
        std::cout << "[report] best_speedup_vs_scalar=" << format_double(best_overall_record.best_speedup_vs_scalar)
                  << " workload=" << best_overall_record.workload
                  << " dataset=" << best_overall_record.dataset
                  << " source=" << best_overall_record.path << "\n";
    }
    if (have_best_real_data) {
        std::cout << "[report] best_real_data_speedup_vs_scalar=" << format_double(best_real_data_record.best_speedup_vs_scalar)
                  << " workload=" << best_real_data_record.workload
                  << " dataset=" << best_real_data_record.dataset
                  << " source=" << best_real_data_record.path << "\n";
    }
    if (have_worst_regret) {
        std::cout << "[report] worst_planner_regret_vs_winner=" << format_double(worst_regret_record.selected_regret_vs_winner)
                  << " selected_backend=" << worst_regret_record.selected_backend
                  << " winning_backend=" << worst_regret_record.winning_backend
                  << " source=" << worst_regret_record.path << "\n";
    }

    std::vector<Metrics::MetricEntry> report = Metrics::collect_host_metadata();
    append_entries(report, {
        {"result_kind", "report"},
        {"command_line", build_command_line("report", "", args)},
        {"result_count", std::to_string(records.size())},
        {"real_data_result_count", std::to_string(real_data_result_count)},
        {"planner_log_count", std::to_string(planner_log_paths.size())},
        {"planner_observation_count", std::to_string(planner_observation_count)},
        {"planner_selected_won_count", std::to_string(planner_selected_won_count)},
        {"planner_accuracy", format_double(planner_accuracy)},
        {"cuda_available_in_any_result", any_cuda_available ? "true" : "false"},
    });

    if (have_best_overall) {
        append_entries(report, {
            {"best_speedup_vs_scalar", format_double(best_overall_record.best_speedup_vs_scalar)},
            {"best_speedup_workload", best_overall_record.workload},
            {"best_speedup_dataset", best_overall_record.dataset},
            {"best_speedup_source", best_overall_record.path},
        });
    }
    if (have_best_real_data) {
        append_entries(report, {
            {"best_real_data_speedup_vs_scalar", format_double(best_real_data_record.best_speedup_vs_scalar)},
            {"best_real_data_workload", best_real_data_record.workload},
            {"best_real_data_dataset", best_real_data_record.dataset},
            {"best_real_data_source", best_real_data_record.path},
        });
    }
    if (have_worst_regret) {
        append_entries(report, {
            {"worst_planner_regret_vs_winner", format_double(worst_regret_record.selected_regret_vs_winner)},
            {"worst_planner_regret_workload", worst_regret_record.workload},
            {"worst_planner_regret_dataset", worst_regret_record.dataset},
            {"worst_planner_regret_selected_backend", worst_regret_record.selected_backend},
            {"worst_planner_regret_winning_backend", worst_regret_record.winning_backend},
            {"worst_planner_regret_source", worst_regret_record.path},
        });
    }

    for (size_t index = 0; index < records.size(); ++index) {
        const std::string prefix = "result_" + std::to_string(index) + "_";
        const auto& record = records[index];
        append_entries(report, {
            {prefix + "path", record.path},
            {prefix + "workload", record.workload},
            {prefix + "dataset", record.dataset},
            {prefix + "selected_backend", record.selected_backend},
            {prefix + "winning_backend", record.winning_backend},
            {prefix + "planner_strategy", record.planner_strategy},
            {prefix + "planner_reason", record.planner_reason},
            {prefix + "median_ms", format_double(record.median_ms)},
            {prefix + "cuda_compiled", record.cuda_compiled ? "true" : "false"},
            {prefix + "cuda_runtime_available", record.cuda_runtime_available ? "true" : "false"},
            {prefix + "cuda_reason", record.cuda_reason},
        });

        if (record.has_selected_speedup) {
            report.push_back({prefix + "selected_speedup_vs_scalar", format_double(record.selected_speedup_vs_scalar)});
        }
        if (record.has_selected_won) {
            report.push_back({prefix + "selected_won", record.selected_won ? "true" : "false"});
        }
        if (record.has_selected_regret) {
            report.push_back({prefix + "selected_regret_vs_winner", format_double(record.selected_regret_vs_winner)});
        }
        if (record.has_selected_effective_gflops) {
            report.push_back({prefix + "selected_effective_gflops", format_double(record.selected_effective_gflops)});
        }
        if (record.has_selected_effective_bandwidth) {
            report.push_back({prefix + "selected_effective_bandwidth_gbps", format_double(record.selected_effective_bandwidth_gbps)});
        }
        if (record.has_avx2_speedup) {
            report.push_back({prefix + "avx2_speedup_vs_scalar", format_double(record.avx2_speedup_vs_scalar)});
        }
        if (record.has_threaded_speedup) {
            report.push_back({prefix + "threaded_speedup_vs_scalar", format_double(record.threaded_speedup_vs_scalar)});
        }
    }

    for (const auto& winner : winner_counts) {
        report.push_back({"planner_winner_" + winner.first + "_count", std::to_string(winner.second)});
    }

    const std::string markdown_path = get_option(args, "--markdown", get_option(args, "--md"));
    if (!markdown_path.empty()) {
        std::ostringstream markdown;
        markdown << "# Helios Proof Report\n\n";
        markdown << "Generated: " << Metrics::utc_now_iso8601() << "\n\n";
        markdown << "## Summary\n\n";
        markdown << "- Results analyzed: " << records.size() << "\n";
        markdown << "- Real-data results analyzed: " << real_data_result_count << "\n";
        markdown << "- Planner observations: " << planner_observation_count << "\n";
        markdown << "- Planner selected-won accuracy: "
                 << format_double(planner_accuracy * 100.0, 2) << "%";
        if (planner_observation_count > 0) {
            markdown << " (" << planner_selected_won_count << "/" << planner_observation_count << ")";
        }
        markdown << "\n\n";
        markdown << "## Strongest Proof Points\n\n";
        if (have_best_overall) {
            markdown << "- Best observed speedup versus scalar: "
                     << format_double(best_overall_record.best_speedup_vs_scalar) << "x"
                     << " on [" << best_overall_record.workload << "] "
                     << best_overall_record.dataset << " from " << best_overall_record.path << "\n";
        }
        if (have_best_real_data) {
            markdown << "- Best real-data speedup versus scalar: "
                     << format_double(best_real_data_record.best_speedup_vs_scalar) << "x"
                     << " on [" << best_real_data_record.workload << "] "
                     << best_real_data_record.dataset << " from " << best_real_data_record.path << "\n";
        }
        if (!any_cuda_available) {
            markdown << "- CUDA proof remains blocked in these artifacts because every scanned result reported cuda_runtime_available=false.\n";
        }
        if (have_worst_regret) {
            markdown << "- Largest planner miss observed: "
                     << format_double(worst_regret_record.selected_regret_vs_winner) << "x regret"
                     << " with selected=" << worst_regret_record.selected_backend
                     << " and winner=" << worst_regret_record.winning_backend
                     << " from " << worst_regret_record.path << "\n";
        }
        markdown << "\n";
        markdown << "## Result Highlights\n\n";
        for (const auto& record : records) {
            markdown << "- [" << record.workload << "] " << record.dataset
                     << " -> backend=" << record.selected_backend
                     << ", median_ms=" << format_double(record.median_ms);
            if (record.has_selected_speedup) {
                markdown << ", speedup_vs_scalar=" << format_double(record.selected_speedup_vs_scalar);
            }
            if (record.has_selected_effective_gflops) {
                markdown << ", effective_gflops=" << format_double(record.selected_effective_gflops);
            }
            if (record.has_selected_effective_bandwidth) {
                markdown << ", effective_bandwidth_gbps=" << format_double(record.selected_effective_bandwidth_gbps);
            }
            if (record.has_selected_won) {
                markdown << ", selected_won=" << (record.selected_won ? "true" : "false");
            }
            if (record.has_selected_regret) {
                markdown << ", regret_vs_winner=" << format_double(record.selected_regret_vs_winner);
            }
            if (record.has_avx2_speedup && record.avx2_speedup_vs_scalar < 1.0) {
                markdown << ", note=avx2_regressed_vs_scalar";
            }
            markdown << "\n";
        }
        markdown << "\n## Planner Winners\n\n";
        if (winner_counts.empty()) {
            markdown << "- No planner observations were provided.\n";
        } else {
            for (const auto& winner : winner_counts) {
                markdown << "- " << winner.first << ": " << winner.second << "\n";
            }
        }
        markdown << "\n## CUDA Status\n\n";
        bool emitted_cuda_status = false;
        for (const auto& record : records) {
            if (!record.cuda_reason.empty() || record.cuda_compiled || record.cuda_runtime_available) {
                markdown << "- From " << record.path
                         << ": cuda_compiled=" << (record.cuda_compiled ? "true" : "false")
                         << ", cuda_runtime_available=" << (record.cuda_runtime_available ? "true" : "false");
                if (!record.cuda_reason.empty()) {
                    markdown << ", note=" << record.cuda_reason;
                }
                markdown << "\n";
                emitted_cuda_status = true;
            }
        }
        if (!emitted_cuda_status) {
            markdown << "- No CUDA status was available in the provided results.\n";
        }

        if (write_text_file(markdown_path, markdown.str())) {
            std::cout << "[output] md=" << markdown_path << "\n";
        } else {
            std::cerr << "Warning: failed to write markdown report to " << markdown_path << "\n";
        }
    }

    emit_outputs(args, report);
    return 0;
}

int handle_compare(const std::vector<std::string>& args) {
    const std::string lhs_path = get_option(args, "--lhs");
    const std::string rhs_path = get_option(args, "--rhs");
    if (lhs_path.empty() || rhs_path.empty()) {
        std::cerr << "Error: compare requires --lhs <result.csv|json> and --rhs <result.csv|json>.\n";
        return 1;
    }

    std::vector<Metrics::MetricEntry> lhs_metrics;
    std::vector<Metrics::MetricEntry> rhs_metrics;
    if (!Metrics::read_metrics(lhs_path, lhs_metrics)) {
        std::cerr << "Error: failed to read metrics from " << lhs_path << "\n";
        return 1;
    }
    if (!Metrics::read_metrics(rhs_path, rhs_metrics)) {
        std::cerr << "Error: failed to read metrics from " << rhs_path << "\n";
        return 1;
    }

    const std::string lhs_median_key = primary_median_key(lhs_metrics);
    const std::string rhs_median_key = primary_median_key(rhs_metrics);
    if (lhs_median_key.empty() || rhs_median_key.empty()) {
        std::cerr << "Error: could not find a primary median metric in one of the result files.\n";
        return 1;
    }

    double lhs_median = 0.0;
    double rhs_median = 0.0;
    if (!parse_metric_double(lhs_metrics, lhs_median_key, lhs_median)
        || !parse_metric_double(rhs_metrics, rhs_median_key, rhs_median)) {
        std::cerr << "Error: failed to parse primary median metrics from result files.\n";
        return 1;
    }

    const std::string lhs_label = Metrics::find_value(lhs_metrics, "selected_backend");
    const std::string rhs_label = Metrics::find_value(rhs_metrics, "selected_backend");
    const std::string lhs_workload = Metrics::find_value(lhs_metrics, "workload");
    const std::string rhs_workload = Metrics::find_value(rhs_metrics, "workload");
    const std::string lhs_dataset = Metrics::find_value(lhs_metrics, "dataset_input_path");
    const std::string rhs_dataset = Metrics::find_value(rhs_metrics, "dataset_input_path");

    const double speedup_rhs_over_lhs = lhs_median / rhs_median;
    const double speedup_lhs_over_rhs = rhs_median / lhs_median;
    const std::string winner =
        std::abs(lhs_median - rhs_median) < 1.0e-12 ? "tie" : (lhs_median < rhs_median ? "lhs" : "rhs");

    std::cout << "[compare] lhs=" << lhs_path
              << " rhs=" << rhs_path << "\n";
    std::cout << "[compare] lhs_workload=" << lhs_workload
              << " rhs_workload=" << rhs_workload << "\n";
    std::cout << "[compare] lhs_dataset=" << lhs_dataset
              << " rhs_dataset=" << rhs_dataset << "\n";
    std::cout << "[compare] lhs_backend=" << lhs_label
              << " rhs_backend=" << rhs_label << "\n";
    std::cout << "[compare] lhs_median_ms=" << format_double(lhs_median)
              << " rhs_median_ms=" << format_double(rhs_median)
              << " winner=" << winner << "\n";
    std::cout << "[compare] rhs_speedup_over_lhs=" << format_double(speedup_rhs_over_lhs)
              << " lhs_speedup_over_rhs=" << format_double(speedup_lhs_over_rhs) << "\n";

    std::vector<Metrics::MetricEntry> comparison = Metrics::collect_host_metadata();
    append_entries(comparison, {
        {"result_kind", "comparison"},
        {"command_line", build_command_line("compare", "", args)},
        {"lhs_path", lhs_path},
        {"rhs_path", rhs_path},
        {"lhs_workload", lhs_workload},
        {"rhs_workload", rhs_workload},
        {"lhs_dataset", lhs_dataset},
        {"rhs_dataset", rhs_dataset},
        {"lhs_backend", lhs_label},
        {"rhs_backend", rhs_label},
        {"lhs_median_key", lhs_median_key},
        {"rhs_median_key", rhs_median_key},
        {"lhs_median_ms", format_double(lhs_median)},
        {"rhs_median_ms", format_double(rhs_median)},
        {"rhs_speedup_over_lhs", format_double(speedup_rhs_over_lhs)},
        {"lhs_speedup_over_rhs", format_double(speedup_lhs_over_rhs)},
        {"winner", winner},
    });
    emit_outputs(args, comparison);
    return 0;
}

} // namespace

Runtime::Runtime() = default;
Runtime::~Runtime() = default;

int Runtime::run(const std::string& command,
                 const std::string& workload,
                 const std::vector<std::string>& args) const {
    if (command == "report") {
        return handle_report(args);
    }
    if (command == "compare") {
        return handle_compare(args);
    }

    DatasetLoader loader;
    Profiler profiler;
    Planner planner;
    const cpu::CapabilityInfo cpu_caps = cpu::capabilities();
    const gpu::CapabilityInfo gpu_caps = gpu::capabilities();

    const std::string requested_path = get_option(args, "--path", "auto");
    ExecutionBackend requested_backend = ExecutionBackend::Auto;
    if (!parse_execution_backend(get_option(args, "--backend", "auto"), requested_backend)) {
        std::cerr << "Error: --backend must be one of auto, scalar, avx2, threaded, cuda, or vendor.\n";
        return 1;
    }

    std::string backend_validation_error;
    if (!validate_backend_request(requested_path, requested_backend, backend_validation_error)) {
        std::cerr << "Error: " << backend_validation_error << "\n";
        return 1;
    }

    if (workload == "sparse") {
        const std::string matrix_path = get_option(args, "--matrix");
        if (matrix_path.empty()) {
            std::cerr << "Error: --matrix <path> is required for sparse workloads.\n";
            return 1;
        }

        SparseMatrix matrix;
        DatasetLoadInfo load_info;
        if (!loader.load_matrix_market(matrix_path, matrix, &load_info)) {
            std::cerr << "Error: failed to load Matrix Market file from " << matrix_path << "\n";
            return 1;
        }

        const WorkloadProfile profile = profiler.profile_matrix(matrix);
        const Planner::Decision decision = planner.select_strategy(profile, requested_path, gpu_caps.runtime_available);

        std::cout << "[dataset] format=matrix_market path=" << matrix_path
                  << " rows=" << matrix.rows
                  << " cols=" << matrix.cols
                  << " nnz=" << matrix.nnz
                  << " field=" << matrix.metadata.field
                  << " symmetry=" << matrix.metadata.symmetry << "\n";
        std::cout << "[dataset] cache_hit=" << (load_info.cache_hit ? "true" : "false")
                  << " cache_path=" << load_info.cache_path << "\n";
        print_profile_summary(profile);
        print_capability_summary(cpu_caps, gpu_caps);
        std::cout << "[planner] selected=" << Planner::to_string(decision.strategy)
                  << " reason=\"" << decision.reason << "\"\n";

        const std::string result_kind = command == "validate" ? "validation" : command;
        auto metrics = make_common_entries(command, workload, result_kind, args, profile, decision, cpu_caps, gpu_caps);
        append_dataset_load_info(metrics, load_info);
        append_entries(metrics, {
            {"matrix_field", matrix.metadata.field},
            {"matrix_symmetry", matrix.metadata.symmetry},
        });

        std::string manifest_path;
        if (maybe_write_manifest(args, load_info, metrics, manifest_path)) {
            metrics.push_back({"dataset_manifest_path", manifest_path});
            std::cout << "[manifest] path=" << manifest_path << "\n";
        }

        if (command == "profile") {
            emit_outputs(args, metrics);
            return 0;
        }

        bool threads_ok = false;
        const size_t threads = get_size_option(args, "--threads", cpu_caps.hardware_threads, &threads_ok);
        if (!threads_ok || threads == 0) {
            std::cerr << "Error: --threads must be a positive integer.\n";
            return 1;
        }

        std::vector<double> x(matrix.cols, 1.0);
        const bool compare_all = has_option(args, "--compare-baselines") || has_option(args, "--compare-all");
        const ExecutionBackend selected_backend = resolve_sparse_backend(requested_backend, decision, profile, cpu_caps, gpu_caps);
        const std::vector<ExecutionBackend> backend_plan =
            build_backend_plan(selected_backend,
                               compare_all,
                               cpu_caps,
                               gpu_caps.runtime_available,
                               gpu_caps.runtime_available && gpu_caps.cusparse_available);

        if (command == "bench") {
            bool warmup_ok = false;
            bool trials_ok = false;
            const size_t warmup = get_size_option(args, "--warmup", 1, &warmup_ok);
            const size_t trials = get_size_option(args, "--trials", 5, &trials_ok);
            if (!warmup_ok || !trials_ok || trials == 0) {
                std::cerr << "Error: --warmup and --trials must be positive integers, with trials > 0.\n";
                return 1;
            }

            std::cout << "[benchmark] warmup=" << warmup
                      << " trials=" << trials
                      << " selected_backend=" << to_string(selected_backend)
                      << " compare_baselines=" << (compare_all ? "true" : "false")
                      << "\n";

            std::vector<BackendRun> runs;
            for (const ExecutionBackend backend : backend_plan) {
                runs.push_back(run_sparse_backend(backend, matrix, x, warmup, trials, threads));
            }

            BackendRun* scalar_run = nullptr;
            BackendRun* selected_run = nullptr;
            for (auto& run : runs) {
                if (run.name == "scalar" && run.executed) {
                    scalar_run = &run;
                }
                if (run.name == to_string(selected_backend)) {
                    selected_run = &run;
                }
            }

            if (selected_run == nullptr || !selected_run->supported || !selected_run->executed) {
                std::cerr << "Error: selected backend " << to_string(selected_backend) << " could not run";
                if (selected_run != nullptr && !selected_run->note.empty()) {
                    std::cerr << ": " << selected_run->note;
                }
                std::cerr << "\n";
                return 1;
            }

            for (const auto& run : runs) {
                print_backend_run(run, scalar_run, workload, profile);
                append_backend_metrics(metrics, run, scalar_run, workload, profile);
            }

            std::cout << "[result] selected_backend=" << selected_run->name
                      << " median_ms=" << format_double(selected_run->summary.median_ms)
                      << " effective_gflops=" << format_double(effective_gflops(profile, selected_run->summary))
                      << " effective_bandwidth_gbps=" << format_double(effective_bandwidth_gbps(profile, selected_run->summary))
                      << " checksum=" << format_double(selected_run->checksum) << "\n";

            append_entries(metrics, {
                {"requested_backend", to_string(requested_backend)},
                {"selected_backend", selected_run->name},
                {"compare_baselines", compare_all ? "true" : "false"},
            });
            append_selection_metrics(metrics, runs, selected_run->name);
            maybe_log_planner_observation(
                args,
                command,
                workload,
                &load_info,
                profile,
                decision,
                selected_run->name,
                runs,
                compare_all);
            emit_outputs(args, metrics);
            return 0;
        }

        bool tolerance_ok = false;
        const double tolerance = get_double_option(args, "--tolerance", 1.0e-9, &tolerance_ok);
        if (!tolerance_ok || tolerance < 0.0) {
            std::cerr << "Error: --tolerance must be a non-negative floating point value.\n";
            return 1;
        }

        std::vector<BackendRun> runs;
        for (const ExecutionBackend backend : backend_plan) {
            runs.push_back(run_sparse_backend(backend, matrix, x, 0, 1, threads));
        }

        BackendRun* scalar_run = nullptr;
        for (auto& run : runs) {
            if (run.name == "scalar" && run.executed) {
                scalar_run = &run;
                break;
            }
        }
        if (scalar_run == nullptr) {
            std::cerr << "Error: scalar baseline did not execute during validation.\n";
            return 1;
        }

        bool validation_passed = true;
        for (auto& run : runs) {
            if (!run.supported) {
                if (run.name == to_string(selected_backend)) {
                    validation_passed = false;
                }
                continue;
            }
            const double diff = run.name == "scalar" ? 0.0 : max_abs_diff(run.output, scalar_run->output);
            const bool pass = run.name == "scalar" || diff <= tolerance;
            run.valid = pass;
            if (!pass) {
                validation_passed = false;
            }
            print_validation_run(run, scalar_run, tolerance);
            append_backend_metrics(metrics, run, scalar_run, workload, profile);
            metrics.push_back({run.name + "_validation_pass", pass ? "true" : "false"});
            if (run.name != "scalar") {
                metrics.push_back({run.name + "_validation_tolerance", format_double(tolerance, 12)});
            }
        }

        append_entries(metrics, {
            {"requested_backend", to_string(requested_backend)},
            {"selected_backend", to_string(selected_backend)},
            {"compare_all", compare_all ? "true" : "false"},
            {"validation_tolerance", format_double(tolerance, 12)},
            {"validation_passed", validation_passed ? "true" : "false"},
        });
        emit_outputs(args, metrics);
        return validation_passed ? 0 : 1;
    }

    if (workload == "graph") {
        const std::string graph_path = get_option(args, "--graph");
        if (graph_path.empty()) {
            std::cerr << "Error: --graph <path> is required for graph workloads.\n";
            return 1;
        }

        GraphData graph;
        DatasetLoadInfo load_info;
        if (!loader.load_snap_edge_list(graph_path, graph, &load_info)) {
            std::cerr << "Error: failed to load SNAP edge list from " << graph_path << "\n";
            return 1;
        }

        const bool force_directed = has_option(args, "--directed");
        const bool force_undirected = has_option(args, "--undirected");
        if (force_directed && force_undirected) {
            std::cerr << "Error: use only one of --directed or --undirected.\n";
            return 1;
        }
        if (force_directed || force_undirected) {
            loader.rebuild_graph_storage(graph, force_directed);
        }

        const std::string algorithm = get_option(args, "--algo", "bfs");
        if (algorithm != "bfs" && algorithm != "pagerank") {
            std::cerr << "Error: --algo must be bfs or pagerank.\n";
            return 1;
        }

        const WorkloadProfile profile = profiler.profile_graph(graph);
        const Planner::Decision decision = planner.select_strategy(profile, requested_path, gpu_caps.runtime_available);

        std::cout << "[dataset] format=snap_edge_list path=" << graph_path
                  << " nodes=" << graph.node_count
                  << " edges=" << graph.edge_count
                  << " stored_edges=" << graph.stored_edge_count
                  << " directed=" << (graph.directed ? "true" : "false") << "\n";
        std::cout << "[dataset] cache_hit=" << (load_info.cache_hit ? "true" : "false")
                  << " cache_path=" << load_info.cache_path << "\n";
        std::cout << "[algorithm] name=" << algorithm << "\n";
        print_profile_summary(profile);
        print_capability_summary(cpu_caps, gpu_caps);
        std::cout << "[planner] selected=" << Planner::to_string(decision.strategy)
                  << " reason=\"" << decision.reason << "\"\n";

        const std::string result_kind = command == "validate" ? "validation" : command;
        auto metrics = make_common_entries(command, workload, result_kind, args, profile, decision, cpu_caps, gpu_caps);
        append_dataset_load_info(metrics, load_info);
        append_entries(metrics, {
            {"algorithm", algorithm},
            {"directed", graph.directed ? "true" : "false"},
            {"edge_count", std::to_string(graph.edge_count)},
            {"stored_edge_count", std::to_string(graph.stored_edge_count)},
        });

        std::string manifest_path;
        if (maybe_write_manifest(args, load_info, metrics, manifest_path)) {
            metrics.push_back({"dataset_manifest_path", manifest_path});
            std::cout << "[manifest] path=" << manifest_path << "\n";
        }

        if (command == "profile") {
            emit_outputs(args, metrics);
            return 0;
        }

        if (command == "bench") {
            bool warmup_ok = false;
            bool trials_ok = false;
            const size_t warmup = get_size_option(args, "--warmup", 1, &warmup_ok);
            const size_t trials = get_size_option(args, "--trials", 5, &trials_ok);
            if (!warmup_ok || !trials_ok || trials == 0) {
                std::cerr << "Error: --warmup and --trials must be positive integers, with trials > 0.\n";
                return 1;
            }

            std::cout << "[benchmark] warmup=" << warmup << " trials=" << trials << " backend=scalar\n";
            metrics.push_back({"selected_backend", "scalar"});

            if (algorithm == "bfs") {
                bool source_ok = false;
                const size_t source = get_size_option(args, "--source", 0, &source_ok);
                if (!source_ok || source >= graph.node_count) {
                    std::cerr << "Error: --source must be a valid node id within the graph.\n";
                    return 1;
                }

                std::vector<int> distances;
                const std::vector<double> samples_ms = benchmark_kernel(warmup, trials, [&]() {
                    cpu::bfs_reference(graph, source, distances);
                });
                const Metrics::BenchmarkSummary summary = Metrics::summarize(samples_ms);
                const size_t reached_nodes = count_reached_nodes(distances);
                const double traversed_edges_per_second =
                    summary_seconds(summary) <= 0.0
                    ? 0.0
                    : static_cast<double>(graph.stored_edge_count) / summary_seconds(summary);

                std::cout << "[result] min_ms=" << format_double(summary.min_ms) << "\n";
                std::cout << "[result] median_ms=" << format_double(summary.median_ms) << "\n";
                std::cout << "[result] p95_ms=" << format_double(summary.p95_ms) << "\n";
                std::cout << "[result] max_ms=" << format_double(summary.max_ms) << "\n";
                std::cout << "[result] mean_ms=" << format_double(summary.mean_ms) << "\n";
                std::cout << "[result] stddev_ms=" << format_double(summary.stddev_ms) << "\n";
                std::cout << "[result] cv_pct=" << format_double(summary.cv_pct) << "\n";
                std::cout << "[result] teps=" << format_double(traversed_edges_per_second, 2) << "\n";
                std::cout << "[result] reached_nodes=" << reached_nodes << "\n";

                append_entries(metrics, {
                    {"source", std::to_string(source)},
                    {"trials", std::to_string(summary.trials)},
                    {"min_ms", format_double(summary.min_ms)},
                    {"median_ms", format_double(summary.median_ms)},
                    {"p95_ms", format_double(summary.p95_ms)},
                    {"max_ms", format_double(summary.max_ms)},
                    {"mean_ms", format_double(summary.mean_ms)},
                    {"stddev_ms", format_double(summary.stddev_ms)},
                    {"cv_pct", format_double(summary.cv_pct)},
                    {"teps", format_double(traversed_edges_per_second, 2)},
                    {"reached_nodes", std::to_string(reached_nodes)},
                });
            } else {
                bool iterations_ok = false;
                const size_t iterations = get_size_option(args, "--iterations", 20, &iterations_ok);
                if (!iterations_ok || iterations == 0) {
                    std::cerr << "Error: --iterations must be a positive integer.\n";
                    return 1;
                }

                std::vector<double> ranks;
                const std::vector<double> samples_ms = benchmark_kernel(warmup, trials, [&]() {
                    cpu::pagerank_reference(graph, iterations, 0.85, ranks);
                });
                const Metrics::BenchmarkSummary summary = Metrics::summarize(samples_ms);
                const double rank_sum = std::accumulate(ranks.begin(), ranks.end(), 0.0);
                const double edges_processed_per_second =
                    summary_seconds(summary) <= 0.0
                    ? 0.0
                    : (static_cast<double>(graph.stored_edge_count) * static_cast<double>(iterations)) / summary_seconds(summary);

                std::cout << "[result] min_ms=" << format_double(summary.min_ms) << "\n";
                std::cout << "[result] median_ms=" << format_double(summary.median_ms) << "\n";
                std::cout << "[result] p95_ms=" << format_double(summary.p95_ms) << "\n";
                std::cout << "[result] max_ms=" << format_double(summary.max_ms) << "\n";
                std::cout << "[result] mean_ms=" << format_double(summary.mean_ms) << "\n";
                std::cout << "[result] stddev_ms=" << format_double(summary.stddev_ms) << "\n";
                std::cout << "[result] cv_pct=" << format_double(summary.cv_pct) << "\n";
                std::cout << "[result] edge_updates_per_second=" << format_double(edges_processed_per_second, 2) << "\n";
                std::cout << "[result] rank_sum=" << format_double(rank_sum) << "\n";

                append_entries(metrics, {
                    {"iterations", std::to_string(iterations)},
                    {"trials", std::to_string(summary.trials)},
                    {"min_ms", format_double(summary.min_ms)},
                    {"median_ms", format_double(summary.median_ms)},
                    {"p95_ms", format_double(summary.p95_ms)},
                    {"max_ms", format_double(summary.max_ms)},
                    {"mean_ms", format_double(summary.mean_ms)},
                    {"stddev_ms", format_double(summary.stddev_ms)},
                    {"cv_pct", format_double(summary.cv_pct)},
                    {"edge_updates_per_second", format_double(edges_processed_per_second, 2)},
                    {"rank_sum", format_double(rank_sum)},
                });
            }

            emit_outputs(args, metrics);
            return 0;
        }

        bool tolerance_ok = false;
        const double tolerance = get_double_option(args, "--tolerance", 1.0e-9, &tolerance_ok);
        if (!tolerance_ok || tolerance < 0.0) {
            std::cerr << "Error: --tolerance must be a non-negative floating point value.\n";
            return 1;
        }

        if (algorithm == "bfs") {
            bool source_ok = false;
            const size_t source = get_size_option(args, "--source", 0, &source_ok);
            if (!source_ok || source >= graph.node_count) {
                std::cerr << "Error: --source must be a valid node id within the graph.\n";
                return 1;
            }

            std::vector<int> distances;
            cpu::bfs_reference(graph, source, distances);
            const bool pass = !distances.empty() && distances[source] == 0 && count_reached_nodes(distances) <= graph.node_count;
            std::cout << "[validation] backend=scalar status=" << (pass ? "pass" : "fail")
                      << " reached_nodes=" << count_reached_nodes(distances) << "\n";
            append_entries(metrics, {
                {"source", std::to_string(source)},
                {"selected_backend", "scalar"},
                {"validation_passed", pass ? "true" : "false"},
                {"reached_nodes", std::to_string(count_reached_nodes(distances))},
            });
            emit_outputs(args, metrics);
            return pass ? 0 : 1;
        }

        bool iterations_ok = false;
        const size_t iterations = get_size_option(args, "--iterations", 20, &iterations_ok);
        if (!iterations_ok || iterations == 0) {
            std::cerr << "Error: --iterations must be a positive integer.\n";
            return 1;
        }

        std::vector<double> ranks;
        cpu::pagerank_reference(graph, iterations, 0.85, ranks);
        const double rank_sum = std::accumulate(ranks.begin(), ranks.end(), 0.0);
        const bool pass = std::abs(rank_sum - 1.0) <= std::max(1.0e-6, tolerance * 10.0);
        std::cout << "[validation] backend=scalar status=" << (pass ? "pass" : "fail")
                  << " rank_sum=" << format_double(rank_sum) << "\n";
        append_entries(metrics, {
            {"iterations", std::to_string(iterations)},
            {"selected_backend", "scalar"},
            {"validation_passed", pass ? "true" : "false"},
            {"rank_sum", format_double(rank_sum)},
        });
        emit_outputs(args, metrics);
        return pass ? 0 : 1;
    }

    if (workload == "dense") {
        bool rows_ok = false;
        bool cols_ok = false;
        bool depth_ok = false;
        const size_t rows = get_size_option(args, "--m", 0, &rows_ok);
        const size_t cols = get_size_option(args, "--n", 0, &cols_ok);
        const size_t depth = get_size_option(args, "--k", 0, &depth_ok);
        if (!rows_ok || !cols_ok || !depth_ok || rows == 0 || cols == 0 || depth == 0) {
            std::cerr << "Error: --m, --n, and --k are required for dense workloads and must be > 0.\n";
            return 1;
        }

        const WorkloadProfile profile = profiler.profile_dense(rows, cols, depth);
        const Planner::Decision decision = planner.select_strategy(profile, requested_path, gpu_caps.runtime_available);

        std::cout << "[dataset] format=synthetic_dense_sanity m=" << rows
                  << " n=" << cols
                  << " k=" << depth << "\n";
        print_profile_summary(profile);
        print_capability_summary(cpu_caps, gpu_caps);
        std::cout << "[planner] selected=" << Planner::to_string(decision.strategy)
                  << " reason=\"" << decision.reason << "\"\n";

        const std::string result_kind = command == "validate" ? "validation" : command;
        auto metrics = make_common_entries(command, workload, result_kind, args, profile, decision, cpu_caps, gpu_caps);
        metrics.push_back({"note", "dense path remains synthetic sanity coverage and is not used for real-data claims"});

        if (command == "profile") {
            emit_outputs(args, metrics);
            return 0;
        }

        bool threads_ok = false;
        const size_t threads = get_size_option(args, "--threads", cpu_caps.hardware_threads, &threads_ok);
        if (!threads_ok || threads == 0) {
            std::cerr << "Error: --threads must be a positive integer.\n";
            return 1;
        }

        const bool compare_all = has_option(args, "--compare-baselines") || has_option(args, "--compare-all");
        const ExecutionBackend selected_backend = resolve_dense_backend(requested_backend, decision, profile, cpu_caps, gpu_caps);
        const std::vector<ExecutionBackend> backend_plan =
            build_backend_plan(selected_backend,
                               compare_all,
                               cpu_caps,
                               gpu_caps.runtime_available,
                               gpu_caps.runtime_available && gpu_caps.cublas_available);
        std::vector<double> A(rows * depth, 1.0);
        std::vector<double> B(depth * cols, 1.0);

        if (command == "bench") {
            bool warmup_ok = false;
            bool trials_ok = false;
            const size_t warmup = get_size_option(args, "--warmup", 1, &warmup_ok);
            const size_t trials = get_size_option(args, "--trials", 3, &trials_ok);
            if (!warmup_ok || !trials_ok || trials == 0) {
                std::cerr << "Error: --warmup and --trials must be positive integers, with trials > 0.\n";
                return 1;
            }

            std::cout << "[benchmark] warmup=" << warmup
                      << " trials=" << trials
                      << " selected_backend=" << to_string(selected_backend)
                      << " compare_baselines=" << (compare_all ? "true" : "false")
                      << "\n";

            std::vector<BackendRun> runs;
            for (const ExecutionBackend backend : backend_plan) {
                runs.push_back(run_dense_backend(backend, A, B, rows, cols, depth, warmup, trials, threads));
            }

            BackendRun* scalar_run = nullptr;
            BackendRun* selected_run = nullptr;
            for (auto& run : runs) {
                if (run.name == "scalar" && run.executed) {
                    scalar_run = &run;
                }
                if (run.name == to_string(selected_backend)) {
                    selected_run = &run;
                }
            }

            if (selected_run == nullptr || !selected_run->supported || !selected_run->executed) {
                std::cerr << "Error: selected backend " << to_string(selected_backend) << " could not run";
                if (selected_run != nullptr && !selected_run->note.empty()) {
                    std::cerr << ": " << selected_run->note;
                }
                std::cerr << "\n";
                return 1;
            }

            for (const auto& run : runs) {
                print_backend_run(run, scalar_run, workload, profile);
                append_backend_metrics(metrics, run, scalar_run, workload, profile);
            }

            std::cout << "[result] selected_backend=" << selected_run->name
                      << " median_ms=" << format_double(selected_run->summary.median_ms)
                      << " effective_gflops=" << format_double(effective_gflops(profile, selected_run->summary))
                      << " effective_bandwidth_gbps=" << format_double(effective_bandwidth_gbps(profile, selected_run->summary))
                      << " checksum=" << format_double(selected_run->checksum) << "\n";

            append_entries(metrics, {
                {"requested_backend", to_string(requested_backend)},
                {"selected_backend", selected_run->name},
                {"compare_baselines", compare_all ? "true" : "false"},
            });
            append_selection_metrics(metrics, runs, selected_run->name);
            maybe_log_planner_observation(
                args,
                command,
                workload,
                nullptr,
                profile,
                decision,
                selected_run->name,
                runs,
                compare_all);
            emit_outputs(args, metrics);
            return 0;
        }

        bool tolerance_ok = false;
        const double tolerance = get_double_option(args, "--tolerance", 1.0e-9, &tolerance_ok);
        if (!tolerance_ok || tolerance < 0.0) {
            std::cerr << "Error: --tolerance must be a non-negative floating point value.\n";
            return 1;
        }

        std::vector<BackendRun> runs;
        for (const ExecutionBackend backend : backend_plan) {
            runs.push_back(run_dense_backend(backend, A, B, rows, cols, depth, 0, 1, threads));
        }

        BackendRun* scalar_run = nullptr;
        for (auto& run : runs) {
            if (run.name == "scalar" && run.executed) {
                scalar_run = &run;
                break;
            }
        }
        if (scalar_run == nullptr) {
            std::cerr << "Error: scalar baseline did not execute during validation.\n";
            return 1;
        }

        bool validation_passed = true;
        for (auto& run : runs) {
            if (!run.supported) {
                if (run.name == to_string(selected_backend)) {
                    validation_passed = false;
                }
                continue;
            }
            const double diff = run.name == "scalar" ? 0.0 : max_abs_diff(run.output, scalar_run->output);
            const bool pass = run.name == "scalar" || diff <= tolerance;
            run.valid = pass;
            if (!pass) {
                validation_passed = false;
            }
            print_validation_run(run, scalar_run, tolerance);
            append_backend_metrics(metrics, run, scalar_run, workload, profile);
            metrics.push_back({run.name + "_validation_pass", pass ? "true" : "false"});
            if (run.name != "scalar") {
                metrics.push_back({run.name + "_validation_tolerance", format_double(tolerance, 12)});
            }
        }

        append_entries(metrics, {
            {"requested_backend", to_string(requested_backend)},
            {"selected_backend", to_string(selected_backend)},
            {"compare_all", compare_all ? "true" : "false"},
            {"validation_tolerance", format_double(tolerance, 12)},
            {"validation_passed", validation_passed ? "true" : "false"},
        });
        emit_outputs(args, metrics);
        return validation_passed ? 0 : 1;
    }

    std::cerr << "Error: unsupported workload type: " << workload << "\n";
    return 1;
}

} // namespace Helios
