#include "helios/metrics.h"
#include "helios/runtime.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_temp_file(const std::filesystem::path& directory,
                                      const std::string& name,
                                      const std::string& contents) {
    const std::filesystem::path path = directory / name;
    std::ofstream output(path);
    assert(output.good());
    output << contents;
    output.close();
    return path;
}

std::string require_metric(const std::vector<Helios::Metrics::MetricEntry>& entries,
                           const std::string& key) {
    const std::string value = Helios::Metrics::find_value(entries, key);
    assert(!value.empty());
    return value;
}

} // namespace

int main() {
    const std::filesystem::path temp_root =
        std::filesystem::temp_directory_path() / "helios_runtime_command_tests";
    std::filesystem::remove_all(temp_root);
    std::filesystem::create_directories(temp_root);

    const std::filesystem::path matrix_path = write_temp_file(
        temp_root,
        "runtime_matrix.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "2 3 2.0\n"
        "3 1 3.0\n"
        "3 3 4.0\n");

    const std::filesystem::path graph_path = write_temp_file(
        temp_root,
        "runtime_graph.txt",
        "# Undirected graph\n"
        "0 1\n"
        "1 2\n");

    Helios::Runtime runtime;

    const std::filesystem::path sparse_profile_json = temp_root / "sparse_profile.json";
    const std::filesystem::path sparse_profile_csv = temp_root / "sparse_profile.csv";
    const std::filesystem::path sparse_manifest_json = temp_root / "sparse_manifest.json";
    assert(runtime.run(
               "profile",
               "sparse",
               {
                   "--matrix", matrix_path.string(),
                   "--csv", sparse_profile_csv.string(),
                   "--json", sparse_profile_json.string(),
                   "--manifest", sparse_manifest_json.string(),
               })
           == 0);
    assert(std::filesystem::exists(sparse_profile_json));
    assert(std::filesystem::exists(sparse_profile_csv));
    assert(std::filesystem::exists(sparse_manifest_json));

    std::vector<Helios::Metrics::MetricEntry> sparse_profile_metrics;
    assert(Helios::Metrics::read_json(sparse_profile_json.string(), sparse_profile_metrics));
    assert(require_metric(sparse_profile_metrics, "result_kind") == "profile");
    assert(!require_metric(sparse_profile_metrics, "dataset_cache_path").empty());

    const std::filesystem::path sparse_validate_json = temp_root / "sparse_validate.json";
    assert(runtime.run(
               "validate",
               "sparse",
               {
                   "--matrix", matrix_path.string(),
                   "--compare-all",
                   "--threads", "2",
                   "--tolerance", "1e-9",
                   "--json", sparse_validate_json.string(),
               })
           == 0);
    std::vector<Helios::Metrics::MetricEntry> sparse_validate_metrics;
    assert(Helios::Metrics::read_json(sparse_validate_json.string(), sparse_validate_metrics));
    assert(require_metric(sparse_validate_metrics, "validation_passed") == "true");

    const std::filesystem::path graph_validate_json = temp_root / "graph_validate.json";
    assert(runtime.run(
               "validate",
               "graph",
               {
                   "--graph", graph_path.string(),
                   "--algo", "pagerank",
                   "--iterations", "5",
                   "--manifest", (temp_root / "graph_manifest.json").string(),
                   "--json", graph_validate_json.string(),
               })
           == 0);
    std::vector<Helios::Metrics::MetricEntry> graph_validate_metrics;
    assert(Helios::Metrics::read_json(graph_validate_json.string(), graph_validate_metrics));
    assert(require_metric(graph_validate_metrics, "validation_passed") == "true");

    const std::filesystem::path dense_scalar_json = temp_root / "dense_scalar.json";
    const std::filesystem::path dense_threaded_json = temp_root / "dense_threaded.json";
    const std::filesystem::path planner_log_jsonl = temp_root / "planner_observations.jsonl";
    assert(runtime.run(
               "bench",
               "dense",
               {
                   "--m", "32",
                   "--n", "32",
                   "--k", "32",
                   "--backend", "scalar",
                   "--warmup", "0",
                   "--trials", "1",
                   "--json", dense_scalar_json.string(),
               })
           == 0);
    assert(runtime.run(
               "bench",
               "dense",
               {
                   "--m", "32",
                   "--n", "32",
                   "--k", "32",
                   "--backend", "threaded",
                   "--threads", "2",
                    "--compare-baselines",
                    "--planner-log", planner_log_jsonl.string(),
                    "--warmup", "0",
                    "--trials", "1",
                    "--json", dense_threaded_json.string(),
                })
           == 0);
    assert(std::filesystem::exists(planner_log_jsonl));

    const std::filesystem::path compare_json = temp_root / "compare.json";
    assert(runtime.run(
               "compare",
               "",
               {
                   "--lhs", dense_scalar_json.string(),
                   "--rhs", dense_threaded_json.string(),
                   "--json", compare_json.string(),
               })
           == 0);
    std::vector<Helios::Metrics::MetricEntry> compare_metrics;
    assert(Helios::Metrics::read_json(compare_json.string(), compare_metrics));
    assert(require_metric(compare_metrics, "result_kind") == "comparison");
    assert(!require_metric(compare_metrics, "winner").empty());

    const std::filesystem::path report_json = temp_root / "report.json";
    const std::filesystem::path report_md = temp_root / "report.md";
    assert(runtime.run(
               "report",
               "",
               {
                   "--result", dense_scalar_json.string(),
                   "--result", dense_threaded_json.string(),
                   "--planner-log", planner_log_jsonl.string(),
                   "--json", report_json.string(),
                   "--md", report_md.string(),
               })
           == 0);
    assert(std::filesystem::exists(report_md));
    std::vector<Helios::Metrics::MetricEntry> report_metrics;
    assert(Helios::Metrics::read_json(report_json.string(), report_metrics));
    assert(require_metric(report_metrics, "result_kind") == "report");
    assert(require_metric(report_metrics, "result_count") == "2");
    assert(require_metric(report_metrics, "real_data_result_count") == "0");
    assert(!require_metric(report_metrics, "planner_observation_count").empty());

    std::ifstream planner_log_input(planner_log_jsonl);
    assert(planner_log_input.good());
    std::string planner_log_line;
    std::getline(planner_log_input, planner_log_line);
    assert(planner_log_line.find("\"winning_backend\"") != std::string::npos);

    std::filesystem::remove_all(temp_root);
    std::cout << "Runtime command integration tests passed.\n";
    return 0;
}
