#pragma once

#include <string>
#include <vector>

namespace Helios {

class Metrics {
public:
    struct MetricEntry {
        std::string name;
        std::string value;
    };

    struct BenchmarkSummary {
        double min_ms = 0.0;
        double median_ms = 0.0;
        double p95_ms = 0.0;
        double max_ms = 0.0;
        double mean_ms = 0.0;
        double stddev_ms = 0.0;
        double cv_pct = 0.0;
        size_t trials = 0;
    };

    static double now();
    static std::string utc_now_iso8601();
    static BenchmarkSummary summarize(const std::vector<double>& samples_ms);
    static std::vector<MetricEntry> collect_host_metadata();
    static bool write_csv(const std::string& path, const std::vector<MetricEntry>& entries);
    static bool write_json(const std::string& path, const std::vector<MetricEntry>& entries);
    static bool append_jsonl(const std::string& path, const std::vector<MetricEntry>& entries);
    static bool read_csv(const std::string& path, std::vector<MetricEntry>& entries);
    static bool read_json(const std::string& path, std::vector<MetricEntry>& entries);
    static bool read_jsonl(const std::string& path, std::vector<std::vector<MetricEntry>>& records);
    static bool read_metrics(const std::string& path, std::vector<MetricEntry>& entries);
    static std::string find_value(const std::vector<MetricEntry>& entries, const std::string& key);
};

} // namespace Helios
