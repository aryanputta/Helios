#include "helios/metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace Helios {

namespace {

bool ensure_parent_directory(const std::string& path) {
    const std::filesystem::path filesystem_path(path);
    if (!filesystem_path.has_parent_path()) {
        return true;
    }
    std::error_code error;
    std::filesystem::create_directories(filesystem_path.parent_path(), error);
    return !error;
}

double percentile(const std::vector<double>& sorted_values, double p) {
    if (sorted_values.empty()) {
        return 0.0;
    }
    const double scaled_index = p * static_cast<double>(sorted_values.size() - 1);
    const size_t lower_index = static_cast<size_t>(scaled_index);
    const size_t upper_index = std::min(lower_index + 1, sorted_values.size() - 1);
    const double fraction = scaled_index - static_cast<double>(lower_index);
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
        case '\\':
            escaped << "\\\\";
            break;
        case '"':
            escaped << "\\\"";
            break;
        case '\n':
            escaped << "\\n";
            break;
        default:
            escaped << character;
            break;
        }
    }
    return escaped.str();
}

std::string trim(std::string text) {
    auto is_space = [](unsigned char character) {
        return std::isspace(character) != 0;
    };

    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [&](unsigned char c) { return !is_space(c); }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [&](unsigned char c) { return !is_space(c); }).base(), text.end());
    return text;
}

std::string unquote(std::string text);

bool parse_json_object_text(const std::string& text, std::vector<Helios::Metrics::MetricEntry>& entries) {
    entries.clear();

    std::string body = trim(text);
    if (!body.empty() && body.front() == '{') {
        body.erase(body.begin());
    }
    body = trim(std::move(body));
    if (!body.empty() && body.back() == '}') {
        body.pop_back();
    }
    body = trim(std::move(body));
    if (body.empty()) {
        return true;
    }

    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;
    bool escaping = false;
    for (const char character : body) {
        if (escaping) {
            current.push_back(character);
            escaping = false;
            continue;
        }

        if (character == '\\') {
            current.push_back(character);
            escaping = true;
            continue;
        }
        if (character == '"') {
            current.push_back(character);
            in_quotes = !in_quotes;
            continue;
        }
        if (character == ',' && !in_quotes) {
            fields.push_back(trim(current));
            current.clear();
            continue;
        }
        current.push_back(character);
    }
    if (!current.empty()) {
        fields.push_back(trim(current));
    }

    for (const auto& field : fields) {
        if (field.empty()) {
            continue;
        }

        size_t colon = std::string::npos;
        in_quotes = false;
        escaping = false;
        for (size_t index = 0; index < field.size(); ++index) {
            const char character = field[index];
            if (escaping) {
                escaping = false;
                continue;
            }
            if (character == '\\') {
                escaping = true;
                continue;
            }
            if (character == '"') {
                in_quotes = !in_quotes;
                continue;
            }
            if (character == ':' && !in_quotes) {
                colon = index;
                break;
            }
        }
        if (colon == std::string::npos) {
            return false;
        }

        Helios::Metrics::MetricEntry entry;
        entry.name = unquote(field.substr(0, colon));
        entry.value = unquote(field.substr(colon + 1));
        entries.push_back(std::move(entry));
    }
    return true;
}

std::string unquote(std::string text) {
    text = trim(std::move(text));
    if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
        text = text.substr(1, text.size() - 2);
    }

    std::string result;
    result.reserve(text.size());
    bool escaping = false;
    for (const char character : text) {
        if (escaping) {
            switch (character) {
            case 'n':
                result.push_back('\n');
                break;
            case '\\':
            case '"':
                result.push_back(character);
                break;
            default:
                result.push_back(character);
                break;
            }
            escaping = false;
            continue;
        }

        if (character == '\\') {
            escaping = true;
            continue;
        }
        result.push_back(character);
    }
    return result;
}

std::string compiler_description() {
#if defined(__clang__)
    return "clang-" + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "gcc-" + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "unknown";
#endif
}

std::string cpu_model() {
#ifdef __APPLE__
    size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) == 0 && size > 0) {
        std::string result(size, '\0');
        if (sysctlbyname("machdep.cpu.brand_string", result.data(), &size, nullptr, 0) == 0) {
            if (!result.empty() && result.back() == '\0') {
                result.pop_back();
            }
            return result;
        }
    }
#endif
    return "unknown";
}

std::string memory_bytes_string() {
#ifdef __APPLE__
    std::uint64_t memory_bytes = 0;
    size_t size = sizeof(memory_bytes);
    if (sysctlbyname("hw.memsize", &memory_bytes, &size, nullptr, 0) == 0) {
        return std::to_string(memory_bytes);
    }
#endif
    return "unknown";
}

} // namespace

double Metrics::now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string Metrics::utc_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm {};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
    return buffer;
}

Metrics::BenchmarkSummary Metrics::summarize(const std::vector<double>& samples_ms) {
    BenchmarkSummary summary;
    if (samples_ms.empty()) {
        return summary;
    }

    std::vector<double> sorted_samples = samples_ms;
    std::sort(sorted_samples.begin(), sorted_samples.end());

    double total_ms = 0.0;
    for (const double sample : sorted_samples) {
        total_ms += sample;
    }

    summary.min_ms = sorted_samples.front();
    summary.median_ms = percentile(sorted_samples, 0.5);
    summary.p95_ms = percentile(sorted_samples, 0.95);
    summary.max_ms = sorted_samples.back();
    summary.mean_ms = total_ms / static_cast<double>(sorted_samples.size());
    summary.trials = sorted_samples.size();
    if (summary.trials > 1) {
        double squared_error_sum = 0.0;
        for (const double sample : sorted_samples) {
            const double delta = sample - summary.mean_ms;
            squared_error_sum += delta * delta;
        }
        summary.stddev_ms = std::sqrt(squared_error_sum / static_cast<double>(summary.trials));
    }
    summary.cv_pct = summary.mean_ms == 0.0 ? 0.0 : (summary.stddev_ms / summary.mean_ms) * 100.0;
    return summary;
}

std::vector<Metrics::MetricEntry> Metrics::collect_host_metadata() {
    std::vector<MetricEntry> entries;
    entries.push_back({"schema_version", "helios_result_v2"});
    entries.push_back({"generated_utc", utc_now_iso8601()});

    struct utsname system_name {};
    if (uname(&system_name) == 0) {
        entries.push_back({"host_sysname", system_name.sysname});
        entries.push_back({"host_release", system_name.release});
        entries.push_back({"host_machine", system_name.machine});
        entries.push_back({"host_nodename", system_name.nodename});
    }

    char hostname_buffer[256];
    if (gethostname(hostname_buffer, sizeof(hostname_buffer)) == 0) {
        hostname_buffer[sizeof(hostname_buffer) - 1] = '\0';
        entries.push_back({"host_hostname", hostname_buffer});
    }

    entries.push_back({"host_cpu_model", cpu_model()});
    entries.push_back({"host_memory_bytes", memory_bytes_string()});
    entries.push_back({"host_hardware_threads", std::to_string(std::thread::hardware_concurrency())});
    entries.push_back({"host_compiler", compiler_description()});
    entries.push_back({"host_cxx_standard", std::to_string(__cplusplus)});
    return entries;
}

bool Metrics::write_csv(const std::string& path, const std::vector<MetricEntry>& entries) {
    if (!ensure_parent_directory(path)) {
        return false;
    }

    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output << "metric,value\n";
    for (const auto& entry : entries) {
        output << entry.name << ",\"" << json_escape(entry.value) << "\"\n";
    }
    return true;
}

bool Metrics::write_json(const std::string& path, const std::vector<MetricEntry>& entries) {
    if (!ensure_parent_directory(path)) {
        return false;
    }

    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output << "{\n";
    for (size_t index = 0; index < entries.size(); ++index) {
        output << "  \"" << json_escape(entries[index].name) << "\": "
               << "\"" << json_escape(entries[index].value) << "\"";
        if (index + 1 < entries.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "}\n";
    return true;
}

bool Metrics::append_jsonl(const std::string& path, const std::vector<MetricEntry>& entries) {
    if (!ensure_parent_directory(path)) {
        return false;
    }

    std::ofstream output(path, std::ios::app);
    if (!output) {
        return false;
    }

    output << "{";
    for (size_t index = 0; index < entries.size(); ++index) {
        output << "\"" << json_escape(entries[index].name) << "\":"
               << "\"" << json_escape(entries[index].value) << "\"";
        if (index + 1 < entries.size()) {
            output << ",";
        }
    }
    output << "}\n";
    return true;
}

bool Metrics::read_csv(const std::string& path, std::vector<MetricEntry>& entries) {
    entries.clear();
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::string line;
    bool first_line = true;
    while (std::getline(input, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }

        const size_t comma = line.find(',');
        if (comma == std::string::npos) {
            return false;
        }
        MetricEntry entry;
        entry.name = trim(line.substr(0, comma));
        entry.value = unquote(line.substr(comma + 1));
        entries.push_back(std::move(entry));
    }
    return true;
}

bool Metrics::read_json(const std::string& path, std::vector<MetricEntry>& entries) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    return parse_json_object_text(contents.str(), entries);
}

bool Metrics::read_jsonl(const std::string& path, std::vector<std::vector<MetricEntry>>& records) {
    records.clear();

    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::string line;
    while (std::getline(input, line)) {
        line = trim(std::move(line));
        if (line.empty()) {
            continue;
        }

        std::vector<MetricEntry> record;
        if (!parse_json_object_text(line, record)) {
            return false;
        }
        records.push_back(std::move(record));
    }
    return true;
}

bool Metrics::read_metrics(const std::string& path, std::vector<MetricEntry>& entries) {
    const std::filesystem::path filesystem_path(path);
    const std::string extension = filesystem_path.extension().string();
    if (extension == ".csv") {
        return read_csv(path, entries);
    }
    if (extension == ".json") {
        return read_json(path, entries);
    }
    return false;
}

std::string Metrics::find_value(const std::vector<MetricEntry>& entries, const std::string& key) {
    for (const auto& entry : entries) {
        if (entry.name == key) {
            return entry.value;
        }
    }
    return {};
}

} // namespace Helios
