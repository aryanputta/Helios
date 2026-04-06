#include "helios/cli.h"

#include "helios/runtime.h"

#include <iostream>

namespace Helios {

CLI::CLI(int argc, char** argv)
    : argc(argc), argv(argv) {}

bool CLI::parse() {
    parse_exit_code = 0;
    show_help = false;
    command.clear();
    workload.clear();
    arguments.clear();

    if (argc < 2) {
        show_help = true;
        return true;
    }

    command = argv[1];
    if (command == "help" || command == "--help" || command == "-h") {
        show_help = true;
        return true;
    }

    if (command != "bench" && command != "profile" && command != "validate"
        && command != "compare" && command != "report") {
        std::cerr << "Unknown command: " << command << "\n";
        parse_exit_code = 1;
        return false;
    }

    if (command == "compare" || command == "report") {
        for (int index = 2; index < argc; ++index) {
            arguments.emplace_back(argv[index]);
        }
        return true;
    }

    if (argc < 3) {
        std::cerr << "Error: " << command
                  << " requires a workload type (dense, sparse, graph).\n";
        parse_exit_code = 1;
        return false;
    }

    workload = argv[2];
    for (int index = 3; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }

    return true;
}

int CLI::execute() {
    if (show_help) {
        print_help();
        return 0;
    }

    Runtime runtime;
    return runtime.run(command, workload, arguments);
}

int CLI::exit_code() const {
    return parse_exit_code;
}

void CLI::print_help() const {
    std::cout
        << "Helios Compute Runtime\n"
        << "Usage:\n"
        << "  helios bench sparse --matrix <path> [--path auto|cpu|gpu] [--backend auto|scalar|avx2|threaded|cuda|vendor] [--threads N] [--compare-baselines] [--planner-log <path>] [--manifest <path>] [--warmup N] [--trials N]\n"
        << "  helios bench graph --graph <path> --algo <bfs|pagerank> [--source N] [--iterations N] [--directed|--undirected] [--manifest <path>]\n"
        << "  helios profile sparse --matrix <path> [--path auto|cpu|gpu] [--backend auto|scalar|avx2|threaded|cuda|vendor] [--manifest <path>]\n"
        << "  helios profile graph --graph <path> [--algo <bfs|pagerank>] [--path auto|cpu|gpu] [--directed|--undirected] [--manifest <path>]\n"
        << "  helios bench dense --m <rows> --n <cols> --k <depth> [--path auto|cpu|gpu] [--backend auto|scalar|avx2|threaded|cuda|vendor] [--threads N] [--compare-baselines] [--planner-log <path>]\n"
        << "  helios profile dense --m <rows> --n <cols> --k <depth> [--path auto|cpu|gpu] [--backend auto|scalar|avx2|threaded|cuda|vendor]\n"
        << "  helios validate sparse --matrix <path> [--backend auto|scalar|avx2|threaded|cuda|vendor] [--threads N] [--tolerance EPS] [--compare-all] [--manifest <path>]\n"
        << "  helios validate dense --m <rows> --n <cols> --k <depth> [--backend auto|scalar|avx2|threaded|cuda|vendor] [--threads N] [--tolerance EPS] [--compare-all]\n"
        << "  helios validate graph --graph <path> --algo <bfs|pagerank> [--source N] [--iterations N] [--directed|--undirected] [--manifest <path>]\n"
        << "  helios compare --lhs <result.csv|json> --rhs <result.csv|json> [--csv <path>] [--json <path>]\n"
        << "  helios report [--result <result.csv|json> ...] [--planner-log <observations.jsonl> ...] [--results-dir <directory>] [--markdown <path>|--md <path>] [--csv <path>] [--json <path>]\n"
        << "  helios --help\n";
}

} // namespace Helios
