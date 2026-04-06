#pragma once

#include <string>
#include <vector>

namespace Helios {

class CLI {
public:
    CLI(int argc, char** argv);

    bool parse();
    int execute();
    int exit_code() const;

private:
    void print_help() const;

    int argc;
    char** argv;
    int parse_exit_code = 0;
    bool show_help = false;
    std::string command;
    std::string workload;
    std::vector<std::string> arguments;
};

} // namespace Helios
