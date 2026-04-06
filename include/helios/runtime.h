#pragma once

#include <string>
#include <vector>

namespace Helios {

class Runtime {
public:
    Runtime();
    ~Runtime();

    int run(const std::string& command,
            const std::string& workload,
            const std::vector<std::string>& args) const;
};

} // namespace Helios
