#pragma once

#include "helios/profiler.h"

#include <string>

namespace Helios {

class Planner {
public:
    enum class Strategy {
        Auto,
        DenseGPU,
        DenseCPU,
        SparseGPU,
        SparseCPU,
        GraphCPU,
        GraphGPU,
    };

    struct Decision {
        Strategy strategy = Strategy::Auto;
        std::string reason;
    };

    Planner();
    ~Planner();

    Decision select_strategy(const WorkloadProfile& profile,
                             const std::string& requested_path = "auto",
                             bool gpu_available = false) const;
    static const char* to_string(Strategy strategy);
};

} // namespace Helios
