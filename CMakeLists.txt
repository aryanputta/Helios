cmake_minimum_required(VERSION 3.24)
project(HeliosComputeRuntime LANGUAGES CXX)

include(CTest)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

check_language(CUDA)

set(HELIOS_CORE_SOURCES
    src/cli/cli.cpp
    src/planner/planner.cpp
    src/profiler/profiler.cpp
    src/io/dataset_loader.cpp
    src/runtime/runtime.cpp
    src/cpu/cpu_kernels.cpp
    src/metrics/metrics.cpp
)

set(HELIOS_PUBLIC_HEADERS
    include/helios/cli.h
    include/helios/planner.h
    include/helios/profiler.h
    include/helios/dataset_loader.h
    include/helios/runtime.h
    include/helios/cpu_kernels.h
    include/helios/gpu_kernels.h
    include/helios/metrics.h
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    list(APPEND HELIOS_CORE_SOURCES src/cpu/cpu_kernels_avx2.cpp)
endif()

if(CMAKE_CUDA_COMPILER)
    list(APPEND HELIOS_CORE_SOURCES src/gpu/gpu_kernels.cu)
else()
    list(APPEND HELIOS_CORE_SOURCES src/gpu/gpu_kernels_stub.cpp)
endif()

add_library(helios_core ${HELIOS_CORE_SOURCES} ${HELIOS_PUBLIC_HEADERS})
target_include_directories(helios_core PUBLIC include)
target_compile_features(helios_core PUBLIC cxx_std_17)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set_target_properties(helios_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )
    target_compile_definitions(helios_core PUBLIC HELIOS_ENABLE_CUDA=1)

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        if(TARGET CUDA::cublas)
            target_link_libraries(helios_core PUBLIC CUDA::cublas)
            target_compile_definitions(helios_core PUBLIC HELIOS_HAVE_CUBLAS=1)
        endif()
        if(TARGET CUDA::cusparse)
            target_link_libraries(helios_core PUBLIC CUDA::cusparse)
            target_compile_definitions(helios_core PUBLIC HELIOS_HAVE_CUSPARSE=1)
        endif()
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    set_source_files_properties(
        src/cpu/cpu_kernels_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
    target_compile_definitions(helios_core PUBLIC HELIOS_HAVE_X86_64=1)
endif()

add_executable(helios src/main.cpp)
target_link_libraries(helios PRIVATE helios_core)

option(HELIOS_VERBOSE_BUILD "Enable verbose build output" OFF)
if(HELIOS_VERBOSE_BUILD)
    target_compile_definitions(helios_core PUBLIC HELIOS_VERBOSE=1)
endif()

option(HELIOS_BUILD_GOOGLE_BENCHMARK "Build optional Google Benchmark harnesses when available" ON)
option(HELIOS_BUILD_NVBENCH "Build optional NVBench harnesses when available" ON)

if(HELIOS_BUILD_GOOGLE_BENCHMARK)
    find_package(benchmark QUIET)
    if(benchmark_FOUND AND TARGET benchmark::benchmark)
        add_executable(helios_gbench_dense bench/google_benchmark/dense_cpu_bench.cpp)
        target_link_libraries(helios_gbench_dense PRIVATE helios_core benchmark::benchmark)

        add_executable(helios_gbench_sparse bench/google_benchmark/sparse_spmv_bench.cpp)
        target_link_libraries(helios_gbench_sparse PRIVATE helios_core benchmark::benchmark)
    endif()
endif()

if(HELIOS_BUILD_NVBENCH AND CMAKE_CUDA_COMPILER)
    find_package(nvbench QUIET)
    if(nvbench_FOUND)
        if(TARGET nvbench::main)
            set(HELIOS_NVBENCH_TARGET nvbench::main)
        elseif(TARGET nvbench)
            set(HELIOS_NVBENCH_TARGET nvbench)
        endif()

        if(HELIOS_NVBENCH_TARGET)
            add_executable(helios_nvbench_dense bench/nvbench/dense_cuda_bench.cu)
            target_link_libraries(helios_nvbench_dense PRIVATE helios_core ${HELIOS_NVBENCH_TARGET})
            set_target_properties(helios_nvbench_dense PROPERTIES CUDA_STANDARD 17)
        endif()
    endif()
endif()

if(BUILD_TESTING)
    add_executable(test_dataset_loader tests/test_dataset_loader.cpp)
    target_link_libraries(test_dataset_loader PRIVATE helios_core)
    add_test(NAME test_dataset_loader COMMAND test_dataset_loader)

    add_executable(test_runtime_commands tests/test_runtime_commands.cpp)
    target_link_libraries(test_runtime_commands PRIVATE helios_core)
    add_test(NAME test_runtime_commands COMMAND test_runtime_commands)
endif()

install(TARGETS helios RUNTIME DESTINATION bin)
