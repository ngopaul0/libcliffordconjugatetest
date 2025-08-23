Include(FetchContent)

set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "download google/benchmark deps" FORCE)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.4
)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(
        libnpy
        GIT_REPOSITORY https://github.com/llohse/libnpy
        GIT_TAG 471fe480d5f1082fd8fd0e746eaf10084a2fb82b
)
FetchContent_MakeAvailable(libnpy)

add_executable(benchmarkmultiqudit CliffConjBenchmark.cpp)
include(../cmake/eigen3.cmake)

target_link_libraries(benchmarkmultiqudit PRIVATE cliffordconjugatetest)
target_link_libraries(benchmarkmultiqudit PRIVATE benchmark::benchmark Eigen3::Eigen)
target_include_directories(benchmarkmultiqudit PRIVATE "${libnpy_SOURCE_DIR}")

# catch_discover_tests(benchmark)
