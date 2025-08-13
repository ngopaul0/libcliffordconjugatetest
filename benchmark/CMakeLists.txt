Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

add_executable(benchmark CliffConjBenchmark.cpp)
include(../cmake/eigen3.cmake)

target_link_libraries(benchmark PRIVATE cliffordconjugatetest)
target_link_libraries(benchmark PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)

# catch_discover_tests(benchmark)
