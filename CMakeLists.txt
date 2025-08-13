cmake_minimum_required(VERSION 3.28)
project(cliffordconjugatetest)

set(CMAKE_CXX_STANDARD 23)

include_directories(lib)
add_subdirectory(lib)

enable_testing()
add_subdirectory(unittest)
add_subdirectory(benchmark)
