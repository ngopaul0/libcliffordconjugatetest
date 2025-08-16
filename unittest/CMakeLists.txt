Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

add_executable(unittest CliffConjTests.cpp
        UtilTests.cpp
        Sp1ZdIteratorTests.cpp
        CliffordGateTests.cpp
        MultiDimensionalArrayTests.cpp
        TupleIteratorTests.cpp
)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Add cross-platform macro to indicate debug build
    target_compile_definitions(unittest PRIVATE CATCH_DEBUG_ONLY)
endif()

include(../cmake/eigen3.cmake)

target_link_libraries(unittest PRIVATE cliffordconjugatetest)
target_link_libraries(unittest PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)

enable_testing()
include(Catch)
catch_discover_tests(unittest)
