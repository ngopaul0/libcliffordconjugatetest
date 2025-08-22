Include(FetchContent)

FetchContent_Declare(
        libnpy
        GIT_REPOSITORY https://github.com/llohse/libnpy
        GIT_TAG 471fe480d5f1082fd8fd0e746eaf10084a2fb82b
)
FetchContent_MakeAvailable(libnpy)

add_executable(multiqudittest main.cpp)

include(../cmake/eigen3.cmake)
target_link_libraries(multiqudittest PRIVATE cliffordconjugatetest)
target_link_libraries(multiqudittest PRIVATE Eigen3::Eigen)
target_include_directories(multiqudittest PRIVATE "${libnpy_SOURCE_DIR}")

find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "Linking OpenMP")
    target_link_libraries(cliffordconjugatetest PUBLIC OpenMP::OpenMP_CXX)
endif()
