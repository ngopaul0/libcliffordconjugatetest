Include(FetchContent)

FetchContent_Declare(
        libnpy
        GIT_REPOSITORY https://github.com/llohse/libnpy
        GIT_TAG 471fe480d5f1082fd8fd0e746eaf10084a2fb82b
)
FetchContent_MakeAvailable(libnpy)


FetchContent_Declare(
        sqlite_modern_cpp
        GIT_REPOSITORY https://github.com/SqliteModernCpp/sqlite_modern_cpp.git
        GIT_TAG 6e3009973025e0016d5573529067714201338c80
)
FetchContent_GetProperties(sqlite_modern_cpp)
if(NOT sqlite_modern_cpp_POPULATED)
    FetchContent_Populate(sqlite_modern_cpp)
endif()

find_package(Boost REQUIRED)

find_package(SQLite3 REQUIRED)

add_executable(multiqudittest main.cpp)

include(../cmake/eigen3.cmake)
target_include_directories(multiqudittest INTERFACE ${SQLITE3_INCLUDE_DIRS})
target_link_libraries(multiqudittest PRIVATE cliffordconjugatetest)
target_link_libraries(multiqudittest PRIVATE sqlite3)
target_link_libraries(multiqudittest PRIVATE ${Boost_LIBRARIES} Eigen3::Eigen)
target_include_directories(multiqudittest PRIVATE ${sqlite_modern_cpp_SOURCE_DIR}/hdr)
target_include_directories(multiqudittest PRIVATE ${Boost_INCLUDE_DIRS} "${libnpy_SOURCE_DIR}")

find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "Linking OpenMP")
    target_link_libraries(cliffordconjugatetest PUBLIC OpenMP::OpenMP_CXX)
endif()
