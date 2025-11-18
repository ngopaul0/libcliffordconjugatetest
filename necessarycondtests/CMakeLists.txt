Include(FetchContent)

add_executable(necessarycondtests main.cpp)

include(../cmake/eigen3.cmake)
target_include_directories(necessarycondtests INTERFACE ${SQLITE3_INCLUDE_DIRS})
target_link_libraries(necessarycondtests PRIVATE cliffordconjugatetest)
target_link_libraries(necessarycondtests PRIVATE ${Boost_LIBRARIES} Eigen3::Eigen)
target_include_directories(necessarycondtests PRIVATE ${Boost_INCLUDE_DIRS} "${libnpy_SOURCE_DIR}")

find_package(Boost REQUIRED)

find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "Linking OpenMP")
    target_link_libraries(necessarycondtests PUBLIC OpenMP::OpenMP_CXX)
endif()
