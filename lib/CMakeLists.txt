include(../cmake/eigen3.cmake)

add_library(cliffordconjugatetest STATIC cliffordconjugacytest.cpp cliffordconjugacytest.hpp internal/util.cpp
        internal/FMap.h
        internal/FMap.cpp
        internal/cliffordgates.cpp
        internal/cliffordgates.h
        internal/symplecticiterator.h
        internal/bruteforcetest.h
        internal/conjtestlemma10.h
        internal/multidimarray.h
        internal/tupleiterator.h
        internal/hash_mix.h
        generalized/vectorisationalgorithm.h
        generalized/vectorisationalgorithm.cpp
        internal/complexexactrepr.h
)
target_link_libraries(cliffordconjugatetest PUBLIC Eigen3::Eigen)
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "Linking OpenMP")
    target_link_libraries(cliffordconjugatetest PUBLIC OpenMP::OpenMP_CXX)
endif()
