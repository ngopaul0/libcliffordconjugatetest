include(../cmake/eigen3.cmake)

add_library(cliffordconjugatetest STATIC cliffordconjugacytest.cpp cliffordconjugacytest.hpp internal/util.cpp
        internal/absvalmap.h
        internal/absvalmap.cpp
        internal/cliffordgates.cpp
        internal/cliffordgates.h
        internal/symplecticiterator.h
        internal/bruteforcetest.h
        internal/conjtestlemma10.h
)
target_link_libraries(cliffordconjugatetest PRIVATE Eigen3::Eigen)
