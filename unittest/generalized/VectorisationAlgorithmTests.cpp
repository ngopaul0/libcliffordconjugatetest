#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <expected>
#include <random>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "generalized/vectorisationalgorithm.h"
#include "internal/tupleiterator.h"

using namespace cliffconjtest;

TEST_CASE("vectorisation", "[vectorisationalgorithm]") {
    SECTION("Performs createXtransposeTensorI correctly for one vector") {
        const size_t d = 5;
        const size_t n = 1;
        Eigen::Matrix<long, 2, 1> X;
        Eigen::Vector<long, 2> v;
        v << 1, 2;
        X.col(0) = v;
        // INFO("X = " << X);
        REQUIRE(X == v);
        auto XtransposeTensorI = createXtransposeTensorI(X);
        auto XtransposeTensorI_from_v = createXtransposeTensorI(v);
        Eigen::Matrix<long, 2, 4> expectedXT_tensor_I;
        expectedXT_tensor_I << 1, 0, 2, 0,
                               0, 1, 0, 2;
        REQUIRE(XtransposeTensorI == expectedXT_tensor_I);
        REQUIRE(XtransposeTensorI_from_v == expectedXT_tensor_I);
        INFO("XtransposeTensorI = " << XtransposeTensorI);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S(2*n, 2*n);
        S << 2, 2, 0, 2;

        Eigen::Vector<long, 2> Sv = modMatrix(S * v, d);

        Eigen::Vector<long, 2> expectedSv(1, 4);
        REQUIRE(Sv == expectedSv);

        auto vecS = vecOperator(S);
        REQUIRE(XtransposeTensorI.cols() == vecS.rows());
        auto XtranposeTensorITimesVecS = modMatrix(XtransposeTensorI * vecS, d);
        INFO("XtranposeTensorITimesVecS = " << XtranposeTensorITimesVecS);
        REQUIRE(XtranposeTensorITimesVecS == expectedSv);
    }

    SECTION("Performs createXtransposeTensorI correctly for multiple vectors") {
        const size_t d = 5;
        const size_t n = 1;
        Eigen::Matrix<long, 2, 2> X;

        Eigen::Vector<long, 2> v1;
        v1 << 1, 2;
        X.col(0) = v1;
        Eigen::Vector<long, 2> v2;
        v2 << 2, 4;
        X.col(1) = v2;

        auto XtransposeTensorI = createXtransposeTensorI(X);
        Eigen::Matrix<long, 4, 4> expectedXT_tensor_I;
        expectedXT_tensor_I << 1, 0, 2, 0,
                               0, 1, 0, 2,
                               2, 0, 4, 0,
                               0, 2, 0, 4;
        REQUIRE(XtransposeTensorI == expectedXT_tensor_I);
        INFO("XtransposeTensorI = " << XtransposeTensorI);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S(2*n, 2*n);
        S << 2, 2,
             0, 2;

        auto SX = modMatrix(S * X, d);

        Eigen::Matrix<long, 2, 2> expectedSX;
        expectedSX << 1, 2,
                      4, 3;
        REQUIRE(SX == expectedSX);

        Eigen::Map<Eigen::Vector<Eigen::Matrix<long, -1, -1>::Scalar, Eigen::Dynamic>> vecS =
            vecOperator(S);
        auto vecExpectedSX = vecOperator(SX);

        REQUIRE(XtransposeTensorI.cols() == vecS.rows());
        auto XtranposeTensorITimesVecS = modMatrix(XtransposeTensorI * vecS, d);
        INFO("XtranposeTensorITimesVecS = " << XtranposeTensorITimesVecS);
        REQUIRE(XtranposeTensorITimesVecS == vecExpectedSX);
    }
}
