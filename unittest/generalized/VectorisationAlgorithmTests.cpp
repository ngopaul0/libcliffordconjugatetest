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
    SECTION("Performs createXtransposeTensorI correctly") {
        const size_t d = 5;
        const size_t n = 1;
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> X(2*n, 1);
        Eigen::Vector<long, 2> v;
        v << 1, 2;
        X.col(0) = v;
        INFO("X = " << X);
        auto XtransposeTensorI = createXtransposeTensorI(X);
        INFO("XtransposeTensorI = " << XtransposeTensorI);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> S(2*n, 2*n);
        S << 2, 2, 0, 2;

        Eigen::Vector<long, 2> Sv = S * v;
        Sv = Sv.array().unaryExpr([&](const long x) { return safeMod(x, d); });

        Eigen::Vector<long, 2> expectedSv(1, 4);
        REQUIRE(Sv == expectedSv);

        Eigen::Vector<long, Eigen::Dynamic> vecS = Eigen::Map<Eigen::Vector<long, Eigen::Dynamic>>(S.data(), S.size());
        REQUIRE(XtransposeTensorI.rows() == S.cols());
        Eigen::Vector<long, Eigen::Dynamic> XtranposeTensorITimesVecS = XtransposeTensorI * vecS;
        XtranposeTensorITimesVecS = XtranposeTensorITimesVecS.array().unaryExpr([&](const long x) { return safeMod(x, d); });
        INFO("XtranposeTensorITimesVecS = " << XtranposeTensorITimesVecS);
        REQUIRE(XtranposeTensorITimesVecS == expectedSv);
    }
}
