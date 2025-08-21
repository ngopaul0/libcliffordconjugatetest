#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <random>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "generalized/vectorisationalgorithm.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
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

    SECTION("Matrix vectorisation algorithm flow - correct mappings") {
        constexpr size_t d = 5;
        constexpr size_t n = 1;

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S(2*n, 2*n);
        S << 2, 2,
             0, 2;

        // INITIAL:
        Eigen::Vector<long, 2> v1(1, 2);
        Eigen::Vector<long, 2> mappedSv1 = modMatrix(S * v1, d);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> system = createSystemForS(v1, mappedSv1);
        Eigen::Matrix<long, 2, 4 + 1> expectedState;
        expectedState << 1, 0, 2, 0, 1,
                         0, 1, 0, 2, 4;
        REQUIRE(expectedState == system);

        const auto rank = reduceToREFAndGetRank(system, d, true);
        REQUIRE(rank == 2);
        // no change
        REQUIRE(expectedState == system);

        // SECOND:
        // Appending a vector that's linearly dependent would result in just zero rows
        Eigen::Vector<long, 2> v2(2, 4);
        Eigen::Vector<long, 2> mappedSv2 = modMatrix(S * v2, d);
        size_t rowsBefore = system.rows();
        CHECK(rowsBefore == 2);
        appendToSystemForS(system, v2, mappedSv2);
        size_t rowsAfter = system.rows();
        CHECK(rowsAfter == 4);
        Eigen::Matrix<long, 4, 4 + 1> expectedState2;
        expectedState2 << 1, 0, 2, 0, 1,
                          0, 1, 0, 2, 4,
                          2, 0, 4, 0, mappedSv2(0),
                          0, 2, 0, 4, mappedSv2(1);
        REQUIRE(expectedState2 == system);

        const auto rank2 = reduceToREFAndGetRank(system, d, true);
        REQUIRE(rank2 == 2);
        // RREF change
        Eigen::Matrix<long, 4, 4 + 1> expectedState2RREF;
        expectedState2RREF << 1, 0, 2, 0, 1,
                              0, 1, 0, 2, 4,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0;
        REQUIRE(expectedState2RREF == system);

        // THIRD:
        // Now add a vector mapping that's linearly independent.
        Eigen::Vector<long, 2> v3(1, 3);
        Eigen::Vector<long, 2> mappedSv3 = modMatrix(S * v3, d);
        rowsBefore = system.rows();
        CHECK(rowsBefore == 4);
        appendToSystemForS(system, v3, mappedSv3);
        rowsAfter = system.rows();
        CHECK(rowsAfter == 6);
        Eigen::Matrix<long, 6, 4 + 1> expectedState3;
        expectedState3 << 1, 0, 2, 0, 1,
                          0, 1, 0, 2, 4,
                          0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0,
                          1, 0, 3, 0, mappedSv3(0),
                          0, 1, 0, 3, mappedSv3(1);
        REQUIRE(expectedState3 == system);
        // RREF should result in finding rank 4, since 2 linearly independent vectors form a basis
        // which completely determines a linear transform in the space
        const auto rank3 = reduceToREFAndGetRank(system, d, true);
        REQUIRE(rank3 == 4);
        Eigen::Matrix<long, 6, 4 + 1> expectedState3RREF;
        expectedState3RREF << 1, 0, 0, 0, 2,
                              0, 1, 0, 0, 0,
                              0, 0, 1, 0, 2,
                              0, 0, 0, 1, 2,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0;
        REQUIRE(expectedState3RREF == system);
        // It's clear the solution of this system gets back S

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> fromSystemS = recoverSFromSystem(system, n);
        REQUIRE(fromSystemS == S);
    }

    SECTION("Matrix vectorisation algorithm flow - mapping from failing test") {
        constexpr size_t d = 3;
        constexpr size_t n = 2;

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S(2*n, 2*n);
        S << 2, 0, 0, 0,
             0, 2, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;

        // INITIAL:
        Eigen::Vector<long, 4> v1(0, 1, 2, 0);
        Eigen::Vector<long, 4> mappedSv1 = modMatrix(S * v1, d);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> system = createSystemForS(v1, mappedSv1);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedState;
        expectedState.resize(system.rows(), system.cols());
        expectedState << 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, mappedSv1(0),
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, mappedSv1(1),
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, mappedSv1(2),
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, mappedSv1(3);
        CHECK(expectedState.rows() == system.rows());
        REQUIRE(expectedState.cols() == system.cols());
        REQUIRE(expectedState == system);

        auto rank = reduceToREFAndGetRank(system, d, true);
        REQUIRE(rank == 4);
        REQUIRE(expectedState == system);

        //e1 = {0, 1, 2, 1};
        //e2 = {0, 2, 2, 1};
        Eigen::Vector<long, 4> v2(0, 1, 2, 1);
        Eigen::Vector<long, 4> mappedSv2 = modMatrix(S * v2, d);

        appendToSystemForS(system, v2, mappedSv2);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedState2;
        expectedState2.resize(system.rows(), system.cols());
        expectedState2 << 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, mappedSv1(0),
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, mappedSv1(1),
                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, mappedSv1(2),
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, mappedSv1(3),
                          0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, mappedSv2(0),
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, mappedSv2(1),
                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, mappedSv2(2),
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, mappedSv2(3);
        CHECK(expectedState2.rows() == system.rows());
        REQUIRE(expectedState2.cols() == system.cols());
        REQUIRE(expectedState2 == system);

        rank = reduceToREFAndGetRank(system, d, true);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedRREF1;
        expectedRREF1.resize(system.rows(), system.cols());
        expectedRREF1 << 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1;
        REQUIRE(rank == 8);
        REQUIRE(expectedRREF1 == system);

        // e1 = {1, 2, 2, 1};
        // e2 = {2, 1, 2, 1};
        Eigen::Vector<long, 4> v3(1, 2, 2, 1);
        Eigen::Vector<long, 4> mappedSv3 = modMatrix(S * v3, d);

        appendToSystemForS(system, v3, mappedSv3);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedState3;
        expectedState3.resize(system.rows(), system.cols());
        expectedState3 << 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                         1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, mappedSv3(0),
                         0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, mappedSv3(1),
                         0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, mappedSv3(2),
                         0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, mappedSv3(3);
        CHECK(expectedState3.rows() == system.rows());
        REQUIRE(expectedState3.cols() == system.cols());
        REQUIRE(expectedState3 == system);

        rank = reduceToREFAndGetRank(system, d, true);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedRREF2;
        expectedRREF2.resize(system.rows(), system.cols());
        expectedRREF2 << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2,
                         0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1;
        REQUIRE(rank == 12);
        REQUIRE(expectedRREF2 == system);
    }

    SECTION("Matrix vectorisation algorithm flow - mapping from failing test 2") {
        constexpr size_t d = 3;
        constexpr size_t n = 2;

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S(2*n, 2*n);
        S << 2, 0, 0, 0,
             0, 2, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;

        // INITIAL:
        Eigen::Vector<long, 4> v1(0, 1, 2, 0);
        Eigen::Vector<long, 4> mappedSv1 = modMatrix(S * v1, d);

        Eigen::Vector<long, 4> v2(0, 1, 2, 1);
        Eigen::Vector<long, 4> mappedSv2 = modMatrix(S * v2, d);

        Eigen::Vector<long, 4> v3(1, 2, 2, 0);
        Eigen::Vector<long, 4> mappedSv3 = modMatrix(S * v3, d);

        Eigen::Vector<long, 4> v4(1, 2, 2, 1);
        Eigen::Vector<long, 4> mappedSv4 = modMatrix(S * v4, d);
        Eigen::Vector<long, 4> v5(2, 2, 2, 0);
        Eigen::Vector<long, 4> mappedSv5 = modMatrix(S * v5, d);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> X(4, 4);
        X.col(0) = v1;
        X.col(1) = v2;
        X.col(2) = v3;
        X.col(3) = v4;
        const size_t rankOfX = reduceToREFAndGetRank(X, d, true);
        CHECK(rankOfX == 3);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> system =
            createSystemForS(v1, mappedSv1);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> runningRREF = system;
        appendToSystemForS(system, v2, mappedSv2);
        reduceToREFAndGetRank(runningRREF, d, true);
        appendToSystemForS(system, v3, mappedSv3);
        reduceToREFAndGetRank(runningRREF, d, true);
        appendToSystemForS(system, v4, mappedSv4);
        reduceToREFAndGetRank(runningRREF, d, true);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedSystem;
        expectedSystem.resize(system.rows(), system.cols());
        expectedSystem << 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1,
                         1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2,
                         0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2,
                         0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1,
                         0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2,
                         0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 1;
        REQUIRE(expectedSystem == system);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> rrefSystem = system;
        const size_t rankSystem = reduceToREFAndGetRank(rrefSystem, d, true);
        CHECK(rankSystem == 12);
        REQUIRE(expectedSystem == system);
        // e1 = {2, 2, 2, 0};
        // e2 = {1, 1, 2, 0};
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> X2(4, 5);
        X2.col(0) = v1;
        X2.col(1) = v2;
        X2.col(2) = v3;
        X2.col(3) = v4;
        X2.col(4) = v5;
        const size_t rankOfX2 = reduceToREFAndGetRank(X2, d, true);
        CHECK(rankOfX2 == 4);

        appendToSystemForS(system, v5, mappedSv5);
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedSystem2;
        expectedSystem2.resize(system.rows(), system.cols());
        expectedSystem2 << 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1,
                         1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2,
                         0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2,
                         0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1,
                         0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2,
                         0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 1,
                         2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2,
                         0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0;
        REQUIRE(expectedSystem2 == system);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> rrefSystem2 = system;
        const size_t rankSystem2 = reduceToREFAndGetRank(rrefSystem2, d, true);
        CHECK(rankSystem2 == 16);
        // REQUIRE(expectedSystem2 == rrefSystem2);
    }
}

TEST_CASE("findSymplecticMatrix", "[vectorisationalgorithm][findSymplecticMatrix]") {
    SECTION("d = 3: Two Pauli basis elements") {
        const int d = 3; // Example dimension
        const int n = 1;
        const int inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));

        const Eigen::Vector<long, 2> v1 = {1, 2};
        const Eigen::Vector<long, 2> v2 = {1, 1};

        const Eigen::MatrixXcd M = W(d, v1(0), v1(1), inv_2, omega) + W(d, v2(0), v2(1), inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd Mprime = C * M * Cstar;

        FMap MMap(d, n, 1e-5, 1e-5);
        for (const auto& tuple : TupleIterator<SingleModulus>(d, 2)) {
            const auto fM = f_multiqudit(M, tuple, d, inv_2, omega);
            MMap.insertEntry(tuple, fM);
        }

        FMap MprimeMap(d, n, 1e-5, 1e-5);
        for (const auto& tuple : TupleIterator<SingleModulus>(d, 2)) {
            const auto fM = f_multiqudit(Mprime, tuple, d, inv_2, omega);
            MprimeMap.insertEntry(tuple, fM);
        }

        const auto M_p = createMpMatrix(M, omega, inv_2);
        const auto Mprime_p = createMpMatrix(Mprime, omega, inv_2);
        std::optional<Lemma10Info> bruteForceResult = bruteForceTestCliffordConjugacy<true>(M, Mprime, omega, M_p, Mprime_p);
        REQUIRE(bruteForceResult.has_value());
        const auto S = bruteForceResult.value().first;

        const Eigen::Vector<long, 2> Sv1 = modMatrix(S * v1, d);
        std::stringstream ssv1;
        ssv1 << Sv1;
        auto strv1 = ssv1.str();
        const Eigen::Vector<long, 2> Sv2 = modMatrix(S * v2, d);
        std::stringstream ssv2;
        ssv2 << Sv2;
        auto strv2 = ssv2.str();

        auto result = findSymplecticMatrix(d, n, omega, M, M_p, Mprime_p, MMap, MprimeMap);
        REQUIRE(result.has_value());
        CHECK(result.value() == S);
    }
}

TEST_CASE("createSystemForPPrimeQPrime", "[generalized]") {
    SECTION("Creates system correctly") {
        const size_t d = 5;
        for (const auto& pq_vec : TupleIterator<SingleModulus>(d, 4)) {
            INFO("pq_vec = " << pq_vec);
            const size_t k = 3;

            REQUIRE(pq_vec.rows() == 4);
            REQUIRE(pq_vec.size() == 4);

            Eigen::RowVector<long, 5> system = createSystemForPPrimeQPrime(d, pq_vec, k);
            Eigen::RowVector<long, 5> expectedSystem =
                {-pq_vec(2), -pq_vec(3), pq_vec(0), pq_vec(1), k};
            expectedSystem = modMatrix(expectedSystem, d);
            REQUIRE(expectedSystem == system);
        }
    }

    SECTION("Appends to system correctly") {
        const size_t d = 5;
        const size_t k = 3;
        const Eigen::Vector<long, 4> pq_vec = {1, 2, 3, 4};
        REQUIRE(pq_vec.rows() == 4);
        REQUIRE(pq_vec.size() == 4);

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> system =
            createSystemForPPrimeQPrime(d, pq_vec, k);
        Eigen::RowVector<long, 5> expectedSystem = {safeMod(-pq_vec(2), d), safeMod(-pq_vec(3), d), safeMod(pq_vec(0), d), pq_vec(1), k};
        REQUIRE(expectedSystem == system);
        REQUIRE(expectedSystem == system.row(0));

        const Eigen::Vector<long, 4> pq_vec_2 = {1, 2, 3, 4};
        CHECK(system.rows() == 1);
        const size_t k2 = 2;
        appendToSystemForPPrimeQPrime(system, d, pq_vec, k2);
        CHECK(system.rows() == 2);
        Eigen::RowVector<long, 5> expectedSystem2ndRow =
           {safeMod(-pq_vec_2(2), d), safeMod(-pq_vec_2(3), d), safeMod(pq_vec_2(0), d), pq_vec_2(1), k2};;
        CHECK(expectedSystem == system.row(0));
        CHECK(expectedSystem2ndRow == system.row(1));
    }
}
