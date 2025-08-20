#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <expected>
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

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> system = createSystem(v1, mappedSv1);
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
        appendToSystem(system, v2, mappedSv2);
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
        appendToSystem(system, v3, mappedSv3);
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


}

TEST_CASE("findSymplecticMatrix", "[vectorisationalgorithm][findSymplecticMatrix]") {
    SECTION("d = 3: Two Pauli basis elements") {
        const int d = 3; // Example dimension
        const int n = 1;
        const int inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 1, 1, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

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

        auto result = findSymplecticMatrix(d, n, omega, M, M_p, Mprime_p, MMap, MprimeMap);
        REQUIRE(result.has_value());
        CHECK(result.value() == S);
    }
}
