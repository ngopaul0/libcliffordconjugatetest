#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <unsupported/Eigen/KroneckerProduct>
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include "internal/tupleiterator.h"

using namespace cliffconjtest;

TEST_CASE("reduceToREFAndGetRank", "[REF]") {
    SECTION("Full-rank square matrix") {
        Eigen::Matrix3i A(3, 3);
        A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
        const int p = 11;
        Eigen::Matrix3i A_copy = A;
        size_t rank = reduceToREFAndGetRank(A, p);
        INFO(A);

        Eigen::MatrixXi expectedRef(3, 3);
        expectedRef << 1, 2, 3,
                        0, 1, 2,
                        0, 0, 1;
        // After reduction, the matrix should be the identity matrix
        REQUIRE(rank == 3);
        REQUIRE(A == expectedRef);
        // Should modify A
        REQUIRE(A != A_copy);

        // Test for idempotence
        size_t rankAgain = reduceToREFAndGetRank(A, p);
        INFO(A);
        REQUIRE(rankAgain == 3);
        REQUIRE(A == expectedRef);

        // RREF
        rankAgain = reduceToREFAndGetRank(A, p, true);
        INFO(A);
        REQUIRE(rankAgain == 3);
        REQUIRE(A.isIdentity());
    }

    // Test case for a singular square matrix.
    SECTION("Singular square matrix") {
        Eigen::MatrixXi A(3, 3);
        A << 1, 1, 2, 2, 2, 4, 3, 3, 6;
        const int p = 7;
        size_t rank = reduceToREFAndGetRank(A, p);

        // The first row is a multiple of the other rows, so the rank is 1.
        Eigen::MatrixXi expectedRef(3, 3);
        expectedRef << 1, 1, 2,
                        0, 0, 0,
                        0, 0, 0;

        REQUIRE(rank == 1);
        REQUIRE(A == (expectedRef));

        // Test for idempotence
        size_t rankAgain = reduceToREFAndGetRank(A, p);
        REQUIRE(rankAgain == 1);
        REQUIRE(A == (expectedRef));

        // Test for idempotence
        rankAgain = reduceToREFAndGetRank(A, p, true);
        REQUIRE(rankAgain == 1);
        REQUIRE(A == (expectedRef));
    }

    // Test case for a rectangular matrix with more columns than rows.
    SECTION("Rectangular matrix (3x4)") {
        Eigen::MatrixXi A(3, 4);
        A << 1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12;
        const int p = 13;

        size_t rank = reduceToREFAndGetRank(A, p);

        // The expected REF for the matrix modulo 13.
        Eigen::MatrixXi expectedRef(3, 4);
        expectedRef << 1, 2, 3, 4,
            0, 1, 2, 3,
            0, 0, 0, 0;

        REQUIRE(rank == 2);
        CHECK(A == (expectedRef));

        // Test for idempotence
        size_t rankAgain = reduceToREFAndGetRank(A, p);
        REQUIRE(rankAgain == 2);
        CHECK(A == (expectedRef));

        rankAgain = reduceToREFAndGetRank(A, p, true);
        // The expected RREF for the matrix modulo 13.
        Eigen::MatrixXi expected_rref(3, 4);
        expectedRef << 1, 0, 12, 11,
            0, 1, 2, 3,
            0, 0, 0, 0;
        REQUIRE(rankAgain == 2);
        REQUIRE(A == (expectedRef));
    }

    // Test case for a rectangular matrix with more columns than rows.
    SECTION("Rectangular matrix already reduced (3x4)") {
        Eigen::MatrixXi A(3, 4);
        A << 1, 2, 3, 4,
             0, 1, 1, 8,
             0, 0, 0, 12;
        const int p = 13;

        size_t rank = reduceToREFAndGetRank(A, p);

        // The expected REF for the matrix modulo 13.
        Eigen::MatrixXi expectedRef(3, 4);
        expectedRef << 1, 2, 3, 4,
             0, 1, 1, 8,
             0, 0, 0, 1;

        REQUIRE(rank == 3);
        REQUIRE(A == (expectedRef));

        // Test for idempotence
        size_t rankAgain = reduceToREFAndGetRank(A, p);
        REQUIRE(rankAgain == 3);
        REQUIRE(A == (expectedRef));
    }

    // Test case for a matrix where a pivot is not 1 and requires modular inverse.
    SECTION("Matrix with non-unity pivots") {
        Eigen::MatrixXi A(2, 2);
        A << 2, 3, 4, 5;
        const int p = 7; // mod 7, inverse of 2 is 4

        size_t rank = reduceToREFAndGetRank(A, p);

        Eigen::MatrixXi expectedRef(2, 2);
        expectedRef << 1, 5, 0, 1;

        REQUIRE(rank == 2);
        REQUIRE(A == expectedRef);

        // Test for idempotence
        size_t rankAgain = reduceToREFAndGetRank(A, p);
        REQUIRE(rankAgain == 2);
        REQUIRE(A == expectedRef);

        rankAgain = reduceToREFAndGetRank(A, p, true);
        REQUIRE(rankAgain == 2);
        REQUIRE(A.isIdentity());

        rankAgain = reduceToREFAndGetRank(A, p, true);
        REQUIRE(rankAgain == 2);
        REQUIRE(A.isIdentity());
    }

    // Test case for an all-zero matrix.
    SECTION("Zero matrix") {
        Eigen::MatrixXi A(3, 3);
        A.setZero();
        const int p = 5;

        size_t rank = reduceToREFAndGetRank(A, p);

        REQUIRE(rank == 0);
        REQUIRE(A.isZero());
    }

    // Test case for a 1x1 matrix.
    SECTION("1x1 matrix") {
        Eigen::MatrixXi A(1, 1);
        A << 5;
        const int p = 7;

        size_t rank = reduceToREFAndGetRank(A, p);
        Eigen::MatrixXi expectedRef(1, 1);
        expectedRef << 1;

        REQUIRE(rank == 1);
        REQUIRE(A == (expectedRef));
    }

    // Sparse example for vectorisation-based algorithm
    SECTION("Tallish matrix (6x5)") {
        const size_t p = 5;
        Eigen::MatrixXi A(6, 5);
        A << 1, 0, 3, 0, 1,
               0, 1, 0, 3, 2,
               0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               3, 0, 4, 0, 3,
               0, 3, 0, 4, 1;

        size_t rank = reduceToREFAndGetRank(A, p, true);
        Eigen::MatrixXi expectedRef(6, 5);
        expectedRef << 1, 0, 3, 0, 1,
               0, 1, 0, 3, 2,
               0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 0, 0;

        CHECK(rank == 2);
        REQUIRE(A == (expectedRef));
    }
}

TEST_CASE("check_phase", "[check_phase]") {
    SECTION("Works on known case") {
        // failed on macOS
        const size_t d = 7;
        const std::complex<double> v1 = {-8.11037, -6.10691}, v2 = {-8.11037,-6.10691};
        REQUIRE_THAT(checkPhase(d, v1, v2), Catch::Matchers::WithinAbs(0, 1e-5));
        REQUIRE_THAT(checkPhase(d, v2, v1), Catch::Matchers::WithinAbs(0, 1e-5));
    }

    SECTION("Works on random cases") {
        const size_t d = 7;
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-9.0, 9.0);
        std::uniform_int_distribution<> randIntegers(0, d - 1);

        const size_t iterations = 5000;
        for (size_t i = 0; i < iterations; i++) {
            const auto v2 = std::complex(dis(gen), dis(gen));
            const auto k = randIntegers(gen);
            const auto omegaPow = std::pow(omega, k);
            const auto v1 = omegaPow * v2;
            INFO("i = " << i << ", v1 = " << v1 << ", v2 = " << v2 << ", k = " << k);
            REQUIRE_THAT(checkPhase(d, v1, v2), Catch::Matchers::WithinAbs(k, 1e-5));
            REQUIRE_THAT(checkPhase(d, v2, v1), Catch::Matchers::WithinAbs((d - k) % d, 1e-5));
        }
    }
}

// A helper function to create a random complex-valued matrix.
Eigen::MatrixXcd create_random_matrix(int d) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-9.0, 9.0);

    Eigen::MatrixXcd matrix(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            matrix(i, j) = std::complex(dis(gen), dis(gen));
        }
    }
    return matrix;
}

TEST_CASE("W function", "[W]") {
    const size_t d = 7;
    const int inv_2 = fastPowerMod(2, d - 2, d);

    const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));

    SECTION("W(0,0) is identity") {
        const size_t dHere = 3;
        const int inv_2Here = fastPowerMod(2, d - 2, dHere);
        const std::complex<double> omegaHere = std::exp(std::complex<double>(0, 2.0 * pi / dHere));

        const auto result = W(dHere, 0, 0, inv_2Here, omegaHere);

        CHECK(result.isIdentity());
    }

    SECTION("Correct for predetermined result") {
        const size_t dHere = 3;
        const int inv_2Here = fastPowerMod(2, d - 2, dHere);
        const auto omegaHere = std::exp(std::complex<double>(0, 2.0 * pi / dHere));
        const Eigen::MatrixXcd result = W(dHere, 2, 1, inv_2Here, omegaHere);
        Eigen::Matrix3cd actual;
        actual << std::complex<double>(0, 0), std::complex<double>(0, 0),
            std::complex<double>(-0.5, -0.866025), std::complex<double>(-0.5, 0.866025),
            std::complex<double>(0, 0), std::complex<double>(0, 0), std::complex<double>(0, 0),
            std::complex<double>(1, 0), std::complex<double>(0, 0);
        INFO(result << " vs " << actual);
        for (size_t i = 0; i < result.rows(); ++i) {
            for (size_t j = 0; j < result.cols(); ++j) {
                CHECK_THAT(result(i, j).imag(),
                           Catch::Matchers::WithinAbs(actual(i, j).imag(), 1e-5));
                CHECK_THAT(result(i, j).real(),
                           Catch::Matchers::WithinAbs(actual(i, j).real(), 1e-5));
            }
        }
    }

    SECTION("Commutation relations") {
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                for (size_t iPrime = 0; iPrime < d; iPrime++) {
                    for (size_t jPrime = 0; jPrime < d; jPrime++) {
                        const Eigen::MatrixXcd left =
                            W(d, i, j, inv_2, omega) * W(d, iPrime, jPrime, inv_2, omega);
                        const Eigen::MatrixXcd right =
                            W(d, iPrime, jPrime, inv_2, omega) * W(d, i, j, inv_2, omega);

                        const auto omegaPow =
                            std::pow(omega, symplecticProduct(d, i, j, iPrime, jPrime));
                        for (size_t row = 0; row < d; row++) {
                            for (size_t col = 0; col < d; col++) {
                                REQUIRE(isApproxEqual(left(row, col), omegaPow * right(row, col)));
                            }
                        }
                    }
                }
            }
        }
    }
}

std::complex<double> fNaive(const Eigen::Ref<const Eigen::MatrixXcd>& M, int p, int q, int inv_2,
                            const std::complex<double>& omega) {

    if (M.rows() != M.cols()) {
        throw std::invalid_argument("Dimension mismatch");
    }
    if (M.rows() == 0) {
        throw std::invalid_argument("Zero matrix");
    }
    const auto d = M.rows();

    const Eigen::MatrixXcd W_matrix = W(d, -p, -q, inv_2, omega);
    const Eigen::MatrixXcd WtimesM = W_matrix * M;
    const auto tr = WtimesM.trace();

    return std::complex(1.0 / d, 0.0) * tr;
}

TEST_CASE("f function", "[f]") {

    const int d = 3; // Example dimension
    const int inv_2 = fastPowerMod(2, d - 2, d);

    const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));

    SECTION("f works on predetermined result") {
        CHECK((2 * inv_2) % d == 1);
        Eigen::Matrix3cd U;
        U << std::complex(-5.57693, -0.765222), std::complex(6.22008, -8.91947),
            std::complex(-0.637812, 0.571667), std::complex(-0.0336533, -1.70245),
            std::complex(7.87921, 3.40459), std::complex(-5.09563, -1.66903),
            std::complex(-8.89951, 2.14251), std::complex(0.366395, 0.542691),
            std::complex(-0.816918, 4.0729);
        auto naiveResult = fNaive(U, 0, 0, inv_2, omega);
        CHECK_THAT(naiveResult.real(), Catch::Matchers::WithinAbs(0.4951206666, 1e-5));
        CHECK_THAT(naiveResult.imag(), Catch::Matchers::WithinAbs(2.23742266, 1e-5));

        auto optimizedResult = f(U, 0, 0, inv_2, omega);
        CHECK_THAT(optimizedResult.real(), Catch::Matchers::WithinAbs(0.4951206666, 1e-5));
        CHECK_THAT(optimizedResult.imag(), Catch::Matchers::WithinAbs(2.23742266, 1e-5));
    }

    SECTION("f_multiqudit works on known result for 2 qubits") {
        constexpr std::complex<double> complexFor2ndTuple = {-1, 3};
        const Eigen::MatrixXcd M = Eigen::kroneckerProduct(
            W(d, 1, 2, inv_2, omega),
            W(d, 0, 1, inv_2, omega)) + complexFor2ndTuple * Eigen::kroneckerProduct(
                W(d, 0,0, inv_2, omega),
                W(d, 1,2, inv_2, omega));

        const Eigen::Vector<long, 4> nonZeroTuple = {1,2,0,1};
        const Eigen::Vector<long, 4> nonZeroTuple2 = {0, 0, 1, 2};
        for (const auto& tuple : TupleIterator<SingleModulus>(d, 4)) {
            INFO("Tuple is " << tuple[0] << ", " << tuple[1] << ", " << tuple[2] << ", " << tuple[3]);
            auto fValue = f_multiqudit(M, tuple, d, inv_2, omega);
            // f_M should be the expected values in the basis elements specified, and 0 everywhere else
            if (tuple == nonZeroTuple) {
                CHECK_THAT(fValue.imag(), Catch::Matchers::WithinAbs(0, 1e-5));
                CHECK_THAT(fValue.real(), Catch::Matchers::WithinAbs(1, 1e-5));
            } else if (tuple == nonZeroTuple2) {
                CHECK_THAT(fValue.imag(), Catch::Matchers::WithinAbs(complexFor2ndTuple.imag(), 1e-5));
                CHECK_THAT(fValue.real(), Catch::Matchers::WithinAbs(complexFor2ndTuple.real(), 1e-5));
            } else {
                CHECK_THAT(fValue.real(), Catch::Matchers::WithinAbs(0, 1e-5));
                CHECK_THAT(fValue.real(), Catch::Matchers::WithinAbs(0, 1e-5));
            }
        }
    }

    SECTION("f_multiqudit works on original 1-qubit case") {
        CHECK((2 * inv_2) % d == 1);
        Eigen::Matrix3cd U;
        U << std::complex(-5.57693, -0.765222), std::complex(6.22008, -8.91947),
            std::complex(-0.637812, 0.571667), std::complex(-0.0336533, -1.70245),
            std::complex(7.87921, 3.40459), std::complex(-5.09563, -1.66903),
            std::complex(-8.89951, 2.14251), std::complex(0.366395, 0.542691),
            std::complex(-0.816918, 4.0729);
        auto naiveResult = fNaive(U, 0, 0, inv_2, omega);
        CHECK_THAT(naiveResult.real(), Catch::Matchers::WithinAbs(0.4951206666, 1e-5));
        CHECK_THAT(naiveResult.imag(), Catch::Matchers::WithinAbs(2.23742266, 1e-5));

        Eigen::Vector<long, 2> zero = Eigen::Vector<long, 2>::Zero();
        auto optimizedResult = f_multiqudit(U, zero, d, inv_2, omega);
        CHECK_THAT(optimizedResult.real(), Catch::Matchers::WithinAbs(0.4951206666, 1e-5));
        CHECK_THAT(optimizedResult.imag(), Catch::Matchers::WithinAbs(2.23742266, 1e-5));
    }

    SECTION("Optimized f equivalent to naive approach") {

        size_t num_iterations = 500;
        for (size_t i = 0; i < num_iterations; ++i) {
            auto U = create_random_matrix(d);
            for (int p = 0; p < d; ++p) {
                for (int q = 0; q < d; ++q) {

                    INFO("Iteration " << i << " p " << p << " q " << q << ", U = " << U);
                    auto naiveValue = fNaive(U, p, q, inv_2, omega);
                    auto optimizedValue = f(U, p, q, inv_2, omega);
                    REQUIRE_THAT(naiveValue.imag(),
                                 Catch::Matchers::WithinAbs(optimizedValue.imag(), 1e-9));
                    REQUIRE_THAT(naiveValue.real(),
                                 Catch::Matchers::WithinAbs(optimizedValue.real(), 1e-9));
                }
            }
        }
    }

    SECTION("f_multiqudit equivalent to naive approach for 1 qudit") {
        size_t num_iterations = 500;
        for (size_t i = 0; i < num_iterations; ++i) {
            auto U = create_random_matrix(d);
            for (int p = 0; p < d; ++p) {
                for (int q = 0; q < d; ++q) {

                    INFO("Iteration " << i << " p " << p << " q " << q << ", U = " << U);
                    auto naiveValue = fNaive(U, p, q, inv_2, omega);
                    const Eigen::Vector<long, 2> pq = {p, q};
                    auto optimizedValue = f_multiqudit(U, pq, d, inv_2, omega);
                    REQUIRE_THAT(naiveValue.imag(),
                                 Catch::Matchers::WithinAbs(optimizedValue.imag(), 1e-9));
                    REQUIRE_THAT(naiveValue.real(),
                                 Catch::Matchers::WithinAbs(optimizedValue.real(), 1e-9));
                }
            }
        }
    }

    SECTION("Lemma 1 should hold") {
        size_t num_iterations = 500;
        for (size_t i = 0; i < num_iterations; ++i) {
            // Generate random matrices U and V
            auto U = create_random_matrix(d);
            auto V = create_random_matrix(d);

            // Pre-calculate the matrix product UV
            auto UV = U * V;
            // Loop through all possible p, q pairs and check the identities
            for (int p = 0; p < d; ++p) {
                for (int q = 0; q < d; ++q) {
                    std::complex<double> sumTemp7 = 0.0;
                    std::complex<double> sumTemp8 = 0.0;

                    // The left side of the identity check
                    std::complex<double> lhs = f(UV, p, q, inv_2, omega);

                    // Loops for Equation 7 and 8
                    for (int i = 0; i < d; ++i) {
                        for (int j = 0; j < d; ++j) {

                            // Omega term for Equation 7
                            int omega_exponent_7 = safeMod(inv_2 * (q * i - p * j), d);
                            std::complex<double> omega_term_7 =
                                std::pow(omega, static_cast<double>(omega_exponent_7));

                            // Equation 7 sum
                            sumTemp7 += omega_term_7 * f(U, i, j, inv_2, omega) *
                                        f(V, p - i, q - j, inv_2, omega);

                            // Omega term for Equation 8
                            int omega_exponent_8 = safeMod(-inv_2 * (q * i - p * j), d);
                            std::complex<double> omega_term_8 =
                                std::pow(omega, static_cast<double>(omega_exponent_8));

                            // Equation 8 sum
                            sumTemp8 += omega_term_8 * f(U, p - i, q - j, inv_2, omega) *
                                        f(V, i, j, inv_2, omega);
                        }
                    }

                    REQUIRE(isApproxEqual(lhs, sumTemp7));

                    REQUIRE(isApproxEqual(lhs, sumTemp8));
                }
            }
        }
    }
}
