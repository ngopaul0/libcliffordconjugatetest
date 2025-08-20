#include <catch2/catch_test_macros.hpp>
#include <complex>

#include <Eigen/Dense>
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "cliffordconjugacytest.hpp"
#include "generalized/vectorisationalgorithm.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/util.h"

using namespace cliffconjtest;

TEST_CASE("Equal matrices are Clifford-conjugate", "[equal]") {
    SECTION("Two identical matrices should be equal") {
        Eigen::Matrix2cd M;
        M << std::complex(1.0, 2.0), std::complex(3.0, 4.0), std::complex(5.0, 6.0),
            std::complex(7.0, 8.0);

        Eigen::Matrix2cd M_prime;
        M_prime << std::complex(1.0, 2.0), std::complex(3.0, 4.0), std::complex(5.0, 6.0),
            std::complex(7.0, 8.0);
        REQUIRE(isCliffordConjugate(M, M_prime));
    }

    SECTION("Works on equal matrices") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const auto M = W(d, 1, 2, inv_2, omega);
        INFO("Matrix is " << M);
        REQUIRE(isCliffordConjugate(M, M));
    }

    SECTION("Brute force on equal matrices") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const auto M = W(d, 1, 2, inv_2, omega);

        INFO("Matrix is " << M);
        const auto M_p = createMpMatrix(M, omega, inv_2);
        REQUIRE(bruteForceTestCliffordConjugacy(M, M, omega, M_p, M_p));
    }
}

TEST_CASE("zero matrix", "[zero]") {
    SECTION("Succeeds with two zero matrices") {
        const int d = 3; // Example dimension
        const Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(d, d);
        const Eigen::MatrixXcd Mprime = Eigen::MatrixXcd::Zero(d, d);

        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("Fails on zero matrix with nonzero matrix") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd M = W(d, 1, 2, inv_2, omega);
        const Eigen::MatrixXcd Mprime = Eigen::MatrixXcd::Zero(d, d);

        REQUIRE(!isCliffordConjugate(M, Mprime));
    }
}

TEST_CASE("single Pauli basis element", "[single]") {
    SECTION("Brute force on single Pauli basis element") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd M = W(d, 1, 2, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd Mprime = C * M * Cstar;

        INFO("Matrix is " << (M));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);

        REQUIRE(bruteForceTestCliffordConjugacy(M, Mprime, omega, M_p, Mprime_p));
    }

    SECTION("Algorithm works on single Pauli basis element") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd M = W(d, 1, 2, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd Mprime = C * M * Cstar;

        INFO("Matrix is " << (M));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);

        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("Algorithm works on non-example single Pauli basis element") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd M = W(d, 1, 2, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd Mprime = C * (2 * M) * Cstar;

        INFO("Matrix is " << (M));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);

        REQUIRE(!isCliffordConjugate(M, Mprime));
    }
}

TEST_CASE("multiple Pauli basis elements", "[multiple]") {
    SECTION("d=3: All basis elements have coeff 1", "[single-qubit][d=3]") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        Eigen::MatrixXcd Mprime = Eigen::MatrixXcd::Zero(d, d);
        for (size_t p = 0; p < d; p++) {
            for (size_t q = 0; q < d; q++) {
                Mprime += 1.0 * W(d, p, q, inv_2, omega);
            }
        }

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;
        INFO("M = " << M);
        INFO("Mprime = " << Mprime);
        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("d=11: Linearly dependent M_p", "[single-qubit][d=11]") {
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega) +
                                        W(d, 3, 6, inv_2, omega) + W(d, 4, 8, inv_2, omega) +
                                        W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;
        INFO("M = " << M);
        INFO("Mprime = " << Mprime);
        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("Generalized algorithm: d=11: Linearly dependent M_p", "[generalized][d=11]") {
        SKIP("Takes too long (48-50s)");
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega) +
                                        W(d, 3, 6, inv_2, omega) + W(d, 4, 8, inv_2, omega) +
                                        W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;
        INFO("M = " << M);
        INFO("Mprime = " << Mprime);
        const auto isCliffordConjugate_M_Mprime = isCliffordConjugateGeneralized(d, 1, M, Mprime);
        INFO("s_numRecursiveCalls = " << returnLastNumRecursiveCalls());
        REQUIRE(isCliffordConjugate_M_Mprime);
    }

    SECTION("d = 3: Two Pauli basis elements") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 1, 1, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        Eigen::MatrixXcd result = M;
        Eigen::Matrix3cd expected;
        expected << std::complex(0.0, 0.0), std::complex(-0.5, 0.8660254040),
            std::complex(-0.5000000002, -0.8660254040), std::complex(-0.5000000002, 0.8660254042),
            std::complex(0.0, 0.0), std::complex(1.0, 0.0),
            std::complex(-0.5000000002, -0.8660254040), std::complex(1.0, 0.0),
            std::complex(0.0, 0.0);
        INFO((result) << " vs " << (expected));
        for (size_t i = 0; i < result.rows(); ++i) {
            for (size_t j = 0; j < result.cols(); ++j) {
                REQUIRE_THAT(result(i, j).imag(),
                             Catch::Matchers::WithinAbs(expected(i, j).imag(), 1e-5));
                REQUIRE_THAT(result(i, j).real(),
                             Catch::Matchers::WithinAbs(expected(i, j).real(), 1e-5));
            }
        }

        result = Mprime;
        Eigen::Matrix3cd expectedMprime;
        expectedMprime << std::complex(0.0, 0.0), std::complex(-0.5000000004, -0.8660254040),
            std::complex(-0.5000000000, 0.8660254040), std::complex(-0.5000000004, -0.8660254040),
            std::complex(0.0, 0.0), std::complex(1.000000001, -0.0000000003233738859),
            std::complex(-0.5000000002, 0.8660254043), std::complex(1.000000000, 0.0),
            std::complex(0.0, 0.0);
        INFO((result) << " vs " << (expected));
        for (size_t i = 0; i < result.rows(); ++i) {
            for (size_t j = 0; j < result.cols(); ++j) {
                REQUIRE_THAT(result(i, j).imag(),
                             Catch::Matchers::WithinAbs(expectedMprime(i, j).imag(), 1e-5));
                REQUIRE_THAT(result(i, j).real(),
                             Catch::Matchers::WithinAbs(expectedMprime(i, j).real(), 1e-5));
            }
        }

        INFO("Matrix is " << (M));
        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("d = 3: Two Pauli basis elements, non-trivial pPrime, qPrime") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 1, 1, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2) * makeX(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        INFO("Matrix is " << (M));
        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("d = 5: Non-example, two linearly dependent Pauli basis elements bypassing histogram "
            "check") {
        // This will actually pass the histogram check.

        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        Eigen::Vector2i v1(1, 2), v2(2, 4), v3(3, 1);

        const Eigen::MatrixXcd Mprime = W(d, v1(0), v1(1), inv_2, omega) +
                                        W(d, v2(0), v2(1), inv_2, omega) +
                                        2 * W(d, v3(0), v3(1), inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2) * makeX(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();

        Eigen::Matrix2i nonSymplecticTransform;
        nonSymplecticTransform << 2, 0, 0, 2;
        // This asserts it's not symplectic
        REQUIRE(safeMod(nonSymplecticTransform.determinant(), d) != 1);

        // Permute the basis element coordinates by a nonsymplectic transformation. Keep the
        // coefficients the same to bypass the histogram check.
        Eigen::Vector2i w1 = nonSymplecticTransform * v1;
        Eigen::Vector2i w2 = nonSymplecticTransform * v2;
        Eigen::Vector2i w3 = nonSymplecticTransform * v3;

        const Eigen::MatrixXcd M = 2 * W(d, w1(0), w1(1), inv_2, omega) +
                                   W(d, w2(0), w2(1), inv_2, omega) +
                                   W(d, w3(0), w3(1), inv_2, omega);

        // Ensure the trace check is bypassed.
        REQUIRE_THAT(Mprime.trace().imag(), Catch::Matchers::WithinAbs(M.trace().imag(), 1e-5));
        REQUIRE_THAT(Mprime.trace().real(), Catch::Matchers::WithinAbs(M.trace().real(), 1e-5));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);
        CHECK(!bruteForceTestCliffordConjugacy(M, Mprime, omega, M_p, Mprime_p));

        INFO("Matrix is " << (M));
        REQUIRE(!isCliffordConjugate(M, Mprime));
    }

    SECTION("d = 5: Non-example, two Pauli basis elements bypassing histogram check") {
        // This will actually pass the histogram check.

        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        Eigen::Vector2i v1(1, 2), v2(2, 2);

        const Eigen::MatrixXcd Mprime =
            2 * W(d, v1(0), v1(1), inv_2, omega) + 3 * W(d, v2(0), v2(1), inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2) * makeX(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();

        Eigen::Matrix2i nonSymplecticTransform;
        nonSymplecticTransform << 2, 0, 0, 2;
        // This asserts it's not symplectic
        REQUIRE(safeMod(nonSymplecticTransform.determinant(), d) != 1);

        // Permute the basis element coordinates by a nonsymplectic transformation. Keep the
        // coefficients the same to bypass the histogram check.
        Eigen::Vector2i w1 = nonSymplecticTransform * v1;
        Eigen::Vector2i w2 = nonSymplecticTransform * v2;

        const Eigen::MatrixXcd M =
            2 * W(d, w1(0), w1(1), inv_2, omega) + 3 * W(d, w2(0), w2(1), inv_2, omega);

        // Ensure the trace check is bypassed.
        REQUIRE_THAT(Mprime.trace().imag(), Catch::Matchers::WithinAbs(M.trace().imag(), 1e-5));
        REQUIRE_THAT(Mprime.trace().real(), Catch::Matchers::WithinAbs(M.trace().real(), 1e-5));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);
        CHECK(!bruteForceTestCliffordConjugacy(M, Mprime, omega, M_p, Mprime_p));

        INFO("Matrix is " << (M));
        REQUIRE(!isCliffordConjugate(M, Mprime));
    }

    SECTION("Generalized algorithm: d = 5: Non-example, two Pauli basis elements bypassing histogram check") {
        // This will actually pass the histogram check.

        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        Eigen::Vector2i v1(1, 2), v2(2, 2);

        const Eigen::MatrixXcd Mprime =
            2 * W(d, v1(0), v1(1), inv_2, omega) + 3 * W(d, v2(0), v2(1), inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2) * makeX(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();

        Eigen::Matrix2i nonSymplecticTransform;
        nonSymplecticTransform << 2, 0, 0, 2;
        // This asserts it's not symplectic
        REQUIRE(safeMod(nonSymplecticTransform.determinant(), d) != 1);

        // Permute the basis element coordinates by a nonsymplectic transformation. Keep the
        // coefficients the same to bypass the histogram check.
        Eigen::Vector2i w1 = nonSymplecticTransform * v1;
        Eigen::Vector2i w2 = nonSymplecticTransform * v2;

        const Eigen::MatrixXcd M =
            2 * W(d, w1(0), w1(1), inv_2, omega) + 3 * W(d, w2(0), w2(1), inv_2, omega);

        // Ensure the trace check is bypassed.
        REQUIRE_THAT(Mprime.trace().imag(), Catch::Matchers::WithinAbs(M.trace().imag(), 1e-5));
        REQUIRE_THAT(Mprime.trace().real(), Catch::Matchers::WithinAbs(M.trace().real(), 1e-5));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);
        CHECK(!bruteForceTestCliffordConjugacy(M, Mprime, omega, M_p, Mprime_p));

        INFO("Matrix is " << (M));
        REQUIRE(!isCliffordConjugateGeneralized(d, 1, M, Mprime));
    }

    SECTION("d = 3, Linearly dependent M_p") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 1, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2) * makeX(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        INFO("Matrix is " << (M));
        REQUIRE(isCliffordConjugate(M, Mprime));
    }

    SECTION("Generalized algorithm: d = 3, Linearly dependent M_p") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 1, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2) * makeX(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        INFO("Matrix is " << (M));
        REQUIRE(isCliffordConjugateGeneralized(d, 1, M, Mprime));
    }

    SECTION("d=5: Algorithm works on linearly dependent M_p") {
        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega) +
                                        W(d, 3, 6, inv_2, omega) + W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        INFO("Matrix is " << (M));
        REQUIRE(isCliffordConjugate(Mprime, M));
    }

    SECTION("d=5: Brute-force works on linearly dependent M_p") {
        // d = 3 and M' = W(1,2) + W(2,1) and cliffordPermutationGate(d, 2) results in nonunique
        // symplectic transforms
        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 1, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);

        INFO("Matrix is " << (M));
        auto result = bruteForceTestCliffordConjugacy<true>(M, Mprime, omega, M_p, Mprime_p);
        REQUIRE(result.has_value());
    }
}

TEST_CASE("generalized algorithm slow cases", "[generalized][slow]") {
    SECTION("d=11: Linearly dependent M_p") {
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega) +  W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;
        bool isCliffordConjugate = isCliffordConjugateGeneralized(d, 1, M, Mprime);
        REQUIRE(isCliffordConjugate);
    }

    SECTION("d=7: Linearly dependent M_p", "[generalized][d=7]") {
        const int d = 7;
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 2, 3, inv_2, omega) *cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        bool isCliffordConjugate = isCliffordConjugateGeneralized(d, 1, M, Mprime);
        REQUIRE(isCliffordConjugate);
    }

    SECTION("Generalized: d=7, Clifford-conjugate, 2 basis elements", "[generalized][d=7]") {
        const int d = 7; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));

        const Eigen::MatrixXcd W1 = W(d, 1, 2, inv_2, omega);
        const Eigen::MatrixXcd W2 = W(d, 1, 1, inv_2, omega);
        const Eigen::MatrixXcd Mprime = W1 + W2;

        const auto X = makeX(d, 2);
        const auto cliffordGate = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd C = cliffordGate * X;
        const auto Cstar = C.adjoint();
        const auto Wmatrix = W(d, 1, 2, inv_2, omega);
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        bool isCliffordConjugate = isCliffordConjugateGeneralized(d, 1, M, Mprime);
        REQUIRE(isCliffordConjugate);
    }
}

Eigen::MatrixXcd computeMSum(size_t d, std::vector<std::pair<long, long>> coords, size_t inv_2, const std::complex<double>& omega) {
    Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(d, d);
    for (const auto& [p, q] : coords) {
        sum += W(d, p, q, inv_2, omega);
    }
    return sum;
}

TEST_CASE("Multi qudit case", "[multiqudit]") {
    SECTION("d=3, 2 qudits", "[generalized][d=3][n=2]") {
        const int d = 3; // Example dimension
        const size_t n = 2;
        const int inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        constexpr std::complex<double> complexFor2ndTuple = {-1, 3};
        const std::vector<std::pair<long, long>> coords1 = {{1,2}, {0,1}, {2,2}};
        const std::vector<std::pair<long, long>> coords2 = {{2,1}, {2,0}};

        std::vector<Eigen::Vector<long, 4>> allTuples;
        for (const auto& [p, q] : coords1) {
            for (const auto& [r, s] : coords2) {
                allTuples.push_back({p, q, r, s});
            }
        }

        const Eigen::MatrixXcd M1 = computeMSum(d, coords1, inv_2, omega);
        const Eigen::MatrixXcd M2 = computeMSum(d, coords2, inv_2, omega);
        const Eigen::MatrixXcd M = Eigen::kroneckerProduct(M1, M2);

        const Eigen::MatrixXcd C1 = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd C2 = W(d, 2,2, inv_2, omega) * makeZX(d, 2, 1);
        const Eigen::MatrixXcd C = Eigen::kroneckerProduct(C1, C2);
        const Eigen::MatrixXcd Mprime = C * M * C.adjoint();

        const Eigen::MatrixXcd M1prime = C1 * M1 * C1.adjoint();
        const Eigen::MatrixXcd M2prime = C2 * M2 * C2.adjoint();

        const auto M1_p = createMpMatrix((M1), omega, inv_2);
        const auto M1prime_p = createMpMatrix((M1prime), omega, inv_2);
        std::optional<Lemma10Info> result1 = bruteForceTestCliffordConjugacy<true>(M1, M1prime, omega, M1_p, M1prime_p);
        std::stringstream ssC1;
        ssC1 << result1->first;
        auto stringC1 = ssC1.str();
        const auto M2_p = createMpMatrix((M2), omega, inv_2);
        const auto M2prime_p = createMpMatrix((M2prime), omega, inv_2);
        std::optional<Lemma10Info> result2 = bruteForceTestCliffordConjugacy<true>(M2, M2prime, omega, M2_p, M2prime_p);
        std::stringstream ssC2;
        ssC2 << result2->first;
        auto stringC2 = ssC2.str();
        const size_t pPrime1 = result1->second.first;
        const size_t qPrime1 = result1->second.second;
        const size_t pPrime2 = result2->second.first;
        const size_t qPrime2 = result2->second.second;

        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S = directSum(result1->first, result2->first);
        std::stringstream ssC;
        ssC << S;
        auto stringC = ssC.str();

        FMap MMap(d, n, 1e-5, 1e-5);
        MpMatrixType M_p(2 * n, d);
        for (auto tuple : M_p.indexIterator()) {
            const auto val = f_multiqudit(M, tuple, d, inv_2, omega);
            M_p.get(tuple) = val;
            MMap.insertEntryNTuple(std::move(tuple), val);
        }

        FMap MprimeMap(d, n, 1e-5, 1e-5);
        MpMatrixType Mprime_p(2 * n, d);
        for (auto tuple : Mprime_p.indexIterator()) {
            const auto val = f_multiqudit(Mprime, tuple, d, inv_2, omega);
            Mprime_p.get(tuple) = val;
            MprimeMap.insertEntryNTuple(std::move(tuple), val);
        }

        std::vector<std::array<long, 4>> allTuplesMapped;
        for (const auto& tuple : allTuples) {
            Eigen::Vector<long, Eigen::Dynamic> prod = modMatrix(S * tuple, d);
            allTuplesMapped.push_back({prod(0), prod(1), prod(2), prod(3)});
        }

        Eigen::Vector<long, 4> pPrimeQPrimeVec(pPrime1, qPrime1, pPrime2, qPrime2);
        REQUIRE(test_clifford_conjugate_lemma_10(d, pPrimeQPrimeVec, omega, M, M_p, Mprime_p, S));

        REQUIRE(isCliffordConjugateGeneralized(d, n, M, Mprime));
    }
}
