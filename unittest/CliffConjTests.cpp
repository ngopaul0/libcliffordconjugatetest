#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <random>

#include <Eigen/Dense>
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "cliffordconjugacytest.hpp"
#include "generalized/vectorisationalgorithm.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/complexexactrepr.h"
#include "internal/util.h"

using namespace cliffconjtest;

TEST_CASE("Seed bit dropping") {
    const std::mt19937::result_type originalSeedValue = 13691651030999647805;
    const unsigned long seedValueUnsigned = originalSeedValue;
    const std::uint_fast32_t finalSeedValue = seedValueUnsigned;
    REQUIRE(originalSeedValue == finalSeedValue);
}

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
        // SKIP("Takes too long (48-50s)");
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

    SECTION("Generalized algorithm: d = 3, varied histogram") {
        const int d = 3; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = 2 * W(d, 1, 2, inv_2, omega) + W(d, 1, 1, inv_2, omega) + W(d, 2, 2, inv_2, omega) + W(d, 0, 0, inv_2, omega);

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
    // Corresponds to the Clifford gate at index i=106092 from n2-c2-gates-d3-asGATES.npy
    // with shuffle seed 13691651030999647805
    SECTION("n = 2, d = 3, slow case", "[generalized][d=3][n=2]") {
        const int d = 3;
        const size_t n = 2;
        const long inv_2 = modInverse(2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * cliffconjtest::pi / d));

        // This example will currently fail at i = 7 without the early checks against error propagation
        // in checkPhase
        const Eigen::MatrixXcd M = // Eigen::kroneckerProduct(M1, M2) + H
            - std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));

        const std::mt19937::result_type seedVal = 13691651030999647805;

        /*
         Processing Gate i=106092, seed = 13691651030999647805:
         */

        Eigen::Matrix<std::complex<double>, 9, 9> C;
        C << ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(),
        ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(),
        ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(),
        ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(),
        ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(),
        ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(),
        ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(),
        ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(),
        ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx();

        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime = C * M * C.adjoint();

        REQUIRE(isCliffordConjugateGeneralized(d, n, M, Mprime, std::make_optional(seedVal)));
    }

    // Corresponds to the Clifford gate at index i=7 from n2-c2-gates-d3-asGATES.npy
    SECTION("d = 3, 2 qudits, 2nd failure from multiqudit test", "[generalized][d=3][n=2]") {
        Eigen::Matrix<std::complex<double>, 9, 9> roundedC;
        roundedC <<
        std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0), std::complex<double>(0.333333, 0),
        std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, -1.32389e-16), std::complex<double>(0.333333, -1.32389e-16), std::complex<double>(0.333333, -1.32389e-16),
        std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, 1.66533e-16), std::complex<double>(0.333333, 1.66533e-16), std::complex<double>(0.333333, 1.66533e-16), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, 0.288675),
        std::complex<double>(0.333333, 0), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, 0), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, 0), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675),
        std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, -1.29526e-16), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, -1.11022e-16), std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, -1.32389e-16), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675),
        std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, -2.77556e-16), std::complex<double>(0.333333, 1.66533e-16), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, -2.40548e-16), std::complex<double>(-0.166667, -0.288675),
        std::complex<double>(0.333333, 0), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, 0), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, 0), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675),
        std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, -1.29526e-16), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, -4.996e-16), std::complex<double>(0.333333, -1.32389e-16), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675),
        std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, -2.77556e-16), std::complex<double>(-0.166667, 0.288675), std::complex<double>(0.333333, 1.66533e-16), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(-0.166667, 0.288675), std::complex<double>(-0.166667, -0.288675), std::complex<double>(0.333333, -6.29126e-16);

        const int d = 3;
        const size_t n = 2;
        const long inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));

        const Eigen::MatrixXcd M = -std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> MprimeFromRounded =
            roundedC * M * roundedC.adjoint();

        REQUIRE(isCliffordConjugateGeneralized(d, n, M, MprimeFromRounded, false));

        // The data directly from numpy has more precision here.
        Eigen::Matrix<std::complex<double>, 9, 9> numpyC;
        numpyC << ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066575,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(),
        ComplexExactRepr(-4628199217061079726,-4624500108022500578).cmplx(), ComplexExactRepr(-4628199217061079726,-4624500108022500578).cmplx(), ComplexExactRepr(-4628199217061079726,-4624500108022500578).cmplx(), ComplexExactRepr(-4628199217061079721,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079728,4598871928832275223).cmplx(), ComplexExactRepr(-4628199217061079722,4598871928832275228).cmplx(), ComplexExactRepr(4599676419421066581,-4854013680152999799).cmplx(), ComplexExactRepr(4599676419421066581,-4854013680152999799).cmplx(), ComplexExactRepr(4599676419421066581,-4854013680152999799).cmplx(),
        ComplexExactRepr(-4628199217061079722,-4624500108022500579).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500579).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500579).cmplx(), ComplexExactRepr(4599676419421066584,4370743438363066368).cmplx(), ComplexExactRepr(4599676419421066584,4370743438363066368).cmplx(), ComplexExactRepr(4599676419421066584,4370743438363066368).cmplx(), ComplexExactRepr(-4628199217061079725,4598871928832275231).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079726,4598871928832275230).cmplx(),
        ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079725,-4624500108022500587).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(-4628199217061079719,-4624500108022500582).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(-4628199217061079719,-4624500108022500582).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275230).cmplx(),
        ComplexExactRepr(-4628199217061079726,-4624500108022500578).cmplx(), ComplexExactRepr(-4628199217061079722,4598871928832275229).cmplx(), ComplexExactRepr(4599676419421066581,-4854129798367499606).cmplx(), ComplexExactRepr(-4628199217061079721,4598871928832275229).cmplx(), ComplexExactRepr(4599676419421066575,-4854880398305394698).cmplx(), ComplexExactRepr(-4628199217061079715,-4624500108022500582).cmplx(), ComplexExactRepr(4599676419421066581,-4854013680152999799).cmplx(), ComplexExactRepr(-4628199217061079715,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079736,4598871928832275231).cmplx(),
        ComplexExactRepr(-4628199217061079722,-4624500108022500579).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066581,-4849250898771181568).cmplx(), ComplexExactRepr(4599676419421066584,4370743438363066368).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500578).cmplx(), ComplexExactRepr(-4628199217061079724,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079725,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066575,-4850001498709076656).cmplx(), ComplexExactRepr(-4628199217061079712,-4624500108022500583).cmplx(),
        ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079738,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066581,0).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(),
        ComplexExactRepr(-4628199217061079726,-4624500108022500578).cmplx(), ComplexExactRepr(4599676419421066581,-4854129798367499606).cmplx(), ComplexExactRepr(-4628199217061079735,4598871928832275231).cmplx(), ComplexExactRepr(-4628199217061079721,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079721,-4624500108022500587).cmplx(), ComplexExactRepr(4599676419421066580,-4845310249097232384).cmplx(), ComplexExactRepr(4599676419421066581,-4854013680152999799).cmplx(), ComplexExactRepr(-4628199217061079736,4598871928832275231).cmplx(), ComplexExactRepr(-4628199217061079704,-4624500108022500588).cmplx(),
        ComplexExactRepr(-4628199217061079722,-4624500108022500579).cmplx(), ComplexExactRepr(4599676419421066581,-4849250898771181568).cmplx(), ComplexExactRepr(-4628199217061079740,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066584,4370743438363066368).cmplx(), ComplexExactRepr(-4628199217061079724,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079711,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079725,4598871928832275231).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500588).cmplx(), ComplexExactRepr(4599676419421066580,-4843996699205915990).cmplx();
        REQUIRE(numpyC != roundedC);
        REQUIRE(numpyC.isApprox(roundedC, 1e-4));
        REQUIRE(numpyC.isApprox(roundedC, 1e-5));
        // Precision is lost when rounding.
        REQUIRE(!numpyC.isApprox(roundedC, 1e-6));
        REQUIRE(!numpyC.isApprox(roundedC, 1e-7));

        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> MprimeFromNumPyC =
            numpyC * M * numpyC.adjoint();

        // This will fail if checkPhase doesn't account for precision issues.
        REQUIRE(isCliffordConjugateGeneralized(d, n, M, MprimeFromNumPyC, false));
    }

    SECTION("d=3, 2 qudits", "[generalized][d=3][n=2]") {
        const int d = 3;
        const size_t n = 2;
        const int inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        constexpr std::complex<double> complexFor2ndTuple = {-1, 3};
        const std::vector<std::pair<long, long>> coords1 = {{1,2}, {0,1}, {2,2}};
        const std::vector<std::pair<long, long>> coords2 = {{2,1}, {2,0}};

        const Eigen::MatrixXcd M1 = computeMSum(d, coords1, inv_2, omega);
        const Eigen::MatrixXcd M2 = computeMSum(d, coords2, inv_2, omega);
        const Eigen::MatrixXcd M = Eigen::kroneckerProduct(M1, M2);

        const Eigen::MatrixXcd C1 = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd C2 = W(d, 2,2, inv_2, omega) * makeZX(d, 2, 1);
        const Eigen::MatrixXcd C = Eigen::kroneckerProduct(C1, C2);
        const Eigen::MatrixXcd Mprime = C * M * C.adjoint();

        REQUIRE(isCliffordConjugateGeneralized(d, n, M, Mprime));
    }

    SECTION("d = 3, 2 qudits, failure from multiqudit test", "[generalized][d=3][n=2]") {
        // Index 18 from n2-c2-gates-d3-asGATES.npy
        Eigen::Matrix<std::complex<double>, 9, 9> C1;
        C1 <<
            std::complex<double>(0.57735, 0),    std::complex<double>(0.57735, 0),    std::complex<double>(0.57735, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),
            std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0.57735, 0),    std::complex<double>(0.57735, 0),    std::complex<double>(0.57735, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),
            std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0.57735, 0),    std::complex<double>(0.57735, 0),    std::complex<double>(0.57735, 0),
            std::complex<double>(0.57735, 0),    std::complex<double>(-0.288675, -0.5), std::complex<double>(-0.288675, 0.5),  std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),
            std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0.57735, 0),    std::complex<double>(-0.288675, -0.5), std::complex<double>(-0.288675, 0.5),  std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),
            std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0.57735, 0),    std::complex<double>(-0.288675, -0.5), std::complex<double>(-0.288675, 0.5),
            std::complex<double>(0.57735, 0),    std::complex<double>(-0.288675, 0.5),  std::complex<double>(-0.288675, -0.5), std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),
            std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0.57735, 0),    std::complex<double>(-0.288675, 0.5),  std::complex<double>(-0.288675, -0.5), std::complex<double>(0, 0),    std::complex<double>(0, 0),    std::complex<double>(0, 0),
            std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0, 0),          std::complex<double>(0.57735, 0),    std::complex<double>(-0.288675, 0.5),  std::complex<double>(-0.288675, -0.5);

        Eigen::Matrix<std::complex<double>, 9, 9> C;

        C <<
        std::complex<double>(0.333333,0),       std::complex<double>(0.333333,-2.04687e-17), std::complex<double>(0.333333,-4.25989e-18), std::complex<double>(-0.166667,0.288675),  std::complex<double>(-0.166667,0.288675),  std::complex<double>(-0.166667,0.288675),  std::complex<double>(0.333333,-7.91653e-17), std::complex<double>(0.333333,-7.91653e-17), std::complex<double>(0.333333,-7.91653e-17),
        std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675),
        std::complex<double>(0.333333,0),       std::complex<double>(0.333333,0),       std::complex<double>(0.333333,0),       std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675),
        std::complex<double>(0.333333,0),       std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(0.333333,-1.38778e-16), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-7.91653e-17), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675),
        std::complex<double>(-0.166667,0.288675), std::complex<double>(0.333333,-2.27195e-16), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(0.333333,-1.68498e-16), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(0.333333,-3.05311e-16),
        std::complex<double>(0.333333,0),       std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(0.333333,-3.56721e-16), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(0.333333,-2.81816e-16),
        std::complex<double>(0.333333,0),       std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-5.27356e-16), std::complex<double>(0.333333,-7.91653e-17), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675),
        std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-6.15773e-16), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-5.40868e-16), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-3.05311e-16), std::complex<double>(-0.166667,0.288675),
        std::complex<double>(0.333333,0),       std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-3.56721e-16), std::complex<double>(-0.166667,0.288675), std::complex<double>(-0.166667,-0.288675), std::complex<double>(0.333333,-2.98024e-16), std::complex<double>(-0.166667,0.288675);

        const int d = 3;
        const size_t n = 2;
        const long inv_2 = modInverse(2, d);
        const std::complex<double> omega =std::exp(std::complex<double>(0, 2.0 * pi / d));
        const std::vector<std::pair<long, long>> coords1 = {{1, 2}, {0, 1}, {2, 2}};
        const std::vector<std::pair<long, long>> coords2 = {{2, 1}};

        const Eigen::MatrixXcd M1 = computeMSum(d, coords1, inv_2, omega);
        const Eigen::MatrixXcd M2 = computeMSum(d, coords2, inv_2, omega);
        const Eigen::MatrixXcd M = Eigen::kroneckerProduct(M1, M2);

        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime = C * M * C.adjoint();

        REQUIRE(isCliffordConjugateGeneralized(d, n, M, Mprime));
    }

    SECTION("d=3, 2 qudits, nonexample", "[generalized][d=3][n=2][nonexample]") {
        const int d = 3;
        const size_t n = 2;
        const int inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        constexpr std::complex<double> complexFor2ndTuple = {-1, 3};
        const std::vector<std::pair<long, long>> coords1 = {{1,2}, {0,1}, {2,2}};
        const std::vector<std::pair<long, long>> coords2 = {{2,1}, {2,0}};

        const Eigen::MatrixXcd M1 = computeMSum(d, coords1, inv_2, omega);
        const Eigen::MatrixXcd M2 = computeMSum(d, coords2, inv_2, omega);
        const Eigen::MatrixXcd M = Eigen::kroneckerProduct(M1, M2);

        const Eigen::MatrixXcd C1 = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd C2 = W(d, 2,2, inv_2, omega) * makeZX(d, 2, 1);
        const Eigen::MatrixXcd C = Eigen::kroneckerProduct(C1, C2);
        Eigen::MatrixXcd Mprime = C * M * C.adjoint();

        Mprime(1, 1) += 0.5;

        REQUIRE(!isCliffordConjugateGeneralized(d, n, M, Mprime));
    }
}
