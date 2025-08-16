#include <catch2/catch_test_macros.hpp>
#include <complex>

#include <Eigen/Dense>
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "cliffordconjugacytest.hpp"
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
    SECTION("d=11: Linearly dependent M_p") {
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
