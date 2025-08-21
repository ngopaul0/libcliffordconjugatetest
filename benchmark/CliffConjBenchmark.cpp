#include <catch2/catch_test_macros.hpp>
#include <complex>

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "cliffordconjugacytest.hpp"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/symplecticiterator.h"
#include "internal/util.h"

using namespace cliffconjtest;

TEST_CASE("Brute force Clifford-conjugate test", "[benchmark][bruteforce]") {
    BENCHMARK_ADVANCED("d=5: Non-example, Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        Eigen::MatrixXcd M = C * Mprime * Cstar;
        M(0,3) += 0.2;

        CliffordGateIterator iterator(d);

        meter.measure([M, Mprime, iterator] {
            for (const auto& C_it : iterator) {
                if (M.isApprox(C_it * Mprime * C_it.adjoint(), 1e-5)) {
                    FAIL("Expected to be not Clifford conjugate");;
                }
            }
            return true;
        });
    };

    BENCHMARK_ADVANCED("d=13: Non-example, Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 13; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        Eigen::MatrixXcd M = C * Mprime * Cstar;
        M(0,3) += 0.2;

        auto M_p = createMpMatrix(M, omega, inv_2);
        auto Mprime_p = createMpMatrix(Mprime, omega, inv_2);
        Sp1ZdGates gates(d);

        meter.measure([M, Mprime, omega, M_p, Mprime_p, gates] {
            bool result = bruteForceTestCliffordConjugacy(
                M, Mprime, omega, M_p, Mprime_p, std::make_optional(std::ref(gates)));
            if (result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("expected to be not Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=11: Linearly dependent M_p -- direct Clifford conjugation")(Catch::Benchmark::Chronometer meter) {
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega) +  W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        CliffordGateIterator iterator(d);

        meter.measure([M, Mprime, iterator] {
            for (const auto& C_it : iterator) {
                if (M.isApprox(C_it * Mprime * C_it.adjoint(), 1e-5)) {
                    return;
                }
            }
            FAIL("Expected to be Clifford conjugate");
        });
    };

    BENCHMARK_ADVANCED("d=11: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega) +  W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        auto M_p = createMpMatrix(M, omega, inv_2);
        auto Mprime_p = createMpMatrix(Mprime, omega, inv_2);
        Sp1ZdGates gates(d);

        meter.measure([M, Mprime, omega, M_p, Mprime_p, gates] {
            bool result = bruteForceTestCliffordConjugacy(
                M, Mprime, omega, M_p, Mprime_p, std::make_optional(std::ref(gates)));
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=7: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 7; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega) + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 2, 3, inv_2, omega) * cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        auto M_p = createMpMatrix(M, omega, inv_2);
        auto Mprime_p = createMpMatrix(Mprime, omega, inv_2);
        Sp1ZdGates gates(d);

        meter.measure([M, Mprime, omega, M_p, Mprime_p, gates] {
            bool result = bruteForceTestCliffordConjugacy(
                M, Mprime, omega, M_p, Mprime_p, std::make_optional(std::ref(gates)));
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=7, Clifford-conjugate, 2 basis elements")(Catch::Benchmark::Chronometer meter) {
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

        auto M_p = createMpMatrix(M, omega, inv_2);
        auto Mprime_p = createMpMatrix(Mprime, omega, inv_2);
        Sp1ZdGates gates(d);

        meter.measure([M, Mprime, omega, M_p, Mprime_p, gates] {
            bool result = bruteForceTestCliffordConjugacy(
                M, Mprime, omega, M_p, Mprime_p, std::make_optional(std::ref(gates)));
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };
}

TEST_CASE("Algorithm for Clifford-conjugate test", "[benchmark][algorithm]") {
    BENCHMARK_ADVANCED("d = 5: Non-example, two linearly dependent Pauli basis elements bypassing histogram check")(Catch::Benchmark::Chronometer meter) {
        // This will actually pass the histogram check.

        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        Eigen::Vector2i v1(1,2), v2(2,4), v3(3,1);

        const Eigen::MatrixXcd Mprime = W(d, v1(0), v1(1), inv_2, omega) + W(d, v2(0), v2(1), inv_2, omega)
            + 2*W(d, v3(0), v3(1), inv_2, omega);

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

        const Eigen::MatrixXcd M = 2*W(d, w1(0), w1(1), inv_2, omega) + W(d, w2(0), w2(1), inv_2, omega)
            + W(d, w3(0), w3(1), inv_2, omega);

        // Ensure the trace check is bypassed.
        REQUIRE_THAT(Mprime.trace().imag(),  Catch::Matchers::WithinAbs(M.trace().imag(), 1e-5));
        REQUIRE_THAT(Mprime.trace().real(),  Catch::Matchers::WithinAbs(M.trace().real(), 1e-5));

        const auto M_p = createMpMatrix((M), omega, inv_2);
        const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);
        CHECK(!bruteForceTestCliffordConjugacy(M, Mprime, omega, M_p, Mprime_p));

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugate(M, Mprime);
            if (result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=13: Non-example, Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 13; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        Eigen::MatrixXcd M = C * Mprime * Cstar;
        M(0,3) += 0.2;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugate(M, Mprime);
            if (result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=13: Non-example, with histogram bypass")(Catch::Benchmark::Chronometer meter) {
        // This will actually pass the histogram check.

        const int d = 13;
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        Eigen::Vector2i v1(1,2), v2(2,2);

        const Eigen::MatrixXcd Mprime = 2*W(d, v1(0), v1(1), inv_2, omega) + 3*W(d, v2(0), v2(1), inv_2, omega);

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

        const Eigen::MatrixXcd M = 2*W(d, w1(0), w1(1), inv_2, omega) + 3*W(d, w2(0), w2(1), inv_2, omega);

        // Ensure the trace check is bypassed.
        REQUIRE_THAT(Mprime.trace().imag(), Catch::Matchers::WithinAbs(M.trace().imag(), 1e-5));
        REQUIRE_THAT(Mprime.trace().real(), Catch::Matchers::WithinAbs(M.trace().real(), 1e-5));

        // const auto M_p = createMpMatrix((M), omega, inv_2);
        // const auto Mprime_p = createMpMatrix((Mprime), omega, inv_2);
        // CHECK(!bruteForceTestCliffordConjugacy(M, Mprime, omega, M_p, Mprime_p));

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugate(M, Mprime);
            if (result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=5: Non-example, Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        Eigen::MatrixXcd M = C * Mprime * Cstar;
        M(0,3) += 0.2;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugate(M, Mprime);
            if (result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=11: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega) +  W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugate(M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are not Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=7: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 7; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega) + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 2, 3, inv_2, omega) *cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugate(M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=7, Clifford-conjugate, 2 basis elements")(Catch::Benchmark::Chronometer meter) {
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
        auto M_p = createMpMatrix(M, omega, inv_2);
        auto Mprime_p = createMpMatrix(Mprime, omega, inv_2);

        meter.measure([M, Mprime, omega, M_p, Mprime_p] {
            bool result = isCliffordConjugate(M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };
}

TEST_CASE("Vectorisation algorithm for Clifford-conjugate test", "[benchmark][vectorisation]") {


    BENCHMARK_ADVANCED("d=5: Non-example, Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 5; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        Eigen::MatrixXcd M = C * Mprime * Cstar;
        M(0,3) += 0.2;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
            if (result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=7: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 7;
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 2, 3, inv_2, omega) *cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };


    BENCHMARK_ADVANCED("d=11: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 11; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega) +  W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are not Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=29: Linearly dependent M_p")(Catch::Benchmark::Chronometer meter) {
        const int d = 29; // Example dimension
        const int inv_2 = fastPowerMod(2, d - 2, d);
        const std::complex<double> omega =
            std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
            + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega) +  W(d, 7, 14, inv_2, omega);

        const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd M = C * Mprime * Cstar;

        meter.measure([M, Mprime] {
            bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpectedly got that the matrices are not Clifford-conjugate");
            }
            return result;
        });
    };

    BENCHMARK_ADVANCED("d=7, Clifford-conjugate, 2 basis elements")(Catch::Benchmark::Chronometer meter) {
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

        meter.measure([M, Mprime, omega] {
            bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
            if (!result) {
                INFO("M = " << M);
                INFO("Mprime = " << Mprime);
                FAIL("unexpected failure");
            }
            return result;
        });
    };
}
