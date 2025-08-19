#include <catch2/catch_test_macros.hpp>
#include <complex>

#include <Eigen/Dense>
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "cliffordconjugacytest.hpp"
#include "internal/FMap.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/util.h"

using namespace cliffconjtest;

TEST_CASE("FMap", "[FMap]") {
    constexpr double mapAbsValPrecision = 1e-5;
    constexpr double mapXPrecision = 1e-4;

    SECTION("Zero key") {
        const auto z = std::complex(0.0, 0.00000000000000057731597280508142);
        const auto key = FMapKey(5, z, mapAbsValPrecision, mapXPrecision);
        CHECK(key.r() == 0.0);
        CHECK(key.x() == 0.0);
    }

    SECTION("Pure imaginary key") {
        const auto z = std::complex(0.0, 0.1);
        const auto key = FMapKey(11, z, mapAbsValPrecision, mapXPrecision);
        CHECK(key.r() == 0.1);
        double dummy;
        CHECK(key.x() == std::modf(11/4.0, &dummy));

        const auto z2 = std::complex(0.0, -0.1);
        const auto key2 = FMapKey(11, z2, mapAbsValPrecision, mapXPrecision);
        CHECK(key2.r() == 0.1);
        CHECK(key2.x() == std::modf(11 * 3/4.0, &dummy));
    }

    SECTION("Key rounding edge case 1") {
        const auto z = std::complex(0.99999999999999966, 0.00000000000000057731597280508142);
        const auto key = FMapKey(5, z, mapAbsValPrecision, mapXPrecision);
        CHECK(key.r() == 1.0);
        CHECK(key.x() == 0.0);
    }

    SECTION("Key rounding edge case 2") {
        const auto z = std::complex<double>(4, 5);
        auto key = FMapKey(5, z, 1e-5, 1e-3);
        CHECK(key.r() == 6.40312);
        CHECK(key.x() == 0.713);

        key = FMapKey(5, z, 1e-6, 1e-5);
        CHECK(key.r() == 6.403124);
        CHECK(key.x() == 0.71306);
    }

    SECTION("Map insertion rounding") {
        FMap map(5, 1e-3, 1e-3);
        const auto z = std::complex<double>(4, 5);
        auto keyFromExact = map.insertEntry(0, 0, z);
        CHECK(map.getCount(keyFromExact) == 1);

        auto zApprox = std::complex(3.9999, 4.9999);
        auto keyFromApprox = map.insertEntry(0, 1, zApprox);
        CHECK_THAT(keyFromExact.x(), Catch::Matchers::WithinAbs(keyFromApprox.x(), 1e-3));
        CHECK_THAT(keyFromExact.r(), Catch::Matchers::WithinAbs(keyFromApprox.r(), 1e-3));
        CHECK(keyFromApprox == keyFromExact);

        std::stringstream ss;
        ss << "keys: [";
        const auto allKeys = map.sortedKeys();
        for (const FMapKey& sKey : allKeys) {
            ss << sKey << ", ";
        }
        ss << "]";
        INFO(ss.str());
        CHECK(allKeys.size() == 1);
        CHECK(map.getCount(keyFromExact) == 2);
    }

    SECTION("Map insertion rounding - lower precision") {
        FMap map(5, 1e-2, 1e-2);
        const auto z = std::complex<double>(4, 5);
        const auto keyFromExact = map.insertEntry(0, 0, z);
        CHECK(map.getCount(keyFromExact) == 1);

        std::vector<std::complex<double>> approxKeys = {
            {3.9999, 4.9999}, {3.9999, 5.0001}, {3.9999, 5.0},
            {3.9998, 4.9999}, {3.9998, 5.0001}, {3.9998, 5.0},
            {4.0000, 4.9999}, {4.0000, 5.0001}, {4.0000, 5.0},
            {4.0001, 4.9999}, {4.0001, 5.0001}, {4.0001, 5.0},
        };

        size_t keyCount = 1;
        for (const auto& zApprox : approxKeys) {
            const auto keyFromApprox = map.insertEntry(0, 1, zApprox);
            keyCount++;
            CHECK_THAT(keyFromExact.x(), Catch::Matchers::WithinAbs(keyFromApprox.x(), 1e-3));
            CHECK_THAT(keyFromExact.r(), Catch::Matchers::WithinAbs(keyFromApprox.r(), 1e-3));
            CHECK(keyFromApprox == keyFromExact);

            std::stringstream ss;
            ss << "keys: [";
            const auto allKeys = map.sortedKeys();
            for (const FMapKey& sKey : allKeys) {
                ss << sKey << ", ";
            }
            ss << "]";
            INFO(ss.str());

            CHECK(allKeys.size() == 1);
            CHECK(map.getCount(keyFromExact) == keyCount);
        }

    }

    SECTION("d=5, known result") {
        const int d = 5; // Example dimension
        const int inv_2 = modInverse(2, d);
        const std::complex<double> omega = std::exp(std::complex<double>(0, 2.0 * pi / d));
        const Eigen::MatrixXcd M = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega) +
                                        W(d, 3, 6, inv_2, omega) + W(d, 4, 8, inv_2, omega);

        const Eigen::MatrixXcd C = cliffordPermutationGate(d, 2);
        const Eigen::MatrixXcd Cstar = C.adjoint();
        const Eigen::MatrixXcd Mprime = C * M * Cstar;

        // precomputed symplectic transform
        Eigen::Matrix2i S;
        S << 2, 0, 0, 3;

        Eigen::Vector2i v1(1, 2);
        Eigen::Vector2i Sv1(2, 1);
        Eigen::Vector2i prod = S * v1;
        prod = prod.array().unaryExpr(
            [&](const int x) { return static_cast<int>(safeMod(x, d)); });
        REQUIRE(prod == Sv1);
        auto const f1 = f(M, v1[0], v1[1], inv_2, omega);
        REQUIRE_THAT(f1.imag(), Catch::Matchers::WithinAbs(0.0, 1e-5));
        REQUIRE_THAT(f1.real(), Catch::Matchers::WithinAbs(1.0, 1e-5));

        auto const fPrime1 = f(Mprime, Sv1[0], Sv1[1], inv_2, omega);
        REQUIRE_THAT(f1.imag(), Catch::Matchers::WithinAbs(fPrime1.imag(), 1e-5));
        REQUIRE_THAT(f1.real(), Catch::Matchers::WithinAbs(fPrime1.real(), 1e-5));

        FMap mapM(d, mapAbsValPrecision, mapXPrecision);
        std::vector<FMapKey> Mkeys;
        for (size_t p = 0; p < M.rows(); p++) {
            for (size_t q = 0; q < M.cols(); q++) {
                const auto key = mapM.insertEntry(p, q, f(M, p, q, inv_2, omega));
                if (!isApproxEqual(key.r(), 0.0)) {
                    Mkeys.push_back(key);
                }
            }
        }
        std::vector<FMapKey> Mprimekeys;
        FMap mapMprime(d, mapAbsValPrecision, mapXPrecision);
        for (size_t p = 0; p < M.rows(); p++) {
            for (size_t q = 0; q < M.cols(); q++) {
                const auto key = mapMprime.insertEntry(p, q, f(M, p, q, inv_2, omega));
                if (!isApproxEqual(key.r(), 0.0)) {
                    Mprimekeys.push_back(key);
                }
            }
        }
        CHECK(mapM.getCountOfZero() == d * d - 4);
        CHECK(mapM.getCount(f(M, 1, 2, inv_2, omega)) == 4);
        CHECK(mapM.size() == mapMprime.size());

        const auto keys = mapM.sortedKeys();
        for (const auto& key : keys) {
            INFO("Key (r,x) = (" << key.r() <<',' << key.x() << ")");
            CHECK(mapM.getCount(key) == mapMprime.getCount(key));
        }
    }
}