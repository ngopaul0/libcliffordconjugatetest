#include "catch2/catch_test_macros.hpp"
#include "internal/symplecticiterator.h"
#include "internal/util.h"

#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "internal/cliffordgates.h"

using namespace cliffconjtest;

void runUniquenessCliffordTest(size_t d) {
    INFO("d = " << d);
    CliffordGateIterator iterator(d);
    size_t i = 0;

    std::set<std::vector<std::pair<double, double>>> all_symplectic_matrices;
    for (const Eigen::MatrixXcd& C : iterator) {
        // Since Eigen matrices don't have a hash function defined, just put all elements
        // into std::vector for simplicity.
        std::vector<std::pair<double, double>> matrix_array(d * d);

        for (int row = 0; row < C.rows(); row++) {
            for (int col = 0; col < C.cols(); col++) {
                matrix_array.emplace_back(C(row, col).real(), C(row, col).imag());
            }
        }

        all_symplectic_matrices.insert(matrix_array);
    }

    REQUIRE(all_symplectic_matrices.size() == (d - 1) * d * (d + 1) * d * d);
}

#define UNIQUENESS_TEST_D(dval)                                                                    \
    SECTION("Generates unique Clifford gates: d = " #dval) { runUniquenessCliffordTest(dval); }

TEST_CASE("makeCliffordGateIterator", "[cliffordgateiterator]") {
    SECTION("Generates valid Clifford gates") {
        for (size_t d : {3, 5}) {
            INFO("d = " << d);
            auto iterator = CliffordGateIterator(d);
            size_t i = 0;
            for (const Eigen::MatrixXcd& C : iterator) {
                INFO("i = " << i);
                INFO("C =\n" << C);

                i++;
            }
            REQUIRE(i == (d - 1) * d * (d + 1) * d * d);
        }
    }

    SECTION("Generates unique Clifford gates") {
        UNIQUENESS_TEST_D(3)
        UNIQUENESS_TEST_D(5)
        UNIQUENESS_TEST_D(7)
        // UNIQUENESS_TEST_D(11)
        // UNIQUENESS_TEST_D(13)
        // UNIQUENESS_TEST_D(17)
        // UNIQUENESS_TEST_D(19)
        // UNIQUENESS_TEST_D(23)
        // UNIQUENESS_TEST_D(29)
    }
}
