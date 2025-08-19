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

using namespace cliffconjtest;

void runUniquenessTest(size_t d) {
    INFO("d = " << d);
    Sp1ZdMatrixRange iterator(d);
    size_t i = 0;

    std::set<std::vector<int>> all_symplectic_matrices;
    for (const auto& S : iterator) {
        // Since Eigen matrices don't have a hash function defined, just put all elements
        // into std::vector for simplicity.
        std::vector<int> matrix_array(d * d);

        for (int row = 0; row < S.rows(); row++) {
            for (int col = 0; col < S.cols(); col++) {
                matrix_array.push_back(S(row, col));
            }
        }

        all_symplectic_matrices.insert(matrix_array);
    }

    REQUIRE(all_symplectic_matrices.size() == (d - 1) * d * (d + 1));
}

#define UNIQUENESS_TEST_D(dval)                                                                    \
    SECTION("Generates unique symplectic matrices: d = " #dval) { runUniquenessTest(dval); }

TEST_CASE("Sp1ZdIterator", "[sp1zditerator]") {
    SECTION("Generates valid symplectic matrices") {
        for (size_t d : {3, 5, 7, 11, 13, 17, 19, 23, 29}) {
            INFO("d = " << d);
            Sp1ZdMatrixRange iterator(d);
            size_t i = 0;
            for (const auto& S : iterator) {
                INFO("i = " << i);
                INFO("S =\n" << S);

                REQUIRE(safeMod(S.determinant(), d) == 1);

                i++;
            }
            REQUIRE(i == (d - 1) * d * (d + 1));
        }
    }

    SECTION("Generates unique symplectic matrices") {
        UNIQUENESS_TEST_D(3)
        UNIQUENESS_TEST_D(5)
        UNIQUENESS_TEST_D(7)
        UNIQUENESS_TEST_D(11)
        UNIQUENESS_TEST_D(13)
        UNIQUENESS_TEST_D(17)
        UNIQUENESS_TEST_D(19)
        // UNIQUENESS_TEST_D(23)
        // UNIQUENESS_TEST_D(29)
    }
}
