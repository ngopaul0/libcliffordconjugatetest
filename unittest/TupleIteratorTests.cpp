#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <random>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "internal/tupleiterator.h"

using namespace cliffconjtest;

TEST_CASE("TupleIterator", "[tupleiterator]") {
    SECTION("Iterates correctly over all elements of Z_3^4 with UseVariableModuli = false") {
        std::vector<std::vector<size_t>> allValues;
        for (const auto& tuple : TupleIterator<false>(3, 4)) {
            allValues.push_back(tuple);
        }

        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 3; b++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        std::vector tuple = {a, b, c, d};
                        REQUIRE(index < allValues.size());
                        REQUIRE(allValues[index] == tuple);
                        index++;
                    }
                }
            }
        }
    }

    SECTION("Iterates correctly over all elements of Z_3^4 with UseVariableModuli = true") {
        std::vector<std::vector<size_t>> allValues;
        for (const auto& tuple : TupleIterator<true>({3, 3, 3, 3})) {
            allValues.push_back(tuple);
        }

        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 3; b++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        std::vector tuple = {a, b, c, d};
                        REQUIRE(index < allValues.size());
                        REQUIRE(allValues[index] == tuple);
                        index++;
                    }
                }
            }
        }
    }

    SECTION(
        "Iterates correctly over all elements of Z_3 x Z_2 x Z_5 with UseVariableModuli = true") {
        auto iterator = TupleIterator<true>({3, 2, 5});
        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 2; b++) {
                for (size_t c = 0; c < 5; c++) {
                    std::vector tuple = {a, b, c};
                    const std::vector<size_t>& val = *iterator;
                    REQUIRE(val == tuple);
                    iterator++;
                    index++;
                }
            }
        }
    }
}
