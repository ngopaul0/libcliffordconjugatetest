#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <expected>
#include <random>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "internal/tupleiterator.h"

using namespace cliffconjtest;

std::vector<size_t> calculateExpectedTupleDimSize3(size_t indexTarget,
                                                   const std::vector<size_t>& dimensions) {
    size_t index = 0;
    for (size_t i = 0; i < dimensions[0]; i++) {
        for (size_t j = 0; j < dimensions[1]; j++) {
            for (size_t k = 0; k < dimensions[2]; k++) {
                if (index == indexTarget) {
                    return {i, j, k};
                }
                index++;
            }
        }
    }
    return {999, 999, 999};
}

TEST_CASE("TupleIterator", "[tupleiterator]") {
    SECTION("Iterates correctly over all elements of Z_3^4 with UseVariableModuli = false") {
        std::vector<std::vector<size_t>> allValues;
        for (const auto& tuple : TupleIterator<SingleModulus>(3, 4)) {
            allValues.push_back(tuple);
        }
        CHECK(allValues.size() == std::pow(3, 4));

        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 3; b++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        std::vector tuple = {a, b, c, d};
                        INFO("index == " << index);
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
        for (const auto& tuple : TupleIterator<VariableModuli>({3, 3, 3, 3})) {
            allValues.push_back(tuple);
        }

        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 3; b++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        std::vector tuple = {a, b, c, d};
                        INFO("index == " << index);
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
        auto iterator = TupleIterator<VariableModuli>({3, 2, 5});
        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 2; b++) {
                for (size_t c = 0; c < 5; c++) {
                    std::vector tuple = {a, b, c};
                    const std::vector<size_t>& val = *iterator;
                    INFO("index == " << index);
                    REQUIRE(val == tuple);
                    ++iterator;
                    index++;
                }
            }
        }
    }

    const std::vector<size_t> dimensions = {2, 3, 4};
    SECTION("Increment and decrement operators") {
        auto it = TupleIterator<VariableModuli>(dimensions);
        REQUIRE((*it) == std::vector<size_t>{0, 0, 0});

        ++it; // Pre-increment
        REQUIRE((*it) == std::vector<size_t>{0, 0, 1});

        auto temp_it = it++; // Post-increment
        REQUIRE((*temp_it) == std::vector<size_t>{0, 0, 1});
        REQUIRE((*it) == std::vector<size_t>{0, 0, 2});

        --it; // Pre-decrement
        REQUIRE((*it) == std::vector<size_t>{0, 0, 1});

        temp_it = it--; // Post-decrement
        REQUIRE((*temp_it) == std::vector<size_t>{0, 0, 1});
        REQUIRE((*it) == std::vector<size_t>{0, 0, 0});
    }

    SECTION("Random access operations") {
        auto it_begin = TupleIterator<VariableModuli>(dimensions);
        auto it_end = it_begin.end();

        CHECK_THROWS_AS(*it_end, std::out_of_range);

        // Operator +
        auto it_plus_5 = it_begin + 5;
        REQUIRE((*it_plus_5) == std::vector<size_t>{0, 1, 1});

        // Operator +=
        auto it_compound = it_begin;
        it_compound += 8;
        REQUIRE((*it_compound) == std::vector<size_t>{0, 2, 0});

        // Operator -
        auto it_minus_2 = it_compound - 2;
        REQUIRE((*it_minus_2) == std::vector<size_t>{0, 1, 2});

        // No change from the += 8 call
        REQUIRE((*it_compound) == std::vector<size_t>{0, 2, 0});

        // Operator -=
        it_compound -= 4;
        // (0,2,0) -> (0,1,3) -> (0,1,2) -> (0,1,1) -> (0,1,0)
        REQUIRE((*it_compound) == std::vector<size_t>{0, 1, 0});

        // Operator []
        REQUIRE(it_begin[10] == calculateExpectedTupleDimSize3(10, dimensions));
        REQUIRE(it_begin[23] == calculateExpectedTupleDimSize3(23, dimensions));

        // Distance operator
        REQUIRE((it_end - it_begin) == 24); // 2 * 3 * 4 = 24
        REQUIRE((it_begin - it_end) == -24);
        REQUIRE((it_plus_5 - it_begin) == 5);
    }

    SECTION("Comparison operators") {
        auto it1 = TupleIterator<VariableModuli>(dimensions);
        auto it2 = it1 + 10;
        auto it3 = it1 + 10;
        auto it_end = it1.end();

        REQUIRE(it1 < it2);
        REQUIRE(it2 > it1);
        REQUIRE(it2 == it3);
        REQUIRE(it2 != it1);
        REQUIRE(it1 <= it2);
        REQUIRE(it2 >= it1);
        REQUIRE(it2 <= it3);
        REQUIRE(it2 >= it3);
        REQUIRE(it_end > it1);
        REQUIRE(it_end >= it1);
    }
}
