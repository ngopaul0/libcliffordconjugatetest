#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <random>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "internal/tupleiterator.h"

TEST_CASE("TupleIterator", "[tupleiterator]") {
    SECTION("Iterates correctly over all elements of Z_3^4") {
        std::vector<std::vector<size_t>> allValues;
        for (const auto& tuple : ZmTupleRange(4, 3)) {
            allValues.push_back(tuple);
        }

        size_t index = 0;
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 3; b++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t d = 0; d < 3; d++) {
                        std::vector tuple = {a, b, c, d};
                        CHECK(allValues[index] == tuple);
                        index++;
                    }
                }
            }
        }
    }
}
