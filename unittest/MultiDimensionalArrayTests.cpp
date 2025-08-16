#include "catch2/catch_test_macros.hpp"
#include "internal/util.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_vector.hpp"
#include "internal/MultiDimensionalArray.h"

struct SomeData {
    size_t a{};
    std::string b;
};

TEST_CASE("MultiDimensionalArray with custom struct", "[multidimensionalarray][tupleiterator]") {
    SECTION("Constructs correctly") {
        auto A = MultiDimensionalArray<SomeData>(4, 2);
        CHECK(A.size() == std::pow(4, 2));

        auto B = MultiDimensionalArray<SomeData>({2, 3, 4, 5});
        CHECK(B.size() == 2 * 3 * 4 * 5);
        CHECK(B.dimensions().size() == 4);
    }

    SECTION("Iterates correctly") {
        auto A = MultiDimensionalArray<SomeData>(4, 2);
        for (const auto& item : A) {
            CHECK(item.a == 0);
            CHECK(item.b.empty());
        }

        const std::vector<size_t> index = {1, 2};
        auto data = A(index);
        data.a = 5;
        CHECK(A(index).a != 5);

        auto& dataRef = A(index);
        dataRef.a = 5;
        CHECK(A(index).a == 5);

        for (const auto& tuple : A.indexIterator()) {
            if (tuple == index) {
                CHECK(A(tuple).a == 5);
            } else {
                CHECK(A(tuple).a == 0);
            }
        }
    }
}

TEST_CASE("MultiDimensionalArray Constructors and Basic Properties", "[MultiDimensionalArray]") {
    SECTION("Constructor with std::vector<size_t>") {
        MultiDimensionalArray<int> arr1({5});
        REQUIRE(arr1.size() == 5);
        REQUIRE(arr1.dimensions() == std::vector<size_t>{5});

        MultiDimensionalArray<double> arr2({2, 3});
        REQUIRE(arr2.size() == 6);
        REQUIRE(arr2.dimensions() == std::vector<size_t>{2, 3});

        MultiDimensionalArray<float> arr3({2, 3, 4});
        REQUIRE(arr3.size() == 24);
        REQUIRE(arr3.dimensions() == std::vector<size_t>{2, 3, 4});
    }

    SECTION("Constructor with n and dimension") {
        MultiDimensionalArray<int> arr1(5, 1);
        REQUIRE(arr1.size() == 5);
        REQUIRE(arr1.dimensions() == std::vector<size_t>{5});

        MultiDimensionalArray<double> arr2(2, 2);
        REQUIRE(arr2.size() == 4);
        REQUIRE(arr2.dimensions() == std::vector<size_t>{2, 2});

        MultiDimensionalArray<float> arr3(3, 3);
        REQUIRE(arr3.size() == 27);
        REQUIRE(arr3.dimensions() == std::vector<size_t>{3, 3, 3});
    }
}

TEST_CASE("MultiDimensionalArray Element Access and Exceptions", "[MultiDimensionalArray]") {
    SECTION("Valid element access") {
        MultiDimensionalArray<int> arr({2, 3, 4});

        // Set values
        arr({0, 0, 0}) = 1;
        arr({1, 2, 3}) = 24;

        // Check values
        REQUIRE(arr({0, 0, 0}) == 1);
        REQUIRE(arr({1, 2, 3}) == 24);
    }

    SECTION("Out of bounds access throws std::out_of_range") {
        MultiDimensionalArray<int> arr({2, 3, 4});

        // Coordinates are too large
        REQUIRE_THROWS_AS(arr({2, 0, 0}), std::out_of_range);
        REQUIRE_THROWS_AS(arr({0, 3, 0}), std::out_of_range);
        REQUIRE_THROWS_AS(arr({0, 0, 4}), std::out_of_range);

        // Negative coordinates (unsigned so large number)
        REQUIRE_THROWS_AS(arr({0, 0, static_cast<size_t>(-1)}), std::out_of_range);
    }

    SECTION("Coordinate size mismatch throws std::invalid_argument") {
        MultiDimensionalArray<int> arr({2, 3, 4});

        // Too few coordinates
        REQUIRE_THROWS_AS(arr({0, 0}), std::invalid_argument);

        // Too many coordinates
        REQUIRE_THROWS_AS(arr({0, 0, 0, 0}), std::invalid_argument);
    }
}

TEST_CASE("MultiDimensionalArray Iterator Functionality", "[MultiDimensionalArray]") {
    MultiDimensionalArray<int> arr({2, 3});
    int counter = 0;
    // Initialize array with values 0, 1, 2, 3, 4, 5
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            arr({i, j}) = counter++;
        }
    }

    SECTION("Range-based for loop") {
        std::vector<int> values;
        for (int val : arr) {
            values.push_back(val);
        }
        REQUIRE_THAT(values, Catch::Matchers::Equals(std::vector<int>{0, 1, 2, 3, 4, 5}));
    }

    SECTION("Iterator increment and decrement") {
        auto it = arr.begin();
        REQUIRE(*it == 0);
        REQUIRE(*(++it) == 1); // pre-increment
        REQUIRE(*(it++) == 1); // post-increment
        REQUIRE(*it == 2);
        REQUIRE(*(--it) == 1); // pre-decrement
        REQUIRE(*(it--) == 1); // post-decrement
        REQUIRE(*it == 0);
    }

    SECTION("Iterator random access") {
        auto it = arr.begin();
        REQUIRE(*(it + 3) == 3);
        REQUIRE(*(it + 5) == 5);
        REQUIRE(*(it + 3 - 2) == 1);

        it += 2;
        REQUIRE(*it == 2);
        it -= 1;
        REQUIRE(*it == 1);

        REQUIRE(it[0] == 1);
        REQUIRE(it[1] == 2);
    }

    SECTION("Iterator difference") {
        auto it1 = arr.begin();
        auto it2 = arr.begin() + 3;
        REQUIRE((it2 - it1) == 3);
        REQUIRE((it1 - it2) == -3);
    }

    SECTION("Iterator comparisons") {
        auto it1 = arr.begin();
        auto it2 = arr.begin() + 3;
        auto it3 = arr.begin();

        REQUIRE(it1 == it3);
        REQUIRE(it1 != it2);
        REQUIRE(it1 < it2);
        REQUIRE(it2 > it1);
        REQUIRE(it1 <= it3);
        REQUIRE(it1 <= it2);
        REQUIRE(it2 >= it1);
        REQUIRE(it3 >= it1);
    }
}

TEST_CASE("MultiDimensionalArray Const Correctness", "[MultiDimensionalArray]") {
    MultiDimensionalArray<int> arr({2, 2});
    arr({0, 0}) = 1;
    arr({0, 1}) = 2;
    arr({1, 0}) = 3;
    arr({1, 1}) = 4;

    const MultiDimensionalArray<int>& constRef = arr;

    SECTION("Const operator()") {
        REQUIRE(constRef({0, 0}) == 1);
        REQUIRE(constRef({1, 1}) == 4);
        REQUIRE(constRef({1, 0}) == 3);
        REQUIRE(constRef({1, 1}) == 4);
    }

    SECTION("Const iterators") {
        auto it_begin = constRef.begin();

        REQUIRE(*it_begin == 1);
        REQUIRE(*(++it_begin) == 2);

        std::vector<int> values;
        for (auto it = constRef.begin(); it != constRef.end(); it++) {
            values.push_back(*it);
        }
        REQUIRE_THAT(values, Catch::Matchers::Equals(std::vector{1, 2, 3, 4}));
    }
}
