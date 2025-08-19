#ifndef SYMPLECTICITERATOR_H
#define SYMPLECTICITERATOR_H

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <iterator>
#include <optional>

#include "util.h"

namespace cliffconjtest {

using Sp1ZdMatrix = Eigen::Matrix<long, 2, 2>;

inline Sp1ZdMatrix B(size_t e) {
    Sp1ZdMatrix  b = Sp1ZdMatrix::Identity();
    b(0, 1) = e;
    return b;
}

inline Sp1ZdMatrix A(size_t f, size_t modulus) {
    if (f == 0) {
        throw std::invalid_argument("f must be nonzero");
    }
    Sp1ZdMatrix a = Sp1ZdMatrix::Zero();
    a(0, 0) = f;
    a(1, 1) = fastPowerMod(f, modulus - 2, modulus);
    return a;
}

/**
 * @class Sp1ZdMatrixIterator
 * @brief A forward iterator for iterating over all symplectic matrices in Sp(1, Z_d), d an odd
 * prime. To use this class, create an instance of Sp1ZdMatrixRange and use it in a for-loop or
 * manually iterate with begin() and end() methods.
 *
 * The iterator will internally over all 3-tuples of the form (x, y, z) where
 * 0 <= x < d + 1; 0 <= y < d, 0 <= z < d - 1.
 *
 * These tuple values correspond to the normal form of Sp(1, Z_d),
 *
 *      ( {I} union B * [[0, 1],[-1,0]) B A
 *
 * where A is the set of matrices of the form [[f, 0], [0, f^(-1)]] and B is the set of matrices
 * of the form [[1,e],[0,1]].
 *
 * There are d + 1 matrices in ( {I} union B * [[0, 1],[-1,0]), d matrices in B, and d - 1 matrices
 * in A.
 *
 * Matrices returned by this iterator should be copied if wanting to use them outside of the
 * iterator scope.
 */
class Sp1ZdMatrixIterator {
  public:
    // Required iterator type aliases for C++17 and later
    using value_type = Sp1ZdMatrix;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    /**
     * @brief Constructor for the beginning iterator (begin()).
     * @param d The prime modulus of Z_d.
     */
    explicit Sp1ZdMatrixIterator(size_t d) : modulus_(d), current_tuple_({0, 0, 0}) {}

    /**
     * @brief Private constructor to create a specific end iterator state.
     *
     * @param d The modulus of Z_d.
     * @param isEnd Where to create an end iterator.
     */
    Sp1ZdMatrixIterator(size_t d, bool isEnd)
        : modulus_(d), current_tuple_({isEnd ? modulus_ + 1 : 0, 0, 0}) {
        // The sentinel is for the first tuple element, x, which ranges from 0 <= d < d + 1.
        // So use modulus_ + 1 as that's an exclusive bound.
    }

    [[nodiscard]] bool isEnd() const {
        return current_tuple_[0] == modulus_ + 1;
    }

    // --- Operators ---

    /**
     * @brief Pre-increment operator.
     *
     * Moves the iterator to the next 3-tuple, then computes the associated symplectic matrix.
     * @return A reference to the incremented iterator.
     */
    Sp1ZdMatrixIterator& operator++() {
        if (modulus_ == 0 || isEnd()) {
            return *this;
        }

        current_tuple_[2]++;
        if (current_tuple_[2] == modulus_ - 1) {
            current_tuple_[2] = 0;
            current_tuple_[1]++;
            if (current_tuple_[1] == modulus_) {
                current_tuple_[1] = 0;
                current_tuple_[0]++;
            }
        }

        setCurrentMatrix();

        return *this;
    }

    /**
     * @brief Post-increment operator.
     *
     * Moves the iterator to the next symplectic matrix and returns the old state.
     * @return The old iterator state.
     */
    Sp1ZdMatrixIterator operator++(int) {
        Sp1ZdMatrixIterator temp = *this;
        ++(*this);
        return temp;
    }

    /**
     * @brief Dereference operator.
     *
     * @return A reference to the current symplectic matrix
     */
    reference operator*() {
        if (isEnd()) {
            throw std::out_of_range("Sp1ZdMatrixIterator end reached");
        }
        if (!current_matrix_.has_value()) {
            setCurrentMatrix();
        }
        return current_matrix_.value();
    }

    /**
     * @brief Equality comparison operator.
     *
     * Compares the internal state (modulus and tuple) of two iterators.
     * @param other The other iterator to compare against.
     * @return true if the iterators are in the same state, false otherwise.
     */
    bool operator==(const Sp1ZdMatrixIterator& other) const {
        return modulus_ == other.modulus_ && current_tuple_ == other.current_tuple_;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param other The other iterator to compare against.
     * @return true if the iterators are not in the same state, false otherwise.
     */
    bool operator!=(const Sp1ZdMatrixIterator& other) const { return !(*this == other); }

  private:
    size_t modulus_;
    std::optional<value_type> current_matrix_;
    std::array<size_t, 3> current_tuple_;

    void setCurrentMatrix() {
        if (isEnd()) {
            current_matrix_ = std::nullopt;
            return;
        }

        Sp1ZdMatrix firstMatrix;
        // The first tuple element, x, ranges from 0 <= x < d + 1, because the first value x = 0
        // represents the identity element. The rest represents the d possible matrices of the
        // form B * [[0, 1],[-1,0]
        if (current_tuple_[0] == 0) {
            firstMatrix = Sp1ZdMatrix::Identity();
        } else {
            Sp1ZdMatrix factor = Sp1ZdMatrix::Zero();
            factor(0, 1) = 1;
            factor(1, 0) = safeMod(-1, modulus_);
            firstMatrix = B(current_tuple_[0] - 1) * factor;
        }

        Sp1ZdMatrix  secondMatrix = B(current_tuple_[1]);

        // The last tuple element, z, ranges from 0 <= z < d - 1, and it is meant to iterate through
        // all the units of Z_d, i.e. non-zero elements. Adding 1 will result in the range being
        // 1 <= z + 1 < d, which captures all invertible elements.
        Sp1ZdMatrix  thirdMatrix = A(current_tuple_[2] + 1, modulus_);

        Sp1ZdMatrix  result = firstMatrix * secondMatrix * thirdMatrix;

        result = result.array().unaryExpr(
            [&](const long x) { return safeMod(x, modulus_); });
        current_matrix_ = std::make_optional(result);
    }
};

/**
 * Helper class for Sp1ZdMatrixIterator so that extra state (current_matrix_, current_tuple_) is not
 * stored until actually iterating.
 */
class Sp1ZdMatrixRange {
    size_t modulus_;

  public:
    explicit Sp1ZdMatrixRange(size_t modulus) : modulus_(modulus) {}
    /**
     * @brief Returns a beginning iterator for the current modulus.
     * @return An iterator pointing to the first symplectic transformation
     */
    [[nodiscard]] Sp1ZdMatrixIterator begin() const { return Sp1ZdMatrixIterator(modulus_); }

    /**
     * @brief Returns an end iterator for the current modulus.
     * @return An iterator pointing to one past the last symplectic transformation
     */
    [[nodiscard]] Sp1ZdMatrixIterator end() const {
        return {modulus_, true};
    }
};

} // namespace cliffconjtest

#endif // SYMPLECTICITERATOR_H
