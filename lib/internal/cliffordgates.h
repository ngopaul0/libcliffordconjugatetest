#ifndef CLIFFORDGATES_H
#define CLIFFORDGATES_H

#include "symplecticiterator.h"

namespace cliffconjtest {

/**
 * Creates the Clifford permutation gate that sends |x> to |ax mod d>
 * @param d The prime
 * @param a The constant
 * @return The Clifford permutation gate
 */
Eigen::MatrixXcd cliffordPermutationGate(int d, int a);

Eigen::MatrixXcd hadamardGate(int d, const std::complex<double>& omega);

Eigen::MatrixXcd diagonalSymplecticCliffordGate(int d, int b, int inv_2,
                                                const std::complex<double>& omega);

/**
 * Iterates over all unique Clifford gates ignoring global phase, i.e. it iterates over the normal
 * form for Clifford gates
 *
 *      M D_1 P Z^p X^q,
 *
 * where
 *   current_tuple_[4] == p
 *   current_tuple_[3] == q
 *   current_tuple_[2], current_tuple_[1], current_tuple_[0] are as in Sp1ZdMatrixIterator
 */
class CliffordGateIterator {
  public:
    // Required iterator type aliases for C++17 and later
    using value_type = Eigen::MatrixXcd;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    /**
     * @brief Constructor for the beginning iterator (begin()).
     * @param d The prime modulus of Z_d.
     */
    explicit CliffordGateIterator(size_t d) : d_(d), current_tuple_({0, 0, 0}) {
        setCurrentMatrix();
    }

    /**
     * @brief Returns a beginning iterator for the current modulus.
     * @return An iterator pointing to the first symplectic transformation
     */
    [[nodiscard]] CliffordGateIterator begin() const { return CliffordGateIterator(d_); }

    /**
     * @brief Returns an end iterator for the current modulus.
     * @return An iterator pointing to one past the last Clifford gate
     */
    [[nodiscard]] CliffordGateIterator end() const {
        // The sentinel is for the first tuple element, x, which ranges from 0 <= d < d + 1.
        // So use d + 1 as that's an exclusive bound.
        return {d_, d_ + 1};
    }

    void reset() {
        current_tuple_ = {0, 0, 0};
    }

    // --- Operators ---

    /**
     * @brief Pre-increment operator.
     *
     * Moves the iterator to the next tuple, then computes the Clifford gate
     * @return A reference to the incremented iterator.
     */
    CliffordGateIterator& operator++() {
        if (d_ == 0) {
            return *this;
        }

        // There's a better way to do this but this works for now.
        current_tuple_[4]++;
        if (current_tuple_[4] == d_) {
            current_tuple_[4] = 0;
            current_tuple_[3]++;
            if (current_tuple_[3] == d_) {
                current_tuple_[3] = 0;

                // The below is from Sp1ZdMatrixIterator
                current_tuple_[2]++;
                if (current_tuple_[2] == d_ - 1) {
                    current_tuple_[2] = 0;
                    current_tuple_[1]++;
                    if (current_tuple_[1] == d_) {
                        current_tuple_[1] = 0;
                        current_tuple_[0]++;
                    }
                }
            }
        }

        setCurrentMatrix();

        return *this;
    }

    /**
     * @brief Post-increment operator.
     *
     * Moves the iterator to the next Clifford gate and returns the old state.
     * @return The old iterator state.
     */
    CliffordGateIterator operator++(int) {
        CliffordGateIterator temp = *this;
        ++(*this);
        return temp;
    }

    /**
     * @brief Dereference operator.
     *
     * @return A reference to the current Clifford gate.
     */
    reference operator*() { return current_matrix_; }

    /**
     * @brief Member access operator.
     *
     * @return A pointer to the current Clifford gate.
     */
    pointer operator->() { return &current_matrix_; }

    /**
     * @brief Equality comparison operator.
     *
     * Compares the internal state (modulus and tuple) of two iterators.
     * @param other The other iterator to compare against.
     * @return true if the iterators are in the same state, false otherwise.
     */
    bool operator==(const CliffordGateIterator& other) const {
        return d_ == other.d_ && current_tuple_ == other.current_tuple_;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * @param other The other iterator to compare against.
     * @return true if the iterators are not in the same state, false otherwise.
     */
    bool operator!=(const CliffordGateIterator& other) const { return !(*this == other); }

  private:
    size_t d_;
    value_type current_matrix_;
    std::array<size_t, 5> current_tuple_;
    size_t inv_2_ = fastPowerMod(2, d_ - 2, d_);
    std::complex<double> omega_ = std::exp(std::complex<double>(0, 2.0 * pi / d_));

    /**
     * @brief Private constructor to create a specific end iterator state.
     *
     * @param d The modulus of Z_d.
     * @param sentinel_value A value to indicate the end state on the first element of the tuple.
     */
    CliffordGateIterator(size_t d, size_t sentinel_value)
        : d_(d), current_tuple_({sentinel_value, 0, 0, 0, 0}) {}

    void setCurrentMatrix() {
        // The code is the exact same as for Sp1ZdMatrixIterator, except the matrices from the
        // normal form of Sp(1, Z_d) are mapped under Neuhauser representation (Theorem 1 from
        // paper) to form Clifford gates:
        //
        // - Elements of A correspond to symplectic Clifford permutation gates
        // - Elements of B correspond to diagonal symplectic Clifford gates
        // - [[0, 1], [-1, 0]] corresponds to the Hadamard gate
        // TODO: Try to refactor this so both Sp1ZdMatrixIterator and CliffordGateIterator can use
        //  it

        Eigen::MatrixXcd firstMatrix;
        // The first tuple element, x, ranges from 0 <= x < d + 1, because the first value x = 0
        // represents the identity element. The rest represents the d possible matrices of the
        // form B * [[0, 1],[-1,0]
        if (current_tuple_[0] == 0) {
            firstMatrix = Eigen::MatrixXcd::Identity(d_, d_);
        } else {
            Eigen::MatrixXcd factor = hadamardGate(d_, omega_);
            firstMatrix = diagonalSymplecticCliffordGate(d_, current_tuple_[0] - 1, inv_2_, omega_) * factor;
        }

        Eigen::MatrixXcd secondMatrix = diagonalSymplecticCliffordGate(d_, current_tuple_[1], inv_2_, omega_);

        // The last tuple element, z, ranges from 0 <= z < d - 1, and it is meant to iterate through
        // all the units of Z_d, i.e. non-zero elements. Adding 1 will result in the range being
        // 1 <= z + 1 < d, which captures all invertible elements.
        Eigen::MatrixXcd thirdMatrix = cliffordPermutationGate(d_, current_tuple_[2] + 1);

        Eigen::MatrixXcd ZX = makeZX(d_, current_tuple_[4], current_tuple_[3]);

        Eigen::MatrixXcd result = firstMatrix * secondMatrix * thirdMatrix * ZX;

        current_matrix_ = result;
    }
};

} // namespace cliffconjtest

#endif // CLIFFORDGATES_H
