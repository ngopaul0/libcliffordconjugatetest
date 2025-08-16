#ifndef TUPLEITERATOR_H
#define TUPLEITERATOR_H

#include <stdexcept>
#include <vector>

namespace cliffconjtest {
template <bool UseVariableModuli>
struct OptionalField {
    // Only defined when UseVariableModuli == true
    const std::vector<size_t> moduli_;
};

template <>
struct OptionalField<false> {
    // Only defined when UseVariableModuli == false
    const size_t modulus_;
    const size_t dimensions_;
};

/**
 * @class TupleIterator
 * @brief Iterator for tuples in Z_{m_1} x Z_{m_2} x ... Z_{m_d}. If m := m_1 = m_2 = ... = m_d,
 * then set UseVariableModuli to false
 *
 * This iterator generates tuples of non-negative integers. The dimensions vector specifies the
 * upper bound for each tuple coordinate, i.e. the moduli for each coordinate. For example, with
 * moduli {2, 3, 4}, it will iterate through all tuples in Z_2 x Z_3 x Z_4.
 *
 * This would allow for a generalization of nested loops, so a loop such as
 * \code
 * for (size_t a = 0; a < 2; a++) {
 *     for (size_t b = 0; b < 3; b++) {
 *         for (size_t c = 0; c < 4; c++) {
 *             for (size_t d = 0; d < 5; d++) {
 *                 f(a,b,c,d);
 *             }
 *         }
 *     }
 * }
 * \endcode
 * can just be replaced with
 * \code
 * for (const auto& tuple : TupleIterator<true>({2, 3, 4, 5})) {
 *     f(tuple[0], tuple[1], tuple[2], tuple[3]);
 * }
 * \endcode
 * For a generalization of Z_m^n, pass false to the template parameter. For example, for Z_3^4:
 * \code
 * size_t modulus = 3;
 * for (size_t a = 0; a < modulus; a++) {
 *     for (size_t b = 0; b < modulus; b++) {
 *         for (size_t c = 0; c < modulus; c++) {
 *             for (size_t d = 0; d < modulus; d++) {
 *                 f(a,b,c,d);
 *             }
 *         }
 *     }
 * }
 * \endcode
 * can just be replaced with
 * \code
 * size_t modulus = 3;
 * for (const auto& tuple : TupleIterator<false>(modulus, 4)) {
 *     f(tuple[0], tuple[1], tuple[2], tuple[3]);
 * }
 * \endcode
 *
 * The iterator's core logic is a counter that increments like a digit system with a variable base
 * for each position. The last dimension (rightmost) increments first, and when it reaches its
 * limit, it "rolls over" and increments the next dimension to the left.
 *
 * @tparam UseVariableModuli Whether to use a variable modulus for each coordinate (e.g. for
 * tuples in Z_2 x Z_3 x Z_4) or to use a single modulus for each coordinate (e.g. for Z_3^3).
 * Using false will ensure that the vector of moduli is not allocated when it's known that the
 * modulus is the same for all coordinates.
 */
template <bool UseVariableModuli>
class TupleIterator : OptionalField<UseVariableModuli> {
public:
    // Required iterator type aliases for C++17 and later
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::vector<size_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    /**
     * @brief Constructs tuple iterator for variable moduli.
     * Used if UseVariableModuli == true
     * @param moduli Vector of moduli. This vector must not be empty. Warning: This will copy the
     * moduli array.
     */
    template <bool B = UseVariableModuli, typename = std::enable_if_t<B>>
    explicit TupleIterator(const std::vector<size_t>& moduli)
        : OptionalField<UseVariableModuli>{moduli}, current_tuple_(moduli.size(), 0) {

        if (this->moduli_.empty()) {
            // An empty dimensions vector means there is nothing to iterate over.
            is_end_ = true;
        } else {
            // Check for any zero dimensions, which would result in an empty iteration.
            for (const size_t dim : this->moduli_) {
                if (dim == 0) {
                    is_end_ = true;
                    break;
                }
            }
        }
    }

    /**
     * @brief Constructs tuple iterator for Z_modulus^dimensions.
     * Used if UseVariableModuli == false
     */
    template <bool B = UseVariableModuli, typename = std::enable_if_t<!B>>
    TupleIterator(const size_t modulus, const size_t dimensions)
        : OptionalField<UseVariableModuli>{modulus, dimensions}, current_tuple_(dimensions, 0) {}

private:
    /**
     * @brief Constructs tuple iterator for variable moduli.
     * Used if UseVariableModuli == true
     * @param moduli Vector of moduli. This vector must not be empty.
     */
    template <bool B = UseVariableModuli, typename = std::enable_if_t<B>>
    TupleIterator(const std::vector<size_t>& moduli, bool is_end)
        : OptionalField<UseVariableModuli>{moduli}, current_tuple_(moduli.size(), 0),
          is_end_(is_end) {}

    /**
     * @brief Constructs tuple iterator for Z_modulus^dimensions.
     * Used if UseVariableModuli == false
     */
    template <bool B = UseVariableModuli, typename = std::enable_if_t<!B>>
    TupleIterator(const size_t modulus, const size_t dimensions, bool is_end)
        : OptionalField<UseVariableModuli>{modulus, dimensions}, current_tuple_(dimensions, 0),
          is_end_(is_end) {}

public:
    /**
     * @brief Dereferences the iterator to get the current tuple.
     * @return A const reference to the current tuple.
     */
    const value_type& operator*() const {
        if (is_end_) {
            throw std::out_of_range("Attempt to dereference an end iterator.");
        }
        return current_tuple_;
    }

    /**
     * @brief Prefix increment operator to advance the iterator.
     *
     * This method implements the core iteration logic. It starts by incrementing
     * the last element of the tuple. If an element reaches its dimension limit,
     * it is reset to 0, and the next element to the left is incremented,
     * simulating a "carry" operation.
     *
     * @return A reference to the advanced iterator.
     */
    TupleIterator& operator++() {
        if (is_end_) {
            return *this; // Already at the end, no-op
        }

        // Start from the last dimension and decrement
        size_t i;
        if constexpr (UseVariableModuli) {
            i = this->moduli_.size();
        } else {
            i = this->dimensions_;
        }

        // e.g. for Z_3^3, suppose we're at (0, 0, 2)
        while (true) {
            i--;

            current_tuple_[i]++;
            // Check if the current dimension has reached its limit.
            // e.g. the example should be incremented from (0, 0, 2) to (0, 0, 3). But we're in Z_3,
            // so the limit of 3 is reached.
            size_t modulus;
            if constexpr (UseVariableModuli) {
                modulus = this->moduli_[i];
            } else {
                modulus = this->modulus_;
            }

            if (current_tuple_[i] < modulus) {
                // If limit not reached, we are done, and can break
                break;
            } else {
                // If it has, reset to 0 and continue to the next dimension to the left
                // e.g. the example should be incremented in the next iteration (0, 1, 0) and then
                // break
                current_tuple_[i] = 0;

                // Can't roll over to anything else; iteration is complete
                if (i == 0) {
                    is_end_ = true;
                    break;
                }
            }
        }

        return *this;
    }

    /**
     * @brief Postfix increment operator.
     * @return A copy of the iterator before it was incremented.
     */
    TupleIterator operator++(int) {
        TupleIterator temp = *this;
        ++(*this);
        return temp;
    }

    /**
     * @brief Equality comparison for iterators.
     * @param other The other iterator to compare against.
     * @return True if the iterators are equal, false otherwise.
     */
    bool operator==(const TupleIterator& other) const {
        if (is_end_ && other.is_end_) {
            return true;
        }

        if constexpr (UseVariableModuli) {
            return this->moduli_ == other.moduli_ && current_tuple_ == other.current_tuple_ &&
                   is_end_ == other.is_end_;
        } else {
            return this->modulus_ == other.modulus_ && this->dimensions_ == other.dimensions_ &&
                   current_tuple_ == other.current_tuple_ && is_end_ == other.is_end_;
        }
    }

    /**
     * @brief Inequality comparison for iterators.
     * @param other The other iterator to compare against.
     * @return True if the iterators are not equal, false otherwise.
     */
    bool operator!=(const TupleIterator& other) const { return !(*this == other); }

    [[nodiscard]] TupleIterator begin() const {
        if constexpr (UseVariableModuli) {
            return TupleIterator(this->moduli_);
        } else {
            return TupleIterator(this->modulus_, this->dimensions_);
        }
    }

    [[nodiscard]] TupleIterator end() const {
        if constexpr (UseVariableModuli) {
            return TupleIterator(this->moduli_, true);
        } else {
            return TupleIterator(this->modulus_, this->dimensions_, true);
        }
    }

private:
    std::vector<size_t> current_tuple_;
    bool is_end_ = false;
};

}

#endif // TUPLEITERATOR_H
