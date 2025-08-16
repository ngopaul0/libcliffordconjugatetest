#ifndef TUPLEITERATOR_H
#define TUPLEITERATOR_H

#include <stdexcept>
#include <vector>

/**
 * @class TupleIterator
 * @brief An iterator for traversing a multidimensional space defined by a vector of dimensions.
 *
 * This iterator generates tuples of non-negative integers. The dimensions vector specifies the
 * upper bound for each tuple coordinate. For example, with dimensions {2, 3, 4}, it will iterate
 * through all tuples in Z_2 x Z_3 x Z_4.
 *
 * This would allow for a generalization of nested loops, so a loop such as
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
 * for (const auto& tuple : TupleIterator({modulus, modulus, modulus, modulus})) {
 *     f(tuple[0], tuple[1], tuple[2], tuple[3]);
 * }
 * \endcode
 *
 * The iterator's core logic is a counter that increments like a digit system with a variable base
 * for each position. The last dimension (rightmost) increments first, and when it reaches its
 * limit, it "rolls over" and increments the next dimension to the left.
 */
class TupleIterator {
  public:
    // Required iterator type aliases for C++17 and later
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::vector<size_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    /**
     * @brief Constructs the beginning iterator.
     * @param dims A const reference to the vector of dimension bounds.
     * This vector must not be empty.
     */
    explicit TupleIterator(const std::vector<size_t>& dims)
        : dimensions_(dims), current_tuple_(dims.size(), 0) {
        if (dimensions_.empty()) {
            // An empty dimensions vector means there is nothing to iterate over.
            is_end_ = true;
        } else {
            // Check for any zero dimensions, which would result in an empty iteration.
            for (const size_t dim : dimensions_) {
                if (dim == 0) {
                    is_end_ = true;
                    break;
                }
            }
        }
    }

    /**
     * @brief Constructs the end iterator.
     * @param dims A const reference to the vector of dimension bounds.
     * The end iterator is conceptually an iterator that has gone past
     * the last valid tuple.
     */
    TupleIterator(const std::vector<size_t>& dims, bool is_end)
        : dimensions_(dims), current_tuple_(dims.size(), 0), is_end_(is_end) {}

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

        // Start from the last dimension and increment
        size_t i = dimensions_.size();
        while (i-- > 0) {
            current_tuple_[i]++;
            // Check if the current dimension has reached its limit
            if (current_tuple_[i] < dimensions_[i]) {
                // If not, we are done, and can break
                break;
            } else {
                // If it has, reset to 0 and continue to the next dimension to the left
                current_tuple_[i] = 0;
            }
        }
        // If the loop finished without a break, it means all dimensions "rolled over"
        // This signifies that the iteration is complete.
        if (i == static_cast<size_t>(-1)) {
            is_end_ = true;
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
        return dimensions_ == other.dimensions_ && current_tuple_ == other.current_tuple_ &&
               is_end_ == other.is_end_;
    }

    /**
     * @brief Inequality comparison for iterators.
     * @param other The other iterator to compare against.
     * @return True if the iterators are not equal, false otherwise.
     */
    bool operator!=(const TupleIterator& other) const { return !(*this == other); }

    [[nodiscard]] TupleIterator begin() const { return TupleIterator(dimensions_); }

    [[nodiscard]] TupleIterator end() const { return TupleIterator(dimensions_, true); }

  private:
    const std::vector<size_t>& dimensions_;
    std::vector<size_t> current_tuple_;
    bool is_end_ = false;
};

/**
 * @brief A container-like class that provides begin() and end() iterators for Z_m^n.
 */
class ZmTupleRange {
  public:
    explicit ZmTupleRange(const size_t n, const size_t modulus)
        : dimensions_(std::vector(n, modulus)) {}

    [[nodiscard]] TupleIterator begin() const { return TupleIterator(dimensions_); }

    [[nodiscard]] TupleIterator end() const { return {dimensions_, true}; }

  private:
    const std::vector<size_t> dimensions_;
};

#endif // TUPLEITERATOR_H
