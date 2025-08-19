#ifndef TUPLEITERATOR_H
#define TUPLEITERATOR_H

#include <stdexcept>
#include <vector>

namespace cliffconjtest {

enum ModuliType {
    VariableModuli,
    SingleModulus,
    __VariableModuliRef,
};

template <ModuliType ModuliType>
struct OptionalField;

template <>
struct OptionalField<__VariableModuliRef> {
    // Only defined when UseVariableModuli == true
    const std::vector<size_t>& moduli_;
};

template <>
struct OptionalField<VariableModuli> {
    // Only defined when UseVariableModuli == true
    std::vector<size_t> moduli_;
};

template <>
struct OptionalField<SingleModulus> {
    // Only defined when UseVariableModuli == false
    size_t modulus_;
    size_t dimensions_;
};

template <ModuliType ModuliType>
constexpr bool isVariableModuli = ModuliType == __VariableModuliRef || ModuliType == VariableModuli;

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
 * For a generalization of Z_m^n, pass SingleModulus to the template parameter. For example, for
 * Z_3^4:
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
 * for (const auto& tuple : TupleIterator<SingleModulus>(modulus, 4)) {
 *     f(tuple[0], tuple[1], tuple[2], tuple[3]);
 * }
 * \endcode
 *
 * The iterator's core logic is that tuples are interpreted as a mixed-radix number. The mixed-radix
 * representation of the number gives the tuple representation. For example, with moduli {2, 3, 4},
 * 0 is (0,0,0), 1 is (0,0,1), 2 is (0,0,2), 3 is (0,0,3), 4 is (0,1,0), ..., 10 is (0,2,2).
 *
 * @tparam Type Whether to use a variable modulus for each coordinate (e.g. for
 * tuples in Z_2 x Z_3 x Z_4) or to use a single modulus for each coordinate (e.g. for Z_3^3).
 *      Using ModuliType::SingleModulus will ensure that the vector of moduli is not allocated when
 * it's known that the modulus is the same for all coordinates.
 *      Using ModuliType::VariableModuli
 * will allow for different moduli for different coordinates, but it will create a copy of the given
 * moduli vector.
 *      ModuliType::VariableModuliRef will avoid the copy, but the caller will be
 * responsible for ensuring the moduli vector is kept in scope with the iterator. Rust would've been
 * a better option with its lifetimes management.
 */
template <ModuliType Type>
class TupleIterator : OptionalField<Type> {
  public:
    // Required iterator type aliases for C++17 and later
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::vector<size_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    // Using SFINAE (Substitution Failure Is Not An Error) to conditionally remove constructors
    // at compile time.
    // isVariableModuli<T> being true would resolve std::enable_if_t into an int. The int value
    // itself doesn't matter at all.
    // isVariableModuli<T> being false would result in compiler removing this constructor (SFINAE)
    // The = 0 is just a way to set a default value for the dummy parameter
    template <ModuliType T = Type, std::enable_if_t<isVariableModuli<T>, int> = 0>
    explicit TupleIterator(const std::vector<size_t>& moduli, const size_t index = 0,
                           const size_t endIndex = 0)
        : OptionalField<Type>{moduli}, index_(index) {
        end_index_ = endIndex == 0 ? computeEndIndex() : endIndex;
    }

    template <ModuliType T = Type, std::enable_if_t<!isVariableModuli<T>, int> = 0>
    TupleIterator(const size_t modulus, const size_t dimensions, size_t index = 0,
                  const size_t endIndex = 0)
        : OptionalField<Type>{modulus, dimensions}, index_(index) {
        end_index_ = endIndex == 0 ? computeEndIndex() : endIndex;
    }

    TupleIterator(const TupleIterator& other)
        : OptionalField<Type>(static_cast<const OptionalField<Type>&>(other)), index_(other.index_),
          end_index_(other.end_index_) {
        // The OptionalField<Type> base class is initialized using a cast from the 'other' object,
        // which correctly handles all ModuliType specializations. Base classes have their own
        // implicit copy constructor.
    }

  public:
    /**
     * @brief Dereferences the iterator to get the current tuple.
     * @return A new object of the current tuple.
     */
    reference operator*() const {
        if (index_ >= end_index_) {
            throw std::out_of_range("Attempt to dereference an end iterator.");
        }
        if constexpr (isVariableModuli<Type>) {
            std::vector<size_t> tuple(this->moduli_.size());
            std::size_t currentIndex = index_;
            // like mixed-radix number system
            for (int i = this->moduli_.size() - 1; i >= 0; i--) {
                tuple[i] = currentIndex % this->moduli_[i];
                currentIndex /= this->moduli_[i];
            }
            return tuple;
        } else {
            std::vector<size_t> tuple(this->dimensions_);
            std::size_t x = index_;
            for (int i = this->dimensions_ - 1; i >= 0; i--) {
                tuple[i] = x % this->modulus_;
                x /= this->modulus_;
            }
            return tuple;
        }
    }

    /**
     * @brief Prefix increment operator to advance the iterator.
     * @return A reference to the advanced iterator.
     */
    TupleIterator& operator++() {
        ++index_;
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

    TupleIterator& operator--() {
        --index_;
        return *this;
    }

    /**
     * @brief Postfix increment operator.
     * @return A copy of the iterator before it was incremented.
     */
    TupleIterator operator--(int) {
        TupleIterator temp = *this;
        --*this;
        return temp;
    }

    TupleIterator& operator+=(const difference_type n) {
        index_ += n;
        return *this;
    }
    TupleIterator& operator-=(const difference_type n) {
        index_ -= n;
        return *this;
    }
    TupleIterator operator+(difference_type n) const {
        if constexpr (isVariableModuli<Type>) {
            return TupleIterator(this->moduli_, index_ + n);
        } else {
            return TupleIterator(this->modulus_, this->dimensions_, index_ + n);
        }
    }
    TupleIterator operator-(difference_type n) const {
        if constexpr (isVariableModuli<Type>) {
            return TupleIterator(this->moduli_, index_ - n);
        } else {
            return TupleIterator(this->modulus_, this->dimensions_, index_ - n);
        }
    }

    TupleIterator& operator=(TupleIterator&& other) noexcept {
        if (this != &other) {
            this->index_ = other.index_;
            this->end_index_ = other.end_index_;
            if constexpr (isVariableModuli<Type>) {
                this->moduli_ = other.moduli_;
            } else {
                this->modulus_ = other.modulus_;
                this->dimensions_ = other.dimensions_;
            }
        }
        return *this;
    }

    template <ModuliType T>
    difference_type operator-(const TupleIterator<T>& other) const {
        return index_ - other.index_;
    }
    reference operator[](difference_type n) const { return *(*this + n); }
    template <ModuliType T>
    bool operator!=(const TupleIterator<T>& other) const {
        return !(*this == other);
    }
    template <ModuliType T>
    bool operator<(const TupleIterator<T>& other) const {
        return index_ < other.index_;
    }
    template <ModuliType T>
    bool operator>(const TupleIterator<T>& other) const {
        return index_ > other.index_;
    }
    template <ModuliType T>
    bool operator<=(const TupleIterator<T>& other) const {
        return index_ <= other.index_;
    }
    template <ModuliType T>
    bool operator>=(const TupleIterator<T>& other) const {
        return index_ >= other.index_;
    }

    /**
     * @brief Equality comparison for iterators.
     * @param other The other iterator to compare against.
     * @return True if the iterators are equal, false otherwise.
     */
    template <ModuliType T>
    bool operator==(const TupleIterator<T>& other) const {
        return index_ == other.index_;
    }

    template <ModuliType T = Type, std::enable_if_t<T == __VariableModuliRef, int> = 0>
    [[nodiscard]] TupleIterator<__VariableModuliRef> begin() const {
        return TupleIterator<__VariableModuliRef>(this->moduli_);
    }

    template <ModuliType T = Type, std::enable_if_t<T == VariableModuli, int> = 0>
    [[nodiscard]] TupleIterator<__VariableModuliRef> begin() const {
        return TupleIterator<__VariableModuliRef>(this->moduli_);
    }

    template <ModuliType T = Type, std::enable_if_t<!isVariableModuli<T>, int> = 0>
    [[nodiscard]] TupleIterator<SingleModulus> begin() const {
        return TupleIterator(this->modulus_, this->dimensions_);
    }

    template <ModuliType T = Type, std::enable_if_t<isVariableModuli<T>, int> = 0>
    [[nodiscard]] TupleIterator<__VariableModuliRef> end() const {
        return TupleIterator<__VariableModuliRef>(this->moduli_, end_index_, end_index_);
    }

    template <ModuliType T = Type, std::enable_if_t<!isVariableModuli<T>, int> = 0>
    [[nodiscard]] TupleIterator<SingleModulus> end() const {
        return TupleIterator(this->modulus_, this->dimensions_, end_index_, end_index_);
    }
    size_t index_;

  private:
    size_t end_index_;

    size_t computeEndIndex() {
        size_t size;
        if constexpr (isVariableModuli<Type>) {
            size = this->moduli_.size();
        } else {
            size = this->dimensions_;
        }
        size_t result = 1;
        for (std::size_t i = 0; i < size; i++) {
            if constexpr (isVariableModuli<Type>) {
                result *= this->moduli_[i];
            } else {
                result *= this->modulus_;
            }
        }
        return result;
    }
};

} // namespace cliffconjtest

#endif // TUPLEITERATOR_H
