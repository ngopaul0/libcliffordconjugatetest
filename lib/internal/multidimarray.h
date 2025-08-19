#ifndef MULTIDIMENSIONALARRAY_H
#define MULTIDIMENSIONALARRAY_H

#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "util.h"
#include "tupleiterator.h"

namespace cliffconjtest {

template <bool UseVariableDimension>
struct OptionalArrayField;

template <>
struct OptionalArrayField<true> {
    // Only defined when UseVariableDimension == true
    const std::vector<size_t> dimensions_;
};

template <>
struct OptionalArrayField<false> {
    // Only defined when UseVariableDimension == false

    /**
     * Specifies the single dimension per coordinate.
     */
    const size_t dimensionPerCoordinate_;
    const size_t numCoordinatePlaces_;
};

/**
 * A class for multidimensional arrays, where each dimension can have a variable size.
 * @tparam T The type to store
 * @tparam UseVariableDimension Whether each each dimension can have a variable size, e.g. indices
 * over Z_2 x Z_3 x Z_4 have different sizes, while indices over Z_3 x Z_3 x Z_3 = Z_3^3 have a
 * uniform size. Using UseVariableDimension == false will result in slightly less memory, as an
 * array for the dimensions doesn't need to be allocated.
 */
template <typename T, bool UseVariableDimension>
class MultiDimensionalArray : OptionalArrayField<UseVariableDimension> {
  public:
    /**
     * The iterator is simple because we can just rely on the backing array.
     */
    class iterator {
      public:
        // Required iterator type aliases for C++17 and later
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        explicit iterator(pointer ptr) : ptr_(ptr) {}

        // Dereference operators
        reference operator*() const { return *ptr_; }
        pointer operator->() { return ptr_; }

        // Pre-increment
        iterator& operator++() {
            ++ptr_;
            return *this;
        }

        // Post-increment
        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        // Pre-decrement
        iterator& operator--() {
            --ptr_;
            return *this;
        }

        // Post-decrement
        iterator operator--(int) {
            iterator temp = *this;
            --(*this);
            return temp;
        }

        // Random access
        iterator& operator+=(difference_type n) {
            ptr_ += n;
            return *this;
        }

        iterator& operator-=(difference_type n) {
            ptr_ -= n;
            return *this;
        }

        iterator operator+(difference_type n) const { return iterator(ptr_ + n); }

        iterator operator-(difference_type n) const { return iterator(ptr_ - n); }

        // Difference between iterators
        difference_type operator-(const iterator& other) const { return ptr_ - other.ptr_; }

        // Subscript operator
        reference operator[](difference_type n) const { return ptr_[n]; }

        // Comparison operators
        bool operator==(const iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const iterator& other) const { return !(*this == other); }
        bool operator<(const iterator& other) const { return ptr_ < other.ptr_; }
        bool operator>(const iterator& other) const { return other < *this; }
        bool operator<=(const iterator& other) const { return !(*this > other); }
        bool operator>=(const iterator& other) const { return !(*this < other); }

      private:
        // A pointer to the underlying array.
        pointer ptr_;
    };

  private:
    std::vector<T> data{};

    template<typename VectorT>
    constexpr long getCoordinate(const VectorT& coords, const size_t index) const {
        if constexpr (std::is_same_v<VectorT, std::vector<size_t>> || std::is_same_v<VectorT, const std::vector<size_t>>){
            return coords[index];
        } else  {
            // Eigen matrix
            return coords(index);
        }
    }

    template<typename VectorT, bool ModOutCoords = false>
    [[nodiscard]] size_t get_index(const VectorT& coords) const {
        int numDimensions;
        if constexpr (UseVariableDimension) {
            numDimensions = this->dimensions_.size();
        } else {
            numDimensions = this->numCoordinatePlaces_;
        }

        if (coords.size() != numDimensions) {
            std::stringstream ss;
            ss << "coordinate size " << coords.size() << " does not match number of dimensions "
               << numDimensions;
            throw std::invalid_argument(ss.str());
        }
        size_t index = 0;
        size_t multiplier = 1;
        // The index is like expansion of a (possibly variable-) base expansion of an integer
        for (int i = numDimensions - 1; i >= 0; i--) {
            size_t thisDimension;
            if constexpr (UseVariableDimension) {
                thisDimension = this->dimensions_[i];
            } else {
                thisDimension = this->dimensionPerCoordinate_;
            }

            size_t coordsAtI = getCoordinate(coords, i);
            if constexpr (ModOutCoords) {
                if (coordsAtI < thisDimension || coordsAtI >= thisDimension) {
                    coordsAtI = safeMod(coordsAtI, thisDimension);
                }
            }
            if (coordsAtI >= thisDimension) {
                std::stringstream ss;
                ss << "coordinate (";
                for (size_t j = 0; j < coords.size(); j++) {
                    size_t coordsAtj = getCoordinate(coords, j);
                    if constexpr (ModOutCoords) {
                        if (coordsAtj < thisDimension || coordsAtj >= thisDimension) {
                            coordsAtj = safeMod(coordsAtj, thisDimension);
                        }
                    }
                    ss << coordsAtj;
                    if (j < coords.size() - 1) {
                        ss << ", ";
                    }
                }
                ss << ") is out of bounds ";
                throw std::out_of_range(ss.str());
            }
            index += coordsAtI * multiplier;
            multiplier *= thisDimension;
        }
        return index;
    }

  public:
    template <bool B = UseVariableDimension, typename = std::enable_if_t<B>>
    explicit MultiDimensionalArray(const std::vector<size_t>& dimensions)
        : OptionalArrayField<UseVariableDimension>({dimensions}) {
        size_t totalElements = std::accumulate(this->dimensions_.begin(), this->dimensions_.end(),
                                               1, std::multiplies());
        data.resize(totalElements);
    }

    template <bool B = UseVariableDimension, typename = std::enable_if_t<!B>>
    explicit MultiDimensionalArray(const size_t numCoordinates,
                                   const size_t dimensionForAllCoordinates)
        : OptionalArrayField<UseVariableDimension>({dimensionForAllCoordinates, numCoordinates}) {
        data.resize(std::pow(dimensionForAllCoordinates, numCoordinates));
    }


    T& operator()(const std::vector<size_t>& coords) { return data[get_index(coords)]; }

    T& get(const Eigen::Vector<long, Eigen::Dynamic>& coords) { return data[get_index(coords)]; }

    const T& operator()(const std::vector<size_t>& coords) const { return data[get_index(coords)]; }

    const T& get(const Eigen::Vector<long, Eigen::Dynamic>& coords) const { return data[get_index(coords)]; }

    const T& getFromInt(const Eigen::Vector<int, Eigen::Dynamic>& coords) const { return data[get_index(coords)]; }

    iterator begin() { return iterator(&data[0]); }
    iterator end() { return iterator(&data[0] + data.size()); }

    iterator begin() const { return iterator(const_cast<T*>(&data[0])); }
    iterator end() const { return iterator(const_cast<T*>(&data[0] + data.size())); }

    template <bool B = UseVariableDimension, typename = std::enable_if_t<B>>
    [[nodiscard]] const std::vector<size_t>& dimensions() const {
        return this->dimensions_;
    }

    template <bool B = UseVariableDimension, typename = std::enable_if_t<!B>>
    [[nodiscard]] size_t dimensionPerCoordinate() const {
        return this->dimensionPerCoordinate_;
    }

    [[nodiscard]] size_t numCoordinatePlaces() const {
        if constexpr (UseVariableDimension) {
            return this->dimensions_.size();
        } else {
            return this->numCoordinatePlaces_;
        }
    }

    [[nodiscard]] size_t size() const { return data.size(); }

    template <bool B = UseVariableDimension, typename = std::enable_if_t<B>>
    [[nodiscard]] TupleIterator<__VariableModuliRef> indexIterator() const {
        // Iterate over a reference
        return TupleIterator<__VariableModuliRef>(this->dimensions_);
    }

    template <bool B = UseVariableDimension, typename = std::enable_if_t<!B>>
    [[nodiscard]] TupleIterator<SingleModulus> indexIterator() const {
        return TupleIterator<SingleModulus>(this->dimensionPerCoordinate_,
                                            this->numCoordinatePlaces_);
    }
};

using MpMatrixType = MultiDimensionalArray<std::complex<double>, false>;

} // namespace cliffconjtest

#endif // MULTIDIMENSIONALARRAY_H
