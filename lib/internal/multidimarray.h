#ifndef MULTIDIMENSIONALARRAY_H
#define MULTIDIMENSIONALARRAY_H

#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tupleiterator.h"

namespace cliffconjtest {
/**
 * A class for multidimensional arrays, where each dimension can have a variable size.
 * @tparam T The type to store
 */
template <typename T>
class MultiDimensionalArray {
public:
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
            ptr_++;
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
            ptr_--;
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
        pointer ptr_;
    };

private:
    std::vector<T> data{};
    std::vector<size_t> dims;

    [[nodiscard]] size_t get_index(const std::vector<size_t>& coords) const {
        if (coords.size() != dims.size()) {
            throw std::invalid_argument("Coordinate size does not match dimension size.");
        }
        size_t index = 0;
        size_t multiplier = 1;
        // Stores data like expansion of a variable-base expansion of number
        for (int i = dims.size() - 1; i >= 0; i--) {
            if (coords[i] >= dims[i]) {
                std::stringstream ss;
                ss << "coordinate (";
                for (size_t j = 0; j < coords.size(); j++) {
                    ss << coords[j];
                    if (j < coords.size() - 1) {
                        ss << ", ";
                    }
                }
                ss << ") is out of bounds ";
                throw std::out_of_range(ss.str());
            }
            index += coords[i] * multiplier;
            multiplier *= dims[i];
        }
        return index;
    }

public:
    explicit MultiDimensionalArray(const std::vector<size_t>& dimensions) : dims(dimensions) {
        size_t totalElements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
        data.resize(totalElements);
    }

    explicit MultiDimensionalArray(const size_t n, const size_t dimension) : dims(dimension) {
        for (size_t i = 0; i < dimension; ++i) {
            dims[i] = n;
        }
        data.resize(std::pow(n, dimension));
    }

    T& operator()(const std::vector<size_t>& coords) { return data[get_index(coords)]; }

    const T& operator()(const std::vector<size_t>& coords) const { return data[get_index(coords)]; }

    iterator begin() { return iterator(&data[0]); }
    iterator end() { return iterator(&data[0] + data.size()); }

    iterator begin() const { return iterator(const_cast<T*>(&data[0])); }
    iterator end() const { return iterator(const_cast<T*>(&data[0] + data.size())); }

    [[nodiscard]] const std::vector<size_t>& dimensions() const { return dims; }

    [[nodiscard]] size_t size() const { return data.size(); }

    [[nodiscard]] TupleIterator<true> indexIterator() const {
        return TupleIterator<true>(dims);
    }
};

}

#endif // MULTIDIMENSIONALARRAY_H
