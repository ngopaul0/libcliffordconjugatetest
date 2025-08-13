#ifndef ABSVALMAP_H
#define ABSVALMAP_H
#include <algorithm>
#include <complex>
#include <list>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>
#include "util.h"

/**
 * Given a complex d x d matrix M, this AbsValMap will store a mapping of absolute values of the
 * entries in M, r, to coordinates (p,q) such that |M(p,q)| = r.
 *
 * i.e. AbsValMap.get(r) = { (p,q) |  |M(p,q)| = r }.
 *
 * std::unordered_map and std::list are used to ensure average constant-time access and
 * modification. Random access of list elements are not needed, and the only modifications are
 * appending elements to lists.
 */
struct AbsValMap {
  private:
    size_t precision_;
    std::unordered_map<std::double_t, std::list<std::pair<size_t, size_t>>> map_;
    static const std::list<std::pair<size_t, size_t>> EMPTY_PAIR_LIST;

    double roundKey(double key) const {
        if (precision_ < 0) {
            return key;
        }
        double multiplier = std::pow(10.0, precision_);
        return std::round(key * multiplier) / multiplier;
    }

  public:
    explicit AbsValMap(size_t precision) : precision_(precision) {};
    explicit AbsValMap(size_t precision, int mapReservation) : precision_(precision) {
        map_.reserve(mapReservation);
    };

    /** Inserts the entry (p,q) and returns the key used */
    double insertEntry(size_t p, size_t q, const std::complex<double>& value) {
        auto key = roundKey(std::abs(value));
        map_[key].emplace_back(p, q);
        return key;
    }

    const std::list<std::pair<size_t, size_t>>& get(const double key) const {
        const auto it = map_.find(roundKey(key));
        return it != map_.end() ? it->second : EMPTY_PAIR_LIST;
    }

    size_t getCount(double key) const {
        // std::list has O(1) size() operator. Note std::forward_list doesnt.
        return get(key).size();
    }

    size_t size() const { return map_.size(); }

    std::optional<double> getNonZeroKey() const {
        // Despite the loop, this is constant-time. If the first element is zero, then the next
        // element (if it exists) has to be nonzero.
        for (const auto& key : map_ | std::views::keys) {
            // keys have been rounded already to some precision
            if (key != 0.0) {
                return std::make_optional(key);
            }
        }
        return std::nullopt;
    }

    std::vector<double> sortedKeys() {
        std::vector<double> keys;
        keys.reserve(map_.size());
        for (const auto& key : map_ | std::views::keys) {
            keys.push_back(key);
        }
        std::ranges::sort(keys, [this](double a, double b) { return getCount(a) < getCount(b); });
        return keys;
    }
};

#endif // ABSVALMAP_H
