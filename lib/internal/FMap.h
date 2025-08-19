#ifndef FMAP_H
#define FMAP_H
#include <algorithm>
#include <complex>
#include <list>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>

#include "hash_mix.h"
#include "util.h"

namespace cliffconjtest {

inline double roundMapKey(double key, double epsilon) {
    if (epsilon == 0.0) {
        return key;
    }
    return std::round(key / epsilon) * epsilon;
}

inline double compute_x_forMap(const size_t d, const std::complex<double>& z, double epsilon) {
    // First express z as z = r * exp(i * theta).
    // Then to express z = r * exp(2*pi*i/d * (n + x)), from z = r * exp(i * theta) we conclude
    // theta = 2*pi/d * (n + x), so n + x = d * theta / (2 * pi)
    double nPlusX;
    if (z.real() == 0.0) {
        if (z.imag() == 0.0) {
            return 0.0;
        }
        // const double theta = z.imag() > 0 ? pi / 2 : 3 * pi / 2;
        // nPlusX = static_cast<double>(d) * theta / (2 * pi);

        if (z.imag() > 0) {
            // theta = pi/2 divided by 2*pi is 1/4
            nPlusX = static_cast<double>(d) * 0.25;
        } else {
            // theta = 3pi/2 divided by 2*pi is 3/4
            nPlusX = static_cast<double>(d) * 0.75;
        }
    } else {
        const double theta = atan2(z.imag(), z.real());
        if (std::abs(theta) < epsilon) {
            nPlusX = 0.0;
        } else if (theta >= 0.0) {
            nPlusX = static_cast<double>(d) * theta / (2 * pi);
        } else {
            // atan2 has range (-pi, pi]. Normalize negative angles by adding 2 * pi
            // theta = atan(Im(z) / Re(z)) + 2*pi
            // Dividing by 2*pi would result in
            // theta / (2 * pi) = 1/(2*pi) atan(Im(z) / Re(z)) + 1
            nPlusX = static_cast<double>(d) * theta / (2 * pi) + static_cast<double>(d);
        }
    }

    double n;
    double x = std::modf(nPlusX, &n);
    double roundedX;
    if (isApproxEqual(x, 1.0, epsilon)) {
        // x is constrained to be in [0,1) for uniqueness
        // n += 1;
        roundedX = 0.0;
    } else {
        roundedX = roundMapKey(x, epsilon);
    }

    const auto rhs = abs(z) * std::exp(std::complex<double>(0, 2 * pi / static_cast<double>(d) * (n + x)));
    assert(isApproxEqual(z, rhs), epsilon);
    return roundedX;
}

/**
 * A key for use in the mapping of (abs(z), x(z)) where z in C is expressed as
 *
 *      z = r * exp(2*pi*i/d * (n + x)), r in R, n in Z, x in [0, 1)
 */
class FMapKey {
    double r_;
    double x_;

  public:
    static const FMapKey ZERO_KEY;

    FMapKey() : r_(0.0), x_(0.0) {}

    FMapKey(const double r, const double x) : r_(r), x_(x) {}

    FMapKey(const size_t d, const std::complex<double>& z, const double precisionFor_r,
            const double precisionFor_x)
        : r_(roundMapKey(std::abs(z), precisionFor_r)),
          x_(r_ != 0.0 ? compute_x_forMap(d, z, precisionFor_x) : 0.0) {}

    [[nodiscard]] double r() const { return r_; }
    [[nodiscard]] double x() const { return x_; }

    // Overload the equality operator for use in unordered_map
    bool operator==(const FMapKey& other) const { return r_ == other.r_ && x_ == other.x_; }
};

struct FMapKeyHash {
    std::size_t operator()(const FMapKey& p) const { return hash_pair(p.r(), p.x()); }
};

/**
 * Given a complex d x d matrix M, FMap will store a mapping of (r, x) in R^2 to the locations in Mp
 * where r in R and x in [0, 1). From
 *
 *      f_M(p,q) =: Mp(p,q) = r * exp(2*pi*i/d * (n + x)), r in R, n in Z, x in [0,1)
 *
 * the key (r, x) is used in the map. The idea is that according to Lemma 10, the every element in
 * M_p = [f_M]_{ij} is a permutation of the elements M'_p = [f_{M'}]_{ij} up to some
 * omega^[(p,q),(p',q')] power. But [(p,q),(p',q')] is an integer, so the omega power is using a
 * multiple of 2*pi/d. So two values that are permuted by a Clifford must have the same x value;
 * multiplying by omega^[(p,q),(p',q')] would only change the n.
 *
 * Let x_forMap be the function that computes this value of x for any complex number z, and
 * fix d a prime. Then
 *
 *      FMap.get(z) = { (p,q) |  |M(p,q)| = abs(z) and x_forMap(M(p,q)) = x_forMap(z) }.
 *
 * std::unordered_map and std::list are used to ensure average constant-time access and
 * modification. Random access of list elements are not needed, and the only modifications are
 * appending elements to lists.
 */
struct FMap {
  private:
    const size_t d_;
    const double precisionFor_r_;
    const double precisionFor_x_;
    std::unordered_map<FMapKey, std::vector<std::pair<size_t, size_t>>, FMapKeyHash> map_;
    static const std::vector<std::pair<size_t, size_t>> EMPTY_PAIR_LIST;

  public:
    explicit FMap(size_t d, double precisionFor_r, double precisionFor_x)
        : d_(d), precisionFor_r_(precisionFor_r), precisionFor_x_(precisionFor_x) {};
    explicit FMap(size_t d, size_t mapReservation, double precisionFor_r, double precisionFor_x)
        : FMap(d, precisionFor_r, precisionFor_x) {
        map_.reserve(mapReservation);
    };

    const auto& getMap() const { return map_; }
    auto& getMap() { return map_; }

    /** Inserts the entry (p,q) and returns the key used */
    FMapKey insertEntry(size_t p, size_t q, const std::complex<double>& value) {
        const auto key = FMapKey(d_, value, precisionFor_r_, precisionFor_x_);
        map_[key].emplace_back(p, q);
        return key;
    }

    const std::vector<std::pair<size_t, size_t>>& get(const FMapKey& key) const {
        const auto it = map_.find(key);
        return it != map_.end() ? it->second : EMPTY_PAIR_LIST;
    }

    size_t getCount(const std::complex<double>& fMValue) const {
        const auto key = FMapKey(d_, fMValue, precisionFor_r_, precisionFor_x_);
        return getCount(key);
    }

    size_t getCountOfZero() const {
        const auto& key = FMapKey::ZERO_KEY;
        return getCount(key);
    }

    size_t getCount(const FMapKey& key) const {
        // std::list has O(1) size() operator. Note std::forward_list doesnt.
        return get(key).size();
    }

    size_t size() const { return map_.size(); }

    /**
     * Returns a key with a nonzero r (abs value)
     */
    std::optional<FMapKey> getNonZeroKey() const {
        // Despite the loop, this is constant-time. If the first element is zero, then the next
        // element (if it exists) has to be nonzero.
        for (const auto& key : map_ | std::views::keys) {
            // keys have been rounded already to some precision
            if (key.r() != 0.0) {
                return std::make_optional(key);
            }
        }
        return std::nullopt;
    }

    std::vector<FMapKey> sortedKeys() {
        std::vector<FMapKey> keys;
        keys.reserve(map_.size());
        for (const auto& key : map_ | std::views::keys) {
            keys.push_back(key);
        }

        std::ranges::sort(
            keys, [this](const FMapKey& a, const FMapKey& b) { return getCount(a) < getCount(b); });
        return keys;
    }
};

} // namespace cliffconjtest

#endif // FMAP_H
