#ifndef FMAP_H
#define FMAP_H
#include <algorithm>
#include <complex>
#include <list>
#include <numeric>
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

inline double compute_x_and_n(const size_t d, const std::complex<double>& z, size_t& nRef,
                              double epsilon) {
    // First express z as z = r * exp(i * theta).
    // Then to express z = r * exp(2*pi*i/d * (n + x)), from z = r * exp(i * theta) we conclude
    // theta = 2*pi/d * (n + x), so n + x = d * theta / (2 * pi)
    double nPlusX;
    // Check to prevent division by zero
    if (isApproxEqual(z.real(), 0.0, epsilon)) {
        if (isApproxEqual(z.imag(), 0.0, epsilon)) {
            return 0.0;
        }

        // z is pure imaginary; it's either straight up or straight down.
        if (z.imag() >= 0.0) {
            // theta = pi/2 divided by 2*pi is 1/4
            nPlusX = static_cast<double>(d) * 0.25;
        } else {
            // theta = 3pi/2 divided by 2*pi is 3/4
            nPlusX = static_cast<double>(d) * 0.75;
        }
    } else {
        const double theta = std::arg(z);
        if (isApproxEqual(theta, 0.0, epsilon)) {
            nPlusX = 0.0;
        } else if (theta >= 0.0) {
            nPlusX = static_cast<double>(d) * theta / (2 * pi);
        } else {
            // atan2 has range (-pi, pi]. Normalize negative angles by adding 2 * pi
            // theta = atan(Im(z) / Re(z)) + 2*pi
            // Dividing by 2*pi would result in
            // theta / (2 * pi) = 1/(2*pi) atan(Im(z) / Re(z)) + 1
            //
            // Pre-compute certain parts to reduce error propagation.
            nPlusX = static_cast<double>(d) * theta / (2 * pi) + static_cast<double>(d);
        }
    }

    double n;
    double x = std::modf(nPlusX, &n);
    if (isApproxEqual(x, 1.0, epsilon)) {
        // x is constrained to be in [0,1) for uniqueness
        n += 1;
        x = 0.0;
    }
#ifndef NDEBUG
    // compiler probably detects and optimizes this, but just be absolutely certain it's not
    // compiled for Release builds
    const auto rhs =
        abs(z) * std::exp(std::complex<double>(0, 2 * pi / static_cast<double>(d) * (n + x)));
    assert(isApproxEqual(z, rhs, epsilon));
#endif
    nRef = n;
    return x;
}

/**
 * A Pauli basis coefficient, i.e. a representation of
 *
 *      f_M(p,q) = r * exp(2*pi*i/d * (n + x)), r in R, n in Z, x in [0, 1)
 */
class PauliCoeff {
    const double r_;
    size_t n_;
    const double x_;

  public:
    PauliCoeff() : r_(0.0), n_(0), x_(0.0) {}

    PauliCoeff(const double r, const size_t n, const double x) : r_(r), n_(n), x_(x) {}

    PauliCoeff(const size_t d, const std::complex<double>& z, const double precisionFor_r,
               const double precisionFor_x)
        : r_(std::abs(z)), n_(0), x_(compute_x_and_n(d, z, n_, precisionFor_x)) {
        assert(isApproxEqual(z, complexVal(d), precisionFor_r));
    }

    [[nodiscard]] double r() const { return r_; }
    [[nodiscard]] double n() const { return n_; }
    [[nodiscard]] double x() const { return x_; }

    [[nodiscard]] std::complex<double> complexVal(const size_t d) const {
        return r_ * std::exp(std::complex<double>(0, 2 * pi / static_cast<double>(d) * (n_ + x_)));
    }

    // Overload the equality operator for use in unordered_map
    bool operator==(const PauliCoeff& other) const { return r_ == other.r_ && x_ == other.x_; }

    friend std::ostream& operator<<(std::ostream& os, const PauliCoeff& obj) {
        os << "PauliCoeff(r=" << obj.r() << ", n=" << obj.n() << ", x=" << obj.x() << ")";
        return os;
    }
};

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

    FMapKey(const PauliCoeff& coeff, const double precisionFor_r, const double precisionFor_x)
        : r_(roundMapKey(coeff.r(), precisionFor_r)), x_(roundMapKey(coeff.x(), precisionFor_x)) {}

    FMapKey(const size_t d, const std::complex<double>& z, const double precisionFor_r,
            const double precisionFor_x)
        : FMapKey(PauliCoeff(d, z, precisionFor_r, precisionFor_x), precisionFor_r,
                  precisionFor_x) {}

    [[nodiscard]] double r() const { return r_; }
    [[nodiscard]] double x() const { return x_; }

    // Overload the equality operator for use in unordered_map
    bool operator==(const FMapKey& other) const { return r_ == other.r_ && x_ == other.x_; }

    friend std::ostream& operator<<(std::ostream& os, const FMapKey& obj) {
        os << "FMapKey(r=" << obj.r() << ", x=" << obj.x() << ")";
        return os;
    }
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
 * In the n-qudigt case, f_M(p) is used where p in Z_d^(2n).
 *
 * std::unordered_map and std::list are used to ensure average constant-time access and
 * modification. Random access of list elements are not needed, and the only modifications are
 * appending elements to lists.
 */
using MatrixCoordinate = Eigen::Vector<long, Eigen::Dynamic>;

/**
 * Entries of coordinates paired with n, where coords[i] returns a vector v in Z_d^(2n) such that
 * n = n[i] and
 *
 *      f_M(v) = r * exp(2*pi*i/d * (n + x)), r in R, n in Z, x in [0,1)
 *
 * r and x can be obtained from the map key..
 */
struct FMapEntry {
    // these are paired together
    std::vector<MatrixCoordinate> coords;
    std::vector<size_t> n;

    size_t size() const {
        assert(coords.size() == n.size());
        return coords.size();
    }

    template <typename _Gen>
    void shuffle(_Gen&& gen) {
        if (size() <= 1) {
            return;
        }
        std::vector<size_t> indices(size());
        // fill with indices
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        // create new matrix based on shuffled indices
        std::vector<MatrixCoordinate> new_coords(size());
        std::vector<size_t> new_n(size());
        for (size_t i = 0; i < size(); ++i) {
            new_coords[i] = coords[indices[i]];
            new_n[i] = n[indices[i]];
        }

        // replace the original vectors
        coords = std::move(new_coords);
        n = std::move(new_n);
    }
};

struct FMap {
  private:
    const size_t d_;
    const size_t tupleSize_;
    const double precisionFor_r_;
    const double precisionFor_x_;
    std::unordered_map<FMapKey, FMapEntry, FMapKeyHash> map_;
    static const FMapEntry EMPTY_ENTRY;

    static constexpr size_t computeTupleLength(const size_t numQubits) {
        // 2 * numQubits
        return 2 * numQubits;
    }

  public:
    explicit FMap(const size_t d, const size_t n, const double precisionFor_r,
                  const double precisionFor_x)
        : d_(d), tupleSize_(computeTupleLength(n)), precisionFor_r_(precisionFor_r),
          precisionFor_x_(precisionFor_x) {};
    explicit FMap(const size_t d, const size_t n, const size_t mapReservation,
                  const double precisionFor_r, double precisionFor_x)
        : FMap(d, n, precisionFor_r, precisionFor_x) {
        map_.reserve(mapReservation);
    };

    const auto& getMap() const { return map_; }
    auto& getMap() { return map_; }

    /** Inserts the entry (p,q) and returns the key used */
    FMapKey insertEntry(Eigen::Vector<long, 2>&& coordinate, const std::complex<double>& value) {
        if (coordinate.rows() != tupleSize_) {
            throw std::invalid_argument("Bad matrix coordinate size");
        }
        const auto pauliCoeff = PauliCoeff(d_, value, precisionFor_r_, precisionFor_x_);
        const auto key = FMapKey(pauliCoeff, precisionFor_r_, precisionFor_x_);
        map_[key].coords.push_back(std::move(coordinate));
        map_[key].n.push_back(pauliCoeff.n());
        return key;
    }

    FMapKey insertEntryNTuple(MatrixCoordinate&& coordinate, const std::complex<double>& value) {
        if (coordinate.rows() != tupleSize_) {
            throw std::invalid_argument("Bad matrix coordinate size");
        }
        const auto pauliCoeff = PauliCoeff(d_, value, precisionFor_r_, precisionFor_x_);
        const auto key = FMapKey(pauliCoeff, precisionFor_r_, precisionFor_x_);
        map_[key].coords.push_back(std::move(coordinate));
        map_[key].n.push_back(pauliCoeff.n());
        return key;
    }

    const std::vector<MatrixCoordinate>& get(const FMapKey& key) const {
        const auto it = map_.find(key);
        return it != map_.end() ? it->second.coords : EMPTY_ENTRY.coords;
    }

    const FMapEntry& getWithN(const FMapKey& key) const {
        const auto it = map_.find(key);
        return it != map_.end() ? it->second : EMPTY_ENTRY;
    }

    std::optional<PauliCoeff> getPauliCoeffIfInSameBin(const std::complex<double>& value,
                                                       const FMapKey& targetKey) const {
        const auto pauliCoeff = PauliCoeff(d_, value, precisionFor_r_, precisionFor_x_);
        const auto key = FMapKey(pauliCoeff, precisionFor_r_, precisionFor_x_);
        return targetKey == key ? std::make_optional(pauliCoeff) : std::nullopt;
    }

    /**
     * @return Mutable reference
     */
    std::optional<std::reference_wrapper<FMapEntry>> getMut(const FMapKey& key) {
        const auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        return std::optional(std::ref(it->second));
    }

    size_t getCount(const std::complex<double>& fMValue) const {
        const auto pauliCoeff = PauliCoeff(d_, fMValue, precisionFor_r_, precisionFor_x_);
        const auto key = FMapKey(pauliCoeff, precisionFor_r_, precisionFor_x_);
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

    std::vector<FMapKey> sortedKeys() const {
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
