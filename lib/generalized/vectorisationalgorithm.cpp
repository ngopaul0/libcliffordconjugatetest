#include "vectorisationalgorithm.h"

#include <Eigen/Dense>
#include <bitset>
#include <optional>
#include <random>
#include <stack>
#include <unordered_set>

#include "internal/FMap.h"
#include "internal/conjtestlemma10.h"

namespace cliffconjtest {

using CliffPermutationSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;
using PPrimeQPrimeSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * Stores information about which vectors in a particular bin have been mapped.
 * The sets are indices for the vector of MatrixCoordinates inside of FMap.
 *
 * The info is stored as sets for O(1) access and modification. Information about a specific mapping
 * is not stored unless in a Debug build.
 */
class BinMapping {
    // could possibly optimize this with std::vector<bool> or std::bitset (fixed, compile-time size)
    // for memory usage
    std::unordered_set<size_t> mappedVecM_;
    std::unordered_set<size_t> mappedVecMprime_;
#ifndef NDEBUG
    std::unordered_map<size_t, size_t> fullMapInfo_;
#endif

  public:
    BinMapping() {}

    bool isVecMIndexMapped(const size_t vecMIndex) const { return mappedVecM_.contains(vecMIndex); }

    bool isVecMprimeIndexMapped(const size_t vecMPrimeIndex) const {
        return mappedVecMprime_.contains(vecMPrimeIndex);
    }

    void markMapping(const size_t vecMIndex, const size_t vecMPrimeIndex) {
        mappedVecM_.insert(vecMIndex);
        mappedVecMprime_.insert(vecMPrimeIndex);
#ifndef NDEBUG
        fullMapInfo_[vecMIndex] = vecMPrimeIndex;
#endif
    }

    void unmarkMapping(const size_t vecMIndex, const size_t vecMPrimeIndex) {
        if (mappedVecM_.erase(vecMIndex) != 1) {
            throw std::runtime_error("invalid map marking state");
        };
        if (mappedVecMprime_.erase(vecMPrimeIndex) != 1) {
            throw std::runtime_error("invalid map marking state");
        }
#ifndef NDEBUG
        if (fullMapInfo_.contains(vecMIndex)) {
            fullMapInfo_.erase(vecMIndex);
        }
#endif
    }

    std::unordered_set<size_t> getMappedVecMIndices() const { return mappedVecM_; }

    size_t size() const {
        assert(mappedVecM_.size() == mappedVecMprime_.size());
        return mappedVecM_.size();
    }

    bool empty() const {
        assert(mappedVecM_.empty() == mappedVecMprime_.empty());
        return mappedVecM_.empty();
    }
};

// these could be space optimized since it's just a map using indices 0..Mmap.size().
// A vector<BinMapping> could just as easily work
using MappingHistory = std::unordered_map<size_t, BinMapping>;
using VecIndicesSetByKey = std::unordered_map<size_t, std::unordered_set<size_t>>;

static size_t s_numRecursiveCalls = 0;

size_t returnLastNumRecursiveCalls() { return s_numRecursiveCalls; }

struct RecursionContext {
    const size_t d_;
    const size_t n_;
    const std::complex<double>& omega_;
    const Eigen::Ref<const Eigen::MatrixXcd>& M_;
    const MpMatrixType& M_p_;
    const MpMatrixType& Mprime_p_;
    const FMap& Mmap_;
    const FMap& Mprimemap_;
    const std::vector<FMapKey>& sortedKeys_;

    /**
     * Modifiable and read across all recursive calls.
     * Stores the maximum (bin) index for sortedKeys_ that we're allowed to look at.
     */
    size_t maxKeyIndex_ = Mmap_.size() - 1;
    /**
     * Modifiable and read across all recursive calls
     * Corresponds to the keys in Mmap_ that we are allowed to look at. The idea behind this is that
     * once the system for S is consistent with one unique solution, there's no need to look at
     * other vectors in MNap, since there will be a basis in the vectors we're looking at.
     */
    VecIndicesSetByKey allowedKeys_;
    /**
     * Modifiable and read across all recursive calls
     * Stores the keys in Mmap_ that correspond to a vector in Mmap_ that's linearly dependent to
     * some other vector in Mmap_. The idea is that we shouldn't try to include this in the system,
     * since it'll add no new information.
     */
    VecIndicesSetByKey linearlyDependentKeys_;
    /**
     * Modifiable and read across all recursive calls
     * Contains the current path. This is managed in a stack-like fashion where the history is
     * appended before doing a recursive call, and history is popped after a recursive call.
     */
    MappingHistory history_ = MappingHistory(Mmap_.size());

    [[nodiscard]] std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
    findSymplecticMatrix() {
        if (sortedKeys_.empty()) {
            return std::nullopt;
        }

        constexpr size_t sortedKeysStartIndex = 0;
        return findSymplecticMatrixRecurse(sortedKeysStartIndex);
    }

  private:
    bool areConstantMultiples(const MatrixCoordinate& v1, const MatrixCoordinate& v2) const {
        // Find the first non-zero element in v2 modulo d
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> M(v1.rows(), 2);
        M.col(0) = v1;
        M.col(0) = v2;
        return reduceToREFAndGetRank(M, d_, false) == 2;
    }

    bool isHistoryContainingLinearlyDependentVector(const MatrixCoordinate& v1) {
        for (const auto& [index, binMapping] : history_) {
            const auto& key = sortedKeys_[index];
            const auto& vecsM = Mmap_.get(key);
            for (const auto& vInd : binMapping.getMappedVecMIndices()) {
                const MatrixCoordinate& v2 = vecsM[vInd];
                assert(v1 != v2 && "vector we're trying to search shouldn't in history");
                if (areConstantMultiples(v1, v2)) {
                    return true;
                }
            }
        }
        return false;
    }

    void useHistoryToFillAllowedKeysIfNeeded(const size_t lastKeyIndex, const size_t i) {
        if (allowedKeys_.empty()) {
            allowedKeys_[lastKeyIndex].insert(i);
            for (const auto& [key, binMapping] : history_) {
                for (const auto& vInd : binMapping.getMappedVecMIndices()) {
                    allowedKeys_[key].insert(vInd);
                }
            }
        }
    }

    bool shouldSkipThisVecMKey(const size_t lastKeyIndex, size_t i) {
        // This will be nonempty when we encounter a full-rank system for S in which case
        // we limit it to examining the vectors that allow us to recover S.
        if (!allowedKeys_.empty()) {
            if (!allowedKeys_[lastKeyIndex].contains(i)) {
                return true;
            }
        }
        // This will be nonempty if we find linearly dependent vectors
        if (!linearlyDependentKeys_.empty()) {
            if (linearlyDependentKeys_[lastKeyIndex].contains(i)) {
                return true;
            }
        }
        return false;
    }

    void unmarkMappingAndPopFromHistory(const size_t sortedMapKeyIndex, const size_t vecMIndex,
                                        const size_t vecMPrimeIndex) {
        // Since the mapping did not work, do not use this mapping in the history. Pop
        // history off like a stack.
#ifndef NDEBUG
        const size_t sizeBefore = history_[sortedMapKeyIndex].size();
#endif
        history_[sortedMapKeyIndex].unmarkMapping(vecMIndex, vecMPrimeIndex);
        assert(sizeBefore == history_[sortedMapKeyIndex].size() + 1);
    }
    void markMappingAndPushToHistory(const size_t sortedMapKeyIndex, const size_t vecMIndex,
                                     const size_t vecMPrimeIndex) {
        // Push history like a stack so that recursive calls are aware of exactly which
        // vectors we have mapped already. As S is a permutation, a one-to-one
        // correspondence is necessary.
        history_[sortedMapKeyIndex].markMapping(vecMIndex, vecMPrimeIndex);
    }
    /**
     * Recursively finds a symplectic matrix S that satisfies Lemma 10. This is a backtracking
     * backed algorithm.
     *
     * @param sortedMapKeyIndex The index in sortedKeys_ to use
     * @param systemForS The running system [X^T tensor I]vec(S) = vec(Y) to construct S based on
     * vectorisation, where SX = Y.
     * @param systemForPPrimeQPrime The running system of equations to construct the vectors p' and
     * q'
     * @param lastSystemSRank The rank of systemForPPrimeQPrime from the last iteration.
     * @return Whether a symplectic matrix was found or not.
     */
    std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> findSymplecticMatrixRecurse(
        const size_t sortedMapKeyIndex,
        std::optional<CliffPermutationSysMatrix>&& systemForS = std::nullopt,
        std::optional<PPrimeQPrimeSysMatrix>&& systemForPPrimeQPrime = std::nullopt,
        const size_t lastSystemSRank = 0) {
        assert(lastSystemSRank < (2 * n_) * (2 * n_) && "should never recurse on a full-rank sys");

        s_numRecursiveCalls++;

        if (sortedMapKeyIndex > maxKeyIndex_ || sortedMapKeyIndex >= sortedKeys_.size()) {
            // RECURSION END: There are no more points to check.
            return std::nullopt;
        }

        const auto& key = sortedKeys_[sortedMapKeyIndex];
        const auto& vecsM = Mmap_.get(key);
        const auto& vecsMprime = Mprimemap_.get(key);
        if (vecsM.size() != vecsMprime.size()) {
            throw std::invalid_argument("map size mismatch");
        }

        if (history_[sortedMapKeyIndex].size() == vecsM.size()) {
            // If we're here, then all the vectors in this bin are mapped.
            // Skip the loops entirely because we're just going to continue everything.
            goto next_bin_key;
        }

        // The loops are used to go to the next mapping possibility if one fails.
        // The loops are not used to go down the bin and choose vectors for mapping.
        // The logic for going down the bins is recursively handled.
        //
        // If a mapping is plausible (i.e. the systems are consistent), a recursive call is made
        // try to map the other vectors.
        for (size_t i = 0; i < vecsM.size(); ++i) {
            if (history_[sortedMapKeyIndex].isVecMIndexMapped(i)) {
                continue;
            }
            if (shouldSkipThisVecMKey(sortedMapKeyIndex, i)) {
                continue;
            }

            const auto& v = vecsM[i];
            // Test the validity of a symplectic matrix mapping v to vMap.
            // If this inner loop continues, that means that particular mapping failed, and the
            // next iteration is looking at another mapping possibility.
            for (size_t j = 0; j < vecsMprime.size(); ++j) {
                if (history_[sortedMapKeyIndex].isVecMprimeIndexMapped(j)) {
                    continue;
                }
                if (shouldSkipThisVecMKey(sortedMapKeyIndex, i)) {
                    break;
                }

                const auto& vMap = vecsMprime[j];
                // Will create a copy of the system
                CliffPermutationSysMatrix systemSForPair;
                PPrimeQPrimeSysMatrix systemPPrimeQPrimeForPair;

                const auto alpha = M_p_.get(v);
                const auto beta = Mprime_p_.get(vMap);
                const double kTest = checkPhase(d_, alpha, beta);
                const size_t k = std::llround(kTest);
                if (std::abs(kTest - k) > 1e-5) {
                    continue;
                }

                size_t systemSRank = 0;
                if (!systemForS || !systemForPPrimeQPrime) {
                    systemSForPair = createSystemForS(v, vMap);
                    systemPPrimeQPrimeForPair = createSystemForPPrimeQPrime(d_, v, k);

                    systemSRank = reduceToREFAndGetRank(systemSForPair, d_, true);
                } else {
                    systemPPrimeQPrimeForPair = systemForPPrimeQPrime.value();
                    systemSForPair = systemForS.value();

                    appendToSystemForPPrimeQPrime(systemPPrimeQPrimeForPair, d_, v, k);

                    // TODO: Optimize this to only reduce the bottom row
                    const size_t rankPQSystem =
                        reduceToREFAndGetRank(systemPPrimeQPrimeForPair, d_, true);

                    if (rankPQSystem > 2 * n_ || isSystemInconsistent(systemPPrimeQPrimeForPair)) {
                        continue;
                    }

                    // TODO: Optimize this to only reduce the bottom row
                    //  An Eigen sparse matrix could also be used to reduce space.
                    appendToSystemForS(systemSForPair, v, vMap);

                    systemSRank = reduceToREFAndGetRank(systemSForPair, d_, true);

                    // Since S is vectorised, the rank should be the number of entries in S, i.e.
                    // it's a (2n) x (2n) symplectic matrix
                    if (systemSRank == (2 * n_) * (2 * n_)) {
                        // Found vectors that can determine S; do not look at other keys for the
                        // rest of the recursive calls.
                        // Now we just have to find the right mapping.
                        maxKeyIndex_ = sortedMapKeyIndex;
                        useHistoryToFillAllowedKeysIfNeeded(sortedMapKeyIndex, i);
                        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S =
                            recoverSFromSystem(systemSForPair, n_);

                        if (rankPQSystem == 2 * n_ && isSymplectic(S, d_)) {
                            const auto pPrime_qPrime_Vec =
                                recoverPPrimeQPrimeVecFromSystem(systemPPrimeQPrimeForPair, n_);
                            if (test_clifford_conjugate_lemma_10(d_, pPrime_qPrime_Vec, omega_, M_,
                                                                 M_p_, Mprime_p_, S)) {
                                // RECURSION END: We have found a valid S that satisfies Lemma 10.
                                return std::make_optional(S);
                            }
                        }
                        continue;
                    }
                    // TODO: Update this with the better method for handling linear dependent
                    // vectors
                    if (systemSRank == lastSystemSRank) {
                        // System rank not changing means what we just added was just a multiple
                        // of some other row
                        if (isHistoryContainingLinearlyDependentVector(v)) {
                            // Leave this out for the rest of the iterations.
                            linearlyDependentKeys_[sortedMapKeyIndex].insert(i);
                            continue;
                        }
                    }

                    if (systemSRank > 2 * n_ * 2 * n_ || isSystemInconsistent(systemSForPair)) {
                        continue;
                    }

                    // Good systems; use them for the next iteration
                    trimZeroRowsFromBottom(systemSForPair);
                    trimZeroRowsFromBottom(systemPPrimeQPrimeForPair);
                    // The recursion below will advance the index i
                }
                // At this point, the attempted mapping resulted in systemSForPair being consistent
                // but with more than one unique solution.
                //
                // With this mapping of vecM[i] to vecMprime[j], the recursion below will
                // use a different index i to map more vectors in vecM for this bin. This is the
                // main step of moving down through the bins. The mapping must be marked to avoid
                // vectors that are already being used.
                const size_t vecMIndex = i;
                const size_t vecMPrimeIndex = j;
                // Push history like a stack so that recursive calls are aware of exactly which
                // vectors we have mapped already. As S is a permutation, a one-to-one
                // correspondence is necessary.
                history_[sortedMapKeyIndex].markMapping(vecMIndex, vecMPrimeIndex);
                // RECURSION DIVE: Use this mapping and check the other vectors in this bin, or
                // go on to the next bin.
                const auto mappingResult = findSymplecticMatrixRecurse(
                    sortedMapKeyIndex, std::make_optional(systemSForPair),
                    std::make_optional(systemPPrimeQPrimeForPair), systemSRank);
                if (mappingResult) {
                    // If we're here, then this mapping is a success
                    return mappingResult;
                }
                // If we're here, then mapping this v from M to vMap from Mprime didn't work, so
                // continue the loop and try to map v to another vector from Mprime with the same
                // map key
                const size_t sizeBefore = history_[sortedMapKeyIndex].size();
                history_[sortedMapKeyIndex].unmarkMapping(i, j);
                assert(sizeBefore == history_[sortedMapKeyIndex].size() + 1);
                // We'll also discard the system for this mapping and recopy the previous system
                // from the start of this recursive call. Again, if the loop advances, it means
                // another possible mapping is being tried.
            }
        }
    next_bin_key:
        // If we are here, then either all the vectors in this bin are already mapped to something,
        // (resulting in a loop exit)
        // or we tried all the mappings for vectors in vecM and failed and exited the loop.

        // Single-element bins have a single, direct mapping. If we got here, then we've failed.
        if (vecsM.size() == 1 && !history_[sortedMapKeyIndex].isVecMIndexMapped(0)) {
            return std::nullopt;
        }
        if (!allowedKeys_.empty()) {
            // If the allowed keys are set, we should only be proceeding if at least one of those
            // keys are mapped to something.
            const auto& allowedVecMIndices = allowedKeys_[sortedMapKeyIndex];
            const auto& binMapping = history_[sortedMapKeyIndex];
            const bool isAtLeastOneAllowedVecBeingMapped =
                std::ranges::any_of(allowedVecMIndices, [&](const size_t vecMIndex) -> bool {
                    return binMapping.isVecMIndexMapped(vecMIndex);
                });
            if (!isAtLeastOneAllowedVecBeingMapped) {
                return std::nullopt;
            }
        }

        // Advance the key index; unable to find in current bin
        const size_t newSortedKeysIndex = sortedMapKeyIndex + 1;
        if (newSortedKeysIndex > maxKeyIndex_ || newSortedKeysIndex >= sortedKeys_.size()) {
            // RECURSION END: There are no more points to check.
            return std::nullopt;
        }
        // RECURSION DIVE: Check the next bin.
        const auto resultWhenNextBinSearched =
            findSymplecticMatrixRecurse(newSortedKeysIndex, std::move(systemForS),
                                        std::move(systemForPPrimeQPrime), lastSystemSRank);
        // Cleanup history
        if (history_.contains(newSortedKeysIndex) && history_[newSortedKeysIndex].empty()) {
            history_.erase(newSortedKeysIndex);
        }
        return resultWhenNextBinSearched;
    }
};

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(const size_t d, const size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, FMap& Mmap, FMap& Mprimemap,
                     const bool shuffleKeys) {
    const std::vector<FMapKey> sortedKeys = Mmap.sortedKeys();

    s_numRecursiveCalls = 0;

    if (shuffleKeys) {
        std::random_device rd;
        std::mt19937 gen(rd());
        for (const auto& key : sortedKeys) {
            auto vecM = Mmap.getMut(key);
            if (vecM) {
                std::ranges::shuffle(vecM->get(), gen);
            }

            auto vecMprime = Mprimemap.getMut(key);
            if (vecMprime) {
                std::ranges::shuffle(vecMprime->get(), gen);
            }
        }
    }

    RecursionContext context{d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys};
    return context.findSymplecticMatrix();
}

} // namespace cliffconjtest
