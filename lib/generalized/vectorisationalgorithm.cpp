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

// Maps indices in sortedKeys to vector mapping by index
using MappingHistory = std::unordered_map<size_t, std::unordered_map<size_t, size_t>>;
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
     * other vectors in MNap.
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

    class UnmappedVecIndicesStack {
        static constexpr size_t MAX_SIZE_FOR_BITSET = 127;
        struct UnmappedIndicesLarge {
            std::vector<bool> unmappedVecMIndices_;
            std::vector<bool> unmappedVecMprimeIndices_;
            explicit UnmappedIndicesLarge(const size_t numVecs)
                : unmappedVecMIndices_(numVecs, true), unmappedVecMprimeIndices_(numVecs, true) {}
        };

        struct UnmappedIndicesSmall {
            std::bitset<1 + MAX_SIZE_FOR_BITSET> unmappedVecMIndices_;
            std::bitset<1 + MAX_SIZE_FOR_BITSET> unmappedVecMprimeIndices_;
            explicit UnmappedIndicesSmall() {
                unmappedVecMIndices_.set();
                unmappedVecMprimeIndices_.set();
            }
        };

        std::stack<std::variant<UnmappedIndicesSmall, UnmappedIndicesLarge>> unmappedIndicesStack_;

      public:
        void push(const size_t numVecs) {
            if (numVecs > MAX_SIZE_FOR_BITSET) {
                unmappedIndicesStack_.emplace(UnmappedIndicesLarge(numVecs));
            } else {
                unmappedIndicesStack_.emplace(UnmappedIndicesSmall());
            }
        }

        void pop() { unmappedIndicesStack_.pop(); }

        [[nodiscard]] bool isIndexAllowedVecM(const size_t idx) const {
            const auto& top = unmappedIndicesStack_.top();
            return std::visit(
                [idx]<typename T0>(T0&& arg) {
                    using T = std::decay_t<T0>;
                    if constexpr (std::is_same_v<T, UnmappedIndicesLarge>) {
                        return arg.unmappedVecMIndices_[idx];
                    } else {
                        return arg.unmappedVecMIndices_.test(idx);
                    }
                },
                top);
        }

        [[nodiscard]] bool isIndexAllowedVecMprime(const size_t idx) const {
            const auto& top = unmappedIndicesStack_.top();
            return std::visit(
                [idx]<typename T0>(T0&& arg) {
                    using T = std::decay_t<T0>;
                    if constexpr (std::is_same_v<T, UnmappedIndicesLarge>) {
                        return arg.unmappedVecMprimeIndices_[idx];
                    } else {
                        return arg.unmappedVecMprimeIndices_.test(idx);
                    }
                },
                top);
        }

        void markMapping(const size_t vecMIndex, const size_t vecMPrimeIndex) {
            auto& top = unmappedIndicesStack_.top();
            return std::visit(
                [vecMIndex, vecMPrimeIndex]<typename T0>(T0&& arg) {
                    using T = std::decay_t<T0>;
                    if constexpr (std::is_same_v<T, UnmappedIndicesLarge>) {
                        arg.unmappedVecMIndices_[vecMIndex] = false;
                        arg.unmappedVecMprimeIndices_[vecMPrimeIndex] = false;
                    } else {
                        arg.unmappedVecMIndices_.reset(vecMIndex);
                        arg.unmappedVecMprimeIndices_.reset(vecMPrimeIndex);
                    }
                },
                top);
        }

        void unmarkMapping(const size_t vecMIndex, const size_t vecMPrimeIndex) {
            auto& top = unmappedIndicesStack_.top();
            return std::visit(
                [vecMIndex, vecMPrimeIndex]<typename T0>(T0&& arg) {
                    using T = std::decay_t<T0>;
                    if constexpr (std::is_same_v<T, UnmappedIndicesLarge>) {
                        arg.unmappedVecMIndices_[vecMIndex] = true;
                        arg.unmappedVecMprimeIndices_[vecMPrimeIndex] = true;
                    } else {
                        arg.unmappedVecMIndices_.set(vecMIndex);
                        arg.unmappedVecMprimeIndices_.set(vecMPrimeIndex);
                    }
                },
                top);
        }
    };

    /**
     * Stores information about which vectors in MMap and MprimeMap are unmapped.
     * Vectors are considered mapped if they're in the system.
     */
    UnmappedVecIndicesStack unmappedVecIndicesStack_;

    [[nodiscard]] std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
    findSymplecticMatrix() {
        if (sortedKeys_.empty()) {
            return std::nullopt;
        }

        constexpr size_t sortedKeysStartIndex = 0;
        pushNewUnmappedIndices(sortedKeysStartIndex);
        return findSymplecticMatrixRecurse(sortedKeysStartIndex);
    }

  private:
    void pushNewUnmappedIndices(const size_t sortedKeysIndex) {
        const auto& key = sortedKeys_[sortedKeysIndex];
        const auto& vecsM = Mmap_.get(key);
        unmappedVecIndicesStack_.push(vecsM.size());
    }

    bool areConstantMultiples(const MatrixCoordinate& v1, const MatrixCoordinate& v2) const {
        // Find the first non-zero element in v2 modulo d
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> M(v1.rows(), 2);
        M.col(0) = v1;
        M.col(0) = v2;
        return reduceToREFAndGetRank(M, d_, false) == 2;
    }

    bool isHistoryContainingLinearlyDependentVector(const MatrixCoordinate& v1) {
        for (const auto& [index, mapForIndex] : history_) {
            const auto& key = sortedKeys_[index];
            const auto& vecsM = Mmap_.get(key);
            for (const auto& vInd : mapForIndex | std::views::keys) {
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
            for (const auto& [key, mapForKey] : history_) {
                for (const auto& vInd : mapForKey | std::views::keys) {
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

    void onMappingFailed(const size_t sortedMapKeyIndex, const size_t vecMIndex,
                         const size_t vecMPrimeIndex) {
        unmappedVecIndicesStack_.unmarkMapping(vecMIndex, vecMPrimeIndex);
        // Since the mapping did not work, do not use this mapping in the history. Pop
        // history off like a stack.
        //
        // We'll also discard the system for this mapping and recopy the previous system
        // from the start of the recursive call. Again, if the loop advances, it means
        // another possible mapping is being tried.
        if (const size_t countRemoved = history_[sortedMapKeyIndex].erase(vecMIndex);
            countRemoved != 1) {
            throw std::runtime_error("unexpected bad history state");
        }
    }
    void onMappingPlausible(const size_t sortedMapKeyIndex, const size_t vecMIndex,
                            const size_t vecMPrimeIndex) {
        unmappedVecIndicesStack_.markMapping(vecMIndex, vecMPrimeIndex);

        // Push history like a stack so that recursive calls are aware of exactly which
        // vectors we have mapped already. As S is a permutation, a one-to-one
        // correspondence is necessary.
        history_[sortedMapKeyIndex][vecMIndex] = vecMPrimeIndex;
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

        // The loops are used to go to the next mapping possibility if one fails.
        // The loops are not used to go down the bin and choose vectors for mapping.
        // The logic for going down the bins is recursively handled.
        //
        // If a mapping is plausible (i.e. the systems are consistent), a recursive call is made
        // try to map the other vectors.
        for (size_t i = 0; i < vecsM.size(); ++i) {
            if (!unmappedVecIndicesStack_.isIndexAllowedVecM(i)) {
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
                if (!unmappedVecIndicesStack_.isIndexAllowedVecMprime(j)) {
                    continue;
                }
                if (shouldSkipThisVecMKey(sortedMapKeyIndex, i)) {
                    goto skip_this_v;
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
                // advance the index i to map more vectors in vecM. This is the main step of
                // moving down through the bins.

                onMappingPlausible(sortedMapKeyIndex, i, j);
                const auto mappingResult = findSymplecticMatrixRecurse(
                    sortedMapKeyIndex, std::make_optional(systemSForPair),
                    std::make_optional(systemPPrimeQPrimeForPair), systemSRank);
                if (mappingResult) {
                    // If we're here, then this mapping is a success
                    return mappingResult;
                }
                // If we're here, then mapping this v from M to vMap from Mprime didn't work, so
                // continue the loop and try to map v to another vector from Mprime with the same
                // map key.
                //
                onMappingFailed(sortedMapKeyIndex, i, j);
            }

        skip_this_v:
        }

        // Advance the key index; unable to find in current bin
        const size_t newSortedKeysIndex = sortedMapKeyIndex + 1;
        if (newSortedKeysIndex > maxKeyIndex_ || newSortedKeysIndex >= sortedKeys_.size()) {
            // RECURSION END: There are no more points to check.
            return std::nullopt;
        }
        pushNewUnmappedIndices(newSortedKeysIndex);
        const auto resultWhenNextBinSearched =
            findSymplecticMatrixRecurse(newSortedKeysIndex, std::move(systemForS),
                                        std::move(systemForPPrimeQPrime), lastSystemSRank);
        unmappedVecIndicesStack_.pop();
        if (history_.contains(newSortedKeysIndex) && history_[newSortedKeysIndex].empty()) {
            history_.erase(newSortedKeysIndex);
        }
        return resultWhenNextBinSearched;
    }
};

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(const size_t d, const size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, FMap& Mmap, FMap& Mprimemap) {
    const std::vector<FMapKey> sortedKeys = Mmap.sortedKeys();

    s_numRecursiveCalls = 0;

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

    RecursionContext context{d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys};
    return context.findSymplecticMatrix();
}

} // namespace cliffconjtest
