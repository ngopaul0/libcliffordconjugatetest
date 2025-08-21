#include "vectorisationalgorithm.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <optional>
#include <random>
#include <unordered_set>

#include "internal/FMap.h"
#include "internal/conjtestlemma10.h"

namespace cliffconjtest {

using CliffPermutationSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;
using PPrimeQPrimeSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;

using MappingHistory =
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_set<size_t>>>;
using AllowedVecMIndicesByKey = std::unordered_map<size_t, std::unordered_set<size_t>>;

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
     * Stores the maximum (bin) index that we're allowed to look at.
     * The index is for sortedKeys.
     */
    size_t maxKeyIndex_ = Mmap_.size() - 1;
    /**
     * Modifiable and read across all recursive calls
     * Stores the keys in Mmap_ that we are allowed to look at. The idea behind this is that
     * once the system for S is consistent with one unique solution, we have found a basis inside of
     * Mmap_.
     */
    AllowedVecMIndicesByKey allowedKeys_;

    MappingHistory history_;

    [[nodiscard]] std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
    findSymplecticMatrix() {
        return findSymplecticMatrixRecurse();
    }

  private:
    void setAllowedKeysIfNeeded(const size_t lastKeyIndex, const size_t i) {
        if (allowedKeys_.empty()) {
            allowedKeys_[lastKeyIndex].insert(i);
            for (auto& [key, vec] : history_) {
                for (const auto& vInd : vec | std::views::keys) {
                    allowedKeys_[key].insert(vInd);
                }
            }
        }
    }

    std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> findSymplecticMatrixRecurse(
        const size_t lastKeyIndex = 0,
        std::optional<CliffPermutationSysMatrix>&& systemForS = std::nullopt,
        std::optional<PPrimeQPrimeSysMatrix>&& systemForPPrimeQPrime = std::nullopt,
        const std::unordered_set<size_t>&& selectedVecMprimeIndices = {}) {

        s_numRecursiveCalls++;

        // assert(maxVecsMIndex < Mmap.size());
        if (lastKeyIndex > maxKeyIndex_ || lastKeyIndex >= sortedKeys_.size()) {
            return std::nullopt;
        }

        const auto& key = sortedKeys_[lastKeyIndex];
        const auto& vecsM = Mmap_.get(key);
        const auto& vecsMprime = Mprimemap_.get(key);
        if (vecsM.size() != vecsMprime.size()) {
            throw std::invalid_argument("map size mismatch");
        }

        for (size_t i = 0; i < vecsM.size(); ++i) {
            if (history_[lastKeyIndex].contains(i)) {
                continue;
            }
            if (!allowedKeys_.empty()) {
                if (!allowedKeys_[lastKeyIndex].contains(i)) {
                    continue;
                }
            }

            const auto& v = vecsM[i];
            // Test the validity of a symplectic matrix mapping v to vMap
            for (size_t j = 0; j < vecsMprime.size(); ++j) {
                if (selectedVecMprimeIndices.contains(j)) {
                    continue;
                }

                const auto& vMap = vecsMprime[j];
                CliffPermutationSysMatrix systemSForPair;
                PPrimeQPrimeSysMatrix systemPPrimeQPrimeForPair;

                const auto alpha = M_p_.get(v);
                const auto beta = Mprime_p_.get(vMap);
                const double kTest = checkPhase(d_, alpha, beta);
                const size_t k = std::round(kTest);
                if (std::abs(kTest - k) > 1e-5) {
                    continue;
                }

                if (!systemForS || !systemForPPrimeQPrime) {
                    systemSForPair = createSystemForS(v, vMap);
                    systemPPrimeQPrimeForPair = createSystemForPPrimeQPrime(d_, v, k);
                } else {
                    systemPPrimeQPrimeForPair = systemForPPrimeQPrime.value();
                    systemSForPair = systemForS.value();

                    appendToSystemForPPrimeQPrime(systemPPrimeQPrimeForPair, d_, v, k);

                    // TODO: Optimize this to only reduce the bottom row
                    const size_t rankPQSystem =
                        reduceToREFAndGetRank(systemPPrimeQPrimeForPair, d_, true);

                    // TODO: Optimize this to only reduce the bottom row
                    appendToSystemForS(systemSForPair, v, vMap);

                    const size_t rank = reduceToREFAndGetRank(systemSForPair, d_, true);

                    // Since S is vectorised, the rank should be the number of entries in S, i.e.
                    // it's an (2n) x (2n) symplectic matrix
                    if (rank == (2 * n_) * (2 * n_)) {
                        // Found vectors that are linearly independent; do not look at other keys.
                        // The vectors that we have selected from MMap contain a basis already.
                        //
                        // These are maintained across all recursive calls.
                        maxKeyIndex_ = lastKeyIndex;
                        setAllowedKeysIfNeeded(lastKeyIndex, i);
                        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S =
                            recoverSFromSystem(systemSForPair, n_);

                        if (isSystemInconsistent(systemPPrimeQPrimeForPair)) {
                            continue;
                        }

                        if (rankPQSystem == 2 * n_ && isSymplectic(S, d_)) {
                            const auto pPrime_qPrime_Vec =
                                recoverPPrimeQPrimeVecFromSystem(systemPPrimeQPrimeForPair, n_);
                            if (test_clifford_conjugate_lemma_10(d_, pPrime_qPrime_Vec, omega_, M_,
                                                                 M_p_, Mprime_p_, S)) {
                                return std::make_optional(S);
                            }
                        }
                        continue;
                    }
                    if (rank > 2 * n_ * 2 * n_ || isSystemInconsistent(systemSForPair)) {
                        continue;
                    }

                    // Good systems; use them for the next iteration
                    trimZeroRowsFromBottom(systemSForPair);
                    trimZeroRowsFromBottom(systemPPrimeQPrimeForPair);
                    // The recursion below will advance the index i
                }


                // Create a separate set to maintain amortized O(1) checks
                std::unordered_set<size_t> thisSelectedMprime = selectedVecMprimeIndices;
                thisSelectedMprime.insert(j);

                // Push history onto "stack"
                history_[lastKeyIndex][i].insert(j);
                const auto isThisMappingCorrect = findSymplecticMatrixRecurse(
                    lastKeyIndex, std::make_optional(systemSForPair),
                    std::make_optional(systemPPrimeQPrimeForPair), std::move(thisSelectedMprime));
                if (isThisMappingCorrect) {
                    return isThisMappingCorrect;
                }
                // Pop history off "stack"
                size_t countRemoved = history_[lastKeyIndex][i].erase(j);
                if (countRemoved != 1) {
                    throw std::runtime_error("unexpected bad history state");
                }
                if (history_[lastKeyIndex][i].empty()) {
                    history_[lastKeyIndex].erase(i);
                }
                // If we're here, then mapping this v to vMap didn't work, so try again with anohter
                // vMap.
            }
        }

        return findSymplecticMatrixRecurse(
            /* Advance the key index; unable to find in current bin */ lastKeyIndex + 1,
            std::move(systemForS), std::move(systemForPPrimeQPrime));
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
