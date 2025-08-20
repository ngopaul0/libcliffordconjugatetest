#include "vectorisationalgorithm.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <optional>
#include <unordered_set>

#include "internal/FMap.h"

namespace cliffconjtest {

using CliffPermutationSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> findSymplecticMatrixRecurse(
    const size_t d, const size_t n, const FMap& Mmap, const FMap& Mprimemap,
    const std::vector<FMapKey>& sortedKeys, size_t maxKeyIndex, const size_t lastKeyIndex = 0,
    const std::optional<CliffPermutationSysMatrix>& systemForS = std::nullopt,
    const size_t lastVecMChecked = 0) {

    // assert(maxVecsMIndex < Mmap.size());
    if (lastKeyIndex >= maxKeyIndex) {
        return std::nullopt;
    }

    CliffPermutationSysMatrix thisSystem;
    bool hasSystemForS = false;
    if (systemForS) {
        hasSystemForS = true;
        thisSystem = *systemForS;
    }

    const auto& key = sortedKeys[lastKeyIndex];
    const auto& vecsM = Mmap.get(key);
    const auto& vecsMprime = Mprimemap.get(key);
    if (vecsM.size() != vecsMprime.size()) {
        throw std::invalid_argument("map size mismatch");
    }

    for (size_t i = lastVecMChecked; i < vecsM.size(); ++i) {
        for (size_t j = 0; j < vecsMprime.size(); ++j) {
            const auto& v = vecsM[i];
            const auto& vMap = vecsMprime[j];
            CliffPermutationSysMatrix systemForThisPair;
            if (!hasSystemForS) {
                systemForThisPair = createSystem(v, vMap);
            } else {
                systemForThisPair = thisSystem;
                appendToSystem(systemForThisPair, v, vMap);
                const size_t rank = reduceToREFAndGetRank(systemForThisPair, d, true);

                // Since S is vectorised, the rank should be the number of entries in S, i.e.
                // it's an (2n) x (2n) symplectic matrix
                if (rank == (2 * n) * (2 * n)) {
                    // Found vectors that are linearly independent; do not look at other keys.
                    maxKeyIndex = i;
                    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S =
                        recoverSFromSystem(systemForThisPair, n);
                    if (isSymplectic(S, d)) {
                        // TODO: Check that it's symplectic and that it works for pPrime, qPrime.
                        // TODO: Add a system for pPrime, qPrime
                        if (S(0, 0) == 2 && S(0, 1) == 0 && S(1, 0) == 0 && S(1, 1) == 2) {
                            return std::make_optional(S);
                        }
                    }
                    continue;
                }

                if (isSystemInconsistent(systemForThisPair)) {
                    continue;
                }
                // Good system; use it for the next iteration
                trimZeroRowsFromBottom(systemForThisPair);
                thisSystem = systemForThisPair;
                // The recursion below will advance the index i
            }

            const auto recursiveResult =
                findSymplecticMatrixRecurse(d, n, Mmap, Mprimemap, sortedKeys, maxKeyIndex,
                                            lastKeyIndex, std::make_optional(systemForThisPair),
                                            /* Advance the v index for this key */ i + 1);
            if (recursiveResult) {
                return recursiveResult;
            }
        }
    }

    return findSymplecticMatrixRecurse(
        d, n, Mmap, Mprimemap, sortedKeys, maxKeyIndex,
        /* Advance the key index; unable to find in current bin */ lastKeyIndex + 1,
        std::make_optional(thisSystem), 0);
}

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(const size_t d, const size_t n, const FMap& Mmap, const FMap& Mprimemap) {
    const std::vector<FMapKey> sortedKeys = Mmap.sortedKeys();

    return findSymplecticMatrixRecurse(d, n, Mmap, Mprimemap, sortedKeys, Mmap.size() - 1);
}

} // namespace cliffconjtest
