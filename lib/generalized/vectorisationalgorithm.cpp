#include "vectorisationalgorithm.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <optional>

#include "internal/FMap.h"
#include "internal/conjtestlemma10.h"
#include "internal/tupleiterator.h"

namespace cliffconjtest {

using CliffPermutationSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;

static size_t s_numRecursiveCalls = 0;

size_t returnLastNumRecursiveCalls() {
    return s_numRecursiveCalls;
}

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> findSymplecticMatrixRecurse(
    const size_t d, const size_t n, const std::complex<double>& omega,
    const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
    const MpMatrixType& Mprime_p, const FMap& Mmap, const FMap& Mprimemap,
    const std::vector<FMapKey>& sortedKeys, size_t maxKeyIndex, const size_t lastKeyIndex = 0,
    std::optional<CliffPermutationSysMatrix>&& systemForS = std::nullopt,
    const size_t lastVecMChecked = 0) {

    s_numRecursiveCalls++;

    // assert(maxVecsMIndex < Mmap.size());
    if (lastKeyIndex > maxKeyIndex || lastKeyIndex >= sortedKeys.size()) {
        return std::nullopt;
    }

    std::optional<CliffPermutationSysMatrix> thisSystem;
    if (systemForS) {
        thisSystem = systemForS;
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
            if (!thisSystem) {
                systemForThisPair = createSystem(v, vMap);
            } else {
                systemForThisPair = thisSystem.value();
                appendToSystem(systemForThisPair, v, vMap);
                const size_t rank = reduceToREFAndGetRank(systemForThisPair, d, true);

                // Since S is vectorised, the rank should be the number of entries in S, i.e.
                // it's an (2n) x (2n) symplectic matrix
                if (rank == (2 * n) * (2 * n)) {
                    // Found vectors that are linearly independent; do not look at other keys.
                    maxKeyIndex = lastKeyIndex;
                    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S =
                        recoverSFromSystem(systemForThisPair, n);
                    if (isSymplectic(S, d)) {
                        // TODO: Add a system for pPrime, qPrime
                        auto tupleIt = TupleIterator<SingleModulus>(d, 2 * n);
                        for (const auto pPrime_qPrime_vec : tupleIt) {
                            if (test_clifford_conjugate_lemma_10(d, std::move(pPrime_qPrime_vec),
                                                                 omega, M, M_p, Mprime_p, S)) {
                                return std::make_optional(S);
                            }
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

            const auto recursiveResult = findSymplecticMatrixRecurse(
                d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys, maxKeyIndex,
                lastKeyIndex, std::make_optional(systemForThisPair),
                /* Advance the v index for this key */ i + 1);
            if (recursiveResult) {
                return recursiveResult;
            }
        }
    }

    return findSymplecticMatrixRecurse(
        d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys, maxKeyIndex,
        /* Advance the key index; unable to find in current bin */ lastKeyIndex + 1,
        std::move(thisSystem), 0);
}

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(const size_t d, const size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, const FMap& Mmap, const FMap& Mprimemap) {
    const std::vector<FMapKey> sortedKeys = Mmap.sortedKeys();

    s_numRecursiveCalls = 0;
    return findSymplecticMatrixRecurse(d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys,
                                       Mmap.size() - 1);
}

} // namespace cliffconjtest
