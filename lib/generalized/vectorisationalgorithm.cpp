#include "vectorisationalgorithm.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <optional>
#include <unordered_set>

#include "internal/FMap.h"
#include "internal/conjtestlemma10.h"

namespace cliffconjtest {

using CliffPermutationSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;
using PPrimeQPrimeSysMatrix = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>;

static size_t s_numRecursiveCalls = 0;

size_t returnLastNumRecursiveCalls() { return s_numRecursiveCalls; }

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> findSymplecticMatrixRecurse(
    const size_t d, const size_t n, const std::complex<double>& omega,
    const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
    const MpMatrixType& Mprime_p, const FMap& Mmap, const FMap& Mprimemap,
    const std::vector<FMapKey>& sortedKeys, size_t& maxKeyIndex, const size_t lastKeyIndex = 0,
    std::optional<CliffPermutationSysMatrix>&& systemForS = std::nullopt,
    std::optional<PPrimeQPrimeSysMatrix>&& systemForPPrimeQPrime = std::nullopt,
    std::unordered_set<size_t>&& selectedVecMIndices = {}) {

    s_numRecursiveCalls++;

    // assert(maxVecsMIndex < Mmap.size());
    if (lastKeyIndex > maxKeyIndex || lastKeyIndex >= sortedKeys.size()) {
        return std::nullopt;
    }

    std::optional<CliffPermutationSysMatrix> thisSystemForS;
    if (systemForS) {
        thisSystemForS = systemForS;
    }
    std::optional<CliffPermutationSysMatrix> thisSystemForPPrimeQPrime;
    if (systemForPPrimeQPrime) {
        thisSystemForPPrimeQPrime = systemForPPrimeQPrime;
    }

    const auto& key = sortedKeys[lastKeyIndex];
    const auto& vecsM = Mmap.get(key);
    const auto& vecsMprime = Mprimemap.get(key);
    if (vecsM.size() != vecsMprime.size()) {
        throw std::invalid_argument("map size mismatch");
    }

    for (size_t i = 0; i < vecsM.size(); ++i) {
        if (selectedVecMIndices.contains(i)) {
            continue;
        }

        // Test the validity of a symplectic matrix mapping v to vMap
        for (size_t j = 0; j < vecsMprime.size(); ++j) {
            const auto& v = vecsM[i];
            const auto& vMap = vecsMprime[j];
            CliffPermutationSysMatrix systemSForPair;
            PPrimeQPrimeSysMatrix systemPPrimeQPrimeForPair;
            if (!thisSystemForS || !thisSystemForPPrimeQPrime) {
                systemSForPair = createSystemForS(v, vMap);
                const auto alpha = M_p.get(v);
                const auto beta = Mprime_p.get(vMap);
                const double kTest = checkPhase(d, alpha, beta);
                const size_t k = std::round(kTest);
                if (std::abs(kTest - k) > 1e-5) {
                    continue;
                }
                systemPPrimeQPrimeForPair = createSystemForPPrimeQPrime(d, v, k);
            } else {
                const auto alpha = M_p.get(v);
                const auto beta = Mprime_p.get(vMap);
                const double kTest = checkPhase(d, alpha, beta);
                const size_t k = std::round(kTest);
                if (std::abs(kTest - k) > 1e-5) {
                    continue;
                }

                systemPPrimeQPrimeForPair = thisSystemForPPrimeQPrime.value();
                systemSForPair = thisSystemForS.value();

                appendToSystemForPPrimeQPrime(systemPPrimeQPrimeForPair, d, v, k);

                // TODO: Optimize this to only reduce the bottom row
                const size_t rankPQSystem =
                    reduceToREFAndGetRank(systemPPrimeQPrimeForPair, d, true);

                if (isSystemInconsistent(systemPPrimeQPrimeForPair)) {
                    continue;
                }

                // TODO: Optimize this to only reduce the bottom row
                appendToSystemForS(systemSForPair, v, vMap);
                const size_t rank = reduceToREFAndGetRank(systemSForPair, d, true);

                // Since S is vectorised, the rank should be the number of entries in S, i.e.
                // it's an (2n) x (2n) symplectic matrix
                if (rank == (2 * n) * (2 * n)) {
                    // Found vectors that are linearly independent; do not look at other keys.
                    // This is a reference, so it's maintained across all recursive calls.
                    maxKeyIndex = lastKeyIndex;
                    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S =
                        recoverSFromSystem(systemSForPair, n);
                    if (rankPQSystem == 2 * n && isSymplectic(S, d)) {
                        const auto pPrime_qPrime_Vec =
                            recoverPPrimeQPrimeVecFromSystem(systemPPrimeQPrimeForPair, n);
                        if (test_clifford_conjugate_lemma_10(d, std::move(pPrime_qPrime_Vec), omega,
                                                             M, M_p, Mprime_p, S)) {
                            return std::make_optional(S);
                        }
                    }
                    continue;
                }

                if (isSystemInconsistent(systemSForPair)) {
                    continue;
                }

                // Good systems; use them for the next iteration
                trimZeroRowsFromBottom(systemSForPair);
                trimZeroRowsFromBottom(systemPPrimeQPrimeForPair);
                thisSystemForS = systemSForPair;
                thisSystemForPPrimeQPrime = systemPPrimeQPrimeForPair;
                // The recursion below will advance the index i
            }

            std::unordered_set<size_t> thisSelected = selectedVecMIndices;
            thisSelected.insert(i);
            const auto recursiveResult = findSymplecticMatrixRecurse(
                d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys, maxKeyIndex,
                lastKeyIndex, std::make_optional(systemSForPair),
                std::make_optional(systemPPrimeQPrimeForPair), std::move(thisSelected));
            if (recursiveResult) {
                return recursiveResult;
            }
        }
    }

    return findSymplecticMatrixRecurse(
        d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys, maxKeyIndex,
        /* Advance the key index; unable to find in current bin */ lastKeyIndex + 1,
        std::move(thisSystemForS), std::move(thisSystemForPPrimeQPrime), {});
}

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(const size_t d, const size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, const FMap& Mmap, const FMap& Mprimemap) {
    const std::vector<FMapKey> sortedKeys = Mmap.sortedKeys();

    s_numRecursiveCalls = 0;

    size_t maxKeyIndex = Mmap.size() - 1;
    return findSymplecticMatrixRecurse(d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys,
                                       maxKeyIndex);
}

} // namespace cliffconjtest
