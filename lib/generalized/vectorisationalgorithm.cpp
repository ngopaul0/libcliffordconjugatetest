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

using MappingHistory = std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>>;
using AllowedVecMIndicesByKey = std::unordered_map<size_t, std::unordered_set<size_t>>;

static size_t s_numRecursiveCalls = 0;

size_t returnLastNumRecursiveCalls() { return s_numRecursiveCalls; }

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> findSymplecticMatrixRecurse(
    const size_t d, const size_t n, const std::complex<double>& omega,
    const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
    const MpMatrixType& Mprime_p, const FMap& Mmap, const FMap& Mprimemap,
    const std::vector<FMapKey>& sortedKeys, size_t& maxKeyIndex,
    AllowedVecMIndicesByKey& allowedKeys,
    const MappingHistory& history = {},
    const size_t lastKeyIndex = 0,
    std::optional<CliffPermutationSysMatrix>&& systemForS = std::nullopt,
    std::optional<PPrimeQPrimeSysMatrix>&& systemForPPrimeQPrime = std::nullopt,
    std::unordered_set<size_t>&& selectedVecMIndices = {}, std::unordered_set<size_t>&& selectedVecMprimeIndices = {},
    std::vector<std::pair<std::vector<long>, std::vector<long>>>&& mappings = {}) {

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
        const auto& v = vecsM[i];
        if (!allowedKeys.empty()) {
            if (!allowedKeys[lastKeyIndex].contains(i)) {
                continue;
            }
        }
        // Test the validity of a symplectic matrix mapping v to vMap
        for (size_t j = 0; j < vecsMprime.size(); ++j) {
            if (selectedVecMprimeIndices.contains(j)) {
                continue;
            }

            const auto& vMap = vecsMprime[j];
#ifndef NDEBUG

            size_t foundCount = 0;
            for (const auto& mapping : mappings) {
                std::vector<long> e1 = {1, 2, 2, 1};
                std::vector<long> e2 = {2, 1, 2, 1};
                if (mapping.first == e1 && mapping.second == e2) {
                    foundCount++;
                    continue;
                }
                e1 = {1, 2, 2, 0};
                e2 = {2, 1, 2, 0};
                if (mapping.first == e1 && mapping.second == e2) {
                    foundCount++;
                    continue;
                }
                e1 = {0, 1, 2, 1};
                e2 = {0, 2, 2, 1};
                if (mapping.first == e1 && mapping.second == e2) {
                    foundCount++;
                    continue;
                }
                e1 = {0, 1, 2, 0};
                e2 = {0, 2, 2, 0};
                if (mapping.first == e1 && mapping.second == e2) {
                    foundCount++;
                    continue;
                }
                e1 = {2, 2, 2, 1};
                e2 = {1, 1, 2, 1};
                if (mapping.first == e1 && mapping.second == e2) {
                    foundCount++;
                    continue;
                }
                e1 = {2, 2, 2, 0};
                e2 = {1, 1, 2, 0};
                if (mapping.first == e1 && mapping.second == e2) {
                    foundCount++;
                    continue;
                }
            }
            if (foundCount >= 2) {
                std::stringstream ss;

                // CliffPermutationSysMatrix system = createSystemForS(v, vMap);
            }



            std::stringstream ssV;
            ssV << v;
            auto stringV = ssV.str();

            std::stringstream ssVmap;
            ssVmap << vMap;
            auto stringVmap = ssVmap.str();

            if (foundCount > 0 && mappings.size() == foundCount) {
                std::stringstream ss;
                for (const auto& vv : vecsMprime) {
                    ss << vv << ", " << std::endl;
                }
                // debugger with i = 4, j = 0 on d=3, 2 qudits test
                // goes to i = 4, j = 1 (target is j=2), foundCount == 4
                auto alLVecs = ss.str();
                if (vMap.rows() >= 4) {
                    if (vMap(0) == 2 && vMap(1) == 1 && vMap(2) == 2 && vMap(3) == 1) {
                        std::stringstream ss2;
                    }
                }
            }

            if (v.rows() >= 4) {
                if (v(0) == 2 && v(1) == 2 && v(2) == 2 && v(3) == 0 &&
                vMap(0) == 1 && vMap(1) == 1 && vMap(2) == 2 && vMap(3) == 0) {
                    std::stringstream ss;
                }
            }


#endif
            CliffPermutationSysMatrix systemSForPair;
            PPrimeQPrimeSysMatrix systemPPrimeQPrimeForPair;
            if (!systemForS || !systemForPPrimeQPrime) {
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

                systemPPrimeQPrimeForPair = systemForPPrimeQPrime.value();
                systemSForPair = systemForS.value();
                const size_t rows = systemSForPair.rows();

                appendToSystemForPPrimeQPrime(systemPPrimeQPrimeForPair, d, v, k);

                // TODO: Optimize this to only reduce the bottom row
                const size_t rankPQSystem =
                    reduceToREFAndGetRank(systemPPrimeQPrimeForPair, d, true);

                // TODO: Optimize this to only reduce the bottom row
                appendToSystemForS(systemSForPair, v, vMap);

                CliffPermutationSysMatrix rrefSystem = systemSForPair;
                const size_t rank = reduceToREFAndGetRank(rrefSystem, d, true);

#ifndef NDEBUG
                std::stringstream ssSystemS;
                ssSystemS << systemSForPair;
                auto stringSSystem = ssSystemS.str();

                std::stringstream ssRREFSystemS;
                ssRREFSystemS << rrefSystem;
                auto stringRREFSSystem = ssRREFSystemS.str();
#endif

                // Since S is vectorised, the rank should be the number of entries in S, i.e.
                // it's an (2n) x (2n) symplectic matrix
                if (rank == (2 * n) * (2 * n)) {
                    // Found vectors that are linearly independent; do not look at other keys.
                    // This is a reference, so it's maintained across all recursive calls.
                    maxKeyIndex = lastKeyIndex;
                    if (allowedKeys.empty()) {
                        allowedKeys[lastKeyIndex].insert(i);
                        for (auto& [key, vec] : history) {
                            for (const auto& vInd : vec | std::views::keys) {
                                allowedKeys[key].insert(vInd);
                            }
                        }
                    }
                    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> S =
                        recoverSFromSystem(rrefSystem, n);

#ifndef NDEBUG
                    std::stringstream sss;
                    sss << S;
                    auto sString = sss.str();
#endif

                    if (isSystemInconsistent(systemPPrimeQPrimeForPair)) {
                        continue;
                    }

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
                if (rank > 2 * n * 2 * n || isSystemInconsistent(rrefSystem)) {
                    continue;
                }

                // Good systems; use them for the next iteration
                trimZeroRowsFromBottom(systemSForPair);
                trimZeroRowsFromBottom(systemPPrimeQPrimeForPair);
                // thisSystemForS = systemSForPair;
                // thisSystemForPPrimeQPrime = systemPPrimeQPrimeForPair;
                // The recursion below will advance the index i
            }

            std::unordered_set<size_t> thisSelected = selectedVecMIndices;
            thisSelected.insert(i);

            std::unordered_set<size_t> thisSelectedMprime = selectedVecMprimeIndices;
            thisSelectedMprime.insert(j);

            MappingHistory thisHistory = history;
            thisHistory[lastKeyIndex].push_back({i, j});
#ifndef NDEBUG
            std::vector<long> vVec;
            for (size_t ind = 0; ind < v.size(); ind++) {
                vVec.push_back(v(ind));
            }
            std::vector<long> vMapVec;
            for (size_t ind = 0; ind < vMap.size(); ind++) {
                vMapVec.push_back(vMap(ind));
            }
#endif
            std::vector newMappings(mappings);
#ifndef NDEBUG
            newMappings.push_back({std::move(vVec), std::move(vMapVec)});
#endif
            const auto recursiveResult = findSymplecticMatrixRecurse(
                d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys, maxKeyIndex, allowedKeys, thisHistory,
                lastKeyIndex, std::make_optional(systemSForPair),
                std::make_optional(systemPPrimeQPrimeForPair), std::move(thisSelected), std::move(thisSelectedMprime), std::move(newMappings));
            if (recursiveResult) {
                return recursiveResult;
            } else {
                std::stringstream ss1;
            }
        }
    }

    return findSymplecticMatrixRecurse(
        d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys, maxKeyIndex, allowedKeys,
        history,
        /* Advance the key index; unable to find in current bin */ lastKeyIndex + 1,
        std::move(thisSystemForS), std::move(thisSystemForPPrimeQPrime), {}, {}, {});
}

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(const size_t d, const size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, FMap& Mmap, FMap& Mprimemap) {
    const std::vector<FMapKey> sortedKeys = Mmap.sortedKeys();

    s_numRecursiveCalls = 0;

    size_t maxKeyIndex = Mmap.size() - 1;
    AllowedVecMIndicesByKey allowedKeys;
    return findSymplecticMatrixRecurse(d, n, omega, M, M_p, Mprime_p, Mmap, Mprimemap, sortedKeys,
                                       maxKeyIndex, allowedKeys);
}

} // namespace cliffconjtest
