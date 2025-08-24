#include "cliffordconjugacytest.hpp"

#include <complex>
#include <iostream>

#include "generalized/vectorisationalgorithm.h"
#include "internal/FMap.h"
#include "internal/bruteforcetest.h"
#include "internal/multidimarray.h"
#include "internal/util.h"

namespace cliffconjtest {

bool isSymplecticTransformation(size_t d, size_t s1, size_t s2, size_t s3, size_t s4) {
    // Symplectic if determinant is 1 (for 2x2 matrices over Z_d)
    return safeMod(s1 * s4 - s2 * s3, d) == 1;
}

std::optional<std::pair<FMapKey, std::pair<MatrixCoordinate, size_t>>>
findLinearIndependentCoord(const size_t d, const FMap& histogramM,
                           const std::vector<FMapKey>& sortedKeysM,
                           const MatrixCoordinate& nonZeroCoord, bool skipZeroKeys) {

    // Given a subset X of Z_d^2, the following holds:
    //      Let (a,b) in X be a nonzero point, (a,b) \neq (0,0).
    //      Then (a,b) is linearly dependent to all points in X iff every point im X is pairwise
    //      linearly dependent.
    // This means it suffices to just loop through all the coordinates once with only one point
    // used as the check (nonZeroCoord). If we go through all points and find that nonZeroCoord is
    // linearly dependent to all of them, then we know all points are in a line.
    //
    //      Proof: (=>) Let (a,b) in X be linearly dependent to all points in X.
    //      Take any (x1, x2), (y1, y2) in X. Assume WLOG that (y1, y2) is nonzero.
    //      Then (x1, x2) = m*(a,b) and (y1, y2) = n*(a,b). n is nonzero since (y1,y2) and (a,b)
    //      nonzero, hence n is invertible in mod d.
    //      Thus we have
    //          (x1, x2) + (-m)*n^{-1}(y1, y2) = m*(a,b) + (-m)*n^{-1}*n(a,b)
    //                                         = m*(a,b) - m(a,b)
    //                                         = 0
    //      (<=) This is clear.
    for (const auto& key : sortedKeysM) {
        if (skipZeroKeys && key.r() == 0.0) {
            continue;
        }

        const auto& coords = histogramM.getWithN(key);
        for (size_t i = 0; i < coords.size(); i++) {
            const auto& coord = coords.coords[i];
            // theDeterminant := (nonZeroCoord[1] * coord[2] - coord[1] * nonZeroCoord[2]) mod d;
            const auto determinant =
                safeMod(nonZeroCoord(0) * coord(1) - coord(0) * nonZeroCoord(1), d);
            if (determinant != 0) {
                return std::make_optional(std::make_pair(key, std::make_pair(coord, coords.n[i])));
            }
        }
    }
    return std::nullopt;
}

Eigen::Vector<long, 2> applyTransformation(size_t d, const MatrixCoordinate& target, size_t x0,
                                           size_t x1, size_t x2, size_t x3) {
    return {(x0 * target(0) + x1 * target(1)) % d, (x2 * target(0) + x3 * target(1)) % d};
}

/**
 * @brief This function will check whether the given vectors v and u are mappings of each other
 * according to Lemma 10. More precisely, it'll validate a necessary condition for u = S(v) to hold.
 *
 * Assuming that the only nonzero entries in M_p have coordinates that are all in a line (all
 * pairwise linearly dependent), validates whether the equations from Lemma 10,
 *
 *      f_M(v) = omega^k * f_{M'}(u) and u = S(v),
 *
 * are satisfied for the given v and u.
 *
 * @param d Prime modulus (odd prime)
 * @param omega dth root of unity
 * @param histogramM Mapping of absolute values of M_p
 * @param M_p Precomputed f_M(p,q) values
 * @param Mprime_p Precomputed f_{M'}(p,q) values
 * @param sortedKeysM Keys of histogramM that are sorted by number of coordinates that have that key
 * @param v The selected vector which is linearly dependent to all coordinates with nonzero entries
 * in M_p
 * @param u The vector u which emits a necessary condition v = S(u) by Lemma 10.
 * @param k Exponent on omega. For Lemma 10, we want this to be [v, (p',q')]
 * @return Whether f_M(v) = omega^k * f_{M'}(u) is satisfied
 */
bool validateNecessaryCondFromLinDepPoints(const Eigen::Index d, const std::complex<double>& omega,
                                           const FMap& histogramM, const MpMatrixType& M_p,
                                           const MpMatrixType& Mprime_p,
                                           const std::vector<FMapKey>& sortedKeysM,
                                           const MatrixCoordinate& v, const MatrixCoordinate& u,
                                           const size_t k) {
    // Testing f_M(v) = omega^k * f_{M'}(u) is trivial. But testing u = S(v) requires more work.
    //
    // From f_M(v) = omega^k * f_{M'}(u), by Lemma 10, we want k = [v, (p',q')].
    //
    // Loop over all (p,q) such that f_M(p,q) is nonzero.
    // Validates that f_M((p,q)) = omega^[(p,q), (p',q')] * f_{M'}(S(p,q)) holds for all nonzero
    // (p,q), using the assumption that u = S(v) and c*v = (p,q) for some c. If u = S(v) holds, then
    // a necessary condition would be that
    //
    //      f_M((p,q)) = omega^[(p,q), (p',q')] * f_{M'}(S(p,q))
    //                 = omega^[c*v, (p',q')] * f_{M'}(S(c*v))
    //                 = omega^(c*[v, (p',q')]) * f_{M'}(c*S(v))
    //                 = omega^(c*k) * f_{M'}(c*u)
    //
    // To do this, loop over all sorted keys and then access the map in nested loop.
    // The overall iterations for the nested loops is O(d), because all nonzero points in M_p are
    // pairwise linearly dependent, so tbere's only 1 line of at most O(d) length
    for (const auto& absVal : sortedKeysM) {
        if (absVal.r() == 0.0) {
            continue;
        }
        for (const auto& coordsForAbsVal : histogramM.get(absVal)) {
            const auto p = coordsForAbsVal(0);
            const auto q = coordsForAbsVal(1);
            if (p == 0 && q == 0) {
                continue;
            }
            assert(safeMod(p * v(1) - v(0) * q, d) == 0);

            // Calculate c where c*v = (p,q), so c = pv1^(-1) and c = qv2^(-1)
            const size_t c = safeMod(p != 0 ? p * modInverse(v(0), d) : q * modInverse(v(1), d), d);
            assert(safeMod(c * v(0), d) == p && safeMod(c * v(1), d) == q);

            // alpha is f_M(p,q)
            const auto& alpha = M_p.get(coordsForAbsVal);
            // beta is f_{M'}(S(p,q)), but by Lemma 10, we would need S(p,q) = S(c*v) = c*S(v) = c*u
            Eigen::Vector<long, 2> cTimesU(safeMod(c * u(0), d), safeMod(c * u(1), d));
            const auto& beta = Mprime_p.get(cTimesU);
            // By Lemma 10, the power of omega is
            //     [(p,q), (p',q')] = [(cv1, cv2), (p', q')]
            //                      = cv1 * q' - p' * cv2
            //                      = c(v1q' - p'v2)
            //                      = c * [v, (p',q')]
            //                      = c * k
            const auto omegaPow = std::pow(omega, safeMod(c * k, d));
            const auto product = omegaPow * beta;
            if (!isApproxEqual(alpha, product)) {
                return false;
            }
        }
    }
    // Here we've successfully validated the necessary condition
    // f_M((p,q)) = omega^(c*k) * f_{M'}(c*u) for all nonzero (p,q) such that f_M((p,q)) != 0.
    // This is not(?) a sufficient condition for Lemma 10; we still need to recover p', q' and S
    // itself.
    return true;
}
bool isCliffordConjugate(const Eigen::Ref<const Eigen::MatrixXcd>& M,
                         const Eigen::Ref<const Eigen::MatrixXcd>& M_prime) {
    constexpr double mapAbsValPrecision = 1e-5;
    constexpr double mapXPrecision = 1e-4;

    if (M.rows() != M.cols() || M_prime.rows() != M_prime.cols() || M_prime.cols() != M.cols()) {
        return false;
    }

    // Since trace is cyclic, if M = CM'C^*, then tr(M) = tr(CM'C^*) = tr(M'C^*C) = tr(M').
    if (!isApproxEqual(M.trace(), M_prime.trace())) {
        return false;
    }

    // earlier assert guarantees M.rows() == M.cols()
    const auto d = M.rows();
    const auto inv2 = modInverse(2, d);
    const auto omega = std::exp(std::complex<double>(0, 2 * pi / d));

    // Via Lemma 10, if M and M' are Clifford-conjugate, then
    // f_M(p,q) = omega^[(p,q),(p',q')]f_{M'}(S(p,q)). Taking the absolute value would leave us with
    // |f_M(p,q)| = |f_{M'}(S(p,q))|. S is a symplectic linear transformation, hence it's a
    // permutation. Now we try to recover this permutation.
    //
    // Let the entries of matrix M_p = [f_M(i,j)]_{ij} and M'_p = [f_{M'}(i,j)]_{ij}. By the above,
    // we must have that the entries of M_p and M'_p are permutations of each other, i.e. they have
    // the same entry values but they're in different locations.
    //
    // Create a mapping of absolute values to matrix coordinates, i.e. if r is a real number, then
    // histogramM.get(r) = { (p,q) |  |f_{M}(p,q)| = r }. This mapping can serve as a histogram of
    // |f_M(p,q)| values by counting the number of elements in the list returned by
    // histogramM.get(r).
    FMap histogramM(d, 1, mapAbsValPrecision, mapXPrecision);
    MpMatrixType M_p(2, d);
    bool allEqual = true;
    for (const auto& coord : M_p.indexIterator()) {
        assert(coord.size() == 2);
        const size_t p = coord[0];
        const size_t q = coord[1];
        if (allEqual && !isApproxEqual(M(p, q), M_prime(p, q))) {
            allEqual = false;
        }
        const auto value = f_multiqudit(M, coord, d, inv2, omega);
        ;
        M_p.get(coord) = value;
        histogramM.insertEntry(std::move(coord), value);
    }

    if (allEqual) {
        // M = M', so true since C = I works
        return true;
    }

    // Now construct a mapping for M' while also checking if the histogram for M' is equal to the
    // one for M. If not, then the two don't have the same entry values.
    FMap histogramMprime(d, 1, histogramM.size(), mapAbsValPrecision, mapXPrecision);
    MpMatrixType Mprime_p(2, d);
    for (const auto& coord : Mprime_p.indexIterator()) {
        assert(coord.size() == 2);

        const auto key = f_multiqudit(M_prime, coord, d, inv2, omega);
        Mprime_p.get(coord) = key;

        const auto mapKey = histogramMprime.insertEntry(std::move(coord), key);
        // Early histogram check
        if (histogramMprime.getCount(mapKey) > histogramM.getCount(mapKey)) {
            return false;
        }
    }

    if (histogramM.size() != histogramMprime.size()) {
        return false;
    }
    // Worst case: Every entry in M_p and M_prime is unique, and this results in O(d^2) absolute
    // values to check.
    for (const auto& key : histogramMprime.getMap() | std::views::keys) {
        if (histogramMprime.getCount(key) != histogramM.getCount(key)) {
            return false;
        }
    }

    // Handle some edge cases
    if (histogramM.size() == 0) {
        return false;
    } else if (histogramM.size() == 1) {
        if (histogramM.getCountOfZero() > 0) {
            // the 0 matrix: histogram has only a key-value mapping for 0
            return histogramMprime.size() == 1 && histogramMprime.getCountOfZero() > 0;
        }

        // At this point, Mp is just a single value in every entry. We cannot determine a
        // permutation reliably. Fall back to brute force (worst case O(d^7) runtime).

        // TODO: Maybe analyze the omegas somehow
        return bruteForceTestCliffordConjugacy(M, M_prime, omega, M_p, Mprime_p);
    }

    // Sort the keys by the number of coordinates associated with each key.
    const auto sortedKeysM = histogramM.sortedKeys();

    FMapKey firstNonZeroKey(0.0, 0.0);
    for (const FMapKey& key : sortedKeysM) {
        if (key.r() != 0.0) {
            firstNonZeroKey = key;
            break;
        }
    }
    if (firstNonZeroKey.r() == 0.0) {
        // Should never reach this point: histogramM.size() >= 2 here, so there must be nonzero key
        throw std::invalid_argument("Missing nonzero coordinate");
    }

    std::optional<std::pair<MatrixCoordinate, size_t>> nonZeroCoordOpt = std::nullopt;
    {
        const auto& nonZeroEntry = histogramM.getWithN(firstNonZeroKey);
        for (size_t i = 0; i < nonZeroEntry.size(); ++i) {
            const auto& coord = nonZeroEntry.coords[i];
            const auto& n = nonZeroEntry.n[i];
            if (coord(0) != 0 || coord(1) != 0) {
                nonZeroCoordOpt = std::make_optional(std::make_pair(coord, n));
                break;
            }
        }
    }
    if (!nonZeroCoordOpt.has_value()) {
        throw std::invalid_argument("Missing nonzero coords");
    }
    const auto& [nonZeroCoord, n_nonZeroCoord] = nonZeroCoordOpt.value();

    // Find a linearly dependent pair. If this doesn't yield a value, then every nonzero coordinate
    // is on a line.
    auto linIndepCoordResult =
        findLinearIndependentCoord(d, histogramM, sortedKeysM, nonZeroCoord, true);
    if (!linIndepCoordResult.has_value()) {
        // If all points are pairwise linearly dependent, then
        // there exists v such that for any (p,q), (p,q) = c*v for some c
        // By Lemma 10, we would have f_M(v) = omega^k * f_{M'}(u) for some u. Then S(v) = u, and
        // k = [v, (p',q')].
        //
        // Now, for any nonzero (p,q), f_M(p,q) = omega^[(p,q), (p',q')] f_{M'}(S(p,q)).
        // Let (p,q) = c*v for some c.
        // But [(p,q), (p',q')] = [(cv1, cv2), (p', q')]
        //                     = cv1 * q' - p' * cv2
        //                     = c(v1q' - p'v2)
        //                     = c[v, (p',q')]
        //                     = c * k
        // Also, S(p,q) = S(cv) = c*S(v) = cu
        // So it remains to check whether f_M(p,q) = omega^(c*k) * f_{M'}(c*u) as a necessary
        // condition

        // Pick a v with the least frequency
        const auto& key = firstNonZeroKey;
        const auto& vEntry = histogramM.getWithN(key);
        const MatrixCoordinate& v = vEntry.coords.front();

        const size_t n_v = vEntry.n.front();

        // alpha_v = f_M(v)
        const auto alpha = M_p.get(v);
        assert(alpha.real() != 0.0 && alpha.imag() != 0.0);
        // The size of this is at most O(d), because if all nonzero entries have linearly dependent
        // coordinates, then all the coordinates are in a line. A line can have at most O(d) points
        // in Z_d^2.
        const auto& possibleUEntry = histogramMprime.getWithN(key);
        const auto& possibleU = possibleUEntry.coords;
        // TODO: Refactor
        for (size_t i = 0; i < possibleU.size(); i++) {
            const auto& u = possibleU[i];
            const size_t n_u = possibleUEntry.n[i];
            const size_t k = safeMod(n_v - n_u, d);
#ifndef NDEBUG
            // For each beta_u with the same absolute value
            const auto& beta = Mprime_p.get(u);
            // Try to find integer k such that alpha_v = omega^k beta_u
            const double kTest = checkPhase(d, alpha, beta);
            const size_t kRounded = std::llround(kTest);
            assert(k == kRounded);
#endif
            // k is the prospective value of the symplectic product [v, (p',q')]
            // This will test whether f_M(v) = omega^k * f_{M'}(u) and u = S(v) can be satisfied.
            // Worst case O(d) to run this (returning true requires going through all O(d) entries)
            if (!validateNecessaryCondFromLinDepPoints(d, omega, histogramM, M_p, Mprime_p,
                                                       sortedKeysM, v, u, k)) {
                continue;
            }

            // We know k = [v, (p',q')] = v1q' - p'v2
            // So v1q' = k + p'v2, hence q' = v1^{-1}(k + p'v2)
            const size_t v1Inv = modInverse(v(0), d);
            for (size_t pPrime = 0; pPrime < d; pPrime++) {
                const size_t qPrime = safeMod(v1Inv * (k + pPrime * v(1)), d);

                // Recall that f_M(v) = omega^k * f_{M'}(u)
                const auto m0 = v(0);
                const auto m1 = v(1);
                const auto n0 = u(0);
                const auto n1 = u(1);
                assert(isApproxEqual(M_p.get(v), std::pow(omega, k) * Mprime_p.get(u)));

                // The solution to S.m = n where
                //      S = [[x0, x1], [x2,x3]], m = [[m0],[m1]], n = [[n0], [n1]]
                // is x0 = -(m1*x1 - n0)/m0, x2 = -(m1*x3 - n1)/m0, x1 and x3 free

                const auto m0_inv = modInverse(m0, d);
                for (size_t x1 = 0; x1 < d; x1++) {
                    for (size_t x3 = 0; x3 < d; x3++) {
                        const auto x0 = safeMod(safeMod(n0 - m1 * x1, d) * m0_inv, d);
                        const auto x2 = safeMod(safeMod(n1 - m1 * x3, d) * m0_inv, d);

                        if (isSymplecticTransformation(d, x0, x1, x2, x3)) {
                            Sp1ZdMatrix S;
                            S << x0, x1, x2, x3;
                            const Eigen::Vector<long, 2> pPrime_qPrime_vec(pPrime, qPrime);
                            if (test_clifford_conjugate_lemma_10(d, pPrime_qPrime_vec, omega, M,
                                                                 M_p, Mprime_p, S)) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }
    const auto linIndepKey = linIndepCoordResult.value().first;
    const auto& [linIndepCoord, n_linIndepCoord] = linIndepCoordResult.value().second;

    // v and v' are linearly independent
    const auto vKey = firstNonZeroKey;
    const auto& v = nonZeroCoord;
    const auto& n_v = n_nonZeroCoord;
    const auto vPrimeKey = linIndepKey;
    const auto& vPrime = linIndepCoord;
    const auto& n_vPrime = n_linIndepCoord;

    // Choose u and u' so that the frequencies histogramM(|u|) and histogramM(|u'|) are minimized.
    // sortedKeysM is sorted for this in increasing order, so just iterate from start.
    auto uKey = sortedKeysM[0];
    const auto& uEntry = histogramM.getWithN(uKey);
    MatrixCoordinate u = uEntry.coords.front();

    const auto uIndepResult = findLinearIndependentCoord(d, histogramM, sortedKeysM, u, false);
    FMapKey uPrimeKey;
    MatrixCoordinate uPrime;
    if (uIndepResult.has_value()) {
        uPrimeKey = uIndepResult->first;
        uPrime = uIndepResult->second.first;
    } else {
        // Fall back to a known pair of linearly independent coordinates
        uKey = vKey;
        u = v;
        uPrimeKey = vPrimeKey;
        uPrime = vPrime;
    }

    // Get all possible places the permutation S could have sent u and u' to.
    // Then since u and u' are linearly independent, try to recover S from this info.
    const auto& possibleMappingsU = histogramMprime.getWithN(uKey);
    const auto& possibleMappingsUPrime = histogramMprime.getWithN(uPrimeKey);

    // Worst case: The mappings are split and these two nested loops have roughly around
    // (d^2 / 2) * (d^2 / 2) = O(d^4) iterations. But it's possible for most of these iterations to
    // exit early without reaching the O(d^2) check further inside.
    for (size_t uIndex = 0; uIndex < possibleMappingsU.size(); uIndex++) {
        for (size_t uPrimeIndex = 0; uPrimeIndex < possibleMappingsU.size(); uPrimeIndex++) {
            const auto& uMap = possibleMappingsU.coords[uIndex];
            const auto& uPrimeMap = possibleMappingsUPrime.coords[uPrimeIndex];

            // This solves the equation S.[u u'] = [S(u) S(u')] for S, where u, u' are in Z_d^2
            // and linearly independent. The linear independence allows us to deterministically
            // recover S for this particular mapping pair.

            const auto uPrimeDeterminant = safeMod(u(0) * uPrime(1) - uPrime(0) * u(1), d);
            const auto uPrimeDetInverse = modInverse(uPrimeDeterminant, d);
            // uUprimeDeterminantInv := (u[1]*uPrime[2] - uPrime[1]*u[2]) &^(-1) mod d

            // Scoord[1] := (uMap[1] * uPrime[2] - u[2] * uPrimeMap[1])*uUprimeDeterminantInv mod d;
            const auto x0 =
                safeMod(uPrimeDetInverse * (uMap(0) * uPrime(1) - u(1) * uPrimeMap(0)), d);
            // Scoord[2] := (u[1] * uPrimeMap[1] - uMap[1] * uPrime[1])*uUprimeDeterminantInv mod d;
            const auto x1 =
                safeMod(uPrimeDetInverse * (u(0) * uPrimeMap(0) - uMap(0) * uPrime(0)), d);
            // Scoord[3] := (uMap[2] * uPrime[2] - u[2] * uPrimeMap[2])*uUprimeDeterminantInv mod d;
            const auto x2 =
                safeMod(uPrimeDetInverse * (uMap(1) * uPrime(1) - u(1) * uPrimeMap(1)), d);
            // Scoord[4] := (u[1] * uPrimeMap[2] - uMap[2] * uPrime[1])*uUprimeDeterminantInv mod d;
            const auto x3 =
                safeMod(uPrimeDetInverse * (u(0) * uPrimeMap(1) - uMap(1) * uPrime(0)), d);

            if (isSymplecticTransformation(d, x0, x1, x2, x3)) {
                // Let alpha_v = f_M(v) and beta_v = f_{M'}(v)

                const size_t n_alphaV = n_v;
                const size_t n_alphaVPrime = n_vPrime;

                const auto& betaVCoord = applyTransformation(d, v, x0, x1, x2, x3);
                const auto& betaV = Mprime_p.get(betaVCoord);
                const auto& betaVPrimeCoord = applyTransformation(d, vPrime, x0, x1, x2, x3);
                const auto& betaVPrime = Mprime_p.get(betaVPrimeCoord);

                const auto betaVEntry = histogramMprime.getPauliCoeffIfInSameBin(betaV, vKey);
                if (!betaVEntry) {
                    continue;
                }
                const size_t n_betaV = betaVEntry->n();
                // Get integer k such that alpha_v = omega^k beta_v
                const size_t k = safeMod(n_alphaV - n_betaV, d);
#ifndef NDEBUG
                const auto& alphaV = M_p.get(v);
                const double kTest = checkPhase(d, alphaV, betaV);
                const size_t kRounded = std::lround(kTest);
                assert(k == kRounded);
#endif
                const auto betaVPrimeEntry =
                    histogramMprime.getPauliCoeffIfInSameBin(betaVPrime, vPrimeKey);
                if (!betaVPrimeEntry) {
                    continue;
                }
                const size_t n_betaVPrime = betaVPrimeEntry->n();
                // Get integer k' such that alpha'_v = omega^k' beta'_v
                const size_t kPrime = safeMod(n_alphaVPrime - n_betaVPrime, d);
#ifndef NDEBUG
                const auto& alphaVPrime = M_p.get(vPrime);
                const double kPrimeTest = checkPhase(d, alphaVPrime, betaVPrime);
                const size_t kPrimeRounded = std::llround(kPrimeTest);
                assert(kPrimeRounded == kPrime);
#endif
                // Via Lemma 10, we must have
                //      alpha_v = omega^k beta_v = omega^[(p,q),(p',q')] beta_v
                // and this gives us a linear (congruence) equation k = [v,(p',q')].
                // Similarly, k' = [v',(p',q')]. As k, k', v and v' are known, this is a system of 2
                // equations in 2 unknowns p' and q'

                // inverseToUse := (v[1] * vPrime[2] - v[2]*vPrime[1])&^(-1) mod d;
                // pPrimeFromThis := (possibleK*vPrime[1] - possibleKPrime*v[1])*inverseToUse mod d;
                // qPrimeFromThis := (possibleK*vPrime[2] - possibleKPrime*v[2])*inverseToUse mod d;
                const auto inverseToUse =
                    modInverse(safeMod(v(0) * vPrime(1) - v(1) * vPrime(0), d), d);
                const auto pPrime = safeMod(inverseToUse * (k * vPrime(0) - kPrime * v(0)), d);
                const auto qPrime = safeMod(inverseToUse * (k * vPrime(1) - kPrime * v(1)), d);

                Sp1ZdMatrix S;
                S << x0, x1, x2, x3;
                // Verify this works for all p, q
                // Check is worst-case O(d^2), but it will exit quickly if it finds a bad value
                if (const Eigen::Vector<long, 2> pPrime_qPrime_vec(pPrime, qPrime);
                    test_clifford_conjugate_lemma_10(d, pPrime_qPrime_vec, omega, M, M_p, Mprime_p,
                                                     S)) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool isCliffordConjugateGeneralized(const std::size_t d, const std::size_t n,
                                    const Eigen::Ref<const Eigen::MatrixXcd>& M,
                                    const Eigen::Ref<const Eigen::MatrixXcd>& M_prime,
                                    const std::optional<unsigned long>& shuffleSeed) {
    constexpr double mapAbsValPrecision = 1e-5;
    constexpr double mapXPrecision = 1e-4;

    const size_t D = computeIntegralPower(d, n);

    if (M.rows() != D || M.cols() != D) {
        return false;
    }
    if (M_prime.rows() != D || M_prime.cols() != D) {
        return false;
    }

    // Since trace is cyclic, if M = CM'C^*, then tr(M) = tr(CM'C^*) = tr(M'C^*C) = tr(M').
    if (!isApproxEqual(M.trace(), M_prime.trace())) {
        return false;
    }

    // earlier assert guarantees M.rows() == M.cols()
    const auto inv2 = modInverse(2, d);
    const auto omega = std::exp(std::complex<double>(0, 2 * pi / d));

    FMap MMap(d, n, mapAbsValPrecision, mapXPrecision);
    MpMatrixType M_p(2 * n, d);
    for (auto tuple : M_p.indexIterator()) {
        const auto val = f_multiqudit(M, tuple, d, inv2, omega);
        M_p.get(tuple) = val;
        MMap.insertEntryNTuple(std::move(tuple), val);
    }

    FMap MprimeMap(d, n, mapAbsValPrecision, mapXPrecision);
    MpMatrixType Mprime_p(2 * n, d);
    for (auto tuple : Mprime_p.indexIterator()) {
        const auto val = f_multiqudit(M_prime, tuple, d, inv2, omega);
        Mprime_p.get(tuple) = val;
        const auto mapKey = MprimeMap.insertEntryNTuple(std::move(tuple), val);
        // Early histogram check
        if (MprimeMap.getCount(mapKey) > MMap.getCount(mapKey)) {
            return false;
        }
    }

    if (MMap.size() != MprimeMap.size()) {
        return false;
    }
    // Worst case: Every entry in M_p and M_prime is unique, and this results in O(d^2) absolute
    // values to check.
    for (const auto& key : MprimeMap.getMap() | std::views::keys) {
        if (MprimeMap.getCount(key) != MMap.getCount(key)) {
            return false;
        }
    }

    return findSymplecticMatrix(d, n, omega, M, M_p, Mprime_p, MMap, MprimeMap, shuffleSeed)
        .has_value();
}

} // namespace cliffconjtest
