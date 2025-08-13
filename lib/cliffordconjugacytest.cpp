#include "cliffordconjugacytest.hpp"

#include <complex>
#include <iostream>
#include "internal/absvalmap.h"
#include "internal/bruteforcetest.h"
#include "internal/util.h"

namespace cliffconjtest {

bool isSymplecticTransformation(size_t d, size_t s1, size_t s2, size_t s3, size_t s4) {
    // Symplectic if determinant is 1 (for 2x2 matrices over Z_d)
    return safeMod(s1 * s4 - s2 * s3, d) == 1;
}

std::optional<std::pair<double, std::pair<size_t, size_t>>>
findLinearIndependentCoord(const size_t d, const AbsValMap& histogramM,
                           std::vector<double> sortedKeysM,
                           const std::pair<size_t, size_t> nonZeroCoord, bool skipZeroKeys) {

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
        if (skipZeroKeys && key == 0.0) {
            continue;
        }

        const auto coords = histogramM.get(key);
        for (const auto& coord : coords) {
            // theDeterminant := (nonZeroCoord[1] * coord[2] - coord[1] * nonZeroCoord[2]) mod d;
            const auto determinant =
                safeMod(nonZeroCoord.first * coord.second - coord.first * nonZeroCoord.second, d);
            if (determinant != 0) {
                return std::make_optional(std::make_pair(key, coord));
            }
        }
    }
    return std::nullopt;
}

std::pair<size_t, size_t> applyTransformation(size_t d, const std::pair<size_t, size_t>& target,
                                              size_t x0, size_t x1, size_t x2, size_t x3) {
    return std::make_pair((x0 * target.first + x1 * target.second) % d,
                          (x2 * target.first + x3 * target.second) % d);
}

bool validate_linear_dependent_points(const Eigen::Index d, const std::complex<double> omega,
                                      const AbsValMap& histogramM, const Eigen::MatrixXcd& M_p,
                                      const Eigen::MatrixXcd& Mprime_p, const std::vector<double>& sortedKeysM,
                                      const std::pair<size_t, size_t>& v,
                                      const std::pair<unsigned long, unsigned long>& u,
                                      const size_t k) {
    // Loop over all (p,q) such that f_M(p,q) is nonzero.
    // To do this, loop over all sorted keys and then access the map in each loop.
    // This will run at most d - 1 times, because there would be only 1 line in the M_p matrix.
    // All nonzero points in M_p are pairwise linearly dependent, so tbere's only 1 line of at most
    // O(d) length
    for (const auto& keyInner : sortedKeysM) {
        if (keyInner == 0.0) {
            continue;
        }
        const auto& possibleUInner = histogramM.get(keyInner);
        for (const auto& [p, q] : possibleUInner) {
            if (p == 0 && q == 0) {
                continue;
            }
            assert(safeMod(p * v.second - v.first * q, d) == 0);

            // Calculate c where c*v = (p,q), so c = pv1^(-1) and c = qv2^(-1)
            const size_t c = safeMod(p != 0 ? p * fastPowerMod(v.first, d - 2, d)
                                            : q * fastPowerMod(v.second, d - 2, d),
                                     d);

            assert(safeMod(c * v.first, d) == p && safeMod(c * v.second, d) == q);

            const auto& alphaInner = M_p(p, q);
            // Then S(p,q) = S(c*v) = c*S(v) = c*u
            const auto& betaInner = Mprime_p(safeMod(c * u.first, d), safeMod(c * u.second, d));
            const auto omegaPow = std::pow(omega, safeMod(c * k, d));
            const auto product = omegaPow * betaInner;
            if (!isApproxEqual(alphaInner, product)) {
                return false;
            }
        }
    }
    return true;
}
bool isCliffordConjugate(const Eigen::Ref<const Eigen::MatrixXcd>& M,
                         const Eigen::Ref<const Eigen::MatrixXcd>& M_prime) {
    if (M.rows() != M.cols() || M_prime.rows() != M_prime.cols() || M_prime.cols() != M.cols()) {
        return false;
    }

    // Since trace is cyclic, if M = CM'C^*, then tr(M) = tr(CM'C^*) = tr(M'C^*C) = tr(M').
    if (!isApproxEqual(M.trace(), M_prime.trace())) {
        return false;
    }

    // earlier assert guarantees M.rows() == M.cols()
    const auto d = M.rows();
    const auto inv2 = fastPowerMod(2, d - 2, d);
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
    AbsValMap histogramM(5);
    Eigen::MatrixXcd M_p = Eigen::MatrixXcd::Zero(d, d);
    bool allEqual = true;
    for (size_t p = 0; p < d; p++) {
        for (size_t q = 0; q < d; q++) {
            if (allEqual && !isApproxEqual(M(p, q), M_prime(p, q))) {
                allEqual = false;
            }
            const auto value = f(M, p, q, inv2, omega);
            M_p(p, q) = value;
            histogramM.insertEntry(p, q, value);
        }
    }

    if (allEqual) {
        // M = M', so true since C = I works
        return true;
    }

    // Now check whether the histogram for M' is equal. If not, then the two
    AbsValMap histogramMprime(5, histogramM.size());
    Eigen::MatrixXcd Mprime_p = Eigen::MatrixXcd::Zero(d, d);
    for (size_t p = 0; p < d; p++) {
        for (size_t q = 0; q < d; q++) {
            Mprime_p(p, q) = f(M_prime, p, q, inv2, omega);

            const auto key = Mprime_p(p, q);
            const auto mapKey = histogramMprime.insertEntry(p, q, key);
            // Early histogram check
            if (histogramMprime.getCount(mapKey) > histogramM.getCount(mapKey)) {
                return false;
            }
        }
    }

    const bool atLeastTwoNonZero = histogramM.getCount(0.0) <= d * d - 2;
    // Handle some edge cases
    if (histogramM.size() == 0) {
        return false;
    } else if (histogramM.size() == 1) {
        if (histogramM.getCount(0.0) > 0) {
            // the 0 matrix: histogram has only a key-value mapping for 0
            return histogramMprime.size() == 1 && histogramMprime.getCount(0.0) > 0;
        }

        // At this point, Mp is just a single value in every entry. We cannot determine a
        // permutation reliably. Fall back to brute force (O(d^7) runtime).

        // TODO: Maybe analyze the omegas somehow
        return bruteForceTestCliffordConjugacy(M, M_prime, omega, M_p, Mprime_p);
    } else if (histogramM.size() == 2) {
        // If there is exactly 1 nonzero entry in Mp
        const double nonZeroKey = histogramM.getNonZeroKey().value_or(0.0);
        if (nonZeroKey != 0.0) {
            const auto& nonZeroLocationsInM = histogramM.get(nonZeroKey);
            const auto& nonZeroLocationsInMprime = histogramMprime.get(nonZeroKey);
            // If there is only a single non-zero value (i.e. M is one of the Pauli basis elements
            // W(p,q))
            if (nonZeroLocationsInM.size() == 1 && nonZeroLocationsInMprime.size() == 1) {
                const auto [m0, m1] = nonZeroLocationsInM.front();
                // Histogram for M' should be the same.
                const auto [n0, n1] = nonZeroLocationsInMprime.front();
                // The solution to S.m = n where
                //      S = [[x0, x1], [x2,x3]], m = [[m0],[m1]], n = [[n0], [n1]]
                // is x0 = -(m1*x1 - n0)/m0, x2 = -(m1*x3 - n1)/m0, x1 and x3 free

                const auto m0_inv = fastPowerMod(m0, d - 2, d);

                for (size_t x1 = 0; x1 < d; x1++) {
                    for (size_t x3 = 0; x3 < d; x3++) {
                        const auto x0 = safeMod(safeMod(n0 - m1 * x1, d) * m0_inv, d);
                        const auto x2 = safeMod(safeMod(n1 - m1 * x3, d) * m0_inv, d);

                        if (isSymplecticTransformation(d, x0, x1, x2, x3)) {
                            Eigen::Matrix2i S;
                            S << x0, x1, x2, x3;

                            // TODO: Optimize this somehow
                            for (size_t pPrime = 0; pPrime < d; pPrime++) {
                                for (size_t qPrime = 0; qPrime < d; qPrime++) {
                                    if (test_clifford_conjugate_lemma_10(pPrime, qPrime, omega, M,
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
        }
    }

    if (!atLeastTwoNonZero) {
        throw std::invalid_argument("TODO: Handle case where there is only 1 nonzero ");
    }

    // Sort the keys by the number of coordinates associated with each key.
    auto sortedKeysM = histogramM.sortedKeys();
    // auto sortedKeysMprime = histogramMprime.sortedKeys();

    auto it_num = std::ranges::find_if(sortedKeysM,
                                       [](const double key) { return !isApproxEqual(key, 0.0); });
    if (it_num == sortedKeysM.end()) {
        throw std::invalid_argument("TODO: Missing nonzero coordinate");
    }

    // Find a non-zero key so that the key itself is nonzero and it has a nonzero coordinate.
    const auto nonZeroKey = *it_num;
    std::optional<std::pair<size_t, size_t>> nonZeroCoordOpt = std::nullopt;
    {
        const auto coords = histogramM.get(nonZeroKey);
        for (const auto& coord : coords) {
            if (coord.first != 0 || coord.second != 0) {
                nonZeroCoordOpt = std::make_optional(coord);
                break;
            }
        }
    }
    if (!nonZeroCoordOpt.has_value()) {
        throw std::invalid_argument("Missing nonzero coords");
    }
    const auto nonZeroCoord = nonZeroCoordOpt.value();

    // Find a linearly dependent pair. If this doesn't yield a value, then every nonzero coordinate
    // is on a line.
    auto linIndepCoordOpt =
        findLinearIndependentCoord(d, histogramM, sortedKeysM, nonZeroCoord, true);
    if (!linIndepCoordOpt.has_value()) {
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
        // So it remains to check whether f_M(p,q) = omega^(c*k) * f_{M'}(c*u)

        const auto& key = sortedKeysM[0];
        std::pair<size_t, size_t> v = histogramM.get(key).front();

        // alpha_v = f_M(v)
        const auto alpha = M_p(v.first, v.second);
        assert(alpha.real() != 0.0 && alpha.imag() != 0.0);
        // The size of this is at most O(d), because if all nonzero entries have linearly dependent
        // coordinates, then all the coordinates are in a line. A line can have at most O(d) points
        // in Z_d^2.
        const auto& possibleU = histogramMprime.get(key);

        // TODO: Refactor
        for (const auto& u : possibleU) {
            // For each beta_u with the same absolute value
            const auto beta = Mprime_p(u.first, u.second);
            // Try to find integer k such that alpha_u = omega^k beta_v
            const double kTest = checkPhase(d, alpha, beta);
            const size_t k = std::round(kTest);
            if (std::abs(kTest - k) > 1e-5) {
                continue;
            }

            // k is the prospective value of the symplectic product [v, (p',q')]
            // O(d) to run this
            if (!validate_linear_dependent_points(d, omega, histogramM, M_p, Mprime_p, sortedKeysM,
                                                  v, u, k)) {
                continue;
            }

            // We know k = [v, (p',q')] = v1q' - p'v2
            // So v1q' = k + p'v2, hence q' = v1^{-1}(k + p'v2)
            const size_t v1Inv = fastPowerMod(v.first, d - 2, d);
            for (size_t pPrime = 0; pPrime < d; pPrime++) {
                const size_t qPrime = safeMod(v1Inv * (k + pPrime * v.second), d);

                // Recall that f_M(v) = omega^k * f_{M'}(u)
                const auto [m0, m1] = v;
                const auto [n0, n1] = u;
                assert(isApproxEqual(M_p(m0, m1), std::pow(omega, k) * Mprime_p(n0, n1)));

                // The solution to S.m = n where
                //      S = [[x0, x1], [x2,x3]], m = [[m0],[m1]], n = [[n0], [n1]]
                // is x0 = -(m1*x1 - n0)/m0, x2 = -(m1*x3 - n1)/m0, x1 and x3 free

                const auto m0_inv = fastPowerMod(m0, d - 2, d);
                for (size_t x1 = 0; x1 < d; x1++) {
                    for (size_t x3 = 0; x3 < d; x3++) {
                        const auto x0 = safeMod(safeMod(n0 - m1 * x1, d) * m0_inv, d);
                        const auto x2 = safeMod(safeMod(n1 - m1 * x3, d) * m0_inv, d);

                        if (isSymplecticTransformation(d, x0, x1, x2, x3)) {
                            Eigen::Matrix2i S;
                            S << x0, x1, x2, x3;
                            if (test_clifford_conjugate_lemma_10(pPrime, qPrime, omega, M, M_p,
                                                                 Mprime_p, S)) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }
    const auto linIndepKey = linIndepCoordOpt.value().first;
    const auto linIndepCoord = linIndepCoordOpt.value().second;

    // v and v' are linearly independent
    const auto vKey = nonZeroKey;
    const auto v = nonZeroCoord;
    const auto vPrimeKey = linIndepKey;
    const auto vPrime = linIndepCoord;

    // Choose u and u' so that the frequencies histogramM(|u|) and histogramM(|u'|) are minimized.
    auto uKey = sortedKeysM[0];
    std::pair<size_t, size_t> u = histogramM.get(uKey).front();
    auto uIndep = findLinearIndependentCoord(d, histogramM, sortedKeysM, u, false);
    double uPrimeKey;
    std::pair<size_t, size_t> uPrime;
    if (uIndep.has_value()) {
        uPrimeKey = uIndep->first;
        uPrime = uIndep->second;
    } else {
        uKey = vKey;
        u = v;
        uPrimeKey = vPrimeKey;
        uPrime = vPrime;
    }

    // Get all possible places the permutation S could have sent u and u' to.
    // Then since u and u' are linearly independent, try to recover S from this info.
    const auto possibleMappingsU = histogramMprime.get(uKey);
    const auto possibleMappingsUPrime = histogramMprime.get(uPrimeKey);

    // Worst case: The mappings are split and these two nested loops have roughly around
    // (d^2 / 2) * (d^2 / 2) = O(d^4) iterations. But it's possible for most of these iterations to
    // exit early without reaching the O(d^2) check further inside.
    for (const auto& uMap : possibleMappingsU) {
        for (const auto& uPrimeMap : possibleMappingsUPrime) {
            // This solves the equation S.[u u'] = [S(u) S(u')] for S, where u, u' are in Z_d^2
            // and linearly independent. The linear independence allows us to deterministically
            // recover S for this particular mapping pair.

            const auto uPrimeDeterminant =
                safeMod(u.first * uPrime.second - uPrime.first * u.second, d);
            const auto uPrimeDetInverse = fastPowerMod(uPrimeDeterminant, d - 2, d);
            // uUprimeDeterminantInv := (u[1]*uPrime[2] - uPrime[1]*u[2]) &^(-1) mod d

            // Scoord[1] := (uMap[1] * uPrime[2] - u[2] * uPrimeMap[1])*uUprimeDeterminantInv mod d;
            const auto x0 = safeMod(
                uPrimeDetInverse * (uMap.first * uPrime.second - u.second * uPrimeMap.first), d);
            // Scoord[2] := (u[1] * uPrimeMap[1] - uMap[1] * uPrime[1])*uUprimeDeterminantInv mod d;
            const auto x1 = safeMod(
                uPrimeDetInverse * (u.first * uPrimeMap.first - uMap.first * uPrime.first), d);
            // Scoord[3] := (uMap[2] * uPrime[2] - u[2] * uPrimeMap[2])*uUprimeDeterminantInv mod d;
            const auto x2 = safeMod(
                uPrimeDetInverse * (uMap.second * uPrime.second - u.second * uPrimeMap.second), d);
            // Scoord[4] := (u[1] * uPrimeMap[2] - uMap[2] * uPrime[1])*uUprimeDeterminantInv mod d;
            const auto x3 = safeMod(
                uPrimeDetInverse * (u.first * uPrimeMap.second - uMap.second * uPrime.first), d);

            if (isSymplecticTransformation(d, x0, x1, x2, x3)) {
                Eigen::Matrix2i S;
                S << x0, x1, x2, x3;

                // Let alpha_v = f_M(v) and beta_v = f_{M'}(v)
                const auto alphaV = M_p(v.first, v.second);
                const auto alphaVPrime = M_p(vPrime.first, vPrime.second);

                const auto betaVCoord = applyTransformation(d, v, x0, x1, x2, x3);
                const auto betaV = Mprime_p(betaVCoord.first, betaVCoord.second);
                const auto betaVPrimeCoord = applyTransformation(d, vPrime, x0, x1, x2, x3);
                const auto betaVPrime = Mprime_p(betaVPrimeCoord.first, betaVPrimeCoord.second);

                // Try to find integer k such that alpha_v = omega^k beta_v
                const double kTest = checkPhase(d, alphaV, betaV);
                const double k = std::round(kTest);
                if (std::abs(kTest - k) > 1e-5) {
                    // too far from an integer
                    continue;
                }

                // Try to find integer k' such that alpha'_v = omega^k' beta'_v
                const double kPrimeTest = checkPhase(d, alphaVPrime, betaVPrime);
                const double kPrime = std::round(kPrimeTest);
                if (std::abs(kPrimeTest - kPrime) > 1e-5) {
                    // too far from an integer
                    continue;
                }

                // Via Lemma 10, we must have
                //      alpha_v = omega^k beta_v = omega^[(p,q),(p',q')] beta_v
                // and this gives us a linear (congruence) equation k = [v,(p',q')].
                // Similarly, k' = [v',(p',q')]. As v and v' are known, this is a system of 2
                // equations in 2 unknowns p' and q'

                // inverseToUse := (v[1] * vPrime[2] - v[2]*vPrime[1])&^(-1) mod d;
                // pPrimeFromThis := (possibleK*vPrime[1] - possibleKPrime*v[1])*inverseToUse mod d;
                // qPrimeFromThis := (possibleK*vPrime[2] - possibleKPrime*v[2])*inverseToUse mod d;
                const auto inverseToUse = fastPowerMod(
                    safeMod(v.first * vPrime.second - v.second * vPrime.first, d), d - 2, d);
                const auto pPrime =
                    safeMod(inverseToUse * (k * vPrime.first - kPrime * v.first), d);
                const auto qPrime =
                    safeMod(inverseToUse * (k * vPrime.second - kPrime * v.second), d);

                // Verify this works for all p, q
                // This check is O(d^2), but it will exit quickly if it finds a bad value.
                if (test_clifford_conjugate_lemma_10(pPrime, qPrime, omega, M, M_p, Mprime_p, S)) {
                    return true;
                }
            }
        }
    }

    return false;
}

} // namespace cliffconjtest
