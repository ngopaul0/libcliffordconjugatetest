#ifndef CONJTESTLEMMA10_H
#define CONJTESTLEMMA10_H

#include <Eigen/Dense>
#include <iostream>

#include "multidimarray.h"
#include "util.h"

namespace cliffconjtest {

/**
 * Checks whether f_M(p,q) = omega^[p, q, pPrime, qPrime) * f_{M'}(S(p,q)) from Lemma 10 by looping
 * through all p,q
 *
 * @param pPrime The parameter to check
 * @param qPrime The parameter to check
 * @param omega dth root of unity
 * @param M The dxd matrix M, where d is an odd prime
 * @param M_p Precomputed values for f_{M}(p,q)
 * @param Mprime_p Precomputed values for f_{M'}(p,q)
 * @param symplectic_transform S
 * @return Whether Lemma 10 is satisfied.
 */
inline bool
test_clifford_conjugate_lemma_10(size_t pPrime, size_t qPrime, const std::complex<double>& omega,
                                 const Eigen::Ref<const Eigen::MatrixXcd>& M,
                                 const MultiDimensionalArray<std::complex<double>, false>& M_p,
                                 const MultiDimensionalArray<std::complex<double>, false>& Mprime_p,
                                 const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& symplectic_transform) {
    const size_t d = M.rows();
    if (d != M.cols()) {
        return false;
    }
    if (M_p.dimensionPerCoordinate() != d || Mprime_p.dimensionPerCoordinate() != d) {
        return false;
    }

    for (const auto& v : M_p.indexIterator()) {
        // it's likely the compiler will optimize this due to Eigen's expression templates
        Eigen::Vector<long, Eigen::Dynamic> vPrime = modMatrix(symplectic_transform * v, d);
        const auto& fM = M_p.get(v);
        const auto& fMPrime = Mprime_p.get(vPrime);
        const auto p = v[0];
        const auto q = v[1];
        const auto omegaTerm = std::pow(omega, symplecticProduct(d, p, q, pPrime, qPrime));

        if (!isApproxEqual(fM, omegaTerm * fMPrime)) {
            return false;
        }

    }

    return true;
}

} // namespace cliffconjtest
#endif // CONJTESTLEMMA10_H
