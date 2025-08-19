#ifndef CLIFFORDCONJUGATETEST_LIBRARY_H
#define CLIFFORDCONJUGATETEST_LIBRARY_H

#include <Eigen/Dense>

namespace cliffconjtest {

/**
 * Given two square dxd matrices, where d is an odd prime, checks whether M and M_prime are
 * Clifford-conjugate, i.e. whether there exists a 1-qudit Clifford gate C in C_2 (normaliser of the
 * group of Pauli gates C_1) such that M = CM'C^*.
 *
 * The algorithm used is based on Lemma 10 from [1], which states
 *
 *     Let M, M' in M_d(C). Then M is Clifford-conjugate to M' if and only if there exists
 *     (S', (p',q')) in Sp(1, Z_d) ⋉ Z_d^2 such that for all p,q in Z_d,
 *
 *          f_M(p,q) = omega^[(p,q),(p',q')]f_{M'}(S(p,q))
 *
 *
 * [1] Nadish de Silva, Oscar Lautsch, "The Clifford hierarchy for one qubit or qudit",
 *     arXiv:2501.07939v1 [quant-ph], 2025. https://arxiv.org/abs/2501.07939v1
 *
 * TODO: Add documentation about runtime
 *
 * @return whether M and M_prime are Clifford-conjugate
 */
bool isCliffordConjugate(const Eigen::Ref<const Eigen::MatrixXcd>& M,
                         const Eigen::Ref<const Eigen::MatrixXcd>& M_prime);

bool isCliffordConjugateGeneralized(std::size_t d, std::size_t n,
                                    const Eigen::Ref<const Eigen::MatrixXcd>& M,
                                    const Eigen::Ref<const Eigen::MatrixXcd>& M_prime);

} // namespace cliffconjtest

#endif // CLIFFORDCONJUGATETEST_LIBRARY_H
