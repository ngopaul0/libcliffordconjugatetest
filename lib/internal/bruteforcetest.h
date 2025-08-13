#ifndef BRUTEFORCETEST_H
#define BRUTEFORCETEST_H
#include <Eigen/Dense>
#include <complex>
#include <vector>

#include "conjtestlemma10.h"
#include "symplecticiterator.h"

namespace cliffconjtest {

class Sp1ZdGates {
    std::vector<Eigen::Matrix2i> gates;

  public:
    explicit Sp1ZdGates(size_t d) {
        Sp1ZdMatrixIterator iterator(d);
        gates.reserve(d * d * d - d);
        for (const auto& g : iterator) {
            gates.push_back(g);
        }
    }

    [[nodiscard]] const std::vector<Eigen::Matrix2i>& getGates() const { return gates; }
};

/**
 * Computes the M_p matrix from the Clifford-conjugate algorithm.
 *
 * @param M dxd matrix
 * @param omega dth root of unity
 * @param inv_2 2^{-1} mod d
 * @return The matrix M_p such that M_p(i,j) = f_M(i,j)
 */
inline Eigen::MatrixXcd createMpMatrix(const Eigen::Ref<const Eigen::MatrixXcd>& M,
                                       const std::complex<double>& omega, const size_t inv_2) {
    const size_t d = M.rows();
    if (d != M.cols()) {
        throw std::invalid_argument("non-square matrix");
    }

    Eigen::MatrixXcd M_p = Eigen::MatrixXcd::Zero(d, d);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            M_p(i, j) = f(M, i, j, inv_2, omega);
        }
    }
    return M_p;
}

using Lemma10Info = std::pair<Eigen::Matrix2i, std::pair<size_t, size_t>>;

template <bool ReturnOptional>
using BruteForceReturnType = std::conditional_t<ReturnOptional, std::optional<Lemma10Info>, bool>;

/**
 * Runs a brute-force Clifford-conjugate test based on Lemma 10
 *
 * @tparam IsReturningInfo Whether to return S and (p',q')
 * @param M dxd matrix, d an odd prime
 * @param Mprime  dxd matrix, d an odd prime
 * @param omega dth root of unity
 * @param M_p precomputed values of f_M(p,q)
 * @param Mprime_p precomputed values of f_{M'}(p,q)
 * @param gates An optional list of all symplectic transformations. If not provided, will iterate
 * through all gates instead without storing an entire list in memory.
 * @return Whether M and M' are Clifford-conjugate, i.e. M = CM'C^* for some CLifford C in C_2.
 */
template <bool IsReturningInfo = false>
BruteForceReturnType<IsReturningInfo> bruteForceTestCliffordConjugacy(
    const Eigen::Ref<const Eigen::MatrixXcd>& M, const Eigen::Ref<const Eigen::MatrixXcd>& Mprime,
    const std::complex<double>& omega, const Eigen::Ref<const Eigen::MatrixXcd> M_p,
    const Eigen::Ref<const Eigen::MatrixXcd>& Mprime_p,
    std::optional<std::reference_wrapper<const Sp1ZdGates>> gates = std::nullopt) {

    const size_t d = M.rows();
    if (d != M.cols() || Mprime.rows() != Mprime.cols() || Mprime.rows() != d) {
        if constexpr (IsReturningInfo) {
            return std::nullopt;
        } else {
            return false;
        }
    }

    if (gates.has_value()) {
        for (size_t pPrime = 0; pPrime < d; pPrime++) {
            for (size_t qPrime = 0; qPrime < d; qPrime++) {
                for (const auto& gate : gates->get().getGates()) {
                    if (test_clifford_conjugate_lemma_10(pPrime, qPrime, omega, M, M_p, Mprime_p,
                                                         gate)) {
                        if constexpr (IsReturningInfo) {
                            return std::make_optional(
                                std::make_pair(gate, std::make_pair(pPrime, qPrime)));
                        } else {
                            return true;
                        }
                    }
                }
            }
        }
    } else {
        for (size_t pPrime = 0; pPrime < d; pPrime++) {
            for (size_t qPrime = 0; qPrime < d; qPrime++) {
                // Use an iterator to avoid storing all the matrices in memory. The iterator
                // is efficient as it's just multiplying 2x2 matrices (constant-time).
                auto gateIterator = Sp1ZdMatrixIterator(d);
                for (const auto& gate : gateIterator) {
                    if (test_clifford_conjugate_lemma_10(pPrime, qPrime, omega, M, M_p, Mprime_p,
                                                         gate)) {
                        if constexpr (IsReturningInfo) {
                            return std::make_optional(
                                std::make_pair(gate, std::make_pair(pPrime, qPrime)));
                        } else {
                            return true;
                        }
                    }
                }
            }
        }
    }

    if constexpr (IsReturningInfo) {
        return std::nullopt;
    } else {
        return false;
    }
}

} // namespace cliffconjtest

#endif // BRUTEFORCETEST_H
