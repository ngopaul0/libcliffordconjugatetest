#ifndef BRUTEFORCETEST_H
#define BRUTEFORCETEST_H
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <variant>

#include "conjtestlemma10.h"
#include "multidimarray.h"
#include "symplecticiterator.h"

namespace cliffconjtest {

class Sp1ZdGates {
    std::vector<Sp1ZdMatrix> gates;

  public:
    explicit Sp1ZdGates(size_t d) {
        Sp1ZdMatrixRange iterator(d);
        gates.reserve(d * d * d - d);
        for (const auto& g : iterator) {
            gates.push_back(g);
        }
    }

    [[nodiscard]] const std::vector<Sp1ZdMatrix>& getGates() const { return gates; }
};

/**
 * Computes the M_p matrix from the Clifford-conjugate algorithm.
 *
 * @param M dxd matrix
 * @param omega dth root of unity
 * @param inv_2 2^{-1} mod d
 * @return The matrix M_p such that M_p(i,j) = f_M(i,j)
 */
inline MpMatrixType createMpMatrix(const Eigen::Ref<const Eigen::MatrixXcd>& M,
                                       const std::complex<double>& omega, const size_t inv_2) {
    const size_t d = M.rows();
    if (d != M.cols()) {
        throw std::invalid_argument("non-square matrix");
    }

    MpMatrixType M_p(2, d);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            M_p({i, j}) = f(M, i, j, inv_2, omega);
        }
    }
    return M_p;
}

using Lemma10Info = std::pair<Sp1ZdMatrix, std::pair<size_t, size_t>>;

template <bool ReturnOptional>
using BruteForceReturnType = std::conditional_t<ReturnOptional, std::optional<Lemma10Info>, bool>;

template <bool IsReturningInfo>
constexpr BruteForceReturnType<IsReturningInfo> notFoundValue() {
    if constexpr (IsReturningInfo) {
        return std::nullopt;
    } else {
        return false;
    }
}

template <bool IsReturningInfo>
constexpr bool isFound(const BruteForceReturnType<IsReturningInfo>& value) {
    if constexpr (IsReturningInfo) {
        return value.has_value();
    } else {
        return value == true;
    }
}

template <bool IsReturningInfo>
constexpr BruteForceReturnType<IsReturningInfo>
createValueFromFound(const Sp1ZdMatrix& S, const size_t pPrime, const size_t qPrime) {
    if constexpr (IsReturningInfo) {
        return std::make_optional(std::make_pair(S, std::make_pair(pPrime, qPrime)));
    } else {
        return true;
    }
}

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
    const std::complex<double>& omega, const MultiDimensionalArray<std::complex<double>, false>& M_p,
    const MultiDimensionalArray<std::complex<double>, false>& Mprime_p,
    std::optional<std::reference_wrapper<const Sp1ZdGates>> gates = std::nullopt) {

    const size_t d = M.rows();
    if (d != M.cols() || Mprime.rows() != Mprime.cols() || Mprime.rows() != d) {
        return notFoundValue<IsReturningInfo>();
    }

    using IterableType = std::variant<const std::vector<Sp1ZdMatrix>*, const Sp1ZdMatrixRange*>;
    IterableType iterable;
    // Optional to manage the lifetime of the dynamically created iterator (like dynamic stack
    // dispatch)
    std::optional<Sp1ZdMatrixRange> sp1ZdIteratorOpt;
    if (gates.has_value()) {
        iterable = &gates->get().getGates();
    } else {
        sp1ZdIteratorOpt.emplace(d);
        iterable = &sp1ZdIteratorOpt.value();
    }

    // Variables for the parallel region
    std::optional<BruteForceReturnType<IsReturningInfo>> result;
    #pragma omp parallel for collapse(2) shared(result) schedule(dynamic)
    for (long pPrime = 0; pPrime < d; pPrime++) {
        for (long qPrime = 0; qPrime < d; qPrime++) {
            if (result.has_value()) {
                continue;
            }

            std::visit(
                [&](auto&& currentSymplecticTransforms) {
                    for (const auto& S : *currentSymplecticTransforms) {
                        if (result.has_value()) {
                            return;
                        }

                        if (test_clifford_conjugate_lemma_10(pPrime, qPrime, omega, M, M_p,
                                                             Mprime_p, S)) {
                            #pragma omp critical(result)
                            {
                                if (!result.has_value()) {
                                    result = std::make_optional(
                                        createValueFromFound<IsReturningInfo>(S, pPrime, qPrime));
                                }
                            }

                            return;
                        }
                    }
                },
                iterable);
        }
    }

    return result.value_or(notFoundValue<IsReturningInfo>());
}

} // namespace cliffconjtest

#endif // BRUTEFORCETEST_H
