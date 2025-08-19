#ifndef VECTORISATIONALGORITHM_H
#define VECTORISATIONALGORITHM_H

#include <Eigen/Dense>
// ReSharper disable once CppUnusedIncludeDirective
#include "unsupported/Eigen/KroneckerProduct"

namespace cliffconjtest {

/**
 * Computes X^T tensor I_m
 * @tparam Derived The Derived Eigen matrix type
 * @param X An m x n matrix representing a list of vectors
 * @return X^T tensor I_m
 */
template <typename Derived>
auto createXtransposeTensorI(const Eigen::MatrixBase<Derived>& X) {
    // For the purposes of our algorithm, the columns of X should be vectors in Z_d^(2n).
    assert(X.rows() % 2 != 0);

    using Scalar = typename Derived::Scalar;
    const auto identity =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(X.rows(), X.rows());

    return kroneckerProduct(X.transpose(), identity);
}

/**
 * @brief If X = [ x1 ... xn ] is an m x n matrix (xi in F^m where F is any set of scalars), then
 * vec(X) = <x1, x2, ..., xn> is a mn x 1 column vector with columns of X stacked on each other
 *
 * @tparam MatrixType The Eigen matrix type
 * @param X The m x m matrix to apply the vectorise operator on
 * @return mn x 1 column vector, vec(X)
 */
template <typename MatrixType>
Eigen::Map<Eigen::Vector<typename MatrixType::Scalar, Eigen::Dynamic>> vecOperator(MatrixType& X) {
    static_assert(!MatrixType::IsRowMajor,
        "The matrix must be in column-major order for efficient vec operator.");
    // Since Eigen matrices here are stored in column-major order, it's trivial to make the
    // vec operator. Eigen::Map allows us to create a vector without any data being copied.
    return Eigen::Map<Eigen::Vector<typename MatrixType::Scalar, Eigen::Dynamic>>(X.data(), X.size());
}

// inline Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> findSymplectic() {

//}

} // namespace cliffconjtest

#endif // VECTORISATIONALGORITHM_H
