#ifndef VECTORISATIONALGORITHM_H
#define VECTORISATIONALGORITHM_H

#include <Eigen/Dense>
#include "internal/FMap.h"
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
    assert(X.rows() % 2 == 0);

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

/**
 * Modifies the matrix A in place to create the matrix [A v] (A augmented with vector v).
 */
template <typename MatrixType, typename VectorType>
void augmentAWithVec(MatrixType& A, const VectorType& v) {
    assert(A.rows() == v.rows());
    A.conservativeResize(Eigen::NoChange, A.cols() + 1);
    A.col(A.cols() - 1) = v;
}

template <typename VectorType>
auto createSystem(const VectorType& v, const VectorType& vMap) {
    using Scalar = typename VectorType::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> newBottomRows = createXtransposeTensorI(v);
    augmentAWithVec(newBottomRows, vMap);
    return newBottomRows;
}

template <typename MatrixType, typename VectorType>
void appendToSystem(MatrixType& existingRREFSystem, const VectorType& v, const VectorType& vMap) {
    using Scalar = typename VectorType::Scalar;
    // Evaluate this explicitly so it can be augmented.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> newBottomRows = createXtransposeTensorI(v);
    augmentAWithVec(newBottomRows, vMap);

    existingRREFSystem.conservativeResize(existingRREFSystem.rows() + newBottomRows.rows(), Eigen::NoChange);
    // Copy newBottomRows into the newly created rows at the bottom of existingRREFSystem
    existingRREFSystem.bottomRows(newBottomRows.rows()) = newBottomRows;
}

template <typename MatrixType>
void trimZeroRowsFromBottom(MatrixType& M) {
    long newRows = 0;

    for (long i = M.rows() - 1; i >= 0; --i) {
        if (!M.row(i).isZero()) {
            newRows = i + 1;
            break;
        }
    }

    M.conservativeResize(newRows, Eigen::NoChange);
}

template <typename MatrixType>
Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> recoverSFromSystem(MatrixType& system, size_t numQubits) {
    static_assert(!MatrixType::IsRowMajor,
        "The matrix must be in column-major order for efficient recovery");
    Eigen::Vector<long, Eigen::Dynamic> vecS = system.col(system.cols() - 1).head((2 * numQubits) * (2 * numQubits));
    return vecS.reshaped((2 * numQubits), (2 * numQubits));
}

template <typename MatrixType>
bool isSystemInconsistent(const MatrixType& matrixRREF) {
    const long colsA = matrixRREF.cols() - 1;
    // Search for a row [0 0 ... 0 | k] where k is nonzero from the bottom up.
    for (long i = matrixRREF.rows() - 1; i >= 0; --i) {
        // Check if the current row's A part is all zeros
        if (matrixRREF.row(i).head(colsA).isZero()) {
            // If it is, check if the corresponding b part is non-zero
            if (matrixRREF.row(i)(matrixRREF.cols() - 1) != 0) {
                return true;
            }
        } else {
            // Assume that any row of all 0s has to be below all other nonzero rows of A
            return false;
        }
    }
    return false;
}

inline bool isSymplectic(const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& S, const size_t d) {
    assert(S.rows() == S.cols());
    assert(S.rows() % 2 == 0);
    const long n = S.rows() / 2;
    if (n == 1) {
        return safeMod(S(0,0) * S(1,1) - S(0, 1) * S(1,0), d) == 1;
    }

    // Construct the standard symplectic matrix, J
    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> J = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>::Zero(2 * n, 2 * n);
    J.topRightCorner(n, n).setIdentity();
    J.bottomLeftCorner(n, n).setIdentity();
    J.bottomLeftCorner(n, n) *= static_cast<long>(d - 1);

    // Definition of symplectic S^T * J * S = J
    return modMatrix(S.transpose() * J * S, d) == J;
}

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(size_t d, size_t n, const FMap& Mmap, const FMap& Mprimemap);

} // namespace cliffconjtest

#endif // VECTORISATIONALGORITHM_H
