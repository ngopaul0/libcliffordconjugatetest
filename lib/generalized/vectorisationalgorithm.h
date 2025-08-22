#ifndef VECTORISATIONALGORITHM_H
#define VECTORISATIONALGORITHM_H

#include "unsupported/Eigen/KroneckerProduct"
#include <Eigen/Dense>
#include "internal/FMap.h"
#include "internal/multidimarray.h"

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

    return Eigen::kroneckerProduct(X.transpose(), identity);
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
    return Eigen::Map<Eigen::Vector<typename MatrixType::Scalar, Eigen::Dynamic>>(X.data(),
                                                                                  X.size());
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
auto createSystemForS(const VectorType& v, const VectorType& vMap) {
    using Scalar = typename VectorType::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> newBottomRows =
        createXtransposeTensorI(v);
    augmentAWithVec(newBottomRows, vMap);
    return newBottomRows;
}

template <typename MatrixType, typename VectorType>
void appendToSystemForS(MatrixType& existingRREFSystem, const VectorType& v,
                        const VectorType& vMap) {
    using Scalar = typename VectorType::Scalar;
    assert(existingRREFSystem.rows() != 0);
    assert(existingRREFSystem.cols() != 0);

    // Evaluate this explicitly so it can be augmented.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> newBottomRows =
        createXtransposeTensorI(v);
    augmentAWithVec(newBottomRows, vMap);

    existingRREFSystem.conservativeResize(existingRREFSystem.rows() + newBottomRows.rows(),
                                          Eigen::NoChange);
    // Copy newBottomRows into the newly created rows at the bottom of existingRREFSystem
    existingRREFSystem.bottomRows(newBottomRows.rows()) = newBottomRows;
}

/**
 * From known pq_vecs = (p_1, ..., p_n, q_1, ..., q_n) and unknown pPrime_qPrime_vec =
 * (p_1',...,p_n',q_1',...,q_n'), computes a system of equations of the form
 *
 *     symplecticProduct(pq_vec, pPrime_qPrime_vec) = k.
 *
 * We have assumption that
 * \code
 *     f_M(pq_vec) = omega^symplecticProduct(pq_vec, pPrime_qPrime_vec) * f_{M'}(S(pq_vec)).
 * \endcode
 *
 * We can compute f_M(pq_vec), f_{M'}(S(pq_vec)), and also the value of
 * symplecticProduct(pq_vec, pPrime_qPrime_vec), which we'll call k.
 *
 * The symplectic product is defined as sum(i = 1..n, p_i * q_i' - q_i * p_i'). So row-wise (i.e.,
 * one equation), one row of the system would be
 * \code
 *                                                |p_1'|
 *                                                |... |
 *                                                |p_n'|
 *            [-q_1  ...  -q_n  p_1  ...  p_n] *  |q_1'| = k
 *                                                |... |
 *                                                |q_n'|
 * \endcode
 * @tparam VectorType
 * @param d Odd prime
 * @param pq_vec
 * @param k The apparent exponent for alpha = omega^k * beta
 * @return
 */
template <typename VectorType>
auto createSystemForPPrimeQPrime(const size_t d, const VectorType& pq_vec, const size_t k) {
    using Scalar = typename VectorType::Scalar;
    const size_t twoTimes_n = pq_vec.rows();
    const long n = twoTimes_n / 2;
    Eigen::RowVector<Scalar, Eigen::Dynamic> row(pq_vec.rows() + 1);
    for (size_t i = 0; i < n; i++) {
        size_t pIndex = i;
        size_t qIndex = n + i;
        long p_i = pq_vec[pIndex];
        long q_i = pq_vec[qIndex];

        // -q_i first
        row(pIndex) = q_i == 0 ? 0 : d - q_i;
        // p_i second
        row(qIndex) = p_i;
        assert(row(pIndex) == safeMod(-q_i, d));
    }
    row(row.cols() - 1) = k;
    return row;
}

template <typename VectorType>
void appendToSystemForPPrimeQPrime(
    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& existingRREFSystem, const size_t d,
    const VectorType& pq_vec, const size_t k) {
    using Scalar = typename VectorType::Scalar;
    assert(existingRREFSystem.rows() != 0);
    assert(existingRREFSystem.cols() != 0);

    // Evaluate this explicitly so it can be augmented.
    Eigen::RowVector<Scalar, Eigen::Dynamic> newBottomRow =
        createSystemForPPrimeQPrime(d, pq_vec, k);

    existingRREFSystem.conservativeResize(existingRREFSystem.rows() + 1, Eigen::NoChange);
    // Copy newBottomRow into the newly created rows at the bottom of existingRREFSystem
    existingRREFSystem.row(existingRREFSystem.rows() - 1) = newBottomRow;
}

template <typename MatrixType>
Eigen::Vector<long, Eigen::Dynamic> recoverPPrimeQPrimeVecFromSystem(MatrixType& system,
                                                                     size_t numQubits) {
    Eigen::Vector<long, Eigen::Dynamic> vecS = system.col(system.cols() - 1).head(2 * numQubits);
    return vecS;
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
Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> recoverSFromSystem(MatrixType& system,
                                                                       size_t numQubits) {
    static_assert(!MatrixType::IsRowMajor,
                  "The matrix must be in column-major order for efficient recovery");
    Eigen::Vector<long, Eigen::Dynamic> vecS =
        system.col(system.cols() - 1).head((2 * numQubits) * (2 * numQubits));
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

/**
 * @return Whether the 2n x 2n matrix M is essentially a block diagonal matrix (2x2 blocks)
 * with symplectic matrices as the diagonal blocks
 */
inline bool checkIfBlockDiagonalAndEachBlockSymplectic(
    const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& M, size_t d) {
    // Get the dimensions of the matrix.
    long rows = M.rows();
    long cols = M.cols();
    assert(rows % 2 == 0);
    assert(rows == cols);

    // Determine the number of blocks, which is n.
    const long n = rows / 2;
    constexpr long blockSize = 2; // The size of each block is 2x2.

    // Check if all off-block-diagonal elements in the matrix are zero.
    // e.g., Consider
    // [2 0 0 0]
    // [0 2 0 0]
    // [0 0 1 0]
    // [0 0 0 1]
    // Each 2x2 block is a symplectic matrix over Z_d, and all off-block-diagonal elements are 0
    for (int i = 0; i < n; ++i) {
        long startRow = i * blockSize;
        long startCol = i * blockSize;

        long rowIndexToCheck = blockSize * i;
        long rowIndexToCheck2 = blockSize * i + 1;
        // If n = 2, then
        //
        // i = 9:
        // [2 0 0 0] <- rowIndexToCheck
        // [0 2 0 0] <- rowIndexToCheck2
        // [0 0 1 0]
        // [0 0 0 1]
        //
        // i = 1
        // [2 0 0 0]
        // [0 2 0 0]
        // [0 0 1 0] <- rowIndexToCheck
        // [0 0 0 1] <- rowIndexToCheck2
        for (int c = 0; c < cols; ++c) {
            if (c >= startCol && c < startCol + blockSize) {
                continue;
            }

            bool inCurrentBlock =
                (rowIndexToCheck >= startRow && rowIndexToCheck < startRow + blockSize);
            // If the element is not in any of the diagonal blocks and is not zero, the matrix is
            // not block diagonal.
            if (!inCurrentBlock && M(rowIndexToCheck, c) != 0) {
                return false;
            }
            inCurrentBlock =
                (rowIndexToCheck2 >= startRow && rowIndexToCheck2 < startRow + blockSize);
            if (!inCurrentBlock && M(rowIndexToCheck2, c) != 0) {
                return false;
            }
        }
    }

    // Iterate through the matrix to check for both conditions.
    for (long i = 0; i < n; ++i) {
        // Define the starting row and column for the current block.
        long startRow = i * blockSize;
        long startCol = i * blockSize;

        // Extract the 2x2 block from the matrix.
        Eigen::Matrix<long, 2, 2> block = M.block<blockSize, blockSize>(startRow, startCol);

        long long det = safeMod(block.determinant(), d);

        // Check if the determinant modulo d is 1.
        if (det != 1) {
            return false;
        }
    }
    return true;
}

inline bool isSymplectic(const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& S,
                         const size_t d) {
    assert(S.rows() == S.cols());
    assert(S.rows() % 2 == 0);
    const long n = S.rows() / 2;
    if (n == 1) {
        return safeMod(S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0), d) == 1;
    }

    // Construct the standard symplectic matrix, Ω
    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> Omega =
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>::Zero(2 * n, 2 * n);
    Omega.topRightCorner(n, n).setIdentity();
    Omega.bottomLeftCorner(n, n).setIdentity();
    Omega.bottomLeftCorner(n, n) *= static_cast<long>(d - 1);
    // Definition of symplectic S^T * Ω * S = Ω
    return modMatrix(S.transpose() * Omega * S, d) == Omega;
}

size_t returnLastNumRecursiveCalls();

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(size_t d, size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, FMap& Mmap, FMap& Mprimemap,
                     bool shuffleKeys = true);

} // namespace cliffconjtest

#endif // VECTORISATIONALGORITHM_H
