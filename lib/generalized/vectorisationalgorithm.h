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

constexpr size_t thresholdForParallelInnerProductComputation = 64;

/**
 * Given two sets U = MMap(key) and V = MprimeMap(key) = S(U), tests whether u in U can be mapped to
 * uMap in V by S, where U, V are subsets of Z_d^(2n).
 *
 * This is based on the property of the symplectic inner product, <u, v> = u^T Omega v, where Omega
 * (some papers use J) is the standard symplectic matrix. Namely, if S in Sp(2n, Z_d), then
 * S^T Omega S = Omega by definition of symplectic matrix, and <u, v> = <Su, Sv>, since
 * (Su)^T Omega (Sv) = u^T S^T Omega Sv = u^T Omega v.
 *
 * To build intuition, note that given U ⊆ Z_d^(2n) and its image under some S in Sp(2n, Z_d),
 * V = S(U), S being invertible means |V| = |U|. We also have the following proposition:
 *
 * Proposition. Suppose u in U and v in V. If v = S*u, then for every x in U, <u, x> = <v, Sx>
 * Proof: <v, Sx> = <Su, Sx> = <u, x>
 *
 * The intuition is that S being symplectic implies it's invertible, i.e. bijection, so given
 * x in U, Sx would be the unique value in V = S(U). So using the above proposition that goes over
 * all x in U, the occurrences of the values of the inner product must line up. We formalize this
 * below:
 *
 * Proposition. Fix u in U, v in V. Let k_U(n) = {x in U : <u, x> = n} and
 * k_V(n) = {y in V : <v, y> = n}. If v = S*u, then |k_U(n)| = |k_V(n)| for all n in Z_d. (i.e. the
 * frequencies of the symplectic inner products must line up).
 * Proof: Fix n in Z_d. We claim there is a bijection k_U(n) to k_V(n) defined by S restricted
 * to U.
 * - This map is well-defined, since if x in U, then Sx in V, and <v, Sx> = <Su, Sx> = <u,x> = n, so
 *   Sx in k_V(n).
 * - Clearly Sx = Sy implies x = y since S is invertible / a bijection.
 * - Let y in k_V(n). Then n = <v,y>, and k_V(n) ⊆ V = S(U) implies y = Sx for some x in U. Now
 *   n = <v, y> = <Su, Sx> = <u, x>, so we conclude x in k_U(n). So restricted S is surjective.
 *
 * Note that false positives exists. This is only a necessary condition on v = S*u.
 *
 * Example (False positive). Consider d = 3, n = 1. Then let
 * \code
 *  S = [2 0], U = {<0, 0>, <0, 1>, <0, 2>, <1, 0>, <2, 0>, <2, 1>, <2, 2>} (column vectors)
 *      [0 2]
 * \endcode
 * Then consider u = <1, 0>. We have Su = <2,0>. But now let v = <1,0>, which happens to be S*<2,0>
 * where <2,0> in U. It can be computed that k_U(n) == k_V(n) for all n in Z_3 (the exact histogram
 * is {0: 3, 1: 2, 2: 2}
 *
 * Also, the problem comes when sets consisting of 1-dimension subspaces (i.e. sets where there
 * are no linearly independent subsets of size > 1; hyperplanes). So we stop once a basis is
 * established.
 *
 * @tparam MatrixType Tests
 * @param MMap
 * @param MprimeMap
 * @param u
 * @param uMap
 * @param key
 * @param Omega
 * @param d
 * @return
 */
template <typename MatrixType>
bool testMapping(const FMap& MMap, const FMap& MprimeMap, const MatrixCoordinate& u,
                 const MatrixCoordinate& uMap, const FMapKey& key, const MatrixType& Omega,
                 const size_t d) {
    const auto& U = MMap.get(key);
    const auto& V = MprimeMap.get(key);
#ifndef NDEBUG
    std::stringstream ssU;
    ssU << "U = \n";
    for (const auto& z : U) {
        ssU << z << ";\n";
    }
    auto sU = ssU.str();

    std::stringstream ssV;
    ssV << "V = \n";
    for (const auto& z : V) {
        ssV << z << ";\n";
    }
    auto sV = ssV.str();

    std::stringstream ss_uVec;
    ss_uVec << u;
    auto sUvec = ss_uVec.str();

    std::stringstream ss_vVec;
    ss_vVec << uMap;
    auto sVvec = ss_vVec.str();
#endif

    // avoid using atomic_size_t if OPENMP is missing
#ifdef _OPENMP
    // TODO: Explicitly enable nested parallelism
    // TODO: Refactor this to be more clean
    if (false && U.size() >= thresholdForParallelInnerProductComputation) {
        std::vector<std::atomic_size_t> innerProductHistogramForU(d);
        std::vector<std::atomic_size_t> innerProductHistogramForV(d);
        // clang-format off
        #pragma omp parallel shared(innerProductHistogramForU, innerProductHistogramForV)
        {
            // clang-format off
            #pragma omp for nowait
            for (int i = 0; i < U.size(); ++i) {
                const auto& u_i = U[i];
                const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& innerProduct =
                    u.transpose() * Omega * u_i;
                if (innerProduct.rows() != 1 && innerProduct.cols() != 1) {
                    throw std::runtime_error("bad symplectic inner product shape");
                }
                const long innerProdVal = safeMod(innerProduct(0, 0), d);
                ++innerProductHistogramForU[innerProdVal];
            }

            // clang-format off
            #pragma omp for
            for (int i = 0; i < V.size(); ++i) {
                const auto& v_i = V[i];
                const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& innerProduct =
                    uMap.transpose() * Omega * v_i;
                if (innerProduct.rows() != 1 && innerProduct.cols() != 1) {
                    throw std::runtime_error("bad symplectic inner product shape");
                }
                const long innerProdVal = safeMod(innerProduct(0, 0), d);
                ++innerProductHistogramForV[innerProdVal];
            }
        }
        return innerProductHistogramForU == innerProductHistogramForV;
    }
#endif

    std::vector<size_t> innerProductHistogramForU(d);
    for (const auto& u_i : U) {
        const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& innerProduct =
            u.transpose() * Omega * u_i;
        if (innerProduct.rows() != 1 && innerProduct.cols() != 1) {
            throw std::runtime_error("bad symplectic inner product shape");
        }
        const long innerProdVal = safeMod(innerProduct(0, 0), d);
        innerProductHistogramForU[innerProdVal] += 1;
    }

    std::vector<size_t> innerProductHistogramForV(d);
    for (const auto& v_i : V) {
        const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& innerProduct =
            uMap.transpose() * Omega * v_i;
        if (innerProduct.rows() != 1 && innerProduct.cols() != 1) {
            throw std::runtime_error("bad symplectic inner product shape");
        }
        const long innerProdVal = safeMod(innerProduct(0, 0), d);
        innerProductHistogramForV[innerProdVal] += 1;
        if (innerProductHistogramForV[innerProdVal] > innerProductHistogramForU[innerProdVal]) {
            // return false;
        }
    }

    std::stringstream ssFMap;
    ssFMap << MMap;
    auto sFMap = ssFMap.str();

    std::stringstream ssFPrimeMap;
    ssFPrimeMap << MprimeMap;
    auto sFMprimeap = ssFPrimeMap.str();

    const bool result = innerProductHistogramForU == innerProductHistogramForV;
    return result;
}

template <typename VectorType>
Eigen::Matrix<typename VectorType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
createSystemForMappingsCheck(const VectorType& v) {
    return v.transpose();
}

template <typename VectorType>
void appendToSystemForMappingsCheck(
    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& existingRREFSystem, const VectorType& v) {
    existingRREFSystem.conservativeResize(existingRREFSystem.rows() + 1, Eigen::NoChange);
    existingRREFSystem.row(existingRREFSystem.rows() - 1) = v.transpose();
}

template <typename MatrixType>
std::optional<std::vector<std::vector<std::vector<size_t>>>>
getPossibleMappings(const FMap& MMap, const FMap& MprimeMap, const MatrixType& Omega,
                    const size_t d, const size_t n, const std::vector<FMapKey>& sortedKeys) {
    std::vector<std::vector<std::vector<size_t>>> possibleMappings(sortedKeys.size());
    assert(possibleMappings.size() == sortedKeys.size());
    assert(sortedKeys.size() == MMap.size());
    assert(sortedKeys.size() == MprimeMap.size());
    size_t vectorsMapped = 0;
    std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>> system;
    size_t lastRank = 0;

    const size_t vectorsNeededForBasis = 2 * n;

    for (size_t binIndex = 0; binIndex < sortedKeys.size(); ++binIndex) {
        assert(possibleMappings[binIndex].empty());

        const auto& keyForBin = sortedKeys[binIndex];
        const auto& allV = MMap.get(keyForBin);

        // Reserve enough vectors. Use a resize to insert an empty slot for each v index. We expect
        // at least 2n vectors.
        const auto numVecsInThisBin = allV.size();
        const auto numVecsLeft = vectorsMapped > vectorsNeededForBasis ? 0 : vectorsNeededForBasis - vectorsMapped;
        const auto numPossibleVecsFromThisV = std::min(numVecsLeft, numVecsInThisBin);
        if (numPossibleVecsFromThisV > 0) {
            possibleMappings[binIndex].resize(numPossibleVecsFromThisV);
            assert(possibleMappings[binIndex].size() == numPossibleVecsFromThisV);
        }
        const auto& vMapPossibilities = MprimeMap.get(keyForBin);
        for (size_t vIndex = 0; vIndex < allV.size(); ++vIndex) {
            const MatrixCoordinate& vHere = allV[vIndex];
            assert(vIndex <= possibleMappings[binIndex].size());

            // Check if access by vIndex would be invalid. We would be here if we ran into false
            // positives.
            if (vIndex == possibleMappings[binIndex].size()) {
                possibleMappings[binIndex].emplace_back();
                assert(vIndex == possibleMappings[binIndex].size() - 1);
            }

            size_t thisRank;
            if (!system) {
                system = std::make_optional(createSystemForMappingsCheck(vHere));
                lastRank = 0;
                thisRank = 1;
            } else {
                appendToSystemForMappingsCheck(*system, vHere);
                thisRank = reduceToREFAndGetRank(*system, d, true);
                // rows of vHere is the dimension of the vector space. See if we have a basis
                // already
                assert(2 * n == vHere.rows());
                assert(thisRank >= lastRank);
                if (thisRank == lastRank) {
                    trimZeroRowsFromBottom(*system);
                    continue;
                }
                lastRank = thisRank;
            }

            if (false && vMapPossibilities.size() > thresholdForParallelInnerProductComputation) {
                // clang-format off
                #pragma omp parallel for shared(possibleMappings) schedule(dynamic)
                for (int vMapIndex = 0; vMapIndex < vMapPossibilities.size(); ++vMapIndex) {
                    const auto& vMapPossible = vMapPossibilities[vMapIndex];
                    if (testMapping(MMap, MprimeMap, vHere, vMapPossible, keyForBin, Omega, d)) {
                        // clang-format off
                        #pragma omp critical(possibleMappings)
                        {
                            possibleMappings[binIndex][vIndex].push_back(vMapIndex);
                        }
                    }
                }
            } else {
                for (size_t vMapIndex = 0; vMapIndex < vMapPossibilities.size(); ++vMapIndex) {
                    const auto& vMapPossible = vMapPossibilities[vMapIndex];
                    if (testMapping(MMap, MprimeMap, vHere, vMapPossible, keyForBin, Omega, d)) {
                        possibleMappings[binIndex][vIndex].push_back(vMapIndex);
                    }
                }
                std::stringstream ssHere;
                for (const auto& vMapIndex : possibleMappings[binIndex][vIndex]) {
                    const auto vMapPossibleHere = vMapPossibilities[vMapIndex];

                    ssHere << vMapPossibleHere << ";" << std::endl;
                }
                auto s = ssHere.str();
                std::stringstream ssHere2;
            }
            // If u couldn't be mapped to anything at all, then the necessary condition failed for all (u, v) pairs for
            // all v in V
            if (possibleMappings[binIndex][vIndex].empty()) {
                return std::nullopt;
            }
            vectorsMapped++;

            if (thisRank == vectorsNeededForBasis) {
                return std::make_optional(possibleMappings);
            }
        }
    }

    return std::make_optional(possibleMappings);
}

size_t returnLastNumRecursiveCalls();

std::optional<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>
findSymplecticMatrix(size_t d, size_t n, const std::complex<double>& omega,
                     const Eigen::Ref<const Eigen::MatrixXcd>& M, const MpMatrixType& M_p,
                     const MpMatrixType& Mprime_p, FMap& Mmap, FMap& Mprimemap,
                     const std::optional<std::uint_fast32_t>& shuffleSeed = std::make_optional(0));

} // namespace cliffconjtest

#endif // VECTORISATIONALGORITHM_H
