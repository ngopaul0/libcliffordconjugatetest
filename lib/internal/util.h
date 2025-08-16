#ifndef UTIL_H
#define UTIL_H
#include <Eigen/Dense>
#include <complex>

// Deal with Windows being different! M_PI needs this to work
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=msvc-170&redirectedfrom=MSDN
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>

namespace cliffconjtest {

// Using M_PI will cause compile error on Windows without specifying the above...
inline constexpr double pi = M_PI;

/**
 * @brief Assuming u = omega^k * v, where omega is the dth root of unity, computes the value of k.
 *
 * @param d The prime for the dth root of unity
 * @param u The first complex number, an instance of std::complex<double>.
 * @param v The second complex number, an instance of std::complex<double>.
 * @return A double representing the scaled phase difference.
 */
inline double checkPhase(size_t d, const std::complex<double>& u, const std::complex<double>& v) {
    // Extract real and imaginary parts from the complex numbers.
    double a1 = u.real();
    double a2 = u.imag();
    double b1 = v.real();
    double b2 = v.imag();

    // If u = omega^k * v, then
    //   omega^k = u / v
    //           = (a1 + i*a2) / (b1 + i*b2)
    //           = (a1 + i*a2)(b1 - i*b2) / (b1 + i*b2)(b1 - i*b2)
    //           = (a1*b1 - a1b2*i + a2b1*i + a2b2) / (b1^2 + b2^2)
    //           = [ (a1*b1 + a2b2) + i*(a2b1 - a1b2) ] / (b1^2 + b2^2)
    // So omega^k = e^(2*pi*i/d * k) = cos(2*pi/d * k) + i*sin(2*pi/d * k).
    // Equating real and imaginary parts, we see
    //   (a2b1 - a1b2) / (b1^2 + b2^2) = sin(2*pi/d * k)  imaginary
    //   (a1b1 + a2b2) / (b1^2 + b2^2) = cos(2*pi/d * k)  real
    //
    // Hence
    //   tan(2*pi/d * k) = sin(2*pi*d * k) / cos(2*pi*d * k)
    //                   = (a2b1 - a1b2) / (a1b1 + a2b2)

    const double cosNumerator = a1 * b1 + a2 * b2;
    const double sinNumerator = a2 * b1 - a1 * b2;

    // Generally, we don't have to worry about cosNumerator == 0, because d is prime.
    // There's no way the root of unity angle will lie on pi/2 or 3pi/2. But let's check anyway
    // in case of floating point roundoff errors.
    if (cosNumerator == 0) {
        return 0;
    }

    double actualAngle = std::atan2(sinNumerator, cosNumerator);

    // std::atan2 has range [-pi, pi].
    if (actualAngle < 0) {
        actualAngle += 2.0 * pi;
    }

    return d * actualAngle / (2.0 * pi);
}

inline bool isApproxEqual(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) <= epsilon;
}

inline bool isApproxEqual(const std::complex<double> a, const std::complex<double> b,
                          double epsilon = 1e-5) {
    return isApproxEqual(a.real(), b.real(), epsilon) && isApproxEqual(a.imag(), b.imag(), epsilon);
}

/**
 * Uses binary exponentiation to efficiently square in the integers mod d.
 *
 * Equivalent to base &^ exponent mod modulus in Maple.
 *
 * @return base ^ exponent % modulus done efficiently
 */
inline long long fastPowerMod(unsigned long long base, unsigned long long exponent,
                              unsigned long long modulus) {
    long long result = 1;
    base %= modulus;
    while (exponent > 0) {
        if (exponent % 2 == 1) { // If the current bit is 1
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exponent /= 2;
    }
    return result;
}

// A helper function to perform modulo arithmetic that correctly handles
// negative numbers, which is a common pitfall with C++'s % operator.
inline long safeMod(const long val, const long modulus) { return (val % modulus + modulus) % modulus; }

inline long modInverse(const long base, const size_t p) {
    // Fermat's little hteorem / Euler's theorem: a^(p-1) = 1 (mod p), so a^(p-2) = a^(-1) (mod p)
    return fastPowerMod(safeMod(base, p), p - 2, p);
}

inline size_t symplecticProduct(size_t d, int p, int q, int pPrime, int qPrime) {
    return safeMod(p * qPrime - pPrime * q, d);
}

/**
 * @brief Creates the Pauli Z gate acting on a single qudit, with exponent
 * @param d Prime dimension
 * @param exponent The exponent used in the complex exponential.
 * @return Z^exponent
 *
 * This function translates the Maple makeZ procedure. The diagonal entries are
 * calculated as Z[i,i] = exp((i-1)*2*Pi*I*exponent/d).
 */
Eigen::MatrixXcd makeZ(int d, int exponent);

/**
 * @brief Creates the product of the Pauli Z and X gates acting on a single qudit.
 * @param d Prime dimension
 * @param zExponent Exponent for Z gate
 * @param xExponent Exponent for X gate
 * @return Z^zExponent * X^xExponent
 */
Eigen::MatrixXcd makeZX(int d, int zExponent, int xExponent);

/**
 * @brief Creates the Pauli gate X^exponent as a Pauli gate on a single qudit.
 * @param d Prime dimension
 * @param exponent_in X gate exponent
 * @return The Pauli gate X^exponent acting on a single qudit.
 */
Eigen::MatrixXcd makeX(int d, int exponent_in);

/**
 * @brief For a given d x d matrix M (d prime), it can be decomposed into a linear combination of
 * the d^2 Pauli basis elements W(p,q), where p,q are integers from 0 to d - 1. This function gets
 * the Pauli basis element W(p,q).
 *
 * Refer to the f() function for calculating the basis linear combination coefficients for a
 * particular matrix.
 *
 * @param d A prime dimension
 * @param p Pauli basis element coordinate
 * @param q Pauli basis element coordinate
 * @param inv_2 The modular multiplicative inverse of 2 modulo d.
 * @param omega Complex dth root of unity.
 * @return The W(p,q) Pauli basis element.
 */
Eigen::MatrixXcd W(long d, int p, int q, int inv_2, const std::complex<double>& omega);

/**
 * @brief For a given d x d matrix M (d prime), it can be decomposed into a linear combination of
 * the d^2 Pauli basis elements W(p,q), where p,q are integers from 0 to d - 1. This function gets
 * the coefficient of W(p,q) in the Pauli basis decomposition of M.
 *
 * Refer to the W() function for the Pauli basis element.
 *
 * @param M A d x d complex matrix, where d is prime
 * @param p Pauli basis element coordinate
 * @param q Pauli basis element coordinate
 * @param inv_2 The modular multiplicative inverse of 2 modulo d.
 * @param omega The complex root of unity (e.g., std::exp(std::complex<double>(0, 2 * M_PI / d))).
 * @return The coefficient on W(p,q) in the W(p,q)-decomposition (Pauli basis) of M
 */
std::complex<double> f(const Eigen::Ref<const Eigen::MatrixXcd>& M, int p, int q, int inv_2,
                       const std::complex<double>& omega);

/**
 * Reduces the matrix A (over Z_p) to REF and returns the rank of A
 * @tparam MatrixType An Eigen integer matrix type like Matrix2i or MatrixXi
 * @param A The matrix over Z_p to reduce to row echelon form (REF). Matrix will be modified.
 * @param p The prime modulus
 * @param shouldReduce Whether do
 * @return The rank of A
 */
template <typename MatrixType>
size_t reduceToREFAndGetRank(MatrixType& A, const int p, bool shouldReduce = false) {
    auto rows = A.rows();
    auto cols = A.cols();

    size_t rank = 0;
    int pivotRow = 0;
    int pivotCol = 0;
    while (pivotRow < rows && pivotCol < cols) {
        // Find a non-zero pivot
        int iMax = pivotRow;
        while (iMax < rows && A(iMax, pivotCol) == 0) {
            iMax++;
        }

        if (iMax == rows) {
            // No pivot found in this column, move to next column
            pivotCol++;
            continue;
        }

        // Swap rows to bring the pivot to the current position
        if (pivotRow != iMax) {
            A.row(pivotRow).swap(A.row(iMax));
        }

        // Normalize the pivot row
        size_t inv_pivot = modInverse(A(pivotRow, pivotCol), p);
        for (int j = pivotCol; j < cols; j++) {
            A(pivotRow, j) = safeMod(A(pivotRow, j) * inv_pivot, p);
        }

        // Zero out everything below (pivotRow, pivotColumn) via row operations
        for (int i = pivotRow + 1; i < rows; i++) {
            if (i != pivotRow) {
                long long factor = A(i, pivotCol);
                for (int j = pivotCol; j < cols; j++) {
                    long long subtractTerm = factor * A(pivotRow, j) % p;
                    A(i, j) = safeMod(A(i, j) - subtractTerm, p);
                }
            }
        }

        pivotRow++;
        pivotCol++;
        rank++;
    }

    if (shouldReduce) {
        pivotRow = rows - 1;
        pivotCol = cols - 1;
        while (pivotRow >= 0 && pivotCol >= 0) {
            // Find first non-zero entry
            int jMin = 0;
            while (jMin <= pivotCol && A(pivotRow, jMin) == 0) {
                jMin++;
            }

            if (jMin > pivotCol) {
                // No pivot found in this row, move to next row
                pivotRow--;
                continue;
            }
            pivotCol = jMin;

            if (A(pivotRow, pivotCol) != 1) {
                auto theValue = A(pivotRow, pivotCol);
                throw std::out_of_range("assertion failed: cannot reduce: not in row echelon form");
            }

            // Zero out everything above (pivotRow, pivotColumn) via row operations
            for (int i = 0; i < pivotRow; i++) {

                long long factor = A(i, pivotCol); // divided by A(pivotRow, pivotCol)
                for (int j = pivotCol; j < cols; j++) {
                    long long subtractTerm = safeMod(factor * A(pivotRow, j), p);
                    A(i, j) = safeMod(A(i, j) - subtractTerm, p);
                }
            }

            pivotRow--;
        }
    }

    return rank;
}

} // namespace cliffconjtest

#endif // UTIL_H
