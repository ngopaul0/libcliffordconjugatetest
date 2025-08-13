#include "util.h"

#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace cliffconjtest {

std::string matrix_to_string(const Eigen::Ref<const Eigen::MatrixXcd>& M) {
    std::stringstream ss;
    ss << "[";

    for (size_t i = 0; i < M.rows(); i++) {
        ss << "["; // Start of the row
        for (size_t j = 0; j < M.cols(); j++) {
            // Format complex number as real + I*imag for Maple
            ss << "" << M(i, j).real() << " + I*(" << M(i, j).imag() << ")";
            if (j < M.cols() - 1) {
                ss << ", "; // Separate elements
            }
        }
        ss << "]"; // End of the row
        if (i < M.size() - 1) {
            ss << ", "; // Separate rows
        }
    }

    ss << "]";
    return ss.str();
}

size_t symplecticProduct(size_t d, int p, int q, int pPrime, int qPrime) {
    return safeMod(p * qPrime - pPrime * q, d);
}

Eigen::MatrixXcd makeZ(int d, int exponent) {
    Eigen::MatrixXcd Z = Eigen::MatrixXcd::Zero(d, d);
    for (int i = 0; i < d; i++) {
        Z(i, i) = std::exp(std::complex(0.0, i * 2.0 * pi * exponent / d));
    }
    return Z;
}

Eigen::MatrixXcd makeZX(int d, int zExponent, int xExponent) {
    Eigen::MatrixXcd Z = Eigen::MatrixXcd::Zero(d, d);

    for (int i = 0; i < d; i++) {
        int permutedRow = safeMod(i + xExponent, d);
        Z(permutedRow, i) =
            std::exp(std::complex(0.0, permutedRow * 2.0 * pi * zExponent / d));
    }
    return Z;
}

Eigen::MatrixXcd makeX(int d, int exponent_in) {
    Eigen::MatrixXcd X = Eigen::MatrixXcd::Zero(d, d);
    int exponent = safeMod(exponent_in, d);
    int col = safeMod(d - exponent, d);
    for (int row = 0; row < d; row++) {
        X(row, col) = 1.0;
        col = safeMod(col + 1, d);
    }
    return X;
}

Eigen::MatrixXcd W(long d, int p, int q, int inv_2, const std::complex<double>& omega) {
    Eigen::MatrixXcd W_matrix = Eigen::MatrixXcd::Zero(d, d);
    const auto omegaExponent = safeMod(-inv_2 * p * q, d);
    auto ZX = makeZX(d, p, q);
    const auto omegaPow = std::pow(omega, omegaExponent);
    for (size_t i = 0; i < ZX.rows(); i++) {
        for (size_t j = 0; j < ZX.cols(); j++) {
            ZX(i, j) *= omegaPow;
        }
    }
    return ZX;
}

std::complex<double> f(const Eigen::Ref<const Eigen::MatrixXcd>& M, int p, int q, int inv_2,
                       const std::complex<double>& omega) {
    if (M.rows() != M.cols()) {
        throw std::invalid_argument("non-square matrix");
    }
    const auto d = M.rows();

    // C++ is 0-indexed, so we loop from 0 to d-1.
    std::complex sumTemp = 0.0;

    // The first omega term in the Maple function's sum is constant for the loop,
    // so we can calculate it once outside the loop for efficiency.
    int outer_omega_exponent = safeMod(-inv_2 * p * q, d);
    std::complex<double> outer_omega_term = std::pow(omega, outer_omega_exponent);

    for (int i = 0; i < d; i++) {
        // Calculate the 0-based row and column indices.
        // Maple's subscript1: ((i-1+q) mod d) + 1  ->  C++ subscript: (i+q) mod d
        int subscript1 = safeMod(i + q, d);
        int subscript2 = i; // The loop variable i is already 0-based

        // Calculate the exponent for the inner omega term.
        // Maple's omegaExponent: modp(-(i-1)*p, d) -> C++ exponent: modp(-i*p, d)
        int omegaExponent = safeMod(-i * p, d);
        std::complex<double> inner_omega_term = std::pow(omega, omegaExponent);

        // Add the term to the sum.
        sumTemp += inner_omega_term * M(subscript1, subscript2);
    }

    // The final result is scaled by d^(-1) and multiplied by the constant omega term.
    return (outer_omega_term * sumTemp) / static_cast<double>(d);
}

} // namespace cliffconjtest