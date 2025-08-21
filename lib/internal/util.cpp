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
        Z(permutedRow, i) = std::exp(std::complex(0.0, permutedRow * 2.0 * pi * zExponent / d));
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

    std::complex sumTemp = 0.0;

    int outer_omega_exponent = safeMod(-inv_2 * p * q, d);
    std::complex<double> outer_omega_term = std::pow(omega, outer_omega_exponent);

    for (int i = 0; i < d; i++) {
        int subscript1 = safeMod(i + q, d);
        int subscript2 = i;
        int omegaExponent = safeMod(-i * p, d);
        std::complex<double> inner_omega_term = std::pow(omega, omegaExponent);
        sumTemp += inner_omega_term * M(subscript1, subscript2);
    }

    // The final result is scaled by d^(-1) and multiplied by the constant omega term.
    return (outer_omega_term * sumTemp) / static_cast<double>(d);
}

std::complex<double> f_multiqudit(const Eigen::Ref<const Eigen::MatrixXcd>& M,
                                  const Eigen::Vector<long, Eigen::Dynamic>& pq_vec, const long d,
                                  int inv_2, const std::complex<double>& omega) {
    if (M.rows() != M.cols()) {
        throw std::invalid_argument("M is not squre");
    }
    assert(pq_vec.size() % 2 == 0);

    const long numQudits = pq_vec.size() / 2;
    if (numQudits == 0) {
        throw std::invalid_argument("empty or single-element pq_vec");
    }

    const size_t totalDim = M.rows();

    // Check if the dimensions are consistent
    if (std::pow(d, numQudits) != totalDim) {
        std::stringstream ss;
        ss << "Inconsistent dimensions, pq_vec size = " << pq_vec.size() << " (" << std::pow(d, numQudits) << " != " << totalDim << ")";
        throw std::invalid_argument(ss.str());
    }

    // Calculate the outer omega term: omega^(-2^(-1) * sum(p_k*q_k))
    int totalOuterExponent = 0;
    for (long i = 0; i < numQudits; i++) {
        const long pIndex = i;
        const long qIndex = numQudits + i;
        totalOuterExponent += safeMod(-inv_2 * pq_vec(pIndex) * pq_vec(qIndex), d);
    }
    std::complex<double> omegaOuter = std::pow(omega, safeMod(totalOuterExponent, d));

    // Perform the inner summation over all d^N states
    // This could be made clearer by using a TupleIterator
    std::complex sumTemp = 0.0;

    for (long i = 0; i < totalDim; i++) {
        long totalInnerExponent = 0;
        long matrixIndex_j = 0;
        long currentBase = 1;
        long temp_i = i;

        // Convert the single integer index 'i' to a tuple of base-d indices
        // and calculate the new index 'j' and the inner omega exponent.
        for (size_t k = 0; k < numQudits; k++) {
            const long i_k = temp_i % d;

            const long pqIndex = numQudits - k - 1;
            // Go through list backwards
            const long pIndex = pqIndex;
            const long qIndex = numQudits + pqIndex;
            const long p_k = pq_vec(pIndex);
            const long q_k = pq_vec(qIndex);

            // Calculate the total inner exponent
            totalInnerExponent += safeMod(-i_k * p_k, d);

            // Calculate the new index j
            matrixIndex_j += safeMod(i_k + q_k, d) * currentBase;

            // Update base and temp index for the next qudit
            currentBase *= d;
            temp_i /= d;
        }

        std::complex<double> inner_omega_term = std::pow(omega, safeMod(totalInnerExponent, d));
        sumTemp += inner_omega_term * M(matrixIndex_j, i);
    }

    // The final result is scaled by D^(-1) and multiplied by the constant omega term.
    return (omegaOuter * sumTemp) / static_cast<double>(totalDim);
}

} // namespace cliffconjtest