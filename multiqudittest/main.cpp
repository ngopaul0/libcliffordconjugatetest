#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include "include/npy.hpp"

#include "unsupported/Eigen/KroneckerProduct"
#include "cliffordconjugacytest.hpp"
#include "internal/FMap.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/util.h"

Eigen::MatrixXcd computeMSumm(size_t d, const std::vector<std::pair<long, long>>& coords,
                              size_t inv_2, const std::complex<double>& omega) {
    Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(d, d);
    for (const auto& [p, q] : coords) {
        sum += cliffconjtest::W(d, p, q, inv_2, omega);
    }
    return sum;
}

// Tests gates in C_2 (Clifford gates
int main() {
    // Load data from .npy file
    std::cout << "Loading Clifford gates on two qudits (d=3)" << std::endl;
    npy::npy_data data = npy::read_npy<std::complex<double>>("n2-c2-gates-d3-asGATES.npy");
    std::cout << "Loaded Clifford gates on two qudits (d=3)" << std::endl;
    std::vector<std::complex<double>> cliffordGateNumPyData = data.data;
    std::vector<npy::ndarray_len_t> shape = data.shape;

    // Verify the shape
    if (shape[0] != 4199040) {
        throw std::runtime_error("Wrong array shape");
    }
    std::cout << shape[0] << " gates loaded" << std::endl;
    if (shape[1] != 9) {
        throw std::runtime_error("Wrong array dim");
    }
    if (shape[2] != 9) {
        throw std::runtime_error("Wrong array dim");
    }
    // std::vector<Eigen::Matrix<std::complex<double>, 9, 9>> matrix_list;

    const size_t numMatrices = shape[0];
    const size_t matrixSize = shape[1] * shape[2];

    const int d = 3;
    const size_t n = 2;
    const long inv_2 = cliffconjtest::modInverse(2, d);
    const std::complex<double> omega =
        std::exp(std::complex<double>(0, 2.0 * cliffconjtest::pi / d));
    const std::vector<std::pair<long, long>> coords1 = {{1, 2}, {0, 1}, {2, 2}};
    const std::vector<std::pair<long, long>> coords2 = {{2, 1}, {2, 0}, {1, 1}};

    const Eigen::MatrixXcd M1 = computeMSumm(d, coords1, inv_2, omega);
    const Eigen::MatrixXcd M2 = computeMSumm(d, coords2, inv_2, omega);
    const Eigen::MatrixXcd M = Eigen::kroneckerProduct(M1, M2);

    std::atomic_bool foundBad = false;
    std::atomic_size_t counter = INT_MAX;
    std::atomic_size_t totalGateCompleteCount = 0;
    std::atomic<unsigned long long> totalDurationMicroS{0};
    std::cout << "Running tests" << std::endl;

    const auto workStartTime = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numMatrices; ++i) {
        if (foundBad) {
            continue;
        }
        // Calculate start of the current block in the flattened data.
        size_t startIndex = i * matrixSize;

        // Get a pointer to the start of this block
        std::complex<double>* data_ptr = cliffordGateNumPyData.data() + startIndex;

        // zero-copy operation; numpy also column major
        Eigen::Map<Eigen::Matrix<std::complex<double>, 9, 9>> C(data_ptr);

        // std::cout << "Gate " << i << "\n" << C << std::endl;
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
            C * M * C.adjoint();
        auto startTime = std::chrono::high_resolution_clock::now();
        const auto result = cliffconjtest::isCliffordConjugateGeneralized(d, n, M, Mprime);
        auto endTime = std::chrono::high_resolution_clock::now();
        if (!result) {
            std::cout << "Gate " << i << "gave an unexpected non Clifford-conjugate result: \n"
                      << C << std::endl;
            foundBad = true;
            continue;
        }

        ++totalGateCompleteCount;
        ++counter;
        auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        totalDurationMicroS +=
            std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        auto durationMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        if (counter > 1000) {
            counter = 0;
            const auto timeSinceStart =
                std::chrono::duration_cast<std::chrono::milliseconds>(endTime - workStartTime);
            const size_t thisGateCount = totalGateCompleteCount;
            long long avgMicroS = thisGateCount > 0 ? totalDurationMicroS / thisGateCount : 0;
            const long double gatesPerMs =
                timeSinceStart.count() > 0
                    ? static_cast<double>(thisGateCount) / timeSinceStart.count()
                    : 0;
            const long double estMsLeft =
                gatesPerMs > 0 ? (numMatrices - thisGateCount) / gatesPerMs : 0;
            const long double estSLeft = estMsLeft / 1000.0;
            const long double estMinLeft = estSLeft / 60.0;
            std::cout << "Verified " << totalGateCompleteCount << " gates out of " << numMatrices
                      << " (i = " << i << " took " << durationNs << " / " << durationMs
                      << ", avg micro seconds " << avgMicroS << ", avg speed " << gatesPerMs
                      << " gates/ms, about " << estSLeft << " seconds / " << estMinLeft
                      << "min left)" << std::endl;
        }
    }

    if (foundBad) {
        std::cout << "Detected failures" << std::endl;
    } else {
        std::cout << "Verified all gates" << std::endl;
    }
}
