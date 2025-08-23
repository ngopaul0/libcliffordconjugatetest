#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include "include/npy.hpp"

#include "unsupported/Eigen/KroneckerProduct"
#include "cliffordconjugacytest.hpp"
#include "internal/FMap.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/complexexactrepr.h"
#include "internal/util.h"

#define ENABLE_OPENMP_MULTITHREADING 1

static constexpr bool ENABLE_KEY_SHUFFLE = false;

Eigen::MatrixXcd computeMSumm(size_t d, const std::vector<std::pair<long, long>>& coords,
                              size_t inv_2, const std::complex<double>& omega) {
    Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(d, d);
    for (const auto& [p, q] : coords) {
        sum += cliffconjtest::W(d, p, q, inv_2, omega);
    }
    return sum;
}

template <typename MatrixType>
std::string getComplexMatrixExactRepo(const MatrixType& M) {
    std::stringstream ss;
    for (long i = 0; i < M.rows(); i++) {
        for (long j = 0; j < M.cols(); j++) {
            // access the same memory location using different types.
            std::complex<double> value = M(i, j);
            ss << ComplexExactRepr(value);
            if (j != M.cols() - 1) {
                ss << ", ";
            }
        }
        if (i != M.rows() - 1) {
            ss << "," << std::endl;
        }
    }
    ss << ";" << std::endl;
    return ss.str();
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

    const std::vector<std::pair<long, long>> coords1 = {{1, 2}, {0, 1}};
    const std::vector<std::pair<long, long>> coords2 = {{0, 0}};

    const std::vector<std::pair<long, long>> coords3 = {{2, 1}, {2, 0}};
    const std::vector<std::pair<long, long>> coords4 = {{1, 0}};
    const Eigen::MatrixXcd H1 = computeMSumm(d, coords1, inv_2, omega);
    const Eigen::MatrixXcd H2 = computeMSumm(d, coords2, inv_2, omega);
    const Eigen::MatrixXcd H = std::complex<double>(1, -3) * Eigen::kroneckerProduct(H1, H2);

    const Eigen::MatrixXcd M1 = computeMSumm(d, coords1, inv_2, omega);
    const Eigen::MatrixXcd M2 = computeMSumm(d, coords2, inv_2, omega);


    // This example will currently fail at i = 7 without the early checks against error propagation
    // in checkPhase
    const Eigen::MatrixXcd M = // Eigen::kroneckerProduct(M1, M2) + H
        - std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));

    std::atomic_bool foundBad = false;
    std::atomic_size_t counter = INT_MAX;
    std::atomic_size_t totalGateCompleteCount = 0;

    std::atomic<unsigned long long> totalDurationMicroS{0};
    std::cout << "Running tests" << std::endl;
    const auto workStartTime = std::chrono::high_resolution_clock::now();

    const auto resultsFileName = "results-" + std::to_string(workStartTime.time_since_epoch().count()) + ".csv";
    {
        if (std::ofstream file(resultsFileName); file) {
            file << "CliffordIndex,DurationMicroS\n";
        } else {
            std::cerr << "Failed to open file for writing\n";
        }
    }

    constexpr size_t counterThresholdForPrintAndHistogramWrite = 2000;

    struct Result {
        unsigned long long ms;
        bool isWritten;
    };

    std::vector<Result> allResultsMicroSeconds(numMatrices);

    std::atomic_size_t longestNonZeroRunEndIndex = 0;

    std::atomic<unsigned long long> longestTimeMs = 0;

#if ENABLE_OPENMP_MULTITHREADING
#pragma omp parallel for schedule(dynamic)
#endif
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

#if !ENABLE_OPENMP_MULTITHREADING
        std::stringstream ssgate;
        ssgate << "====" << std::endl;
        ssgate << "Processing Gate i=" << i << ":"
        << std::endl << getComplexMatrixExactRepo(C) << std::endl;
        ssgate << "====" << std::endl;
        std::cout << ssgate.str() << std::endl;
#endif

        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
            C * M * C.adjoint();
        auto startTime = std::chrono::high_resolution_clock::now();
        const auto result = cliffconjtest::isCliffordConjugateGeneralized(d, n, M, Mprime, ENABLE_KEY_SHUFFLE);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto durationMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        auto durationMicroS =
            std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        const auto durationToRecord = durationMicroS.count();
#pragma omp critical(resultWriting)
        {
            allResultsMicroSeconds[i].ms = durationToRecord == 0 ? 1 : durationToRecord;
        }
        if (!result) {
            const auto timeSinceStart =
                std::chrono::duration_cast<std::chrono::milliseconds>(endTime - workStartTime);
            std::stringstream ss;
            ss << "====" << std::endl;
            ss << "Gate i=" << i << " (" << timeSinceStart << ") gave an unexpected non Clifford-conjugate result:"
            << std::endl << getComplexMatrixExactRepo(C) << std::endl;
            ss << "====" << std::endl;
            std::cout << ss.str();
            foundBad = true;
            continue;
        }

        ++totalGateCompleteCount;
        ++counter;
        auto durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        totalDurationMicroS +=
            std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        bool takesVeryLong = false;
        {
            const size_t thisGateCount = totalGateCompleteCount;
            long long avgMicroS = thisGateCount > 0 ? totalDurationMicroS / thisGateCount : 0;
            // not proper
            if (durationMs.count() >= longestTimeMs - avgMicroS) {
                takesVeryLong = true;
            }
        }

        if (durationMs.count() > longestTimeMs) {
            longestTimeMs = durationMs.count();
        }
#if ENABLE_OPENMP_MULTITHREADING
        if (takesVeryLong || counter > counterThresholdForPrintAndHistogramWrite || totalGateCompleteCount == numMatrices - 1) {
#else
        {
#endif
            if (!takesVeryLong) {
                counter = 0;
            }
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


            const std::string startString = takesVeryLong ? "(LONG) " : "";

            std::cout << startString << "Verified " << totalGateCompleteCount << " gates out of " << numMatrices
                      << std::endl
                      << " (i = " << i << " took " << durationNs << " / " << durationMs
                      << ", avgMicroS " << avgMicroS
                      << ", avg speed " << gatesPerMs << " gates/ms"
                      << ", max millis " << longestTimeMs
                      << ", " << std::endl
                      << " about " << estSLeft << " seconds / " << estMinLeft
                      << "min left)" << std::endl;
            if (!takesVeryLong) {
#pragma omp critical(resultWriting)
                {
                    if (std::ofstream resultFile(resultsFileName, std::ios::app); resultFile) {
                        const size_t thisLongestIndex = longestNonZeroRunEndIndex;
                        const auto loopStart = i <= longestNonZeroRunEndIndex ? 0 : thisLongestIndex;
                        size_t thisNonZeroRunIndex = 0;
                        bool foundZero = false;
                        for (size_t resultIdx = loopStart; resultIdx <= i; resultIdx++) {
                            if (allResultsMicroSeconds[resultIdx].ms != 0) {
                                if (!foundZero) {
                                    thisNonZeroRunIndex = resultIdx;
                                }
                                if (!allResultsMicroSeconds[resultIdx].isWritten) {
                                    allResultsMicroSeconds[resultIdx].isWritten = true;
                                    resultFile << resultIdx << "," << allResultsMicroSeconds[resultIdx].ms << std::endl;
                                }
                            } else {
                                foundZero = true;
                            }
                        }
                        longestNonZeroRunEndIndex = thisNonZeroRunIndex;
                        std::cout << "Wrote results. longestNonZeroRunEndIndex = " << thisNonZeroRunIndex << std::endl;
                    } else {
                        std::cerr << "Failed to open resultFile for writing\n";
                    }
                }
            }
        }
    }

    if (foundBad) {
        std::cout << "Detected failures" << std::endl;
    } else {
        std::cerr << "Writing all results\n";
        if (std::ofstream resultFile(resultsFileName, std::ios::app); resultFile) {
            for (size_t resultIdx = 0; resultIdx < numMatrices; resultIdx++) {
                if (allResultsMicroSeconds[resultIdx].ms != 0 && !allResultsMicroSeconds[resultIdx].isWritten) {
                    resultFile << resultIdx << "," << allResultsMicroSeconds[resultIdx].ms << std::endl;
                }
            }
            std::cerr << "Wrote all results\n";
        } else {
            std::cerr << "Failed to open resultFile for writing\n";
        }
        std::cout << "Verified all gates" << std::endl;
    }
}
