// get_as_unsigned and MaybeReenterWithoutASLR are from Google Benchmark
// (https://github.com/google/benchmark). LICENSE notice for Google
// Benchmark code:
//
// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Eigen/Dense>
#include <chrono>
#include <complex>
#ifdef __linux__
#include <sys/personality.h>
#endif

#include <random>
#include <unordered_set>

#include "include/npy.hpp"

#include "unsupported/Eigen/KroneckerProduct"
#include "cliffordconjugacytest.hpp"
#include "internal/FMap.h"
#include "internal/bruteforcetest.h"
#include "internal/cliffordgates.h"
#include "internal/complexexactrepr.h"
#include "internal/util.h"

#define ENABLE_OPENMP_MULTITHREADING 0
#define ENABLE_KEY_SHUFFLE 1

constexpr std::optional<size_t> customStartIndex = std::nullopt;

constexpr size_t counterThresholdForPrintAndHistogramWrite = 20000;

constexpr bool showEveryMatrixWhenMultithreadingDisabled = false;

// From Google Benchmark
template <typename T>
std::make_unsigned_t<T> get_as_unsigned(T v) {
    using UnsignedT = std::make_unsigned_t<T>;
    return static_cast<UnsignedT>(v);
}

// From Google Benchmark
// Try to disable ASLR (Address Space Layout Randomization) to prevent unreproducible noise
void MaybeReenterWithoutASLR(int /*argc*/, char** argv) {
    // On e.g. Hexagon simulator, argv may be NULL.
    if (!argv) return;

#ifdef __linux__
    const auto curr_personality = personality(0xffffffff);

    // We should never fail to read-only query the current personality,
    // but let's be cautious.
    if (curr_personality == -1) return;

    // If ASLR is already disabled, we have nothing more to do.
    if (get_as_unsigned(curr_personality) & ADDR_NO_RANDOMIZE) return;

    // Try to change the personality to disable ASLR.
    const auto proposed_personality =
        get_as_unsigned(curr_personality) | ADDR_NO_RANDOMIZE;
    const auto prev_personality = personality(proposed_personality);

    // Have we failed to change the personality? That may happen.
    if (prev_personality == -1) return;

    // Make sure the parsona has been updated with the no-ASLR flag,
    // otherwise we will try to reenter infinitely.
    // This seems impossible, but can happen in some docker configurations.
    const auto new_personality = personality(0xffffffff);
    if ((get_as_unsigned(new_personality) & ADDR_NO_RANDOMIZE) == 0)
        return;

    execv(argv[0], argv);
    // The exec() functions return only if an error has occurred,
    // in which case we want to just continue as-is.
#else
    return;
#endif
}

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

std::optional<std::unordered_set<size_t>> getOutlierIndices(std::string& outliersFileName) {
    std::ifstream inputFile(outliersFileName);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << outliersFileName << std::endl;
        return std::nullopt;
    }

    std::unordered_set<size_t> outlierIndices;
    std::string line;

    std::string headerLine;
    std::getline(inputFile, headerLine);
    if (headerLine != "CliffordIndex,DurationMicroS") {
        std::cerr << "Bad header of " << headerLine << " in " << outliersFileName << std::endl;
        return std::nullopt;
    }

    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        std::string indexString;
        // use , as delimiter
        if (std::getline(ss, indexString, ',')) {
            size_t index = std::stoul(indexString);
            outlierIndices.insert(index);
        }
    }

    return outlierIndices.empty() ? std::nullopt : std::make_optional(outlierIndices);
}

// Tests gates in C_2 (Clifford gates)
int main(int argc, char** argv) {
    MaybeReenterWithoutASLR(argc, argv);
    std::optional<std::string> inputSeedFilename;
    std::optional<std::string> outliersFilename;
    if (argc > 2) {
        inputSeedFilename = std::make_optional(std::string(argv[1]));
        std::cout << "Detected seed file arg " << *inputSeedFilename << std::endl;
        if (argc == 3) {
            outliersFilename = std::make_optional(std::string(argv[2]));
#if ENABLE_OPENMP_MULTITHREADING
            assert(false && "Cannot use outliers and multithreading");
#endif
            std::cout << "Detected outliers file arg " << *outliersFilename << std::endl;
        }
    }

    std::optional<std::unordered_set<size_t>> outlierIndices;
    if (outliersFilename) {
        outlierIndices = getOutlierIndices(*outliersFilename);
        if (!outlierIndices) {
            std::cout << "Failed to get outliers from " << *outliersFilename << std::endl;
            return 1;
        }
        std::cout << "Outliers parsed: " << outlierIndices->size() << std::endl;
    }

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
    std::cout << "Key shuffling: " << ENABLE_KEY_SHUFFLE;
    if (shape[1] != 9) {
        throw std::runtime_error("Wrong array dim");
    }
    if (shape[2] != 9) {
        throw std::runtime_error("Wrong array dim");
    }

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

    const auto startUnixEpochMs =  std::chrono::duration_cast<std::chrono::milliseconds>(workStartTime.time_since_epoch());
    const auto resultsFileName = "results-" + std::to_string(startUnixEpochMs.count()) + ".csv";
    {
        if (std::ofstream file(resultsFileName); file) {
            file << "CliffordIndex,DurationMicroS\n";
        } else {
            std::cerr << "Failed to open file for writing\n";
        }
    }

#ifdef ENABLE_KEY_SHUFFLE
    std::vector<std::mt19937::result_type> seeds(numMatrices);
    std::mt19937::result_type baseSeed;
    if (inputSeedFilename) {
        std::cout << "Reading seed from " << *inputSeedFilename << std::endl;
        std::ifstream inputFile(*inputSeedFilename, std::ios::binary);
        // Check if the file was opened successfully.
        if (!inputFile.is_open()) {
            std::cerr << "Error opening file: " << *inputSeedFilename << std::endl;
            return 1;
        }

        // Read the size of the vector first.
        baseSeed = 0;
        inputFile.read(reinterpret_cast<char*>(&baseSeed), sizeof(std::mt19937::result_type));
        if (!inputFile || baseSeed == 0) {
            std::cerr << "failed to read a seed" << std::endl;
            return 1;
        }

        std::cout << "Successfully read baseSeed " << baseSeed << std::endl;
    } else {
        std::random_device rd;
        baseSeed = rd();

        const auto seedFileName = "baseseed-" + std::to_string(startUnixEpochMs.count()) + ".bin";
        std::cout << "Writing base seed " << baseSeed << " to " << seedFileName << std::endl;

        if (std::ofstream file(seedFileName, std::ios::binary); file) {
            file.write(reinterpret_cast<const char*>(&baseSeed), sizeof( std::mt19937::result_type));
        } else {
            std::cerr << "Failed to open base seed file for writing\n";
            exit(1);
        }
        std::cout << "Wrote base seed " << baseSeed << " to " << seedFileName << std::endl;
    }
    std::mt19937 gen(baseSeed);
    std::uniform_int_distribution<std::mt19937::result_type> distrib;

    for (size_t i = 0; i < numMatrices; i++) {
        seeds[i] = distrib(gen);
    }

    std::cout << "Finished generating shuffle seeds for each Clifford (baseSeed=" << baseSeed
        << "). First 2 seeds are " << seeds[0] << ", " << seeds[1] << std::endl;
#endif

    struct Result {
        unsigned long long ms;
        bool isWritten;
    };

    std::vector<Result> allResultsMicroSeconds(numMatrices);

    std::atomic_size_t longestNonZeroRunEndIndex = 0;

    std::atomic<unsigned long long> longestTimeMs = 0;

    std::atomic_bool shouldExitEarly = false;

    // constexpr size_t actualStartIndex = customStartIndex.value_or(0);

#if ENABLE_OPENMP_MULTITHREADING
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < numMatrices; ++i) {
#if !ENABLE_OPENMP_MULTITHREADING
        if (outlierIndices) {
            if (outlierIndices->empty()) {
                break;
            }
            if (!outlierIndices->contains(i)) {
                continue;
            }
            outlierIndices->erase(i);
        }
#endif

        if (foundBad || shouldExitEarly) {
            continue;
        }
        // Calculate start of the current block in the flattened data.
        size_t startIndex = i * matrixSize;

        // Get a pointer to the start of this block
        std::complex<double>* data_ptr = cliffordGateNumPyData.data() + startIndex;

        // zero-copy operation; numpy also column major
        Eigen::Map<Eigen::Matrix<std::complex<double>, 9, 9>> C(data_ptr);

#ifdef ENABLE_KEY_SHUFFLE
        const auto seed = std::make_optional(seeds[i]);
#else
        const auto seed = std::nullopt;
#endif

#if !ENABLE_OPENMP_MULTITHREADING
        if (showEveryMatrixWhenMultithreadingDisabled) {
            std::stringstream ssgate;
            ssgate << "====" << std::endl;
            ssgate << "Processing Gate i=" << i << ", seed = " << seed.value_or(0) << ":"
            << std::endl << getComplexMatrixExactRepo(C) << std::endl;
            ssgate << "====" << std::endl;
            std::cout << ssgate.str() << std::endl;
        }
#endif

        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
            C * M * C.adjoint();

        auto startTime = std::chrono::high_resolution_clock::now();
        const auto result = cliffconjtest::isCliffordConjugateGeneralized(d, n, M, Mprime, seed);
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
        const bool shouldPrintOut = takesVeryLong || counter > counterThresholdForPrintAndHistogramWrite || totalGateCompleteCount == numMatrices - 1;
#else
        constexpr bool shouldPrintOut = true;
#endif
        if (shouldPrintOut) {
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
                      << " bin shuffling: " << ENABLE_KEY_SHUFFLE
                      << ", seed = " << seed.value_or(0)
                      << ", about " << estSLeft << " seconds / " << estMinLeft
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
                        std::cout << "Wrote results to " << resultsFileName <<". longestNonZeroRunEndIndex = " << thisNonZeroRunIndex << std::endl;
                    } else {
                        std::cerr << "Failed to open resultFile for writing\n";
                    }
                }
            }
        }


        if constexpr (customStartIndex.has_value()) {
#if ENABLE_OPENMP_MULTITHREADING
            shouldExitEarly = true;
#else
            break;
#endif
        }
    }

    if (foundBad) {
        std::cout << "Detected failures" << std::endl;
        return 1;
    } else {
        if (!customStartIndex.has_value()) {
            std::cerr << "Writing all results\n";
            if (std::ofstream resultFile(resultsFileName); resultFile) {
                resultFile << "CliffordIndex,DurationMicroS\n";
                for (size_t resultIdx = 0; resultIdx < numMatrices; resultIdx++) {
                    if (outlierIndices && allResultsMicroSeconds[resultIdx].ms == 0) {
                        continue;
                    }
                    resultFile << resultIdx << "," << allResultsMicroSeconds[resultIdx].ms << std::endl;
                }
                std::cerr << "Wrote all results\n";
            } else {
                std::cerr << "Failed to open resultFile for writing\n";
            }
        }
        std::cout << "Verified " << totalGateCompleteCount << "/" << numMatrices << " gates" << std::endl;
        return 0;
    }
}
