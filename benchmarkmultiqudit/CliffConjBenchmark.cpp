#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <complex>
#include <include/npy.hpp>
#include <internal/util.h>
#include <iostream>
#include <random>

#include "unsupported/Eigen/KroneckerProduct"
#include "cliffordconjugacytest.hpp"
#include "internal/cliffordgates.h"

using namespace cliffconjtest;

// Helper function to check if a number is a prime.
constexpr bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i = i + 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

// A helper function to count primes up to N.
constexpr int count_primes(int n) {
    int count = 0;
    for (int i = 3; i <= n; ++i) {
        if (is_prime(i)) {
            count++;
        }
    }
    return count;
}

// The main constexpr function to generate the prime array.
template <size_t N>
constexpr auto generate_primes(int max_val) {
    std::array<int, N> primes{};
    int prime_index = 0;
    for (int i = 3; i <= max_val; ++i) {
        if (is_prime(i)) {
            primes[prime_index++] = i;
        }
    }
    return primes;
}


// Use the constexpr functions to create the static array.
constexpr int prime_count = count_primes(50);
constexpr auto primes_up_to_1000 = generate_primes<prime_count>(50);

void BM_CliffordConjugateTest(benchmark::State& state) {
    // All of the original setup code, but using the template parameter 'd'
    const int index = static_cast<int>(state.range(0));
    const int d = primes_up_to_1000[index];

    const int inv_2 = modInverse(2, d);
    const std::complex<double> omega =
        std::exp(std::complex<double>(0, 2.0 * pi / d));
    const Eigen::MatrixXcd Mprime = W(d, 1, 2, inv_2, omega) + W(d, 2, 4, inv_2, omega)
        + W(d, 3, 6, inv_2, omega) +  W(d, 4, 8, inv_2, omega);
    const Eigen::MatrixXcd C = W(d, 3, 4, inv_2, omega) * cliffordPermutationGate(d, 7);
    const Eigen::MatrixXcd Cstar = C.adjoint();
    Eigen::MatrixXcd M = C * Mprime * Cstar;
    M(0, 3) += 0.2;

    for (auto _ : state) {
        bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(d);
}

// A function to get the static data.
npy::npy_data<std::complex<double>>& getCliffordGatesD3N2NumPyData() {
    // The static variable is initialized only on the first call.
    static npy::npy_data<std::complex<double>> cliffordGateNumPyData = []() {
        std::cout << "Loading Clifford gates on two qudits (d=3)" << std::endl;
        npy::npy_data<std::complex<double>> data = npy::read_npy<std::complex<double>>("n2-c2-gates-d3-asGATES.npy");
        std::cout << "Loaded Clifford gates on two qudits (d=3)" << std::endl;
        std::cout << "Loaded Clifford gates on two qudits (d=3)" << std::endl;
        return data;
    }();
    return cliffordGateNumPyData;
}

void BM_CliffordConjugateTest2Qudits(benchmark::State& state) {
    auto& numpyData = getCliffordGatesD3N2NumPyData();
    const auto& shape = numpyData.shape;
    auto& cliffordGateNumPyData = numpyData.data;

    const size_t numMatrices = shape[0];
    const size_t matrixSize = shape[1] * shape[2];

    const int d = 3;
    const size_t n = 2;
    const long inv_2 = cliffconjtest::modInverse(2, d);
    const std::complex<double> omega =
        std::exp(std::complex<double>(0, 2.0 * cliffconjtest::pi / d));

    // This example will currently fail at i = 7 without the early checks against error propagation
    // in checkPhase
    const Eigen::MatrixXcd M = // Eigen::kroneckerProduct(M1, M2) + H
        - std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, numMatrices - 1);

    const auto i = distrib(gen);
    // Calculate start of the current block in the flattened data.
    size_t startIndex = i * matrixSize;

    // Get a pointer to the start of this block
    std::complex<double>* data_ptr = cliffordGateNumPyData.data() + startIndex;

    // zero-copy operation; numpy also column major
    Eigen::Map<Eigen::Matrix<std::complex<double>, 9, 9>> C(data_ptr);

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
        C * M * C.adjoint();

    for (auto _ : state) {
        bool result = isCliffordConjugateGeneralized(d, n, M, Mprime);
        if (!result) {
            throw std::runtime_error("Clifford conjugate generation failed for i = " + i);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(d);
}

static void BM_SortComplexity(benchmark::State& state) {
    std::vector<int> data(state.range(0));
    for (auto& x: data) x = rand();
    for (auto _: state) {
        std::sort(data.begin(), data.end());
    }
    state.SetComplexityN(state.range(0));
}

// Custom main function to register and run the benchmarks.
int main(int argc, char** argv) {
    {
        const auto& allCliffordGates = getCliffordGatesD3N2NumPyData();
        const size_t gateCount = allCliffordGates.shape[0];
        std::cout << "Loaded " << gateCount << " gates"<< std::endl;
    }

    // Register a benchmark for each odd prime inside the main function.
    /*
    for (int p : primes) {
        benchmark::RegisterBenchmark(
            "BM_CliffordConjugateTest",
            BM_CliffordConjugateTest)
            ->Args({p})->Complexity();
    }
    */
    benchmark::RegisterBenchmark(
            "BM_CliffordConjugateTest2Qudits",
            BM_CliffordConjugateTest2Qudits)
            ->Iterations(100)
            ->Unit(benchmark::kMillisecond);

    benchmark::RegisterBenchmark(
            "BM_CliffordConjugateTest",
            BM_CliffordConjugateTest)
            ->Range(0, prime_count - 1)
            ->Complexity();

    //for (int i = 1; i < 10; i++) {
        benchmark::RegisterBenchmark(
            "BM_SortComplexity",
            BM_SortComplexity)
            ->Range(1, 10)
            ->Complexity();
    //}


    const char* custom_argv[] = {
        argv[0],
        //"--benchmark_report_aggregates_only=true",
        //"--benchmark_display_aggregates_only=true"
    };
    int custom_argc = std::size(custom_argv);

    benchmark::Initialize(&custom_argc, const_cast<char**>(custom_argv));
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
