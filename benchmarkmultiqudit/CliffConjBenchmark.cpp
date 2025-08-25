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
#include "internal/complexexactrepr.h"

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
    M(0, 2) += 0.2;

    for (auto _ : state) {
        bool result = isCliffordConjugateGeneralized(d, 1, M, Mprime);
        if (result) {
            const std::string failMsg = "Unexpected Clifford-conjugate failure (d = " + std::to_string(d) + ")";
            state.SkipWithError(failMsg);
            break; // REQUIRED to prevent all further iterations.
        }
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
            const std::string failMsg = "Unexpected Clifford-conjugate failure (i = " + std::to_string(i) + ")";
            state.SkipWithError(failMsg);
            break; // REQUIRED to prevent all further iterations.
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(d);
}

/*
Processing Gate i=106092, seed = 13691651030999647805:
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(),
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(),
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx();

 */

void BM_CliffordConjugateTest2Qudits_ExtremelySlow2(benchmark::State& state) {
    const int d = 3;
    const size_t n = 2;
    const long inv_2 = modInverse(2, d);
    const std::complex<double> omega =
        std::exp(std::complex<double>(0, 2.0 * cliffconjtest::pi / d));

    // This example will currently fail at i = 7 without the early checks against error propagation
    // in checkPhase
    const Eigen::MatrixXcd M = // Eigen::kroneckerProduct(M1, M2) + H
        - std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));

    const std::mt19937::result_type seedValue = 13691651030999647805;

    Eigen::Matrix<std::complex<double>, 9, 9> C;
    C << ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(),
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(),
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx();

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
        C * M * C.adjoint();

    for (auto _ : state) {
        bool result = isCliffordConjugateGeneralized(d, n, M, Mprime, std::make_optional(seedValue));
        if (!result) {
            const std::string failMsg = "Unexpected Clifford-conjugate failure";
            state.SkipWithError(failMsg);
            break; // REQUIRED to prevent all further iterations.
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(d);
}

void BM_CliffordConjugateTest2Qudits_ExtremelySlow(benchmark::State& state) {
    auto& numpyData = getCliffordGatesD3N2NumPyData();
    const auto& shape = numpyData.shape;
    auto& cliffordGateNumPyData = numpyData.data;

    const size_t numMatrices = shape[0];
    const size_t matrixSize = shape[1] * shape[2];

    const int d = 3;
    const size_t n = 2;
    const long inv_2 = modInverse(2, d);
    const std::complex<double> omega =
        std::exp(std::complex<double>(0, 2.0 * cliffconjtest::pi / d));

    // This example will currently fail at i = 7 without the early checks against error propagation
    // in checkPhase
    const Eigen::MatrixXcd M = // Eigen::kroneckerProduct(M1, M2) + H
        - std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));

    // const size_t i = 114515;
    // const unsigned int seedVal = 17315759047394915809;

   //  const size_t i = 112547;
    // const std::mt19937::result_type seedVal = 11134845882598934268;
    //const size_t i = 23590;
    //const std::mt19937::result_type seedVal = 8388112314913335105;
    //const size_t i = 106092;
    // const std::mt19937::result_type seedVal = 13691651030999647805;

    const size_t i = 115494;
    const std::mt19937::result_type seedVal = 1758570121290292570;
    /*
Processing Gate i=115494, seed = 1758570121290292570:
ComplexExactRepr(4599676419421066579,0).cmplx(), ComplexExactRepr(-4628199217061079723,-4624500108022500582).cmplx(), ComplexExactRepr(4599676419421066580,-4854264943845182893).cmplx(), ComplexExactRepr(-4628199217061079724,4598871928832275226).cmplx(), ComplexExactRepr(4599676419421066579,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275231).cmplx(), ComplexExactRepr(-4628199217061079724,-4624500108022500580).cmplx(), ComplexExactRepr(-4628199217061079723,4598871928832275227).cmplx(), ComplexExactRepr(-4628199217061079717,-4624500108022500581).cmplx(),
ComplexExactRepr(-4628199217061079723,-4624500108022500579).cmplx(), ComplexExactRepr(-4628199217061079724,4598871928832275227).cmplx(), ComplexExactRepr(-4628199217061079719,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079714,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079728,4598871928832275227).cmplx(), ComplexExactRepr(-4628199217061079717,-4624500108022500585).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500582).cmplx(), ComplexExactRepr(-4628199217061079732,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079714,-4624500108022500584).cmplx(),
ComplexExactRepr(-4628199217061079725,4598871928832275228).cmplx(), ComplexExactRepr(4599676419421066583,4361526221081163428).cmplx(), ComplexExactRepr(-4628199217061079728,4598871928832275228).cmplx(), ComplexExactRepr(4599676419421066579,-4850001498709076653).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500582).cmplx(), ComplexExactRepr(4599676419421066578,-4847562048910917632).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079742,4598871928832275233).cmplx(), ComplexExactRepr(-4628199217061079705,-4624500108022500587).cmplx(),
ComplexExactRepr(-4628199217061079728,-4624500108022500578).cmplx(), ComplexExactRepr(4599676419421066582,-4863887597560135680).cmplx(), ComplexExactRepr(4599676419421066582,4370743438363066368).cmplx(), ComplexExactRepr(-4628199217061079723,-4624500108022500582).cmplx(), ComplexExactRepr(4599676419421066580,-4854063821197985058).cmplx(), ComplexExactRepr(4599676419421066579,-4875466071359286940).cmplx(), ComplexExactRepr(-4628199217061079721,-4624500108022500583).cmplx(), ComplexExactRepr(4599676419421066582,-4847562048910917632).cmplx(), ComplexExactRepr(4599676419421066579,-4857132198119079936).cmplx(),
ComplexExactRepr(-4628199217061079720,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066581,-4849813848724602880).cmplx(), ComplexExactRepr(4599676419421066583,-4852628598491709440).cmplx(), ComplexExactRepr(-4628199217061079724,4598871928832275227).cmplx(), ComplexExactRepr(-4628199217061079719,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079723,-4624500108022500579).cmplx(), ComplexExactRepr(4599676419421066579,-4850376798678024192).cmplx(), ComplexExactRepr(-4628199217061079738,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079733,4598871928832275232).cmplx(),
ComplexExactRepr(-4628199217061079708,-4624500108022500584).cmplx(), ComplexExactRepr(4599676419421066581,-4845310249097232384).cmplx(), ComplexExactRepr(4599676419421066581,-4848124998864338944).cmplx(), ComplexExactRepr(4599676419421066583,4360721730492372092).cmplx(), ComplexExactRepr(-4628199217061079728,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079725,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079736,4598871928832275234).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500585).cmplx(),
ComplexExactRepr(-4628199217061079728,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079722,4598871928832275227).cmplx(), ComplexExactRepr(4599676419421066580,4356482039543059795).cmplx(), ComplexExactRepr(4599676419421066582,-4863887597560135680).cmplx(), ComplexExactRepr(4599676419421066582,4370743438363066368).cmplx(), ComplexExactRepr(-4628199217061079728,-4624500108022500578).cmplx(), ComplexExactRepr(-4628199217061079716,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079721,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(),
ComplexExactRepr(-4628199217061079716,-4624500108022500583).cmplx(), ComplexExactRepr(-4628199217061079712,-4624500108022500580).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275228).cmplx(), ComplexExactRepr(4599676419421066581,-4849813848724602880).cmplx(), ComplexExactRepr(4599676419421066583,-4852628598491709440).cmplx(), ComplexExactRepr(-4628199217061079720,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079725,4598871928832275229).cmplx(), ComplexExactRepr(4599676419421066579,-4856381598181184856).cmplx(),
ComplexExactRepr(4599676419421066580,-4846999098957496320).cmplx(), ComplexExactRepr(4599676419421066580,-4850376798678024193).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500580).cmplx(), ComplexExactRepr(4599676419421066581,-4845310249097232384).cmplx(), ComplexExactRepr(4599676419421066581,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500584).cmplx(), ComplexExactRepr(4599676419421066580,-4850376798678024196).cmplx(), ComplexExactRepr(4599676419421066580,-4852628598491709440).cmplx(), ComplexExactRepr(-4628199217061079719,-4624500108022500579).cmplx();
     */

    //const size_t i = 115898;
    //const std::mt19937::result_type seedVal = 1734907290151951969;

    // Calculate start of the current block in the flattened data.
    size_t startIndex = i * matrixSize;

    // Get a pointer to the start of this block
    std::complex<double>* data_ptr = cliffordGateNumPyData.data() + startIndex;

    // zero-copy operation; numpy also column major
    Eigen::Map<Eigen::Matrix<std::complex<double>, 9, 9>> C(data_ptr);
    /*
     Processing Gate i=106092, seed = 13691651030999647805:
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(4599676419421066582,-4871321393266982593).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275229).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066583,-4853962019326894909).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275230).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079722,-4624500108022500581).cmplx(), ComplexExactRepr(4599676419421066583,-4853630297885617527).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(),
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(-4628199217061079730,4598871928832275228).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079718,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079710,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079713,-4624500108022500581).cmplx(), ComplexExactRepr(-4628199217061079708,-4624500108022500586).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(),
ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(),
ComplexExactRepr(-4628199217061079727,4598871928832275230).cmplx(), ComplexExactRepr(-4628199217061079734,4598871928832275232).cmplx(), ComplexExactRepr(4599676419421066579,-4842495499330125824).cmplx(), ComplexExactRepr(4599676419421066582,0).cmplx(), ComplexExactRepr(4599676419421066579,-4848124998864338944).cmplx(), ComplexExactRepr(-4628199217061079702,-4624500108022500586).cmplx(), ComplexExactRepr(-4628199217061079727,4598871928832275225).cmplx(), ComplexExactRepr(-4628199217061079731,4598871928832275231).cmplx(), ComplexExactRepr(4599676419421066578,-4842776974306836480).cmplx();

     */

    /*
     *i = 23590, seed 8388112314913335105
ComplexExactRepr(4603375528459645722,0).cmplx(), ComplexExactRepr(4603375528459645726,0).cmplx(), ComplexExactRepr(4603375528459645723,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(),
ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(-4624500108022500581,-4620693217682128902).cmplx(), ComplexExactRepr(-4624500108022500577,-4620693217682128895).cmplx(), ComplexExactRepr(-4624500108022500580,-4620693217682128900).cmplx(),
ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(4603375528459645721,-4854183689018395878).cmplx(), ComplexExactRepr(4603375528459645725,-4854183689018395874).cmplx(), ComplexExactRepr(4603375528459645722,-4854183689018395877).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(),
ComplexExactRepr(-4624500108022500579,4602678819172646910).cmplx(), ComplexExactRepr(-4624500108022500581,-4620693217682128898).cmplx(), ComplexExactRepr(4603375528459645726,-4854532043661895281).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(),
ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(4603375528459645723,-4863190888273136869).cmplx(), ComplexExactRepr(-4624500108022500583,4602678819172646910).cmplx(), ComplexExactRepr(-4624500108022500575,-4620693217682128896).cmplx(),
ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(-4624500108022500582,4602678819172646909).cmplx(), ComplexExactRepr(-4624500108022500580,-4620693217682128902).cmplx(), ComplexExactRepr(4603375528459645725,-4849854266712775082).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(),
ComplexExactRepr(-4624500108022500578,4602678819172646914).cmplx(), ComplexExactRepr(4603375528459645725,4364684748209009435).cmplx(), ComplexExactRepr(-4624500108022500579,-4620693217682128899).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(),
ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(4603375528459645726,-4852883611789803546).cmplx(), ComplexExactRepr(-4624500108022500579,-4620693217682128896).cmplx(), ComplexExactRepr(-4624500108022500585,4602678819172646912).cmplx(),
ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(-4624500108022500581,4602678819172646914).cmplx(), ComplexExactRepr(4603375528459645724,-4858687288645766367).cmplx(), ComplexExactRepr(-4624500108022500577,-4620693217682128903).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx(), ComplexExactRepr(0,0).cmplx();
     */

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
        C * M * C.adjoint();

    for (auto _ : state) {
        bool result = isCliffordConjugateGeneralized(d, n, M, Mprime, std::make_optional(seedVal));
        if (!result) {
            const std::string failMsg = "Unexpected Clifford-conjugate failure (i = " + std::to_string(i) + ")";
            state.SkipWithError(failMsg);
            break; // REQUIRED to prevent all further iterations.
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(d);
}

bool bruteForceTest(const size_t d, const std::complex<double>& omega, const Eigen::MatrixXcd& M, const Eigen::MatrixXcd& Mprime, const std::vector<size_t>& indices) {
    auto& numpyData = getCliffordGatesD3N2NumPyData();
    const auto& shape = numpyData.shape;
    auto& cliffordGateNumPyData = numpyData.data;

    const size_t matrixSize = shape[1] * shape[2];
    const size_t numMatrices = shape[0];
    if (indices.size() != numMatrices) {
        throw std::invalid_argument("indices should be the same length");
    }

    for (const size_t i : indices) {
        const size_t startIndex = i * matrixSize;
        std::complex<double>* data_ptr = cliffordGateNumPyData.data() + startIndex;
        Eigen::Map<Eigen::Matrix<std::complex<double>, 9, 9>> C(data_ptr);
        Eigen::MatrixXcd MprimeTest = C * M * C.adjoint();

        if (MprimeTest.isApprox(Mprime)) {
            return true;
        }
        /*
        for (int k = 1; k < d; k++) {
            const auto omegaPow = std::pow(omega, k);
            Eigen::MatrixXcd MprimeTestWithPhase = (omegaPow * C) * M * (omegaPow * C).adjoint();
            if (MprimeTestWithPhase.isApprox(Mprime)) {
                return true;
            }
        }
        */
    }

    return false;
}

void BM_CliffordConjugateTest2Qudits_BruteForce(benchmark::State& state) {
    auto& numpyData = getCliffordGatesD3N2NumPyData();
    const auto& shape = numpyData.shape;
    auto& cliffordGateNumPyData = numpyData.data;

    const size_t numMatrices = shape[0];
    const size_t matrixSize = shape[1] * shape[2];

    const int d = 3;
    const size_t n = 2;
    const long inv_2 = modInverse(2, d);
    const std::complex<double> omega =
        std::exp(std::complex<double>(0, 2.0 * cliffconjtest::pi / d));

    // This example will currently fail at i = 7 without the early checks against error propagation
    // in checkPhase
    const Eigen::MatrixXcd M = // Eigen::kroneckerProduct(M1, M2) + H
        - std::sqrt(2) * Eigen::kroneckerProduct(cliffconjtest::W(d, 1, 3, inv_2, omega), cliffconjtest::W(d, 1, 3, inv_2, omega));

    const size_t i = numMatrices - 1;
    const std::mt19937::result_type seedVal = 1758570121290292570;

    // Calculate start of the current block in the flattened data.
    size_t startIndex = i * matrixSize;

    // Get a pointer to the start of this block
    std::complex<double>* data_ptr = cliffordGateNumPyData.data() + startIndex;

    // zero-copy operation; numpy also column major
    Eigen::Map<Eigen::Matrix<std::complex<double>, 9, 9>> C(data_ptr);

    const std::complex<double> omegaPower = std::pow(omega, 1);

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Mprime =
        (omegaPower * C) * M *(omegaPower * C).adjoint();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indices(numMatrices);
    // fill with indices
    std::iota(indices.begin(), indices.end(), 0);
    //std::ranges::shuffle(indices, gen);


    for (auto _ : state) {
        bool result = bruteForceTest(d, omega, M, Mprime, indices);
        if (!result) {
            const std::string failMsg = "Unexpected Clifford-conjugate failure (i = " + std::to_string(i) + ")";
            state.SkipWithError(failMsg);
            break; // REQUIRED to prevent all further iterations.
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
    benchmark::MaybeReenterWithoutASLR(argc, argv);

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
            "BM_CliffordConjugateTest2Qudits_BruteForce",
            BM_CliffordConjugateTest2Qudits_BruteForce)
            // ->Threads(32)
            ->Unit(benchmark::kNanosecond);

    benchmark::RegisterBenchmark(
            "BM_CliffordConjugateTest2Qudits_ExtremelySlow",
            BM_CliffordConjugateTest2Qudits_ExtremelySlow)
            // ->Threads(32)
            ->Unit(benchmark::kNanosecond);

    benchmark::RegisterBenchmark(
            "BM_CliffordConjugateTest2Qudits_ExtremelySlow2",
            BM_CliffordConjugateTest2Qudits_ExtremelySlow2)
            // ->Threads(32)
            ->Unit(benchmark::kNanosecond);

    benchmark::RegisterBenchmark(
            "BM_CliffordConjugateTest2Qudits",
            BM_CliffordConjugateTest2Qudits)
            // ->Threads(32)
            // ->Iterations(5000)
            ->Unit(benchmark::kNanosecond);


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

    /*
    const char* custom_argv[] = {
        argv[0],
        //"--benchmark_report_aggregates_only=true",
        //"--benchmark_display_aggregates_only=true"
    };
    int custom_argc = std::size(custom_argv);
    */

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    benchmark::Shutdown();
    return 0;
}
