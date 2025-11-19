#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <strstream>
#include <unordered_set>

#include "unsupported/Eigen/KroneckerProduct"
#include <Eigen/Dense>

#include "internal/cliffordgates.h"

// validates the symplectic inner product function against the naive multiplication x^t Omega y
#define CHECK_SYMPLECTIC_INNER_PRODUCT false
#define FINE_GRAINED_RESULT_WRITING true
#define SIMULATE_BASIS_FINDING true

void modReduce(Eigen::MatrixXi& M, int d) {
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            long long x = M(i, j);
            long long r = x % d;
            if (r < 0) r += d;
            M(i, j) = static_cast<int>(r);
        }
    }
}

void modReduce(Eigen::VectorXi& v, int d) {
    for (int i = 0; i < v.size(); ++i) {
        long long x = v(i);
        long long r = x % d;
        if (r < 0) r += d;
        v(i) = static_cast<int>(r);
    }
}

namespace symplectic {

Eigen::MatrixXi canonicalSymplecticForm(int n, int d) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive.");
    }
    if (d <= 1) {
        throw std::invalid_argument("d must be a prime >= 2.");
    }
    int m = 2 * n;
    Eigen::MatrixXi J = Eigen::MatrixXi::Zero(m, m);

    // Top-right block: I
    J.block(0, n, n, n).setIdentity();
    // Bottom-left block: -I mod d == (d-1)*I
    J.block(n, 0, n, n).setIdentity();
    J.block(n, 0, n, n) = ((d - 1) * J.block(n, 0, n, n)).eval();
    modReduce(J, d);
    return J;
}

template <class URNG>
Eigen::VectorXi randomNonzeroVector(int m, int d, URNG& rng) {
    if (m <= 0) {
        throw std::invalid_argument("m must be positive.");
    }
    std::uniform_int_distribution<int> dist(0, d - 1);
    Eigen::VectorXi v(m);
    while (true) {
        for (int i = 0; i < m; ++i) {
            v(i) = dist(rng);
        }
        // Check not all zero
        bool allZero = true;
        for (int i = 0; i < m; ++i) {
            if (v(i) != 0) {
                allZero = false;
                break;
            }
        }
        if (!allZero) {
            return v;
        }
    }
}

void applyLeftTransvection(Eigen::MatrixXi& A,
                                  const Eigen::VectorXi& v_in,
                                  const Eigen::MatrixXi& J,
                                  int d) {
    const int m = A.rows();
    if (A.cols() != m || v_in.size() != m || J.rows() != m || J.cols() != m) {
        throw std::invalid_argument("Dimension mismatch in applyLeftTransvection.");
    }
    // Reduce inputs just to be safe
    Eigen::VectorXi v = v_in;
    modReduce(v, d);

    // Compute S = v^T J A  (1 x m row vector)
    // We do: tmp = J * A (m x m), then S = v^T * tmp
    Eigen::MatrixXi tmp = (J * A).unaryExpr([&](int x){ return ((x % d) + d) % d; });
    Eigen::RowVectorXi S = (v.transpose() * tmp);
    for (int i = 0; i < S.size(); ++i) {
        long long r = S(i) % d;
        if (r < 0) r += d;
        S(i) = static_cast<int>(r);
    }

    // Rank-1 update: A += v * S
    // (v * S) is (m x 1) * (1 x m) = (m x m)
    A.noalias() += v * S;
    modReduce(A, d);
}

bool isSymplectic(const Eigen::MatrixXi& A, const Eigen::MatrixXi& J, int d) {
    if (A.rows() != A.cols() || J.rows() != J.cols() || A.rows() != J.rows()) {
        return false;
    }
    Eigen::MatrixXi test = A.transpose() * J * A;
    modReduce(test, d); // safe: we discard local copy
    // Compare with J mod d
    Eigen::MatrixXi Jmod = J;
    modReduce(Jmod, d);
    return test == Jmod;
}

template <class URNG>
Eigen::MatrixXi randomSymplecticMatrix(int n, int d, URNG& rng, int steps) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive.");
    }
    if (d <= 1) {
        throw std::invalid_argument("d must be a prime >= 2.");
    }
    int m = 2 * n;
    if (steps < 0) {
        steps = 10 * n;
    }

    Eigen::MatrixXi J = canonicalSymplecticForm(n, d);
    Eigen::MatrixXi A = Eigen::MatrixXi::Identity(m, m);

    for (int i = 0; i < steps; ++i) {
        Eigen::VectorXi v = randomNonzeroVector(m, d, rng);
        applyLeftTransvection(A, v, J, d);
    }

    modReduce(A, d);

    if (!isSymplectic(A, J, d)) {
        throw std::runtime_error("Generated matrix failed symplectic check (numerical bug).");
    }
    return A;
}

} // namespace symplectic

namespace sampling {

/* Hash for std::vector<int> so we can use it in unordered_set for deduplication. */
struct VecHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        // 64-bit FNV-1a-like mixing with integer hashing
        std::size_t h = 1469598103934665603ull;
        for (int x : v) {
            std::size_t y = std::hash<int>{}(x);
            h ^= y + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h *= 1099511628211ull;
        }
        return h;
    }
};

/*
 * Return: std::vector<Eigen::MatrixXi> of length m,
 *         each element is a (1 x 2n) row matrix with entries in [0, d-1].
 *
 * Seed: pass an optional 64-bit seed; if seed == std::nullopt, we seed from std::random_device.
 */
inline std::vector<Eigen::MatrixXi>
random_distinct_vectors_Zd(int d, int n, int m, std::optional<std::uint64_t> seed = std::nullopt)
{
    const int dim = 2 * n;

    // RNG setup
    std::mt19937_64 rng;
    if (seed.has_value()) {
        rng.seed(*seed);
    } else {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        rng.seed(seq);
    }
    std::uniform_int_distribution<int> dist(0, d - 1);

    // Deduplication set keyed by the raw coordinate vector
    std::unordered_set<std::vector<int>, VecHash> seen;
    seen.reserve(static_cast<std::size_t>(m) * 2);

    std::vector<Eigen::MatrixXi> out;
    out.reserve(static_cast<std::size_t>(m));

    while (static_cast<int>(out.size()) < m) {
        // Sample a candidate vector in (Z_d)^(2n)
        std::vector<int> key;
        key.resize(static_cast<std::size_t>(dim));
        for (int i = 0; i < dim; ++i) {
            key[static_cast<std::size_t>(i)] = dist(rng);
        }

        // Try to insert (deduplicate)
        if (auto [it, inserted] = seen.insert(key); !inserted) {
            continue; // duplicate, resample
        }

        // Convert to Eigen vector (1 x 2n)
        Eigen::MatrixXi row(dim, 1);
        for (int i = 0; i < dim; i++) {
            row(i, 0) = key[static_cast<std::size_t>(i)];
        }
        out.emplace_back(std::move(row));
    }

    return out;
}

} // namespace sampling

namespace printing {

// Format seconds as h:mm:ss
static std::string fmt_hms(double seconds) {
    if (seconds < 0 || !std::isfinite(seconds)) return "--:--:--";
    auto s = static_cast<std::uint64_t>(seconds + 0.5);
    const std::uint64_t h = s / 3600; s %= 3600;
    const std::uint64_t m = s / 60;   s %= 60;
    std::ostringstream oss;
    oss << h << ':' << std::setw(2) << std::setfill('0') << m
        << ':' << std::setw(2) << std::setfill('0') << s;
    return oss.str();
}

// Print a single-line progress bar (overwrites the same line)
// Call with current in [0,total]. Prints a newline automatically when current == total.
inline void print_progress(std::uint64_t current,
                           std::uint64_t total,
                           const std::chrono::steady_clock::time_point& start,
                           bool clearLine = false,
                           std::size_t bar_width = 40) {
    static std::size_t prev_len = 0;

    if (clearLine) {
        if (prev_len > 0) {
            std::cout << '\r' << std::string(prev_len, ' ') << '\r' << std::flush;
            prev_len = 0;
        }
        return;
    }

    if (total == 0) return;

    double ratio = std::min<double>(1.0, static_cast<double>(current) / static_cast<double>(total));
    int filled = static_cast<int>(std::round(ratio * bar_width));

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start).count();
    double speed = (elapsed > 0) ? (current / elapsed) : 0.0; // units per second
    double eta = (speed > 0 && current < total) ? (double(total - current) / speed) : 0.0;

    // Build bar: [=====>-----]
    std::ostringstream bar;
    bar << '[';
    for (int i = 0; i < static_cast<int>(bar_width); ++i) {
        if (i < filled) bar << '=';
        else if (i == filled) bar << '>';
        else bar << '-';
    }
    bar << ']';

    // Compose line
    std::ostringstream oss;
    oss << bar.str() << ' '
        << std::setw(6) << std::fixed << std::setprecision(2) << (ratio * 100.0) << "%  "
        << current << '/' << total << "  "
        << std::fixed << std::setprecision(2) << speed << "/s  "
        << "elapsed " << fmt_hms(elapsed) << "  "
        << "ETA " << (current < total ? fmt_hms(eta) : "00:00:00");

    std::string line = oss.str();

    // Write line in-place
    std::cout << '\r' << line;
    if (prev_len > line.size()) {
        std::cout << std::string(prev_len - line.size(), ' ');
    }
    std::cout << std::flush;
    prev_len = line.size();

    //if (current >= total) {
    //    std::cout << '\n';
    //    prev_len = 0;
    //}
}

} // namespace printing

/**
 * Tests using the necessary condition whether v = S*u is possible.
 */
bool necessary_condition_check(
    int d,
    const std::vector<Eigen::MatrixXi>& U,
    const std::vector<Eigen::MatrixXi>& V,
    const Eigen::MatrixXi& omega,
    const Eigen::MatrixXi& u,
    const Eigen::MatrixXi& v
) {
    std::vector<size_t> innerProductHistogramForU(d);
    for (const auto& x : U) {
        // Using this faster way brought the time down for d = 2, n = 5, k = 40
        // from 10.5ms to 4.7ms
        const long innerProdVal = cliffconjtest::symplecticProductMultiQudit(d, u, x);
#if CHECK_SYMPLECTIC_INNER_PRODUCT
        const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>& innerProduct =
            u.transpose() * omega * x;
        if (innerProduct.rows() != 1 && innerProduct.cols() != 1) {
            throw std::runtime_error("bad symplectic inner product shape");
        }
        const auto slowVal = cliffconjtest::safeMod(innerProduct(0, 0), d);
        if (innerProdVal != slowVal) {
            std::stringstream ss;
            ss << innerProdVal << "!=" << slowVal;
            throw std::runtime_error("symplectic inner product doesn't match " + ss.str());
        }
#endif
        innerProductHistogramForU[innerProdVal] += 1;
    }

    std::vector<size_t> innerProductHistogramForV(d);
    for (const auto& x : V) {
        const long innerProdVal = cliffconjtest::symplecticProductMultiQudit(d, v, x);
#if CHECK_SYMPLECTIC_INNER_PRODUCT
        const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>& innerProduct =
            v.transpose() * omega * x;
        if (innerProduct.rows() != 1 && innerProduct.cols() != 1) {
            throw std::runtime_error("bad symplectic inner product shape");
        }
        const auto slowVal = cliffconjtest::safeMod(innerProduct(0, 0), d);
        if (innerProdVal != slowVal) {
            std::stringstream ss;
            ss << innerProdVal << "!=" << slowVal;
            throw std::runtime_error("symplectic inner product doesn't match " + ss.str());
        }
#endif
        innerProductHistogramForV[innerProdVal] += 1;
        if (innerProductHistogramForV[innerProdVal] > innerProductHistogramForU[innerProdVal]) {
            return false;
        }
    }

    return innerProductHistogramForU == innerProductHistogramForV;
}

size_t count_necessary_cond_accepts(
    int d,
    const std::vector<Eigen::MatrixXi>& U,
    const std::vector<Eigen::MatrixXi>& V,
    const Eigen::MatrixXi& omega,
    size_t u_index
) {
    auto u_now = U[u_index];
    std::atomic_size_t count = 0;

#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < V.size(); i++) {
        const auto& v = V[i];
        // tests whether v = S*u is possible via necessary condition
        if (necessary_condition_check(d, U, V, omega, u_now, v)) {
            ++count;
        }
    }

    return count;
}

std::pair<std::vector<size_t>, std::vector<size_t>> runTrial(
    size_t trial_index, size_t num_trials, std::atomic_size_t& vectors_finished,
    const std::chrono::steady_clock::time_point& trials_start_time,
    int n, int d, const std::vector<Eigen::MatrixXi>& U,
    const std::vector<Eigen::MatrixXi>& V, const Eigen::MatrixXi& omega,
    const std::string& output_results_filename
) {

    std::vector<size_t> all_counts(U.size());
    std::vector<size_t> all_times(U.size());

    const std::chrono::time_point<std::chrono::steady_clock> start =
        std::chrono::steady_clock::now();
    auto lastPrintTime = std::chrono::high_resolution_clock::now();

    const size_t vectors_to_take = SIMULATE_BASIS_FINDING ? 2 * n : U.size();

// making it parallel here is not close to how the algorithm is done in practice, since the u's
// are normally checked for linear independence in order to form a basis of 2n vectors
// #pragma omp parallel for schedule(dynamic)
    for (size_t u_index = 0; u_index < vectors_to_take; ++u_index) {
        auto startTime = std::chrono::high_resolution_clock::now();

        auto count = count_necessary_cond_accepts(d, U, V, omega, u_index);

        auto endTime = std::chrono::high_resolution_clock::now();
        ++vectors_finished;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        all_counts[u_index] = count;
        all_times[u_index] = us.count();

        auto thisTime = std::chrono::high_resolution_clock::now();
        auto msSinceLastPrint = std::chrono::duration_cast<std::chrono::milliseconds>(thisTime - lastPrintTime);
        if (u_index == 0 || msSinceLastPrint.count() > 50) {

#if SIMULATE_BASIS_FINDING
            printing::print_progress(vectors_finished,vectors_to_take * num_trials,
                trials_start_time);
#else
            printing::print_progress(u_index + 1, vectors_to_take, start);
#endif
        }

#if FINE_GRAINED_RESULT_WRITING
        if (std::ofstream resultFile(output_results_filename, std::ios::app); resultFile) {
            resultFile << count << "," << us.count() << std::endl;
        } else {
            std::cerr << "Failed to open " << output_results_filename << " for writing" << std::endl;
        }
#endif

    }
#if !SIMULATE_BASIS_FINDING
    printing::print_progress(0, 1, start, true);
#endif


#if !SIMULATE_BASIS_FINDING
    auto the_mean = mean(times);
    double mean_time_to_display = the_mean > 1000 ? the_mean / 1000 : the_mean;
    std::string unit = the_mean > 1000 ? "ms" : "us";
    std::cout << "Trial " << trial_index << " done" << std::endl;
#endif

    return std::make_pair(all_counts, all_times);
}


int main(int argc, char** argv) {
    std::cout << "Running necessary_condition test (argc: " << argc << ")" << std::endl;
    int d = 2;
    int n = 4;
    int k = 4;
    int num_trials = SIMULATE_BASIS_FINDING ? 200 : 100;
    if (argc == 1) {
        // defaults defined above
    } else if (argc == 4 || argc == 5) {
        d = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        if (argc == 5) {
            num_trials = atoi(argv[4]);
        }
    } else {
        std::cerr << "Not enough arguments. Expected " << argv[0] << "d n k [num_trials], " <<
            "where d is prime and n,k is pos int. Trials over Z_d^(2n) " <<
            "with random subset of size d^(2n) - k" << std::endl;
        return 1;
    }

    const int vector_space_size = cliffconjtest::computeIntegralPower(d, 2 * n);
    const int set_size = vector_space_size - k;

    std::cout << "d = " << d << ", n = " << n << ", k = " << k << std::endl;
    std::cout << "Size of Z_d^(2n): " << vector_space_size << std::endl;
    std::cout << "Size of U (|Z_d^(2n)| - k): " << set_size << std::endl;
    std::cout << "Running " << num_trials << " trials where " << (SIMULATE_BASIS_FINDING ? (2 * n) : set_size) << " vectors are selected in each" << std::endl;

    const Eigen::MatrixXi Omega = symplectic::canonicalSymplecticForm(n, d);


    std::string output_results_filename;
    {
        std::stringstream ss_filename;
        const auto startUnixEpochMs =  std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch());
        const std::string basis_finding_str = SIMULATE_BASIS_FINDING ? "bf" : "";
        ss_filename << "necesscondtrial" << "-d" << d << "n" << n
            << "k" << k << "trials" << num_trials
            << basis_finding_str << "-" << startUnixEpochMs.count() << ".csv";
        output_results_filename = ss_filename.str();

        if (std::ofstream file(output_results_filename); file) {
            file << "VectorsAcceptedCount,TimeFor_u_FilteringInMicroS" << std::endl;
        } else {
            std::cerr << "Failed to open file for writing" << std::endl;
        }
    }
    std::cout << "Writing results to " << output_results_filename << std::endl;

    const auto start = std::chrono::steady_clock::now();


    std::mutex mu;

    std::atomic_size_t vectors_finished = 0;

    for (int trial_num = 0; trial_num < num_trials; trial_num++) {
        std::mt19937 rng;
        Eigen::MatrixXi S = symplectic::randomSymplecticMatrix(n, d, rng, 40);
        std::vector<Eigen::MatrixXi> U = sampling::random_distinct_vectors_Zd(d, n, set_size);
        if (U.size() != set_size) {
            throw std::invalid_argument("U must be a vector of size N.");
        }
        std::vector<Eigen::MatrixXi> V;
        V.reserve(set_size);
        for (const auto& u : U) {
            assert(u.rows() == 2 * n);
            assert(u.cols() == 1);

            Eigen::MatrixXi multiply = S * u;
            modReduce(multiply, d);
            assert(multiply.rows() == 2 * n);
            assert(multiply.cols() == 1);
            V.push_back(multiply);
        }

        assert(U.size() == V.size());

        auto [counts, times] = runTrial(trial_num, num_trials,
            vectors_finished, start, n, d, U, V, Omega, output_results_filename);

        if (counts.size() != times.size()) {
            throw std::invalid_argument("counts size different from times");
        }

#if !FINE_GRAINED_RESULT_WRITING
        {
            std::lock_guard lk(mu);
            if (std::ofstream resultFile(output_results_filename, std::ios::app); resultFile) {
                for (size_t idx = 0; idx < counts.size(); idx++) {
                    auto count = counts[idx];
                    auto meanTimeMs = times[idx];
                    resultFile << count << "," << meanTimeMs << std::endl;
                }
            } else {
                std::cerr << "Failed to open " << output_results_filename << " for writing" << std::endl;
            }
        }
#endif

    }
#if SIMULATE_BASIS_FINDING
    printing::print_progress(0, 1, start, true);
#endif

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start).count();

    std::cout << num_trials << " trials (" << vectors_finished <<  " vectors) all done in "
        << printing::fmt_hms(elapsed) << std::endl;
    std::cout << "Wrote all accepted vectors to " << output_results_filename << std::endl;

    return 0;
}
