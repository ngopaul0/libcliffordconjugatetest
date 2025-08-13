#include "cliffordgates.h"
#include <Eigen/Dense>

#include "symplecticiterator.h"
#include "util.h"

namespace cliffconjtest {

Eigen::MatrixXcd cliffordPermutationGate(int d, int a) {
    Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(d, d);
    // Iterate through the columns, i.e. iterate and get where |j> would be sent to, |a*j mod d>
    for (int j = 0; j < d; j++) {
        int i = safeMod(a * j, d);
        gate(i, j) = 1.0;
    }
    return gate;
}

Eigen::MatrixXcd hadamardGate(int d, const std::complex<double>& omega) {
    Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(d, d);
    const double outsideFactor = 1.0 / d;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            gate(i, j) = outsideFactor * std::pow(omega, (i * j) % d);
        }
    }
    return gate;
}

Eigen::MatrixXcd diagonalSymplecticCliffordGate(int d, int b, int inv_2,
                                                const std::complex<double>& omega) {
    Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(d, d);
    for (int z = 0; z < d; z++) {
        gate(z, z) = std::pow(omega, safeMod(inv_2 * b * z * z, d));
    }
    return gate;
}

} // namespace cliffconjtest
