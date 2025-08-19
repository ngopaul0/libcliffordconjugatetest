#ifndef VECTORISATIONALGORITHM_H
#define VECTORISATIONALGORITHM_H

#include <Eigen/Dense>
#include "internal/util.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace cliffconjtest {

inline Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> createXtransposeTensorI(const Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>& X) {
    if (X.rows() % 2 != 0) {
        throw std::invalid_argument("Vector size must be even");
    }
    const size_t n = X.rows() / 2;

    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> identity(X.rows(), X.rows());
    for (size_t i = 0; i < identity.rows(); ++i) {
        for (size_t j = 0; j < identity.cols(); ++j) {
            if (i == j) {
                identity(i, j) = 1;
            } else {
                identity(i, j) = 0;
            }
        }
    }

    Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> theResult = kroneckerProduct(X.transpose(), identity);
    return theResult;
}

//inline Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> findSymplectic() {

//}

} // namespace cliffconjtest

#endif //VECTORISATIONALGORITHM_H
