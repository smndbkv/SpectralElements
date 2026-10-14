#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "sem/core/dense_matrix.hpp"
#include "sem/core/types.hpp"

namespace sem::test {

inline Real maxAbsDiff(const DenseMatrix& a, const DenseMatrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("maxAbsDiff: matrix sizes differ");
    }
    Real result = 0;
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            result = std::max(result, std::abs(a(i, j) - b(i, j)));
        }
    }
    return result;
}

// n равномерно распределённых точек на [a, b], включая концы.
inline std::vector<Real> linspace(Real a, Real b, int n) {
    std::vector<Real> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = a + (b - a) * i / (n - 1);
    }
    return x;
}

// Точное значение интеграла x^k по [-1, 1].
inline Real monomialIntegral(int k) {
    return k % 2 == 1 ? 0.0 : 2.0 / (k + 1);
}

} // namespace sem::test
