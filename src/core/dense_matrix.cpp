#include "sem/core/dense_matrix.hpp"

#include <stdexcept>

namespace sem {

DenseMatrix::DenseMatrix(int rows, int cols, Real value)
    : rows_(rows), cols_(cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("DenseMatrix: dimensions must be non-negative");
    }
    data_.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), value);
}

DenseMatrix DenseMatrix::identity(int n) {
    DenseMatrix m(n, n);
    for (int i = 0; i < n; ++i) {
        m(i, i) = 1;
    }
    return m;
}

std::span<Real> DenseMatrix::row(int i) {
    return data().subspan(index(i, 0), static_cast<std::size_t>(cols_));
}

std::span<const Real> DenseMatrix::row(int i) const {
    return data().subspan(index(i, 0), static_cast<std::size_t>(cols_));
}

DenseMatrix DenseMatrix::transposed() const {
    DenseMatrix t(cols_, rows_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            t(j, i) = (*this)(i, j);
        }
    }
    return t;
}

DenseMatrix operator*(const DenseMatrix& a, const DenseMatrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("DenseMatrix product: inner dimensions do not match");
    }
    DenseMatrix c(a.rows(), b.cols());
    for (int i = 0; i < a.rows(); ++i) {
        for (int k = 0; k < a.cols(); ++k) {
            const Real aik = a(i, k);
            for (int j = 0; j < b.cols(); ++j) {
                c(i, j) += aik * b(k, j);
            }
        }
    }
    return c;
}

void multiply(const DenseMatrix& a, std::span<const Real> x, std::span<Real> y) {
    if (x.size() != static_cast<std::size_t>(a.cols())
        || y.size() != static_cast<std::size_t>(a.rows())) {
        throw std::invalid_argument("multiply: vector sizes do not match the matrix");
    }
    for (int i = 0; i < a.rows(); ++i) {
        const auto row = a.row(i);
        Real sum = 0;
        for (std::size_t j = 0; j < row.size(); ++j) {
            sum += row[j] * x[j];
        }
        y[static_cast<std::size_t>(i)] = sum;
    }
}

} // namespace sem
