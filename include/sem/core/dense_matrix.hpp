#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include "sem/core/types.hpp"

namespace sem {

// Небольшая плотная матрица с хранением по строкам (row-major).
// Предназначена для одномерных операторов размера (N+1) x (N+1)
// и матриц интерполяции M x (N+1); для больших систем не используется.
class DenseMatrix {
public:
    DenseMatrix() = default;
    DenseMatrix(int rows, int cols, Real value = Real{0});

    static DenseMatrix identity(int n);

    Real& operator()(int i, int j) { return data_[index(i, j)]; }
    Real  operator()(int i, int j) const { return data_[index(i, j)]; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    std::span<Real>       data() { return data_; }
    std::span<const Real> data() const { return data_; }

    std::span<Real>       row(int i);
    std::span<const Real> row(int i) const;

    DenseMatrix transposed() const;

private:
    std::size_t index(int i, int j) const {
        assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_)
             + static_cast<std::size_t>(j);
    }

    int rows_ = 0;
    int cols_ = 0;
    std::vector<Real> data_;
};

// C = A B
DenseMatrix operator*(const DenseMatrix& a, const DenseMatrix& b);

// y = A x; x и y не должны перекрываться.
void multiply(const DenseMatrix& a, std::span<const Real> x, std::span<Real> y);

} // namespace sem
