#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "sem/basis/tensor_apply.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "sem/core/types.hpp"

namespace sem {

// y = K x — матрица жёсткости (дискретный -Laplace) на эталонном элементе [-1, 1]^Dim:
//   K_ij = int grad(phi_i) . grad(phi_j) dx.
// Квадратура GLL интегрирует произведения производных точно (степень 2N - 2 по каждой
// переменной), так что K — точная матрица жёсткости полиномиального пространства.
//
// Матрица не собирается (matrix-free):
//   K x = sum_d D_d^T W D_d x,
// где D_d — одномерная матрица дифференцирования вдоль направления d (applyAlong),
// W — диагональ весов квадратуры w_{i_0} ... w_{i_{Dim-1}}. Стоимость O(Dim N^{Dim+1}).
//
// x и y — узловые значения в раскладке TensorBasis, не должны перекрываться.
// Бросает std::invalid_argument при неверных размерах.
template <int Dim>
void applyK(const TensorBasis<Dim>& basis, std::span<const Real> x, std::span<Real> y) {
    const auto size = static_cast<std::size_t>(basis.numPoints());
    if (x.size() != size || y.size() != size) {
        throw std::invalid_argument("applyK: x and y must hold numPoints() values");
    }

    const SpectralBasis1D& b1 = basis.basis1D();
    std::array<int, Dim> shape{};
    shape.fill(b1.numPoints());

    std::vector<Real> gradient(size);  // производная по x_d в узлах, затем умноженная на веса
    std::vector<Real> term(size);      // D_d^T W D_d x
    std::fill(y.begin(), y.end(), Real{0});

    for (int d = 0; d < Dim; ++d) {
        applyAlong<Dim>(b1.D(), d, shape, x, gradient);
        for (std::size_t k = 0; k < size; ++k) {
            gradient[k] *= basis.weight(static_cast<int>(k));
        }
        applyAlong<Dim>(b1.Dt(), d, shape, gradient, term);
        for (std::size_t k = 0; k < size; ++k) {
            y[k] += term[k];
        }
    }
}

// Диагональ матрицы жёсткости без её сборки (например, для предобусловливателя Якоби).
// Из K = sum_d M ⊗ ... ⊗ A ⊗ ... ⊗ M (A на месте d) с одномерными A = D^T W D, M = W:
//   K_ii = sum_d A_{i_d i_d} * prod_{e != d} w_{i_e},   A_jj = sum_k w_k D_kj^2.
template <int Dim>
std::vector<Real> diagonalK(const TensorBasis<Dim>& basis) {
    const SpectralBasis1D& b1 = basis.basis1D();
    const auto n = static_cast<std::size_t>(b1.numPoints());
    const auto w = b1.weights();

    std::vector<Real> a(n, Real{0});  // диагональ одномерной жёсткости
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < n; ++k) {
            const Real dkj = b1.D()(static_cast<int>(k), static_cast<int>(j));
            a[j] += w[k] * dkj * dkj;
        }
    }

    std::vector<Real> diagonal(static_cast<std::size_t>(basis.numPoints()));
    for (int flat = 0; flat < basis.numPoints(); ++flat) {
        const std::array<int, Dim> index = basis.multiIndex(flat);
        Real sum = 0;
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            Real term = a[static_cast<std::size_t>(index[d])];
            for (std::size_t e = 0; e < static_cast<std::size_t>(Dim); ++e) {
                if (e != d) {
                    term *= w[static_cast<std::size_t>(index[e])];
                }
            }
            sum += term;
        }
        diagonal[static_cast<std::size_t>(flat)] = sum;
    }
    return diagonal;
}

} // namespace sem
