#pragma once

#include <span>
#include <vector>

#include "sem/core/dense_matrix.hpp"
#include "sem/core/types.hpp"

namespace sem {

// Интерполяционный базис Лагранжа на произвольных различных узлах x_0..x_N:
//   ell_j(x_i) = delta_ij.
// Вычисления ведутся в барицентрической форме (Berrut & Trefethen, SIAM Review, 2004):
//   lambda_j = 1 / prod_{k != j} (x_j - x_k),
//   ell_j(x) = (lambda_j / (x - x_j)) / sum_k (lambda_k / (x - x_k)).
// Класс ничего не знает о квадратурах и элементах — это чистый инструмент
// интерполяции и дифференцирования.
//
// Соглашение об именах во всех базисах библиотеки:
//   evaluateBasis(x, values) — значения всех базисных функций в точке;
//   evaluate(u, x)           — значение разложения u_N(x) = sum_j u_j ell_j(x);
//   interpolate(f)           — узловые значения функции (оператор интерполяции I_N),
//                              есть у SpectralBasis1D и TensorBasis.
class LagrangeBasis1D {
public:
    explicit LagrangeBasis1D(std::vector<Real> nodes);

    int degree() const { return numNodes() - 1; }
    int numNodes() const { return static_cast<int>(nodes_.size()); }

    std::span<const Real> nodes() const { return nodes_; }
    std::span<const Real> barycentricWeights() const { return baryWeights_; }

    // Значения всех базисных функций: values[j] = ell_j(x); values.size() == numNodes().
    void evaluateBasis(Real x, std::span<Real> values) const;

    // Значение разложения в точке: u_N(x) = sum_j u_j ell_j(x).
    Real evaluate(std::span<const Real> nodalValues, Real x) const;

    // I(i, j) = ell_j(targets[i]); размер targets.size() x numNodes().
    DenseMatrix interpolationMatrix(std::span<const Real> targets) const;

    // D(i, j) = ell_j'(x_i); размер numNodes() x numNodes().
    DenseMatrix derivativeMatrix() const;

private:
    std::vector<Real> nodes_;
    std::vector<Real> baryWeights_;
};

} // namespace sem
