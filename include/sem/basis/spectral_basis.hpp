#pragma once

#include <span>
#include <vector>

#include "sem/basis/lagrange_basis.hpp"
#include "sem/basis/quadrature.hpp"
#include "sem/core/dense_matrix.hpp"
#include "sem/core/types.hpp"

namespace sem {

// Эталонный одномерный элемент SEM степени N на [-1, 1].
//
// Узлы базиса совпадают с узлами квадратуры GLL (коллокация). Отсюда инварианты,
// на которые опираются операторы:
//   * ell_i(x_k) = delta_ik  =>  матрица масс диагональна: M = diag(w);
//   * узловые значения и есть коэффициенты разложения;
//   * концы ±1 входят в узлы  =>  непрерывность между элементами через dssum;
//   * W D + D^T W = diag(-1, 0, ..., 0, 1)  (дискретное интегрирование по частям).
// Конструктор принимает только степень, поэтому нарушить эти инварианты нельзя.
class SpectralBasis1D {
public:
    explicit SpectralBasis1D(int degree);  // degree >= 1

    int degree() const { return degree_; }
    int numPoints() const { return degree_ + 1; }

    const Quadrature1D&    quadrature() const { return quad_; }
    std::span<const Real>  points() const { return quad_.points; }
    std::span<const Real>  weights() const { return quad_.weights; }
    const LagrangeBasis1D& lagrange() const { return lagrange_; }

    // D(i, j) = ell_j'(x_i) и её транспонированная (хранится заранее для ядер).
    const DenseMatrix& D() const { return D_; }
    const DenseMatrix& Dt() const { return Dt_; }

    // Узловые значения функции: u[i] = f(x_i) (оператор интерполяции I_N).
    // Это и есть решение задачи u = f при квадратуре GLL (матрица масс диагональна).
    template <class F>
    std::vector<Real> interpolate(F&& f) const {
        std::vector<Real> u;
        u.reserve(quad_.points.size());
        for (const Real x : quad_.points) {
            u.push_back(f(x));
        }
        return u;
    }

    // Значение разложения в точке x из [-1, 1]: u_N(x) = sum_j u_j ell_j(x).
    Real evaluate(std::span<const Real> u, Real x) const { return lagrange_.evaluate(u, x); }

private:
    int             degree_;
    Quadrature1D    quad_;
    LagrangeBasis1D lagrange_;
    DenseMatrix     D_;
    DenseMatrix     Dt_;
};

} // namespace sem
