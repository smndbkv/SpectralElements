#pragma once

#include <cstddef>
#include <vector>

#include "sem/core/types.hpp"

namespace sem {

enum class PointsType {
    GaussLobattoLegendre,  // включает концы ±1; точна для полиномов степени <= 2Q-3
    GaussLegendre,         // только внутренние точки; точна для степени <= 2Q-1
};

// Квадратурная формула на [-1, 1]; Q = size() — число точек.
struct Quadrature1D {
    PointsType        type = PointsType::GaussLobattoLegendre;
    std::vector<Real> points;   // строго по возрастанию, симметричны относительно нуля
    std::vector<Real> weights;

    int size() const { return static_cast<int>(points.size()); }
};

// Аргумент — ЧИСЛО ТОЧЕК, а не степень полинома.
// Для GLL с N+1 точками внутренние узлы — корни L_N'.
Quadrature1D gaussLobattoLegendre(int numPoints);  // numPoints >= 2
Quadrature1D gaussLegendre(int numPoints);         // numPoints >= 1
Quadrature1D makeQuadrature(PointsType type, int numPoints);

// sum_i w_i f(x_i)
template <class F>
Real integrate(const Quadrature1D& quad, F&& f) {
    Real sum = 0;
    for (std::size_t i = 0; i < quad.points.size(); ++i) {
        sum += quad.weights[i] * f(quad.points[i]);
    }
    return sum;
}

} // namespace sem
