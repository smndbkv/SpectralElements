#pragma once

// Общая часть примеров laplace_3d и laplace_caverna_3d: скалярная задача Дирихле
//   Delta w = 0 в (-1, 1)^3,   w = g на границе
// на одном спектральном элементе степени N (Галёркин с квадратурой GLL).
//
// Неоднородное граничное условие — через «подъём»: w = w0 + wg, где wg = g в граничных
// узлах и 0 внутри, а w0 = 0 на границе и находится из
//   P K P w0 = -P K wg,
// P — маска (0 в граничных узлах, 1 во внутренних). Система решается методом
// сопряжённых градиентов без сборки матрицы (applyK) с предобусловливателем Якоби.

#include <cstddef>
#include <span>
#include <vector>

#include "sem/basis/tensor_basis.hpp"
#include "sem/operators/stiffness.hpp"
#include "sem/solvers/conjugate_gradient.hpp"

struct LaplaceSolution {
    std::vector<sem::Real> values; // узловые значения решения
    sem::CgResult cg;              // статистика метода сопряжённых градиентов
};

// g — функция точки (const sem::Point<3>&), вызывается только в граничных узлах.
template<class G>
LaplaceSolution solveLaplaceDirichlet(const sem::TensorBasis<3> &basis, G &&g) {
    using sem::Real;
    constexpr int Dim = 3;
    const int N = basis.degree();
    const auto size = static_cast<std::size_t>(basis.numPoints());

    // Маска: граничный узел — если хотя бы один индекс равен 0 или N.
    std::vector<bool> boundary(size);
    for (std::size_t k = 0; k < size; ++k) {
        for (const int i: basis.multiIndex(static_cast<int>(k))) {
            if (i == 0 || i == N) boundary[k] = true;
        }
    }

    // Подъём граничного условия: wg = g на границе, 0 внутри.
    std::vector<Real> wg(size, 0.0);
    for (std::size_t k = 0; k < size; ++k) {
        if (boundary[k]) wg[k] = g(basis.point(static_cast<int>(k)));
    }

    // S x = P K P x.
    const auto applyS = [&](std::span<const Real> in, std::span<Real> out) {
        std::vector<Real> px(in.begin(), in.end());
        for (std::size_t k = 0; k < size; ++k) {
            if (boundary[k]) px[k] = 0;
        }
        sem::applyK<Dim>(basis, px, out);
        for (std::size_t k = 0; k < size; ++k) {
            if (boundary[k]) out[k] = 0;
        }
    };

    // Правая часть b = -P K wg: известные граничные значения переносятся вправо.
    std::vector<Real> b(size);
    sem::applyK<Dim>(basis, wg, b);
    for (std::size_t k = 0; k < size; ++k) {
        b[k] = boundary[k] ? 0 : -b[k];
    }

    // Якоби: диагональ K во внутренних узлах, 1 в граничных (там невязка всегда 0).
    std::vector<Real> diagonal = sem::diagonalK<Dim>(basis);
    for (std::size_t k = 0; k < size; ++k) {
        if (boundary[k]) diagonal[k] = 1;
    }

    std::vector<Real> w0(size, 0.0);
    LaplaceSolution solution;
    solution.cg = sem::conjugateGradient(applyS, sem::JacobiPreconditioner(diagonal), b, w0,
                                         {.relativeTolerance = 1e-13});
    solution.values.resize(size);
    for (std::size_t k = 0; k < size; ++k) {
        solution.values[k] = w0[k] + wg[k];
    }
    return solution;
}
