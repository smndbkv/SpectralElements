// «Каверна» для векторного уравнения Лапласа в кубе:
//
//   Delta u = 0 в (-1, 1)^3,   u = (u1, u2, u3),
//   u = (1, 0, 0) на верхней грани z = 1 («крышка» движется вдоль x, включая её рёбра),
//   u = 0 на остальных гранях.
//
// Компоненты не связаны, поэтому решаются три скалярные задачи Дирихле
// (solveLaplaceDirichlet, laplace_dirichlet.hpp). У u2 и u3 граничные данные нулевые,
// так что u2 = u3 = 0 тождественно: вся задача — это u1. (Настоящее течение в каверне
// с циркуляцией получается только из уравнений Стокса/Навье–Стокса, где компоненты
// связаны через давление.)
//
// Точного решения в замкнутом виде нет, поэтому качество проверяется иначе:
//   * значение в центре известно точно: u1(0, 0, 0) = 1/6. Сумма шести задач с единицей
//     на каждой грани по очереди — это задача с u = 1 на всей границе, её решение u = 1,
//     а по симметрии в центре все шесть дают одно и то же.
//     Численное решение даёт 1/6 с точностью до округления при ЛЮБОМ N: матрица
//     K = A⊗M⊗M + M⊗A⊗M + M⊗M⊗A с диагональной M не связывает внутренние узлы с узлами
//     на рёбрах и в углах, поэтому то же рассуждение верно и для дискретной задачи.
//     Это проверка симметрии решателя, но не мера сходимости;
//   * самосходимость: отличие u_N от решения при N_ref = 32 во внутренних точках сетки —
//     основная мера ошибки;
//   * принцип максимума: гармоническая функция лежит между 0 и 1. Граничные данные
//     разрывны на верхних рёбрах, поэтому сходимость алгебраическая, а у рёбер возможны
//     колебания (эффект Гиббса) — их видно по min/max численного решения.
//
// Вывод: таблица по N, график сходимости, срез u1 при y = 0 и поле скоростей стрелками.
// Запуск с флагом --no-show только сохраняет картинки, без окон.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "sem/basis/tensor_basis.hpp"
#include "sem/core/timer.hpp"
#include "sem/io/plot.hpp"

#include "laplace_dirichlet.hpp"

using sem::Point;
using sem::Real;

constexpr int Dim = 3;

// Граничные данные: (1, 0, 0) на крышке z = 1 (узлы GLL на ней имеют z ровно 1), иначе 0.
Point<Dim> lid(const Point<Dim> &p) {
    return p[2] == 1 ? Point<Dim>{1, 0, 0} : Point<Dim>{0, 0, 0};
}

std::vector<Real> uniformAxis(int n) {
    std::vector<Real> axis(n);
    for (int i = 0; i < n; ++i) axis[i] = -1 + 2.0 * i / (n - 1);
    return axis;
}

// Значения узловых данных u на равномерной сетке axis^3 (одним tensorApply).
std::vector<Real> onGrid(const sem::TensorBasis<Dim> &basis, const std::vector<Real> &u,
                         const std::vector<Real> &axis) {
    const sem::DenseMatrix I = basis.basis1D().lagrange().interpolationMatrix(axis);
    std::vector<Real> values(axis.size() * axis.size() * axis.size());
    sem::tensorApply<Dim>({&I, &I, &I}, u, values);
    return values;
}

int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    const int refDegree = 32;
    const Real centerExact = 1.0 / 6;

    // Сетка для сравнения решений: 21^3 точек, сравниваются только внутренние.
    const std::vector<Real> axis = uniformAxis(21);
    const int n = static_cast<int>(axis.size());
    const auto interior = [n](int flat) {
        const int i = flat % n, j = (flat / n) % n, k = flat / (n * n);
        return i > 0 && i < n - 1 && j > 0 && j < n - 1 && k > 0 && k < n - 1;
    };

    // Эталон: решение при N_ref.
    sem::Timer refTimer;
    const sem::TensorBasis<Dim> refBasis(refDegree);
    const LaplaceSolution reference = solveLaplaceDirichlet(refBasis, [](const Point<Dim> &p) { return lid(p)[0]; });
    const std::vector<Real> refGrid = onGrid(refBasis, reference.values, axis);
    std::printf("reference N = %d: %d nodes, %d CG iterations, %.0f ms, u1(0,0,0) = %.10f (exact 1/6 = %.10f)\n\n",
                refDegree, refBasis.numPoints(), reference.cg.iterations, refTimer.milliseconds(),
                refBasis.evaluate(reference.values, {0, 0, 0}), centerExact);

    std::printf("%3s %7s %6s %9s %14s %11s %12s %12s %9s %9s %10s\n", "N", "nodes", "CG it", "time, ms",
                "u1(center)", "|err cen|", "max|u-uref|", "rms|u-uref|", "min u1", "max u1", "max|u2,u3|");

    std::vector<Real> degrees, maxDiff, rmsDiff;
    for (int N = 4; N <= 24; N += 2) {
        sem::Timer timer;
        const sem::TensorBasis<Dim> basis(N);
        std::array<LaplaceSolution, Dim> u;
        for (int c = 0; c < Dim; ++c) {
            u[c] = solveLaplaceDirichlet(basis, [c](const Point<Dim> &p) { return lid(p)[c]; });
        }
        const double ms = timer.milliseconds();

        const Real center = basis.evaluate(u[0].values, {0, 0, 0});

        // Самосходимость и принцип максимума на равномерной сетке.
        const std::vector<Real> grid = onGrid(basis, u[0].values, axis);
        Real maxD = 0, sumD2 = 0, minU = grid[0], maxU = grid[0];
        int count = 0;
        for (int k = 0; k < static_cast<int>(grid.size()); ++k) {
            minU = std::min(minU, grid[k]);
            maxU = std::max(maxU, grid[k]);
            if (!interior(k)) continue;
            const Real d = std::abs(grid[k] - refGrid[k]);
            maxD = std::max(maxD, d);
            sumD2 += d * d;
            ++count;
        }
        const Real rmsD = std::sqrt(sumD2 / count);

        Real otherComponents = 0;
        for (int c = 1; c < Dim; ++c) {
            for (const Real v: u[c].values) otherComponents = std::max(otherComponents, std::abs(v));
        }

        std::printf("%3d %7d %6d %9.1f %14.10f %11.3e %12.3e %12.3e %9.4f %9.4f %10.1e\n", N, basis.numPoints(),
                    u[0].cg.iterations, ms, center, std::abs(center - centerExact), maxD, rmsD, minU, maxU,
                    otherComponents);

        degrees.push_back(N);
        maxDiff.push_back(maxD);
        rmsDiff.push_back(rmsD);
    }

    // 1. Сходимость в логарифмическом масштабе по обеим осям: алгебраическая
    //    сходимость выглядит как прямая с наклоном, равным порядку.
    sem::io::Figure convergence(R"(Lid-driven cube: $\Delta\mathbf{u} = 0$, $\mathbf{u} = (1, 0, 0)$ on the lid $z = 1$)");
    convergence.setXLabel("degree $N$ (logarithmic scale)")
            .setYLabel("error (logarithmic scale)")
            .setXScale(sem::io::AxisScale::Log)
            .setYScale(sem::io::AxisScale::Log)
            .plot(degrees, maxDiff, R"($\max|u_{1,N} - u_{1,\mathrm{ref}}|$, interior)", "s-")
            .plot(degrees, rmsDiff, R"(rms $|u_{1,N} - u_{1,\mathrm{ref}}|$, interior)", "^-");

    // Направляющая C N^{-3} (через первую точку rms): прямая с наклоном -3 в осях log-log.
    std::vector<Real> guide;
    for (const Real N: degrees) guide.push_back(rmsDiff.front() * std::pow(degrees.front() / N, 3));
    convergence.plot(degrees, guide, R"($\sim N^{-3}$)", "k--");

    // 2. Срез u1 при y = 0 для эталонного решения: по x и z — 161 точка, по y — одна строка
    //    ell(0). Один tensorApply даёт массив 161 x 1 x 161 с раскладкой i + 161 k.
    const std::vector<Real> fine = uniformAxis(161);
    const sem::DenseMatrix Ix = refBasis.basis1D().lagrange().interpolationMatrix(fine);
    const std::vector<Real> yZero{0.0};
    const sem::DenseMatrix Iy = refBasis.basis1D().lagrange().interpolationMatrix(yZero);
    std::vector<Real> slice(fine.size() * fine.size());
    sem::tensorApply<Dim>({&Ix, &Iy, &Ix}, reference.values, slice);

    sem::io::Figure2D sliceFigure(R"(Slice $y = 0$ of $u_1$, $N = )" + std::to_string(refDegree) + "$");
    sliceFigure.setXLabel("$x$").setYLabel("$z$").setColorLabel("$u_1$")
            .setField(fine, fine, slice)
            .addOverlay(fine, fine, slice, "isolines");

    // 3. Поле скоростей стрелками для эталонного решения на сетке 7 x 7 x 7.
    //    u2 = u3 = 0 тождественно (см. таблицу), поэтому все стрелки направлены вдоль x,
    //    а длина и цвет показывают, как «течение» от крышки затухает вглубь куба.
    const std::vector<Real> coarse = uniformAxis(7);
    const std::vector<Real> u1 = onGrid(refBasis, reference.values, coarse);
    const std::vector<Real> zero(u1.size(), 0.0);
    std::vector<Real> px, py, pz; // точки сетки в той же раскладке (x меняется быстрее всего)
    for (const Real z: coarse) {
        for (const Real y: coarse) {
            for (const Real x: coarse) {
                px.push_back(x);
                py.push_back(y);
                pz.push_back(z);
            }
        }
    }
    sem::io::VectorFigure3D velocity(R"(Velocity $\mathbf{u}_N = (u_1, 0, 0)$, lid $z = 1$ moves along $x$, $N = )" +
                                     std::to_string(refDegree) + "$");
    velocity.setColorLabel(R"($|\mathbf{u}_N|$)")
            .setArrowLength(0.9 * (coarse[1] - coarse[0]))
            .setVectors(px, py, pz, u1, zero, zero);

    sem::io::renderAll(std::vector<sem::io::PlotItem>{
                           {convergence.toJson(), "laplace_caverna_3d_convergence.json",
                            "laplace_caverna_3d_convergence.png"},
                           {sliceFigure.toJson(), "laplace_caverna_3d_slice.json", "laplace_caverna_3d_slice.png"},
                           {velocity.toJson(), "laplace_caverna_3d_velocity.json", "laplace_caverna_3d_velocity.png"}
                       },
                       showWindow);
    return 0;
}
