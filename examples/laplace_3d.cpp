// Векторное уравнение Лапласа в кубе — поле скоростей потенциального течения:
//
//   Delta u = 0 в (-1, 1)^3,   u = g на границе,   u = (u1, u2, u3).
//
// Точное решение — скорость u = grad(phi) для гармонического потенциала
//   phi(x, y, z) = e^{sqrt(2) x} sin y cos z   (Delta phi = (2 - 1 - 1) phi = 0):
//   u1 = sqrt(2) e^{sqrt(2) x} sin y cos z,
//   u2 =         e^{sqrt(2) x} cos y cos z,
//   u3 =        -e^{sqrt(2) x} sin y sin z.
// Каждая компонента гармоническая (Delta(d phi / dx_i) = d(Delta phi) / dx_i = 0), а само поле
// бездивергентное (div u = Delta phi = 0) и безвихревое.
//
// Уравнения для компонент не связаны: матрица системы блочно-диагональная diag(K, K, K),
// поэтому решаются три независимые скалярные задачи Дирихле с одним и тем же оператором.
// Каждая — функцией solveLaplaceDirichlet (laplace_dirichlet.hpp): один спектральный элемент степени N, Галёркин
// с квадратурой GLL (K w = 0 во внутренних узлах), подъём граничного условия
// w = w0 + wg и система P K P w0 = -P K wg с маской P, метод сопряжённых градиентов
// без сборки матрицы (applyK) с предобусловливателем Якоби.
//
// Вывод: таблица для N = 2..16 — итерации, время, ошибки поля скоростей
// (C = max по компонентам, L2 = sqrt(сумма квадратов) по компонентам) и норма
// дивергенции численного поля; график сходимости и поле скоростей стрелками при N = 16.
// Запуск с флагом --no-show только сохраняет картинки, без окон.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "sem/core/timer.hpp"
#include "sem/io/plot.hpp"
#include "sem/operators/stiffness.hpp"
#include "sem/solvers/conjugate_gradient.hpp"

#include "laplace_dirichlet.hpp"

using sem::Point;
using sem::Real;

constexpr int Dim = 3;

int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    const int maxDegree = 16;
    const Real a = std::sqrt(2.0);

    // Точное поле скоростей u = grad(phi) и его производные: jacobian(p)[c] = grad u_c
    // (это матрица Гессе потенциала phi).
    const auto velocity = [a](const Point<Dim> &p) {
        const Real e = std::exp(a * p[0]);
        return Point<Dim>{
            a * e * std::sin(p[1]) * std::cos(p[2]),
            e * std::cos(p[1]) * std::cos(p[2]),
            -e * std::sin(p[1]) * std::sin(p[2])
        };
    };
    const auto jacobian = [a](const Point<Dim> &p) {
        const Real e = std::exp(a * p[0]);
        const Real sy = std::sin(p[1]), cy = std::cos(p[1]), sz = std::sin(p[2]), cz = std::cos(p[2]);
        return std::array<Point<Dim>, Dim>{
            Point<Dim>{2 * e * sy * cz, a * e * cy * cz, -a * e * sy * sz},
            Point<Dim>{a * e * cy * cz, -e * sy * cz, -e * cy * sz},
            Point<Dim>{-a * e * sy * sz, -e * cy * sz, -e * sy * cz}
        };
    };

    // Нормы считаются для каждой из трёх компонент; в 3D перебор точек для C и C1 растёт
    // как (s N)^3 — берём меньше точек, чем по умолчанию.
    sem::analysis::NormOptions normOptions;
    normOptions.quadraturePoints = 6;
    normOptions.samplesPerInterval = 4;

    std::vector<Real> degrees, errorC, errorL2, divergenceL2;
    std::array<std::vector<Real>, Dim> finalVelocity; // решение при N = maxDegree

    std::printf("%3s %8s %12s %10s %12s %12s %14s\n", "N", "nodes", "CG it", "time, ms", "C", "L2",
                "||div u_N||");
    for (int N = 2; N <= maxDegree; ++N) {
        sem::Timer timer;
        const sem::TensorBasis<Dim> basis(N);

        // Три независимые скалярные задачи — по одной на компоненту.
        std::array<LaplaceSolution, Dim> u;
        for (int c = 0; c < Dim; ++c) {
            u[c] = solveLaplaceDirichlet(basis, [&](const Point<Dim> &p) { return velocity(p)[c]; });
        }
        const double ms = timer.milliseconds();

        // Ошибки поля: C — максимум по компонентам, L2 — корень из суммы квадратов.
        Real c = 0, l2 = 0;
        for (int comp = 0; comp < Dim; ++comp) {
            const auto norms = sem::analysis::errorNorms<Dim>(
                basis, u[comp].values,
                [&](const Point<Dim> &p) { return velocity(p)[comp]; },
                [&](const Point<Dim> &p) { return jacobian(p)[comp]; },
                normOptions);
            c = std::max(c, norms.c);
            l2 += norms.l2 * norms.l2;
        }
        l2 = std::sqrt(l2);

        // Дивергенция численного поля в узлах: sum_d du_d / dx_d (точное значение 0),
        // её L2-норма — квадратурой GLL.
        std::vector<Real> divergence(u[0].values.size(), 0.0), derivative(divergence.size());
        for (int d = 0; d < Dim; ++d) {
            basis.derivative(u[d].values, d, derivative);
            for (std::size_t k = 0; k < divergence.size(); ++k) divergence[k] += derivative[k];
        }
        for (Real &v: divergence) v *= v;
        const Real divNorm = std::sqrt(basis.integrate(divergence));

        const std::string iterations = std::to_string(u[0].cg.iterations) + "/" +
                                       std::to_string(u[1].cg.iterations) + "/" +
                                       std::to_string(u[2].cg.iterations);
        std::printf("%3d %8d %12s %10.1f %12.3e %12.3e %14.3e\n", N, basis.numPoints(), iterations.c_str(), ms,
                    c, l2, divNorm);

        degrees.push_back(N);
        errorC.push_back(c);
        errorL2.push_back(l2);
        divergenceL2.push_back(divNorm);
        if (N == maxDegree) {
            for (int d = 0; d < Dim; ++d) finalVelocity[d] = u[d].values;
        }
    }

    // 1. Сходимость: ошибки поля скоростей и дивергенция численного решения.
    sem::io::Figure convergence(
        R"(Potential flow $\mathbf{u} = \nabla\varphi$, $\varphi = e^{\sqrt{2}\,x}\sin y\,\cos z$:)"
        R"( $\Delta\mathbf{u} = 0$ in $(-1, 1)^3$)");
    convergence.setXLabel("degree $N$")
            .setYLabel(R"(error norm (logarithmic scale))")
            .setYScale(sem::io::AxisScale::Log)
            .plot(degrees, errorC, R"($\Vert\mathbf{u} - \mathbf{u}_N\Vert_C$)", "o-")
            .plot(degrees, errorL2, R"($\Vert\mathbf{u} - \mathbf{u}_N\Vert_{L^2}$)", "s-")
            .plot(degrees, divergenceL2, R"($\Vert\nabla\cdot\mathbf{u}_N\Vert_{L^2}$)", "^--")
            .plot(degrees, std::vector<Real>(degrees.size(), std::numeric_limits<Real>::epsilon()),
                  R"(machine $\varepsilon$)", "k:");

    // 2. Поле скоростей стрелками при N = maxDegree. Значения трёх компонент на сетке
    //    7 x 7 x 7 — по одному tensorApply с матрицей интерполяции I (7 x (N+1)).
    const sem::TensorBasis<Dim> finalBasis(maxDegree);
    const int samples = 7;
    std::vector<Real> axis(samples);
    for (int i = 0; i < samples; ++i) {
        axis[i] = -1 + 2.0 * i / (samples - 1);
    }
    const sem::DenseMatrix I = finalBasis.basis1D().lagrange().interpolationMatrix(axis);
    const auto total = static_cast<std::size_t>(samples) * samples * samples;
    std::array<std::vector<Real>, Dim> grid;
    for (int d = 0; d < Dim; ++d) {
        grid[d].resize(total);
        sem::tensorApply<Dim>({&I, &I, &I}, finalVelocity[d], grid[d]);
    }
    std::vector<Real> px, py, pz; // точки сетки в той же раскладке (x меняется быстрее всего)
    for (int k = 0; k < samples; ++k) {
        for (int j = 0; j < samples; ++j) {
            for (int i = 0; i < samples; ++i) {
                px.push_back(axis[i]);
                py.push_back(axis[j]);
                pz.push_back(axis[k]);
            }
        }
    }
    sem::io::VectorFigure3D field(R"(Velocity $\mathbf{u}_N$ of the potential flow, $N = )" +
                                  std::to_string(maxDegree) + "$");
    field.setColorLabel(R"($|\mathbf{u}_N|$)")
            .setArrowLength(0.9 * (axis[1] - axis[0]))
            .setVectors(px, py, pz, grid[0], grid[1], grid[2]);

    sem::io::renderAll(std::vector<sem::io::PlotItem>{
                           {convergence.toJson(), "laplace_3d_convergence.json", "laplace_3d_convergence.png"},
                           {field.toJson(), "laplace_3d_velocity.json", "laplace_3d_velocity.png"}
                       },
                       showWindow);
    return 0;
}
