// Краевая задача для уравнения Гельмгольца (со знаком «+») на одном спектральном элементе:
//
//   u''(x) + u(x) = f(x),   x in (-1, 1),
//   u(-1) = u(1) = 0.
//
// Метод: Галёркин на одном элементе (K = 1, J = 1) с базисом Лагранжа степени N на узлах GLL.
// Слабая форма (после интегрирования по частям, граничный член исчезает, т.к. v(±1) = 0):
//   int u' v' dx - int u v dx = -int f v dx   для всех v с v(±1) = 0.
// С квадратурой GLL получается система (N+1) x (N+1):
//   (D^T W D - W) u = -W f,   W = diag(w_i),   D_ij = ell_j'(x_i),
// где D^T W D — матрица жёсткости (точная), W — диагональная матрица масс.
// Граничные условия: строки 0 и N заменяются уравнениями u_0 = 0, u_N = 0.
// Система решается методом Гаусса с выбором главного элемента.
//
// Тестовые задачи строятся методом искусственных решений: выбирается u с u(±1) = 0
// и вычисляется f = u'' + u. Сейчас включён узкий гауссов пик (beta = 50, x0 = 0.3),
// из которого вычтена прямая, чтобы выполнялись граничные условия; закомментированы
// гладкая (1 - x^2) e^x, осцилляции sin(4 pi x) и f = sin x с точным решением
// (cot(1) sin x - x cos x) / 2.
//
// Вывод: нормы ошибки C, C1, L1, L2 и график точного и численного решений.
// Замечание: на отрезке длины k*pi задача резонансная, и матрица системы вырождена.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <ostream>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/quadrature.hpp"
#include "sem/basis/spectral_basis.hpp"
#include "sem/core/linear_solve.hpp"
#include "sem/io/plot.hpp"


int main() {
    const int N = 36; //degree
    // const sem::Real cot1 = std::cos(1.0) / std::sin(1.0);
    // const auto f = [](sem::Real x) { return std::sin(x); };
    // const auto u = [cot1](sem::Real x) { return (cot1 * std::sin(x) - x * std::cos(x)) / 2; };
    // const auto du = [cot1](sem::Real x) { return ((cot1 - 1) * std::cos(x) + x * std::sin(x)) / 2; };

    // const sem::Real kw = 4 * std::numbers::pi;
    // const auto f = [kw](sem::Real x) { return (1 - kw * kw) * std::sin(kw * x); };
    // const auto u = [kw](sem::Real x) { return std::sin(kw * x); };
    // const auto du = [kw](sem::Real x) { return kw * std::cos(kw * x); };

    // const auto f = [](sem::Real x) { return -2 * x * (2 + x) * std::exp(x); };
    // const auto u = [](sem::Real x) { return (1 - x * x) * std::exp(x); };
    // const auto du = [](sem::Real x) { return (1 - 2 * x - x * x) * std::exp(x); };

    const sem::Real beta = 50, x0 = 0.3;
    const auto g = [=](sem::Real x) { return std::exp(-beta * (x - x0) * (x - x0)); };
    const sem::Real gp = g(1), gm = g(-1);
    const auto lin = [=](sem::Real x) { return (gp * (1 + x) + gm * (1 - x)) / 2; };

    const auto u = [=](sem::Real x) { return g(x) - lin(x); };
    const auto du = [=](sem::Real x) { return -2 * beta * (x - x0) * g(x) - (gp - gm) / 2; };
    const auto f = [=](sem::Real x) {
        return (4 * beta * beta * (x - x0) * (x - x0) - 2 * beta) * g(x) + u(x); // g'' + u
    };

    auto basis = sem::SpectralBasis1D(N);
    const int n = basis.numPoints();
    auto weights = basis.weights();
    auto points = basis.points();
    const auto &D = basis.D();
    const auto &Dt = basis.Dt();

    sem::DenseMatrix WD = D;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            WD(k, j) *= weights[k];
        }
    }
    sem::DenseMatrix A = Dt * WD;

    for (int k = 0; k < n; ++k)
        A(k, k) -= weights[k];

    std::vector<sem::Real> b(n);
    for (int i = 0; i < n; i++)
        b[i] = -weights[i] * f(points[i]);


    //Boundary condition u(+-1) = 0
    for (const int k: {0, N}) {
        for (int j = 0; j <= N; ++j) {
            A(k, j) = 0;
        }
        A(k, k) = 1;
        b[k] = 0;
    }

    const auto coeff = sem::solveGauss(A, b);

    // auto u_approximation = [&basis, &coeff](sem::Real x) -> sem::Real {
    //     sem::Real res = 0;
    //     std::vector<sem::Real> l(N + 1);
    //     basis.lagrange().evaluateBasis(x, l);
    //     for (int i = 0; i < N + 1; i++) {
    //         res += coeff[i] * l[i];
    //     }
    //     return res;
    // };

    auto u_approximation = [&basis, &coeff](sem::Real x) -> sem::Real {
        return basis.evaluate(coeff, x);
    };

    const std::vector<sem::Real> bp{-1, 1}; // один элемент [-1, 1]
    const auto norms = sem::analysis::errorNorms(basis, bp, coeff, u, du);

    printf("Norms of error: C = %.3e  C1 = %.3e  L1 = %.3e  L2 = %.3e\n", norms.c, norms.c1, norms.l1, norms.l2);


    sem::io::Figure fig("Приближение u(x)");
    fig.setXLabel("x").setYLabel("u");
    fig.plotFunction(u, -1, 1, 200, "точная", "-"); // функция на равномерной сетке
    fig.plotFunction(u_approximation, -1, 1, 200, "SEM", "--");
    fig.plot(basis.points(), coeff, "узлы GLL", "o"); // точки (x, y)
    fig.render("approx.json", {.image = "approx.png"}); // сохранить картинку и открыть окно

    return 0;
}
