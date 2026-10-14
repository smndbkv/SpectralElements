// Приближение функции на [a, b], разбитом на K спектральных элементов степени N.
//
// Элемент k = [x_k, x_{k+1}] получается из эталонного [-1, 1] аффинным отображением
//   x(xi) = x_k + (xi + 1) * J_k,   J_k = (x_{k+1} - x_k) / 2 = dx/dxi.
// Отсюда:
//   производная:  du/dx = (1 / J_k) du/dxi
//   интеграл:     int_{x_k}^{x_{k+1}} g dx = J_k int_{-1}^{1} g dxi ≈ J_k sum_i w_i g_i
//
// Запуск с флагом --no-show только сохраняет картинки, без окон.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/spectral_basis.hpp"
#include "sem/io/plot.hpp"

int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    const int N = 16; // степень полинома в каждом элементе
    const int K = 50; // число элементов
    const double a = -4; // отрезок [a, b]
    const double b = 4;

    // Точная функция и её производная.
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
    const sem::SpectralBasis1D basis(N);
    const int n = basis.numPoints(); // N + 1 узлов на элемент
    const auto xi = basis.points();
    const auto w = basis.weights();

    // 1. Разбиение: границы элементов x_0 = a < x_1 < ... < x_K = b.
    std::vector<double> bp(K + 1);
    for (int k = 0; k <= K; ++k) {
        bp[k] = a + (b - a) * k / K;
    }

    // 2. Отображение эталонного элемента на элемент k и обратно.
    const auto J = [&](int k) { return (bp[k + 1] - bp[k]) / 2; };
    const auto toPhysical = [&](int k, double s) { return bp[k] + (s + 1) * J(k); };
    const auto toReference = [&](int k, double x) { return (x - bp[k]) / J(k) - 1; };

    // 3. Узловые значения, поэлементное хранение: nodal[k * n + i] = u(x_{k,i}).
    //    Общий узел соседних элементов записан дважды с одним и тем же значением,
    //    поэтому приближение непрерывно.
    std::vector<double> nodal(K * n);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < n; ++i) {
            nodal[k * n + i] = u(toPhysical(k, xi[i]));
        }
    }
    const auto element = [&](int k) {
        return std::span<const double>(nodal).subspan(k * n, n);
    };

    // 4. Узловые значения производной: (1 / J_k) * D * u_k.
    std::vector<double> dnodal(K * n);
    for (int k = 0; k < K; ++k) {
        const std::span<double> dk = std::span<double>(dnodal).subspan(k * n, n);
        sem::multiply(basis.D(), element(k), dk);
        for (double &d: dk) {
            d /= J(k);
        }
    }

    // 5. Значение приближения в произвольной точке: найти элемент, перейти в xi, интерполировать.
    const auto findElement = [&](double x) {
        // Число внутренних границ x_1..x_{K-1}, не превосходящих x.
        const auto it = std::upper_bound(bp.begin() + 1, bp.end() - 1, x);
        return static_cast<int>(it - (bp.begin() + 1));
    };
    const auto approx = [&](double x) {
        const int k = findElement(x);
        return basis.evaluate(element(k), toReference(k, x));
    };

    // 6. Интеграл по всему отрезку: sum_k J_k sum_i w_i u_{k,i}.
    double integral = 0;
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < n; ++i) {
            integral += J(k) * w[i] * nodal[k * n + i];
        }
    }
    std::printf("N = %d, K = %d on [%g, %g]\n", N, K, a, b);
    std::printf("integral of u (SEM quadrature) = %.15f\n", integral);
    std::printf("u(0.3) = %.15f, approx = %.15f\n", u(0.3), approx(0.3));
    std::printf("u'(b)  = %.15f, approx = %.15f\n", du(b), dnodal.back());

    // 7. Нормы ошибки.
    const auto norms = sem::analysis::errorNorms(basis, bp, nodal, u, du);
    std::printf("C = %.3e  C1 = %.3e  L1 = %.3e  L2 = %.3e\n\n",
                norms.c, norms.c1, norms.l1, norms.l2);

    // 8. Сходимость при измельчении: ожидаемые порядки N+1 (L2) и N (C1).
    std::printf("%4s %12s %6s %12s %6s\n", "K", "L2", "rate", "C1", "rate");
    sem::analysis::ErrorNorms previous{};
    for (int numElements = 1; numElements <= 32; numElements *= 2) {
        std::vector<double> breaks(numElements + 1);
        for (int k = 0; k <= numElements; ++k) {
            breaks[k] = a + (b - a) * k / numElements;
        }
        std::vector<double> values;
        for (int k = 0; k < numElements; ++k) {
            const double Jk = (breaks[k + 1] - breaks[k]) / 2;
            for (int i = 0; i < n; ++i) {
                values.push_back(u(breaks[k] + (xi[i] + 1) * Jk));
            }
        }
        const auto e = sem::analysis::errorNorms(basis, breaks, values, u, du);
        if (numElements == 1) {
            std::printf("%4d %12.3e %6s %12.3e %6s\n", numElements, e.l2, "", e.c1, "");
        } else {
            std::printf("%4d %12.3e %6.2f %12.3e %6.2f\n", numElements,
                        e.l2, std::log2(previous.l2 / e.l2), e.c1, std::log2(previous.c1 / e.c1));
        }
        previous = e;
    }

    // 9. Графики: приближение и ошибка (ошибка обнуляется в узлах GLL каждого элемента).
    std::vector<double> nodesX, nodesY;
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < n; ++i) {
            nodesX.push_back(toPhysical(k, xi[i]));
            nodesY.push_back(nodal[k * n + i]);
        }
    }
    std::vector<double> boundsY;
    for (const double x: bp) {
        boundsY.push_back(u(x));
    }

    sem::io::Figure fig("Приближение: N = " + std::to_string(N) + ", K = " + std::to_string(K));
    fig.setXLabel("x").setYLabel("u");
    fig.plotFunction(u, a, b, 2000, "точная", "-");
    fig.plotFunction(approx, a, b, 2000, "SEM", "--");
    fig.plot(nodesX, nodesY, "узлы GLL", "o");
    fig.plot(bp, boundsY, "границы элементов", "ks");
    fig.render("approximation_1d.json", {.image = "approximation_1d.png", .showWindow = showWindow});

    sem::io::Figure err("Ошибка u - u_h");
    err.setXLabel("x").setYLabel("u - u_h");
    err.plotFunction([&](double x) { return u(x) - approx(x); }, a, b, 2000, "ошибка", "-");
    err.plot(bp, std::vector<double>(bp.size(), 0.0), "границы элементов", "ks");
    err.render("approximation_1d_error.json",
               {.image = "approximation_1d_error.png", .showWindow = showWindow});
    return 0;
}
