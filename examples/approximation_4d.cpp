#include <cmath>
#include <cstdio>
#include <format>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/quadrature.hpp"
#include "sem/basis/spectral_basis.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "sem/core/linear_solve.hpp"
#include "sem/core/timer.hpp"
#include "sem/io/plot.hpp"

// Приближение функции четырёх переменных на эталонном элементе [-1, 1]^4 (K = 1, J = Id)
// и сходимость ошибки при росте степени N0 = 1, 2, ..., N: таблица норм C, C1, L1, L2
// и график всех норм на одном рисунке в логарифмическом масштабе.
//
// Запуск с флагом --no-show только сохраняет картинку, без окна.
int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    const int N = 10; // максимальная степень: считаются все N0 = 1, ..., N
    constexpr int Dim = 4;

    // const auto u = [](sem::Point<Dim> p)-> sem::Real { return p[0] + p[1]; }; // z = x + y
    // const auto grad_u = [](sem::Point<Dim>) -> sem::Point<Dim> { return {1, 1}; }; // (u_x, u_y)

    // const auto u = [](sem::Point<Dim> p)-> sem::Real { return std::sin(p[0] + p[1]); }; // z = sin(x + y)
    // const auto grad_u = [](sem::Point<Dim> p) -> sem::Point<Dim> {
    //     return {std::cos(p[0] + p[1]), std::cos(p[0] + p[1])};
    // }; // (u_x, u_y)

    // const auto u = [](sem::Point<Dim> p)-> sem::Real {
    //     return 1 / ((p[0] * p[0] + 1) * (p[1] * p[1] + 1));
    // }; // z =1/(x^2+1)(y^2+1)
    // const auto grad_u = [](sem::Point<Dim> p) -> sem::Point<Dim> {
    //     const sem::Real ax = p[0] * p[0] + 1, ay = p[1] * p[1] + 1;
    //     return {-2 * p[0] / (ax * ax * ay), -2 * p[1] / (ax * ay * ay)};
    // };

    // const auto u = [](sem::Point<Dim> p)-> sem::Real {
    //     return std::sin(6 * p[0] * p[0] + 8 * p[1] * p[1]);
    // }; // z = sin(6x^2 + 8y^2)
    // const auto grad_u = [](sem::Point<Dim> p) -> sem::Point<Dim> {
    //     return {
    //         12 * p[0] * std::cos(6 * p[0] * p[0] + 8 * p[1] * p[1]),
    //         16 * p[1] * std::cos(6 * p[0] * p[0] + 8 * p[1] * p[1])
    //     };
    // }; // (u_x, u_y)
    //
    //
    // const auto u = [](sem::Point<Dim> p)-> sem::Real {
    //     return std::exp(-p[0] * p[0] - p[1] * p[1]);
    // }; // z = e^(-x^2-y^2)
    // const auto grad_u = [](sem::Point<Dim> p) -> sem::Point<Dim> {
    //     return {
    //         -2 * p[0] * std::exp(-p[0] * p[0] - p[1] * p[1]),
    //         -2 * p[1] * std::exp(-p[0] * p[0] - p[1] * p[1])
    //     };
    // }; // (u_x, u_y)

    // Точки: p = (x, y, z, t). Формула для подписи графика (TeX, mathtext) —
    // при смене u её нужно поменять вместе с u и grad_u.
    const std::string uFormula = R"($u(x, y, z, t) = e^{x}\sin 3y + z^2 + t\,\sin t$)";
    const auto u = [](sem::Point<Dim> p) -> sem::Real {
        return std::exp(p[0]) * std::sin(3 * p[1]) + p[2] * p[2] + p[3] * std::sin(p[3]);
    };
    const auto grad_u = [](sem::Point<Dim> p) -> sem::Point<Dim> {
        return {
            std::exp(p[0]) * std::sin(3 * p[1]),
            3 * std::exp(p[0]) * std::cos(3 * p[1]),
            2 * p[2],
            std::sin(p[3]) + p[3] * std::cos(p[3])
        };
    };

    // В 4D перебор точек для норм C и C1 растёт как (samplesPerInterval * N0)^4, поэтому
    // берём меньше точек, чем по умолчанию (8 и 16): нормы меняются лишь в 3-4 знаке,
    // а считаются в десятки раз быстрее.
    sem::analysis::NormOptions options;
    options.quadraturePoints = 6;
    options.samplesPerInterval = 6;

    sem::Timer total; // всё время работы программы

    std::vector<sem::Real> degrees, errorC, errorC1, errorL1, errorL2;
    std::cout << std::format("{:>3} {:>7} {:>11} {:>11} {:>11} {:>11} {:>10}\n",
                             "N", "nodes", "C", "C1", "L1", "L2", "time, ms");

    for (int N0 = 1; N0 <= N; ++N0) {
        sem::Timer timer;
        const sem::TensorBasis<Dim> basis(N0);
        const std::vector<sem::Real> coeff = basis.interpolate(u); // задача u = f: узловые значения

        // Нормы ошибки: C, C1 = max|e| + sum_d max|de/dx_d|, L1, L2 по [-1, 1]^4.
        const auto norms = sem::analysis::errorNorms<Dim>(basis, coeff, u, grad_u, options);
        const double ms = timer.milliseconds();

        std::cout << std::format("{:>3} {:>7} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>10.1f}\n",
                                 N0, basis.numPoints(), norms.c, norms.c1, norms.l1, norms.l2, ms);
        degrees.push_back(N0);
        errorC.push_back(norms.c);
        errorC1.push_back(norms.c1);
        errorL1.push_back(norms.l1);
        errorL2.push_back(norms.l2);
    }
    std::cout << std::format("total: {:.1f} ms\n", total.milliseconds());

    // Все нормы на одном графике: по оси y — логарифмический масштаб, поэтому
    // экспоненциальная сходимость выглядит как прямая линия.
    // Строки с TeX собираются сложением, а не через std::format: в std::format фигурные
    // скобки TeX пришлось бы удваивать.
    // Подписи на английском: картинки используются в README (docs/images).
    sem::io::Figure errors("Approximation of " + uFormula + R"( on $[-1, 1]^4$)");
    errors.setXLabel(R"(degree $N$)")
            .setYLabel(R"(error norm of $u - u_N$)" "\n" "(logarithmic scale)")
            .setYScale(sem::io::AxisScale::Log)
            .plot(degrees, errorC, "$C$", "o-")
            .plot(degrees, errorC1, "$C^1$", "s-")
            .plot(degrees, errorL1, "$L^1$", "^-")
            .plot(degrees, errorL2, "$L^2$", "v-");
    // Те же нормы в обычной (линейной) шкале: видно, насколько велика ошибка при малых N
    // и как быстро она становится неотличимой от нуля.
    sem::io::Figure errorsLinear("Approximation of " + uFormula + R"( on $[-1, 1]^4$)");
    errorsLinear.setXLabel(R"(degree $N$)")
            .setYLabel(R"(error norm of $u - u_N$)" "\n" "(linear scale)")
            .plot(degrees, errorC, "$C$", "o-")
            .plot(degrees, errorC1, "$C^1$", "s-")
            .plot(degrees, errorL1, "$L^1$", "^-")
            .plot(degrees, errorL2, "$L^2$", "v-");

    // Оба графика одним запуском Python: два окна открываются одновременно.
    sem::io::renderAll(std::vector<sem::io::PlotItem>{
                           {errors.toJson(), "approximation_4d_errors.json", "approximation_4d_errors.png"},
                           {
                               errorsLinear.toJson(), "approximation_4d_errors_linear.json",
                               "approximation_4d_errors_linear.png"
                           }
                       },
                       showWindow);

    return 0;
}
