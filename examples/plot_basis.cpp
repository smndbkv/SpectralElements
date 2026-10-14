// Пример модуля sem::io: графики базисных функций Лагранжа на узлах GLL
// и сходимость интерполяции sin(pi x).
//
// Запуск с флагом --no-show только сохраняет картинки, без окон.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

#include "sem/basis/spectral_basis.hpp"
#include "sem/io/plot.hpp"

int main(int argc, char **argv) {
    const bool showWindow = !(argc > 1 && std::string_view(argv[1]) == "--no-show");

    // 1. Базисные функции ell_j(x) при N = 6.
    const int degree = 6;
    const sem::SpectralBasis1D basis(degree);

    const int numSamples = 401;
    std::vector<sem::Real> x(numSamples);
    for (int k = 0; k < numSamples; ++k) {
        x[k] = -1 + 2.0 * k / (numSamples - 1);
    }
    const sem::DenseMatrix L = basis.lagrange().interpolationMatrix(x); // L(k, j) = ell_j(x_k)

    sem::io::Figure basisFigure("Базис Лагранжа на узлах GLL, N = " + std::to_string(degree));
    basisFigure.setXLabel("x").setYLabel(R"($\ell_j(x)$)");

    std::vector<sem::Real> column(numSamples);
    for (int j = 0; j <= degree; ++j) {
        for (int k = 0; k < numSamples; ++k) {
            column[k] = L(k, j);
        }
        basisFigure.plot(x, column, "j = " + std::to_string(j));
    }
    const std::vector<sem::Real> zeros(basis.numPoints(), 0.0);
    basisFigure.plot(basis.points(), zeros, "узлы GLL", "ko");

    basisFigure.render("lagrange_basis.json",
                       {.image = "lagrange_basis.png", .showWindow = showWindow});

    // 2. Сходимость интерполяции: ошибка в логарифмическом масштабе.
    const auto f = [](sem::Real t) { return std::sin(std::numbers::pi * t); };

    std::vector<sem::Real> degrees, errors;
    for (int N = 2; N <= 24; N += 2) {
        const sem::SpectralBasis1D b(N);
        std::vector<sem::Real> nodal;
        for (const sem::Real p: b.points()) {
            nodal.push_back(f(p));
        }
        sem::Real err = 0;
        for (const sem::Real t: x) {
            err = std::max(err, std::abs(b.evaluate(nodal, t) - f(t)));
        }
        degrees.push_back(N);
        errors.push_back(err);
    }

    sem::io::Figure convergence("Сходимость интерполяции sin(πx)");
    convergence.setXLabel("степень N")
            .setYLabel(R"($\max|u - I_N u|$)")
            .setYScale(sem::io::AxisScale::Log)
            .plot(degrees, errors, "узлы GLL", "o-");

    convergence.render("interpolation_convergence.json",
                       {.image = "interpolation_convergence.png", .showWindow = showWindow});
    return 0;
}
