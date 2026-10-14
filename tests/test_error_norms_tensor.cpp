#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/tensor_basis.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using sem::Point;
using sem::Real;
using sem::TensorBasis;
using sem::analysis::ErrorNorms;
using sem::analysis::errorNorms;

namespace {

const Real pi = std::numbers::pi;

} // namespace

TEST_CASE("errorNorms<2>: exact values for e = x^2 y^2", "[analysis][norms][normsNd]") {
    // Приближение — ноль, поэтому ошибка равна самой функции.
    const TensorBasis<2> basis(4);
    const std::vector<Real> zero(static_cast<std::size_t>(basis.numPoints()), 0.0);

    const ErrorNorms n = errorNorms<2>(
        basis, zero,
        [](const Point<2>& p) { return p[0] * p[0] * p[1] * p[1]; },
        [](const Point<2>& p) {
            return Point<2>{2 * p[0] * p[1] * p[1], 2 * p[0] * p[0] * p[1]};
        });

    CHECK_THAT(n.c, WithinAbs(1, 1e-14));            // max в углах
    CHECK_THAT(n.c1, WithinAbs(1 + 2 + 2, 1e-13));    // 1 + max|e_x| + max|e_y|
    CHECK_THAT(n.l1, WithinAbs(4.0 / 9, 1e-14));      // (2/3)^2
    CHECK_THAT(n.l2, WithinAbs(2.0 / 5, 1e-14));      // sqrt((2/5)^2)
}

TEST_CASE("errorNorms<2>: exact values for e = sin(pi x) sin(pi y)", "[analysis][norms][normsNd]") {
    // Максимум в (±1/2, ±1/2) не совпадает с точками перебора: проверяем уточнение.
    // N чётное: x = 0 — узел GLL, изломы |e| совпадают с границами ячеек.
    const TensorBasis<2> basis(4);
    const std::vector<Real> zero(static_cast<std::size_t>(basis.numPoints()), 0.0);

    const ErrorNorms n = errorNorms<2>(
        basis, zero,
        [](const Point<2>& p) { return std::sin(pi * p[0]) * std::sin(pi * p[1]); },
        [](const Point<2>& p) {
            return Point<2>{pi * std::cos(pi * p[0]) * std::sin(pi * p[1]),
                            pi * std::sin(pi * p[0]) * std::cos(pi * p[1])};
        });

    CHECK_THAT(n.c, WithinRel(1.0, 1e-4));
    CHECK_THAT(n.c1, WithinRel(1 + 2 * pi, 1e-4));
    CHECK_THAT(n.l1, WithinAbs(16 / (pi * pi), 1e-12));  // (4/pi)^2
    CHECK_THAT(n.l2, WithinAbs(1, 1e-12));               // sqrt(1 * 1)
}

TEST_CASE("errorNorms<3>: exact values for e = x y z", "[analysis][norms][normsNd]") {
    const TensorBasis<3> basis(2);
    const std::vector<Real> zero(static_cast<std::size_t>(basis.numPoints()), 0.0);
    sem::analysis::NormOptions options;
    options.samplesPerInterval = 6;

    const ErrorNorms n = errorNorms<3>(
        basis, zero,
        [](const Point<3>& p) { return p[0] * p[1] * p[2]; },
        [](const Point<3>& p) { return Point<3>{p[1] * p[2], p[0] * p[2], p[0] * p[1]}; },
        options);

    CHECK_THAT(n.c, WithinAbs(1, 1e-14));
    CHECK_THAT(n.c1, WithinAbs(4, 1e-14));
    CHECK_THAT(n.l1, WithinAbs(1, 1e-14));                          // (int |x|)^3 = 1
    CHECK_THAT(n.l2, WithinAbs(std::sqrt(8.0 / 27), 1e-14));        // sqrt((2/3)^3)
}

TEST_CASE("errorNorms<2>: interpolant of a polynomial of degree <= N has zero error",
          "[analysis][norms][normsNd]") {
    const TensorBasis<2> basis(5);
    const auto p = [](const Point<2>& q) {
        return std::pow(q[0], 5) * q[1] - 2 * q[0] * std::pow(q[1], 4) + 0.5;
    };
    const auto gradP = [](const Point<2>& q) {
        return Point<2>{5 * std::pow(q[0], 4) * q[1] - 2 * std::pow(q[1], 4),
                        std::pow(q[0], 5) - 8 * q[0] * std::pow(q[1], 3)};
    };

    const ErrorNorms n = errorNorms<2>(basis, basis.interpolate(p), p, gradP);
    CHECK(n.c < 1e-13);
    CHECK(n.c1 < 1e-12);
    CHECK(n.l1 < 1e-13);
    CHECK(n.l2 < 1e-13);
}

TEST_CASE("errorNorms<1>: agrees with the 1D SpectralBasis1D version", "[analysis][norms][normsNd]") {
    const int N = 6;
    const TensorBasis<1> tb(N);
    const sem::SpectralBasis1D sb(N);
    const auto u  = [](Real x) { return std::exp(x) * std::sin(2 * x); };
    const auto du = [](Real x) { return std::exp(x) * (std::sin(2 * x) + 2 * std::cos(2 * x)); };

    const std::vector<Real> nodal = tb.interpolate([&](const Point<1>& p) { return u(p[0]); });
    const std::vector<Real> bp{-1, 1};

    const ErrorNorms a = errorNorms<1>(
        tb, nodal, [&](const Point<1>& p) { return u(p[0]); },
        [&](const Point<1>& p) { return Point<1>{du(p[0])}; });
    const ErrorNorms b = errorNorms(sb, bp, nodal, u, du);

    CHECK_THAT(a.c, WithinRel(b.c, 1e-12));
    CHECK_THAT(a.c1, WithinRel(b.c1, 1e-12));
    CHECK_THAT(a.l1, WithinRel(b.l1, 1e-12));
    CHECK_THAT(a.l2, WithinRel(b.l2, 1e-12));
}

TEST_CASE("errorNorms<2>: spectral convergence of the interpolant", "[analysis][norms][normsNd]") {
    const auto f = [](const Point<2>& p) { return std::exp(p[0]) * std::sin(2 * p[1]); };
    const auto gradF = [](const Point<2>& p) {
        return Point<2>{std::exp(p[0]) * std::sin(2 * p[1]), 2 * std::exp(p[0]) * std::cos(2 * p[1])};
    };

    Real previous = 1;
    for (int N = 4; N <= 12; N += 2) {
        const TensorBasis<2> basis(N);
        const ErrorNorms n = errorNorms<2>(basis, basis.interpolate(f), f, gradF);
        CHECK(n.l2 < previous / 10);
        CHECK(n.c1 > n.c);
        previous = n.l2;
    }
}

TEST_CASE("errorNorms<2>: invalid arguments", "[analysis][norms][normsNd]") {
    const TensorBasis<2> basis(3);
    const auto f = [](const Point<2>&) { return Real{0}; };
    const auto g = [](const Point<2>&) { return Point<2>{0, 0}; };

    CHECK_THROWS_AS(errorNorms<2>(basis, std::vector<Real>(15), f, g), std::invalid_argument);

    sem::analysis::NormOptions bad;
    bad.quadraturePoints = 0;
    CHECK_THROWS_AS(errorNorms<2>(basis, std::vector<Real>(16), f, g, bad), std::invalid_argument);
}
