#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/spectral_basis.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using sem::Real;
using sem::SpectralBasis1D;
using sem::analysis::ErrorNorms;
using sem::analysis::errorNorms;

namespace {

const Real pi = std::numbers::pi;

const auto zero = [](Real) { return Real{0}; };

// Равномерное разбиение [a, b] на numElements элементов.
std::vector<Real> uniformBreakpoints(Real a, Real b, int numElements) {
    std::vector<Real> bp(static_cast<std::size_t>(numElements + 1));
    for (int k = 0; k <= numElements; ++k) {
        bp[static_cast<std::size_t>(k)] = a + (b - a) * k / numElements;
    }
    return bp;
}

// Поэлементные узловые значения интерполянта f на разбиении.
template <class F>
std::vector<Real> interpolant(const SpectralBasis1D& basis, const std::vector<Real>& bp, F&& f) {
    std::vector<Real> values;
    for (std::size_t k = 0; k + 1 < bp.size(); ++k) {
        const Real J = (bp[k + 1] - bp[k]) / 2;
        for (const Real xi : basis.points()) {
            values.push_back(f(bp[k] + (xi + 1) * J));
        }
    }
    return values;
}

} // namespace

TEST_CASE("errorNorms: exact values for e = x^2", "[analysis][norms]") {
    const std::vector<Real> bp{-1, 1};
    const ErrorNorms n = errorNorms([](Real x) { return x * x; }, [](Real x) { return 2 * x; },
                                    zero, zero, bp);
    CHECK_THAT(n.c, WithinAbs(1, 1e-15));
    CHECK_THAT(n.c1, WithinAbs(3, 1e-15));                       // 1 + max|2x|
    CHECK_THAT(n.l1, WithinAbs(2.0 / 3, 1e-14));
    CHECK_THAT(n.l2, WithinAbs(std::sqrt(2.0 / 5), 1e-14));
}

TEST_CASE("errorNorms: exact values for e = sin(pi x)", "[analysis][norms]") {
    const std::vector<Real> bp{-1, 1};
    const ErrorNorms n = errorNorms([](Real x) { return std::sin(pi * x); },
                                    [](Real x) { return pi * std::cos(pi * x); },
                                    zero, zero, bp);
    CHECK_THAT(n.c, WithinAbs(1, 1e-14));
    CHECK_THAT(n.c1, WithinAbs(1 + pi, 1e-14));
    CHECK_THAT(n.l1, WithinAbs(4 / pi, 1e-13));                  // знак меняется в нуле
    CHECK_THAT(n.l2, WithinAbs(1, 1e-13));
}

TEST_CASE("errorNorms (SEM): interpolant of a polynomial of degree <= N has zero error",
          "[analysis][norms]") {
    const int N = GENERATE(range(1, 11));
    const SpectralBasis1D basis(N);
    const auto bp = uniformBreakpoints(-1, 2, 3);

    const auto p  = [N](Real x) { return std::pow(x, N) - 0.5 * x + 1; };
    const auto dp = [N](Real x) { return N * std::pow(x, N - 1) - 0.5; };

    // Ошибка — только округление, поэтому порог относителен масштаба:
    // на [-1, 2] |p| <= 2^N + 2, |p'| <= N 2^(N-1) + 1.
    const Real scale      = std::pow(2.0, N) + 2;
    const Real scaleDeriv = N * std::pow(2.0, N - 1) + 1;

    const ErrorNorms n = errorNorms(basis, bp, interpolant(basis, bp, p), p, dp);
    CHECK(n.c < 1e-13 * scale);
    CHECK(n.c1 < 1e-13 * (scale + scaleDeriv));
    CHECK(n.l1 < 1e-13 * scale);
    CHECK(n.l2 < 1e-13 * scale);
}

TEST_CASE("errorNorms (SEM): agrees with the generic version on a fine subdivision",
          "[analysis][norms]") {
    const SpectralBasis1D basis(5);
    const auto bp = uniformBreakpoints(-1, 1, 3);
    const auto u  = [](Real x) { return std::exp(2 * x); };
    const auto du = [](Real x) { return 2 * std::exp(2 * x); };
    const auto nodal = interpolant(basis, bp, u);

    const ErrorNorms sem = errorNorms(basis, bp, nodal, u, du);

    // То же приближение как обычная функция: поиск элемента и интерполяция.
    const auto element = [&](Real x) {
        const auto k = static_cast<std::size_t>(std::min(2.0, std::floor((x + 1) * 1.5)));
        return k;
    };
    const auto v = [&](Real x) {
        const std::size_t k = element(x);
        const Real J = (bp[k + 1] - bp[k]) / 2;
        return basis.lagrange().evaluate(
            std::span<const Real>(nodal).subspan(k * 6, 6), (x - bp[k]) / J - 1);
    };
    const auto dv = [&](Real x) {
        const std::size_t k = element(x);
        const Real J = (bp[k + 1] - bp[k]) / 2;
        std::vector<Real> d(6);
        sem::multiply(basis.D(), std::span<const Real>(nodal).subspan(k * 6, 6), d);
        return basis.lagrange().evaluate(d, (x - bp[k]) / J - 1) / J;
    };

    sem::analysis::NormOptions fine;
    fine.subdivisions       = 600;
    fine.samplesPerInterval = 8;
    const ErrorNorms ref = errorNorms(u, du, v, dv, bp, fine);

    CAPTURE(sem.c, ref.c, sem.c1, ref.c1, sem.l1, ref.l1, sem.l2, ref.l2);
    CHECK_THAT(sem.c, WithinRel(ref.c, 1e-4));
    CHECK_THAT(sem.c1, WithinRel(ref.c1, 1e-4));
    CHECK_THAT(sem.l1, WithinRel(ref.l1, 1e-4));
    CHECK_THAT(sem.l2, WithinRel(ref.l2, 1e-6));
}

TEST_CASE("errorNorms (SEM): h-convergence rates N+1 (C, L1, L2) and N (C1)",
          "[analysis][norms]") {
    const int N = 3;
    const SpectralBasis1D basis(N);
    const auto u  = [](Real x) { return std::sin(pi * x); };
    const auto du = [](Real x) { return pi * std::cos(pi * x); };

    const auto norms = [&](int numElements) {
        const auto bp = uniformBreakpoints(-1, 1, numElements);
        return errorNorms(basis, bp, interpolant(basis, bp, u), u, du);
    };
    const ErrorNorms coarse = norms(8);
    const ErrorNorms fine   = norms(16);

    CHECK_THAT(std::log2(coarse.c / fine.c), WithinAbs(N + 1, 0.2));
    CHECK_THAT(std::log2(coarse.l1 / fine.l1), WithinAbs(N + 1, 0.2));
    CHECK_THAT(std::log2(coarse.l2 / fine.l2), WithinAbs(N + 1, 0.2));
    CHECK_THAT(std::log2(coarse.c1 / fine.c1), WithinAbs(N, 0.2));
}

TEST_CASE("errorNorms (SEM): spectral convergence in N", "[analysis][norms]") {
    const auto u  = [](Real x) { return std::sin(pi * x); };
    const auto du = [](Real x) { return pi * std::cos(pi * x); };
    const std::vector<Real> bp{-1, 1};

    Real previous = 1;
    for (int N = 6; N <= 16; N += 2) {
        const SpectralBasis1D basis(N);
        const ErrorNorms n = errorNorms(basis, bp, interpolant(basis, bp, u), u, du);
        CHECK(n.l2 < previous / 10);
        previous = n.l2;
    }
}

TEST_CASE("errorNorms (SEM): kink on an element boundary, one-sided derivatives",
          "[analysis][norms]") {
    // sin|x| гладкая на каждом элементе [-1, 0] и [0, 1]; в x = 0 у производной скачок.
    // Производная точного решения на конце элемента должна браться изнутри элемента,
    // иначе C1 ложно показывает скачок |(+1) - (-1)| = 2.
    const auto u  = [](Real x) { return std::sin(std::abs(x)); };
    const auto du = [](Real x) { return (x >= 0 ? 1.0 : -1.0) * std::cos(x); };
    const std::vector<Real> bp{-1, 0, 1};

    const SpectralBasis1D basis(12);
    const ErrorNorms n = errorNorms(basis, bp, interpolant(basis, bp, u), u, du);
    CHECK(n.c < 1e-13);
    CHECK(n.c1 < 1e-11);

    // То же для версии с произвольными функциями: v = u, dv = du.
    const ErrorNorms g = errorNorms(u, du, u, du, bp);
    CHECK(g.c1 == 0);
}

TEST_CASE("errorNorms: invalid arguments", "[analysis][norms]") {
    const SpectralBasis1D basis(2);
    const std::vector<Real> bp{-1, 1};
    const std::vector<Real> wrongSize(4);
    const std::vector<Real> notIncreasing{0, 0};
    const auto f = [](Real x) { return x; };

    CHECK_THROWS_AS(errorNorms(basis, bp, wrongSize, f, f), std::invalid_argument);
    CHECK_THROWS_AS(errorNorms(f, f, f, f, notIncreasing), std::invalid_argument);
    CHECK_THROWS_AS(errorNorms(f, f, f, f, std::vector<Real>{0}), std::invalid_argument);

    sem::analysis::NormOptions bad;
    bad.samplesPerInterval = 1;
    CHECK_THROWS_AS(errorNorms(f, f, f, f, bp, bad), std::invalid_argument);
}
