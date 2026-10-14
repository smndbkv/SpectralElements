#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "sem/basis/legendre.hpp"
#include "sem/basis/spectral_basis.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::DenseMatrix;
using sem::Real;
using sem::SpectralBasis1D;

namespace {

// Явная формула матрицы дифференцирования на GLL-узлах — независимая проверка
// общей барицентрической формулы.
DenseMatrix analyticGllDerivative(const SpectralBasis1D& basis) {
    const int N = basis.degree();
    const auto x = basis.points();
    DenseMatrix D(N + 1, N + 1);
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            const auto ui = static_cast<std::size_t>(i);
            const auto uj = static_cast<std::size_t>(j);
            if (i != j) {
                D(i, j) = sem::legendre(N, x[ui]).value
                          / (sem::legendre(N, x[uj]).value * (x[ui] - x[uj]));
            }
        }
    }
    D(0, 0) = -N * (N + 1) / 4.0;
    D(N, N) = N * (N + 1) / 4.0;
    return D;
}

DenseMatrix diag(std::span<const Real> w) {
    DenseMatrix W(static_cast<int>(w.size()), static_cast<int>(w.size()));
    for (int i = 0; i < W.rows(); ++i) {
        W(i, i) = w[static_cast<std::size_t>(i)];
    }
    return W;
}

} // namespace

TEST_CASE("SpectralBasis1D: sizes and consistency", "[basis][spectral]") {
    const SpectralBasis1D basis(6);
    CHECK(basis.degree() == 6);
    CHECK(basis.numPoints() == 7);
    CHECK(basis.points().size() == 7);
    CHECK(basis.weights().size() == 7);
    CHECK(basis.D().rows() == 7);
    CHECK(basis.D().cols() == 7);
    CHECK(sem::test::maxAbsDiff(basis.Dt(), basis.D().transposed()) == 0);

    // Узлы базиса совпадают с узлами квадратуры — это и есть коллокация.
    const auto lagrangeNodes = basis.lagrange().nodes();
    for (std::size_t i = 0; i < basis.points().size(); ++i) {
        CHECK(lagrangeNodes[i] == basis.points()[i]);
    }
}

TEST_CASE("SpectralBasis1D: D matches the analytic GLL formula", "[basis][spectral]") {
    const int N = GENERATE(range(1, 25));
    const SpectralBasis1D basis(N);
    CHECK(sem::test::maxAbsDiff(basis.D(), analyticGllDerivative(basis)) < 1e-12 * N * N);
}

TEST_CASE("SpectralBasis1D: D differentiates polynomials of degree <= N exactly",
          "[basis][spectral]") {
    const int N = GENERATE(range(1, 25));
    const SpectralBasis1D basis(N);
    const auto x = basis.points();

    std::vector<Real> u(x.size()), du(x.size());
    for (int k = 0; k <= N; ++k) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            u[i] = std::pow(x[i], k);
        }
        sem::multiply(basis.D(), u, du);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const Real exact = k == 0 ? 0 : k * std::pow(x[i], k - 1);
            CHECK_THAT(du[i], WithinAbs(exact, 1e-12 * N * N));
        }
    }
}

TEST_CASE("SpectralBasis1D: summation-by-parts W D + D^T W = diag(-1, 0, ..., 0, 1)",
          "[basis][spectral]") {
    const int N = GENERATE(range(1, 33));
    const SpectralBasis1D basis(N);

    const DenseMatrix W = diag(basis.weights());
    const DenseMatrix Q = W * basis.D();  // Q_ij = int ell_i ell_j' dx (точно для GLL)
    const DenseMatrix Qt = Q.transposed();

    DenseMatrix sum(N + 1, N + 1);
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            sum(i, j) = Q(i, j) + Qt(i, j);
        }
    }

    DenseMatrix B(N + 1, N + 1);  // граничный член [ell_i ell_j] на концах
    B(0, 0) = -1;
    B(N, N) = 1;

    CHECK(sem::test::maxAbsDiff(sum, B) < 1e-12);
}

TEST_CASE("SpectralBasis1D: stiffness matrix D^T W D is symmetric with constant kernel",
          "[basis][spectral]") {
    const int N = GENERATE(range(1, 21));
    const SpectralBasis1D basis(N);
    const DenseMatrix K = basis.Dt() * diag(basis.weights()) * basis.D();

    CHECK(sem::test::maxAbsDiff(K, K.transposed()) < 1e-11 * N * N);

    const std::vector<Real> ones(static_cast<std::size_t>(N + 1), 1.0);
    std::vector<Real> Kones(ones.size());
    sem::multiply(K, ones, Kones);
    for (const Real v : Kones) {
        CHECK_THAT(v, WithinAbs(0, 1e-11 * N * N));
    }
}

TEST_CASE("SpectralBasis1D: spectral convergence of the derivative of sin(pi x)",
          "[basis][spectral]") {
    const Real pi = std::numbers::pi;
    const auto maxError = [pi](int N) {
        const SpectralBasis1D basis(N);
        const auto x = basis.points();
        std::vector<Real> u(x.size()), du(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(pi * x[i]);
        }
        sem::multiply(basis.D(), u, du);
        Real err = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            err = std::max(err, std::abs(du[i] - pi * std::cos(pi * x[i])));
        }
        return err;
    };

    // Для производной асимптотический режим наступает примерно с N = 6.
    Real previous = maxError(6);
    for (int N = 8; N <= 16; N += 2) {
        const Real current = maxError(N);
        CHECK(current < previous / 10);
        previous = current;
    }
    CHECK(maxError(24) < 1e-11);
}

TEST_CASE("SpectralBasis1D: interpolate and evaluate", "[basis][spectral]") {
    const SpectralBasis1D basis(5);
    const auto p = [](Real x) { return 3 * std::pow(x, 5) - x * x + 0.25; };  // степень <= N

    const std::vector<Real> u = basis.interpolate(p);
    REQUIRE(static_cast<int>(u.size()) == basis.numPoints());
    for (std::size_t i = 0; i < u.size(); ++i) {
        CHECK(u[i] == p(basis.points()[i]));  // узловые значения
    }
    for (const Real x : {-0.9, -0.33, 0.0, 0.41, 0.77}) {
        CHECK_THAT(basis.evaluate(u, x), WithinAbs(p(x), 1e-13));  // полином воспроизводится точно
    }
}

TEST_CASE("SpectralBasis1D: invalid degree", "[basis][spectral]") {
    CHECK_THROWS_AS(SpectralBasis1D(0), std::invalid_argument);
}
