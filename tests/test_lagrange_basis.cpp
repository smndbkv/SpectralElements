#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sem/basis/lagrange_basis.hpp"
#include "sem/basis/quadrature.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::LagrangeBasis1D;
using sem::Real;

namespace {

std::vector<Real> gllNodes(int degree) {
    return sem::gaussLobattoLegendre(degree + 1).points;
}

// Полином степени `degree` с фиксированными «случайными» коэффициентами.
Real testPolynomial(int degree, Real x) {
    Real value = 0;
    for (int k = degree; k >= 0; --k) {
        value = value * x + std::sin(1.0 + k);
    }
    return value;
}

} // namespace

TEST_CASE("Lagrange: Kronecker property at the nodes", "[basis][lagrange]") {
    // GLL-узлы и равномерные узлы: класс работает на любых различных узлах.
    const std::vector<std::vector<Real>> nodeSets{gllNodes(8), sem::test::linspace(-1, 1, 6)};

    for (const auto& nodes : nodeSets) {
        const LagrangeBasis1D basis(nodes);
        std::vector<Real> values(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            basis.evaluateBasis(nodes[i], values);
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                CHECK(values[j] == (i == j ? 1.0 : 0.0));
            }
        }
    }
}

TEST_CASE("Lagrange: partition of unity", "[basis][lagrange]") {
    const int N = GENERATE(range(1, 17));
    const LagrangeBasis1D basis(gllNodes(N));
    std::vector<Real> values(static_cast<std::size_t>(N + 1));

    for (const Real x : sem::test::linspace(-1, 1, 101)) {
        basis.evaluateBasis(x, values);
        CHECK_THAT(std::accumulate(values.begin(), values.end(), Real{0}), WithinAbs(1, 1e-13));
    }
}

TEST_CASE("Lagrange: reproduces polynomials of degree <= N", "[basis][lagrange]") {
    const int N = GENERATE(range(1, 17));
    const LagrangeBasis1D basis(gllNodes(N));

    std::vector<Real> nodal;
    for (const Real x : basis.nodes()) {
        nodal.push_back(testPolynomial(N, x));
    }
    for (const Real x : sem::test::linspace(-1, 1, 57)) {
        CHECK_THAT(basis.evaluate(nodal, x), WithinAbs(testPolynomial(N, x), 1e-12));
    }
}

TEST_CASE("Lagrange: interpolation matrix agrees with evaluateBasis", "[basis][lagrange]") {
    const LagrangeBasis1D basis(gllNodes(7));
    const std::vector<Real> targets = sem::gaussLegendre(11).points;

    const sem::DenseMatrix I = basis.interpolationMatrix(targets);
    REQUIRE(I.rows() == 11);
    REQUIRE(I.cols() == 8);

    std::vector<Real> values(8);
    for (int i = 0; i < I.rows(); ++i) {
        basis.evaluateBasis(targets[static_cast<std::size_t>(i)], values);
        for (int j = 0; j < I.cols(); ++j) {
            CHECK(I(i, j) == values[static_cast<std::size_t>(j)]);
        }
    }
}

TEST_CASE("Lagrange: spectral convergence for sin(pi x) on GLL nodes", "[basis][lagrange]") {
    const auto f = [](Real x) { return std::sin(std::numbers::pi * x); };
    const std::vector<Real> samples = sem::test::linspace(-1, 1, 201);

    const auto maxError = [&](int N) {
        const LagrangeBasis1D basis(gllNodes(N));
        std::vector<Real> nodal;
        for (const Real x : basis.nodes()) {
            nodal.push_back(f(x));
        }
        Real err = 0;
        for (const Real x : samples) {
            err = std::max(err, std::abs(basis.evaluate(nodal, x) - f(x)));
        }
        return err;
    };

    // При N < 4 функция ещё не разрешена (доасимптотический режим), поэтому
    // скорость проверяем начиная с N = 4.
    Real previous = maxError(4);
    for (int N = 6; N <= 16; N += 2) {
        const Real current = maxError(N);
        CHECK(current < previous / 10);  // каждые +2 к степени — минимум порядок точности
        previous = current;
    }
    CHECK(maxError(20) < 1e-13);
}

TEST_CASE("Lagrange: invalid arguments", "[basis][lagrange]") {
    CHECK_THROWS_AS(LagrangeBasis1D(std::vector<Real>{}), std::invalid_argument);
    CHECK_THROWS_AS(LagrangeBasis1D(std::vector<Real>{0.0, 0.5, 0.5}), std::invalid_argument);

    const LagrangeBasis1D basis(gllNodes(3));
    std::vector<Real> wrongSize(3);
    CHECK_THROWS_AS(basis.evaluateBasis(0.1, wrongSize), std::invalid_argument);
}
