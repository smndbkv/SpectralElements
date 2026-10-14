#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sem/basis/quadrature.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::PointsType;
using sem::Quadrature1D;
using sem::Real;

namespace {

void checkRule(const Quadrature1D& q, const std::vector<Real>& points,
               const std::vector<Real>& weights) {
    REQUIRE(q.points.size() == points.size());
    REQUIRE(q.weights.size() == weights.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        CHECK_THAT(q.points[i], WithinAbs(points[i], 1e-15));
        CHECK_THAT(q.weights[i], WithinAbs(weights[i], 1e-15));
    }
}

void checkStructure(const Quadrature1D& q) {
    const std::size_t n = q.points.size();
    REQUIRE(q.weights.size() == n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(q.points[i] >= -1);
        CHECK(q.points[i] <= 1);
        CHECK(q.weights[i] > 0);
        // Симметрия задаётся построением и потому точная.
        CHECK(q.points[i] == -q.points[n - 1 - i]);
        CHECK(q.weights[i] == q.weights[n - 1 - i]);
        if (i > 0) {
            CHECK(q.points[i] > q.points[i - 1]);
        }
    }
    CHECK_THAT(std::accumulate(q.weights.begin(), q.weights.end(), Real{0}),
               WithinAbs(2, 1e-13));
}

} // namespace

TEST_CASE("GLL: known low-order rules", "[basis][quadrature]") {
    const Real s5  = 1 / std::sqrt(5.0);
    const Real s37 = std::sqrt(3.0 / 7.0);

    checkRule(sem::gaussLobattoLegendre(2), {-1, 1}, {1, 1});
    checkRule(sem::gaussLobattoLegendre(3), {-1, 0, 1}, {1.0 / 3, 4.0 / 3, 1.0 / 3});
    checkRule(sem::gaussLobattoLegendre(4), {-1, -s5, s5, 1},
              {1.0 / 6, 5.0 / 6, 5.0 / 6, 1.0 / 6});
    checkRule(sem::gaussLobattoLegendre(5), {-1, -s37, 0, s37, 1},
              {1.0 / 10, 49.0 / 90, 32.0 / 45, 49.0 / 90, 1.0 / 10});
}

TEST_CASE("GL: known low-order rules", "[basis][quadrature]") {
    const Real s3  = 1 / std::sqrt(3.0);
    const Real s35 = std::sqrt(3.0 / 5.0);

    checkRule(sem::gaussLegendre(1), {0}, {2});
    checkRule(sem::gaussLegendre(2), {-s3, s3}, {1, 1});
    checkRule(sem::gaussLegendre(3), {-s35, 0, s35}, {5.0 / 9, 8.0 / 9, 5.0 / 9});
}

TEST_CASE("GLL: structure for many orders", "[basis][quadrature]") {
    const int numPoints = GENERATE(range(2, 66));
    const Quadrature1D q = sem::gaussLobattoLegendre(numPoints);

    CHECK(q.type == PointsType::GaussLobattoLegendre);
    CHECK(q.size() == numPoints);
    CHECK(q.points.front() == -1);
    CHECK(q.points.back() == 1);
    checkStructure(q);
}

TEST_CASE("GL: structure for many orders", "[basis][quadrature]") {
    const int numPoints = GENERATE(range(1, 66));
    const Quadrature1D q = sem::gaussLegendre(numPoints);

    CHECK(q.type == PointsType::GaussLegendre);
    CHECK(q.size() == numPoints);
    CHECK(q.points.front() > -1);
    CHECK(q.points.back() < 1);
    checkStructure(q);
}

TEST_CASE("GLL: exact for polynomials of degree <= 2N-1", "[basis][quadrature]") {
    const int numPoints = GENERATE(range(2, 21));
    const int N = numPoints - 1;
    const Quadrature1D q = sem::gaussLobattoLegendre(numPoints);

    for (int k = 0; k <= 2 * N - 1; ++k) {
        const Real approx = sem::integrate(q, [k](Real x) { return std::pow(x, k); });
        CHECK_THAT(approx, WithinAbs(sem::test::monomialIntegral(k), 1e-13));
    }
}

TEST_CASE("GLL: not exact for degree 2N (inexact SEM mass matrix)", "[basis][quadrature]") {
    const int numPoints = GENERATE(range(2, 9));
    const int k = 2 * (numPoints - 1);
    const Quadrature1D q = sem::gaussLobattoLegendre(numPoints);

    const Real approx = sem::integrate(q, [k](Real x) { return std::pow(x, k); });
    CHECK(std::abs(approx - sem::test::monomialIntegral(k)) > 1e-8);
}

TEST_CASE("GL: exact for polynomials of degree <= 2M-1", "[basis][quadrature]") {
    const int M = GENERATE(range(1, 21));
    const Quadrature1D q = sem::gaussLegendre(M);

    for (int k = 0; k <= 2 * M - 1; ++k) {
        const Real approx = sem::integrate(q, [k](Real x) { return std::pow(x, k); });
        CHECK_THAT(approx, WithinAbs(sem::test::monomialIntegral(k), 1e-13));
    }
}

TEST_CASE("Quadrature: factory and invalid arguments", "[basis][quadrature]") {
    CHECK(sem::makeQuadrature(PointsType::GaussLobattoLegendre, 6).points
          == sem::gaussLobattoLegendre(6).points);
    CHECK(sem::makeQuadrature(PointsType::GaussLegendre, 6).points
          == sem::gaussLegendre(6).points);

    CHECK_THROWS_AS(sem::gaussLobattoLegendre(1), std::invalid_argument);
    CHECK_THROWS_AS(sem::gaussLegendre(0), std::invalid_argument);
}
