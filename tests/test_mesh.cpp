#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "sem/mesh/box_mesh.hpp"
#include "sem/operators/stiffness.hpp"
#include "sem/solvers/conjugate_gradient.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using sem::BoxMesh;
using sem::Point;
using sem::Real;

namespace {

Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real s = 0;
    for (std::size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
    return s;
}

std::vector<Real> pseudoRandom(std::size_t size, int seed) {
    std::vector<Real> v(size);
    for (std::size_t k = 0; k < size; ++k) v[k] = std::sin(1.7 * static_cast<Real>(k) + seed);
    return v;
}

template <int Dim>
std::vector<Real> applyK(const BoxMesh<Dim>& mesh, const std::vector<Real>& x) {
    std::vector<Real> y(x.size());
    mesh.applyK(x, y);
    return y;
}

// u^T K u для u = x_0^2 + x_1 + ... + x_{Dim-1} на [-1, 1]^Dim (квадратура точна при N >= 2):
//   int |grad u|^2 = int 4 x_0^2 + (Dim - 1) = 4 (2/3) 2^(Dim-1) + (Dim - 1) 2^Dim.
template <int Dim>
void checkEnergy(int degree, int elements) {
    const BoxMesh<Dim> mesh(degree, elements);
    const auto u = mesh.interpolate([](const Point<Dim>& p) {
        Real v = p[0] * p[0];
        for (std::size_t d = 1; d < static_cast<std::size_t>(Dim); ++d) v += p[d];
        return v;
    });
    const Real exact = 4.0 * (2.0 / 3) * std::pow(2.0, Dim - 1) + (Dim - 1) * std::pow(2.0, Dim);
    CHECK_THAT(dot(u, applyK(mesh, u)), WithinRel(exact, 1e-12));
}

// Сумма диагонали масс — объём куба (upper - lower)^Dim.
template <int Dim>
void checkVolume(int degree, int elements, Real lower, Real upper) {
    const BoxMesh<Dim> mesh(degree, elements, lower, upper);
    const std::vector<Real> mass = mesh.massDiagonal();
    Real sum = 0;
    for (const Real m : mass) sum += m;
    CHECK_THAT(sum, WithinRel(std::pow(upper - lower, Dim), 1e-13));
}

} // namespace

TEST_CASE("BoxMesh: sizes, node coordinates and shared nodes", "[mesh]") {
    const BoxMesh<2> mesh(3, 2, 0, 4);  // N = 3, K = 2, h = 2
    CHECK(mesh.numElements() == 4);
    CHECK(mesh.nodesPerDirection() == 7);
    CHECK(mesh.numNodes() == 49);
    CHECK(mesh.elementSize() == 2);
    CHECK(mesh.jacobian() == 1);

    CHECK(mesh.node(0) == Point<2>{0, 0});
    CHECK(mesh.node(6) == Point<2>{4, 0});
    CHECK(mesh.node(48) == Point<2>{4, 4});
    CHECK(mesh.node(3)[0] == 2);  // узел на границе элементов

    // Правый край элемента 0 и левый край элемента 1 — одни и те же глобальные узлы.
    const auto& basis = mesh.basis();
    for (int j = 0; j <= 3; ++j) {
        CHECK(mesh.globalIndex(0, basis.flatIndex({3, j})) == mesh.globalIndex(1, basis.flatIndex({0, j})));
    }
    // Узлы элемента совпадают с отображением узлов эталонного элемента.
    for (int e = 0; e < mesh.numElements(); ++e) {
        for (int k = 0; k < basis.numPoints(); ++k) {
            const Point<2> a = mesh.node(mesh.globalIndex(e, k));
            const Point<2> b = mesh.toPhysical(e, basis.point(k));
            CHECK_THAT(a[0], WithinAbs(b[0], 1e-14));
            CHECK_THAT(a[1], WithinAbs(b[1], 1e-14));
        }
    }

    // Граничных узлов M^2 - (M - 2)^2.
    int boundary = 0;
    for (const bool b : mesh.boundaryMask()) boundary += b ? 1 : 0;
    CHECK(boundary == 49 - 25);
}

TEST_CASE("BoxMesh: mass matrix integrates over the whole box", "[mesh]") {
    checkVolume<1>(4, 5, -1, 2);
    checkVolume<2>(3, 4, 0, 1);
    checkVolume<3>(2, 3, -2, 1);

    // Интеграл многочлена степени <= 2N - 1 по каждой переменной — точный.
    const BoxMesh<2> mesh(3, 3, 0, 3);
    const auto u = mesh.interpolate([](const Point<2>& p) { return p[0] * p[0] * p[1]; });
    CHECK_THAT(mesh.integrate(u), WithinRel(9.0 * 4.5, 1e-13));  // int x^2 = 9, int y = 4.5
}

TEST_CASE("BoxMesh: stiffness energy of a quadratic is exact", "[mesh]") {
    checkEnergy<1>(2, 3);
    checkEnergy<2>(2, 3);
    checkEnergy<2>(5, 4);
    checkEnergy<3>(3, 2);
}

TEST_CASE("BoxMesh: assembled K is symmetric, kills constants and matches one element", "[mesh]") {
    const BoxMesh<2> mesh(4, 3, -1, 2);
    const auto size = static_cast<std::size_t>(mesh.numNodes());

    const std::vector<Real> one(size, 1.0);
    for (const Real v : applyK(mesh, one)) CHECK_THAT(v, WithinAbs(0.0, 1e-12));

    const auto x = pseudoRandom(size, 1), y = pseudoRandom(size, 2);
    CHECK_THAT(dot(x, applyK(mesh, y)), WithinAbs(dot(y, applyK(mesh, x)), 1e-11));
    CHECK(dot(x, applyK(mesh, x)) > 0);

    // K = 1 на [-1, 1]^2 — это просто applyK эталонного элемента.
    const BoxMesh<2> single(5, 1);
    const auto z = pseudoRandom(static_cast<std::size_t>(single.numNodes()), 3);
    std::vector<Real> expected(z.size());
    sem::applyK<2>(single.basis(), z, expected);
    const auto actual = applyK(single, z);
    for (std::size_t k = 0; k < z.size(); ++k) CHECK_THAT(actual[k], WithinAbs(expected[k], 1e-12));
}

TEST_CASE("BoxMesh: diagonalK is the diagonal of the assembled K", "[mesh]") {
    const BoxMesh<2> mesh(3, 2, 0, 1);
    const auto diagonal = mesh.diagonalK();
    std::vector<Real> unit(static_cast<std::size_t>(mesh.numNodes()), 0.0);
    for (std::size_t k = 0; k < unit.size(); ++k) {
        unit[k] = 1;
        CHECK_THAT(applyK(mesh, unit)[k], WithinAbs(diagonal[k], 1e-12));
        unit[k] = 0;
    }
}

TEST_CASE("BoxMesh: evaluate reproduces polynomials, also on element faces", "[mesh]") {
    const BoxMesh<2> mesh(3, 3, -1, 2);
    const auto f = [](const Point<2>& p) { return p[0] * p[0] * p[0] - 2 * p[0] * p[1] + p[1] * p[1]; };
    const auto u = mesh.interpolate(f);
    for (const Point<2> p : {Point<2>{-1, -1}, Point<2>{2, 2}, Point<2>{0, 0.5}, Point<2>{1, 1},
                             Point<2>{0.3, -0.7}, Point<2>{1.9, 0.01}}) {
        CHECK_THAT(mesh.evaluate(u, p), WithinAbs(f(p), 1e-12));
    }

    // locate: точка на общей грани относится к элементу слева.
    const auto [element, xi] = mesh.locate({0, 0.5});
    CHECK(mesh.elementIndex(element) == std::array<int, 2>{0, 1});
    CHECK_THAT(xi[0], WithinAbs(1.0, 1e-14));
    CHECK_THAT(xi[1], WithinAbs(0.0, 1e-14));
}

TEST_CASE("BoxMesh: Poisson problem converges under h-refinement", "[mesh]") {
    // -Laplace u = 2 pi^2 sin(pi x) sin(pi y) в (0, 1)^2, u = 0 на границе; точное u = sin(pi x) sin(pi y).
    const Real pi = std::numbers::pi;
    const auto exact = [pi](const Point<2>& p) { return std::sin(pi * p[0]) * std::sin(pi * p[1]); };
    const int degree = 3;

    std::vector<Real> errors;
    for (const int elements : {2, 4, 8}) {
        const BoxMesh<2> mesh(degree, elements, 0, 1);
        const auto size = static_cast<std::size_t>(mesh.numNodes());
        const auto boundary = mesh.boundaryMask();
        const auto mass = mesh.massDiagonal();
        const auto f = mesh.interpolate([&](const Point<2>& p) { return 2 * pi * pi * exact(p); });

        std::vector<Real> b(size), diagonal = mesh.diagonalK(), masked(size);
        for (std::size_t k = 0; k < size; ++k) {
            b[k] = boundary[k] ? 0 : mass[k] * f[k];
            if (boundary[k]) diagonal[k] = 1;
        }
        const auto applyS = [&](std::span<const Real> in, std::span<Real> out) {
            for (std::size_t k = 0; k < size; ++k) masked[k] = boundary[k] ? 0 : in[k];
            mesh.applyK(masked, out);
            for (std::size_t k = 0; k < size; ++k) if (boundary[k]) out[k] = 0;
        };
        std::vector<Real> u(size, 0.0);
        const auto cg = sem::conjugateGradient(applyS, sem::JacobiPreconditioner(diagonal), b, u);
        REQUIRE(cg.converged);

        Real error = 0;
        for (std::size_t k = 0; k < size; ++k) {
            error = std::max(error, std::abs(u[k] - exact(mesh.node(static_cast<int>(k)))));
        }
        errors.push_back(error);
    }
    // Порядок около N + 1 = 4: при вдвое меньшем h ошибка падает хотя бы в 8 раз.
    CHECK(errors[0] / errors[1] > 8);
    CHECK(errors[1] / errors[2] > 8);
}

TEST_CASE("BoxMesh: invalid arguments", "[mesh]") {
    CHECK_THROWS_AS(BoxMesh<2>(3, 0), std::invalid_argument);
    CHECK_THROWS_AS(BoxMesh<2>(3, 2, 1, 1), std::invalid_argument);
    const BoxMesh<2> mesh(2, 2);
    std::vector<Real> wrong(5), y(static_cast<std::size_t>(mesh.numNodes()));
    CHECK_THROWS_AS(mesh.applyK(wrong, y), std::invalid_argument);
    CHECK_THROWS_AS(mesh.evaluate(wrong, {0, 0}), std::invalid_argument);
}
