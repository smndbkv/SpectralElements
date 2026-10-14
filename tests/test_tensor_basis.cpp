#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sem/basis/tensor_apply.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::DenseMatrix;
using sem::Point;
using sem::Real;
using sem::TensorBasis;

namespace {

// Детерминированная «случайная» матрица.
DenseMatrix pseudoRandom(int rows, int cols, int seed) {
    DenseMatrix a(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            a(i, j) = std::sin(1.0 + seed + 3 * i + 7 * j);
        }
    }
    return a;
}

std::vector<Real> pseudoRandomVector(int size, int seed) {
    std::vector<Real> v(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
        v[static_cast<std::size_t>(i)] = std::cos(0.5 + seed + 1.3 * i);
    }
    return v;
}

// Кронекерово произведение B ⊗ C: индекс B — внешний (медленный), индекс C — внутренний.
// При раскладке flat = i + n_x * j оператор "A_y по y, A_x по x" равен kron(A_y, A_x).
DenseMatrix kron(const DenseMatrix& b, const DenseMatrix& c) {
    DenseMatrix k(b.rows() * c.rows(), b.cols() * c.cols());
    for (int i1 = 0; i1 < b.rows(); ++i1) {
        for (int j1 = 0; j1 < b.cols(); ++j1) {
            for (int i2 = 0; i2 < c.rows(); ++i2) {
                for (int j2 = 0; j2 < c.cols(); ++j2) {
                    k(i1 * c.rows() + i2, j1 * c.cols() + j2) = b(i1, j1) * c(i2, j2);
                }
            }
        }
    }
    return k;
}

std::vector<Real> matVec(const DenseMatrix& a, const std::vector<Real>& x) {
    std::vector<Real> y(static_cast<std::size_t>(a.rows()));
    sem::multiply(a, x, y);
    return y;
}

void checkClose(std::span<const Real> actual, std::span<const Real> expected, Real tol) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK_THAT(actual[i], WithinAbs(expected[i], tol));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// applyAlong / tensorApply
// ---------------------------------------------------------------------------

TEST_CASE("applyAlong<1> is a plain matrix-vector product", "[tensor][apply]") {
    const DenseMatrix A = pseudoRandom(4, 3, 0);
    const std::vector<Real> in = pseudoRandomVector(3, 0);
    std::vector<Real> out(4);
    sem::applyAlong<1>(A, 0, {3}, in, out);
    checkClose(out, matVec(A, in), 1e-14);
}

TEST_CASE("applyAlong<2> matches Kronecker products with the identity", "[tensor][apply]") {
    const std::array<int, 2> shape{3, 4};  // nx = 3, ny = 4
    const std::vector<Real> in = pseudoRandomVector(12, 1);

    SECTION("along x: I_ny ⊗ A, i.e. U A^T") {
        const DenseMatrix A = pseudoRandom(5, 3, 1);  // 3 -> 5 точек по x
        std::vector<Real> out(5 * 4);
        sem::applyAlong<2>(A, 0, shape, in, out);
        checkClose(out, matVec(kron(DenseMatrix::identity(4), A), in), 1e-14);
    }
    SECTION("along y: A ⊗ I_nx, i.e. A U") {
        const DenseMatrix A = pseudoRandom(2, 4, 2);  // 4 -> 2 точки по y
        std::vector<Real> out(3 * 2);
        sem::applyAlong<2>(A, 1, shape, in, out);
        checkClose(out, matVec(kron(A, DenseMatrix::identity(3)), in), 1e-14);
    }
}

TEST_CASE("applyAlong<3> matches Kronecker products with the identity", "[tensor][apply]") {
    const std::array<int, 3> shape{2, 3, 4};  // flat = i + 2 (j + 3 k)
    const std::vector<Real> in = pseudoRandomVector(24, 2);

    SECTION("along y: I_nz ⊗ A ⊗ I_nx") {
        const DenseMatrix A = pseudoRandom(2, 3, 3);
        std::vector<Real> out(2 * 2 * 4);
        sem::applyAlong<3>(A, 1, shape, in, out);
        const DenseMatrix full = kron(DenseMatrix::identity(4), kron(A, DenseMatrix::identity(2)));
        checkClose(out, matVec(full, in), 1e-14);
    }
    SECTION("along z: A ⊗ I_(nx*ny)") {
        const DenseMatrix A = pseudoRandom(3, 4, 4);
        std::vector<Real> out(2 * 3 * 3);
        sem::applyAlong<3>(A, 2, shape, in, out);
        checkClose(out, matVec(kron(A, DenseMatrix::identity(6)), in), 1e-14);
    }
}

TEST_CASE("tensorApply<2> matches the Kronecker product A_y ⊗ A_x", "[tensor][apply]") {
    const DenseMatrix Ax = pseudoRandom(4, 3, 5);  // 3 -> 4 по x
    const DenseMatrix Ay = pseudoRandom(5, 3, 6);  // 3 -> 5 по y
    const std::vector<Real> in = pseudoRandomVector(9, 3);
    std::vector<Real> out(4 * 5);

    sem::tensorApply<2>({&Ax, &Ay}, in, out);
    checkClose(out, matVec(kron(Ay, Ax), in), 1e-14);
}

TEST_CASE("tensorApply<3> matches the Kronecker product A_z ⊗ A_y ⊗ A_x", "[tensor][apply]") {
    const DenseMatrix Ax = pseudoRandom(2, 3, 7);
    const DenseMatrix Ay = pseudoRandom(4, 3, 8);
    const DenseMatrix Az = pseudoRandom(3, 3, 9);
    const std::vector<Real> in = pseudoRandomVector(27, 4);
    std::vector<Real> out(2 * 4 * 3);

    sem::tensorApply<3>({&Ax, &Ay, &Az}, in, out);
    checkClose(out, matVec(kron(Az, kron(Ay, Ax)), in), 1e-13);
}

TEST_CASE("applyAlong / tensorApply: invalid arguments", "[tensor][apply]") {
    const DenseMatrix A = pseudoRandom(3, 3, 0);
    std::vector<Real> in(9), out(9), wrong(8);

    CHECK_THROWS_AS(sem::applyAlong<2>(A, 2, {3, 3}, in, out), std::invalid_argument);
    CHECK_THROWS_AS(sem::applyAlong<2>(A, -1, {3, 3}, in, out), std::invalid_argument);
    CHECK_THROWS_AS(sem::applyAlong<2>(A, 0, {4, 3}, std::vector<Real>(12), out),
                    std::invalid_argument);  // A.cols() != shape[0]
    CHECK_THROWS_AS(sem::applyAlong<2>(A, 0, {3, 3}, wrong, out), std::invalid_argument);
    CHECK_THROWS_AS(sem::applyAlong<2>(A, 0, {3, 3}, in, wrong), std::invalid_argument);
    CHECK_THROWS_AS(sem::tensorApply<2>({&A, &A}, in, wrong), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TensorBasis
// ---------------------------------------------------------------------------

TEST_CASE("TensorBasis: sizes", "[tensor][basis2d]") {
    CHECK(TensorBasis<1>(4).numPoints() == 5);
    CHECK(TensorBasis<2>(4).numPoints() == 25);
    CHECK(TensorBasis<3>(4).numPoints() == 125);
    CHECK(TensorBasis<2>(4).numPointsPerDirection() == 5);
    CHECK(TensorBasis<2>(4).degree() == 4);
}

TEST_CASE("TensorBasis: flat index layout, x is fastest", "[tensor][basis2d]") {
    const TensorBasis<3> b(2);  // n = 3
    CHECK(b.flatIndex({0, 0, 0}) == 0);
    CHECK(b.flatIndex({1, 0, 0}) == 1);
    CHECK(b.flatIndex({0, 1, 0}) == 3);
    CHECK(b.flatIndex({0, 0, 1}) == 9);
    CHECK(b.flatIndex({2, 1, 2}) == 2 + 3 * (1 + 3 * 2));

    for (int flat = 0; flat < b.numPoints(); ++flat) {
        CHECK(b.flatIndex(b.multiIndex(flat)) == flat);
    }
}

TEST_CASE("TensorBasis: node coordinates and weights", "[tensor][basis2d]") {
    const TensorBasis<2> b(4);
    const auto x = b.basis1D().points();
    const auto w = b.basis1D().weights();

    // flat = i + n j  ->  точка (x_i, x_j), вес w_i w_j
    const Point<2> p = b.point(b.flatIndex({3, 1}));
    CHECK(p[0] == x[3]);
    CHECK(p[1] == x[1]);
    CHECK(b.weight(b.flatIndex({3, 1})) == w[3] * w[1]);

    Real sum2 = 0;
    for (int flat = 0; flat < b.numPoints(); ++flat) {
        sum2 += b.weight(flat);
    }
    CHECK_THAT(sum2, WithinAbs(4, 1e-13));  // площадь [-1,1]^2

    const TensorBasis<3> b3(3);
    Real sum3 = 0;
    for (int flat = 0; flat < b3.numPoints(); ++flat) {
        sum3 += b3.weight(flat);
    }
    CHECK_THAT(sum3, WithinAbs(8, 1e-13));  // объём [-1,1]^3
}

TEST_CASE("TensorBasis: interpolate gives f at the nodes", "[tensor][basis2d]") {
    const TensorBasis<2> b(5);
    const auto f = [](const Point<2>& p) { return std::sin(p[0]) * std::exp(p[1]); };
    const std::vector<Real> u = b.interpolate(f);

    REQUIRE(static_cast<int>(u.size()) == b.numPoints());
    for (int flat = 0; flat < b.numPoints(); ++flat) {
        CHECK(u[static_cast<std::size_t>(flat)] == f(b.point(flat)));
    }
}

TEST_CASE("TensorBasis: integrate", "[tensor][basis2d]") {
    const TensorBasis<2> b(3);  // точно до степени 5 по каждой переменной
    const auto u1 = b.interpolate([](const Point<2>& p) { return p[0] * p[0] * p[1] * p[1]; });
    CHECK_THAT(b.integrate(u1), WithinAbs(4.0 / 9, 1e-14));  // (2/3)^2

    const auto u2 = b.interpolate([](const Point<2>& p) { return std::pow(p[0], 4) * p[1] * p[1]; });
    CHECK_THAT(b.integrate(u2), WithinAbs(4.0 / 15, 1e-14));  // (2/5)(2/3)

    const TensorBasis<3> b3(2);
    const auto u3 = b3.interpolate([](const Point<3>& p) { return 1 + p[0] * p[1] * p[2]; });
    CHECK_THAT(b3.integrate(u3), WithinAbs(8, 1e-13));
}

TEST_CASE("TensorBasis<2>: evaluate reproduces polynomials of degree <= N", "[tensor][basis2d]") {
    const int N = GENERATE(range(1, 9));
    const TensorBasis<2> b(N);

    // p(x, y) = sum_{a, b <= N} c_ab x^a y^b, коэффициенты фиксированы
    const auto p = [N](const Point<2>& q) {
        Real s = 0;
        for (int a = 0; a <= N; ++a) {
            for (int c = 0; c <= N; ++c) {
                s += std::sin(1.0 + a + 2 * c) * std::pow(q[0], a) * std::pow(q[1], c);
            }
        }
        return s;
    };
    const std::vector<Real> u = b.interpolate(p);

    for (const Real x : sem::test::linspace(-1, 1, 7)) {
        for (const Real y : {-0.93, -0.31, 0.0, 0.44, 0.87}) {
            CHECK_THAT(b.evaluate(u, {x, y}), WithinAbs(p({x, y}), 1e-12));
        }
    }
    // В узле значение совпадает с узловым.
    const int flat = b.numPoints() / 2;
    CHECK_THAT(b.evaluate(u, b.point(flat)), WithinAbs(u[static_cast<std::size_t>(flat)], 1e-14));
}

TEST_CASE("TensorBasis<3>: evaluate reproduces a polynomial", "[tensor][basis2d]") {
    const TensorBasis<3> b(3);
    const auto p = [](const Point<3>& q) { return q[0] * q[1] * q[1] - 2 * q[2] * q[2] * q[2] + 0.5; };
    const std::vector<Real> u = b.interpolate(p);
    const Point<3> x{0.3, -0.7, 0.55};
    CHECK_THAT(b.evaluate(u, x), WithinAbs(p(x), 1e-13));
}

TEST_CASE("TensorBasis<2>: derivatives along x and y (non-symmetric polynomial)", "[tensor][basis2d]") {
    // u = x^3 y^2: если перепутать направления, результат будет другим.
    const TensorBasis<2> b(4);
    const auto u  = b.interpolate([](const Point<2>& p) { return std::pow(p[0], 3) * p[1] * p[1]; });
    const auto ux = b.interpolate([](const Point<2>& p) { return 3 * p[0] * p[0] * p[1] * p[1]; });
    const auto uy = b.interpolate([](const Point<2>& p) { return 2 * std::pow(p[0], 3) * p[1]; });

    std::vector<Real> du(u.size());
    b.derivative(u, 0, du);
    checkClose(du, ux, 1e-12);
    b.derivative(u, 1, du);
    checkClose(du, uy, 1e-12);
}

TEST_CASE("TensorBasis<3>: derivative along z", "[tensor][basis2d]") {
    const TensorBasis<3> b(3);
    const auto u  = b.interpolate([](const Point<3>& p) { return p[0] * p[1] * p[1] * std::pow(p[2], 3); });
    const auto uz = b.interpolate([](const Point<3>& p) { return 3 * p[0] * p[1] * p[1] * p[2] * p[2]; });
    std::vector<Real> du(u.size());
    b.derivative(u, 2, du);
    checkClose(du, uz, 1e-12);
}

TEST_CASE("TensorBasis<1>: consistent with SpectralBasis1D", "[tensor][basis2d]") {
    const TensorBasis<1> b(6);
    const auto u = b.interpolate([](const Point<1>& p) { return std::exp(p[0]); });
    std::vector<Real> du(u.size()), expected(u.size());
    b.derivative(u, 0, du);
    sem::multiply(b.basis1D().D(), u, expected);
    checkClose(du, expected, 1e-14);
}

TEST_CASE("TensorBasis: invalid arguments", "[tensor][basis2d]") {
    const TensorBasis<2> b(3);
    std::vector<Real> u(16), du(16), wrong(15);
    CHECK_THROWS_AS(b.derivative(u, 2, du), std::invalid_argument);
    CHECK_THROWS_AS(b.derivative(wrong, 0, du), std::invalid_argument);
    CHECK_THROWS_AS(b.evaluate(wrong, {0, 0}), std::invalid_argument);
}
