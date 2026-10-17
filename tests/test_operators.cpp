#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "sem/basis/tensor_apply.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "sem/operators/stiffness.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::DenseMatrix;
using sem::Point;
using sem::Real;
using sem::TensorBasis;

namespace {

Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real s = 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        s += a[k] * b[k];
    }
    return s;
}

std::vector<Real> pseudoRandom(std::size_t size, int seed) {
    std::vector<Real> v(size);
    for (std::size_t k = 0; k < size; ++k) {
        v[k] = std::sin(1.0 + seed + 0.7 * static_cast<Real>(k));
    }
    return v;
}

template <int Dim>
std::vector<Real> K(const TensorBasis<Dim>& basis, const std::vector<Real>& x) {
    std::vector<Real> y(x.size());
    sem::applyK<Dim>(basis, x, y);
    return y;
}

} // namespace

TEST_CASE("applyK: constants are in the kernel", "[operators][stiffness]") {
    const TensorBasis<1> b1(6);
    const TensorBasis<2> b2(5);
    const TensorBasis<3> b3(3);
    for (const Real v : K(b1, std::vector<Real>(7, 1.0))) CHECK_THAT(v, WithinAbs(0, 1e-12));
    for (const Real v : K(b2, std::vector<Real>(36, 1.0))) CHECK_THAT(v, WithinAbs(0, 1e-12));
    for (const Real v : K(b3, std::vector<Real>(64, 1.0))) CHECK_THAT(v, WithinAbs(0, 1e-12));
}

TEST_CASE("applyK: energy u^T K u equals the integral of |grad u|^2", "[operators][stiffness]") {
    // u = x^2 y: int (2xy)^2 + (x^2)^2 over [-1,1]^2 = 16/9 + 4/5 = 116/45.
    const TensorBasis<2> b2(4);
    const auto u2 = b2.interpolate([](const Point<2>& p) { return p[0] * p[0] * p[1]; });
    CHECK_THAT(dot(u2, K(b2, u2)), WithinAbs(116.0 / 45, 1e-13));

    // u = x y z: int (yz)^2 + (xz)^2 + (xy)^2 over [-1,1]^3 = 3 * 8/9 = 8/3.
    const TensorBasis<3> b3(3);
    const auto u3 = b3.interpolate([](const Point<3>& p) { return p[0] * p[1] * p[2]; });
    CHECK_THAT(dot(u3, K(b3, u3)), WithinAbs(8.0 / 3, 1e-13));
}

TEST_CASE("applyK: symmetric operator", "[operators][stiffness]") {
    const TensorBasis<2> basis(7);
    const auto x = pseudoRandom(64, 1);
    const auto z = pseudoRandom(64, 2);
    CHECK_THAT(dot(z, K(basis, x)), WithinAbs(dot(x, K(basis, z)), 1e-11));
}

TEST_CASE("applyK<1>: equals the 1D stiffness matrix D^T W D", "[operators][stiffness]") {
    const TensorBasis<1> basis(6);
    const auto& b1 = basis.basis1D();
    DenseMatrix WD = b1.D();
    for (int k = 0; k < WD.rows(); ++k) {
        for (int j = 0; j < WD.cols(); ++j) {
            WD(k, j) *= b1.weights()[static_cast<std::size_t>(k)];
        }
    }
    const DenseMatrix A = b1.Dt() * WD;

    const auto x = pseudoRandom(7, 3);
    std::vector<Real> expected(7);
    sem::multiply(A, x, expected);
    const auto actual = K(basis, x);
    for (std::size_t k = 0; k < x.size(); ++k) {
        CHECK_THAT(actual[k], WithinAbs(expected[k], 1e-12));
    }
}

TEST_CASE("applyK<2>: equals the Kronecker form M (x) A + A (x) M", "[operators][stiffness]") {
    const TensorBasis<2> basis(5);
    const auto& b1 = basis.basis1D();
    const int n = b1.numPoints();
    DenseMatrix WD = b1.D(), M(n, n);
    for (int k = 0; k < n; ++k) {
        const Real w = b1.weights()[static_cast<std::size_t>(k)];
        M(k, k) = w;
        for (int j = 0; j < n; ++j) {
            WD(k, j) *= w;
        }
    }
    const DenseMatrix A = b1.Dt() * WD;

    const auto x = pseudoRandom(36, 4);
    std::vector<Real> expected(36), tmp(36);
    sem::tensorApply<2>({&A, &M}, x, expected);
    sem::tensorApply<2>({&M, &A}, x, tmp);
    const auto actual = K(basis, x);
    for (std::size_t k = 0; k < x.size(); ++k) {
        CHECK_THAT(actual[k], WithinAbs(expected[k] + tmp[k], 1e-12));
    }
}

TEST_CASE("diagonalK: equals the diagonal of K obtained by applying it to unit vectors",
          "[operators][stiffness]") {
    const auto check = [](const auto& basis) {
        const auto diagonal = sem::diagonalK(basis);
        const auto size = static_cast<std::size_t>(basis.numPoints());
        std::vector<Real> e(size, 0.0), column(size);
        for (std::size_t j = 0; j < size; ++j) {
            e[j] = 1;
            sem::applyK(basis, std::span<const Real>(e), std::span<Real>(column));
            CHECK_THAT(diagonal[j], WithinAbs(column[j], 1e-12));
            e[j] = 0;
        }
    };
    check(TensorBasis<1>(6));
    check(TensorBasis<2>(4));
    check(TensorBasis<3>(2));
}

TEST_CASE("applyK: invalid sizes", "[operators][stiffness]") {
    const TensorBasis<2> basis(3);
    std::vector<Real> x(16), y(16), wrong(15);
    CHECK_THROWS_AS(sem::applyK<2>(basis, wrong, y), std::invalid_argument);
    CHECK_THROWS_AS(sem::applyK<2>(basis, x, wrong), std::invalid_argument);
}
