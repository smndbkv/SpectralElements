#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "sem/analysis/error_norms.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "sem/core/dense_matrix.hpp"
#include "sem/core/linear_solve.hpp"
#include "sem/operators/stiffness.hpp"
#include "sem/solvers/conjugate_gradient.hpp"

using Catch::Matchers::WithinAbs;
using sem::CgOptions;
using sem::CgResult;
using sem::DenseMatrix;
using sem::Point;
using sem::Real;

namespace {

// Симметричная положительно определённая матрица A = B^T B + I.
DenseMatrix spdMatrix(int n) {
    DenseMatrix B(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B(i, j) = std::sin(1.0 + i + 3 * j);
        }
    }
    DenseMatrix A = B.transposed() * B;
    for (int i = 0; i < n; ++i) {
        A(i, i) += 1;
    }
    return A;
}

auto matrixOperator(const DenseMatrix& A) {
    return [&A](std::span<const Real> in, std::span<Real> out) { sem::multiply(A, in, out); };
}

std::vector<Real> rhs(int n) {
    std::vector<Real> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        b[static_cast<std::size_t>(i)] = std::cos(0.3 * i);
    }
    return b;
}

} // namespace

TEST_CASE("conjugateGradient: dense SPD system agrees with Gaussian elimination", "[solvers][cg]") {
    const int n = 20;
    const DenseMatrix A = spdMatrix(n);
    const std::vector<Real> b = rhs(n);
    const std::vector<Real> exact = sem::solveGauss(A, b);

    std::vector<Real> x(static_cast<std::size_t>(n), 0.0);
    const CgResult result = sem::conjugateGradient(matrixOperator(A), b, x, {.relativeTolerance = 1e-13});

    CHECK(result.converged);
    CHECK(result.relativeResidual <= 1e-13);
    CHECK(result.iterations <= 3 * n);  // в точной арифметике не больше n
    for (std::size_t k = 0; k < x.size(); ++k) {
        CHECK_THAT(x[k], WithinAbs(exact[k], 1e-10));
    }
}

TEST_CASE("conjugateGradient: 2D problem with applyK, Dirichlet mask and Jacobi",
          "[solvers][cg]") {
    // Delta u + u = f на (-1, 1)^2, u = 0 на границе: (K - M) u = -M f.
    // Положительно определено, т. к. min собств. знач. -Delta = pi^2 / 2 > 1.
    const Real pi = std::numbers::pi;
    const auto u = [pi](const Point<2>& p) { return std::sin(pi * p[0]) * std::sin(pi * p[1]); };
    const auto gradU = [pi](const Point<2>& p) {
        return Point<2>{pi * std::cos(pi * p[0]) * std::sin(pi * p[1]),
                        pi * std::sin(pi * p[0]) * std::cos(pi * p[1])};
    };

    const sem::TensorBasis<2> basis(16);
    const auto size = static_cast<std::size_t>(basis.numPoints());
    std::vector<bool> boundary(size);
    for (std::size_t k = 0; k < size; ++k) {
        const auto [i, j] = basis.multiIndex(static_cast<int>(k));
        boundary[k] = i == 0 || i == basis.degree() || j == 0 || j == basis.degree();
    }

    // S x = P (K - M) P x: маска P обнуляет граничные узлы до и после оператора.
    const auto applyS = [&](std::span<const Real> in, std::span<Real> out) {
        std::vector<Real> px(in.begin(), in.end());
        for (std::size_t k = 0; k < size; ++k) {
            if (boundary[k]) px[k] = 0;
        }
        sem::applyK<2>(basis, px, out);
        for (std::size_t k = 0; k < size; ++k) {
            out[k] = boundary[k] ? 0 : out[k] - basis.weight(static_cast<int>(k)) * px[k];
        }
    };

    std::vector<Real> b(size), diagonal = sem::diagonalK<2>(basis);
    for (std::size_t k = 0; k < size; ++k) {
        const Real w = basis.weight(static_cast<int>(k));
        b[k] = boundary[k] ? 0 : -w * (1 - 2 * pi * pi) * u(basis.point(static_cast<int>(k)));
        diagonal[k] = boundary[k] ? 1 : diagonal[k] - w;
    }

    std::vector<Real> plain(size, 0.0), jacobi(size, 0.0);
    const CgResult r1 = sem::conjugateGradient(applyS, b, plain);
    const CgResult r2 = sem::conjugateGradient(applyS, sem::JacobiPreconditioner(diagonal), b, jacobi);

    CHECK(r1.converged);
    CHECK(r2.converged);
    for (std::size_t k = 0; k < size; ++k) {
        CHECK_THAT(plain[k], WithinAbs(jacobi[k], 1e-10));
    }
    const auto norms = sem::analysis::errorNorms<2>(basis, jacobi, u, gradU);
    CHECK(norms.l2 < 1e-9);  // спектральная точность при N = 16
}

TEST_CASE("conjugateGradient: indefinite operator is detected", "[solvers][cg]") {
    DenseMatrix A(2, 2);
    A(0, 0) = 1;
    A(1, 1) = -1;
    const std::vector<Real> b{1, 1};
    std::vector<Real> x(2, 0.0);
    CHECK_THROWS_AS(sem::conjugateGradient(matrixOperator(A), b, x), std::runtime_error);
}

TEST_CASE("conjugateGradient: iteration limit, zero rhs and exact initial guess", "[solvers][cg]") {
    const int n = 20;
    const DenseMatrix A = spdMatrix(n);
    const std::vector<Real> b = rhs(n);

    std::vector<Real> x(static_cast<std::size_t>(n), 0.0);
    const CgResult limited = sem::conjugateGradient(matrixOperator(A), b, x, {.maxIterations = 2});
    CHECK_FALSE(limited.converged);
    CHECK(limited.iterations == 2);

    std::vector<Real> zero(static_cast<std::size_t>(n), 1.0);
    const CgResult trivial = sem::conjugateGradient(matrixOperator(A), std::vector<Real>(n, 0.0), zero);
    CHECK(trivial.converged);
    CHECK(trivial.iterations == 0);
    CHECK(zero == std::vector<Real>(static_cast<std::size_t>(n), 0.0));

    std::vector<Real> exact = sem::solveGauss(A, b);
    const CgResult already = sem::conjugateGradient(matrixOperator(A), b, exact, {.relativeTolerance = 1e-10});
    CHECK(already.converged);
    CHECK(already.iterations == 0);
}

TEST_CASE("conjugateGradient and Jacobi: invalid arguments", "[solvers][cg]") {
    const DenseMatrix A = spdMatrix(3);
    const std::vector<Real> b(3, 1.0);
    std::vector<Real> wrong(2, 0.0);
    CHECK_THROWS_AS(sem::conjugateGradient(matrixOperator(A), b, wrong), std::invalid_argument);

    const std::vector<Real> badDiagonal{1, 0, 2};
    CHECK_THROWS_AS(sem::JacobiPreconditioner(badDiagonal), std::invalid_argument);
}
