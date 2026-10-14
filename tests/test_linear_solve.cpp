#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "sem/core/dense_matrix.hpp"
#include "sem/core/linear_solve.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::DenseMatrix;
using sem::Real;
using sem::solveGauss;

namespace {

DenseMatrix fromRows(const std::vector<std::vector<Real>>& rows) {
    DenseMatrix a(static_cast<int>(rows.size()), static_cast<int>(rows.front().size()));
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            a(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    return a;
}

std::vector<Real> matVec(const DenseMatrix& a, const std::vector<Real>& x) {
    std::vector<Real> y(static_cast<std::size_t>(a.rows()));
    sem::multiply(a, x, y);
    return y;
}

void checkClose(const std::vector<Real>& actual, const std::vector<Real>& expected, Real tol) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK_THAT(static_cast<double>(actual[i]),
                   WithinAbs(static_cast<double>(expected[i]), static_cast<double>(tol)));
    }
}

} // namespace

TEST_CASE("solveGauss: small system with a zero leading entry", "[core][solve]") {
    // Без перестановки строк деление на A(0,0) = 0 было бы невозможно.
    const DenseMatrix A = fromRows({{0, 2, 1},
                                    {1, 1, 1},
                                    {2, 1, 0}});
    const std::vector<Real> exact{1, 2, 3};
    checkClose(solveGauss(A, matVec(A, exact)), exact, 1e-14);
}

TEST_CASE("solveGauss: pivoting keeps a tiny leading entry stable", "[core][solve]") {
    // Классический пример: без выбора главного элемента x_0 теряется полностью.
    const DenseMatrix A = fromRows({{1e-20, 1},
                                    {1, 1}});
    const std::vector<Real> b{1, 2};
    // Точное решение: x_0 = 1 / (1 - 1e-20), x_1 = (1 - 2e-20) / (1 - 1e-20); оба ~ 1.
    checkClose(solveGauss(A, b), {1, 1}, 1e-14);
}

TEST_CASE("solveGauss: symmetric indefinite matrix", "[core][solve]") {
    // Собственные значения 3 и -1: Холецкий здесь не работает, Гаусс — работает.
    const DenseMatrix A = fromRows({{1, 2},
                                    {2, 1}});
    const std::vector<Real> exact{-0.5, 2};
    checkClose(solveGauss(A, matVec(A, exact)), exact, 1e-14);
}

TEST_CASE("solveGauss: larger system, solution and residual", "[core][solve]") {
    const int n = 60;
    DenseMatrix A(n, n);
    std::vector<Real> exact(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = std::sin(Real(1 + i + 2 * j)) + (i == j ? Real(3) : Real(0));
        }
        exact[static_cast<std::size_t>(i)] = std::cos(Real(i));
    }
    const std::vector<Real> b = matVec(A, exact);
    const std::vector<Real> x = solveGauss(A, b);

    checkClose(x, exact, 1e-11);
    checkClose(matVec(A, x), b, 1e-11);
}

TEST_CASE("solveGauss: input matrix is not modified", "[core][solve]") {
    const DenseMatrix A = fromRows({{0, 1},
                                    {1, 0}});
    const DenseMatrix copy = A;
    const std::vector<Real> b{1, 2};
    (void)solveGauss(A, b);
    CHECK(sem::test::maxAbsDiff(A, copy) == 0);
}

TEST_CASE("solveGauss: edge sizes", "[core][solve]") {
    CHECK(solveGauss(DenseMatrix(0, 0), std::vector<Real>{}).empty());

    const std::vector<Real> x = solveGauss(fromRows({{4}}), std::vector<Real>{2});
    REQUIRE(x.size() == 1);
    CHECK(x[0] == Real(0.5));
}

TEST_CASE("solveGauss: singular matrix and invalid arguments", "[core][solve]") {
    const DenseMatrix singular = fromRows({{1, 2},
                                           {2, 4}});
    CHECK_THROWS_AS(solveGauss(singular, std::vector<Real>{1, 2}), std::runtime_error);
    CHECK_THROWS_AS(solveGauss(DenseMatrix(2, 2), std::vector<Real>{1, 2}), std::runtime_error);

    CHECK_THROWS_AS(solveGauss(DenseMatrix(2, 3), std::vector<Real>{1, 2}),
                    std::invalid_argument);
    CHECK_THROWS_AS(solveGauss(DenseMatrix::identity(2), std::vector<Real>{1, 2, 3}),
                    std::invalid_argument);
}
