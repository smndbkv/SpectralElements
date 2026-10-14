#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "sem/core/dense_matrix.hpp"
#include "test_helpers.hpp"

using sem::DenseMatrix;
using sem::Real;

namespace {

DenseMatrix example2x3() {
    DenseMatrix a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    return a;
}

} // namespace

TEST_CASE("DenseMatrix: construction and element access", "[core]") {
    DenseMatrix a(2, 3, 7.0);
    CHECK(a.rows() == 2);
    CHECK(a.cols() == 3);
    CHECK(a.data().size() == 6);
    CHECK(a(1, 2) == 7.0);

    a(1, 2) = -1;
    CHECK(a.row(1)[2] == -1);

    CHECK_THROWS_AS(DenseMatrix(-1, 2), std::invalid_argument);
}

TEST_CASE("DenseMatrix: transpose and identity", "[core]") {
    const DenseMatrix a = example2x3();
    const DenseMatrix t = a.transposed();
    REQUIRE(t.rows() == 3);
    REQUIRE(t.cols() == 2);
    CHECK(t(2, 1) == 6);
    CHECK(t(0, 1) == 4);

    CHECK(sem::test::maxAbsDiff(DenseMatrix::identity(2) * a, a) == 0);
    CHECK(sem::test::maxAbsDiff(a * DenseMatrix::identity(3), a) == 0);
}

TEST_CASE("DenseMatrix: products", "[core]") {
    const DenseMatrix a = example2x3();

    const DenseMatrix aat = a * a.transposed();
    CHECK(aat(0, 0) == 14);
    CHECK(aat(0, 1) == 32);
    CHECK(aat(1, 1) == 77);

    const std::vector<Real> x{1, 0, -1};
    std::vector<Real> y(2);
    sem::multiply(a, x, y);
    CHECK(y[0] == -2);
    CHECK(y[1] == -2);

    CHECK_THROWS_AS(a * a, std::invalid_argument);
    std::vector<Real> wrong(3);
    CHECK_THROWS_AS(sem::multiply(a, x, wrong), std::invalid_argument);
}
