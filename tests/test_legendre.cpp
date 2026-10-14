#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>

#include "sem/basis/legendre.hpp"
#include "test_helpers.hpp"

using Catch::Matchers::WithinAbs;
using sem::Real;

TEST_CASE("legendre: matches explicit low-degree formulas", "[basis][legendre]") {
    for (const Real x : sem::test::linspace(-1, 1, 17)) {
        const Real x2 = x * x;
        CHECK_THAT(sem::legendre(0, x).value, WithinAbs(1, 1e-15));
        CHECK_THAT(sem::legendre(1, x).value, WithinAbs(x, 1e-15));
        CHECK_THAT(sem::legendre(2, x).value, WithinAbs((3 * x2 - 1) / 2, 1e-14));
        CHECK_THAT(sem::legendre(3, x).value, WithinAbs((5 * x2 - 3) * x / 2, 1e-14));
        CHECK_THAT(sem::legendre(4, x).value, WithinAbs((35 * x2 * x2 - 30 * x2 + 3) / 8, 1e-14));

        CHECK_THAT(sem::legendre(0, x).derivative, WithinAbs(0, 1e-15));
        CHECK_THAT(sem::legendre(1, x).derivative, WithinAbs(1, 1e-15));
        CHECK_THAT(sem::legendre(2, x).derivative, WithinAbs(3 * x, 1e-14));
        CHECK_THAT(sem::legendre(3, x).derivative, WithinAbs((15 * x2 - 3) / 2, 1e-14));
        CHECK_THAT(sem::legendre(4, x).derivative, WithinAbs((140 * x2 * x - 60 * x) / 8, 1e-14));
    }
}

TEST_CASE("legendre: endpoint values", "[basis][legendre]") {
    const int n = GENERATE(range(0, 51));
    const Real sign = n % 2 == 0 ? 1 : -1;
    const Real endSlope = n * (n + 1) / 2.0;  // L_n'(1)

    CHECK_THAT(sem::legendre(n, 1).value, WithinAbs(1, 1e-13));
    CHECK_THAT(sem::legendre(n, -1).value, WithinAbs(sign, 1e-13));
    CHECK_THAT(sem::legendre(n, 1).derivative, WithinAbs(endSlope, 1e-12 * (1 + endSlope)));
    CHECK_THAT(sem::legendre(n, -1).derivative, WithinAbs(-sign * endSlope, 1e-12 * (1 + endSlope)));
}

TEST_CASE("lobattoQ: q = (2n+1)/(n(n+1)) (x^2-1) L_n' and q' = (2n+1) L_n", "[basis][legendre]") {
    const int n = GENERATE(range(1, 31));
    for (const Real x : sem::test::linspace(-1, 1, 23)) {
        const sem::LobattoQ r = sem::lobattoQ(n, x);
        const sem::LegendreValue L = sem::legendre(n, x);
        const Real expectedQ = (2.0 * n + 1) / (n * (n + 1.0)) * (x * x - 1) * L.derivative;

        CHECK_THAT(r.q, WithinAbs(expectedQ, 1e-12));
        CHECK_THAT(r.dq, WithinAbs((2 * n + 1) * L.value, 1e-12));
        CHECK_THAT(r.legendre, WithinAbs(L.value, 1e-13));
    }
}

TEST_CASE("legendre: invalid arguments", "[basis][legendre]") {
    CHECK_THROWS_AS(sem::legendre(-1, 0.5), std::invalid_argument);
    CHECK_THROWS_AS(sem::lobattoQ(0, 0.5), std::invalid_argument);
}
