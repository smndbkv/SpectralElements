#include "sem/basis/quadrature.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "sem/basis/legendre.hpp"

namespace sem {

namespace {

// Уточнение корня методом Ньютона; f(x) возвращает пару {f(x), f'(x)}.
// Начальное приближение должно быть близко к искомому корню.
template <class F>
Real newtonRoot(Real x, F&& f) {
    constexpr int  kMaxIterations = 100;
    constexpr Real kTolerance     = 1e-14;

    for (int it = 0; it < kMaxIterations; ++it) {
        const auto [value, derivative] = f(x);
        const Real delta = value / derivative;
        x -= delta;
        if (std::abs(delta) <= kTolerance) {
            // Ещё один шаг: квадратичная сходимость доводит до машинной точности.
            const auto [v, dv] = f(x);
            return x - v / dv;
        }
    }
    throw std::runtime_error("newtonRoot: iteration did not converge");
}

} // namespace

Quadrature1D gaussLobattoLegendre(int numPoints) {
    if (numPoints < 2) {
        throw std::invalid_argument("gaussLobattoLegendre: at least 2 points are required");
    }
    const int n = numPoints - 1;  // степень L_n, корни L_n' — внутренние узлы

    Quadrature1D quad;
    quad.type = PointsType::GaussLobattoLegendre;
    quad.points.assign(static_cast<std::size_t>(numPoints), 0);
    quad.weights.assign(static_cast<std::size_t>(numPoints), 0);

    // w_j = 2 / (n(n+1) L_n(x_j)^2), а на концах L_n(±1)^2 = 1.
    const Real endWeight = 2.0 / (n * (n + 1));
    quad.points.front()  = -1;
    quad.points.back()   = 1;
    quad.weights.front() = endWeight;
    quad.weights.back()  = endWeight;

    const auto q = [n](Real x) {
        const LobattoQ r = lobattoQ(n, x);
        return std::pair{r.q, r.dq};
    };

    // Ищем только левую половину, правую получаем отражением: так узлы
    // симметричны точно, а при чётном n центральный узел равен ровно нулю.
    for (int j = 1; j < numPoints / 2; ++j) {
        // Начальное приближение — узлы Чебышёва–Гаусса–Лобатто.
        const Real x0 = -std::cos(std::numbers::pi * j / n);
        const Real x  = newtonRoot(x0, q);
        const Real L  = legendre(n, x).value;
        const Real w  = endWeight / (L * L);

        quad.points[static_cast<std::size_t>(j)]      = x;
        quad.points[static_cast<std::size_t>(n - j)]  = -x;
        quad.weights[static_cast<std::size_t>(j)]     = w;
        quad.weights[static_cast<std::size_t>(n - j)] = w;
    }

    if (n % 2 == 0) {
        const Real L = legendre(n, 0).value;
        quad.points[static_cast<std::size_t>(n / 2)]  = 0;
        quad.weights[static_cast<std::size_t>(n / 2)] = endWeight / (L * L);
    }
    return quad;
}

Quadrature1D gaussLegendre(int numPoints) {
    if (numPoints < 1) {
        throw std::invalid_argument("gaussLegendre: at least 1 point is required");
    }
    const int m = numPoints;  // узлы — корни L_m

    Quadrature1D quad;
    quad.type = PointsType::GaussLegendre;
    quad.points.assign(static_cast<std::size_t>(m), 0);
    quad.weights.assign(static_cast<std::size_t>(m), 0);

    const auto L = [m](Real x) {
        const LegendreValue r = legendre(m, x);
        return std::pair{r.value, r.derivative};
    };
    // w_j = 2 / ((1 - x_j^2) L_m'(x_j)^2)
    const auto weightAt = [m](Real x) {
        const Real dL = legendre(m, x).derivative;
        return 2 / ((1 - x * x) * dL * dL);
    };

    for (int j = 0; j < m / 2; ++j) {
        // Асимптотическое приближение j-го корня L_m (по возрастанию).
        const Real x0 = -std::cos(std::numbers::pi * (4 * j + 3) / (4 * m + 2));
        const Real x  = newtonRoot(x0, L);
        const Real w  = weightAt(x);

        quad.points[static_cast<std::size_t>(j)]          = x;
        quad.points[static_cast<std::size_t>(m - 1 - j)]  = -x;
        quad.weights[static_cast<std::size_t>(j)]         = w;
        quad.weights[static_cast<std::size_t>(m - 1 - j)] = w;
    }

    if (m % 2 == 1) {
        quad.points[static_cast<std::size_t>(m / 2)]  = 0;
        quad.weights[static_cast<std::size_t>(m / 2)] = weightAt(0);
    }
    return quad;
}

Quadrature1D makeQuadrature(PointsType type, int numPoints) {
    switch (type) {
    case PointsType::GaussLobattoLegendre:
        return gaussLobattoLegendre(numPoints);
    case PointsType::GaussLegendre:
        return gaussLegendre(numPoints);
    }
    throw std::invalid_argument("makeQuadrature: unknown points type");
}

} // namespace sem
