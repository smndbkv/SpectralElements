#include "sem/analysis/error_norms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "sem/core/dense_matrix.hpp"

namespace sem::analysis {

namespace {

void checkBreakpoints(std::span<const Real> breakpoints) {
    if (breakpoints.size() < 2) {
        throw std::invalid_argument("errorNorms: at least 2 breakpoints are required");
    }
    for (std::size_t k = 1; k < breakpoints.size(); ++k) {
        if (!(breakpoints[k] > breakpoints[k - 1])) {
            throw std::invalid_argument("errorNorms: breakpoints must be strictly increasing");
        }
    }
}

// Сдвигает концы отрезка [a, b] на одно машинное число внутрь. Производные на концах
// элемента вычисляются как односторонние пределы изнутри: если у точной функции
// в узле стыка излом, иначе бралась бы производная соседнего элемента.
Real insideInterval(Real x, Real a, Real b) {
    return std::clamp(x, std::nextafter(a, b), std::nextafter(b, a));
}

} // namespace

void detail::checkOptions(const NormOptions& options) {
    if (options.quadraturePoints < 1) {
        throw std::invalid_argument("NormOptions: quadraturePoints must be at least 1");
    }
    if (options.samplesPerInterval < 2) {
        throw std::invalid_argument("NormOptions: samplesPerInterval must be at least 2");
    }
    if (options.subdivisions < 1) {
        throw std::invalid_argument("NormOptions: subdivisions must be at least 1");
    }
}

NormAccumulator::NormAccumulator(const NormOptions& options)
    : options_(options) {
    detail::checkOptions(options);
    gauss_ = gaussLegendre(options.quadraturePoints);
    samplesE_.resize(static_cast<std::size_t>(options.samplesPerInterval));
    samplesDE_.resize(static_cast<std::size_t>(options.samplesPerInterval));
}

Real NormAccumulator::refinedMaxAbs(const std::vector<Real>& f) {
    // Парабола через три отсчёта вокруг лучшего — одномерный случай общей квадратичной модели.
    return detail::refinedMaxAbsGrid<1>(f, static_cast<int>(f.size()));
}

ErrorNorms NormAccumulator::result() const {
    return {maxAbs_, maxAbs_ + maxAbsDeriv_, integralAbs_, std::sqrt(integralSquare_)};
}

ErrorNorms errorNorms(const Function1D& u, const Function1D& du,
                      const Function1D& v, const Function1D& dv,
                      std::span<const Real> breakpoints, const NormOptions& options) {
    checkBreakpoints(breakpoints);

    NormAccumulator acc(options);
    const auto e = [&](Real x) { return u(x) - v(x); };

    const int parts = options.subdivisions;
    for (std::size_t k = 0; k + 1 < breakpoints.size(); ++k) {
        const Real a = breakpoints[k];
        const Real b = breakpoints[k + 1];
        const auto de = [&](Real x) {
            const Real xs = insideInterval(x, a, b);
            return du(xs) - dv(xs);
        };
        const Real h = (b - a) / parts;
        for (int p = 0; p < parts; ++p) {
            acc.addInterval(a + p * h, a + (p + 1) * h, e, de);
        }
    }
    return acc.result();
}

ErrorNorms errorNorms(const SpectralBasis1D& basis, std::span<const Real> breakpoints,
                      std::span<const Real> nodalValues,
                      const Function1D& u, const Function1D& du,
                      const NormOptions& options) {
    checkBreakpoints(breakpoints);

    const auto n           = static_cast<std::size_t>(basis.numPoints());
    const std::size_t numElements = breakpoints.size() - 1;
    if (nodalValues.size() != numElements * n) {
        throw std::invalid_argument(
            "errorNorms: nodalValues must hold (N+1) values for every element");
    }

    const auto xi = basis.points();
    const LagrangeBasis1D& lagrange = basis.lagrange();
    NormAccumulator acc(options);
    std::vector<Real> dLocal(n);

    for (std::size_t k = 0; k < numElements; ++k) {
        const Real a = breakpoints[k];
        const Real b = breakpoints[k + 1];
        const Real J = (b - a) / 2;  // dx/dxi
        const auto local = nodalValues.subspan(k * n, n);

        // Производная приближения — полином степени N-1, поэтому её узловые
        // значения D u / J и интерполяция по ним дают её точно в любой точке.
        multiply(basis.D(), local, dLocal);
        for (Real& d : dLocal) {
            d /= J;
        }

        const auto toReference = [a, J](Real x) { return (x - a) / J - 1; };
        const auto e  = [&](Real x) { return u(x) - lagrange.evaluate(local, toReference(x)); };
        const auto de = [&](Real x) {
            return du(insideInterval(x, a, b)) - lagrange.evaluate(dLocal, toReference(x));
        };

        // Подотрезки между узлами GLL: ошибка интерполяции обращается в ноль в узлах,
        // так что |e| не имеет изломов внутри подотрезков и квадратура остаётся точной.
        for (std::size_t i = 0; i + 1 < n; ++i) {
            acc.addInterval(a + (xi[i] + 1) * J, a + (xi[i + 1] + 1) * J, e, de);
        }
    }
    return acc.result();
}

} // namespace sem::analysis
