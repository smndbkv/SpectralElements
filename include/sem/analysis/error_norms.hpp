#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sem/basis/quadrature.hpp"
#include "sem/basis/spectral_basis.hpp"
#include "sem/basis/tensor_apply.hpp"
#include "sem/basis/tensor_basis.hpp"
#include "sem/core/dense_matrix.hpp"
#include "sem/core/linear_solve.hpp"
#include "sem/core/point.hpp"
#include "sem/core/types.hpp"

namespace sem::analysis {

// Нормы ошибки e = u - v на области Omega:
//   C   = max |e|
//   C1  = max |e| + sum_d max |de/dx_d|   (в 1D: max |e| + max |e'|; производные
//                                           берутся внутри элементов, на стыках они разрывны)
//   L1  = int |e| dx
//   L2  = sqrt( int e^2 dx )
struct ErrorNorms {
    Real c  = 0;
    Real c1 = 0;
    Real l1 = 0;
    Real l2 = 0;
};

struct NormOptions {
    // Точек Гаусса на каждом подотрезке (интегралы для L1 и L2); в Dim измерениях
    // в каждой ячейке берётся тензорная сетка quadraturePoints^Dim.
    int quadraturePoints = 8;
    // Равномерных точек на каждом подотрезке, включая концы (максимумы для C и C1);
    // в Dim измерениях — сетка samplesPerInterval^Dim в каждой ячейке.
    // Максимум ищется перебором и уточняется квадратичной моделью по соседним точкам
    // (в 1D — парабола через три точки), поэтому C и C1 — оценки снизу
    // с относительной погрешностью около 1e-5.
    int samplesPerInterval = 16;
    // Только для errorNorms по функциям: на сколько равных частей делить каждый
    // отрезок разбиения. Для SEM подотрезки задаются узлами GLL.
    int subdivisions = 32;
};

// Накопитель норм по подотрезкам, на каждом из которых e и e' гладкие.
// Нужен для собственных типов данных; обычно достаточно функций errorNorms ниже.
class NormAccumulator {
public:
    explicit NormAccumulator(const NormOptions& options = {});

    // e(x) и de(x) должны быть определены на [a, b] (пределы изнутри отрезка).
    template <class E, class DE>
    void addInterval(Real a, Real b, E&& e, DE&& de);

    ErrorNorms result() const;

private:
    // Максимум |f| по равномерным отсчётам с уточнением параболой у лучшего отсчёта.
    static Real refinedMaxAbs(const std::vector<Real>& samples);

    NormOptions  options_;
    Quadrature1D gauss_;
    std::vector<Real> samplesE_;
    std::vector<Real> samplesDE_;
    Real maxAbs_        = 0;
    Real maxAbsDeriv_   = 0;
    Real integralAbs_   = 0;
    Real integralSquare_ = 0;
};

using Function1D = std::function<Real(Real)>;

// Нормы u - v на [breakpoints.front(), breakpoints.back()]. Функции должны быть
// гладкими внутри каждого отрезка разбиения; на стыках производные могут разрываться.
ErrorNorms errorNorms(const Function1D& u, const Function1D& du,
                      const Function1D& v, const Function1D& dv,
                      std::span<const Real> breakpoints, const NormOptions& options = {});

// Нормы ошибки приближения SEM в 1D. Элемент k — это [breakpoints[k], breakpoints[k+1]],
// на нём решение задано значениями в узлах GLL: nodalValues[k * (N+1) + i].
// Хранение поэлементное: общий узел соседних элементов записан дважды.
// Для одного элемента [-1, 1]: breakpoints = {-1, 1}, nodalValues — N+1 значений.
ErrorNorms errorNorms(const SpectralBasis1D& basis, std::span<const Real> breakpoints,
                      std::span<const Real> nodalValues,
                      const Function1D& u, const Function1D& du,
                      const NormOptions& options = {});

template <int Dim>
using FunctionND = std::function<Real(const Point<Dim>&)>;

template <int Dim>
using GradientND = std::function<Point<Dim>(const Point<Dim>&)>;

// Нормы ошибки приближения на эталонном элементе [-1, 1]^Dim (одна ячейка сетки, J = Id).
// nodalValues — узловые значения в раскладке TensorBasis (x меняется быстрее всего),
// u — точная функция, gradU — её градиент (gradU(x)[d] = du/dx_d).
//
// Элемент делится на ячейки — произведения подотрезков между соседними узлами GLL.
// В каждой ячейке: интегралы — тензорной квадратурой Гаусса, максимумы — перебором
// по равномерной сетке с квадратичным уточнением. Значения приближения во всех точках
// ячейки вычисляются сразу через tensorApply с матрицами интерполяции.
//
// Бросает std::invalid_argument при неверном размере nodalValues или неверных options.
template <int Dim>
ErrorNorms errorNorms(const TensorBasis<Dim>& basis, std::span<const Real> nodalValues,
                      const std::type_identity_t<FunctionND<Dim>>& u,
                      const std::type_identity_t<GradientND<Dim>>& gradU,
                      const NormOptions& options = {});

namespace detail {

void checkOptions(const NormOptions& options);

constexpr int ipow(int base, int exponent) {
    int result = 1;
    for (int k = 0; k < exponent; ++k) {
        result *= base;
    }
    return result;
}

// Разряды числа value в системе с основанием base (младший — направление 0).
template <int Dim>
std::array<std::size_t, Dim> digits(int value, int base) {
    std::array<std::size_t, Dim> result{};
    for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
        result[d] = static_cast<std::size_t>(value % base);
        value /= base;
    }
    return result;
}

// Максимум |f| по равномерной сетке s^Dim отсчётов (раскладка: x быстрее всего).
// Около лучшего отсчёта строится квадратичная модель по соседям
// f(t) ≈ f0 + g·t + t^T H t / 2 (t — смещение в шагах сетки) и берётся её вершина
// t* = -H^{-1} g, если она лежит в пределах одного шага. В 1D это парабола через три точки.
// Уточнение идёт только по «активным» направлениям, где у отсчёта есть соседи с обеих
// сторон; по остальным координата фиксирована (максимум на грани ячейки).
template <int Dim>
Real refinedMaxAbsGrid(std::span<const Real> f, int s) {
    std::size_t best = 0;
    for (std::size_t p = 1; p < f.size(); ++p) {
        if (std::abs(f[p]) > std::abs(f[best])) {
            best = p;
        }
    }
    const Real f0 = f[best];
    const Real result = std::abs(f0);

    const auto index = digits<Dim>(static_cast<int>(best), s);
    std::vector<std::size_t> stride;  // шаги в памяти для активных направлений
    for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
        if (index[d] > 0 && index[d] + 1 < static_cast<std::size_t>(s)) {
            stride.push_back(static_cast<std::size_t>(ipow(s, static_cast<int>(d))));
        }
    }
    const std::size_t k = stride.size();
    if (k == 0) {
        return result;
    }

    DenseMatrix H(static_cast<int>(k), static_cast<int>(k));
    std::vector<Real> minusG(k);
    for (std::size_t a = 0; a < k; ++a) {
        const Real fp = f[best + stride[a]];
        const Real fm = f[best - stride[a]];
        minusG[a] = -(fp - fm) / 2;
        H(static_cast<int>(a), static_cast<int>(a)) = fp - 2 * f0 + fm;
        for (std::size_t b = a + 1; b < k; ++b) {
            const Real mixed = (f[best + stride[a] + stride[b]] - f[best + stride[a] - stride[b]]
                              - f[best - stride[a] + stride[b]] + f[best - stride[a] - stride[b]]) / 4;
            H(static_cast<int>(a), static_cast<int>(b)) = mixed;
            H(static_cast<int>(b), static_cast<int>(a)) = mixed;
        }
    }

    std::vector<Real> t;
    try {
        t = solveGauss(H, minusG);
    } catch (const std::runtime_error&) {
        return result;  // вырожденная модель: остаёмся с лучшим отсчётом
    }
    Real vertex = f0;
    for (std::size_t a = 0; a < k; ++a) {
        if (std::abs(t[a]) > 1) {
            return result;
        }
        vertex -= minusG[a] * t[a] / 2;  // f0 + g·t*/2
    }
    return std::max(result, std::abs(vertex));
}

} // namespace detail

template <int Dim>
ErrorNorms errorNorms(const TensorBasis<Dim>& basis, std::span<const Real> nodalValues,
                      const std::type_identity_t<FunctionND<Dim>>& u,
                      const std::type_identity_t<GradientND<Dim>>& gradU,
                      const NormOptions& options) {
    detail::checkOptions(options);
    if (nodalValues.size() != static_cast<std::size_t>(basis.numPoints())) {
        throw std::invalid_argument("errorNorms: nodalValues must hold numPoints() values");
    }
    constexpr auto dims = static_cast<std::size_t>(Dim);

    const SpectralBasis1D& b1 = basis.basis1D();
    const int N = b1.degree();
    const int q = options.quadraturePoints;
    const int s = options.samplesPerInterval;
    const auto xi = b1.points();
    const Quadrature1D gauss = gaussLegendre(q);

    // Одномерные таблицы для каждого подотрезка [xi_c, xi_{c+1}] между узлами GLL:
    // точки и веса Гаусса, равномерные точки и матрицы, переводящие узловые значения
    // в значения (I) и производные (I D) в этих точках.
    struct Interval {
        std::vector<Real> gaussX;
        std::vector<Real> gaussW;
        std::vector<Real> sampleX;
        DenseMatrix toGauss;
        DenseMatrix toSamples;
        DenseMatrix derivToSamples;
    };
    std::vector<Interval> intervals(static_cast<std::size_t>(N));
    for (std::size_t c = 0; c < intervals.size(); ++c) {
        Interval& iv = intervals[c];
        const Real a = xi[c];
        const Real b = xi[c + 1];
        for (std::size_t k = 0; k < gauss.points.size(); ++k) {
            iv.gaussX.push_back((a + b) / 2 + (b - a) / 2 * gauss.points[k]);
            iv.gaussW.push_back((b - a) / 2 * gauss.weights[k]);
        }
        for (int t = 0; t < s; ++t) {
            iv.sampleX.push_back(a + (b - a) * static_cast<Real>(t) / static_cast<Real>(s - 1));
        }
        iv.toGauss        = b1.lagrange().interpolationMatrix(iv.gaussX);
        iv.toSamples      = b1.lagrange().interpolationMatrix(iv.sampleX);
        iv.derivToSamples = iv.toSamples * b1.D();
    }

    const int numCells   = detail::ipow(N, Dim);
    const int numGauss   = detail::ipow(q, Dim);
    const int numSamples = detail::ipow(s, Dim);
    std::vector<Real> approx(static_cast<std::size_t>(numGauss));
    std::vector<Real> approxSamples(static_cast<std::size_t>(numSamples));
    std::vector<Real> error(static_cast<std::size_t>(numSamples));
    std::array<std::vector<Real>, Dim> exactGrad;
    for (auto& g : exactGrad) {
        g.resize(static_cast<std::size_t>(numSamples));
    }
    std::array<const DenseMatrix*, Dim> matrices{};

    Real maxAbs = 0, integralAbs = 0, integralSquare = 0;
    std::array<Real, Dim> maxAbsDeriv{};

    for (int cell = 0; cell < numCells; ++cell) {
        const auto c = detail::digits<Dim>(cell, N);  // номер подотрезка по каждому направлению

        // Интегралы: приближение во всех точках Гаусса ячейки одним tensorApply.
        for (std::size_t d = 0; d < dims; ++d) {
            matrices[d] = &intervals[c[d]].toGauss;
        }
        tensorApply<Dim>(matrices, nodalValues, approx);
        for (int p = 0; p < numGauss; ++p) {
            const auto k = detail::digits<Dim>(p, q);
            Point<Dim> x{};
            Real w = 1;
            for (std::size_t d = 0; d < dims; ++d) {
                x[d] = intervals[c[d]].gaussX[k[d]];
                w *= intervals[c[d]].gaussW[k[d]];
            }
            const Real e = u(x) - approx[static_cast<std::size_t>(p)];
            integralAbs += w * std::abs(e);
            integralSquare += w * e * e;
        }

        // Максимумы: ошибка и производные ошибки на равномерной сетке ячейки.
        for (std::size_t d = 0; d < dims; ++d) {
            matrices[d] = &intervals[c[d]].toSamples;
        }
        tensorApply<Dim>(matrices, nodalValues, approxSamples);
        for (int p = 0; p < numSamples; ++p) {
            const auto k = detail::digits<Dim>(p, s);
            Point<Dim> x{};
            for (std::size_t d = 0; d < dims; ++d) {
                x[d] = intervals[c[d]].sampleX[k[d]];
            }
            const auto up = static_cast<std::size_t>(p);
            error[up] = u(x) - approxSamples[up];
            const Point<Dim> g = gradU(x);
            for (std::size_t d = 0; d < dims; ++d) {
                exactGrad[d][up] = g[d];
            }
        }
        maxAbs = std::max(maxAbs, detail::refinedMaxAbsGrid<Dim>(error, s));

        for (std::size_t d = 0; d < dims; ++d) {
            // Производная по x_d: I D вдоль d, просто I вдоль остальных направлений.
            for (std::size_t e = 0; e < dims; ++e) {
                matrices[e] = (e == d) ? &intervals[c[e]].derivToSamples : &intervals[c[e]].toSamples;
            }
            tensorApply<Dim>(matrices, nodalValues, approxSamples);
            for (std::size_t p = 0; p < error.size(); ++p) {
                error[p] = exactGrad[d][p] - approxSamples[p];
            }
            maxAbsDeriv[d] = std::max(maxAbsDeriv[d], detail::refinedMaxAbsGrid<Dim>(error, s));
        }
    }

    ErrorNorms result;
    result.c  = maxAbs;
    result.c1 = maxAbs;
    for (const Real m : maxAbsDeriv) {
        result.c1 += m;
    }
    result.l1 = integralAbs;
    result.l2 = std::sqrt(integralSquare);
    return result;
}

template <class E, class DE>
void NormAccumulator::addInterval(Real a, Real b, E&& e, DE&& de) {
    const Real mid  = (a + b) / 2;
    const Real half = (b - a) / 2;
    for (int k = 0; k < gauss_.size(); ++k) {
        const auto uk = static_cast<std::size_t>(k);
        const Real x  = mid + half * gauss_.points[uk];
        const Real w  = half * gauss_.weights[uk];
        const Real ex = e(x);
        integralAbs_ += w * std::abs(ex);
        integralSquare_ += w * ex * ex;
    }

    const std::size_t n = samplesE_.size();
    for (std::size_t s = 0; s < n; ++s) {
        const Real x  = a + (b - a) * static_cast<Real>(s) / static_cast<Real>(n - 1);
        samplesE_[s]  = e(x);
        samplesDE_[s] = de(x);
    }
    maxAbs_      = std::max(maxAbs_, refinedMaxAbs(samplesE_));
    maxAbsDeriv_ = std::max(maxAbsDeriv_, refinedMaxAbs(samplesDE_));
}

} // namespace sem::analysis
