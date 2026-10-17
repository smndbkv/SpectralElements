#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sem/core/types.hpp"

namespace sem {

struct CgOptions {
    Real relativeTolerance = 1e-12;  // остановка, когда ||b - A x|| <= relativeTolerance * ||b||
    int  maxIterations     = 1000;
};

struct CgResult {
    int  iterations       = 0;
    Real residualNorm     = 0;  // ||b - A x|| в конце (рекуррентная невязка)
    Real relativeResidual = 0;  // residualNorm / ||b||
    bool converged        = false;
};

// Метод сопряжённых градиентов с предобусловливанием для A x = b.
//
// A задаётся только применением (matrix-free): applyA(in, out) записывает out = A in,
// где in — std::span<const Real>, out — std::span<Real>. Подходит, например,
//   [&](auto in, auto out) { sem::applyK<2>(basis, in, out); }.
// precondition(r, z) записывает z = P^{-1} r (P — предобусловливатель).
// A и P должны быть симметричными положительно определёнными.
//
// x на входе — начальное приближение (обычно нули), на выходе — решение.
// Каждая итерация: одно применение A, одно применение P^{-1} и O(n) операций.
//
// Бросает std::invalid_argument при разных размерах b и x и std::runtime_error,
// если обнаружено p^T A p <= 0 или r^T P^{-1} r <= 0, то есть A или P не положительно
// определены: тогда метод неприменим (нужен Гаусс, MINRES или GMRES).
template <class ApplyA, class Precondition>
CgResult conjugateGradient(ApplyA&& applyA, Precondition&& precondition,
                           std::span<const Real> b, std::span<Real> x,
                           const CgOptions& options = {}) {
    const std::size_t n = b.size();
    if (x.size() != n) {
        throw std::invalid_argument("conjugateGradient: b and x must have the same size");
    }
    const auto dot = [](std::span<const Real> u, std::span<const Real> v) {
        Real s = 0;
        for (std::size_t k = 0; k < u.size(); ++k) {
            s += u[k] * v[k];
        }
        return s;
    };

    std::vector<Real> r(n), z(n), p(n), Ap(n);
    CgResult result;

    // Начальная невязка r = b - A x.
    applyA(std::span<const Real>(x.data(), n), std::span<Real>(Ap));
    for (std::size_t k = 0; k < n; ++k) {
        r[k] = b[k] - Ap[k];
    }
    const Real normB = std::sqrt(dot(b, b));
    if (normB == 0) {  // b = 0: решение x = 0
        std::fill(x.begin(), x.end(), Real{0});
        result.converged = true;
        return result;
    }
    const Real target = options.relativeTolerance * normB;
    Real normR = std::sqrt(dot(r, r));

    if (normR > target) {
        precondition(std::span<const Real>(r), std::span<Real>(z));
        p = z;
        Real rz = dot(r, z);

        for (int it = 1; it <= options.maxIterations; ++it) {
            if (!(rz > 0)) {
                throw std::runtime_error(
                    "conjugateGradient: preconditioner is not positive definite (r^T P^-1 r <= 0)");
            }
            applyA(std::span<const Real>(p), std::span<Real>(Ap));
            const Real pAp = dot(p, Ap);
            if (!(pAp > 0)) {
                throw std::runtime_error(
                    "conjugateGradient: operator is not positive definite (p^T A p <= 0)");
            }

            const Real alpha = rz / pAp;  // шаг вдоль направления p
            for (std::size_t k = 0; k < n; ++k) {
                x[k] += alpha * p[k];
                r[k] -= alpha * Ap[k];
            }
            normR = std::sqrt(dot(r, r));
            result.iterations = it;
            if (normR <= target) {
                break;
            }

            precondition(std::span<const Real>(r), std::span<Real>(z));
            const Real rzNew = dot(r, z);
            const Real beta = rzNew / rz;  // новое направление A-сопряжено со всеми прежними
            rz = rzNew;
            for (std::size_t k = 0; k < n; ++k) {
                p[k] = z[k] + beta * p[k];
            }
        }
    }

    result.residualNorm     = normR;
    result.relativeResidual = normR / normB;
    result.converged        = normR <= target;
    return result;
}

// Без предобусловливания (P = I).
template <class ApplyA>
CgResult conjugateGradient(ApplyA&& applyA, std::span<const Real> b, std::span<Real> x,
                           const CgOptions& options = {}) {
    const auto identity = [](std::span<const Real> r, std::span<Real> z) {
        std::copy(r.begin(), r.end(), z.begin());
    };
    return conjugateGradient(std::forward<ApplyA>(applyA), identity, b, x, options);
}

// Предобусловливатель Якоби: P = diag(A), z = r / diag(A).
// Диагональ должна быть положительной (для SPD-матрицы это так всегда).
class JacobiPreconditioner {
public:
    explicit JacobiPreconditioner(std::span<const Real> diagonal) {
        inverse_.reserve(diagonal.size());
        for (const Real d : diagonal) {
            if (!(d > 0)) {
                throw std::invalid_argument("JacobiPreconditioner: diagonal must be positive");
            }
            inverse_.push_back(1 / d);
        }
    }

    void operator()(std::span<const Real> r, std::span<Real> z) const {
        for (std::size_t k = 0; k < r.size(); ++k) {
            z[k] = inverse_[k] * r[k];
        }
    }

private:
    std::vector<Real> inverse_;
};

} // namespace sem
