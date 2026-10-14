#include "sem/basis/lagrange_basis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace sem {

LagrangeBasis1D::LagrangeBasis1D(std::vector<Real> nodes)
    : nodes_(std::move(nodes)) {
    if (nodes_.empty()) {
        throw std::invalid_argument("LagrangeBasis1D: at least one node is required");
    }

    // Разности масштабируются к интервалу длины 4: тогда произведения в lambda_j
    // не переполняются и не теряют порядок даже при больших N.
    const auto [minIt, maxIt] = std::minmax_element(nodes_.begin(), nodes_.end());
    const Real length = *maxIt - *minIt;
    const Real scale  = length > 0 ? 4 / length : 1;

    const std::size_t n = nodes_.size();
    baryWeights_.assign(n, 1);
    for (std::size_t j = 0; j < n; ++j) {
        Real product = 1;
        for (std::size_t k = 0; k < n; ++k) {
            if (k == j) {
                continue;
            }
            const Real diff = scale * (nodes_[j] - nodes_[k]);
            if (diff == 0) {
                throw std::invalid_argument("LagrangeBasis1D: nodes must be distinct");
            }
            product *= diff;
        }
        baryWeights_[j] = 1 / product;
    }

    // Общий множитель сокращается в барицентрических формулах — нормируем на максимум.
    Real maxAbs = 0;
    for (const Real w : baryWeights_) {
        maxAbs = std::max(maxAbs, std::abs(w));
    }
    for (Real& w : baryWeights_) {
        w /= maxAbs;
    }
}

void LagrangeBasis1D::evaluateBasis(Real x, std::span<Real> values) const {
    if (values.size() != nodes_.size()) {
        throw std::invalid_argument("LagrangeBasis1D::evaluateBasis: wrong output size");
    }

    Real sum = 0;
    for (std::size_t j = 0; j < nodes_.size(); ++j) {
        const Real diff = x - nodes_[j];
        if (diff == 0) {
            // Точка совпала с узлом: формула вырождается, ответ — дельта-символ.
            std::fill(values.begin(), values.end(), Real{0});
            values[j] = 1;
            return;
        }
        values[j] = baryWeights_[j] / diff;
        sum += values[j];
    }
    for (Real& v : values) {
        v /= sum;
    }
}

Real LagrangeBasis1D::evaluate(std::span<const Real> nodalValues, Real x) const {
    if (nodalValues.size() != nodes_.size()) {
        throw std::invalid_argument("LagrangeBasis1D::evaluate: wrong input size");
    }

    Real numerator = 0, denominator = 0;
    for (std::size_t j = 0; j < nodes_.size(); ++j) {
        const Real diff = x - nodes_[j];
        if (diff == 0) {
            return nodalValues[j];
        }
        const Real t = baryWeights_[j] / diff;
        numerator += t * nodalValues[j];
        denominator += t;
    }
    return numerator / denominator;
}

DenseMatrix LagrangeBasis1D::interpolationMatrix(std::span<const Real> targets) const {
    DenseMatrix I(static_cast<int>(targets.size()), numNodes());
    for (int i = 0; i < I.rows(); ++i) {
        evaluateBasis(targets[static_cast<std::size_t>(i)], I.row(i));
    }
    return I;
}

DenseMatrix LagrangeBasis1D::derivativeMatrix() const {
    const int n = numNodes();
    DenseMatrix D(n, n);
    for (int i = 0; i < n; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        Real rowSum = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const auto uj = static_cast<std::size_t>(j);
            D(i, j) = (baryWeights_[uj] / baryWeights_[ui]) / (nodes_[ui] - nodes_[uj]);
            rowSum += D(i, j);
        }
        // Диагональ через сумму строки: производная константы равна нулю
        // с машинной точностью, что заметно точнее прямой формулы при больших N.
        D(i, i) = -rowSum;
    }
    return D;
}

} // namespace sem
