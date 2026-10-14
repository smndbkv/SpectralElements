#include "sem/core/linear_solve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sem {

std::vector<Real> solveGauss(DenseMatrix A, std::span<const Real> b) {
    const int n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("solveGauss: matrix must be square");
    }
    if (b.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("solveGauss: right-hand side size does not match the matrix");
    }

    std::vector<Real> x(b.begin(), b.end());

    // Порог вырожденности относительно масштаба матрицы: главный элемент
    // такого размера неотличим от ошибок округления.
    Real scale = 0;
    for (const Real v : A.data()) {
        scale = std::max(scale, std::abs(v));
    }
    const Real tiny = n * std::numeric_limits<Real>::epsilon() * scale;

    // Прямой ход: приводим A к верхнетреугольному виду.
    for (int k = 0; k < n; ++k) {
        // Главный элемент — наибольший по модулю в столбце k среди строк k..n-1.
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A(i, k)) > std::abs(A(pivot, k))) {
                pivot = i;
            }
        }
        if (!(std::abs(A(pivot, k)) > tiny)) {
            throw std::runtime_error("solveGauss: matrix is singular or nearly singular");
        }
        if (pivot != k) {
            const auto rowK = A.row(k);
            const auto rowP = A.row(pivot);
            std::swap_ranges(rowK.begin(), rowK.end(), rowP.begin());
            std::swap(x[static_cast<std::size_t>(k)], x[static_cast<std::size_t>(pivot)]);
        }

        // Обнуляем столбец k под диагональю. Благодаря выбору главного
        // элемента |factor| <= 1, и ошибки округления не растут.
        const Real diag = A(k, k);
        for (int i = k + 1; i < n; ++i) {
            const Real factor = A(i, k) / diag;
            if (factor == 0) {
                continue;
            }
            for (int j = k + 1; j < n; ++j) {
                A(i, j) -= factor * A(k, j);
            }
            A(i, k) = 0;
            x[static_cast<std::size_t>(i)] -= factor * x[static_cast<std::size_t>(k)];
        }
    }

    // Обратный ход: решаем треугольную систему снизу вверх.
    for (int i = n - 1; i >= 0; --i) {
        Real sum = x[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= A(i, j) * x[static_cast<std::size_t>(j)];
        }
        x[static_cast<std::size_t>(i)] = sum / A(i, i);
    }
    return x;
}

} // namespace sem
