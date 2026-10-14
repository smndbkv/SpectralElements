#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "sem/core/dense_matrix.hpp"
#include "sem/core/types.hpp"

namespace sem {

// Тензорные (sum factorization) операции над данными одного элемента.
//
// Раскладка данных: Dim-мерный массив размера shape[0] x shape[1] x ...,
// индекс по направлению 0 (x) меняется быстрее всего:
//   flat = i_0 + shape[0] * (i_1 + shape[1] * i_2).
// В 2D это матрица U по строкам: строка — индекс j по y, столбец — индекс i по x.

// Применяет матрицу A (m x n) вдоль направления `direction`:
//   out[..., r, ...] = sum_k A(r, k) * in[..., k, ...],
// где r, k — индекс по направлению direction, остальные индексы не меняются.
// shape — размеры ВХОДА; должно быть shape[direction] == A.cols().
// Размеры выхода: те же, кроме direction, где вместо n будет m = A.rows().
// В 2D: direction 0 — это U A^T, direction 1 — это A U.
//
// Математически: out = (I_outer ⊗ A ⊗ I_inner) in, где
//   inner = shape[0] * ... * shape[direction-1],  outer = shape[direction+1] * ... ,
// то есть покомпонентно
//   out[s + inner*(r + m*o)] = sum_k A(r, k) * in[s + inner*(k + n*o)].
// Пара (s, o) нумерует линии вдоль direction, k и r — положение внутри линии.
//
// in и out не должны перекрываться.
// Бросает std::invalid_argument, если direction вне [0, Dim), A.cols() != shape[direction]
// или размеры in/out не соответствуют shape.
template <int Dim>
void applyAlong(const DenseMatrix& A, int direction, const std::array<int, Dim>& shape,
                std::span<const Real> in, std::span<Real> out) {
    if (direction < 0 || direction >= Dim) {
        throw std::invalid_argument("applyAlong: direction out of range");
    }
    const auto d = static_cast<std::size_t>(direction);
    if (A.cols() != shape[d]) {
        throw std::invalid_argument("applyAlong: A.cols() must equal shape[direction]");
    }

    std::size_t inner = 1;  // шаг между соседними элементами линии
    for (std::size_t e = 0; e < d; ++e) {
        inner *= static_cast<std::size_t>(shape[e]);
    }
    std::size_t outer = 1;  // число блоков по направлениям после direction
    for (std::size_t e = d + 1; e < static_cast<std::size_t>(Dim); ++e) {
        outer *= static_cast<std::size_t>(shape[e]);
    }
    const auto n = static_cast<std::size_t>(A.cols());
    const auto m = static_cast<std::size_t>(A.rows());

    if (in.size() != inner * n * outer) {
        throw std::invalid_argument("applyAlong: input size does not match shape");
    }
    if (out.size() != inner * m * outer) {
        throw std::invalid_argument("applyAlong: output size does not match shape");
    }

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t r = 0; r < m; ++r) {
            // Все inner линий блока o обрабатываются одновременно: самый внутренний
            // цикл идёт по s, то есть по соседним ячейкам памяти.
            const std::span<Real> outRow = out.subspan(inner * (r + m * o), inner);
            std::fill(outRow.begin(), outRow.end(), Real{0});

            const auto Ar = A.row(static_cast<int>(r));
            for (std::size_t k = 0; k < n; ++k) {
                const Real a = Ar[k];
                const std::span<const Real> inRow = in.subspan(inner * (k + n * o), inner);
                for (std::size_t s = 0; s < inner; ++s) {
                    outRow[s] += a * inRow[s];
                }
            }
        }
    }
}

// out = (A[Dim-1] ⊗ ... ⊗ A[1] ⊗ A[0]) in — матрица A[d] применяется вдоль направления d.
// Вход: prod_d A[d].cols() значений, выход: prod_d A[d].rows() значений.
// В 2D: out = A[1] * U * A[0]^T.
// Матрицы могут быть прямоугольными (например, интерполяция на другую сетку).
//
// Реализация: A[0] ⊗ ... раскладывается в произведение множителей I ⊗ ... ⊗ A[d] ⊗ ... ⊗ I,
// каждый из которых — applyAlong вдоль d. После шага d форма меняется: shape[d] = A[d].rows().
//
// in и out не должны перекрываться.
// Бросает std::invalid_argument при несоответствии размеров.
template <int Dim>
void tensorApply(const std::array<const DenseMatrix*, Dim>& A,
                 std::span<const Real> in, std::span<Real> out) {
    const auto product = [](const std::array<int, Dim>& s) {
        std::size_t p = 1;
        for (const int v : s) {
            p *= static_cast<std::size_t>(v);
        }
        return p;
    };

    std::array<int, Dim> shape{};
    std::array<int, Dim> finalShape{};
    for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
        if (A[d] == nullptr) {
            throw std::invalid_argument("tensorApply: matrix pointer is null");
        }
        shape[d]      = A[d]->cols();
        finalShape[d] = A[d]->rows();
    }
    if (in.size() != product(shape)) {
        throw std::invalid_argument("tensorApply: input size does not match the matrices");
    }
    if (out.size() != product(finalShape)) {
        throw std::invalid_argument("tensorApply: output size does not match the matrices");
    }

    // Промежуточные результаты — в двух буферах по очереди; последний шаг пишет прямо в out.
    std::array<std::vector<Real>, 2> buffers;
    std::span<const Real> source = in;
    for (int d = 0; d < Dim; ++d) {
        const auto ud = static_cast<std::size_t>(d);
        std::array<int, Dim> nextShape = shape;
        nextShape[ud] = A[ud]->rows();

        if (d == Dim - 1) {
            applyAlong<Dim>(*A[ud], d, shape, source, out);
        } else {
            std::vector<Real>& target = buffers[ud % 2];
            target.assign(product(nextShape), Real{0});
            applyAlong<Dim>(*A[ud], d, shape, source, target);
            source = target;
        }
        shape = nextShape;
    }
}

} // namespace sem
