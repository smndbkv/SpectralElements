#pragma once

#include <span>
#include <vector>

#include "sem/core/dense_matrix.hpp"
#include "sem/core/types.hpp"

namespace sem {

// Решает A x = b методом Гаусса с выбором главного элемента по столбцу
// (частичный выбор). Подходит для любых невырожденных квадратных матриц,
// в том числе симметричных знаконеопределённых. Стоимость O(n^3), память O(n^2).
//
// A передаётся по значению: внутри исключение идёт на копии, исходная матрица
// не меняется.
//
// Исключения:
//   std::invalid_argument — A не квадратная или размер b не совпадает;
//   std::runtime_error    — матрица вырождена или близка к вырожденной
//                            (главный элемент <= n * eps * max|A_ij|).
std::vector<Real> solveGauss(DenseMatrix A, std::span<const Real> b);

} // namespace sem
