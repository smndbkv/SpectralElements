#pragma once

#include "sem/core/types.hpp"

namespace sem {

// Значение полинома Лежандра L_n(x) и его производной L_n'(x).
struct LegendreValue {
    Real value;
    Real derivative;
};

// L_n(x) и L_n'(x) по трёхчленной рекуррентной формуле:
//   k L_k = (2k-1) x L_{k-1} - (k-1) L_{k-2},   L_k' = L_{k-2}' + (2k-1) L_{k-1}.
// Требует n >= 0.
LegendreValue legendre(int n, Real x);

// Вспомогательная функция для узлов Гаусса–Лобатто–Лежандра:
//   q(x) = L_{n+1}(x) - L_{n-1}(x) = (2n+1) / (n(n+1)) * (x^2 - 1) L_n'(x),
//   q'(x) = (2n+1) L_n(x).
// Внутренние узлы GLL — корни q на (-1, 1), при этом q' не требует L_n''.
// Требует n >= 1.
struct LobattoQ {
    Real q;
    Real dq;
    Real legendre;  // L_n(x)
};

LobattoQ lobattoQ(int n, Real x);

} // namespace sem
