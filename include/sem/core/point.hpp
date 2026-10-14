#pragma once

#include <array>

#include "sem/core/types.hpp"

namespace sem {

// Точка в Dim-мерном пространстве: p[0] = x, p[1] = y, p[2] = z.
template <int Dim>
using Point = std::array<Real, Dim>;

} // namespace sem
