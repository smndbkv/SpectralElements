#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sem/basis/spectral_basis.hpp"
#include "sem/basis/tensor_apply.hpp"
#include "sem/core/dense_matrix.hpp"
#include "sem/core/point.hpp"
#include "sem/core/types.hpp"

namespace sem {
    // Эталонный элемент [-1, 1]^Dim с тензорным базисом
    //   phi_{i_0 ... i_{Dim-1}}(x) = ell_{i_0}(x_0) * ... * ell_{i_{Dim-1}}(x_{Dim-1})
    // на узлах GLL по каждому направлению (одна и та же степень N).
    //
    // Узловые значения хранятся в раскладке из tensor_apply.hpp:
    //   flat = i_0 + n * (i_1 + n * i_2),   n = N + 1.
    template<int Dim>
    class TensorBasis {
        static_assert(Dim >= 1 && Dim <= 4, "TensorBasis supports Dim = 1, 2, 3, 4"); // What? 4? x,y,z...and t! Ok...

    public:
        explicit TensorBasis(int degree) : basis_(degree) {
        }

        const SpectralBasis1D &basis1D() const { return basis_; }

        int degree() const { return basis_.degree(); }
        int numPointsPerDirection() const { return basis_.numPoints(); } // n = N + 1

        int numPoints() const {
            // n^Dim
            int total = 1;
            for (int d = 0; d < Dim; ++d) {
                total *= numPointsPerDirection();
            }
            return total;
        }

        // (i_0, ..., i_{Dim-1}) -> flat и обратно.
        // flat — запись мультииндекса в системе счисления с основанием n,
        // младший разряд — направление 0 (x).
        int flatIndex(const std::array<int, Dim> &index) const {
            const int n = numPointsPerDirection();
            int flat = 0;
            for (int d = Dim - 1; d >= 0; --d) {
                // схема Горнера: (i_2 * n + i_1) * n + i_0
                const int i = index[static_cast<std::size_t>(d)];
                assert(i >= 0 && i < n);
                flat = flat * n + i;
            }
            return flat;
        }

        std::array<int, Dim> multiIndex(int flat) const {
            assert(flat >= 0 && flat < numPoints());
            const int n = numPointsPerDirection();
            std::array<int, Dim> index{};
            for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
                index[d] = flat % n; // очередной разряд
                flat /= n;
            }
            return index;
        }

        // Координаты узла flat на [-1, 1]^Dim: (xi_{i_0}, xi_{i_1}, ...).
        Point<Dim> point(int flat) const {
            const std::array<int, Dim> index = multiIndex(flat);
            const auto xi = basis_.points();
            Point<Dim> p{};
            for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
                p[d] = xi[static_cast<std::size_t>(index[d])];
            }
            return p;
        }

        // Вес квадратуры в узле flat: w_{i_0} * w_{i_1} * ...
        Real weight(int flat) const {
            const std::array<int, Dim> index = multiIndex(flat);
            const auto w = basis_.weights();
            Real product = 1;
            for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
                product *= w[static_cast<std::size_t>(index[d])];
            }
            return product;
        }

        // Узловые значения функции: u[flat] = f(point(flat)); f принимает const Point<Dim>&.
        // Это и есть решение задачи u = f при квадратуре GLL (матрица масс диагональна).
        template<class F>
        std::vector<Real> interpolate(F &&f) const {
            std::vector<Real> u(static_cast<std::size_t>(numPoints()));
            for (int flat = 0; flat < numPoints(); ++flat) {
                u[static_cast<std::size_t>(flat)] = f(point(flat));
            }
            return u;
        }

        // Значение u_N(x) = sum u[flat] * phi_flat(x) в произвольной точке x из [-1, 1]^Dim.
        // Бросает std::invalid_argument, если u.size() != numPoints().
        //
        // Через sum factorization: по каждому направлению d берётся матрица 1 x n
        // со строкой (ell_0(x_d), ..., ell_N(x_d)), и tensorApply сворачивает все
        // направления по очереди до одного числа.
        Real evaluate(std::span<const Real> u, const Point<Dim> &x) const {
            checkSize(u, "TensorBasis::evaluate");

            const int n = numPointsPerDirection();
            std::array<DenseMatrix, Dim> rows;
            std::array<const DenseMatrix *, Dim> matrices{};
            for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
                rows[d] = DenseMatrix(1, n);
                basis_.lagrange().evaluateBasis(x[d], rows[d].row(0));
                matrices[d] = &rows[d];
            }

            Real value = 0;
            tensorApply<Dim>(matrices, u, std::span<Real>(&value, 1));
            return value;
        }

        // Узловые значения производной du_N / dx_direction.
        // Бросает std::invalid_argument при неверном direction или размерах.
        //
        // Производная по x_d — одномерная матрица D, применённая вдоль направления d.
        // Результат точен: производная полинома степени N представима в том же базисе.
        void derivative(std::span<const Real> u, int direction, std::span<Real> du) const {
            std::array<int, Dim> shape{};
            shape.fill(numPointsPerDirection());
            applyAlong<Dim>(basis_.D(), direction, shape, u, du); // проверяет все размеры
        }

        // Интеграл по [-1, 1]^Dim квадратурой GLL: sum weight(flat) * u[flat].
        // Точен, если по каждой переменной степень <= 2N - 1.
        // Бросает std::invalid_argument, если u.size() != numPoints().
        Real integrate(std::span<const Real> u) const {
            checkSize(u, "TensorBasis::integrate");
            Real sum = 0;
            for (int flat = 0; flat < numPoints(); ++flat) {
                sum += weight(flat) * u[static_cast<std::size_t>(flat)];
            }
            return sum;
        }

    private:
        void checkSize(std::span<const Real> u, const char *where) const {
            if (u.size() != static_cast<std::size_t>(numPoints())) {
                throw std::invalid_argument(std::string(where) + ": wrong number of nodal values");
            }
        }

        SpectralBasis1D basis_;
    };
} // namespace sem
