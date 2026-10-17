#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sem/basis/tensor_basis.hpp"
#include "sem/core/point.hpp"
#include "sem/core/types.hpp"
#include "sem/operators/stiffness.hpp"

namespace sem {

// Равномерная сетка спектральных элементов в кубе [lower, upper]^Dim: по каждому
// направлению K одинаковых элементов длины h = (upper - lower) / K, в каждом —
// тензорный базис степени N на узлах GLL.
//
// Глобальные узлы. Соседние элементы делят узлы на общей грани, поэтому функция
// непрерывна (пространство C^0). Узлы образуют сетку M^Dim, M = K N + 1, по каждому
// направлению: глобальный индекс g = e N + i, где e — номер элемента, i — локальный
// индекс узла. Раскладка — как у TensorBasis: x меняется быстрее всего.
// Элементы нумеруются так же (x быстрее всего).
//
// Отображение эталонного элемента [-1, 1]^Dim на элемент e:
//   x_d = lower + h e_d + (xi_d + 1) h / 2,   якобиан J = h / 2 по каждому направлению.
// Поэтому на элементе
//   M_e = J^Dim  M_ref,      K_e = J^(Dim - 2)  K_ref
// (у K производные дают 1 / J^2, а объём — J^Dim). В 2D K_e = K_ref.
//
// Глобальные операторы собираются из элементных (direct stiffness summation):
//   y = sum_e Q_e  A_e  Q_e^T x,
// где Q_e^T (gather) выбирает из глобального вектора значения в узлах элемента,
// а Q_e (scatter) складывает результат обратно: в общих узлах вклады суммируются.
// Матрицы не собираются; M остаётся диагональной (квадратура GLL).
template <int Dim>
class BoxMesh {
public:
    BoxMesh(int degree, int elementsPerDirection, Real lower = -1, Real upper = 1)
        : basis_(degree), K_(elementsPerDirection), lower_(lower), upper_(upper) {
        if (elementsPerDirection < 1) {
            throw std::invalid_argument("BoxMesh: elementsPerDirection must be at least 1");
        }
        if (!(upper > lower)) {
            throw std::invalid_argument("BoxMesh: upper must be greater than lower");
        }
        h_ = (upper - lower) / K_;

        // Таблица локальный узел -> глобальный для всех элементов.
        const int local = basis_.numPoints();
        map_.resize(static_cast<std::size_t>(numElements()) * static_cast<std::size_t>(local));
        for (int e = 0; e < numElements(); ++e) {
            const std::array<int, Dim> element = elementIndex(e);
            for (int k = 0; k < local; ++k) {
                const std::array<int, Dim> i = basis_.multiIndex(k);
                std::array<int, Dim> g{};
                for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
                    g[d] = element[d] * degree + i[d];
                }
                map_[static_cast<std::size_t>(e * local + k)] = flatNode(g);
            }
        }
    }

    const TensorBasis<Dim>& basis() const { return basis_; }
    int degree() const { return basis_.degree(); }
    int elementsPerDirection() const { return K_; }
    int numElements() const { return ipow(K_); }
    Real lower() const { return lower_; }
    Real upper() const { return upper_; }
    Real elementSize() const { return h_; }
    Real jacobian() const { return h_ / 2; }  // J = h / 2 по каждому направлению

    int nodesPerDirection() const { return K_ * degree() + 1; }  // M = K N + 1
    int numNodes() const { return ipow(nodesPerDirection()); }

    // Мультииндекс элемента по его номеру (x быстрее всего).
    std::array<int, Dim> elementIndex(int element) const {
        std::array<int, Dim> index{};
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            index[d] = element % K_;
            element /= K_;
        }
        return index;
    }

    // Глобальный номер k-го локального узла элемента.
    int globalIndex(int element, int local) const {
        return map_[static_cast<std::size_t>(element * basis_.numPoints() + local)];
    }

    // Глобальный мультииндекс узла (по каждому направлению 0..M-1).
    std::array<int, Dim> nodeIndex(int node) const {
        const int m = nodesPerDirection();
        std::array<int, Dim> index{};
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            index[d] = node % m;
            node /= m;
        }
        return index;
    }

    // Координаты глобального узла.
    Point<Dim> node(int node) const {
        const std::array<int, Dim> g = nodeIndex(node);
        const auto xi = basis_.basis1D().points();
        const int N = degree();
        Point<Dim> p{};
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            // Последний узел принадлежит последнему элементу, остальные — элементу g / N.
            const int e = std::min(g[d] / N, K_ - 1);
            const int i = g[d] - e * N;
            p[d] = lower_ + h_ * e + (xi[static_cast<std::size_t>(i)] + 1) * jacobian();
        }
        return p;
    }

    // true для узлов на границе куба.
    std::vector<bool> boundaryMask() const {
        const int last = nodesPerDirection() - 1;
        std::vector<bool> mask(static_cast<std::size_t>(numNodes()));
        for (int k = 0; k < numNodes(); ++k) {
            for (const int g : nodeIndex(k)) {
                if (g == 0 || g == last) mask[static_cast<std::size_t>(k)] = true;
            }
        }
        return mask;
    }

    // Узловые значения функции f(const Point<Dim>&).
    template <class F>
    std::vector<Real> interpolate(F&& f) const {
        std::vector<Real> u(static_cast<std::size_t>(numNodes()));
        for (int k = 0; k < numNodes(); ++k) {
            u[static_cast<std::size_t>(k)] = f(node(k));
        }
        return u;
    }

    // Q_e^T: значения глобального вектора в узлах элемента.
    void gather(int element, std::span<const Real> global, std::span<Real> local) const {
        for (int k = 0; k < basis_.numPoints(); ++k) {
            local[static_cast<std::size_t>(k)] = global[static_cast<std::size_t>(globalIndex(element, k))];
        }
    }

    // Q_e: прибавить элементный вектор к глобальному (в общих узлах вклады складываются).
    void scatterAdd(int element, std::span<const Real> local, std::span<Real> global) const {
        for (int k = 0; k < basis_.numPoints(); ++k) {
            global[static_cast<std::size_t>(globalIndex(element, k))] += local[static_cast<std::size_t>(k)];
        }
    }

    // Диагональ собранной матрицы масс: sum_e J^Dim w_k по элементам, содержащим узел.
    std::vector<Real> massDiagonal() const {
        const Real scale = std::pow(jacobian(), Dim);
        std::vector<Real> local(static_cast<std::size_t>(basis_.numPoints()));
        for (int k = 0; k < basis_.numPoints(); ++k) {
            local[static_cast<std::size_t>(k)] = scale * basis_.weight(k);
        }
        std::vector<Real> mass(static_cast<std::size_t>(numNodes()), Real{0});
        for (int e = 0; e < numElements(); ++e) {
            scatterAdd(e, local, mass);
        }
        return mass;
    }

    // y = K x — собранная матрица жёсткости (дискретный -Laplace) без сборки:
    // для каждого элемента gather, applyK на эталонном элементе, множитель J^(Dim-2), scatter.
    void applyK(std::span<const Real> x, std::span<Real> y) const {
        checkSize(x, "BoxMesh::applyK");
        checkSize(y, "BoxMesh::applyK");
        const Real scale = std::pow(jacobian(), Dim - 2);
        std::vector<Real> in(static_cast<std::size_t>(basis_.numPoints()));
        std::vector<Real> out(in.size());
        std::fill(y.begin(), y.end(), Real{0});
        for (int e = 0; e < numElements(); ++e) {
            gather(e, x, in);
            sem::applyK<Dim>(basis_, in, out);
            for (Real& v : out) v *= scale;
            scatterAdd(e, out, y);
        }
    }

    // Диагональ собранной матрицы жёсткости (например, для предобусловливателя Якоби).
    std::vector<Real> diagonalK() const {
        std::vector<Real> local = sem::diagonalK<Dim>(basis_);
        const Real scale = std::pow(jacobian(), Dim - 2);
        for (Real& v : local) v *= scale;
        std::vector<Real> diagonal(static_cast<std::size_t>(numNodes()), Real{0});
        for (int e = 0; e < numElements(); ++e) {
            scatterAdd(e, local, diagonal);
        }
        return diagonal;
    }

    // Интеграл по кубу (квадратура GLL): sum_i M_ii u_i.
    Real integrate(std::span<const Real> u) const {
        checkSize(u, "BoxMesh::integrate");
        const std::vector<Real> mass = massDiagonal();
        Real sum = 0;
        for (std::size_t k = 0; k < mass.size(); ++k) sum += mass[k] * u[k];
        return sum;
    }

    // Элемент, содержащий точку x, и её координаты на эталонном элементе.
    // Точка на общей грани относится к элементу с меньшим номером по этому направлению.
    std::pair<int, Point<Dim>> locate(const Point<Dim>& x) const {
        int element = 0, stride = 1;
        Point<Dim> xi{};
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            const Real s = (x[d] - lower_) / h_;  // координата в «элементах»
            int e = static_cast<int>(std::ceil(s)) - 1;
            e = std::clamp(e, 0, K_ - 1);
            xi[d] = 2 * (s - e) - 1;
            element += e * stride;
            stride *= K_;
        }
        return {element, xi};
    }

    // Отображение эталонного элемента на элемент element.
    Point<Dim> toPhysical(int element, const Point<Dim>& xi) const {
        const std::array<int, Dim> e = elementIndex(element);
        Point<Dim> x{};
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            x[d] = lower_ + h_ * e[d] + (xi[d] + 1) * jacobian();
        }
        return x;
    }

    // Значение приближения в точке x.
    Real evaluate(std::span<const Real> u, const Point<Dim>& x) const {
        checkSize(u, "BoxMesh::evaluate");
        const auto [element, xi] = locate(x);
        std::vector<Real> local(static_cast<std::size_t>(basis_.numPoints()));
        gather(element, u, local);
        return basis_.evaluate(local, xi);
    }

private:
    int ipow(int base) const {
        int result = 1;
        for (int d = 0; d < Dim; ++d) result *= base;
        return result;
    }

    int flatNode(const std::array<int, Dim>& g) const {
        const int m = nodesPerDirection();
        int flat = 0;
        for (int d = Dim - 1; d >= 0; --d) {
            flat = flat * m + g[static_cast<std::size_t>(d)];
        }
        return flat;
    }

    void checkSize(std::span<const Real> u, const char* where) const {
        if (u.size() != static_cast<std::size_t>(numNodes())) {
            throw std::invalid_argument(std::string(where) + ": wrong number of nodal values");
        }
    }

    TensorBasis<Dim> basis_;
    int              K_;
    Real             lower_, upper_, h_ = 0;
    std::vector<int> map_;  // map_[e * n^Dim + k] — глобальный номер k-го узла элемента e
};

} // namespace sem
