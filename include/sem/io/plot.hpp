#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sem/core/point.hpp"
#include "sem/core/types.hpp"

namespace sem::io {

enum class AxisScale { Linear, Log };

struct RenderOptions {
    std::filesystem::path image;  // файл картинки (.png, .pdf, .svg); пусто — не сохранять
    bool showWindow = true;       // окно matplotlib; программа ждёт, пока его не закроют
};

// Один график для renderAll: его JSON (toJson() у Figure или Figure2D), куда этот JSON
// сохранить и куда сохранить картинку (пусто — не сохранять).
struct PlotItem {
    std::string           json;
    std::filesystem::path dataFile;
    std::filesystem::path image;
};

// Рисует несколько графиков одним запуском Python: все окна открываются одновременно,
// и программа ждёт, пока их не закроют. (Отдельные вызовы render открывали бы окна
// по очереди: следующее — только после закрытия предыдущего.)
//
//   sem::io::renderAll(std::vector<sem::io::PlotItem>{
//       {logFigure.toJson(), "log.json", "log.png"},
//       {linearFigure.toJson(), "linear.json", "linear.png"}});
//
// Возвращает true, если скрипт отработал без ошибок.
bool renderAll(std::span<const PlotItem> items, bool showWindow = true);

// Описание графика на стороне C++. Сам рисунок строит скрипт tools/plot.py
// (matplotlib), данные передаются через JSON-файл. Поэтому сборка не зависит
// от Python, а сохранённый JSON можно перерисовать без повторного расчёта:
//   python tools/plot.py data.json -o plot.png
class Figure {
public:
    explicit Figure(std::string title = {});

    Figure& setTitle(std::string title);
    Figure& setXLabel(std::string label);
    Figure& setYLabel(std::string label);
    Figure& setXScale(AxisScale scale);
    Figure& setYScale(AxisScale scale);

    // Линия или точки; style — строка формата matplotlib: "-", "--", "o", "s-", "k." ...
    // В подписях работает mathtext: R"($\ell_j(x)$)".
    Figure& plot(std::span<const Real> x, std::span<const Real> y,
                 std::string label = {}, std::string style = "-");

    // График функции f на [a, b] по numPoints равномерно распределённым точкам.
    template <class F>
    Figure& plotFunction(F&& f, Real a, Real b, int numPoints,
                         std::string label = {}, std::string style = "-");

    std::size_t numSeries() const { return series_.size(); }

    std::string toJson() const;
    void saveJson(const std::filesystem::path& path) const;

    // Сохраняет данные в dataFile и запускает tools/plot.py.
    // Возвращает true, если скрипт отработал без ошибок.
    bool render(const std::filesystem::path& dataFile, const RenderOptions& options = {}) const;

private:
    struct Series {
        std::string       label;
        std::string       style;
        std::vector<Real> x;
        std::vector<Real> y;
    };

    std::string         title_;
    std::string         xLabel_;
    std::string         yLabel_;
    AxisScale           xScale_ = AxisScale::Linear;
    AxisScale           yScale_ = AxisScale::Linear;
    std::vector<Series> series_;
};

enum class FieldStyle {
    Heatmap,  // цветовая карта (pcolormesh)
    Contour,  // заливка изолиниями (contourf)
    Surface,  // 3D-поверхность (plot_surface)
};

// Скалярное поле z(x, y) на прямоугольной сетке. Раскладка значений — та же, что у
// узловых значений TensorBasis<2>: z[i + nx * j] — значение в точке (x[i], y[j]).
// Рисуется тем же скриптом tools/plot.py.
class Figure2D {
public:
    explicit Figure2D(std::string title = {});

    Figure2D& setTitle(std::string title);
    Figure2D& setXLabel(std::string label);
    Figure2D& setYLabel(std::string label);
    Figure2D& setColorLabel(std::string label);  // подпись цветовой шкалы (оси z у поверхности)
    Figure2D& setStyle(FieldStyle style);
    // Log — для модуля ошибки: значения <= 0 не рисуются; у поверхности по оси z
    // откладывается log10(z).
    Figure2D& setColorScale(AxisScale scale);

    // Поле на сетке x (nx точек) на y (ny точек), z.size() == nx * ny.
    Figure2D& setField(std::span<const Real> x, std::span<const Real> y, std::span<const Real> z);

    // Поле функции f(const Point<2>&) на равномерной сетке nx x ny прямоугольника
    // [x0, x1] x [y0, y1].
    template <class F>
    Figure2D& setFunction(F&& f, Real x0, Real x1, Real y0, Real y1, int nx, int ny);

    // Подпись основного поля в легенде (нужна, когда есть дополнительные поля).
    Figure2D& setFieldLabel(std::string label);

    // Дополнительное поле для сравнения с основным (например, точное решение).
    // У поверхности рисуется каркасом (wireframe), у цветовой карты и изолиний — линиями уровня.
    // Сетка может отличаться от сетки основного поля; раскладка та же: z[i + nx * j].
    Figure2D& addOverlay(std::span<const Real> x, std::span<const Real> y, std::span<const Real> z,
                         std::string label = {});

    template <class F>
    Figure2D& addOverlayFunction(F&& f, Real x0, Real x1, Real y0, Real y1, int nx, int ny,
                                 std::string label = {});

    // Точки поверх поля (например, узлы GLL); у поверхности не рисуются.
    Figure2D& addPoints(std::span<const Real> x, std::span<const Real> y,
                        std::string label = {}, std::string style = "k.");

    std::string toJson() const;
    void saveJson(const std::filesystem::path& path) const;
    bool render(const std::filesystem::path& dataFile, const RenderOptions& options = {}) const;

private:
    struct PointSet {
        std::string       label;
        std::string       style;
        std::vector<Real> x;
        std::vector<Real> y;
    };

    struct Field {
        std::string       label;
        std::vector<Real> x;
        std::vector<Real> y;
        std::vector<Real> z;
    };

    // Значения f на равномерной сетке nx x ny, раскладка z[i + nx * j].
    template <class F>
    static Field sample(F&& f, Real x0, Real x1, Real y0, Real y1, int nx, int ny, const char* who);

    static void checkField(std::span<const Real> x, std::span<const Real> y,
                           std::span<const Real> z, const char* who);

    std::string           title_;
    std::string           xLabel_;
    std::string           yLabel_;
    std::string           colorLabel_;
    FieldStyle            style_      = FieldStyle::Heatmap;
    AxisScale             colorScale_ = AxisScale::Linear;
    Field                 field_;
    std::vector<Field>    overlays_;
    std::vector<PointSet> points_;
};

// Векторное поле в 3D: стрелки (quiver) в точках (x_k, y_k, z_k) с векторами (vx_k, vy_k, vz_k).
// Цвет стрелки — модуль вектора (шкала справа), длина — пропорциональна модулю:
// самая длинная стрелка имеет длину arrowLength. Если normalize = true, все стрелки
// одной длины arrowLength и показывают только направление.
class VectorFigure3D {
public:
    explicit VectorFigure3D(std::string title = {});

    VectorFigure3D& setTitle(std::string title);
    VectorFigure3D& setXLabel(std::string label);
    VectorFigure3D& setYLabel(std::string label);
    VectorFigure3D& setZLabel(std::string label);
    VectorFigure3D& setColorLabel(std::string label);  // подпись шкалы цвета (модуль вектора)
    VectorFigure3D& setArrowLength(Real length);       // <= 0: подобрать автоматически
    VectorFigure3D& setNormalize(bool normalize);

    // Все шесть массивов одной длины.
    VectorFigure3D& setVectors(std::span<const Real> x, std::span<const Real> y, std::span<const Real> z,
                               std::span<const Real> vx, std::span<const Real> vy,
                               std::span<const Real> vz);

    // Поле f(const Point<3>&) -> Point<3> на равномерной сетке n x n x n параллелепипеда
    // [lower, upper]. Если длина стрелок не задана, берётся 0.9 шага сетки.
    template <class F>
    VectorFigure3D& setFunction(F&& f, const Point<3>& lower, const Point<3>& upper, int n);

    std::string toJson() const;
    void saveJson(const std::filesystem::path& path) const;
    bool render(const std::filesystem::path& dataFile, const RenderOptions& options = {}) const;

private:
    std::string       title_;
    std::string       xLabel_ = "$x$";
    std::string       yLabel_ = "$y$";
    std::string       zLabel_ = "$z$";
    std::string       colorLabel_;
    Real              arrowLength_ = 0;
    bool              normalize_   = false;
    std::vector<Real> x_, y_, z_, vx_, vy_, vz_;
};

enum class Scalar3DStyle {
    Faces,    // видимые грани параллелепипеда, раскрашенные по значению
    Scatter,  // облако точек всей сетки, раскрашенных по значению
};

// Скалярное поле в 3D на регулярной сетке nx x ny x nz. Раскладка — как у TensorBasis<3>:
// values[i + nx * (j + ny * k)] — значение в точке (x[i], y[j], z[k]).
class ScalarFigure3D {
public:
    explicit ScalarFigure3D(std::string title = {});

    ScalarFigure3D& setTitle(std::string title);
    ScalarFigure3D& setXLabel(std::string label);
    ScalarFigure3D& setYLabel(std::string label);
    ScalarFigure3D& setZLabel(std::string label);
    ScalarFigure3D& setColorLabel(std::string label);
    ScalarFigure3D& setStyle(Scalar3DStyle style);

    ScalarFigure3D& setField(std::span<const Real> x, std::span<const Real> y, std::span<const Real> z,
                             std::span<const Real> values);

    // Значения f(const Point<3>&) на равномерной сетке n x n x n параллелепипеда [lower, upper].
    template <class F>
    ScalarFigure3D& setFunction(F&& f, const Point<3>& lower, const Point<3>& upper, int n);

    std::string toJson() const;
    void saveJson(const std::filesystem::path& path) const;
    bool render(const std::filesystem::path& dataFile, const RenderOptions& options = {}) const;

private:
    std::string       title_;
    std::string       xLabel_ = "$x$";
    std::string       yLabel_ = "$y$";
    std::string       zLabel_ = "$z$";
    std::string       colorLabel_;
    Scalar3DStyle     style_ = Scalar3DStyle::Faces;
    std::vector<Real> x_, y_, z_, values_;
};

template <class F>
ScalarFigure3D& ScalarFigure3D::setFunction(F&& f, const Point<3>& lower, const Point<3>& upper,
                                            int n) {
    if (n < 2) {
        throw std::invalid_argument("ScalarFigure3D::setFunction: at least 2 points per direction");
    }
    std::array<std::vector<Real>, 3> axes;
    for (std::size_t d = 0; d < 3; ++d) {
        for (int i = 0; i < n; ++i) {
            axes[d].push_back(lower[d] + (upper[d] - lower[d]) * static_cast<Real>(i) /
                                             static_cast<Real>(n - 1));
        }
    }
    std::vector<Real> values;
    values.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (const Real zk : axes[2]) {
        for (const Real yj : axes[1]) {
            for (const Real xi : axes[0]) {
                values.push_back(f(Point<3>{xi, yj, zk}));
            }
        }
    }
    return setField(axes[0], axes[1], axes[2], values);
}

template <class F>
VectorFigure3D& VectorFigure3D::setFunction(F&& f, const Point<3>& lower, const Point<3>& upper,
                                            int n) {
    if (n < 2) {
        throw std::invalid_argument("VectorFigure3D::setFunction: at least 2 points per direction");
    }
    std::vector<Real> x, y, z, vx, vy, vz;
    Real minStep = 0;
    for (std::size_t d = 0; d < 3; ++d) {
        const Real step = (upper[d] - lower[d]) / static_cast<Real>(n - 1);
        minStep = d == 0 ? step : std::min(minStep, step);
    }
    const auto coordinate = [&](std::size_t d, int i) {
        return lower[d] + (upper[d] - lower[d]) * static_cast<Real>(i) / static_cast<Real>(n - 1);
    };
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const Point<3> p{coordinate(0, i), coordinate(1, j), coordinate(2, k)};
                const Point<3> v = f(p);
                x.push_back(p[0]);
                y.push_back(p[1]);
                z.push_back(p[2]);
                vx.push_back(v[0]);
                vy.push_back(v[1]);
                vz.push_back(v[2]);
            }
        }
    }
    if (!(arrowLength_ > 0)) {
        arrowLength_ = Real(0.9) * minStep;
    }
    return setVectors(x, y, z, vx, vy, vz);
}

template <class F>
Figure2D::Field Figure2D::sample(F&& f, Real x0, Real x1, Real y0, Real y1, int nx, int ny,
                                 const char* who) {
    if (nx < 2 || ny < 2) {
        throw std::invalid_argument(std::string(who) + ": at least 2 points per direction");
    }
    Field field;
    field.x.resize(static_cast<std::size_t>(nx));
    field.y.resize(static_cast<std::size_t>(ny));
    for (std::size_t i = 0; i < field.x.size(); ++i) {
        field.x[i] = x0 + (x1 - x0) * static_cast<Real>(i) / static_cast<Real>(nx - 1);
    }
    for (std::size_t j = 0; j < field.y.size(); ++j) {
        field.y[j] = y0 + (y1 - y0) * static_cast<Real>(j) / static_cast<Real>(ny - 1);
    }
    field.z.reserve(field.x.size() * field.y.size());
    for (const Real yj : field.y) {
        for (const Real xi : field.x) {
            field.z.push_back(f(Point<2>{xi, yj}));
        }
    }
    return field;
}

template <class F>
Figure2D& Figure2D::setFunction(F&& f, Real x0, Real x1, Real y0, Real y1, int nx, int ny) {
    const Field field = sample(f, x0, x1, y0, y1, nx, ny, "Figure2D::setFunction");
    return setField(field.x, field.y, field.z);
}

template <class F>
Figure2D& Figure2D::addOverlayFunction(F&& f, Real x0, Real x1, Real y0, Real y1, int nx, int ny,
                                       std::string label) {
    const Field field = sample(f, x0, x1, y0, y1, nx, ny, "Figure2D::addOverlayFunction");
    return addOverlay(field.x, field.y, field.z, std::move(label));
}

template <class F>
Figure& Figure::plotFunction(F&& f, Real a, Real b, int numPoints,
                             std::string label, std::string style) {
    if (numPoints < 2) {
        throw std::invalid_argument("Figure::plotFunction: at least 2 points are required");
    }
    std::vector<Real> x(static_cast<std::size_t>(numPoints));
    std::vector<Real> y(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] = a + (b - a) * static_cast<Real>(k) / static_cast<Real>(numPoints - 1);
        y[k] = f(x[k]);
    }
    return plot(x, y, std::move(label), std::move(style));
}

} // namespace sem::io
