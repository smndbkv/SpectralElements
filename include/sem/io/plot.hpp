#pragma once

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
