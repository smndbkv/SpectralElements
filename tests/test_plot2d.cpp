#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "sem/core/point.hpp"
#include "sem/io/plot.hpp"

using sem::Point;
using sem::Real;
using sem::io::AxisScale;
using sem::io::FieldStyle;
using sem::io::Figure2D;

namespace {

bool contains(const std::string& text, const std::string& fragment) {
    return text.find(fragment) != std::string::npos;
}

} // namespace

TEST_CASE("Figure2D: JSON header and settings", "[io][plot][plot2d]") {
    const std::vector<Real> x{0, 1}, y{0, 1}, z{1, 2, 3, 4};
    Figure2D fig("поле");
    fig.setXLabel("x").setYLabel("y").setColorLabel("|e|")
        .setStyle(FieldStyle::Contour).setColorScale(AxisScale::Log).setField(x, y, z);

    const std::string json = fig.toJson();
    CHECK(contains(json, R"("kind": "field2d")"));
    CHECK(contains(json, R"("title": "поле")"));
    CHECK(contains(json, R"("zlabel": "|e|")"));
    CHECK(contains(json, R"("style": "contour")"));
    CHECK(contains(json, R"("zscale": "log")"));
    CHECK(contains(json, R"("points": [])"));

    fig.setStyle(FieldStyle::Surface);
    CHECK(contains(fig.toJson(), R"("style": "surface")"));
    fig.setStyle(FieldStyle::Heatmap);
    CHECK(contains(fig.toJson(), R"("style": "heatmap")"));
}

TEST_CASE("Figure2D: setFunction uses the x-fastest layout", "[io][plot][plot2d]") {
    // z = x + 10 y на сетке x = {0, 1, 2}, y = {1, 2}: z[i + 3 j] = x_i + 10 y_j.
    Figure2D fig;
    fig.setFunction([](const Point<2>& p) { return p[0] + 10 * p[1]; }, 0, 2, 1, 2, 3, 2);

    const std::string json = fig.toJson();
    CHECK(contains(json, R"("x": [0, 1, 2])"));
    CHECK(contains(json, R"("y": [1, 2])"));
    CHECK(contains(json, R"("z": [10, 11, 12, 20, 21, 22])"));
}

TEST_CASE("Figure2D: overlay points", "[io][plot][plot2d]") {
    const std::vector<Real> x{0, 1}, y{0, 1}, z{0, 0, 0, 0};
    const std::vector<Real> px{0.5}, py{-0.5};
    Figure2D fig;
    fig.setField(x, y, z).addPoints(px, py, "узлы", "ko");
    CHECK(contains(fig.toJson(), R"({"label": "узлы", "style": "ko",)"));
    CHECK(contains(fig.toJson(), R"("x": [0.5],)"));
}

TEST_CASE("Figure2D: field label and overlays", "[io][plot][plot2d]") {
    Figure2D fig;
    fig.setFunction([](const Point<2>& p) { return p[0]; }, 0, 1, 0, 1, 2, 2)
        .setFieldLabel("SEM")
        .addOverlayFunction([](const Point<2>& p) { return 2 * p[1]; }, 0, 1, 0, 1, 2, 3, "точное");

    const std::string json = fig.toJson();
    CHECK(contains(json, R"("label": "SEM")"));
    CHECK(contains(json, R"("overlays": [)"));
    CHECK(contains(json, R"({"label": "точное",)"));
    CHECK(contains(json, R"("y": [0, 0.5, 1],)"));            // своя сетка у дополнительного поля
    CHECK(contains(json, R"("z": [0, 0, 1, 1, 2, 2]})"));      // z = 2y, x меняется быстрее

    Figure2D plain;
    plain.setFunction([](const Point<2>&) { return Real{0}; }, 0, 1, 0, 1, 2, 2);
    CHECK(contains(plain.toJson(), R"("overlays": [],)"));
}

TEST_CASE("renderAll: invalid arguments", "[io][plot]") {
    using sem::io::PlotItem;
    const std::vector<PlotItem> none;
    CHECK_THROWS_AS(sem::io::renderAll(none), std::invalid_argument);

    // Без окна и без картинок рисовать незачем.
    const std::vector<PlotItem> noImages{{"{}", "a.json", {}}, {"{}", "b.json", {}}};
    CHECK_THROWS_AS(sem::io::renderAll(noImages, false), std::invalid_argument);

    const std::vector<PlotItem> noJson{{"", "a.json", "a.png"}};
    CHECK_THROWS_AS(sem::io::renderAll(noJson, false), std::invalid_argument);
}

TEST_CASE("Figure2D: invalid arguments", "[io][plot][plot2d]") {
    const std::vector<Real> x{0, 1}, y{0, 1, 2}, one{0};
    Figure2D fig;

    CHECK_THROWS_AS(fig.setField(x, y, std::vector<Real>(5)), std::invalid_argument);  // нужно 6
    CHECK_THROWS_AS(fig.setField(one, y, std::vector<Real>(3)), std::invalid_argument);
    CHECK_THROWS_AS(fig.addPoints(x, y), std::invalid_argument);
    CHECK_THROWS_AS(fig.addOverlay(x, y, std::vector<Real>(4)), std::invalid_argument);  // нужно 6
    CHECK_THROWS_AS(fig.setFunction([](const Point<2>&) { return Real{0}; }, 0, 1, 0, 1, 1, 5),
                    std::invalid_argument);
    // Нет данных — рисовать нечего.
    CHECK_THROWS_AS(fig.render("unused.json", {.image = "unused.png", .showWindow = false}),
                    std::invalid_argument);
}
