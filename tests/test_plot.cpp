#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "sem/io/plot.hpp"

using sem::Real;
using sem::io::AxisScale;
using sem::io::Figure;

namespace {

bool contains(const std::string& text, const std::string& fragment) {
    return text.find(fragment) != std::string::npos;
}

} // namespace

TEST_CASE("Figure: JSON contains settings and series", "[io][plot]") {
    const std::vector<Real> x{0, 0.5, 1};
    const std::vector<Real> y{1, 0.1, -2.5e-17};

    Figure fig("title");
    fig.setXLabel("x").setYLabel("y").setYScale(AxisScale::Log).plot(x, y, "data", "o-");
    REQUIRE(fig.numSeries() == 1);

    const std::string json = fig.toJson();
    CHECK(contains(json, R"("title": "title")"));
    CHECK(contains(json, R"("xscale": "linear")"));
    CHECK(contains(json, R"("yscale": "log")"));
    CHECK(contains(json, R"("label": "data", "style": "o-")"));
    // Кратчайшее точное представление чисел.
    CHECK(contains(json, R"("x": [0, 0.5, 1])"));
    CHECK(contains(json, R"("y": [1, 0.1, -2.5e-17])"));
}

TEST_CASE("Figure: strings are escaped, UTF-8 is kept", "[io][plot]") {
    Figure fig(R"(a "quoted" \ell)");
    fig.setXLabel("строка\nвторая");

    const std::string json = fig.toJson();
    CHECK(contains(json, R"("title": "a \"quoted\" \\ell")"));
    CHECK(contains(json, R"("xlabel": "строка\nвторая")"));
    CHECK(contains(json, R"("series": [])"));
}

TEST_CASE("Figure: non-finite values", "[io][plot]") {
    const std::vector<Real> x{0, 1, 2};
    const std::vector<Real> y{std::numeric_limits<Real>::quiet_NaN(),
                              std::numeric_limits<Real>::infinity(),
                              -std::numeric_limits<Real>::infinity()};
    Figure fig;
    fig.plot(x, y);
    CHECK(contains(fig.toJson(), R"("y": [NaN, Infinity, -Infinity])"));
}

TEST_CASE("Figure: plotFunction samples the function uniformly", "[io][plot]") {
    Figure fig;
    fig.plotFunction([](Real t) { return 2 * t; }, 0, 1, 3, "line");
    CHECK(contains(fig.toJson(), R"("x": [0, 0.5, 1])"));
    CHECK(contains(fig.toJson(), R"("y": [0, 1, 2])"));
}

TEST_CASE("Figure: saveJson writes exactly toJson()", "[io][plot]") {
    Figure fig("файл");
    const std::vector<Real> x{1, 2};
    fig.plot(x, x);

    const auto path = std::filesystem::temp_directory_path() / "sem_plot_test.json";
    fig.saveJson(path);

    std::ifstream file(path, std::ios::binary);
    const std::string content{std::istreambuf_iterator<char>(file), {}};
    file.close();
    std::filesystem::remove(path);

    CHECK(content == fig.toJson());
}

TEST_CASE("Figure: invalid arguments", "[io][plot]") {
    const std::vector<Real> a{1, 2, 3};
    const std::vector<Real> b{1, 2};
    Figure fig;

    CHECK_THROWS_AS(fig.plot(a, b), std::invalid_argument);
    CHECK_THROWS_AS(fig.plotFunction([](Real t) { return t; }, 0, 1, 1), std::invalid_argument);
    CHECK_THROWS_AS(fig.render("unused.json", {.image = {}, .showWindow = false}),
                    std::invalid_argument);
}
