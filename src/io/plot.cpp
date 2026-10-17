#include "sem/io/plot.hpp"

#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string_view>
#include <system_error>

#include "sem/io/plot_config.hpp"

namespace sem::io {

namespace {

void appendString(std::string& out, std::string_view s) {
    out += '"';
    for (const char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x",
                              static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += buf;
            } else {
                out += c;  // байты UTF-8 (в том числе кириллица) передаются как есть
            }
        }
    }
    out += '"';
}

void appendNumber(std::string& out, Real v) {
    // NaN и бесконечности не входят в стандарт JSON, но модуль json в Python
    // их понимает, а matplotlib просто не рисует такие точки.
    if (std::isnan(v)) {
        out += "NaN";
        return;
    }
    if (std::isinf(v)) {
        out += v > 0 ? "Infinity" : "-Infinity";
        return;
    }
    // Кратчайшее представление, которое читается обратно без потерь и не зависит от локали.
    char buf[32];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v);
    assert(ec == std::errc{});
    out.append(buf, end);
}

void appendArray(std::string& out, const std::vector<Real>& values) {
    out += '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        appendNumber(out, values[i]);
    }
    out += ']';
}

std::string_view scaleName(AxisScale scale) {
    return scale == AxisScale::Log ? "log" : "linear";
}

// Запускает tools/plot.py для одного или нескольких графиков одним процессом:
//   python plot.py data1.json data2.json ... --output img1 --output img2 ... [--no-show]
// Картинка "-" означает «не сохранять». Все окна открываются одновременно.
int runPlotScript(std::span<const PlotItem> items, bool showWindow) {
    const std::filesystem::path python(config::kPythonExecutable);
    const std::filesystem::path script(config::kPlotScript);
    const std::filesystem::path noImage("-");

#ifdef _WIN32
    // Широкие строки: пути с кириллицей не должны проходить через ANSI-кодировку.
    const auto quote = [](const std::filesystem::path& p) {
        return L"\"" + p.wstring() + L"\"";
    };
    std::wstring command = quote(python) + L" " + quote(script);
    for (const PlotItem& item : items) {
        command += L" " + quote(item.dataFile);
    }
    for (const PlotItem& item : items) {
        command += L" --output " + quote(item.image.empty() ? noImage : item.image);
    }
    if (!showWindow) {
        command += L" --no-show";
    }
    // cmd.exe /c снимает внешнюю пару кавычек, поэтому вся команда оборачивается ещё раз.
    command = L"\"" + command + L"\"";
    return _wsystem(command.c_str());
#else
    const auto quote = [](const std::filesystem::path& p) { return "'" + p.string() + "'"; };
    std::string command = quote(python) + " " + quote(script);
    for (const PlotItem& item : items) {
        command += " " + quote(item.dataFile);
    }
    for (const PlotItem& item : items) {
        command += " --output " + quote(item.image.empty() ? noImage : item.image);
    }
    if (!showWindow) {
        command += " --no-show";
    }
    return std::system(command.c_str());
#endif
}

void writeTextFile(const std::filesystem::path& path, const std::string& text, const char* who) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string(who) + ": cannot open " + path.string());
    }
    file << text;
    if (!file) {
        throw std::runtime_error(std::string(who) + ": write failed for " + path.string());
    }
}

// Общая часть render и renderAll: проверки, запись JSON и запуск tools/plot.py.
bool renderItems(std::span<const PlotItem> items, bool showWindow, const char* who) {
    if (items.empty()) {
        throw std::invalid_argument(std::string(who) + ": nothing to render");
    }
    bool anyImage = false;
    for (const PlotItem& item : items) {
        if (item.json.empty() || item.dataFile.empty()) {
            throw std::invalid_argument(std::string(who) + ": every item needs json and a data file");
        }
        anyImage = anyImage || !item.image.empty();
    }
    if (!showWindow && !anyImage) {
        throw std::invalid_argument(
            std::string(who) + ": nothing to do, set an image path or enable the window");
    }
    for (const PlotItem& item : items) {
        writeTextFile(item.dataFile, item.json, who);
    }

    const int status = runPlotScript(items, showWindow);
    if (status != 0) {
        std::cerr << who << ": plot script failed (exit code " << status
                  << "). Check SEM_PYTHON_EXECUTABLE in CMake and that matplotlib is installed.\n";
        return false;
    }
    return true;
}

bool renderJson(const std::string& json, const std::filesystem::path& dataFile,
                const RenderOptions& options, const char* who) {
    const PlotItem item{json, dataFile, options.image};
    return renderItems(std::span<const PlotItem>(&item, 1), options.showWindow, who);
}

std::string_view fieldStyleName(FieldStyle style) {
    switch (style) {
    case FieldStyle::Heatmap: return "heatmap";
    case FieldStyle::Contour: return "contour";
    case FieldStyle::Surface: return "surface";
    }
    return "heatmap";
}

} // namespace

Figure::Figure(std::string title) : title_(std::move(title)) {}

Figure& Figure::setTitle(std::string title) {
    title_ = std::move(title);
    return *this;
}

Figure& Figure::setXLabel(std::string label) {
    xLabel_ = std::move(label);
    return *this;
}

Figure& Figure::setYLabel(std::string label) {
    yLabel_ = std::move(label);
    return *this;
}

Figure& Figure::setXScale(AxisScale scale) {
    xScale_ = scale;
    return *this;
}

Figure& Figure::setYScale(AxisScale scale) {
    yScale_ = scale;
    return *this;
}

Figure& Figure::plot(std::span<const Real> x, std::span<const Real> y,
                     std::string label, std::string style) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Figure::plot: x and y must have the same size");
    }
    series_.push_back({std::move(label), std::move(style),
                       std::vector<Real>(x.begin(), x.end()),
                       std::vector<Real>(y.begin(), y.end())});
    return *this;
}

std::string Figure::toJson() const {
    std::string out = "{\n";
    out += "  \"title\": ";
    appendString(out, title_);
    out += ",\n  \"xlabel\": ";
    appendString(out, xLabel_);
    out += ",\n  \"ylabel\": ";
    appendString(out, yLabel_);
    out += ",\n  \"xscale\": ";
    appendString(out, scaleName(xScale_));
    out += ",\n  \"yscale\": ";
    appendString(out, scaleName(yScale_));
    out += ",\n  \"series\": [";
    for (std::size_t s = 0; s < series_.size(); ++s) {
        const Series& series = series_[s];
        out += s == 0 ? "\n" : ",\n";
        out += "    {\"label\": ";
        appendString(out, series.label);
        out += ", \"style\": ";
        appendString(out, series.style);
        out += ",\n     \"x\": ";
        appendArray(out, series.x);
        out += ",\n     \"y\": ";
        appendArray(out, series.y);
        out += '}';
    }
    out += series_.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

void Figure::saveJson(const std::filesystem::path& path) const {
    writeTextFile(path, toJson(), "Figure::saveJson");
}

bool Figure::render(const std::filesystem::path& dataFile, const RenderOptions& options) const {
    return renderJson(toJson(), dataFile, options, "sem::io::Figure::render");
}

// ---------------------------------------------------------------------------
// Figure2D
// ---------------------------------------------------------------------------

Figure2D::Figure2D(std::string title) : title_(std::move(title)) {}

Figure2D& Figure2D::setTitle(std::string title) {
    title_ = std::move(title);
    return *this;
}

Figure2D& Figure2D::setXLabel(std::string label) {
    xLabel_ = std::move(label);
    return *this;
}

Figure2D& Figure2D::setYLabel(std::string label) {
    yLabel_ = std::move(label);
    return *this;
}

Figure2D& Figure2D::setColorLabel(std::string label) {
    colorLabel_ = std::move(label);
    return *this;
}

Figure2D& Figure2D::setStyle(FieldStyle style) {
    style_ = style;
    return *this;
}

Figure2D& Figure2D::setColorScale(AxisScale scale) {
    colorScale_ = scale;
    return *this;
}

void Figure2D::checkField(std::span<const Real> x, std::span<const Real> y,
                          std::span<const Real> z, const char* who) {
    if (x.size() < 2 || y.size() < 2) {
        throw std::invalid_argument(std::string(who) + ": at least 2 points per direction");
    }
    if (z.size() != x.size() * y.size()) {
        throw std::invalid_argument(std::string(who) + ": z must hold x.size() * y.size() values");
    }
}

Figure2D& Figure2D::setField(std::span<const Real> x, std::span<const Real> y,
                             std::span<const Real> z) {
    checkField(x, y, z, "Figure2D::setField");
    field_.x.assign(x.begin(), x.end());
    field_.y.assign(y.begin(), y.end());
    field_.z.assign(z.begin(), z.end());
    return *this;
}

Figure2D& Figure2D::setFieldLabel(std::string label) {
    field_.label = std::move(label);
    return *this;
}

Figure2D& Figure2D::addOverlay(std::span<const Real> x, std::span<const Real> y,
                               std::span<const Real> z, std::string label) {
    checkField(x, y, z, "Figure2D::addOverlay");
    overlays_.push_back({std::move(label), std::vector<Real>(x.begin(), x.end()),
                         std::vector<Real>(y.begin(), y.end()),
                         std::vector<Real>(z.begin(), z.end())});
    return *this;
}

Figure2D& Figure2D::addPoints(std::span<const Real> x, std::span<const Real> y,
                              std::string label, std::string style) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Figure2D::addPoints: x and y must have the same size");
    }
    points_.push_back({std::move(label), std::move(style),
                       std::vector<Real>(x.begin(), x.end()),
                       std::vector<Real>(y.begin(), y.end())});
    return *this;
}

std::string Figure2D::toJson() const {
    std::string out = "{\n  \"kind\": \"field2d\",\n  \"title\": ";
    appendString(out, title_);
    out += ",\n  \"xlabel\": ";
    appendString(out, xLabel_);
    out += ",\n  \"ylabel\": ";
    appendString(out, yLabel_);
    out += ",\n  \"zlabel\": ";
    appendString(out, colorLabel_);
    out += ",\n  \"style\": ";
    appendString(out, fieldStyleName(style_));
    out += ",\n  \"zscale\": ";
    appendString(out, scaleName(colorScale_));
    out += ",\n  \"label\": ";
    appendString(out, field_.label);
    out += ",\n  \"x\": ";
    appendArray(out, field_.x);
    out += ",\n  \"y\": ";
    appendArray(out, field_.y);
    out += ",\n  \"z\": ";
    appendArray(out, field_.z);
    out += ",\n  \"overlays\": [";
    for (std::size_t s = 0; s < overlays_.size(); ++s) {
        const Field& overlay = overlays_[s];
        out += s == 0 ? "\n" : ",\n";
        out += "    {\"label\": ";
        appendString(out, overlay.label);
        out += ",\n     \"x\": ";
        appendArray(out, overlay.x);
        out += ",\n     \"y\": ";
        appendArray(out, overlay.y);
        out += ",\n     \"z\": ";
        appendArray(out, overlay.z);
        out += '}';
    }
    out += overlays_.empty() ? "]" : "\n  ]";
    out += ",\n  \"points\": [";
    for (std::size_t s = 0; s < points_.size(); ++s) {
        const PointSet& set = points_[s];
        out += s == 0 ? "\n" : ",\n";
        out += "    {\"label\": ";
        appendString(out, set.label);
        out += ", \"style\": ";
        appendString(out, set.style);
        out += ",\n     \"x\": ";
        appendArray(out, set.x);
        out += ",\n     \"y\": ";
        appendArray(out, set.y);
        out += '}';
    }
    out += points_.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

void Figure2D::saveJson(const std::filesystem::path& path) const {
    writeTextFile(path, toJson(), "Figure2D::saveJson");
}

bool Figure2D::render(const std::filesystem::path& dataFile, const RenderOptions& options) const {
    if (field_.z.empty()) {
        throw std::invalid_argument("Figure2D::render: no field data, call setField or setFunction");
    }
    return renderJson(toJson(), dataFile, options, "sem::io::Figure2D::render");
}

// ---------------------------------------------------------------------------
// VectorFigure3D
// ---------------------------------------------------------------------------

VectorFigure3D::VectorFigure3D(std::string title) : title_(std::move(title)) {}

VectorFigure3D& VectorFigure3D::setTitle(std::string title) {
    title_ = std::move(title);
    return *this;
}

VectorFigure3D& VectorFigure3D::setXLabel(std::string label) {
    xLabel_ = std::move(label);
    return *this;
}

VectorFigure3D& VectorFigure3D::setYLabel(std::string label) {
    yLabel_ = std::move(label);
    return *this;
}

VectorFigure3D& VectorFigure3D::setZLabel(std::string label) {
    zLabel_ = std::move(label);
    return *this;
}

VectorFigure3D& VectorFigure3D::setColorLabel(std::string label) {
    colorLabel_ = std::move(label);
    return *this;
}

VectorFigure3D& VectorFigure3D::setArrowLength(Real length) {
    arrowLength_ = length;
    return *this;
}

VectorFigure3D& VectorFigure3D::setNormalize(bool normalize) {
    normalize_ = normalize;
    return *this;
}

VectorFigure3D& VectorFigure3D::setVectors(std::span<const Real> x, std::span<const Real> y,
                                           std::span<const Real> z, std::span<const Real> vx,
                                           std::span<const Real> vy, std::span<const Real> vz) {
    const std::size_t n = x.size();
    if (y.size() != n || z.size() != n || vx.size() != n || vy.size() != n || vz.size() != n) {
        throw std::invalid_argument("VectorFigure3D::setVectors: all arrays must have the same size");
    }
    x_.assign(x.begin(), x.end());
    y_.assign(y.begin(), y.end());
    z_.assign(z.begin(), z.end());
    vx_.assign(vx.begin(), vx.end());
    vy_.assign(vy.begin(), vy.end());
    vz_.assign(vz.begin(), vz.end());
    return *this;
}

std::string VectorFigure3D::toJson() const {
    std::string out = "{\n  \"kind\": \"vector3d\",\n  \"title\": ";
    appendString(out, title_);
    out += ",\n  \"xlabel\": ";
    appendString(out, xLabel_);
    out += ",\n  \"ylabel\": ";
    appendString(out, yLabel_);
    out += ",\n  \"zlabel\": ";
    appendString(out, zLabel_);
    out += ",\n  \"clabel\": ";
    appendString(out, colorLabel_);
    out += ",\n  \"arrow_length\": ";
    appendNumber(out, arrowLength_);
    out += ",\n  \"normalize\": ";
    out += normalize_ ? "true" : "false";
    const std::pair<const char*, const std::vector<Real>*> arrays[] = {
        {"x", &x_}, {"y", &y_}, {"z", &z_}, {"vx", &vx_}, {"vy", &vy_}, {"vz", &vz_}};
    for (const auto& [name, values] : arrays) {
        out += ",\n  \"";
        out += name;
        out += "\": ";
        appendArray(out, *values);
    }
    out += "\n}\n";
    return out;
}

void VectorFigure3D::saveJson(const std::filesystem::path& path) const {
    writeTextFile(path, toJson(), "VectorFigure3D::saveJson");
}

bool VectorFigure3D::render(const std::filesystem::path& dataFile, const RenderOptions& options) const {
    if (x_.empty()) {
        throw std::invalid_argument("VectorFigure3D::render: no vectors, call setVectors or setFunction");
    }
    return renderJson(toJson(), dataFile, options, "sem::io::VectorFigure3D::render");
}

// ---------------------------------------------------------------------------
// ScalarFigure3D
// ---------------------------------------------------------------------------

ScalarFigure3D::ScalarFigure3D(std::string title) : title_(std::move(title)) {}

ScalarFigure3D& ScalarFigure3D::setTitle(std::string title) {
    title_ = std::move(title);
    return *this;
}

ScalarFigure3D& ScalarFigure3D::setXLabel(std::string label) {
    xLabel_ = std::move(label);
    return *this;
}

ScalarFigure3D& ScalarFigure3D::setYLabel(std::string label) {
    yLabel_ = std::move(label);
    return *this;
}

ScalarFigure3D& ScalarFigure3D::setZLabel(std::string label) {
    zLabel_ = std::move(label);
    return *this;
}

ScalarFigure3D& ScalarFigure3D::setColorLabel(std::string label) {
    colorLabel_ = std::move(label);
    return *this;
}

ScalarFigure3D& ScalarFigure3D::setStyle(Scalar3DStyle style) {
    style_ = style;
    return *this;
}

ScalarFigure3D& ScalarFigure3D::setField(std::span<const Real> x, std::span<const Real> y,
                                         std::span<const Real> z, std::span<const Real> values) {
    if (x.size() < 2 || y.size() < 2 || z.size() < 2) {
        throw std::invalid_argument("ScalarFigure3D::setField: at least 2 points per direction");
    }
    if (values.size() != x.size() * y.size() * z.size()) {
        throw std::invalid_argument(
            "ScalarFigure3D::setField: values must hold x.size() * y.size() * z.size() numbers");
    }
    x_.assign(x.begin(), x.end());
    y_.assign(y.begin(), y.end());
    z_.assign(z.begin(), z.end());
    values_.assign(values.begin(), values.end());
    return *this;
}

std::string ScalarFigure3D::toJson() const {
    std::string out = "{\n  \"kind\": \"scalar3d\",\n  \"title\": ";
    appendString(out, title_);
    out += ",\n  \"xlabel\": ";
    appendString(out, xLabel_);
    out += ",\n  \"ylabel\": ";
    appendString(out, yLabel_);
    out += ",\n  \"zlabel\": ";
    appendString(out, zLabel_);
    out += ",\n  \"clabel\": ";
    appendString(out, colorLabel_);
    out += ",\n  \"style\": ";
    appendString(out, style_ == Scalar3DStyle::Scatter ? "scatter" : "faces");
    out += ",\n  \"x\": ";
    appendArray(out, x_);
    out += ",\n  \"y\": ";
    appendArray(out, y_);
    out += ",\n  \"z\": ";
    appendArray(out, z_);
    out += ",\n  \"values\": ";
    appendArray(out, values_);
    out += "\n}\n";
    return out;
}

void ScalarFigure3D::saveJson(const std::filesystem::path& path) const {
    writeTextFile(path, toJson(), "ScalarFigure3D::saveJson");
}

bool ScalarFigure3D::render(const std::filesystem::path& dataFile, const RenderOptions& options) const {
    if (values_.empty()) {
        throw std::invalid_argument("ScalarFigure3D::render: no data, call setField or setFunction");
    }
    return renderJson(toJson(), dataFile, options, "sem::io::ScalarFigure3D::render");
}

bool renderAll(std::span<const PlotItem> items, bool showWindow) {
    return renderItems(items, showWindow, "sem::io::renderAll");
}

} // namespace sem::io
