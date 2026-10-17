#!/usr/bin/env python3
"""Рисует графики по JSON-файлам, которые записывают классы из sem/io/plot.hpp.

Использование:
    python tools/plot.py data.json                       # открыть окно
    python tools/plot.py data.json -o plot.png           # сохранить и открыть окно
    python tools/plot.py data.json -o plot.png --no-show # только сохранить
    python tools/plot.py a.json b.json -o a.png -o -     # два окна сразу; b не сохранять

Формат линейного графика (Figure):
    {
      "title": "...", "xlabel": "...", "ylabel": "...",
      "xscale": "linear" | "log", "yscale": "linear" | "log",
      "series": [{"label": "...", "style": "o-", "x": [...], "y": [...]}, ...]
    }

Формат поля (Figure2D):
    {
      "kind": "field2d",
      "title": "...", "xlabel": "...", "ylabel": "...", "zlabel": "...",
      "style": "heatmap" | "contour" | "surface", "zscale": "linear" | "log",
      "label": "...",                       (подпись основного поля в легенде)
      "x": [nx значений], "y": [ny значений],
      "z": [nx * ny значений, z[i + nx * j] — в точке (x[i], y[j])],
      "overlays": [{"label": "...", "x": [...], "y": [...], "z": [...]}, ...],
      "points": [{"label": "...", "style": "k.", "x": [...], "y": [...]}, ...]
    }

Формат векторного поля в 3D (VectorFigure3D):
    {
      "kind": "vector3d",
      "title": "...", "xlabel": "...", "ylabel": "...", "zlabel": "...", "clabel": "...",
      "arrow_length": 0.2,                  (длина самой длинной стрелки; <= 0 — автоматически)
      "normalize": false,                   (true — все стрелки одной длины)
      "x": [...], "y": [...], "z": [...],   (точки)
      "vx": [...], "vy": [...], "vz": [...] (векторы)
    }

Формат скалярного поля в 3D (ScalarFigure3D):
    {
      "kind": "scalar3d",
      "title": "...", "xlabel": "...", "ylabel": "...", "zlabel": "...", "clabel": "...",
      "style": "faces" | "scatter",
      "x": [nx], "y": [ny], "z": [nz],
      "values": [nx * ny * nz значений, values[i + nx * (j + ny * k)] — в точке (x[i], y[j], z[k])]
    }

"""

import argparse
import json
import sys
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def draw_lines(data, fig):
    ax = fig.add_subplot()
    has_labels = False
    for series in data.get("series", []):
        label = series.get("label") or None
        has_labels = has_labels or label is not None
        ax.plot(series["x"], series["y"], series.get("style") or "-", label=label)

    ax.set_title(data.get("title", ""))
    ax.set_xlabel(data.get("xlabel", ""))
    ax.set_ylabel(data.get("ylabel", ""))
    ax.set_xscale(data.get("xscale", "linear"))
    ax.set_yscale(data.get("yscale", "linear"))
    ax.grid(True, which="major", alpha=0.3)
    if has_labels:
        ax.legend()


OVERLAY_COLORS = ["tab:red", "black", "tab:orange", "tab:pink"]


def grid(field):
    """Сетка и значения поля: z[i + nx * j] -> массив (ny, nx), строка j — при y = y[j]."""
    import numpy as np

    x = np.asarray(field["x"], dtype=float)
    y = np.asarray(field["y"], dtype=float)
    z = np.asarray(field["z"], dtype=float).reshape(len(y), len(x))
    X, Y = np.meshgrid(x, y)
    return X, Y, z


def draw_field(data, fig):
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    X, Y, z = grid(data)
    overlays = data.get("overlays", [])

    style = data.get("style", "heatmap")
    log = data.get("zscale", "linear") == "log"
    zlabel = data.get("zlabel", "")
    cmap = "viridis"

    def log10_or_nan(values):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)

    if style == "surface":
        ax = fig.add_subplot(projection="3d")
        if log:
            z = log10_or_nan(z)
            zlabel = f"log10({zlabel})" if zlabel else "log10(z)"
        main_label = data.get("label") or None
        mappable = ax.plot_surface(X, Y, z, cmap=cmap, linewidth=0, antialiased=True,
                                   alpha=0.85 if overlays else 1.0, label=main_label)

        # Дополнительные поля — каркасом примерно из 20 линий по каждому направлению.
        has_labels = main_label is not None
        for k, overlay in enumerate(overlays):
            OX, OY, oz = grid(overlay)
            if log:
                oz = log10_or_nan(oz)
            label = overlay.get("label") or None
            has_labels = has_labels or label is not None
            ax.plot_wireframe(OX, OY, oz, color=OVERLAY_COLORS[k % len(OVERLAY_COLORS)],
                              linewidth=0.7, rstride=max(1, oz.shape[0] // 20),
                              cstride=max(1, oz.shape[1] // 20), label=label)
        if has_labels:
            ax.legend(loc="upper left")
        ax.set_zlabel(zlabel)
    else:
        ax = fig.add_subplot()
        if log:
            z = np.ma.masked_where(~(z > 0), z)  # ноль и отрицательные не рисуются
        if style == "contour":
            if log:
                # Изолинии по log10: у contourf с LogNorm шкала остаётся без подписей.
                z = np.ma.log10(z)
                zlabel = f"log10({zlabel})" if zlabel else "log10(z)"
            mappable = ax.contourf(X, Y, z, levels=30, cmap=cmap)
        else:
            norm = LogNorm(vmin=z.min(), vmax=z.max()) if log and z.count() > 0 else None
            mappable = ax.pcolormesh(X, Y, z, shading="auto", norm=norm, cmap=cmap)

        # Дополнительные поля — линиями уровня; в легенду — через линию-заместитель.
        handles = []
        for k, overlay in enumerate(overlays):
            OX, OY, oz = grid(overlay)
            if log:
                oz = log10_or_nan(oz) if style == "contour" else np.ma.masked_where(~(oz > 0), oz)
            color = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
            ax.contour(OX, OY, oz, levels=10, colors=color, linewidths=0.9,
                       norm=None if style == "contour" or not log else mappable.norm)
            if overlay.get("label"):
                handles.append(Line2D([], [], color=color, linewidth=0.9, label=overlay["label"]))

        for points in data.get("points", []):
            (line,) = ax.plot(points["x"], points["y"], points.get("style") or "k.",
                              label=points.get("label") or None)
            if points.get("label"):
                handles.append(line)
        if handles:
            ax.legend(handles=handles, loc="upper right", framealpha=0.8)
        ax.set_aspect("equal")

    fig.colorbar(mappable, ax=ax, label=zlabel, shrink=0.8)
    ax.set_title(data.get("title", ""))
    ax.set_xlabel(data.get("xlabel", ""))
    ax.set_ylabel(data.get("ylabel", ""))


def draw_vector3d(data, fig):
    """Стрелки в 3D: цвет — модуль вектора, длина — пропорциональна модулю (или одинаковая)."""
    import numpy as np
    from matplotlib import cm, colormaps
    from matplotlib.colors import Normalize

    x, y, z, u, v, w = (np.asarray(data[key], dtype=float) for key in ("x", "y", "z", "vx", "vy", "vz"))
    magnitude = np.sqrt(u * u + v * v + w * w)

    length = float(data.get("arrow_length", 0) or 0)
    if length <= 0:  # автоматически: десятая часть наибольшего размера области
        length = 0.1 * max(np.ptp(x), np.ptp(y), np.ptp(z), 1e-12)

    if data.get("normalize", False):
        scale = np.where(magnitude > 0, length / np.where(magnitude > 0, magnitude, 1), 0)
    else:
        scale = np.full_like(magnitude, length / magnitude.max() if magnitude.max() > 0 else 0)

    ax = fig.add_subplot(projection="3d")
    norm = Normalize(vmin=magnitude.min(), vmax=magnitude.max())
    cmap = colormaps["viridis"]
    colors = cmap(norm(magnitude))
    # У каждой стрелки 3D quiver три отрезка: стержень и две половинки наконечника.
    # matplotlib складывает их блоками — все стержни, все левые половинки, все правые
    # (lines = [*shafts, *heads[::2], *heads[1::2]]), поэтому цвета повторяются блоками.
    colors = np.concatenate((colors, colors, colors))
    ax.quiver(x, y, z, u * scale, v * scale, w * scale, length=1.0, normalize=False,
              colors=colors, linewidth=1.2, arrow_length_ratio=0.3)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7,
                 label=data.get("clabel", ""))
    ax.set_title(data.get("title", ""))
    ax.set_xlabel(data.get("xlabel", ""))
    ax.set_ylabel(data.get("ylabel", ""))
    ax.set_zlabel(data.get("zlabel", ""))
    ax.set_box_aspect((np.ptp(x) or 1, np.ptp(y) or 1, np.ptp(z) or 1))


def draw_scalar3d(data, fig):
    """Скалярное поле на регулярной сетке в 3D: раскрашенные грани или облако точек."""
    import numpy as np
    from matplotlib import cm, colormaps
    from matplotlib.colors import Normalize

    x, y, z = (np.asarray(data[key], dtype=float) for key in ("x", "y", "z"))
    # values[i + nx * (j + ny * k)] -> массив (nz, ny, nx)
    v = np.asarray(data["values"], dtype=float).reshape(len(z), len(y), len(x))

    norm = Normalize(vmin=v.min(), vmax=v.max())
    cmap = colormaps["viridis"]
    ax = fig.add_subplot(projection="3d")

    if data.get("style") == "scatter":
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=v.ravel(), cmap=cmap, norm=norm,
                   s=10, alpha=0.5, depthshade=False, linewidths=0)
    else:
        # Грани, видимые с ракурса matplotlib по умолчанию (elev=30, azim=-60):
        # x = max (справа), y = min (спереди), z = max (сверху). Внутренних плоскостей нет,
        # поэтому нет и артефактов перекрытия.
        common = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
        YY, ZZ = np.meshgrid(y, z)
        ax.plot_surface(np.full_like(YY, x[-1]), YY, ZZ, facecolors=cmap(norm(v[:, :, -1])), **common)
        XX, ZZ = np.meshgrid(x, z)
        ax.plot_surface(XX, np.full_like(XX, y[0]), ZZ, facecolors=cmap(norm(v[:, 0, :])), **common)
        XX, YY = np.meshgrid(x, y)
        ax.plot_surface(XX, YY, np.full_like(XX, z[-1]), facecolors=cmap(norm(v[-1, :, :])), **common)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7,
                 label=data.get("clabel", ""))
    ax.set_title(data.get("title", ""))
    ax.set_xlabel(data.get("xlabel", ""))
    ax.set_ylabel(data.get("ylabel", ""))
    ax.set_zlabel(data.get("zlabel", ""))
    ax.set_box_aspect((np.ptp(x) or 1, np.ptp(y) or 1, np.ptp(z) or 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw sem::io::Figure / Figure2D JSON files.")
    parser.add_argument("data", type=Path, nargs="+",
                        help="JSON files written by sem::io::Figure or Figure2D")
    parser.add_argument("-o", "--output", type=Path, action="append", default=[],
                        help="image for the corresponding data file (in the same order); "
                             "'-' means do not save")
    parser.add_argument("--no-show", action="store_true", help="do not open windows")
    args = parser.parse_args(argv)

    outputs = args.output
    if outputs and len(outputs) != len(args.data):
        parser.error("give either no --output or one --output per data file")
    outputs = [None if out is None or str(out) == "-" else out for out in outputs] or [None] * len(args.data)
    if args.no_show and all(out is None for out in outputs):
        parser.error("nothing to do: use --output or allow the window")

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")  # без окна: работает и без графической среды
    import matplotlib.pyplot as plt

    # Каждый файл — отдельная фигура; plt.show() в конце открывает все окна сразу.
    for path, output in zip(args.data, outputs):
        data = load(path)
        if data.get("kind") == "field2d":
            fig = plt.figure(figsize=(8, 6.5), layout="constrained")
            draw_field(data, fig)
        elif data.get("kind") == "vector3d":
            fig = plt.figure(figsize=(8, 7), layout="constrained")
            draw_vector3d(data, fig)
        elif data.get("kind") == "scalar3d":
            fig = plt.figure(figsize=(8, 7), layout="constrained")
            draw_scalar3d(data, fig)
        else:
            fig = plt.figure(figsize=(8, 5), layout="constrained")
            draw_lines(data, fig)
        if output is not None:
            fig.savefig(output, dpi=150)
            print(f"Saved {output}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
