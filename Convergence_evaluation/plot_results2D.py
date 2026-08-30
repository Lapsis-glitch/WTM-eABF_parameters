#!/usr/bin/env python3
"""Render mean and standard-deviation convergence surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .convergence_summary import read_summary
    from .outputs import analysis_output
    from .plotting import PlotConfig, add_plotting_arguments, close_figure, flatten_axes
    from .plotting import make_figure, publication_style, save_figure
except ImportError:
    from convergence_summary import read_summary
    from outputs import analysis_output
    from plotting import PlotConfig, add_plotting_arguments, close_figure, flatten_axes
    from plotting import make_figure, publication_style, save_figure


def _surface(ax, rows, x_name, y_name, field, divisor, cmap, label):
    points = []
    for row in rows:
        parameters = row.get("parameter_values") or {}
        if x_name in parameters and y_name in parameters:
            points.append((float(parameters[x_name]), float(parameters[y_name]), float(row[field]) / divisor))
    if not points:
        raise RuntimeError(f"no rows contain selected parameters {x_name!r} and {y_name!r}")
    x_values = np.array(sorted({point[0] for point in points}))
    y_values = np.array(sorted({point[1] for point in points}))
    grid = np.full((len(y_values), len(x_values)), np.nan)
    for x, y, value in points:
        grid[np.where(y_values == y)[0][0], np.where(x_values == x)[0][0]] = value
    X, Y = np.meshgrid(x_values, y_values)
    if len(x_values) > 1 and len(y_values) > 1:
        mappable = ax.contourf(X, Y, grid, levels=40, cmap=cmap)
    else:
        mappable = ax.scatter([point[0] for point in points], [point[1] for point in points],
                              c=[point[2] for point in points], cmap=cmap)
    ax.set(xlabel=x_name, ylabel=y_name, title=label)
    ax.figure.canvas.draw_idle()
    import pubready as pr
    pr.add_colorbar(ax.figure, mappable, ax=ax, location="right", label=label)
    return ax


def plot_surface(rows, x_name, y_name, *, output_root="Results", config=None, divisor=20.0):
    config = config or PlotConfig(output_root=str(output_root))
    output = analysis_output(output_root, "convergence_surface")
    renderers = [("mean_convergence", "mean", "viridis", "Mean Convergence Time (ns)"),
                 ("standard_deviation", "std", "magma", "Standard Deviation (ns)")]
    with publication_style(config):
        fig, axes = make_figure(config, kind="multipanel", ncols=2)
        for axis, (_, field, cmap, label) in zip(flatten_axes(axes), renderers):
            _surface(axis, rows, x_name, y_name, field, divisor, cmap, label)
        save_figure(fig, output.figures / "convergence_surface_multipanel", config)
        close_figure(fig)
        for name, field, cmap, label in renderers:
            panel_fig, panel_ax = make_figure(config, kind="panel")
            _surface(panel_ax, rows, x_name, y_name, field, divisor, cmap, label)
            save_figure(panel_fig, output.panels / name, config)
            close_figure(panel_fig)
    return output.directory


def main():
    parser = argparse.ArgumentParser(description="Plot 2D convergence surfaces")
    parser.add_argument("file", help="Structured convergence_summary.csv")
    parser.add_argument("--x-parameter")
    parser.add_argument("--y-parameter")
    parser.add_argument("--divisor", type=float, default=20.0)
    add_plotting_arguments(parser)
    args = parser.parse_args()
    config = PlotConfig(publisher=args.publisher, multipanel_target=args.multipanel_target,
                        multipanel_fraction=args.multipanel_fraction, panel_target=args.panel_target,
                        panel_fraction=args.panel_fraction, formats=tuple(args.figure_formats),
                        dpi=args.dpi, output_root=args.output_root)
    rows = read_summary(args.file)
    names = sorted({name for row in rows for name in (row.get("parameter_values") or {})})
    if args.x_parameter is None or args.y_parameter is None:
        if len(names) != 2:
            parser.error("select --x-parameter and --y-parameter when the summary does not contain exactly two parameters")
        args.x_parameter, args.y_parameter = names
    plot_surface(rows, args.x_parameter, args.y_parameter,
                 output_root=args.output_root, config=config, divisor=args.divisor)


if __name__ == "__main__":
    main()
