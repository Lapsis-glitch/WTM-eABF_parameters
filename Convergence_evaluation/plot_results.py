#!/usr/bin/env python3
"""Plot structured convergence summaries with reusable PubReady panels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from .convergence_summary import read_summary
    from .outputs import analysis_output
    from .plotting import (PlotConfig, add_plotting_arguments, close_figure,
                           finalize_multipanel_layout, flatten_axes, multipanel_grid,
                           make_figure, publication_style, save_figure, set_shared_labels)
except ImportError:
    from convergence_summary import read_summary
    from outputs import analysis_output
    from plotting import (PlotConfig, add_plotting_arguments, close_figure,
                          finalize_multipanel_layout, flatten_axes, multipanel_grid,
                          make_figure, publication_style, save_figure, set_shared_labels)


LABELS = {"MTDheight": "hillWeight", "MTDnewhill": "newHillFrequency",
          "MTDtemp": "biasTemperature", "MTDwidth": "hillWidth",
          "colvarWidth": "colvarWidth", "extDamp": "extendedLangevinDamping",
          "extFluc": "extendedFluctuation", "extTime": "extendedTimeConstant",
          "fullSamp": "fullSamples"}


def _read_legacy(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 6:
                continue
            name, mean, std, minimum, maximum, n = fields[:6]
            pieces = name.rsplit("_", 1)
            if len(pieces) != 2:
                continue
            try:
                value = float(pieces[1])
                rows.append({"group": pieces[0], "parameter_name": pieces[0],
                             "parameter_value": value, "parameter_values": {pieces[0]: value},
                             "mean": float(mean), "std": float(std), "min": float(minimum),
                             "max": float(maximum), "n": int(n)})
            except ValueError:
                continue
    return rows


def load_rows(path):
    return read_summary(path) if Path(path).suffix.lower() == ".csv" else _read_legacy(path)


def _value_sort(value):
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _plain_tick_label(value, _position):
    if value == 0:
        return "0"
    label = format(Decimal(str(value)), "f")
    return label.rstrip("0").rstrip(".") if "." in label else label


def plot_summary_panel(ax, rows, parameter_name, divisor=20.0, show_ylabel=True):
    grouped = defaultdict(list)
    for row in rows:
        params = row.get("parameter_values") or {}
        if parameter_name not in params:
            continue
        grouped[str(row.get("group", "all"))].append(row)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color_index, (group, group_rows) in enumerate(sorted(grouped.items())):
        valid = sorted(group_rows, key=lambda row: _value_sort(row["parameter_values"][parameter_name]))
        x = [row["parameter_values"][parameter_name] for row in valid]
        mean = [row["mean"] / divisor for row in valid]
        std = [row["std"] / divisor for row in valid]
        minimum = [row["min"] / divisor for row in valid]
        maximum = [row["max"] / divisor for row in valid]
        color = color_cycle[color_index % len(color_cycle)]
        label = group if len(grouped) > 1 else None
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color=color, alpha=0.2)
        ax.fill_between(x, minimum, maximum, color=color, alpha=0.1)
    ax.set_xlabel(LABELS.get(parameter_name, parameter_name))
    if show_ylabel:
        ax.set_ylabel("Convergence (ns)")
    else:
        ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
    ax.grid(True, color="lightgray")
    if len(grouped) > 1:
        ax.legend(loc="best")
    return ax


def plot_summary(rows, *, output_root="Results", config=None, divisor=20.0):
    config = config or PlotConfig(output_root=str(output_root))
    names = sorted({name for row in rows for name in (row.get("parameter_values") or {})})
    if not names:
        raise RuntimeError("summary contains no parameter metadata; provide a manifest with parameter columns")
    output = analysis_output(output_root, "convergence_summary")
    with publication_style(config):
        rows_count, cols = multipanel_grid(len(names), config)
        fig, axes = make_figure(config, kind="multipanel", nrows=rows_count, ncols=cols, sharey=True)
        axes_list = flatten_axes(axes)
        for axis, name in zip(axes_list, names):
            plot_summary_panel(axis, rows, name, divisor=divisor, show_ylabel=False)
        for axis in axes_list[len(names):]:
            axis.set_visible(False)
        set_shared_labels(fig, ylabel="Convergence (ns)")
        finalize_multipanel_layout(fig)
        save_figure(fig, output.figures / "convergence_summary_multipanel", config, fit=False)
        close_figure(fig)
        for name in names:
            panel_fig, panel_ax = make_figure(config, kind="panel")
            plot_summary_panel(panel_ax, rows, name, divisor=divisor)
            save_figure(panel_fig, output.panels / f"{name}_convergence", config)
            close_figure(panel_fig)
    return output.directory


def main():
    parser = argparse.ArgumentParser(description="Plot mean convergence with standard deviation")
    parser.add_argument("file", help="CSV summary or legacy results.dat")
    parser.add_argument("--divisor", type=float, default=20.0)
    add_plotting_arguments(parser)
    args = parser.parse_args()
    config = PlotConfig(publisher=args.publisher, multipanel_target=args.multipanel_target,
                        multipanel_fraction=args.multipanel_fraction, panel_target=args.panel_target,
                        panel_fraction=args.panel_fraction, formats=tuple(args.figure_formats),
                        dpi=args.dpi, output_root=args.output_root)
    plot_summary(load_rows(args.file), output_root=args.output_root, config=config, divisor=args.divisor)


if __name__ == "__main__":
    main()
