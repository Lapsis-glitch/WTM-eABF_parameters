"""Reusable grouped and seed-resolved RMSD analyses."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .analyze_ND import PMFAnalyzer
    from .input_discovery import RunRecord, records_from_inputs
    from .outputs import analysis_output, safe_name
    from .plotting import PlotConfig, add_plotting_arguments, close_figure, flatten_axes
    from .plotting import make_figure, publication_style, save_figure
except ImportError:
    from analyze_ND import PMFAnalyzer
    from input_discovery import RunRecord, records_from_inputs
    from outputs import analysis_output, safe_name
    from plotting import PlotConfig, add_plotting_arguments, close_figure, flatten_axes
    from plotting import make_figure, publication_style, save_figure


LOGGER = logging.getLogger(__name__)
LINESTYLES = ["-", (0, (5, 1)), (0, (5, 5)), (0, (5, 10)), (0, (1, 1)), (0, (1, 5))]


def add_input_arguments(parser):
    parser.add_argument("root", nargs="?", help="Input root for recursive discovery")
    parser.add_argument("--manifest")
    parser.add_argument("--pmf-file")
    parser.add_argument("--count-file")
    parser.add_argument("--pmf-pattern", default="**/*czar.pmf")
    parser.add_argument("--count-pattern")
    parser.add_argument("--metadata-regex")
    parser.add_argument("--reference-pmf")
    parser.add_argument("--n-recent", type=int, default=4)
    parser.add_argument("--slope-thresh", type=float, default=1e-3)
    parser.add_argument("--rmsd-thresh", type=float, default=0.01)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--divisor", type=float, default=1.0)
    add_plotting_arguments(parser)


def _records(args):
    if not args.root and not args.manifest and not args.pmf_file:
        raise ValueError("provide root, --manifest, or --pmf-file")
    result = records_from_inputs(root=args.root if not args.manifest and not args.pmf_file else None,
                                 manifest=args.manifest, pmf_file=args.pmf_file,
                                 count_file=args.count_file, pmf_pattern=args.pmf_pattern,
                                 count_pattern=args.count_pattern, metadata_regex=args.metadata_regex)
    if not result.runs:
        raise RuntimeError("No usable PMF/count pairs were discovered")
    return result.runs


def _analyzers(records, args):
    items = []
    for record in records:
        try:
            analyzer = PMFAnalyzer(record.pmf_file, record.count_file,
                                   n_recent=args.n_recent, slope_thresh=args.slope_thresh,
                                   reference_pmf_file=args.reference_pmf,
                                   rmsd_thresh=args.rmsd_thresh)
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.warning("Skipping run %s (PMF=%s, count=%s): %s",
                           record.run_id, record.pmf_file, record.count_file, exc)
            continue
        if len(analyzer.rmsd_raw) < args.min_frames:
            LOGGER.warning("Skipping run %s: only %d RMSD frames", record.run_id, len(analyzer.rmsd_raw))
            continue
        items.append((record, analyzer))
    if not items:
        raise RuntimeError("No usable runs remained for RMSD analysis")
    return items


def _parameter_label(record: RunRecord):
    return record.parameter_value() if record.parameter_values else record.run_id


def _sort_value(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _write_run_table(items, output):
    path = output.directory / "rmsd_runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "pmf_file", "count_file", "group", "seed", "parameter_values", "convergence_idx"])
        for record, analyzer in items:
            writer.writerow([record.run_id, record.pmf_file, record.count_file or "", record.group or "",
                             record.seed if record.seed is not None else "", record.parameter_values,
                             analyzer.convergence_idx if analyzer.convergence_idx is not None else ""])


def plot_group_panel(ax, entries, divisor=1.0):
    for record, analyzer in sorted(entries, key=lambda item: _sort_value(_parameter_label(item[0]))):
        ax.plot(analyzer.t / divisor, analyzer.rmsd_raw, label=str(_parameter_label(record)), linewidth=1.5)
    group = entries[0][0].group or "RMSD"
    ax.set(title=str(group), xlabel="Time (ns)" if divisor != 1 else "Snapshot Index", ylabel="RMSD")
    ax.legend(loc="best")
    ax.grid(True)
    return ax


def plot_seed_panel(ax, entries, divisor=1.0):
    by_value = defaultdict(list)
    for record, analyzer in entries:
        by_value[_parameter_label(record)].append((record, analyzer))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = []
    for color_index, value in enumerate(sorted(by_value, key=_sort_value)):
        color = colors[color_index % len(colors)]
        seed_entries = sorted(by_value[value], key=lambda item: _sort_value(item[0].seed))
        for seed_index, (record, analyzer) in enumerate(seed_entries):
            ax.plot(analyzer.t / divisor, analyzer.rmsd_raw, color=color,
                    linestyle=LINESTYLES[seed_index % len(LINESTYLES)], linewidth=1.5)
        handles.append(Line2D([0], [0], color=color, lw=2, label=str(value)))
    group = entries[0][0].group or "RMSD by seed"
    ax.set(title=str(group), xlabel="Time (ns)" if divisor != 1 else "Snapshot Index", ylabel="RMSD")
    ax.legend(handles=handles, loc="best")
    ax.grid(True)
    return ax


def run(items, *, output_root="Results", analysis_name="rmsd_curves", config=None,
        seed_mode=False, divisor=1.0):
    config = config or PlotConfig(output_root=str(output_root))
    output = analysis_output(output_root, analysis_name)
    grouped = defaultdict(list)
    for record, analyzer in items:
        grouped[record.group or record.run_id].append((record, analyzer))
    groups = sorted(grouped.items(), key=lambda item: str(item[0]))
    renderer = plot_seed_panel if seed_mode else plot_group_panel
    with publication_style(config):
        cols = min(3, max(1, math.ceil(math.sqrt(len(groups)))))
        rows = math.ceil(len(groups) / cols)
        fig, axes = make_figure(config, kind="multipanel", nrows=rows, ncols=cols, sharex=True)
        axes_list = flatten_axes(axes)
        for axis, (_, entries) in zip(axes_list, groups):
            renderer(axis, entries, divisor=divisor)
        for axis in axes_list[len(groups):]:
            axis.set_visible(False)
        save_figure(fig, output.figures / f"{safe_name(analysis_name)}_multipanel", config)
        close_figure(fig)
        for group, entries in groups:
            panel_fig, panel_ax = make_figure(config, kind="panel")
            renderer(panel_ax, entries, divisor=divisor)
            save_figure(panel_fig, output.panels / safe_name(str(group)), config)
            close_figure(panel_fig)
    _write_run_table(items, output)
    return output.directory


def cli(seed_mode=False):
    parser = argparse.ArgumentParser(description="Grouped RMSD comparison analysis")
    add_input_arguments(parser)
    args = parser.parse_args()
    config = PlotConfig(publisher=args.publisher, multipanel_target=args.multipanel_target,
                        multipanel_fraction=args.multipanel_fraction, panel_target=args.panel_target,
                        panel_fraction=args.panel_fraction, formats=tuple(args.figure_formats),
                        dpi=args.dpi, output_root=args.output_root)
    items = _analyzers(_records(args), args)
    run(items, output_root=args.output_root,
        analysis_name="rmsd_seed_curves" if seed_mode else "rmsd_curves",
        config=config, seed_mode=seed_mode, divisor=args.divisor)
