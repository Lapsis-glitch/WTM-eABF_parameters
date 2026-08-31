"""Reusable grouped and seed-resolved RMSD analyses."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from .analyze_ND import PMFAnalyzer
    from .input_discovery import RunRecord, records_from_inputs
    from .outputs import analysis_output, safe_name
    from .plotting import (PlotConfig, add_plotting_arguments, close_figure,
                           finalize_multipanel_layout, flatten_axes, multipanel_grid,
                           make_figure, publication_style, save_figure, set_shared_labels)
except ImportError:
    from analyze_ND import PMFAnalyzer
    from input_discovery import RunRecord, records_from_inputs
    from outputs import analysis_output, safe_name
    from plotting import (PlotConfig, add_plotting_arguments, close_figure,
                          finalize_multipanel_layout, flatten_axes, multipanel_grid,
                          make_figure, publication_style, save_figure, set_shared_labels)


LOGGER = logging.getLogger(__name__)
LINESTYLES = ["-", (0, (5, 1)), (0, (5, 5)), (0, (5, 10)), (0, (1, 1)), (0, (1, 5))]
RMSD_SERIES_FILENAME = "rmsd_series.csv"
RMSD_METADATA_FILENAME = "rmsd_plot_metadata.json"


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
    parser.add_argument("--temperature", type=float, default=298.0,
                        help="temperature in K recorded with the RMSD plotting data")
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


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _reference_payload(analyzer):
    reference = getattr(analyzer, "reference_pmf", None)
    if reference is None:
        return ""
    coords = [np.asarray(axis).tolist() for axis in getattr(analyzer, "pmf_coords", ())]
    return json.dumps({"coords": coords, "values": np.asarray(reference).tolist()},
                      default=_json_default, separators=(",", ":"))


def _as_csv_number(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(value)


def _effective_threshold(items, threshold=None):
    if threshold is not None:
        return float(threshold)
    for _, analyzer in items:
        value = getattr(analyzer, "rmsd_thresh", None)
        if value is not None:
            return float(value)
    return None


def _write_rmsd_series(items, output, *, divisor, temperature_k, kbt_threshold):
    """Write complete per-run RMSD series for later plot-only rendering."""
    threshold = _effective_threshold(items, kbt_threshold)
    series_path = output.directory / RMSD_SERIES_FILENAME
    metadata_path = output.directory / RMSD_METADATA_FILENAME
    columns = ["run_id", "group", "seed", "parameter_values", "snapshot_index", "time",
               "raw_rmsd", "smoothed_rmsd", "fitted_rmsd", "reference_pmf_used",
               "kbt_threshold", "temperature_K", "convergence_index", "convergence_time"]
    with series_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record, analyzer in items:
            times = np.asarray(analyzer.t)
            raw = np.asarray(analyzer.rmsd_raw)
            smooth = np.asarray(getattr(analyzer, "rmsd_smooth", np.full(raw.shape, np.nan)))
            fitted = np.asarray(getattr(analyzer, "rmsd_fit", np.full(raw.shape, np.nan)))
            reference = _reference_payload(analyzer)
            convergence_index = getattr(analyzer, "convergence_idx", None)
            convergence_time = (float(convergence_index) / float(divisor)
                                if convergence_index is not None else None)
            for index, snapshot in enumerate(times):
                writer.writerow({
                    "run_id": record.run_id,
                    "group": record.group or record.run_id,
                    "seed": "" if record.seed is None else record.seed,
                    "parameter_values": json.dumps(record.parameter_values,
                                                   default=_json_default, sort_keys=True),
                    "snapshot_index": _as_csv_number(snapshot),
                    "time": _as_csv_number(float(snapshot) / float(divisor)),
                    "raw_rmsd": _as_csv_number(raw[index]),
                    "smoothed_rmsd": _as_csv_number(smooth[index]),
                    "fitted_rmsd": _as_csv_number(fitted[index]),
                    "reference_pmf_used": reference,
                    "kbt_threshold": _as_csv_number(threshold),
                    "temperature_K": _as_csv_number(temperature_k),
                    "convergence_index": _as_csv_number(convergence_index),
                    "convergence_time": _as_csv_number(convergence_time),
                })
    metadata = {
        "format_version": 1,
        "analysis_name": output.name,
        "seed_mode": output.name == "rmsd_seed_curves",
        "temperature_K": float(temperature_k),
        "temperature_units": "K",
        "kbt_threshold": threshold,
        "kbt_units": "kcal/mol",
        "time_divisor": float(divisor),
        "time_units": "ns",
        "time_label": "Time (ns)" if float(divisor) != 1 else "Snapshot Index",
        "columns": columns,
        "reference_pmf_encoding": "compact JSON object with coords and values in each row",
        "source_runs": [
            {"run_id": record.run_id,
             "reference_pmf_file": getattr(analyzer, "reference_pmf_file", None)}
            for record, analyzer in items
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True,
                                        default=_json_default) + "\n")
    return series_path, metadata_path


def load_rmsd_series(series_path, metadata_path=None):
    """Deserialize saved RMSD plotting data without touching source histories."""
    series_path = Path(series_path)
    metadata_path = (Path(metadata_path) if metadata_path is not None else
                     series_path.with_name(RMSD_METADATA_FILENAME))
    metadata = json.loads(metadata_path.read_text())
    grouped = {}
    with series_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = row["run_id"]
            item = grouped.setdefault(run_id, {"row": row, "rows": []})
            item["rows"].append(row)
    items = []
    for run_id, item in grouped.items():
        first = item["row"]
        rows = item["rows"]
        parameter_values = json.loads(first["parameter_values"] or "{}")
        reference = json.loads(first["reference_pmf_used"]) if first["reference_pmf_used"] else None
        convergence_index = (float(first["convergence_index"])
                             if first["convergence_index"] else None)
        threshold = float(first["kbt_threshold"]) if first["kbt_threshold"] else None
        record = SimpleNamespace(run_id=run_id, group=first["group"] or None,
                                 seed=first["seed"] or None, parameter_values=parameter_values)
        record.parameter_value = lambda values=parameter_values: next(iter(values.values()), None)
        analyzer = SimpleNamespace(
            t=np.asarray([float(row["snapshot_index"]) for row in rows]),
            rmsd_raw=np.asarray([float(row["raw_rmsd"]) for row in rows]),
            rmsd_smooth=np.asarray([float(row["smoothed_rmsd"]) for row in rows]),
            rmsd_fit=np.asarray([float(row["fitted_rmsd"]) for row in rows]),
            reference_pmf=None if reference is None else np.asarray(reference["values"]),
            reference_pmf_file=None,
            rmsd_thresh=threshold,
            convergence_idx=convergence_index,
        )
        items.append((record, analyzer))
    if not items:
        raise ValueError(f"RMSD series contains no runs: {series_path}")
    return items, metadata


def _rmsd_legend_kwargs():
    return {"loc": "best", "ncols": 2,
            "handlelength": plt.rcParams["legend.handlelength"] / 3.0}


def _plot_rmsd_guides(ax, entries, divisor, kbt_threshold=None):
    threshold = _effective_threshold(entries, kbt_threshold)
    if threshold is not None:
        ax.axhline(threshold, color="C3", linestyle=":",
                   label=f"kBT threshold ({threshold:g} kcal/mol)")
    convergence = sorted({float(analyzer.convergence_idx) for _, analyzer in entries
                          if getattr(analyzer, "convergence_idx", None) is not None})
    for index, snapshot in enumerate(convergence):
        ax.axvline(snapshot / float(divisor), color="C4", linestyle="--",
                   label="Convergence time" if index == 0 else "_nolegend_")


def plot_group_panel(ax, entries, divisor=1.0, show_xlabel=True, show_ylabel=True,
                     kbt_threshold=None):
    for record, analyzer in sorted(entries, key=lambda item: _sort_value(_parameter_label(item[0]))):
        ax.plot(analyzer.t / divisor, analyzer.rmsd_raw, label=str(_parameter_label(record)), linewidth=1.5)
    _plot_rmsd_guides(ax, entries, divisor, kbt_threshold=kbt_threshold)
    group = entries[0][0].group or "RMSD"
    ax.set_title(str(group))
    if show_xlabel:
        ax.set_xlabel("Time (ns)" if divisor != 1 else "Snapshot Index")
    if show_ylabel:
        ax.set_ylabel("RMSD")
    ax.legend(**_rmsd_legend_kwargs())
    ax.grid(True)
    return ax


def plot_seed_panel(ax, entries, divisor=1.0, show_xlabel=True, show_ylabel=True,
                    kbt_threshold=None):
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
    _plot_rmsd_guides(ax, entries, divisor, kbt_threshold=kbt_threshold)
    group = entries[0][0].group or "RMSD by seed"
    ax.set_title(str(group))
    if show_xlabel:
        ax.set_xlabel("Time (ns)" if divisor != 1 else "Snapshot Index")
    if show_ylabel:
        ax.set_ylabel("RMSD")
    ax.legend(handles=handles, **_rmsd_legend_kwargs())
    ax.grid(True)
    return ax


def _render_rmsd_figures(items, *, output_root="Results", analysis_name="rmsd_curves",
                         config=None, seed_mode=False, divisor=1.0,
                         kbt_threshold=None):
    config = config or PlotConfig(output_root=str(output_root))
    output = analysis_output(output_root, analysis_name)
    grouped = defaultdict(list)
    for record, analyzer in items:
        grouped[record.group or record.run_id].append((record, analyzer))
    groups = sorted(grouped.items(), key=lambda item: str(item[0]))
    renderer = plot_seed_panel if seed_mode else plot_group_panel
    with publication_style(config):
        rows, cols = multipanel_grid(len(groups), config)
        fig, axes = make_figure(config, kind="multipanel", nrows=rows, ncols=cols, sharex=True)
        axes_list = flatten_axes(axes)
        for axis, (_, entries) in zip(axes_list, groups):
            renderer(axis, entries, divisor=divisor, show_xlabel=False, show_ylabel=False,
                     kbt_threshold=kbt_threshold)
            axis.label_outer()
        for axis in axes_list[len(groups):]:
            axis.set_visible(False)
        set_shared_labels(fig, xlabel="Time (ns)" if divisor != 1 else "Snapshot Index", ylabel="RMSD")
        finalize_multipanel_layout(fig)
        save_figure(fig, output.figures / f"{safe_name(analysis_name)}_multipanel", config, fit=False)
        close_figure(fig)
        for group, entries in groups:
            panel_fig, panel_ax = make_figure(config, kind="panel")
            renderer(panel_ax, entries, divisor=divisor, kbt_threshold=kbt_threshold)
            save_figure(panel_fig, output.panels / safe_name(str(group)), config)
            close_figure(panel_fig)
    return output


def run(items, *, output_root="Results", analysis_name="rmsd_curves", config=None,
        seed_mode=False, divisor=1.0, temperature_k=298.0, kbt_threshold=None):
    config = config or PlotConfig(output_root=str(output_root))
    output = analysis_output(output_root, analysis_name)
    threshold = _effective_threshold(items, kbt_threshold)
    _write_rmsd_series(items, output, divisor=divisor, temperature_k=temperature_k,
                       kbt_threshold=threshold)
    _render_rmsd_figures(items, output_root=output_root, analysis_name=analysis_name,
                         config=config, seed_mode=seed_mode, divisor=divisor,
                         kbt_threshold=threshold)
    _write_run_table(items, output)
    return output.directory


def render_saved_rmsd_figures(items, *, output_root="Results", config=None,
                              seed_mode=False, divisor=1.0, kbt_threshold=None):
    """Render figures from deserialized RMSD data only."""
    analysis_name = "rmsd_seed_curves" if seed_mode else "rmsd_curves"
    return _render_rmsd_figures(items, output_root=output_root, analysis_name=analysis_name,
                                config=config, seed_mode=seed_mode, divisor=divisor,
                                kbt_threshold=kbt_threshold)


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
        config=config, seed_mode=seed_mode, divisor=args.divisor,
        temperature_k=args.temperature, kbt_threshold=args.rmsd_thresh)
