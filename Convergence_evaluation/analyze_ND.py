#!/usr/bin/env python3
"""PMF convergence analysis and PubReady publication figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

try:
    from .outputs import analysis_output
    from .plotting import PlotConfig, add_plotting_arguments, close_figure, flatten_axes
    from .plotting import make_figure, publication_style, save_figure
    from .pmf_io import (interpolate_pmf, read_sequential_counts, read_sequential_pmf,
                         read_sequential_pmf_blocks)
except ImportError:  # Support direct script execution.
    from outputs import analysis_output
    from plotting import PlotConfig, add_plotting_arguments, close_figure, flatten_axes
    from plotting import make_figure, publication_style, save_figure
    from pmf_io import interpolate_pmf, read_sequential_counts, read_sequential_pmf, read_sequential_pmf_blocks


class PMFAnalyzer:
    """Analyze PMF snapshots while preserving the established calculations."""

    def __init__(self, pmf_file, count_file=None, n_recent=4, slope_thresh=1e-3,
                 use_sliding_window=False, count_std_thresh=None,
                 reference_pmf_file=None, rmsd_thresh=None, use_ref_and_slope=False):
        self.pmf_file = str(pmf_file)
        self.count_file = str(count_file) if count_file is not None else None
        self.n_recent = n_recent
        self.slope_thresh = slope_thresh
        self.use_sliding_window = use_sliding_window
        self.count_std_thresh = count_std_thresh
        self.reference_pmf_file = reference_pmf_file
        self.rmsd_thresh = rmsd_thresh
        self.use_ref_and_slope = use_ref_and_slope
        try:
            # File role is determined from its contents, not an experiment-
            # specific filename token such as ``hist``.
            try:
                coords, pmf = read_sequential_pmf(self.pmf_file)
                pmf_blocks = [(coords, pmf)]
            except (AssertionError, IndexError, ValueError):
                pmf_blocks = read_sequential_pmf_blocks(self.pmf_file)
        except Exception as exc:
            raise RuntimeError(f"Failed to read PMF file '{self.pmf_file}': {exc}") from exc
        if not pmf_blocks:
            raise RuntimeError(f"PMF file '{self.pmf_file}' contains no PMFs")
        self.pmfs = pmf_blocks
        self.pmf_coords = self.pmfs[0][0]
        self.pmf_values = [block[1] for block in self.pmfs]
        if self.count_file is None:
            self.counts, self.count_coords = [], self.pmf_coords
        else:
            try:
                self.counts, self.count_coords = read_sequential_counts(self.count_file)
            except Exception as exc:
                raise RuntimeError(f"Failed to read counts file '{self.count_file}': {exc}") from exc
        self.normed_counts = self._normalize_counts(self.counts)
        self.reference_pmf = None
        if self.reference_pmf_file is not None:
            try:
                ref_coords, ref_pmf = read_sequential_pmf(self.reference_pmf_file)
                self.reference_pmf = interpolate_pmf(ref_coords, ref_pmf, self.pmf_coords)
            except Exception as exc:
                raise RuntimeError(f"Failed to read reference PMF '{self.reference_pmf_file}': {exc}") from exc
        try:
            self._compute_rmsd_series()
            self.rmsd_smooth = self._smooth_rmsd(self.rmsd_raw)
            self.params, self.rmsd_fit = self._fit_exp_decay()
            self.convergence_idx = self._detect_convergence()
        except Exception as exc:
            raise RuntimeError(f"RMSD/convergence setup failed: {exc}") from exc

    def _compute_rmsd_series(self):
        def zero_min(arr):
            return arr - np.nanmin(arr)
        if self.reference_pmf is not None:
            ref = zero_min(self.reference_pmf)
            self.rmsd_raw = np.array([np.sqrt(np.nanmean((zero_min(pmf) - ref) ** 2)) for pmf in self.pmf_values])
            self.t = np.arange(len(self.rmsd_raw))
            return
        if not self.use_sliding_window:
            final = zero_min(self.pmf_values[-1])
            self.rmsd_raw = np.array([np.sqrt(np.mean((zero_min(pmf) - final) ** 2)) for pmf in self.pmf_values])
            self.t = np.arange(len(self.rmsd_raw))
            return
        if len(self.pmf_values) <= self.n_recent:
            raise ValueError(f"Not enough PMF snapshots ({len(self.pmf_values)}) for sliding window of size {self.n_recent}")
        rmsd_vals = []
        for i in range(self.n_recent, len(self.pmf_values)):
            ref = zero_min(np.mean(self.pmf_values[i - self.n_recent:i], axis=0))
            rmsd_vals.append(np.sqrt(np.mean((zero_min(self.pmf_values[i]) - ref) ** 2)))
        self.rmsd_raw = np.array(rmsd_vals)
        self.t = np.arange(self.n_recent, len(self.pmf_values))

    @staticmethod
    def _normalize_counts(counts):
        if not counts:
            return []
        c_min = min(np.min(c) for c in counts)
        c_max = max(np.max(c) for c in counts)
        return [(c - c_min) / (c_max - c_min + 1e-12) for c in counts]

    @staticmethod
    def _smooth_rmsd(rmsd, window_length=11, polyorder=3):
        if len(rmsd) < 3:
            return rmsd
        max_wl = min(window_length, len(rmsd))
        if max_wl % 2 == 0:
            max_wl -= 1
        wl = max(3, max_wl)
        return savgol_filter(rmsd, window_length=wl, polyorder=min(polyorder, wl - 1))

    @staticmethod
    def _exp_decay(t, A, B, C):
        return A * np.exp(-B * t) + C

    def _fit_exp_decay(self):
        try:
            params, _ = curve_fit(self._exp_decay, self.t, self.rmsd_smooth,
                                  p0=(1, 0.1, 0.01), maxfev=10000)
            return params, self._exp_decay(self.t, *params)
        except Exception:
            return None, np.full_like(self.t, np.nan)

    def _detect_convergence(self):
        if self.use_ref_and_slope and self.reference_pmf is not None:
            if np.isnan(self.rmsd_fit).all():
                return None
            slope = np.gradient(self.rmsd_fit, self.t)
            for idx, (raw_rmsd, value) in enumerate(zip(self.rmsd_raw, slope)):
                if self.rmsd_thresh is not None and raw_rmsd < self.rmsd_thresh and abs(value) < self.slope_thresh:
                    return self.t[idx]
        if self.rmsd_thresh is not None:
            for idx, value in enumerate(self.rmsd_raw):
                if value < self.rmsd_thresh:
                    return self.t[idx]
            return None
        if np.isnan(self.rmsd_fit).all():
            return None
        slope = np.gradient(self.rmsd_fit, self.t)
        if self.count_std_thresh is None or not self.normed_counts:
            for idx, value in enumerate(slope):
                if abs(value) < self.slope_thresh:
                    return self.t[idx]
            return None
        for idx, value in enumerate(slope):
            if abs(value) < self.slope_thresh:
                window = self.normed_counts[idx:idx + self.n_recent]
                if window and np.mean([np.std(item) for item in window]) < self.count_std_thresh:
                    return self.t[idx]
        return None

    def detect_features(self, window=3, grad_thresh=0.1):
        pmf_grid = self.pmf_values[-1]
        coords = self.pmf_coords
        local_min = pmf_grid == minimum_filter(pmf_grid, size=window)
        local_max = pmf_grid == maximum_filter(pmf_grid, size=window)
        grads = np.gradient(pmf_grid, *coords)
        grad_mag = np.sqrt(sum(g ** 2 for g in grads))
        self.features = {"minima": np.argwhere(local_min), "maxima": np.argwhere(local_max),
                         "plateaus": np.argwhere(grad_mag < grad_thresh)}

    def annotate_comparison(self, ax, fs=None):
        self.detect_features()
        pmf_grid, coords = self.pmf_values[-1], self.pmf_coords
        if pmf_grid.ndim == 1:
            for idx in self.features["minima"]:
                x, y = coords[0][idx[0]], pmf_grid[idx[0]]
                ax.plot(x, y, "bo")
                ax.text(x, y, f"Min\\n{x:.2f}", ha="center", color="blue")
            for idx in self.features["maxima"]:
                x, y = coords[0][idx[0]], pmf_grid[idx[0]]
                ax.plot(x, y, "ro")
                ax.text(x, y, f"Max\\n{x:.2f}", ha="center", color="red")
        else:
            coords_x, coords_y = coords[0], coords[1]
            for idx in self.features["minima"]:
                ax.plot(coords_x[idx[0]], coords_y[idx[1]], "bo")
                ax.text(coords_x[idx[0]], coords_y[idx[1]], "Min", ha="center", color="blue")
            for idx in self.features["maxima"]:
                ax.plot(coords_x[idx[0]], coords_y[idx[1]], "ro")
                ax.text(coords_x[idx[0]], coords_y[idx[1]], "Max", ha="center", color="red")

    def plot(self, show_annotations=True, save_path=None, output_dir=None,
             config=None, show=False, snapshots_dir=None, **_ignored):
        """Save one multipanel and one standalone figure for every applicable panel."""
        config = config or PlotConfig()
        output = analysis_output(config.output_root, "pmf_convergence") if output_dir is None else None
        figures_dir = Path(output_dir) / "Figures" if output_dir is not None else output.figures
        panels_dir = figures_dir / "panels"
        figures_dir.mkdir(parents=True, exist_ok=True)
        panels_dir.mkdir(parents=True, exist_ok=True)
        panel_specs = [("rmsd_convergence", plot_rmsd_panel, {}),
                       ("sequential_pmfs", plot_sequential_pmf_panel, {})]
        if self.normed_counts:
            panel_specs.append(("sampling_evolution", plot_sampling_panel, {}))
        if self.convergence_idx is not None:
            panel_specs.append(("post_convergence_vs_final", plot_post_convergence_panel,
                                {"show_annotations": show_annotations}))
        with publication_style(config):
            fig, axes = make_figure(config, kind="multipanel", nrows=2, ncols=2)
            axes_list = flatten_axes(axes)
            for axis, (_, renderer, options) in zip(axes_list, panel_specs):
                renderer(axis, self, **options)
            for axis in axes_list[len(panel_specs):]:
                axis.set_visible(False)
            base = Path(save_path).with_suffix("") if save_path else figures_dir / "pmf_convergence_multipanel"
            save_figure(fig, base, config)
            close_figure(fig)
            for name, renderer, options in panel_specs:
                panel_fig, panel_ax = make_figure(config, kind="panel")
                renderer(panel_ax, self, **options)
                save_figure(panel_fig, panels_dir / name, config)
                close_figure(panel_fig)
        if snapshots_dir:
            Path(snapshots_dir).mkdir(parents=True, exist_ok=True)
        if show:
            plt.show()

    def plot_snapshot(self, idx, out_path=None, config=None, annotate=True, show=False, **_ignored):
        """Optional per-snapshot export, kept separate from semantic panel exports."""
        if idx < 0 or idx >= len(self.pmf_values):
            raise IndexError("Snapshot index out of range")
        config = config or PlotConfig()
        destination = Path(out_path) if out_path else analysis_output(config.output_root, "pmf_convergence").snapshots / f"pmf_snapshot_{idx:04d}"
        with publication_style(config):
            if self.pmf_values[idx].ndim == 1:
                fig, axes = make_figure(config, kind="panel", ncols=2)
                axes = flatten_axes(axes)
                axes[0].plot(self.pmf_coords[0], self.pmf_values[idx], color="C0", lw=2)
                axes[0].set(xlabel=r"$\xi$", ylabel="PMF", title=f"PMF snapshot {idx}")
                if idx < len(self.normed_counts):
                    axes[1].plot(self.count_coords[0], self.normed_counts[idx], color="C1", lw=2)
                    axes[1].set(xlabel=r"$\xi$", ylabel="Normalized Count", title=f"Sampling snapshot {idx}")
                else:
                    axes[1].set_visible(False)
                if annotate and idx == len(self.pmf_values) - 1:
                    self.annotate_comparison(axes[0])
            else:
                fig, ax = make_figure(config, kind="panel")
                mappable = _plot_2d_field(ax, self.pmf_coords, self.pmf_values[idx], "viridis")
                if idx < len(self.normed_counts):
                    X, Y = np.meshgrid(self.count_coords[0], self.count_coords[1], indexing="ij")
                    ax.contour(X, Y, self.normed_counts[idx], levels=8,
                               colors="k", linewidths=0.6, alpha=0.6)
                import pubready as pr
                pr.add_colorbar(fig, mappable, ax=ax, location="right", label="PMF")
                ax.set_title(f"PMF snapshot {idx}")
            save_figure(fig, destination, config)
            close_figure(fig)
        if show:
            plt.show()

    def _plot_sequence(self, ax, values_list, coords, title, **kwargs):
        return plot_sequence_panel(ax, values_list, coords, title, **kwargs)


def _plot_2d_field(ax, coords, values, cmap):
    X, Y = np.meshgrid(coords[0], coords[1], indexing="ij")
    return ax.contourf(X, Y, values, levels=30, cmap=cmap)


def plot_rmsd_panel(ax, analysis, **_kwargs):
    ax.plot(analysis.t, analysis.rmsd_raw, color="gray", alpha=0.6, label="Raw RMSD")
    ax.plot(analysis.t, analysis.rmsd_smooth, color="C0", label="Smoothed")
    if not np.isnan(analysis.rmsd_fit).all():
        ax.plot(analysis.t, analysis.rmsd_fit, "--", color="C1", label="Fit")
    if analysis.convergence_idx is not None:
        ax.axvline(analysis.convergence_idx, color="C3", linestyle="--", label="Converged")
        ax.axvspan(analysis.convergence_idx, analysis.t[-1], color="C3", alpha=0.2)
    ax.set(title="PMF Convergence", xlabel="PMF Snapshot Index", ylabel="RMSD")
    ax.grid(True)
    ax.legend(loc="best")
    return ax


def plot_sequence_panel(ax, values_list, coords, title, is_count=False):
    if not values_list:
        return ax
    last_idx = len(values_list) - 1
    if values_list[0].ndim == 1:
        for i, values in enumerate(values_list):
            color = "black" if i == last_idx else str(0.3 + 0.7 * i / max(1, last_idx))
            ax.plot(coords[0], values, color=color, linewidth=2 if i == last_idx else 1)
        ax.set(xlabel=r"$\xi$", ylabel="Normalized Count" if is_count else "PMF", title=title)
    else:
        X, Y = np.meshgrid(coords[0], coords[1], indexing="ij")
        for i, values in enumerate(values_list):
            color = "black" if i == last_idx else str(0.3 + 0.7 * i / max(1, last_idx))
            ax.contour(X, Y, values, levels=20, colors=[color], linewidths=1)
        ax.set(xlabel="Coord 1", ylabel="Coord 2", title=title)
    ax.grid(True)
    return ax


def plot_sequential_pmf_panel(ax, analysis, **_kwargs):
    return plot_sequence_panel(ax, analysis.pmf_values, analysis.pmf_coords, "Sequential PMFs")


def plot_sampling_panel(ax, analysis, **_kwargs):
    return plot_sequence_panel(ax, analysis.normed_counts, analysis.count_coords,
                               "Sampling Evolution", is_count=True)


def plot_post_convergence_panel(ax, analysis, show_annotations=True, **_kwargs):
    idx = next((i for i, value in enumerate(analysis.t) if value >= analysis.convergence_idx), None)
    if idx is None:
        return ax
    pmf_index = analysis.n_recent + idx if analysis.use_sliding_window else idx
    pmf_post, pmf_final = analysis.pmf_values[pmf_index], analysis.pmf_values[-1]
    if pmf_final.ndim == 1:
        ax.plot(analysis.pmf_coords[0], pmf_post, color="C0", linewidth=2, label="Post-Conv")
        ax.plot(analysis.pmf_coords[0], pmf_final, "--", color="black", linewidth=2, label="Final")
        ax.set(xlabel=r"$\xi$", ylabel="PMF")
    else:
        X, Y = np.meshgrid(analysis.pmf_coords[0], analysis.pmf_coords[1], indexing="ij")
        mappable = ax.contourf(X, Y, pmf_final, levels=30, cmap="viridis")
        ax.contour(X, Y, pmf_post, levels=30, colors="C0", linewidths=1)
        import pubready as pr
        pr.add_colorbar(ax.figure, mappable, ax=ax, location="right", label="Final PMF")
        ax.set(xlabel="Coord 1", ylabel="Coord 2")
    ax.set_title("Post-Convergence vs Final")
    ax.grid(True)
    ax.legend(loc="best")
    if show_annotations:
        analysis.annotate_comparison(ax)
    return ax


def main():
    parser = argparse.ArgumentParser(description="Generate PMF convergence and comparison plots")
    parser.add_argument("pmf_file", help="Path to PMF history or single PMF file")
    parser.add_argument("counts_file", nargs="?", default=None, help="Path to sampling counts history file")
    parser.add_argument("--font-size", type=int, default=None, help="Deprecated compatibility option")
    parser.add_argument("--annotation-fs", type=int, default=None, help="Deprecated compatibility option")
    parser.add_argument("--no-annotations", dest="show_annotations", action="store_false")
    parser.add_argument("--save-path", default=None, help="Compatibility base path; Results/ is preferred")
    parser.add_argument("--save-each", default=None, help="Optional directory for per-snapshot figures")
    parser.add_argument("--conv-threshold", type=float, default=0.01)
    parser.add_argument("--rmsd-threshold", type=float, default=None)
    parser.add_argument("--n-recent", type=int, default=10)
    parser.add_argument("--use-sliding-window", action="store_true")
    parser.add_argument("--counts-std-thresh", type=float, default=None)
    parser.add_argument("--reference-pmf", default=None)
    parser.add_argument("--use-ref-and-slope", action="store_true")
    add_plotting_arguments(parser)
    args = parser.parse_args()
    config = PlotConfig(publisher=args.publisher, multipanel_target=args.multipanel_target,
                        multipanel_fraction=args.multipanel_fraction, panel_target=args.panel_target,
                        panel_fraction=args.panel_fraction, formats=tuple(args.figure_formats),
                        dpi=args.dpi, output_root=args.output_root)
    analyzer = PMFAnalyzer(args.pmf_file, args.counts_file, slope_thresh=args.conv_threshold,
                           n_recent=args.n_recent, use_sliding_window=args.use_sliding_window,
                           count_std_thresh=args.counts_std_thresh, reference_pmf_file=args.reference_pmf,
                           rmsd_thresh=args.rmsd_threshold, use_ref_and_slope=args.use_ref_and_slope)
    output = analysis_output(config.output_root, "pmf_convergence")
    analyzer.plot(show_annotations=args.show_annotations, save_path=args.save_path,
                  output_dir=output.directory, config=config, snapshots_dir=args.save_each)
    if args.save_each:
        for index in range(len(analyzer.pmf_values)):
            analyzer.plot_snapshot(index, out_path=Path(args.save_each) / f"pmf_snapshot_{index:04d}", config=config)


if __name__ == "__main__":
    main()
