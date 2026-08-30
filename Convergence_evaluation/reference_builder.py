"""Reference-PMF statistics and PubReady panel renderers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .pmf_io import interpolate_pmf, write_sequential_pmf
    from .outputs import analysis_output
    from .plotting import (PlotConfig, close_figure, finalize_multipanel_layout,
                           make_figure, publication_style, save_figure)
except ImportError:
    from pmf_io import interpolate_pmf, write_sequential_pmf
    from outputs import analysis_output
    from plotting import (PlotConfig, close_figure, finalize_multipanel_layout,
                          make_figure, publication_style, save_figure)


kB = 0.008314462618
REFERENCE_XLABEL = "Coordinate"
REFERENCE_YLABEL = "PMF (kcal/mol)"


def pmf_to_prob(F, T):
    beta = 1.0 / (kB * T)
    P = np.exp(-beta * F)
    return P / np.sum(P)


def prob_to_pmf(P, T):
    beta = 1.0 / (kB * T)
    F = -1.0 / beta * np.log(P)
    return F - np.min(F)


def compute_reference_pmf_with_outliers(coords_tuple, F_list, T, mad_cut=3.5,
                                        write_prefix="reference"):
    """Compute the original median/all/filtered PMFs and uncertainty arrays."""
    P_list = [pmf_to_prob(F, T) for F in F_list]
    P_stack = np.stack(P_list, axis=0)
    P_median = np.median(P_stack, axis=0)
    F_median = prob_to_pmf(P_median, T)
    P_all = np.mean(P_stack, axis=0)
    P_all_std = np.std(P_stack, axis=0)
    F_all = prob_to_pmf(P_all, T)
    beta = 1.0 / (kB * T)
    F_all_err = (1.0 / beta) * (P_all_std / (P_all + 1e-12))
    flat = P_stack.reshape(P_stack.shape[0], -1)
    flat_median = P_median.reshape(-1)
    deviations = np.sqrt(np.mean((flat - flat_median) ** 2, axis=1))
    med_dev = np.median(deviations)
    mad = np.median(np.abs(deviations - med_dev)) + 1e-12
    cutoff = med_dev + mad_cut * mad
    keep_mask = deviations < cutoff
    P_kept = P_stack[keep_mask]
    P_filt = np.mean(P_kept, axis=0)
    P_filt_std = np.std(P_kept, axis=0)
    F_filt = prob_to_pmf(P_filt, T)
    F_filt_err = (1.0 / beta) * (P_filt_std / (P_filt + 1e-12))
    prefix = Path(write_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_sequential_pmf(coords_tuple, F_median, f"{prefix}_median.pmf")
    write_sequential_pmf(coords_tuple, F_all, f"{prefix}_average_all.pmf")
    write_sequential_pmf(coords_tuple, F_all_err, f"{prefix}_average_all_err.pmf")
    write_sequential_pmf(coords_tuple, F_filt, f"{prefix}_average_filtered.pmf")
    write_sequential_pmf(coords_tuple, F_filt_err, f"{prefix}_average_filtered_err.pmf")
    return {"F_median": F_median, "F_all": F_all, "F_all_err": F_all_err,
            "F_filtered": F_filt, "F_filtered_err": F_filt_err,
            "keep_mask": keep_mask, "deviations": deviations, "cutoff": cutoff}


def plot_median_reference(ax, x, median, *, xlabel=REFERENCE_XLABEL,
                          ylabel=REFERENCE_YLABEL, show_legend=False,
                          show_title=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot(x, median, label="Median", linewidth=2, color=colors[0])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_ylim(-0.1, None)
    if show_legend:
        ax.legend(loc="best")
    if not show_title:
        ax.set_title("")
    return ax


def plot_pmf_comparison_panel(ax, data, x):
    plot_median_reference(ax, x, data["F_median"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot(x, data["F_all"], "--", label="Average (all)", linewidth=2, color=colors[2])
    ax.plot(x, data["F_filtered"], "-.", label="Average (filtered)", linewidth=2, color=colors[1])
    ax.fill_between(x, data["F_all"] - data["F_all_err"], data["F_all"] + data["F_all_err"],
                    alpha=0.4, color="lightgray", label="Error (all)")
    ax.fill_between(x, data["F_filtered"] - data["F_filtered_err"],
                    data["F_filtered"] + data["F_filtered_err"], alpha=0.2, color=colors[1],
                    label="Error (filtered)")
    ax.legend(loc="best")
    return ax


def plot_simple_reference_panel(ax, data, x, *, xlabel=None):
    """Render only the already-computed median reference PMF."""
    return plot_median_reference(ax, x, data["F_median"],
                                 xlabel=REFERENCE_XLABEL if xlabel is None else xlabel)


def plot_outlier_panel(ax, data):
    ax.plot(data["deviations"], "o", label="Deviation from median")
    ax.axhline(data["cutoff"], color="C3", linestyle="--", label="Outlier cutoff")
    ax.set(xlabel="Simulation index", ylabel="Deviation\n(RMSD in P-space)",
           title="Outlier Diagnostics\n(MAD-based)")
    ax.legend(loc="best")
    return ax


def render_reference_figures(data, coords_tuple, *, output_root="Results", config=None,
                             simple_reference_plot=False, xlabel=None):
    config = config or PlotConfig(output_root=str(output_root))
    output = analysis_output(output_root, "reference_pmf")
    if len(coords_tuple) != 1:
        return output.directory
    renderers = [("pmf_comparison", lambda ax: plot_pmf_comparison_panel(ax, data, coords_tuple[0])),
                 ("outlier_diagnostics", lambda ax: plot_outlier_panel(ax, data))]
    with publication_style(config):
        fig, axes = make_figure(config, kind="multipanel", ncols=2)
        for axis, (_, renderer) in zip(np.asarray(axes, dtype=object).flat, renderers):
            renderer(axis)
        finalize_multipanel_layout(fig)
        save_figure(fig, output.figures / "reference_pmf_multipanel", config, fit=False)
        close_figure(fig)
        for name, renderer in renderers:
            panel_fig, panel_ax = make_figure(config, kind="panel")
            renderer(panel_ax)
            save_figure(panel_fig, output.panels / name, config)
            close_figure(panel_fig)
        if simple_reference_plot:
            simple_fig, simple_ax = make_figure(config, kind="panel")
            plot_simple_reference_panel(simple_ax, data, coords_tuple[0], xlabel=xlabel)
            save_figure(simple_fig, output.figures / "reference_median", config)
            close_figure(simple_fig)
    return output.directory
