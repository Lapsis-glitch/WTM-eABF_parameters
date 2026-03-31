"""
reference_builder.py
--------------------
Statistical aggregation pipeline for building robust reference PMFs from
multiple simulation replicas.

Workflow
--------
1. Convert each PMF to a Boltzmann probability distribution.
2. Compute **median**, **mean (all)**, and **mean (outlier-filtered)** PMFs.
3. Detect outliers via MAD (Median Absolute Deviation) on RMSD-in-P-space.
4. Write output PMFs + error envelopes, and generate comparison plots.

IO utilities (``interpolate_pmf``, ``write_sequential_pmf``) are imported
from ``pmf_io`` to avoid duplication.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pmf_io import interpolate_pmf, write_sequential_pmf  # noqa: F401 (re-export)
from plotting_config import configure_plotting

configure_plotting()
sns.set_palette("colorblind")

# Boltzmann constant in kJ·mol⁻¹·K⁻¹
kB = 0.008314462618


# ============================================================
# PMF ↔ probability conversions
# ============================================================

def pmf_to_prob(F: np.ndarray, T: float) -> np.ndarray:
    """Convert a PMF (kJ/mol) to a normalised Boltzmann probability at temperature *T*."""
    beta = 1.0 / (kB * T)
    P = np.exp(-beta * F)
    return P / np.sum(P)


def prob_to_pmf(P: np.ndarray, T: float) -> np.ndarray:
    """Convert a probability distribution back to a PMF; shift minimum to zero."""
    beta = 1.0 / (kB * T)
    F = -1.0 / beta * np.log(P)
    return F - np.min(F)


# ============================================================
# Plotting helpers
# ============================================================

def plot_pmf_set(
    x, F_median,
    F_all, F_all_err,
    F_filt, F_filt_err,
    deviations, cutoff,
    prefix="reference",
):
    """
    Produce two figures:
      * PMF comparison (median / all-average / filtered-average ± error).
      * Outlier diagnostics (per-simulation deviation from median).

    Parameters
    ----------
    x : 1-D array
        Coordinate grid.
    F_median, F_all, F_filt : 1-D arrays
        The three reference PMF variants.
    F_all_err, F_filt_err : 1-D arrays
        Standard-error envelopes.
    deviations : 1-D array
        Per-simulation RMSD from the median in probability space.
    cutoff : float
        MAD-based outlier threshold.
    prefix : str
        Output file prefix (without extension).
    """
    colors = sns.color_palette("colorblind")

    # --- PMF comparison ---
    plt.figure(figsize=(3.25, 3))
    plt.plot(x, F_median, label="Median", linewidth=2, color=colors[0])
    plt.plot(x, F_all, "--", label="Average (all)", linewidth=2, color=colors[2])
    plt.plot(x, F_filt, "-.", label="Average (filtered)", linewidth=2, color=colors[1])
    plt.fill_between(x, F_all - F_all_err, F_all + F_all_err,
                     alpha=0.4, color="lightgray", label="Error (all)")
    plt.fill_between(x, F_filt - F_filt_err, F_filt + F_filt_err,
                     alpha=0.2, color=colors[1], label="Error (filtered)")
    plt.xlabel("Coordinate")
    plt.ylabel("PMF (kcal/mol)")
    plt.ylim(-0.1, None)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{prefix}_pmf_comparison.pdf", dpi=300)
    plt.close()

    # --- Outlier diagnostics ---
    plt.figure(figsize=(7, 5))
    plt.plot(deviations, "o", label="Deviation from median")
    plt.axhline(cutoff, color="red", linestyle="--", label="Outlier cutoff")
    plt.xlabel("Simulation index")
    plt.ylabel("Deviation (RMSD in P-space)")
    plt.title("Outlier Diagnostics (MAD-based)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_outlier_diagnostics.png", dpi=300)
    plt.close()


# ============================================================
# Main pipeline
# ============================================================

def compute_reference_pmf_with_outliers(
    coords_tuple, F_list, T,
    mad_cut=3.5,
    write_prefix="reference",
):
    """
    Full aggregation pipeline:

    1. Convert each PMF → probability.
    2. Compute median, all-average, and outlier-filtered average PMFs.
    3. Detect outliers via MAD on deviation from the median (in P-space).
    4. Write three PMFs + error envelopes to disk.
    5. Generate comparison / diagnostic plots (1-D only).

    Parameters
    ----------
    coords_tuple : tuple of 1-D arrays
        Coordinate grid (one array per dimension).
    F_list : list of N-D arrays
        PMF values from each replica.
    T : float
        Temperature in Kelvin.
    mad_cut : float
        Number of MAD units for outlier rejection.
    write_prefix : str
        Path prefix for output files.

    Returns
    -------
    dict with keys: F_median, F_all, F_all_err, F_filtered, F_filtered_err,
                    keep_mask, deviations, cutoff.
    """
    # Convert PMFs → probability distributions
    P_list = [pmf_to_prob(F, T) for F in F_list]
    P_stack = np.stack(P_list, axis=0)

    # --- Median PMF ---
    P_median = np.median(P_stack, axis=0)
    F_median = prob_to_pmf(P_median, T)

    # --- All-average PMF ---
    P_all = np.mean(P_stack, axis=0)
    P_all_std = np.std(P_stack, axis=0)
    F_all = prob_to_pmf(P_all, T)
    beta = 1.0 / (kB * T)
    F_all_err = (1.0 / beta) * (P_all_std / (P_all + 1e-12))

    # --- Outlier detection (MAD in flattened P-space) ---
    flat = P_stack.reshape(P_stack.shape[0], -1)
    flat_median = P_median.reshape(-1)
    deviations = np.sqrt(np.mean((flat - flat_median) ** 2, axis=1))

    med_dev = np.median(deviations)
    mad = np.median(np.abs(deviations - med_dev)) + 1e-12
    cutoff = med_dev + mad_cut * mad
    keep_mask = deviations < cutoff

    # --- Filtered-average PMF ---
    P_kept = P_stack[keep_mask]
    P_filt = np.mean(P_kept, axis=0)
    P_filt_std = np.std(P_kept, axis=0)
    F_filt = prob_to_pmf(P_filt, T)
    F_filt_err = (1.0 / beta) * (P_filt_std / (P_filt + 1e-12))

    # --- Write output PMFs ---
    write_sequential_pmf(coords_tuple, F_median, f"{write_prefix}_median.pmf")
    write_sequential_pmf(coords_tuple, F_all, f"{write_prefix}_average_all.pmf")
    write_sequential_pmf(coords_tuple, F_all_err, f"{write_prefix}_average_all_err.pmf")
    write_sequential_pmf(coords_tuple, F_filt, f"{write_prefix}_average_filtered.pmf")
    write_sequential_pmf(coords_tuple, F_filt_err, f"{write_prefix}_average_filtered_err.pmf")

    # --- Plots (1-D only) ---
    if len(coords_tuple) == 1:
        plot_pmf_set(
            coords_tuple[0], F_median,
            F_all, F_all_err,
            F_filt, F_filt_err,
            deviations, cutoff,
            prefix=write_prefix,
        )

    return {
        "F_median": F_median,
        "F_all": F_all,
        "F_all_err": F_all_err,
        "F_filtered": F_filt,
        "F_filtered_err": F_filt_err,
        "keep_mask": keep_mask,
        "deviations": deviations,
        "cutoff": cutoff,
    }

