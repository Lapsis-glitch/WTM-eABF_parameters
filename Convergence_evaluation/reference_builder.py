import numpy as np
import matplotlib.pyplot as plt

# Reuse interpolation / IO utilities from pmf_io to avoid duplication
from pmf_io import interpolate_pmf, write_sequential_pmf

kB = 0.008314462618  # kJ/mol/K


# ============================================================
# 1. PMF <-> Probability conversions
# ============================================================

def pmf_to_prob(F, T):
    """Convert PMF (kJ/mol) to a normalized probability distribution at temperature T."""
    beta = 1.0 / (kB * T)
    P = np.exp(-beta * F)
    return P / np.sum(P)


def prob_to_pmf(P, T):
    """Convert probability distribution to PMF; shift to zero minimum."""
    beta = 1.0 / (kB * T)
    F = -1.0 / beta * np.log(P)
    return F - np.min(F)


# Note: interpolate_pmf and write_sequential_pmf are imported from pmf_io above.
# This file focuses on the statistical aggregation + plotting pipeline.

# ============================================================
# 4. Unified plotting function (1D only)
# ============================================================

def plot_pmf_set(
    x,
    F_median,
    F_all, F_all_err,
    F_filt, F_filt_err,
    deviations,
    cutoff,
    prefix="reference"
):
    """Unified plotting for all PMF curves + outlier diagnostics."""

    # --- PMF comparison plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(x, F_median, label="Median PMF", linewidth=2)
    plt.plot(x, F_all, '--', label="Average (all)", linewidth=2, color='C1')
    plt.plot(x, F_filt, '-.', label="Average (filtered)", linewidth=2, color='C2')

    plt.fill_between(x, F_all - F_all_err, F_all + F_all_err,
                     alpha=0.2, color='C1', label="Error (all)")
    plt.fill_between(x, F_filt - F_filt_err, F_filt + F_filt_err,
                     alpha=0.2, color='C2', label="Error (filtered)")

    plt.xlabel("Coordinate")
    plt.ylabel("PMF (kcal/mol)")
    plt.title("Reference PMFs: Median vs Averages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_pmf_comparison.png", dpi=300)
    plt.close()

    # --- Outlier diagnostics ---
    plt.figure(figsize=(7, 5))
    plt.plot(deviations, 'o', label="Deviation from median")
    plt.axhline(cutoff, color='red', linestyle='--', label="Outlier cutoff")
    plt.xlabel("Simulation index")
    plt.ylabel("Deviation (RMSD in P-space)")
    plt.title("Outlier Diagnostics (MAD-based)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_outlier_diagnostics.png", dpi=300)
    plt.close()


# ============================================================
# 5. Main: median + all-average + filtered-average PMFs
# ============================================================

def compute_reference_pmf_with_outliers(
    coords_tuple,
    F_list,
    T,
    mad_cut=3.5,
    write_prefix="reference"
):
    """
    Full pipeline:
    - Convert PMFs to probabilities
    - Compute:
        * Median PMF
        * Average PMF over ALL simulations (+ error)
        * Average PMF over NON-OUTLIER simulations (+ error)
    - Outliers detected via MAD on deviation from median
    - Write three PMFs + their errors (for averages)
    - Unified plotting
    """

    # Convert to probabilities
    P_list = [pmf_to_prob(F, T) for F in F_list]
    P_stack = np.stack(P_list, axis=0)

    # --- Median ---
    P_median = np.median(P_stack, axis=0)
    F_median = prob_to_pmf(P_median, T)

    # --- Average over ALL ---
    P_all = np.mean(P_stack, axis=0)
    P_all_std = np.std(P_stack, axis=0)
    F_all = prob_to_pmf(P_all, T)
    beta = 1.0 / (kB * T)
    F_all_err = (1.0 / beta) * (P_all_std / (P_all + 1e-12))

    # --- Outlier detection (N-dimensional safe) ---
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

    # --- Write PMFs + errors ---
    write_sequential_pmf(coords_tuple, F_median, f"{write_prefix}_median.pmf")

    write_sequential_pmf(coords_tuple, F_all, f"{write_prefix}_average_all.pmf")
    write_sequential_pmf(coords_tuple, F_all_err, f"{write_prefix}_average_all_err.pmf")

    write_sequential_pmf(coords_tuple, F_filt, f"{write_prefix}_average_filtered.pmf")
    write_sequential_pmf(coords_tuple, F_filt_err, f"{write_prefix}_average_filtered_err.pmf")

    # --- Plotting (1D only) ---
    if len(coords_tuple) == 1:
        x = coords_tuple[0]
        plot_pmf_set(
            x,
            F_median,
            F_all, F_all_err,
            F_filt, F_filt_err,
            deviations,
            cutoff,
            prefix=write_prefix
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