import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

kB = 0.008314462618  # kJ/mol/K


# ============================================================
# 1. PMF <-> Probability conversions
# ============================================================

def pmf_to_prob(F, T):
    beta = 1.0 / (kB * T)
    P = np.exp(-beta * F)
    return P / np.sum(P)


def prob_to_pmf(P, T):
    beta = 1.0 / (kB * T)
    F = -1.0 / beta * np.log(P)
    return F - np.min(F)


# ============================================================
# 2. Interpolation to uniform grid (N-dimensional)
# ============================================================

def interpolate_pmf(coords_tuple, pmf, n_points):
    new_coords = [np.linspace(c[0], c[-1], n_points) for c in coords_tuple]
    new_coords_tuple = tuple(new_coords)

    interpolator = RegularGridInterpolator(
        coords_tuple, pmf, bounds_error=False, fill_value=np.nan
    )

    mesh = np.meshgrid(*new_coords_tuple, indexing="ij")
    points = np.stack([m.flatten() for m in mesh], axis=-1)

    new_pmf = interpolator(points).reshape([n_points] * len(coords_tuple))

    # Replace NaNs with large values
    nan_mask = np.isnan(new_pmf)
    if np.any(nan_mask):
        new_pmf[nan_mask] = np.nanmax(new_pmf) + 50.0

    return new_coords_tuple, new_pmf


# ============================================================
# 3. Write PMF in sequential format
# ============================================================

def write_sequential_pmf(coords_tuple, pmf, filename):
    """
    For ND:
      - header with dimension metadata
      - data rows in nested order
      - blank line after each sweep of the first dimension (i index)
    For 1D:
      - no blank blank lines between rows.
    """
    ndim = len(coords_tuple)
    shape = pmf.shape

    starts = [c[0] for c in coords_tuple]
    steps = [(c[1] - c[0]) if len(c) > 1 else 0.0 for c in coords_tuple]
    sizes = [len(c) for c in coords_tuple]

    with open(filename, "w") as f:
        f.write(f"# {ndim}\n")
        for start, step, size in zip(starts, steps, sizes):
            f.write(f"# {start: .14e}  {step: .14e}  {size:8d}  1\n")
        f.write("\n")

        if ndim == 1:
            # Simple 1D: no blank lines between rows
            x = coords_tuple[0]
            for i in range(shape[0]):
                f.write(f"{x[i]: .14e}   {pmf[i]: .14e}\n")
        else:
            # ND: blank line after each sweep of the first dimension
            for i in range(shape[0]):
                it = np.nditer(pmf[i], flags=['multi_index'])
                for val in it:
                    idx = (i,) + it.multi_index
                    coords = [coords_tuple[d][idx[d]] for d in range(ndim)]
                    f.write("  ".join(f"{c: .14e}" for c in coords) + f"   {val: .14e}\n")
                f.write("\n")


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
    F_all_err = (1.0 / beta) * (P_all_std / P_all)

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
    F_filt_err = (1.0 / beta) * (P_filt_std / P_filt)

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