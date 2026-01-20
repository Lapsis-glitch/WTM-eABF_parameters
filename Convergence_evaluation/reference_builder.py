import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

kB = 0.008314462618  # kJ/mol/K


# ============================================================
# 1. PMF <-> Probability conversions
# ============================================================

def pmf_to_prob(F, T):
    """
    Convert PMF F(x) to normalized probability P(x).
    F in kJ/mol, T in Kelvin.
    """
    beta = 1.0 / (kB * T)
    P_unnorm = np.exp(-beta * F)

    # Normalize using sum or integral depending on dimensionality
    Z = np.sum(P_unnorm)
    return P_unnorm / Z


def prob_to_pmf(P, T):
    """
    Convert probability P(x) to PMF F(x).
    """
    beta = 1.0 / (kB * T)
    F = -1.0 / beta * np.log(P)

    # Shift minimum to zero
    F -= np.min(F)
    return F


# ============================================================
# 2. Combine probabilities + compute uncertainties
# ============================================================

def combine_probabilities(P_list, weights=None):
    """
    Combine multiple probability distributions into a reference distribution.
    Also compute per-bin standard deviation across simulations.
    """
    n = len(P_list)

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights) / np.sum(weights)

    # Weighted mean probability
    P_ref = np.zeros_like(P_list[0])
    for w, P in zip(weights, P_list):
        P_ref += w * P

    # Standard deviation across simulations (unweighted)
    P_stack = np.stack(P_list, axis=0)
    P_std = np.std(P_stack, axis=0)

    return P_ref, P_std


# ============================================================
# 3. Interpolation to uniform grid (N-dimensional)
# ============================================================

def interpolate_pmf(coords_tuple, pmf, n_points):
    """
    Interpolate an N-dimensional PMF onto a new uniform grid
    with n_points per dimension.

    coords_tuple: tuple of coordinate arrays (x1, x2, ..., xN)
    pmf: ndarray of shape (len(x1), len(x2), ..., len(xN))
    n_points: int, number of points per dimension in the new grid

    Returns:
        new_coords_tuple: tuple of new coordinate arrays
        new_pmf: interpolated PMF array
    """
    ndim = len(coords_tuple)

    # Build new uniform grid
    new_coords = []
    for coords in coords_tuple:
        new_coords.append(np.linspace(coords[0], coords[-1], n_points))

    new_coords_tuple = tuple(new_coords)

    # Create interpolator
    interpolator = RegularGridInterpolator(
        coords_tuple,
        pmf,
        bounds_error=False,
        fill_value=np.nan
    )

    # Build meshgrid
    mesh = np.meshgrid(*new_coords_tuple, indexing="ij")
    points = np.stack([m.flatten() for m in mesh], axis=-1)

    # Interpolate
    new_pmf_flat = interpolator(points)
    new_pmf = new_pmf_flat.reshape([n_points] * ndim)

    # Replace NaNs with large values
    nan_mask = np.isnan(new_pmf)
    if np.any(nan_mask):
        max_val = np.nanmax(new_pmf)
        new_pmf[nan_mask] = max_val + 50.0

    return new_coords_tuple, new_pmf


# ============================================================
# 4. Write PMF in your sequential format
# ============================================================

def write_sequential_pmf(coords_tuple, pmf, filename):
    """
    Write an N-dimensional PMF in the same sequential format used by analyze_ND.py.
    """
    ndim = len(coords_tuple)
    shape = pmf.shape

    # Infer grid metadata
    starts = [coords[0] for coords in coords_tuple]
    steps = [coords[1] - coords[0] if len(coords) > 1 else 0.0 for coords in coords_tuple]
    sizes = [len(coords) for coords in coords_tuple]

    with open(filename, "w") as f:
        # Header: number of dimensions
        f.write(f"# {ndim}\n")

        # Header: start, step, size, 0 for each dimension
        header_parts = []
        for start, step, size in zip(starts, steps, sizes):
            header_parts.append(f"{start: .14e}  {step: .14e}  {size:8d}  0")

        f.write("#  " + "   ".join(header_parts) + "\n\n")

        # Flatten grid in C-order
        it = np.nditer(pmf, flags=['multi_index'])
        for val in it:
            idx = it.multi_index
            coords = [coords_tuple[d][idx[d]] for d in range(ndim)]
            line = "  ".join(f"{c: .14e}" for c in coords) + f"   {val: .14e}\n"
            f.write(line)


# ============================================================
# 5. Main function: compute reference PMF + errors
# ============================================================

def compute_reference_pmf(coords_tuple, F_list, T, weights=None):
    """
    Full pipeline:
    - Convert PMFs to probabilities
    - Combine probabilities
    - Convert back to PMF
    - Compute uncertainties
    """
    # Convert each PMF to probability
    P_list = [pmf_to_prob(F, T) for F in F_list]

    # Combine probabilities + compute std dev
    P_ref, P_std = combine_probabilities(P_list, weights)

    # Convert reference probability to PMF
    F_ref = prob_to_pmf(P_ref, T)

    # Convert probability std to PMF std via error propagation:
    # σ_F = (kT / P) * σ_P
    beta = 1.0 / (kB * T)
    F_err = (1.0 / beta) * (P_std / P_ref)

    return F_ref, F_err


# ============================================================
# 6. Plotting (1D only)
# ============================================================

def plot_reference_pmf(coords_tuple, F_ref, F_err, filename="reference_pmf.png"):
    """
    Plot 1D PMF with error bars.
    """
    if len(coords_tuple) != 1:
        print("Plotting is only implemented for 1D PMFs.")
        return

    x = coords_tuple[0]

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, F_ref, yerr=F_err, fmt='-o', markersize=3, capsize=3)
    plt.xlabel("Coordinate")
    plt.ylabel("PMF (kJ/mol)")
    plt.title("Reference PMF with Error Bars")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()