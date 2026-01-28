#!/usr/bin/env python3
import os
import argparse
import numpy as np

from reference_builder import (
    interpolate_pmf,
    write_sequential_pmf,
    compute_reference_pmf_with_outliers,
)

# ------------------------------------------------------------
# Helper: read PMFs in sequential format
# ------------------------------------------------------------

def read_sequential_pmf_file(filename):
    """
    Reads a single PMF file in your sequential format.
    Returns (coords_tuple, pmf_array).
    """
    tmp = []
    coords_tuple = None
    pmf = None

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                if tmp:
                    arr = np.array(tmp, float)
                    coords = [np.unique(arr[:, i]) for i in range(arr.shape[1]-1)]
                    shape = tuple(len(c) for c in coords)
                    pmf = arr[:, -1].reshape(shape)
                    coords_tuple = tuple(coords)
                    tmp = []
                continue

            if line.strip():
                tmp.append(line.split())

    if tmp:
        arr = np.array(tmp, float)
        coords = [np.unique(arr[:, i]) for i in range(arr.shape[1]-1)]
        shape = tuple(len(c) for c in coords)
        pmf = arr[:, -1].reshape(shape)
        coords_tuple = tuple(coords)

    return coords_tuple, pmf


# ------------------------------------------------------------
# Main run function
# ------------------------------------------------------------

def run(base_dir, temperature=300, name="abf_00.abf1", n_points=100):
    """
    Look inside each immediate subdirectory of base_dir.
    In each subdirectory, look for:
        {name}.czar.pmf

    Collect all PMFs, interpolate them to a common grid,
    compute:
        - median PMF
        - average PMF (all)
        - average PMF (filtered)
    and save results + PNG plots.
    """

    pmf_filename = f"{name}.czar.pmf"
    pmf_paths = []

    # Scan subdirectories
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, pmf_filename)
            if os.path.isfile(candidate):
                pmf_paths.append(candidate)

    if not pmf_paths:
        print(f"No PMFs named {pmf_filename} found in subdirectories of {base_dir}")
        return

    print(f"Found {len(pmf_paths)} PMFs:")
    for p in pmf_paths:
        print("  ", p)

    # Read + interpolate all PMFs
    coords_tuple = None
    F_list = []

    for path in pmf_paths:
        coords, F = read_sequential_pmf_file(path)

        # Interpolate to consistent grid
        coords_interp, F_interp = interpolate_pmf(coords, F, n_points)

        if coords_tuple is None:
            coords_tuple = coords_interp
        else:
            if any(len(c1) != len(c2) for c1, c2 in zip(coords_tuple, coords_interp)):
                print("Warning: grid mismatch after interpolation")

        F_list.append(F_interp)

    # --------------------------------------------------------
    # Compute robust reference PMFs (median + averages)
    # --------------------------------------------------------
    results = compute_reference_pmf_with_outliers(
        coords_tuple,
        F_list,
        T=temperature,
        mad_cut=3.5,
        write_prefix=os.path.join(base_dir, "reference")
    )

    print("Saved:")
    print("  reference_median.pmf")
    print("  reference_average_all.pmf")
    print("  reference_average_filtered.pmf")
    print("  reference_pmf_comparison.png")
    print("  reference_outlier_diagnostics.png")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute robust reference PMF from multiple simulations."
    )
    parser.add_argument("--dir", required=True,
                        help="Base directory containing subdirectories with PMFs.")
    parser.add_argument("--temp", type=float, default=300,
                        help="Temperature in Kelvin (default: 300).")
    parser.add_argument("--name", type=str, default="abf_00.abf1",
                        help="Prefix of PMF files (default: abf_00.abf1).")
    parser.add_argument("--npoints", type=int, default=100,
                        help="Number of grid points per dimension (default: 100).")

    args = parser.parse_args()

    run(args.dir, temperature=args.temp, name=args.name, n_points=args.npoints)