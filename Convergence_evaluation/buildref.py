#!/usr/bin/env python3
import os
import argparse
import numpy as np

from reference_builder import (
    compute_reference_pmf,
    write_sequential_pmf,
    plot_reference_pmf,
    interpolate_pmf
)

# ------------------------------------------------------------
# Helper: read PMFs using your existing analyze_ND logic
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
    compute reference PMF + errors, and save results.
    """

    pmf_filename = f"{name}.czar.pmf"
    pmf_paths = []

    # Only check immediate subdirectories
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
            # Optional: check consistency
            if any(len(c1) != len(c2) for c1, c2 in zip(coords_tuple, coords_interp)):
                print("Warning: grid mismatch after interpolation")

        F_list.append(F_interp)

    # Compute reference PMF + errors
    F_ref, F_err = compute_reference_pmf(coords_tuple, F_list, temperature)

    # Save outputs in the base directory
    out_pmf = os.path.join(base_dir, "reference.czar.pmf")
    out_err = os.path.join(base_dir, "reference.czar.errors")

    write_sequential_pmf(coords_tuple, F_ref, out_pmf)
    write_sequential_pmf(coords_tuple, F_err, out_err)

    print(f"Saved reference PMF to: {out_pmf}")
    print(f"Saved reference errors to: {out_err}")

    # Plot (only for 1D)
    plot_reference_pmf(coords_tuple, F_ref, F_err,
                       filename=os.path.join(base_dir, "reference_pmf.png"))

    print("Plot saved as reference_pmf.png")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute reference PMF from multiple simulations."
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