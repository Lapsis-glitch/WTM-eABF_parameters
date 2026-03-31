#!/usr/bin/env python3
"""
buildref.py
-----------
CLI tool to compute a robust reference PMF from multiple simulation replicas.
Scans subdirectories of a base directory for PMF files matching a given name,
interpolates them onto a common uniform grid, then delegates to
``reference_builder.compute_reference_pmf_with_outliers`` for statistical
aggregation, outlier rejection, and plotting.
Usage
-----
::
    python buildref.py --dir /path/to/base --npoints 100 --temp 300
Output files (written into *base_dir*):
  * ``reference_median.pmf``
  * ``reference_average_all.pmf``  /  ``…_err.pmf``
  * ``reference_average_filtered.pmf``  /  ``…_err.pmf``
  * ``reference_pmf_comparison.pdf``
  * ``reference_outlier_diagnostics.png``
"""
import os
import argparse
from pmf_io import read_sequential_pmf, interpolate_pmf
from reference_builder import compute_reference_pmf_with_outliers
# ============================================================
# Core routine
# ============================================================
def run(base_dir: str, temperature: float = 300, name: str = "abf_00.abf1",
        n_points: int = 100) -> None:
    """
    Collect PMFs from subdirectories, interpolate, and build reference PMFs.
    Parameters
    ----------
    base_dir : str
        Parent directory whose immediate children contain PMF files.
    temperature : float
        Simulation temperature in Kelvin.
    name : str
        PMF file stem (looks for ``{name}.czar.pmf`` in each subdirectory).
    n_points : int
        Number of grid points per dimension for the common interpolation grid.
    """
    pmf_filename = f"{name}.czar.pmf"
    pmf_paths = []
    # Scan immediate subdirectories for the target PMF file
    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, pmf_filename)
            if os.path.isfile(candidate):
                pmf_paths.append(candidate)
    if not pmf_paths:
        print(f"No PMFs named '{pmf_filename}' found in subdirectories of {base_dir}")
        return
    print(f"Found {len(pmf_paths)} PMFs:")
    for p in pmf_paths:
        print(f"  {p}")
    # Read and interpolate all PMFs onto a common grid
    coords_tuple = None
    F_list = []
    for path in pmf_paths:
        coords, F = read_sequential_pmf(path)
        # Interpolate to a uniform grid with *n_points* bins per dimension
        coords_interp, F_interp = interpolate_pmf(coords, F, n_points)
        if coords_tuple is None:
            coords_tuple = coords_interp
        F_list.append(F_interp)
    # Compute robust reference PMFs (median + averages) and write output
    compute_reference_pmf_with_outliers(
        coords_tuple,
        F_list,
        T=temperature,
        mad_cut=3.5,
        write_prefix=os.path.join(base_dir, "reference"),
    )
    print("\nSaved:")
    print("  reference_median.pmf")
    print("  reference_average_all.pmf")
    print("  reference_average_filtered.pmf")
    print("  reference_pmf_comparison.pdf")
    print("  reference_outlier_diagnostics.png")
# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute robust reference PMF from multiple simulations."
    )
    parser.add_argument("--dir", required=True,
                        help="Base directory containing subdirectories with PMFs.")
    parser.add_argument("--temp", type=float, default=300,
                        help="Temperature in Kelvin (default: 300).")
    parser.add_argument("--name", type=str, default="abf_00.abf1",
                        help="PMF file stem, e.g. 'abf_00.abf1' (default).")
    parser.add_argument("--npoints", type=int, default=100,
                        help="Grid points per dimension (default: 100).")
    args = parser.parse_args()
    run(args.dir, temperature=args.temp, name=args.name, n_points=args.npoints)
if __name__ == "__main__":
    main()
