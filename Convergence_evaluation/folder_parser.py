#!/usr/bin/env python3
"""
folder_parser.py
----------------
Batch convergence analysis over a directory tree of WTM-eABF simulations.

For every subfolder that contains PMF + count history files, a
:class:`~analyze_ND.PMFAnalyzer` is constructed and its convergence index is
recorded.  Results are grouped by a *base name* (folder name without the
``_seed_X`` suffix) and summarised as mean / std / min / max.

Two naming modes are supported via ``--mode``:

* **1d** (default) — strip the ``_seed_X`` suffix to get the base name.
* **2d** — use a regex to extract multi-parameter base names of the form
  ``Param1_Value1_Param2_Value2``.

Usage
-----
::

    python folder_parser.py /path/to/runs --reference-pmf ref.pmf
    python folder_parser.py /path/to/runs --reference-pmf ref.pmf --mode 2d
"""

import argparse
import os
import re
import numpy as np
from collections import defaultdict

from analyze_ND import PMFAnalyzer


# ============================================================
# Name extraction helpers
# ============================================================

def extract_base_name_1d(folder: str) -> str:
    """Remove ``_seed_X`` suffix from a folder name."""
    if "_seed_" in folder:
        return folder.rsplit("_seed_", 1)[0]
    return folder


def extract_base_name_2d(folder: str) -> str:
    """
    Extract the multi-parameter portion of a folder name.

    Example: ``Param1_1.0_Param2_2.0_seed_42`` → ``Param1_1.0_Param2_2.0``
    """
    folder_no_seed = folder.rsplit("_seed_", 1)[0]
    match = re.match(r"^([A-Za-z0-9.]+(?:_[A-Za-z0-9.]+)+)", folder_no_seed)
    return match.group(1) if match else folder_no_seed


# ============================================================
# Main parsing routine
# ============================================================

def parse(path: str, reference_pmf: str | None, mode: str = "1d") -> None:
    """
    Scan *path* for simulation folders, compute convergence, and write
    ``results.dat``.

    Parameters
    ----------
    path : str
        Root directory containing one subfolder per simulation run.
    reference_pmf : str or None
        Path to an external reference PMF file (optional).
    mode : str
        ``'1d'`` or ``'2d'`` — controls how folder names map to group keys.
    """
    extract_fn = extract_base_name_2d if mode == "2d" else extract_base_name_1d

    print(f"Scanning: {path}  (mode={mode})")

    grouped_results: dict[str, list] = defaultdict(list)

    # --- First pass: compute convergence per folder ---
    for folder in sorted(os.listdir(path)):
        if folder.startswith(("reference", "common")):
            continue

        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        base_name = extract_fn(folder)

        try:
            # Detect the PMF filename convention used in this folder
            candidate = os.path.join(folder_path, "output", "abf_00.abf1.hist.czar.pmf")
            if os.path.isfile(candidate):
                stem = os.path.join(folder_path, "output", "abf_00.abf1.hist.")
            else:
                stem = os.path.join(folder_path, "output", "window1.abf1.hist.")

            analyzer = PMFAnalyzer(
                stem + "czar.pmf",
                stem + "count",
                slope_thresh=0.01,
                n_recent=5,
                use_sliding_window=False,
                count_std_thresh=None,
                reference_pmf_file=reference_pmf,
                rmsd_thresh=0.592186869182,
                use_ref_and_slope=True,
            )

            grouped_results[base_name].append(analyzer.convergence_idx)

        except Exception:
            continue  # skip problematic folders silently

    # --- Second pass: aggregate stats and write output ---
    results_file = os.path.join(path, "results.dat")
    with open(results_file, "w") as f:
        for base_name, values in sorted(grouped_results.items()):
            clean = [v for v in values if v is not None]
            if not clean:
                print(f"  {base_name}: skipped (no valid convergence)")
                continue

            mean_val = np.mean(clean)
            std_val = np.std(clean)
            min_val = np.min(clean)
            max_val = np.max(clean)
            n = len(clean)

            print(f"  {base_name}  mean={mean_val:.3f}  std={std_val:.3f}  "
                  f"min={min_val:.3f}  max={max_val:.3f}  (n={n})")
            f.write(f"{base_name} {mean_val:.3f} {std_val:.3f} "
                    f"{min_val:.3f} {max_val:.3f} {n}\n")

    print(f"Results written to {results_file}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch convergence analysis over a directory of ABF simulations."
    )
    parser.add_argument("path", help="Root directory containing simulation folders.")
    parser.add_argument("--reference-pmf", type=str, default=None,
                        help="External reference PMF for RMSD computation.")
    parser.add_argument("--mode", choices=["1d", "2d"], default="1d",
                        help="Name-extraction mode: '1d' (strip seed) or "
                             "'2d' (regex multi-param). Default: 1d.")
    args = parser.parse_args()
    parse(args.path, args.reference_pmf, mode=args.mode)


if __name__ == "__main__":
    main()

