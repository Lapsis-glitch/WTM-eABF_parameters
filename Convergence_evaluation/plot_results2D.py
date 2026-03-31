#!/usr/bin/env python3
"""
plot_results2D.py
-----------------
Plot a 2-D heatmap of mean convergence time and its standard deviation from a
``results.dat`` file whose base names encode **two** parameters, e.g.::
    Param1_1.0_Param2_2.0  12.5  3.1  8.0  17.0  5
Usage
-----
::
    python plot_results2D.py results.dat --divisor 20
"""
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401 (palette)
from plotting_config import configure_plotting, LABELS
configure_plotting()
# Regex: <Name1>_<Val1>_<Name2>_<Val2>
_PARAM_RE = re.compile(r"([A-Za-z0-9]+)_([0-9.]+)_([A-Za-z0-9]+)_([0-9.]+)")
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="2-D heatmap of convergence results (two-parameter sweep)."
    )
    parser.add_argument("file", help="Path to results.dat")
    parser.add_argument("--divisor", type=float, default=20.0,
                        help="Divisor for Y values (default: 20)")
    args = parser.parse_args()
    # ---- Parse results file ----
    p1_vals, p2_vals, mean_vals, std_vals = [], [], [], []
    p1_name = p2_name = ""
    with open(args.file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            match = _PARAM_RE.match(parts[0])
            if not match:
                continue
            p1_name, v1, p2_name, v2 = match.groups()
            try:
                p1_vals.append(float(v1))
                p2_vals.append(float(v2))
                mean_vals.append(float(parts[1]))
                std_vals.append(float(parts[2]))
            except ValueError:
                continue
    if not p1_vals:
        print("No valid two-parameter entries found.")
        return
    # ---- Build grids ----
    p1_unique = np.unique(p1_vals)
    p2_unique = np.unique(p2_vals)
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)
    mean_grid = np.full(p1_grid.shape, np.nan)
    std_grid = np.full(p1_grid.shape, np.nan)
    for x, y, m, s in zip(p1_vals, p2_vals, mean_vals, std_vals):
        i = np.where(p2_unique == y)[0][0]
        j = np.where(p1_unique == x)[0][0]
        mean_grid[i, j] = m / args.divisor
        std_grid[i, j] = s / args.divisor
    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    im0 = axes[0].contourf(p1_grid, p2_grid, mean_grid, levels=40, cmap="viridis")
    axes[0].set_xlabel(LABELS.get(p1_name, p1_name))
    axes[0].set_ylabel(LABELS.get(p2_name, p2_name))
    fig.colorbar(im0, ax=axes[0]).set_label("Mean Convergence Time (ns)")
    im1 = axes[1].contourf(p1_grid, p2_grid, std_grid, levels=40, cmap="magma")
    axes[1].set_xlabel(LABELS.get(p1_name, p1_name))
    axes[1].set_ylabel(LABELS.get(p2_name, p2_name))
    fig.colorbar(im1, ax=axes[1]).set_label("Standard Deviation (ns)")
    plt.tight_layout()
    plt.savefig("convergence_2D.png", dpi=300)
if __name__ == "__main__":
    main()
