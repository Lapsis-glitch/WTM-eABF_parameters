#!/usr/bin/env python3
"""
plot_results.py
---------------
Plot mean convergence time with standard-deviation error bands from a
``results.dat`` file produced by :mod:`folder_parser`.
Each row in the input has the format::
    BaseName  Mean  Std  Min  Max  N
The *BaseName* is split on ``_`` to extract a numeric X value (last token)
and a group key (everything before).  Groups are plotted in a grid of
subplots.
Usage
-----
::
    python plot_results.py results.dat --divisor 20
"""
import argparse
import math
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotting_config import configure_plotting, LABELS
configure_plotting()
sns.set_palette("colorblind")
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot mean convergence with std deviation from results.dat."
    )
    parser.add_argument("file", help="Path to results.dat")
    parser.add_argument("--divisor", type=float, default=20.0,
                        help="Divisor for Y values (default: 20)")
    args = parser.parse_args()
    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found: {args.file}")
    # ---- Read & prepare data ----
    df = pd.read_csv(
        args.file, sep=r"\s+", header=None,
        names=["Name", "Mean", "Std", "Min", "Max", "N"],
    )
    for col in ("Mean", "Std", "Min", "Max"):
        df[col] = pd.to_numeric(df[col], errors="coerce") / args.divisor
    df = df.dropna(subset=["Mean", "Std", "Min", "Max"])
    # Extract numeric X from the last token of the name
    df["X"] = df["Name"].apply(lambda x: x.split("_")[-1])
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df = df.dropna(subset=["X"])
    # Group by base name (everything except the last token)
    df["Base"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:-1]))
    groups = df.groupby("Base")
    # ---- Subplot layout ----
    n_plots = len(groups)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    colors = sns.color_palette("colorblind")
    fig, axes = plt.subplots(rows, cols,
                             figsize=(7, 7 / cols * rows), sharey=True)
    axes = axes.flatten()
    for i, (name, group) in enumerate(groups):
        group = group.sort_values("X")
        # Mean curve
        axes[i].plot(group["X"], group["Mean"],
                     marker="o", linestyle="-", color=colors[0])
        # Highlight under-sampled points (N < 3)
        mask_low = group["N"] < 3
        axes[i].scatter(group["X"].values[mask_low],
                        group["Mean"].values[mask_low],
                        marker="o", color=colors[3], zorder=10)
        # +/-1 std band
        axes[i].fill_between(group["X"],
                             group["Mean"] - group["Std"],
                             group["Mean"] + group["Std"],
                             color=colors[0], alpha=0.2)
        # Min-max band
        axes[i].fill_between(group["X"], group["Min"], group["Max"],
                             color=colors[4], alpha=0.2)
        axes[i].grid(True, linestyle="-", color="lightgray")
        axes[i].set_xlabel(LABELS.get(name, name))
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.supylabel("Convergence (ns)", fontsize=12)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    plt.savefig("convergence.png", dpi=400)
if __name__ == "__main__":
    main()
