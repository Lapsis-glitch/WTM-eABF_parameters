#!/usr/bin/env python3
import argparse
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(
        description="Plot X-Y lines from results.dat, split by Name."
    )
    parser.add_argument(
        "file",
        help="file containing results"
    )
    parser.add_argument(
        "--divisor",
        type=float,
        default=20.0,
        help="Divisor for Y values (default: 20)"
    )
    args = parser.parse_args()

    # Locate results.dat

    if not os.path.isfile(args.file):
        sys.exit(f"Error: results.dat not found in {args.folder}")
    file = args.file

    # Read file
    df = pd.read_csv(file, sep=r"\s+", header=None, names=["Name", "X", "Y"])

    # Remove invalid Y values
    df = df[~df["Y"].isin(["None", "err"])]

    # Convert Y to numeric and scale
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce") / args.divisor
    df = df.dropna(subset=["Y"])

    # Group by Name
    groups = df.groupby("Name")
    num_plots = len(groups)

    # Determine subplot layout
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    # Use a colorblind-friendly palette
    sns.set_palette("colorblind")
    color = sns.color_palette("colorblind")[0]

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = axes.flatten()

    for i, (name, group) in enumerate(groups):
        # Sort by X so the line connects in the right order
        group = group.sort_values(by="X")

        axes[i].plot(group["X"], group["Y"], marker='o', linestyle='-', color=color)
        axes[i].set_title(name)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].set_xlabel(name)
        if i % cols == 0:
            axes[i].set_ylabel("Convergence (ns)")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()