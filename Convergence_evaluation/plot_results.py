#!/usr/bin/env python3
import argparse
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.font_manager as font_manager
def load_matplotlib_local_fonts():

    # Load a font from TTF file, 
    # relative to this Python module
    # https://stackoverflow.com/a/69016300/315168
    #font_path = os.path.join(os.path.dirname(__file__), '/home/lia/gchen/miniconda3/envs/nn4/fonts/arial.ttf')
    font_path = '/home/lia/gchen/miniconda3/envs/nn/fonts/arial.ttf'
    assert os.path.exists(font_path)
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    #  Set it as default matplotlib font
    matplotlib.rc('font', family='sans-serif') 
    matplotlib.rcParams.update({
        'font.size': 12,
        'font.sans-serif': prop.get_name(),
    })
try:
    load_matplotlib_local_fonts()
except:
    pass

labels = {"MTDheight": "hillWeight",
          "MTDnewhill": "newHillFrequency",
          "MTDtemp": "biasTemperature",
          "MTDwidth": "hillWidth",
          "colvarWidth": "colvarWidth",
          "extDamp": "extendedLangevinDamping",
          "extFluc": "extendedFluctuation",
          "extTime": "extendedTimeConstant",
          "fullSamp": "fullSamples",
          }

def main():
    """Plot mean convergence with standard deviation error bars."""
    parser = argparse.ArgumentParser(
        description="Plot mean convergence with std deviation from results.dat."
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

    # Locate results file
    file_path = args.file
    if not os.path.isfile(file_path):
        sys.exit(f"Error: results file not found: {file_path}")

    # Read file (new format)
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["Name", "Mean", "Std", "Min", "Max", "N"]
    )

    # Convert to numeric and scale
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce") / args.divisor
    df["Std"] = pd.to_numeric(df["Std"], errors="coerce") / args.divisor
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce") / args.divisor
    df["Max"] = pd.to_numeric(df["Max"], errors="coerce") / args.divisor

    df = df.dropna(subset=["Mean", "Std", "Min", "Max"])

    # Optional: extract X from Name if needed
    # Assumes format like: something_X
    df["X"] = df["Name"].apply(lambda x: x.split("_")[-1])
    df["X"] = pd.to_numeric(df["X"], errors="coerce")

    # Drop rows where X couldn't be parsed
    df = df.dropna(subset=["X"])

    # Group by base name (everything except last part)
    df["Base"] = df["Name"].apply(lambda x: "_".join(x.split("_")[:-1]))
    groups = df.groupby("Base")

    num_plots = len(groups)

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    sns.set_palette("colorblind")
    colors = sns.color_palette("colorblind")
    color = colors[0]

    #fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    fig, axes = plt.subplots(rows, cols, figsize=(7, 7/ cols * rows), sharey=True)
    axes = axes.flatten()

    for i, (name, group) in enumerate(groups):
        group = group.sort_values(by="X")

        axes[i].plot(
            group["X"],
            group["Mean"],
            #yerr=group["Std"],
            marker='o',
            linestyle='-',
            color=color,
            #capsize=4
        )

        axes[i].scatter(
            group["X"].values[group["N"] < 3],
            group["Mean"].values[group["N"] < 3],
            #yerr=group["Std"],
            marker='o',
            color=colors[3],
            #capsize=4
            zorder=10,
        )

        axes[i].fill_between(
            group["X"],
            group["Mean"] - group["Std"],
            group["Mean"] + group["Std"],
            color=color,
            alpha=0.2
        )
        
        axes[i].fill_between(
            group["X"],
            group["Min"],
            group["Max"],
            color=colors[4],
            alpha=0.2
        )

        #axes[i].set_title(name)
        axes[i].grid(True, linestyle='-', color="lightgray", #alpha=0.6
        )
        axes[i].set_xlabel(labels[name])


    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.supylabel("Convergence (ns)", fontsize=12)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    plt.savefig("convergence.png", dpi=400)
    # plt.show()


if __name__ == "__main__":
    main()