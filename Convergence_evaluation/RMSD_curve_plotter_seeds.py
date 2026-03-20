#!/usr/bin/env python3
import os
import math
import matplotlib.pyplot as plt
from analyze_ND import PMFAnalyzer
import numpy as np
import seaborn as sns
from collections import defaultdict
from matplotlib.lines import Line2D
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

# ============================================================
# CONFIGURATION
# ============================================================

parent_dir = "./"
pmf_filename = "output/window1.abf1.hist.czar.pmf"
count_filename = "output/window1.abf1.hist.zcount"
divisor = 5.0

reference_pmf = os.path.join(parent_dir, "reference_median.pmf")

n_recent = 4
slope_thresh = 1e-3
rmsd_thresh = 0.01
min_frames = 6
use_sliding = False


# ============================================================
# HELPERS
# ============================================================

def parse_folder_name(name):
    parts = name.split("_")
    if len(parts) < 2:
        return None

    prefix = parts[0]

    try:
        X = float(parts[1])
    except ValueError:
        X = parts[1]

    seed = None
    if len(parts) >= 4 and parts[2] == "seed":
        try:
            seed = int(parts[3])
        except ValueError:
            seed = parts[3]

    return prefix, X, seed


def build_groups(base_dir):
    groups = defaultdict(list)

    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)

        if not os.path.isdir(full_path):
            continue
        if "reference" in entry or "common" in entry:
            continue

        parsed = parse_folder_name(entry)
        if parsed is None:
            continue

        prefix, X, seed = parsed
        groups[prefix].append((full_path, X, seed))

    return groups


def sort_group(folder_list):
    def key(item):
        _, X, seed, _ = item
        return (X, seed if seed is not None else -1)

    return sorted(folder_list, key=key)


def validate_and_collect(base_dir, groups):
    analyzers_cache = {}
    valid_groups = {}

    for prefix, folders in groups.items():
        cleaned = []

        for folder_path, X, seed in folders:
            pmf_path = os.path.join(folder_path, pmf_filename)
            count_path = os.path.join(folder_path, count_filename)

            if not (os.path.exists(pmf_path) and os.path.exists(count_path)):
                continue

            key = (pmf_path, count_path)

            if key in analyzers_cache:
                analyzer = analyzers_cache[key]
            else:
                try:
                    analyzer = PMFAnalyzer(
                        pmf_file=pmf_path,
                        count_file=count_path,
                        n_recent=n_recent,
                        slope_thresh=slope_thresh,
                        use_sliding_window=use_sliding,
                        reference_pmf_file=reference_pmf,
                        rmsd_thresh=rmsd_thresh
                    )
                except Exception as e:
                    print(f"Skipping {folder_path}: {e}")
                    continue

                analyzers_cache[key] = analyzer

            if len(analyzer.rmsd_raw) < min_frames:
                print(f"Skipping {folder_path}: only {len(analyzer.rmsd_raw)} frames")
                continue

            cleaned.append((folder_path, X, seed, analyzer))

        if cleaned:
            valid_groups[prefix] = sort_group(cleaned)

    return valid_groups


# ============================================================
# MAIN
# ============================================================

groups = build_groups(parent_dir)
valid_groups = validate_and_collect(parent_dir, groups)

if not valid_groups:
    print("No valid PMF/count folders found.")
    exit()


# ---- Subplots ----

n_groups = len(valid_groups)
n_cols = math.ceil(math.sqrt(n_groups))
n_rows = math.ceil(n_groups / n_cols)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(7, 7/ n_cols * n_rows), sharex=True)

if hasattr(axes, "flatten"):
    axes_list = list(axes.flatten())
else:
    axes_list = [axes]


# ---- Colors ----

palette = sns.color_palette("colorblind", n_colors=10)
linestyles = ['-', (0, (5, 1)), (0, (5, 5)), (0, (5, 10)),(0, (1, 1)), (0, (1, 5)), (0, (1, 10))]
white = np.array([1.0, 1.0, 1.0])


# ---- Plot ----

group_items = list(valid_groups.items())

for ax_idx, (prefix, folder_list) in enumerate(group_items):
    ax = axes_list[ax_idx]

    by_X = defaultdict(list)
    for folder_path, X, seed, analyzer in folder_list:
        by_X[X].append((seed, analyzer))

    sorted_X = sorted(by_X.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else x)

    legend_handles = []

    for color_idx, X in enumerate(sorted_X):
        base_color = np.array(palette[color_idx % len(palette)])
        entries = by_X[X]

        entries = sorted(entries, key=lambda x: x[0] if x[0] is not None else -1)
        n_seeds = len(entries)

        # ---- Plot all seeds (no legend labels) ----
        for i, (seed, analyzer) in enumerate(entries):

            linestyle = linestyles[i % len(linestyles)]

            ax.plot(
                analyzer.t / divisor,
                analyzer.rmsd_raw,
                color=base_color,
                linestyle=linestyle,
                linewidth=1.5
            )

        # ---- Add ONE legend entry per X (base color) ----
        legend_handles.append(
            Line2D([0], [0],
                   color=base_color,
                   lw=2,
                   label=f"{X}")
        )

    ax.set_title(f"{labels[prefix]}", fontsize=9)
    #ax.set_xlabel("Time (ns)")
    #ax.set_ylabel("RMSD (kcal/mol)")
    ax.legend(handles=legend_handles, fontsize=8, loc='best',ncols=2, columnspacing=0.5)
    ax.set_aspect('auto')


# ---- Hide unused axes ----

for j in range(len(group_items), len(axes_list)):
    fig.delaxes(axes_list[j])

fig.supxlabel("Time (ns)")
fig.supylabel("RMSD (kcal/mol)")
plt.tight_layout()
#fig.subplots_adjust(wspace=0.2)

plt.savefig("RMSD_convergence_seeds.png", dpi=300)