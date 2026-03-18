#!/usr/bin/env python3
import os
import math
import matplotlib.pyplot as plt
from pathlib import Path
from analyze_ND import PMFAnalyzer
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

parent_dir = "/home/lia/gchen/WTM-eABF/deca_ala_seed_100ns/"
parent_dir = "/home/lia/gchen/WTM-eABF/deca_ala_seed_12-32_10ns"
parent_dir = "/home/lia/gchen/WTM-eABF/deca_ala_final/"
pattern = "MTDwidth*"
pmf_filename = "output/abf_00.abf1.hist.czar.pmf"
count_filename = "output/abf_00.abf1.hist.zcount"

#parent_dir = "/home/lia/gchen/WTM-eABF/ethanol_scripted_long"
#pmf_filename = "output/window1.abf1.hist.czar.pmf"
#count_filename = "output/window1.abf1.hist.zcount"

reference_pmf = os.path.join(parent_dir, "reference_median.pmf")

n_recent = 4
slope_thresh = 1e-3
rmsd_thresh = 0.01
min_frames = 6
use_sliding = False


# ---- Helpers ----

def build_groups(base_dir):
    """Scan base_dir and group folders by prefix before the first underscore."""
    groups = {}
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if "reference" in entry or "common" in entry:
            continue
        if "_" not in entry:
            continue
        group, label = entry.split("_", 1)
        groups.setdefault(group, []).append((full_path, label))
    return groups

from pathlib import Path

def build_groups_from_pattern(pattern):
    """
    Use glob pattern to select folders and group them.
    Grouping is based on removing _seed_X if present.
    """
    groups = {}

    for path in Path().glob(pattern):
        if not path.is_dir():
            continue

        name = path.name

        if "reference" in name or "common" in name:
            continue

        # Remove _seed_X if present
        if "_seed_" in name:
            group_name = name.rsplit("_seed_", 1)[0]
            label = name.split("_seed_")[-1]
        else:
            # fallback: split last underscore
            parts = name.split("_")
            if len(parts) > 1:
                group_name = "_".join(parts[:-1])
                label = parts[-1]
            else:
                group_name = name
                label = name

        groups.setdefault(group_name, []).append((str(path), label))

    return groups

def sort_group(folder_list):
    """Sort by numeric label when possible."""
    def key(item):
        try:
            return float(item[1])
        except Exception:
            return item[1]
    return sorted(folder_list, key=key)


def validate_and_collect(base_dir, groups):
    """
    For each group, keep only folders that have both PMF and count files
    and produce a PMFAnalyzer (cached) for reuse.
    Returns dict: group_name -> [(folder_path, label, analyzer), ...]
    """
    analyzers_cache = {}
    valid_groups = {}
    for group_name, folders in groups.items():
        cleaned = []
        for folder_path, label in folders:
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
                    # skip problematic folders
                    print(f"Skipping {folder_path}: {e}")
                    continue
                analyzers_cache[key] = analyzer

            if len(analyzer.rmsd_raw) < min_frames:
                print(f"Skipping {folder_path}: only {len(analyzer.rmsd_raw)} frames")
                continue

            cleaned.append((folder_path, label, analyzer))

        if cleaned:
            # Preserve analyzers and keep sorted order (by label)
            valid_groups[group_name] = sort_group(cleaned)

    return valid_groups


# ---- Main flow ----
#groups = build_groups(parent_dir)
groups = build_groups_from_pattern(pattern)
valid_groups = validate_and_collect(None, groups)

if not valid_groups:
    print("No valid PMF/count folders found.")
    exit()

# Prepare subplot layout
n_groups = len(valid_groups)
n_cols = math.ceil(math.sqrt(n_groups))
n_rows = math.ceil(n_groups / n_cols)

fig_width = 4 * n_cols
fig_height = 3 * n_rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

# Normalize axes to a 1D list for easy indexing (handle ndarray, single Axes, list)
if hasattr(axes, "flatten"):
    axes_list = list(axes.flatten())
elif isinstance(axes, (list, tuple)):
    axes_list = list(axes)
else:
    axes_list = [axes]

# Plot each group
color_cycle = plt.get_cmap("tab10")

group_items = list(valid_groups.items())
used_count = len(group_items)

for ax_idx, (group_name, folder_list) in enumerate(group_items):
    ax = axes_list[ax_idx]
    for idx, (folder_path, label, analyzer) in enumerate(folder_list):
        # re-use analyzer already created in validation
        ax.plot(analyzer.t, analyzer.rmsd_raw, label=label,
                color=color_cycle(idx % 10), linewidth=1.5)

    ax.set_title(f"{group_name} — RMSD Convergence")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("RMSD")
    ax.legend(fontsize=8, loc='best')
    ax.set_aspect('auto')

# Hide unused axes (based on used_count)
for j in range(used_count, len(axes_list)):
    fig.delaxes(axes_list[j])

plt.tight_layout()
plt.savefig("rmsd.png")
plt.show()