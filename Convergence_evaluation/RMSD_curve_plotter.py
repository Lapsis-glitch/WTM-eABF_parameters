#!/usr/bin/env python3
import os
import math
import matplotlib.pyplot as plt
from Convergence_evaluation.analyze import PMFAnalyzer  # your PMFAnalyzer class

# --- CONFIG ---
parent_dir = "/home/rat/Nancy_D/ABF_parameter_test/decaalanine_parameter_scripted"
pmf_filename = "output/abf_00.abf1.hist.czar.pmf"
count_filename = "output/abf_00.abf1.hist.zcount"

parent_dir = "/home/rat/Nancy_D/ABF_parameter_test/ethanol_scripted"
pmf_filename = "output/window1.abf1.hist.czar.pmf"
count_filename = "output/window1.abf1.hist.zcount"

use_final_rmsd = True
n_recent = 4
slope_thresh = 1e-3
min_frames = 6  # minimum PMF snapshots required to plot

# --- STEP 1: Build groups automatically ---
groups = {}  # { group_name: [(folder_path, label), ...] }

for entry in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, entry)
    if not os.path.isdir(full_path):
        continue
    if "reference" in entry or "common" in entry:
        continue
    if "_" not in entry:
        continue

    parts = entry.split("_", 1)
    group_name = parts[0]
    label = parts[1] if len(parts) > 1 else entry

    groups.setdefault(group_name, []).append((full_path, label))

# --- STEP 2: Sort each group by label (numeric if possible) ---
def sort_key(item):
    label = item[1]
    try:
        return float(label)
    except ValueError:
        return label

for g in groups:
    groups[g].sort(key=sort_key)

# --- STEP 3: Filter out empty or too-short groups ---
non_empty_groups = {}
for group_name, folder_list in groups.items():
    valid_folders = []
    for folder_path, label in folder_list:
        pmf_path = os.path.join(folder_path, pmf_filename)
        count_path = os.path.join(folder_path, count_filename)
        if not (os.path.exists(pmf_path) and os.path.exists(count_path)):
            continue

        # Quick length check before creating analyzer
        try:
            analyzer = PMFAnalyzer(
                pmf_file=pmf_path,
                count_file=count_path,
                n_recent=n_recent,
                slope_thresh=slope_thresh,
                use_final_rmsd=use_final_rmsd
            )
        except ValueError as e:
            # In case PMFAnalyzer fails due to too few points
            print(f"Skipping {folder_path}: {e}")
            continue

        if len(analyzer.rmsd_raw) < min_frames:
            print(f"Skipping {folder_path}: only {len(analyzer.rmsd_raw)} frames")
            continue

        valid_folders.append((folder_path, label))

    if valid_folders:
        non_empty_groups[group_name] = valid_folders

n_groups = len(non_empty_groups)
if n_groups == 0:
    print("No groups with valid PMF+count files found.")
    exit()

# --- STEP 4: Auto-compute columns for landscape layout ---
n_cols = math.ceil(math.sqrt(n_groups))
if n_cols < n_groups:  # ensure more columns than rows
    n_cols = min(n_groups, n_cols + 1)
n_rows = math.ceil(n_groups / n_cols)

# Each subplot cell is 4:3 ratio → total figure size:
fig_width = 4 * n_cols
fig_height = 3 * n_rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=False)
axes = axes.flatten() if n_groups > 1 else [axes]

# --- STEP 5: Plot ---
for ax_idx, (group_name, folder_list) in enumerate(non_empty_groups.items()):
    ax = axes[ax_idx]
    for folder_path, label in folder_list:
        pmf_path = os.path.join(folder_path, pmf_filename)
        count_path = os.path.join(folder_path, count_filename)

        analyzer = PMFAnalyzer(
            pmf_file=pmf_path,
            count_file=count_path,
            n_recent=n_recent,
            slope_thresh=slope_thresh,
            use_final_rmsd=use_final_rmsd
        )

        ax.plot(analyzer.t, analyzer.rmsd_raw, label=label)

    ax.set_title(f"{group_name} RMSD to Final Frame")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("RMSD")
    ax.legend(fontsize=8)
    ax.set_box_aspect(3/4)  # height/width ratio for the plot area

# Hide unused axes
for j in range(ax_idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()