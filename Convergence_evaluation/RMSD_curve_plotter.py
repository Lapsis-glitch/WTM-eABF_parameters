#!/usr/bin/env python3
import os
import math
import matplotlib.pyplot as plt
from Convergence_evaluation.analyze_ND import PMFAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

parent_dir = "/home/rat/Nancy_D/ABF_parameter_test/deca_ala_seed"
pmf_filename = "output/abf_00.abf1.hist.czar.pmf"
count_filename = "output/abf_00.abf1.hist.zcount"

# parent_dir = "/home/rat/Nancy_D/ABF_parameter_test/ethanol_scripted"
# pmf_filename   = "output/window1.abf1.hist.czar.pmf"
# count_filename = "output/window1.abf1.hist.zcount"

reference_pmf = os.path.join(parent_dir, "reference_average_filtered.pmf")

# Convergence settings
n_recent      = 4
slope_thresh  = 1e-3
rmsd_thresh   = 0.01      # fixed RMSD threshold
min_frames    = 6         # minimum snapshots required
use_sliding   = False     # default: final/reference PMF RMSD


# ============================================================
# STEP 1 — Build groups automatically
# ============================================================

groups = {}   # { group_name: [(folder_path, label), ...] }

for entry in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, entry)

    if not os.path.isdir(full_path):
        continue
    if "reference" in entry or "common" in entry:
        continue
    if "_" not in entry:
        continue

    group, label = entry.split("_", 1)
    groups.setdefault(group, []).append((full_path, label))


# ============================================================
# STEP 2 — Sort each group by label (numeric if possible)
# ============================================================

def sort_key(item):
    label = item[1]
    try:
        return float(label)
    except ValueError:
        return label

for g in groups:
    groups[g].sort(key=sort_key)


# ============================================================
# STEP 3 — Filter out invalid or too-short groups
# ============================================================

valid_groups = {}

for group_name, folder_list in groups.items():
    cleaned = []

    for folder_path, label in folder_list:
        pmf_path   = os.path.join(folder_path, pmf_filename)
        count_path = os.path.join(folder_path, count_filename)

        if not (os.path.exists(pmf_path) and os.path.exists(count_path)):
            continue

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

        if len(analyzer.rmsd_raw) < min_frames:
            print(f"Skipping {folder_path}: only {len(analyzer.rmsd_raw)} frames")
            continue

        cleaned.append((folder_path, label))

    if cleaned:
        valid_groups[group_name] = cleaned


if not valid_groups:
    print("No valid PMF/count folders found.")
    exit()


# ============================================================
# STEP 4 — Determine subplot layout
# ============================================================

n_groups = len(valid_groups)
n_cols = math.ceil(math.sqrt(n_groups))
n_rows = math.ceil(n_groups / n_cols)

fig_width  = 4 * n_cols
fig_height = 3 * n_rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
axes = axes.flatten() if n_groups > 1 else [axes]


# ============================================================
# STEP 5 — Plot each group
# ============================================================

for ax_idx, (group_name, folder_list) in enumerate(valid_groups.items()):
    ax = axes[ax_idx]

    for folder_path, label in folder_list:
        pmf_path   = os.path.join(folder_path, pmf_filename)
        count_path = os.path.join(folder_path, count_filename)

        analyzer = PMFAnalyzer(
            pmf_file=pmf_path,
            count_file=count_path,
            n_recent=n_recent,
            slope_thresh=slope_thresh,
            use_sliding_window=use_sliding,
            reference_pmf_file=reference_pmf,
            rmsd_thresh=rmsd_thresh
        )

        ax.plot(analyzer.t, analyzer.rmsd_raw, label=label)

    ax.set_title(f"{group_name} — RMSD Convergence")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("RMSD")
    ax.legend(fontsize=8)
    ax.set_box_aspect(3/4)


# Hide unused axes
for j in range(ax_idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()