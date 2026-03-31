#!/usr/bin/env python3
"""
RMSD_curve_plotter.py
---------------------
Overlay RMSD convergence curves from multiple simulation folders, grouped by
a glob pattern.  Folders matching the pattern are grouped by stripping the
``_seed_X`` suffix (or by splitting at the last ``_``).
Usage
-----
::
    python RMSD_curve_plotter.py --pattern "MTDwidth*" \\
        --pmf output/abf_00.abf1.hist.czar.pmf \\
        --counts output/abf_00.abf1.hist.zcount \\
        --reference reference_median.pmf
"""
import argparse
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from analyze_ND import PMFAnalyzer
from plotting_config import configure_plotting
configure_plotting()
# ============================================================
# Helpers
# ============================================================
def build_groups(pattern: str):
    """
    Glob *pattern* for directories and group them.
    Grouping strips ``_seed_X`` when present; otherwise splits on the last
    underscore.  Returns ``{group_name: [(path, label), ...]}``.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(Path().glob(pattern)):
        if not path.is_dir():
            continue
        name = path.name
        if "reference" in name or "common" in name:
            continue
        if "_seed_" in name:
            group_name = name.rsplit("_seed_", 1)[0]
            label = name.split("_seed_")[-1]
        else:
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
    """Sort entries by numeric label when possible."""
    def key(item):
        try:
            return float(item[1])
        except (ValueError, TypeError):
            return item[1]
    return sorted(folder_list, key=key)
def validate_and_collect(groups, pmf_rel, count_rel, ref_pmf,
                         n_recent, slope_thresh, rmsd_thresh,
                         use_sliding, min_frames):
    """
    Filter groups to folders that have valid PMF + count files and create
    :class:`PMFAnalyzer` instances (cached).
    Returns ``{group: [(path, label, analyzer), ...]}``.
    """
    cache: dict[tuple, PMFAnalyzer] = {}
    valid: dict[str, list] = {}
    for group_name, folders in groups.items():
        cleaned = []
        for folder_path, label in folders:
            pmf_path = os.path.join(folder_path, pmf_rel)
            count_path = os.path.join(folder_path, count_rel)
            if not (os.path.exists(pmf_path) and os.path.exists(count_path)):
                continue
            key = (pmf_path, count_path)
            if key not in cache:
                try:
                    cache[key] = PMFAnalyzer(
                        pmf_file=pmf_path,
                        count_file=count_path,
                        n_recent=n_recent,
                        slope_thresh=slope_thresh,
                        use_sliding_window=use_sliding,
                        reference_pmf_file=ref_pmf,
                        rmsd_thresh=rmsd_thresh,
                    )
                except Exception as e:
                    print(f"Skipping {folder_path}: {e}")
                    continue
            analyzer = cache[key]
            if len(analyzer.rmsd_raw) < min_frames:
                print(f"Skipping {folder_path}: only {len(analyzer.rmsd_raw)} frames")
                continue
            cleaned.append((folder_path, label, analyzer))
        if cleaned:
            valid[group_name] = sort_group(cleaned)
    return valid
# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Overlay RMSD curves by group.")
    parser.add_argument("--pattern", default="*",
                        help="Glob pattern to match simulation folders (default: '*')")
    parser.add_argument("--pmf", default="output/abf_00.abf1.hist.czar.pmf",
                        help="Relative path to PMF file inside each folder")
    parser.add_argument("--counts", default="output/abf_00.abf1.hist.zcount",
                        help="Relative path to count file inside each folder")
    parser.add_argument("--reference", default=None,
                        help="Path to a reference PMF file")
    parser.add_argument("--n-recent", type=int, default=4)
    parser.add_argument("--slope-thresh", type=float, default=1e-3)
    parser.add_argument("--rmsd-thresh", type=float, default=0.01)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--use-sliding", action="store_true")
    parser.add_argument("--output", default="rmsd.png",
                        help="Output figure filename (default: rmsd.png)")
    args = parser.parse_args()
    groups = build_groups(args.pattern)
    valid_groups = validate_and_collect(
        groups, args.pmf, args.counts, args.reference,
        args.n_recent, args.slope_thresh, args.rmsd_thresh,
        args.use_sliding, args.min_frames,
    )
    if not valid_groups:
        print("No valid PMF/count folders found.")
        return
    # ---- Subplot layout ----
    n_groups = len(valid_groups)
    n_cols = math.ceil(math.sqrt(n_groups))
    n_rows = math.ceil(n_groups / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows))
    axes_list = list(np.array(axes).flatten()) if n_groups > 1 else [axes]
    cmap = plt.get_cmap("tab10")
    for ax_idx, (group_name, folder_list) in enumerate(valid_groups.items()):
        ax = axes_list[ax_idx]
        for idx, (_, label, analyzer) in enumerate(folder_list):
            ax.plot(analyzer.t, analyzer.rmsd_raw, label=label,
                    color=cmap(idx % 10), linewidth=1.5)
        ax.set_title(f"{group_name}")
        ax.set_xlabel("Snapshot Index")
        ax.set_ylabel("RMSD")
        ax.legend(fontsize=8, loc="best")
    # Hide unused axes
    for j in range(len(valid_groups), len(axes_list)):
        fig.delaxes(axes_list[j])
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved {args.output}")
if __name__ == "__main__":
    main()
