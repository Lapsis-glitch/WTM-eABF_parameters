#!/usr/bin/env python3
"""
RMSD_curve_plotter_seeds.py
---------------------------
Overlay RMSD convergence curves from seeded simulation replicas.
Folders are expected to follow the naming convention::
    <Prefix>_<Value>_seed_<N>
All seeds sharing the same ``(Prefix, Value)`` are overlaid on one subplot
with the same colour but different line-styles.  Each *Prefix* gets its own
subplot panel.
Usage
-----
::
    python RMSD_curve_plotter_seeds.py --dir ./ \\
        --pmf output/window1.abf1.hist.czar.pmf \\
        --counts output/window1.abf1.hist.zcount \\
        --reference reference_median.pmf \\
        --divisor 5
"""
import argparse
import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from analyze_ND import PMFAnalyzer
from plotting_config import configure_plotting, LABELS
configure_plotting()
# ============================================================
# Folder-name parsing
# ============================================================
def parse_folder_name(name: str):
    """
    Parse ``<Prefix>_<Value>[_seed_<N>]`` into ``(prefix, value, seed)``.
    Returns *None* if the name cannot be parsed.
    """
    parts = name.split("_")
    if len(parts) < 2:
        return None
    prefix = parts[0]
    try:
        value = float(parts[1])
    except ValueError:
        value = parts[1]
    seed = None
    if len(parts) >= 4 and parts[2] == "seed":
        try:
            seed = int(parts[3])
        except ValueError:
            seed = parts[3]
    return prefix, value, seed
# ============================================================
# Group discovery & validation
# ============================================================
def build_groups(base_dir: str):
    """Scan *base_dir* and group folders by prefix."""
    groups: dict[str, list] = defaultdict(list)
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if "reference" in entry or "common" in entry:
            continue
        parsed = parse_folder_name(entry)
        if parsed is None:
            continue
        prefix, value, seed = parsed
        groups[prefix].append((full_path, value, seed))
    return groups
def _sort_key(item):
    """Sort by (value, seed)."""
    _, value, seed, _ = item
    return (value, seed if seed is not None else -1)
def validate_and_collect(groups, pmf_rel, count_rel, reference,
                         n_recent, slope_thresh, rmsd_thresh,
                         use_sliding, min_frames):
    """Build :class:`PMFAnalyzer` for each valid folder; return sorted groups."""
    cache: dict[tuple, PMFAnalyzer] = {}
    valid: dict[str, list] = {}
    for prefix, folders in groups.items():
        cleaned = []
        for folder_path, value, seed in folders:
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
                        reference_pmf_file=reference,
                        rmsd_thresh=rmsd_thresh,
                    )
                except Exception as exc:
                    print(f"Skipping {folder_path}: {exc}")
                    continue
            analyzer = cache[key]
            if len(analyzer.rmsd_raw) < min_frames:
                print(f"Skipping {folder_path}: only {len(analyzer.rmsd_raw)} frames")
                continue
            cleaned.append((folder_path, value, seed, analyzer))
        if cleaned:
            valid[prefix] = sorted(cleaned, key=_sort_key)
    return valid
# ============================================================
# Main
# ============================================================
LINESTYLES = [
    "-",
    (0, (5, 1)),
    (0, (5, 5)),
    (0, (5, 10)),
    (0, (1, 1)),
    (0, (1, 5)),
    (0, (1, 10)),
]
def main():
    parser = argparse.ArgumentParser(
        description="Overlay RMSD curves from seeded replicas."
    )
    parser.add_argument("--dir", default="./",
                        help="Parent directory to scan (default: ./)")
    parser.add_argument("--pmf", default="output/window1.abf1.hist.czar.pmf",
                        help="Relative path to PMF file inside each folder")
    parser.add_argument("--counts", default="output/window1.abf1.hist.zcount",
                        help="Relative path to count file inside each folder")
    parser.add_argument("--reference", default=None,
                        help="Path to reference PMF (e.g. reference_median.pmf)")
    parser.add_argument("--divisor", type=float, default=5.0,
                        help="Divisor to convert snapshot index to time (ns)")
    parser.add_argument("--n-recent", type=int, default=4)
    parser.add_argument("--slope-thresh", type=float, default=1e-3)
    parser.add_argument("--rmsd-thresh", type=float, default=0.01)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--use-sliding", action="store_true")
    parser.add_argument("--output", default="RMSD_convergence_seeds.png",
                        help="Output figure filename")
    args = parser.parse_args()
    # Default reference path if not provided
    ref = args.reference
    if ref is None:
        candidate = os.path.join(args.dir, "reference_median.pmf")
        if os.path.isfile(candidate):
            ref = candidate
    groups = build_groups(args.dir)
    valid_groups = validate_and_collect(
        groups, args.pmf, args.counts, ref,
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
                             figsize=(7, 7 / n_cols * n_rows), sharex=True)
    axes_list = list(np.array(axes).flatten()) if n_groups > 1 else [axes]
    palette = sns.color_palette("colorblind", n_colors=10)
    # ---- Plot each group ----
    group_items = list(valid_groups.items())
    for ax_idx, (prefix, folder_list) in enumerate(group_items):
        ax = axes_list[ax_idx]
        # Sub-group by parameter value
        by_value: dict = defaultdict(list)
        for folder_path, value, seed, analyzer in folder_list:
            by_value[value].append((seed, analyzer))
        sorted_values = sorted(by_value.keys(),
                               key=lambda v: float(v) if isinstance(v, (int, float)) else v)
        legend_handles = []
        for color_idx, value in enumerate(sorted_values):
            base_color = np.array(palette[color_idx % len(palette)])
            entries = sorted(by_value[value],
                             key=lambda x: x[0] if x[0] is not None else -1)
            for i, (seed, analyzer) in enumerate(entries):
                ax.plot(
                    analyzer.t / args.divisor,
                    analyzer.rmsd_raw,
                    color=base_color,
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    linewidth=1.5,
                )
            legend_handles.append(
                Line2D([0], [0], color=base_color, lw=2, label=f"{value}")
            )
        ax.set_title(LABELS.get(prefix, prefix), fontsize=9)
        ax.legend(handles=legend_handles, fontsize=8, loc="best",
                  ncols=2, columnspacing=0.5)
        ax.set_aspect("auto")
    # Hide unused axes
    for j in range(len(group_items), len(axes_list)):
        fig.delaxes(axes_list[j])
    fig.supxlabel("Time (ns)")
    fig.supylabel("RMSD (kcal/mol)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved {args.output}")
if __name__ == "__main__":
    main()
