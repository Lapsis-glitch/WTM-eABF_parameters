#!/usr/bin/env python3
"""Recreate RMSD figures from saved long-form series only."""

from __future__ import annotations

import argparse

try:
    from .plotting import add_plotting_arguments, config_from_args
    from .rmsd_analysis import load_rmsd_series, render_saved_rmsd_figures
except ImportError:
    from plotting import add_plotting_arguments, config_from_args
    from rmsd_analysis import load_rmsd_series, render_saved_rmsd_figures


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recreate RMSD figures without reading PMF or count histories"
    )
    parser.add_argument("series", help="rmsd_series.csv produced by RMSD analysis")
    parser.add_argument("--metadata", help="metadata JSON (default: rmsd_plot_metadata.json beside series)")
    parser.add_argument("--seed-mode", action="store_true",
                        help="render seed-resolved figures under rmsd_seed_curves")
    add_plotting_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()
    items, metadata = load_rmsd_series(args.series, args.metadata)
    config = config_from_args(args)
    render_saved_rmsd_figures(
        items,
        output_root=config.output_root,
        config=config,
        seed_mode=args.seed_mode,
        divisor=float(metadata["time_divisor"]),
        kbt_threshold=metadata.get("kbt_threshold"),
    )


if __name__ == "__main__":
    main()
