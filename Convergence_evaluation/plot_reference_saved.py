#!/usr/bin/env python3
"""Recreate reference-PMF figures from saved numerical plotting data."""

from __future__ import annotations

import argparse

try:
    from .plotting import add_plotting_arguments, config_from_args
    from .reference_builder import load_reference_plot_data, render_reference_figures
except ImportError:
    from plotting import add_plotting_arguments, config_from_args
    from reference_builder import load_reference_plot_data, render_reference_figures


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recreate reference-PMF figures without reading simulation histories"
    )
    parser.add_argument("data", help="reference_plot_data.npz produced by buildref.py")
    parser.add_argument("--metadata", help="metadata JSON (default: reference_plot_metadata.json beside data)")
    parser.add_argument("--simple-reference-plot", action="store_true",
                        help="also recreate the median-only reference figure")
    parser.add_argument("--xlabel", default=None, metavar="LABEL",
                        help="x-axis label for the optional median-only figure")
    add_plotting_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()
    data, coords_tuple, metadata = load_reference_plot_data(args.data, args.metadata)
    config = config_from_args(args)
    simple_reference_plot = args.simple_reference_plot or metadata.get("simple_reference_plot", False)
    render_reference_figures(
        data,
        coords_tuple,
        output_root=config.output_root,
        config=config,
        simple_reference_plot=simple_reference_plot,
        xlabel=args.xlabel if args.xlabel is not None else metadata.get("simple_xlabel"),
    )


if __name__ == "__main__":
    main()
