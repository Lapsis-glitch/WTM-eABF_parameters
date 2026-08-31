#!/usr/bin/env python3
"""Build robust reference PMFs from explicit, manifest, or discovered runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    from .input_discovery import records_from_inputs
    from .outputs import analysis_output
    from .plotting import PlotConfig, add_plotting_arguments, config_from_args
    from .pmf_io import interpolate_pmf
    from .reference_builder import (compute_reference_pmf_with_outliers,
                                    render_reference_figures, save_reference_plot_data)
except ImportError:
    from input_discovery import records_from_inputs
    from outputs import analysis_output
    from plotting import PlotConfig, add_plotting_arguments, config_from_args
    from pmf_io import interpolate_pmf
    from reference_builder import (compute_reference_pmf_with_outliers,
                                   render_reference_figures, save_reference_plot_data)


def read_sequential_pmf_file(filename):
    """Read the legacy single-PMF sequential format used by existing data."""
    blocks, current = [], []
    with Path(filename).open() as handle:
        for line in handle:
            if line.startswith("#"):
                if current:
                    blocks.append(np.array(current, float))
                    current = []
            elif line.strip():
                current.append(line.split())
    if current:
        blocks.append(np.array(current, float))
    if not blocks:
        raise ValueError(f"PMF file is empty: {filename}")
    data = blocks[-1]
    coords = tuple(np.unique(data[:, index]) for index in range(data.shape[1] - 1))
    return coords, data[:, -1].reshape(tuple(len(item) for item in coords))


def run(base_dir=None, temperature=300, name="abf_00.abf1", n_points=100, *,
        output_root="Results", manifest=None, pmf_file=None, count_file=None,
        pmf_pattern=None, count_pattern=None, metadata_regex=None, config=None,
        simple_reference_plot=False, xlabel=None):
    pattern = pmf_pattern or f"**/{name}.czar.pmf"
    discovered = records_from_inputs(root=base_dir if manifest is None and pmf_file is None else None,
                                     manifest=manifest, pmf_file=pmf_file, count_file=count_file,
                                     pmf_pattern=pattern, count_pattern=count_pattern,
                                     metadata_regex=metadata_regex, require_count=False)
    pmf_paths = [record.pmf_file for record in discovered.runs]
    if not pmf_paths:
        raise RuntimeError("No PMF files were discovered for reference construction")
    coords_tuple = None
    values = []
    for path in pmf_paths:
        try:
            coords, pmf = read_sequential_pmf_file(path)
            interpolated_coords, interpolated = interpolate_pmf(coords, pmf, n_points)
        except (OSError, ValueError, IndexError) as exc:
            logging.warning("Skipping PMF %s: %s", path, exc)
            continue
        coords_tuple = coords_tuple or interpolated_coords
        values.append(interpolated)
    if not values:
        raise RuntimeError("No usable PMFs remained for reference construction")
    output = analysis_output(output_root, "reference_pmf")
    data = compute_reference_pmf_with_outliers(coords_tuple, values, temperature,
                                               write_prefix=output.directory / "reference")
    save_reference_plot_data(data, coords_tuple, temperature, output.directory,
                             simple_reference_plot=simple_reference_plot,
                             simple_xlabel=xlabel)
    render_reference_figures(data, coords_tuple, output_root=output_root,
                             config=config or PlotConfig(output_root=str(output_root)),
                             simple_reference_plot=simple_reference_plot, xlabel=xlabel)
    return data


def build_parser():
    parser = argparse.ArgumentParser(description="Compute robust reference PMFs")
    parser.add_argument("--dir", help="Input root for recursive discovery")
    parser.add_argument("--manifest")
    parser.add_argument("--pmf-file")
    parser.add_argument("--count-file")
    parser.add_argument("--temp", type=float, default=300)
    parser.add_argument("--name", default="abf_00.abf1")
    parser.add_argument("--pmf-pattern")
    parser.add_argument("--count-pattern")
    parser.add_argument("--metadata-regex")
    parser.add_argument("--npoints", type=int, default=100)
    parser.add_argument("--simple-reference-plot", action="store_true",
                        help="also write a median-only publication reference figure")
    parser.add_argument("--xlabel", default=None, metavar="LABEL",
                        help="x-axis label for the simple reference figure")
    add_plotting_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.dir and not args.manifest and not args.pmf_file:
        parser.error("provide --dir, --manifest, or --pmf-file")
    config = config_from_args(args)
    run(args.dir, temperature=args.temp, name=args.name, n_points=args.npoints,
        output_root=config.output_root, manifest=args.manifest, pmf_file=args.pmf_file,
        count_file=args.count_file, pmf_pattern=args.pmf_pattern,
        count_pattern=args.count_pattern, metadata_regex=args.metadata_regex, config=config,
        simple_reference_plot=args.simple_reference_plot, xlabel=args.xlabel)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
