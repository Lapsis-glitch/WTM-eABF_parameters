#!/usr/bin/env python3
"""Compute convergence summaries from explicit, manifest, or discovered inputs."""

from __future__ import annotations

import argparse
import logging

try:
    from .convergence_summary import summarize_runs, write_summary_csv
    from .input_discovery import records_from_inputs
    from .outputs import analysis_output
    from .plotting import add_plotting_arguments, config_from_args
except ImportError:
    from convergence_summary import summarize_runs, write_summary_csv
    from input_discovery import records_from_inputs
    from outputs import analysis_output
    from plotting import add_plotting_arguments, config_from_args


def parse(path=None, reference_pmf=None, *, output_root="Results", pmf_pattern="**/*czar.pmf",
          count_pattern=None, manifest=None, pmf_file=None, count_file=None, metadata_regex=None):
    result = records_from_inputs(root=path if manifest is None and pmf_file is None else None,
                                 manifest=manifest, pmf_file=pmf_file, count_file=count_file,
                                 pmf_pattern=pmf_pattern, count_pattern=count_pattern,
                                 metadata_regex=metadata_regex)
    summaries = summarize_runs(result.runs, reference_pmf=reference_pmf)
    output = analysis_output(output_root, "convergence_summary")
    write_summary_csv(summaries, output.directory / "convergence_summary.csv")
    with (output.directory / "results.dat").open("w") as handle:
        for item in summaries:
            handle.write(f"{item.group} {item.mean:.3f} {item.std:.3f} {item.minimum:.3f} {item.maximum:.3f} {item.n}\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Compute structured convergence summaries")
    parser.add_argument("path", nargs="?", help="Input root for recursive discovery")
    parser.add_argument("--manifest")
    parser.add_argument("--pmf-file")
    parser.add_argument("--count-file")
    parser.add_argument("--reference-pmf")
    parser.add_argument("--pmf-pattern", default="**/*czar.pmf")
    parser.add_argument("--count-pattern", default=None)
    parser.add_argument("--metadata-regex", default=None)
    add_plotting_arguments(parser)
    args = parser.parse_args()
    if not args.path and not args.manifest and not args.pmf_file:
        parser.error("provide path, --manifest, or --pmf-file")
    config = config_from_args(args)
    parse(args.path, args.reference_pmf, output_root=config.output_root,
          pmf_pattern=args.pmf_pattern, count_pattern=args.count_pattern,
          manifest=args.manifest, pmf_file=args.pmf_file, count_file=args.count_file,
          metadata_regex=args.metadata_regex)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
