#!/usr/bin/env python3
"""Backward-compatible entry point for the generalized convergence parser."""

from __future__ import annotations

import argparse
import logging

try:
    from .convergence_summary import summarize_runs, write_summary_csv
    from .input_discovery import records_from_inputs
    from .outputs import analysis_output
except ImportError:
    from convergence_summary import summarize_runs, write_summary_csv
    from input_discovery import records_from_inputs
    from outputs import analysis_output


def parse(path=None, reference_pmf=None, *, output_root="Results", manifest=None,
          pmf_file=None, count_file=None, pmf_pattern="**/*czar.pmf", count_pattern=None,
          metadata_regex=None):
    result = records_from_inputs(root=path if manifest is None and pmf_file is None else None,
                                 manifest=manifest, pmf_file=pmf_file, count_file=count_file,
                                 pmf_pattern=pmf_pattern, count_pattern=count_pattern,
                                 metadata_regex=metadata_regex)
    summaries = summarize_runs(result.runs, reference_pmf=reference_pmf)
    output = analysis_output(output_root, "convergence_surface")
    write_summary_csv(summaries, output.directory / "convergence_summary.csv")
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Discover runs for a convergence surface")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--manifest")
    parser.add_argument("--pmf-file")
    parser.add_argument("--count-file")
    parser.add_argument("--reference-pmf")
    parser.add_argument("--pmf-pattern", default="**/*czar.pmf")
    parser.add_argument("--count-pattern")
    parser.add_argument("--metadata-regex")
    parser.add_argument("--output-root", default="Results")
    args = parser.parse_args()
    if not args.path and not args.manifest and not args.pmf_file:
        parser.error("provide path, --manifest, or --pmf-file")
    parse(args.path, args.reference_pmf, output_root=args.output_root, manifest=args.manifest,
          pmf_file=args.pmf_file, count_file=args.count_file, pmf_pattern=args.pmf_pattern,
          count_pattern=args.count_pattern, metadata_regex=args.metadata_regex)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
