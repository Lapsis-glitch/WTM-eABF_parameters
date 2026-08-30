#!/usr/bin/env python3
"""CLI entry point for grouped RMSD curves."""

try:
    from .rmsd_analysis import cli
except ImportError:
    from rmsd_analysis import cli


if __name__ == "__main__":
    cli(seed_mode=False)
