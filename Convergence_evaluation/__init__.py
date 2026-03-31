"""
Convergence_evaluation
======================
Toolkit for analysing PMF convergence and building robust reference PMFs
from WTM-eABF simulations.

Public API
----------
- :class:`analyze_ND.PMFAnalyzer` — N-D PMF convergence analyser.
- :mod:`pmf_io` — read / write / interpolate PMF files.
- :mod:`reference_builder` — statistical aggregation of multiple PMFs.
"""

from .analyze_ND import PMFAnalyzer
from .pmf_io import (
    interpolate_pmf,
    read_sequential_counts,
    read_sequential_pmf,
    read_sequential_pmf_blocks,
    write_sequential_pmf,
)
from .reference_builder import compute_reference_pmf_with_outliers

__all__ = [
    "PMFAnalyzer",
    "read_sequential_pmf",
    "read_sequential_pmf_blocks",
    "read_sequential_counts",
    "write_sequential_pmf",
    "interpolate_pmf",
    "compute_reference_pmf_with_outliers",
]

