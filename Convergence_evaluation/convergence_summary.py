"""Structured convergence-summary analysis shared by 1D and 2D plotters."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .analyze_ND import PMFAnalyzer
    from .input_discovery import RunRecord
except ImportError:
    from analyze_ND import PMFAnalyzer
    from input_discovery import RunRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvergenceSummary:
    group: str
    parameter_values: dict
    mean: float
    std: float
    minimum: float
    maximum: float
    n: int

    @property
    def parameter_name(self):
        return next(iter(self.parameter_values), "")

    @property
    def parameter_value(self):
        return self.parameter_values.get(self.parameter_name, "")


def summarize_runs(records: Iterable[RunRecord], *, reference_pmf=None,
                   slope_thresh=0.01, n_recent=5, rmsd_thresh=0.592186869182,
                   use_ref_and_slope=True) -> list[ConvergenceSummary]:
    values = defaultdict(list)
    for record in records:
        try:
            analyzer = PMFAnalyzer(record.pmf_file, record.count_file,
                                   slope_thresh=slope_thresh, n_recent=n_recent,
                                   use_sliding_window=False, reference_pmf_file=reference_pmf,
                                   rmsd_thresh=rmsd_thresh, use_ref_and_slope=use_ref_and_slope)
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.warning("Skipping run %s (PMF=%s, count=%s): %s",
                           record.run_id, record.pmf_file, record.count_file, exc)
            continue
        if analyzer.convergence_idx is None:
            LOGGER.warning("Run %s has no detected convergence", record.run_id)
            continue
        parameters = dict(record.parameter_values)
        key = (record.group or record.run_id,
               json.dumps(parameters, sort_keys=True, default=str))
        values[key].append(float(analyzer.convergence_idx))
    summaries = []
    for (group, encoded), entries in sorted(values.items()):
        parameters = json.loads(encoded)
        summaries.append(ConvergenceSummary(
            group=group, parameter_values=parameters,
            mean=float(np.mean(entries)), std=float(np.std(entries)),
            minimum=float(np.min(entries)), maximum=float(np.max(entries)), n=len(entries)))
    if not summaries:
        raise RuntimeError("No usable runs remained for convergence summary")
    return summaries


def write_summary_csv(summaries: Iterable[ConvergenceSummary], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "parameter_name", "parameter_value",
                                                     "parameter_values", "mean", "std", "min", "max", "n"])
        writer.writeheader()
        for item in summaries:
            writer.writerow({"group": item.group, "parameter_name": item.parameter_name,
                             "parameter_value": item.parameter_value,
                             "parameter_values": json.dumps(item.parameter_values, sort_keys=True),
                             "mean": item.mean, "std": item.std, "min": item.minimum,
                             "max": item.maximum, "n": item.n})
    return destination


def read_summary(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["mean"] = float(row["mean"])
        row["std"] = float(row["std"])
        row["min"] = float(row["min"])
        row["max"] = float(row["max"])
        row["n"] = int(row["n"])
        row["parameter_values"] = json.loads(row.get("parameter_values") or "{}")
    return rows
