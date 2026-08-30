"""Centralized, safe output-directory management."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def safe_name(value: str) -> str:
    """Return a deterministic filesystem-safe name for an analysis label."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "analysis"


@dataclass(frozen=True)
class AnalysisOutput:
    root: Path
    name: str

    @property
    def directory(self) -> Path:
        path = self.root / safe_name(self.name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figures(self) -> Path:
        path = self.directory / "Figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def panels(self) -> Path:
        path = self.figures / "panels"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def snapshots(self) -> Path:
        path = self.figures / "snapshots"
        path.mkdir(parents=True, exist_ok=True)
        return path


def analysis_output(output_root: str | Path = "Results", name: str = "analysis") -> AnalysisOutput:
    return AnalysisOutput(Path(output_root).expanduser(), safe_name(name))
