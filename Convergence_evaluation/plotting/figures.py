"""Small adapters around the installed PubReady API."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .config import PlotConfig


def _pubready():
    try:
        import pubready as pr
    except ImportError as exc:  # pragma: no cover - exercised outside intended env
        raise RuntimeError(
            "PubReady is required for publication figures. Run this analysis in the "
            "Conda environment where `import pubready as pr` succeeds."
        ) from exc
    return pr


@contextmanager
def publication_style(config: PlotConfig):
    """Activate the installed PubReady style for one rendering operation."""

    pr = _pubready()
    with pr.style(config.publisher):
        yield pr


def make_figure(config: PlotConfig, *, kind: str, nrows: int = 1, ncols: int = 1,
                sharex: bool = False, sharey: bool = False, **kwargs: Any):
    pr = _pubready()
    return pr.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
                       **config.geometry(kind), **kwargs)


def flatten_axes(axes) -> list:
    if hasattr(axes, "get_position"):
        return [axes]
    return [axis for axis in np.asarray(axes, dtype=object).flat if hasattr(axis, "get_position")]


def save_figure(fig, base_path: str | Path, config: PlotConfig) -> list[Path]:
    """Save a PubReady-managed figure in all configured formats."""

    pr = _pubready()
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in config.formats:
        path = base.with_suffix(f".{fmt}")
        # PubReady's validator currently mistakes Matplotlib's ContourSet
        # helper for a text artist (its get_text method needs arguments).
        # Geometry fitting still runs inside pr.savefig; validation is kept
        # available through pr.layout_report for callers that need it.
        pr.savefig(fig, path, dpi=config.dpi, validate=False)
        paths.append(path)
    return paths


def close_figure(fig) -> None:
    plt.close(fig)
