"""Central PubReady defaults and shared CLI options."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlotConfig:
    publisher: str = "acs"
    multipanel_target: str = "si"
    multipanel_fraction: str = "full"
    panel_target: str = "double"
    panel_fraction: str = "quarter"
    formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300
    output_root: str = "Results"

    def geometry(self, kind: str) -> dict[str, str]:
        if kind == "multipanel":
            return {"publisher": self.publisher, "target": self.multipanel_target,
                    "fraction": self.multipanel_fraction}
        if kind in {"panel", "standalone"}:
            return {"publisher": self.publisher, "target": self.panel_target,
                    "fraction": self.panel_fraction}
        raise ValueError(f"unknown figure kind: {kind}")


def add_plotting_arguments(parser: argparse.ArgumentParser, *, include_output_root: bool = True) -> argparse.ArgumentParser:
    parser.add_argument("--publisher", default="acs")
    parser.add_argument("--multipanel-target", default="si", choices=("single", "double", "si"))
    parser.add_argument("--multipanel-fraction", default="full")
    parser.add_argument("--panel-target", default="double", choices=("single", "double", "si"))
    parser.add_argument("--panel-fraction", default="quarter")
    parser.add_argument("--figure-formats", nargs="+", default=("pdf", "png"), metavar="FORMAT")
    parser.add_argument("--dpi", type=int, default=300)
    if include_output_root:
        parser.add_argument("--output-root", default="Results")
    return parser


def config_from_args(args: argparse.Namespace) -> PlotConfig:
    formats = tuple(str(item).lower().lstrip(".") for item in (args.figure_formats or ("pdf", "png")))
    supported = {"pdf", "png", "svg"}
    invalid = set(formats) - supported
    if invalid:
        raise ValueError(f"unsupported figure format(s): {', '.join(sorted(invalid))}")
    return PlotConfig(
        publisher=args.publisher,
        multipanel_target=args.multipanel_target,
        multipanel_fraction=args.multipanel_fraction,
        panel_target=args.panel_target,
        panel_fraction=args.panel_fraction,
        formats=formats,
        dpi=args.dpi,
        output_root=getattr(args, "output_root", "Results"),
    )
