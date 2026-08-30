"""Shared PubReady publication plotting infrastructure."""

from .config import PlotConfig, add_plotting_arguments, config_from_args
from .figures import (
    close_figure,
    finalize_multipanel_layout,
    make_figure,
    multipanel_grid,
    publication_style,
    save_figure,
    set_shared_labels,
    flatten_axes,
)

__all__ = [
    "PlotConfig", "add_plotting_arguments", "config_from_args", "close_figure",
    "make_figure", "publication_style", "save_figure", "flatten_axes",
    "finalize_multipanel_layout", "multipanel_grid", "set_shared_labels",
]
