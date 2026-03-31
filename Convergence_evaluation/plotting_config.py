"""
plotting_config.py
------------------
Shared Matplotlib configuration and parameter-label mappings used by all
plotting scripts in the Convergence_evaluation package.

Font resolution order:
  1. Path in the ``MATPLOTLIB_FONT_PATH`` environment variable.
  2. System default sans-serif font (DejaVu Sans on most installs).
"""

import os
import matplotlib
import matplotlib.font_manager as font_manager

# ============================================================
# Parameter name → human-readable axis label
# ============================================================
LABELS = {
    "MTDheight":   "hillWeight",
    "MTDnewhill":  "newHillFrequency",
    "MTDtemp":     "biasTemperature",
    "MTDwidth":    "hillWidth",
    "colvarWidth": "colvarWidth",
    "extDamp":     "extendedLangevinDamping",
    "extFluc":     "extendedFluctuation",
    "extTime":     "extendedTimeConstant",
    "fullSamp":    "fullSamples",
}


def configure_plotting(font_size: int = 12) -> None:
    """
    Apply a consistent Matplotlib style across the project.

    Tries to load a custom font from ``$MATPLOTLIB_FONT_PATH``; on failure
    falls back to the default sans-serif family.
    """
    font_path = os.environ.get("MATPLOTLIB_FONT_PATH", "")

    if font_path and os.path.isfile(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        matplotlib.rc("font", family="sans-serif")
        matplotlib.rcParams["font.sans-serif"] = [prop.get_name()]
    else:
        matplotlib.rc("font", family="sans-serif")

    matplotlib.rcParams.update({"font.size": font_size})

