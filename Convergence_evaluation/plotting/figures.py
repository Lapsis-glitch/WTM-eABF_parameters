"""Small adapters around the installed PubReady API."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .config import PlotConfig


PT_PER_INCH = 72.0
LAYOUT_PAD_IN = 2.0 / PT_PER_INCH


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
    result = pr.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
                         **config.geometry(kind), **kwargs)
    if kind == "multipanel":
        result[0]._wtm_multipanel_shape = (int(nrows), int(ncols))
    return result


def flatten_axes(axes) -> list:
    if hasattr(axes, "get_position"):
        return [axes]
    return [axis for axis in np.asarray(axes, dtype=object).flat if hasattr(axis, "get_position")]


def save_figure(fig, base_path: str | Path, config: PlotConfig, *, fit: bool = True) -> list[Path]:
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
        pr.savefig(fig, path, dpi=config.dpi, fit=fit, validate=False)
        paths.append(path)
    return paths


def close_figure(fig) -> None:
    plt.close(fig)


def multipanel_grid(n_panels: int, config: PlotConfig) -> tuple[int, int]:
    """Return a compact automatic grid for a strict multipanel figure."""
    n_panels = int(n_panels)
    if n_panels < 1:
        raise ValueError("n_panels must be positive")
    is_default_si = (str(config.publisher).lower() == "acs"
                     and str(config.multipanel_target).lower() == "si"
                     and str(config.multipanel_fraction).lower() == "full")
    max_columns = 2 if is_default_si else 3
    ncols = min(max_columns, max(1, int(np.ceil(np.sqrt(n_panels)))))
    return int(np.ceil(n_panels / ncols)), ncols


def set_shared_labels(fig, *, xlabel: str | None = None, ylabel: str | None = None):
    """Add figure-level labels and retain references for layout finalization."""
    if xlabel is not None:
        fig._wtm_shared_xlabel = fig.supxlabel(xlabel)
    if ylabel is not None:
        fig._wtm_shared_ylabel = fig.supylabel(ylabel)
    return fig


def _core_axes(fig):
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def _physical_box(artist, dpi):
    box = artist.get_window_extent()
    return (box.x0 / dpi, box.y0 / dpi, box.x1 / dpi, box.y1 / dpi)


def _report_boxes(decoration):
    names = ("x_tick_labels", "y_tick_labels", "x_offset", "y_offset",
             "xlabel", "ylabel", "title", "annotations")
    boxes = []
    for name in names:
        value = getattr(decoration, name, None)
        values = value if isinstance(value, tuple) else (value,)
        boxes.extend(item for item in values if item is not None)
    return [(box.x0, box.y0, box.x1, box.y1) for box in boxes]


def _report_side_boxes(decoration, side):
    names = {
        "bottom": ("x_tick_labels", "x_offset", "xlabel"),
        "left": ("y_tick_labels", "y_offset", "ylabel"),
    }[side]
    boxes = []
    for name in names:
        value = getattr(decoration, name, None)
        values = value if isinstance(value, tuple) else (value,)
        boxes.extend(item for item in values if item is not None)
    return [(box.x0, box.y0, box.x1, box.y1) for box in boxes]


def _axis_boxes(fig, report, axes):
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    result = []
    for index, axis in enumerate(axes):
        boxes = _report_boxes(report.decorations[index]) if index < len(report.decorations) else []
        legend = axis.get_legend()
        if legend is not None and legend.get_visible():
            box = legend.get_window_extent(renderer)
            boxes.append((box.x0 / dpi, box.y0 / dpi, box.x1 / dpi, box.y1 / dpi))
        result.append(boxes)
    return result


def _live_text_boxes(fig, axis):
    """Return visible text-bearing decorations from the current renderer."""
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    artists = [axis.title, axis.xaxis.label, axis.yaxis.label,
               *axis.get_xticklabels(), *axis.get_yticklabels(), axis.get_legend()]
    artists.extend(item for item in axis.get_children()
                   if item.__class__.__name__ == "Annotation")
    boxes = []
    for artist in artists:
        if artist is None or not artist.get_visible():
            continue
        text = artist.get_text() if hasattr(artist, "get_text") else ""
        if hasattr(artist, "get_texts"):
            text = any(item.get_text() for item in artist.get_texts())
        if not text:
            continue
        box = artist.get_window_extent(renderer)
        boxes.append((box.x0 / dpi, box.y0 / dpi, box.x1 / dpi, box.y1 / dpi))
    return boxes


def _interval_overlap(low_a, high_a, low_b, high_b):
    return max(0.0, min(high_a, high_b) - max(low_a, low_b))


def _required_horizontal_gap(left_axis, right_axis):
    extra = 0.0
    for left in left_axis:
        for right in right_axis:
            if _interval_overlap(left[1], left[3], right[1], right[3]) > 0:
                extra = max(extra, left[2] - right[0] + LAYOUT_PAD_IN)
    return max(0.0, extra)


def _required_vertical_gap(upper_axis, lower_axis):
    extra = 0.0
    for upper in upper_axis:
        for lower in lower_axis:
            if _interval_overlap(upper[0], upper[2], lower[0], lower[2]) > 0:
                extra = max(extra, lower[3] - upper[1] + LAYOUT_PAD_IN)
    return max(0.0, extra)


def _reposition_colorbars(fig, old_axes, new_axes, old_colorbars, new_width, new_height):
    """Move attached colorbars with their nearest plotting axes after spacing changes."""
    if not old_colorbars:
        return
    for colorbar, old_box in old_colorbars:
        old_center = ((old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2)
        axis_index = min(
            range(len(old_axes)),
            key=lambda index: abs(old_center[1] - ((old_axes[index][1] + old_axes[index][3]) / 2))
                             + 0.25 * abs(old_center[0] - old_axes[index][2]),
        )
        old_axis, new_axis = old_axes[axis_index], new_axes[axis_index]
        dx = new_axis[0] - old_axis[0]
        dy = new_axis[1] - old_axis[1]
        x0, y0, x1, y1 = old_box
        colorbar.set_position(((x0 + dx) / new_width, (y0 + dy) / new_height,
                               (x1 - x0) / new_width, (y1 - y0) / new_height))


def finalize_multipanel_layout(fig):
    """Resolve PubReady geometry, then separate neighboring decorations physically.

    PubReady intentionally fixes the publisher width and its public API does not
    expose a multipanel gutter setting.  Its renderer-backed resolution is used
    first; this final pass only changes inter-panel positions and, as a last
    resort, reduces axes width enough to keep the fixed publisher canvas valid.
    """
    pr = _pubready()
    report = pr.layout_report(fig)
    if report is None:
        raise ValueError("finalize_multipanel_layout requires a PubReady-managed figure")
    fig.canvas.draw()
    axes = _core_axes(fig)
    nrows, ncols = getattr(fig, "_wtm_multipanel_shape", (1, len(axes)))
    if len(axes) != nrows * ncols:
        raise ValueError("multipanel shape does not match managed axes")
    visible = [axis.get_visible() for axis in axes]
    dpi = fig.dpi
    old_width, old_height = fig.bbox.width / dpi, fig.bbox.height / dpi
    old_axes = [_physical_box(axis, dpi) for axis in axes]
    old_colorbars = [(axis, _physical_box(axis, dpi)) for axis in fig.axes
                     if axis.get_label() == "<colorbar>"]
    decorations = _axis_boxes(fig, report, axes)

    horizontal_extra = [0.0] * max(0, ncols - 1)
    for row in range(nrows):
        for col in range(ncols - 1):
            left_index, right_index = row * ncols + col, row * ncols + col + 1
            if visible[left_index] and visible[right_index]:
                horizontal_extra[col] = max(
                    horizontal_extra[col],
                    _required_horizontal_gap(decorations[left_index], decorations[right_index]),
                )
    vertical_extra = [0.0] * max(0, nrows - 1)
    for row in range(nrows - 1):
        for col in range(ncols):
            upper_index, lower_index = row * ncols + col, (row + 1) * ncols + col
            if visible[upper_index] and visible[lower_index]:
                vertical_extra[row] = max(
                    vertical_extra[row],
                    _required_vertical_gap(decorations[upper_index], decorations[lower_index]),
                )

    col_shifts = [0.0]
    for extra in horizontal_extra:
        col_shifts.append(col_shifts[-1] + extra)
    row_shifts = [0.0]
    for extra in vertical_extra:
        row_shifts.append(row_shifts[-1] - extra)
    total_vertical_extra = sum(vertical_extra)

    shared_xlabel = getattr(fig, "_wtm_shared_xlabel", None)
    shared_ylabel = getattr(fig, "_wtm_shared_ylabel", None)
    fig.canvas.draw()
    xlabel_height = _physical_box(shared_xlabel, dpi)[3] - _physical_box(shared_xlabel, dpi)[1] if shared_xlabel else 0.0
    ylabel_width = _physical_box(shared_ylabel, dpi)[2] - _physical_box(shared_ylabel, dpi)[0] if shared_ylabel else 0.0
    bottom_decoration = min(
        (box[1] + row_shifts[index // ncols]
         for index, decoration in enumerate(report.decorations)
         if visible[index] for box in _report_side_boxes(decoration, "bottom")),
        default=0.0,
    )
    bottom_shift = max(0.0, (2 * LAYOUT_PAD_IN + xlabel_height)
                       - (bottom_decoration + total_vertical_extra)) if shared_xlabel else 0.0
    left_decoration = min(
        (box[0] + col_shifts[index % ncols]
         for index, decoration in enumerate(report.decorations)
         if visible[index] for box in _report_side_boxes(decoration, "left")),
        default=0.0,
    )
    left_shift = max(0.0, 2 * LAYOUT_PAD_IN + ylabel_width - left_decoration) if shared_ylabel else 0.0

    new_height = old_height + total_vertical_extra + bottom_shift
    proposed = []
    for index, (x0, y0, x1, y1) in enumerate(old_axes):
        row, col = divmod(index, ncols)
        proposed.append([x0 + col_shifts[col] + left_shift,
                         y0 + row_shifts[row] + total_vertical_extra + bottom_shift,
                         x1 - x0, y1 - y0])
    core_width = report.core_width
    right_edge = max((item[0] + item[2] for index, item in enumerate(proposed) if visible[index]), default=core_width)
    width_reduction = max(0.0, right_edge + LAYOUT_PAD_IN - core_width) / max(1, ncols)
    if width_reduction:
        for item in proposed:
            item[2] = max(0.6, item[2] - width_reduction)

    max_height = float(pr.get_publisher(report.publisher).max_height)
    if new_height > max_height:
        raise ValueError(
            f"Finalized multipanel height {new_height:.3f} in exceeds "
            f"{report.publisher} maximum of {max_height:.3f} in."
        )
    fig.set_size_inches(old_width, new_height, forward=True)
    new_axes = []
    for axis, (x0, y0, width, height) in zip(axes, proposed):
        axis.set_position((x0 / old_width, y0 / new_height, width / old_width, height / new_height))
        new_axes.append((x0, y0, x0 + width, y0 + height))
    _reposition_colorbars(fig, old_axes, new_axes, old_colorbars, old_width, new_height)
    if shared_xlabel or shared_ylabel:
        fig.canvas.draw()
    if shared_xlabel:
        xlabel_box = _physical_box(shared_xlabel, dpi)
        bottom_boxes = [box for index, axis in enumerate(axes)
                        if visible[index] and index // ncols == nrows - 1
                        for box in _live_text_boxes(fig, axis)]
        extra_bottom = max(
            (xlabel_box[3] - box[1] + LAYOUT_PAD_IN
             for box in bottom_boxes
             if _interval_overlap(xlabel_box[0], xlabel_box[2], box[0], box[2]) > 0),
            default=0.0,
        )
        if extra_bottom > 0:
            current_axes = new_axes
            current_colorbars = [(axis, _physical_box(axis, dpi)) for axis in fig.axes
                                 if axis.get_label() == "<colorbar>"]
            new_height += extra_bottom
            if new_height > max_height:
                raise ValueError(
                    f"Finalized multipanel height {new_height:.3f} in exceeds "
                    f"{report.publisher} maximum of {max_height:.3f} in."
                )
            new_axes = [(x0, y0 + extra_bottom, x1, y1 + extra_bottom)
                        for x0, y0, x1, y1 in current_axes]
            fig.set_size_inches(old_width, new_height, forward=True)
            for axis, (x0, y0, x1, y1) in zip(axes, new_axes):
                axis.set_position((x0 / old_width, y0 / new_height,
                                   (x1 - x0) / old_width, (y1 - y0) / new_height))
            _reposition_colorbars(fig, current_axes, new_axes, current_colorbars,
                                  old_width, new_height)
    if shared_ylabel:
        fig.canvas.draw()
        ylabel_box = _physical_box(shared_ylabel, dpi)
        left_boxes = [box for index, axis in enumerate(axes)
                      if visible[index] and index % ncols == 0
                      for box in _live_text_boxes(fig, axis)]
        extra_left = max(
            (ylabel_box[2] - box[0] + LAYOUT_PAD_IN
             for box in left_boxes
             if _interval_overlap(ylabel_box[1], ylabel_box[3], box[1], box[3]) > 0),
            default=0.0,
        )
        if extra_left > 0:
            current_axes = new_axes
            current_colorbars = [(axis, _physical_box(axis, dpi)) for axis in fig.axes
                                 if axis.get_label() == "<colorbar>"]
            new_axes = [(x0 + extra_left, y0, x1 + extra_left, y1)
                        for x0, y0, x1, y1 in current_axes]
            for axis, (x0, y0, x1, y1) in zip(axes, new_axes):
                axis.set_position((x0 / old_width, y0 / new_height,
                                   (x1 - x0) / old_width, (y1 - y0) / new_height))
            _reposition_colorbars(fig, current_axes, new_axes, current_colorbars,
                                  old_width, new_height)
    if shared_xlabel:
        # ``supxlabel`` uses top alignment: its y position is the top edge,
        # not its vertical center.
        shared_xlabel.set_position((0.5, (LAYOUT_PAD_IN + xlabel_height) / new_height))
    if shared_ylabel:
        shared_ylabel.set_position(((LAYOUT_PAD_IN + ylabel_width / 2) / old_width, 0.5))
    fig.canvas.draw()
    if shared_xlabel:
        xlabel_box = _physical_box(shared_xlabel, dpi)
        bottom_boxes = [box for index, axis in enumerate(axes)
                        if visible[index] and index // ncols == nrows - 1
                        for box in _live_text_boxes(fig, axis)]
        extra_bottom = max(
            (xlabel_box[3] - box[1] + LAYOUT_PAD_IN
             for box in bottom_boxes
             if _interval_overlap(xlabel_box[0], xlabel_box[2], box[0], box[2]) > 0),
            default=0.0,
        )
        if extra_bottom > 0:
            x_position, y_position = shared_xlabel.get_position()
            shared_xlabel.set_position((x_position, y_position - extra_bottom / new_height))
    if shared_ylabel:
        ylabel_box = _physical_box(shared_ylabel, dpi)
        left_boxes = [box for index, axis in enumerate(axes)
                      if visible[index] and index % ncols == 0
                      for box in _live_text_boxes(fig, axis)]
        extra_left = max(
            (ylabel_box[2] - box[0] + LAYOUT_PAD_IN
             for box in left_boxes
             if _interval_overlap(ylabel_box[1], ylabel_box[3], box[1], box[3]) > 0),
            default=0.0,
        )
        if extra_left > 0:
            x_position, y_position = shared_ylabel.get_position()
            shared_ylabel.set_position((x_position - extra_left / old_width, y_position))
    fig.canvas.draw()
    fig._wtm_multipanel_finalized = True
    resolved = pr.layout_report(fig, resolve=False)
    axis_metrics = tuple(
        replace(metric, x0=position[0], y0=position[1],
                width=position[2] - position[0], height=position[3] - position[1])
        for metric, position in zip(resolved.axes_bboxes, new_axes)
    )
    fig._wtm_final_layout_report = replace(
        resolved,
        panel_height=new_height,
        axes_width=min(item[2] for item in proposed),
        axes_height=min(item[3] for item in proposed),
        axes_bboxes=axis_metrics,
        canvas_height=new_height,
    )
    return fig._wtm_final_layout_report
