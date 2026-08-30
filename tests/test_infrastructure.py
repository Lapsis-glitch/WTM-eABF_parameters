import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Convergence_evaluation.input_discovery import discover_runs, read_manifest
from Convergence_evaluation.buildref import build_parser
from Convergence_evaluation.outputs import analysis_output
from Convergence_evaluation.plotting import (PlotConfig, close_figure, make_figure,
                                             publication_style, save_figure)
from Convergence_evaluation.analyze_ND import plot_rmsd_panel
from Convergence_evaluation.plot_results import plot_summary, plot_summary_panel
from Convergence_evaluation.plot_results2D import _surface
from Convergence_evaluation.rmsd_analysis import plot_group_panel, plot_seed_panel
from Convergence_evaluation.reference_builder import (compute_reference_pmf_with_outliers,
                                                      plot_pmf_comparison_panel,
                                                      plot_simple_reference_panel)
from Convergence_evaluation.plotting import (finalize_multipanel_layout, multipanel_grid,
                                             flatten_axes, set_shared_labels)


def test_recursive_discovery_pairs_arbitrary_paths(tmp_path):
    first = tmp_path / "random_project" / "run_alpha" / "data"
    second = tmp_path / "completely_different" / "nested"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "strange_name.czar.pmf").write_text("data")
    (first / "strange_name.count").write_text("data")
    (second / "test123.czar.pmf").write_text("data")
    (second / "test123.zcount").write_text("data")
    result = discover_runs(tmp_path)
    assert len(result.runs) == 2
    assert {run.count_file.suffix for run in result.runs} == {".count", ".zcount"}


def test_manifest_resolves_relative_paths_and_arbitrary_parameters(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pmf = data / "pmf.dat"
    count = data / "counts.dat"
    pmf.write_text("pmf")
    count.write_text("count")
    manifest = tmp_path / "runs.csv"
    manifest.write_text("run_id,pmf_file,count_file,group,seed,parameter_name,parameter_value\n"
                        "alpha,data/pmf.dat,data/counts.dat,set-A,7,my_test_parameter,2.5\n")
    result = read_manifest(manifest)
    run = result.runs[0]
    assert run.pmf_file == pmf.resolve()
    assert run.count_file == count.resolve()
    assert run.seed == 7
    assert run.parameter_values == {"my_test_parameter": 2.5}


def test_output_names_are_sanitized_and_nested(tmp_path):
    output = analysis_output(tmp_path, "my analysis/one")
    assert output.directory == tmp_path / "my_analysis_one"
    assert output.panels == tmp_path / "my_analysis_one" / "Figures" / "panels"


def test_panel_renderer_receives_same_data():
    class Analysis:
        t = [0, 1, 2]
        rmsd_raw = [3, 2, 1]
        rmsd_smooth = [3, 2, 1]
        rmsd_fit = __import__("numpy").array([float("nan")] * 3)
        convergence_idx = None

    import matplotlib.pyplot as plt
    first, second = plt.subplots(), plt.subplots()
    plot_rmsd_panel(first[1], Analysis())
    plot_rmsd_panel(second[1], Analysis())
    assert first[1].lines[0].get_ydata().tolist() == second[1].lines[0].get_ydata().tolist()
    plt.close(first[0])
    plt.close(second[0])


def test_pubready_defaults_are_strict():
    pytest.importorskip("pubready")
    config = PlotConfig()
    with publication_style(config):
        fig, ax = make_figure(config, kind="multipanel")
        report = __import__("pubready").layout_report(fig)
        assert report.target == "si"
        assert report.fraction == "full"
        close_figure(fig)
        fig, ax = make_figure(config, kind="panel")
        report = __import__("pubready").layout_report(fig)
        assert report.target == "double"
        assert report.fraction == "quarter"
        close_figure(fig)


def test_summary_writes_multipanel_and_standalone_outputs(tmp_path):
    pytest.importorskip("pubready")
    rows = [{"group": "set-A", "parameter_name": "my_test_parameter",
             "parameter_value": 1, "parameter_values": {"my_test_parameter": 1},
             "mean": 10, "std": 1, "min": 8, "max": 12, "n": 3},
            {"group": "set-A", "parameter_name": "my_test_parameter",
             "parameter_value": 2, "parameter_values": {"my_test_parameter": 2},
             "mean": 12, "std": 1, "min": 10, "max": 14, "n": 3}]
    plot_summary(rows, output_root=tmp_path)
    figures = tmp_path / "convergence_summary" / "Figures"
    assert (figures / "convergence_summary_multipanel.pdf").is_file()
    assert (figures / "convergence_summary_multipanel.png").is_file()
    assert (figures / "panels" / "my_test_parameter_convergence.pdf").is_file()


def test_summary_panel_has_no_title_and_plain_x_ticks(tmp_path):
    rows = [{"group": "set-A", "parameter_values": {"parameter": 1000},
             "mean": 10, "std": 1, "min": 8, "max": 12},
            {"group": "set-A", "parameter_values": {"parameter": 10000},
             "mean": 12, "std": 1, "min": 10, "max": 14}]
    config = PlotConfig(formats=("pdf",))
    with publication_style(config):
        fig, ax = make_figure(config, kind="panel")
        plot_summary_panel(ax, rows, "parameter")
        save_path = tmp_path / "plain_ticks"
        save_figure(fig, save_path, config)
        fig.canvas.draw()
        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert ax.get_title() == ""
        assert ax.xaxis.get_offset_text().get_text() == ""
        assert all("e" not in label.lower() and "10^" not in label for label in labels)
        assert "10000" in labels
        close_figure(fig)


def _rmsd_entry(group="group", value=1, seed=1):
    record = SimpleNamespace(run_id=f"{group}-{value}-{seed}", group=group,
                             parameter_values={"parameter": value}, seed=seed)
    record.parameter_value = lambda: value
    analyzer = SimpleNamespace(t=np.arange(12), rmsd_raw=np.linspace(1, 0.1, 12))
    return record, analyzer


def _text_boxes(fig, artists):
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in artists:
        if artist is None or not artist.get_visible():
            continue
        text = artist.get_text() if hasattr(artist, "get_text") else ""
        if hasattr(artist, "get_texts"):
            text = any(item.get_text() for item in artist.get_texts())
        if text:
            boxes.append(artist.get_window_extent(renderer))
    return boxes


def _artist_text_boxes(fig, artists):
    renderer = fig.canvas.get_renderer()
    result = []
    for artist in artists:
        if artist is None or not artist.get_visible():
            continue
        text = artist.get_text() if hasattr(artist, "get_text") else ""
        if hasattr(artist, "get_texts"):
            text = any(item.get_text() for item in artist.get_texts())
        if text:
            result.append((artist, artist.get_window_extent(renderer)))
    return result


def _axis_decoration_artists(axis):
    return [axis.title, axis.xaxis.label, axis.yaxis.label,
            *axis.get_xticklabels(), *axis.get_yticklabels(), axis.get_legend()]


def _assert_no_decoration_overlap(fig, axes, nrows, ncols):
    for row in range(nrows):
        for col in range(ncols - 1):
            left = _text_boxes(fig, _axis_decoration_artists(axes[row * ncols + col]))
            right = _text_boxes(fig, _axis_decoration_artists(axes[row * ncols + col + 1]))
            assert not any(a.overlaps(b) for a in left for b in right)
    for row in range(nrows - 1):
        for col in range(ncols):
            upper = _artist_text_boxes(fig, _axis_decoration_artists(axes[row * ncols + col]))
            lower = _artist_text_boxes(fig, _axis_decoration_artists(axes[(row + 1) * ncols + col]))
            for upper_artist, a in upper:
                for lower_artist, b in lower:
                    if a.overlaps(b):
                        raise AssertionError(f"vertical overlap: {upper_artist!r} with {lower_artist!r}")


def test_default_multipanel_grid_is_two_columns():
    config = PlotConfig()
    assert [multipanel_grid(count, config) for count in range(1, 7)] == [
        (1, 1), (1, 2), (2, 2), (2, 2), (3, 2), (3, 2)
    ]


def test_rmsd_multipanel_uses_shared_labels_and_compact_legends(tmp_path):
    pytest.importorskip("pubready")
    config = PlotConfig(formats=("pdf",))
    groups = [[_rmsd_entry(group=f"long group {index}", value=value)
               for value in range(1, 4)] for index in range(4)]
    with publication_style(config):
        fig, axes = make_figure(config, kind="multipanel", nrows=2, ncols=2, sharex=True)
        axes_list = flatten_axes(axes)
        for axis, entries in zip(axes_list, groups):
            plot_group_panel(axis, entries, divisor=20, show_xlabel=False, show_ylabel=False)
            axis.label_outer()
        set_shared_labels(fig, xlabel="Time (ns)", ylabel="RMSD")
        report = finalize_multipanel_layout(fig)
        save_figure(fig, tmp_path / "rmsd_multipanel", config, fit=False)
        fig.canvas.draw()
        assert report.feasible
        assert report.target == "si"
        assert report.fraction == "full"
        assert [axis.get_xlabel() for axis in axes_list] == ["", "", "", ""]
        assert [axis.get_ylabel() for axis in axes_list] == ["", "", "", ""]
        assert fig._wtm_shared_xlabel.get_text() == "Time (ns)"
        assert fig._wtm_shared_ylabel.get_text() == "RMSD"
        assert len(fig.texts) == 2
        expected_handle_length = plt.rcParams["legend.handlelength"] / 3.0
        for axis in axes_list:
            legend = axis.get_legend()
            assert getattr(legend, "_ncols", None) == 2
            assert legend.handlelength == pytest.approx(expected_handle_length)
        _assert_no_decoration_overlap(fig, axes_list, 2, 2)
        xlabel_box = _text_boxes(fig, [fig._wtm_shared_xlabel])[0]
        ylabel_box = _text_boxes(fig, [fig._wtm_shared_ylabel])[0]
        bottom_tick_boxes = _text_boxes(fig, [*axes_list[2].get_xticklabels(), *axes_list[3].get_xticklabels()])
        left_tick_boxes = _text_boxes(fig, [*axes_list[0].get_yticklabels(), *axes_list[2].get_yticklabels()])
        assert not any(xlabel_box.overlaps(box) for box in bottom_tick_boxes), (xlabel_box.bounds, [box.bounds for box in bottom_tick_boxes])
        assert not any(ylabel_box.overlaps(box) for box in left_tick_boxes), (ylabel_box.bounds, [box.bounds for box in left_tick_boxes])
        close_figure(fig)


def test_rmsd_standalone_panels_keep_labels_and_two_column_legends():
    entries = [_rmsd_entry(value=1), _rmsd_entry(value=2)]
    first = plt.subplots()[1]
    plot_group_panel(first, entries, divisor=20)
    assert first.get_xlabel() == "Time (ns)"
    assert first.get_ylabel() == "RMSD"
    assert getattr(first.get_legend(), "_ncols", None) == 2
    plt.close(first.figure)

    second = plt.subplots()[1]
    plot_seed_panel(second, entries, divisor=1)
    assert second.get_xlabel() == "Snapshot Index"
    assert second.get_ylabel() == "RMSD"
    assert getattr(second.get_legend(), "_ncols", None) == 2
    plt.close(second.figure)


def test_seed_rmsd_legend_handle_length_uses_active_style():
    axis = plt.subplots()[1]
    plot_seed_panel(axis, [_rmsd_entry(value=1, seed=1), _rmsd_entry(value=1, seed=2)])
    legend = axis.get_legend()
    assert legend.handlelength == pytest.approx(plt.rcParams["legend.handlelength"] / 3.0)
    plt.close(axis.figure)


def test_convergence_summary_multipanel_has_one_common_ylabel():
    pytest.importorskip("pubready")
    names = ["very_long_parameter_name_one", "parameter_two", "parameter_three",
             "parameter_four", "parameter_five", "parameter_six"]
    rows = [{"group": "set-A", "parameter_values": {name: value},
             "mean": 10 + value, "std": 1, "min": 8, "max": 12}
            for value, name in enumerate(names, 1)]
    config = PlotConfig(formats=("pdf",))
    with publication_style(config):
        nrows, ncols = multipanel_grid(len(names), config)
        fig, axes = make_figure(config, kind="multipanel", nrows=nrows, ncols=ncols, sharey=True)
        axes_list = flatten_axes(axes)
        for axis, name in zip(axes_list, names):
            plot_summary_panel(axis, rows, name, show_ylabel=False)
        set_shared_labels(fig, ylabel="Convergence (ns)")
        report = finalize_multipanel_layout(fig)
        fig.canvas.draw()
        assert report.feasible
        assert [axis.get_xlabel() for axis in axes_list] == names
        assert all(axis.get_ylabel() == "" for axis in axes_list)
        assert fig._wtm_shared_ylabel.get_text() == "Convergence (ns)"
        standalone = make_figure(config, kind="panel")
        standalone_axis = standalone[1] if hasattr(standalone[1], "get_ylabel") else standalone[1]
        plot_summary_panel(standalone_axis, rows, names[0])
        assert standalone_axis.get_ylabel() == "Convergence (ns)"
        close_figure(standalone[0])
        close_figure(fig)


def test_two_dimensional_multipanel_preserves_pubready_colorbars(tmp_path):
    pytest.importorskip("pubready")
    values = (1, 2, 3)
    rows = [{"parameter_values": {"x": x, "y": y}, "mean": x + y,
             "std": x * y} for x in values for y in values]
    config = PlotConfig(formats=("pdf",))
    with publication_style(config):
        fig, axes = make_figure(config, kind="multipanel", ncols=2)
        axes_list = flatten_axes(axes)
        _surface(axes_list[0], rows, "x", "y", "mean", 1, "viridis", "Mean")
        _surface(axes_list[1], rows, "x", "y", "std", 1, "magma", "Std")
        report = finalize_multipanel_layout(fig)
        save_figure(fig, tmp_path / "surface_multipanel", config, fit=False)
        assert report.feasible
        assert report.target == "si"
        assert report.fraction == "full"
        colorbars = [axis for axis in fig.axes if axis.get_label() == "<colorbar>"]
        assert len(colorbars) == 2
        fig.canvas.draw()
        colorbar_labels = [axis.yaxis.label.get_window_extent(fig.canvas.get_renderer())
                           for axis in colorbars]
        assert not colorbar_labels[0].overlaps(colorbar_labels[1])
        close_figure(fig)


def test_simple_reference_panel_is_median_only_with_custom_xlabel(tmp_path):
    pytest.importorskip("pubready")
    coords = (np.linspace(-1, 1, 8),)
    pmfs = [np.linspace(0, 2, 8), np.linspace(0.2, 2.2, 8), np.linspace(0.1, 1.9, 8)]
    data = compute_reference_pmf_with_outliers(coords, pmfs, 300,
                                                write_prefix=tmp_path / "reference")
    config = PlotConfig(formats=("pdf",))
    with publication_style(config):
        fig, ax = make_figure(config, kind="panel")
        plot_simple_reference_panel(ax, data, coords[0], xlabel="Reaction coordinate")
        save_figure(fig, tmp_path / "reference_median", config)
        fig.canvas.draw()
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), data["F_median"])
        assert ax.get_legend() is None
        assert ax.get_xlabel() == "Reaction coordinate"
        assert ax.get_ylabel() == "PMF (kcal/mol)"
        assert ax.get_title() == ""
        report = __import__("pubready").layout_report(fig)
        assert report.publisher == "acs"
        assert report.target == "double"
        assert report.fraction == "quarter"
        assert (tmp_path / "reference_median.pdf").is_file()
        close_figure(fig)


def test_simple_reference_panel_uses_default_xlabel_and_shared_median(tmp_path):
    coords = (np.linspace(0, 1, 5),)
    median = np.array([0.0, 0.4, 0.8, 0.3, 0.1])
    data = {"F_median": median}
    fig, ax = plt.subplots()
    plot_simple_reference_panel(ax, data, coords[0])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), median)
    assert ax.get_xlabel() == "Coordinate"
    assert ax.get_title() == ""
    assert ax.get_legend() is None
    plt.close(fig)


def test_reference_comparison_panel_keeps_rich_diagnostic_curves():
    x = np.linspace(0, 1, 5)
    data = {"F_median": np.array([0, 1, 2, 1, 0.5]),
            "F_all": np.array([0, 0.8, 1.8, 1.1, 0.4]),
            "F_filtered": np.array([0, 0.9, 1.9, 1.0, 0.45]),
            "F_all_err": np.full(5, 0.1),
            "F_filtered_err": np.full(5, 0.05)}
    fig, ax = plt.subplots()
    plot_pmf_comparison_panel(ax, data, x)
    assert [line.get_label() for line in ax.lines] == [
        "Median", "Average (all)", "Average (filtered)"
    ]
    assert len(ax.collections) == 2
    assert ax.get_legend() is not None
    plt.close(fig)


def test_buildref_parser_accepts_simple_reference_options():
    args = build_parser().parse_args(["--pmf-file", "input.pmf",
                                      "--simple-reference-plot",
                                      "--xlabel", "Reaction coordinate"])
    assert args.simple_reference_plot is True
    assert args.xlabel == "Reaction coordinate"
