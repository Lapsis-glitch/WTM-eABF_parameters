import csv
from pathlib import Path

import pytest

from Convergence_evaluation.input_discovery import discover_runs, read_manifest
from Convergence_evaluation.outputs import analysis_output
from Convergence_evaluation.plotting import PlotConfig, close_figure, make_figure, publication_style
from Convergence_evaluation.analyze_ND import plot_rmsd_panel
from Convergence_evaluation.plot_results import plot_summary


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
