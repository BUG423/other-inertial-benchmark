"""Plot contracts: physical error dimensions and complete model coverage."""

from pathlib import Path

import numpy as np
import pytest

from inertial_benchmark.engine.results import RunResult, SequenceResult
from inertial_benchmark.utils import plotting
from inertial_benchmark.utils.reports import make_plots


@pytest.mark.parametrize("requested,predicted,expected,effective", [
    (2, 3, 4.0, 2),
    (3, 3, 5.0, 3),
    (3, 2, 4.0, 2),
])
def test_position_plots_match_metric_dimensions(
        tmp_path, monkeypatch, requested, predicted, expected, effective):
    # A 4 m horizontal error and 3 m vertical error form a 3-4-5 triangle.
    seq = SequenceResult(
        sequence_id="triangle",
        t=np.arange(3, dtype=float) / 200.0,
        pos_pred=np.tile(np.array([4.0, 0.0, 3.0])[:predicted], (3, 1)),
        pos_gt=np.zeros((3, 3)),
        pos_oracle=None,
        valid_pose=np.array([True, False, True]),
    )
    assert seq.compute_metrics(dims=requested)["ate"] == pytest.approx(expected)
    run = RunResult([seq], model="triangle", cfg={"metric_dims": requested})
    figures = {}
    original_cdf = plotting.plot_error_cdf
    original_time = plotting.plot_error_over_time

    def capture_cdf(errors, path=None, **kwargs):
        figures["cdf"] = original_cdf(errors, path=None, **kwargs)

    def capture_time(curves, path=None, max_n=12, **kwargs):
        figures["time"] = original_time(curves, path=None, max_n=max_n, **kwargs)

    for name in ("plot_trajectory", "plot_trajectory_grid", "plot_boxplot", "plot_length_ratio"):
        monkeypatch.setattr(plotting, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting, "plot_error_cdf", capture_cdf)
    monkeypatch.setattr(plotting, "plot_error_over_time", capture_time)
    run.plot(tmp_path)

    cdf = figures["cdf"].axes[0]
    np.testing.assert_allclose(cdf.lines[0].get_xdata(), [expected, expected])
    assert cdf.get_xlabel() == f"{effective}D position error (m)"
    curve = figures["time"].axes[0]
    np.testing.assert_allclose(curve.lines[0].get_ydata(), [expected, np.nan, expected],
                               equal_nan=True)
    assert curve.get_ylabel() == f"{effective}D position error (m)"


def _report_inputs(count):
    pd = pytest.importorskip("pandas")
    models = [f"m{k:02d}" for k in range(count)]
    rows, summary = [], []
    for k, model in enumerate(models):
        # The final model dominates all earlier pages in both coordinates.
        value = float(count - k)
        rows.extend({"dataset": "syn", "split": "test", "model": model,
                     "sequence_id": f"s{j}", "ate": value + j / 10,
                     "plr": 1.0, "distance_m": 10.0 + j} for j in range(2))
        summary.append({"dataset": "syn", "split": "test", "model": model,
                        "params": 1000.0 * value, "ate_mean": value})
    return pd.DataFrame(rows), pd.DataFrame(summary), models


def _capture_report_plots(monkeypatch):
    captures = {"box": [], "cdf": [], "plr": [], "pareto": []}
    original_cdf = plotting.plot_error_cdf
    original_pareto = plotting.plot_pareto

    def capture_box(values, path=None, **kwargs):
        captures["box"].append((Path(path).name, tuple(values)))

    def capture_cdf(values, path=None, **kwargs):
        captures["cdf"].append((Path(path).name, tuple(values)))
        # Exercise the real palette/render path: an unpaged 17-series call fails.
        return original_cdf(values, path=None, **kwargs)

    def capture_plr(points, path=None):
        captures["plr"].append((Path(path).name, tuple(points)))

    def capture_pareto(labels, x, y, path=None, **kwargs):
        fig = original_pareto(labels, x, y, path=None, **kwargs)
        captures["pareto"].append((tuple(labels), fig))
        return fig

    monkeypatch.setattr(plotting, "plot_boxplot", capture_box)
    monkeypatch.setattr(plotting, "plot_error_cdf", capture_cdf)
    monkeypatch.setattr(plotting, "plot_length_ratio", capture_plr)
    monkeypatch.setattr(plotting, "plot_pareto", capture_pareto)
    return captures


def test_seventeen_models_are_paged_without_losing_global_pareto(tmp_path, monkeypatch):
    data, summary, models = _report_inputs(17)
    captures = _capture_report_plots(monkeypatch)
    paths = make_plots(data, summary, tmp_path)

    for kind in ("box", "cdf", "plr"):
        pages = captures[kind]
        assert [len(labels) for _, labels in pages] == [8, 8, 1]
        assert [model for _, labels in pages for model in labels] == models
        assert all(f"page{k:02d}" in name for k, (name, _) in enumerate(pages, 1))
    assert len(paths) == 10  # Three plot types per page plus one global Pareto plot.
    assert len(captures["pareto"]) == 1
    labels, fig = captures["pareto"][0]
    assert set(labels) == set(models)
    assert len(fig.axes[0].collections[0].get_offsets()) == 17
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1000.0])
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [1.0])


def test_eight_model_report_keeps_existing_filenames(tmp_path, monkeypatch):
    data, summary, models = _report_inputs(8)
    captures = _capture_report_plots(monkeypatch)
    paths = make_plots(data, summary, tmp_path)
    assert {path.name for path in paths} == {
        "box_ate_syn_test.png", "cdf_ate_syn_test.png", "plr_syn_test.png",
        "pareto_params_ate_syn_test.png",
    }
    assert [model for _, labels in captures["cdf"] for model in labels] == models
