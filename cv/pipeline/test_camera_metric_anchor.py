"""Automatic anchor continuation must use existing physical calibration, not raw rank."""

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import camera_cal, court_topology_runner as runner


def anchor(index):
    return runner.AnchorSolution(index, f"f_{index:04d}.jpg", np.eye(3), "test", 1, 1, None, {}, {})


def test_metric_mode_continues_in_original_order_and_stops(monkeypatch, tmp_path):
    frames = [tmp_path / f"f_{i:04d}.jpg" for i in range(101)]
    visited = []
    monkeypatch.setattr(runner, "_solve_anchor_frame", lambda i, _path: anchor(i))

    def qualify(candidate, _path, **_kw):
        visited.append(candidate.index)
        return {"reliable": candidate.index != 30, "reason": "test"}

    monkeypatch.setattr(runner, "qualify_metric_anchor", qualify)
    selected, attempts = runner.select_anchor(frames, "first_metric_success")
    assert visited == [30, 70]
    assert selected.index == 70
    assert selected.metric_evidence["reliable"]
    assert attempts[0]["frame"] == frames[30].name
    assert attempts[0]["error"] == "metric_calibration_rejected"
    visited.clear()
    assert runner.select_anchor(frames)[0].index == 30
    assert visited == []


def test_metric_failures_do_not_promote_fallback_or_expand_solved_candidates(monkeypatch, tmp_path):
    frames = [tmp_path / f"f_{i:04d}.jpg" for i in range(101)]
    visited = []

    def solve(i, _path, *, proposal_pool=48):
        visited.append((i, proposal_pool))
        return anchor(i)

    monkeypatch.setattr(runner, "_solve_anchor_frame", solve)
    monkeypatch.setattr(runner, "qualify_metric_anchor", lambda *_a, **_k: {"reliable": False})
    selected, attempts = runner.select_anchor(frames, "first_metric_success")
    assert selected is None
    assert len(attempts) == 10
    assert all(pool == 48 for _, pool in visited)


def test_metric_qualification_uses_native_size_and_emitted_edge_convention(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.cv2, "imread", lambda _p: np.zeros((120, 240, 3), np.uint8))
    transform = np.diag([2.0, 3.0, 1.0])
    monkeypatch.setattr(runner, "edge_convention_world_transform", lambda _: (transform, {}))
    jobs = []

    def measure(job):
        jobs.append(job)
        return 0, Path(job[2]).name, ["observed"], 400.0, "native_tape"

    monkeypatch.setattr(camera_cal, "measure_point_net", measure)
    monkeypatch.setattr(
        camera_cal, "direct_projection_from_net", lambda *_a, **_k: (None, {"reliable": False})
    )
    result = runner.qualify_metric_anchor(anchor(0), tmp_path / "f_0000.jpg")
    np.testing.assert_allclose(jobs[0][1], np.linalg.inv(transform))
    assert jobs[0][3:5] == (240, 120)
    assert not result["reliable"]
    assert result["net_observation_source"] == "native_tape"


@pytest.mark.parametrize(
    "ground,net,check,height,accepted",
    [
        (0.5, 8, 0.5, 3, True),
        (0.501, 1, 0, 10, False),
        (0, 8.001, 0, 10, False),
        (0, float("nan"), 0, 10, False),
        (0, 1, 0.501, 10, False),
        (0, 1, 0, 2.999, False),
    ],
)
def test_direct_projection_preserves_existing_boundaries(
    monkeypatch, ground, net, check, height, accepted
):
    P = np.arange(12).reshape(3, 4)
    monkeypatch.setattr(camera_cal, "fixed_f_projection_from_ground", lambda *_a, **_k: P)
    monkeypatch.setattr(camera_cal, "calibration_residuals", lambda *_a: (ground, net))
    monkeypatch.setattr(
        camera_cal,
        "net_reprojection_check",
        lambda *_a: {"ground_err_px": check, "net_px": [height]},
    )
    result, evidence = camera_cal.direct_projection_from_net(
        np.eye(3), ["observed"], 500, w=240, h=120
    )
    assert (result is P) == accepted
    assert evidence["reliable"] == accepted


def test_missing_net_cannot_be_replaced_by_intrinsic_projection(monkeypatch):
    monkeypatch.setattr(
        camera_cal,
        "fixed_f_projection_from_ground",
        lambda *_a, **_k: pytest.fail("no direct evidence"),
    )
    result, evidence = camera_cal.direct_projection_from_net(np.eye(3), None, 500, w=240, h=120)
    assert result is None
    assert evidence["reason"] == "missing_net_or_focal"


def test_match_prior_cannot_bypass_metric_qualification(monkeypatch, tmp_path):
    class InlinePool:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def submit(self, function, *args, **kwargs):
            future = Future()
            future.set_result(function(*args, **kwargs))
            return future

    clip = tmp_path / "frames/pt0001"
    clip.mkdir(parents=True)
    (clip / "f_0000.jpg").touch()
    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(
        runner,
        "calibrate_clip",
        lambda *_a, **_k: (
            1,
            {"point": 1, "status": "abstained", "attempts": []},
            np.full((3, 3), np.nan),
            {},
        ),
    )
    monkeypatch.setattr(runner, "recover_anchor_from_match_prior", lambda *_a: anchor(0))
    monkeypatch.setattr(runner, "qualify_metric_anchor", lambda *_a, **_k: {"reliable": False})
    result = runner.calibrate_points(tmp_path, "frames", jobs=1, anchor_mode="first_metric_success")
    assert result["direct"] == 0
    assert result["match_prior_recovered"] == 0
    with np.load(tmp_path / "court_H_per_point.npz") as data:
        assert np.isnan(data["H"]).all()
