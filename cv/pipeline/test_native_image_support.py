"""Reflected crop pixels cannot become automatic native observations."""

import json

import cv2
import numpy as np
import pytest

from cv.pipeline import ball_local_refine as scalar
from cv.pipeline import ball_local_refine_batched as batched
from cv.pipeline import resolution as res
from cv.pipeline.s6_automatic_observations import native_ball_support


@pytest.mark.parametrize("method", ["argmax", "centroid", "parabolic"])
def test_fully_native_peak_arithmetic_is_exact(method):
    rng = np.random.default_rng(4)
    heatmap = rng.random((24, 32)).astype(np.float32)
    region = scalar.RecoveryRegion("pt1", 3, 100, 100, 64, 48, "observed")
    assert scalar.supported_peaks(
        heatmap, region, res.FrameSize(200, 200), 5, 0.05, 3, method
    ) == scalar.refined_peaks(heatmap, 5, 0.05, 3, method)


def test_padding_peak_is_masked_before_topk_and_entire_padding_abstains():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[8, 5] = 1.0  # stronger reflected ghost below image
    heatmap[3, 5] = 0.6  # genuine native-supported candidate
    region = scalar.RecoveryRegion("pt1", 3, 50, 100, 20, 20, "forward")
    audit = {}
    peaks = scalar.supported_peaks(
        heatmap, region, res.FrameSize(100, 100), 1, 0.05, 1, "argmax", audit
    )
    assert peaks == [(5.0, 3.0, pytest.approx(0.6))]
    assert audit["unsupported_heatmap_cells"] == 50
    region = scalar.RecoveryRegion("pt1", 3, 50, 130, 20, 20, "forward")
    assert (
        scalar.supported_peaks(
            heatmap, region, res.FrameSize(100, 100), 1, 0.05, 1, "argmax", audit
        )
        == []
    )
    assert audit["fully_unsupported_regions"] == 1


def test_actual_reflected_rgb_crop_has_ghost_but_only_native_peak_admitted():
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(frame, (50, 90), 2, (0, 255, 0), -1)
    region = scalar.RecoveryRegion("pt1", 1, 50, 110, 40, 80, "forward")
    tensor = scalar._crop_tensor(frame, region, res.FrameSize(100, 100))
    green = (tensor[1] * scalar.STD[1] + scalar.MEAN[1]).clip(0, 1)
    raw = scalar.refined_peaks(green, 2, 0.5, 30, "argmax")
    raw_points = [
        scalar.heatmap_to_artifact(x, y, green.shape[1], green.shape[0], region) for x, y, _ in raw
    ]
    assert any(not res.points_inside_image(p, res.FrameSize(100, 100)) for p in raw_points)
    kept = scalar.supported_peaks(green, region, res.FrameSize(100, 100), 2, 0.5, 30, "argmax")
    assert kept
    assert all(
        res.points_inside_image(
            scalar.heatmap_to_artifact(x, y, green.shape[1], green.shape[0], region),
            res.FrameSize(100, 100),
        )
        for x, y, _ in kept
    )


def test_scalar_and_batched_production_rows_have_identical_native_support(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    frames = tmp_path / "pt1"
    frames.mkdir()
    for f in range(1, 4):
        cv2.imwrite(str(frames / f"f_{f:04d}.jpg"), np.zeros((100, 100, 3), np.uint8))
    region = scalar.RecoveryRegion("pt1", 2, 50, 100, 20, 20, "forward")
    heatmap = np.full((10, 10), 0.001, np.float32)
    heatmap[8, 5] = 0.99
    heatmap[3, 5] = 0.6
    probabilities = torch.from_numpy(np.tile(heatmap, (1, 3, 1, 1)))
    logits = torch.logit(probabilities)
    original_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **kw: (
            self if a and str(a[0]).startswith("cuda") else original_to(self, *a, **kw)
        ),
    )

    def model(x):
        return (logits,)

    scalar_audit = {}
    actual = scalar.infer_regions(
        model,
        tmp_path,
        [region],
        0,
        res.FrameSize(100, 100),
        k_best=1,
        temporal_ensemble=False,
        subpixel="argmax",
        native_support_audit=scalar_audit,
    )
    batch_audit = {}
    expected = batched._rows_from_outputs(
        [region],
        logits.sigmoid().numpy(),
        batched._contexts(False),
        1,
        0.05,
        4,
        False,
        "argmax",
        res.FrameSize(100, 100),
        batch_audit,
    )
    assert actual == expected
    assert scalar_audit == batch_audit
    assert len(actual) == 1 and actual[0]["y"] == 97


@pytest.mark.parametrize("xy", [[-1, 50], [50, 1080], [1920, 50], [1028, 1374], [50, float("nan")]])
def test_old_automatic_artifact_outside_centres_are_retained_but_not_observed(xy):
    raw = {
        "frame": 285,
        "status": "visible",
        "x1080": xy[0],
        "y1080": xy[1],
        "automatic_sources": "coarse_lock",
        "guide_support": {"sources": "detector"},
    }
    rows, receipt = native_ball_support([raw], res.NATIVE_SIZE)
    assert raw["status"] == "visible"
    assert rows[0]["status"] == "unsupported"
    assert rows[0]["x1080"] == raw["x1080"]
    assert receipt["refused_rows"] == 1
    assert receipt["refusals"][0]["original_status"] == "visible"


def test_native_support_preserves_in_picture_rows_exactly():
    rows = [
        {"frame": 1, "status": "visible", "x1080": 1919.99, "y1080": 1079.99},
        {"frame": 2, "status": "derived_estimate", "x1080": 0, "y1080": 0},
    ]
    before = json.dumps(rows)
    after, receipt = native_ball_support(rows, res.NATIVE_SIZE)
    assert json.dumps(after) == before
    assert receipt["refused_rows"] == 0
