import numpy as np
import pytest
import hashlib
from pathlib import Path

from cv.validation import s6_owner_court_audit as audit
from cv.validation.s6_owner_ground_camera import project
from cv.pipeline import camera_bundle


def test_original_owner_export_bytes_remain_immutable():
    source = Path(__file__).parent / "labels/s6_owner_inputs_v1/s6_inputs_20260906_v1_labels.json"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "cf34bd389fa6c6bf01f4a44eba603425fa27308b1ed54dd67e777f6d03308c6f"
    )


def fixture():
    matrix = camera_bundle.projection_matrix(
        np.array([5.4, -20.5, 7.6]), 1.57, -0.25, 2900, (1920, 1080)
    )
    points = {**audit.GROUND, **audit.NET}
    pixels = project(matrix, np.array(list(points.values())))
    return dict(
        case_id="test",
        complete=False,
        window_status="ambiguous",
        notes="keep original",
        frames=[
            dict(
                target_id=k,
                status="visible",
                x1080=float(p[0]),
                y1080=float(p[1]),
                uncertainty_radius_px1080=3,
            )
            for k, p in zip(points, pixels)
        ],
    )


def test_independent_net_prediction_preserves_owner_status_and_notes():
    row = fixture()
    result = audit.evaluate(row)
    assert result["fit"]["native_rms_px"] < 1e-6
    assert max(x["error_px"] for x in result["withheld_net"]) < 1e-5
    assert result["owner_notes"] == "keep original" and not result["owner_complete"]
    assert not result["labels_changed"] and not result["airborne_accuracy_certified"]
    row["frames"].reverse()
    assert audit.evaluate(row)["fit"] == result["fit"]


def test_landmark_reprojection_keeps_ambiguous_supports_out_of_denominator():
    row = fixture()
    row["frames"][-1]["status"] = "ambiguous"
    matrix = camera_bundle.projection_matrix(
        np.array([5.4, -20.5, 7.6]), 1.57, -0.25, 2900, (1920, 1080)
    )
    result = audit.landmark_reprojection_errors(row, matrix)
    assert result["visible_landmarks"] == 10
    assert result["withheld_or_ambiguous_landmarks"] == 1
    assert result["maximum_px"] < 1e-5


@pytest.mark.parametrize("kind", ["duplicate", "missing", "hidden", "invalid"])
def test_unsupported_landmarks_fail_without_repair(kind):
    row = fixture()
    if kind == "duplicate":
        row["frames"].append(row["frames"][0])
    elif kind == "missing":
        row["frames"].pop()
    elif kind == "hidden":
        row["frames"][0]["status"] = "occluded"
    else:
        row["frames"][0]["x1080"] = float("nan")
    with pytest.raises(ValueError):
        audit.evaluate(row)
