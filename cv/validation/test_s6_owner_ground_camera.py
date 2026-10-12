from copy import deepcopy

import numpy as np
import pytest

from cv.pipeline import camera_bundle, provenance
from cv.validation import s6_owner_ground_camera as control


def fixture():
    xyz = np.array(
        [
            [0, 0, 0],
            [10.97, 0, 0],
            [0, 23.77, 0],
            [10.97, 23.77, 0],
            [1.37, 5.485, 0],
            [9.6, 5.485, 0],
            [1.37, 18.285, 0],
            [9.6, 18.285, 0],
        ]
    )
    matrix = camera_bundle.projection_matrix(
        np.array([5.4, -20.5, 7.6]), 1.57, -0.25, 2900, (1920, 1080)
    )
    pixels = control.project(matrix, xyz)
    return xyz, pixels, matrix


def test_exact_physical_camera_recovers_ground_and_airborne_under_its_assumptions():
    xyz, pixels, expected = fixture()
    result = control.fit_ground(xyz, pixels)
    matrix = np.array(result["P"])
    assert result["native_rms_px"] < 1e-6
    assert result["focal_native_px"] == pytest.approx(2900, abs=1e-4)
    assert result["camera_center_m"] == pytest.approx([5.4, -20.5, 7.6], abs=1e-6)
    air = xyz + [0, 0, 2.5]
    assert np.max(np.abs(control.project(matrix, air) - control.project(expected, air))) < 1e-5
    # The camera is a single rigid P, never a ground-exact spliced matrix.
    parameters = np.array(result["parameters"])
    rotation = control.Rotation.from_rotvec(parameters[:3]).as_matrix()
    assert np.linalg.det(rotation) == pytest.approx(1)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_leave_one_out_is_separate_and_does_not_certify_depth_or_temporal_transfer():
    xyz, pixels, _ = fixture()
    pixels[0] += [2, -1]
    labels = [
        dict(
            landmark_id=str(i),
            court_x_m=str(x),
            court_y_m=str(y),
            corrected_x540=str(u / 2),
            corrected_y540=str(v / 2),
            coordinate_convention="itf_outside_edge_v1",
        )
        for i, ((x, y, _), (u, v)) in enumerate(zip(xyz, pixels, strict=True))
    ]
    original = deepcopy(labels)
    row = control.evaluate(labels)
    assert labels == original
    assert len(row["leave_one_landmark_out"]) == 8
    assert all(r["status"] == "measured" for r in row["leave_one_landmark_out"])
    assert row["leave_one_landmark_out"][0]["native_error_px"] == pytest.approx(
        np.sqrt(5), abs=1e-5
    )
    assert not row["airborne_metric_accuracy_certified"]
    assert not row["temporal_camera_validated"]
    assert not row["automatic_inference_eligible"]


@pytest.mark.parametrize(
    "kind", ["short", "duplicate", "collinear", "above_ground", "nan", "pixel_shape"]
)
def test_rejects_invalid_ground_inputs(kind):
    xyz, pixels, _ = fixture()
    if kind == "short":
        xyz, pixels = xyz[:5], pixels[:5]
    elif kind == "duplicate":
        xyz[0] = xyz[1]
    elif kind == "collinear":
        xyz[:, 1] = 0
    elif kind == "above_ground":
        xyz[0, 2] = 1
    elif kind == "nan":
        pixels[0, 0] = np.nan
    else:
        pixels = pixels[:, :1]
    with pytest.raises(ValueError, match="ground landmarks"):
        control.fit_ground(xyz, pixels)


def test_behind_camera_projection_is_not_a_finite_valid_fit():
    with pytest.raises(ValueError, match="in front"):
        control.project(np.eye(3, 4), np.array([[1, 1, -1]]))


def test_source_records_are_verified_and_confined(monkeypatch, tmp_path):
    monkeypatch.setattr(control.paths, "REPO_ROOT", tmp_path)
    source = tmp_path / "source.json"
    source.write_text("{}")
    record = provenance.file_record(source)
    assert control.resolve_record(record) == source
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="bytes changed"):
        control.resolve_record(record)
    with pytest.raises(ValueError, match="root"):
        control.resolve_record({**record, "path": "../source.json"})


def test_overwrite_is_rejected_before_reading_packet(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv", ["control", "--packet", "missing.json", "--output", str(tmp_path)]
    )
    with pytest.raises(FileExistsError):
        control.main()
