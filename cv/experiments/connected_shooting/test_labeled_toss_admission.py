"""Admission preserves original evidence and cannot cross the contact boundary."""

import numpy as np
import pytest

from cv.experiments.connected_shooting.labeled_toss_admission import SCHEMA, admit
from cv.experiments.connected_shooting.labeled_prefix_joint_impact import _prefix_rows


@pytest.fixture
def source():
    points = [
        dict(frame=f, status="visible", x1080=float(f), y1080=20.0, uncertainty_radius_px1080=3.0)
        for f in range(1, 31)
    ]
    points[4]["status"] = "ambiguous"
    labels = {"ball": {"records": [{"clip": "p", "frames": points}]}}
    cameras = {
        "cameras": [
            dict(frame=f, status="supported", P=np.eye(3, 4).tolist()) for f in range(1, 31)
        ]
    }
    original = {
        "status": "supported",
        "minimum_observations": 6,
        "rows": [
            dict(
                frame=f,
                pixel=[float(f), 20.0],
                camera=np.eye(3, 4).tolist(),
                uncertainty_px=3.0,
                source="frozen_labeled_front",
            )
            for f in range(20, 30)
        ],
    }
    support = dict(
        schema=SCHEMA,
        status="supported",
        frame_interval=[5, 29],
        rationale="free after held-ball frames",
        annotation_origin="test",
    )
    return original, dict(
        labels=labels,
        cameras=cameras,
        clip="p",
        original_contact_interval=(29.5, 30.5),
        exposure_frames=0.25,
        support=support,
        enabled=True,
    )


def test_extended_original_rows_reach_prefix_without_rewindow(source):
    original, args = source
    original["rows"][0]["uncertainty_px"] = 7.0
    out, receipt = admit(original, **args)
    rows = _prefix_rows(out, 29.5, 0.25)
    assert [r["frame"] for r in rows] == list(range(6, 30))
    assert receipt["added_frames"] == list(range(6, 20))
    assert out["abstained"] == [dict(frame=5, reason="original ball absent or abstained")]
    assert rows[0]["pixel"] == [6.0, 20.0] and rows[0]["uncertainty_px"] == 3.0
    assert next(r for r in rows if r["frame"] == 20)["uncertainty_px"] == 7.0
    assert out["upstream_observation_receipt"]["minimum_observations"] == 6
    assert len(original["rows"]) == 10


def test_disabled_absent_same_span_and_explicit_no_extension_are_exact(source):
    original, args = source
    assert admit(original, **(args | dict(enabled=False, labels={})))[0] is original
    assert admit(original, **(args | dict(support=None)))[1]["status"] == "absent_support"
    args["support"]["frame_interval"] = [20, 29]
    out, receipt = admit(original, **args)
    assert out is original and receipt["status"] == "same_observations"
    args["support"]["status"] = "no_extension"
    assert admit(original, **args)[1]["status"] == "explicit_no_extension"


@pytest.mark.parametrize(
    "change,match",
    [
        ({"frame_interval": [5, 30]}, "ORIGINAL"),
        ({"frame_interval": [21, 29]}, "remove original"),
        ({"frame_interval": [5.5, 29]}, "native integer"),
        ({"rationale": ""}, "native free-motion"),
    ],
)
def test_invalid_support_never_silently_falls_back(source, change, match):
    original, args = source
    args["support"].update(change)
    with pytest.raises(ValueError, match=match):
        admit(original, **args)


def test_bound_camera_pixels_and_foreign_clip_required(source):
    original, args = source
    with pytest.raises(ValueError, match="supported original"):
        admit(original, **(args | dict(clip="other")))
    original["rows"][0]["pixel"] = [99.0, 20.0]
    with pytest.raises(ValueError, match="differs from original"):
        admit(original, **args)


def test_unsupported_camera_abstention_and_nonzero_radial_rejection(source):
    original, args = source
    args["cameras"]["cameras"][5]["status"] = "unsupported"
    out, receipt = admit(original, **args)
    assert 6 not in receipt["admitted_frames"]
    args["cameras"]["cameras"][6].update(k1=0.1, dist_center=[0, 0])
    with pytest.raises(ValueError, match="radial"):
        admit(original, **args)
