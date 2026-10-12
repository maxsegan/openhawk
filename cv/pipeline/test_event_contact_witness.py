import pytest

from cv.pipeline import event_contact_witness as witness


def test_audio_clock_does_not_reset_source_epoch_or_bridge_gap():
    receipt = witness.validate_audio_frames([(0, 123.5, 160), (1, 123.51, 160)], 320, 16000)
    assert receipt["first_pts_seconds"] == 123.5
    with pytest.raises(ValueError, match="gap"):
        witness.validate_audio_frames([(0, 123.5, 160), (1, 123.54, 160)], 320, 16000)
    with pytest.raises(ValueError, match="every emitted"):
        witness.validate_audio_frames([(0, 123.5, 160)], 159, 16000)


def test_multiple_audio_transients_do_not_choose_nearest_or_strongest():
    peaks = [
        {"source_pts_seconds": 10.02, "support": 0.6},
        {"source_pts_seconds": 10.05, "support": 1.0},
    ]
    assert not witness.audio_support(peaks, (9.99, 10.01))["supported"]
    supported = witness.audio_support(peaks[:1], (9.99, 10.01))
    assert supported["supported"]
    assert not supported["event_time_refined"]
    assert not witness.audio_support(peaks[:1], (10.03, 10.04))["supported"]


def pose_row(f):
    return {
        "frame": f,
        "source_pts_seconds": f * 0.04 + 100,
        "track_id": "near-1",
        "box_xyxy_native": [0, 0, 100, 200],
        "joints": {j: [30 + f * f, 60, 0.8] for j in witness.JOINTS},
    }


def test_pose_does_not_interpolate_missing_frames_or_cross_cuts():
    rows = [pose_row(f) for f in range(1, 10)]
    series = witness.pose_motion(rows, set())
    assert series[4]["supported"]
    missing = witness.pose_motion([r for r in rows if r["frame"] != 5], set())
    assert not next(r for r in missing if r["frame"] == 4)["supported"]
    assert not next(r for r in missing if r["frame"] == 6)["supported"]
    cut = witness.pose_motion(rows, {5})
    assert not cut[3]["supported"] and not cut[4]["supported"]
    assert cut[6]["supported"]


def test_static_pose_and_unobserved_joint_do_not_create_motion():
    rows = [pose_row(f) for f in range(1, 10)]
    for row in rows:
        row["joints"] = {j: [30, 60, 0.8] for j in witness.JOINTS}
    assert all(r.get("score_z", 0) == 0 for r in witness.pose_motion(rows, set()))
    for row in rows:
        row["joints"] = {j: [30, 60, 0.01] for j in witness.JOINTS}
    assert not any(r["supported"] for r in witness.pose_motion(rows, set()))
