from pathlib import Path

import pytest

from cv.pipeline.frame_identity import frame_number_from_name


@pytest.mark.parametrize("index", [0, 1, 9999, 10000, 10001, 999999])
def test_frame_number_has_no_four_digit_ceiling(index):
    assert frame_number_from_name(Path("/frames") / f"f_{index:04d}.jpg") == index


@pytest.mark.parametrize(
    "name", ["frame_0001.jpg", "f_.jpg", "f_-1.jpg", "f_1.5.jpg", "f_001x.jpg"]
)
def test_malformed_frame_names_fail_closed(name):
    with pytest.raises(ValueError, match="frame identity"):
        frame_number_from_name(name)


def test_numeric_sort_preserves_the_transition_past_four_digits():
    names = ["f_10000.jpg", "f_0002.jpg", "f_9999.jpg"]
    assert sorted(names, key=frame_number_from_name) == ["f_0002.jpg", "f_9999.jpg", "f_10000.jpg"]


def test_event_and_physics_readers_preserve_long_ids():
    from cv.pipeline import bounce_detect, rich_ball_physics

    for module in (bounce_detect, rich_ball_physics):
        assert module.frame_num("f_10001.jpg") == 10001
        assert module.frame_num("10001") == 10001


def test_consensus_does_not_merge_distinct_long_frames(tmp_path):
    from cv.pipeline.ball_track_consensus import DecoderConfig, load_candidate_streams

    from cv.pipeline import resolution as res

    path = tmp_path / "candidates.csv"
    path.write_text(
        "clip,frame,x,y,score,rank\n"
        "pt0001,f_10000.jpg,10,20,0.9,0\n"
        "pt0001,f_10001.jpg,20,30,0.9,0\n"
    )
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.LEGACY_TRACKING_SIZE,
        source="test fixture",
        subnative_flagged=True,
        subnative_justification="fixture mirrors the shipped legacy candidate space",
    )
    candidates = load_candidate_streams([path], ["test"], None, DecoderConfig())
    assert sorted(candidates["pt0001"]) == [10000, 10001]


def test_pose_box_reader_preserves_long_frame_name(tmp_path):
    from cv.pipeline.ball import load_pose_boxes

    (tmp_path / "poses.csv").write_text("clip,frame,x0,y0,x1,y1\npt0001,f_10001.jpg,1,2,3,4\n")
    boxes = load_pose_boxes(str(tmp_path), "poses.csv")
    assert ("pt0001", "f_10001.jpg") in boxes


def test_classical_tracker_retains_its_standalone_help_entrypoint(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ball.py")), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
