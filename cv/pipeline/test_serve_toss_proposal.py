from __future__ import annotations

import numpy as np

from cv.pipeline import serve_toss_proposal as toss

FPS = 50.0
FAR_BOX = (1000.0, 80.0, 1040.0, 220.0)  # height 140; the contact is near its top
NEAR_BOX = (800.0, 600.0, 900.0, 850.0)


def _serve_track(*, contact: int = 100, server: str = "far") -> dict[int, np.ndarray]:
    """A far toss rising to an apex at y=40 and struck six frames later at ``contact``."""

    track = {}
    apex = contact - 6
    g = 0.9  # px per frame^2 in the picture
    for frame in range(apex - 20, contact + 1):
        y = 40.0 + g * (frame - apex) ** 2 / 2
        track[frame] = np.asarray([1020.0 + 0.1 * (frame - apex), y])
    last = track[contact]
    velocity = np.asarray([-6.0, 14.0]) if server == "far" else np.asarray([-6.0, -14.0])
    for step in range(1, 16):
        track[contact + step] = last + velocity * step
    return track


def _boxes(box, frames) -> dict[int, dict[str, tuple]]:
    return {frame: {"far": box, "near": NEAR_BOX} for frame in frames}


def test_far_serve_is_proposed_at_the_departure() -> None:
    track = _serve_track()
    found = toss.propose(track, _boxes(FAR_BOX, track), FPS)
    assert len(found) == 1
    assert found[0].server == "far"
    assert abs(found[0].frame - 100) <= 1.0


def test_ball_that_keeps_falling_on_the_toss_is_not_a_serve() -> None:
    track = {
        frame: np.asarray([1020.0, 40.0 + 0.45 * (frame - 88) ** 2]) for frame in range(68, 120)
    }
    assert toss.propose(track, _boxes(FAR_BOX, track), FPS) == []


def test_departure_away_from_the_other_end_is_refused() -> None:
    # A far server's ball that leaves upward is not heading to the near court.
    track = _serve_track(server="near")
    assert toss.propose(track, _boxes(FAR_BOX, track), FPS) == []


def test_apex_must_be_above_the_server_box() -> None:
    track = _serve_track()
    low_box = (1000.0, 20.0, 1040.0, 160.0)  # the apex (y=40) is inside, not above
    assert toss.propose(track, _boxes(low_box, track), FPS) == []


def _row(frame: float, event_type: str = "contact", abstain: bool = False) -> dict:
    return {
        "match_id": "m",
        "clip": "m__pt0001",
        "event_type": event_type,
        "frame": frame,
        "abstain": abstain,
        "model_abstain": abstain,
    }


def _admit(rows):
    track = _serve_track()
    boxes = {frame: [("far", FAR_BOX), ("near", NEAR_BOX)] for frame in track}
    return toss.admit_serve_proposals(
        rows, match_id="m", clip="pt0001", track=track, boxes_by_frame=boxes, fps=FPS
    )


def test_admission_adds_a_row_when_the_decoder_has_no_serve() -> None:
    rows, census = _admit([_row(160.0)])
    assert census["added"] == 1
    added = [row for row in rows if row.get("source") == toss.SOURCE]
    assert len(added) == 1 and added[0]["abstain"] is False
    assert abs(added[0]["location"]["frame_subpixel"] - 100) <= 1.0


def test_admission_releases_an_abstained_decoder_row_at_the_serve() -> None:
    rows, census = _admit([_row(101.0, abstain=True), _row(160.0)])
    assert census["released"] == 1 and census["added"] == 0
    released = [row for row in rows if toss.POLICY_KEY in row]
    assert released[0]["frame"] == 101.0 and released[0]["abstain"] is False


def test_admission_refuses_after_an_accepted_contact_or_at_a_bounce() -> None:
    _rows, census = _admit([_row(60.0)])
    assert census["added"] == 0
    assert census["refused"][0]["reason"] == "after_an_accepted_contact"
    _rows, census = _admit([_row(100.0, "bounce", abstain=True)])
    assert census["refused"][0]["reason"] == "departure_is_a_decoder_bounce"
    _rows, census = _admit([_row(101.0)])
    assert census["refused"][0]["reason"] == "decoder_accepted_contact_here"
