"""A shirt that sits still is dropped. A ball that crosses the torso is kept."""

from cv.pipeline.torso_lock import persistent_torso_mask

PERSON = (100.0, 100.0, 180.0, 400.0)  # torso is the upper 60 percent: y 100-280


def test_a_shirt_chain_is_refused_and_the_other_blob_stays() -> None:
    frames = []
    points = []
    boxes = {}
    for frame in range(1, 9):
        boxes[frame] = [PERSON]
        frames.extend([frame, frame])
        points.append((140.0, 180.0))  # on the shirt
        points.append((40.0 + frame, 300.0))  # the ball, outside the box
    mask = persistent_torso_mask(frames, points, boxes, min_run=6)
    assert mask[0::2] == [True] * 8
    assert mask[1::2] == [False] * 8


def test_a_ball_crossing_the_torso_for_two_frames_is_kept() -> None:
    frames = [1, 2, 3, 4]
    points = [(140.0, 180.0), (142.0, 182.0), (200.0, 190.0), (220.0, 170.0)]
    boxes = {frame: [PERSON] for frame in frames}
    # The last two are outside the torso (x > 180).
    mask = persistent_torso_mask(frames, points, boxes, min_run=6)
    assert mask == [False, False, False, False]


def test_two_players_do_not_chain_into_one_lock() -> None:
    other = (400.0, 100.0, 480.0, 400.0)
    frames = [1, 2, 3, 4, 5, 6]
    points = [(140.0, 180.0)] * 3 + [(440.0, 180.0)] * 3
    boxes = {frame: [PERSON, other] for frame in frames}
    mask = persistent_torso_mask(frames, points, boxes, min_run=6)
    assert mask == [False] * 6


def test_streams_are_filtered_on_the_pooled_mask(tmp_path) -> None:
    import csv

    from cv.pipeline.torso_lock import filter_candidate_streams

    boxes = tmp_path / "boxes.csv"
    with boxes.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "frame", "x0_native", "y0_native", "x1_native", "y1_native"])
        writer.writerows(["c", f, *PERSON] for f in range(1, 9))
    # Two detectors each see the shirt on alternate frames: four frames apiece, one chain of eight.
    for name, parity in (("a.csv", 0), ("b.csv", 1)):
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["clip", "frame", "x_native", "y_native"])
            for frame in range(1, 9):
                if frame % 2 == parity:
                    writer.writerow(["c", frame, 140.0, 180.0])
                writer.writerow(["c", frame, 40.0 + frame, 300.0])
    out = tmp_path / "out"
    dropped = filter_candidate_streams(tmp_path, ["a.csv", "b.csv"], boxes, out, min_run=6)
    assert dropped == {"a.csv": 4, "b.csv": 4}
    for name in ("a.csv", "b.csv"):
        rows = list(csv.DictReader((out / name).open()))
        assert len(rows) == 8 and all(float(row["x_native"]) < 100 for row in rows)


FAR = (500.0, 100.0, 530.0, 180.0)  # far player: torso y 100-148, 48 px tall


def _far_contact(outside_after: bool):
    """A ball flying into a far torso, dwelling seven frames, leaving (or not)."""

    frames, points = [], []
    for frame in range(1, 4):  # arriving from the left, outside the torso
        frames.append(frame)
        points.append((470.0 + 8 * frame, 120.0))
    for frame in range(4, 11):  # inside the torso box
        frames.append(frame)
        points.append((502.0 + 3 * (frame - 4), 120.0 + (frame - 4)))
    if outside_after:
        for frame in range(11, 14):
            frames.append(frame)
            points.append((531.0 + 8 * (frame - 10), 126.0))
    boxes = {frame: [FAR] for frame in range(1, 14)}
    return frames, points, boxes


def test_keep_joined_keeps_a_far_ball_that_enters_and_leaves() -> None:
    frames, points, boxes = _far_contact(outside_after=True)
    assert any(persistent_torso_mask(frames, points, boxes, min_run=6, keep_joined=False))
    assert not any(persistent_torso_mask(frames, points, boxes, min_run=6, keep_joined=True))


def test_keep_joined_needs_both_ends() -> None:
    frames, points, boxes = _far_contact(outside_after=False)
    assert any(persistent_torso_mask(frames, points, boxes, min_run=6, keep_joined=True))


def test_keep_joined_leaves_a_near_shirt_dropped() -> None:
    frames, points, boxes = [], [], {}
    for frame in range(1, 12):
        boxes[frame] = [PERSON]  # 180 px torso
        frames.append(frame)
        # on the shirt for frames 3-9, just outside the box edge before and after
        points.append((140.0, 180.0) if 3 <= frame <= 9 else (185.0, 180.0))
    mask = persistent_torso_mask(frames, points, boxes, min_run=6, keep_joined=True)
    assert mask[2:9] == [True] * 7


def _row(frame: int, x: float, y: float) -> dict:
    return {"clip": "pt0001", "frame": f"f_{frame:04d}.jpg", "x_native": str(x), "y_native": str(y)}


def _line(frames, x0=100.0, step=10.0) -> list[dict]:
    return [_row(f, x0 + step * f, 200.0) for f in frames]


def test_gap_restore_fills_a_bracketed_gap() -> None:
    from cv.pipeline.torso_lock import restore_lock_gaps

    lock = _line([*range(0, 5), *range(9, 14)])
    free = _line(range(0, 14))
    rows, count = restore_lock_gaps(lock, free)
    assert count == 4
    assert [r["frame"] for r in rows] == [f"f_{f:04d}.jpg" for f in range(14)]


def test_gap_restore_needs_agreement_on_both_sides_and_never_overwrites() -> None:
    from cv.pipeline.torso_lock import restore_lock_gaps

    lock = _line([*range(0, 5), *range(9, 14)])
    free = _line(range(0, 14))
    free[9] = _row(9, 500.0, 500.0)  # the unfiltered track is elsewhere after the gap
    assert restore_lock_gaps(lock, free)[1] == 0
    lock_only = _line(range(0, 5))
    rows, count = restore_lock_gaps(lock_only, _line(range(0, 9), step=11.0))
    assert count == 0 and rows == lock_only


def test_gap_restore_near_player_needs_a_continuous_path() -> None:
    from cv.pipeline.torso_lock import restore_lock_gaps

    big = {"pt0001": {f: [(0.0, 0.0, 1000.0, 400.0)] for f in range(14)}}  # torso 240 px
    lock = _line([*range(0, 5), *range(9, 14)])
    free = _line(range(0, 14))
    assert restore_lock_gaps(lock, free, boxes_by_clip=big)[1] == 4
    jumped = [dict(r) for r in free]
    for f in range(5, 9):
        jumped[f] = _row(f, 100.0 + 10.0 * f, 230.0)  # hands, 30 px off the path
    assert restore_lock_gaps(lock, jumped, boxes_by_clip=big)[1] == 0
    assert restore_lock_gaps(lock, jumped)[1] == 4  # no boxes: agreement alone
