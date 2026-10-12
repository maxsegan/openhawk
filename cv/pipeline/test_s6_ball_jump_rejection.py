import pytest

from cv.pipeline.s6_ball_jump_rejection import JumpPolicy, island_flags, teleports


def flight(start, count, origin, velocity):
    return [
        (start + i, origin[0] + velocity[0] * i, origin[1] + velocity[1] * i) for i in range(count)
    ]


def test_static_island_between_agreeing_flight_rows_is_rejected():
    # source01_case001_short_a1 f229-232: the track sits on a shoe ~80 px off for four
    # frames and then returns to the same outgoing flight.
    rows = flight(219, 10, (893.0, 73.0), (8.0, 12.0))
    shoe = [(229, 903.0, 223.0), (230, 904.0, 226.0), (231, 905.0, 229.0), (232, 906.0, 232.0)]
    rows += shoe + flight(233, 8, (893.0 + 8.0 * 14, 73.0 + 12.0 * 14), (8.0, 12.0))
    assert sorted(island_flags(rows, 59.94)) == [229, 230, 231, 232]


def test_soft_return_is_still_found_from_the_hard_entry():
    # source01_case003_medium_a2 f309-312: 95 px entry, only a 34 px step back.
    rows = flight(300, 9, (921.0, 60.0), (0.0, 14.0))
    rows += [(309, 899.0, 223.0), (310, 896.0, 222.0), (311, 893.0, 220.0), (312, 891.0, 218.0)]
    rows += flight(313, 6, (921.0, 60.0 + 14.0 * 13), (0.0, 14.0))
    assert sorted(island_flags(rows, 59.94)) == [309, 310, 311, 312]


def test_a_racket_contact_is_not_an_island():
    # The velocity changes once and the new speed persists: no teleport at all.
    rows = flight(100, 8, (900.0, 300.0), (1.0, -3.0)) + flight(
        108, 10, (908.0, 276.0), (25.0, 30.0)
    )
    assert teleports(rows, 59.94) == []
    assert island_flags(rows, 59.94) == {}


def test_a_jump_with_no_return_is_kept():
    # The tracker switched object and stayed there; the track alone cannot say which is the ball.
    rows = flight(100, 10, (900.0, 300.0), (6.0, 8.0)) + flight(110, 10, (700.0, 600.0), (6.0, 8.0))
    assert island_flags(rows, 59.94) == {}


def test_an_island_before_a_detector_gap_needs_the_far_side_to_continue_the_near_side():
    island = [(110, 700.0, 600.0), (111, 701.0, 601.0), (112, 702.0, 602.0)]
    near = flight(100, 10, (900.0, 300.0), (6.0, 8.0))
    continues = flight(130, 10, (900.0 + 6.0 * 30, 300.0 + 8.0 * 30), (6.0, 8.0))
    assert sorted(island_flags(near + island + continues, 59.94)) == [110, 111, 112]
    elsewhere = flight(130, 10, (400.0, 200.0), (6.0, 8.0))
    assert island_flags(near + island + elsewhere, 59.94) == {}
    too_late = flight(160, 10, (900.0 + 6.0 * 60, 300.0 + 8.0 * 60), (6.0, 8.0))
    assert island_flags(near + island + too_late, 59.94) == {}


def test_an_island_at_a_contact_is_kept_because_the_sides_disagree():
    rows = flight(100, 8, (900.0, 300.0), (2.0, 10.0))
    rows += [(108, 980.0, 420.0), (109, 981.0, 421.0)]
    rows += flight(110, 8, (918.0, 392.0), (-20.0, -15.0))
    assert island_flags(rows, 59.94) == {}


def test_island_length_is_bounded_in_seconds_not_frames():
    rows = flight(100, 10, (900.0, 300.0), (6.0, 8.0))
    rows += [(110 + i, 700.0 + i, 600.0) for i in range(20)]
    rows += flight(130, 10, (900.0 + 6.0 * 30, 300.0 + 8.0 * 30), (6.0, 8.0))
    assert island_flags(rows, 59.94) == {}
    assert len(island_flags(rows, 59.94, JumpPolicy(maximum_island_seconds=0.5))) == 20


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        island_flags([(1, 0.0, 0.0), (1, 1.0, 1.0)], 60.0)
    with pytest.raises(ValueError):
        island_flags([(1, 0.0, 0.0)], 0.0)
    with pytest.raises(ValueError):
        JumpPolicy(bridge_fraction=1.0).validate()


def _track_rows():
    from cv.pipeline.s6_ball_jump_rejection import apply  # noqa: F401  (import check)

    rows = flight(219, 10, (893.0, 73.0), (8.0, 12.0))
    rows += [(229, 903.0, 223.0), (230, 904.0, 226.0), (232, 906.0, 232.0)]
    rows += flight(233, 8, (893.0 + 8.0 * 14, 73.0 + 12.0 * 14), (8.0, 12.0))
    documents = [{"frame": f, "status": "visible", "x1080": x, "y1080": y} for f, x, y in rows]
    documents.append({"frame": 231, "status": "derived_estimate", "x1080": 905.0, "y1080": 229.0})
    return sorted(documents, key=lambda r: r["frame"])


def test_apply_off_returns_the_same_rows_and_no_receipt():
    from cv.pipeline.s6_ball_jump_rejection import apply

    rows = _track_rows()
    result, receipt = apply(rows, 59.94, "off")
    assert result is rows and receipt is None


def test_apply_on_refuses_island_rows_and_the_estimate_between_them_without_moving_anything():
    from cv.pipeline.s6_ball_jump_rejection import apply

    rows = _track_rows()
    result, receipt = apply(rows, 59.94, "on")
    refused = {r["frame"]: r for r in result if r["status"] == "unsupported"}
    assert sorted(refused) == [229, 230, 231, 232]
    assert refused[231]["original_tracker_status"] == "derived_estimate"
    assert all(
        (new["x1080"], new["y1080"]) == (old["x1080"], old["y1080"])
        for new, old in zip(result, rows, strict=True)
    )
    assert receipt["refused_rows"] == 4 and receipt["labels_read"] is False
    with pytest.raises(ValueError):
        apply(rows, None, "on")
    with pytest.raises(ValueError):
        apply(rows, 59.94, "sometimes")


def test_native_ball_support_is_byte_identical_with_the_key_off():
    import json

    from cv.pipeline import resolution
    from cv.pipeline.s6_automatic_observations import native_ball_support

    rows = _track_rows()
    before = native_ball_support(rows, resolution.NATIVE_SIZE)
    off = native_ball_support(rows, resolution.NATIVE_SIZE, jump_rejection="off", fps=59.94)
    assert json.dumps(before, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert "jump_rejection" not in before[1]
    on_rows, on_receipt = native_ball_support(
        rows, resolution.NATIVE_SIZE, jump_rejection="on", fps=59.94
    )
    assert on_receipt["jump_rejection"]["refused_rows"] == 4
    assert sum(r["status"] == "unsupported" for r in on_rows) == 4


def test_shared_settings_keeps_the_key_absent_unless_declared():
    from cv.pipeline import s6_labeled_stage as stage

    base = stage.shared_settings({})
    assert "automatic_ball_jump_rejection" not in base
    declared = stage.shared_settings({"automatic_ball_jump_rejection": "on"})
    assert declared["automatic_ball_jump_rejection"] == "on"
    assert {k: v for k, v in declared.items() if k != "automatic_ball_jump_rejection"} == base
    with pytest.raises(ValueError):
        stage.shared_settings({"automatic_ball_jump_rejection": "maybe"})


def test_gap_step_extension_finds_an_island_entered_and_left_through_detector_gaps():
    # source01_case004_long_a1 f1122-1128: a re-lock 840 px away between two gaps while a
    # slow lob continues either side. Off by default; its own declared mode.
    rows = flight(1100, 14, (968.0, 68.0), (-0.2, -2.8))
    rows += [(1122 + i, 875.0 - 3.0 * i, 853.0 + 1.5 * i) for i in range(7)]
    rows += flight(1143, 12, (938.0, 73.0), (-1.0, 5.0))
    assert island_flags(rows, 59.94) == {}
    flagged = island_flags(rows, 59.94, JumpPolicy(gap_step_seconds=0.25))
    assert sorted(flagged) == list(range(1122, 1129))


def test_gap_step_extension_keeps_a_ball_that_was_hit_inside_the_gap():
    rows = flight(100, 10, (900.0, 300.0), (2.0, 3.0))
    rows += flight(120, 12, (1100.0, 500.0), (20.0, 18.0))
    assert island_flags(rows, 59.94, JumpPolicy(gap_step_seconds=0.25)) == {}


def test_gap_steps_is_its_own_mode():
    from cv.pipeline import s6_labeled_stage as stage
    from cv.pipeline.s6_ball_jump_rejection import apply

    assert (
        stage.shared_settings({"automatic_ball_jump_rejection": "gap_steps"})[
            "automatic_ball_jump_rejection"
        ]
        == "gap_steps"
    )
    _, receipt = apply(_track_rows(), 59.94, "gap_steps")
    assert receipt["policy"]["gap_step_seconds"] == 0.25
    _, receipt = apply(_track_rows(), 59.94, "on")
    assert receipt["policy"]["gap_step_seconds"] == 0.0
