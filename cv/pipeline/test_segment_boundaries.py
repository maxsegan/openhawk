import numpy as np
import pytest

from cv.pipeline import segment_boundaries as boundaries


def _motion(times, near_y, far_y, near_x=None, far_x=None, fps=25.0):
    times = np.asarray(times, dtype=float)
    out = {}
    for side, y, x in (("near", near_y, near_x), ("far", far_y, far_x)):
        if y is None:
            continue
        court = np.column_stack(
            [np.zeros(len(times)) if x is None else np.asarray(x, float), np.asarray(y, float)]
        )
        step = np.maximum(np.gradient(times), 1e-3)
        speed = np.hypot(np.gradient(court[:, 0]), np.gradient(court[:, 1])) / step
        window = max(1, int(round(boundaries.SPEED_SMOOTHING_SECONDS * fps)))
        out[side] = (times, court, np.convolve(speed, np.ones(window) / window, mode="same"))
    return out


def test_ending_follows_rally_motion_and_snaps_to_an_impact() -> None:
    fps = 25.0
    times = np.arange(100.0, 112.0, 1 / fps)
    # the near player runs until t=106 and then walks
    near = np.where(times < 106.0, 4.0 * (times - 100.0), 24.0 + 0.2 * (times - 106.0))
    far = np.zeros(len(times)) + 20.0
    peaks = np.array([[101.0, 20.0], [106.2, 14.0], [108.5, 40.0]])

    witness = boundaries.physical_ending_witness(
        _motion(times, near, far),
        peaks,
        first_impact_seconds=100.5,
        coarse_end_seconds=111.5,
        fps=fps,
    )

    assert not witness["abstained"]
    assert witness["seconds"] == pytest.approx(106.2)
    assert witness["snapped_to_impact"]
    assert witness["witness"] == "rally_player_motion"


def test_short_attempt_without_running_uses_the_impact_after_contact() -> None:
    fps = 25.0
    times = np.arange(50.0, 56.0, 1 / fps)
    still = np.full(len(times), -2.0)
    peaks = np.array([[50.4, 30.0], [50.9, 12.0], [54.0, 60.0]])

    witness = boundaries.physical_ending_witness(
        _motion(times, still, still + 26.0),
        peaks,
        first_impact_seconds=50.0,
        coarse_end_seconds=55.5,
        fps=fps,
    )

    assert witness["witness"] == "short_attempt_impact_train"
    assert witness["seconds"] == pytest.approx(0.9 + 50.0)


def test_ending_abstains_without_player_evidence_instead_of_guessing() -> None:
    witness = boundaries.physical_ending_witness(
        {}, np.array([[10.0, 5.0]]), first_impact_seconds=8.0, coarse_end_seconds=20.0, fps=25.0
    )

    assert witness["abstained"]
    assert witness["abstention_reason"] == "no_player_motion_evidence_in_segment"
    assert witness["frame"] is None


def test_toss_release_is_one_service_motion_before_the_automatic_contact() -> None:
    witness = boundaries.toss_release_witness(first_impact_seconds=42.0, fps=25.0)

    assert witness["seconds"] == pytest.approx(42.0 - boundaries.SERVE_MOTION_SECONDS)
    assert boundaries.toss_release_witness(first_impact_seconds=None, fps=25.0)["abstained"]


def test_a_confirmed_second_serve_stance_splits_one_coarse_segment() -> None:
    fps = 25.0
    times = np.arange(0.0, 24.0, 1 / fps)
    # near player serves, rallies, returns behind the baseline and serves again
    near = np.piecewise(
        times,
        [
            times < 2.0,
            (times >= 2.0) & (times < 9.0),
            (times >= 9.0) & (times < 15.0),
            times >= 15.0,
        ],
        [
            -2.0,
            lambda t: 6.0 * (t - 2.0),
            -2.0,
            lambda t: -2.0 + 6.0 * (t - 15.0),
        ],
    )
    far = np.full(len(times), 25.0)
    peaks = np.array([[1.0, 40.0], [8.5, 20.0], [15.3, 35.0], [19.0, 25.0]])
    rows = [
        {
            "pt": "1",
            "point_index": "1",
            "rally_t_start": "0.5",
            "rally_t_end": "20.0",
            "first_impact_t": "1.0",
        }
    ]

    refined, report = boundaries.refine_rows(rows, _motion(times, near, far), peaks, fps=fps)

    assert report["counts"]["second_serve_splits"] == 1
    assert len(refined) == 2
    assert refined[1]["rally_t_start"] >= refined[0]["rally_t_end"] - 1e-6
    assert [row["pt"] for row in refined] == [1, 2]


def test_a_coarse_segment_is_never_dropped_when_every_witness_abstains() -> None:
    rows = [
        {"pt": "1", "point_index": "1", "rally_t_start": "10", "rally_t_end": "20"},
        {"pt": "2", "point_index": "2", "rally_t_start": "40", "rally_t_end": "48"},
    ]

    refined, report = boundaries.refine_rows(rows, {}, np.empty((0, 2)), fps=25.0)

    assert [(row["rally_t_start"], row["rally_t_end"]) for row in refined] == [
        (10.0, 20.0),
        (40.0, 48.0),
    ]
    assert report["counts"]["ending_abstentions"] == 2
    assert report["counts"]["toss_abstentions"] == 2


def test_unobserved_gap_never_rejoins_two_real_attempts() -> None:
    """Absence of player tracks is not evidence that no second serve happened."""
    rows = [
        {
            "pt": "1",
            "point_index": "5",
            "rally_t_start": "10",
            "rally_t_end": "14",
            "first_impact_t": "11",
        },
        {
            "pt": "2",
            "point_index": "5",
            "rally_t_start": "15",
            "rally_t_end": "22",
            "first_impact_t": "16",
        },
    ]

    refined, report = boundaries.refine_rows(rows, {}, np.empty((0, 2)), fps=25.0)

    assert len(refined) == 2
    assert "false_splits_rejoined" not in report["counts"]


def test_a_pulled_back_toss_never_crosses_the_previous_attempt_ending() -> None:
    rows = [
        {
            "pt": "1",
            "point_index": "1",
            "rally_t_start": "10",
            "rally_t_end": "20",
            "first_impact_t": "11",
        },
        {
            "pt": "2",
            "point_index": "2",
            "rally_t_start": "20.4",
            "rally_t_end": "26",
            "first_impact_t": "20.5",
        },
    ]

    refined, report = boundaries.refine_rows(rows, {}, np.empty((0, 2)), fps=25.0)

    assert len(refined) == 2
    assert refined[1]["rally_t_start"] >= refined[0]["rally_t_end"]
    assert report["counts"]["toss_clamped_to_previous_end"] == 1


def test_native_motion_clock_joins_exact_clip_and_frame_without_changing_rows(tmp_path):
    import csv

    path = tmp_path / "players.csv"
    clock = {f: 724.6 + (f - 1) / 25 for f in range(1, 11)}
    rows = [
        dict(clip="wanted", frame=f"f_{f:04d}.jpg", side="near", court_x=f / 10, court_y=-2, t="")
        for f in clock
    ]
    # Same local frame in another clip is neither an observation nor a clock error.
    rows += [dict(rows[0], clip="other", t="9999", court_x=999)]
    rows += [dict(rows[0], frame="f_0999.jpg", court_x=999)]

    def write():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    write()
    original = path.read_bytes()
    joined = boundaries.load_player_motion(path, fps=25, clip="wanted", native_pts_by_frame=clock)
    assert path.read_bytes() == original
    np.testing.assert_array_equal(joined["near"][0], np.round(list(clock.values()), 3))
    np.testing.assert_array_equal(joined["near"][1][:, 0], np.arange(1, 11) / 10)
    for row, time in zip(rows[:10], clock.values()):
        row["t"] = time
    write()
    present = boundaries.load_player_motion(path, fps=25, clip="wanted", native_pts_by_frame=clock)
    for a, b in zip(joined["near"], present["near"]):
        np.testing.assert_array_equal(a, b)
    rows[2]["t"] = clock[3] + 1 / 25
    write()
    with pytest.raises(ValueError, match="timestamp differs"):
        boundaries.load_player_motion(path, fps=25, clip="wanted", native_pts_by_frame=clock)
    with pytest.raises(ValueError, match="supplied together"):
        boundaries.load_player_motion(path, fps=25, native_pts_by_frame=clock)
    with pytest.raises(ValueError, match="strictly increasing"):
        boundaries.load_player_motion(path, fps=25, clip="wanted", native_pts_by_frame={1: 4, 2: 3})
