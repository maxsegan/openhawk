from __future__ import annotations

import numpy as np

from cv.viz.point_overlay_video import (
    PROVENANCE_COLORS,
    _events_for_point,
    _fitted_trajectories,
    _scoreboard_names,
    _server_side,
    accepted_flight_counts,
    draw_overlay_frame,
    frame_number,
    project,
    provenance_kind,
    select_six_points,
    select_twelve_points,
    write_s1_point_table,
)


def test_frame_number_and_provenance() -> None:
    assert frame_number("f_0123.jpg") == 123
    assert provenance_kind("crop_refine") == "crop"
    assert provenance_kind("wasb_far_native") == "far-native"
    assert provenance_kind("motion_prediction+provenance:interpolated") == "interpolated"
    assert provenance_kind("wasb_full") == "full"
    assert provenance_kind("far_native_wasb+provenance:crop") == "crop"
    assert provenance_kind("coarse_lock+provenance:coarse") == "full"


def test_projection_scales_camera_pixels_to_native() -> None:
    P = np.array([[10.0, 0.0, 0.0, 100.0], [0.0, 10.0, 0.0, 50.0], [0, 0, 0, 1]])
    assert project(P, (2.0, 3.0, 0.0), 2.0) == (240, 160)
    assert project(np.zeros((3, 4)), (1.0, 1.0, 0.0)) is None


def test_select_six_is_ranked_then_seeded() -> None:
    ledger = {
        "rows": [
            {"clip": "pt0002", "status": "valid"},
            {"clip": "pt0002", "status": "provisional_valid"},
            {"clip": "pt0004", "status": "provisional_valid"},
            {"clip": "pt0001", "status": "invalid"},
        ]
    }
    clips = [f"pt{index:04d}" for index in range(1, 9)]
    selected = select_six_points(clips, ledger, seed=17)
    assert selected[:3] == ["pt0002", "pt0004", "pt0001"]
    assert len(selected) == len(set(selected)) == 6
    assert accepted_flight_counts(ledger) == {"pt0002": 1, "pt0004": 1}


def test_select_twelve_uses_six_ranked_and_six_seeded_in_play() -> None:
    clips = [f"pt{index:04d}" for index in range(1, 17)]
    ledger = {
        "rows": [
            {"clip": f"pt{index:04d}", "status": "provisional_valid"}
            for index in range(1, 7)
            for _ in range(7 - index)
        ]
        + [{"clip": f"pt{index:04d}", "status": "recoverable"} for index in range(7, 16)]
    }
    active = {f"match/{clip}": {"point_valid": True, "active_spans": [[1, 2]]} for clip in clips}

    selected = select_twelve_points(clips, ledger, active, "match", seed=17)

    assert selected[:6] == clips[:6]
    assert len(selected) == len(set(selected)) == 12
    assert set(selected[6:]).issubset(set(clips[6:]))
    assert "pt0016" not in selected


def test_s1_table_names_server_and_score(tmp_path) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / "automatic_point_ledger_v1.csv").write_text(
        "pt,rally_t_start,rally_t_end,set_number,sets_won_1,sets_won_2,games_1,games_2,"
        "points_1,points_2,server,confidence,outcome_hint\n"
        "1,10,20,2,1,0,3,2,30,15,2,high,in_play\n"
    )
    (match_dir / "vlm_reads.jsonl").write_text(
        '{"raw":{"rows":[{"name":"SINNER"},{"name":"ALCARAZ"}]}}\n'
    )

    output = write_s1_point_table(match_dir, "rg2025f", tmp_path / "table.csv")

    rows = output.read_text().splitlines()
    assert "player_1_name" in rows[0]
    assert "Jannik Sinner,Carlos Alcaraz,2,Carlos Alcaraz,high,in_play" in rows[1]


def test_fitted_trajectories_include_rejected_fits() -> None:
    detail = {
        "fits": [
            {
                "flight_index": 0,
                "trajectory": [{"frame": 10, "xyz": [1.0, 2.0, 3.0]}],
            },
            {
                "flight_index": 1,
                "trajectory": [{"frame": 11, "xyz": [1.0, 2.0, 3.0]}],
            },
        ]
    }
    ledger = {
        "rows": [
            {
                "match_id": "match",
                "clip": "pt0001",
                "flight_index": 0,
                "status": "provisional_valid",
            },
            {
                "match_id": "match",
                "clip": "pt0001",
                "flight_index": 1,
                "status": "invalid",
            },
        ]
    }

    trajectories = _fitted_trajectories(detail, ledger, "match", "pt0001")

    assert trajectories[10][0]["flight_status"] == "provisional_valid"
    assert trajectories[11][0]["flight_status"] == "invalid"


def test_scoreboard_names_use_modal_row_order(tmp_path) -> None:
    reads = tmp_path / "vlm_reads.jsonl"
    reads.write_text(
        "\n".join(
            [
                '{"raw":{"rows":[{"name":"P. Martinez"},{"name":"A. Fils"}]}}',
                '{"raw":{"rows":[{"name":"TSITSIPAS"},{"name":"FILS"}]}}',
                '{"raw":{"rows":[{"name":"TSITSIPAS"},{"name":"FILS"}]}}',
            ]
        )
    )
    assert _scoreboard_names(reads, "unused") == ("Stefanos Tsitsipas", "Arthur Fils")


def test_server_side_falls_back_to_early_ball_near_player() -> None:
    players = {
        1: [
            {
                "side": "far",
                "x0_native": "100",
                "y0_native": "100",
                "x1_native": "140",
                "y1_native": "180",
            },
            {
                "side": "near",
                "x0_native": "700",
                "y0_native": "700",
                "x1_native": "800",
                "y1_native": "900",
            },
        ]
    }
    track = {1: [{"x_native": "120", "y_native": "150"}]}
    assert _server_side({}, [], players, track) == "far"


def test_missing_point_end_is_explicit_abstention() -> None:
    events = _events_for_point([], {}, "match", "pt0001", terminal_frame=27)
    assert events[27] == [
        {
            "clip": "match__pt0001",
            "match_id": "match",
            "frame": 27,
            "event_type": "point_end",
            "gate_held": True,
            "production_scope": False,
        }
    ]


def test_draw_overlay_keeps_native_shape_and_marks_track() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    track = {
        10: [{"x_native": "500", "y_native": "400", "sources": "crop"}],
        15: [{"x_native": "700", "y_native": "500", "sources": "full"}],
    }
    output = draw_overlay_frame(
        image,
        frame=10,
        clip="pt0001",
        track=track,
        players={},
        side_names={},
        events={
            15: [
                {
                    "event_type": "contact",
                    "gate_held": False,
                    "production_scope": True,
                    "point_grammar": {"verdict": "in_play"},
                }
            ]
        },
        arcs=[{"arc_id": 0, "start_frame": 1, "end_frame": 20, "decision": "retain"}],
        camera={},
        trajectories={},
        camera_scale=2.0,
    )
    assert output.shape == image.shape
    assert tuple(output[400, 500]) == PROVENANCE_COLORS["crop"]
    assert np.any(output[485:516, 685:716] != 0)
    assert np.any(output != image)


def test_overlay_distinguishes_model_abstention_and_event_only_impulse_support(monkeypatch):
    labels = []
    monkeypatch.setattr(
        "cv.viz.point_overlay_video._draw_text",
        lambda canvas, text, *args, **kwargs: labels.append(text),
    )
    draw_overlay_frame(
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        frame=10,
        clip="pt0001",
        track={},
        players={},
        side_names={},
        events={
            10: [
                {"event_type": "contact", "abstain": True, "gate_held": False},
                {"event_type": "bounce", "abstain": False, "impulse_support": {"supported": True}},
            ]
        },
        arcs=[],
        camera={},
        trajectories={},
        camera_scale=1.0,
    )
    assert "contact @ f0010 | ABSTAIN" in labels
    assert "bounce @ f0010 | EMIT" in labels
    assert "two-wing event support; track held" in labels
