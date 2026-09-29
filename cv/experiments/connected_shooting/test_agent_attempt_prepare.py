from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import agent_attempt_prepare as prepare


def _timing_uncertain_attempt(event):
    labels = {
        "benchmark_id": "uncertain_bounce",
        "match_id": "match",
        "ball_convention": "front",
        "source_pack": {"fps": 25},
        "events": {
            "records": [
                dict(
                    status="labeled",
                    event_type="contact",
                    frame=2,
                    frame_interval=[2, 2],
                    note="contact",
                ),
                event,
                dict(
                    status="labeled",
                    event_type="contact",
                    frame=7,
                    frame_interval=[7, 7],
                    note="next contact",
                ),
                dict(
                    status="labeled",
                    event_type="ending",
                    frame=9,
                    frame_interval=[9, 9],
                    note="ending",
                ),
            ]
        },
    }
    intake = {
        "complete_ball_windows": [
            dict(
                case_id="ball",
                clip="pt",
                complete=True,
                window_status="correct",
                notes="",
                frames=[dict(frame=frame, status="visible") for frame in range(2, 10)],
            )
        ]
    }
    return prepare.attempt_record(labels, intake)[0]


def test_bounce_exists_when_only_exact_epoch_is_abstained():
    event = dict(
        event_type="bounce",
        frame=5.5,
        frame_interval=[4, 6],
        note="descent and rebound visible; player hides impact",
        status="ambiguous",
        timing_status="abstained_exact_epoch",
        reason_class="occluded_by_player",
    )
    attempt = _timing_uncertain_attempt(event)
    bounce = next(row for row in attempt["events"] if row["event_type"] == "bounce")
    assert bounce["frame"] == 5.5
    assert bounce["frame_interval"] == [4, 6]
    assert bounce["status"] == "ambiguous"
    assert bounce["timing_status"] == "abstained_exact_epoch"
    assert bounce["exact_epoch_observed"] is False
    assert attempt["agent_event_rows"][1] == event


@pytest.mark.parametrize(
    "event_type,timing_status",
    [
        ("bounce", None),
        ("bounce", "abstained"),
        ("contact", "abstained_exact_epoch"),
        ("net_hit", "abstained_exact_epoch"),
    ],
)
def test_generic_event_ambiguity_is_not_promoted(event_type, timing_status):
    event = dict(
        event_type=event_type,
        frame=5,
        frame_interval=[4, 6],
        note="existence unresolved",
        status="ambiguous",
        timing_status=timing_status,
    )
    attempt = _timing_uncertain_attempt(event)
    assert all(row["frame"] != 5 for row in attempt["events"])


@pytest.mark.parametrize("interval", [[6, 4], [5, 5], [4, float("nan")], [4], [6, 8]])
def test_uncertain_bounce_requires_valid_nonzero_interval(interval):
    event = dict(
        event_type="bounce",
        frame=5,
        frame_interval=interval,
        status="ambiguous",
        timing_status="abstained_exact_epoch",
    )
    with pytest.raises(ValueError, match="timing-uncertain bounce"):
        prepare.timing_uncertain_bounce(event)


def test_uncertain_bounce_cannot_cross_a_contact_boundary():
    event = dict(
        event_type="bounce",
        frame=6,
        frame_interval=[5, 8],
        note="uncertain flight membership",
        status="ambiguous",
        timing_status="abstained_exact_epoch",
    )
    with pytest.raises(ValueError, match="must belong to one flight"):
        _timing_uncertain_attempt(event)


def test_attempt_record_keeps_context_outside_physical_window():
    labels = {
        "benchmark_id": "case",
        "match_id": "match",
        "ball_convention": "front",
        "source_pack": {"fps": 25},
        "events": {
            "records": [
                {
                    "status": "labeled",
                    "event_type": "contact",
                    "frame": 2.5,
                    "frame_interval": [2, 3],
                    "note": "serve",
                },
                {
                    "status": "labeled",
                    "event_type": "bounce",
                    "frame": 5,
                    "frame_interval": [4.5, 5.5],
                    "note": "bounce",
                },
                {
                    "status": "labeled",
                    "event_type": "ending",
                    "frame": 5,
                    "frame_interval": [4.5, 5.5],
                    "note": "end",
                },
            ]
        },
    }
    rows = [dict(frame=frame, status="visible") for frame in range(1, 7)]
    intake = {
        "complete_ball_windows": [
            dict(
                case_id="ball",
                clip="pt",
                complete=True,
                window_status="correct",
                notes="",
                frames=rows,
            )
        ]
    }
    attempt, physical, context = prepare.attempt_record(labels, intake)
    assert physical == [3, 4, 5]
    assert context == [1, 2, 6]
    assert [row["frame"] for row in attempt["owner_ball_labels"]] == physical
    assert attempt["first_event_frame"] == 2.5


def test_attempt_record_does_not_claim_replayability_for_second_terminal_bounce():
    labels = {
        "benchmark_id": "case",
        "match_id": "match",
        "ball_convention": "front",
        "source_pack": {"fps": 25},
        "events": {
            "records": [
                {
                    "status": "labeled",
                    "event_type": "contact",
                    "frame": 2,
                    "frame_interval": [2, 2],
                    "note": "hit",
                },
                {
                    "status": "labeled",
                    "event_type": "bounce",
                    "frame": 4,
                    "frame_interval": [4, 4],
                    "note": "first",
                },
                {
                    "status": "labeled",
                    "event_type": "bounce",
                    "frame": 5,
                    "frame_interval": [5, 5],
                    "note": "second",
                },
                {
                    "status": "labeled",
                    "event_type": "ending",
                    "frame": 5,
                    "frame_interval": [5, 5],
                    "note": "end",
                },
            ]
        },
    }
    intake = {
        "complete_ball_windows": [
            dict(
                case_id="ball",
                clip="pt",
                complete=True,
                window_status="correct",
                notes="",
                frames=[dict(frame=frame, status="visible") for frame in range(2, 6)],
            )
        ]
    }
    attempt, _, _ = prepare.attempt_record(labels, intake)
    assert not attempt["structurally_ground_replayable"]


def test_clay_registration_accepts_relaxed_court_evidence_with_same_threshold(monkeypatch):
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    anchor = {
        "P": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        "parameters": [0] * 7,
        "camera_center_m": [0, 0, 1],
    }
    monkeypatch.setattr(prepare.cv2, "imread", lambda _: image)
    monkeypatch.setattr(
        prepare.court_topology, "register_static_scene", lambda *_: (np.eye(3), {"inliers": 50})
    )
    monkeypatch.setattr(prepare.court_topology, "_standard_broadcast_view", lambda *_: True)
    monkeypatch.setattr(
        prepare.court_topology,
        "court_line_masks",
        lambda _: (("standard", 0), ("relaxed_chroma", 1)),
    )
    monkeypatch.setattr(prepare.court_topology, "topology_score", lambda _, mask: (0.5, 0.9)[mask])
    monkeypatch.setattr(
        prepare.transport, "fit_view", lambda *_: {"P": anchor["P"], "parameters": [0] * 7}
    )
    result = prepare.transport_labeled_frame((2, "target", "anchor", anchor, 1))
    assert result["status"] == "supported"
    assert result["court_support_mask"] == "relaxed_chroma"
    assert result["court_support_score"] == 0.9


def test_contact_anchor_policy_avoids_midpoint_parameter_handoffs():
    anchors = [{"frame": 10}, {"frame": 20}, {"frame": 30}]
    selected = [
        prepare.transport_reference(anchors, frame, 19.5, "contact_anchor")
        for frame in (11, 19, 21, 29)
    ]
    assert [row["frame"] for row in selected] == [20, 20, 20, 20]
    assert prepare.transport_reference(anchors, 29, 19.5, "nearest_anchor")["frame"] == 30


def test_joint_label_fit_keeps_partially_occluded_landmark_frames(monkeypatch):
    def record(frame, hidden=()):
        return {
            "frames": [
                {
                    "frame": frame,
                    "target_id": landmark,
                    "status": "occluded" if landmark in hidden else "visible",
                    "x1080": 100.0 + index,
                    "y1080": 200.0 + index,
                }
                for index, landmark in enumerate(prepare.camera_cal.ITF_LANDMARKS)
            ]
        }

    records = [record(10), record(20, {"near_left_doubles"})]
    seed = np.array([[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]])
    anchors = [{"frame": 10, "fit": {"P": seed.tolist()}}]
    captured = {}
    monkeypatch.setattr(prepare.ground, "fit_ground", lambda *_: {"P": seed.tolist()})
    monkeypatch.setattr(
        prepare.camera_cal,
        "pixel_to_plane_residual_transform",
        lambda *_: np.eye(2),
    )

    def fit(observations, initial, **keywords):
        captured["observations"] = observations
        captured["initial"] = initial
        captured["keywords"] = keywords
        return SimpleNamespace(success=True)

    monkeypatch.setattr(prepare.camera_cal, "fit_joint_camera", fit)
    prepare.joint_label_fit(records, anchors, 0.05)
    assert set(captured["initial"]) == {10, 20}
    assert {row.frame for row in captured["observations"]} == {10, 20}
    assert len(captured["observations"]) == 21
    assert captured["keywords"] == {"fixed_k1": None}
    prepare.joint_label_fit(records, anchors, 0.05, shared_k1="zero")
    assert captured["keywords"] == {"fixed_k1": 0.0}


def test_the_packet_camera_default_is_the_measured_winner():
    # The prepared packet is the fitter's camera contract, so the default it ships
    # under is asserted rather than read off the help text. On the additive landmark
    # frames neither arm fitted, the anchor transport holds 0.0610 m of ground error
    # against the joint rig's 0.0729 m, so the joint arm stays opt-in.
    assert prepare.CAMERA_MODELS == ("independent_anchor_transport", "joint_rigid_court_v1")
    defaults = prepare.build_parser().parse_args(["--labels", "labels.json", "--output", "out"])
    assert defaults.camera_model == "independent_anchor_transport"
    assert defaults.transport_reference_policy == "contact_anchor"
    assert defaults.landmark_addendum == "all"
    assert defaults.shared_k1 == "solved"


def test_the_fitter_reads_the_shared_k1_the_joint_packet_ships():
    from cv.experiments.connected_shooting import agent_whole_point_search as search

    attempt = {
        "point_clip": "pt",
        "match_id": "match",
        "fps": 25.0,
        "owner_end_frame": 30.0,
        "events": [
            {"event_type": "contact", "frame": 10.0},
            {"event_type": "bounce", "frame": 20.0},
        ],
        "owner_ball_labels": [
            {"frame": frame, "status": "visible", "x1080": 900.0 + frame, "y1080": 500.0 + frame}
            for frame in range(10, 31)
        ],
    }
    projection = np.array(
        [[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 40.0]]
    )
    rows = [
        {
            "frame": frame,
            "status": "supported",
            "P": projection.tolist(),
            "k1": 5.5e-09,
            "dist_center": [960.0, 540.0],
        }
        for frame in range(10, 31)
    ]
    document = {"cameras": rows, "clip": "pt", "match_id": "match"}
    train, _heldout, _bounces, _native, _outside = search.prepare_attempt(attempt, document, "hard")
    assert train.camera_distortion is not None
    assert np.allclose(train.camera_distortion[0][:, 0], 5.5e-09)
    assert np.allclose(train.camera_distortion[0][:, 1:], [960.0, 540.0])

    pinhole = {
        "cameras": [
            {k: v for k, v in row.items() if k not in ("k1", "dist_center")} for row in rows
        ],
        "clip": "pt",
        "match_id": "match",
    }
    plain, *_ = search.prepare_attempt(attempt, pinhole, "hard")
    assert plain.camera_distortion is None


def test_a_partly_radial_camera_packet_is_rejected():
    import pytest

    from cv.experiments.connected_shooting import agent_whole_point_search as search

    attempt = {
        "point_clip": "pt",
        "match_id": "match",
        "fps": 25.0,
        "owner_end_frame": 30.0,
        "events": [{"event_type": "contact", "frame": 10.0}],
        "owner_ball_labels": [
            {"frame": frame, "status": "visible", "x1080": 900.0, "y1080": 500.0}
            for frame in range(10, 31)
        ],
    }
    projection = np.eye(3, 4).tolist()
    rows = [{"frame": frame, "status": "supported", "P": projection} for frame in range(10, 31)]
    rows[3]["k1"] = 1e-8
    document = {"cameras": rows, "clip": "pt", "match_id": "match"}
    with pytest.raises(ValueError, match="every row"):
        search.prepare_attempt(attempt, document, "hard")
