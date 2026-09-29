import pytest

from point_validity import compose_point_gate


def test_gate_prioritizes_hard_camera_then_tracking_risk_within_budget():
    active = {
        "match/pt0001": {"point_valid": True, "reasons": [], "trim": [1, 10]},
        "match/pt0002": {
            "point_valid": False,
            "reasons": ["phase:multiple_camera_shots"],
            "trim": None,
        },
        "match/pt0003": {
            "point_valid": False,
            "reasons": ["phase:activity_ambiguous"],
            "trim": [2, 9],
        },
        "match/pt0004": {"point_valid": True, "reasons": [], "trim": [1, 10]},
        "match/pt0005": {"point_valid": True, "reasons": [], "trim": [1, 10]},
    }
    tracking = [
        {
            "match_id": "match",
            "clip": f"pt{point:04d}",
            "quality_risk": risk,
            "decision": "hold" if risk >= 8 else "retain",
        }
        for point, risk in enumerate((10, 1, 9, 8, 2), start=1)
    ]

    report = compose_point_gate(
        active,
        tracking,
        maximum_invalid_fraction=0.4,
    )

    held = {row["point"] for row in report["rows"] if row["decision"] == "hold"}
    assert held == {"match/pt0001", "match/pt0002"}
    ambiguous = next(row for row in report["rows"] if row["point"] == "match/pt0003")
    assert ambiguous["decision"] == "retain"
    assert ambiguous["reasons"] == []
    assert "active_play_ambiguous" in ambiguous["diagnostic_reasons"]


def test_gate_treats_invalid_court_geometry_as_hard_camera_failure():
    active = {
        "match/pt0001": {"point_valid": True, "reasons": []},
        "match/pt0002": {"point_valid": True, "reasons": []},
    }
    tracking = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "quality_risk": -2.0,
            "decision": "retain",
        },
        {
            "match_id": "match",
            "clip": "pt0002",
            "quality_risk": 2.0,
            "decision": "hold",
        },
    ]
    geometry = [
        {"point": "match/pt0001", "valid": False, "reasons": ["offscreen"]},
        {"point": "match/pt0002", "valid": True, "reasons": []},
    ]

    report = compose_point_gate(
        active,
        tracking,
        maximum_invalid_fraction=0.5,
        court_geometry_rows=geometry,
    )
    by_point = {row["point"]: row for row in report["rows"]}

    assert by_point["match/pt0001"]["decision"] == "hold"
    assert by_point["match/pt0001"]["hard_camera_failure"]
    assert by_point["match/pt0001"]["reasons"] == ["hard_camera_failure"]


def test_gate_treats_systematic_duplicate_frames_as_hard_timing_failure():
    active = {
        "match/pt0001": {"point_valid": True, "reasons": []},
        "match/pt0002": {"point_valid": True, "reasons": []},
    }
    tracking = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "quality_risk": -10.0,
            "decision": "retain",
        },
        {
            "match_id": "match",
            "clip": "pt0002",
            "quality_risk": 10.0,
            "decision": "hold",
        },
    ]
    cadence = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "decision": "timing_hold",
            "active_duplicate_rate": 0.167,
        },
        {
            "match_id": "match",
            "clip": "pt0002",
            "decision": "timing_usable",
            "active_duplicate_rate": 0.0,
        },
    ]

    report = compose_point_gate(
        active,
        tracking,
        maximum_invalid_fraction=0.5,
        frame_cadence_rows=cadence,
    )
    by_point = {row["point"]: row for row in report["rows"]}

    assert by_point["match/pt0001"]["decision"] == "hold"
    assert by_point["match/pt0001"]["hard_timing_failure"]
    assert by_point["match/pt0001"]["reasons"] == ["systematic_repeated_visual_frames"]


def test_hard_failures_can_exceed_legacy_risk_budget():
    active = {f"match/pt{point:04d}": {"point_valid": True, "reasons": []} for point in range(1, 5)}
    tracking = [
        {
            "match_id": "match",
            "clip": f"pt{point:04d}",
            "quality_risk": 0.0,
            "decision": "retain",
        }
        for point in range(1, 5)
    ]
    cadence = [
        {
            "match_id": "match",
            "clip": f"pt{point:04d}",
            "decision": "timing_hold" if point <= 2 else "timing_usable",
            "active_duplicate_rate": 0.5 if point <= 2 else 0.0,
        }
        for point in range(1, 5)
    ]

    report = compose_point_gate(
        active,
        tracking,
        maximum_invalid_fraction=0.25,
        frame_cadence_rows=cadence,
    )

    assert report["budget"] == 1
    assert report["mandatory_holds"] == 2
    assert report["held"] == 2


def test_absolute_gate_retains_good_arcs_from_legacy_held_point():
    active = {"match/pt0001": {"point_valid": True, "reasons": []}}
    tracking = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "quality_risk": 10.0,
            "decision": "hold",
            "arcs": [
                {"arc_id": 0, "start_frame": 1, "end_frame": 12, "decision": "retain"},
                {"arc_id": 1, "start_frame": 13, "end_frame": 15, "decision": "hold"},
            ],
        }
    ]

    report = compose_point_gate(active, tracking, maximum_invalid_fraction=None)
    row = report["rows"][0]

    assert row["decision"] == "retain"
    assert row["tracking_decision"] == "hold"
    assert row["tracking_arc_decision"] == "partial"
    assert row["retained_tracking_arc_count"] == 1
    assert row["held_tracking_arc_count"] == 1
    assert row["reasons"] == []
    assert row["diagnostic_reasons"] == ["tracking_arc_abstentions"]


def test_absolute_gate_keeps_all_held_arcs_as_a_whole_point_diagnostic():
    active = {"match/pt0001": {"point_valid": True, "reasons": []}}
    tracking = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "quality_risk": -1.0,
            "decision": "retain",
            "arcs": [
                {"arc_id": 0, "start_frame": 1, "end_frame": 3, "decision": "hold"},
            ],
        }
    ]

    row = compose_point_gate(active, tracking, maximum_invalid_fraction=None)["rows"][0]

    assert row["decision"] == "retain"
    assert row["tracking_arc_decision"] == "hold"
    assert row["tracking_whole_point_decision"] == "hold"
    assert row["reasons"] == []
    assert row["diagnostic_reasons"] == ["tracking_risk"]


def test_absolute_gate_still_holds_a_camera_abstention_with_retained_arcs():
    active = {
        "match/pt0001": {
            "point_valid": False,
            "reasons": ["no_play_camera_shot"],
        }
    }
    tracking = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "quality_risk": 0.0,
            "decision": "retain",
            "arcs": [
                {"arc_id": 0, "start_frame": 1, "end_frame": 10, "decision": "retain"},
            ],
        }
    ]

    row = compose_point_gate(active, tracking, maximum_invalid_fraction=None)["rows"][0]

    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]


# --- retained play scope (default off) ----------------------------------------------

SHOT_CUT_ACTIVE = {
    "point_valid": False,
    "reasons": ["phase:multiple_camera_shots"],
    "trim": [100.0, 900.0],
    "n_frames": 1000,
    "active_spans": [[100.0, 400.0], [600.0, 900.0]],
}
TRACKING = [
    {
        "match_id": "match",
        "clip": "pt0001",
        "quality_risk": 0.0,
        "decision": "retain",
        "arcs": [{"arc_id": 0, "start_frame": 1, "end_frame": 1000, "decision": "retain"}],
    }
]
GEOMETRY_VALID = [{"point": "match/pt0001", "valid": True, "reasons": []}]
# The reliable per-frame camera transport supports the whole clip unless a case says
# otherwise, so a test about play spans is not silently also a test about support.
CAMERA_SUPPORT = [
    {
        "point": "match/pt0001",
        "match_id": "match",
        "clip": "pt0001",
        "artifact_present": True,
        "supported_frames": 1000,
        "supported_spans": [[1, 1000]],
    }
]


def _support(spans, **extra):
    return [dict(CAMERA_SUPPORT[0], supported_spans=spans, **extra)]


def _scoped_row(active, geometry=GEOMETRY_VALID, camera_support_rows=CAMERA_SUPPORT, **kwargs):
    return compose_point_gate(
        {"match/pt0001": active},
        TRACKING,
        maximum_invalid_fraction=None,
        court_geometry_rows=geometry,
        retained_play_scope_enabled=True,
        camera_support_rows=camera_support_rows,
        **kwargs,
    )["rows"][0]


def test_shot_composition_hold_is_unchanged_while_the_scope_is_off():
    report = compose_point_gate(
        {"match/pt0001": SHOT_CUT_ACTIVE},
        TRACKING,
        maximum_invalid_fraction=None,
        court_geometry_rows=GEOMETRY_VALID,
    )

    row = report["rows"][0]
    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]
    assert "play_scope" not in row
    assert "play_scope_native_interval" not in row
    assert "retained_play_scope" not in report


def test_scope_retains_a_shot_composition_point_and_publishes_its_play_spans():
    row = _scoped_row(SHOT_CUT_ACTIVE)

    assert row["decision"] == "retain"
    assert row["reasons"] == []
    assert row["hard_camera_failure"] is False
    assert row["play_scope_native_interval"] == [100.0, 900.0]
    # The hole the broadcast cut left between the two rallies is preserved.
    assert row["play_scope_native_spans"] == [[100.0, 400.0], [600.0, 900.0]]
    assert row["play_scope"]["shot_composition_reasons"] == ["phase:multiple_camera_shots"]
    assert "multiple_camera_shots_scoped_to_play_interval" in row["diagnostic_reasons"]


def test_scope_report_declares_itself_and_counts_the_scoped_points():
    report = compose_point_gate(
        {"match/pt0001": SHOT_CUT_ACTIVE},
        TRACKING,
        maximum_invalid_fraction=None,
        court_geometry_rows=GEOMETRY_VALID,
        retained_play_scope_enabled=True,
        camera_support_rows=CAMERA_SUPPORT,
    )

    assert report["retained_play_scope"] is True
    assert report["play_interval_scoped_points"] == 1
    assert report["play_scope_camera_support"] == {
        "transport": {"mode": "reliable_per_frame", "missing": "hold"},
        "play_span_frames": 602,
        "supported_play_span_frames": 602,
    }


def test_scope_never_softens_an_invalid_court_geometry():
    geometry = [{"point": "match/pt0001", "valid": False, "reasons": ["offscreen"]}]

    row = _scoped_row(SHOT_CUT_ACTIVE, geometry=geometry)

    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]
    assert "play_scope" not in row


def test_scope_requires_an_explicitly_valid_geometry_verdict():
    row = _scoped_row(SHOT_CUT_ACTIVE, geometry=None)

    assert row["decision"] == "hold"
    assert "play_scope" not in row


def test_scope_leaves_a_camera_failure_sharing_the_point_held():
    active = dict(SHOT_CUT_ACTIVE, reasons=["phase:multiple_camera_shots", "no_play_camera_shot"])

    row = _scoped_row(active)

    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]
    assert "play_scope" not in row


def test_scope_holds_a_cadence_failure_that_shares_the_point():
    cadence = [
        {"match_id": "match", "clip": "pt0001", "decision": "timing_hold", "active_duplicate_rate": 0.4}
    ]

    row = _scoped_row(SHOT_CUT_ACTIVE, frame_cadence_rows=cadence)

    assert row["decision"] == "hold"
    assert row["reasons"] == ["systematic_repeated_visual_frames"]
    # The camera verdict is separable; the cadence verdict is not touched by it.
    assert row["hard_camera_failure"] is False
    assert row["play_scope"] is not None


def test_scope_preserves_a_held_tracking_arc_diagnostic():
    tracking = [
        {
            "match_id": "match",
            "clip": "pt0001",
            "quality_risk": 0.0,
            "decision": "retain",
            "arcs": [
                {"arc_id": 0, "start_frame": 1, "end_frame": 500, "decision": "retain"},
                {"arc_id": 1, "start_frame": 501, "end_frame": 1000, "decision": "hold"},
            ],
        }
    ]

    report = compose_point_gate(
        {"match/pt0001": SHOT_CUT_ACTIVE},
        tracking,
        maximum_invalid_fraction=None,
        court_geometry_rows=GEOMETRY_VALID,
        retained_play_scope_enabled=True,
        camera_support_rows=CAMERA_SUPPORT,
    )

    row = report["rows"][0]
    assert row["decision"] == "retain"
    assert row["tracking_arc_decision"] == "partial"
    assert "tracking_arc_abstentions" in row["diagnostic_reasons"]


@pytest.mark.parametrize(
    "override",
    [
        {"trim": None},
        {"trim": []},
        {"trim": [100.0]},
        {"trim": [900.0, 100.0]},
        {"trim": [100.0, 100.0]},
        {"trim": [float("nan"), 900.0]},
        {"trim": ["a", "b"]},
        {"n_frames": 0},
        {"n_frames": None},
        {"n_frames": 800},  # the trim leaves the clip's own native window
        {"trim": [-5.0, 900.0]},
        {"active_spans": []},
        {"active_spans": None},
        {"active_spans": "1-10"},
        {"active_spans": [[100.0, 400.0], [350.0, 900.0]]},  # overlapping
        {"active_spans": [[600.0, 900.0], [100.0, 400.0]]},  # out of order
        {"active_spans": [[50.0, 400.0]]},  # not contained in the trim
        {"active_spans": [[100.0, 950.0]]},  # not contained in the trim
        {"active_spans": [[100.0, 100.0]]},  # empty span
        {"active_spans": [[100.0, float("inf")]]},
        {"active_spans": [[100.0, 400.0], None]},
    ],
)
def test_scope_refuses_malformed_or_missing_play_evidence(override):
    active = dict(SHOT_CUT_ACTIVE)
    active.update(override)
    for key, value in override.items():
        if value is None and key in ("n_frames",):
            active.pop(key)

    row = _scoped_row(active)

    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]
    assert "play_scope" not in row


def test_scope_does_not_touch_a_point_with_no_shot_composition_reason():
    active = {"point_valid": True, "reasons": [], "trim": [1.0, 10.0], "n_frames": 20}

    row = _scoped_row(active)

    assert row["decision"] == "retain"
    assert "play_scope" not in row


# --- per-frame reliable-camera support ----------------------------------------------
#
# An active span is a shot-composition verdict, not a per-frame view verdict: the
# source13 pt0004 broadcast marks 129..969 `is_play_camera` true while a close-up runs
# inside it.  These cases pin the separate condition that catches that.


def test_a_play_span_covering_an_unsupported_close_up_admits_only_the_supported_frames():
    # One active span over the whole trim, with the automatic per-frame transport
    # registering the court only on its second half: the span-only verdict would have
    # certified the close-up too.
    active = dict(SHOT_CUT_ACTIVE, active_spans=[[100.0, 900.0]])

    row = _scoped_row(active, camera_support_rows=_support([[500, 900]]))

    assert row["decision"] == "retain"
    # The play span is published verbatim; the support is published beside it, so a
    # consumer can refuse the individual unsupported rows instead of the whole point.
    assert row["play_scope_native_spans"] == [[100.0, 900.0]]
    assert row["play_scope_camera_supported_spans"] == [[500.0, 900.0]]
    support = row["play_scope"]["camera_support"]
    assert support["play_span_frames"] == 801
    assert support["supported_play_span_frames"] == 401
    assert support["admitted_spans"] == [[500.0, 900.0]]
    assert support["transport"] == {"mode": "reliable_per_frame", "missing": "hold"}


def test_scope_refuses_a_play_interval_with_no_camera_support_anywhere_in_it():
    # Every frame the producer called live play is a close-up.  There is nothing to
    # admit, so the original hold stands rather than a scope that accepts nothing.
    row = _scoped_row(SHOT_CUT_ACTIVE, camera_support_rows=_support([[410, 590]]))

    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]
    assert "play_scope" not in row


def test_scope_keeps_a_single_supported_native_frame():
    # A support run is a closed set of native frame integers and may be one frame wide.
    row = _scoped_row(SHOT_CUT_ACTIVE, camera_support_rows=_support([[400, 400]]))

    assert row["decision"] == "retain"
    assert row["play_scope"]["camera_support"]["supported_play_span_frames"] == 1


@pytest.mark.parametrize(
    "supported_spans",
    [
        [],
        None,
        "1-1000",
        [[1, 1001]],  # outside the clip's own native window
        [[-1, 500]],
        [[100, 400], [300, 900]],  # overlapping
        [[600, 900], [100, 400]],  # out of order
        [[float("nan"), 900]],
        [[100, float("inf")]],
        [[400, 100]],  # reversed
        [[100, 400], None],
        [[100]],
    ],
)
def test_scope_refuses_malformed_or_missing_camera_support(supported_spans):
    row = _scoped_row(SHOT_CUT_ACTIVE, camera_support_rows=_support(supported_spans))

    assert row["decision"] == "hold"
    assert row["reasons"] == ["hard_camera_failure"]
    assert "play_scope" not in row


def test_scope_holds_a_point_absent_from_the_camera_support_rows():
    row = _scoped_row(SHOT_CUT_ACTIVE, camera_support_rows=[])

    assert row["decision"] == "hold"
    assert "play_scope" not in row


def test_scope_refuses_to_compose_without_camera_support_rows_at_all():
    # A caller that cannot supply per-frame support does not silently fall back to
    # the span-only verdict this followup replaced.
    with pytest.raises(ValueError, match="reliable per-frame camera support"):
        compose_point_gate(
            {"match/pt0001": SHOT_CUT_ACTIVE},
            TRACKING,
            maximum_invalid_fraction=None,
            court_geometry_rows=GEOMETRY_VALID,
            retained_play_scope_enabled=True,
        )


def _compose_cli(tmp_path, monkeypatch, *extra):
    import json

    from cv.pipeline import compose_point_validity_gate

    (tmp_path / "active.json").write_text(json.dumps({"match/pt0001": SHOT_CUT_ACTIVE}))
    (tmp_path / "tracking.json").write_text(json.dumps({"rows": TRACKING}))
    (tmp_path / "geometry.json").write_text(json.dumps({"rows": GEOMETRY_VALID}))
    (tmp_path / "support.json").write_text(
        json.dumps(
            {"schema": "reliable_per_frame_camera_support_v1", "rows": _support([[1, 700]])}
        )
    )
    output = tmp_path / f"gate{len(extra)}.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compose_point_validity_gate",
            "--active-play",
            str(tmp_path / "active.json"),
            "--tracking-gate",
            str(tmp_path / "tracking.json"),
            "--court-geometry",
            str(tmp_path / "geometry.json"),
            "--maximum-invalid-fraction",
            "absolute",
            "--output",
            str(output),
            *extra,
        ],
    )
    compose_point_validity_gate.main()
    return json.loads(output.read_text())


def test_producer_cli_publishes_the_play_interval_only_when_asked(tmp_path, monkeypatch):
    default = _compose_cli(tmp_path, monkeypatch)
    assert default["rows"][0]["decision"] == "hold"
    assert "retained_play_scope" not in default

    selected = _compose_cli(
        tmp_path,
        monkeypatch,
        "--retained-play-scope",
        "--camera-frame-support",
        str(tmp_path / "support.json"),
    )
    assert selected["retained_play_scope"] is True
    assert selected["rows"][0]["decision"] == "retain"
    assert selected["rows"][0]["play_scope_native_spans"] == [[100.0, 400.0], [600.0, 900.0]]
    assert selected["rows"][0]["play_scope_camera_supported_spans"] == [[1.0, 700.0]]


def test_producer_cli_requires_the_camera_support_document_with_the_scope(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _compose_cli(tmp_path, monkeypatch, "--retained-play-scope")


def test_producer_cli_refuses_an_unknown_camera_support_schema(tmp_path, monkeypatch):
    import json

    (tmp_path / "other.json").write_text(json.dumps({"schema": "other_v1", "rows": []}))
    with pytest.raises(ValueError, match="unsupported camera support schema"):
        _compose_cli(
            tmp_path,
            monkeypatch,
            "--retained-play-scope",
            "--camera-frame-support",
            str(tmp_path / "other.json"),
        )
