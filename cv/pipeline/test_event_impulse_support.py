from copy import deepcopy

import numpy as np
import pytest

from cv.pipeline.event_impulse_support import TrackSnapshot, certificate, recover_events
from cv.pipeline.resolution import (
    FrameSize,
    coordinate_manifest_path,
    write_native_dual_coordinate_manifest,
)


def inputs(knot=10.25):
    def state(t):
        dt = t - knot
        return np.array([100 + (2 if dt <= 0 else 10) * dt, 100 + (4 if dt <= 0 else -6) * dt])

    track = {t: state(t) for t in range(1, 21)}
    event = {
        "event_type": "contact",
        "frame": 10.0,
        "model_abstain": False,
        "gate_held": True,
        "point_gate_verdict": "retain",
        "point_gate_failure_reasons": ["tracking_arc_abstained"],
        "probability": 0.99,
        "path_marginal": 0.99,
        "decision_threshold": 0.95,
        "location": {
            "fps": 25.0,
            "image_coordinate_space": "native_1920x1080",
            "image_x": state(10)[0],
            "image_y": state(10)[1],
        },
    }
    arcs = [
        {
            "arc_id": 0,
            "start_frame": 1,
            "end_frame": 9,
            "regime": "ballistic",
            "decision": "retain",
        },
        {
            "arc_id": 1,
            "start_frame": 10,
            "end_frame": 10,
            "regime": "impulse",
            "decision": "hold",
            "failure_reasons": ["too_few_arc_observations", "excessive_arc_innovation_covariance"],
            "candidate_support_rate": 1.0,
            "coverage_rate": 1.0,
        },
        {
            "arc_id": 2,
            "start_frame": 11,
            "end_frame": 20,
            "regime": "ballistic",
            "decision": "retain",
        },
    ]
    return event, arcs, track


@pytest.mark.parametrize("knot", [9.75, 10.25])
def test_two_wings_meet_without_retiming_or_changing_inputs(knot):
    event, arcs, track = inputs(knot)
    before_event, before_arcs = deepcopy(event), deepcopy(arcs)
    result = certificate(event, arcs, track)
    assert result["supported"]
    assert result["join_offset_frames"] == pytest.approx(knot - 10, abs=1e-6)
    assert result["event_frame_unchanged"] == 10
    assert result["minimum_join_native_px"] < 1e-6
    assert result["event_pixel_error_native_px"] < 1e-6
    assert event == before_event and arcs == before_arcs


@pytest.mark.parametrize(
    "change",
    [
        {"model_abstain": True},
        {"point_gate_verdict": "hold"},
        {"event_type": "net_hit"},
        {"point_gate_failure_reasons": ["tracking_arc_abstained", "hard_camera_failure"]},
        {"probability": -0.01},
        {"probability": 1.01},
        {"probability": float("nan")},
    ],
)
def test_never_overrides_other_abstentions_or_weak_event_evidence(change):
    event, arcs, track = inputs()
    assert not certificate({**event, **change}, arcs, track)["supported"]


def test_refuses_missing_wing_pixels_and_bad_arc_support():
    event, arcs, track = inputs()
    del track[7]
    assert certificate(event, arcs, track)["reason"] == "insufficient_observed_wing"
    event, arcs, track = inputs()
    arcs[0]["decision"] = "hold"
    assert not certificate(event, arcs, track)["supported"]
    event, arcs, track = inputs()
    arcs[1]["failure_reasons"].append("insufficient_arc_candidate_support")
    assert not certificate(event, arcs, track)["supported"]
    event, arcs, track = inputs()
    arcs.append(deepcopy(arcs[1]))
    assert certificate(event, arcs, track)["reason"] == "missing_or_overlapping_arc"


def test_refuses_position_jump_wrong_event_pixel_and_noisy_wing():
    event, arcs, track = inputs()
    for t in range(11, 21):
        track[t] += [100, 100]
    assert not certificate(event, arcs, track)["supported"]
    event, arcs, track = inputs()
    event["location"]["image_x"] += 30
    assert not certificate(event, arcs, track)["supported"]
    event, arcs, track = inputs()
    track[7] += [0, 100]
    assert certificate(event, arcs, track)["reason"] == "wing_not_locally_consistent"


def test_constant_coincident_wings_do_not_crash_polynomial_minimization():
    event, arcs, track = inputs()
    track = {t: np.array([100.0, 100.0]) for t in track}
    event["location"].update(image_x=100.0, image_y=100.0)
    result = certificate(event, arcs, track)
    # Positional agreement without a velocity impulse is insufficient.
    assert np.isfinite(result["minimum_join_native_px"])
    assert not result["supported"]


@pytest.mark.parametrize("field", ["candidate_support_rate", "coverage_rate"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.1])
def test_nonfinite_or_impossible_support_rates_cannot_restore_events(field, value):
    event, arcs, track = inputs()
    arcs[1][field] = value
    assert not certificate(event, arcs, track)["supported"]


def test_recovery_keeps_track_holds_and_original_event_evidence():
    event, arcs, track = inputs()
    event.update(
        clip="m__pt1",
        match_id="m",
        abstain=True,
        tracking_arc_gate={"decision": "hold", "retained_arc_ids": []},
    )
    snapshot = TrackSnapshot({"m__pt1": track}, [], [])
    result = recover_events(
        [event], {"rows": [{"match_id": "m", "clip": "pt1", "arcs": arcs}]}, snapshot
    )[0]
    assert result["abstain"] is False and result["model_abstain"] is False
    assert result["pre_impulse_abstain"] is True
    assert result["tracking_arc_gate"]["decision"] == "supported_impulse_event_only"
    assert result["tracking_arc_gate"]["track_frames_restored"] is False
    assert result["tracking_arc_gate"]["retained_arc_ids"] == []
    assert result["pre_impulse_tracking_arc_gate"] == event["tracking_arc_gate"]
    assert result["frame"] == event["frame"] and result["location"] == event["location"]
    assert event["abstain"] is True


def native_track(tmp_path):
    root = tmp_path / "root"
    (root / "m").mkdir(parents=True)
    path = root / "m/ball_track_joint_native1080_arc_augmented_v2.csv"
    path.write_text("clip,frame,x,y,x_native,y_native\npt1,f_0001.jpg,5,10,10,20\n")
    write_native_dual_coordinate_manifest(
        coordinate_manifest_path(path),
        image_size=FrameSize(1920, 1080),
        legacy_size=FrameSize(960, 540),
        source="test",
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
    )
    return root, path


def test_snapshot_requires_native_contract_and_detects_concurrent_changes(tmp_path):
    root, path = native_track(tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    snapshot = TrackSnapshot.load(root, ["m"], extra_paths=(gate,))
    assert len(snapshot.records) == 3
    np.testing.assert_equal(snapshot.tracks["m__pt1"][1], [10, 20])
    snapshot.assert_unchanged()
    gate.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        snapshot.assert_unchanged()
    path.write_text("clip,frame,x,y\npt1,f_0001.jpg,5,10\n")
    with pytest.raises(ValueError, match="native track columns"):
        TrackSnapshot.load(root, ["m"])


def test_snapshot_rejects_duplicate_exposures(tmp_path):
    root, path = native_track(tmp_path)
    path.write_text(path.read_text() + "pt1,f_0001.jpg,5,10,10,20\n")
    with pytest.raises(ValueError, match="duplicate"):
        TrackSnapshot.load(root, ["m"])


def test_event_pixel_association_uses_original_interval_not_rounded_frame():
    event, arcs, track = inputs(knot=10.25)

    def state(t):
        dt = t - 10.25
        return np.array([300 + (80 if dt <= 0 else -40) * dt, 300 + (20 if dt <= 0 else -50) * dt])

    track = {t: state(t) for t in track}
    event["location"].update(frame_subpixel=10.25, image_x=state(9.6)[0], image_y=state(9.6)[1])
    event["frame_interval"] = [9.25, 11.25]
    before = deepcopy(event)
    result = certificate(event, arcs, track)
    assert result["supported"]
    assert result["rounded_event_pixel_error_native_px"] > 12
    assert result["event_pixel_error_native_px"] < 1e-6
    assert result["pixel_support_offset_frames"] == pytest.approx(-0.4, abs=1e-6)
    assert result["prediction_interval_frames"] == event["frame_interval"]
    assert result["localization_status"] == "interval_supported_only"
    assert not result["certifies_physical_ending"]
    assert event == before
    # A narrower supplied interval cannot borrow association from outside it.
    event["frame_interval"] = [10.15, 10.35]
    assert not certificate(event, arcs, track)["supported"]
    event["frame_interval"] = [11.0, 10.0]
    assert certificate(event, arcs, track)["reason"] == "invalid_original_prediction_interval"


def test_snapshot_excludes_explicit_and_copied_interpolation(tmp_path):
    root, path = native_track(tmp_path)
    guide = path.with_name("guide.csv")
    guide.write_text("clip,frame,x,y,x_native,y_native,sources\npt1,2,6,11,12,22,interpolated\n")
    coordinate_manifest_path(guide).write_text(coordinate_manifest_path(path).read_text())
    import json

    manifest = json.loads(coordinate_manifest_path(path).read_text())
    manifest["source"] = str(guide)
    coordinate_manifest_path(path).write_text(json.dumps(manifest))
    path.write_text(
        "clip,frame,x,y,x_native,y_native,sources\n"
        "pt1,f_0001.jpg,5,10,10,20,interpolated\n"
        "pt1,f_0002.jpg,6,11,12,22,coarse_lock\n"
        "pt1,f_0003.jpg,7,12,14,24,observed_detector\n"
    )
    snapshot = TrackSnapshot.load(root, ["m"])
    assert list(snapshot.tracks["m__pt1"]) == [3]
    assert snapshot.excluded_derived_frames["m__pt1"] == [1, 2]
    assert guide in snapshot.paths and coordinate_manifest_path(guide) in snapshot.paths


def test_identity_only_recovery_cannot_certify_an_ending():
    from cv.pipeline.event_grammar_decoder import point_end_rows

    def row(kind, frame, **extras):
        return dict(
            clip="m__pt1",
            match_id="m",
            event_type=kind,
            frame=frame,
            confidence=0.999,
            probability=0.999,
            abstain=False,
            gate_held=False,
            location=dict(court_x_fraction=1.5, court_y_fraction=0.5),
            **extras,
        )

    contact, bounce = row("contact", 1), row("bounce", 10)
    assert len(point_end_rows([contact, bounce])) == 1
    identity = dict(
        status="supported_by_model_and_observed_impulse", certifies_physical_ending=False
    )
    assert point_end_rows([contact, bounce | dict(event_identity_support=identity)]) == []
    # An interior recovery does not suppress an independently supported net ending.
    terminal = row("bounce", 30)
    net = row("net_hit", 20)
    ended = point_end_rows([contact | dict(event_identity_support=identity), net, terminal])
    assert ended[0]["point_end"]["termination_kind"] == "ground_after_net_hit"
    # A recovered first bounce cannot independently certify second-bounce legality.
    first = row("bounce", 10, event_identity_support=identity)
    first["location"]["court_x_fraction"] = 0.5
    second = row(
        "bounce",
        30,
        terminal_evidence=dict(
            termination_kind="second_bounce",
            preceding_bounce_candidate_frame=10,
            preceding_bounce_in_bounds=True,
        ),
    )
    second["location"]["court_x_fraction"] = 0.5
    assert point_end_rows([contact, first, second]) == []


def test_snapshot_rejects_duplicate_even_when_first_row_is_derived(tmp_path):
    root, path = native_track(tmp_path)
    path.write_text(
        "clip,frame,x,y,x_native,y_native,sources\npt1,f_0001.jpg,5,10,10,20,interpolated\npt1,f_0001.jpg,5,10,10,20,observed\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        TrackSnapshot.load(root, ["m"])


def test_snapshot_refuses_missing_declared_csv_guide(tmp_path):
    import json

    root, path = native_track(tmp_path)
    sidecar = coordinate_manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    manifest["source"] = str(path.with_name("missing.csv"))
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="guide source is unavailable"):
        TrackSnapshot.load(root, ["m"])


@pytest.mark.parametrize("raw_probability", [0.64, 0.94, 0.99])
def test_qualified_model_identity_does_not_rethreshold_raw_class_probability(raw_probability):
    event, arcs, track = inputs()
    event.update(probability=raw_probability, acceptance_marginal=0.98157)
    event["location"]["frame_subpixel"] = 10.25
    event["frame_interval"] = [9.25, 11.25]
    before = deepcopy(event)
    result = certificate(event, arcs, track)
    assert result["supported"]
    assert result["model_identity_admission"] == {
        "policy": "original_decoder_decision",
        "score_field": "acceptance_marginal",
        "score": 0.98157,
        "decision_threshold": 0.95,
        "original_model_abstain": False,
        "raw_class_probability_rethresholded": False,
    }
    assert result["prediction_interval_frames"] == event["frame_interval"]
    assert result["predicted_epoch_unchanged"] == 10.25
    assert event == before


@pytest.mark.parametrize(
    "change",
    [
        {"decision_threshold": None},
        {"decision_threshold": float("nan")},
        {"decision_threshold": -0.1},
        {"path_marginal": float("inf")},
        {"path_marginal": 0.94},
        # A terminal suffix may raise full-lattice confidence without admitting
        # its underlying physical witness under the original decoder threshold.
        {"acceptance_marginal": 0.87, "path_marginal": 0.999, "confidence": 0.999},
    ],
)
def test_inconsistent_original_model_admission_stays_held(change):
    event, arcs, track = inputs()
    result = certificate({**event, **change}, arcs, track)
    assert not result["supported"]
    assert result["reason"] == "original_decoder_admission_missing_or_inconsistent"


def test_model_abstained_toss_catch_cannot_be_restored_by_strong_motion_support():
    event, arcs, track = inputs()
    # A hand catch has a real motion impulse but is not an admitted in-play
    # contact. Native kinematic support must not override model/phase abstention.
    event.update(model_abstain=True, probability=0.99, path_marginal=0.68)
    before = deepcopy(event)
    result = certificate(event, arcs, track)
    assert not result["supported"]
    assert result["reason"] == "not_exclusively_tracking_held_physical_event"
    assert event == before


def test_available_native_wings_use_original_sparse_timestamps():
    event, arcs, track = inputs()
    del track[12]
    assert not certificate(event, arcs, track)["supported"]
    result = certificate(event, arcs, track, native_wing_sampling="available_native")
    assert result["supported"]
    assert result["wing_frames"][1] == [11, 13, 14, 15, 16]
    assert result["join_offset_frames"] == pytest.approx(0.25, abs=1e-6)
    assert 12 not in track
    assert max(result["wing_rms_native_px"]) < 1e-8


def test_sparse_wings_do_not_reach_past_duration_or_arc():
    event, arcs, track = inputs()
    del track[12]
    del track[13]
    assert not certificate(event, arcs, track, native_wing_sampling="available_native")["supported"]
    event, arcs, track = inputs()
    del track[12]
    arcs[-1]["end_frame"] = 15
    assert not certificate(event, arcs, track, native_wing_sampling="available_native")["supported"]


def test_complete_wings_keep_identical_numerics_with_available_sampling():
    event, arcs, track = inputs()
    assert certificate(event, arcs, track, native_wing_sampling="available_native") == certificate(
        event, arcs, track
    )


def test_invalid_native_sampling_fails_closed():
    event, arcs, track = inputs()
    with pytest.raises(ValueError, match="sampling"):
        certificate(event, arcs, track, native_wing_sampling="guess")


@pytest.mark.parametrize("missing", [9, 11])
def test_sparse_wings_do_not_extend_extrapolation_across_junction(missing):
    event, arcs, track = inputs()
    del track[missing]
    assert (
        certificate(event, arcs, track, native_wing_sampling="available_native")["reason"]
        == "missing_native_wing_junction"
    )


def test_higher_native_cadence_preserves_minimum_sample_count():
    event, arcs, track = inputs()
    event["location"]["fps"] = 50.0
    # At50Hz ten actual samples are required. Nine incoming observations are
    # insufficient even though a flat five-sample rule would accept this arc.
    assert (
        certificate(event, arcs, track, native_wing_sampling="available_native")["reason"]
        == "insufficient_observed_wing"
    )
