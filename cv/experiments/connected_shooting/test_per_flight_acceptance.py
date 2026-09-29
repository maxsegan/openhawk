from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    event_constraints,
    model,
    per_flight_acceptance as acceptance,
    physical_compatibility,
)


def test_acceptance_and_search_share_runback_envelope():
    assert acceptance.CONTACT_XY_ENVELOPE_M is event_constraints.CONTACT_XY_ENVELOPE_M


def test_projection_scope_separates_bad_context_from_flight_and_reserved_pixels():
    rows = [
        {"frame": 9.0, "split": "training", "error_px": 500.0},
        {"frame": 10.0, "split": "training", "error_px": 3.0},
        {"frame": 19.0, "split": "withheld", "error_px": 7.0},
        {"frame": 20.0, "split": "training", "error_px": 4.0},
        {"frame": 21.0, "split": "training", "error_px": 200.0},
    ]
    result = acceptance.projection_scope_diagnostics(rows, 10.0, 20.0)
    scopes = result["scopes"]
    assert scopes["in_flight"]["training"] == {
        "pictures": 2,
        "rms_px": pytest.approx(np.sqrt(12.5)),
    }
    assert scopes["in_flight"]["withheld"] == {"pictures": 1, "rms_px": 7.0}
    assert scopes["before_flight"]["training"]["rms_px"] == 500.0
    assert scopes["after_flight"]["training"]["rms_px"] == 200.0
    assert result["gate_evidence_changed"] is False
    assert len(rows) == 5


CAMERA = np.array([[1000.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1000.0, 0.0], [0.0, 1.0, 0.0, 20.0]])


def _project(xyz: np.ndarray) -> np.ndarray:
    q = CAMERA @ np.r_[np.asarray(xyz, float), 1.0]
    return q[:2] / q[2]


def fixture():
    """Two connected flights whose pictures are the exact projected centres."""
    fps = 25.0
    spin = np.array([2.0, 0.0, 0.0])
    start = np.array([5.0, 2.0, 1.2])
    velocities = [np.array([0.4, 19.0, 4.0]), np.array([-0.3, -18.0, 4.5])]
    parameters = np.r_[start, np.concatenate(velocities), np.tile(spin, 2), [1.0, 1.0]]

    def build(contacts):
        frames = tuple(
            np.arange(int(np.ceil(a)), int(np.floor(b)) + 1, 2.0)
            for a, b in zip(contacts[:-1], contacts[1:])
        )
        scene = model.Scene(
            contact_frames=np.asarray(contacts, float),
            observation_frames=frames,
            cameras=tuple(np.repeat(CAMERA[None], len(f), axis=0) for f in frames),
            pixels=tuple(np.zeros((len(f), 2)) for f in frames),
            spin_parameters=np.tile(spin, (2, 1)),
            fps=fps,
            surface="hard",
            dynamics="measured_240hz",
            rebound_mode="point_scales",
        )
        scene.validate()
        return scene, frames

    # The point ends at its own first terminal impact, which the dynamics -- not
    # the fixture -- decides; probe it once, then close the scene there.
    probe, _ = build([10.0, 34.0, 80.0])
    terminal_impact = model.chain(probe, parameters)[1]["bounces"][0]
    contacts = [10.0, 34.0, float(terminal_impact["frame"])]
    scene, frames = build(contacts)
    fitted = model.chain(scene, parameters)
    scene = replace(
        scene,
        pixels=tuple(
            np.asarray([_project(row) for row in flight["positions"]]) for flight in fitted
        ),
    )
    scene.validate()
    heldout_frames = tuple(f[:1] + 1.0 for f in frames)
    heldout = replace(
        scene,
        observation_frames=heldout_frames,
        cameras=tuple(np.repeat(CAMERA[None], len(f), axis=0) for f in heldout_frames),
        pixels=tuple(np.zeros((len(f), 2)) for f in heldout_frames),
    )
    held_fit = model.chain(heldout, parameters)
    heldout = replace(
        heldout,
        pixels=tuple(
            np.asarray([_project(row) for row in flight["positions"]]) for flight in held_fit
        ),
    )
    heldout.validate()
    # The generator's terminal root is the declared physical observation. A
    # root a few ulps beyond its rounded query endpoint must not erase that
    # label from the fixture while the scorer correctly owns it by tolerance.
    fitted[-1]["bounces"] = [terminal_impact]
    bounces = tuple(np.asarray([b["frame"] for b in flight["bounces"]], float) for flight in fitted)
    targets = [
        [
            {
                "event_frame": float(b["frame"]),
                "xyz_m": np.asarray(b["x"], float).tolist(),
                "uncertainty_sigma_m": 0.0,
                "witness_mode": "subframe_graded_circle",
            }
            for b in flight["bounces"]
        ]
        for flight in fitted
    ]
    events = [{"event_type": "contact", "frame": contacts[0]}]
    for index, flight in enumerate(fitted):
        events.extend(
            {"event_type": "bounce", "frame": float(b["frame"])} for b in flight["bounces"]
        )
        events.append({"event_type": "contact", "frame": contacts[index + 1]})
    native = tuple(np.unique(np.r_[a, b]) for a, b in zip(frames, heldout_frames, strict=True))
    axes = np.zeros((sum(map(len, frames)), 2))
    axes[:, 0] = 1.0
    return dict(
        scene=scene,
        heldout=heldout,
        parameters=parameters,
        bounces=bounces,
        native=native,
        axes=axes,
        targets=targets,
        events=events,
    )


def measured(case=None, **overrides):
    case = {**(case or fixture()), **overrides}
    return acceptance.measure(
        case["scene"],
        case["heldout"],
        case["parameters"],
        case["bounces"],
        case["native"],
        case["axes"],
        case["targets"],
        None,
        case["events"],
        termination_kind="terminal_bounce",
        duration=None,
    )


def test_graded_circle_floor_is_explicit_and_two_sigma_still_widens_it():
    assert acceptance.graded_circle_radius_m(None, 0.20) == 0.20
    assert acceptance.graded_circle_radius_m(0.05, 0.20) == 0.20
    assert acceptance.graded_circle_radius_m(0.05, 0.40) == 0.40
    assert acceptance.graded_circle_radius_m(0.30, 0.20) == pytest.approx(0.60)


def test_flight_roles_read_like_a_rally():
    assert acceptance.flight_role(0, 1) == "serve"
    assert [acceptance.flight_role(i, 4) for i in range(4)] == [
        "serve",
        "return",
        "mid_rally",
        "final",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounce_circle_floor_m": -0.1},
        {"directional_rms_limit_px": float("nan")},
        {"bounce_uncertainty_frames": 0.0},
        {"ending_uncertainty_frames": 2.5},
        # 3.5 is the 2026-09-24 owner cap. Past it still fails closed.
        {"bounce_uncertainty_frames": 4.0},
        {"serve_bounce_ray_limit_m": 1.5},
    ],
)
def test_unusable_thresholds_fail_closed(kwargs):
    with pytest.raises(ValueError):
        acceptance.Thresholds(**kwargs).validate()


def test_every_directional_window_belongs_to_exactly_one_flight():
    case = fixture()
    result = measured(case)
    windows = acceptance.directional_windows(
        [
            {"frame": float(frame), "split": "training", "error_px": 1.0}
            for group in case["scene"].observation_frames
            for frame in group
        ],
        case["events"],
        np.asarray(case["scene"].contact_frames, float),
    )
    assert windows and all(row["flight_index"] in (0, 1) for row in windows)
    assert {row["flight_index"] for row in result["flights"][0]["directional_windows"]} == {0}
    assert {row["flight_index"] for row in result["flights"][1]["directional_windows"]} == {1}


def test_directional_window_records_zero_and_best_shift_residuals():
    rows = [
        {
            "frame": frame,
            "split": "training",
            "error_px": 34.0,
            "time_shift_error_px": {
                "-1.0": 12.0,
                "-0.5": 20.0,
                "0.5": 40.0,
                "1.0": 50.0,
            },
        }
        for frame in (11, 12, 13)
    ]
    windows = acceptance.directional_windows(
        rows,
        [{"event_type": "contact", "frame": 10.5}],
        np.asarray([10.0, 20.0]),
    )
    short = next(
        row for row in windows if row["direction"] == "forward" and row["horizon"] == "short"
    )
    assert short["zero_shift_rms_px"] == 34.0
    assert short["best_time_shift_frames"] == -1.0
    assert short["best_shift_rms_px"] == 12.0


def test_an_exactly_projected_point_accepts_every_flight_and_the_whole_point():
    case = fixture()
    verdict = acceptance.score(measured(case), acceptance.Thresholds())
    assert verdict["accepted_flight_count"] == 2
    assert verdict["complete_point"] and not verdict["partial_point"]
    assert verdict["gaps"] == []
    assert not verdict["complete_real_point_accepted"]
    whole = physical_compatibility.evaluate(
        case["scene"],
        case["parameters"],
        case["bounces"],
        "terminal_bounce",
        case["native"],
    )
    assert whole["compatible"]


def test_one_displaced_bounce_witness_rejects_only_its_own_flight():
    case = fixture()
    targets = [[dict(row) for row in group] for group in case["targets"]]
    targets[0][0]["xyz_m"] = [
        targets[0][0]["xyz_m"][0] + 3.0,
        targets[0][0]["xyz_m"][1],
        targets[0][0]["xyz_m"][2],
    ]
    verdict = acceptance.score(measured(case, targets=targets), acceptance.Thresholds())
    assert verdict["accepted_flight_indices"] == [1]
    assert verdict["rejected_flight_indices"] == [0]
    assert verdict["partial_point"] and not verdict["complete_point"]
    assert verdict["gaps"][0]["failures"] == ["bounce_rays_agree"]
    assert verdict["flights"][0]["role"] == "serve"


def test_a_wider_bounce_circle_readmits_that_flight_and_nothing_else_changes():
    case = fixture()
    targets = [[dict(row) for row in group] for group in case["targets"]]
    witness = targets[0][0]["xyz_m"]
    targets[0][0]["xyz_m"] = [witness[0] + 1.2, witness[1], witness[2]]
    base = measured(case, targets=targets)
    strict = acceptance.score(base, acceptance.Thresholds())
    relaxed = acceptance.score(base, acceptance.Thresholds(bounce_circle_floor_m=0.40))
    assert strict["rejected_flight_indices"] == [0]
    assert relaxed["accepted_flight_count"] == 2 and relaxed["complete_point"]


def test_frozen_legacy_witness_preserves_an_existing_accept_when_reversal_abstains():
    case = fixture()
    targets = [[dict(row) for row in group] for group in case["targets"]]
    for group in targets:
        for target in group:
            target["legacy_subframe_witness"] = {
                "xyz_m": list(target["xyz_m"]),
                "uncertainty_sigma_m": 0.0,
            }
    original = list(targets[0][0]["xyz_m"])
    targets[0][0].update(
        xyz_m=[original[0] + 3.0, original[1], original[2]],
        uncertainty_sigma_m=1.0,
        ground_covariance_xy_m2=[[1.0, 0.0], [0.0, 1.0]],
        impact_epoch_witness={"delta_from_supplied_frames": 0.0},
        legacy_subframe_witness={"xyz_m": original, "uncertainty_sigma_m": 0.0},
    )
    verdict = acceptance.score(measured(case, targets=targets), acceptance.Thresholds())
    witness = verdict["flights"][0]["bounce_witness"][0]
    assert witness["legacy_agrees"] and not witness["reversal_agrees"]
    assert verdict["complete_point"]


def test_reversal_witness_cannot_newly_accept_an_unresolved_covariance():
    case = fixture()
    targets = [[dict(row) for row in group] for group in case["targets"]]
    original = list(targets[0][0]["xyz_m"])
    targets[0][0].update(
        uncertainty_sigma_m=0.31,
        ground_covariance_xy_m2=[[0.1, 0.0], [0.0, 0.1]],
        impact_epoch_witness={"delta_from_supplied_frames": 0.0},
        legacy_subframe_witness={
            "xyz_m": [original[0] + 3.0, original[1], original[2]],
            "uncertainty_sigma_m": 0.0,
        },
    )
    verdict = acceptance.score(measured(case, targets=targets), acceptance.Thresholds())
    witness = verdict["flights"][0]["bounce_witness"][0]
    assert not witness["covariance_resolved"] and not witness["agrees"]
    assert not verdict["complete_point"]


def test_grass_reversal_requires_a_tighter_timing_witness():
    case = fixture()
    targets = [[dict(row) for row in group] for group in case["targets"]]
    original = list(targets[0][0]["xyz_m"])
    targets[0][0].update(
        uncertainty_sigma_m=0.1,
        ground_covariance_xy_m2=[[0.01, 0.0], [0.0, 0.01]],
        impact_epoch_witness={"delta_from_supplied_frames": 0.9},
        legacy_subframe_witness={
            "xyz_m": [original[0] + 3.0, original[1], original[2]],
            "uncertainty_sigma_m": 0.0,
        },
    )
    hard = acceptance.score(measured(case, targets=targets), acceptance.Thresholds())
    grass_measurement = measured(case, targets=targets)
    grass_measurement["surface"] = "grass"
    grass = acceptance.score(grass_measurement, acceptance.Thresholds())
    assert hard["flights"][0]["bounce_witness"][0]["timing_witness_resolved"]
    assert not grass["flights"][0]["bounce_witness"][0]["timing_witness_resolved"]


def test_velocity_relaxation_never_refits_an_already_accepted_flight():
    case = fixture()
    verdict = acceptance.evaluate(
        case["scene"],
        case["heldout"],
        case["parameters"],
        case["bounces"],
        case["native"],
        case["axes"],
        case["targets"],
        None,
        case["events"],
        termination_kind="terminal_bounce",
        thresholds=acceptance.Thresholds(velocity_slack_mps=2.0),
        duration=None,
    )
    assert verdict["relaxation"]["attempted"] == 0
    assert verdict["accepted_flight_count"] == 2 and verdict["complete_point"]
    assert verdict["maximum_junction_gap_m"] == pytest.approx(0.0, abs=1e-6)


def test_the_measurement_never_moves_the_fitted_state_or_an_exposure():
    case = fixture()
    parameters = case["parameters"].copy()
    contacts = case["scene"].contact_frames.copy()
    measured(case)
    np.testing.assert_array_equal(case["parameters"], parameters)
    np.testing.assert_array_equal(case["scene"].contact_frames, contacts)


SIGMA = acceptance.Thresholds(net_clearance_sigma_m=0.10)


def _with_clearance(result, flight_index, centre_clearance_m):
    """The same measurement with one modelled crossing moved in height only."""
    flights = [dict(row) for row in result["flights"]]
    row = flights[flight_index]
    crossing = dict(row["net_crossings"][0])
    band = crossing["net_band_height_m"]
    crossing["ball_centre_clearance_m"] = centre_clearance_m
    crossing["ball_surface_clearance_m"] = centre_clearance_m - model.R_BALL
    crossing["penetration"] = bool(crossing["ball_surface_clearance_m"] < -1e-3)
    crossing["xyz"] = [crossing["xyz"][0], crossing["xyz"][1], band + centre_clearance_m]
    row["net_crossings"] = [crossing]
    row["net_penetrations"] = [crossing] if crossing["penetration"] else []
    return {**result, "flights": flights}


def test_the_net_clearance_verdict_is_the_frozen_three_way_reading():
    assert acceptance.net_crossing_verdict(0.30, 0.10) == "cleared"
    assert acceptance.net_crossing_verdict(-0.30, 0.10) == "into_net"
    for margin in (-0.073, 0.0, 0.10, -0.10):
        assert acceptance.net_crossing_verdict(margin, 0.10) == "net_contact"


def test_a_net_ending_is_read_from_the_free_text_kind_not_the_scoring_family():
    assert acceptance.net_ending_labeled("net_failure_ground")
    assert acceptance.net_ending_labeled("serve_fault_net")
    assert not acceptance.net_ending_labeled("out_long")
    assert not acceptance.net_ending_labeled(None)


def test_a_crossing_inside_one_sigma_is_accepted_where_the_binary_check_rejects_it():
    # atpf2023sf_pt0003's shape: the ball centre passes 0.073 m below the band,
    # which the geometry cannot separate from a cord clip.
    result = _with_clearance(measured(), 1, -0.073)
    assert acceptance.score(result, acceptance.Thresholds())["flights"][1]["failures"] == [
        "no_net_penetration"
    ]
    verdict = acceptance.score(result, SIGMA, ending_kind="out")
    assert verdict["flights"][1]["accepted"]
    assert "no_net_penetration" not in verdict["flights"][1]["checks"]
    assert verdict["flights"][1]["checks"]["no_net_mesh_traversal"]
    assert verdict["flights"][1]["net_clearance"][0]["verdict"] == "net_contact"


def test_a_fitted_mesh_traversal_stays_wrong_however_the_point_ended():
    result = _with_clearance(measured(), 1, -0.35)
    for kind in ("out", "net_failure", None):
        verdict = acceptance.score(result, SIGMA, ending_kind=kind)
        assert not verdict["flights"][1]["checks"]["no_net_mesh_traversal"]
        assert not verdict["flights"][1]["accepted"]


def test_a_net_ending_label_requires_net_contact_on_the_flight_that_ends_the_point():
    # The fitted terminal flight sails 0.28 m clear of the band, which is a
    # clearance the geometry can call, so a net ending is not supported.
    result = measured()
    assert acceptance.score(result, SIGMA, ending_kind="out")["flights"][1]["accepted"]
    verdict = acceptance.score(result, SIGMA, ending_kind="return_into_net")
    assert not verdict["flights"][1]["checks"]["net_ending_supported"]
    assert not verdict["flights"][1]["accepted"]
    # A crossing inside the band is the cord clip the label describes.
    inside = _with_clearance(result, 1, -0.05)
    assert acceptance.score(inside, SIGMA, ending_kind="return_into_net")["flights"][1]["accepted"]
    # The requirement is on the terminal flight only; an earlier flight of the
    # same point still just has to stay out of the mesh.
    assert acceptance.score(result, SIGMA, ending_kind="return_into_net")["flights"][0]["accepted"]


def test_a_labeled_net_hit_the_fit_never_produces_is_still_rejected_under_the_sigma():
    case = fixture()
    scene = replace(case["scene"], net_hit_frames=(np.empty(0), np.asarray([41.5], float)))
    scene.validate()
    result = measured(case, scene=scene)
    verdict = acceptance.score(result, SIGMA, ending_kind="net_failure")
    assert not verdict["flights"][1]["checks"]["supplied_net_transitions_met"]
    assert not verdict["flights"][1]["accepted"]


def _trailing_front_case():
    """The fixture with one native front recorded after the ground impact."""
    case = fixture()
    scene = case["scene"]
    impact = float(scene.contact_frames[-1])
    trailing = float(np.ceil(impact))
    frames = (scene.observation_frames[0], np.r_[scene.observation_frames[1], trailing])
    extended = replace(
        scene,
        contact_frames=np.r_[scene.contact_frames[:-1], trailing],
        observation_frames=frames,
        cameras=tuple(np.repeat(CAMERA[None], len(f), axis=0) for f in frames),
        pixels=tuple(np.zeros((len(f), 2)) for f in frames),
    )
    fitted = model.chain(extended, case["parameters"])
    extended = replace(
        extended,
        pixels=tuple(
            np.asarray([_project(row) for row in flight["positions"]]) for flight in fitted
        ),
    )
    extended.validate()
    heldout = replace(case["heldout"], contact_frames=np.r_[scene.contact_frames[:-1], trailing])
    heldout.validate()
    native = (case["native"][0], np.unique(np.r_[case["native"][1], trailing]))
    axes = np.zeros((sum(map(len, frames)), 2))
    axes[:, 0] = 1.0
    return {
        **case,
        "scene": extended,
        "heldout": heldout,
        "native": native,
        "axes": axes,
        "impact_frame": impact,
        "trailing_frame": trailing,
    }


def test_a_front_after_the_ending_holds_the_point_with_the_passive_arm_off():
    case = _trailing_front_case()
    result = measured(case)
    assert not result["ending_completed"]
    assert (
        result["terminal_completion"]["reason"] == "completion_would_discard_observations_or_flight"
    )
    verdict = acceptance.score(result, acceptance.Thresholds())
    assert "physical_ground_ending" in verdict["flights"][1]["failures"]
    assert not verdict["complete_point"]


def test_the_point_ends_at_its_impact_and_the_later_front_stays_passive_context():
    case = _trailing_front_case()
    result = acceptance.measure(
        case["scene"],
        case["heldout"],
        case["parameters"],
        case["bounces"],
        case["native"],
        case["axes"],
        case["targets"],
        None,
        case["events"],
        termination_kind="terminal_bounce",
        duration=None,
        ending_passive_context_frames=1.0,
    )
    assert result["ending_completed"]
    assert result["terminal_completion"]["end_frame"] == pytest.approx(case["impact_frame"])
    assert result["terminal_completion"]["observations_discarded"] == 0
    row = result["flights"][1]
    # The exposure is retained at its own timestamp, is named as passive, and
    # the flight's own native inventory still counts it.
    assert row["post_ending_passive_exposure_frames"] == [case["trailing_frame"]]
    assert row["native_exposures"] == len(case["native"][1])
    assert row["end_frame"] == pytest.approx(case["impact_frame"])
    scopes = row["projection_scope_diagnostics"]["scopes"]
    assert sum(r["pictures"] for r in scopes["after_flight"].values()) == 1
    assert scopes["in_flight"]["training"]["pictures"] > 0
    verdict = acceptance.score(result, acceptance.Thresholds(ending_passive_context_frames=1.0))
    assert verdict["flights"][1]["checks"]["physical_ground_ending"]
    assert verdict["complete_point"]
    assert verdict["flights"][1]["post_ending_passive_exposure_frames"] == [case["trailing_frame"]]


def test_passive_context_never_turns_a_post_ending_contact_into_a_live_flight():
    case = _trailing_front_case()
    events = case["events"] + [{"event_type": "contact", "frame": case["trailing_frame"]}]
    result = acceptance.measure(
        case["scene"],
        case["heldout"],
        case["parameters"],
        case["bounces"],
        case["native"],
        case["axes"],
        case["targets"],
        None,
        events,
        termination_kind="terminal_bounce",
        duration=None,
        ending_passive_context_frames=1.0,
    )
    # The point has two flights before and after; the pickup adds no shot, and
    # the ending fails closed rather than absorbing something that may be live.
    assert result["flight_count"] == 2
    assert not result["ending_completed"]
    assert result["terminal_completion"]["reason"] == (
        "passive_post_ending_context:live_contact_inside_passive_span"
    )


def test_terminal_native_integer_roundoff_query_preserves_observations():
    case = fixture()
    scene, parameters = case["scene"], case["parameters"]
    end = float(np.nextafter(np.nextafter(np.nextafter(159.0, -np.inf), -np.inf), -np.inf))
    contacts = scene.contact_frames + (end - scene.contact_frames[-1])
    contacts[-1] = end
    start = float(contacts[-2])
    interior = (start + end) / 2
    native = np.array([start, interior, 159.0, 159.01])
    original = native.copy()
    queries = acceptance._flight_queries(start, end, native, scene.fps)
    np.testing.assert_array_equal(native, original)
    assert queries[-1] == end
    assert interior in queries
    assert 159.0 not in queries
    assert 159.01 not in queries
    actual = model.chain(
        replace(scene, contact_frames=contacts),
        parameters,
        query_frames=(np.array([contacts[0], contacts[1]]), queries),
    )
    assert len(actual[-1]["positions"]) == len(queries)


def test_materially_early_native_query_remains_an_error():
    with pytest.raises(ValueError, match="precedes trajectory domain"):
        acceptance._flight_queries(10.0, 20.0, np.array([9.99, 15.0]), 25.0)


def test_tolerance_probe_capacity_refuses_only_last_flight(monkeypatch):
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    case = fixture()
    baseline = measured(case)
    original = model.chain
    end = baseline["flights"][-1]["end_frame"]

    def fail_probe(scene, *args, **kwargs):
        if abs(scene.contact_frames[-1] - (end + 1e-6)) < 1e-12:
            raise BounceCapacityError(
                end + 0.5e-6, len(original(case["scene"], case["parameters"])[-1]["bounces"]) + 1
            )
        return original(scene, *args, **kwargs)

    monkeypatch.setattr(model, "chain", fail_probe)
    actual = measured(case)
    assert actual["flights"][0] == baseline["flights"][0]
    last = actual["flights"][-1]
    receipt = last["terminal_impact_census"]
    assert receipt["complete"] is False
    assert receipt["tolerance_domain_impact_count"] is None
    assert receipt["retained_domain_impact_count"] == last["modeled_bounce_count"]
    assert last["modeled_bounce_count_scope"] == "retained_domain_only_incomplete_census"
    verdict = acceptance.score(
        actual,
        acceptance.Thresholds(),
        extra_flight_checks={
            1: {"terminal_impact_census_supported": True, "bounce_count_timing": True}
        },
    )
    ordinary = acceptance.score(baseline, acceptance.Thresholds())
    assert all("terminal_impact_census" not in row for row in baseline["flights"])
    assert all("terminal_impact_census" not in row for row in ordinary["flights"])
    assert verdict["flights"][0] == ordinary["flights"][0]
    assert verdict["flights"][-1]["accepted"] is False
    assert "terminal_impact_census_supported" in verdict["flights"][-1]["failures"]
    assert verdict["flights"][-1]["checks"]["bounce_count_timing"] is False
    assert verdict["flights"][-1]["terminal_impact_census"] == receipt


@pytest.mark.parametrize(
    "failure", ["ordinary", "within_retained", "outside_probe", "count_too_low", "count_too_high"]
)
def test_tolerance_probe_does_not_hide_other_domain_failures(monkeypatch, failure):
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    case = fixture()
    baseline = measured(case)
    end = baseline["flights"][-1]["end_frame"]
    original = model.chain

    def fail_probe(scene, *args, **kwargs):
        if abs(scene.contact_frames[-1] - (end + 1e-6)) < 1e-12:
            if failure == "ordinary":
                raise ValueError("different physical-domain failure")
            if failure in {"count_too_low", "count_too_high"}:
                retained = len(original(case["scene"], case["parameters"])[-1]["bounces"])
                increment = 0 if failure == "count_too_low" else 2
                raise BounceCapacityError(end + 0.5e-6, retained + increment)
            offset = -0.5e-6 if failure == "within_retained" else 2e-6
            raise BounceCapacityError(end + offset, 8)
        return original(scene, *args, **kwargs)

    monkeypatch.setattr(model, "chain", fail_probe)
    with pytest.raises(ValueError if failure == "ordinary" else BounceCapacityError):
        measured(case)


@pytest.mark.parametrize("backend", ["measured", "net"])
def test_real_impact_capacity_exposes_same_physical_boundary(backend):
    from cv.experiments.connected_shooting import measured_dynamics, net_collision

    frames = np.arange(1.0, 501.0)
    if backend == "net":
        producer = net_collision
        theta = np.array([5.5, 18, 1, 0, -20, -1, 0, 0, 0.0])
        kwargs = {"net_frame": 8.64}
    else:
        producer = measured_dynamics
        theta = np.array([2, 3, 1, 1, 2, -2, 0, 0, 0.0])
        kwargs = {}
    with pytest.raises(measured_dynamics.BounceCapacityError) as caught:
        producer.simulate(theta, 1, frames, 25, "hard", ground_settling=False, **kwargs)
    error = caught.value
    assert str(error) == "measured dynamics bounce cap reached"
    assert error.resolved_impacts == 8
    boundary = error.frame
    before = producer.simulate(
        theta,
        1,
        np.array([1.0, boundary - 0.5e-6]),
        25,
        "hard",
        ground_settling=False,
        **kwargs,
    )
    assert len(before[3]) == 7
    with pytest.raises(measured_dynamics.BounceCapacityError) as repeated:
        producer.simulate(
            theta,
            1,
            np.array([1.0, boundary + 0.5e-6]),
            25,
            "hard",
            ground_settling=False,
            **kwargs,
        )
    assert repeated.value.frame == pytest.approx(boundary, abs=1e-9)


def test_real_net_capacity_boundary_through_measure_and_score():
    from cv.experiments.connected_shooting import measured_dynamics, net_collision

    theta = np.array([5.5, 18, 1, 0, -20, -1, 0, 0, 0.0])
    with pytest.raises(measured_dynamics.BounceCapacityError) as caught:
        net_collision.simulate(
            theta,
            1,
            np.arange(1.0, 501.0),
            25,
            "hard",
            net_frame=8.64,
            ground_settling=False,
        )
    end = caught.value.frame - 0.5e-6
    frames = np.arange(2.0, np.floor(end) - 2, 2)
    scene = model.Scene(
        contact_frames=np.array([1.0, end]),
        observation_frames=(frames,),
        cameras=(np.repeat(CAMERA[None], len(frames), axis=0),),
        pixels=(np.zeros((len(frames), 2)),),
        spin_parameters=np.zeros((1, 3)),
        fps=25,
        surface="hard",
        dynamics="measured_240hz",
        net_hit_frames=(np.array([8.64]),),
    )
    parameters = theta[:6]
    fitted = model.chain(scene, parameters)[0]
    assert len(fitted["bounces"]) == 7
    scene = replace(scene, pixels=(np.array([_project(x) for x in fitted["positions"]]),))
    bounces = (np.array([b["frame"] for b in fitted["bounces"]]),)
    targets = [
        [
            dict(
                event_frame=b["frame"],
                xyz_m=b["x"].tolist(),
                uncertainty_sigma_m=0,
                witness_mode="subframe_graded_circle",
            )
            for b in fitted["bounces"]
        ]
    ]
    raw = acceptance.measure(
        scene,
        scene,
        parameters,
        bounces,
        (frames,),
        np.tile([1.0, 0.0], (len(frames), 1)),
        targets,
        None,
        [{"event_type": "contact", "frame": 1}, {"event_type": "net_hit", "frame": 8.64}],
        termination_kind="terminal_bounce",
        duration=None,
        apply_terminal_completion=False,
        preserve_observation_horizon=True,
    )
    receipt = raw["flights"][0]["terminal_impact_census"]
    assert receipt["guard_impact_frame"] == pytest.approx(caught.value.frame, abs=1e-9)
    assert receipt["guard_resolved_impacts"] == 8
    assert receipt["retained_domain_impact_count"] == 7
    verdict = acceptance.score(raw, acceptance.Thresholds())
    assert verdict["accepted_flight_count"] == 0
    assert "terminal_impact_census_supported" in verdict["flights"][0]["failures"]


@pytest.mark.parametrize("offset,expected_refusal", [(0.0, True), (1e-6, False)])
def test_tolerance_probe_exact_boundary_ownership(monkeypatch, offset, expected_refusal):
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    case = fixture()
    baseline = measured(case)
    end = baseline["flights"][-1]["end_frame"]
    original = model.chain
    retained = len(original(case["scene"], case["parameters"])[-1]["bounces"])

    def fail_probe(scene, *args, **kwargs):
        if abs(scene.contact_frames[-1] - (end + 1e-6)) < 1e-12:
            raise BounceCapacityError(end + offset, retained + 1)
        return original(scene, *args, **kwargs)

    monkeypatch.setattr(model, "chain", fail_probe)
    if expected_refusal:
        with pytest.raises(BounceCapacityError):
            measured(case)
    else:
        raw = measured(case)
        assert raw["flights"][-1]["terminal_impact_census"]["complete"] is False


@pytest.mark.parametrize("pieces", [1, 2])
def test_velocity_relaxation_retains_incomplete_terminal_census(monkeypatch, pieces):
    case = fixture()
    original_measure = acceptance.measure

    def incomplete_measure(*args, **kwargs):
        raw = original_measure(*args, **kwargs)
        last = raw["flights"][-1]
        last["terminal_impact_census"] = {
            "status": "unavailable",
            "complete": False,
            "retained_domain_impact_count": last["modeled_bounce_count"],
            "tolerance_domain_impact_count": None,
        }
        last["modeled_bounce_count_scope"] = "retained_domain_only_incomplete_census"
        return raw

    monkeypatch.setattr(acceptance, "measure", incomplete_measure)
    scene = case["scene"]
    index = 1
    fitted = model.chain(scene, case["parameters"])[index]
    row = acceptance.relax_flight(
        scene,
        scene,
        case["parameters"],
        index,
        fitted["start_xyz"],
        np.tile([1.0, 0.0], (len(scene.observation_frames[index]), 1)),
        case["bounces"][index],
        case["native"][index],
        case["targets"][index],
        None,
        case["events"],
        slack_mps=0.1,
        duration=None,
        termination_kind="terminal_bounce",
        ending_uncertainty_frames=1.0,
        ending_passive_context_frames=0,
        terminal=True,
        flight_total=2,
        athlete=None,
        split_frame=None if pieces == 1 else 43.5,
        max_nfev=2,
        thresholds=acceptance.Thresholds(),
        depth_bounds=None,
        athlete_prior_mode="stature_pose_soft",
    )
    assert row is not None
    diagnostic = row["projection_scope_diagnostics"]
    if pieces == 2:
        assert len(diagnostic["pieces"]) == 2
        assert diagnostic["pieces"][0]["end_frame"] == pytest.approx(43.5)
        assert diagnostic["pieces"][1]["start_frame"] == pytest.approx(43.5)
    assert row["terminal_impact_census"]["complete"] is False
    assert (
        row["terminal_impact_census"]["retained_domain_impact_count"] == row["modeled_bounce_count"]
    )
    assert row["modeled_bounce_count_scope"] == "retained_domain_only_incomplete_census"
    base = measured(case)
    base["flights"][-1] = row
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert "terminal_impact_census_supported" in verdict["flights"][-1]["failures"]


def test_capacity_error_survives_worker_exception_transport():
    import pickle
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    original = BounceCapacityError(115.0000003, 8)
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is BounceCapacityError
    assert restored.frame == original.frame
    assert restored.resolved_impacts == original.resolved_impacts
    assert str(restored) == str(original)


def test_boundary_close_must_leave_as_a_struck_shot():
    base = measured()
    row = dict(base["flights"][-1])
    row.update(
        terminal=True,
        supported_ending_kind="point_end",
        partial_flight=True,
        supported_ending_evidence="labelled_boundary",
        start_xyz_m=[4.8, -0.9, 1.3],
    )
    # A dead ball tapped backwards after play ended.
    base["flights"][-1] = dict(row, start_velocity_mps=[-2.6, -0.8, 4.2])
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert "boundary_close_struck_shot" in verdict["flights"][-1]["failures"]
    base["flights"][-1] = dict(row, start_velocity_mps=[1.0, 18.0, 3.0])
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert verdict["flights"][-1]["checks"]["boundary_close_struck_shot"] is True
    # A contact slid 8 m up the viewing ray is not a racket.
    base["flights"][-1] = dict(
        row, start_xyz_m=[4.9, -6.7, 8.0], start_velocity_mps=[1.0, 17.0, -6.0]
    )
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert "boundary_close_struck_shot" in verdict["flights"][-1]["failures"]


def test_boundary_close_must_reach_an_abstained_landing_or_net_row():
    base = measured()
    row = dict(base["flights"][-1])
    row.update(
        terminal=True,
        supported_ending_kind="point_end",
        partial_flight=True,
        supported_ending_evidence="labelled_boundary",
        start_xyz_m=[4.0, 1.0, 1.0],
        start_velocity_mps=[0.0, 20.0, 3.0],
        boundary_row_event="bounce",
        boundary_row_lead_s=0.02,
        end_velocity_mps=[0.0, 20.0, -5.0],
    )
    # Still 1.2 m up one frame before the landing row: wrong depth, not a landing.
    base["flights"][-1] = dict(row, end_xyz_m=[4.0, 18.0, 1.2])
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert "boundary_close_reaches_row" in verdict["flights"][-1]["failures"]
    base["flights"][-1] = dict(row, end_xyz_m=[4.0, 18.0, 0.15])
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert verdict["flights"][-1]["checks"]["boundary_close_reaches_row"] is True
    # A net row needs the ball near the net plane.
    base["flights"][-1] = dict(row, boundary_row_event="net_hit", end_xyz_m=[4.0, 7.1, 1.4])
    verdict = acceptance.score(base, acceptance.Thresholds())
    assert "boundary_close_reaches_row" in verdict["flights"][-1]["failures"]
