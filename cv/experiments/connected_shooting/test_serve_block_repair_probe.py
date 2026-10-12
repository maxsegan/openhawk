"""Boundary repair scope, original witness route and missing-toss behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import model, serve_block_repair_probe as probe


def test_two_flight_serve_moves_first_contacts_but_not_right_endpoint_or_rebound():
    scene = SimpleNamespace(contact_frames=np.arange(3))
    blocks = model.shared_parameter_slices(scene)
    p = np.arange(blocks["rebound_scales"].stop, dtype=float)
    q = p.copy()
    indices = probe.serve_indices(scene)
    q[indices] += 1
    assert len(indices) == 18
    np.testing.assert_array_equal(p[6:9], q[6:9])
    np.testing.assert_array_equal(p[blocks["rebound_scales"]], q[blocks["rebound_scales"]])
    assert np.all(p[:6] != q[:6])


@pytest.mark.parametrize("count", [1, 3])
def test_single_flight_net_and_larger_points_outside_this_bounded_arm(count):
    with pytest.raises(ValueError, match="exactly two flights"):
        probe.serve_indices(SimpleNamespace(contact_frames=np.arange(count + 1)))


def test_existing_chord_uses_tighter_serve_gate_without_resolving_primary_covariance():
    witness = dict(
        legacy_subframe_witness={"xyz_m": [1.0, 2.0, 0.0325], "available": True},
        supplied_frame=100.5,
        legacy_graded_circle_radius_m=0.2,
        covariance_resolved=False,
    )
    context = dict(
        repair_reference_verdict={"flights": [{"bounce_witness": [witness]}]},
        repair_thresholds=SimpleNamespace(
            serve_bounce_ray_limit_m=0.5, bounce_ray_limit_m=0.75, bounce_uncertainty_frames=2.0
        ),
    )
    plan = probe.existing_serve_bounce_plan(context)
    assert plan[0]["maximum_distance_m"] == pytest.approx(0.6999)
    assert not plan[0]["primary_covariance_resolved"]
    witness.pop("legacy_subframe_witness")
    with pytest.raises(ValueError, match="existing frozen chord"):
        probe.existing_serve_bounce_plan(context)


def test_missing_precontact_toss_stays_explicitly_absent(monkeypatch):
    monkeypatch.setattr(
        probe.toss_witness,
        "contact_constraint_observations",
        lambda *a, **kw: dict(status="abstained", rows=[]),
    )
    context = dict(
        events=[dict(event_type="contact", frame=10.5, frame_interval=[10.0, 11.0])],
        attempt={"point_clip": "test"},
    )
    result = probe.toss_evidence(context, {}, {}, None, 1.0, np.zeros(3), np.ones(3))
    assert result["status"] == "abstained"
    assert result["estimate"] is None
    assert result["observations"]["rows"] == []


def test_no_invented_incoming_image_wing_at_initial_serve():
    scene = SimpleNamespace(
        contact_frames=np.array([10.5, 20.5, 30.5]),
        observation_frames=(np.array([11.0, 12.0, 18.0, 19.0]), np.array([21.0, 22.0, 29.0])),
    )
    assert [x.tolist() for x in probe.serve_contact_image_groups(scene)] == [[0, 1], [2, 3], [4, 5]]


def test_stricter_serve_guard_reselects_and_updates_all_selection_fields():
    before = {"accepted_flight_count": 1}
    good = dict(
        start="reference",
        eligible_improvement=True,
        contact_image_constraints_valid=True,
        all_frozen_gate_constraints_valid=True,
        verdict={"accepted_flight_count": 2},
        cost=10.0,
    )
    failed = dict(
        start="toss_contact",
        eligible_improvement=True,
        contact_image_constraints_valid=True,
        all_frozen_gate_constraints_valid=False,
        verdict={"accepted_flight_count": 2},
        cost=9.0,
    )
    result = dict(status="measured", before=before, trials=[good, failed], selected="toss_contact")
    probe.finalize_selection(result)
    assert result["selected"] == result["selected_trial_start"] == "reference"
    assert result["selected_source"] == "repair_trial"
    assert not failed["eligible_improvement"]
    good["all_frozen_gate_constraints_valid"] = False
    probe.finalize_selection(result)
    assert result["selected_source"] == "original_reference"
    assert result["selected_trial_start"] is None
    assert result["selected_verdict"] is before
