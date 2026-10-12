"""Movable-contact block scope and native-image support contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import block_flight_repair_probe as probe
from cv.experiments.connected_shooting import model


def test_three_flight_indices_preserve_outer_contacts_and_external_parameters():
    scene = SimpleNamespace(contact_frames=np.arange(8))
    blocks = model.shared_parameter_slices(scene)
    initial = np.arange(blocks["rebound_scales"].stop, dtype=float)
    changed = initial.copy()
    indices = probe.block_indices(scene, 3)
    assert len(indices) == len(set(indices)) == 24
    changed[indices] += 0.5
    for contact in [0, 1, 2, 5, 6, 7]:
        np.testing.assert_array_equal(
            initial[3 * contact : 3 * contact + 3], changed[3 * contact : 3 * contact + 3]
        )
    for block in ("velocities", "spins"):
        old, new = initial[blocks[block]].reshape(-1, 3), changed[blocks[block]].reshape(-1, 3)
        np.testing.assert_array_equal(old[[0, 1, 5, 6]], new[[0, 1, 5, 6]])
        assert np.all(new[[2, 3, 4]] != old[[2, 3, 4]])
    np.testing.assert_array_equal(
        initial[blocks["rebound_scales"]], changed[blocks["rebound_scales"]]
    )


@pytest.mark.parametrize("index", [-1, 0, 6, 7])
def test_block_needs_both_neighbors(index):
    with pytest.raises(ValueError, match="three-flight block"):
        probe.block_indices(SimpleNamespace(contact_frames=np.arange(8)), index)


def test_contact_constraints_use_native_images_from_each_side():
    scene = SimpleNamespace(
        contact_frames=np.array([0, 10, 20, 30, 40]),
        observation_frames=tuple(np.array([i + 1, i + 3, i + 7, i + 9]) for i in (0, 10, 20, 30)),
    )
    groups = probe.contact_image_groups(scene, 2)
    assert [group.tolist() for group in groups] == [[6, 7], [8, 9], [10, 11], [12, 13]]
    residual = np.zeros((16, 2))
    residual[6:8] = [3, 4]
    np.testing.assert_allclose(probe.image_budgets(residual, groups), [25, 0, 0, 0])


def test_unsupported_contact_image_wing_cannot_move():
    scene = SimpleNamespace(
        contact_frames=np.array([0, 10, 20, 30]),
        observation_frames=(np.array([1]), np.array([11, 19]), np.array([21])),
    )
    with pytest.raises(ValueError, match="unsupported original image wing"):
        probe.contact_image_groups(scene, 1)


def test_exported_downstream_drift_is_checked_even_with_frozen_launch_parameters():
    before = [{"positions": np.array([[0.0, 0, 0], [1.0, 1, 1]])} for _ in range(5)]
    after = [{"positions": row["positions"].copy()} for row in before]
    after[2]["positions"] += 100
    assert probe.unrelated_path_drift(before, after, [1, 2, 3]) == 0
    after[4]["positions"][1, 0] += 0.002
    assert probe.unrelated_path_drift(before, after, [1, 2, 3]) == pytest.approx(0.002)


def witness_context(**overrides):
    witness = dict(
        witness_xyz_m=[2.0, 3.0, 0.0325],
        supplied_frame=100.5,
        raw_distance_limit_m=0.95,
        graded_circle_radius_m=0.47,
        two_sided_reversal=True,
        covariance_resolved=True,
        timing_witness_resolved=True,
        serve_witness_covariance_resolved=True,
    )
    witness.update(overrides)
    return dict(
        repair_thresholds=SimpleNamespace(bounce_ray_limit_m=0.75, bounce_uncertainty_frames=2.0),
        repair_reference_verdict={"flights": [{"bounce_witness": [witness]}]},
    )


def test_feasibility_uses_raw_cap_even_when_covariance_circle_is_larger():
    plan = probe.frozen_bounce_plan(witness_context(), 0)
    assert plan[0]["maximum_distance_m"] == pytest.approx(0.9499)
    assert plan[0]["maximum_time_error_frames"] < 2
    # A wide graded circle must not turn a one-metre landing miss into feasible.
    assert probe.bounce_feasibility_slack([dict(x=[3.0, 3.0, 0.0325], frame=100.5)], plan)[0] < 0
    assert (
        np.min(probe.bounce_feasibility_slack([dict(x=[2.5, 3.0, 0.0325], frame=101.5)], plan)) >= 0
    )


def test_feasibility_cannot_move_or_replace_frozen_witness():
    context = witness_context()
    before = context["repair_reference_verdict"]["flights"][0]["bounce_witness"][0].copy()
    plan = probe.frozen_bounce_plan(context, 0)
    assert context["repair_reference_verdict"]["flights"][0]["bounce_witness"][0] == before
    assert plan[0]["event_frame"] == 100.5
    assert plan[0]["xy_m"] == [2.0, 3.0]


def test_missing_bounce_or_wrong_epoch_is_infeasible():
    plan = probe.frozen_bounce_plan(witness_context(), 0)
    assert np.all(probe.bounce_feasibility_slack([], plan) < 0)
    assert probe.bounce_feasibility_slack([dict(x=[2.0, 3.0, 0.0325], frame=103.0)], plan)[1] < 0


@pytest.mark.parametrize(
    "field",
    [
        "two_sided_reversal",
        "covariance_resolved",
        "timing_witness_resolved",
        "serve_witness_covariance_resolved",
    ],
)
def test_unresolved_witness_cannot_become_an_optimizer_target(field):
    with pytest.raises(ValueError, match="resolved frozen two-sided"):
        probe.frozen_bounce_plan(witness_context(**{field: False}), 0)


def test_neighbor_preserves_existing_accepted_chord_but_cannot_invent_one():
    context = witness_context()
    flight = context["repair_reference_verdict"]["flights"][0]
    flight.update(accepted=True, role="mid_rally")
    witness = flight["bounce_witness"][0]
    witness.update(
        from_terminal_completion=False,
        reversal_agrees=False,
        legacy_agrees=True,
        legacy_subframe_witness={"xyz_m": [8.0, 9.0, 0.0325]},
        legacy_graded_circle_radius_m=0.2,
    )
    plan = probe.accepted_neighbor_bounce_plan(context, 0)
    assert plan[0]["xy_m"] == [8.0, 9.0]
    assert plan[0]["maximum_distance_m"] == pytest.approx(0.9499)
    assert plan[0]["witness_route"] == "already_accepted_frozen_legacy"
    witness.pop("legacy_subframe_witness")
    with pytest.raises(ValueError, match="no reproducible frozen bounce route"):
        probe.accepted_neighbor_bounce_plan(context, 0)


def test_neighbor_serve_keeps_its_tighter_original_ray_limit():
    context = witness_context()
    context["repair_thresholds"].serve_bounce_ray_limit_m = 0.5
    flight = context["repair_reference_verdict"]["flights"][0]
    flight.update(accepted=True, role="serve")
    flight["bounce_witness"][0].update(
        from_terminal_completion=False,
        reversal_agrees=False,
        legacy_agrees=True,
        legacy_subframe_witness={"xyz_m": [8.0, 9.0, 0.0325]},
        legacy_graded_circle_radius_m=0.2,
    )
    assert probe.accepted_neighbor_bounce_plan(context, 0)[0][
        "maximum_distance_m"
    ] == pytest.approx(0.6999)


def test_window_plan_freezes_native_membership_and_permitted_shift():
    scene = SimpleNamespace(observation_frames=(np.array([1.0, 2.0, 3.0]),))
    threshold = SimpleNamespace(
        directional_rms_limit_px=32.0,
        directional_time_shift_frames=1.0,
        flight_reprojection_rms_limit_px=32.0,
    )
    window = dict(
        horizon="short",
        event_frame=4.0,
        direction="backward",
        native_training_pictures=3,
        zero_shift_rms_px=45.0,
        selected_time_shift_frames=-0.5,
    )
    context = dict(
        scene=scene,
        repair_thresholds=threshold,
        repair_reference_verdict={"flights": [{"directional_windows": [window]}]},
    )
    plan = probe.block_window_plan(context, [0])
    assert plan[0]["indices"] == [0, 1, 2]
    assert plan[0]["shift_frames"] == -0.5
    assert plan[0]["rms_limit_px"] < 32
    window["native_training_pictures"] = 2
    with pytest.raises(ValueError, match="membership failed"):
        probe.block_window_plan(context, [0])
