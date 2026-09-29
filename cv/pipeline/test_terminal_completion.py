import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cv.pipeline.anchor_first_fit import simulate_measured_bounce_knot
from cv.pipeline.terminal_completion import complete_second_bounce, completion_window
from cv.pipeline.trajectory_contract import terminal_coverage_report
from physics.bounce_reference import DWELL_SECONDS, court_bounce
from physics.flight import ground_impact_in_interval


def case():
    fps = 25.0
    node_frame = 20.37
    xyz, incoming, spin = np.array([5.2, 15.0, 0.0325]), np.array([0.5, -11.0, -4.5]), np.zeros(3)
    rebound = court_bounce(incoming, spin, "hard")
    impact = ground_impact_in_interval(
        xyz, rebound.velocity, rebound.spin, 0.2, 2.0, plane_z_m=0.0325
    )
    assert impact is not None
    exact = node_frame + (impact.time_seconds + DWELL_SECONDS) * fps
    observed = float(math.floor(exact))
    boundary = {
        "event_type": "point_end",
        "frame": observed,
        "point_end": {"source": "test_physical_ending", "termination_kind": "second_bounce"},
    }
    contact = {
        "frame": observed,
        "terminal": True,
        "terminal_cutoff": {
            "point_end": boundary,
            "forward_completion_window": completion_window(boundary, observed + 5),
        },
    }

    def state(frame, fps, surface):
        result = simulate_measured_bounce_knot(
            incoming, spin, 0.0, np.array([frame]), fps, surface, {"frame": node_frame, "x": xyz}
        )
        return result[0][0], result[1][0]

    fit = SimpleNamespace(
        obs_frames=np.arange(1, observed + 1),
        state=state,
        _bounce_node_split={
            "frame": node_frame,
            "xyz": xyz,
            "sampling_model": "measured_bounce_v1",
            "dwell_seconds": DWELL_SECONDS,
            "outgoing_velocity": rebound.velocity,
            "outgoing_spin": rebound.spin,
        },
    )
    return fit, contact, exact


def test_forward_root_preserves_existing_path_observations_and_declared_event():
    fit, contact, exact = case()
    before = copy.deepcopy(contact)
    observations = fit.obs_frames.copy()
    result = complete_second_bounce(fit, contact, 25.0, "hard")
    assert result["status"] == "completed"
    assert result["end_frame"] == pytest.approx(exact, abs=1e-7)
    assert result["end_xyz"][2] == pytest.approx(0.0325, abs=1e-7)
    assert result["bounce"]["v_in"][2] < 0
    assert result["bounce"]["v_out"][2] > 0
    assert 0 < result["time_shift_frames"] < 1
    assert result["fitted_observations_discarded"] == 0
    assert contact == before
    np.testing.assert_array_equal(fit.obs_frames, observations)


def test_earlier_impact_is_not_completed_without_postimpact_observation_model():
    fit, contact, exact = case()
    boundary = contact["terminal_cutoff"]["point_end"]
    boundary["frame"] = float(math.ceil(exact))
    contact["frame"] = boundary["frame"]
    contact["terminal_cutoff"]["forward_completion_window"] = completion_window(boundary, exact + 5)
    result = complete_second_bounce(fit, contact, 25.0, "hard")
    assert result["status"] == "abstain"
    assert result["reason"] == "no_descending_ground_root_in_window"


@pytest.mark.parametrize("kind", ["terminal_bounce", "net_stop", "out_of_view", None])
def test_non_second_bounce_cannot_create_a_completion_window(kind):
    assert (
        completion_window(
            {
                "event_type": "point_end",
                "frame": 40.0,
                "point_end": {"source": "test", "termination_kind": kind},
            },
            45.0,
        )
        is None
    )


def test_window_cannot_extend_active_span_or_treat_missing_context_as_permission():
    _, contact, _ = case()
    boundary = contact["terminal_cutoff"]["point_end"]
    frame = boundary["frame"]
    assert completion_window(boundary, frame) is None
    window = completion_window(boundary, frame + 0.1)
    assert window["bounds_frames"] == [frame, frame + 0.1]


def test_existing_observations_cannot_be_dropped_to_create_a_valid_ending():
    fit, contact, _ = case()
    fit.obs_frames = np.r_[fit.obs_frames, contact["frame"] + 1]
    result = complete_second_bounce(fit, contact, 25.0, "hard")
    assert result["reason"] == "would_discard_fitted_observations"


def test_unrecognized_or_inconsistent_sampling_model_abstains():
    fit, contact, _ = case()
    fit.state = lambda *args: (np.array([1, 2, 3]), np.array([0, 1, -2]))
    assert (
        complete_second_bounce(fit, contact, 25.0, "hard")["reason"]
        == "export_sampler_disagrees_with_ground_root"
    )
    fit._net_split = {"frame": 22}
    assert (
        complete_second_bounce(fit, contact, 25.0, "hard")["reason"]
        == "unsupported_incoming_physics"
    )


def test_oversized_window_is_not_trusted():
    fit, contact, _ = case()
    contact["terminal_cutoff"]["forward_completion_window"]["bounds_frames"][1] += 100
    assert (
        complete_second_bounce(fit, contact, 25.0, "hard")["reason"]
        == "invalid_ending_state_or_interval"
    )


def test_coverage_accepts_only_matching_completion_record_and_real_ground_state():
    fit, contact, _ = case()
    completed = complete_second_bounce(fit, contact, 25.0, "hard")
    compact = {
        "terminal_end": True,
        "end_frame": completed["end_frame"],
        "end_xyz": completed["end_xyz"],
        "terminal_completion": completed,
    }
    report = terminal_coverage_report([contact], [compact])
    assert report["valid"] is True
    assert report["ending_time_policy"] == "bounded_forward_completion"
    bad = copy.deepcopy(compact)
    bad["terminal_completion"]["window"]["bounds_frames"][1] += 10
    assert terminal_coverage_report([contact], [bad])["valid"] is False
    bad = copy.deepcopy(compact)
    bad["end_xyz"][2] += 0.09
    assert terminal_coverage_report([contact], [bad])["valid"] is False
    bad = copy.deepcopy(compact)
    bad["terminal_completion"]["fitted_observations_discarded"] = 1
    assert terminal_coverage_report([contact], [bad])["valid"] is False


def test_append_terminal_contact_only_adds_window_when_opted_in():
    from cv.pipeline.reconstruction import append_terminal_contacts

    fit, contact, _ = case()
    boundary = contact["terminal_cutoff"]["point_end"]
    original = [{"frame": 1.0, "side": "near", "span": 0}]
    track = {int(frame): np.array([1.0, 2.0]) for frame in fit.obs_frames}
    spans = [[1.0, boundary["frame"] + 5]]
    default = append_terminal_contacts(original, track, spans, 25.0, boundary_events=[boundary])
    armed = append_terminal_contacts(
        original, track, spans, 25.0, boundary_events=[boundary], forward_ground_completion=True
    )
    assert default[-1]["terminal_cutoff"]["forward_completion_window"] is None
    assert (
        armed[-1]["terminal_cutoff"]["forward_completion_window"]["observed_frame"]
        == boundary["frame"]
    )
