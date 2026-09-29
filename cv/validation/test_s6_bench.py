from __future__ import annotations

import numpy as np
import pytest

from cv.validation.s6_bench import (
    EmissionRealismSamplers,
    EmpiricalSamplers,
    _side_from_homography,
    _striker_ledger,
    _contiguous_runs,
    evaluate_thresholds,
    proposed_thresholds,
    summarize_rows,
)


def _noise_payload() -> dict:
    return {
        "track_error": {
            "sequences": [
                {
                    "case_id": "case",
                    "samples": [
                        {
                            "side": "near",
                            "contact_adjacent": False,
                            "dx_px": 1.0,
                            "dy_px": 2.0,
                            "error_px": 2.24,
                            "innovation_cov_trace_native": 4.0,
                        },
                        {
                            "side": "near",
                            "contact_adjacent": False,
                            "dx_px": 2.0,
                            "dy_px": 3.0,
                            "error_px": 3.61,
                            "innovation_cov_trace_native": 9.0,
                        },
                        {
                            "side": "near",
                            "contact_adjacent": False,
                            "dx_px": 110.0,
                            "dy_px": 0.0,
                            "error_px": 110.0,
                            "innovation_cov_trace_native": 400.0,
                        },
                    ],
                },
                {
                    "case_id": "far-case",
                    "samples": [
                        {
                            "side": "far",
                            "contact_adjacent": False,
                            "dx_px": -7.0,
                            "dy_px": 5.0,
                            "error_px": 8.60,
                            "innovation_cov_trace_native": 25.0,
                        }
                    ],
                },
            ]
        },
        "wrong_object_and_dropout": {
            "wrong_object_arcs": [
                {"duration_frames": 2, "detour_offsets_px": [[10.0, 0.0], [11.0, 1.0]]}
            ],
            "wrong_object_arc_start_rate_per_frame": 1.0,
            "dropout_run_lengths": [2],
            "dropout_start_rate_per_frame": 1.0,
        },
        "camera_registration": {
            "registration_residual_vectors_px": [[1.0, -1.0]],
            "fallback_residual_vectors_px": [[5.0, -5.0]],
        },
        "event_timing": {"signed_error_frames": [-0.5, 0.0, 0.5]},
        "anchor_availability": {"patterns": [[], ["net_crossing", "bounce"]]},
    }


def test_contiguous_runs_preserves_lengths() -> None:
    assert _contiguous_runs([5, 2, 3, 9, 10, 11]) == [[2, 3], [5], [9, 10, 11]]


def test_empirical_samplers_draw_only_measured_values() -> None:
    sampler = EmpiricalSamplers(_noise_payload())
    rng = np.random.default_rng(4)
    errors, covariance = sampler.sample_track_errors(rng, ["near", "near"])
    assert {tuple(row) for row in errors}.issubset({(1.0, 2.0), (2.0, 3.0), (110.0, 0.0)})
    assert set(covariance).issubset({4.0, 9.0, 400.0})
    tail_draws = [sampler.sample_track_errors(rng, ["near"])[0][0, 0] for _ in range(20)]
    assert 110.0 in tail_draws
    side_errors, _ = sampler.sample_track_errors(rng, ["near", "far"])
    assert tuple(side_errors[1]) == (-7.0, 5.0)
    assert tuple(sampler.sample_camera_jitter(rng, [False, True])[0]) == (1.0, -1.0)
    assert tuple(sampler.sample_camera_jitter(rng, [False, True])[1]) == (5.0, -5.0)
    assert set(sampler.sample_event_timing(rng, 20)).issubset({-0.5, 0.0, 0.5})


def test_summary_includes_aligned_shared_contact_gap() -> None:
    rows = [
        {
            "solved": True,
            "accepted": True,
            "held_out_median_px": 1.0,
            "held_out_p90_px": 2.0,
            "bounce_position_error_m": 0.0,
            "contact_position_error_m": 0.1,
            "spin_identifiable": True,
            "spin_error_rpm": 20.0,
            "point": "p",
            "start_frame": 0.0,
            "end_frame": 10.0,
            "start_position_error_xyz": [0.0, 0.0, 0.0],
            "end_position_error_xyz": [0.1, 0.0, 0.0],
            "net_anchor_supplied": True,
            "bounce_anchor_supplied": True,
        },
        {
            "solved": True,
            "accepted": False,
            "held_out_median_px": 3.0,
            "held_out_p90_px": 5.0,
            "bounce_position_error_m": None,
            "contact_position_error_m": 0.3,
            "spin_identifiable": None,
            "spin_error_rpm": None,
            "point": "p",
            "start_frame": 10.0,
            "end_frame": 20.0,
            "start_position_error_xyz": [-0.1, 0.0, 0.0],
            "end_position_error_xyz": [0.0, 0.0, 0.0],
            "net_anchor_supplied": True,
            "bounce_anchor_supplied": False,
        },
    ]
    summary = summarize_rows("clean", "default_multistart", rows)
    assert summary["solved"] == 2
    assert summary["accepted"] == 1
    assert summary["held_out_median_px"] == 2.0
    assert summary["junction_pairs"] == 1
    assert np.isclose(summary["junction_gap_m"], 0.2)


def test_threshold_gate_scales_small_smoke_runs() -> None:
    limits = proposed_thresholds(10)
    assert limits["clean"]["minimum_accepted"] == 10
    table = [
        {
            "rung": "clean",
            "lever": "default_multistart",
            "solved": 10,
            "accepted": 10,
            "held_out_median_px": 1.0,
            "held_out_p90_px": 2.0,
        }
    ]
    assert evaluate_thresholds(table, 10)["passed"]


def _emission_realism_payload() -> dict:
    return {
        "schema": "s6_emission_realism_v1",
        "pixel_head": {
            "contact|near": {"count": 1, "offsets_px": [[10.0, -4.0]]},
            "contact|far": {"count": 1, "offsets_px": [[1.0, 1.0]]},
            "bounce|near": {"count": 1, "offsets_px": [[2.0, 0.0]]},
            "bounce|far": {"count": 0, "offsets_px": []},
        },
        "emitted_frame_offset": {
            "contact|near": {"count": 2, "offsets_frames": [0.5, 1.0]},
            "contact|far": {"count": 1, "offsets_frames": [0.0]},
            "bounce|near": {"count": 1, "offsets_frames": [-1.0]},
        },
        "off_track_frame": {
            "contact": {"count": 100, "off_track": 0, "rate": 0.0},
            "bounce": {"count": 100, "off_track": 100, "rate": 1.0},
        },
        "contact_track_gap": {
            "contacts": 100,
            "window_frames": 5,
            "with_gap_in_window": 100,
            "rate": 1.0,
            "missing_frames_in_window": [3],
        },
        "missing_bounce_emission": {"rate": 0.0, "terminal_rate": 1.0},
        "camera_abstention": {"points": 100, "without_camera": 0, "rate": 0.0},
        "striker": {"available": True, "wrong_body_rate": 1.0, "wrong_reach_m": [4.0]},
    }


def test_emission_realism_samplers_draw_only_measured_rows() -> None:
    sampler = EmissionRealismSamplers(_emission_realism_payload())
    rng = np.random.default_rng(3)
    assert tuple(sampler.sample_pixel_head(rng, "contact", "near")) == (10.0, -4.0)
    assert set(sampler.sample_frame_offset(rng, "contact", "near") for _ in range(20)) == {0.5, 1.0}
    # A cell with no measured row falls back to the same event type's other cells rather
    # than inventing a distribution.
    assert tuple(sampler.sample_pixel_head(rng, "bounce", "far")) == (2.0, 0.0)
    assert sampler.sample_frame_offset(rng, "bounce", "far") == -1.0
    assert sampler.allow_off_track_frame(rng, "contact") is False
    assert sampler.allow_off_track_frame(rng, "bounce") is True
    assert sampler.sample_contact_gap(rng) == 3
    assert sampler.drop_bounce_emission(rng) is False
    assert sampler.drop_bounce_emission(rng, terminal=True) is True
    assert sampler.camera_abstains(rng) is False
    assert sampler.striker_wrong_body(rng) == 4.0


def test_emission_realism_samplers_refuse_another_payload() -> None:
    with pytest.raises(ValueError):
        EmissionRealismSamplers({"schema": "s6_3d_bench_empirical_noise_v1"})


def test_the_striker_ledger_rate_is_over_the_contacts_that_have_a_box(tmp_path) -> None:
    path = tmp_path / "contacts.csv"
    path.write_text(
        "cause,striker_reach,striker_ray_reach_m\n"
        "ok,0.20,0.70\n"
        "ok,0.30,0.90\n"
        "wrong_player_same_side,2.10,8.00\n"
        "no_court_homography,,\n"
    )
    ledger = _striker_ledger(path)
    assert ledger["contacts_with_box"] == 3
    assert ledger["wrong_body"] == 1
    assert ledger["wrong_body_rate"] == pytest.approx(1.0 / 3.0)
    assert ledger["wrong_reach_m"] == [8.0]


def test_the_court_side_comes_from_the_homography() -> None:
    homography = np.eye(3)
    assert _side_from_homography(homography, (0.0, 1.0)) == "near"
    assert _side_from_homography(homography, (0.0, 20.0)) == "far"
    assert _side_from_homography(None, (0.0, 1.0)) == "unknown"
