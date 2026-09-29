"""Pytest configuration: keep the default suite fast, keep the slow suite reachable.

The full suite is about 1,400 tests and was believed to hang. It does not — it is dominated by a
handful of physics fits that run real optimisations: measured 42 s
(`test_ball_events.py`), 64 s (`test_phys_fit_rally.py`) and >10 min
(`test_phys_fit_segment.py`). With `pytest -q` buffering its output until the end, a run
that is merely grinding is indistinguishable from one that is stuck, so nobody had a green
signal for the whole suite.

Default runs now skip anything marked `slow`, which makes the ordinary feedback loop usable.
Pass `--runslow` to include them, and do that before any promotion or release — a fast suite
that silently drops the physics tests would be worse than a slow one.

    uv run pytest -q                     # fast: everything except the marked physics fits
    uv run pytest -q --runslow           # everything
    uv run pytest -q -m slow --runslow   # only the slow physics fits

Environment note: run `uv sync --extra cv --extra dev` first. Invoking pytest through a
transient `uv run --with pytest` environment omits scikit-learn and joblib, which surfaces
as import errors in ~10 unrelated test modules and looks like a broken tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow (real physics optimisations; minutes, not seconds)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: runs a real optimisation or fit; excluded unless --runslow is given",
    )


#: Tests that read data this public tree does not ship; skipped while that data is absent.
DATA_TESTS = {'model weights (docs/MODELS.md)': ('$TENNIS_DATA_ROOT/models/pipeline/yolov8m.pt',
                                    ('cv/pipeline/test_canonical_runner.py::test_base_runner_emits_player_tracks_with_a_coordinate_contract',
                                     'cv/pipeline/test_canonical_runner.py::test_camera_mask_policy_reaches_court_command_and_receipt',
                                     'cv/pipeline/test_canonical_runner.py::test_canonical_runner_omits_legacy_contact_and_event_exports',
                                     'cv/pipeline/test_canonical_runner.py::test_court_worker_count_reaches_command_and_provenance',
                                     'cv/pipeline/test_canonical_runner.py::test_default_profile_allows_point_scoped_cadence_holds',
                                     'cv/pipeline/test_canonical_runner.py::test_guard_request_is_preserved_when_camera_admission_produces_no_clips',
                                     'cv/pipeline/test_canonical_runner.py::test_interpolation_policy_reaches_camera_command_and_receipt',
                                     'cv/pipeline/test_canonical_runner.py::test_keep_unique_admissible_frames_is_on_by_default',
                                     'cv/pipeline/test_canonical_runner.py::test_keep_unique_admissible_frames_reaches_the_association_command',
                                     'cv/pipeline/test_canonical_runner.py::test_keep_unique_off_moves_the_association_stage_identity',
                                     'cv/pipeline/test_canonical_runner.py::test_lost_revival_body_history_reaches_the_association_command',
                                     'cv/pipeline/test_canonical_runner.py::test_motion_profile_allows_only_point_scoped_cadence_holds',
                                     'cv/pipeline/test_canonical_runner.py::test_overlap_fragment_recovery_moves_the_association_stage_identity',
                                     'cv/pipeline/test_canonical_runner.py::test_overlap_fragment_recovery_reaches_the_association_command',
                                     'cv/pipeline/test_canonical_runner.py::test_overlap_recovery_request_is_preserved_when_camera_admission_produces_no_clips',
                                     'cv/pipeline/test_canonical_runner.py::test_runner_camera_bundle_is_explicit_and_recorded',
                                     'cv/pipeline/test_canonical_runner.py::test_runner_default_threads_argmax_ball_decode',
                                     'cv/pipeline/test_canonical_runner.py::test_runner_threads_named_subpixel_alternative_and_records_provenance',
                                     'cv/pipeline/test_canonical_runner.py::test_shot_homography_propagation_is_off_by_default',
                                     'cv/pipeline/test_canonical_runner.py::test_shot_homography_propagation_reaches_the_association_command',
                                     'cv/pipeline/test_canonical_runner.py::test_surface_witness_reaches_court_command_and_provenance')),
 'a fixture derived from development broadcasts': ('cv/experiments/connected_shooting/fixtures/observed_horizon_event_boundary.json',
                                                   ('cv/experiments/connected_shooting/test_observed_horizon_event_boundary.py::test_qualified_ground_after_original_net_reaches_actual_scene',
                                                    'cv/experiments/connected_shooting/test_observed_horizon_event_boundary.py::test_unknown_horizon_does_not_admit_later_context_events')),
 'an operations script': ('cv/validation/wk3_nightly.py',
                          ('cv/validation/test_s6_point_bench.py::test_nightly_argument_contract_is_the_shipped_one',)),
 'development labels (see docs/EVALUATION.md)': ('cv/validation/labels',
                                                 ('cv/experiments/connected_shooting/test_cohort.py::test_every_complete_attempt_is_found_once',
                                                  'cv/experiments/connected_shooting/test_cohort.py::test_the_cohort_manifest_is_serialisable',
                                                  'cv/experiments/connected_shooting/test_cohort.py::test_the_frozen_seventeen_keep_their_published_configuration',
                                                  'cv/experiments/connected_shooting/test_labeled_attempt_sweep.py::test_all_discovered_attempts_have_toss_or_dense_prior_evidence',
                                                  'cv/experiments/connected_shooting/test_labeled_attempt_sweep.py::test_every_cohort_case_names_a_present_frozen_label',
                                                  'cv/experiments/connected_shooting/test_labeled_attempt_sweep.py::test_regional_v2_is_a_grass_only_hard_centred_broadcast_model',
                                                  'cv/validation/test_ball_streak_reference.py::test_attempt_comparison_keeps_missing_owner_frames_in_denominator',
                                                  'cv/validation/test_ball_streak_reference.py::test_complete_additive_reference_and_immutable_owner',
                                                  'cv/validation/test_ball_streak_reference.py::test_frozen_early_serve_reference_and_opened_owner_agreement',
                                                  'cv/validation/test_ball_streak_reference.py::test_frozen_reference_checksum_and_opened_owner_agreement',
                                                  'cv/validation/test_ball_streak_reference.py::test_owner_comparison_requires_freeze_before_opening_owner',
                                                  'cv/validation/test_ball_streak_reference.py::test_reject_malformed_reference',
                                                  'cv/validation/test_flight_gate_audit.py::test_owner_position_index_is_native_and_nonempty',
                                                  'cv/validation/test_s6_owner_court_audit.py::test_original_owner_export_bytes_remain_immutable',
                                                  'cv/validation/test_s6_owner_inputs.py::test_agent_intake_preserves_annotator_events_streaks_and_source_pts',
                                                  'cv/validation/test_s6_owner_inputs.py::test_agent_intake_rejects_corrupted_evidence',
                                                  'cv/validation/test_s6_owner_inputs.py::test_speed_crop_must_match_native_pixels_even_with_updated_png_digest',
                                                  'cv/validation/test_s6_owner_inputs.py::test_speed_is_separate_preserved_evidence',
                                                  'cv/validation/test_s6_owner_inputs.py::test_speed_rejects_unbound_stale_or_inconsistent_witness',
                                                  'cv/validation/test_score_contact_strikers.py::test_derived_truth_is_only_used_when_the_labels_name_no_hitter',
                                                  'cv/validation/test_score_contact_strikers.py::test_labelled_truth_is_read_from_the_hitter_fields_not_from_alternation'))}


def _data_missing(path: str) -> bool:
    root = os.environ.get("TENNIS_DATA_ROOT", "data")
    here = Path(__file__).resolve().parent
    return not (here / path.replace("$TENNIS_DATA_ROOT", root)).exists()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for missing, (path, nodeids) in DATA_TESTS.items():
            if item.nodeid.split("[")[0] in nodeids and _data_missing(path):
                item.add_marker(pytest.mark.skip(reason=f"needs {missing} not in this tree"))
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; pass --runslow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
