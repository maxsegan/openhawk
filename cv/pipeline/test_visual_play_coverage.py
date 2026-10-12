"""Calibration uncertainty must not masquerade as a proven visual camera cut."""

from copy import deepcopy
from pathlib import Path

import pytest

from cv.pipeline.active_play import apply_leakage_trims
from cv.pipeline.broadcast_runner import point_gate_commands
from cv.pipeline.play_camera_leakage import FrameEvidence, visual_play_support


def packet():
    run = dict(start_frame=11, end_frame=34, reasons=["court_support_lost"])
    evidence = [
        FrameEvidence(f, 0.02, 40, True, 1.6, 1.5, True, True, False) for f in range(11, 35)
    ]
    run["visual_play_support"] = visual_play_support(run, evidence)
    entry = dict(
        active_spans=[[4, 273]],
        event_spans=[[1, 273]],
        fps=25,
        n_frames=275,
        point_valid=True,
        phase_reason="ok",
        shots=[dict(start_frame=1, end_frame=273, is_play_camera=True)],
    )
    return entry, dict(proposed_trims=[run]), evidence


def test_uniform_optional_policy_retains_pictures_not_camera_or_event_truth():
    entry, proposal, _ = packet()
    original = deepcopy((entry, proposal))
    off = apply_leakage_trims(entry, proposal)
    on = apply_leakage_trims(entry, proposal, court_loss_policy="preserve_supported_play")
    assert off["active_spans"] == [[4, 10], [35, 273]]
    assert off["event_spans"] == [[1, 10], [35, 273]]
    assert on["active_spans"] == [[4, 273]]
    assert on["event_spans"] == [[1, 273]]
    receipt = on["leakage_trim"]["visual_scope_policy"]
    assert receipt["competitive_phase_inferred"] is False
    assert (
        receipt["retained_court_only_runs"][0]["visual_play_support"]["camera_reliability_upgraded"]
        is False
    )
    assert (entry, proposal) == original


@pytest.mark.parametrize(
    "control", ["cut", "boundary", "nonplay", "invalid", "giant", "legacy", "wrong_interval"]
)
def test_known_visual_exclusions_and_unbound_legacy_evidence_still_trim(control):
    entry, proposal, _ = packet()
    run = proposal["proposed_trims"][0]
    if control == "cut":
        entry["shots"] = [
            dict(start_frame=1, end_frame=20, is_play_camera=True),
            dict(start_frame=24, end_frame=273, is_play_camera=True),
        ]
    elif control == "boundary":
        entry["shots"][0]["start_frame"] = 11
    elif control == "nonplay":
        entry["shots"][0]["is_play_camera"] = False
    elif control == "invalid":
        entry["point_valid"] = False
    elif control == "giant":
        run["reasons"].append("giant_player_box")
    elif control == "legacy":
        del run["visual_play_support"]
    else:
        run["visual_play_support"]["start_frame"] = 12
    result = apply_leakage_trims(entry, proposal, court_loss_policy="preserve_supported_play")
    assert result["event_spans"] == [[1, 10], [35, 273]]
    assert not result["leakage_trim"]["visual_scope_policy"]["retained_court_only_runs"]


@pytest.mark.parametrize(
    "control", ["missing", "unreadable", "one_player", "scale", "side_conflict", "giant"]
)
def test_native_actor_support_requires_observations_and_rejects_conflicts(control):
    _, proposal, evidence = packet()
    if control == "missing":
        evidence = evidence[1:]
    else:
        for row in evidence:
            if control == "unreadable":
                row.frame_readable = False
            elif control == "one_player":
                row.near_present = False
            elif control == "scale":
                row.apparent_near_m = 0.4
            elif control == "side_conflict":
                row.side_conflict = True
            else:
                row.apparent_far_m = 3.5
    assert visual_play_support(proposal["proposed_trims"][0], evidence)["supported"] is False


def test_phase_and_afterplay_exclusions_are_not_expanded_by_visual_support():
    entry, proposal, _ = packet()
    entry["active_spans"] = [[40, 273]]
    entry["event_spans"] = [[35, 273]]
    result = apply_leakage_trims(entry, proposal, court_loss_policy="preserve_supported_play")
    assert result["event_spans"] == [[35, 273]]
    assert result["active_spans"] == [[40, 273]]


def test_normal_runner_reaches_same_optional_active_play_consumer():
    commands = point_gate_commands(
        "python", Path("out"), Path("manifest"), court_loss_policy="preserve_supported_play"
    )
    active = commands[0]
    assert active[active.index("--court-loss-policy") + 1] == "preserve_supported_play"
    default = point_gate_commands("python", Path("out"), Path("manifest"))[0]
    assert default[default.index("--court-loss-policy") + 1] == "strict"


def test_reused_proposal_positive_visual_claim_is_remeasured(monkeypatch, tmp_path):
    from cv.pipeline import play_camera_leakage as leakage

    _, proposal, evidence = packet()
    for row in evidence:
        row.near_present = False
    captured = []

    def measured(match_dir, clip, frames, boxes, camera, *, measure_court_lines):
        captured.append((match_dir, clip, frames, measure_court_lines))
        return evidence

    monkeypatch.setattr(leakage, "evidence_for_clip", measured)
    inputs = leakage.MatchLeakageInputs(tmp_path, {}, {}, [], None)
    result = leakage.enrich_visual_support(tmp_path, "pt0001", proposal, inputs=inputs)
    assert proposal["proposed_trims"][0]["visual_play_support"]["supported"] is True
    assert result["proposed_trims"][0]["visual_play_support"]["supported"] is False
    assert captured == [(tmp_path, "pt0001", list(range(11, 35)), False)]
    with pytest.raises(ValueError, match="different match"):
        leakage.enrich_visual_support(tmp_path / "other", "pt0001", proposal, inputs=inputs)


def test_visual_only_evidence_reads_pixels_without_static_line_sweep(monkeypatch, tmp_path):
    import cv2
    import numpy as np
    from cv.pipeline import play_camera_leakage as leakage

    native = tmp_path / "audit_frames_native_1080/pt0001"
    native.mkdir(parents=True)
    cv2.imwrite(str(native / "f_0001.jpg"), np.zeros((64, 64), dtype=np.uint8))
    monkeypatch.setattr(leakage, "court_support", lambda *_: pytest.fail("unrequested line sweep"))
    camera = leakage.CameraEvidence(homography=np.eye(3))
    result = leakage.evidence_for_clip(
        tmp_path, "pt0001", [1, 2], {"near": {}, "far": {}}, camera, measure_court_lines=False
    )
    assert result[0].frame_readable is True
    assert result[0].court_support is None
    assert result[1].frame_readable is False
