"""Source-only 2D continuity must survive H holds without fabricating metric state."""

from copy import deepcopy
from pathlib import Path

import pytest

from cv.pipeline.native_actor_continuity import ActorInputs, NativeBox, bridge_gap, witnesses
from cv.pipeline.active_play import apply_leakage_trims
from cv.pipeline.broadcast_runner import point_gate_commands
from cv.pipeline import play_camera_leakage as leakage


def sample():
    raw, sided = {}, {}
    for f in range(1, 10):
        for side, y, track in [("near", 600, "1"), ("far", 200, "0")]:
            box = NativeBox((100 + f * 3, y, 160 + f * 3, y + 100), 0.9)
            raw.setdefault(("pt0001", f), []).append(box)
            if f in (1, 9):
                sided[("pt0001", f, side)] = [NativeBox(box.box, 0.9, track)]
    return ActorInputs(
        raw,
        sided,
        {"pt0001": {1, 9}},
        {"pt0001": set(range(1, 10))},
        [],
        {("pt0001", f): 10 + (f - 1) / 25 for f in range(1, 10)},
    )


def run(inputs, shots=None):
    return bridge_gap(
        inputs,
        "pt0001",
        2,
        8,
        fps=25,
        shots=shots or [dict(start_frame=1, end_frame=20, is_play_camera=True)],
    )


def test_pan_camera_gap_keeps_original_observed_boxes_and_unknown_metric_state():
    inputs = sample()
    original = deepcopy(inputs)
    result = run(inputs)
    assert result["supported"]
    assert len(result["observations"]) == 7
    assert result["court_coordinates"] is None
    assert not result["camera_reliability_upgraded"]
    assert not result["competitive_phase_inferred"]
    assert inputs == original
    for row in result["observations"]:
        for side in ("near", "far"):
            assert tuple(row[side + "_box"]) in {
                b.box for b in inputs.raw[("pt0001", row["frame"])]
            }


@pytest.mark.parametrize(
    "control",
    [
        "missing",
        "crowd",
        "wrong_anchor",
        "identity_swap",
        "cut",
        "nonplay",
        "missing_inventory",
        "reliable_interior",
        "giant",
        "wrong_clock",
    ],
)
def test_abstains_without_unique_native_continuity(control):
    inputs = sample()
    shots = None
    if control == "missing":
        del inputs.raw[("pt0001", 5)]
    elif control == "crowd":
        b = inputs.raw[("pt0001", 5)][0]
        inputs.raw[("pt0001", 5)].append(NativeBox((b.box[0] + 1, *b.box[1:]), 0.9))
    elif control == "wrong_anchor":
        inputs.sided[("pt0001", 9, "near")] = [NativeBox((600, 600, 660, 700), 0.9, "1")]
    elif control == "identity_swap":
        b = inputs.sided[("pt0001", 9, "near")][0]
        inputs.sided[("pt0001", 9, "near")] = [NativeBox(b.box, 0.9, "2")]
    elif control == "cut":
        shots = [
            dict(start_frame=1, end_frame=5, is_play_camera=True),
            dict(start_frame=6, end_frame=20, is_play_camera=True),
        ]
    elif control == "nonplay":
        shots = [dict(start_frame=1, end_frame=20, is_play_camera=False)]
    elif control == "missing_inventory":
        inputs.known["pt0001"].remove(5)
    elif control == "reliable_interior":
        inputs.reliable["pt0001"].add(5)
    elif control == "wrong_clock":
        inputs.times[("pt0001", 5)] += 0.08
    else:
        inputs.raw[("pt0001", 5)][0] = NativeBox((110, 400, 450, 1000), 0.9)
    assert not run(inputs, shots)["supported"]


def test_never_shortens_original_long_gap_to_requested_favorable_window():
    inputs = sample()
    inputs.known["pt0001"] = set(range(1, 50))
    inputs.reliable["pt0001"] = {1, 49}
    r = witnesses(
        inputs,
        "pt0001",
        {5, 6},
        shots=[dict(start_frame=1, end_frame=50, is_play_camera=True)],
        fps=25,
    )
    assert r[0]["start_frame"] == 2 and r[0]["end_frame"] == 48
    assert r[0]["reason"] == "gap_not_short_and_bounded"


def test_shared_coverage_consumer_uses_witness_without_making_metric_boxes(monkeypatch, tmp_path):
    inputs = sample()
    bound = leakage.MatchLeakageInputs(tmp_path, {}, {}, [], None, inputs)
    evidence = [
        leakage.FrameEvidence(f, None, 0, True, None, None, False, False, False)
        for f in range(2, 9)
    ]
    monkeypatch.setattr(leakage, "evidence_for_clip", lambda *a, **kw: evidence)
    entry = dict(
        fps=25,
        point_valid=True,
        phase_reason="ok",
        active_spans=[[1, 100]],
        event_spans=[[1, 100]],
        shots=[dict(start_frame=1, end_frame=20, is_play_camera=True)],
    )
    proposal = dict(
        proposed_trims=[dict(start_frame=2, end_frame=8, reasons=["court_support_lost"])]
    )
    enriched = leakage.enrich_visual_support(
        tmp_path, "pt0001", proposal, inputs=bound, entry=entry
    )
    support = enriched["proposed_trims"][0]["visual_play_support"]
    assert support["native_actor_continuity_frames"] == 7
    assert support["metric_scaled_actor_support_frames"] == 0
    assert all(e.apparent_near_m is None for e in evidence)
    on = apply_leakage_trims(entry, enriched, court_loss_policy="preserve_supported_play")
    assert on["event_spans"] == [[1, 100]]
    off = apply_leakage_trims(entry, enriched)
    assert off["event_spans"] == [[1, 1], [9, 100]]
    # Visibility remains required even with actual matched raw detections.
    for e in evidence:
        e.frame_readable = False
    assert not leakage.enrich_visual_support(
        tmp_path, "pt0001", proposal, inputs=bound, entry=entry
    )["proposed_trims"][0]["visual_play_support"]["supported"]


def test_normal_runner_flag_reaches_actual_gate_and_default_is_off():
    args = ("python", Path("output"), Path("manifest"))
    assert "--native-actor-continuity" not in point_gate_commands(*args)[0]
    cmd = point_gate_commands(
        *args, court_loss_policy="preserve_supported_play", native_actor_continuity=True
    )[0]
    assert "--native-actor-continuity" in cmd


@pytest.mark.parametrize("fps", [25.0, 30.0, 30000 / 1001, 50.0, 60.0])
def test_original_centisecond_rounding_is_compatible_with_native_cadence(fps):
    inputs = sample()
    inputs.times = {("pt0001", f): round(100.003 + (f - 1) / fps, 2) for f in range(1, 10)}
    result = bridge_gap(
        inputs,
        "pt0001",
        2,
        8,
        fps=fps,
        shots=[dict(start_frame=1, end_frame=20, is_play_camera=True)],
    )
    assert result["supported"]


def test_wrong_cadence_cannot_accumulate_individually_small_adjacent_errors():
    inputs = sample()  # original25Hz observations
    result = bridge_gap(
        inputs,
        "pt0001",
        2,
        8,
        fps=30,
        shots=[dict(start_frame=1, end_frame=20, is_play_camera=True)],
    )
    assert result["reason"] == "native_actor_clock_not_frame_adjacent"
