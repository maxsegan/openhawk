"""Input-only net qualification, response-scoped same-context output and restoration."""

from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_net_recipe as net


def context(net_frame=20, status="labeled"):
    return dict(
        scene=SimpleNamespace(contact_frames=np.array([10, 40]), pixels=[[]]),
        bounces=[np.array([30])],
        events=[
            dict(
                event_type="net_hit",
                frame=net_frame,
                frame_interval=[net_frame - 1, net_frame + 1],
                status=status,
            ),
            dict(event_type="bounce", frame=30, frame_interval=[29, 31]),
        ],
    )


def test_requires_original_terminal_net_and_later_ground():
    assert net.qualify(context())["terminal_flight"] == 0
    with pytest.raises(ValueError, match="resolved net membership"):
        net.qualify(context(status="ambiguous"))
    with pytest.raises(ValueError, match="terminal-flight net"):
        net.qualify(context(net_frame=5))
    c = context()
    c["events"] = c["events"][:1]
    with pytest.raises(ValueError, match="separate arm"):
        net.qualify(c)
    c = context()
    c["bounces"] = [np.array([30, 35])]
    c["events"].append(dict(event_type="bounce", frame=35, frame_interval=[34, 36]))
    assert [e["frame"] for e in net.qualify(c)["ground_events"]] == [30, 35]
    c["bounces"] = [np.array([30])]
    with pytest.raises(ValueError, match="matching the physical inventory"):
        net.qualify(c)


def test_two_ground_events_cannot_overlap_or_have_unresolved_membership():
    c = context()
    c["bounces"] = [np.array([30, 35])]
    c["events"].append(dict(event_type="bounce", frame=35, frame_interval=[31, 36]))
    with pytest.raises(ValueError, match="ordered ground"):
        net.qualify(c)
    c["events"][-1]["frame_interval"] = [34, 36]
    c["events"][-1]["status"] = "ambiguous"
    with pytest.raises(ValueError, match="resolved ordered ground"):
        net.qualify(c)


def test_ground_witness_does_not_cross_another_ground_interval():
    grounds = [dict(frame=30, frame_interval=[29, 31]), dict(frame=35, frame_interval=[34, 36])]
    frames = list(range(22, 40))
    first, a = net.ground_bounded_frames(grounds[0], frames, grounds)
    second, b = net.ground_bounded_frames(grounds[1], frames, grounds)
    assert first == list(range(22, 34))
    assert second == list(range(32, 40))
    assert a["next_ground_lower_frame"] == 34
    assert b["previous_ground_upper_frame"] == 31


@pytest.mark.parametrize("scoped", [False, True])
def test_preparation_preserves_centers_and_distinguishes_unknown_from_legacy_radius(
    monkeypatch, scoped
):
    c = context()
    c["attempt"] = dict(point_clip="pt")
    if scoped:
        c["attempt"]["observation_scope"] = dict(observation_horizon_frame=40)
    c["targets"] = [[dict(event_frame=30)]]
    c["scene"].net_hit_frames = [np.array([20])]
    c["scene"].observation_frames = [np.array([29, 30, 31])]
    c["heldout"] = SimpleNamespace(observation_frames=[np.array([])])
    records = [
        dict(
            frame=29,
            status="visible",
            x1080=101.5,
            y1080=202.5,
            uncertainty_radius_px1080=None,
            observation_semantics="detector_heatmap_nominal_centre",
        ),
        dict(frame=30, status="visible", x1080=103.5, y1080=204.5),
        dict(frame=31, status="visible", x1080=105.5, y1080=206.5, uncertainty_radius_px1080=3.5),
    ]
    labels = dict(ball=dict(records=[dict(clip="pt", frames=records)]))
    before = deepcopy(labels)
    if not scoped:
        monkeypatch.setattr(net.census, "extended_context", lambda *args: (c, {}))
    monkeypatch.setattr(net.original.followup, "fit_check_copy", lambda *args: ("checks", []))
    seen = []

    def target(event, cameras, pixels, radii, **kwargs):
        assert radii == {30: 2.0, 31: 3.5}
        for row in records:
            np.testing.assert_array_equal(pixels[row["frame"]], [row["x1080"], row["y1080"]])
        seen.append(event)
        return dict(event_frame=30, witnessed=True)

    monkeypatch.setattr(net.original.whole, "event_ground_target", target)
    bundle = dict(context=c, labels=labels, cameras=dict(cameras=[]), search={}, duration=None)
    result = net.prepare(bundle)
    assert len(seen) == 1
    assert result["context"]["targets"] == [[dict(event_frame=30, witnessed=True)]]
    assert bundle["duration"] is None
    assert labels == before
    assert result["context"]["scene"] is c["scene"]
    assert result["context"]["heldout"] is c["heldout"]
    assert result["context"]["events"] is c["events"]
    if scoped:
        assert result["context_receipt"]["status"] == "original_observation_horizon_preserved"
        assert result["context"]["attempt"]["observation_scope"] == dict(
            observation_horizon_frame=40
        )


def test_unscoped_unsupported_net_context_still_abstains(monkeypatch):
    c = context()
    c["attempt"] = dict(point_clip="pt")
    monkeypatch.setattr(
        net.census, "extended_context", lambda *args: (None, {"status": "missing_native_tail"})
    )
    with pytest.raises(net.UnsupportedNet, match="missing_native_tail"):
        net.prepare(dict(context=c, labels={}, cameras={}, search={}))


def test_measurement_and_verdict_use_same_final_context(monkeypatch):
    active = {"targets": "bounded", "scene": "active"}
    legacy = {"targets": "legacy"}
    seen = []

    def score(ctx, p, bundle):
        seen.append(ctx)
        return {"verdict_targets": ctx["targets"]}, {"measurement_targets": ctx["targets"]}

    @contextmanager
    def using_response(law):
        assert law == "explicit_response"
        yield

    monkeypatch.setattr(net, "score", score)
    monkeypatch.setattr(net.free_response, "response_from_record", lambda _: "explicit_response")
    monkeypatch.setattr(net.epoch.tape, "using_response", using_response)
    monkeypatch.setattr(net.recipe.full.model, "chain", lambda *_: [{"net_hits": []}])
    result = net.measure_result(dict(context=active, original_target_context=legacy), [], {}, {})
    assert seen == [active, legacy]
    assert (
        result["after"]["verdict_targets"]
        == result["measurement"]["measurement_targets"]
        == "bounded"
    )
    assert result["original_targets_diagnostic"]["measurement"]["measurement_targets"] == "legacy"


def test_continuous_ground_restores_shared_callable_after_failure(monkeypatch):
    def evaluate(*_args, **_kwargs):
        return [], np.array([1.0]), {}

    monkeypatch.setattr(net.epoch.interval.block.event_constraints, "evaluate", evaluate)
    monkeypatch.setattr(net.original.horizon, "apply", lambda *args: args[2:5])
    with pytest.raises(RuntimeError):
        with net.continuous_ground(context()):
            assert net.epoch.interval.block.event_constraints.evaluate is not evaluate
            raise RuntimeError("test worker exit")
    assert net.epoch.interval.block.event_constraints.evaluate is evaluate


def test_cli_serializes_path_configuration_and_preserves_setup_failure(tmp_path):
    args = [
        value
        for name in ["search-report", "labels", "packet", "cameras", "pose-csv", "output"]
        for value in ["--" + name, str(tmp_path / name)]
    ]
    net.main(args)
    result = json.loads((tmp_path / "output/report.json").read_text())
    assert result["status"] == "execution_failed"
    assert result["stage"] == "source_loading"
    assert isinstance(result["configuration"]["search_report"], str)


def reviewed_ground_context():
    from copy import deepcopy
    from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

    c = context()
    raw = dict(
        event_type="bounce",
        frame=30,
        frame_interval=[29, 31],
        status="ambiguous",
        clip="pt",
        note="hidden impact, both wings visible",
    )
    row = deepcopy(raw)
    row.update(
        occurrence_status="supported",
        timing_status="abstained_exact_epoch",
        occurrence_evidence=dict(
            reviewer="Astra",
            basis="native descent and rebound",
            source_label_sha256="a" * 64,
            original_event_sha256=occurrence.record_hash(raw),
            original_event=deepcopy(raw),
        ),
    )
    c["attempt"] = dict(point_clip="pt", agent_event_rows=[row])
    c["events"][-1] = occurrence.physical_record(row, occurrence.timing_admission(row))
    return c


def test_net_consumes_supported_ground_occurrence_without_claiming_exact_timing():
    from copy import deepcopy

    c = reviewed_ground_context()
    before = deepcopy(c["events"])
    q = net.qualify(c)
    assert q["ground_events"] == [before[-1]]
    assert q["ground_events"][0]["status"] == "ambiguous"
    assert q["ground_events"][0]["frame_interval"] == [29, 31]
    assert q["ground_events"][0]["exact_epoch_observed"] is False
    assert c["events"] == before


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "wrong_clip",
        "wrong_interval",
        "unresolved",
        "tampered_evidence",
        "contradictory_impact",
    ],
)
def test_net_refuses_unbound_or_unresolved_ground_occurrence(change):
    from copy import deepcopy

    c = reviewed_ground_context()
    row = c["attempt"]["agent_event_rows"][0]
    if change == "missing":
        c["attempt"]["agent_event_rows"] = []
    elif change == "duplicate":
        c["attempt"]["agent_event_rows"].append(deepcopy(row))
    elif change == "wrong_clip":
        row["clip"] = "other"
    elif change == "wrong_interval":
        row["frame_interval"] = [28, 31]
    elif change == "unresolved":
        row["occurrence_status"] = "ambiguous"
    elif change == "tampered_evidence":
        row["occurrence_evidence"]["original_event_sha256"] = "b" * 64
    else:
        c["events"][-1]["occurrence_status"] = "unsupported"
    with pytest.raises(ValueError):
        net.qualify(c)
