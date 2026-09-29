"""Source proposals reach the shared bounded topology consumer without firm-event mutation."""

from copy import deepcopy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cv.pipeline import event_paths, event_proposals, s6_optional_contacts as optional
from cv.pipeline import s6_independent_event_proposals as independent


def source_fixture(tmp_path):
    probabilities = np.asarray([[0.01, 0.97, 0.01, 0.01]] * 2)
    pred = tmp_path / "predictions.npz"
    np.savez(
        pred,
        probabilities=probabilities,
        frames=[10, 70],
        clips=["m__pt0001"] * 2,
        broadcasts=["m"] * 2,
        court_y=[0.1, 0.9],
    )
    rows = [
        dict(
            clip="m__pt0001",
            frame=f,
            candidate_frame=f,
            event_type="contact",
            abstain=False,
            class_probabilities=dict(
                zip(("none", "contact", "bounce", "net_hit"), probabilities[i])
            ),
        )
        for i, f in enumerate([10, 70])
    ]
    emissions = tmp_path / "events.json"
    emissions.write_text(json.dumps(rows))
    source = tmp_path / "proposals.json"
    proposal = event_proposals.EventProposal(
        "m__pt0001",
        40,
        38,
        42,
        ("contact",),
        "pose_swing",
        0.7,
        {"support_family": "pose", "predicted_x": 300, "predicted_y": 250},
    )
    source.write_text(json.dumps(event_proposals.proposal_document([proposal])))
    event_paths.build_packets(pred, emissions, source, tmp_path / "sidecars", evidence_only=True)
    inputs = SimpleNamespace(
        match_id="m",
        clip="pt0001",
        events=emissions,
        independent_event_candidates=tmp_path / "sidecars/m__pt0001.event_candidates.json",
        source_video=pred,
        source_pts=pred,
        players=pred,
        optional_contact_views=None,
    )
    events = [dict(event_type="contact", frame=f, frame_interval=[f, f]) for f in [10, 70]]
    labels = [dict(frame=f, status="visible", x1080=100 + f, y1080=200 + f) for f in range(10, 71)]
    attempt = dict(events=events, owner_ball_labels=labels)
    images = [dict(frame=f, native_pts_seconds=100 + f * 0.04) for f in range(10, 71)]
    cameras = dict(cameras=[dict(frame=f, status="supported") for f in range(10, 71)])
    return inputs, attempt, images, cameras


def test_independent_pose_reaches_actual_bounded_alternative_consumer(tmp_path):
    inputs, attempt, images, cameras = source_fixture(tmp_path)
    before = deepcopy((attempt, images, cameras))
    # No classifier crop or held emission exists at40. The original source cue survives.
    document = optional.build_witness(
        inputs, attempt, images, cameras, independent_event_proposals="on"
    )
    branches = optional.search_hypotheses(document, attempt)
    assert len(branches) == 1 and branches[0]["added"][0]["frame_interval"] == [38, 42]
    assert branches[0]["occurrence_log_odds"] == 0
    assert branches[0]["independent_event_proposal"]["source_proposal"]["source"] == "pose_swing"
    calls = []
    budget = SimpleNamespace(seconds=90, started=time.monotonic(), exhausted=False)

    def run(events, end, name):
        calls.append(dict(events=events, end=end, name=name, budget=budget.seconds))
        return dict(status="construction_reached")

    result, receipts = optional.fit_alternatives(run, branches, 70, budget, 90)
    assert len(result) == 1 and receipts[0]["status"] == "fitted"
    assert calls[0]["events"] == branches[0]["events"] and calls[0]["budget"] == 90
    assert budget.seconds == 90  # no extra clock or per-proposal budget
    assert (attempt, images, cameras) == before
    assert event_paths.load_evidence_packet(
        inputs.independent_event_candidates, inputs.events, "m__pt0001"
    )["firm"] == json.loads(inputs.events.read_text())


def test_off_is_exact_legacy_behavior(tmp_path):
    inputs, attempt, images, cameras = source_fixture(tmp_path)
    a = optional.build_witness(inputs, attempt, images, cameras)
    b = optional.build_witness(inputs, attempt, images, cameras, independent_event_proposals="off")
    assert a == b and independent.KEY not in a
    assert optional.search_hypotheses(a, attempt) == []


@pytest.mark.parametrize("missing", ["camera", "observations", "outside", "overlap"])
def test_source_candidate_is_retained_with_explicit_physical_admission_refusal(tmp_path, missing):
    inputs, attempt, images, cameras = source_fixture(tmp_path)
    if missing == "camera":
        cameras["cameras"][25]["status"] = "held"
    if missing == "observations":
        attempt["owner_ball_labels"] = []
    if missing == "outside":
        attempt["events"][0].update(frame=45, frame_interval=[45, 45])
    if missing == "overlap":
        attempt["events"].insert(1, dict(event_type="bounce", frame=40, frame_interval=[39, 41]))
    document = optional.build_witness(
        inputs, attempt, images, cameras, independent_event_proposals="on"
    )
    receipt = document[independent.KEY]
    assert len(receipt["inventory"]) == 1 and len(receipt["exclusions"]) == 1
    assert optional.search_hypotheses(document, attempt) == []


def test_independent_additions_cannot_expand_existing_alternative_cap(tmp_path):
    inputs, attempt, images, cameras = source_fixture(tmp_path)
    doc = optional.build_witness(inputs, attempt, images, cameras, independent_event_proposals="on")
    old = [dict(name="existing1"), dict(name="existing2")]
    assert independent.extend_hypotheses(doc, attempt["events"], old) == old


def test_research_handoff_preserves_assistance_and_all_original_channels(tmp_path):
    from cv.pipeline import provenance

    inputs, attempt, images, cameras = source_fixture(tmp_path)
    original = dict(
        attempts=[attempt],
        human_derived=True,
        automatic_inference_eligible=False,
        assistance="supplied clips and labeled ball; automatic event source",
        stream_origins=dict(ball="labeled", events="automatic"),
        native_images=images,
        cameras=cameras,
    )
    path = tmp_path / "original.json"
    path.write_text(json.dumps(original))
    witness = optional.build_witness(
        inputs, attempt, images, cameras, independent_event_proposals="on"
    )
    wp = tmp_path / "witness.json"
    wp.write_text(json.dumps(witness))
    packet = independent.bind_research_packet(path, wp)
    lookup = {provenance.file_record(p)["path"]: p for p in [path, wp]}

    def resolve(record):
        return lookup[record["path"]]

    independent.validate_research_packet(packet, witness, resolve)
    assert packet["human_derived"] and not packet["automatic_inference_eligible"]
    assert json.loads(path.read_text()) == original
    changed = deepcopy(packet)
    changed["attempts"][0]["owner_ball_labels"][0]["x1080"] += 1
    with pytest.raises(ValueError, match="changed original input channels"):
        independent.validate_research_packet(changed, witness, resolve)
    changed = deepcopy(packet)
    changed["automatic_inference_eligible"] = True
    with pytest.raises(ValueError, match="changed original input channels"):
        independent.validate_research_packet(changed, witness, resolve)
    with pytest.raises(ValueError, match="explicit research handoff"):
        independent.validate_research_packet(original, witness, resolve)


@pytest.mark.parametrize("tamper", [None, "event", "name", "family", "witness"])
def test_selected_source_contact_reaches_family_reader_and_retains_exact_proof(
    tmp_path, monkeypatch, tamper
):
    from cv.pipeline import provenance, s6_optional_event_union

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    inputs, attempt, images, cameras = source_fixture(tmp_path)
    before = deepcopy(attempt)
    document = optional.build_witness(
        inputs, attempt, images, cameras, independent_event_proposals="on"
    )
    hypothesis = optional.search_hypotheses(document, attempt)[0]
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(document))
    report = dict(
        events=deepcopy(hypothesis["events"]),
        optional_contact_selection=dict(
            selected_topology=hypothesis["name"],
            policy="source_witness_optional_contact_v1",
            witness_record=provenance.file_record(path),
        ),
    )
    if tamper == "event":
        report["events"][1]["frame"] += 1
    elif tamper == "name":
        report["optional_contact_selection"]["selected_topology"] = "independent_proposal_999"
    elif tamper == "family":
        report["optional_contact_selection"]["policy"] = "source_witness_optional_bounce_v1"
    elif tamper == "witness":
        path.write_text(path.read_text() + " ")
    if tamper is None:
        assert s6_optional_event_union.validate_report_events(attempt, report)
    else:
        with pytest.raises(ValueError):
            s6_optional_event_union.validate_report_events(attempt, report)
    assert attempt == before
