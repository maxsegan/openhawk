import itertools
import json

import pytest

from cv.pipeline import event_cascade as cascade
from cv.pipeline import event_cascade_rebind as rebind
from cv.pipeline.event_cascade import Answer

ROW = {"id": "a__3", "event_type": "bounce", "frame": 100.0, "marginal": 0.9}


def _lookup(replies: dict):
    return lambda tier, draw: replies.get((tier, draw))


def _settle(replies, *, contradicts=False, **chosen):
    return cascade.settle(ROW, _lookup(replies), contradicts=contradicts, chosen=chosen or None)


def test_tier_confirms_only_the_emitted_type_inside_the_window():
    confirm = [Answer("bounce", 101.0), Answer("bounce", 99.0)]
    assert (
        cascade.occurrence_decision(confirm, emitted_type="bounce", emitted_frame=100.0)
        == "confirm"
    )
    moved = [Answer("bounce", 103.0), Answer("bounce", 103.0)]
    assert (
        cascade.occurrence_decision(moved, emitted_type="bounce", emitted_frame=100.0)
        == "unsettled"
    )
    retitled = [Answer("contact", 100.0), Answer("contact", 100.0)]
    assert (
        cascade.occurrence_decision(retitled, emitted_type="bounce", emitted_frame=100.0)
        == "unsettled"
    )
    reject = [Answer("no_event", None), Answer("none", None)]
    assert (
        cascade.occurrence_decision(reject, emitted_type="bounce", emitted_frame=100.0) == "reject"
    )


def test_first_settling_tier_stops_the_climb_with_a_receipt_per_call():
    replies = {
        ("qwen", 1): Answer("contact", 100.0),
        ("qwen", 2): Answer("contact", 100.0),
        ("gemini", 1): Answer(
            "bounce", 100.0, usd=0.009, tokens={"prompt": 5000, "completion": 1300}
        ),
        ("gemini", 2): Answer(
            "bounce", 101.0, usd=0.008, tokens={"prompt": 5000, "completion": 1200}
        ),
    }
    walk = _settle(replies, second_draw="always", first_tier="qwen")
    assert (walk["owed"], walk["action"], walk["tier"]) == (None, "confirm", "gemini")
    assert walk["model"] == "google/gemini-3.8-flash"
    assert [(r["tier"], r["draw"], r["trigger"]) for r in walk["receipts"]] == [
        (1, 1, "enter"),
        (1, 2, "second_draw"),
        (2, 1, "tier_disagreement"),
        (2, 2, "second_draw"),
    ]
    assert walk["receipts"][2]["usd"] == 0.009 and walk["receipts"][2]["tokens"]["prompt"] == 5000


def test_first_tier_gemini_never_asks_qwen_and_enters_at_flash():
    replies = {
        ("gemini", 1): Answer("bounce", 100.0, usd=0.009),
        ("gemini", 2): Answer("bounce", 101.0, usd=0.008),
    }
    walk = _settle(replies, first_tier="gemini")
    assert (walk["owed"], walk["action"], walk["tier"]) == (None, "confirm", "gemini")
    assert [(r["tier"], r["draw"], r["trigger"]) for r in walk["receipts"]] == [
        (2, 1, "enter"),
        (2, 2, "second_draw"),
    ]
    assert _settle({}, first_tier="gemini")["owed"] == ("gemini", 1)
    walk = _settle(
        {("gemini", 1): Answer("contact", 100.0), ("opus", 1): Answer("none", None)},
        first_tier="gemini",
        contradicts=True,
    )
    assert (walk["action"], walk["tier"]) == ("unsettled", "opus")
    with pytest.raises(ValueError):
        cascade.settings({"first_tier": "opus"})


def test_second_draw_is_skipped_only_when_the_first_cannot_settle():
    replies = {("qwen", 1): Answer("contact", 100.0)}
    assert _settle(replies, second_draw="always", first_tier="qwen")["owed"] == ("qwen", 2)
    assert _settle(replies, second_draw="settle_only", first_tier="qwen")["owed"] == ("gemini", 1)
    assert _settle(
        {("qwen", 1): Answer("bounce", 100.0)}, second_draw="settle_only", first_tier="qwen"
    )["owed"] == (
        "qwen",
        2,
    )


def test_skipping_second_draws_never_changes_a_decision():
    # Every combination of a few answer kinds on every draw: the cheaper walk
    # reaches the same action and settling tier as the recorded one.
    kinds = [
        Answer("bounce", 100.0),
        Answer("bounce", 104.0),
        Answer("contact", 100.0),
        Answer("no_event", None),
        Answer("abstain", None),
    ]
    draws = [("qwen", 1), ("qwen", 2), ("gemini", 1), ("gemini", 2), ("opus", 1)]
    for combo in itertools.product(kinds, repeat=len(draws)):
        replies = dict(zip(draws, combo))
        for contradicts in (False, True):
            full = _settle(replies, contradicts=contradicts, second_draw="always")
            short = _settle(replies, contradicts=contradicts, second_draw="settle_only")
            assert (full["action"], full["tier"]) == (short["action"], short["tier"])
            assert len(short["receipts"]) <= len(full["receipts"])


def test_contradictory_reject_climbs_and_is_not_removed_after_opus():
    replies = {
        key: Answer("no_event", None)
        for key in [("qwen", 1), ("qwen", 2), ("gemini", 1), ("gemini", 2), ("opus", 1)]
    }
    assert _settle(replies)["action"] == "reject"
    walk = _settle(replies, contradicts=True)
    assert (walk["action"], walk["tier"]) == ("unsettled", "opus")
    assert walk["receipts"][2]["trigger"] == "contradictory_reject"


def test_track_contradiction_rules():
    reverse = dict(pre=(10.0, 0.0), post=(-10.0, 0.0))
    assert cascade.contradictory_reject(event_type="contact", marginal=0.96, **reverse)
    assert not cascade.contradictory_reject(event_type="contact", marginal=0.9, **reverse)
    assert cascade.contradictory_reject(
        event_type="bounce", marginal=None, pre=(2.0, 8.0), post=(2.0, -7.0)
    )
    assert not cascade.contradictory_reject(
        event_type="bounce", marginal=None, pre=(6.0, 8.0), post=(-6.0, -7.0)
    )


def test_switches_default_to_the_cheaper_policy_and_keep_caches_apart():
    assert cascade.settings() == {
        "second_draw": "settle_only",
        "opus_evidence": "1280",
        "first_tier": "gemini",
    }
    assert cascade.reply_profile("opus", cascade.settings()) == "zero@1280"
    assert cascade.reply_profile("opus", cascade.RECORDED_SETTINGS) == "zero"
    assert cascade.reply_profile("qwen", cascade.settings()) == "guided_nothink"
    with pytest.raises(ValueError):
        cascade.settings({"opus_evidence": "960"})


def test_opus_evidence_is_resized_on_both_grids(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from cv.pipeline import event_cascade_models as models

    images = []
    for name in ("clean", "trail"):
        path = tmp_path / "evidence" / "one" / "pergrid" / "clip" / name / "a__3.jpg"
        path.parent.mkdir(parents=True)
        cv2.imwrite(str(path), np.zeros((640, 640, 3), dtype=np.uint8))
        images.append({"path": str(path), "width": 640, "height": 640})
    nom = {"clip": "clip", "id": "a__3", "images": images}
    small = models.resized(nom, 320, tmp_path / "evidence" / "one")
    assert [cv2.imread(i["path"]).shape[:2] for i in small["images"]] == [(320, 320), (320, 320)]
    assert models.resized(nom, None, tmp_path)["images"] == images


def test_candidates_are_unsupported_rows_on_flights_the_plan_did_not_accept():
    ambiguous = {
        "event_type": "bounce",
        "occurrence_status": "ambiguous",
        "automatic_abstention": {"acceptance_marginal": 0.8},
    }
    plan = {
        "match_id": "m",
        "clip": "p",
        "original_slots": [
            {
                "original_contact_index": 0,
                "start_frame": 10.0,
                "end_frame": 50.0,
                "status": "prepared",
            },
            {"original_contact_index": 1, "start_frame": 50.0, "end_frame": 90.0, "status": "held"},
        ],
        "original_events": [dict(ambiguous, frame=30.0), dict(ambiguous, frame=70.0)],
    }
    rows = cascade.select_from_plan(plan, {("m__p", "bounce", 70.0): {"gate_held": False}})
    assert [(r["event_index"], r["places"], r["band"]) for r in rows] == [
        (1, ["inside:1"], "marginal_band")
    ]


def test_wider_scopes_ask_on_prepared_slots_the_fit_did_not_accept():
    ambiguous = {
        "event_type": "bounce",
        "occurrence_status": "ambiguous",
        "automatic_abstention": {"acceptance_marginal": 0.8},
    }
    slots = [
        {"original_contact_index": i, "start_frame": a, "end_frame": b, "status": "prepared"}
        for i, (a, b) in enumerate(((10.0, 50.0), (50.0, 90.0), (90.0, 130.0)))
    ]
    plan = {
        "match_id": "m",
        "clip": "p",
        "original_slots": slots,
        "original_events": [dict(ambiguous, frame=f) for f in (30.0, 70.0, 100.0, 120.0)],
    }
    verdict = {
        "flights": [
            {"original_contact_index": 0, "start_frame": 10.0, "end_frame": 50.0, "accepted": True},
            {
                "original_contact_index": 1,
                "start_frame": 50.0,
                "end_frame": 90.0,
                "accepted": False,
            },
            {
                "original_contact_index": 2,
                "start_frame": 90.0,
                "end_frame": 110.0,
                "accepted": True,
            },
        ]
    }

    def asked(scope):
        rows = cascade.select_from_plan(plan, scope=scope, verdict=verdict)
        return [row["frame"] for row in rows]

    assert asked("unresolved") == []
    assert asked("unresolved_or_rejected") == [70.0]
    assert asked("unresolved_or_uncovered") == [70.0, 120.0]
    with pytest.raises(ValueError, match="verdict"):
        cascade.select_from_plan(plan, scope="unresolved_or_rejected")


def test_settling_ignores_unsettled_and_refuses_duplicates():
    rows = [
        {"event_type": "bounce", "frame": 70.0, "action": "confirm"},
        {"event_type": "contact", "frame": 80.0, "action": "unsettled"},
    ]
    assert list(rebind.settling(rows)) == [("bounce", 70.0)]
    with pytest.raises(ValueError):
        rebind.settling(rows + [dict(rows[0])])


def test_rebind_refuses_a_packet_with_labelled_events():
    with pytest.raises(ValueError, match="automatic"):
        rebind.require_automatic({"stream_origins": {"ball": "automatic", "events": "labeled"}})
    rebind.require_automatic({"stream_origins": {"ball": "automatic", "events": "automatic"}})


def test_confirm_names_the_settling_tier_and_model():
    event = {
        "event_type": "bounce",
        "frame": 70.0,
        "occurrence_status": "ambiguous",
        "status": "predicted",
        "annotation_origin": "automatic",
        "exact_epoch_observed": False,
        "timing_status": "predicted",
        "frame_interval": [69.0, 71.0],
        "automatic_abstention": {"acceptance_marginal": 0.8},
    }
    row = rebind.confirm_event(event, {"action": "confirm", "tier": "opus"})
    assert (row["cascade_tier"], row["cascade_source"], row["cascade_model"]) == (
        3,
        "opus",
        "anthropic/claude-opus-5.5",
    )
    assert "automatic_abstention" not in row and json.dumps(row)


def test_fitter_confirm_is_a_paid_confirm_without_a_model():
    event = {
        "event_type": "bounce",
        "frame": 70.0,
        "occurrence_status": "ambiguous",
        "status": "predicted",
        "annotation_origin": "automatic",
        "exact_epoch_observed": False,
        "timing_status": "predicted",
        "frame_interval": [69.0, 71.0],
        "automatic_abstention": {"acceptance_marginal": 0.8},
    }
    walk = cascade.fitter_settle(ROW)
    assert (walk["action"], walk["tier"], walk["model"], walk["receipts"]) == (
        "confirm",
        "fitter_adjudicated",
        None,
        [],
    )
    fitter = rebind.confirm_event(event, walk)
    paid = rebind.confirm_event(event, {"action": "confirm", "tier": "gemini"})
    assert (fitter["cascade_tier"], fitter["cascade_source"], fitter["cascade_model"]) == (
        0,
        "fitter_adjudicated",
        None,
    )
    # S6 reads the note, status and occurrence fields; only the receipt names differ.
    drop = ("cascade_tier", "cascade_source", "cascade_model")
    assert {k: v for k, v in fitter.items() if k not in drop} == {
        k: v for k, v in paid.items() if k not in drop
    }


def test_fitter_adjudication_is_the_default():
    assert cascade.DEFAULT_ADJUDICATION == "fitter"
    assert cascade.candidate_scope_for(cascade.DEFAULT_ADJUDICATION) == "unresolved_or_rejected"


def test_adjudication_switch_picks_the_candidate_scope():
    assert cascade.candidate_scope_for("fitter") == cascade.FITTER_CANDIDATE_SCOPE
    assert cascade.candidate_scope_for("paid") == cascade.DEFAULT_CANDIDATE_SCOPE
    assert cascade.candidate_scope_for("fitter", "unresolved") == "unresolved"
    with pytest.raises(ValueError):
        cascade.candidate_scope_for("free")


def test_fitter_adjudicate_confirms_every_candidate_at_zero_cost(tmp_path):
    from cv.pipeline import event_cascade_runner as runner

    rows = [
        {
            "id": f"a__{i}",
            "attempt": "a",
            "panel": "p",
            "clip": "c",
            "event_index": i,
            "event_type": kind,
            "frame": frame,
            "band": "low",
        }
        for i, (kind, frame) in enumerate((("bounce", 70.0), ("contact", 90.5)))
    ]
    (tmp_path / "candidates.json").write_text(
        json.dumps({"rows": rows, "wave_attempts": 23, "candidate_scope": "unresolved_or_rejected"})
    )
    document, cost = runner.fitter_adjudicate(tmp_path)
    saved = json.loads((tmp_path / "adjudicate" / "decisions.json").read_text())
    assert saved == document
    assert document["actions"] == {"confirm": 2} and document["pending"] == 0
    assert {d["tier"] for d in document["decisions"]} == {"fitter_adjudicated"}
    assert document["receipt_usd"] == 0 and document["by_model"] == {}
    assert cost["usd_per_match"] == 0 and cost["wave_attempts"] == 23
    assert "no model is asked" in document["rule"]
