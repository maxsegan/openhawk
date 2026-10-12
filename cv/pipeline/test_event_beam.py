import json

import numpy as np

from cv.pipeline import event_beam as eb


def _attempt(**kw):
    a = eb.Attempt("m__pt0001", {"rows": [{}]}, {}, "/nonexistent/result.json", 60.0)
    a.fps, a.clip = 50.0, "pt0001"
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_candidates_merge_witnesses_and_relate_to_automatic_contacts(monkeypatch):
    a = _attempt(contacts=[100.0])
    raw = [
        {"family": "audio", "frame": 150.0, "interval": [148.0, 152.0], "z": 12.0},
        {"family": "s5_held", "frame": 151.5, "interval": [150.0, 153.0]},
        {"family": "audio", "frame": 108.0, "interval": [106.0, 110.0], "z": 9.0},
        {"family": "s5_peak", "frame": 101.0, "interval": [99.0, 103.0]},
    ]
    monkeypatch.setattr(eb, "witnesses", lambda *_: raw)
    out = eb.candidates(None, a, {"modes": [0.0], "spread": 1.0})
    by_frame = {c["frame"]: c for c in out}
    assert by_frame[101.0]["relation"] == "duplicate"
    assert by_frame[108.0]["relation"] == "retime"
    # The audio witness times the merged candidate; both families are kept.
    assert by_frame[150.0]["families"] == ["audio", "s5_held"]
    assert by_frame[150.0]["relation"] == "insert"
    assert by_frame[150.0]["interval"] == [148.0, 153.0]


def test_split_test_prefers_one_arc_on_a_smooth_track():
    t = np.arange(80, 121)
    smooth = {int(k): np.array([3.0 * k, 0.05 * (k - 100) ** 2]) for k in t}
    hinge = {int(k): np.array([3.0 * k, 400 - 8.0 * abs(k - 100)]) for k in t}
    assert not eb.split_ok(smooth, 100.0)
    assert eb.split_ok(hinge, 100.0)
    assert abs(eb.best_break(hinge, 103.0) - 100.0) <= 0.5


def test_bounce_guard_drops_a_candidate_just_before_a_bounce(monkeypatch):
    a = _attempt(bounces=[74.2])
    assert not eb.bounce_ok(a, 72.0)
    assert eb.bounce_ok(a, 70.0)  # outside the guard
    # After the bounce: kept only when the ball then crosses the net.
    monkeypatch.setattr(eb, "crosses_back", lambda _a, _f: False)
    assert not eb.bounce_ok(a, 75.5)
    monkeypatch.setattr(eb, "crosses_back", lambda _a, _f: True)
    assert eb.bounce_ok(a, 75.5)


def test_edits_never_retime_a_row_the_cascade_settled(monkeypatch):
    monkeypatch.setattr(
        eb, "contact_event", lambda f, i, fam: {"event_type": "contact", "frame": f}
    )
    row = {"event_type": "contact", "frame": 100.0, "clip": "x"}
    from cv.pipeline.event_cascade_rebind import event_key

    a = _attempt(inventory=[row, {"event_type": "bounce", "frame": 140.0}])
    retime = {"frame": 108.0, "interval": [107.0, 109.0], "families": ["audio"]}
    assert eb.edit_for(retime, a, set(), [])["action"] == "retime"
    a.settled = {event_key(row): {"action": "confirm"}}
    assert eb.edit_for(retime, a, set(), []) is None
    near_bounce = {"frame": 141.0, "interval": [140.0, 142.0], "families": ["gap"]}
    assert eb.edit_for(near_bounce, a, set(), []) is None
    insert = {"frame": 160.0, "interval": [158.0, 162.0], "families": ["gap"]}
    assert eb.edit_for(insert, a, set(), [])["action"] == "insert"


def _result(flights):
    return {
        "verdict": {
            "flights": [
                {"start_frame": s, "end_frame": e, "accepted": ok, "contacts": []}
                for s, e, ok in flights
            ]
        }
    }


def test_decide_moves_only_to_a_fit_with_more_flights_and_no_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "guard_ok", lambda *_: True)
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_result([(10, 50, True), (50, 90, False)])))

    def fit(name, flights):
        folder = tmp_path / name
        (folder / "stage").mkdir(parents=True)
        (folder / "stage" / "result.json").write_text(json.dumps(_result(flights)))
        (folder / "run.json").write_text(json.dumps({"status": "completed"}))
        return str(folder)

    loses = fit("b0", [(10, 30, True), (30, 50, True), (50, 90, True)])  # loses 10-50
    gains = fit("b1", [(10, 50, True), (50, 70, True), (70, 90, True)])
    a = _attempt()
    st = {"incumbent": str(base), "cands": [], "history": []}
    c0 = {"frame": 30.0}
    c1 = {"frame": 70.0}
    st["rounds"] = {
        "1": {
            "branches": [
                {"name": "r1b0", "cands": ["30.00"], "folder": loses},
                {"name": "r1b1", "cands": ["70.00"], "folder": gains},
            ],
            "cands": {"30.00": c0, "70.00": c1},
        }
    }
    eb._decide_attempt(a, st, 1)
    assert st["rounds"]["1"]["decision"]["kept"] == "r1b1"
    assert st["incumbent"].endswith("b1/stage/result.json")
    assert st["cands"] == [c1]
    assert st["incumbent_manifest"].endswith("b1/packet/manifest.json")


def test_composed_reuses_the_branch_fit_with_identical_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "guard_ok", lambda *_: True)
    monkeypatch.setattr(eb, "edits_for", lambda a, cands: [{"id": c["frame"]} for c in cands])
    monkeypatch.setattr(eb, "write_branch", lambda *_: (_ for _ in ()).throw(AssertionError))
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_result([(10, 50, True), (50, 90, False)])))
    folder = tmp_path / "b1"
    (folder / "stage").mkdir(parents=True)
    gained = _result([(10, 50, True), (50, 70, True), (70, 90, True)])
    (folder / "stage" / "result.json").write_text(json.dumps(gained))
    (folder / "run.json").write_text(json.dumps({"status": "completed"}))
    c1 = {"frame": 70.0}
    st = {"incumbent": str(base), "cands": [], "history": []}
    st["rounds"] = {
        "1": {
            "slots": [{"slot": [50, 90], "ranked": ["70.00"]}],
            "branches": [
                {
                    "name": "r1b0",
                    "cands": ["70.00"],
                    "folder": str(folder),
                    "edits_sha256": eb._edits_digest([{"id": 70.0}]),
                }
            ],
            "cands": {"70.00": c1},
        }
    }
    a = _attempt()
    assert eb._compose_attempt(a, st, tmp_path, 1) == []
    rec = st["rounds"]["1"]
    assert rec["composed"]["reused_branch"] == "r1b0"
    eb._decide_attempt(a, st, 1)
    assert rec["decision"]["kept"] == "composed" and st["cands"] == [c1]
    assert st["incumbent"] == str(folder / "stage" / "result.json")


def test_time_witness_prune_keeps_audio_and_s5_peak_witnesses():
    def cand(families, z=None):
        return {"families": families, "audio_z": z}

    assert eb.pruned(cand(["gap"])) and eb.pruned(cand(["break", "gap"]))
    assert eb.pruned(cand(["s5_held"])) and eb.pruned(cand(["audio"], 6.0))
    assert not eb.pruned(cand(["audio"], 9.0)) and not eb.pruned(cand(["s5_peak"]))
    assert not eb.pruned(cand(["audio", "break"], 5.7)) and not eb.pruned(cand(["audio", "gap"], 5))
    assert eb.PRUNE_MODES[0] == "off"
