import json
from pathlib import Path

import pytest

from cv.pipeline import event_cascade_rebind as rebind
from cv.pipeline import product_runner as runner
from cv.pipeline import provenance


def _args(tmp_path, *extra):
    return runner.parser().parse_args(
        [
            "--video",
            str(tmp_path / "match.mp4"),
            "--match-id",
            "m",
            "--surface",
            "clay",
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_upstream_command_is_the_production_profile_and_stops_before_s6(tmp_path):
    args = _args(tmp_path)
    command = runner.upstream_command(args)
    text = " ".join(command)
    for flag in runner.UPSTREAM_PROFILE:
        assert flag in command
    assert "--stop-after upstream" in text
    assert "--s6-backend" not in text
    assert "--event-marginal-threshold 0.9907168846560898" in text
    assert "--resume" in command
    # torso lock and court-anchor play camera stay at the runner's own defaults (on)
    assert "--no-torso-lock" not in command
    assert "--no-active-play-court-anchor-camera" not in command
    assert "label" not in text.lower()


def test_policy_file_is_complete_after_the_pipeline_defaults(tmp_path, monkeypatch):
    from cv.pipeline import s6_broadcast_backend as backend
    from cv.pipeline import s6_labeled_stage as stage

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    prior = tmp_path / "prior.json"
    prior.write_text("{}")
    policy = dict(json.loads(runner.POLICY_FILE.read_text())["policy"])
    policy["serve_location_prior"] = provenance.file_record(prior)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    loaded = backend.load_policy(path)
    for key, value in stage.PIPELINE_COMPONENT_POLICY.items():
        assert loaded[key] == policy.get(key, value)


def test_fit_attempts_expands_split_children(tmp_path):
    fit = tmp_path / "fit"
    fit.mkdir()
    summary = {
        "match_id": "m",
        "attempts": [
            {
                "key": "m__pt0001",
                "clip": "pt0001",
                "status": "completed",
                "ledger_row": {"pt": "1"},
            },
            {
                "key": "m__pt0002",
                "clip": "pt0002",
                "status": "children_incomplete",
                "ledger_row": {"pt": "2"},
                "children": [
                    {"key": "m__pt0002__a01", "child_index": 1, "status": "completed"},
                    {"key": "m__pt0002__a02", "child_index": 2, "status": "preparation_held"},
                ],
            },
            {"key": "m__pt0003", "clip": "pt0003", "status": "preparation_held", "reason": "x"},
        ],
    }
    (fit / "summary.json").write_text(json.dumps(summary))
    _, attempts = runner.fit_attempts(fit)
    assert [a["key"] for a in attempts] == [
        "m__pt0001",
        "m__pt0002__a01",
        "m__pt0002__a02",
        "m__pt0003",
    ]
    assert attempts[1]["stage"] == fit / "attempts/m__pt0002/child_01/stage"
    assert attempts[3]["reason"] == "x"


def _dense(start, end):
    return {
        "start_frame": start,
        "end_frame": end,
        "start_xyz": [0, 0, 1],
        "end_xyz": [0, 20, 0],
        "positions": [[0, y, 1] for y in range(12)],
        "bounces": [{"x": [0, 18, 0.03]}],
        "net_hits": [],
    }


def test_attempt_flights_keeps_accepted_flights_with_their_fitted_states():
    result = {
        "component_results": [
            {"component_index": 0, "result": {"measurement": {"dense_flights": [_dense(10, 30)]}}},
            {
                "component_index": 1,
                "result": {"measurement": {"dense_flights": [_dense(40, 60), _dense(60, 80)]}},
            },
        ],
        "verdict": {
            "flights": [
                {"accepted": True, "component_index": 0, "flight_index": 0, "start_frame": 10.0},
                {"accepted": False, "component_index": 1, "flight_index": 0, "start_frame": 40.0},
                {"accepted": True, "component_index": 1, "flight_index": 1, "start_frame": 60.0},
                {"accepted": False, "original_flight_index": 7, "start_frame": 90.0},
            ]
        },
    }
    flights = runner.attempt_flights(result, step=4)
    assert [f["start_frame"] for f in flights] == [10.0, 60.0]
    assert flights[1]["bounces_xyz_m"] == [[0, 18, 0.03]]
    assert len(flights[0]["positions_m"]) == 3


def test_edited_automatic_packet_gets_a_producer_that_binds_it(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    old = tmp_path / "prep"
    old.mkdir()
    for name in ("packet.json", "observations.json", "cameras.json", "boxes.csv"):
        (old / name).write_text(name)
    video = {"path": "video.mp4", "path_base": "TENNIS_DATA_ROOT", "sha256": "v" * 64}
    parent = provenance.build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=provenance.AUTOMATIC_MODE,
        source_videos=[video],
        models=[{"name": "event_model"}],
        configuration={"stage": "prep"},
        reused_artifacts=[
            provenance.file_record(old / name)
            for name in ("packet.json", "observations.json", "cameras.json", "boxes.csv")
        ],
    )
    parent_path = provenance.write_provenance(old / "automatic_provenance.json", parent)
    row = {
        "observation_origin": "automatic",
        "packet": provenance.file_record(old / "packet.json"),
        "labels": provenance.file_record(old / "observations.json"),
        "automatic_provenance": provenance.file_record(parent_path),
    }
    folder = tmp_path / "edited"
    folder.mkdir()
    (folder / "packet.json").write_text("new packet")
    (folder / "observations.json").write_text("new document")
    decisions = {
        ("bounce", 10.0): {
            "event_type": "bounce",
            "frame": 10.0,
            "action": "confirm",
            "receipts": [
                {"tier": 1, "model": "Qwen/Qwen3.8-27B-FP8", "usd": 0.0},
                {"tier": 2, "model": "google/gemini-3.8-flash", "usd": 0.001},
            ],
        }
    }
    path = rebind.automatic_producer(row, folder, decisions)
    document = provenance.load_provenance(path, require_automatic=True)
    bound = {(r["path"], r["sha256"]) for r in document["reused_artifacts"]}
    new_packet = provenance.file_record(folder / "packet.json")
    assert (new_packet["path"], new_packet["sha256"]) in bound
    assert (row["packet"]["path"], row["packet"]["sha256"]) not in bound
    assert (row["labels"]["path"], row["labels"]["sha256"]) not in bound
    cameras = provenance.file_record(old / "cameras.json")
    assert (cameras["path"], cameras["sha256"]) in bound
    assert document["source_videos"] == [video]
    roles = {m.get("role"): m["name"] for m in document["models"]}
    assert roles["event_cascade_gemini"] == "google/gemini-3.8-flash"
    assert roles["event_cascade_qwen"] == "Qwen/Qwen3.8-27B-FP8"
    assert json.loads((folder / "cascade_decisions.json").read_text())["decisions"][0][
        "action"
    ] == ("confirm")


def test_write_manifest_refuses_an_automatic_edit_without_decisions(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    manifest = {"policy": {}, "rows": [{"observation_origin": "automatic"}]}
    with pytest.raises(ValueError, match="settled decisions"):
        rebind.write_manifest(manifest, {}, {}, tmp_path / "edited")


def test_fitter_adjudicated_packet_names_no_model(tmp_path, monkeypatch):
    from cv.pipeline import event_cascade as cascade

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    old = tmp_path / "prep"
    old.mkdir()
    for name in ("packet.json", "observations.json", "cameras.json"):
        (old / name).write_text(name)
    video = {"path": "video.mp4", "path_base": "TENNIS_DATA_ROOT", "sha256": "v" * 64}
    parent = provenance.build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=provenance.AUTOMATIC_MODE,
        source_videos=[video],
        models=[{"name": "event_model"}],
        configuration={"stage": "prep"},
        reused_artifacts=[
            provenance.file_record(old / name)
            for name in ("packet.json", "observations.json", "cameras.json")
        ],
    )
    parent_path = provenance.write_provenance(old / "automatic_provenance.json", parent)
    row = {
        "observation_origin": "automatic",
        "packet": provenance.file_record(old / "packet.json"),
        "labels": provenance.file_record(old / "observations.json"),
        "automatic_provenance": provenance.file_record(parent_path),
    }
    folder = tmp_path / "edited"
    folder.mkdir()
    (folder / "packet.json").write_text("new packet")
    (folder / "observations.json").write_text("new document")
    walk = cascade.fitter_settle({})
    decisions = {("bounce", 10.0): {"event_type": "bounce", "frame": 10.0, **walk}}
    document = provenance.load_provenance(
        rebind.automatic_producer(row, folder, decisions), require_automatic=True
    )
    assert document["configuration"]["producer"] == "fitter_adjudication_rebind"
    assert document["configuration"]["confirms"] == 1
    assert [m["name"] for m in document["models"]] == ["event_model"]


def test_event_adjudication_flag_defaults_to_the_cascade_default(tmp_path):
    assert _args(tmp_path).event_adjudication is None
    assert _args(tmp_path, "--event-adjudication", "paid").event_adjudication == "paid"
