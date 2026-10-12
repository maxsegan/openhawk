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


def test_resumed_refit_redoes_an_interrupted_attempt(tmp_path, monkeypatch):
    import argparse

    from cv.pipeline import s6_broadcast_backend

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": [{"key": "a"}], "policy": {}}))
    wave = tmp_path / "cascade" / "CASCADE_WAVE.json"
    wave.parent.mkdir(parents=True)
    wave.write_text(
        json.dumps({"jobs": [{"attempt": "a", "manifest": str(manifest), "timeout_seconds": 5}]})
    )
    (tmp_path / "refit" / "a" / "stage_run").mkdir(parents=True)  # killed mid-refit

    def run_stage(job, stage, timeout):
        assert not stage.exists() and not (stage.parent / "stage_run").exists()
        return {"status": "completed"}

    monkeypatch.setattr(s6_broadcast_backend, "run_stage", run_stage)
    detail = runner.step_refit(argparse.Namespace(out=tmp_path, workers=1))
    assert {k: detail[k] for k in ("refits", "completed")} == {"refits": 1, "completed": 1}
    assert list((tmp_path / "refit" / "a").glob("stage_run.partial.*"))


def test_a_step_running_in_a_live_runner_is_not_started_twice(tmp_path, monkeypatch):
    marker = tmp_path / "steps" / "refit.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"status": "running", "pid": 424242}))
    monkeypatch.setattr(runner, "_runner_alive", lambda pid: pid == 424242)
    with pytest.raises(RuntimeError, match="already running"):
        runner.metered("refit", tmp_path, lambda: {})
    monkeypatch.setattr(runner, "_runner_alive", lambda pid: False)
    assert runner.metered("refit", tmp_path, lambda: {"ok": 1})["detail"] == {"ok": 1}


def test_event_beam_and_flight_validation_default_on_with_rollbacks(tmp_path):
    args = _args(tmp_path)
    assert (args.event_beam, args.flight_validation) == ("on", "on")
    off = _args(tmp_path, "--event-beam", "off", "--flight-validation", "off")
    assert (off.event_beam, off.flight_validation) == ("off", "off")
    assert runner.step_beam(off) == {"event_beam": "off"}
    assert runner.STEPS.index("beam") == runner.STEPS.index("refit") + 1


def test_validate_flights_keeps_failing_flights_unconfirmed(tmp_path):
    cameras = tmp_path / "cameras.json"
    cameras.write_text(json.dumps({"cameras": []}))
    result = {
        "verdict": {
            "flights": [
                {
                    "start_frame": 10.0,
                    "end_frame": 60.0,
                    "accepted": True,
                    "contacts": [{"end": "start", "kind": "contact", "contact_xyz_m": [5, 2, 1.0]}],
                },
                {
                    "start_frame": 60.0,
                    "end_frame": 90.0,
                    "accepted": True,
                    "contacts": [
                        {"end": "start", "kind": "contact", "contact_xyz_m": [5, 22, 4.5]}
                    ],
                },
            ]
        }
    }
    flights = runner.attempt_flights(result)
    tally = runner.validate_flights(result, {"cameras": str(cameras)}, flights)
    assert tally == {"launch_height": 1}
    assert flights[0]["confirmed"] is True
    assert flights[1]["confirmed"] is False
    assert flights[1]["validation"]["reasons"] == ["launch_height"]


def test_upstream_decodes_video_on_the_leased_card(tmp_path, monkeypatch):
    # The dispatcher's index would name another card (0 = the scoreboard card); under the
    # step's CUDA_VISIBLE_DEVICES the leased card is device 0.
    monkeypatch.setenv("TENNIS_VIDEO_DECODE", "nvdec:2")
    seen = []
    monkeypatch.setattr(runner, "_broadcast_runner", lambda command, env, log: seen.append(env))
    args = _args(tmp_path, "--gpu", "3", "--score-vlm-url", "http://hq:1/v1")
    runner.step_upstream(args)
    assert seen and all(
        (e["CUDA_VISIBLE_DEVICES"], e["TENNIS_VIDEO_DECODE"], e["TENNIS_NVDEC_GPU"])
        == ("3", "nvdec", "0")
        for e in seen
    )
    monkeypatch.delenv("TENNIS_VIDEO_DECODE")
    seen.clear()
    runner.step_upstream(args)
    assert "TENNIS_VIDEO_DECODE" not in seen[0]


def test_recorded_job_paths_are_read_from_the_current_copy_of_the_match(tmp_path):
    # A cascade wave or BEAM.json written on another box (or before the match moved)
    # records that copy's absolute paths; the steps read this copy's file instead.
    from cv.pipeline.product_runner import in_match

    here = tmp_path / "data_b" / "processed" / "q" / "m1"
    there = tmp_path / "box" / "data" / "processed" / "q" / "m1"
    for root in (here, there):
        (root / "cascade" / "manifests").mkdir(parents=True)
        (root / "cascade" / "manifests" / "m1__pt0001.json").write_text(str(root))
    recorded = str(there / "cascade" / "manifests" / "m1__pt0001.json")
    assert in_match(recorded, here) == here / "cascade" / "manifests" / "m1__pt0001.json"
    # Unmoved: the recorded path itself; a file this copy lacks: the recorded path.
    assert in_match(recorded, there) == Path(recorded)
    assert in_match(str(there / "cascade" / "x.json"), here) == there / "cascade" / "x.json"


def test_a_match_without_processable_attempts_ends_done_with_a_reason(tmp_path, monkeypatch):
    from cv.pipeline import broadcast_runner, product_outcomes

    args = _args(tmp_path)
    args.video.write_bytes(b"")
    upstream = args.out / "upstream" / "m"

    def upstream_step(_args):
        upstream.mkdir(parents=True)
        (upstream / "automatic_point_ledger_v1.csv").write_text("rally_t_start,rally_t_end\n")
        (upstream / "automatic_point_ledger_v1.json").write_text(json.dumps({"attempts": 0}))
        assert broadcast_runner.declare_no_processable_attempts(
            upstream,
            upstream / "automatic_point_ledger_v1.csv",
            upstream / "automatic_point_ledger_v1.json",
        )
        return {}

    ran = []
    monkeypatch.setattr(runner, "step_upstream", upstream_step)
    monkeypatch.setattr(runner, "step_compact", lambda _args: ran.append("compact") or {})
    for step in ("fit", "cascade", "refit", "beam", "portal"):
        monkeypatch.setattr(runner, f"step_{step}", lambda _args, s=step: ran.append(s))
    runner.run(args)
    assert ran == ["compact"]
    outcomes = json.loads((args.out / product_outcomes.FILE).read_text())
    assert outcomes["status"] == "done" and outcomes["failure"] is None
    assert outcomes["counts"]["accepted_flights"] == 0
    assert outcomes["no_processable_attempts"]["reason"] == "point_ledger_empty"
    assert outcomes["stages"]["point_segmentation"]["reason"] == "ledger_empty"
    result = json.loads((args.out / "MATCH_RESULT.json").read_text())
    assert result["counts"]["attempts_with_flights"] == 0 and result["attempts"] == []
    # The failure index is not part of the public release (cv/pipeline/failure_index.py).
    assert "failed_flights" not in outcomes and "missed_play" not in outcomes
    assert not (args.out / "FAILED_FLIGHTS.json").exists()


def test_long_feed_switches_reach_the_upstream_command_only_when_asked(tmp_path):
    plain = runner.upstream_command(_args(tmp_path))
    assert "--long-shot-view-windows" not in plain
    assert "--scoreboard-map-agreement" not in plain
    asked = runner.upstream_command(
        _args(tmp_path, "--long-shot-view-windows", "--scoreboard-map-agreement")
    )
    assert {"--long-shot-view-windows", "--scoreboard-map-agreement"} <= set(asked)


def test_trim_record_cuts_the_flight():
    positions = [[5.0, 0.4 * i, 1.0] for i in range(97)]  # 0.4 s at 240 Hz, frames 0..20
    dense = {
        "start_frame": 0.0,
        "end_frame": 20.0,
        "positions": positions,
        "bounces": [{"frame": 5.0, "x": [5, 8, 0]}, {"frame": 15.0, "x": [5, 30, 0]}],
        "net_hits": [],
    }
    flight = {"start_frame": 0.0, "end_frame": 20.0, "positions_m": positions[::4]}
    runner._trim_record(flight, {"end_frame": 10.0, "reason": "past_back_wall"}, dense)
    assert flight["end_frame"] == 10.0
    assert flight["end_xyz_m"] == positions[48]
    assert len(flight["positions_m"]) == 13
    assert flight["bounces_xyz_m"] == [[5, 8, 0]]
    assert flight["trim"] == {
        "end_frame": 10.0,
        "reason": "past_back_wall",
        "end_frame_before": 20.0,
    }
