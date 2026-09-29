"""Automatic preparation contracts: native evidence and ancestry, no numerical fit."""

from dataclasses import replace
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cv.pipeline import artifact_cache, paths, provenance
from cv.pipeline.s6_automatic_observations import AutomaticInputs, build_observations


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")
    return path


def _csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _bind(inputs, *, extra=()):
    required = [
        inputs.manifest,
        inputs.point_ledger,
        inputs.extraction_receipt,
        inputs.source_pts,
        inputs.ball,
        Path(str(inputs.ball) + ".coordinates.json"),
        inputs.events,
        inputs.cameras,
        inputs.players,
        Path(str(inputs.players) + ".coordinates.json"),
        *extra,
    ]
    prov = provenance.build_provenance(
        root=paths.REPO_ROOT,
        mode="automatic",
        source_videos=[provenance.file_record(inputs.source_video)],
        configuration={"producer": "automatic_fixture"},
        reused_artifacts=[
            provenance.file_record(p, role="automatic_producer_output") for p in required
        ],
    )
    _json(inputs.ancestry[0], prov)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    frames = tmp_path / "frames"
    frames.mkdir()
    # The fixture stands in for a producer-validated video. Preparation never decodes it.
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-video-identity")
    pts = tmp_path / "source_pts.txt"
    pts.write_text("".join(f"frame|pts_time={0.003 + i / 25:.6f}\n" for i in range(50)))
    ledger = _csv(
        tmp_path / "ledger.csv",
        [
            {
                "pt": 1,
                "rally_t_start": 0.4,
                "rally_t_end": 1.2,
                "point_index": 7,
                "attempt_index": 1,
                "attempts_in_point": 2,
            }
        ],
    )
    manifest = _json(
        tmp_path / "manifest.json",
        {"matches": [{"id": "match", "source_fps": 25.0, "surface": "grass", "point_ids": [1]}]},
    )
    source_identity = {
        "stage": "player_frame_extraction",
        "command": [],
        "code": [],
        "configuration": {
            "native": True,
            "preserve_source_fps": True,
            "fps": 25.0,
            "native_epoch_capture": "ffmpeg_copyts_showinfo_v1",
        },
        "inputs": [artifact_cache.path_record(video)],
        "upstream_receipts": [],
    }
    source_identity["fingerprint"] = artifact_cache._digest_json(source_identity)
    pictures = []
    for f in range(1, 21):
        picture = frames / f"f_{f:04d}.jpg"
        Image.new("RGB", (1920, 1080), (f, 30, 10)).save(picture)
        pictures.append({"name": picture.name, "sha256": provenance.file_sha256(picture)})
    extraction = _json(
        frames / "extraction_receipt.json",
        {
            "schema": "player_frame_extraction_v1",
            "identity": {
                "source_fingerprint": source_identity["fingerprint"],
                "point": {"pt": 1, "t0": 0.4, "t1": 1.2},
                "video_duration": 2,
            },
            "source_identity": source_identity,
            "native_clock": {
                "schema": "native_extracted_picture_clock_v1",
                "method": "ffmpeg_copyts_showinfo",
                "seek_timestamp": True,
                "seek_seconds": 0.4,
                "time_base": "1/1000",
                "frames": [
                    {"name": row["name"], "source_pts": 403 + index * 40}
                    for index, row in enumerate(pictures)
                ],
            },
            "frames": pictures,
        },
    )
    ball = _csv(
        tmp_path / "ball.csv",
        [
            {
                "clip": "pt0001",
                "frame": f"f_{f:04d}.jpg",
                "x": 900.0 + f,
                "y": 400.0 + f,
                "confidence": 0.82,
                "sources": "interpolated" if f == 13 else "detector",
            }
            for f in range(1, 21)
            if f != 8
        ],
    )
    players = _csv(
        tmp_path / "players.csv",
        [
            {
                "clip": "pt0001",
                "frame": f"f_{f:04d}.jpg",
                "side": side,
                "x0": 850,
                "y0": 700,
                "x1": 950,
                "y1": 1000,
                "court_x": 4.0,
                "court_y": 0.0 if side == "near" else 23.77,
            }
            for f in range(1, 21)
            for side in ("near", "far")
        ],
    )
    for artifact in (ball, players):
        _json(
            Path(str(artifact) + ".coordinates.json"),
            {
                "schema": "tennis.coordinate-space.v1",
                "artifact": artifact.name,
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 1920, "height": 1080},
                "source": "automatic_fixture",
            },
        )
    events = _json(
        tmp_path / "events.json",
        [
            {
                "match_id": "match",
                "clip": "pt0001",
                "event_type": kind,
                "frame": rounded,
                "abstain": False,
                "confidence": 0.9,
                "probability": 0.88,
                "location": {"frame_subpixel": fine},
                "frame_interval": interval,
                **(
                    {
                        "point_end": {
                            "termination_kind": "second_ground",
                            "terminal_event_type": "bounce",
                        }
                    }
                    if kind == "point_end"
                    else {}
                ),
            }
            for kind, rounded, fine, interval in [
                ("contact", 4, 4.2, [4.0, 5.0]),
                ("bounce", 18, 17.6, [17.0, 18.0]),
                ("point_end", 18, 17.6, [17.0, 18.0]),
            ]
        ],
    )
    camera = tmp_path / "camera.npz"
    np.savez(
        camera,
        clips=np.repeat("pt0001", 20),
        frames=np.arange(1, 21),
        P=np.tile(
            np.array([[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]]),
            (20, 1, 1),
        ),
        reliable=np.arange(1, 21) != 7,
        source=np.repeat("bundle", 20),
        confidence=np.ones(20),
        k1=np.full(20, 1e-8),
        dist_center=np.tile([960.0, 540.0], (20, 1)),
    )
    result = AutomaticInputs(
        "match",
        "pt0001",
        manifest,
        ledger,
        extraction,
        frames,
        video,
        pts,
        ball,
        events,
        camera,
        players,
        (tmp_path / "parent.json",),
    )
    _bind(result)
    return result


def test_prepares_automatic_inputs_without_any_label_file(inputs, tmp_path):
    assert not list(tmp_path.rglob("*label*"))
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "prepared"
    row = result["row"]
    assert row["observation_origin"] == "automatic" and row["declared_flights"] is None
    assert row["player_order"] == [] and row["player_statures_m"] == []
    packet = json.loads((tmp_path / "prepared/packet.json").read_text())
    consumer = json.loads((tmp_path / "prepared/observations.json").read_text())
    prov = json.loads((tmp_path / "prepared/automatic_provenance.json").read_text())
    provenance.validate_provenance(prov)
    assert packet["human_derived"] is False
    assert set(packet["stream_origins"].values()) == {"automatic"}
    assert packet["ball_observation_operator"]["kind"] == "nominal_center"
    assert packet["external_label_binding"]["record"]["sha256"] == row["labels"]["sha256"]
    assert {row[k]["sha256"] for k in ("labels", "packet", "cameras", "pose_csv")} <= {
        r["sha256"] for r in prov["reused_artifacts"]
    }
    attempt = packet["attempts"][0]
    assert attempt["events"][0]["frame"] == 4.2
    assert attempt["events"][0]["frame_interval"] == [4.0, 5.0]
    assert attempt["events"][0]["exact_epoch_observed"] is False
    assert attempt["events"][0]["automatic_probability"] == 0.88
    assert attempt["labeled_native_frames"] == 0
    assert attempt["owner_ball_labels"] == attempt["ball_observations"]
    rows = consumer["ball"]["records"][0]["frames"]
    assert [r["frame"] for r in rows] == list(range(1, 21))
    assert rows[7]["status"] == "missing"
    assert rows[12]["status"] == "derived_estimate"
    assert rows[0]["native_pts_seconds"] == 0.403
    assert rows[0]["automatic_confidence"] == 0.82
    assert consumer["source_pack"]["images"][0]["source_frame_index"] == 11
    assert set([1, 2, 3, 4, 18, 19, 20]) <= set(attempt["context_native_frames"])
    assert "serve_speed_evidence" not in consumer and "court" not in consumer
    cameras = json.loads((tmp_path / "prepared/cameras.json").read_text())
    assert cameras["cameras"][0]["k1"] == 1e-8
    assert cameras["cameras"][6]["supported"] is False
    # The automatic packet path admits tracked intrinsic fallbacks unless a
    # caller names the rollback. This fixture has no such frames.
    assert cameras["intrinsic_fallback_registration"] == "admit"
    assert cameras["admitted_intrinsic_fallback_frames"] == 0
    assert result["runtime_model_calls"] == 0 and result["fitted_states_read"] is False


@pytest.mark.parametrize("fps", [25.0, 50.0, 60000 / 1001])
@pytest.mark.parametrize("fraction", [-0.4, 0.0, 0.4])
def test_service_scope_uses_native_picture_epoch_without_retagging(
    inputs, tmp_path, monkeypatch, fps, fraction
):
    from cv.pipeline import service_attempt_scope

    # Exercise real source-clock validation independently of stance estimation.
    monkeypatch.setattr(
        service_attempt_scope.segment_boundaries, "load_player_motion", lambda *a, **k: {}
    )
    # Fractional requested windows select the same original native ordinals.
    start, end = (10 + fraction) / fps, (30 + fraction) / fps
    ledger = list(csv.DictReader(inputs.point_ledger.open()))
    ledger[0].update(rally_t_start=start, rally_t_end=end)
    _csv(inputs.point_ledger, ledger)
    manifest = json.loads(inputs.manifest.read_text())
    manifest["matches"][0]["source_fps"] = fps
    _json(inputs.manifest, manifest)
    inputs.source_pts.write_text("".join(f"frame|pts_time={i / fps:.6f}\n" for i in range(50)))
    receipt = json.loads(inputs.extraction_receipt.read_text())
    source = receipt["source_identity"]
    source["configuration"]["fps"] = fps
    source["fingerprint"] = artifact_cache._digest_json(
        {k: v for k, v in source.items() if k != "fingerprint"}
    )
    receipt["identity"]["source_fingerprint"] = source["fingerprint"]
    receipt["identity"]["point"].update(t0=start, t1=end)
    if "native_clock" in receipt:
        # The measured-clock fixture carries the same exact original ordinals.
        receipt["native_clock"].update(
            seek_seconds=10 / fps,
            time_base="1/3000000",
            frames=[
                {"name": row["name"], "source_pts": round((10 + i) * 3000000 / fps)}
                for i, row in enumerate(receipt["frames"])
            ],
        )
    _json(inputs.extraction_receipt, receipt)
    _bind(inputs)
    original_ledger = inputs.point_ledger.read_bytes()
    original_pts = inputs.source_pts.read_bytes()
    original_pictures = receipt["frames"]

    output = tmp_path / "native_service_scope"
    result = build_observations(inputs, output, service_attempt_split="on")
    assert result["status"] == "prepared"
    plan = json.loads((output / "service_attempt_scope.json").read_text())
    observed = json.loads((output / "observations.json").read_text())
    first = observed["source_pack"]["images"][0]
    assert first["source_frame_index"] == 11
    assert plan["source_start_seconds"] == first["native_pts_seconds"]
    assert plan["source_start_seconds"] == pytest.approx(round(10 / fps, 6), abs=1e-10)
    assert plan["requested_ledger_interval_seconds"] == [start, end]
    assert plan["source_start_basis"] == "first_authenticated_native_picture_pts"
    assert not plan["children"]
    assert inputs.point_ledger.read_bytes() == original_ledger
    assert inputs.source_pts.read_bytes() == original_pts
    assert json.loads(inputs.extraction_receipt.read_text())["frames"] == original_pictures


def test_unknown_ending_keeps_automatic_scope_without_point_completion(inputs, tmp_path):
    events = json.loads(inputs.events.read_text())
    _json(inputs.events, [e for e in events if e["event_type"] != "point_end"])
    _bind(inputs)
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "prepared"
    packet = json.loads((tmp_path / "prepared/packet.json").read_text())
    consumer = json.loads((tmp_path / "prepared/observations.json").read_text())
    evidence = json.loads((tmp_path / "prepared/observed_evidence.json").read_text())
    from cv.experiments.connected_shooting import observation_scope

    contract = observation_scope.validate(packet["attempts"][0], consumer)
    assert contract["schema"] == "s6_observed_ground_scope_v2"
    assert contract["segmentation_assistance"] is None
    assert contract["segmentation_source"]["record"]["sha256"] == provenance.file_sha256(
        inputs.point_ledger
    )
    assert evidence["physical_ending"] is None
    assert "ending_frame" not in consumer["attempt"]
    assert all(e["event_type"] != "ending" for e in consumer["events"]["records"])


def test_missing_contact_retains_counted_preparation_hold_and_video(inputs, tmp_path):
    _json(inputs.events, [])
    _bind(inputs)
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "preparation_held" and result["attempt_count"] == 1
    assert "no contact" in result["reason"]
    assert result["row"] is None
    evidence = json.loads((tmp_path / "prepared/observed_evidence.json").read_text())
    assert len(evidence["source_pack"]["images"]) == 20
    assert len(evidence["ball_observations"]) == 20


@pytest.mark.parametrize("field", ["reviewed_inputs", "manual_overrides", "human_inputs"])
def test_rejects_assisted_ancestry_even_with_automatic_mode(inputs, tmp_path, field):
    doc = json.loads(inputs.ancestry[0].read_text())
    doc[field] = [{"role": "assisted_input"}]
    _json(inputs.ancestry[0], doc)
    with pytest.raises(provenance.ProvenanceError, match="forbidden"):
        build_observations(inputs, tmp_path / "prepared")


def test_rejects_changed_producer_input(inputs, tmp_path):
    inputs.ball.write_text(inputs.ball.read_text().replace("901.0", "950.0"))
    with pytest.raises(ValueError, match="digest changed"):
        build_observations(inputs, tmp_path / "prepared")


def test_rejects_unbound_inputs_and_diagnostic_root(inputs, tmp_path):
    doc = json.loads(inputs.ancestry[0].read_text())
    doc["reused_artifacts"] = []
    _json(inputs.ancestry[0], doc)
    with pytest.raises(ValueError, match="does not bind inputs"):
        build_observations(inputs, tmp_path / "prepared")
    doc["mode"] = "diagnostic"
    _json(inputs.ancestry[0], doc)
    with pytest.raises(ValueError, match="nonautomatic"):
        build_observations(inputs, tmp_path / "prepared")


@pytest.mark.parametrize(
    "change", ["image", "missing_pts", "reordered_pts", "native_cadence", "window"]
)
def test_refuses_broken_native_identity_even_after_parent_rebind(inputs, tmp_path, change):
    if change == "image":
        (inputs.frames_directory / "f_0004.jpg").write_bytes(b"changed")
    elif change in {"missing_pts", "reordered_pts"}:
        lines = inputs.source_pts.read_text().splitlines()
        if change == "missing_pts":
            lines[15] = "frame|pts_time=N/A"
        else:
            lines[15], lines[16] = lines[16], lines[15]
        inputs.source_pts.write_text("\n".join(lines))
    elif change == "native_cadence":
        doc = json.loads(inputs.extraction_receipt.read_text())
        doc["source_identity"]["configuration"]["preserve_source_fps"] = False
        _json(inputs.extraction_receipt, doc)
    else:
        inputs.point_ledger.write_text(inputs.point_ledger.read_text().replace("0.4", "0.5"))
    _bind(inputs)
    with pytest.raises(ValueError):
        build_observations(inputs, tmp_path / "prepared")


def test_rejects_unrelated_receipt_beside_automatic_parent(inputs, tmp_path):
    receipt = artifact_cache.write_stage_receipt(
        out_dir=tmp_path, stage="unrelated", command=[], inputs=[], outputs=[]
    )
    altered = replace(inputs, ancestry=(*inputs.ancestry, receipt))
    with pytest.raises(ValueError, match="unbound automatic ancestry"):
        build_observations(altered, tmp_path / "prepared")


def test_checks_nested_stage_input_provenance(inputs, tmp_path):
    nested = tmp_path / "nested_provenance.json"
    doc = provenance.build_provenance(
        root=paths.REPO_ROOT,
        mode="diagnostic",
        source_videos=[provenance.file_record(inputs.source_video)],
        reviewed_inputs=[{"role": "reviewed_court"}],
    )
    _json(nested, doc)
    receipt = artifact_cache.write_stage_receipt(
        out_dir=tmp_path,
        stage="camera_producer",
        command=[],
        inputs=[nested],
        outputs=[inputs.cameras],
    )
    _bind(inputs, extra=[receipt])
    with pytest.raises(provenance.ProvenanceError, match="reviewed_inputs"):
        build_observations(inputs, tmp_path / "prepared")


def test_checks_bound_stage_receipt_fingerprint(inputs, tmp_path):
    receipt = artifact_cache.write_stage_receipt(
        out_dir=tmp_path,
        stage="camera_producer",
        command=[],
        inputs=[inputs.source_video],
        outputs=[inputs.cameras],
    )
    doc = json.loads(receipt.read_text())
    doc["identity"]["configuration"] = {"changed": True}
    _json(receipt, doc)
    _bind(inputs, extra=[receipt])
    with pytest.raises(ValueError, match="fingerprint changed"):
        build_observations(inputs, tmp_path / "prepared")


def test_rejects_nonautomatic_event_origin_despite_automatic_parent(inputs, tmp_path):
    events = json.loads(inputs.events.read_text())
    events[0]["annotation_origin"] = "agent"
    _json(inputs.events, events)
    _bind(inputs)
    with pytest.raises(ValueError, match="nonautomatic observations"):
        build_observations(inputs, tmp_path / "prepared")


def test_requires_declared_tracker_guide_ancestry(inputs, tmp_path):
    guide = tmp_path / "guide.csv"
    guide.write_text(inputs.ball.read_text())
    guide_sidecar = Path(str(guide) + ".coordinates.json")
    sidecar = Path(str(inputs.ball) + ".coordinates.json")
    guide_sidecar.write_text(sidecar.read_text())
    doc = json.loads(sidecar.read_text())
    doc["source"] = "guide.csv + automatic composer"
    _json(sidecar, doc)
    _bind(inputs)
    with pytest.raises(ValueError, match="does not bind inputs"):
        build_observations(inputs, tmp_path / "refused")
    _bind(inputs, extra=[guide, guide_sidecar])
    result = build_observations(inputs, tmp_path / "prepared")
    assert provenance.file_sha256(guide) in {r["sha256"] for r in result["input_bindings"]}


@pytest.mark.parametrize("mutate_directory", [False, True])
def test_binds_native_receipt_through_producer_directory_output(inputs, tmp_path, mutate_directory):
    receipt = artifact_cache.write_stage_receipt(
        out_dir=tmp_path,
        stage="native_images",
        command=[],
        inputs=[inputs.source_video],
        outputs=[inputs.frames_directory],
    )
    _bind(inputs, extra=[receipt])
    parent = json.loads(inputs.ancestry[0].read_text())
    parent["reused_artifacts"] = [
        r
        for r in parent["reused_artifacts"]
        if r["sha256"] != provenance.file_sha256(inputs.extraction_receipt)
    ]
    _json(inputs.ancestry[0], parent)
    if mutate_directory:
        (inputs.frames_directory / "unexpected.json").write_text("{}")
        with pytest.raises(ValueError, match="output directory changed"):
            build_observations(inputs, tmp_path / "prepared")
    else:
        assert build_observations(inputs, tmp_path / "prepared")["status"] == "prepared"


def test_generated_automatic_inputs_reach_shared_stage_validation(inputs, tmp_path):
    from cv.pipeline import s6_labeled_stage

    prepared = build_observations(inputs, tmp_path / "shared_inputs")
    row = prepared["row"]
    assert row["declared_flights"] is None
    assert set(s6_labeled_stage.validate_row(row)) == set(s6_labeled_stage.INPUTS)


def test_external_video_requires_exact_explicit_source_binding(tmp_path, monkeypatch):
    from cv.pipeline.s6_automatic_observations import _checked_record

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(data))
    source = tmp_path / "videos" / "source.mp4"
    source.parent.mkdir()
    source.write_bytes(b"original video")
    record = provenance.file_record(source)
    assert record["path_base"] == "unconfigured_external"
    assert _checked_record(record, source) == source.resolve()
    with pytest.raises(ValueError, match="configured portable path"):
        _checked_record(record)
    other = tmp_path / "other" / "source.mp4"
    other.parent.mkdir()
    other.write_bytes(b"different video")
    with pytest.raises(ValueError, match="explicitly declared source"):
        _checked_record(record, other)
    with pytest.raises(ValueError, match="explicitly declared source"):
        _checked_record(record | {"path": "unrelated.mp4"}, source)


def test_scope_refusal_retains_original_event_emissions_without_admitting_them(inputs, tmp_path):
    events = [e for e in json.loads(inputs.events.read_text()) if e["event_type"] != "point_end"]
    before = dict(events[1], frame=2, location={"frame_subpixel": 2.0}, frame_interval=[1.5, 2.5])
    rejected = dict(events[0], frame=3, abstain=True, gate_held=True)
    original = [before, rejected, *events]
    _json(inputs.events, original)
    _bind(inputs)
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "preparation_held"
    assert "originating contact" in result["reason"]
    evidence = json.loads((tmp_path / "prepared/observed_evidence.json").read_text())
    assert evidence["original_event_emissions"] == original
    assert result["row"] is None
    assert evidence["physical_events"] == []


def test_known_ending_does_not_make_a_truncated_automatic_origin_complete(inputs, tmp_path):
    events = json.loads(inputs.events.read_text())
    before = dict(events[1], frame=2, location={"frame_subpixel": 2.0}, frame_interval=[1.5, 2.5])
    _json(inputs.events, [before, *events])
    _bind(inputs)
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "preparation_held"
    assert "originating contact" in result["reason"]
    assert result["row"] is None
    evidence = json.loads((tmp_path / "prepared/observed_evidence.json").read_text())
    assert evidence["original_event_emissions"] == [before, *events]


def historical_code_fixture(tmp_path, monkeypatch):
    import hashlib
    import subprocess

    repo = tmp_path / "old_source"
    repo.mkdir()
    source = repo / "producer.py"
    original = b"# original automatic implementation\n"
    source.write_bytes(original)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "producer.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Record producer",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    binding = dict(
        kind="file",
        path_base="repository",
        path="producer.py",
        bytes=len(original),
        sha256=hashlib.sha256(original).hexdigest(),
    )
    identity = dict(
        code=[binding.copy()], configuration=dict(inference_source=dict(commit=commit, dirty=False))
    )
    source.write_text("# changed current implementation\n")
    return binding, identity


def test_historical_code_uses_authentic_declared_revision(tmp_path, monkeypatch):
    from cv.pipeline.s6_automatic_observations import historical_implementation_input

    binding, identity = historical_code_fixture(tmp_path, monkeypatch)
    verified = historical_implementation_input(binding, identity)
    assert verified["verification"] == "git_blob"
    assert verified["commit"] == identity["configuration"]["inference_source"]["commit"]
    assert verified["sha256"] == binding["sha256"]
    # A repository observation is not implementation merely because an old Git
    # revision exists; it must be the exact declared Python closure member.
    identity["code"] = []
    assert historical_implementation_input(binding, identity) is None


@pytest.mark.parametrize("bad", ["revision", "content", "dirty", "traversal"])
def test_historical_code_rejects_unverifiable_revision_or_content(tmp_path, monkeypatch, bad):
    from cv.pipeline.s6_automatic_observations import historical_implementation_input

    binding, identity = historical_code_fixture(tmp_path, monkeypatch)
    if bad == "revision":
        identity["configuration"]["inference_source"]["commit"] = "f" * 40
    elif bad == "content":
        binding["sha256"] = "f" * 64
    elif bad == "dirty":
        identity["configuration"]["inference_source"]["dirty"] = True
    else:
        binding["path"] = "../producer.py"
    identity["code"] = [binding.copy()]
    with pytest.raises(ValueError):
        historical_implementation_input(binding, identity)


def test_ancestry_records_historical_implementation_verification(inputs, tmp_path, monkeypatch):
    from cv.pipeline.s6_automatic_observations import validate_ancestry

    binding, historical = historical_code_fixture(tmp_path, monkeypatch)
    receipt = artifact_cache.write_stage_receipt(
        out_dir=tmp_path, stage="event_inference", command=[], inputs=[], outputs=[inputs.events]
    )
    document = json.loads(receipt.read_text())
    identity = document["identity"]
    identity.update(historical, inputs=[binding])
    identity["fingerprint"] = artifact_cache._digest_json(
        {k: v for k, v in identity.items() if k != "fingerprint"}
    )
    _json(receipt, document)
    _bind(inputs, extra=[receipt])
    records = validate_ancestry(inputs, [inputs.events])
    verified = [r for r in records if "historical_implementation_verification" in r]
    assert len(verified) == 1
    assert verified[0]["historical_implementation_verification"][0]["sha256"] == binding["sha256"]
    # Ordinary observed data still requires its exact current bytes.
    inputs.events.write_text("[]")
    with pytest.raises(ValueError, match="digest changed"):
        validate_ancestry(inputs, [inputs.events])


def test_verified_ancestry_metadata_does_not_look_like_changed_file(inputs, tmp_path, monkeypatch):
    from cv.pipeline.s6_automatic_observations import ancestry_unchanged, validate_ancestry

    binding, historical = historical_code_fixture(tmp_path, monkeypatch)
    receipt = artifact_cache.write_stage_receipt(
        out_dir=tmp_path, stage="event_inference", command=[], inputs=[], outputs=[inputs.events]
    )
    document = json.loads(receipt.read_text())
    identity = document["identity"]
    identity.update(historical, inputs=[binding])
    identity["fingerprint"] = artifact_cache._digest_json(
        {k: v for k, v in identity.items() if k != "fingerprint"}
    )
    _json(receipt, document)
    _bind(inputs, extra=[receipt])
    records = validate_ancestry(inputs, [inputs.events])
    assert any("historical_implementation_verification" in r for r in records)
    assert ancestry_unchanged(records)
    receipt.write_text(receipt.read_text() + "\n")
    assert not ancestry_unchanged(records)


def test_native_padding_rows_remain_bound_but_cannot_enter_fitting(inputs, tmp_path):
    with inputs.ball.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["frame"] == "f_0006.jpg":
            row["x"], row["y"] = "1028", "1374"
    _csv(inputs.ball, rows)
    _bind(inputs)
    raw_hash = provenance.file_sha256(inputs.ball)
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "prepared"
    evidence = json.loads((tmp_path / "prepared/observed_evidence.json").read_text())
    consumer = json.loads((tmp_path / "prepared/observations.json").read_text())
    frame = next(r for r in consumer["ball"]["records"][0]["frames"] if r["frame"] == 6)
    assert frame["status"] == "unsupported"
    assert (frame["x1080"], frame["y1080"]) == (1028, 1374)
    assert evidence["native_image_support"]["refused_rows"] == 1
    assert provenance.file_sha256(inputs.ball) == raw_hash
    proof = json.loads((tmp_path / "prepared/automatic_provenance.json").read_text())
    assert any(r["sha256"] == raw_hash for r in proof["reused_artifacts"])
    from cv.experiments.connected_shooting.agent_whole_point_search import observation_accounting

    assert observation_accounting([frame])["status_counts"] == {"unsupported": 1}


def test_first_flight_prefix_retains_original_events_video_and_shared_contract(inputs, tmp_path):
    from cv.experiments.connected_shooting import observation_scope, real_exposure_replay
    from cv.pipeline import s6_labeled_stage, s6_preparation_policy

    source = json.loads(inputs.events.read_text())
    first = source[0]
    ground = dict(
        source[1], frame=10, location={"frame_subpixel": 10.2}, frame_interval=[9.5, 10.5]
    )
    last = dict(first, frame=18, location={"frame_subpixel": 18.2}, frame_interval=[17.5, 18.5])
    _json(inputs.events, [first, ground, last])
    _bind(inputs)
    off = build_observations(inputs, tmp_path / "off_prefix")
    assert off["status"] == "preparation_held"
    on = build_observations(inputs, tmp_path / "on_prefix", observed_first_flight="on")
    assert on["status"] == "prepared"
    resolved = s6_labeled_stage.validate_row(on["row"])
    packet = json.loads(resolved["packet"].read_text())
    labels = json.loads(resolved["labels"].read_text())
    attempt = packet["attempts"][0]
    assert len(attempt["events"]) == 2 and len(labels["events"]["records"]) == 3
    assert attempt["owner_end_frame"] == 17
    assert len(labels["source_pack"]["images"]) == 20
    assert len(labels["ball"]["records"][0]["frames"]) == 20
    assert labels["attempt"]["ending_kind"] == "unresolved"
    assert "ending_frame" not in labels["attempt"]
    observation_scope.validate(attempt, labels)
    prepared, _ = s6_preparation_policy.prepare_packet(packet, labels, "on")
    observation_scope.validate(prepared["attempts"][0], labels)
    from cv.experiments.connected_shooting import agent_whole_point_search as whole

    scene, _, bounces, _, _ = whole.prepare_attempt(
        attempt,
        json.loads(resolved["cameras"].read_text()),
        "grass",
        observation_fallback=True,
    )
    assert len(scene.pixels) == 1
    np.testing.assert_array_equal(scene.contact_frames, [4.2, 17.0])
    np.testing.assert_array_equal(bounces[0], [10.2])
    barrier = real_exposure_replay.next_physical_event(labels, "pt0001", 10.2)
    assert barrier["frame"] == 18.2
    assert json.loads(inputs.events.read_text()) == [first, ground, last]
    assert on["first_flight_scope"]["original_preparation_hold"]
    # The ordinary already-accepted path remains byte-equivalent when the option is ON.
    _json(inputs.events, source)
    _bind(inputs)
    ordinary = build_observations(inputs, tmp_path / "ordinary_on", observed_first_flight="on")
    assert ordinary["status"] == "prepared" and ordinary.get("first_flight_scope") is None
    baseline = build_observations(inputs, tmp_path / "ordinary_off")
    for name in ("labels", "cameras"):
        a = s6_labeled_stage.validate_row(ordinary["row"])[name]
        b = s6_labeled_stage.validate_row(baseline["row"])[name]
        assert a.read_bytes() == b.read_bytes()


def test_optional_contact_packet_reaches_shared_stage_with_source_binding(inputs, tmp_path):
    from cv.pipeline import s6_labeled_stage as stage

    result = build_observations(inputs, tmp_path / "optional", optional_contacts="on")
    row = result["row"]
    packet = json.loads(stage.resolve(row["packet"]).read_text())
    assert "optional_contact_witness" in packet
    witness = json.loads(stage.resolve(packet["optional_contact_witness"]).read_text())
    assert witness["source_attempt_events"] == packet["attempts"][0]["events"]
    assert witness["source_events"]["sha256"] == provenance.file_sha256(inputs.events)
    assert witness["candidates"] == []
    policy = {
        "optional_contacts": "on",
        "coarse_iterations": 1,
        "refine_iterations": 1,
        "search_seconds": 3,
        "athlete_evidence": "optional",
        "serve_location_prior": provenance.file_record(inputs.source_video),
    }
    output = tmp_path / "stage"
    output.mkdir()
    command = stage.search_arguments(row, stage.validate_row(row), output, policy)
    assert command[command.index("--optional-contact-witness") + 1] == str(
        stage.resolve(packet["optional_contact_witness"])
    )


def multi_command_code_fixture(tmp_path, monkeypatch):
    import hashlib
    import subprocess

    binding, old = historical_code_fixture(tmp_path, monkeypatch)
    repo = paths.REPO_ROOT
    originals = {
        "producer.py": b"import helper\n",
        "second.py": b"import helper\n",
        "helper.py": b"VALUE = 1\n",
    }
    for name, content in originals.items():
        (repo / name).write_bytes(content)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Record complete commands",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    records = [
        dict(
            kind="file",
            path_base="repository",
            path=name,
            bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for name, content in originals.items()
    ]
    identity = dict(
        code=records[:1],
        inputs=deepcopy(records),
        configuration=dict(
            source_commit=commit,
            receipt_timing="retrospective_completed_direct_commands",
            executed_commands=[["python", "-m", "producer"], ["python", "-m", "second"]],
        ),
    )
    for name in originals:
        (repo / name).write_text("# changed consumer checkout\n")
    return records, identity


def test_multi_command_historical_code_replays_full_original_union(tmp_path, monkeypatch):
    from cv.pipeline.s6_automatic_observations import historical_implementation_input

    records, identity = multi_command_code_fixture(tmp_path, monkeypatch)
    for binding in records:
        proof = historical_implementation_input(binding, identity)
        assert proof["verification"] == "git_command_closure" and proof["closure_files"] == 3
    # A Python-looking model/data input has no historical exemption.
    model = dict(records[0], path="model_weights.py")
    identity["inputs"].append(model)
    assert historical_implementation_input(model, identity) is None


@pytest.mark.parametrize(
    "bad", ["blob", "missing_dependency", "unrecorded_entry", "model_as_code", "missing_revision"]
)
def test_multi_command_closure_rejects_incomplete_or_fabricated_history(tmp_path, monkeypatch, bad):
    from cv.pipeline.s6_automatic_observations import historical_implementation_input

    records, identity = multi_command_code_fixture(tmp_path, monkeypatch)
    if bad == "blob":
        identity["inputs"][2]["sha256"] = "f" * 64
    elif bad == "missing_dependency":
        identity["inputs"].pop()
    elif bad == "unrecorded_entry":
        identity["configuration"]["executed_commands"].pop()
    elif bad == "model_as_code":
        identity["code"].append(dict(records[0], path="model_weights.py"))
    elif bad == "missing_revision":
        identity["configuration"]["source_commit"] = "f" * 40
    if bad == "unrecorded_entry":
        # A code file outside the recorded commands cannot masquerade as an imported dependency.
        identity["code"].append(records[1])
    with pytest.raises(ValueError):
        historical_implementation_input(records[0], identity)


@pytest.mark.parametrize(
    "marker",
    [
        {"manual_overrides": ["changed"]},
        {"labels_loaded": True},
        {"human_derived": True},
        {"human_derived_inputs": ["changed"]},
        {"annotation_origin": "agent"},
        {"annotation_origin": "human"},
        {"annotation_origin": "reviewed"},
        {"annotation_origin": "manual"},
    ],
)
def test_source_origin_checks_reject_deeply_nested_markers(marker):
    from cv.pipeline.s6_automatic_observations import _reject_assistance

    document = {"events": [{"native": {"evidence": [0, None, {"detail": marker}]}}]}
    with pytest.raises((ValueError, provenance.ProvenanceError)):
        _reject_assistance(document, "nested")


def test_source_origin_checks_scan_forbidden_tree_once_without_changing_allowed_values(monkeypatch):
    from cv.pipeline.s6_automatic_observations import _reject_assistance

    original = provenance.assert_automatic_document
    calls = []

    def checked(document, *, context):
        calls.append(document)
        original(document, context=context)

    monkeypatch.setattr(provenance, "assert_automatic_document", checked)
    allowed = {
        "manual_overrides": [],
        "labels_loaded": False,
        "human_derived": False,
        "annotation_origin": "automatic",
        "children": [{"ball": [{"x": 3.5, "y": 8.2} for _ in range(80)]}, None],
    }
    _reject_assistance(allowed, "allowed")
    assert calls == [allowed]


@pytest.mark.parametrize("physical_ending", [False, True])
def test_contact_prefix_preserves_original_active_observation_accounting(
    inputs, tmp_path, physical_ending
):
    source = json.loads(inputs.events.read_text())
    first = source[0]
    ground = dict(
        source[1], frame=10, location={"frame_subpixel": 10.2}, frame_interval=[9.5, 10.5]
    )
    last = dict(first, frame=18, location={"frame_subpixel": 18.2}, frame_interval=[17.5, 18.5])
    events = [first, ground, last]
    if physical_ending:
        events.extend(
            [
                dict(
                    source[1],
                    frame=20,
                    location={"frame_subpixel": 19.5},
                    frame_interval=[19.0, 20.0],
                ),
                dict(
                    source[2],
                    frame=20,
                    location={"frame_subpixel": 19.5},
                    frame_interval=[19.0, 20.0],
                ),
            ]
        )
    _json(inputs.events, events)
    _bind(inputs)
    output = tmp_path / "contact_prefix"
    result = build_observations(
        inputs, output, contact_prefix_scope="coverage", observed_horizon_tail="on"
    )
    assert result["status"] == "prepared", result.get("reason")
    attempt = json.loads((output / "packet.json").read_text())["attempts"][0]
    consumer = json.loads((output / "observations.json").read_text())
    scope = attempt["observation_scope"]
    assert scope["right_boundary_kind"] == "original_contact"
    original = [
        r
        for r in consumer["ball"]["records"][0]["frames"]
        if first["location"]["frame_subpixel"] <= r["frame"] <= scope["original_owner_end_frame"]
    ]
    assert attempt["owner_ball_labels"] == attempt["ball_observations"] == original
    assert attempt["visible_native_frames"] == sum(r["status"] == "visible" for r in original)
    assert not set(attempt["context_native_frames"]) & {r["frame"] for r in original}
    assert any(r["frame"] > scope["modeled_horizon"] for r in original)
    assert len(consumer["ball"]["records"][0]["frames"]) == 20


def _leading_events(inputs):
    """The same automatic stream opening on an accepted bounce before its contacts."""
    template = next(
        e for e in json.loads(inputs.events.read_text()) if e["event_type"] == "contact"
    )
    rows = []
    for kind, rounded, fine, interval in [
        ("bounce", 2, 2.4, [2.0, 3.0]),
        ("contact", 4, 4.2, [4.0, 5.0]),
        ("bounce", 9, 9.4, [9.0, 10.0]),
        ("contact", 13, 13.2, [13.0, 14.0]),
        ("bounce", 18, 17.6, [17.0, 18.0]),
    ]:
        row = deepcopy(template)
        row.update(
            event_type=kind,
            frame=rounded,
            frame_interval=interval,
            location={"frame_subpixel": fine},
        )
        rows.append(row)
    _json(inputs.events, rows)
    _bind(inputs)
    return rows[0]


def test_leading_automatic_evidence_is_refused_by_default(inputs, tmp_path):
    _leading_events(inputs)
    result = build_observations(inputs, tmp_path / "prepared")
    assert result["status"] == "preparation_held"
    assert "needs an originating contact" in result["reason"]


def test_leading_automatic_evidence_needs_both_dependent_policies(inputs, tmp_path):
    _leading_events(inputs)
    for extra in (
        {"leading_event_components": "declared_prefix"},
        {"leading_event_components": "declared_prefix", "contact_components": "unresolved_ending"},
        {
            "leading_event_components": "declared_prefix",
            "contact_prefix_scope": "unresolved_ending",
        },
    ):
        with pytest.raises(ValueError, match="leading physical evidence requires"):
            build_observations(inputs, tmp_path / f"held_{len(extra)}_{len(str(extra))}", **extra)


def test_leading_automatic_evidence_is_declared_under_the_explicit_policy(inputs, tmp_path):
    leading = _leading_events(inputs)
    original = json.loads(inputs.events.read_text())
    result = build_observations(
        inputs,
        tmp_path / "prepared",
        contact_prefix_scope="unresolved_ending",
        contact_components="unresolved_ending",
        leading_event_components="declared_prefix",
    )
    assert result["status"] == "prepared", result.get("reason")
    packet = json.loads((tmp_path / "prepared/packet.json").read_text())
    evidence = json.loads((tmp_path / "prepared/observed_evidence.json").read_text())
    attempt = packet["attempts"][0]
    # The adapter binds the same unresolved-ending contact prefix as the
    # supplied-clip runtime; the declared inventory is its original scope.
    assert attempt["observation_scope"]["schema"] == "s6_observed_contact_prefix_v1"
    contract = attempt["original_observation_scope"]
    assert contract["schema"] == "s6_unresolved_ending_input_v1"
    assert contract["physical_ending"] is None and evidence["physical_ending"] is None
    assert contract["segmentation_source"]["record"]["sha256"] == provenance.file_sha256(
        inputs.point_ledger
    )
    declaration = contract["leading_physical_prefix"]
    assert declaration["event_count"] == 1 and declaration["dropped_original_rows"] == 0
    assert declaration["fabricated_events"] is False and declaration["modelled"] is False
    assert declaration["origin_status"] == "unknown"
    assert declaration["role_candidates"] == ["serve", "rally"]
    assert "originating contact" in declaration["source_refusal"]
    # The leading row survives verbatim at its own native epoch, and the first
    # modelled contact still opens the owned observation domain.
    assert declaration["events"][0]["frame"] == 2.4
    assert [e["event_type"] for e in attempt["events"]] == [
        "bounce",
        "contact",
        "bounce",
        "contact",
    ]
    assert attempt["events"][0]["frame"] == 2.4
    assert attempt["first_event_frame"] == 4.2
    assert attempt["contact_count"] == 2
    assert attempt["original_physical_events"][0]["frame"] == 2.4
    # No source row was moved, dropped or invented anywhere in this preparation.
    assert json.loads(inputs.events.read_text()) == original
    assert original[0]["frame"] == leading["frame"]
    assert result["runtime_model_calls"] == 0 and result["fitted_states_read"] is False
    # The automatic packet reaches ordinary component preparation with an
    # unknown origin, and the leading row stays outside every component.
    from cv.pipeline import s6_contact_components as components

    consumer = json.loads((tmp_path / "prepared/observations.json").read_text())
    cameras = json.loads((tmp_path / "prepared/cameras.json").read_text())
    plan = components.prepare(
        packet,
        consumer,
        cameras,
        mode=components.MODE,
        leading_event_components=components.LEADING_PREFIX,
    )
    assert plan["leading_physical_prefix_count"] == 1
    assert [c["first_contact_role"] for c in plan["components"]] == ["unknown"]
    assert plan["components"][0]["events"][0]["event_type"] == "contact"
    assert plan["complete_original_source"] is False
