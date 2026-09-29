"""The ordinary runner's own optional-contact pose arm, from CLI flag to forwarded input."""

import json
import sys

import pytest

from cv.pipeline import broadcast_runner, s6_broadcast_backend as backend
from cv.pipeline.artifact_cache import write_stage_receipt
from cv.pipeline.s6_broadcast_backend import optional_pose_ancestry

POSE_NAME = "player_pose_optional_contact_native_v1.csv"
RECEIPTED_STAGES = ("postseg_base", "pose_player_crop")


class Stopped(RuntimeError):
    """Raised at the first stage after the independent pose arm, to end the run there."""


def _materialize(name: str, outputs: list) -> None:
    if name == "point_ledger":
        outputs[0].write_text("pt,rally_t_start,rally_t_end,confidence\n1,10,20,high\n")
        outputs[1].write_text('{"schema":"automatic_point_ledger_v1"}\n')
    elif name == "postseg_materialization":
        outputs[0].write_text('{"points":[]}\n')
        outputs[1].write_text(
            json.dumps({"matches": [{"id": "match_a", "source_fps": 25.0, "point_ids": [1]}]})
        )
    elif name == "postseg_base":
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("camera\n" if path.suffix == ".npz" else "clip,frame,side,conf\n")
        boxes = outputs[-1]
        boxes.with_suffix(boxes.suffix + ".coordinates.json").write_text('{"fps":25}\n')
        (boxes.parent / "audit_frames_native_1080.coordinates.json").write_text('{"fps":25}\n')
        clip = boxes.parent / "audit_frames_native_1080" / "pt0001"
        clip.mkdir(parents=True, exist_ok=True)
        (clip / "extraction_receipt.json").write_text(
            json.dumps(
                {
                    "schema": "player_frame_extraction_v1",
                    "identity": {"point": {"pt": 1}},
                    "source_identity": {"fingerprint": "abc"},
                    "frames": [{"name": "f_0000000040.jpg", "sha256": "0" * 64}],
                }
            )
        )
        receipts = boxes.parent / "run_manifests" / "stage_receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        for stage in ("frame_extraction", "player_detection", "player_side_association"):
            (receipts / f"{stage}.json").write_text('{"schema":"canonical"}\n')
    elif name == "pose_player_crop":
        pose, coordinates, coverage = outputs
        pose.write_text("clip,frame,side\npt0001,f_0000000040.jpg,near\n")
        coordinates.write_text('{"image_size":{"width":1920,"height":1080}}\n')
        coverage.write_text(
            json.dumps(
                {
                    "schema": "tracked_crop_pose_coverage_v1",
                    "observation_scope": "retained_native",
                    "requested_rows": 3,
                    "presented_rows": 2,
                    "decode_failed_rows": 1,
                    "pose_rows": 1,
                }
            )
        )


def _recording_runner(monkeypatch, stages: list) -> None:
    """A runner that writes the real stage receipts the producer binding depends on."""

    class RecordingRunner:
        def __init__(self, out_root, resume) -> None:
            self.out_root = out_root
            self.receipts: dict = {}

        @property
        def upstream_receipts(self):
            return list(self.receipts.values())

        def stage(self, name, command, inputs, outputs, callback=None, dependencies=()):
            stages.append(
                dict(
                    name=name,
                    command=[str(value) for value in command],
                    inputs=list(inputs),
                    outputs=list(outputs),
                    dependencies=tuple(dependencies),
                )
            )
            if name == "event_inference":
                raise Stopped(name)
            _materialize(name, outputs)
            if name in RECEIPTED_STAGES:
                self.receipts[name] = write_stage_receipt(
                    out_dir=self.out_root,
                    stage=name,
                    command=command,
                    inputs=inputs,
                    outputs=outputs,
                    upstream_receipts=[
                        self.receipts[d] for d in dependencies if d in self.receipts
                    ],
                )

    monkeypatch.setattr(broadcast_runner, "Runner", RecordingRunner)


def _scene(monkeypatch, tmp_path, *, healed: bool = False) -> dict:
    """A data root with the shipped weights, a source video and an audited cadence."""
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    models = tmp_path / "models" / "pipeline"
    models.mkdir(parents=True)
    (models / "yolov8m.pt").write_bytes(b"person-detector")
    (models / "yolo26m-pose.pt").write_bytes(b"pose-model")
    finetune = models / "ball_finetune_v1"
    finetune.mkdir()
    for model in ("wasb", "tracknetv2"):
        (finetune / f"{model}_native_crop_ep3.pth.tar").write_bytes(b"ball-detector")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"broadcast")
    event_model = tmp_path / "event_video_model.pt"
    event_model.write_bytes(b"events")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"optional_contacts": "on"}')
    match_out = tmp_path / "run" / "match_a"
    match_out.mkdir(parents=True)
    cadence = {"decision": "usable", "processing_fps": 25.0}
    processing = video
    if healed:
        processing = match_out / "normalized_v1.mp4"
        processing.write_bytes(b"normalized")
        cadence = {
            "decision": "healed",
            "processing_fps": 25.0,
            "normalized_video": processing.name,
        }
    (match_out / "source_integrity.json").write_text(json.dumps({"cadence": cadence, "fps": 25.0}))
    monkeypatch.setattr(broadcast_runner, "probe", lambda _: {"fps": 25.0})
    return dict(
        video=video,
        processing_video=processing,
        out=tmp_path / "run",
        match_out=match_out,
        event_model=event_model,
        policy=policy_path,
    )


def _arguments(scene, *, flags=(), backend_name="shared_s6") -> list[str]:
    return [
        "broadcast_runner",
        "--video",
        str(scene["video"]),
        "--out",
        str(scene["out"]),
        "--match-id",
        "match_a",
        "--surface",
        "hard",
        "--segmentation",
        "legacy",
        "--event-model",
        str(scene["event_model"]),
        "--s6-backend",
        backend_name,
        "--shared-s6-policy",
        str(scene["policy"]),
        *flags,
    ]


def _run(monkeypatch, scene, *, flags=(), policy=None, backend_name="shared_s6"):
    """Drive the real CLI to the first stage after the independent pose arm."""
    stages: list = []
    captured: dict = {}
    _recording_runner(monkeypatch, stages)
    monkeypatch.setattr(backend, "load_policy", lambda _: policy or {"optional_contacts": "on"})
    real_run = broadcast_runner.run

    def capture(args):
        captured["args"] = args
        return real_run(args)

    monkeypatch.setattr(broadcast_runner, "run", capture)
    monkeypatch.setattr(sys, "argv", _arguments(scene, flags=flags, backend_name=backend_name))
    return stages, captured


def _forwarded(monkeypatch, tmp_path, args, processing_video) -> dict:
    captured: dict = {}
    monkeypatch.setattr(
        backend, "run_cached", lambda **kwargs: captured.update(kwargs) or tmp_path / "summary"
    )
    broadcast_runner.run_shared_s6_backend(args, processing_video)
    return captured


def _named(stages, name):
    return [stage for stage in stages if stage["name"] == name]


def test_default_run_has_no_independent_pose_stage_and_forwards_nothing(monkeypatch, tmp_path):
    scene = _scene(monkeypatch, tmp_path)
    stages, captured = _run(monkeypatch, scene)

    with pytest.raises(Stopped):
        broadcast_runner.main()

    assert _named(stages, "pose_player_crop") == []
    assert _named(stages, "physical_player_motion") == []
    assert not (scene["match_out"] / POSE_NAME).exists()
    forwarded = _forwarded(monkeypatch, tmp_path, captured["args"], scene["video"])

    # The default-off call is the call it always was: the keywords are absent, not None.
    assert "optional_contact_pose" not in forwarded
    assert "optional_contact_pose_provenance" not in forwarded
    assert forwarded["camera_name"] == "camera_P_per_frame_v1.npz"


def test_enabled_arm_runs_one_ordinary_stage_and_forwards_its_actual_receipt(monkeypatch, tmp_path):
    scene = _scene(monkeypatch, tmp_path)
    stages, captured = _run(monkeypatch, scene, flags=["--shared-s6-independent-contact-pose"])

    with pytest.raises(Stopped):
        broadcast_runner.main()

    produced = _named(stages, "pose_player_crop")
    assert len(produced) == 1
    stage = produced[0]
    assert stage["dependencies"] == ("postseg_base",)
    command = stage["command"]
    assert command[1:3] == ["-m", "cv.pipeline.pose_player_crop"]
    assert command[command.index("--observation-scope") + 1] == "retained_native"
    assert command[command.index("--output") + 1] == POSE_NAME
    assert command[command.index("--boxes") + 1] == "player_boxes_25_native_sided_v1.csv"
    assert "--active-play" not in command
    pose = scene["match_out"] / POSE_NAME
    producer = scene["match_out"] / f"{pose.stem}.producer.json"
    assert stage["outputs"][0] == pose
    # The original per-clip extraction receipts decide which pictures the scope may read,
    # so a re-extraction invalidates this stage through its declared inputs.
    frames = scene["match_out"] / "audit_frames_native_1080"
    assert frames / "pt0001" / "extraction_receipt.json" in stage["inputs"]

    # The shared validator, not a local re-implementation, has to accept the binding.
    receipt, parent = optional_pose_ancestry(pose, producer, scene["video"])
    assert receipt == scene["match_out"] / "run_manifests/stage_receipts/pose_player_crop.json"
    assert parent == producer
    document = json.loads(producer.read_text())
    configuration = document["configuration"]
    assert configuration["observation_scope"] == "retained_native"
    assert configuration["receipt_written_by_runner_stage"] is True
    assert configuration["coverage"]["decode_failed_rows"] == 1
    assert configuration["coverage"]["pose_rows"] == 1
    assert document["human_inputs"] == []
    # The ordinary frame, player and side receipts this scope descends from stay bound.
    bound = [r["path"] for r in document["reused_artifacts"]]
    for stage_name in ("frame_extraction", "player_detection", "player_side_association"):
        assert any(path.endswith(f"{stage_name}.json") for path in bound)

    manifest = json.loads(
        (scene["match_out"] / "run_manifests" / "independent_contact_pose.json").read_text()
    )
    assert manifest["extraction_receipts_declared"] == 1
    assert manifest["observation_scope"] == "retained_native"

    forwarded = _forwarded(monkeypatch, tmp_path, captured["args"], scene["video"])
    assert forwarded["optional_contact_pose"] == pose
    assert forwarded["optional_contact_pose_provenance"] == producer
    assert forwarded["camera_name"] == "camera_P_per_frame_v1.npz"


def test_enabling_the_arm_adds_the_pose_stage_and_changes_no_other_command(monkeypatch, tmp_path):
    scene = _scene(monkeypatch, tmp_path)
    default_stages, _ = _run(monkeypatch, scene)
    with pytest.raises(Stopped):
        broadcast_runner.main()
    enabled_stages, _ = _run(monkeypatch, scene, flags=["--shared-s6-independent-contact-pose"])
    with pytest.raises(Stopped):
        broadcast_runner.main()

    def declared(stages):
        return [(s["name"], s["command"], s["dependencies"]) for s in stages]

    assert [row for row in declared(enabled_stages) if row[0] != "pose_player_crop"] == declared(
        default_stages
    )
    assert len(_named(enabled_stages, "pose_player_crop")) == 1


def test_physical_and_independent_arms_keep_separate_artifacts_and_camera(monkeypatch, tmp_path):
    scene = _scene(monkeypatch, tmp_path)
    stages, captured = _run(
        monkeypatch,
        scene,
        flags=["--shared-s6-independent-contact-pose", "--physical-player-motion"],
    )

    with pytest.raises(Stopped):
        broadcast_runner.main()

    physical = _named(stages, "physical_player_motion")[0]
    independent = _named(stages, "pose_player_crop")[0]
    physical_pose = scene["match_out"] / "player_pose_tracked_crop_native_v1.csv"
    assert physical["outputs"][0] == physical_pose
    assert independent["outputs"][0] == scene["match_out"] / POSE_NAME
    assert not set(physical["outputs"]) & set(independent["outputs"])
    # The physical arm keeps its active-play scope and its own artifact name.
    assert "--observation-scope" not in physical["command"]
    assert physical_pose.name not in independent["command"]

    forwarded = _forwarded(monkeypatch, tmp_path, captured["args"], scene["video"])
    assert forwarded["camera_name"] == "camera_P_metric_v1.npz"
    assert forwarded["optional_contact_pose"] == scene["match_out"] / POSE_NAME


def test_healed_processing_video_binds_both_identities_without_inventing_one(monkeypatch, tmp_path):
    scene = _scene(monkeypatch, tmp_path, healed=True)
    stages, _ = _run(monkeypatch, scene, flags=["--shared-s6-independent-contact-pose"])

    with pytest.raises(Stopped):
        broadcast_runner.main()

    assert _named(stages, "pose_player_crop")
    pose = scene["match_out"] / POSE_NAME
    producer = scene["match_out"] / f"{pose.stem}.producer.json"
    document = json.loads(producer.read_text())

    (processing,) = document["source_videos"]
    assert processing["role"] == "processing_video"
    assert processing["path"].endswith("normalized_v1.mp4")
    assert document["configuration"]["processing_video_is_parent_source"] is False
    records = {record["role"]: record for record in document["reused_artifacts"]}
    roles = [record["role"] for record in document["reused_artifacts"]]
    assert roles.count("automatic_pose_inference_receipt") == 1
    # The original broadcast identity stays bound, and the audited normalization -- not a
    # claim about the original broadcast -- is what attests the processing stream.
    assert records["original_automatic_source_broadcast"]["path"].endswith("source.mp4")
    assert records["automatic_processing_source_audit"]["path"].endswith("source_integrity.json")

    # The declared processing video is the one the shared stage is actually given.
    receipt, _ = optional_pose_ancestry(pose, producer, scene["processing_video"])
    assert receipt.name == "pose_player_crop.json"
    with pytest.raises(ValueError, match="another processing video"):
        optional_pose_ancestry(pose, producer, scene["video"])


def test_independent_pose_is_rejected_before_any_stage_runs(monkeypatch, tmp_path):
    scene = _scene(monkeypatch, tmp_path)
    stages, _ = _run(
        monkeypatch,
        scene,
        flags=["--shared-s6-independent-contact-pose"],
        policy={"optional_contacts": "off"},
    )

    with pytest.raises(ValueError, match="optional_contacts on"):
        broadcast_runner.main()

    assert stages == []


def test_independent_pose_requires_the_shared_backend(monkeypatch, tmp_path, capsys):
    scene = _scene(monkeypatch, tmp_path)
    stages, _ = _run(
        monkeypatch,
        scene,
        flags=["--shared-s6-independent-contact-pose"],
        backend_name="legacy",
    )

    with pytest.raises(SystemExit):
        broadcast_runner.main()

    assert "--s6-backend shared_s6" in capsys.readouterr().err
    assert stages == []
