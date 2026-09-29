import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import canonical_runner
from cv.pipeline.paths import data_root


def test_player_weights_resolve_in_shared_model_store():
    assert canonical_runner.YOLO_WEIGHTS == data_root() / "models/pipeline/yolov8m.pt"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_court_worker_counts_fail_before_inference(value):
    with pytest.raises(ValueError, match="positive integer"):
        canonical_runner.resolve_court_jobs("postseg", value)


@pytest.mark.parametrize(
    "profile,requested,expected",
    [("postseg", None, 4), ("postseg", 16, 16), ("motion", None, 16), ("motion", 2, 2)],
)
def test_court_worker_count_reaches_command_and_provenance(
    tmp_path, monkeypatch, profile, requested, expected
):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, profile=profile, court_jobs=requested
    )
    court = next(command for command in commands if "cv.pipeline.court_topology_runner" in command)
    if requested is None and profile == "postseg":
        assert "--jobs" not in court  # preserve the old default command/receipt
    else:
        assert court[court.index("--jobs") + 1] == str(expected)
    document = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert document["provenance"]["configuration"]["court_jobs"] == expected


def test_canonical_runner_omits_legacy_contact_and_event_exports(tmp_path, monkeypatch) -> None:
    match_out = tmp_path / "match"
    match_out.mkdir()
    (match_out / canonical_runner.REEL_NAME).write_bytes(b"video")
    (match_out / canonical_runner.POINT_MAP_NAME).write_text("pt,rally_t_start,rally_t_end\n")
    commands = []
    monkeypatch.setattr(canonical_runner, "_run", commands.append)
    monkeypatch.setattr(canonical_runner, "require_paths", lambda _: None)

    def run_cached(**kwargs):
        commands.append(kwargs["command"])
        for output in kwargs["outputs"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".npz":
                np.savez_compressed(output, pts=np.array([]), H=np.empty((0, 3, 3)))
            else:
                output.write_text("output")
        path = canonical_runner.receipt_path(match_out, kwargs["stage"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    monkeypatch.setattr(canonical_runner, "_run_cached_command", run_cached)
    monkeypatch.setattr(canonical_runner, "stage_receipt_matches", lambda **_: False)
    monkeypatch.setattr(canonical_runner, "remove_receipt", lambda *_: None)

    def write_receipt(**kwargs):
        path = canonical_runner.receipt_path(match_out, kwargs["stage"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    monkeypatch.setattr(canonical_runner, "write_stage_receipt", write_receipt)

    def expand(_match_out: Path, _frames_dir: str) -> Path:
        path = match_out / "camera_P_per_frame_v1.npz"
        np.savez_compressed(path, clips=np.array(["pt0001"]))
        return path

    monkeypatch.setattr(canonical_runner, "expand_point_cameras", expand)

    def bundle(_match_out: Path, _frames_dir: str) -> Path:
        path = expand(_match_out, _frames_dir)
        (match_out / "camera_bundle_v1.json").write_text("{}")
        return path

    monkeypatch.setattr(canonical_runner, "bundle_match_cameras", bundle)
    canonical_runner.run_match(
        {"id": "match", "source_fps": 25.0, "surface": "hard"},
        tmp_path,
        0,
    )

    rendered = "\n".join(" ".join(command) for command in commands)
    assert "contacts_longitudinal" not in rendered
    assert "contacts_fused" not in rendered
    assert "contacts_serve" not in rendered
    assert "contacts_player_gate" not in rendered
    assert "track_arcs" not in rendered
    assert "ball_events" not in rendered
    assert "audio_evidence.py" not in rendered
    assert "-m cv.pipeline.run_contacts_pipeline" in rendered
    assert "-m cv.pipeline.court" in rendered
    assert "-m cv.pipeline.camera_cal" in rendered
    assert "-m cv.pipeline.player_side_association" in rendered
    document = __import__("json").loads(
        (match_out / "run_manifests" / "canonical_runner.json").read_text()
    )
    assert document["provenance"]["mode"] == "automatic"
    assert document["match"] == "match"


def _run_match_capturing_commands(
    tmp_path, monkeypatch, *, camera_clips=None, **run_match_kwargs
) -> tuple[list, Path]:
    match_out = tmp_path / "match"
    match_out.mkdir()
    (match_out / canonical_runner.REEL_NAME).write_bytes(b"video")
    (match_out / canonical_runner.POINT_MAP_NAME).write_text("pt,rally_t_start,rally_t_end\n")
    commands = []
    monkeypatch.setattr(canonical_runner, "_run", commands.append)
    monkeypatch.setattr(canonical_runner, "require_paths", lambda _: None)

    def run_cached(**kwargs):
        commands.append(kwargs["command"])
        for output in kwargs["outputs"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".npz":
                np.savez_compressed(output, pts=np.array([]), H=np.empty((0, 3, 3)))
            else:
                output.write_text("output")
        path = canonical_runner.receipt_path(match_out, kwargs["stage"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    monkeypatch.setattr(canonical_runner, "_run_cached_command", run_cached)
    monkeypatch.setattr(canonical_runner, "stage_receipt_matches", lambda **_: False)
    monkeypatch.setattr(canonical_runner, "remove_receipt", lambda *_: None)

    def write_receipt(**kwargs):
        path = canonical_runner.receipt_path(match_out, kwargs["stage"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    monkeypatch.setattr(canonical_runner, "write_stage_receipt", write_receipt)

    def expand(_match_out: Path, _frames_dir: str) -> Path:
        path = match_out / "camera_P_per_frame_v1.npz"
        np.savez_compressed(
            path, clips=np.array(["pt0001"] if camera_clips is None else camera_clips)
        )
        return path

    monkeypatch.setattr(canonical_runner, "expand_point_cameras", expand)

    def bundle(_match_out: Path, _frames_dir: str) -> Path:
        commands.append(["cv.pipeline.camera_bundle"])
        path = expand(_match_out, _frames_dir)
        (match_out / "camera_bundle_v1.json").write_text("{}")
        return path

    monkeypatch.setattr(canonical_runner, "bundle_match_cameras", bundle)
    canonical_runner.run_match(
        {"id": "match", "source_fps": 25.0, "surface": "hard"},
        tmp_path,
        0,
        **run_match_kwargs,
    )
    return commands, match_out


def _evidence_command(commands: list) -> list:
    return next(command for command in commands if "cv.pipeline.run_contacts_pipeline" in command)


def test_base_runner_emits_player_tracks_with_a_coordinate_contract(tmp_path, monkeypatch):
    commands, match_out = _run_match_capturing_commands(tmp_path, monkeypatch)
    command = next(c for c in commands if "cv.pipeline.player_side_association" in c)
    output = match_out / "player_tracks_native_v1.csv"
    assert command[command.index("--tracks-output") + 1] == str(output)
    assert output.is_file()
    assert output.with_suffix(".csv.coordinates.json").is_file()
    document = __import__("json").loads(
        (match_out / "run_manifests" / "canonical_runner.json").read_text()
    )
    assert output.name in {row["path"] for row in document["outputs"]}


def test_runner_default_threads_argmax_ball_decode(tmp_path, monkeypatch) -> None:
    commands, match_out = _run_match_capturing_commands(tmp_path, monkeypatch)
    evidence = _evidence_command(commands)

    assert evidence[evidence.index("--subpixel") + 1] == "argmax"
    document = __import__("json").loads(
        (match_out / "run_manifests" / "canonical_runner.json").read_text()
    )
    assert document["provenance"]["configuration"]["ball_subpixel_decode"] == "argmax"


def test_runner_threads_named_subpixel_alternative_and_records_provenance(
    tmp_path, monkeypatch
) -> None:
    commands, match_out = _run_match_capturing_commands(tmp_path, monkeypatch, subpixel="centroid")
    evidence = _evidence_command(commands)

    assert evidence[evidence.index("--subpixel") + 1] == "centroid"
    document = __import__("json").loads(
        (match_out / "run_manifests" / "canonical_runner.json").read_text()
    )
    assert document["provenance"]["configuration"]["ball_subpixel_decode"] == "centroid"


def test_runner_rejects_unknown_subpixel_decode(tmp_path, monkeypatch) -> None:
    import pytest

    with pytest.raises(ValueError, match="sub-pixel"):
        _run_match_capturing_commands(tmp_path, monkeypatch, subpixel="bilinear")


def test_runner_camera_bundle_is_explicit_and_recorded(tmp_path, monkeypatch) -> None:
    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, camera_model="bundle_v1"
    )

    document = __import__("json").loads(
        (match_out / "run_manifests" / "canonical_runner.json").read_text()
    )
    assert document["provenance"]["configuration"]["camera_model"] == "bundle_v1"
    assert (match_out / "camera_bundle_v1.json").is_file()
    receipt = __import__("json").loads(
        canonical_runner.receipt_path(match_out, "camera_expand").read_text()
    )
    del receipt
    assert any("cv.pipeline.camera_bundle" in command for command in commands)


def test_runner_rejects_unknown_camera_model(tmp_path, monkeypatch) -> None:
    import pytest

    with pytest.raises(ValueError, match="camera model"):
        _run_match_capturing_commands(tmp_path, monkeypatch, camera_model="free_camera")


def test_canonical_inference_modules_do_not_depend_on_validation_or_machine_paths() -> None:
    names = {
        "canonical_runner.py",
        "tracking_composition.py",
        "event_model_v3.py",
        "event_model_v2_features.py",
        "point_grammar.py",
        "reconstruction.py",
        "frame_cadence.py",
        "cadence_normalize.py",
        "camera_artifacts.py",
        "broadcast_runner.py",
        "broadcast_source.py",
        "point_ledger.py",
        "artifact_cache.py",
        "event_inference.py",
        "event_model.py",
    }

    for name in names:
        source = (canonical_runner.PIPELINE_DIR / name).read_text()
        tree = ast.parse(source)
        imports = [
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert all(not imported.startswith("cv.validation") for imported in imports)
        assert "/home/" not in source


def test_tracking_composition_does_not_emit_legacy_review_artifacts() -> None:
    source = (canonical_runner.PIPELINE_DIR / "tracking_composition.py").read_text()

    assert 'PIPELINE / "track_arcs.py"' not in source
    assert 'PIPELINE / "ball_events.py"' not in source
    assert 'PIPELINE / "player_side_association.py"' not in source


def test_base_manifest_does_not_claim_downstream_stages() -> None:
    capabilities = [name for name, _ in canonical_runner.BASE_STAGE_REGISTRY]

    assert "tracking_composition" not in capabilities
    assert "event_proposals" not in capabilities
    assert "reconstruction_3d" not in capabilities


def test_motion_profile_allows_only_point_scoped_cadence_holds(tmp_path, monkeypatch) -> None:
    match_out = tmp_path / "match"
    match_out.mkdir()
    (match_out / canonical_runner.REEL_NAME).write_bytes(b"video")
    (match_out / canonical_runner.POINT_MAP_NAME).write_text("pt,rally_t_start,rally_t_end\n")
    commands = []
    monkeypatch.setattr(canonical_runner, "_run", commands.append)
    monkeypatch.setattr(canonical_runner, "require_paths", lambda _: None)

    def run_cached(**kwargs):
        commands.append(kwargs["command"])
        for output in kwargs["outputs"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".npz":
                np.savez_compressed(output, pts=np.array([]), H=np.empty((0, 3, 3)))
            else:
                output.write_text("output")
        return canonical_runner.receipt_path(match_out, kwargs["stage"])

    monkeypatch.setattr(canonical_runner, "_run_cached_command", run_cached)
    monkeypatch.setattr(canonical_runner, "stage_receipt_matches", lambda **_: True)
    np.savez_compressed(match_out / "camera_P_per_frame_v1.npz", clips=np.array([]))

    canonical_runner.run_match(
        {"id": "match", "source_fps": 25.0, "surface": "hard"},
        tmp_path,
        0,
        profile="motion",
    )

    evidence = commands[0]
    assert "--skip-audio-evidence" in evidence
    assert "--allow-partial-timing-holds" in evidence
    assert "--skip-player-detection" in evidence
    player_commands = [
        command
        for command in commands
        if "cv.pipeline.run_contacts_pipeline" in command
        and "--skip-player-detection" not in command
    ]
    assert len(player_commands) == 1
    assert "--skip-audio-evidence" in player_commands[0]
    court_command = next(
        command for command in commands if "cv.pipeline.court_topology_runner" in command
    )
    jobs_index = court_command.index("--jobs")
    assert court_command[jobs_index + 1] == "16"


def test_default_profile_allows_point_scoped_cadence_holds(tmp_path, monkeypatch) -> None:
    commands, _ = _run_match_capturing_commands(tmp_path, monkeypatch)

    assert "--allow-partial-timing-holds" in _evidence_command(commands)


def test_substrate_module_cli_loads() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cv.pipeline.run_contacts_pipeline", "--help"],
        cwd=canonical_runner.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--preserve-source-fps" in result.stdout


@pytest.mark.parametrize("policy", ["off", "court_observation", "court_observation_fallback"])
def test_camera_mask_policy_reaches_court_command_and_receipt(tmp_path, monkeypatch, policy):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, court_registration_mask=policy
    )
    command = next(c for c in commands if "cv.pipeline.court_topology_runner" in c)
    if policy == "off":
        assert "--court-registration-mask" not in command
    else:
        assert command[command.index("--court-registration-mask") + 1] == policy
    receipt = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert receipt["provenance"]["configuration"]["court_registration_mask"] == policy


@pytest.mark.parametrize("policy", ["off", "qualify_retry"])
def test_interpolation_policy_reaches_camera_command_and_receipt(tmp_path, monkeypatch, policy):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path,
        monkeypatch,
        court_registration_mask="court_observation",
        interpolated_camera_policy=policy,
    )
    command = next(c for c in commands if "cv.pipeline.court_topology_runner" in c)
    assert command[command.index("--court-registration-mask") + 1] == "court_observation"
    if policy == "off":
        assert "--interpolated-camera-policy" not in command
    else:
        assert command[command.index("--interpolated-camera-policy") + 1] == policy
    receipt = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert receipt["provenance"]["configuration"]["interpolated_camera_policy"] == policy


@pytest.mark.parametrize("enabled", [False, True])
def test_keep_unique_admissible_frames_reaches_the_association_command(
    tmp_path, monkeypatch, enabled
):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, keep_unique_admissible_frames=enabled
    )
    sided = next(
        command for command in commands if "cv.pipeline.player_side_association" in command
    )
    assert ("--keep-unique-admissible-frames" in sided) is enabled
    assert ("--no-keep-unique-admissible-frames" in sided) is (not enabled)
    document = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert document["provenance"]["configuration"]["keep_unique_admissible_frames"] is enabled


def test_shot_homography_propagation_is_off_by_default(tmp_path, monkeypatch):
    commands, match_out = _run_match_capturing_commands(tmp_path, monkeypatch)
    sided = next(
        command for command in commands if "cv.pipeline.player_side_association" in command
    )
    assert "--no-propagate-shot-homography" in sided
    document = __import__("json").loads(
        (match_out / "run_manifests/canonical_runner.json").read_text()
    )
    assert document["provenance"]["configuration"]["propagate_shot_homography"] is False


def test_shot_homography_propagation_reaches_the_association_command(tmp_path, monkeypatch):
    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, propagate_shot_homography=True
    )
    sided = next(
        command for command in commands if "cv.pipeline.player_side_association" in command
    )
    assert "--propagate-shot-homography" in sided
    assert "--frames-root" in sided
    assert "--no-propagate-shot-homography" not in sided
    document = __import__("json").loads(
        (match_out / "run_manifests/canonical_runner.json").read_text()
    )
    assert document["provenance"]["configuration"]["propagate_shot_homography"] is True


def test_keep_unique_admissible_frames_is_on_by_default(tmp_path, monkeypatch):
    commands, match_out = _run_match_capturing_commands(tmp_path, monkeypatch)
    sided = next(
        command for command in commands if "cv.pipeline.player_side_association" in command
    )
    assert "--keep-unique-admissible-frames" in sided
    assert "--no-keep-unique-admissible-frames" not in sided
    document = __import__("json").loads(
        (match_out / "run_manifests/canonical_runner.json").read_text()
    )
    assert document["provenance"]["configuration"]["keep_unique_admissible_frames"] is True


def test_keep_unique_off_moves_the_association_stage_identity(tmp_path, monkeypatch):
    """OFF and ON are different work, so an ON run cannot resume an unflagged association."""
    from cv.pipeline import artifact_cache

    def association_command(root, enabled):
        root.mkdir()
        commands, _ = _run_match_capturing_commands(
            root, monkeypatch, keep_unique_admissible_frames=enabled
        )
        command = next(c for c in commands if "cv.pipeline.player_side_association" in c)
        return [token.replace(str(root), "<root>") for token in command]

    on = association_command(tmp_path / "on", True)
    off = association_command(tmp_path / "off", False)
    assert "--keep-unique-admissible-frames" in on
    assert "--no-keep-unique-admissible-frames" in off
    identity = artifact_cache.stage_identity
    before = identity(stage="player_side_association", command=on, inputs=[])
    after = identity(stage="player_side_association", command=off, inputs=[])
    assert before["fingerprint"] != after["fingerprint"]


@pytest.mark.parametrize("enabled", [False, True])
def test_lost_revival_body_history_reaches_the_association_command(tmp_path, monkeypatch, enabled):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, lost_revival_body_history=enabled
    )
    sided = next(
        command for command in commands if "cv.pipeline.player_side_association" in command
    )
    assert ("--lost-revival-body-history" in sided) is enabled
    document = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert document["provenance"]["configuration"]["lost_revival_body_history"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_overlap_fragment_recovery_reaches_the_association_command(tmp_path, monkeypatch, enabled):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, overlap_fragment_recovery=enabled
    )
    sided = next(
        command for command in commands if "cv.pipeline.player_side_association" in command
    )
    assert ("--overlap-fragment-recovery" in sided) is enabled
    # Independent arm: requesting it never turns on the other default-off association guard.
    assert "--lost-revival-body-history" not in sided
    document = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    configuration = document["provenance"]["configuration"]
    assert configuration["overlap_fragment_recovery"] is enabled
    assert configuration["lost_revival_body_history"] is False


def test_overlap_fragment_recovery_moves_the_association_stage_identity(tmp_path, monkeypatch):
    """OFF and ON are different work, so the ON arm can never resume an OFF association."""
    from cv.pipeline import artifact_cache

    def association_command(root, enabled):
        # Each arm runs under its own root; the root is normalized away so that only the
        # requested option can distinguish the two commands.
        root.mkdir()
        commands, _ = _run_match_capturing_commands(
            root, monkeypatch, overlap_fragment_recovery=enabled
        )
        command = next(c for c in commands if "cv.pipeline.player_side_association" in c)
        return [token.replace(str(root), "<root>") for token in command]

    default = association_command(tmp_path / "off", False)
    requested = association_command(tmp_path / "on", True)

    keep = default.index("--keep-unique-admissible-frames")
    assert requested == [
        *default[:keep],
        "--overlap-fragment-recovery",
        *default[keep:],
    ]
    identity = artifact_cache.stage_identity
    before = identity(stage="player_side_association", command=default, inputs=[])
    after = identity(stage="player_side_association", command=requested, inputs=[])
    assert before["fingerprint"] != after["fingerprint"]


def test_overlap_recovery_request_is_preserved_when_camera_admission_produces_no_clips(
    tmp_path, monkeypatch
):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, camera_clips=[], overlap_fragment_recovery=True
    )
    assert not any("cv.pipeline.player_side_association" in c for c in commands)
    document = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert document["provenance"]["configuration"]["overlap_fragment_recovery"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_canonical_cli_forwards_the_overlap_recovery_request(tmp_path, monkeypatch, enabled):
    import json

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"matches": []}))
    forwarded: list = []
    monkeypatch.setattr(canonical_runner, "run_manifest", lambda *a: forwarded.append(a))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "canonical_runner",
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "out"),
            *(["--overlap-fragment-recovery"] if enabled else []),
        ],
    )

    canonical_runner.main()

    manifest_document, out, device, *options = forwarded[0]
    assert options[-1] is False
    assert options[-2] is True
    assert options[-3] == "off"
    assert options[-4] is enabled
    # The other default-off arms and the original camera choices are untouched by this flag.
    assert options[-5] is False
    assert options[:6] == ["postseg", "argmax", "transport", None, "off", "off"]


def test_guard_request_is_preserved_when_camera_admission_produces_no_clips(tmp_path, monkeypatch):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, camera_clips=[], lost_revival_body_history=True
    )
    assert not any("cv.pipeline.player_side_association" in c for c in commands)
    doc = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert doc["provenance"]["configuration"]["lost_revival_body_history"] is True


@pytest.mark.parametrize("policy", ["off", "illumination_split"])
def test_surface_witness_reaches_court_command_and_provenance(tmp_path, monkeypatch, policy):
    import json

    commands, match_out = _run_match_capturing_commands(
        tmp_path, monkeypatch, court_surface_witness=policy
    )
    command = next(c for c in commands if "cv.pipeline.court_topology_runner" in c)
    if policy == "off":
        assert "--court-surface-witness" not in command
    else:
        assert command[command.index("--court-surface-witness") + 1] == policy
    receipt = json.loads((match_out / "run_manifests/canonical_runner.json").read_text())
    assert receipt["provenance"]["configuration"]["court_surface_witness"] == policy


def test_surface_witness_rejects_unknown_policy_before_running(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unknown surface witness policy"):
        _run_match_capturing_commands(
            tmp_path, monkeypatch, court_surface_witness="unregistered_override"
        )
