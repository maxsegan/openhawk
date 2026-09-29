"""Smoke tests for modules launched by the canonical tracking composition."""

from __future__ import annotations

import importlib
import json

import pytest

from cv.pipeline.tracking_composition import completed_match_result


@pytest.fixture(autouse=True)
def isolated_crop_weight_paths(tmp_path, monkeypatch):
    """Command-construction tests do not require installed model weights or a .env."""
    from cv.pipeline import tracking_composition as tc

    root = tmp_path / "test_data"
    monkeypatch.setattr(tc, "data_root", lambda: root)
    for model in ("wasb", "tracknetv2"):
        path = root / tc.CROP_FINETUNE_TEMPLATE.format(model=model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test path only; detector execution is mocked")
    # Tracker runs are mocked, so the torso-lock gap restore has no tracks to read.
    monkeypatch.setattr(tc, "restore_torso_lock_gaps", lambda *a, **k: 0)


@pytest.mark.parametrize(
    "module",
    [
        "ball_neural",
        "ball_neural_batched",
        "ball_local_refine",
        "ball_local_refine_batched",
        "ball_motion_tracker",
        "ball_track_consensus",
        "ball_track_detour_repair",
        "ball_track_local_augment",
    ],
)
def test_canonical_tracking_module_imports(module: str) -> None:
    importlib.import_module(f"cv.pipeline.{module}")


def _capture_run_match_commands(tmp_path, monkeypatch, **kwargs) -> list[list[str]]:
    from cv.pipeline import tracking_composition as tc

    commands: list[list[str]] = []

    def fake_run(command):
        commands.append(command)
        return 0.0

    monkeypatch.setattr(tc, "run", fake_run)
    monkeypatch.setattr(tc, "decode", lambda *a, **k: 0.0)
    monkeypatch.setattr(tc, "require_paths", lambda _: None)
    monkeypatch.setattr(tc, "resolve_player_boxes", lambda *a, **k: tmp_path / "boxes.csv")
    monkeypatch.setattr(tc, "filter_candidate_streams", lambda *a, **k: {})
    tc.run_match({"id": "match", "source_fps": 25.0}, tmp_path, 0, **kwargs)
    return commands


def test_torso_lock_tracks_the_filtered_streams_and_rollback_restores_them(
    tmp_path, monkeypatch
) -> None:
    from cv.pipeline.tracking_composition import TORSO_LOCK_INPUTS_NAME

    def motion(commands):
        return next(c for c in commands if "cv.pipeline.ball_motion_tracker" in c)

    on = motion(_capture_run_match_commands(tmp_path, monkeypatch))
    streams = [on[i + 1] for i, part in enumerate(on) if part == "--candidates"]
    assert len(streams) == 7
    assert all(f"/{TORSO_LOCK_INPUTS_NAME}/" in path for path in streams)
    assert on[on.index("--torso-lock-boxes") + 1] == str(tmp_path / "boxes.csv")
    off = motion(_capture_run_match_commands(tmp_path, monkeypatch, torso_lock=False))
    assert "--torso-lock-boxes" not in off
    assert not any(TORSO_LOCK_INPUTS_NAME in part for part in off)


def _sliding_ball_command(commands: list[list[str]]) -> list[str]:
    return next(command for command in commands if "native1080_sliding_k5_v1" in command)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("motion", [False, True])
def test_launched_tracker_modules_are_declared_for_cache_identity(
    tmp_path, monkeypatch, optimized, motion
):
    from cv.pipeline.tracking_composition import RUNTIME_MODULE_DEPENDENCIES

    commands = _capture_run_match_commands(
        tmp_path,
        monkeypatch,
        optimized_full_runtime=optimized,
        optimized_local_runtime=optimized,
        motion_tracker=motion,
    )
    launched = {command[command.index("-m") + 1] for command in commands if "-m" in command}
    assert launched <= set(RUNTIME_MODULE_DEPENDENCIES)


def test_model_identity_covers_crop_weights_and_external_source_but_not_bytecode(
    tmp_path, monkeypatch
):
    from cv.pipeline import tracking_composition as tc
    from cv.pipeline.artifact_cache import stage_identity

    monkeypatch.setattr(tc, "EXTERNAL_ROOT", tmp_path)
    production = tmp_path / "pretrained_weights"
    production.mkdir()
    for model in ("wasb", "tracknetv2"):
        (production / f"{model}_tennis_best.pth.tar").write_bytes(b"production")
        (tmp_path / f"{model}_crop.pt").write_bytes(b"crop")
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.py").write_text("MODEL = 1\n")
    (src / "model.yaml").write_text("name: test\n")
    (src / "ignored.pyc").write_bytes(b"bytecode")
    template = str(tmp_path / "{model}_crop.pt")
    inputs = tc.tracking_model_inputs(template)
    assert set(path.name for path in inputs) == {
        "wasb_tennis_best.pth.tar",
        "tracknetv2_tennis_best.pth.tar",
        "wasb_crop.pt",
        "tracknetv2_crop.pt",
        "model.py",
        "model.yaml",
    }
    first = stage_identity(stage="tracking", command=["tracker"], inputs=inputs)["fingerprint"]
    (tmp_path / "wasb_crop.pt").write_bytes(b"new crop model")
    second = stage_identity(stage="tracking", command=["tracker"], inputs=inputs)["fingerprint"]
    assert first != second
    (src / "model.py").write_text("MODEL = 2\n")
    assert (
        stage_identity(stage="tracking", command=["tracker"], inputs=inputs)["fingerprint"]
        != second
    )


def test_tracking_ball_decode_defaults_to_centroid(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(tmp_path, monkeypatch)
    ball = _sliding_ball_command(commands)

    assert ball[ball.index("--subpixel") + 1] == "centroid"


def test_tracking_ball_decode_threads_named_alternative(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(
        tmp_path,
        monkeypatch,
        subpixel="argmax",
        optimized_local_runtime=True,
    )
    ball = _sliding_ball_command(commands)
    crop = _module_command(commands, "ball_local_refine_batched")

    assert ball[ball.index("--subpixel") + 1] == "argmax"
    assert crop[crop.index("--subpixel") + 1] == "argmax"


def _far_native_command(commands: list[list[str]]) -> list[str]:
    return next(command for command in commands if "native1080_far_native_v1" in command)


def test_far_native_tiling_is_on_by_default(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(tmp_path, monkeypatch)

    assert "--far-native" in _far_native_command(commands)


def test_far_native_tiling_runs_against_the_native_frames(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(tmp_path, monkeypatch, far_native=True)
    far = _far_native_command(commands)

    assert "--far-native" in far
    assert far[far.index("--native-frames-dir") + 1] == "audit_frames_native_1080"
    assert far[far.index("--camera-npz") + 1] == "camera_P_per_point.npz"


def test_far_native_tiling_can_be_disabled(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(tmp_path, monkeypatch, far_native=False)

    assert not any("--far-native" in command for command in commands)


def _module_command(commands: list[list[str]], module: str) -> list[str]:
    return next(command for command in commands if f"cv.pipeline.{module}" in command)


def test_crop_authority_is_opt_in(tmp_path, monkeypatch) -> None:
    command = _module_command(
        _capture_run_match_commands(tmp_path, monkeypatch, motion_tracker=False),
        "ball_track_local_augment",
    )

    assert "--crop-authoritative" not in command


def test_crop_authority_carries_the_measured_gate(tmp_path, monkeypatch) -> None:
    command = _module_command(
        _capture_run_match_commands(
            tmp_path,
            monkeypatch,
            crop_authoritative=True,
            motion_tracker=False,
        ),
        "ball_track_local_augment",
    )

    assert "--crop-authoritative" in command
    assert command[command.index("--crop-authority-px1080") + 1] == "4.0"


def test_detour_repair_passthrough_is_opt_in(tmp_path, monkeypatch) -> None:
    repair = _module_command(
        _capture_run_match_commands(tmp_path, monkeypatch, motion_tracker=False),
        "ball_track_detour_repair",
    )

    assert "--passthrough" not in repair

    disabled = _module_command(
        _capture_run_match_commands(
            tmp_path,
            monkeypatch,
            detour_repair=False,
            motion_tracker=False,
        ),
        "ball_track_detour_repair",
    )

    assert "--passthrough" in disabled


def test_motion_tracker_is_default_and_reads_every_candidate_family(tmp_path, monkeypatch) -> None:
    command = _module_command(
        _capture_run_match_commands(tmp_path, monkeypatch),
        "ball_motion_tracker",
    )

    assert command.count("--candidates") == 7
    assert "coarse_lock" in command
    assert "branched_crop_wasb" in command
    assert "far_native_tracknetv2" in command
    assert command[command.index("--camera-projections") + 1].endswith("camera_P_per_frame_v1.npz")


def test_legacy_consensus_skips_motion_tracker(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(tmp_path, monkeypatch, motion_tracker=False)

    assert not any("cv.pipeline.ball_motion_tracker" in command for command in commands)


def test_batched_infer_clip_accepts_baseline_positional_subpixel() -> None:
    """Regression: ball_neural.main() passes subpixel positionally into the monkeypatched
    batched infer_clip; the batched signature must accept it or the tracking stage crashes."""
    import inspect

    from cv.pipeline import ball_neural, ball_neural_batched

    baseline_params = list(inspect.signature(ball_neural.infer_clip).parameters)
    batched_params = list(inspect.signature(ball_neural_batched.infer_clip).parameters)

    assert baseline_params[-1] == "subpixel"
    assert batched_params[-1] == "subpixel"
    assert batched_params == baseline_params


def test_completed_match_does_not_require_sided_boxes(tmp_path) -> None:
    match = {"id": "match", "source_fps": 24.0}
    match_root = tmp_path / "match"
    match_root.mkdir()
    track = match_root / "ball_track_joint_native1080_arc_augmented_v2.csv"
    track.write_text("clip,frame,x,y\n")
    track.with_suffix(track.suffix + ".coordinates.json").write_text(json.dumps({}))
    (match_root / "camera_P_per_frame_v1.npz").write_bytes(b"camera")
    (match_root / "contact_audio_scores_16k_native_v1.npz").write_bytes(b"audio")

    result = completed_match_result(match, tmp_path)

    assert result is not None
    assert result["reused_complete"] is True


def test_crop_pass_uses_the_native_crop_finetune_by_default(tmp_path, monkeypatch) -> None:
    from cv.pipeline import tracking_composition as tc

    crop = _module_command(_capture_run_match_commands(tmp_path, monkeypatch), "ball_local_refine")
    weights = crop[crop.index("--weights") + 1]

    assert weights.endswith("ball_finetune_v1/wasb_native_crop_ep3.pth.tar")
    # The full-frame and far-native passes are a different input regime and keep the
    # production checkpoints.
    assert "--weights" not in _sliding_ball_command(
        _capture_run_match_commands(tmp_path, monkeypatch)
    )
    assert tc.production_weights("wasb").name == "wasb_tennis_best.pth.tar"


def test_production_crop_weights_are_still_reachable(tmp_path, monkeypatch) -> None:
    from cv.pipeline import tracking_composition as tc

    crop = _module_command(
        _capture_run_match_commands(tmp_path, monkeypatch, crop_weights=None),
        "ball_local_refine",
    )

    assert crop[crop.index("--weights") + 1] == str(tc.production_weights("wasb"))


def test_crop_weights_flag_selects_a_checkpoint_per_detector(tmp_path, monkeypatch) -> None:
    for model in ("wasb", "tracknetv2"):
        (tmp_path / f"{model}_ft.pth.tar").write_bytes(b"")
    commands = _capture_run_match_commands(
        tmp_path,
        monkeypatch,
        crop_weights=str(tmp_path / "{model}_ft.pth.tar"),
    )
    crops = [command for command in commands if "cv.pipeline.ball_local_refine" in command]

    assert [command[command.index("--weights") + 1] for command in crops] == [
        str(tmp_path / "wasb_ft.pth.tar"),
        str(tmp_path / "tracknetv2_ft.pth.tar"),
    ]
    # The full-frame sliding pass keeps the production checkpoint: a crop fine-tune is only a
    # drop-in for the regime it was trained on.
    assert "--weights" not in _sliding_ball_command(commands)


def test_far_native_weights_flag_is_independent_of_the_crop_flag(tmp_path, monkeypatch) -> None:
    from cv.pipeline import tracking_composition as tc

    for model in ("wasb", "tracknetv2"):
        (tmp_path / f"{model}_ft.pth.tar").write_bytes(b"")
    commands = _capture_run_match_commands(
        tmp_path,
        monkeypatch,
        far_native_weights=str(tmp_path / "{model}_ft.pth.tar"),
    )
    far = _far_native_command(commands)
    crop = _module_command(commands, "ball_local_refine")

    assert far[far.index("--weights") + 1] == str(tmp_path / "wasb_ft.pth.tar")
    assert crop[crop.index("--weights") + 1].endswith("wasb_native_crop_ep3.pth.tar")
    assert tc.production_weights("tracknetv2").name == "tracknetv2_tennis_best.pth.tar"


def test_missing_override_weights_fail_closed(tmp_path) -> None:
    from cv.pipeline import tracking_composition as tc

    with pytest.raises(FileNotFoundError):
        tc.resolve_weights(str(tmp_path / "{model}_absent.pth.tar"), "wasb")


def test_run_record_states_which_checkpoint_every_pass_loaded(tmp_path, monkeypatch) -> None:
    from cv.pipeline import tracking_composition as tc

    (tmp_path / "wasb_ft.pth.tar").write_bytes(b"")
    (tmp_path / "tracknetv2_ft.pth.tar").write_bytes(b"")
    monkeypatch.setattr(tc, "run", lambda command: 0.0)
    monkeypatch.setattr(tc, "decode", lambda *a, **k: 0.0)
    monkeypatch.setattr(tc, "require_paths", lambda _: None)
    monkeypatch.setattr(tc, "resolve_player_boxes", lambda *a, **k: tmp_path / "boxes.csv")
    monkeypatch.setattr(tc, "filter_candidate_streams", lambda *a, **k: {})
    record = tc.run_match(
        {"id": "match", "source_fps": 25.0},
        tmp_path,
        0,
        crop_weights=str(tmp_path / "{model}_ft.pth.tar"),
    )

    assert record["crop_weights"]["wasb"] == str(tmp_path / "wasb_ft.pth.tar")
    assert record["far_native_weights"]["wasb"].endswith("wasb_tennis_best.pth.tar")


def _motion_command(commands: list[list[str]]) -> list[str]:
    return _module_command(commands, "ball_motion_tracker")


def _lock_stream(command: list[str]) -> str:
    """The path paired with the ``coarse_lock`` source in the tracker's candidate list."""
    index = command.index("coarse_lock")
    assert command[index - 1] == "--source"
    assert command[index - 3] == "--candidates"
    return command[index - 2]


def test_motion_tracker_locks_onto_the_native_crop_decode_by_default(tmp_path, monkeypatch) -> None:
    lock = _lock_stream(_motion_command(_capture_run_match_commands(tmp_path, monkeypatch)))

    assert lock.endswith("ball_track_joint_native1080_branched_crop_integrity_v2.csv")


def test_full_frame_lock_stream_is_still_reachable(tmp_path, monkeypatch) -> None:
    lock = _lock_stream(
        _motion_command(_capture_run_match_commands(tmp_path, monkeypatch, crop_lock_stream=False))
    )

    assert lock.endswith("ball_track_joint_native1080_integrity_v1.csv")


def test_crop_first_association_is_opt_in_and_declares_its_lock(tmp_path, monkeypatch) -> None:
    default = _motion_command(_capture_run_match_commands(tmp_path, monkeypatch))
    enabled = _motion_command(_capture_run_match_commands(tmp_path, monkeypatch, crop_first=True))

    assert "--crop-first" not in default
    assert "--crop-first" in enabled
    # The default lock stream is itself a crop decode, so the tracker must be told.
    assert "--lock-is-crop" in enabled


def test_run_record_states_the_lock_stream(tmp_path, monkeypatch) -> None:
    from cv.pipeline import tracking_composition as tc

    monkeypatch.setattr(tc, "run", lambda command: 0.0)
    monkeypatch.setattr(tc, "decode", lambda *a, **k: 0.0)
    monkeypatch.setattr(tc, "require_paths", lambda _: None)
    monkeypatch.setattr(tc, "resolve_player_boxes", lambda *a, **k: tmp_path / "boxes.csv")
    monkeypatch.setattr(tc, "filter_candidate_streams", lambda *a, **k: {})
    record = tc.run_match({"id": "match", "source_fps": 25.0}, tmp_path, 0)

    assert record["crop_lock_stream"] is True
    assert record["crop_first"] is False
    assert record["crop_weights"]["wasb"].endswith("wasb_native_crop_ep3.pth.tar")


# --- optional ball-identity arms ------------------------------------------------------------


def _prepare_consumed_track(tmp_path):
    """The motion tracker's own output, as the anchor-extension hook finds it."""
    match_out = tmp_path / "match"
    match_out.mkdir(parents=True, exist_ok=True)
    track = match_out / "ball_track_joint_native1080_arc_augmented_v2.csv"
    track.write_text("clip,frame,x,y,x_native,y_native\n")
    track.with_name(track.name + ".coordinates.json").write_text(json.dumps({}))
    return track


def test_anchor_extension_is_rejected_before_any_detector_work(tmp_path) -> None:
    from cv.pipeline import tracking_composition as tc

    # No stage is mocked here: reaching require_paths would raise something else.
    with pytest.raises(ValueError, match="requires primary_ball_ownership"):
        tc.run_match({"id": "match", "source_fps": 25.0}, tmp_path, 0, anchor_extension=True)
    with pytest.raises(ValueError, match="motion tracker"):
        tc.run_match(
            {"id": "match", "source_fps": 25.0},
            tmp_path,
            0,
            motion_tracker=False,
            primary_ball_ownership=True,
        )


def test_optional_ball_identity_arms_are_off_by_default(tmp_path, monkeypatch) -> None:
    import inspect

    from cv.pipeline import tracking_composition as tc

    commands = _capture_run_match_commands(tmp_path, monkeypatch)
    launched = {command[command.index("-m") + 1] for command in commands if "-m" in command}

    assert "cv.pipeline.ball_anchor_extension" not in launched
    assert "--primary-ball-ownership" not in _motion_command(commands)
    assert inspect.signature(tc.run_match).parameters["primary_ball_ownership"].default is False
    assert inspect.signature(tc.run_match).parameters["anchor_extension"].default is False


def test_the_extension_consumes_the_tracker_arms_and_replaces_the_consumed_track(
    tmp_path, monkeypatch
) -> None:
    from cv.pipeline import tracking_composition as tc
    from cv.pipeline.ball_motion_tracker import ownership_artifact_path

    track = _prepare_consumed_track(tmp_path)
    match_out = track.parent
    commands = _capture_run_match_commands(
        tmp_path, monkeypatch, primary_ball_ownership=True, anchor_extension=True
    )
    motion = _motion_command(commands)
    extension = _module_command(commands, "ball_anchor_extension")

    assert "--primary-ball-ownership" in motion
    # The side artifact the extension reads is the one the tracker actually writes.
    assert ownership_artifact_path(track).name == tc.OWNERSHIP_ARTIFACT_NAME
    assert extension[extension.index("--ownership") + 1] == str(
        match_out / tc.OWNERSHIP_ARTIFACT_NAME
    )
    # Extension input is the preserved pre-extension copy; output is the consumed artifact
    # the downstream stage reads, so the extended rows are the ones that are consumed.
    assert extension[extension.index("--track") + 1] == str(
        match_out / tc.PRE_ANCHOR_EXTENSION_NAME
    )
    assert extension[extension.index("--output") + 1] == str(track)
    assert (match_out / tc.PRE_ANCHOR_EXTENSION_NAME).is_file()
    assert (match_out / (tc.PRE_ANCHOR_EXTENSION_NAME + ".coordinates.json")).is_file()
    # Exactly the tracker's own arms, in the same order; nothing is autodiscovered.
    assert _arm_pairs(extension) == _arm_pairs(motion)
    assert "--extra-candidates" not in extension


def _arm_pairs(command: list[str]) -> list[tuple[str, str]]:
    return [
        (command[index + 1], command[index + 3])
        for index, token in enumerate(command)
        if token == "--candidates" and command[index + 2] == "--source"
    ]


def test_completed_match_reuse_needs_the_arm_that_was_requested(tmp_path) -> None:
    from cv.pipeline import tracking_composition as tc

    match = {"id": "match", "source_fps": 24.0}
    track = _prepare_consumed_track(tmp_path)
    match_root = track.parent
    (match_root / "camera_P_per_frame_v1.npz").write_bytes(b"camera")
    (match_root / "contact_audio_scores_16k_native_v1.npz").write_bytes(b"audio")

    assert completed_match_result(match, tmp_path) is not None
    assert completed_match_result(match, tmp_path, primary_ball_ownership=True) is None
    (match_root / tc.OWNERSHIP_ARTIFACT_NAME).write_text("clip,frame,x,y,owner\n")
    assert completed_match_result(match, tmp_path, primary_ball_ownership=True) is not None
    assert completed_match_result(match, tmp_path, anchor_extension=True) is None
    (match_root / tc.ANCHOR_EXTENSION_REPORT_NAME).write_text("{}")
    assert completed_match_result(match, tmp_path, anchor_extension=True) is not None


# --- optional guide-admission fallback ------------------------------------------------------


def test_guide_sequence_association_is_off_by_default(tmp_path, monkeypatch) -> None:
    import inspect

    from cv.pipeline import tracking_composition as tc

    commands = _capture_run_match_commands(tmp_path, monkeypatch)

    assert "--guide-sequence-association" not in _motion_command(commands)
    assert inspect.signature(tc.run_match).parameters["guide_sequence_association"].default is False
    assert inspect.signature(tc.run_lane).parameters["guide_sequence_association"].default is False


def test_guide_sequence_association_reaches_only_the_motion_tracker(tmp_path, monkeypatch) -> None:
    commands = _capture_run_match_commands(tmp_path, monkeypatch, guide_sequence_association=True)
    motion = _motion_command(commands)

    assert "--guide-sequence-association" in motion
    # Independent of the ball-identity arms: neither is implied, and no other stage is told.
    assert "--primary-ball-ownership" not in motion
    # The torso-lock gap restore reruns the same tracker on the unfiltered pool.
    told = [command for command in commands if "--guide-sequence-association" in command]
    assert motion in told
    assert all(any("ball_motion_tracker" in part for part in command) for command in told)


def test_guide_sequence_association_needs_the_motion_tracker_stage(tmp_path) -> None:
    from cv.pipeline import tracking_composition as tc

    # No stage is mocked here: reaching require_paths would raise something else.
    with pytest.raises(ValueError, match="guide_sequence_association requires"):
        tc.run_match(
            {"id": "match", "source_fps": 25.0},
            tmp_path,
            0,
            motion_tracker=False,
            guide_sequence_association=True,
        )


def test_run_record_states_the_guide_admission_arm(tmp_path, monkeypatch) -> None:
    from cv.pipeline import tracking_composition as tc

    monkeypatch.setattr(tc, "run", lambda command: 0.0)
    monkeypatch.setattr(tc, "decode", lambda *a, **k: 0.0)
    monkeypatch.setattr(tc, "require_paths", lambda _: None)
    monkeypatch.setattr(tc, "resolve_player_boxes", lambda *a, **k: tmp_path / "boxes.csv")
    monkeypatch.setattr(tc, "filter_candidate_streams", lambda *a, **k: {})
    record = tc.run_match({"id": "match", "source_fps": 25.0}, tmp_path, 0)

    assert record["guide_sequence_association"] is False


def test_guide_sequence_association_writes_no_side_artifact_for_reuse(tmp_path) -> None:
    """The arm changes tracked rows only, so reuse is separated by the stage command."""
    match = {"id": "match", "source_fps": 24.0}
    track = _prepare_consumed_track(tmp_path)
    (track.parent / "camera_P_per_frame_v1.npz").write_bytes(b"camera")
    (track.parent / "contact_audio_scores_16k_native_v1.npz").write_bytes(b"audio")

    assert completed_match_result(match, tmp_path) is not None


def test_the_guide_admission_arm_is_rejected_at_the_composition_cli(monkeypatch, capsys) -> None:
    import sys

    from cv.pipeline import tracking_composition as tc

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tracking_composition",
            "--manifest",
            "manifest.json",
            "--protocol",
            "protocol.json",
            "--out",
            "out",
            "--legacy-consensus",
            "--guide-sequence-association",
        ],
    )
    # The CLI must refuse before it reads the manifest, which does not exist here.
    with pytest.raises(SystemExit):
        tc.main()

    assert "--guide-sequence-association needs the motion tracker stage" in capsys.readouterr().err


def test_the_tracking_receipt_binds_the_ownership_modules_and_the_pinned_tracker() -> None:
    import sys

    from cv.pipeline import tracking_composition as tc
    from cv.pipeline.artifact_cache import stage_identity

    # A stale pin would let the composed stage run a tracker it did not declare.
    assert tc.file_sha256(tc.ROOT / tc.MOTION_TRACKER_PATH) == tc.MOTION_TRACKER_SHA256
    identity = stage_identity(
        stage="tracking",
        command=[sys.executable, str(tc.ROOT / "cv/pipeline/tracking_composition.py")],
        inputs=[],
    )
    paths = {row["path"] for row in identity["code"]}
    assert {
        "cv/pipeline/ball_motion_tracker.py",
        "cv/pipeline/ball_ownership.py",
        "cv/pipeline/ball_anchor_extension.py",
    } <= paths
