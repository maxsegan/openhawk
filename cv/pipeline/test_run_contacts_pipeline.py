from argparse import Namespace
import json

from cv.pipeline.artifact_cache import write_stage_receipt

from run_contacts_pipeline import (
    Stage,
    build_stages,
    model_records,
    plan_stages,
    stage_outputs_ready,
    write_manifest,
)


def _args(tmp_path) -> Namespace:
    return Namespace(
        out=str(tmp_path),
        match="match-id",
        video=str(tmp_path / "match.mp4"),
        frames_dir="frames",
        point_map="point_video_map_v2.csv",
        artifact_width=960,
        artifact_height=540,
        fps=50.0,
        preserve_source_fps=True,
        device=0,
        seed=4,
        contact_tag="contact_v1",
        player_model="yolo.pt",
        player_batch=64,
        audio_cache="shared_audio.npz",
        skip_audio_evidence=False,
        allow_partial_timing_holds=False,
        resume=False,
        dry_run=False,
    )


def test_default_stage_graph_contains_only_current_event_inputs(tmp_path) -> None:
    stages = build_stages(_args(tmp_path))

    assert [stage.name for stage in stages] == [
        "frame_extraction",
        "frame_cadence",
        "player_detection",
        "audio_evidence",
    ]
    audio = stages[-1]
    assert audio.command[audio.command.index("--audio-cache") + 1] == "shared_audio.npz"
    assert audio.outputs == [str(tmp_path / "shared_audio.npz")]
    assert "--preserve-source-fps" in stages[0].command
    assert "--preserve-source-fps" in stages[2].command
    assert all(stage.command[1] == "-m" for stage in stages)
    for name in {"frame_extraction", "player_detection", "audio_evidence"}:
        command = next(stage.command for stage in stages if stage.name == name)
        assert command[command.index("--point-map") + 1] == "point_video_map_v2.csv"
    cadence = stages[1]
    assert cadence.command[cadence.command.index("--frames-root") + 1] == str(tmp_path / "frames")
    assert "--fail-on-timing-hold" in cadence.command


def test_subprocess_stages_are_declared_for_parent_cache_identity(tmp_path):
    from cv.pipeline.run_contacts_pipeline import RUNTIME_MODULE_DEPENDENCIES

    stages = build_stages(_args(tmp_path))
    assert {stage.command[2] for stage in stages} <= set(RUNTIME_MODULE_DEPENDENCIES)


def test_default_manifest_does_not_claim_unused_ball_model(tmp_path) -> None:
    args = _args(tmp_path)

    assert model_records(args) == [{"name": "yolo.pt", "role": "player_detector"}]


def test_motion_profile_omits_audio_without_changing_spatial_stages(tmp_path) -> None:
    args = _args(tmp_path)
    args.skip_audio_evidence = True

    stages = build_stages(args)

    assert [stage.name for stage in stages] == [
        "frame_extraction",
        "frame_cadence",
        "player_detection",
    ]


def test_motion_orchestrator_can_defer_player_detection(tmp_path) -> None:
    args = _args(tmp_path)
    args.skip_audio_evidence = True
    args.skip_player_detection = True

    stages = build_stages(args)

    assert [stage.name for stage in stages] == ["frame_extraction", "frame_cadence"]
    assert model_records(args) == []


def test_partial_timing_holds_require_usable_points(tmp_path) -> None:
    from run_contacts_pipeline import accept_partial_timing_holds

    args = _args(tmp_path)
    args.allow_partial_timing_holds = True
    stage = build_stages(args)[1]
    cadence = tmp_path / "frame_cadence_v1.json"
    cadence.write_text(json.dumps({"points": 10, "timing_holds": 2}))

    assert accept_partial_timing_holds(args, stage, 2) == {
        "timing_usable_points": 8,
        "timing_held_points": 2,
        "policy": "preserve_native_timestamps_and_exclude_held_points_downstream",
    }
    cadence.write_text(json.dumps({"points": 10, "timing_holds": 10}))
    assert accept_partial_timing_holds(args, stage, 2) is None
    assert accept_partial_timing_holds(args, stage, 1) is None


def test_non_frame_stage_requires_every_nonempty_output(tmp_path) -> None:
    stage = build_stages(_args(tmp_path))[2]
    for output in stage.outputs:
        with open(output, "w") as handle:
            handle.write("ok")
    assert stage_outputs_ready(stage, str(tmp_path), "frames")
    open(stage.outputs[0], "w").close()
    assert not stage_outputs_ready(stage, str(tmp_path), "frames")


def test_frame_stage_uses_selected_versioned_point_map(tmp_path) -> None:
    frames = tmp_path / "frames"
    (frames / "pt0002").mkdir(parents=True)
    (frames / "pt0002" / "frame_000001.jpg").write_text("image")
    (tmp_path / "frames.coordinates.json").write_text("{}")
    (tmp_path / "point_video_map.csv").write_text("pt,rally_t_start,rally_t_end\n1,1,2\n")
    (tmp_path / "point_video_map_v2.csv").write_text("pt,rally_t_start,rally_t_end\n2,1,2\n")
    stage = Stage("frames", ["true"], [str(frames)], frames=True)

    assert not stage_outputs_ready(stage, str(tmp_path), "frames")
    assert stage_outputs_ready(stage, str(tmp_path), "frames", "point_video_map_v2.csv")


def test_missing_stage_invalidates_all_downstream_outputs(tmp_path) -> None:
    outputs = [tmp_path / f"stage{i}.csv" for i in range(3)]
    outputs[0].write_text("ready")
    outputs[2].write_text("stale downstream")
    stages = [Stage(f"stage{i}", ["true"], [str(output)]) for i, output in enumerate(outputs)]
    write_stage_receipt(
        out_dir=tmp_path,
        stage="stage0",
        command=["true"],
        inputs=[],
        outputs=[outputs[0]],
    )

    assert [reuse for _, reuse in plan_stages(stages, str(tmp_path), "frames", True)] == [
        True,
        False,
        False,
    ]


def test_resume_rejects_changed_stage_command(tmp_path) -> None:
    output = tmp_path / "stage.csv"
    output.write_text("ready")
    stage = Stage("stage", ["true"], [str(output)])
    write_stage_receipt(
        out_dir=tmp_path,
        stage=stage.name,
        command=stage.command,
        inputs=[],
        outputs=stage.outputs,
    )

    assert plan_stages([stage], str(tmp_path), "frames", True) == [(stage, True)]
    changed = Stage("stage", ["true", "--changed"], [str(output)])
    assert plan_stages([changed], str(tmp_path), "frames", True) == [(changed, False)]


def test_pipeline_manifest_is_provenance_carrying_and_portable(tmp_path) -> None:
    args = _args(tmp_path)
    (tmp_path / "match.mp4").write_bytes(b"video")
    (tmp_path / "point_video_map_v2.csv").write_text("pt,rally_t_start,rally_t_end\n")

    path = write_manifest(args, 0.0, "2026-08-01T00:00:00+00:00", "ok", [])
    manifest = json.loads(open(path).read())

    assert manifest["provenance"]["mode"] == "automatic"
    assert str(tmp_path) not in json.dumps(manifest)
