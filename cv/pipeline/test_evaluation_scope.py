"""The declared first-attempts execution scope: selection, roles, guards and accounting."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cv.pipeline import broadcast_runner, evaluation_scope
from cv.pipeline import s6_broadcast_backend as backend
from cv.pipeline import window_camera_inference
from cv.pipeline.canonical_runner import POINT_MAP_NAME

LEDGER_HEADER = "pt,rally_t_start,rally_t_end,confidence,point_index,attempt_role,attempts_in_point"


def write_ledger(path: Path, rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        LEDGER_HEADER + "\n" + "".join(",".join(str(v) for v in row) + "\n" for row in rows)
    )
    return path


def default_rows(count: int = 5) -> list[tuple]:
    # Non-contiguous original identities, one zero-confidence attempt and one score point
    # served twice, so nothing in the prefix depends on tidy input.
    rows = []
    for n in range(count):
        pt = 3 + n * 2
        rows.append((pt, n * 10.0, n * 10.0 + 5.0, 0.0 if n == 1 else 0.8, n, "first", 1))
    if count >= 4:
        # pt of row 3 continues the score point of row 2.
        start, end = rows[2][1], rows[2][2]
        rows[3] = (rows[3][0], end + 1.0, end + 6.0, 0.7, rows[2][4], "continuation", 2)
        rows[2] = (rows[2][0], start, end, rows[2][3], rows[2][4], "first", 2)
    return rows


def materialize(tmp_path: Path, requested: int | None, count: int = 5):
    root = tmp_path / "out"
    match_out = root / "match"
    ledger = write_ledger(match_out / "automatic_point_ledger_v1.csv", default_rows(count))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"broadcast")
    scope = None
    if requested is not None:
        scope = evaluation_scope.build_scope(
            ledger=ledger,
            rows=evaluation_scope.ledger_rows(ledger),
            requested=requested,
            match_id="match",
            source_video=video,
            upstream_root=root,
        )
    broadcast_runner.materialize_postseg(
        video=video,
        match_out=match_out,
        ledger=ledger,
        manifest_path=root / "broadcast_manifest.json",
        match_id="match",
        fps=25.0,
        surface="hard",
        scope=scope,
    )
    return root, match_out, ledger, scope


def test_default_materialization_writes_one_roster_and_no_scope_artifacts(tmp_path):
    root, match_out, ledger, scope = materialize(tmp_path, None)
    assert scope is None
    assert (match_out / POINT_MAP_NAME).read_bytes() == ledger.read_bytes()
    full = (root / "broadcast_manifest.json").read_text()
    assert (root / "manifest.json").read_text() == full
    assert json.loads(full)["matches"][0]["point_ids"] == [3, 5, 7, 9, 11]
    assert not evaluation_scope.scope_path(root).exists()
    assert not (root / evaluation_scope.WORK_MANIFEST_NAME).exists()
    # The opt-in flag is absent by default and the runner keeps its default namespace.
    assert (
        broadcast_runner.requested_evaluation_scope(
            broadcast_runner.argparse.Namespace(evaluation_first_attempts=None)
        )
        is None
    )


@pytest.mark.parametrize("requested, expected", [(1, [3]), (3, [3, 5, 7]), (5, [3, 5, 7, 9, 11])])
def test_scope_materializes_only_the_prefix_and_keeps_the_full_identity_roster(
    tmp_path, requested, expected
):
    root, match_out, ledger, scope = materialize(tmp_path, requested)
    full = json.loads((root / "broadcast_manifest.json").read_text())
    work = json.loads((root / evaluation_scope.WORK_MANIFEST_NAME).read_text())
    assert full["matches"][0]["point_ids"] == [3, 5, 7, 9, 11]
    assert full["points_per_match"] == 5
    assert work["matches"][0]["point_ids"] == expected
    assert work["points_per_match"] == len(expected)
    # The conventional name the ordinary `event_video_model --root` and point grammar read
    # is the selected work roster, not the full one.
    assert (root / "manifest.json").read_text() == (
        root / evaluation_scope.WORK_MANIFEST_NAME
    ).read_text()
    lines = (match_out / POINT_MAP_NAME).read_text().splitlines()
    ledger_lines = ledger.read_text().splitlines()
    assert lines[0] == ledger_lines[0]
    # Original identities, declared native windows and continuation metadata verbatim.
    assert lines[1:] == ledger_lines[1 : 1 + len(expected)]
    document = json.loads(evaluation_scope.scope_path(root).read_text())
    assert document["selected_point_ids"] == expected
    assert document["excluded_point_ids"] == [i for i in [3, 5, 7, 9, 11] if i not in expected]
    assert document["requested_attempts"] == requested
    assert document["full_ledger_attempts"] == 5
    assert document["point_ledger"]["sha256"] == evaluation_scope.provenance.file_sha256(ledger)
    assert document["labels_loaded"] is False


def test_scope_larger_than_the_ledger_selects_the_whole_ledger(tmp_path):
    root, _, _, scope = materialize(tmp_path, 9)
    assert scope["requested_attempts"] == 9 and scope["selected_attempts"] == 5
    assert scope["excluded_point_ids"] == []
    assert json.loads((root / evaluation_scope.WORK_MANIFEST_NAME).read_text())["matches"][0][
        "point_ids"
    ] == [3, 5, 7, 9, 11]


@pytest.mark.parametrize("requested", [0, -1, -5])
def test_nonpositive_scope_is_refused(requested):
    with pytest.raises(evaluation_scope.ScopeError, match="at least one"):
        evaluation_scope.validate_requested(requested)


def test_ambiguous_ledger_prefixes_are_refused(tmp_path):
    duplicate = write_ledger(
        tmp_path / "dup.csv",
        [(3, 0.0, 5.0, 0.8, 0, "first", 1), (3, 10.0, 15.0, 0.8, 1, "first", 1)],
    )
    with pytest.raises(evaluation_scope.ScopeError, match="repeats an attempt identity"):
        evaluation_scope.select_prefix(evaluation_scope.ledger_rows(duplicate), 1)
    unordered = write_ledger(
        tmp_path / "unordered.csv",
        [(3, 40.0, 45.0, 0.8, 0, "first", 1), (5, 10.0, 15.0, 0.8, 1, "first", 1)],
    )
    with pytest.raises(evaluation_scope.ScopeError, match="chronological"):
        evaluation_scope.select_prefix(evaluation_scope.ledger_rows(unordered), 1)


def test_zero_confidence_and_continuation_rows_stay_in_the_selection(tmp_path):
    root, match_out, _, scope = materialize(tmp_path, 4)
    rows = (match_out / POINT_MAP_NAME).read_text().splitlines()[1:]
    assert rows[1].split(",")[3] == "0.0"
    assert rows[3].split(",")[5] == "continuation"
    assert scope["selected_windows"][3]["attempt_role"] == "continuation"
    # A prefix that ends inside a score point discloses the members it does not run.
    cut = materialize(tmp_path / "cut", 3)[3]
    assert cut["continuation_outside_scope"] == [
        {
            "pt": 7,
            "point_index": "2",
            "attempt_role": "first",
            "remaining_attempts_outside_scope": [9],
        }
    ]
    assert scope["continuation_outside_scope"] == []


def test_scope_verification_rejects_a_tampered_ledger_and_a_foreign_match(tmp_path):
    root, match_out, ledger, _ = materialize(tmp_path, 2)
    assert evaluation_scope.verify_scope(root=root, match_id="match", ledger=ledger) is not None
    with pytest.raises(evaluation_scope.ScopeError, match="names another match"):
        evaluation_scope.verify_scope(root=root, match_id="other", ledger=ledger)
    ledger.write_text(ledger.read_text().replace("0.8", "0.9", 1))
    with pytest.raises(evaluation_scope.ScopeError, match="does not bind this automatic"):
        evaluation_scope.verify_scope(root=root, match_id="match", ledger=ledger)


def test_scope_that_is_not_the_ledger_prefix_is_rejected(tmp_path):
    root, _, ledger, _ = materialize(tmp_path, 2)
    path = evaluation_scope.scope_path(root)
    document = json.loads(path.read_text())
    document["selected_point_ids"] = [5, 7]
    path.write_text(json.dumps(document))
    with pytest.raises(evaluation_scope.ScopeError, match="first-attempts prefix"):
        evaluation_scope.verify_scope(root=root, match_id="match", ledger=ledger)


def test_resume_accepts_the_same_scope_and_refuses_changed_or_full_execution(tmp_path):
    root, match_out, ledger, scope = materialize(tmp_path, 2)
    selected = list(scope["selected_point_ids"])
    for pt in selected:
        (match_out / "audit_frames_native_1080" / f"pt{pt:04d}").mkdir(parents=True)
    evaluation_scope.reject_incompatible_root(
        root=root, match_directory=match_out, requested=2, selected_ids=selected
    )
    with pytest.raises(evaluation_scope.ScopeError, match="different execution scope"):
        evaluation_scope.reject_incompatible_root(
            root=root, match_directory=match_out, requested=3, selected_ids=[3, 5, 7]
        )
    with pytest.raises(evaluation_scope.ScopeError, match="full run needs a new output root"):
        evaluation_scope.reject_incompatible_root(
            root=root, match_directory=match_out, requested=None, selected_ids=[]
        )
    # Data outside the declared scope is never removed, only refused.
    (match_out / "audit_frames_native_1080" / "pt0009").mkdir()
    with pytest.raises(evaluation_scope.ScopeError, match="outside its declared scope"):
        evaluation_scope.reject_incompatible_root(
            root=root, match_directory=match_out, requested=2, selected_ids=selected
        )
    assert (match_out / "audit_frames_native_1080" / "pt0009").is_dir()


def test_a_materialized_full_root_refuses_to_adopt_a_scope_but_a_bare_one_may(tmp_path):
    root = tmp_path / "out"
    match_out = root / "match"
    write_ledger(match_out / "automatic_point_ledger_v1.csv", default_rows())
    evaluation_scope.reject_incompatible_root(
        root=root, match_directory=match_out, requested=2, selected_ids=[3, 5]
    )
    root, match_out, _, _ = materialize(tmp_path, None)
    # A materialized full root is incompatible even before extraction, or when its only
    # extracted frame directory happens to lie inside the proposed new prefix.
    for with_frames in (False, True):
        if with_frames:
            (match_out / "audit_frames_native_1080" / "pt0003").mkdir(parents=True)
        with pytest.raises(evaluation_scope.ScopeError, match="full-broadcast materialization"):
            evaluation_scope.reject_incompatible_root(
                root=root, match_directory=match_out, requested=2, selected_ids=[3, 5]
            )
    evaluation_scope.reject_incompatible_root(
        root=root, match_directory=match_out, requested=None, selected_ids=[]
    )


def test_materialization_declares_every_written_file_and_a_distinct_stage_identity(tmp_path):
    root, match_out, _, scope = materialize(tmp_path, 2)
    outputs = evaluation_scope.materialized_outputs(root, match_out)
    assert [path.name for path in outputs] == [
        POINT_MAP_NAME,
        "broadcast_manifest.json",
        evaluation_scope.WORK_MANIFEST_NAME,
        "manifest.json",
        evaluation_scope.SCOPE_NAME,
    ]
    assert all(path.is_file() for path in outputs)
    identity = evaluation_scope.scope_stage_identity(scope)
    assert identity[:2] == ["--evaluation-first-attempts", "2"]
    assert scope["point_ledger"]["sha256"] in identity
    assert "3,5" in identity
    other = materialize(tmp_path / "other", 3)[3]
    assert evaluation_scope.scope_stage_identity(other) != identity


@pytest.mark.parametrize(
    "maximum, expected", [(None, 2), (2, 2)], ids=["omitted_uses_scope", "same_value"]
)
def test_selected_limit_adopts_the_scope_and_refuses_a_second_selection_rule(maximum, expected):
    scope = {"selected_attempts": 2}
    assert evaluation_scope.selected_limit(scope, maximum) == expected
    with pytest.raises(evaluation_scope.ScopeError, match="contradicts the declared"):
        evaluation_scope.selected_limit(scope, 3)
    assert evaluation_scope.selected_limit(None, 4) == 4


def test_summary_excludes_capped_and_out_of_scope_parents_from_child_counts():
    rows = [
        {"status": "completed", "verdict": {"complete_point": True}, "children": [{}, {}]},
        {"status": "not_run_cap"},
        {"status": "not_run_scope"},
        {"status": "not_run_scope"},
        {"status": "execution_failed"},
    ]
    counts = backend.summarize(rows, selected=2)
    assert counts["automatic_ledger_attempts"] == 5
    assert counts["selected_attempts"] == 2
    assert counts["not_run_cap_attempts"] == 1
    assert counts["not_run_scope_attempts"] == 2
    # One split parent's two children plus one unsplit selected failure; no phantom child
    # from either unselected status.
    assert counts["split_parent_count"] == 1
    assert counts["all_selected_child_attempts"] == 3
    assert counts["full_roster_evaluated"] is False


def test_window_camera_scope_limits_attempts_and_keeps_every_roster_row(tmp_path):
    ledger = write_ledger(tmp_path / "ledger.csv", default_rows(5))
    manifest = {"matches": [{"id": "match", "source_fps": 25, "point_ids": [3, 5, 7, 9, 11]}]}
    rows = evaluation_scope.ledger_rows(ledger)
    scope = evaluation_scope.build_scope(
        ledger=ledger,
        rows=rows,
        requested=2,
        match_id="match",
        source_video=ledger,
        upstream_root=tmp_path,
    )
    _, scoped = window_camera_inference.source_windows(manifest, rows, "match", None, scope)
    assert [row["status"] for row in scoped] == ["pending"] * 2 + ["not_run_scope"] * 3
    assert [row["clip"] for row in scoped] == [f"pt{i:04d}" for i in (3, 5, 7, 9, 11)]
    _, full = window_camera_inference.source_windows(manifest, rows, "match", None, None)
    assert [row["status"] for row in full] == ["pending"] * 5
    _, capped = window_camera_inference.source_windows(manifest, rows, "match", 2, None)
    assert [row["status"] for row in capped] == ["pending"] * 2 + ["not_run_cap"] * 3
    with pytest.raises(evaluation_scope.ScopeError, match="contradicts"):
        window_camera_inference.source_windows(manifest, rows, "match", 3, scope)


def test_window_camera_guards_a_scoped_root_and_a_foreign_forwarded_marker(tmp_path, monkeypatch):
    root, match_out, ledger, _ = materialize(tmp_path, 2)
    video = tmp_path / "video.mp4"
    monkeypatch.setattr(window_camera_inference.paths, "data_relative", str)
    monkeypatch.setattr(window_camera_inference, "discover_upstream", lambda *_: ([], []))
    with pytest.raises(ValueError, match="not this upstream root's scope"):
        window_camera_inference.run(
            upstream_root=root,
            match_id="match",
            source_video=video,
            output=tmp_path / "camera",
            evaluation_scope=tmp_path / "elsewhere.json",
        )
    ledger.write_text(ledger.read_text().replace("0.8", "0.85", 1))
    with pytest.raises(evaluation_scope.ScopeError, match="does not bind this automatic"):
        window_camera_inference.run(
            upstream_root=root,
            match_id="match",
            source_video=video,
            output=tmp_path / "camera2",
        )
    assert not (tmp_path / "camera2").exists()


def test_backend_scope_limits_attempts_while_the_full_roster_identity_stays_strict(
    tmp_path, monkeypatch
):
    root, match_out, ledger, scope = materialize(tmp_path, 2)
    backend._save(
        root / "broadcast_manifest.json",
        {"matches": [{"id": "match", "source_fps": 25, "point_ids": [3, 5, 7, 9, 11]}]},
    )
    video, policy_path = tmp_path / "video.mp4", tmp_path / "policy.json"
    policy = {"composed": True}
    backend._save(policy_path, policy)
    monkeypatch.setattr(backend, "load_policy", lambda _: policy)
    monkeypatch.setattr(backend.paths, "data_relative", str)
    monkeypatch.setattr(
        backend,
        "source_pts_cache",
        lambda *_: (_ for _ in ()).throw(RuntimeError("stop after roster")),
    )
    summary = backend.run_cached(
        upstream_root=root,
        match_id="match",
        source_video=video,
        policy_path=policy_path,
        output=tmp_path / "s6",
        evaluation_scope=evaluation_scope.scope_path(root),
    )
    document = json.loads(Path(summary).read_text())
    assert [row["status"] for row in document["attempts"]] == [
        "preparation_failed",
        "preparation_failed",
        "not_run_scope",
        "not_run_scope",
        "not_run_scope",
    ]
    assert [row["key"] for row in document["attempts"]] == [
        f"match__pt{i:04d}" for i in (3, 5, 7, 9, 11)
    ]
    assert document["counts"]["automatic_ledger_attempts"] == 5
    assert document["counts"]["selected_attempts"] == 2
    assert document["counts"]["not_run_scope_attempts"] == 3
    assert document["processing_scope"]["selected_point_ids"] == [3, 5]
    assert document["processing_scope_input"]["sha256"]
    # An identity disagreement between the full ledger and the full manifest still fails,
    # scope or no scope.
    backend._save(
        root / "broadcast_manifest.json",
        {"matches": [{"id": "match", "source_fps": 25, "point_ids": [3, 5]}]},
    )
    with pytest.raises(ValueError, match="attempt identities disagree"):
        backend.run_cached(
            upstream_root=root,
            match_id="match",
            source_video=video,
            policy_path=policy_path,
            output=tmp_path / "s6_b",
        )


def test_backend_refuses_a_maximum_that_contradicts_the_scope(tmp_path, monkeypatch):
    root, _, _, _ = materialize(tmp_path, 2)
    backend._save(
        root / "broadcast_manifest.json",
        {"matches": [{"id": "match", "source_fps": 25, "point_ids": [3, 5, 7, 9, 11]}]},
    )
    policy_path = tmp_path / "policy.json"
    backend._save(policy_path, {"composed": True})
    monkeypatch.setattr(backend, "load_policy", lambda _: {"composed": True})
    with pytest.raises(evaluation_scope.ScopeError, match="contradicts the declared"):
        backend.run_cached(
            upstream_root=root,
            match_id="match",
            source_video=tmp_path / "video.mp4",
            policy_path=policy_path,
            output=tmp_path / "s6",
            maximum=4,
        )
    assert not (tmp_path / "s6").exists()


def test_ordinary_runner_forwards_the_root_scope_marker_to_both_consumers(tmp_path, monkeypatch):
    root, match_out, ledger, _ = materialize(tmp_path, 2)
    captured = {}

    def fake_run_cached(**kwargs):
        captured["s6"] = kwargs
        return tmp_path / "summary.json"

    monkeypatch.setattr(backend, "run_cached", fake_run_cached)
    args = broadcast_runner.argparse.Namespace(
        out=root,
        match_id="match",
        shared_s6_policy=tmp_path / "policy.json",
        max_points=None,
        physical_player_motion=False,
        shared_s6_camera_backend="cached",
    )
    broadcast_runner.run_shared_s6_backend(args, tmp_path / "video.mp4")
    assert captured["s6"]["evaluation_scope"] == evaluation_scope.scope_path(root)
    # An unscoped root names nothing, so the ordinary dispatch keeps its exact arguments.
    evaluation_scope.scope_path(root).unlink()
    broadcast_runner.run_shared_s6_backend(args, tmp_path / "video.mp4")
    assert "evaluation_scope" not in captured["s6"]


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--evaluation-first-attempts", "0"], "at least one"),
        (["--evaluation-first-attempts", "2", "--max-points", "3"], "contradicts"),
        (
            ["--evaluation-first-attempts", "2", "--s6-backend", "legacy"],
            "requires --s6-backend shared_s6",
        ),
        (
            [
                "--evaluation-first-attempts",
                "2",
                "--s6-backend",
                "shared_s6",
                "--shared-s6-policy",
                "policy.json",
                "--physical-player-motion",
            ],
            "does not support --physical-player-motion",
        ),
    ],
)
def test_command_line_rejects_unusable_scopes_before_any_work(tmp_path, arguments, message):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cv.pipeline.broadcast_runner",
            "--video",
            "missing.mp4",
            "--out",
            str(tmp_path / "out"),
            "--match-id",
            "match",
            "--surface",
            "hard",
            *arguments,
        ],
        cwd=Path(broadcast_runner.__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert message in completed.stderr
    # Nothing was created: the refusal happens before the first stage.
    assert not (tmp_path / "out").exists()


def test_scope_flag_is_absent_by_default_in_the_command_line(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "cv.pipeline.broadcast_runner", "--help"],
        cwd=Path(broadcast_runner.__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--evaluation-first-attempts" in completed.stdout
    assert "Default absent" in " ".join(completed.stdout.split())


@pytest.mark.parametrize("mutation", ["request", "window", "continuation", "rule"])
def test_scope_rejects_internally_consistent_ids_with_tampered_declaration(tmp_path, mutation):
    root, _, ledger, scope = materialize(tmp_path, 2)
    if mutation == "request":
        scope["requested_attempts"] = 3
    elif mutation == "window":
        scope["selected_windows"][0]["rally_t_end"] = "999"
    elif mutation == "continuation":
        scope["continuation_outside_scope"] = [{"pt": 3}]
    else:
        scope["selection_rule"] = "best camera outputs"
    evaluation_scope.write_scope(evaluation_scope.scope_path(root), scope)
    with pytest.raises(evaluation_scope.ScopeError):
        evaluation_scope.verify_scope(root=root, match_id="match", ledger=ledger)


def test_scope_consumers_accept_requested_limit_above_full_ledger(tmp_path):
    root, _, ledger, scope = materialize(tmp_path, 9)
    assert evaluation_scope.selected_limit(scope, 9) == 5
    assert (
        evaluation_scope.guard_consumer_root(
            upstream_root=root,
            match_id="match",
            ledger=ledger,
            rows=evaluation_scope.ledger_rows(ledger),
            maximum=9,
        )[1]
        == 5
    )


@pytest.mark.parametrize("mutation", ["point_map", "manifest", "extra_directory", "video"])
def test_consumer_checks_actual_materialized_scope_inputs(tmp_path, mutation):
    root, match_out, ledger, _ = materialize(tmp_path, 2)
    video = tmp_path / "video.mp4"
    if mutation == "point_map":
        write_ledger(match_out / POINT_MAP_NAME, default_rows(3))
    elif mutation == "manifest":
        path = root / "manifest.json"
        doc = json.loads(path.read_text())
        doc["matches"][0]["point_ids"] = [3, 5, 7]
        path.write_text(json.dumps(doc))
    elif mutation == "extra_directory":
        extra = match_out / "audit_frames_native_1080/pt0007"
        extra.mkdir(parents=True)
    else:
        video.write_bytes(b"different source")
    with pytest.raises(evaluation_scope.ScopeError):
        evaluation_scope.guard_consumer_root(
            upstream_root=root,
            match_id="match",
            ledger=ledger,
            rows=evaluation_scope.ledger_rows(ledger),
            maximum=None,
            source_video=video,
        )


def test_same_ids_with_changed_ledger_windows_require_new_root(tmp_path):
    root, match_out, ledger, scope = materialize(tmp_path, 2)
    rows = default_rows()
    rows[0] = (3, 0, 6, 0.8, 0, "first", 1)
    write_ledger(ledger, rows)
    next_scope = evaluation_scope.build_scope(
        ledger=ledger,
        rows=evaluation_scope.ledger_rows(ledger),
        requested=2,
        match_id="match",
        source_video=tmp_path / "video.mp4",
        upstream_root=root,
    )
    with pytest.raises(evaluation_scope.ScopeError, match="scope inputs changed"):
        evaluation_scope.reject_incompatible_root(
            root=root,
            match_directory=match_out,
            requested=2,
            selected_ids=scope["selected_point_ids"],
            expected_scope=next_scope,
        )


def test_runner_rejects_different_existing_scope_before_source_work(tmp_path, monkeypatch):
    root, _, _, _ = materialize(tmp_path, 2)
    args = broadcast_runner.argparse.Namespace(
        out=root,
        match_id="match",
        evaluation_first_attempts=3,
        s6_backend="shared_s6",
        physical_player_motion=False,
        max_points=None,
    )
    monkeypatch.setattr(
        broadcast_runner, "reject_unusable_independent_contact_pose", lambda _: None
    )
    with pytest.raises(evaluation_scope.ScopeError, match="different execution scope"):
        broadcast_runner.run(args)


def test_automatic_preparation_requires_scope_producer_binding(tmp_path, monkeypatch):
    from types import SimpleNamespace

    supplied = tmp_path / "input.json"
    supplied.write_text("{}")
    evaluation_scope.scope_path(tmp_path).write_text("{}")
    inputs = SimpleNamespace(
        manifest=supplied,
        point_ledger=supplied,
        ball=supplied,
        events=supplied,
        cameras=supplied,
        players=supplied,
        source_video=supplied,
    )
    monkeypatch.setattr(
        backend,
        "_producer_bindings",
        lambda *args: {backend._identity(backend.provenance.file_record(supplied))},
    )
    with pytest.raises(ValueError, match="processing_scope.json"):
        backend.prepare_provenance(
            inputs=inputs,
            pts_receipt=supplied,
            receipts=[],
            parents=[],
            output=tmp_path / "pre_reconstruction.json",
        )
    assert not (tmp_path / "pre_reconstruction.json").exists()
