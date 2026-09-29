from types import SimpleNamespace
import json
import subprocess
import sys

import pytest

from cv.pipeline import artifact_cache, provenance, s6_broadcast_backend as backend
from cv.pipeline.broadcast_runner import run_shared_s6_backend


def test_pts_cache_uses_actual_decoder_records_once_and_rejects_mutation(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"native video")
    calls = []

    def decode(command, *, stdout, stderr, check):
        calls.append(command)
        stdout.write("frame|pts_time=1.120000\nframe|pts_time=1.160000\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend.subprocess, "run", decode)
    pts, receipt = backend.source_pts_cache(video, 25, tmp_path / "cache")
    assert pts.read_text().startswith("frame|pts_time=1.120000")
    assert backend._read(receipt)["audit"]["first_pts_seconds"] == 1.12
    assert backend.source_pts_cache(video, 25.0, tmp_path / "cache") == (pts, receipt)
    assert len(calls) == 1
    assert "frame=pts_time" in calls[0]
    assert "-threads" in calls[0] and "-select_streams" in calls[0]
    pts.write_text("frame|pts_time=0\nframe|pts_time=0.04\n")
    with pytest.raises(ValueError, match="cache identity or bytes changed"):
        backend.source_pts_cache(video, 25, tmp_path / "cache")


@pytest.mark.parametrize(
    "content,stderr,returncode",
    [
        ("frame|pts_time=0\nframe|pts_time=0.08\n", "", 0),
        ("frame|pts_time=0\nframe|pts_time=0.04\n", "decode error", 0),
        ("frame|pts_time=0\nframe|pts_time=0.04\n", "", 1),
    ],
)
def test_pts_decode_failures_never_become_cached_evidence(
    tmp_path, monkeypatch, content, stderr, returncode
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")

    def decode(command, **kwargs):
        kwargs["stdout"].write(content)
        kwargs["stderr"].write(stderr)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(backend.subprocess, "run", decode)
    with pytest.raises(ValueError, match="PTS"):
        backend.source_pts_cache(video, 25, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("receipt.json"))


def test_upstream_discovery_excludes_legacy_fitted_output_receipts(tmp_path):
    receipts = tmp_path / "match/run_manifests/stage_receipts"
    receipts.mkdir(parents=True)
    for name in ("tracking", "event_inference", "reconstruction_inputs", "reconstruction_3d"):
        (receipts / f"{name}.json").write_text("{}")
    (receipts.parent / "broadcast_runner.json").write_text("{}")
    (receipts.parent / "canonical_runner.json").write_text("{}")
    selected, parents = backend.discover_upstream(tmp_path, "match")
    assert {p.stem for p in selected} == {"tracking", "event_inference"}
    assert [p.name for p in parents] == ["canonical_runner.json"]


def test_producer_identity_rejects_manual_ancestry_and_mutation(tmp_path):
    parent = tmp_path / "parent.json"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    document = provenance.build_provenance(
        root=backend.REPO,
        mode=provenance.AUTOMATIC_MODE,
        source_videos=[provenance.file_record(video)],
    )
    backend._save(parent, document)
    receipt = tmp_path / "receipt.json"
    identity = {"configuration": {}, "inputs": [], "upstream_receipts": []}
    identity["fingerprint"] = artifact_cache._digest_json(identity)
    backend._save(
        receipt,
        {
            "schema": artifact_cache.SCHEMA,
            "identity": identity,
            "outputs": [
                {
                    "kind": "file",
                    "path_base": "TENNIS_DATA_ROOT",
                    "path": "automatic.csv",
                    "sha256": "abc",
                }
            ],
        },
    )
    assert backend._producer_bindings([receipt], [parent], video) == {
        ("TENNIS_DATA_ROOT", "automatic.csv", "abc")
    }
    identity["configuration"]["hand_edits"] = True
    backend._save(receipt, {"schema": artifact_cache.SCHEMA, "identity": identity, "outputs": []})
    with pytest.raises(ValueError):
        backend._producer_bindings([receipt], [parent], video)
    document["manual_overrides"] = [{"type": "reviewed_choice"}]
    backend._save(parent, document)
    with pytest.raises(provenance.ProvenanceError):
        backend._producer_bindings([receipt], [parent], video)


def test_primary_artifacts_need_upstream_output_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_producer_bindings", lambda *_: set())
    p = tmp_path / "observations.csv"
    p.write_text("automatically produced")
    inputs = SimpleNamespace(
        manifest=p, point_ledger=p, ball=p, events=p, cameras=p, players=p, source_video=p
    )
    with pytest.raises(ValueError, match="not producer-bound"):
        backend.prepare_provenance(inputs=inputs, pts_receipt=p, receipts=[], parents=[], output=p)


def test_regenerated_player_receipt_is_discovered_and_binds_actual_rows(tmp_path):
    directory = tmp_path / "match/run_manifests"
    receipt = directory / "stage_receipts/player_side_association.json"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"original source")
    players = tmp_path / "players.csv"
    players.write_text("native measured player rows")
    backend._save(players.with_suffix(".csv.coordinates.json"), {})
    parent = tmp_path / "provenance.json"
    backend._save(
        parent,
        provenance.build_provenance(
            root=backend.REPO,
            mode=provenance.AUTOMATIC_MODE,
            source_videos=[provenance.file_record(video)],
        ),
    )
    identity = {
        "stage": "player_side_association",
        "configuration": {},
        "inputs": [],
        "upstream_receipts": [],
    }
    identity["fingerprint"] = artifact_cache._digest_json(identity)
    backend._save(
        receipt,
        {
            "schema": artifact_cache.SCHEMA,
            "identity": identity,
            "outputs": [{"kind": "file", **provenance.file_record(players)}],
        },
    )
    # A fitted-output receipt cannot authorize a changed source observation.
    backend._save(directory / "stage_receipts/reconstruction_3d.json", {})
    receipts, parents = backend.discover_upstream(tmp_path, "match")
    assert receipts == [receipt]
    assert parents == [parent]
    inputs = SimpleNamespace(
        manifest=players,
        point_ledger=players,
        ball=players,
        events=players,
        cameras=players,
        players=players,
        source_video=video,
        extraction_receipt=players,
        source_pts=players,
        match_id="match",
        clip="pt0001",
    )
    output = backend.prepare_provenance(
        inputs=inputs,
        pts_receipt=players,
        receipts=receipts,
        parents=parents,
        output=tmp_path / "prepared.json",
    )
    assert output.is_file()
    players.write_text("unbound replacement rows")
    with pytest.raises(ValueError, match="not producer-bound.*players.csv"):
        backend.prepare_provenance(
            inputs=inputs,
            pts_receipt=players,
            receipts=receipts,
            parents=parents,
            output=tmp_path / "mutated.json",
        )


def test_stage_is_subprocess_and_rejects_foreign_or_assisted_result(tmp_path, monkeypatch):
    job = tmp_path / "job.json"
    backend._save(
        job,
        {"row": {"key": "match__pt0001", "automatic_provenance": {"sha256": "x"}}, "policy": {}},
    )
    output = tmp_path / "stage"
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        backend._save(
            output / "result.json",
            {
                "key": "match__pt0001",
                "observation_origin": "labeled",
                "human_derived_inputs": True,
                "verdict": {"complete_point": True},
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend.subprocess, "run", execute)
    result = backend.run_stage(job, output, 100)
    assert result["status"] == "execution_failed"
    assert "does not bind" in result["reason"]
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-m", "cv.pipeline.s6_labeled_stage"]
    assert kwargs["timeout"] == 100
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"


def test_summary_keeps_failures_and_cap_and_does_not_invent_reference_flights():
    rows = [
        {
            "status": "completed",
            "verdict": {"complete_point": True, "accepted_flight_count": 3, "flight_count": 4},
        },
        {"status": "preparation_held"},
        {"status": "execution_failed"},
        {"status": "not_run_cap"},
    ]
    counts = backend.summarize(rows, 3)
    assert counts["automatic_ledger_attempts"] == 4
    assert counts["selected_attempts"] == 3
    assert counts["complete_attempt_gate_percent"] == pytest.approx(100 / 3)
    assert counts["complete_roster_gate_lower_bound_percent"] == 25
    assert counts["not_run_cap_attempts"] == 1
    assert counts["full_roster_evaluated"] is False
    assert counts["emitted_flight_gate_percent"] == 75
    assert counts["reference_flights"] is None
    assert counts["reference_flight_yield_percent"] is None
    assert counts["useful_complete_yield_percent"] is None


def test_broadcast_opt_in_passes_original_root_and_bound_policy(tmp_path, monkeypatch):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return kwargs["output"] / "summary.json"

    monkeypatch.setattr(backend, "run_cached", run)
    args = SimpleNamespace(
        out=tmp_path,
        match_id="match",
        max_points=2,
        physical_player_motion=False,
        shared_s6_policy=tmp_path / "policy.json",
        shared_s6_workers=3,
        shared_s6_service_attempt_split="on",
    )
    result = run_shared_s6_backend(args, tmp_path / "original.mp4")
    assert result == tmp_path / "shared_s6/summary.json"
    assert captured["upstream_root"] == tmp_path
    assert captured["policy_path"] == args.shared_s6_policy
    assert captured["workers"] == 3 and captured["maximum"] == 2
    assert captured["camera_name"] == "camera_P_per_frame_v1.npz"
    assert captured["service_attempt_split"] == "on"
    assert "point_inputs" not in str(captured)
    args.shared_s6_policy = None
    with pytest.raises(ValueError, match="requires --shared-s6-policy"):
        run_shared_s6_backend(args, tmp_path / "original.mp4")


def test_backend_help_runs_without_importing_numerical_stage():
    result = subprocess.run(
        [sys.executable, "-m", "cv.pipeline.s6_broadcast_backend", "--help"],
        cwd=backend.REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy" in result.stdout and "--upstream-root" in result.stdout


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("player_override", [False, True])
@pytest.mark.parametrize("pose_override", [False, True])
@pytest.mark.parametrize("leading_components", [False, True])
@pytest.mark.parametrize("component_routing", [False, True])
def test_cached_orchestrator_counts_every_selected_failure_and_uses_one_policy(
    tmp_path,
    monkeypatch,
    override,
    player_override,
    pose_override,
    leading_components,
    component_routing,
):
    from dataclasses import field, make_dataclass
    from types import ModuleType

    root = tmp_path / "upstream"
    directory = root / "match"
    directory.mkdir(parents=True)
    (directory / "shot_views_v1.csv").write_text("shot_index,t_start,t_end\n0,0,100\n")
    backend._save(
        root / "broadcast_manifest.json",
        {"matches": [{"id": "match", "source_fps": 25, "point_ids": [1, 2, 3, 4, 5]}]},
    )
    (directory / "automatic_point_ledger_v1.csv").write_text(
        "pt,rally_t_start,rally_t_end\n"
        + "".join(f"{i},{i * 10},{i * 10 + 5}\n" for i in range(1, 6))
    )
    video, policy_path = tmp_path / "video.mp4", tmp_path / "policy.json"
    video.write_bytes(b"native")
    policy = {"composed": True} | ({"optional_contacts": "on"} if pose_override else {})
    if leading_components:
        policy.update(
            leading_event_components="declared_prefix",
            contact_components="unresolved_ending",
            contact_prefix_scope="unresolved_ending",
        )
    if component_routing:
        policy.update(
            contact_component_routing="unsupported_original_span",
            contact_components="unresolved_ending",
            contact_prefix_scope="unresolved_ending",
        )
    backend._save(policy_path, policy)
    monkeypatch.setattr(backend, "load_policy", lambda _: policy)
    monkeypatch.setattr(backend.paths, "data_relative", str)
    monkeypatch.setattr(backend, "source_pts_cache", lambda *_: (video, policy_path))
    monkeypatch.setattr(backend, "discover_upstream", lambda *_: ([], []))
    monkeypatch.setattr(backend, "prepare_provenance", lambda **kwargs: policy_path)
    adapter = ModuleType("cv.pipeline.s6_automatic_observations")
    adapter.AutomaticInputs = make_dataclass(
        "AutomaticInputs",
        [
            "match_id",
            "clip",
            "manifest",
            "point_ledger",
            "extraction_receipt",
            "frames_directory",
            "source_video",
            "source_pts",
            "ball",
            "events",
            "cameras",
            "players",
            "ancestry",
            ("optional_contact_pose", object, field(default=None)),
            ("optional_contact_views", object, field(default=None)),
        ],
    )
    prepared = []
    event_override = tmp_path / "qualified_events.json"
    event_override.write_text("[]")
    event_proof = tmp_path / "qualification_proof.json"
    event_proof.write_text("{}")
    monkeypatch.setattr(
        backend, "event_override_ancestry", lambda e, p, v: (event_proof, event_proof)
    )

    player_csv = tmp_path / "fresh_players.csv"
    player_csv.write_text("clip,frame,side\n")
    backend.resolution.coordinate_manifest_path(player_csv).write_text("{}")
    player_proof = tmp_path / "player_proof.json"
    player_proof.write_text("{}")
    monkeypatch.setattr(
        backend, "player_override_ancestry", lambda *_: (player_proof, player_proof)
    )
    pose_csv = tmp_path / "independent_pose.csv"
    pose_csv.write_text("clip,frame,side\n")
    backend.resolution.coordinate_manifest_path(pose_csv).write_text("{}")
    pose_proof = tmp_path / "pose_proof.json"
    pose_proof.write_text("{}")
    monkeypatch.setattr(backend, "optional_pose_ancestry", lambda *_: (pose_proof, pose_proof))

    def build(inputs, output, **options):
        prepared.append(inputs)
        assert inputs.manifest == root / "broadcast_manifest.json"
        assert inputs.events == (event_override if override else root / "event_emissions.json")
        assert inputs.ball.parent == directory
        assert inputs.players == (
            player_csv if player_override else directory / "player_boxes_25_native_sided_v1.csv"
        )
        assert inputs.optional_contact_pose == (pose_csv if pose_override else None)
        assert inputs.optional_contact_views == (
            directory / "shot_views_v1.csv" if pose_override else None
        )
        assert options.get("optional_contacts", "off") == ("on" if pose_override else "off")
        for name in (
            "leading_event_components",
            "contact_components",
            "contact_prefix_scope",
            "contact_component_routing",
        ):
            assert options.get(name, "off") == policy.get(name, "off")
        assert inputs.clip != "pt0005"
        if inputs.clip == "pt0003":
            raise ValueError("ancestry changed")
        record = {
            "status": "preparation_held" if inputs.clip == "pt0002" else "prepared",
            "reason": "missing event" if inputs.clip == "pt0002" else None,
            "evidence": {"path": "original_native_evidence.json", "sha256": "x"},
            "row": None
            if inputs.clip == "pt0002"
            else {
                "key": f"match__{inputs.clip}",
                "declared_flights": None,
                "automatic_provenance": {"sha256": "producer"},
            },
        }
        backend._save(output / "preparation.json", record)
        return record

    adapter.build_observations = build
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    jobs = []

    def stage(job, output, timeout):
        payload = backend._read(job)
        jobs.append(payload)
        if payload["row"]["key"].endswith("0004"):
            return {"status": "execution_failed", "reason": "solver exception"}
        return {
            "status": "completed",
            "verdict": {"complete_point": True, "accepted_flight_count": 2, "flight_count": 3},
        }

    monkeypatch.setattr(backend, "run_stage", stage)
    summary = backend.run_cached(
        upstream_root=root,
        match_id="match",
        source_video=video,
        policy_path=policy_path,
        output=tmp_path / "s6",
        maximum=4,
        workers=2,
        event_emissions=event_override if override else None,
        event_provenance=event_proof if override else None,
        player_observations=player_csv if player_override else None,
        player_provenance=player_proof if player_override else None,
        optional_contact_pose=pose_csv if pose_override else None,
        optional_contact_pose_provenance=pose_proof if pose_override else None,
    )
    document = backend._read(summary)
    assert [r["status"] for r in document["attempts"]] == [
        "completed",
        "preparation_held",
        "preparation_failed",
        "execution_failed",
        "not_run_cap",
    ]
    assert len(prepared) == 4
    assert ("player_input_override" in document) == player_override
    assert ("optional_contact_pose_input" in document) == pose_override
    if pose_override:
        assert document["optional_contact_pose_input"]["actor_players_replaced"] is False
    if override:
        assert document["event_input_override"]["emissions"] == provenance.file_record(
            event_override
        )
        assert document["event_input_override"]["producer"] == provenance.file_record(event_proof)
    else:
        assert "event_input_override" not in document
    assert len(jobs) == 2 and all(j["policy"] == policy for j in jobs)
    assert document["counts"]["selected_attempts"] == 4
    assert document["counts"]["automatic_ledger_attempts"] == 5
    assert document["counts"]["complete_attempt_gate_percent"] == 25
    assert document["counts"]["reference_flights"] is None
    assert document["attempts"][1]["evidence"]["path"] == "original_native_evidence.json"
    assert document["attempts"][1]["observation_row"] is None
    assert not (root / "reconstruction_3d.json").exists()
    assert not (root / "flight_ledger.json").exists()


def test_unrelated_automatic_parent_cannot_bless_another_video(tmp_path):
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(b"first video")
    second.write_bytes(b"second video")
    parent = tmp_path / "parent.json"
    backend._save(
        parent,
        provenance.build_provenance(
            root=backend.REPO,
            mode=provenance.AUTOMATIC_MODE,
            source_videos=[provenance.file_record(first)],
        ),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="do not bind the declared processing video"):
        backend._producer_bindings([receipt], [parent], second)


def test_actual_stage_cli_reaches_input_validation_with_fresh_directory(tmp_path):
    job = tmp_path / "job.json"
    backend._save(job, {"row": {"key": "automatic_missing_inputs"}, "policy": {}})
    output = tmp_path / "stage"
    result = backend.run_stage(job, output, 30)
    assert result["status"] == "execution_failed"
    log = (tmp_path / "stage.log").read_text()
    assert "fresh attempt output required" not in log
    assert "validate_row" in log and "surface" in log
    assert not output.exists()


@pytest.mark.parametrize("mode", ["off", "soft_l1"])
def test_prefix_pixel_loss_reaches_ordinary_shared_policy(tmp_path, monkeypatch, mode):
    from cv.pipeline import s6_labeled_stage as stage

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    prior = tmp_path / "prior.json"
    backend._save(prior, {"trained_prior": "fixture"})
    policy = dict(
        **stage.NUMERICAL_POLICY,
        coarse_iterations=1,
        refine_iterations=1,
        search_seconds=10,
        preparation="on",
        refinement="on",
        athlete_evidence="optional",
        serve_location_prior=provenance.file_record(prior),
        prefix_pixel_loss=mode,
    )
    path = tmp_path / "policy.json"
    backend._save(path, policy)
    assert backend.load_policy(path)["prefix_pixel_loss"] == mode
    policy["prefix_pixel_loss"] = "huber"
    backend._save(path, policy)
    with pytest.raises(ValueError, match="prefix_pixel_loss"):
        backend.load_policy(path)


@pytest.mark.parametrize("mode", ["off", "prediction_hinge"])
def test_prefix_following_ground_timing_reaches_ordinary_shared_policy(tmp_path, monkeypatch, mode):
    from cv.pipeline import s6_labeled_stage as stage

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    prior = tmp_path / "prior.json"
    backend._save(prior, {"trained_prior": "fixture"})
    policy = dict(
        **stage.NUMERICAL_POLICY,
        coarse_iterations=1,
        refine_iterations=1,
        search_seconds=10,
        preparation="on",
        refinement="on",
        athlete_evidence="optional",
        serve_location_prior=provenance.file_record(prior),
        prefix_following_ground_timing=mode,
    )
    path = tmp_path / "policy.json"
    backend._save(path, policy)
    assert backend.load_policy(path)["prefix_following_ground_timing"] == mode
    policy["prefix_following_ground_timing"] = "on"
    backend._save(path, policy)
    with pytest.raises(ValueError, match="prefix_following_ground_timing"):
        backend.load_policy(path)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_independent_toss_horizontal_prior_reaches_ordinary_shared_policy(
    tmp_path, monkeypatch, mode
):
    from cv.pipeline import s6_labeled_stage as stage

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    prior = tmp_path / "prior.json"
    backend._save(prior, {"trained_prior": "fixture"})
    policy = dict(
        **stage.NUMERICAL_POLICY,
        coarse_iterations=1,
        refine_iterations=1,
        search_seconds=10,
        preparation="on",
        refinement="on",
        athlete_evidence="optional",
        serve_location_prior=provenance.file_record(prior),
        independent_toss_horizontal_prior=mode,
    )
    path = tmp_path / "policy.json"
    backend._save(path, policy)
    assert backend.load_policy(path)["independent_toss_horizontal_prior"] == mode
    policy["independent_toss_horizontal_prior"] = "invalid"
    backend._save(path, policy)
    with pytest.raises(ValueError, match="independent_toss_horizontal_prior"):
        backend.load_policy(path)


def _production_policy_file(tmp_path, monkeypatch, **overrides):
    """A minimal valid shared policy for the production automatic entrypoint."""
    from cv.pipeline import s6_labeled_stage as stage

    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    prior = tmp_path / "prior.json"
    backend._save(prior, {"trained_prior": "fixture"})
    policy = dict(
        **stage.NUMERICAL_POLICY,
        coarse_iterations=1,
        refine_iterations=1,
        search_seconds=10,
        preparation="on",
        refinement="on",
        athlete_evidence="optional",
        observation_scope="on",
        serve_location_prior=provenance.file_record(prior),
        **overrides,
    )
    path = tmp_path / "policy.json"
    backend._save(path, policy)
    return path, policy


#: The six mechanisms the cold panels of 2026-09-18 and 2026-09-19 settled, with the
#: measurement that settled each. Five adopted, one rejected; all six are pinned here so
#: that neither an adoption nor a rejection can be reverted silently.
#:
#:   mechanism_ablation_v1         -- 12 arms, 72 cold cells, 926 fits
#:   abstained_occurrence_v1       -- 3 arms, 18 cold cells, 258 fits, anchor re-run in wave
#:   event_operating_point_v1      -- 4 arms, 24 cold cells, 374 fits, anchor re-run in wave
#:   transfer_evaluation_20260919  -- 3 arms, 15 cold cells, 153 fits, control re-run in
#:                                    wave, on FIVE BROADCASTS DISJOINT from the six above
ADOPTED = {
    # +-7 LA / +-8 AA: until the abstention key, the only mechanism that moved the product
    # metric at all.
    "leading_event_components": "declared_prefix",
    # +-9 LL / +-6-7 AL.
    "uncertain_original_occurrence": "retained_inventory",
    # +18 LA / +14 AA of 118 with zero false accepts -- the largest single-mechanism move
    # on the product metric measured on this panel. `declared_ending` is the strict subset
    # (+6 LA / +4 AA) that `retained_inventory` contains exactly.
    "automatic_abstained_occurrence": "retained_inventory",
    # event_operating_point_v1, 2026-09-19: 4 arms, 24 cold cells, 374 fits. A REPAIR --
    # the upstream runs an explicit uncalibrated 0.9907169 while the model ships this
    # calibrated operating point. +11 AA / +12 LA over the previous default with both
    # false-accept columns still 0; the 0.20 band floor reaches AA 59 but is the first
    # configuration ever scored with a false accept, and that trade was deferred.
    "automatic_event_operating_point": 0.9521754400734156,
    # transfer_evaluation_20260919, on five broadcasts DISJOINT from the panel the four keys
    # above were tuned on. The ablation left this key off at the noise floor (+2 LA / +1 AA)
    # EXPLICITLY "pending a second panel"; this is that panel, so the deferral is resolved,
    # not overridden. +12 LA / +3 AA of 72 -- and what it buys is a per-broadcast collapse the
    # other three adoptions CAUSE: without it they add five `unresolved contact prefix` holds
    # and lose source04_long_a2 LA outright, 9 of 15 matched flights to 0. The three-key arm's
    # pooled LA delta there is exactly zero, which is +8 on one broadcast and -9 on another.
    "contact_component_routing": "unsupported_original_span",
    # ball_jump_rejection_v1, 2026-09-19: whole fresh36 panel, 12 cold cells, 196 fits.
    # AL 84 -> 87 and AA 59 -> 61 of 118 with zero new extra accepts in any arm; the two
    # labelled-ball arms moved by exactly 0, so the wave's own noise was 0, not +-1. The one
    # lost AL flight is in an attempt the filter did not touch (1203 visible rows in both
    # arms), so it is wall-clock search noise. Tradeoffs, on the record: one development
    # panel, gains concentrated in source01, and fresh12 does not move at all.
    "automatic_ball_jump_rejection": "on",
    # The two acceptance-gate changes the owner approved in person on 2026-09-19 (memory
    # `owner-approved-gate-changes-20260919.md`). These are the only two entries here whose
    # authority is an owner decision rather than a panel: they change what "correct" means,
    # which AGENTS.md says may not happen silently. Pinned so neither can be reverted --
    # or widened to a third gate change -- without this test being edited deliberately.
    "bounce_interval_timing": "on",
    "automatic_ball_witness_sigma": "on",
    # stacked-fresh holdout 2026-09-21: AL 23 -> 41, AA 9 -> 10 of 102, crashes 5 -> 1,
    # zero extra accepts. Owner: make it default. Rollback: delete this line from
    # PIPELINE_COMPONENT_POLICY.
    "whole_point_seed_fallback": "on",
    # Owner decision 2026-09-23 ("yes definitely change this rule"): a flight whose ending is in
    # the ball's own track is searchable without a known next contact. Fresh panel A labelled
    # 44 -> 49 of 72, nothing lost, zero extra accepts (FINAL_REPORT_terminal_endpoint.md).
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "terminal_track_endpoint": "ball_track_last_sample",
    # Owner 2026-09-24: server-pose fallback ("no-regrets") and the failed-component split
    # fallback (FINAL_REPORT_split.md). Rollback: delete these lines.
    "server_pose_association": "nearest_detected",
    "failed_component_split_fallback": "on",
    # Owner 2026-09-24: "I'm okay with this loosening." Four acceptance checks, measured
    # in cv/experiments/ceiling_census/FINAL_REPORT_gates.md. Rollback: delete this line.
    "owner_gate_loosening_20260924": "near_miss_20260924",
    # Rome near-baseline refit, 2026-09-25. Same-wave panel A source03: no flight gained
    # or lost against the switch off, no extra accept, near-contact error 1.96 m -> 0.10 m.
    # Rollback: delete this line. FINAL_REPORT_fix_rome_camera.md.
    "near_baseline_refinement": "on",
    # Owner GO 2026-09-25. Fresh AA 40 -> 65 of 174, development AA 72 -> 78 of 190.
    # Rollback: delete these three lines from PIPELINE_COMPONENT_POLICY.
    # Numbers: cv/experiments/fix_gates/FINAL_REPORT_gates.md.
    "live_shot_camera_eligibility": "on",
    "serve_outgoing_release": "on",
    "preparation_prefix_isolation": "on",
    # Same-wave 2026-09-25 (FINAL_REPORT_fix_endings.md): fresh AL 118 -> 128 and
    # AA 40 -> 43 of 174, development AA 72 -> 73, nothing lost, no extra accepts.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "supported_flight_endings": "on",
    # Owner GO 2026-09-26 (FINAL_REPORT_endings_v2.md). Fresh AL 127 -> 132 and
    # AA 68 -> 70 of 174, cascade 106 -> 110, panel C AL 51 -> 53 of 77,
    # development LL +1 and AL +1. Nothing lost, no new extra accept.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "ending_ownership": "on",
    # Rome player sides, 2026-09-26 (FINAL_REPORT_rome_players.md): panel C
    # labelled 0 -> 8 of 16, nothing lost, no wrong-object accept. Rollback:
    # delete this line from PIPELINE_COMPONENT_POLICY.
    "shot_homography_propagation": "on",
    # Owner GO 2026-09-26. Same-wave AA: 40 gained, none lost, no extra accept.
    # Fresh AA 68 -> 85 of 174. Panel C AA 28 -> 40 of 77.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    # Numbers: cv/experiments/connected_shooting/FINAL_REPORT_held_release.md.
    "held_release_20260926": "on",
    # Panel D dead-ball fix (FINAL_REPORT_wrong_output_d.md). Rollback: delete this line.
    "out_bounce_ending": "on",
    # Serve toss proposal 2026-09-27: same-wave AA 6 gained, none lost, no extra accept.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "serve_toss_proposal": "on",
    # Ceiling census D 2026-09-27: split children of an exhausted joined component
    # get a search share. Fits +7, cascade +12, none lost, one origin-offset extra.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "component_search_budget": "exhausted_split",
    # Owner GO 2026-09-27: held flights close on a labelled boundary; fresh AL 134 -> 145.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    # Numbers: cv/experiments/ceiling_endings/FINAL_REPORT_ceiling_endings.md.
    "boundary_endings": "on",
    # Ceiling census D 2026-09-27: an interior flight held only by an ambiguous bounce is fitted
    # with it supplied. Fits 174 -> 209 of 322, cascade 27 -> 33, none lost, no new extra.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "ambiguous_interior_ground": "supplied",
    # Ceiling census D T5 2026-09-27: the camera standing height may vouch for a player the
    # ground-scale body gate refuses. Fits 2 -> 12 of 20, none lost, no extra.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "player_body_scale": "camera",
    # Held release v2 (FINAL_REPORT_held_release_v2.md). Rollback: delete this line.
    "held_release_v2_20260927": "on",
    # Fit seeds (c) 2026-09-28: serve side by serving reach. +2, none lost, no extra.
    # Rollback: delete this line from PIPELINE_COMPONENT_POLICY.
    "serve_side_association": "serve_reach",
    # Fit seeds (a) 2026-09-28: a labeller abstention between two contacts is an ambiguous
    # bounce. AL +1, none lost, no extra. Rollback: delete this line.
    "abstained_interior_ground": "ambiguous",
    "exit_partial_endings": "on",
    # Ceiling v2 2026-09-29: refine the camera once, refit off-paint courts, hold runs still off
    # the paint. Panel D AL/AA +1; Rome refused pieces recovered. Rollback: delete both lines.
    "near_baseline_refinement_once": "on",
    "court_paint_refinement": "abstain",
}
#: name -> the value a policy WITHOUT it normalizes to, or None when the key stays absent.
REJECTED = {
    # A measured NET NEGATIVE: removing it gains 3 LL, adding it alone loses 3, three ways.
    "fit_ground_witness": "off",
}


@pytest.mark.parametrize("flag", [False, True])
def test_cached_backend_refuses_sided_boxes_that_do_not_match_the_selection_default(
    tmp_path, monkeypatch, flag
):
    from cv.pipeline import player_side_association as psa
    from cv.pipeline import resolution as res

    root = tmp_path / "upstream"
    directory = root / "match"
    directory.mkdir(parents=True)
    backend._save(
        root / "broadcast_manifest.json",
        {"matches": [{"id": "match", "source_fps": 25, "point_ids": [1]}]},
    )
    boxes = directory / "player_boxes_25_native_sided_v1.csv"
    boxes.write_text("clip,frame,side\n")
    sidecar = {
        "artifact": boxes.name,
        "artifact_identity": res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
        psa.KEEP_UNIQUE_SIDECAR_KEY: flag,
    }
    boxes.with_name(boxes.name + ".coordinates.json").write_text(json.dumps(sidecar))
    video, policy_path = tmp_path / "video.mp4", tmp_path / "policy.json"
    video.write_bytes(b"native")
    backend._save(policy_path, {"composed": True})
    monkeypatch.setattr(backend, "load_policy", lambda _: {"composed": True})
    monkeypatch.setattr(backend.paths, "data_relative", str)
    kwargs = dict(
        upstream_root=root,
        match_id="match",
        source_video=video,
        policy_path=policy_path,
        output=tmp_path / "out",
    )
    if not flag:
        with pytest.raises(ValueError, match="keep_unique_admissible_frames"):
            backend.run_cached(**kwargs)
        return
    (directory / "automatic_point_ledger_v1.csv").write_text(
        "pt,rally_t_start,rally_t_end\n1,0,5\n"
    )
    try:
        backend.run_cached(**kwargs)
    except ValueError as error:
        assert "keep_unique_admissible_frames" not in str(error)


def test_the_production_policy_declares_exactly_the_adopted_mechanisms(tmp_path, monkeypatch):
    from cv.pipeline import s6_labeled_stage as stage

    assert stage.PIPELINE_COMPONENT_POLICY == ADOPTED
    path, _ = _production_policy_file(
        tmp_path,
        monkeypatch,
        contact_prefix_scope="unresolved_ending",
        contact_components="unresolved_ending",
    )
    loaded = backend.load_policy(path)
    for name, value in ADOPTED.items():
        assert loaded[name] == value, name
    for name, value in REJECTED.items():
        if value is None:
            assert name not in loaded, name
        else:
            assert loaded[name] == value, name


def test_emptying_the_mapping_is_the_whole_revert(tmp_path, monkeypatch):
    """The one-line revert property: no adoption is wired anywhere else."""
    from cv.pipeline import s6_labeled_stage as stage

    path, _ = _production_policy_file(
        tmp_path,
        monkeypatch,
        contact_prefix_scope="unresolved_ending",
        contact_components="unresolved_ending",
    )
    monkeypatch.setattr(stage, "PIPELINE_COMPONENT_POLICY", {})
    loaded = backend.load_policy(path)
    for name in ADOPTED:
        assert name not in loaded, name


@pytest.mark.parametrize("name", sorted(ADOPTED))
def test_an_explicit_production_policy_value_beats_the_measured_default(
    tmp_path, monkeypatch, name
):
    path, _ = _production_policy_file(
        tmp_path,
        monkeypatch,
        contact_prefix_scope="unresolved_ending",
        contact_components="unresolved_ending",
        **{key: "off" for key in ADOPTED},
    )
    loaded = backend.load_policy(path)
    assert loaded[name] == "off"


def test_the_operating_point_reaches_ordinary_video_inference(tmp_path, monkeypatch):
    """The adopted floor must be a TRIGGER, not only a forwarded value.

    `uncertain_original_occurrence` was adopted and then found inert on the automatic
    route because nothing there consumed it. This key is consumed directly by
    `s6_automatic_observations.build_observations`, so it has to be able to reach that
    call on its own.
    """
    assert "automatic_event_operating_point" in backend.COMPONENT_TRIGGERS
    assert "automatic_event_operating_point" in backend.COMPONENT_FORWARDED
    import inspect

    from cv.pipeline import s6_automatic_observations as automatic

    parameters = inspect.signature(automatic.build_observations).parameters
    assert "automatic_event_operating_point" in parameters
    assert parameters["automatic_event_operating_point"].default == "off"


def test_a_policy_without_the_component_contracts_keeps_its_original_receipt(tmp_path, monkeypatch):
    """Both admissions refuse to run as bare flags, so the default cannot reach a policy
    that does not declare the unresolved-ending prefix and contact-component contracts."""
    from cv.pipeline import s6_labeled_stage as stage

    path, policy = _production_policy_file(tmp_path, monkeypatch)
    loaded = backend.load_policy(path)
    for name in ADOPTED:
        assert name not in loaded, name
    # The low-level defaults and the absent-key pop are untouched, so an existing policy
    # receipt that never declared these keys still normalizes to exactly what it did before.
    settings = stage.shared_settings(policy)
    for name in ADOPTED:
        assert name not in settings, name
    assert loaded == {**policy, **settings}
    assert stage.pipeline_policy(policy) == policy


@pytest.mark.parametrize(
    "contracts",
    [
        {},
        {"contact_prefix_scope": "unresolved_ending"},
        {"contact_components": "unresolved_ending"},
        {"contact_prefix_scope": "coverage", "contact_components": "unresolved_ending"},
    ],
)
def test_pipeline_policy_needs_both_contracts_before_it_declares_anything(contracts):
    from cv.pipeline import s6_labeled_stage as stage

    assert stage.pipeline_policy(contracts) == contracts


def test_player_body_scale_is_omitted_when_absent():
    from cv.pipeline import s6_labeled_stage as stage

    assert "player_body_scale" not in stage.shared_settings({})
    on = stage.shared_settings({"player_body_scale": "camera", "shot_homography_propagation": "on"})
    assert on["player_body_scale"] == "camera"


def test_component_policy_runs_through_the_shared_executor(tmp_path, monkeypatch):
    from cv.pipeline import s6_attempt_execution

    row = {"key": "match__pt0001", "automatic_provenance": {"sha256": "x"}}
    job = tmp_path / "job.json"
    backend._save(job, {"row": row, "policy": {"contact_components": "unresolved_ending"}})
    seen = []

    def execute(row, output, policy, timeout):
        seen.append((output, policy, timeout))
        result = {
            "schema": "labeled_s6_contact_component_source_v1",
            "key": row["key"],
            "status": "components_measured_requires_native_review",
            "verdict": {"accepted_flight_count": 2},
        }
        backend._save(output / "cases" / row["key"] / "result.json", result)
        backend._save(output / "cases" / row["key"] / "component_plan.json", {})
        return result

    monkeypatch.setattr(s6_attempt_execution, "execute", execute)
    result = backend.run_stage(job, tmp_path / "stage", 100)
    assert result["status"] == "completed"
    assert result["verdict"] == {"accepted_flight_count": 2}
    assert seen[0][0] == tmp_path / "stage_run" and seen[0][2] == 100
    assert (tmp_path / "stage" / "component_plan.json").is_file()


def test_shared_executor_failure_keeps_its_status(tmp_path, monkeypatch):
    from cv.pipeline import s6_attempt_execution

    job = tmp_path / "job.json"
    backend._save(job, {"row": {"key": "k"}, "policy": {"whole_point_seed_fallback": "on"}})

    def execute(row, output, policy, timeout):
        result = {"key": "k", "status": "timeout", "reason": "budget exhausted"}
        backend._save(output / "cases" / "k" / "result.json", result)
        return result

    monkeypatch.setattr(s6_attempt_execution, "execute", execute)
    result = backend.run_stage(job, tmp_path / "stage", 100)
    assert (result["status"], result["reason"]) == ("timeout", "budget exhausted")
