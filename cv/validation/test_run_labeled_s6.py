"""Contracts that keep the cold observation-only denominator honest."""

from __future__ import annotations

import json

import pytest

from cv.pipeline import provenance, s6_labeled_stage as stage
from cv.validation.run_labeled_s6 import failure, parent_case_summary, summarize


def test_split_parent_requires_both_serves_even_in_a_subset():
    rows = [{"key": key} for key in ("fault", "second", "other")]
    cohort = {"rows": rows, "child_parent_map": {"fault": "source", "second": "source"}}

    def passed(key):
        return {
            "key": key,
            "verdict": {"complete_point": True, "accepted_flight_count": 1},
        }

    incomplete = parent_case_summary(rows, [passed("fault"), passed("other")], cohort)
    assert incomplete["parent_case_denominator"] == 2
    assert incomplete["gate_accepted_complete_parent_cases"] == 1
    subset = parent_case_summary(rows[:1], [passed("fault")], cohort)
    assert subset["parent_case_denominator"] == 1
    assert subset["gate_accepted_complete_parent_cases"] == 0
    assert not subset["parent_case_rows"][0]["all_children_selected"]
    complete = parent_case_summary(rows, list(map(passed, ("fault", "second", "other"))), cohort)
    assert complete["gate_accepted_complete_parent_cases"] == 2


def test_parent_mapping_rejects_unknown_child():
    with pytest.raises(ValueError, match="unknown child"):
        parent_case_summary([], [], {"rows": [], "child_parent_map": {"missing": "p"}})


def make_row(tmp_path, monkeypatch):
    monkeypatch.setattr(stage.paths, "REPO_ROOT", tmp_path)
    row = {
        "key": "point1",
        "declared_flights": 2,
        "surface": "clay",
        "player_order": ["A", "B"],
        "player_statures_m": [["A", 1.8], ["B", 1.9]],
    }
    for name in stage.INPUTS:
        path = tmp_path / f"{name}.json"
        document = {"attempt": {"ending_kind": "out_wide"}}
        if name == "packet":
            document = {"attempts": [{}], "external_label_binding": {"record": row["labels"]}}
        path.write_text(json.dumps(document))
        row[name] = {
            "path": path.name,
            "path_base": "repository",
            "sha256": provenance.file_sha256(path),
        }
    return row


def test_frozen_input_digest_and_packet_binding(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    stage.validate_row(row)
    path = tmp_path / "packet.json"
    path.write_text(
        json.dumps({"attempts": [{}], "external_label_binding": {"record": {"sha256": "wrong"}}})
    )
    row["packet"]["sha256"] = provenance.file_sha256(path)
    with pytest.raises(ValueError, match="does not bind"):
        stage.validate_row(row)
    (tmp_path / "labels.json").write_text("{}")
    with pytest.raises(ValueError, match="digest changed"):
        stage.validate_row(row)


@pytest.mark.parametrize(
    "field", ["search_report", "parameters", "candidate_index", "solver_flags"]
)
def test_no_per_point_fitted_inputs_or_policy(tmp_path, monkeypatch, field):
    row = make_row(tmp_path, monkeypatch)
    row[field] = "forbidden"
    with pytest.raises(ValueError, match="unsupported per-attempt"):
        stage.validate_row(row)


def test_cold_command_and_unknown_serve_number(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    command = stage.search_arguments(
        row,
        stage.validate_row(row),
        tmp_path / "out",
        {"coarse_iterations": 20, "refine_iterations": 80},
    )
    assert "--baseline" not in command
    assert "--search-report" not in command
    assert "--shared-state-second-stage-reference" not in command
    for name, expected in stage.NUMERICAL_POLICY.items():
        assert command[command.index("--" + name.replace("_", "-")) + 1] == expected
    assert "--serve-number-uncertain" in command
    assert command[command.index("--surface") + 1] == "clay"


def test_failures_and_pending_rows_stay_in_denominator():
    rows = [{"key": str(i), "surface": "hard", "declared_flights": i + 1} for i in range(3)]
    measured = {
        "key": "0",
        "status": "measured_requires_native_review",
        "verdict": {"complete_point": True, "accepted_flight_count": 1, "flight_count": 1},
    }
    result = summarize(rows, [measured, failure(rows[1], "preparation_failed", "missing")])
    assert result["attempt_denominator"] == 3
    assert result["declared_flight_denominator"] == 6
    assert result["gate_accepted_complete_attempts"] == 1
    assert result["pending_attempts"] == 1
    assert result["correct_complete_yield"] is None
    assert result["incorrect_accept_count"] is None


def test_search_incumbent_is_off_by_default_and_needs_a_deadline(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    assert stage.shared_settings(base)["search_incumbent"] == "off"
    assert "--search-incumbent" not in stage.search_arguments(row, inputs, tmp_path / "a", base)
    on = base | {"search_seconds": 30.0, "search_incumbent": "on"}
    command = stage.search_arguments(row, inputs, tmp_path / "b", on)
    assert command[command.index("--search-incumbent") + 1] == "on"
    assert command[command.index("--search-seconds") + 1] == "30.0"
    with pytest.raises(ValueError, match="search deadline"):
        stage.search_arguments(row, inputs, tmp_path / "c", base | {"search_incumbent": "on"})
    with pytest.raises(ValueError, match="unsupported global search_incumbent"):
        stage.shared_settings(base | {"search_incumbent": "maybe"})


def test_fitting_witness_flag_is_shared_default_off_and_acceptance_mode_unchanged(
    tmp_path, monkeypatch
):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    output = tmp_path / "out"
    original = stage.search_arguments(row, inputs, output, base)
    off = stage.search_arguments(row, inputs, output, base | {"fit_ground_witness": "off"})
    on = stage.search_arguments(
        row, inputs, output, base | {"fit_ground_witness": "interval_ballistic_center"}
    )
    assert original == off
    assert "--fit-ground-witness" not in off
    index = on.index("--fit-ground-witness")
    assert on[index + 1] == "interval_ballistic_center"
    assert on[:index] + on[index + 2 :] == off
    with pytest.raises(ValueError, match="unsupported global fit_ground_witness"):
        stage.shared_settings(base | {"fit_ground_witness": "case_selected"})


@pytest.mark.parametrize("enabled", [False, True])
def test_fitting_witness_runner_serializes_only_explicit_opt_in(tmp_path, monkeypatch, enabled):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "no numerical solve in CLI contract test")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--fit-ground-witness", "interval_ballistic_center"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    policy = json.loads((output / "manifest.json").read_text())["policy"]
    assert ("fit_ground_witness" in policy) == enabled
    assert policy == seen[0]
    assert stage.shared_settings(policy)["fit_ground_witness"] == (
        "interval_ballistic_center" if enabled else "off"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_prefix_net_revisit_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--prefix-net-revisit", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["prefix_net_revisit"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["prefix_net_revisit"] == expected
    assert stage.shared_settings(written["policy"])["prefix_net_revisit"] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_toss_player_prior_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--toss-player-prior", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["toss_player_prior"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["toss_player_prior"] == expected
    assert stage.shared_settings(written["policy"])["toss_player_prior"] == expected
    with pytest.raises(ValueError, match="unsupported global toss_player_prior"):
        stage.shared_settings({"toss_player_prior": "clay_only"})


@pytest.mark.parametrize("enabled", [False, True])
def test_player_camera_coordinates_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--player-camera-coordinates", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["player_camera_coordinates"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["player_camera_coordinates"] == expected
    assert stage.shared_settings(written["policy"])["player_camera_coordinates"] == expected
    with pytest.raises(ValueError, match="unsupported global player_camera_coordinates"):
        stage.shared_settings({"player_camera_coordinates": "clay_only"})


def test_pose_space_sidecar_is_read_only_when_the_artifact_ships_one(tmp_path):
    csv_path = tmp_path / "player_pose.csv"
    csv_path.write_text("clip,frame\n")
    assert stage.pose_space_sidecar(csv_path) is None
    sidecar = tmp_path / "player_pose.csv.coordinates.json"
    sidecar.write_text(
        json.dumps(
            {
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 960, "height": 540},
            }
        )
    )
    space = stage.pose_space_sidecar(csv_path)
    assert space["artifact_size"] == {"width": 960, "height": 540}
    assert space["record"]["sha256"] == provenance.file_sha256(sidecar)


@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_ground_normal_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--terminal-ground-normal", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["terminal_ground_normal"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["terminal_ground_normal"] == expected
    assert stage.shared_settings(written["policy"])["terminal_ground_normal"] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_net_coupling_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--terminal-net-coupling", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["terminal_net_coupling"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["terminal_net_coupling"] == expected
    assert stage.shared_settings(written["policy"])["terminal_net_coupling"] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_ground_coupling_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--terminal-ground-coupling", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["terminal_ground_coupling"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["terminal_ground_coupling"] == expected
    assert stage.shared_settings(written["policy"])["terminal_ground_coupling"] == expected


def test_terminal_net_seed_is_global_default_off_and_forwarded(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    assert stage.shared_settings(base)["terminal_net_seed"] == "off"
    assert "--terminal-net-seed" not in stage.search_arguments(row, inputs, tmp_path / "off", base)
    command = stage.search_arguments(
        row, inputs, tmp_path / "on", base | {"terminal_net_seed": "on"}
    )
    assert command[command.index("--terminal-net-seed") + 1] == "on"
    with pytest.raises(ValueError, match="unsupported global terminal_net_seed"):
        stage.shared_settings(base | {"terminal_net_seed": "one_case_only"})


def test_observation_partition_is_global_explicit_and_default_legacy(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    assert stage.shared_settings(base)["observation_partition"] == "fifth_frame_withheld"
    assert "--observation-partition" not in stage.search_arguments(
        row, inputs, tmp_path / "legacy", base
    )
    command = stage.search_arguments(
        row, inputs, tmp_path / "all", base | {"observation_partition": "all_native"}
    )
    assert command[command.index("--observation-partition") + 1] == "all_native"
    with pytest.raises(ValueError, match="unsupported global observation_partition"):
        stage.shared_settings(base | {"observation_partition": "just_bad_frames"})


@pytest.mark.parametrize("enabled", [False, True])
def test_interior_restoration_geometry_only_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--interior-restoration-geometry-only", "on", "--interior-ground-normal", "sweep"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["interior_restoration_geometry_only"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["interior_restoration_geometry_only"] == expected
    assert (
        stage.shared_settings(written["policy"])["interior_restoration_geometry_only"] == expected
    )


@pytest.mark.parametrize("override", [None, 360.0])
def test_interior_block_seconds_is_default_180_and_recorded_in_manifest(
    tmp_path, monkeypatch, override
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append((policy, timeout))
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    argv += ["--timeout-seconds", "4200"]
    if override is not None:
        argv += ["--interior-block-seconds", str(override)]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = 180.0 if override is None else override
    policy, timeout = seen[0]
    assert policy["interior_block_seconds"] == expected and timeout == 4200
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["interior_block_seconds"] == expected
    assert written["timeout_seconds"] == 4200
    assert stage.shared_settings(written["policy"])["interior_block_seconds"] == expected
    assert stage.shared_settings({})["interior_block_seconds"] == 180.0


@pytest.mark.parametrize("value", ["0", "-5", "inf", "nan"])
def test_interior_block_seconds_rejects_non_positive_or_non_finite(tmp_path, monkeypatch, value):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    argv = ["run", "--manifest", str(manifest), "--output", str(tmp_path / "out"), "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv + ["--interior-block-seconds", value])
    with pytest.raises(ValueError, match="positive finite shared interior block budget"):
        runner.main()
    assert not (tmp_path / "out").exists()
    for bad in (0, -1.0, float("inf"), float("nan"), "360", True, None):
        with pytest.raises(ValueError, match="unsupported global interior_block_seconds"):
            stage.shared_settings({"interior_block_seconds": bad})
    assert stage.shared_settings({"interior_block_seconds": 360})["interior_block_seconds"] == 360.0


def test_athlete_evidence_is_required_by_default_and_optional_is_explicit(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    assert stage.shared_settings(base)["athlete_evidence"] == "required"
    strict = stage.search_arguments(row, inputs, tmp_path / "a", base)
    assert "--athlete-evidence" not in strict
    optional = base | {"athlete_evidence": "optional"}
    # Known evidence resolves identically under both policies.
    known = stage.search_arguments(row, inputs, tmp_path / "a", optional)
    assert [a for a in known if a != "optional" and a != "--athlete-evidence"] == strict
    assert known[known.index("--athlete-evidence") + 1] == "optional"
    assert known.count("--player-order") == 2 and known.count("--player-stature") == 2
    # Absent roster evidence: the original policy refuses, optional declares the absence.
    absent = {k: v for k, v in row.items() if k not in ("player_order", "player_statures_m")}
    with pytest.raises(ValueError, match="required athlete evidence"):
        stage.search_arguments(absent, inputs, tmp_path / "c", base)
    command = stage.search_arguments(absent, inputs, tmp_path / "d", optional)
    assert "--player-order" not in command and "--player-stature" not in command
    assert command[command.index("--athlete-evidence") + 1] == "optional"
    assert "--athlete-prior-mode" in command  # geometry constraints stay on the stature arm
    # Statures without an order are malformed under both policies, never silently dropped.
    malformed = absent | {"player_statures_m": [["A", 1.8]]}
    for policy in (base, optional):
        with pytest.raises(ValueError, match="malformed"):
            stage.search_arguments(malformed, inputs, tmp_path / "e", policy)
    with pytest.raises(ValueError, match="unsupported global athlete_evidence"):
        stage.shared_settings(base | {"athlete_evidence": "on"})


def test_shared_cauchy_policy_is_explicit_and_never_a_per_row_choice(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    off = stage.search_arguments(row, inputs, tmp_path / "off", base)
    assert "--athlete-root-reach-loss" not in off
    on = stage.search_arguments(
        row, inputs, tmp_path / "on", base | {"athlete_root_reach_loss": "cauchy"}
    )
    index = on.index("--athlete-root-reach-loss")
    assert on[index + 1] == "cauchy"
    # No artifact or numerical settings are changed apart from the declared loss.
    normalized = [
        a.replace(str(tmp_path / "on"), str(tmp_path / "off")) for a in on[:index] + on[index + 2 :]
    ]
    assert normalized == off
    with pytest.raises(ValueError, match="unsupported per-attempt"):
        stage.validate_row(row | {"athlete_root_reach_loss": "cauchy"})


def test_observed_net_seed_is_global_explicit_and_scale_is_validated(tmp_path, monkeypatch):
    row = make_row(tmp_path, monkeypatch)
    inputs = stage.validate_row(row)
    base = {"coarse_iterations": 20, "refine_iterations": 80}
    assert stage.shared_settings(base)["observation_net_seed"] == "off"
    assert "--observation-net-seed" not in stage.search_arguments(
        row, inputs, tmp_path / "off", base
    )
    command = stage.search_arguments(
        row,
        inputs,
        tmp_path / "on",
        base | {"observation_net_seed": "on", "net_seed_speed_scale_mps": 15.0},
    )
    assert command[command.index("--observation-net-seed") + 1] == "on"
    assert float(command[command.index("--net-seed-speed-scale-mps") + 1]) == 15.0
    for value in (True, 0.0, -1.0, float("inf"), float("nan"), "15"):
        with pytest.raises(ValueError, match="net seed speed"):
            stage.shared_settings(base | {"net_seed_speed_scale_mps": value})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("field", ["interior_sparse_wings", "terminal_ground_sparse_wings"])
def test_sparse_wings_flags_are_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled, field
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += [
            "--" + field.replace("_", "-"),
            "on",
            "--interior-ground-normal",
            "sweep",
            "--terminal-ground-coupling",
            "on",
        ]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0][field] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"][field] == expected
    assert stage.shared_settings(written["policy"])[field] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_independent_toss_horizontal_prior_flag_is_default_off_and_recorded_in_manifest(
    tmp_path, monkeypatch, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--independent-toss-horizontal-prior", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "on" if enabled else "off"
    assert seen[0]["independent_toss_horizontal_prior"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["independent_toss_horizontal_prior"] == expected
    assert stage.shared_settings(written["policy"])["independent_toss_horizontal_prior"] == expected
    with pytest.raises(ValueError, match="unsupported global independent_toss_horizontal_prior"):
        stage.shared_settings({"independent_toss_horizontal_prior": "clay_only"})


@pytest.mark.parametrize("override", [None, "prediction_hinge"])
def test_prefix_following_ground_timing_cli_defaults_off_and_reaches_the_policy(
    tmp_path, monkeypatch, override
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if override is not None:
        argv += ["--prefix-following-ground-timing", override]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    expected = "off" if override is None else override
    assert seen[0]["prefix_following_ground_timing"] == expected
    written = json.loads((output / "manifest.json").read_text())
    assert written["policy"]["prefix_following_ground_timing"] == expected
    assert stage.shared_settings(written["policy"])["prefix_following_ground_timing"] == expected


def test_prefix_following_ground_timing_cli_rejects_an_unsupported_mode(tmp_path, monkeypatch):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out"),
            "--prefix-following-ground-timing",
            "on",
        ],
    )
    with pytest.raises(SystemExit):
        runner.main()


@pytest.mark.parametrize(
    "flag,key,on_value",
    [
        ("--failed-component-split-fallback", "failed_component_split_fallback", "on"),
        ("--component-search-budget", "component_search_budget", "exhausted_retry"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_split_and_search_budget_cli_are_absent_unless_named(
    tmp_path, monkeypatch, flag, key, on_value, enabled
):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += [flag, on_value]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    policy = json.loads((output / "manifest.json").read_text())["policy"]
    assert (key in policy) == enabled
    assert policy == seen[0]
    assert stage.shared_settings(policy).get(key, "off") == (on_value if enabled else "off")


@pytest.mark.parametrize("enabled", [False, True])
def test_whole_point_seed_fallback_cli_is_absent_unless_on(tmp_path, monkeypatch, enabled):
    import sys

    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": [row]}))
    seen = []

    def execute(row, output, policy, timeout):
        seen.append(policy)
        return failure(row, "execution_failed", "stubbed")

    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "out"
    argv = ["run", "--manifest", str(manifest), "--output", str(output), "--workers", "1"]
    if enabled:
        argv += ["--whole-point-seed-fallback", "on"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    policy = json.loads((output / "manifest.json").read_text())["policy"]
    assert ("whole_point_seed_fallback" in policy) == enabled
    assert policy == seen[0]
    assert stage.shared_settings(policy).get("whole_point_seed_fallback", "off") == (
        "on" if enabled else "off"
    )


@pytest.mark.parametrize("fallback", [None, "off", "on"])
def test_execute_whole_point_seed_death_respects_declared_fallback(tmp_path, monkeypatch, fallback):
    from cv.validation import run_labeled_s6 as runner

    row = make_row(tmp_path, monkeypatch)
    recipe = dict(stage.NUMERICAL_POLICY, coarse_iterations=1, refine_iterations=1)
    if fallback is not None:
        recipe = {**recipe, "whole_point_seed_fallback": fallback}
    output = tmp_path / "out"
    (output / "cases").mkdir(parents=True)
    (output / "jobs").mkdir()
    (output / "logs").mkdir()

    class Process:
        pid = 1

        def wait(self, timeout=None):
            return 1

    def popen(command, stdout=None, **kwargs):
        if stdout is not None:
            stdout.write("CandidateFailure: no completed candidate after 20 numerical failures\n")
            stdout.flush()
        return Process()

    called = []

    def run_source(*args, **kwargs):
        called.append(kwargs)
        return dict(status="components_measured_requires_native_review", seed_failure_fallback=True)

    monkeypatch.setattr("cv.pipeline.s6_attempt_execution.subprocess.Popen", popen)
    monkeypatch.setattr(
        "cv.pipeline.s6_contact_components.as_unresolved_source", lambda packet: packet
    )
    monkeypatch.setattr("cv.pipeline.s6_component_orchestration.run_source", run_source)
    result = runner.execute(row, output, recipe, 30)
    if fallback == "on":
        assert called
        assert result["seed_failure_fallback"] is True
        return
    assert not called
    assert result["status"] == "execution_failed"
    assert "no completed candidate after 20 numerical failures" in result["reason"]
    assert not result.get("seed_failure_fallback")
