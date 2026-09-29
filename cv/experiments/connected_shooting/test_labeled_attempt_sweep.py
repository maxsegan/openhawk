import json

from cv.experiments.connected_shooting import labeled_attempt_sweep as sweep
from cv.pipeline import provenance


def test_cached_preparation_requires_exact_label_binding(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}\n")
    cache = tmp_path / "cache"
    source = cache / "case" / "inputs"
    source.mkdir(parents=True)
    packet = {
        "external_label_binding": {"record": provenance.file_record(labels)},
        "configuration": {"transport_reference_policy": "contact_anchor"},
    }
    (source / "packet.json").write_text(json.dumps(packet))
    (source / "cameras.json").write_text("{}\n")
    (source / "qualification.json").write_text("{}\n")
    case = sweep.AttemptCase("case", "labels.json", "hard", 1)

    destination = tmp_path / "output" / "inputs"
    result = sweep._cached_preparation(case, labels, cache, destination)

    assert result["status"] == "reused_provenance_matched_cache"
    assert (destination / "packet.json").read_text() == json.dumps(packet)
    assert (destination / "packet.json").stat().st_ino == (source / "packet.json").stat().st_ino


def test_cached_preparation_rejects_changed_label(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}\n")
    cache = tmp_path / "cache"
    source = cache / "case" / "inputs"
    source.mkdir(parents=True)
    packet = {
        "external_label_binding": {"record": provenance.file_record(labels)},
        "configuration": {"transport_reference_policy": "contact_anchor"},
    }
    (source / "packet.json").write_text(json.dumps(packet))
    (source / "cameras.json").write_text("{}\n")
    (source / "qualification.json").write_text("{}\n")
    labels.write_text('{"changed": true}\n')
    case = sweep.AttemptCase("case", "labels.json", "hard", 1)

    assert sweep._cached_preparation(case, labels, cache, tmp_path / "output") is None


def test_cached_preparation_rejects_erased_uncertain_bounce_with_same_label_hash(tmp_path):
    event = dict(
        event_type="bounce",
        frame=160.5,
        frame_interval=[159, 162],
        status="ambiguous",
        timing_status="abstained_exact_epoch",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"events": {"records": [event]}}))
    source = tmp_path / "cache" / "case" / "inputs"
    source.mkdir(parents=True)
    packet = {
        "external_label_binding": {"record": provenance.file_record(labels)},
        "configuration": {"transport_reference_policy": "contact_anchor"},
        "attempts": [{"events": []}],
    }
    (source / "packet.json").write_text(json.dumps(packet))
    for name in ("cameras.json", "qualification.json"):
        (source / name).write_text("{}\n")
    case = sweep.AttemptCase("case", "labels.json", "grass", 1)
    destination = tmp_path / "out"
    assert sweep._cached_preparation(case, labels, tmp_path / "cache", destination) is None
    assert not destination.exists()
    packet["attempts"][0]["events"] = [{**event, "exact_epoch_observed": False}]
    (source / "packet.json").write_text(json.dumps(packet))
    assert sweep._cached_preparation(case, labels, tmp_path / "cache", destination) is not None


def test_prepared_uncertain_bounce_preserves_interval_and_does_not_claim_exact_epoch():
    event = dict(
        event_type="bounce",
        frame=160.5,
        frame_interval=[159, 162],
        status="ambiguous",
        timing_status="abstained_exact_epoch",
    )
    labels = {"events": {"records": [event]}}
    for changed in (
        {"frame_interval": [160, 161]},
        {"exact_epoch_observed": True},
        {"status": "labeled"},
    ):
        cached = {**event, "exact_epoch_observed": False, **changed}
        assert not sweep._timing_uncertain_events_match(
            {"attempts": [{"events": [cached]}]}, labels
        )


def test_every_cohort_case_names_a_present_frozen_label():
    root = sweep.paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1"
    for case in sweep.COHORTS["labeled17"]:
        assert (root / case.label_file).is_file()
    assert len(sweep.COHORTS["nine"]) == 9
    assert len(sweep.COHORTS["labeled17"]) == 17
    assert sweep.COHORTS["labeled17"][:9] == sweep.COHORTS["nine"]
    keys = [case.key for case in sweep.COHORTS["labeled17"]]
    assert len(set(keys)) == len(keys)


def test_the_athlete_arm_has_one_stature_for_every_named_hitter():
    for case in sweep.COHORTS["labeled17"]:
        names = [name for name, _ in case.player_statures_m]
        assert names and len(set(names)) == len(names)
        assert all(1.4 < value < 2.2 for _, value in case.player_statures_m)


def test_covers_contacts_rejects_an_unsided_or_incomplete_artifact(tmp_path):
    labels = {
        "attempt": {"clip": "pt0001"},
        "events": {
            "records": [
                {"status": "labeled", "event_type": "contact", "frame": 10.5},
                {"status": "labeled", "event_type": "bounce", "frame": 20.0},
            ]
        },
    }
    unsided = tmp_path / "unsided.csv"
    unsided.write_text("clip,frame,x0,y0,x1,y1\npt0001,f_0011.jpg,1,2,3,4\n")
    assert sweep._covers_contacts(unsided, labels) is False

    missing = tmp_path / "missing.csv"
    missing.write_text("clip,frame,side,court_x,court_y\npt0001,f_0012.jpg,near,1.0,2.0\n")
    assert sweep._covers_contacts(missing, labels) is False

    complete = tmp_path / "complete.csv"
    complete.write_text("clip,frame,side,court_x,court_y\npt0001,f_0011.jpg,near,1.0,2.0\n")
    assert sweep._covers_contacts(complete, labels) is True


def test_branch_optimizer_exits_publish_every_stage_and_the_feasibility_pass():
    def candidate(stage, status, iterations, survived, feasibility=None):
        return {
            "stage": stage,
            "depth_hypothesis_m": 23.0,
            "wall_seconds": 1.0,
            "measurement": {
                "rms_px": {"training": 12.5, "withheld": 13.0},
                "fit": {
                    "success": status == 0,
                    "status": status,
                    "message": "message",
                    "iterations": iterations,
                    "initial_cost": 10.0,
                    "final_cost": 5.0,
                    "primary_optimizer": {
                        "success": status == 0,
                        "status": status,
                        "message": "primary",
                        "iterations": iterations,
                    },
                    "final_gate_feasibility": feasibility,
                },
            },
            "evidence": {"survived": survived, "death_reasons": [] if survived else ["gate"]},
        }

    feasibility = {
        "method": "SLSQP_local_parameter_scaling_zero_displacement",
        "success": True,
        "status": 0,
        "message": "certified",
        "iterations": 3,
        "terminal_slack_at_final_limits": [0.1],
    }
    rows = sweep.branch_optimizer_exits(
        {
            "coarse_candidates": [candidate("coarse", 8, 5, False)],
            "refined_candidates": [candidate("refined", 8, 5, False, feasibility)],
        }
    )

    assert [row["stage"] for row in rows] == ["coarse", "refined"]
    assert rows[0]["primary_optimizer"]["status"] == 8
    assert rows[0]["final_gate_feasibility"] is None
    assert rows[1]["final_gate_feasibility"]["method"].startswith("SLSQP_local")
    assert rows[1]["training_rms_px"] == 12.5
    assert rows[1]["passes_unchanged_acceptance_gates"] is False
    assert rows[1]["solver_experimental_arm"] is None


def test_the_surface_arm_reaches_the_search_subprocess(tmp_path):
    """End to end through the exact runner the sweep uses, not just the dictionary."""
    import sys

    from cv.experiments.connected_shooting import labeled_attempt_sweep as sweep

    code = (
        "import json;from physics import surface_model as sm;"
        "print(json.dumps(sm.parse('grass').as_dict()))"
    )
    environment = sweep.surface_model_environment("grass@w=1.0600,a=0.1500,base=fresh")
    log = tmp_path / "with.log"
    sweep._run([sys.executable, "-c", code], log, check=True, environment=environment)
    selected = json.loads(log.read_text())
    assert selected["wear"] == 1.06
    assert selected["base_source"] == "literature"

    log = tmp_path / "without.log"
    sweep._run([sys.executable, "-c", code], log, check=True, environment={})
    plain = json.loads(log.read_text())
    assert plain["wear"] == 1.0 and plain["base_source"] == "reference"
    assert plain["provenance"] == "inert"


def test_the_surface_arm_is_none_unless_it_is_asked_for():
    from cv.experiments.connected_shooting import labeled_attempt_sweep as sweep

    case = sweep.COHORTS["labeled17"][0]
    assert sweep.surface_model_spec(case, "hard", "off", {}) is None
    assert sweep.surface_model_environment(None) == {}


def test_regional_v2_is_a_grass_only_hard_centred_broadcast_model():
    root = sweep.paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1"
    cases = sweep.discovered_cases(root)
    grass = next(case for case in cases if case.key == "wim2025f_pt0002_a01")
    spec = sweep.surface_model_spec(grass, "regional_v2", "off", {})
    model = sweep.surface_model.parse(spec)
    assert model.surface == "grass"
    assert model.base_source == "hard_reference"
    assert model.provenance == "spec"

    controls = [
        next(case for case in cases if case.surface == surface) for surface in ("hard", "clay")
    ]
    for control in controls:
        assert sweep.fitted_surface(control, "regional_v2") == control.surface
        assert sweep.surface_model_spec(control, "regional_v2", "off", {}) is None
        assert sweep.surface_model_environment(None) == {}


def test_all_discovered_attempts_have_toss_or_dense_prior_evidence():
    root = sweep.paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1"
    cases = sweep.discovered_cases(root)

    assert len(cases) == 89
    assert all(sweep.toss_label_path(case, root) is not None for case in cases)
