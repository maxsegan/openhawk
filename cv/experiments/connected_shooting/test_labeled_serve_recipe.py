"""Explicit source/role and unchanged isolated dispatch contracts."""

import sys

import pytest

from cv.experiments.connected_shooting import labeled_serve_recipe as recipe


def candidate(n, rank, depth, accepted=False):
    return dict(
        depth_hypothesis_m=depth,
        evidence=dict(input_only_rank_score=rank),
        measurement=dict(fit=dict(parameters=[0.0] * (5 + 6 * n))),
        accepted=accepted,
    )


def test_source_rank_ignores_acceptance_and_wrong_topology():
    search = dict(
        refined_candidates=[
            candidate(1, 0, 0, True),
            candidate(5, 4, 1, True),
            candidate(5, 2, 2, False),
        ]
    )
    selected, receipt = recipe.select_candidate(search, 5, "input-ranked-refined", None)
    assert selected is search["refined_candidates"][2]
    assert receipt["gate_used"] is False
    with pytest.raises(ValueError, match="cannot override"):
        recipe.select_candidate(search, 5, "input-ranked-refined", 1)
    selected, receipt = recipe.select_candidate(search, 5, "explicit-refined", 1)
    assert selected is search["refined_candidates"][1]
    assert receipt["source_stage"] == "explicit-refined"


def bounce_search():
    from cv.pipeline import s6_optional_contacts as optional

    rows = [candidate(2, 0, 8, True), candidate(2, 99, 9, False)]
    for row, rms in zip(rows, (8, 3), strict=True):
        row["measurement"]["rms_px"] = {"training": rms}
        row["evidence"]["input_geometry_penalty"] = 2
    selection = {
        "policy": "source_witness_optional_bounce_v1",
        "added_contact_count": 0,
        "added_bounce_count": 1,
        "added_parameter_count": 0,
        "occurrence_log_odds": 0.4,
        "training_observations": 100,
        "score": optional.physical_rank(rows[1], 1, 0.4, 100, added_parameters=0),
    }
    return {
        "coarse_candidates": rows,
        "refined_candidates": [],
        "optional_contact_selection": selection,
    }


def test_optional_bounce_replays_zero_extra_dimensions_and_source_rank():
    search = bounce_search()
    chosen, receipt = recipe.select_candidate(search, 2, "input-ranked-refined", None)
    assert chosen is search["coarse_candidates"][1]
    assert chosen["accepted"] is False
    assert receipt["source_rank"] == search["optional_contact_selection"]["score"]
    assert receipt["source_stage"] == "input-ranked-optional-contact"
    assert receipt["gate_used"] is False


@pytest.mark.parametrize(
    "field,value",
    [("added_contact_count", 1), ("added_bounce_count", -1), ("added_parameter_count", 6)],
)
def test_optional_bounce_cannot_claim_changed_contact_dimensions(field, value):
    search = bounce_search()
    search["optional_contact_selection"][field] = value
    with pytest.raises(ValueError, match="preserve contact parameter dimensions"):
        recipe.select_candidate(search, 2, "input-ranked-refined", None)


def event_inputs(role="serve"):
    events = [
        dict(event_type="contact", frame=10, shot_type=role),
        dict(event_type="contact", frame=30),
    ]
    labels = dict(attempt=dict(first_contact_frame=10), events=dict(records=events))
    packet = dict(attempts=[dict(point_clip="example", events=events)])
    return labels, packet, dict(events=events)


def test_shot_type_serve_qualifies_without_owner_flag():
    assert (
        recipe.prefix_serve_evidence(*event_inputs(), qualified=False)["origin"]
        == "original_typed_contact"
    )


def test_explicit_flag_never_overrides_nonserve_or_epoch_mismatch():
    with pytest.raises(ValueError, match="non-serve"):
        recipe.prefix_serve_evidence(*event_inputs("forehand"), qualified=True)
    labels, packet, search = event_inputs()
    search = dict(
        events=[
            dict(event_type="contact", frame=11, shot_type="serve"),
            dict(event_type="contact", frame=30),
        ]
    )
    with pytest.raises(ValueError, match="epochs disagree"):
        recipe.prefix_serve_evidence(labels, packet, search, qualified=True)


def test_isolated_delegation_preserves_arguments_and_restores_argv(monkeypatch, tmp_path):
    original = sys.argv
    seen = []
    monkeypatch.setattr(recipe.isolated_cli, "main", lambda: seen.append(sys.argv[:]))
    args = ["--output", str(tmp_path), "--contact-mode", "fixed", "--max-nfev", "17"]
    recipe.delegate_isolated(args, "input-ranked-refined", None)
    assert seen == [[recipe.isolated_cli.__file__, "--experiment", "isolated-serve-joint", *args]]
    assert sys.argv is original
    assert (tmp_path / "adapter.json").exists()


def test_prefix_preflight_records_paths_without_start_failure(monkeypatch, tmp_path):
    args = recipe.prefix_parser().parse_args(
        [
            item
            for name in ["search-report", "labels", "packet", "cameras", "pose-csv", "output"]
            for item in ["--" + name, str(tmp_path / name)]
        ]
    )
    recipe.run_prefix(args, "input-ranked-refined", None)
    import json

    report = json.loads((args.output / "report.json").read_text())
    assert report["status"] == "execution_failed"
    assert report["stage"] == "input_preflight"
    assert isinstance(report["configuration"]["labels"], str)


def test_explicit_net_ending_rejected_without_typed_net_event():
    labels, packet, search = event_inputs()
    labels["attempt"]["ending_kind"] = "net_error"
    with pytest.raises(ValueError, match="net ending"):
        recipe.prefix_serve_evidence(labels, packet, search, qualified=False)


def test_common_prefix_default_and_explicit_legacy_initializer(tmp_path):
    args = [
        item
        for name in ["search-report", "labels", "packet", "cameras", "pose-csv", "output"]
        for item in ["--" + name, str(tmp_path / name)]
    ]
    parser = recipe.prefix_parser()
    assert parser.parse_args(args).incoming_initializer == "bounded_front"
    assert (
        parser.parse_args(args + ["--incoming-initializer", "linear_clipped"]).incoming_initializer
        == "linear_clipped"
    )
