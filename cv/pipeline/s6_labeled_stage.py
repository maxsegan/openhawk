"""Cold, deterministic S6 research stage on explicitly supplied observations.

The historical ``agent_whole_point_search`` name denotes ordinary numerical
Python code: this entrypoint never invokes a model, agent or labeling service.
Prepared camera/observation packets are inputs; fitted ball states are not.
Run via ``python -m cv.validation.run_labeled_s6 --help``.

``whole_point_seed_fallback`` is production default-on via
``PIPELINE_COMPONENT_POLICY`` (applied by ``s6_broadcast_backend.load_policy``).
``shared_settings`` still defaults the key off and pops it when absent, so an
existing policy receipt that does not name it is unchanged. Rollback: delete
the ``whole_point_seed_fallback`` line from ``PIPELINE_COMPONENT_POLICY``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from cv.pipeline import paths, provenance

SCHEMA = "labeled_s6_observations_v1"
INPUTS = ("labels", "packet", "cameras", "pose_csv")
NUMERICAL_POLICY = {
    "refine_inequalities": "terminal_only",
    "anchor_residuals": "present",
    "seed_restarts": "three",
    "reference_restart": "on",
}
ROW_FIELDS = {
    "key",
    "declared_flights",
    "surface",
    *INPUTS,
    "pose_image_scale",
    "player_order",
    "player_statures_m",
    "serve_number",
    "serve_number_uncertain",
    "notes",
    "observation_origin",
    "automatic_provenance",
}


def resolve(record: dict) -> Path:
    base = record.get("path_base")
    if base == "repository":
        return paths.REPO_ROOT / record["path"]
    if base == "TENNIS_DATA_ROOT":
        return paths.data_root() / record["path"]
    raise ValueError("observation records require repository or TENNIS_DATA_ROOT paths")


def validate_row(row: dict) -> dict[str, Path]:
    unknown = set(row) - ROW_FIELDS
    if unknown:
        raise ValueError(f"unsupported per-attempt fields: {sorted(unknown)}")
    if not row.get("key") or Path(row["key"]).name != row["key"]:
        raise ValueError("one safe attempt key required")
    from cv.pipeline import s6_input_origin

    automatic = s6_input_origin.origin(row) == s6_input_origin.AUTOMATIC
    if row["surface"] not in ("hard", "clay", "grass"):
        raise ValueError("known surface required")
    if automatic:
        if row.get("declared_flights") is not None:
            raise ValueError("automatic inference cannot accept a reference-flight denominator")
    elif int(row["declared_flights"]) < 1:
        raise ValueError("positive flight denominator required")
    result = {}
    for name in INPUTS:
        record = row[name]
        if record.get("missing"):
            raise FileNotFoundError(
                f"missing prepared observation input: {name}: {record.get('path')}"
            )
        path = resolve(record)
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared observation input: {name}: {path}")
        if provenance.file_sha256(path) != record.get("sha256"):
            raise ValueError(f"frozen observation digest changed: {name}")
        result[name] = path
    packet = json.loads(result["packet"].read_text())
    binding = packet.get("external_label_binding", {}).get("record", {})
    if binding.get("sha256") != row["labels"]["sha256"]:
        raise ValueError("prepared packet does not bind the frozen label document")
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("prepared packet must contain one observed attempt")
    s6_input_origin.validate_automatic(row, result, resolve)
    return result


def search_arguments(row: dict, inputs: dict[str, Path], output: Path, policy: dict) -> list[str]:
    """Build an allowlisted cold command; no per-case numerical flags are accepted."""
    labels = json.loads(inputs["labels"].read_text())
    from cv.pipeline import s6_contact_prefix_runtime

    labels = s6_contact_prefix_runtime.labels_for_scope(
        labels, json.loads(inputs["packet"].read_text())
    )
    evidence = output / "serve_number.json"
    save(
        evidence,
        {
            "source": (
                "automatic attempt metadata; serve number uncertain"
                if row.get("observation_origin") == "automatic"
                else "frozen observed attempt metadata"
            ),
            "number": row.get("serve_number", 1),
            "uncertain": row.get("serve_number_uncertain", True),
        },
    )
    args = []
    for name, path in inputs.items():
        args.extend(["--" + name.replace("_", "-"), str(path)])
    args.extend(
        [
            "--output",
            str(output / "search"),
            "--serve-number",
            str(row.get("serve_number", 1)),
            "--serve-number-evidence",
            str(evidence),
            "--surface",
            row["surface"],
            "--pose-image-scale",
            str(row.get("pose_image_scale", 1.0)),
            "--athlete-prior-mode",
            "stature_pose_soft",
            "--observation-fallback",
            "on",
            *(
                ["--server-pose-association", "nearest_detected"]
                if shared_settings(policy).get("server_pose_association") == "nearest_detected"
                else []
            ),
            "--coarse-iterations",
            str(policy["coarse_iterations"]),
            "--refine-iterations",
            str(policy["refine_iterations"]),
            "--bounce-bracket-frames",
            "2",
            "--ending-kind",
            labels["attempt"]["ending_kind"],
        ]
    )
    from cv.experiments.connected_shooting import observation_scope
    from cv.pipeline import s6_contact_prefix_scope

    packet = json.loads(inputs["packet"].read_text())
    if s6_contact_prefix_scope.unresolved_input(packet["attempts"][0]):
        # A retained original inventory is owned by the contact-component stage,
        # which builds one command per prepared component. There is no source
        # topology here for a search to model.
        raise ValueError(
            "a retained unresolved-ending inventory is owned by the contact-component "
            "stage, not by a source search command"
        )
    contract = observation_scope.validate(packet["attempts"][0], labels)
    if contract is not None:
        if shared_settings(policy)["observation_scope"] != "on":
            raise ValueError("observation-scoped inputs require explicit shared policy")
        from cv.pipeline import s6_contact_prefix_runtime

        from cv.pipeline import s6_component_scope

        if s6_component_scope.active(packet["attempts"][0]):
            if shared_settings(policy).get("contact_components", "off") != "unresolved_ending":
                raise ValueError("component inputs require explicit shared policy")
            args.extend(["--observation-scope", "on", "--contact-components", "unresolved_ending"])
        elif s6_contact_prefix_runtime.active(packet["attempts"][0]):
            # The bound contract's own mode, never a policy string the packet
            # does not carry.
            mode = s6_contact_prefix_runtime.bound_mode(packet["attempts"][0])
            if shared_settings(policy)["contact_prefix_scope"] != mode:
                raise ValueError("contact-prefix inputs require their explicit shared scope policy")
            args.extend(["--observation-scope", "on", "--contact-prefix-scope", mode])
        else:
            args.extend(["--observation-scope", "on", "--terminal-rebound", "on"])
    if shared_settings(policy)["terminal_context_ownership"] == "on":
        args.extend(["--terminal-context-ownership", "on"])
    if shared_settings(policy)["optional_contacts"] == "on":
        from cv.pipeline import s6_optional_contacts

        binding = packet.get("optional_contact_witness")
        if binding is None:
            raise ValueError("optional contact policy requires normal prepared witness input")
        witness_path = resolve(binding)
        if provenance.file_sha256(witness_path) != binding["sha256"]:
            raise ValueError("optional contact witness changed")
        if row.get("observation_origin") != "automatic":
            from cv.pipeline import s6_independent_event_proposals

            s6_independent_event_proposals.validate_research_packet(
                packet, json.loads(witness_path.read_text()), resolve
            )
        final = shared_settings(policy)["optional_final_contacts"]
        # The prepared witness declares the scope it was built under; a policy
        # that disagrees with it is a preparation error, not a runtime choice.
        if s6_optional_contacts.declared_final_scope(json.loads(witness_path.read_text())) != (
            final == "on"
        ):
            raise ValueError("optional contact witness scope differs from the shared policy")
        from cv.pipeline import s6_contact_timing

        timing_policy = shared_settings(policy)["optional_contact_timing"]
        if s6_contact_timing.policy(json.loads(witness_path.read_text())) != timing_policy:
            raise ValueError("optional contact timing witness differs from shared policy")
        from cv.pipeline import s6_contact_composition

        composition_policy = shared_settings(policy)["optional_contact_composition"]
        if (
            s6_contact_composition.policy(json.loads(witness_path.read_text()))
            != composition_policy
        ):
            raise ValueError("optional contact composition witness differs from shared policy")
        from cv.pipeline import s6_witnessed_interior_contacts

        interior_policy = shared_settings(policy)["optional_interior_contacts"]
        if (
            s6_witnessed_interior_contacts.policy(json.loads(witness_path.read_text()))
            != interior_policy
        ):
            raise ValueError("optional interior contact witness differs from shared policy")
        from cv.pipeline import s6_independent_event_proposals as independent

        if (
            independent.policy(json.loads(witness_path.read_text()))
            != shared_settings(policy)["independent_event_proposals"]
        ):
            raise ValueError("independent proposal witness differs from shared policy")
        args.extend(["--optional-contact-witness", str(witness_path)])
        if interior_policy != "off":
            args.extend(["--optional-interior-contacts", interior_policy])
        if composition_policy != "off":
            args.extend(["--optional-contact-composition", composition_policy])
        if timing_policy != "off":
            args.extend(["--optional-contact-timing", timing_policy])
        if final == "on":
            args.extend(["--optional-final-contacts", "on"])
    elif shared_settings(policy)["optional_contact_composition"] != "off":
        raise ValueError("optional contact composition requires optional contacts enabled")
    elif shared_settings(policy)["optional_contact_timing"] != "off":
        raise ValueError("optional contact timing requires optional contacts enabled")
    elif shared_settings(policy)["optional_final_contacts"] == "on":
        raise ValueError("optional final contacts require the optional contact policy")
    elif shared_settings(policy)["independent_event_proposals"] == "on":
        raise ValueError("independent event proposals require optional contacts")
    elif shared_settings(policy)["optional_interior_contacts"] == "on":
        raise ValueError("optional interior contacts require the optional contact policy")
    if shared_settings(policy)["optional_bounces"] != "off":
        if row.get("observation_origin") != "automatic":
            raise ValueError("optional bounce witnesses require explicit automatic inputs")
        binding = packet.get("optional_bounce_witness")
        if binding is None or binding != packet["attempts"][0].get("optional_bounce_witness"):
            raise ValueError("optional bounce policy requires its prepared source witness")
        witness_path = resolve(binding)
        if provenance.file_sha256(witness_path) != binding["sha256"]:
            raise ValueError("optional bounce witness changed")
        from cv.pipeline import s6_optional_bounces

        s6_optional_bounces.validate_interior_policy_binding(
            json.loads(witness_path.read_text()), shared_settings(policy)["optional_bounces"]
        )
        args.extend(["--optional-bounce-witness", str(witness_path)])
    if shared_settings(policy)["optional_conditional_grounds"] != "off":
        # A composition policy over both prepared witnesses: it changes no
        # witness document, so it is declared to the search and replayed by
        # the readers rather than written into preparation.
        if (
            shared_settings(policy)["optional_contacts"] != "on"
            or shared_settings(policy)["optional_interior_contacts"] != "on"
            or shared_settings(policy)["optional_bounces"] == "off"
        ):
            raise ValueError(
                "optional conditional grounds require optional contacts, the interior "
                "contact scope and an optional bounce witness"
            )
        args.extend(
            [
                "--optional-conditional-grounds",
                shared_settings(policy)["optional_conditional_grounds"],
            ]
        )
    if policy.get("search_seconds") is not None:
        args.extend(["--search-seconds", str(policy["search_seconds"]), "--skip-render"])
    if shared_settings(policy)["search_incumbent"] == "on":
        if policy.get("search_seconds") is None:
            raise ValueError("search incumbent retention requires a shared search deadline")
        args.extend(["--search-incumbent", "on"])
    if shared_settings(policy)["observation_partition"] != "fifth_frame_withheld":
        args.extend(["--observation-partition", shared_settings(policy)["observation_partition"]])
    if shared_settings(policy)["interior_contact_epochs"] != "off":
        args.extend(
            ["--interior-contact-epochs", shared_settings(policy)["interior_contact_epochs"]]
        )
    if shared_settings(policy).get("depth_conditioned_seed", "off") == "on":
        args.extend(["--depth-conditioned-seed", "on"])
    if shared_settings(policy)["event_ground_seed"] == "on":
        args.extend(["--event-ground-seed", "on"])
    # Owner-approved 2026-09-19; absent (the pre-sign-off default) emits no argument.
    if shared_settings(policy).get("bounce_interval_timing", "off") == "on":
        args.extend(["--bounce-interval-timing", "on"])
    if shared_settings(policy).get("automatic_ball_witness_sigma", "off") == "on":
        args.extend(["--automatic-ball-witness-sigma", "on"])
    if shared_settings(policy).get("serve_side_association", "off") != "off":
        args.extend(["--serve-side-association", shared_settings(policy)["serve_side_association"]])
    if shared_settings(policy).get("final_bounce_anchor", "off") == "on":
        args.extend(["--final-bounce-anchor", "on"])
    if shared_settings(policy).get("exit_partial_endings", "off") == "on":
        args.extend(["--exit-partial-endings", "on"])
    if "net_cord_response" in policy:
        args.extend(["--net-cord-response", shared_settings(policy)["net_cord_response"]])
    if "net_cord_tape_band_m" in policy:
        args.extend(
            ["--net-cord-tape-band-m", str(shared_settings(policy)["net_cord_tape_band_m"])]
        )
    if shared_settings(policy)["fit_ground_witness"] != "off":
        args.extend(["--fit-ground-witness", shared_settings(policy)["fit_ground_witness"]])
    if shared_settings(policy)["passive_bounce_response"] != "off":
        args.extend(
            ["--passive-bounce-response", shared_settings(policy)["passive_bounce_response"]]
        )
    if shared_settings(policy)["ground_settling"] == "on":
        args.extend(["--ground-settling", "on"])
    if shared_settings(policy)["observation_net_seed"] != "off":
        args.extend(
            [
                "--observation-net-seed",
                shared_settings(policy)["observation_net_seed"],
                "--net-seed-speed-scale-mps",
                str(shared_settings(policy)["net_seed_speed_scale_mps"]),
            ]
        )
    if shared_settings(policy)["terminal_net_seed"] == "on":
        args.extend(["--terminal-net-seed", "on"])
    for name, value in NUMERICAL_POLICY.items():
        if policy.get(name, value) != value:
            raise ValueError("unsupported shared numerical policy; change the stage explicitly")
        args.extend(["--" + name.replace("_", "-"), value])
    if row.get("serve_number_uncertain", True):
        args.append("--serve-number-uncertain")
    args.extend(athlete_evidence_arguments(row, shared_settings(policy)["athlete_evidence"]))
    if shared_settings(policy)["athlete_root_reach_loss"] != "quadratic":
        args.extend(
            ["--athlete-root-reach-loss", shared_settings(policy)["athlete_root_reach_loss"]]
        )
    if shared_settings(policy)["missing_player_position"] != "off":
        args.extend(
            ["--missing-player-position", shared_settings(policy)["missing_player_position"]]
        )
    from cv.pipeline import s6_input_origin

    from cv.pipeline import s6_first_contact_role

    s6_first_contact_role.validate(
        packet["attempts"][0], labels, requested=shared_settings(policy)["first_contact_role"]
    )
    requested_role = shared_settings(policy)["first_contact_role"]
    if requested_role in s6_first_contact_role.BOUND_ROLES:
        args.extend(["--first-contact-role", requested_role])
    prior_path = s6_input_origin.serve_prior_path(row, policy, resolve)
    if prior_path is not None:
        args.extend(["--serve-location-prior", str(prior_path)])
    return args


def athlete_evidence_arguments(row: dict, athlete_evidence: str) -> list[str]:
    """Pass supplied roster evidence through unchanged; declare its absence explicitly.

    ``required`` is the original labeled contract: a roster order and statures
    are row inputs and a row without them refuses.  ``optional`` uses the same
    evidence identically when present and otherwise tells the search that no
    athlete evidence exists, so its stature-scaled terms abstain rather than
    receiving invented heights.  Statures without an order are malformed under both.
    """
    order = list(row.get("player_order") or [])
    statures = list(row.get("player_statures_m") or [])
    if statures and not order:
        raise ValueError("player statures without a player order are malformed athlete evidence")
    if athlete_evidence == "required" and not (order and statures):
        raise ValueError(
            "required athlete evidence needs row player_order and player_statures_m; "
            "declare the shared optional policy to run without them"
        )
    args = [] if athlete_evidence == "required" else ["--athlete-evidence", athlete_evidence]
    for name in order:
        args.extend(["--player-order", name])
    for name, stature in statures:
        args.extend(["--player-stature", f"{name}={stature}"])
    return args


def save(path: Path, value: dict) -> None:
    from cv.experiments.connected_shooting.serve_timing_profile import jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pose_space_sidecar(pose_csv: Path) -> dict | None:
    """The frozen artifact's declared coordinate space, when it ships one; never inferred."""
    sidecar = pose_csv.with_name(pose_csv.name + ".coordinates.json")
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text()) | {"record": provenance.file_record(sidecar)}


def shared_settings(policy: dict) -> dict:
    """Global input/preparation choices; row-level overrides are forbidden."""
    result = {
        name: policy.get(name, default)
        for name, default in (
            ("contact_prefix_scope", "off"),
            ("contact_components", "off"),
            ("whole_point_seed_fallback", "off"),
            ("failed_component_split_fallback", "off"),
            ("component_search_budget", "off"),
            ("leading_event_components", "off"),
            ("contact_component_routing", "off"),
            ("terminal_track_endpoint", "off"),
            ("supported_flight_endings", "off"),
            ("ending_ownership", "off"),
            ("out_bounce_ending", "off"),
            ("boundary_endings", "off"),
            ("ambiguous_interior_ground", "off"),
            ("abstained_interior_ground", "off"),
            ("serve_side_association", "off"),
            ("contact_reversal_timing", "off"),
            ("net_cord_mid_flight", "off"),
            ("serve_tape_clearance", "off"),
            ("server_pose_association", "off"),
            ("uncertain_original_occurrence", "off"),
            ("automatic_abstained_occurrence", "off"),
            ("automatic_event_operating_point", "off"),
            ("automatic_ball_jump_rejection", "off"),
            ("automatic_ball_streak_centre", "off"),
            ("net_cord_response", "tape_clip"),
            ("net_cord_tape_band_m", 0.05),
            ("observation_scope", "off"),
            ("optional_contacts", "off"),
            ("optional_final_contacts", "off"),
            ("optional_interior_contacts", "off"),
            ("independent_event_proposals", "off"),
            ("optional_contact_timing", "off"),
            ("optional_contact_composition", "off"),
            ("interior_contact_epochs", "off"),
            ("optional_bounces", "off"),
            ("optional_conditional_grounds", "off"),
            ("terminal_context_ownership", "off"),
            ("preparation", "off"),
            ("terminal_net_tail", "off"),
            ("observed_horizon_tail", "off"),
            ("terminal_net_membership", "off"),
            ("refinement", "off"),
            ("following_bounce_intervals", "off"),
            ("joint_toss_requalification", "off"),
            ("earlier_toss_support", "off"),
            ("net_physical_eligibility", "off"),
            ("net_ground_normal", "off"),
            ("net_mesh_height", "off"),
            ("net_first_ground_candidates", "off"),
            ("net_first_contact_toss", "off"),
            ("net_ground_horizontal", "off"),
            ("prefix_local_boundary", "off"),
            ("prefix_source_incumbent", "off"),
            ("prefix_pixel_loss", "off"),
            ("prefix_following_ground_timing", "off"),
            ("prefix_net_revisit", "off"),
            ("terminal_impact_intervals", "off"),
            ("terminal_ground_normal", "off"),
            ("terminal_net_coupling", "off"),
            ("terminal_ground_coupling", "off"),
            ("interior_ground_normal", "off"),
            ("terminal_net_seed", "off"),
            ("observation_net_seed", "off"),
            ("ground_settling", "off"),
            ("passive_bounce_response", "off"),
            ("net_seed_speed_scale_mps", 15.0),
            ("terminal_ground_sparse_wings", "off"),
            ("interior_sparse_wings", "off"),
            ("interior_short_blocks", "off"),
            ("interior_ground_horizontal", "off"),
            ("interior_restoration_geometry_only", "off"),
            ("first_contact_role", "unspecified"),
            ("observation_partition", "fifth_frame_withheld"),
            ("search_incumbent", "off"),
            ("event_ground_seed", "off"),
            # Owner-approved acceptance-gate changes, 2026-09-19 (see
            # PIPELINE_COMPONENT_POLICY, which turns both on at the production entrypoint).
            ("bounce_interval_timing", "off"),
            ("automatic_ball_witness_sigma", "off"),
            ("owner_gate_loosening_20260924", "off"),
            ("fit_ground_witness", "off"),
            ("toss_player_prior", "off"),
            ("independent_toss_horizontal_prior", "off"),
            ("serve_ground_normal", "off"),
            ("player_camera_coordinates", "off"),
            ("athlete_evidence", "required"),
            ("athlete_root_reach_loss", "quadratic"),
            ("missing_player_position", "off"),
            ("near_baseline_refinement", "off"),
            ("court_paint_refinement", "off"),
            ("near_baseline_refinement_once", "off"),
            ("shot_homography_propagation", "off"),
            ("player_body_scale", "off"),
            ("interior_block_seconds", 180.0),
            # Upstream gate releases. Absent is popped below, so an existing policy
            # receipt is unchanged. Production default-on is the three lines in
            # PIPELINE_COMPONENT_POLICY. Explicit ``off`` is the rollback.
            ("live_shot_camera_eligibility", "off"),
            ("serve_outgoing_release", "off"),
            ("preparation_prefix_isolation", "off"),
            # High-confidence residual holds and orphan contacts. Absent is popped
            # below, so an existing receipt stays off. Production default-on is
            # PIPELINE_COMPONENT_POLICY. Explicit ``off`` is the rollback.
            ("held_release_20260926", "off"),
            # Supported band abstentions (0.8 to the threshold, live span, track turn,
            # interior to the accepted rally). Absent is popped below. ``off`` is the rollback.
            ("held_release_v2_20260927", "off"),
            # Serve contacts proposed from the ball toss and departure. Absent is
            # popped below, so an existing receipt stays off. Explicit ``off`` is the rollback.
            ("serve_toss_proposal", "off"),
            # Picture-agreed final-bounce anchor (seed restart + 1-frame timing prior).
            # Absent is popped below. Explicit ``off`` is the rollback.
            ("final_bounce_anchor", "off"),
            # Picture-exit / last-sample final flights are partial (like a camera cut).
            # Absent is popped below. Explicit ``off`` is the rollback.
            ("exit_partial_endings", "off"),
            # Net-stop close three frames past the net event. Absent is popped below.
            ("net_stop_tail", "off"),
        )
    }
    for name, choices in (
        ("contact_prefix_scope", ("off", "coverage", "terminal_identity", "unresolved_ending")),
        ("contact_components", ("off", "unresolved_ending")),
        ("whole_point_seed_fallback", ("off", "on")),
        ("failed_component_split_fallback", ("off", "on")),
        ("component_search_budget", ("off", "exhausted_retry", "exhausted_split")),
        ("leading_event_components", ("off", "declared_prefix")),
        ("contact_component_routing", ("off", "unsupported_original_span")),
        ("terminal_track_endpoint", ("off", "ball_track", "ball_track_last_sample")),
        ("supported_flight_endings", ("off", "on")),
        ("ending_ownership", ("off", "on")),
        ("out_bounce_ending", ("off", "on")),
        ("boundary_endings", ("off", "on")),
        ("ambiguous_interior_ground", ("off", "supplied")),
        ("abstained_interior_ground", ("off", "ambiguous")),
        ("serve_side_association", ("off", "serve_reach")),
        ("contact_reversal_timing", ("off", "track_reversal")),
        ("net_cord_mid_flight", ("off", "admissible_set")),
        ("serve_tape_clearance", ("off", "hard")),
        ("server_pose_association", ("off", "nearest_detected")),
        ("uncertain_original_occurrence", ("off", "retained_inventory")),
        ("automatic_abstained_occurrence", ("off", "declared_ending", "retained_inventory")),
        ("observation_scope", ("on", "off")),
        ("optional_contacts", ("on", "off")),
        ("optional_final_contacts", ("on", "off")),
        ("optional_interior_contacts", ("on", "off")),
        ("independent_event_proposals", ("on", "off")),
        ("optional_contact_timing", ("off", "pmf_peaks")),
        ("optional_contact_composition", ("off", "bounded_pairs")),
        ("interior_contact_epochs", ("off", "source_interval")),
        ("optional_bounces", ("off", "classifier", "classifier_v2", "classifier_interior")),
        ("optional_conditional_grounds", ("off", "on")),
        ("terminal_context_ownership", ("on", "off")),
        ("preparation", ("on", "off")),
        ("terminal_net_tail", ("on", "off")),
        ("observed_horizon_tail", ("on", "off")),
        ("terminal_net_membership", ("off", "predicted")),
        ("refinement", ("on", "off")),
        ("following_bounce_intervals", ("on", "off")),
        ("joint_toss_requalification", ("on", "off")),
        ("earlier_toss_support", ("on", "off")),
        ("net_physical_eligibility", ("on", "off")),
        ("net_ground_normal", ("on", "off")),
        ("net_mesh_height", ("on", "off")),
        ("net_first_ground_candidates", ("on", "off")),
        ("net_first_contact_toss", ("on", "off")),
        ("net_ground_horizontal", ("off", "retention", "heading")),
        ("prefix_local_boundary", ("on", "off")),
        ("prefix_source_incumbent", ("on", "off")),
        ("prefix_pixel_loss", ("off", "soft_l1")),
        ("prefix_following_ground_timing", ("off", "prediction_hinge")),
        ("prefix_net_revisit", ("on", "off")),
        ("terminal_impact_intervals", ("on", "off")),
        ("terminal_ground_normal", ("on", "off")),
        ("terminal_net_coupling", ("on", "off")),
        ("terminal_ground_coupling", ("on", "off")),
        ("interior_ground_normal", ("on", "off", "sweep", "sweep_revisit")),
        ("terminal_net_seed", ("on", "off")),
        ("observation_net_seed", ("on", "off", "candidate")),
        ("ground_settling", ("on", "off")),
        ("passive_bounce_response", ("off", "coupled_slip")),
        ("terminal_ground_sparse_wings", ("on", "off")),
        ("interior_sparse_wings", ("on", "off")),
        ("interior_short_blocks", ("on", "off")),
        ("interior_ground_horizontal", ("off", "retention")),
        ("interior_restoration_geometry_only", ("on", "off")),
        ("first_contact_role", ("serve", "unspecified", "rally", "unknown")),
        ("observation_partition", ("fifth_frame_withheld", "all_native")),
        ("search_incumbent", ("on", "off")),
        ("event_ground_seed", ("on", "off")),
        ("bounce_interval_timing", ("on", "off")),
        ("automatic_ball_witness_sigma", ("on", "off")),
        ("owner_gate_loosening_20260924", ("off", "near_miss_20260924")),
        ("fit_ground_witness", ("off", "interval_ballistic_center")),
        ("toss_player_prior", ("on", "off")),
        ("independent_toss_horizontal_prior", ("on", "off")),
        ("serve_ground_normal", ("on", "off")),
        ("player_camera_coordinates", ("on", "off")),
        ("athlete_evidence", ("required", "optional")),
        ("athlete_root_reach_loss", ("quadratic", "cauchy")),
        ("missing_player_position", ("off", "abstain")),
        ("near_baseline_refinement", ("off", "on")),
        ("court_paint_refinement", ("off", "on", "abstain")),
        ("near_baseline_refinement_once", ("off", "on")),
        ("shot_homography_propagation", ("off", "on")),
        ("player_body_scale", ("off", "camera")),
        ("net_cord_response", ("tape_clip", "admissible_set", "evidence_bound")),
        ("live_shot_camera_eligibility", ("off", "on")),
        ("serve_outgoing_release", ("off", "on")),
        ("preparation_prefix_isolation", ("off", "on")),
        ("held_release_20260926", ("off", "on")),
        ("held_release_v2_20260927", ("off", "on")),
        ("serve_toss_proposal", ("off", "on")),
        ("final_bounce_anchor", ("off", "on")),
        ("exit_partial_endings", ("off", "on")),
        ("net_stop_tail", ("off", "on")),
    ):
        if result[name] not in choices:
            raise ValueError(f"unsupported global {name}")
    seconds = result["interior_block_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            "unsupported global interior_block_seconds; positive finite seconds required"
        )
    result["interior_block_seconds"] = float(seconds)
    speed_scale = result["net_seed_speed_scale_mps"]
    if type(speed_scale) not in (int, float) or not math.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError("positive finite net seed speed scale required")
    result["net_seed_speed_scale_mps"] = float(speed_scale)
    if "contact_components" not in policy:
        result.pop("contact_components")
    # Absent or "off": whole-point numerical seed death still kills the attempt.
    # "on" degrades to the component partition. Production default-on is declared
    # in PIPELINE_COMPONENT_POLICY, not here; shared_settings stays off-unless-
    # named so every existing policy receipt still reproduces. Rollback: delete
    # the whole_point_seed_fallback line from PIPELINE_COMPONENT_POLICY.
    if "whole_point_seed_fallback" not in policy:
        result.pop("whole_point_seed_fallback")
    if "failed_component_split_fallback" not in policy:
        result.pop("failed_component_split_fallback")
    if "component_search_budget" not in policy:
        result.pop("component_search_budget")
    # Absent keys stay absent so every existing policy receipt is unchanged.
    if "leading_event_components" not in policy:
        result.pop("leading_event_components")
    if "contact_component_routing" not in policy:
        result.pop("contact_component_routing")
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # Production default-on, if the measurement earns it, belongs in
    # PIPELINE_COMPONENT_POLICY. Explicit ``off`` is the rollback.
    if "terminal_track_endpoint" not in policy:
        result.pop("terminal_track_endpoint")
    # Absent stays off and is omitted. ``on`` is the measured arm: a net stop is a
    # terminal, and a flight is trimmed at the first camera cut, held camera,
    # point end or dead-ball boundary. Explicit ``off`` is the rollback.
    if "supported_flight_endings" not in policy:
        result.pop("supported_flight_endings")
    # Absent stays off and is omitted. ``on`` keeps aftermath rows as dead-ball
    # boundaries, drops an automatic bounce more than 1.5 s after the previous
    # live bounce, and stops a last-flight tail at the first picture exit or
    # track gap. Explicit ``off`` is the rollback. Production default, if the
    # measurement earns it, belongs in PIPELINE_COMPONENT_POLICY.
    if "ending_ownership" not in policy:
        result.pop("ending_ownership")
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # ``on`` refuses a flight whose origin follows a bounce read outside the
    # singles court by more than 0.75 m (wrong_output_d). Not a production default.
    if "out_bounce_ending" not in policy:
        result.pop("out_bounce_ending")
    # Absent stays off and is omitted. ``on`` closes a held flight before its
    # first labelled boundary row and keeps a short tail after a dead-ball bounce.
    if "boundary_endings" not in policy:
        result.pop("boundary_endings")
    if "final_bounce_anchor" not in policy:
        result.pop("final_bounce_anchor")
    if "exit_partial_endings" not in policy:
        result.pop("exit_partial_endings")
    if "net_stop_tail" not in policy:
        result.pop("net_stop_tail")
    if "ambiguous_interior_ground" not in policy:
        result.pop("ambiguous_interior_ground")
    # Absent stays off and is omitted. ``ambiguous`` reads a labeller abstention
    # lying between two contacts with no bounce as an ambiguous bounce at
    # preparation (prepare_validation_s6.abstained_interior_grounds).
    if "abstained_interior_ground" not in policy:
        result.pop("abstained_interior_ground")
    # Absent stays off and is omitted. ``serve_reach`` picks the serving box by
    # its serving reach, not the nearest centre (connected_shooting/serve_side.py).
    if "serve_side_association" not in policy:
        result.pop("serve_side_association")
    # Absent stays off and is omitted. ``track_reversal`` moves labelled contacts to
    # the ball-track turn at preparation (s6_contact_reversal.retime).
    if "contact_reversal_timing" not in policy:
        result.pop("contact_reversal_timing")
    # Absent stays off and is omitted. ``admissible_set`` switches net_cord_response
    # to the admissible set only for a fit whose events hold a mid-flight net hit
    # (net_cord_response.mid_flight_net_hits). ``hard`` holds the serve flight's
    # tape clearance at HARD_SERVE_CLEARANCE_SCALE_M during a serve-origin fit.
    for name in ("net_cord_mid_flight", "serve_tape_clearance"):
        if name not in policy:
            result.pop(name)
    # Absent stays off and is omitted. ``nearest_detected`` is the measured arm:
    # a hole at the contact picture uses the nearest detected pose instead of
    # refusing the component. Explicit ``off`` is the current lookup.
    if "server_pose_association" not in policy:
        result.pop("server_pose_association")
    if "uncertain_original_occurrence" not in policy:
        result.pop("uncertain_original_occurrence")
    if "automatic_abstained_occurrence" not in policy:
        result.pop("automatic_abstained_occurrence")
    if "automatic_event_operating_point" not in policy:
        result.pop("automatic_event_operating_point")
    else:
        # Validated here so an unusable operating point is refused by the policy reader
        # rather than by the first attempt that happens to carry a band row.
        from cv.pipeline.s6_event_operating_point import event_operating_point

        event_operating_point(result["automatic_event_operating_point"])
    # Owner-approved 2026-09-19. Absent keys are popped, exactly like the jump-rejection
    # key above, so every policy that predates the sign-off reproduces byte for byte; the
    # production entrypoint declares them through PIPELINE_COMPONENT_POLICY instead.
    for owner_approved in (
        "bounce_interval_timing",
        "automatic_ball_witness_sigma",
        "owner_gate_loosening_20260924",
    ):
        if owner_approved not in policy:
            result.pop(owner_approved)
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # The production on-switch is PIPELINE_COMPONENT_POLICY. Explicit ``off`` is the rollback.
    if "near_baseline_refinement" not in policy:
        result.pop("near_baseline_refinement")
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # ``on`` refits a supported camera whose court lines miss the paint (a whole band
    # off on Monte Carlo 2025 and on some Rome 2026 shots). Explicit ``off`` is the rollback.
    if "court_paint_refinement" not in policy:
        result.pop("court_paint_refinement")
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # ``on`` skips the near-baseline refit here when the camera document is already
    # marked refined (the product observation stage refines it first). A second pass
    # moves P on an off-paint camera and the component source check then refuses every
    # piece of the point. Explicit ``off`` is the rollback.
    if "near_baseline_refinement_once" not in policy:
        result.pop("near_baseline_refinement_once")
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # Production on-switch is PIPELINE_COMPONENT_POLICY. ``on`` fills side
    # association and fallback-camera admission from the nearest registered
    # homography of the same camera shot. A close-up is not painted. Explicit
    # ``off`` is the rollback.
    if "shot_homography_propagation" not in policy:
        result.pop("shot_homography_propagation")
    # Absent stays off and is omitted. ``camera`` lets the automatic camera's standing
    # height vouch for a player the ground-scale body-size gate refuses. It acts only
    # inside the shot-propagated side association, so with shot_homography_propagation
    # off (its rollback) it is inert.
    if "player_body_scale" not in policy:
        result.pop("player_body_scale")
    if "automatic_ball_jump_rejection" not in policy:
        result.pop("automatic_ball_jump_rejection")
    else:
        from cv.pipeline.s6_ball_jump_rejection import validate_mode

        validate_mode(result["automatic_ball_jump_rejection"])
    # Absent stays off and is omitted, so an existing policy receipt is unchanged.
    # ``on`` moves automatic centres back along travel to the streak middle.
    if "automatic_ball_streak_centre" not in policy:
        result.pop("automatic_ball_streak_centre")
    else:
        from cv.pipeline.s6_ball_streak_centre import validate_mode as validate_streak

        validate_streak(result["automatic_ball_streak_centre"])
    # Owner-directed 2026-09-20. Absent is the VR tape clip, byte for byte; the
    # named value "tape_clip" is the same law reachable by name. Not in
    # PIPELINE_COMPONENT_POLICY: extra accepts decide whether this ships.
    if "net_cord_response" not in policy:
        result.pop("net_cord_response")
    else:
        from cv.pipeline.net_cord_response import validate_mode as validate_net_cord

        validate_net_cord(result["net_cord_response"])
    if "net_cord_tape_band_m" not in policy:
        result.pop("net_cord_tape_band_m")
    else:
        from cv.pipeline.net_cord_response import validate_h_tol

        result["net_cord_tape_band_m"] = validate_h_tol(result["net_cord_tape_band_m"])
    for gate_release in (
        "live_shot_camera_eligibility",
        "serve_outgoing_release",
        "preparation_prefix_isolation",
        "held_release_20260926",
        "held_release_v2_20260927",
        "serve_toss_proposal",
    ):
        if gate_release not in policy:
            result.pop(gate_release)
    if "depth_conditioned_seed" in policy:
        if policy["depth_conditioned_seed"] not in ("off", "on"):
            raise ValueError("unsupported global depth_conditioned_seed")
        result["depth_conditioned_seed"] = policy["depth_conditioned_seed"]
    return result


#: Measured defaults for the PRODUCTION entrypoints only; `shared_settings` above keeps its
#: strict defaults and its absent-key pop, so every existing policy receipt still reproduces
#: byte for byte.  This is the `illumination_split` shape: the low-level default stays off and
#: the pipeline entrypoint declares the permissive policy explicitly.
#:
#: Source: the 12-arm ablation of the four composed S6 mechanisms, 2026-09-18, 72 cold cells /
#: 926 fits, control reproduced twice at LL 90 / AL 77 / LA 7 / AA 8 of 118 with a +-1 noise
#: floor, at
#: processed/s6_local_20260911/systemic_flight_audit_v1/current_transfer_recheck_v1/
#:   matched_fresh36_v1/fresh_evaluation_v1/overnight_20260918/mechanism_ablation_v1
#: (ABLATION_TABLE.json).  Best measured configuration is the composed candidate MINUS
#: `fit_ground_witness`: LL 99 / AL 82 / LA 16 / AA 17.  `leading_event_components` moves
#: +-7 LA / +-8 AA and is the only mechanism that moves the product metric;
#: `uncertain_original_occurrence` moves +-9 LL / +-6-7 AL.  `contact_component_routing` sat
#: at the noise floor on that panel (+2 LA / +1 AA alone) and was left off EXPLICITLY PENDING
#: A SECOND PANEL; that deferral was resolved on 2026-09-19 and the key is adopted below, so
#: a reader sees a deferral closed rather than a rejection overridden.  The one absent
#: mechanism is deliberate: `fit_ground_witness` is a measured NET NEGATIVE (removing it gains
#: 3 LL, adding it alone loses 3 LL, reproduced three ways).  Revert by emptying this mapping.
#:
#: `automatic_abstained_occurrence=retained_inventory` was added 2026-09-18 on the strength
#: of the `abstained_occurrence_v1` panel (…/overnight_20260918/abstained_occurrence_v1/
#: RESULT.json): three arms x six sources, one wave, the anchor re-run inside it and
#: reproducing LL 99 / LA 16 / AA 17, and the key worth **+18 LA / +14 AA of 118 with
#: `extra_accepted_competitive_or_unbound` and `extra_accepted_post_ending` 0 in every arm
#: of every cell** -- the largest single-mechanism move on the product metric measured on
#: this panel, against +8 AA for `leading_event_components`.  `declared_ending`, the strict
#: subset that touches only a resolved automatic ending, is worth +6 LA / +4 AA and is
#: contained exactly by `retained_inventory`, so the wider mode is the one declared.
#:
#: WHAT CERTIFIES THIS FOR THE PRODUCTION ROUTE, since the panel is a supplied-input
#: measurement and production is not that route: a packet-equality gate, not a second fit
#: panel.  `s6_automatic_observations.build_observations` and
#: `cv.validation.prepare_s6_input_swaps` now reach ONE decision through
#: `s6_contact_prefix_scope.qualify_or_retain` / `unresolved_boundaries`, and over 58
#: production automatic attempts the two build an identical event inventory, identical
#: boundary intervals, an identical retain/qualify decision and an identical recorded
#: refusal on 54 of 54 that reach the seam, in all three arms
#: (abstained_occurrence_v1/runtime_equality_v1/).  The panel measured its result by
#: fitting packets; if production builds the same packet the fit consumes the same input,
#: so the downstream result is not a separate empirical question and a production fit panel
#: would differ only by the 900 s wall-clock budget's own noise.
#:
#: THE RESIDUAL, so a reader can see which guarantee they are relying on: that equality was
#: demonstrated on 58 *production* attempts, NOT on the panel's own six sources, because
#: `validate_ancestry` correctly refuses the panel's reviewed-input upstream as automatic
#: ancestry.  So the inference is "the routes agree wherever they were compared, and they
#: now share the deciding code" -- shared code being the stronger of the two -- and not
#: "the routes were compared on the panel's own attempts".  The standing structural gap
#: behind that is named in docs/PIPELINE_IMPLEMENTATION.md.
#: `automatic_event_operating_point` was added 2026-09-19 and IS A REPAIR, NOT A TUNING
#: CHOICE -- but the defect is subtler than a hard-coded number, so it is stated exactly.
#: The upstream this pipeline consumes runs its own retrained checkpoint
#: (`upstream_event_feature_diagnosis_v1/translation_robustness_v1/training_v1/off`,
#: sha fc265709…, NOT main's calibrated 72875763…), and that checkpoint IS calibrated:
#: `…/translation_robustness_v1/calibration_v3/off/COMPLETED.json` is `status: "calibrated"`
#: and selects 0.9907168846560898.  The emission manifest's
#: `calibration_scope: "reference_only_not_calibrated_for_this_variant"` is
#: `runtime_calibration_identity` correctly refusing to attach main's NAMED operating point
#: to a different checkpoint's bytes -- that guard working, not an uncalibrated threshold.
#:
#: THE ACTUAL DEFECT IS THE OBJECTIVE.  That calibration's
#: `PLAN.json:threshold_rule.method` is "existing calibrate_abstention
#: **bounded_false_positives** maximum recall" with `false_positive_rate_ceiling: 0.01`.
#: `bounded_false_positives` is one of this module's three named operating points
#: (0.9898053678698258); the runtime default is `complete_points_one_fp_per_100`
#: (0.9521754400734156), and `complete_points_one_fp_per_100` EXISTS precisely because
#: maximising per-event precision is not the same as maximising complete event points --
#: which is the quantity a 3D fit needs.  The plan even carries
#: `historical_threshold_sensitivity: 0.9521754400734156`, so the complete-points value was
#: in hand and the stricter rule was selected anyway, over 8 opened broadcasts / 29 clips /
#: 239 truth events rather than main's 32 broadcasts / 1176 truth events.  So the shipped
#: state is a per-event-precision optimum being consumed by a complete-point consumer.
#:
#: That the adopted value is right FOR THIS CHECKPOINT is measured on that checkpoint's own
#: calibration set, not borrowed from main: at tolerance 3 it scores P 0.992 / R 0.536 at
#: 0.9907169 and P 0.967 / R 0.611 at 0.9521754 -- +7.5 recall points for -2.5 precision,
#: the same shape the sealed `vlm_event_benchmark_v1` measured independently on the OTHER
#: checkpoint (+10.7 / -5.1).
#:
#: THE SOURCE CANNOT BE FIXED FROM THIS LANE and is named here so the owning one can:
#: re-deriving the threshold means re-running `translation_robustness_v1/calibration_v3`
#: under the `complete_points_one_fp_per_100` rule and rebuilding every frozen upstream that
#: binds it -- and a partial rebuild aborts the whole panel (SHARED_UPSTREAM parity).  Until
#: that happens this key is what keeps the consumer at the consumer's own operating point,
#: and it will re-diverge the next time an upstream is rebuilt under the old rule.
#:
#: Source: `event_operating_point_v1`, 2026-09-19 (…/overnight_20260919/
#: event_operating_point_v1/RESULT.json): four arms x six sources, 24 cold cells, 374 fits,
#: one wave, the anchor re-run inside it and reproducing LL 99 / LA 34 / AA 31.  Walking the
#: floor down is monotone on both automatic arms -- AA 31 / 42 / 48 / 59 and LA 34 / 46 / 55
#: / 59 of 118 at floors 0.9907169 / 0.9521754 / 0.80 / 0.20 -- against a +-1 noise floor,
#: and LL and AL are unchanged per source in every arm, which is the correct signature: the
#: key cannot reach an arm whose events are labels.  **0.9521754400734156 is adopted because
#: it is the interior optimum, not the endpoint: +11 AA / +12 LA over the previous default
#: with `extra_accepted_competitive_or_unbound` and `extra_accepted_post_ending` STILL 0 in
#: every arm of every cell and no attempt losing a match.**  0.20 reaches AA 59 but is the
#: first configuration this instrument has ever scored with a false accept (1 competitive,
#: 2 post-ending, across two attempts); that trade was put to the owner and deferred, not
#: taken.  It is not an admission and carries no contract of its own, but it is declared
#: here rather than in `shared_settings` because the only configuration in which it was
#: MEASURED is the one that declares both component contracts, which this mapping gates on.
#: Revert by emptying this mapping.
#:
#: LIMITATION, carried from the panel and from the sealed benchmark's own winning arm: the
#: chain point grammar is NOT re-run over the enlarged accepted set.  Both routes
#: re-threshold the decision the producer already wrote rather than re-decoding, so no
#: `point_end` row is ever restated and the resolved/unresolved ending shape is unchanged.
#: `contact_component_routing=unsupported_original_span` was added 2026-09-19, and it closes
#: the ablation's own deferral rather than overriding its verdict: that panel recorded the key
#: at the noise floor and said it "stays off pending a second panel".  This is that panel --
#: processed/s6_local_20260911/systemic_flight_audit_v1/current_transfer_recheck_v1/
#:   matched_fresh12_v1/transfer_evaluation_20260919 (RESULT.json) -- five broadcasts
#: DISJOINT by match_key from the six the other three keys were tuned on, 72 confirmed
#: origins, three declared policies x five sources, 153 cold fits, one wave, the control
#: re-run inside it.  A full-tree grep finds zero prior occurrences of any of these five keys
#: anywhere under that panel, so it had never seen one of them.
#:
#: On it the key is worth **+12 LA and +3 AA of 72**, and what it actually buys is not a
#: margin, it is a per-broadcast collapse that the OTHER THREE ADOPTIONS CAUSE.  Declaring
#: three of the four leaves five `unresolved contact prefix` preparation holds that neither
#: the pre-adoption control nor the four-key arm produces (source04_long_a1 LA/AA,
#: source04_long_a2 LA, source05_long_a1 LA/AA, source08_long_a1 LA/AA), and loses
#: source04_long_a2 LA outright -- **9 of 15 matched flights to 0**, on an attempt the
#: control prepares and the four-key arm scores 10 of 15.  The mechanism is exact: the three
#: admissions widen the retained inventory, the widening creates contact-prefix demands, and
#: routing is the key that owns the spans around a contradictory original span.  Adopting
#: three of four leaves a demand with nothing to satisfy it.
#:
#: READ THE POOLED NUMBER CAREFULLY, because it is the reason this was nearly missed: on that
#: panel the three-key arm's pooled LA delta is EXACTLY ZERO, and that zero is `+8` on one
#: broadcast and `-9` on another.  **A pooled delta of zero is indistinguishable from no
#: effect and from two large effects cancelling.**  Only the per-broadcast table separates
#: them, which is why docs/operations/S6_PANEL_ROTATION.md requires one.
#:
#: The key is already in `s6_broadcast_backend.COMPONENT_TRIGGERS` and `COMPONENT_FORWARDED`,
#: so unlike `uncertain_original_occurrence` it is not inert on the automatic route the day it
#: is adopted.  Revert by emptying this mapping.
#: The last two entries are the two acceptance-gate changes the owner approved in person on
#: 2026-09-19, after a plain-language explanation of each (memory
#: `owner-approved-gate-changes-20260919.md`).  They change correctness definitions, which
#: AGENTS.md says must not change silently; that sign-off is the authority, and it covers
#: only these two.  Both were found by the loss-scoping lane
#: (`processed/loss_scoping_20260919/REPORT.md`).
#:
#: `bounce_interval_timing` -- "We should accept bounces in range, yes."  A modelled bounce
#: passes the timing check when it lands inside the label's OWN observed frame range, not
#: only within `bounce_uncertainty_frames` of that range's midpoint.  Strictly additive:
#: the midpoint test survives as an OR, so nothing that passes today can fail.
#:
#: `automatic_ball_witness_sigma` -- "increasing ball labeling uncertainty also makes sense
#: - agreed."  Automatic ball rows stop receiving the 2.0 px human-click sigma floor in the
#: two-wing bounce witness.  They carry no measured radius of their own, so the floor was
#: letting a denser automatic wing out-certify the labelled arm on the same flight and
#: close the 0.75 m bounce-ray gate.  The replacement is measured, not assumed: see
#: `agent_whole_point_search.AUTOMATIC_BALL_SIGMA_FLOOR_PX`.  It moves no fitted position,
#: only the covariance, and is inert on a labelled ball arm.
#:
#: `automatic_ball_jump_rejection` is adopted on the coordinator's decision of 2026-09-19,
#: on the whole fresh36 panel, same-wave cold OFF/ON, 196 fits, 118 reference flights per arm:
#: AL 84 -> 87 and AA 59 -> 61 with ZERO new extra accepts in any arm.  The two labelled-ball
#: arms read no automatic row and both moved by exactly 0, so this wave's own noise measured 0
#: rather than the assumed +-1, which is why the gains are not read as noise.
#:
#: STATE THE TRADEOFFS, because they are real:
#:  * the evidence is ONE development panel, and the panel cohorts are development data;
#:  * the gains are CONCENTRATED IN ONE BROADCAST -- five of the six are source01, one source05;
#:  * fresh12 does not move at all, because it has no wrong-object bursts of this kind (the
#:    filter catches 1 of 43 label-flagged rows in point there, against 56 of 137 on fresh36);
#:  * one AL flight was lost, `source06_long_p005_a2:a2_c7`, and it is NOT attributable to the
#:    filter -- that attempt has 1203 visible rows in both arms, the filter removed none of
#:    them, so it is search nondeterminism under the 900 s wall-clock budget;
#:  * in-point false rejections are 0 of 10,631 good rows across both panels, but whole clip
#:    the filter does refuse 10 of 9,748 good fresh36 rows (0.10%) outside point spans.
#: The `gap_steps` mode stays declared and measured, NOT default: it changed two attempts and
#: still cannot see the 16-frame-gap case at 59.94 fps.
#:
#: `whole_point_seed_fallback` is adopted 2026-09-21 on the owner's "make them default"
#: decision after the stacked-fresh holdout: sealed panel 0 AL 23 -> 41 and AA 9 -> 10 of
#: 102, attempt crashes 5 -> 1, zero extra accepts, inert on the development panels. The
#: low-level default in `shared_settings` stays off (absent-key pop); this mapping is the
#: one-line production on-switch. Pair with `SelectionConfig.keep_unique_admissible_frames`
#: (upstream of S6). Rollback: delete this line.
#:
#: Revert any one by deleting its line here; revert everything by emptying this mapping.
PIPELINE_COMPONENT_POLICY = {
    "leading_event_components": "declared_prefix",
    "uncertain_original_occurrence": "retained_inventory",
    "automatic_abstained_occurrence": "retained_inventory",
    "automatic_event_operating_point": 0.9521754400734156,
    "contact_component_routing": "unsupported_original_span",
    "automatic_ball_jump_rejection": "on",
    "bounce_interval_timing": "on",
    "automatic_ball_witness_sigma": "on",
    "whole_point_seed_fallback": "on",
    # Owner 2026-09-24: last flight may end at the last visible ball sample when no other ball-track
    # terminal is seen (it fired and moved no accepted flight on the fresh holdout). Rollback: "ball_track".
    "terminal_track_endpoint": "ball_track_last_sample",
    # Owner 2026-09-24 "no-regrets": a missing server pose uses the nearest detected pose inside the wide
    # pre-contact window (fresh panel A AL 49 -> 55). Rollback: delete this line.
    "server_pose_association": "nearest_detected",
    # Joined multi-flight components that die or reject are refit as one-flight children (fresh AL +9, AA +1,
    # development +4; none lost, no extra accepts; FINAL_REPORT_split.md). Rollback: delete this line.
    "failed_component_split_fallback": "on",
    # Owner 2026-09-24: "I'm okay with this loosening." Rollback: delete this line.
    # The numbers and the measurement are in s6_owner_gate_loosening and
    # cv/experiments/ceiling_census/FINAL_REPORT_gates.md.
    "owner_gate_loosening_20260924": "near_miss_20260924",
    # Rome near baseline, 2026-09-25. Intrinsic-fallback frames were drawing the near
    # baseline up to 172 px below the paint, so accepted near contacts sat about 2 m
    # off the court and one (A2_C03) was reported inside the baseline while the player
    # stands behind it. Refitting from the painted line keeps the same flights and
    # drops that error to about 0.1 m. A frame whose near line cannot be put back on
    # the paint is held instead of accepted. Rollback: delete this line.
    # Measurement: cv/experiments/connected_shooting/FINAL_REPORT_fix_rome_camera.md.
    "near_baseline_refinement": "on",
    # Owner GO 2026-09-25. Release a confident event the clip gate, the tracking-arc
    # serve check, or a bad first-event prefix had vetoed. Fresh AA 40 -> 65 of 174,
    # no fresh extra accept. Development AA 72 -> 78 of 190. The one development extra
    # is an origin offset on source08_long_a1 (contact frame 955.15, the labelled
    # return e028 about 64 ms later), not a wrong object. Rollback: delete these three
    # lines. Numbers: cv/experiments/fix_gates/FINAL_REPORT_gates.md.
    "live_shot_camera_eligibility": "on",
    "serve_outgoing_release": "on",
    "preparation_prefix_isolation": "on",
    # Same-wave 2026-09-25: a net event with no later competitive landing closes
    # the flight, and a camera cut, held camera, point end or dead-ball phase
    # trims it (FINAL_REPORT_fix_endings.md). Fresh AL 118 -> 128, AA 40 -> 43
    # of 174, nothing lost, no extra accepts. Rollback: delete this line.
    "supported_flight_endings": "on",
    # Owner GO 2026-09-26. An aftermath row, or an automatic bounce more than
    # 1.5 s after the last live bounce, ends the flight. A last-flight tail
    # stops at the picture exit or a track gap. A net stop that is earlier
    # stays the ending. Fresh AL 127 -> 132 and AA 68 -> 70 of 174, cascade
    # 106 -> 110, panel C AL 51 -> 53 of 77, development LL 166 -> 167 and
    # AL 152 -> 153. Nothing lost. No new extra accept.
    # Rollback: delete this line. FINAL_REPORT_endings_v2.md.
    "ending_ownership": "on",
    # Rome player sides, 2026-09-26. A static wide shot registers one frame and
    # stores that homography as unreliable anchor_static_fallback, so side
    # association kept no player rows. ON reuses that homography on the other
    # frames of the same camera shot, for sides and for a fallback point camera.
    # A different shot, including a close-up, is not painted. Panel C Rome
    # labelled 0 -> 8 of 16, nothing lost, no wrong-object accept. Rollback:
    # delete this line. Numbers: FINAL_REPORT_rome_players.md.
    "shot_homography_propagation": "on",
    # Owner GO 2026-09-26. A model-accepted event at or above 0.99 is released
    # when every hold is a tracking-arc abstention, a camera failure on a frame
    # the local camera check supports, or a repeated-frame hold after the native
    # cadence audit shows short groups and the frame is not a repeated picture.
    # An attempt with no accepted contact keeps an abstained contact instead of
    # refusing. Same-wave AA on the attempts whose inventory changes: 40 flights
    # gained, none lost, no extra accept. Fresh AA 68 -> 85 of 174. Panel C AA
    # 28 -> 40 of 77. Rollback: delete this line.
    # Numbers: cv/experiments/connected_shooting/FINAL_REPORT_held_release.md.
    "held_release_20260926": "on",
    # Panel D wrong output, 2026-09-27 (FINAL_REPORT_wrong_output_d.md). After a supported bounce read more
    # than 0.75 m outside the singles court on a reliable camera, later contacts open no flight. Removes the
    # panel D dead-ball accept (source08_point007_long_a1), loses no real flight on fresh 0/A, panel C, panel D
    # or development. The margin was set after looking at panel D. Rollback: delete this line.
    "out_bounce_ending": "on",
    # Serve toss proposal, 2026-09-27. When the decoder has no serve row, a toss
    # over the server box followed by a fast outgoing flight adds one serve
    # contact at the departure, still through the gates. Same-wave AA on the 15
    # attempts whose inventory changes: 6 flights gained (panel D 2, development
    # 4), none lost, no extra accept; fresh 0/A inert. Rollback: delete this line.
    # Numbers: cv/experiments/serve_contacts/FINAL_REPORT_serve_contacts.md.
    "serve_toss_proposal": "on",
    # Ceiling census D, 2026-09-27. When a joined component exhausts its search,
    # the split fallback's one-flight children get a search share instead of zero
    # (the parent is not retried). Same-wave on the 21 receipts it can act on:
    # fits 82 -> 89 of 107, cascade 55 -> 67 of 94, nothing lost. One new extra,
    # an origin offset on panel D source08_point007_long_a1 cascade (the right
    # ball, automatic contact 3.5 frames before the labelled C03). Job wall +5%.
    # Rollback: delete this line. FINAL_REPORT_ceiling_census_d.md.
    "component_search_budget": "exhausted_split",
    # Owner GO 2026-09-27 ("1 wrong flight out of 105 is totally fine"). A held flight
    # with an unknown next contact closes before its first labelled boundary row, and a
    # dead-ball close keeps its short post-bounce tail; the close must be a struck shot
    # and must reach an adjacent abstained landing or net row. Same-wave fresh AL
    # 134 -> 145 and AA 88 -> 106 of 174, cascade 111 -> 112; panel C AL 67 -> 71, AA
    # 40 -> 47; development AL 155 -> 161, AA 95 -> 111. Nothing lost. One disclosed
    # wrong AA accept: fresh source08_case002_short_a1 E008 (depth about 2 m short).
    # Rollback: delete this line. Numbers: FINAL_REPORT_ceiling_endings.md.
    "boundary_endings": "on",
    # Ceiling census D, 2026-09-27. An interior flight held only by one bounce whose status is
    # `ambiguous` (and its duplicate source boundary), with both contacts supported and nothing
    # else crossing, is fitted with that bounce as supplied, in its own component. Same-wave on
    # the 46 attempts it can act on, on main with boundary_endings: fits 174 -> 209 of 322,
    # cascade 27 -> 33 of 43; AL +7 (fresh A 2, panel C 3, panel D 2), development LL +1, AA
    # +27. Nothing lost, no new extra; every gained flight viewed. Rollback: delete this line.
    # Numbers: cv/experiments/ceiling_census_d/FINAL_REPORT_ceiling_census_d.md.
    "ambiguous_interior_ground": "supplied",
    # Ceiling census D T5, 2026-09-27. Inside the shot-propagated side association a
    # player box keeps whichever of the ground scale and the automatic camera's standing
    # height puts its body-size ratio nearer 1, so the gate refuses only what both call
    # the wrong size (a steep Monte Carlo camera foreshortened the near player to 0.57
    # against the 0.62 floor). Of 144 attempts on fresh 0/A, panel C, panel D and
    # development only two change; same-wave fits 2 -> 12 of 20 (panel D
    # source03_r1_medium_001_a1 C01-C05 on AL and AA), nothing lost, no extra, viewed.
    # Inert when shot_homography_propagation is off. Rollback: delete this line.
    # Numbers: cv/experiments/ceiling_census_d/FINAL_REPORT_ceiling_census_d.md.
    "player_body_scale": "camera",
    # Held release v2, 2026-09-27 (FINAL_REPORT_held_release_v2.md). A contact or bounce with marginal in
    # [0.8, threshold), only v1-clearable holds, inside a live play span, a >=30 deg track turn within +-2 frames
    # and interior to the rally is released. Same-wave AA +21 / -3 (fit failures), no extra; cascade panel D +3.
    # Rollback: delete this line.
    "held_release_v2_20260927": "on",
    # Fit seeds (c), 2026-09-28. The first hitter's box is the one whose serving reach
    # holds the ball, not the nearest centre (a high toss picked the far box). Same-wave
    # on the 3 jobs it changes: fresh source09_case001_short_a2 a2_serve AL and panel A
    # source03_point_002_a1 C01 AA gained, nothing lost, no extra, both viewed.
    # Rollback: delete this line. Numbers: cv/experiments/fit_seeds/FINAL_REPORT_fit_seeds.md.
    "serve_side_association": "serve_reach",
    # Fit seeds (a), 2026-09-28. A labeller abstention strictly between two contacts with
    # no labelled bounce becomes an ambiguous bounce at preparation (label arms only), so
    # ambiguous_interior_ground fits it. Same-wave on the 2 jobs it changes: panel D
    # source03_r1_long_001_a3 A3_C12 AL gained and viewed, nothing lost, no extra; fresh
    # C5 stays rejected (the ball is off-picture across the whole abstention).
    # Rollback: delete this line.
    "abstained_interior_ground": "ambiguous",
    # Bounce-anchor lane, 2026-09-28 (FINAL_REPORT_bounce_anchor.md). A final flight that ends
    # by leaving the picture (fov_exit / last_visible_sample) is a partial flight, like a camera
    # cut: its last supplied bounce is no longer pinned to the exit frame and
    # physical_ground_ending is replaced by partial_flight_labelled. Same-wave on every
    # combined-v3 attempt with such an ending: AL +6 (fresh C6, a2_return, A1_contact_1; panel
    # C C05; panel D a1_c01, a1_c1), AA +8, nothing lost, no new extra, every gain viewed.
    # Rollback: delete this line.
    "exit_partial_endings": "on",
    # Ceiling v2, 2026-09-29 (cv/experiments/ceiling_v2/FINAL_REPORT_ceiling_v2.md). The product
    # observation stage already refines the near baseline; a second pass here moved P on off-paint
    # cameras and the component source check refused the piece (Rome: ~50 flights per match never
    # reached the fitter). Refine once, then refit cameras whose court misses the paint and hold the
    # frame runs still off it (wrong far-court depth). Labelled same-wave: panel D AL +1, AA +1
    # (source03_r1_short_001 C01, viewed), nothing else changed. Rome, 29 off-paint attempts: ~32
    # accepted vs production's 30, now mostly at the right depth; one disclosed new wrong-depth accept
    # (pt0061). Needs native frames: a remote worker without staged images raises instead of
    # silently fitting unrefined. Rollback: delete these two lines.
    "near_baseline_refinement_once": "on",
    "court_paint_refinement": "abstain",
}


def pipeline_policy(policy: dict) -> dict:
    """Apply `PIPELINE_COMPONENT_POLICY` to a supplied production policy.

    Every admission here refuses to run as a bare flag: each requires the unresolved-ending
    contact prefix contract to hold the original inventory and the contact-component policy
    to model the supported spans, and raises otherwise.  (`automatic_abstained_occurrence`
    inherits both by being a scope on the uncertain-occurrence admission rather than a
    policy of its own.)  A policy that does not declare both contracts is therefore returned
    unchanged rather than turned into a hard failure, exactly as the panels measured them --
    their own control arms declare both.  An explicit value in the supplied policy always
    wins, so an arm that names `off` still runs off.
    """
    from cv.pipeline import s6_contact_components as components
    from cv.pipeline import s6_contact_prefix_scope as prefix

    if (
        policy.get("contact_prefix_scope") != prefix.UNRESOLVED_ENDING
        or policy.get("contact_components") != components.MODE
    ):
        return dict(policy)
    return {**PIPELINE_COMPONENT_POLICY, **policy}


def serialize_context(context: dict) -> dict:
    """Export both final Scene copies, original evidence and selected event epochs."""
    from cv.experiments.connected_shooting import observation_partition as partition

    result = context | {name: asdict(context[name]) for name in ("scene", "heldout")}
    if partition.all_native(context["scene"]):
        result["observation_partition"] = partition.receipt(context["scene"], context["heldout"])
        result["heldout_scene_role"] = (
            "in_sample_check_copy; legacy internal key, no independent holdout"
        )
    return result


def implementation_records() -> list[dict]:
    """Bind loaded numerical dependencies, including the explicit optional adapters."""
    filenames = {
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if module is not None
        and getattr(module, "__file__", None)
        and name.startswith(("cv.experiments.connected_shooting.", "physics.", "cv.pipeline.s6_"))
        and Path(module.__file__).suffix == ".py"
    }
    filenames.add(Path(__file__).resolve())
    return [provenance.file_record(path) for path in sorted(filenames)]


def implementation_unchanged(records: list[dict]) -> bool:
    """Resolve provenance bases explicitly, independent of the caller's cwd."""
    return all(provenance.file_sha256(resolve(record)) == record["sha256"] for record in records)


def final_state_binding(result: dict) -> dict:
    """Bind final geometry and its matching evaluation without rerunning a fit."""
    from cv.experiments.connected_shooting.serve_timing_profile import jsonable

    values = {
        name: result[name]
        for name in ("parameters", "measurement", "verdict", "net_response", "evaluation_context")
    }
    if "ground_response" in result:
        values["ground_response"] = result["ground_response"]
    if "passive_bounce_response" in result:
        values["passive_bounce_response"] = result["passive_bounce_response"]
    if "net_collision_policy" in result:
        values["net_collision_policy"] = result["net_collision_policy"]
    if "ball_observation_operator" in result:
        values["ball_observation_operator"] = result["ball_observation_operator"]
    if "automatic_provenance" in result:
        values["automatic_provenance"] = result["automatic_provenance"]
    values["initial_parameters"] = result["initial_search"]["parameters"]
    from cv.pipeline import s6_first_contact_role

    if s6_first_contact_role.FIELD in result:
        values[s6_first_contact_role.FIELD] = result[s6_first_contact_role.FIELD]
    if "net_response" in result["initial_search"]:
        values["initial_net_response"] = result["initial_search"]["net_response"]
        values["initial_net_response_initialization"] = result["initial_search"][
            "net_response_initialization"
        ]
    return {
        "schema": "labeled_s6_final_state_binding_v1",
        "canonical_json": "sort_keys=True,separators=(',',':'),allow_nan=False; numpy converted to lists/scalars",
        "sha256": {
            name: hashlib.sha256(
                json.dumps(
                    jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            for name, value in values.items()
        },
        "selected_search": result["selected_by"]["selected_search"],
    }


def _frame_paths(labels: dict) -> dict[int, str]:
    return {
        int(image["frame"]): image["source"]["path"]
        for image in (labels.get("source_pack") or {}).get("images") or []
        if image.get("source", {}).get("path")
    }


def _near_baseline_cameras(
    cameras: dict, labels: dict, settings: dict
) -> tuple[dict, dict | None, bool]:
    """Near-baseline refit of the input cameras; returns (cameras, receipt, changed)."""
    already_refined = (
        settings.get("near_baseline_refinement_once") == "on"
        and cameras.get("near_baseline_refinement") == "on"
    )
    if settings.get("near_baseline_refinement") != "on":
        return cameras, None, False
    if already_refined:
        receipt = {
            "mode": "on",
            "changed_frames": 0,
            "unresolved_frames": 0,
            "skipped": "already_refined",
        }
        return cameras, receipt, False
    from cv.pipeline.court_near_baseline_refinement import refine_camera_document

    refined, receipt = refine_camera_document(cameras, mode="on", frame_paths=_frame_paths(labels))
    unread = [
        group for group in receipt.get("groups") or [] if group.get("reason") == "image_unavailable"
    ]
    if unread:
        raise FileNotFoundError(
            "near-baseline refinement could not read a native frame for "
            f"{sum(int(group['frames']) for group in unread)} fallback frames"
        )
    changed = bool(receipt["changed_frames"] or receipt["unresolved_frames"])
    return (refined if changed else cameras), receipt, changed


#: Component jobs of one source share its camera document and native frames, so the
#: source case keeps one content-addressed paint refit they all reuse.
PAINT_CACHE_DIR = "court_paint_cache"


def _paint_cache_dir(output: Path) -> Path | None:
    for parent in list(output.parents)[:6]:
        if (parent / "component_plan.json").is_file():
            return parent / PAINT_CACHE_DIR
    return None


def _paint_cameras(
    cameras: dict, labels: dict, mode: str, cache_dir: Path | None
) -> tuple[dict, dict]:
    """Court paint refit, reused from ``cache_dir`` when its exact inputs were refit before.

    The key covers the camera document, mode, native frame paths and the module source,
    so a hit returns exactly what a fresh refit would compute.
    """
    from cv.pipeline import court_paint_refinement

    frame_paths = _frame_paths(labels)
    path = None
    if cache_dir is not None:
        digest = hashlib.sha256(
            json.dumps(
                [
                    Path(court_paint_refinement.__file__).read_text(),
                    mode,
                    sorted(frame_paths.items()),
                    cameras,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        path = cache_dir / f"{digest}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            if not (cached["receipt"]["changed_frames"] or cached["receipt"]["abstained_frames"]):
                return cameras, cached["receipt"] | {"cache": "hit"}
            return cached["cameras"], cached["receipt"] | {"cache": "hit"}
    refined, receipt = court_paint_refinement.refine_camera_document(
        cameras, mode=mode, frame_paths=frame_paths
    )
    if receipt.get("unread_frames"):
        raise FileNotFoundError(
            "court paint refinement could not read a native frame for "
            f"{receipt['unread_frames']} supported frames"
        )
    if path is not None:
        save(path, {"receipt": receipt, "cameras": refined})
    return refined, receipt


def warm_paint_cache(labels: dict, cameras: dict, policy: dict, cache_dir: Path) -> dict | None:
    """Refit a component source's cameras once, before its components run.

    Every component would otherwise repeat the whole-point refit inside its own wall cap.
    Returns None when the refit is off or cannot run here (components then decide).
    """
    settings = shared_settings(policy)
    mode = settings.get("court_paint_refinement")
    if mode not in ("on", "abstain"):
        return None
    try:
        cameras, _, _ = _near_baseline_cameras(cameras, labels, settings)
        _, receipt = _paint_cameras(cameras, labels, mode, cache_dir)
    except (FileNotFoundError, ValueError) as error:
        return {"status": "not_warmed", "reason": f"{type(error).__name__}: {error}"}
    return {
        "status": "warmed",
        "refits": receipt.get("refits"),
        "changed_frames": receipt["changed_frames"],
    }


def _run_attempt(row: dict, output: Path, policy: dict) -> dict:
    """Execute search from observations, choose by input score, then score once."""
    from cv.experiments.connected_shooting import agent_whole_point_search as search
    from cv.experiments.connected_shooting.candidate_attempts import completed_source_allowed
    from cv.experiments.connected_shooting import labeled_common_source as loader
    from cv.experiments.connected_shooting import labeled_context_census as census
    from cv.experiments.connected_shooting import labeled_context_witness_scope as scope
    from cv.experiments.connected_shooting import labeled_isolated_serve as isolated
    from cv.pipeline import s6_preparation_policy as preparation
    from cv.pipeline import s6_shared_refinement as refinement
    from cv.pipeline import s6_terminal_context

    if output.exists():
        raise ValueError("fresh attempt output required; previous fitted states cannot enter")
    from cv.experiments.connected_shooting import net_collision, passive_bounce

    requested_policy = policy
    settings = shared_settings(policy)
    inputs = validate_row(row)
    labels, packet, cameras = (
        json.loads(inputs[name].read_text()) for name in ("labels", "packet", "cameras")
    )
    from cv.pipeline import s6_first_flight_scope as prefix
    from cv.pipeline import s6_first_contact_role

    from cv.pipeline import s6_component_scope

    if s6_component_scope.active(packet["attempts"][0]):
        from cv.pipeline import s6_component_orchestration

        s6_component_orchestration.validate_policy(policy, packet["attempts"][0])
    policy, component_applicability = s6_component_scope.applied_policy(policy, packet)
    if s6_component_scope.active(packet["attempts"][0]):
        s6_component_scope.validate_inputs(
            packet["attempts"][0], labels, cameras, settings["observation_partition"]
        )
    policy, rally_receipt = s6_first_contact_role.applied_policy(
        policy, packet["attempts"][0], labels
    )
    settings = shared_settings(policy)
    role_receipt = rally_receipt or prefix.role_policy(packet, settings["first_contact_role"])
    from cv.pipeline import s6_contact_prefix_runtime as contact_prefix

    prepared, prefix_receipt = contact_prefix.prepare_packet(
        packet, labels, cameras, settings["contact_prefix_scope"], settings["observation_partition"]
    )
    applied_policy, applicability = contact_prefix.applied_policy(policy, prepared)
    applicability = component_applicability or applicability
    settings = shared_settings(applied_policy)
    mid_flight_receipt = None
    if settings.get("net_cord_mid_flight", "off") != "off":
        from cv.pipeline import net_cord_response as _mid_flight

        # Labelled net hits only; an automatic net row keeps the tape clip.
        labelled = (prepared["attempts"][0].get("stream_origins") or {}).get(
            "events"
        ) != "automatic"
        frames = (
            _mid_flight.mid_flight_net_hits(prepared["attempts"][0]["events"]) if labelled else []
        )
        mid_flight_receipt = {
            "mode": settings["net_cord_mid_flight"],
            "mid_flight_net_hit_frames": frames,
            "net_cord_response_before": settings.get("net_cord_response", "absent"),
        }
        if frames:
            applied_policy = applied_policy | {"net_cord_response": _mid_flight.ADMISSIBLE_SET}
            settings = shared_settings(applied_policy)
        mid_flight_receipt["net_cord_response_applied"] = settings.get(
            "net_cord_response", "absent"
        )
    if role_receipt is not None:
        settings = settings | {"first_contact_role": role_receipt["effective_role"]}
    from cv.experiments.connected_shooting import event_constraints as _clearance

    serve_origin = role_receipt is None or role_receipt.get("effective_role") not in {
        "rally",
        "unknown",
    }
    _clearance.SERVE_CLEARANCE_SCALE_M = (
        _clearance.HARD_SERVE_CLEARANCE_SCALE_M
        if settings.get("serve_tape_clearance", "off") == "hard" and serve_origin
        else None
    )
    prepared, preparation_receipt = preparation.prepare_packet(
        prepared, labels, settings["preparation"]
    )
    if settings["contact_prefix_scope"] != "off":
        preparation_receipt["contact_prefix_scope"] = prefix_receipt
        preparation_receipt["scope_applicability"] = applicability
        if applicability is not None:
            preparation_receipt["first_contact_role"] = {
                "requested_role": settings["first_contact_role"],
                "effective_role": settings["first_contact_role"],
                "reason": "original shared input role preserved at a contact-prefix boundary",
            }
    if component_applicability is not None:
        preparation_receipt["contact_component_scope"] = prepared["attempts"][0][
            "observation_scope"
        ]
        preparation_receipt["scope_applicability"] = component_applicability
    if role_receipt is not None:
        preparation_receipt["first_contact_role"] = role_receipt
    if mid_flight_receipt is not None:
        preparation_receipt["net_cord_mid_flight"] = mid_flight_receipt
    if settings.get("serve_tape_clearance", "off") != "off":
        preparation_receipt["serve_tape_clearance"] = {
            "mode": settings["serve_tape_clearance"],
            "serve_origin": serve_origin,
            "serve_flight_clearance_scale_m": _clearance.SERVE_CLEARANCE_SCALE_M,
        }
    if settings["observed_horizon_tail"] == "on":
        from cv.experiments.connected_shooting import observed_horizon_tail as horizon_tail

        try:
            prepared, horizon_receipt = horizon_tail.qualify(prepared, labels)
            preparation_receipt["observed_horizon_tail"] = {
                "status": "qualified",
                **horizon_receipt,
            }
        except (ValueError, KeyError, TypeError) as error:
            preparation_receipt["observed_horizon_tail"] = {
                "status": "not_applicable",
                "reason": str(error),
            }
    before_net_tail = prepared
    if settings["terminal_net_tail"] == "on":
        from cv.experiments.connected_shooting import labeled_terminal_net_tail as net_tail

        try:
            prepared, net_tail_receipt = net_tail.qualify(prepared, labels)
            preparation_receipt["terminal_net_tail"] = {"status": "qualified", **net_tail_receipt}
        except ValueError as error:
            preparation_receipt["terminal_net_tail"] = {
                "status": "not_applicable",
                "reason": str(error),
            }
    if settings["terminal_net_membership"] != "off":
        # Both branches are qualified once, from the original packet as it entered
        # the net-tail contract, and bound into one receipt the search, replay and
        # export all resolve their effective attempt from.
        from cv.pipeline import s6_terminal_net_membership as net_membership

        prepared, membership_receipt = net_membership.qualify(
            prepared, before_net_tail, labels, settings["terminal_net_membership"]
        )
        if membership_receipt is not None:
            preparation_receipt["terminal_net_membership"] = membership_receipt
    output.mkdir(parents=True)
    cameras, near_receipt, near_changed = _near_baseline_cameras(cameras, labels, settings)
    if near_receipt is not None:
        # An unchanged document (a direct camera, or a fallback already on the
        # paint) must keep the file the component plan was hashed against.
        # Installing the flag-only copy fails the source-identity check.
        if near_changed:
            refined_cameras = output / "cameras_near_baseline.json"
            save(refined_cameras, cameras)
            inputs = dict(inputs)
            inputs["cameras"] = refined_cameras
        preparation_receipt["near_baseline_refinement"] = near_receipt
    if settings.get("court_paint_refinement") in ("on", "abstain"):
        cameras, paint_receipt = _paint_cameras(
            cameras, labels, settings["court_paint_refinement"], _paint_cache_dir(output)
        )
        # An unchanged document keeps the file the component plan was hashed against.
        if paint_receipt["changed_frames"] or paint_receipt["abstained_frames"]:
            refined_cameras = output / "cameras_court_paint.json"
            save(refined_cameras, cameras)
            inputs = dict(inputs)
            inputs["cameras"] = refined_cameras
        preparation_receipt["court_paint_refinement"] = {
            key: value for key, value in paint_receipt.items() if key not in ("groups", "runs")
        } | {
            "abstained_runs": [run for run in paint_receipt.get("runs") or [] if run["abstained"]],
            "applied_groups": [
                {"frames": group["frames"], "tried": group["tried"]}
                for group in paint_receipt.get("groups") or []
                if group.get("applied")
            ],
        }
    from cv.pipeline import s6_player_camera

    inputs, player_camera_receipt = s6_player_camera.prepare(
        inputs, row, packet, cameras, output, settings["player_camera_coordinates"]
    )
    preparation_receipt["player_camera_coordinates"] = player_camera_receipt
    if prepared is not packet:
        prepared_path = output / "prepared_packet.json"
        save(prepared_path, prepared)
        inputs = inputs | {"packet": prepared_path}
    command = search_arguments(row, inputs, output, applied_policy)
    if preparation_receipt.get("terminal_net_tail", {}).get("status") == "qualified":
        command.extend(["--terminal-net-tail", "on"])
    if preparation_receipt.get("observed_horizon_tail", {}).get("status") == "qualified":
        command.extend(["--observed-horizon-tail", "on"])
    if preparation_receipt.get("terminal_net_membership", {}).get("status") == "qualified":
        command.extend(["--terminal-net-membership", settings["terminal_net_membership"]])
    nets = preparation_receipt["resolved_net_events"]
    if nets:
        command.extend(["--terminal-rebound", "on"])
    association_policy = "observed-contact" if settings["preparation"] == "on" else "original"
    scoped_labels = preparation.attempt_labels(labels, prepared)
    if settings["fit_ground_witness"] != "off":
        # Bind this opt-in dependency before capturing the invocation, even
        # though search imports it lazily only on its fitting branch.
        from cv.experiments.connected_shooting import fit_ground_witness_center  # noqa: F401

    if settings.get("depth_conditioned_seed", "off") == "on":
        from cv.experiments.connected_shooting import depth_conditioned_seed  # noqa: F401

    implementation_files = implementation_records()

    save(
        output / "invocation.json",
        {
            "key": row["key"],
            "inputs": row,
            "policy": requested_policy,
            "shared_settings": settings,
            "scope_applicability": applicability,
            "preparation": preparation_receipt,
            "executed_inputs": {
                name: provenance.file_record(path) for name, path in inputs.items()
            },
            "arguments": command,
            "cold_search": True,
            "runtime_model_calls": 0,
            "prepared_observations_reused": True,
            "fitted_input_states_reused": False,
            "code": provenance.git_record(paths.REPO_ROOT),
            "implementation_files": implementation_files,
        },
    )
    previous = sys.argv
    try:
        with preparation.bounded_net_context(nets) as search_net:
            with preparation.association.context(
                scoped_labels, cameras, association_policy
            ) as search_association:
                sys.argv = [search.__file__, *command]
                search.main()
    finally:
        sys.argv = previous
    search_path = output / "search" / "report.json"
    report = json.loads(search_path.read_text())
    if player_camera_receipt["enabled"]:
        report["s6_player_camera_coordinates"] = player_camera_receipt
    report["net_collision_policy"] = net_collision.active_policy()
    recorded_passive = report.get(
        "passive_bounce_response",
        report.get("configuration", {}).get("passive_bounce_response", "off"),
    )
    if recorded_passive != passive_bounce.active():
        raise ValueError("actual search passive bounce policy differs from stage invocation")
    if passive_bounce.active() != "off":
        report["passive_bounce_response"] = recorded_passive
    save(search_path, report)
    if settings["preparation"] == "on":
        report["experimental_contact_association"] = search_association
        if nets:
            report["experimental_net_support_policy"] = preparation.net.POLICY
        report["s6_preparation_policy"] = preparation_receipt | {
            "net_support": search_net,
            "association": search_association,
        }
        save(search_path, report)
    initialization = report.get("initialization", {})
    if initialization.get("uses_truth_xyz_or_velocity") is not False:
        raise ValueError("search did not attest observation-only initialization")
    args = SimpleNamespace(
        **inputs,
        search_report=search_path,
        output=output,
        pose_image_scale=row.get("pose_image_scale", 1.0),
        player_order=list(row.get("player_order") or []),
        athlete_evidence=settings["athlete_evidence"],
        athlete_root_reach_loss=settings["athlete_root_reach_loss"],
        missing_player_position=settings["missing_player_position"],
        observation_fallback="on",
        witness_surface=row["surface"],
        source_stage=(
            "input-ranked-completed" if completed_source_allowed(report) else "input-ranked-refined"
        ),
        association_policy=association_policy,
    )
    # This adapter replays only the search produced immediately above, never an
    # externally supplied/cached fit. Selection precedes reading acceptance bits.
    from cv.pipeline import s6_terminal_net_membership as net_membership

    with (
        preparation.bounded_net_context(nets) as replay_net,
        preparation.declared_absent_nets(
            net_membership.absent_source_nets(report.get("configuration", {}))
        ),
    ):
        bundle = loader.load(args)
        from cv.pipeline import s6_owner_gate_loosening as owner_gates

        bundle["threshold"] = owner_gates.overlay(bundle["threshold"], settings)
        from cv.experiments.connected_shooting import observation_net_seed

        from contextlib import nullcontext as _nullcontext

        from cv.pipeline import net_cord_response as _cord

        _mode = settings.get("net_cord_response", _cord.TAPE_CLIP)
        _h_tol = settings.get("net_cord_tape_band_m")
        if _mode == _cord.ADMISSIBLE_SET:
            _mode_cm = _cord.using_mode(_cord.ADMISSIBLE_SET, h_tol=_h_tol)
        elif _mode == _cord.EVIDENCE_BOUND:
            _mode_cm = _cord.using_evidence((), h_tol=_h_tol)
        else:
            _mode_cm = _nullcontext()
        with _mode_cm, observation_net_seed.response_context(bundle.get("initial_search_fit", {})):
            context = bundle["context"]
            context_receipt = {"status": "source_context"}
            if settings["terminal_context_ownership"] == "on":
                context, context_receipt = s6_terminal_context.prepare(
                    context,
                    bundle["parameters"],
                    bundle["labels"],
                    bundle["cameras"],
                    bundle["search"],
                )
            else:
                try:
                    extended, receipt = census.extended_context(
                        context, bundle["labels"], bundle["cameras"], bundle["search"]
                    )
                    context_receipt = receipt
                    if extended is not None:
                        context = extended
                except ValueError as error:
                    context_receipt = {"status": "source_context_fallback", "reason": str(error)}
            initial_parameters = bundle["parameters"].copy()
            initial_verdict, initial_measurement = census.score(
                context,
                initial_parameters,
                bundle["threshold"],
                bundle["duration"],
                bundle["ending"],
                scorer=scope.LOCAL_SCORE,
            )
            parameters, response, ground_response = (
                initial_parameters,
                bundle.get("initial_search_fit", {}).get("net_response"),
                None,
            )
            verdict, measurement = initial_verdict, initial_measurement
            refinement_receipt = {"status": "disabled", "applied_mechanisms": []}
            if settings["refinement"] == "on":
                active = bundle | {"context": context}
                if (
                    settings["interior_ground_normal"] != "off"
                    or settings["toss_player_prior"] == "on"
                    or settings["serve_ground_normal"] == "on"
                    or settings["net_first_contact_toss"] == "on"
                ):
                    with inputs["pose_csv"].open() as stream:
                        active["observed_pose_rows"] = list(csv.DictReader(stream))
                    active["pose_image_scale"] = row.get("pose_image_scale", 1.0)
                    active["observed_pose_record"] = provenance.file_record(inputs["pose_csv"])
                    active["observed_pose_space"] = pose_space_sidecar(inputs["pose_csv"])
                if settings["first_contact_role"] == "serve":
                    active["serve_role_evidence"] = {
                        "role": "serve",
                        "origin": "supplied_observation",
                        "scope": "common cohort first-contact role; no serve ordinal inferred",
                    }
                refined = refinement.refine(
                    active,
                    s6_component_scope.refinement_policy(
                        refinement.Policy(
                            following_bounce_intervals=settings["following_bounce_intervals"]
                            == "on",
                            joint_toss_requalification=settings["joint_toss_requalification"]
                            == "on",
                            earlier_toss_support=settings["earlier_toss_support"] == "on",
                            net_ground_normal=settings["net_ground_normal"] == "on",
                            net_mesh_height=settings["net_mesh_height"] == "on",
                            net_first_ground_candidates=settings["net_first_ground_candidates"]
                            == "on",
                            net_first_contact_toss=settings["net_first_contact_toss"] == "on",
                            net_ground_horizontal=settings["net_ground_horizontal"],
                            prefix_local_boundary=settings["prefix_local_boundary"] == "on",
                            prefix_source_incumbent=settings["prefix_source_incumbent"] == "on",
                            prefix_pixel_loss=settings["prefix_pixel_loss"],
                            prefix_following_ground_timing=settings[
                                "prefix_following_ground_timing"
                            ],
                            prefix_net_revisit=settings["prefix_net_revisit"] == "on",
                            terminal_impact_intervals=settings["terminal_impact_intervals"] == "on",
                            terminal_ground_normal=settings["terminal_ground_normal"] == "on",
                            terminal_net_coupling=settings["terminal_net_coupling"] == "on",
                            terminal_ground_coupling=settings["terminal_ground_coupling"] == "on",
                            interior_ground_normal=settings["interior_ground_normal"] != "off",
                            terminal_ground_sparse_wings=settings["terminal_ground_sparse_wings"]
                            == "on",
                            interior_sparse_wings=settings["interior_sparse_wings"] == "on",
                            interior_short_blocks=settings["interior_short_blocks"] == "on",
                            interior_ground_horizontal=settings["interior_ground_horizontal"]
                            == "retention",
                            interior_restoration_geometry_only=settings[
                                "interior_restoration_geometry_only"
                            ]
                            == "on",
                            interior_schedule=(
                                settings["interior_ground_normal"]
                                if settings["interior_ground_normal"] in ("sweep", "sweep_revisit")
                                else "first"
                            ),
                            interior_seconds=settings["interior_block_seconds"],
                            toss_player_prior=settings["toss_player_prior"] == "on",
                            independent_toss_horizontal_prior=settings[
                                "independent_toss_horizontal_prior"
                            ]
                            == "on",
                            serve_ground_normal=settings["serve_ground_normal"] == "on",
                        ),
                        applied_policy,
                    ),
                )
                context, parameters = refined.pop("context"), refined.pop("parameters")
                verdict, measurement = refined.pop("verdict"), refined.pop("measurement")
                response = refined.pop("net_response")
                ground_response = refined.pop("ground_response", None)
                refinement_receipt = refined
    unchanged = implementation_unchanged(implementation_files)
    if not unchanged:
        raise ValueError("implementation source changed during cold invocation")
    result = {
        "schema": "labeled_s6_cold_attempt_v1",
        "key": row["key"],
        "status": "measured_requires_native_review",
        **(
            {"observation_policy": measurement["observation_partition"]}
            if "observation_partition" in measurement
            else {}
        ),
        "declared_flights": row["declared_flights"],
        "surface": row["surface"],
        "cold_search": True,
        "runtime_model_calls": 0,
        "selected_by": {
            **{
                key: value
                for key, value in bundle["selection"].items()
                if key != "from_scratch_initialization"
            },
            "selection_replays_same_invocation_search": True,
            "selected_search": provenance.file_record(search_path),
        },
        "context": context_receipt,
        "evaluation_context": serialize_context(context),
        "preparation": preparation_receipt
        | {"net_support": search_net, "association": search_association},
        "replay_preparation": {
            "net_support": replay_net,
            "association": bundle["association_preparation"],
        },
        "refinement": refinement_receipt,
        "net_response": response,
        "ground_response": ground_response,
        "implementation_files_unchanged": unchanged,
        "implementation_files": implementation_files,
        "initial_search": {
            "parameters": initial_parameters,
            "measurement": initial_measurement,
            "verdict": initial_verdict,
            "selection": bundle["selection"],
            "projection_replay": bundle["projection_replay"],
            **(
                {
                    name: bundle["initial_search_fit"][name]
                    for name in ("net_response", "net_response_initialization")
                }
                if "initial_search_fit" in bundle
                else {}
            ),
        },
        "initialization": initialization,
        "verdict": verdict,
        "measurement": measurement,
        "parameters": parameters,
        "search_budget": report.get("search_budget"),
        "replay": bundle["projection_replay"],
        "scorer": {"rung": isolated.RUNG, "serve_covariance_scope": "serve_only"},
        "independent_xyz_truth_available": False,
        "human_derived_inputs": True,
        "upstream_automatic_inference_eligible": False,
        "incorrect_accept_count": None,
    }
    from cv.pipeline import s6_input_origin

    if s6_input_origin.origin(row) == s6_input_origin.AUTOMATIC:
        result.update(
            observation_origin=s6_input_origin.AUTOMATIC,
            automatic_provenance=row["automatic_provenance"],
            human_derived_inputs=False,
            upstream_automatic_inference_eligible=True,
        )
    result["net_collision_policy"] = net_collision.active_policy()
    if passive_bounce.active() != "off":
        result["passive_bounce_response"] = passive_bounce.active()
    result["ball_observation_operator"] = bundle["observation_operator"]
    from cv.experiments.connected_shooting import native_seed_check_pixels

    # Search-seed use and subsequent all-native refinement are distinct stages.
    # Preserve the actual scoring chronology without claiming independent checks.
    result["check_pixel_provenance"] = {
        "search_native_seed_selection": report.get("native_seed_check_pixel_usage"),
        "selected_native_seed": native_seed_check_pixels.fit_usage(result["initial_search"]),
        "shared_refinement": native_seed_check_pixels.refinement_disclosures(refinement_receipt),
        "residual_scope": "nominal check-set residuals; independence is stage-specific",
        "scoring_timing_changed": False,
    }
    s6_first_contact_role.mark_result(
        result, s6_first_contact_role.validate(prepared["attempts"][0], labels)
    )
    result["final_state_binding"] = final_state_binding(result)
    save(output / "result.json", result)
    return result


def run_attempt(row: dict, output: Path, policy: dict) -> dict:
    """Run one shared preparation policy from fixed observations, with explicit receipts."""
    from cv.experiments.connected_shooting import labeled_preparation_recovery_inputs as recovery

    from cv.experiments.connected_shooting import net_collision, ground_contact, passive_bounce
    from cv.experiments.connected_shooting import event_constraints

    settings = shared_settings(policy)
    if settings.get("contact_components", "off") != "off":
        from cv.pipeline import s6_component_scope

        source = validate_row(row)
        packet = json.loads(source["packet"].read_text())
        if not s6_component_scope.active(packet["attempts"][0]):
            raise ValueError("component source jobs require the source component orchestrator")
        applied, _ = s6_component_scope.applied_policy(policy, packet)
        settings = shared_settings(applied)
    if settings["contact_prefix_scope"] != "off":
        from cv.pipeline import s6_contact_prefix_runtime

        source = validate_row(row)
        labels, packet, cameras = (
            json.loads(source[k].read_text()) for k in ("labels", "packet", "cameras")
        )
        prepared, _ = s6_contact_prefix_runtime.prepare_packet(
            packet,
            labels,
            cameras,
            settings["contact_prefix_scope"],
            settings["observation_partition"],
        )
        applied, _ = s6_contact_prefix_runtime.applied_policy(policy, prepared)
        settings = shared_settings(applied)
    enabled = settings["preparation"] == "on"
    receipt = {
        "policy": "bounded_observation_seed_and_associated_contact_feet",
        "enabled": enabled,
        "applications": [],
        "status": "running",
        "native_observations_changed": False,
        "fitting_bounds_changed": False,
    }
    try:
        with (
            net_collision.physical_eligibility(settings["net_physical_eligibility"] == "on"),
            ground_contact.using(settings["ground_settling"] == "on"),
            passive_bounce.using(settings["passive_bounce_response"]),
            # _run_attempt sets the serve clearance for a serve origin; restore it here.
            event_constraints.serve_clearance(None),
            recovery.recovery_scope(
                launch_feasibility=enabled,
                contact_player_feet=enabled,
                receipts=receipt["applications"],
            ),
        ):
            result = _run_attempt(row, output, policy)
        receipt["status"] = "completed"
        result["initialization_recovery"] = receipt
        save(output / "result.json", result)
        return result
    except BaseException as error:
        receipt.update(status="execution_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if output.is_dir():
            save(output / "initialization_recovery.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    run_attempt(job["row"], args.output, job["policy"])


if __name__ == "__main__":
    main()
