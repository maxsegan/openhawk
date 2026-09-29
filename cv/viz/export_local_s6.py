"""Stage actual cold S6 results for the existing 3D viewer, without selecting a new fit.

Usage: uv run python -m cv.viz.export_local_s6 --run RUN --out STAGED_3D [--case KEY]
The output is a 3D viewer directory, not a new site. --merge preserves historical index
entries in an existing staged copy. This command refuses the live canonical directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace

from cv.experiments.connected_shooting.candidate_attempts import completed_source_allowed
from cv.viz import export_connected_3d as legacy
from cv.viz.portal_3d_src.build import copy_static
from scripts.shared_data import repository_root, resolve_shared_root

GROUP = "local_s6_cold"
GROUP_LABEL = "Local S6 — cold labeled inputs; gate results, visual review pending"


AUTOMATIC_GROUP = "automatic_shared_s6"


def checked_export_origin(row: dict) -> dict:
    """Automatic display provenance requires the same producer checks as inference."""
    origin = row.get("observation_origin", "labeled")
    if origin == "labeled" and row.get("automatic_provenance") is None:
        return {"human_derived": True, "automatic_inference_eligible": False}
    from cv.pipeline import s6_input_origin, s6_labeled_stage

    if s6_input_origin.origin(row) != s6_input_origin.AUTOMATIC:
        raise ValueError("unsupported export observation origin")
    s6_labeled_stage.validate_row(row)
    return {
        "human_derived": False,
        "automatic_inference_eligible": True,
        "observation_origin": "automatic",
        "automatic_provenance": row["automatic_provenance"],
    }


def export_namespace(origin: dict) -> tuple[str, str]:
    return (
        ("automatic_s6", AUTOMATIC_GROUP)
        if origin.get("observation_origin") == "automatic"
        else ("local_s6", GROUP)
    )


def mark_automatic_observations(doc: dict, origin: dict) -> None:
    """Legacy viewer field names remain compatible; visible semantics are explicit."""
    if origin.get("observation_origin") != "automatic":
        return
    doc["video_overlay"]["labeled_marker"] = "blue dot: automatic observed native detector center"
    doc["video_overlay"]["observation_operator"] = "nominal_center"
    doc["observed_events"] = doc["labeled_events"]
    doc["event_origin"] = "automatic"
    for tick in doc.get("timeline_ticks", []):
        if tick.get("source") == "label":
            tick["source"] = "automatic_event"


def native_context_bounds(
    label: dict,
    source_clip: str,
    original_range: tuple[int, int],
    start_frame: float,
    end_frame: float,
    data_root: Path,
    seconds: float = 2.0,
) -> tuple[int, int]:
    """Extend within the original point's contiguous image pool, never the next reel clip."""
    images = [r for r in label["source_pack"]["images"] if r["clip"] == source_clip]
    if legacy._declared_native_clock(label) or legacy._explicit_native_pts(label):
        # Prepared native windows already include explicitly bound context. A
        # neighboring filesystem image is not evidence of another playable epoch.
        legacy._native_timebase(label)
        lo, hi = original_range
        if not lo <= start_frame <= end_frame <= hi:
            raise ValueError("competitive attempt exceeds the declared native context")
        available = {r["frame"] for r in images}
        if not set(range(lo, hi + 1)) <= available:
            raise ValueError("declared native context lacks original images")
        return lo, hi
    segment = label.get("segmentation", {}).get("view_window")
    if segment is not None:
        if (
            not isinstance(segment, list)
            or len(segment) != 2
            or any(type(value) is not int for value in segment)
            or not 1 <= segment[0] <= start_frame <= end_frame <= segment[1]
        ):
            raise ValueError("segmented native context must contain the competitive attempt")
        available = {int(row["frame"]) for row in images}
        if not set(range(segment[0], segment[1] + 1)) <= available:
            raise ValueError("segmented native context lacks source images")
        # A source container can also carry a later speed graphic or another
        # serve. Those evidence images must not extend this attempt's playback.
        return segment[0], segment[1]
    directories = {(data_root / r["image_url"]).parent for r in images}
    if len(directories) != 1:
        raise ValueError("one original native point-image pool required")
    directory = next(iter(directories))
    suffixes = {Path(r["image_url"]).suffix for r in images}
    if len(suffixes) != 1:
        raise ValueError("one native image format required per clip")
    suffix = next(iter(suffixes))
    timebase, _ = legacy._native_timebase(label)
    padding = seconds * timebase["fps"]
    desired_lo = max(1, math.floor(start_frame - padding))
    desired_hi = math.ceil(end_frame + padding)
    lo, hi = original_range
    while lo > desired_lo and (directory / f"f_{lo - 1:04d}{suffix}").is_file():
        lo -= 1
    while hi < desired_hi and (directory / f"f_{hi + 1:04d}{suffix}").is_file():
        hi += 1
    return lo, hi


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checked_passive_bounce_policy(result, search, invocation, manifest):
    """Old OFF artifacts remain readable; enabled response must be explicitly bound."""
    from cv.experiments.connected_shooting import passive_bounce

    requested = passive_bounce.validate(manifest["policy"].get("passive_bounce_response", "off"))
    if invocation["policy"].get("passive_bounce_response", "off") != requested:
        raise ValueError("invocation passive bounce policy differs from manifest")
    for record in (result, search):
        if record.get("passive_bounce_response", "off") != requested:
            raise ValueError("search/result passive bounce policy differs from manifest")
    if "passive_bounce_response" in result and result.get("final_state_binding", {}).get(
        "sha256", {}
    ).get("passive_bounce_response") != value_sha256(result["passive_bounce_response"]):
        raise ValueError("final passive bounce policy differs from state binding")
    return requested


def checked_net_policy(result: dict, search: dict, invocation: dict, manifest: dict) -> str:
    """Display only geometry bound to one explicit collision policy throughout."""
    requested = manifest["policy"].get("net_physical_eligibility", "off")
    if requested not in ("off", "on"):
        raise ValueError("unsupported net physical eligibility policy")
    expected = "physical_mesh_v1" if requested == "on" else "timing_window_v1"
    if invocation["policy"].get("net_physical_eligibility", "off") != requested:
        raise ValueError("invocation net collision policy differs from manifest")
    for record in (result, search):
        if record.get("net_collision_policy", "timing_window_v1") != expected:
            raise ValueError("search/result net collision policy differs from manifest")
    if "net_collision_policy" in result:
        if result.get("final_state_binding", {}).get("sha256", {}).get(
            "net_collision_policy"
        ) != value_sha256(result["net_collision_policy"]):
            raise ValueError("final net collision policy differs from invocation binding")
    elif expected != "timing_window_v1":
        raise ValueError("physical-mode result requires an explicit net collision policy")
    return expected


def checked_observation_operator(result: dict, search: dict, packet: dict) -> dict:
    """Require exported geometry to retain its original center/front contract."""
    from cv.experiments.connected_shooting import observation_operator

    expected = observation_operator.from_report(search, packet)
    declared = result.get("ball_observation_operator")
    if declared is None:
        if expected["kind"] != "leading_front":
            raise ValueError("center result requires a bound observation operator")
    elif declared != expected or result.get("final_state_binding", {}).get("sha256", {}).get(
        "ball_observation_operator"
    ) != value_sha256(declared):
        raise ValueError("final observation operator differs from source or binding")
    return expected


def checked_final_state(result: dict, seed: list) -> None:
    """Match final output to the same invocation's refinement and response receipt."""
    receipt = result["refinement"]
    binding = result["final_state_binding"]
    if binding.get("schema") != "labeled_s6_final_state_binding_v1":
        raise ValueError("final state binding schema required")
    for name in ("parameters", "measurement", "verdict", "net_response", "evaluation_context"):
        if binding.get("sha256", {}).get(name) != value_sha256(result[name]):
            raise ValueError(f"final {name} differs from invocation binding")
    from cv.pipeline import s6_first_contact_role

    if s6_first_contact_role.FIELD in result and binding.get("sha256", {}).get(
        s6_first_contact_role.FIELD
    ) != value_sha256(result[s6_first_contact_role.FIELD]):
        raise ValueError("final rally role differs from invocation binding")
    if "ground_response" in result and binding.get("sha256", {}).get(
        "ground_response"
    ) != value_sha256(result["ground_response"]):
        raise ValueError("final ground response differs from invocation binding")
    if "net_collision_policy" in result and binding.get("sha256", {}).get(
        "net_collision_policy"
    ) != value_sha256(result["net_collision_policy"]):
        raise ValueError("final net collision policy differs from invocation binding")
    if binding.get("sha256", {}).get("initial_parameters") != value_sha256(seed):
        raise ValueError("refinement initial parameters differ from selected seed")
    if result.get("implementation_files_unchanged") is not True:
        raise ValueError("invocation implementation identity was not preserved")
    if receipt.get("status") != "disabled" and (
        receipt.get("external_fitted_inputs_used") is not False
        or receipt.get("runtime_model_calls") != 0
    ):
        raise ValueError("refinement must use only this invocation's local fitted state")
    initial_fit = result.get("initial_search", {})
    response = initial_fit.get("net_response")
    if response is not None or "net_response_initialization" in initial_fit:
        from cv.experiments.connected_shooting import observation_net_seed

        with observation_net_seed.response_context(initial_fit):
            pass
        for name in ("net_response", "net_response_initialization"):
            if binding.get("sha256", {}).get("initial_" + name) != value_sha256(initial_fit[name]):
                raise ValueError("initial net response differs from invocation binding")
    vector, ground_response = seed, None
    applied = []
    for stage in receipt.get("stages", []):
        if stage.get("status") != "applied":
            continue
        fit = stage["fit"]
        if stage["stage"] == "serve":
            mechanism = stage["mechanism"]
            if mechanism == "isolated_serve":
                vector = fit["best"]["parameters"]
            elif mechanism == "serve_prefix":
                if "ground_response" in fit:
                    from cv.experiments.connected_shooting import (
                        labeled_interior_ground_response as ground,
                    )

                    declared = fit.get("fit_policy", {}).get("direct_first_ground_normal", {})
                    normalized_ground = ground.normalize(fit["ground_response"])
                    if normalized_ground is None:
                        raise ValueError("serve first-ground response is missing")
                    routes = normalized_ground["routes"]
                    if (
                        declared.get("enabled") is not True
                        or fit.get("initial_ground_response") != ground_response
                        or len(routes) != 1
                        or routes[0]["flight_index"] != 0
                        or routes[0]["ground_ordinal"] != 0
                        or routes[0]["native_contact_epoch"] != fit["contact_epoch"]
                        or declared.get("active_epoch") != fit["contact_epoch"]
                        or fit["initial_source"] != vector
                    ):
                        raise ValueError(
                            "serve first-ground response differs from its source or latent contact"
                        )
                    ground_response = fit["ground_response"]
                vector = fit["full_vector"]
            else:
                raise ValueError("unsupported applied serve refinement")
            applied.append(mechanism)
        elif stage["stage"] == "net":
            if "initial_source" in stage and stage["initial_source"] != vector:
                raise ValueError("net refinement source differs from preceding stage")
            vector, response = fit["best"]["parameters"], fit["best"]["response"]
            applied.append("terminal_net")
        elif stage["stage"] == "prefix_net_revisit":
            # Response-aware declared-net prefix revisit: same-invocation post-net
            # vector and net response carried in; only the prefix vector changes.
            if stage.get("mechanism") != "serve_prefix_net_revisit":
                raise ValueError("unsupported applied prefix revisit mechanism")
            if applied[-1:] != ["terminal_net"]:
                raise ValueError("prefix revisit requires an immediately preceding net stage")
            if fit["initial_source"] != vector:
                raise ValueError("prefix revisit source differs from preceding stage")
            if response is None or fit["retained_later_net_response"] != response:
                raise ValueError("prefix revisit changed its carried net response")
            if stage.get("carried_ground_response") != ground_response:
                raise ValueError("prefix revisit changed its carried ground response")
            vector = fit["full_vector"]
            applied.append("serve_prefix_net_revisit")
        elif stage["stage"] == "net_follow_on":
            if applied[-1:] != ["serve_prefix_net_revisit"]:
                raise ValueError("net follow-on requires an immediately preceding prefix revisit")
            if stage.get("initial_source") != vector:
                raise ValueError("net follow-on source differs from the revisited prefix")
            vector, response = fit["best"]["parameters"], fit["best"]["response"]
            applied.append("terminal_net_follow_on")
        elif stage["stage"] == "interior_ground":
            if fit["initial_source"] != vector:
                raise ValueError("interior refinement source differs from preceding stage")
            if (
                fit["initial_ground_response"] != ground_response
                or fit["retained_net_response"] != response
            ):
                raise ValueError("interior refinement changed its preceding response identity")
            if (
                fit.get("gate_selection") is not False
                or fit.get("external_fitted_inputs_used") is not False
            ):
                raise ValueError(
                    "interior refinement must select only same-invocation input objective"
                )
            vector, ground_response = fit["full_vector"], fit["ground_response"]
            applied.append("interior_ground_normal")
        elif stage["stage"] == "terminal_net_coupling":
            if fit["initial_source"] != vector:
                raise ValueError("terminal coupling source differs from preceding stage")
            if (
                fit["initial_ground_response"] != ground_response
                or fit["initial_net_response"] != response
            ):
                raise ValueError("terminal coupling changed its preceding response identity")
            if (
                fit.get("gate_selection") is not False
                or fit.get("external_fitted_inputs_used") is not False
            ):
                raise ValueError("terminal coupling must use the same-invocation input objective")
            vector, ground_response, response = (
                fit["full_vector"],
                fit["ground_response"],
                fit["net_response"],
            )
            applied.append("terminal_net_contact_coupling")
        elif (
            stage["stage"] == "terminal_ground"
            and stage.get("mechanism") == "terminal_ground_normal"
        ):
            # Persistent terminal first-ground route: the registry may gain one final-flight
            # route; the source vector and every other response identity are retained.
            if fit["initial_source"] != vector:
                raise ValueError("terminal refinement source differs from preceding stage")
            if (
                fit["initial_ground_response"] != ground_response
                or fit["retained_net_response"] != response
            ):
                raise ValueError("terminal refinement changed its preceding response identity")
            if (
                fit.get("gate_selection") is not False
                or fit.get("external_fitted_inputs_used") is not False
            ):
                raise ValueError(
                    "terminal refinement must select only same-invocation input objective"
                )
            vector, ground_response = fit["full_vector"], fit["ground_response"]
            applied.append("terminal_ground_normal")
        elif stage["stage"] == "terminal_ground_coupling":
            if fit["initial_source"] != vector:
                raise ValueError("terminal ground coupling source differs from preceding stage")
            if (
                fit["initial_ground_response"] != ground_response
                or fit["retained_net_response"] != response
            ):
                raise ValueError("terminal ground coupling changed its preceding response identity")
            if (
                fit.get("gate_selection") is not False
                or fit.get("external_fitted_inputs_used") is not False
                or fit.get("replay", {}).get("status") != "verified"
            ):
                raise ValueError(
                    "terminal ground coupling requires replayed input-objective selection"
                )
            vector, ground_response = fit["full_vector"], fit["ground_response"]
            applied.append("terminal_ground_contact_coupling")
        elif stage["stage"] == "terminal_ground":
            if fit.get("retained_ground_response") != ground_response:
                raise ValueError("terminal refinement changed the retained ground response")
            if fit["initial_source"] != vector:
                raise ValueError("terminal refinement source differs from preceding stage")
            if fit["retained_net_response"] != response:
                raise ValueError("terminal refinement changed the retained net response")
            vector = fit["full_vector"]
            applied.append("terminal_impact_interval")
        else:
            raise ValueError("unsupported applied refinement stage")
    if receipt.get("applied_mechanisms", []) != applied:
        raise ValueError("applied refinement list differs from stage receipts")
    if (
        result["parameters"] != vector
        or result["net_response"] != response
        or result.get("ground_response") != ground_response
    ):
        raise ValueError("final vector or response differs from applied refinement")
    flights = result["measurement"]["dense_flights"]
    contacts = result["evaluation_context"]["scene"]["contact_frames"]
    if [flight["start_frame"] for flight in flights] != contacts[:-1]:
        raise ValueError("final measured contact epochs differ from evaluation context")


def checked_independent_proposal_binding(document: dict, policy: dict) -> None:
    """Bind the independent source-proposal policy to the contact witness that ran.

    ``extend_hypotheses`` silently contributes nothing when the witness declares
    the arm off, so a ranked list alone cannot distinguish an on-policy run from
    an off-policy witness. Compare the two declarations directly instead.
    """
    from cv.pipeline import s6_independent_event_proposals as independent

    if independent.policy(document) != policy.get("independent_event_proposals", "off"):
        raise ValueError("independent proposal witness differs from the shared policy")


def checked_optional_contact_source(
    search: dict,
    packet: dict,
    cameras: dict,
    policy: dict,
    context: dict,
    *,
    automatic: bool,
    data_root: Path,
    repo: Path,
) -> None:
    """Bind the optional selector to original observations, never an evaluation winner.

    Other topologies retain scored execution receipts rather than complete candidate
    pools. Check that inventory and its deterministic winner; independently replay
    the winning pool below using the ordinary shared selector.
    """
    from cv.pipeline import s6_optional_contacts as optional

    from cv.pipeline import s6_optional_event_union as union

    from cv.pipeline import s6_terminal_net_membership as net_membership

    from cv.pipeline import s6_independent_event_proposals as independent

    selection = search["optional_contact_selection"]
    joint = selection.get("joint_source_families")
    membership_block = selection.get(net_membership.RECEIPT_KEY)
    # A requested ``predicted`` policy legitimately qualifies nothing: an ordinary
    # ground ending, a human net or an unqualified tail all return an explicit
    # not_applicable reason, which the preparation replay above checks. The
    # comparison here is therefore against the membership actually applied to
    # this packet, so an ON-but-ineligible search keeps its ordinary optional
    # selection and its ordinary export unchanged.
    membership_applied = net_membership.receipt_of(packet["attempts"][0]) is not None
    if policy.get("terminal_net_membership", "off") == "off" and membership_applied:
        raise ValueError("terminal net membership receipt requires the explicit shared policy")
    if (membership_block is not None) != membership_applied:
        raise ValueError("terminal net membership selection differs from the applied membership")
    membership_only = selection.get("policy") == net_membership.SELECTION_POLICY
    if membership_only and membership_block is None:
        raise ValueError("membership selection requires its declared branch provenance")
    policy_family = None if membership_only else union.selection_family(selection)
    bounce = policy_family == "bounce"
    composed = policy_family == union.JOINT
    contacts_on = policy.get("optional_contacts", "off") == "on"
    bounce_policy = policy.get("optional_bounces", "off")
    from cv.pipeline import s6_optional_bounces

    bounces_on = bounce_policy in ("classifier", *s6_optional_bounces.INTERIOR_MODES)
    final_on = policy.get("optional_final_contacts", "off") == "on"
    conditional = union.conditional_policy(policy.get("optional_conditional_grounds"))
    conditional_on = conditional == "on"
    if joint is not None and (
        union.conditional_policy(selection.get("optional_conditional_grounds")) != conditional
    ):
        raise ValueError("optional conditional ground selection differs from the shared policy")
    family = selection.get("source_family")
    # The declared contact scope is part of the shared policy, so a v1 export may
    # not replay a final-scope selection and the reverse.
    if (
        joint is None
        and not bounce
        and (selection.get("scope") == optional.FINAL_SCOPE) != (final_on and family != union.JOINT)
    ):
        raise ValueError("optional contact selection scope differs from the shared policy")
    if composed:
        # A composed winner requires an explicit interior bounce mode or a
        # declared final-contact scope, with both source arms prepared.
        enabled = (
            contacts_on
            and (
                bounce_policy in s6_optional_bounces.INTERIOR_MODES
                or (final_on and bounces_on)
                or (conditional_on and bounces_on)
            )
            and sorted(joint or ()) == sorted(union.FAMILIES)
            and family == union.JOINT
            and union.family_of(selection.get("selected_topology")) == union.JOINT
        )
    elif joint is not None:
        enabled = (
            contacts_on
            and bounces_on
            and sorted(joint) == sorted(union.FAMILIES)
            and family in (*union.FAMILIES, union.SUPPLIED)
            # The no-addition branch is qualified by the bounce witness, so only a
            # contact winner may carry the contact policy.
            and bounce == (family != "contact")
        )
    elif membership_only:
        # A membership run with no ordinary optional witness: the declared branch
        # family is the only source of alternatives and owns the whole selection.
        enabled = not contacts_on and not bounces_on and membership_block is not None
    else:
        enabled = (
            bounces_on and not contacts_on
            if bounce
            else contacts_on
            and not bounces_on
            and selection.get("policy") == "source_witness_optional_contact_v1"
        )
    # The independent source-proposal arm is its own explicit shared policy over
    # the ordinary contact witness: the winner names an ``independent_proposal_``
    # topology that the same contact reader rebuilds below. Its declared policy is
    # checked against the witness that actually carried it, not assumed from the
    # legacy contact policy string alone.
    independent_on = policy.get("independent_event_proposals", "off") == "on"
    # Source proposals are prepared on the assisted research inputs, whose origin
    # is labeled rather than automatic. That handoff is admitted here by exactly
    # the check the run itself applied in the shared stage: the frozen original
    # packet is preserved byte for byte and the witness keeps every original firm
    # event. Every other origin still requires an automatic stage input.
    origin_checked = automatic
    if not automatic and independent_on and not bounce and joint is None and not membership_only:
        handoff = packet.get("optional_contact_witness")
        if not handoff:
            raise ValueError("research proposal handoff requires its prepared contact witness")
        independent.validate_research_packet(
            packet,
            legacy._read_json(legacy._verify_binding(handoff, data_root, repo)),
            lambda binding: legacy._verify_binding(binding, data_root, repo),
        )
        origin_checked = True
    if (
        not origin_checked
        or not enabled
        or selection.get("selection_uses_gates") is not False
        or selection.get("selection_uses_reference") is not False
        or search.get("selector_uses_withheld_pixels") is not False
        or selection.get("total_search_seconds") != policy.get("search_seconds")
    ):
        raise ValueError("optional contact selection requires its source-only shared policy")
    original_attempt = packet["attempts"][0]
    # The selection is validated against the effective attempt its own declared
    # branch was solved on, rebuilt from the original packet and the receipt.
    attempt = net_membership.replay_attempt(original_attempt, selection)
    document = None
    if membership_only:
        binding = None
        alternatives = []
    elif joint is not None:
        # Rebuild the union from both original witnesses; neither arm's inventory
        # may stand in for the other and no document is concatenated.
        bindings = {name: packet.get(f"optional_{name}_witness") for name in union.FAMILIES}
        if not all(bindings.values()):
            raise ValueError("joint optional selection requires both prepared source witnesses")
        documents = {
            name: legacy._read_json(legacy._verify_binding(record, data_root, repo))
            for name, record in bindings.items()
        }
        checked_independent_proposal_binding(documents["contact"], policy)
        s6_optional_bounces.validate_interior_policy_binding(documents["bounce"], bounce_policy)
        if selection.get("source_bindings") != union.check_source_binding(
            documents, attempt, conditional
        ):
            raise ValueError("joint optional source bindings differ from original witnesses")
        # Older union contact winners omitted the redundant selection.scope.
        # Derive it only from both hash-verified original witnesses and the exact
        # replayed union binding above; never modify a saved selection or fit.
        declared_final = optional.declared_final_scope(documents["contact"])
        if declared_final != final_on or (
            "scope" in selection
            and selection["scope"]
            != (optional.FINAL_SCOPE if declared_final and family == "contact" else None)
        ):
            raise ValueError("optional contact selection scope differs from the shared policy")
        alternatives, receipts = union.hypotheses(documents, attempt, conditional)
        if selection.get("optional_family_receipts") != receipts:
            raise ValueError("joint optional family eligibility differs from original witnesses")
        if composed and not (union.composes(documents) or conditional_on):
            # Either composing scope is admissible here: the v2 bounce witness, or the
            # declared final-contact scope under the v1 bounce witness.
            raise ValueError(
                "composed optional winner requires the v2 bounce or final-contact source scope"
            )
        binding = (
            None
            if composed
            else packet.get(
                "optional_contact_witness" if family == "contact" else "optional_bounce_witness"
            )
        )
        if union.family_of(selection.get("selected_topology")) != family:
            raise ValueError("joint optional winner does not belong to its declared family")
    else:
        binding = packet.get("optional_bounce_witness" if bounce else "optional_contact_witness")
        alternatives = None
    if composed:
        # A composed winner belongs to both witnesses and to neither arm alone.
        if selection.get("witness_records") != bindings or selection.get("witness_record"):
            raise ValueError("composed optional selection must name both original witnesses")
    elif membership_only:
        if selection.get("witness_record") or selection.get("witness_records"):
            raise ValueError("membership selection names no ordinary optional witness")
    elif not binding or selection.get("witness_record") != binding:
        raise ValueError("optional contact witness differs from original packet")
    if alternatives is None:
        document = legacy._read_json(legacy._verify_binding(binding, data_root, repo))
        if bounce:
            s6_optional_bounces.validate_interior_policy_binding(document, bounce_policy)
        else:
            checked_independent_proposal_binding(document, policy)
        alternatives = optional.search_hypotheses(
            net_membership.bounce_witness(attempt, document) if bounce else document, attempt
        )
    if selection.get("source_ranked_hypotheses") != alternatives:
        raise ValueError("optional contact hypotheses differ from source witness ranking")
    choices = [
        {"name": "supplied", "events": attempt["events"], "added": [], "occurrence_log_odds": 0},
        *alternatives,
    ]
    chosen = next((r for r in choices if r["name"] == selection.get("selected_topology")), None)
    if chosen is None or chosen["events"] != search["events"]:
        raise ValueError("optional contact selected events are not a source hypothesis")
    if composed:
        added = len(chosen["added"])
        complexity_changed = (
            selection.get("added_contact_count") != chosen["added_contact_count"]
            or selection.get("added_bounce_count") != chosen["added_bounce_count"]
            or chosen["added_contact_count"] + chosen["added_bounce_count"] != added
            or selection.get("added_parameter_count") != chosen["added_parameter_count"]
            # Bounces add no continuous dimensions; only contacts do.
            or chosen["added_parameter_count"] != 6 * chosen["added_contact_count"]
            or selection.get("joint_composition") != chosen["joint_composition"]
        )
    elif joint is not None:
        added = len(chosen["added"])
        complexity_changed = (
            selection.get("added_contact_count") != (added if family == "contact" else 0)
            or selection.get("added_bounce_count") != (added if family == "bounce" else 0)
            or selection.get("added_parameter_count")
            != (0 if family == "supplied" else chosen.get("added_parameter_count"))
        )
    else:
        complexity_changed = selection.get("added_contact_count") != (
            0 if bounce else len(chosen["added"])
        )
        if bounce:
            complexity_changed |= (
                selection.get("added_bounce_count") != len(chosen["added"])
                or selection.get("added_parameter_count") != 0
                or chosen.get("added_parameter_count", 0) != 0
            )
    if complexity_changed or selection.get("occurrence_log_odds") != chosen["occurrence_log_odds"]:
        raise ValueError("optional contact complexity or occurrence prior changed")
    partition = policy.get("observation_partition", "fifth_frame_withheld")
    if search["configuration"].get("observation_partition", "fifth_frame_withheld") != partition:
        raise ValueError("optional contact observation partition changed")
    # The shared source builder retains any normally qualified terminal context.
    signature = optional.observation_signature(context["attempt"], cameras, partition)
    n = selection.get("training_observations")
    if type(n) is not int or n < 1 or n != len(signature):
        raise ValueError("optional contact fixed observation count differs from original inputs")

    def alternatives_for(branch_attempt: dict) -> list[dict]:
        """The same source beams this branch scope enumerates, in the same order."""
        if membership_only:
            return []
        if joint is not None:
            return union.hypotheses(
                {
                    **documents,
                    "bounce": net_membership.bounce_witness(branch_attempt, documents["bounce"]),
                },
                branch_attempt,
                conditional,
            )[0]
        return optional.search_hypotheses(
            net_membership.bounce_witness(branch_attempt, document) if bounce else document,
            branch_attempt,
        )

    branches = selection.get("branches", [])
    if membership_block is None:
        expected_inventory = [r["name"] for r in choices]
        actual_inventory = [r.get("name") for r in branches]
    else:
        selected_membership = membership_block["selected_membership"]
        expected_inventory = []
        for name in net_membership.MEMBERSHIPS:
            branch_attempt = net_membership.effective_attempt(original_attempt, name)
            rows = alternatives if name == selected_membership else alternatives_for(branch_attempt)
            expected_inventory.append((name, "supplied"))
            expected_inventory.extend((name, row["name"]) for row in rows)
        actual_inventory = [(r.get("membership"), r.get("name")) for r in branches]
        if selection.get("terminal_net_membership_parameter_offset") != membership_block.get(
            "added_parameter_count_offset"
        ):
            raise ValueError("membership parameter complexity differs from its declared branch")
    if actual_inventory != expected_inventory:
        raise ValueError("optional contact branch inventory lost a source hypothesis")
    allowed_statuses = {"fitted", "no_fit", "topology_failed"}
    if selection.get("input_domain_selection") == "original_physical_domain_before_rank_v1":
        allowed_statuses.add("input_domain_refused")
    if any(r.get("status") not in allowed_statuses for r in branches):
        raise ValueError("unsupported optional contact branch receipt")
    scores = selection.get("candidate_scores", [])
    if membership_block is None:
        scored = [r.get("topology") for r in scores]
        completed = [r["name"] for r in branches if r["status"] == "fitted"]
    else:
        scored = [(r.get("membership"), r.get("topology")) for r in scores]
        completed = [(r.get("membership"), r["name"]) for r in branches if r["status"] == "fitted"]
    if scored != completed:
        raise ValueError("optional contact score inventory differs from completed branches")
    if not scores or any(
        not isinstance(r.get("score"), (int, float)) or not math.isfinite(r["score"])
        for r in scores
    ):
        raise ValueError("finite optional contact branch scores required")
    if membership_block is None:
        winner = min(scores, key=lambda r: (r["score"], r["topology"]))
        expected_winner = {"topology": chosen["name"], "score": selection.get("score")}
    else:
        winner = min(
            scores,
            key=lambda r: (
                r["score"],
                net_membership.index_of(r.get("membership")),
                r["topology"],
            ),
        )
        expected_winner = {
            "topology": chosen["name"],
            "score": selection.get("score"),
            "membership": membership_block["selected_membership"],
        }
    if winner != expected_winner:
        raise ValueError("optional contact topology is not the recorded physical-objective winner")


def checked_candidate(
    result: dict,
    search: dict,
    *,
    original_context: dict | None = None,
    optional_source_checked: bool = False,
) -> dict:
    """Verify the cold input-ranked seed and any in-invocation final refinement."""
    selection = result["selected_by"]
    if result.get("cold_search") is not True or selection.get("gate_used") is not False:
        raise ValueError("cold, gate-independent stage selection required")
    stage = selection.get("source_stage")
    if (search.get("optional_contact_selection") is not None) != (
        stage == "input-ranked-optional-contact"
    ):
        raise ValueError("optional contact search and result selection policies differ")
    if stage == "input-ranked-optional-contact":
        if not optional_source_checked or original_context is None:
            raise ValueError("original optional contact inputs must be checked before export")
        candidates = [*search.get("coarse_candidates", []), *search.get("refined_candidates", [])]
    elif stage == "input-ranked-refined":
        candidates = search.get("refined_candidates", [])
    elif stage == "input-ranked-completed" and completed_source_allowed(search):
        candidates = search.get("completed_candidates", [])
    else:
        raise ValueError("unsupported stage selection policy")
    admission = None
    seeded_net = search.get("configuration", {}).get("observation_net_seed", "off") != "off"
    if search.get("configuration", {}).get("terminal_net_tail") is not None or seeded_net:
        if original_context is None or (
            not seeded_net and getattr(original_context["scene"], "terminal_net_tail", None) is None
        ):
            raise ValueError("original terminal input domain must be physically replayed")
        from cv.experiments.connected_shooting.labeled_common_source import input_domain_admission

        admission = input_domain_admission(original_context, include_seeded_final_net=seeded_net)
    refined = "initial_search" in result
    parameters = result["initial_search"]["parameters"] if refined else result["parameters"]
    if stage == "input-ranked-optional-contact":
        from cv.experiments.connected_shooting import labeled_serve_recipe as recipe

        candidate, replayed = recipe.select_candidate(
            search,
            len(original_context["scene"].pixels),
            "refined",
            None,
            input_admission=admission,
        )
        if (
            any(selection.get(name) != value for name, value in replayed.items())
            or "selected_index" in selection
        ):
            raise ValueError("recorded optional contact selection differs from shared selector")
    else:
        options = []
        for index, candidate in enumerate(candidates):
            rank = candidate.get("evidence", {}).get("input_only_rank_score")
            vector = candidate.get("measurement", {}).get("fit", {}).get("parameters", [])
            if (
                isinstance(rank, (int, float))
                and math.isfinite(rank)
                and len(vector) == len(parameters)
                and all(math.isfinite(v) for v in vector)
            ):
                if admission is None or admission(candidate)["admissible"]:
                    options.append((rank, candidate["depth_hypothesis_m"], index))
        if not options or min(options)[2] != selection["selected_index"]:
            raise ValueError("recorded candidate is not the input-objective winner")
        candidate = candidates[selection["selected_index"]]
    if candidate["measurement"]["fit"]["parameters"] != parameters:
        raise ValueError("result parameters differ from selected cold candidate")
    if (
        stage != "input-ranked-optional-contact"
        and candidate["evidence"]["input_only_rank_score"] != selection["source_rank"]
    ):
        raise ValueError("recorded objective differs from selected candidate")
    from cv.experiments.connected_shooting import interior_contact_epochs

    timing = candidate["measurement"]["fit"].get(interior_contact_epochs.FIELD)
    enabled = (
        search.get("configuration", {}).get("interior_contact_epochs", "off") == "source_interval"
    )
    if enabled != (timing is not None):
        raise ValueError("selected contact timing differs from its declared search policy")
    if timing is not None:
        if original_context is None:
            raise ValueError("original contact timing inputs required for export")
        from cv.experiments.connected_shooting.per_flight_rescore import context_at_candidate_epoch

        expected = context_at_candidate_epoch(original_context, candidate)
        final_context = result["evaluation_context"]
        if final_context.get(interior_contact_epochs.FIELD) != timing:
            raise ValueError("final contact timing receipt differs from selected search")
        for name in ("scene", "heldout"):
            if list(final_context[name]["contact_frames"][1:-1]) != list(
                expected[name].contact_frames[1:-1]
            ):
                raise ValueError("final interior contact epochs differ from selected search")
    if refined:
        for name in ("net_response", "net_response_initialization"):
            if result["initial_search"].get(name) != candidate["measurement"]["fit"].get(name):
                raise ValueError("initial net response differs from selected cold candidate")
        if selection.get("selection_replays_same_invocation_search") is not True:
            raise ValueError("refinement seed must be from the same cold invocation")
        selection_fields = ("source_stage", "selected_index", "source_rank", "gate_used")
        if stage == "input-ranked-optional-contact":
            selection_fields += ("rule", "candidate_count")
        for name in selection_fields:
            if result["initial_search"]["selection"].get(name) != selection.get(name):
                raise ValueError("initial search selection differs from recorded seed")
        checked_final_state(result, parameters)
    # Always render final measured geometry, never the earlier seed's trajectory.
    return {"measurement": result["measurement"]}


def original_candidate_context(
    row: dict, search: dict, bindings: dict, policy: dict, data_root: Path, repo: Path
) -> dict | None:
    """Rebuild the source scene from verified observations; do not fit or read final XYZ."""
    if (
        search.get("configuration", {}).get("terminal_net_tail") is None
        and search.get("optional_contact_selection") is None
        and search.get("configuration", {}).get("terminal_net_membership") is None
        and search.get("configuration", {}).get("interior_contact_epochs", "off") == "off"
        and search.get("configuration", {}).get("observation_net_seed", "off") == "off"
    ):
        return None
    from cv.experiments.connected_shooting import labeled_common_source as common
    from cv.pipeline import s6_preparation_policy as preparation

    paths = {
        name: legacy._binding_path(binding, data_root, repo) for name, binding in bindings.items()
    }
    labels = legacy._read_json(paths["labels"])
    cameras = legacy._read_json(paths["cameras"])
    original_packet = legacy._read_json(legacy._binding_path(row["packet"], data_root, repo))
    from cv.pipeline import s6_contact_prefix_runtime as contact_prefix
    from cv.pipeline import s6_labeled_stage as stage

    # Same shared helper, same effective settings, applied to the original inputs
    # in the stage's order: the prefix is bound before ordinary preparation.
    settings = stage.shared_settings(policy)
    original_packet, _ = contact_prefix.prepare_packet(
        original_packet,
        labels,
        cameras,
        settings["contact_prefix_scope"],
        settings["observation_partition"],
    )
    preparation_policy = policy.get("preparation", "off")
    _, receipt = preparation.prepare_packet(original_packet, labels, preparation_policy)
    association_policy = "observed-contact" if preparation_policy == "on" else "original"
    declared = search.get("experimental_contact_association")
    if declared is not None and declared["policy"] != association_policy:
        raise ValueError("source association differs from frozen invocation policy")
    args = SimpleNamespace(
        **paths,
        dense_labels=None,
        player_ledger=None,
        athlete_prior_mode="stature_pose_soft",
        observation_fallback="on",
        witness_surface=row["surface"],
        pose_image_scale=row.get("pose_image_scale", 1.0),
        player_order=list(row.get("player_order") or []),
        athlete_evidence=policy.get("athlete_evidence", "required"),
        athlete_root_reach_loss=policy.get("athlete_root_reach_loss", "quadratic"),
    )
    from cv.pipeline import s6_terminal_net_membership as net_membership

    with preparation.bounded_net_context(receipt["resolved_net_events"]):
        with preparation.declared_absent_nets(
            net_membership.absent_source_nets(search.get("configuration", {}))
        ):
            with common.association.context(labels, cameras, association_policy):
                return common.recipe.rescore.build_context(args, search)


def contact_component_scope(result: dict) -> dict | None:
    return result.get("verdict", {}).get("contact_component_reconstruction", {}).get("scope")


def checked_contact_component_output(result, search, contract):
    """Keep separate component geometry bound to original source identity."""
    from cv.pipeline import s6_contact_components as components

    reconstruction = result["verdict"].get("contact_component_reconstruction", {})
    component = contract["component"]
    # A ball-track terminal closes the component on its supplied ending, not on
    # an original contact, so the scene carries one boundary more than indices.
    boundaries = len(component["original_contact_indices"]) + bool(
        component.get(components.TERMINAL_TRACK_FIELD)
    )
    scene = result["evaluation_context"]["scene"]
    flights = result["verdict"]["flights"]
    if (
        result["verdict"].get("complete_point") is not False
        or reconstruction.get("scope") != contract
        or reconstruction.get("complete_original_source") is not False
        or reconstruction.get("local_to_original") != component["local_to_original"]
        or result.get("preparation", {}).get("contact_component_scope") != contract
        or len(scene["contact_frames"]) != boundaries
        or scene["contact_frames"][-1] != component["end_frame"]
        or any(
            not contract["native_window"][0] <= frame < component["end_frame"]
            for frame in scene["contact_frames"][:-1]
        )
        or len(flights) != len(component["original_flight_indices"])
        or search.get("configuration", {}).get("contact_components") != "unresolved_ending"
    ):
        raise ValueError("exported component differs from original source binding")


def contact_prefix_scope(result: dict) -> dict | None:
    """The bound original-contact prefix contract a displayed result reconstructs."""
    scope = result.get("verdict", {}).get("contact_prefix_reconstruction", {}).get("scope")
    return scope if isinstance(scope, dict) else None


def checked_contact_prefix_output(result: dict, search: dict, contract: dict) -> None:
    """A prefix export may never claim the missing outgoing flight or a point ending.

    The measured roster is the retained prefix alone. The original inventory and
    its unresolved slots stay a separate receipt, so no count here is padded to
    the original source size and no source-level completion can become true.
    """
    from cv.pipeline import s6_contact_prefix_scope as prefix

    configuration = search.get("configuration", {})
    mode = contract.get("mode")
    if (
        mode not in (prefix.COVERAGE, prefix.TERMINAL_IDENTITY, prefix.UNRESOLVED_ENDING)
        or configuration.get("contact_prefix_scope") != mode
        or configuration.get("right_boundary_kind") != "original_contact"
    ):
        raise ValueError("contact-prefix output requires its explicit shared search policy")
    verdict = result["verdict"]
    reconstruction = verdict.get("contact_prefix_reconstruction") or {}
    if (
        verdict.get("complete_point") is not False
        or verdict.get("complete_original_point") is True
        or reconstruction.get("scope") != contract
        or reconstruction.get("complete_original_source") is not False
        or reconstruction.get("retained_original_flight_indices")
        != contract["retained_flight_indices"]
        or reconstruction.get("unresolved_original_slots") != contract["unresolved_original_slots"]
    ):
        raise ValueError("contact-prefix output cannot claim legacy complete-point success")
    if verdict.get("observed_scope_reconstruction") is not None or verdict.get(
        "observed_horizon_tail"
    ):
        raise ValueError("a contact prefix carries no observed-horizon tail grammar")
    # The scope module binds both horizon aliases to the same original contact.
    horizon = float(contract["modeled_horizon"])
    contacts = result["evaluation_context"]["scene"]["contact_frames"]
    if contacts[-1] != horizon:
        raise ValueError("exported scene lost the original right contact")
    # k measured flights against k+1 original contacts: the final contact opens no
    # modeled flight, and the unresolved original slots are never measured rows.
    # Under the scope-local interior mode an admitted optional contact adds one
    # modeled flight strictly inside the retained span; it adds no original slot,
    # so the original denominator and the unresolved roster above are unchanged.
    retained = len(contract["retained_flight_indices"])
    inserted = len(contacts) - 1 - retained
    start = float(contract["native_window"][0])
    horizon_frames = [float(frame) for frame in contacts[:-1]]
    if (
        inserted < 0
        or verdict.get("flight_count") != len(contacts) - 1
        or (inserted and mode != prefix.TERMINAL_IDENTITY)
        or any(not start <= frame < horizon for frame in horizon_frames)
    ):
        raise ValueError("exported prefix roster differs from the retained original flights")


def checked_observation_scope(result: dict, search: dict, packet: dict, labels: dict) -> None:
    """A replay cannot turn an explicitly unresolved horizon into a point ending."""
    from cv.experiments.connected_shooting import observation_scope
    from cv.pipeline import s6_contact_prefix_scope as prefix

    contract = observation_scope.validate(packet["attempts"][0], labels, search)
    if contract is None:
        return
    from cv.pipeline import s6_component_scope

    if contract.get("schema") == s6_component_scope.SCHEMA:
        checked_contact_component_output(result, search, contract)
        return
    if contract.get("schema") == prefix.SCHEMA:
        checked_contact_prefix_output(result, search, contract)
        return
    verdict = result["verdict"]
    if (
        verdict.get("complete_point") is not False
        or verdict.get("observed_scope_reconstruction", {}).get("scope") != contract
    ):
        raise ValueError("observation-scoped output cannot claim legacy complete-point success")
    from cv.experiments.connected_shooting.observed_horizon_tail import modeled_horizon

    if result["evaluation_context"]["scene"]["contact_frames"][-1] != modeled_horizon(
        packet["attempts"][0], contract
    ):
        raise ValueError("exported scene lost the original observation horizon")
    if contract.get("terminal_kind") == "observed_horizon":
        # The export must keep saying the ground count is unknown. A missing or
        # resolved tail receipt would display a physical ending the source
        # never witnessed.
        tail = verdict.get("observed_horizon_tail") or {}
        if (
            tail.get("terminal_ground_count") != "unknown"
            or tail.get("physical_ending") is not None
            or tail.get("witnessed_complete_point") is not False
            or tail.get("latent_ground_is_point_ending") is not False
            or verdict.get("complete_original_point") is not False
        ):
            raise ValueError("exported unresolved tail must keep its unknown terminal ground count")


def contact_prefix_context(
    result: dict,
    scope: dict,
    native_time: Callable[[float], float],
    lo: int,
    hi: int,
    *,
    track_terminal: bool = False,
) -> dict:
    """Display a prefix through its original right contact, keeping the video whole.

    The boundary is a source contact, not an observation horizon and not a point
    ending. Modeled XYZ stops there; the original native pictures continue to the
    end of the source window, so the viewer keeps the future frames as context.
    A component closed by its own ball-track terminal (``track_terminal``) ends on
    that supplied, still-unresolved ending instead of an original contact.
    """
    flights = result["verdict"]["flights"]
    start = min(f["start_frame"] for f in flights)
    fitted_end = max(f["end_frame"] for f in flights)
    horizon = scope.get("modeled_horizon")
    boundary = "supplied_end" if track_terminal else "original_contact"
    if (
        scope.get("right_boundary_kind") != boundary
        or scope.get("complete_original_source") is not False
        or scope.get("physical_ending") is not None
        or not isinstance(horizon, (int, float))
        or not math.isfinite(horizon)
        or not start < horizon <= hi
    ):
        raise ValueError("finite original right contact inside the native view required")
    if scope.get("observation_horizon") != horizon:
        raise ValueError("prefix horizon aliases must denote the original contact")
    scene_bounds = result.get("evaluation_context", {}).get("scene", {}).get("contact_frames")
    if scene_bounds is not None and scene_bounds[-1] != horizon:
        raise ValueError("displayed observation horizon differs from actual modeled support")
    return {
        # Reuse the viewer's unresolved-scope presentation: modeled support ends
        # here and the competitive ending stays unknown.
        "kind": "local_s6_observation_scope",
        "right_boundary_kind": boundary,
        "score_start_t": native_time(start),
        "score_end_t": native_time(horizon),
        "view_start_t": native_time(lo),
        "view_end_t": native_time(hi),
        "score_start_frame": start,
        "score_end_frame": horizon,
        "display_end_frame": horizon,
        "display_end_t": native_time(horizon),
        "observed_end_frame": None,
        "fitted_end_frame": fitted_end,
        "fitted_end_semantics": "legacy physical gate endpoint, not competitive ending",
        "modeled_horizon_frame": horizon,
        "physical_ending_frame": None,
        "ending_source": (
            "unresolved; the right boundary is a ball-track terminal, not a declared ending"
            if track_terminal
            else "unresolved; the right boundary is an original contact, not an ending"
        ),
        "observation_scope": scope,
        "observed_live_continuation": False,
        "source_native_window": list(scope["native_window"]),
        "source_ending_frame": scope.get("original_owner_end_frame"),
        "source_ending_kind": scope.get("original_ending_kind"),
        "complete_original_source": False,
        "retained_original_flight_indices": list(scope["retained_flight_indices"]),
        "unresolved_original_flight_count": scope["unresolved_original_flight_count"],
        "scope": (
            "Original contact-to-contact prefix. The last modeled boundary is an original "
            "source contact, not a physical ending: its outgoing flight was never modeled and "
            "the original source stays incomplete. Later original pictures remain playable "
            "context with no reconstructed ball."
        ),
    }


def contact_prefix_coverage(scope: dict, measured_flight_count: int | None = None) -> dict:
    """Original roster and unresolved slots, kept out of every measured array.

    The exporter enforces ``len(measured) == verdict.flight_count``; the original
    source denominator therefore lives here instead of padding the flight arrays
    with trajectories nothing measured.
    """
    result = {
        "schema": "local_s6_contact_prefix_coverage_v1",
        "right_boundary_kind": scope["right_boundary_kind"],
        "complete_original_source": False,
        "original_flight_slot_count": len(scope["original_inventory"]),
        "original_flight_slots_semantics": (
            "original source contact/endpoint slots; not an independent reference-flight count"
        ),
        "reference_flight_count": scope["reference_flight_count"],
        "retained_original_flight_indices": list(scope["retained_flight_indices"]),
        "measured_flight_count": len(scope["retained_flight_indices"]),
        "unresolved_original_flight_count": scope["unresolved_original_flight_count"],
        "unresolved_original_slots": deepcopy(scope["unresolved_original_slots"]),
        "first_coverage_failure": scope["first_coverage_failure"],
        "original_native_window": list(scope["native_window"]),
        "modeled_native_window": list(scope["modeled_native_window"]),
        "original_ending_kind": scope["original_ending_kind"],
        "original_ending_frame": scope["original_ending_frame"],
        "original_event_count": scope["original_event_count"],
        "retained_event_count": scope["retained_event_count"],
        "right_contact_frame": scope["modeled_horizon"],
        "right_contact_supported_native_frames": list(
            scope["right_contact_supported_native_frames"]
        ),
        "not_applicable_features": list(scope["not_applicable_features"]),
        "note": (
            "Unresolved original slots are unmodeled source flights, never measured rows. "
            "The original video continues past the right contact; modeled XYZ does not."
        ),
    }
    if scope.get("mode") == "terminal_identity":
        if measured_flight_count is None or measured_flight_count < len(
            scope["retained_flight_indices"]
        ):
            raise ValueError("terminal-identity coverage requires the actual measured flight count")
        result.update(
            mode=scope["mode"],
            terminal_identity_cut=scope["terminal_identity_cut"],
            terminal_identity_refusal=deepcopy(scope["terminal_identity_refusal"]),
            measured_flight_count=measured_flight_count,
            retained_original_flight_slot_count=len(scope["retained_flight_indices"]),
            inserted_interior_contact_count=measured_flight_count
            - len(scope["retained_flight_indices"]),
        )
    elif scope.get("mode") == "unresolved_ending":
        result.update(mode=scope["mode"], unresolved_ending_cut=scope["unresolved_ending_cut"])
    return result


def competitive_context(
    result: dict,
    native_time: Callable[[float], float],
    lo: int,
    hi: int,
    *,
    strict_native_clock: bool = False,
) -> dict:
    """Separate the supplied ending from replay aftermath without changing any verdict."""
    observation_scope = (
        result.get("verdict", {}).get("observed_scope_reconstruction", {}).get("scope")
    )
    component_scope = contact_component_scope(result)
    if component_scope is not None:
        # Shared display clipping only; no prefix preparation or admission receipt.
        view = contact_prefix_context(
            result,
            component_scope
            | {
                "unresolved_original_flight_count": len(
                    component_scope["unresolved_original_slots"]
                )
            },
            native_time,
            lo,
            hi,
            track_terminal=bool(component_scope.get("terminal_track_endpoint")),
        )
        view["scope"] = (
            "Independent original contact component; all original gaps remain unresolved."
        )
        return view
    prefix_scope = contact_prefix_scope(result)
    if prefix_scope is not None:
        if observation_scope is not None:
            raise ValueError("one right-boundary reconstruction per exported result")
        return contact_prefix_context(result, prefix_scope, native_time, lo, hi)
    flights = result["verdict"]["flights"]
    start = min(f["start_frame"] for f in flights)
    fitted_end = max(f["end_frame"] for f in flights)
    observed_end = result.get("context", {}).get("supplied_terminal_bounce_frame")
    if not isinstance(observed_end, (int, float)) or not math.isfinite(observed_end):
        observed_end = None
    # A rejected fit can end beyond the last original picture. Retain its
    # measured endpoint, but playback cannot invent an exposure timestamp there.
    clipped_fitted_end = strict_native_clock and observed_end is None and fitted_end > hi
    end = observed_end if observed_end is not None else (hi if clipped_fitted_end else fitted_end)
    if end <= start:
        raise ValueError("competitive ending must follow the first contact")
    # A winner's first in-bounds bounce is a scoring boundary in the old
    # benchmark, not the moment the visible ball stops being live. Preserve
    # that benchmark boundary and expose the observed continuation separately.
    ending = result.get("verdict", {}).get("labeled_ending_kind")
    live_after_first_bounce = (
        ending
        in {
            "ace",
            "winner",
            "unreturned_serve",
            "unreturned_winner",
            "unreturned_volley_winner",
            "unreturned_shot",
            "unreturned_overhead",
            "missed_return",
        }
        and result.get("evaluation_context", {}).get("termination_kind") != "second_bounce"
    )
    display_end = end
    if live_after_first_bounce:
        observed = [
            row["frame"]
            for row in result.get("measurement", {}).get("native_projection", [])
            if isinstance(row.get("frame"), (int, float)) and math.isfinite(row["frame"])
        ]
        dense = result.get("measurement", {}).get("dense_flights", [])
        if observed and dense:
            last_fit = max(flight["end_frame"] for flight in dense)
            display_end = max(end, min(max(observed), last_fit, hi))
    if observation_scope is not None:
        horizon = observation_scope.get("observation_horizon")
        if (
            not isinstance(horizon, (int, float))
            or not math.isfinite(horizon)
            or not start < horizon <= hi
        ):
            raise ValueError("finite modeled observation horizon inside native view required")
        source_horizon = horizon
        tail = result.get("verdict", {}).get("observed_horizon_tail")
        if tail is not None and tail.get("kind") == "observed_horizon":
            # The source clip can continue after the last supported ball row.
            # Display only the bound modeled tail, while retaining video context.
            horizon = tail.get("observation_horizon")
            if (
                not isinstance(horizon, (int, float))
                or not math.isfinite(horizon)
                or not start < horizon <= source_horizon
            ):
                raise ValueError("modeled tail must lie inside the source observation horizon")
        scene_bounds = result.get("evaluation_context", {}).get("scene", {}).get("contact_frames")
        if scene_bounds is not None and scene_bounds[-1] != horizon:
            raise ValueError("displayed observation horizon differs from actual modeled support")
        return {
            "kind": "local_s6_observation_scope",
            **(
                {"source_observation_horizon_frame": source_horizon}
                if horizon != source_horizon
                else {}
            ),
            **(
                {
                    # An unresolved tail is displayed as the partial visible
                    # flight it is: the modelled landings after the last
                    # supplied contact are latent, not a physical ending.
                    "terminal_ground_count": "unknown",
                    "supplied_ground_count": 0,
                    "observed_horizon_reason": tail["horizon_reason"],
                    "latent_terminal_ground_frames": tail["latent_terminal_ground_frames"],
                    "latent_ground_origin": tail["latent_ground_origin"],
                    "latent_ground_is_point_ending": False,
                    "visible_tail_semantics": (
                        "partial visible tail; the source stopped witnessing the ball in flight"
                    ),
                }
                if tail is not None
                else {}
            ),
            "score_start_t": native_time(start),
            "score_end_t": native_time(horizon),
            "view_start_t": native_time(lo),
            "view_end_t": native_time(hi),
            "score_start_frame": start,
            "score_end_frame": horizon,
            "display_end_frame": horizon,
            "display_end_t": native_time(horizon),
            "observed_end_frame": None,
            "fitted_end_frame": fitted_end,
            "fitted_end_semantics": "legacy physical gate endpoint, not competitive ending",
            "modeled_horizon_frame": horizon,
            "physical_ending_frame": None,
            "ending_source": "unresolved; supplied observation horizon is not a physical ending",
            "observation_scope": observation_scope,
            "observed_live_continuation": False,
            "scope": "Original native observation scope; ending type and complete-reference coverage remain unresolved."
            if tail is None
            else (
                "Original native observation scope with an UNKNOWN terminal ground count. "
                "The displayed tail is the partial visible flight; any modelled landing after "
                "the last supplied contact is latent, not a source event and not a point ending."
            ),
        }
    return {
        "kind": "local_s6_competitive",
        "score_start_t": native_time(start),
        "score_end_t": native_time(end),
        "view_start_t": native_time(lo),
        "view_end_t": native_time(hi),
        "score_start_frame": start,
        "score_end_frame": end,
        "display_end_frame": display_end,
        "display_end_t": native_time(display_end),
        "observed_live_continuation": display_end > end,
        "observed_end_frame": observed_end,
        "fitted_end_frame": fitted_end,
        **(
            {
                "modeled_endpoint_outside_native_view": True,
                "physical_ending_frame": None,
                "fitted_end_semantics": "measured endpoint beyond original pictures; gates unchanged",
            }
            if clipped_fitted_end
            else {}
        ),
        "ending_source": "supplied terminal observation"
        if observed_end is not None
        else (
            "native view boundary; modeled endpoint is beyond original pictures"
            if clipped_fitted_end
            else "measured competitive flight endpoint"
        ),
        "scope": "The original scoring boundary is unchanged. For an explicitly labeled "
        "unreturned first-bounce shot, observed live continuation stays visible through the "
        "last observed fitted exposure. Other aftermath remains neutral; flight gates are unchanged.",
    }


def checked_first_contact_role(result: dict, search: dict) -> dict | None:
    """Validate actual numerical rally semantics, not only the displayed stroke name."""
    from cv.pipeline import s6_first_contact_role as role

    bound = search.get("configuration", {}).get(role.FIELD)
    if result.get(role.FIELD) != bound:
        raise ValueError("exported rally role differs from original search")
    context = result.get("evaluation_context", {})
    players = context.get("players") or [{}]
    if players[0].get(role.FIELD) != bound:
        raise ValueError("final player context lost original rally role")
    if bound is None:
        return None
    role.validate(context["attempt"], report=search)
    if result.get("final_state_binding", {}).get("sha256", {}).get(role.FIELD) != value_sha256(
        bound
    ):
        raise ValueError("exported rally role lacks final-state binding")
    preparation = result.get("preparation", {}).get("first_contact_role", {})
    if (
        preparation.get("effective_role") not in role.BOUND_ROLES
        or preparation.get("effective_role") != bound.get("role")
        or preparation.get("source") != bound
    ):
        raise ValueError("exported rally role differs from applied input policy")
    if any(f.get("role") == "serve" for f in result["verdict"].get("flights", [])):
        raise ValueError("rally output cannot relabel a numerical serve")
    return bound


def display_contact_roles(
    result: dict, contacts: list[dict], per_flight: dict
) -> tuple[list[dict], dict]:
    """Present explicitly unknown stroke roles without altering the raw scorer verdict."""
    scope = (
        result.get("verdict", {}).get("observed_scope_reconstruction", {}).get("scope")
        or contact_prefix_scope(result)
        or contact_component_scope(result)
        or {}
    )
    role = result.get("preparation", {}).get("first_contact_role", {}).get("effective_role")
    if role in ("rally", "unknown"):
        contacts = deepcopy(contacts)
        for contact in contacts:
            if contact.get("phase") == "serve":
                contact.update(
                    phase="contact",
                    stroke_role=role,
                    role_source=(
                        "bound original rally contact"
                        if role == "rally"
                        else "unobserved original origin; stroke role not established"
                    ),
                )
        return contacts, per_flight
    if role != "unspecified" and scope.get("first_contact_role") != "unspecified":
        return contacts, per_flight
    contacts, per_flight = deepcopy(contacts), deepcopy(per_flight)
    if contacts:
        contacts[0].update(
            legacy_index_phase=contacts[0].get("phase"),
            phase="contact",
            stroke_role="unspecified",
            role_source="explicit effective input role",
        )
    if per_flight.get("flights"):
        first = per_flight["flights"][0]
        first.update(
            legacy_scorer_role=first.get("role"),
            role="unspecified",
            role_source="explicit effective input role; legacy gates unchanged",
        )
    return contacts, per_flight


def mark_context_frames(frames: dict, context: dict) -> None:
    """Keep native samples and coordinates intact; context has no acceptance color."""
    for index, frame in enumerate(frames["frame"]):
        if frame < context["score_start_frame"] or frame > context.get(
            "display_end_frame", context["score_end_frame"]
        ):
            frames["acceptance"][index] = "context"


def replayed_contact_prefix(
    invocation: dict, packet: dict, labels: dict, cameras: dict
) -> tuple[dict, dict]:
    """Re-derive the prefix packet and its abstentions from the original inputs.

    The same shared helper and the same effective settings the stage used, applied
    to the frozen originals: a tampered cut, roster, right contact, horizon or
    silently re-enabled terminal/topology setting cannot survive this.
    """
    from cv.pipeline import s6_contact_prefix_runtime as runtime
    from cv.pipeline import s6_labeled_stage as stage

    requested = invocation["policy"]
    settings = stage.shared_settings(requested)
    prepared, receipt = runtime.prepare_packet(
        packet,
        labels,
        cameras,
        settings["contact_prefix_scope"],
        settings["observation_partition"],
    )
    applied, applicability = runtime.applied_policy(requested, prepared)
    if invocation.get("scope_applicability") != applicability:
        raise ValueError("recorded scope applicability differs from the replayed prefix policy")
    # Only the scope-local abstention is asserted here; the rest of the shared
    # recipe is the ordinary frozen policy already checked against the manifest.
    declared = invocation.get("shared_settings", {})
    expected = stage.shared_settings(applied)
    # Older producers omit settings introduced later. An unrequested default
    # remains implicit; explicitly requested settings still require a receipt.
    defaults = stage.shared_settings({})
    if any(
        declared.get(name, defaults[name] if name not in requested else None) != expected[name]
        for name in ("contact_prefix_scope", *runtime.NOT_APPLICABLE_SETTINGS)
        if name in expected
    ):
        raise ValueError("executed shared settings differ from the replayed prefix abstention")
    return prepared, receipt


def checked_near_baseline_cameras(
    original: Path,
    actual: Path,
    result_path: Path,
    row: dict,
    preparation: dict,
    data_root: Path,
    repo: Path,
) -> None:
    """A changed camera document must be the stage's near-baseline refinement, replayed."""
    from cv.pipeline.court_near_baseline_refinement import refine_camera_document

    receipt = preparation.get("near_baseline_refinement") or {}
    if actual != (result_path.parent / "cameras_near_baseline.json").resolve() or not (
        receipt.get("changed_frames") or receipt.get("unresolved_frames")
    ):
        raise ValueError("executed cameras is not the frozen input or its declared preparation")
    labels = legacy._read_json(legacy._verify_binding(row["labels"], data_root, repo))
    frame_paths = {
        int(image["frame"]): image["source"]["path"]
        for image in (labels.get("source_pack") or {}).get("images") or []
        if image.get("source", {}).get("path")
    }
    expected, replayed = refine_camera_document(
        legacy._read_json(original), mode="on", frame_paths=frame_paths
    )
    if replayed != receipt or expected != legacy._read_json(actual):
        raise ValueError("near-baseline cameras do not replay the frozen cameras and receipt")


def checked_input_bindings(
    row: dict, search: dict, invocation: dict, result_path: Path, data_root: Path, repo: Path
) -> dict:
    """Verify frozen observations and an optional same-invocation prepared packet."""
    bindings = {(v.get("path_base"), v.get("path")): v for v in search["inputs"]}
    executed = invocation.get("executed_inputs", {})
    preparation = invocation.get("preparation", {})
    player_receipt = preparation.get("player_camera_coordinates", {})
    player_prepared = player_receipt.get("enabled", False)
    if player_prepared != (
        invocation.get("shared_settings", {}).get("player_camera_coordinates", "off") == "on"
    ):
        raise ValueError("player camera preparation policy differs from executed receipt")
    if player_prepared:
        from cv.pipeline import s6_player_camera

        actual_pose = legacy._verify_binding(executed["pose_csv"], data_root, repo).resolve()
        if (
            actual_pose != (result_path.parent / "prepared_player_pose.csv").resolve()
            or invocation.get("fitted_input_states_reused") is not False
            or search.get("s6_player_camera_coordinates") != player_receipt
        ):
            raise ValueError("executed player camera preparation is not bound to this invocation")
        # Player roots are derived from the executed cameras; a near-baseline
        # camera document is itself replayed below as the cameras binding.
        s6_player_camera.verify_derivation(
            {
                n: legacy._verify_binding(
                    executed.get(n, row[n]) if n == "cameras" else row[n], data_root, repo
                )
                for n in ("labels", "packet", "cameras", "pose_csv")
            },
            row,
            actual_pose,
            player_receipt,
        )
    # A declared prefix policy always replays, whether or not it changed the
    # packet: the recorded abstention and applicability are checked either way.
    prefix_expected = None
    prefix_applied = False
    if invocation.get("shared_settings", {}).get("contact_prefix_scope", "off") != "off":
        prefix_expected, replayed_receipt = replayed_contact_prefix(
            invocation,
            legacy._read_json(legacy._verify_binding(row["packet"], data_root, repo)),
            legacy._read_json(legacy._verify_binding(row["labels"], data_root, repo)),
            legacy._read_json(legacy._verify_binding(row["cameras"], data_root, repo)),
        )
        declared_receipt = preparation.get("contact_prefix_scope")
        if replayed_receipt != declared_receipt:
            raise ValueError("contact prefix does not replay frozen observations and receipt")
        prefix_applied = declared_receipt.get("status") == "qualified"
    from cv.pipeline import s6_first_contact_role

    source_packet = legacy._read_json(legacy._verify_binding(row["packet"], data_root, repo))
    source_labels = legacy._read_json(legacy._verify_binding(row["labels"], data_root, repo))
    from cv.pipeline import s6_component_scope, s6_labeled_stage as stage

    role_policy = invocation.get("policy", {})
    if s6_component_scope.active(next(iter(source_packet.get("attempts", [])), {})):
        source_cameras = legacy._read_json(legacy._verify_binding(row["cameras"], data_root, repo))
        contract = s6_component_scope.validate_inputs(
            source_packet["attempts"][0],
            source_labels,
            source_cameras,
            invocation["shared_settings"]["observation_partition"],
        )
        role_policy, applicability = s6_component_scope.applied_policy(role_policy, source_packet)
        if (
            preparation.get("contact_component_scope") != contract
            or invocation.get("scope_applicability") != applicability
        ):
            raise ValueError("component preparation differs from original source replay")
        expected = stage.shared_settings(role_policy)
        declared = invocation["shared_settings"]
        if any(
            declared.get(k) != expected[k]
            for k in (
                "contact_components",
                "contact_prefix_scope",
                "first_contact_role",
                "observation_scope",
                *s6_component_scope.prefix_runtime.NOT_APPLICABLE_SETTINGS,
            )
        ):
            raise ValueError("component executed policy differs from source replay")
    _, role_receipt = s6_first_contact_role.applied_policy(
        role_policy,
        next(iter(source_packet.get("attempts", [])), {}),
        source_labels,
    )
    if role_receipt is not None and preparation.get("first_contact_role") != role_receipt:
        raise ValueError("rally role applicability differs from original source policy")
    output = {}
    for name in ("labels", "packet", "cameras", "pose_csv"):
        supplied = row[name]
        original = legacy._verify_binding(supplied, data_root, repo).resolve()
        used = executed.get(name, supplied)
        actual = legacy._verify_binding(used, data_root, repo).resolve()
        changed = actual != original or used.get("sha256") != supplied["sha256"]
        preparation = invocation.get("preparation", {})
        occurrence_applied = (
            preparation.get("event_occurrence", {}).get("status") == "inserted_original_occurrences"
        )
        net_tail_applied = preparation.get("terminal_net_tail", {}).get("status") == "qualified"
        horizon_tail_applied = (
            preparation.get("observed_horizon_tail", {}).get("status") == "qualified"
        )
        membership_applied = (
            preparation.get("terminal_net_membership", {}).get("status") == "qualified"
        )
        # A qualified contact prefix is its own authorization for a changed packet:
        # it is bound before ordinary preparation and needs no other mechanism.
        if changed and name == "cameras":
            checked_near_baseline_cameras(
                original, actual, result_path, row, preparation, data_root, repo
            )
        elif (
            changed
            and not (name == "pose_csv" and player_prepared)
            and (
                name != "packet"
                or actual != (result_path.parent / "prepared_packet.json").resolve()
                or (
                    invocation.get("shared_settings", {}).get("preparation") != "on"
                    and not prefix_applied
                )
                or (
                    preparation.get("terminal_context", {}).get("status") != "applied"
                    and not occurrence_applied
                    and not net_tail_applied
                    and not horizon_tail_applied
                    and not membership_applied
                    and not prefix_applied
                )
                or invocation.get("fitted_input_states_reused") is not False
            )
        ):
            raise ValueError(f"executed {name} is not the frozen input or its declared preparation")
        if changed and name == "packet" and prefix_applied:
            # The executed packet must be exactly the replayed prefix; ordinary
            # preparation abstains on this scope and leaves it untouched.
            if prefix_expected != json.loads(actual.read_text()):
                raise ValueError("contact prefix does not replay frozen observations and receipt")
        elif (
            changed
            and name == "packet"
            and (
                occurrence_applied or net_tail_applied or horizon_tail_applied or membership_applied
            )
        ):
            # Recreate only observation preparation, never a fitted trajectory. A
            # receipt flag alone cannot authorize changed pixels, epochs or events.
            from cv.pipeline.s6_preparation_policy import prepare_packet

            labels_path = legacy._verify_binding(row["labels"], data_root, repo)
            expected, receipt = prepare_packet(
                json.loads(original.read_text()), json.loads(labels_path.read_text()), "on"
            )
            if (
                horizon_tail_applied
                or invocation.get("shared_settings", {}).get("observed_horizon_tail") == "on"
            ):
                # Replay the stage's own order: the unresolved-tail contract is
                # qualified from source observations before the net tail.
                from cv.experiments.connected_shooting.observed_horizon_tail import (
                    qualify as qualify_horizon,
                )

                try:
                    expected, details = qualify_horizon(
                        expected, json.loads(labels_path.read_text())
                    )
                    receipt["observed_horizon_tail"] = {"status": "qualified", **details}
                except (ValueError, KeyError, TypeError) as error:
                    receipt["observed_horizon_tail"] = {
                        "status": "not_applicable",
                        "reason": str(error),
                    }
            before_net_tail = expected
            if (
                net_tail_applied
                or invocation.get("shared_settings", {}).get("terminal_net_tail") == "on"
            ):
                from cv.experiments.connected_shooting.labeled_terminal_net_tail import qualify

                try:
                    expected, details = qualify(expected, json.loads(labels_path.read_text()))
                    receipt["terminal_net_tail"] = {"status": "qualified", **details}
                except ValueError as error:
                    receipt["terminal_net_tail"] = {
                        "status": "not_applicable",
                        "reason": str(error),
                    }
            membership_setting = invocation.get("shared_settings", {}).get(
                "terminal_net_membership", "off"
            )
            if membership_applied or membership_setting != "off":
                # Replay the stage's own order and the same shared resolver: both
                # branches are requalified from the original packet, never from a
                # saved receipt flag.
                from cv.pipeline import s6_terminal_net_membership as net_membership

                expected, details = net_membership.qualify(
                    expected,
                    before_net_tail,
                    json.loads(labels_path.read_text()),
                    membership_setting,
                )
                if details is not None:
                    receipt["terminal_net_membership"] = details
            if prefix_expected is not None:
                # A coverage check that abstained can precede ordinary event
                # preparation. Its independent replay was verified above.
                receipt["contact_prefix_scope"] = replayed_receipt
                receipt["scope_applicability"] = invocation.get("scope_applicability")
            packet_preparation = {
                k: v
                for k, v in preparation.items()
                # Camera-side receipts: their document is checked as the cameras binding.
                if k not in ("player_camera_coordinates", "near_baseline_refinement")
                and not (role_receipt is not None and k == "first_contact_role")
            }
            if receipt != packet_preparation or expected != json.loads(actual.read_text()):
                raise ValueError(
                    "event preparation does not replay frozen observations and receipt"
                )
        bound = bindings.get((used.get("path_base"), used.get("path")))
        if not bound or bound.get("sha256") != used.get("sha256"):
            raise ValueError(f"search input differs from executed {name}")
        output[name] = used
    executed_packet = legacy._read_json(legacy._verify_binding(output["packet"], data_root, repo))
    s6_first_contact_role.validate(
        next(iter(executed_packet.get("attempts", [])), {}),
        source_labels,
        search,
        requested=invocation.get("policy", {}).get("first_contact_role", "unspecified"),
    )
    from cv.pipeline import s6_rally_origin_cue

    if s6_rally_origin_cue.FIELD in search.get("configuration", {}):
        s6_rally_origin_cue.verify_report(
            search,
            source_labels,
            legacy._read_json(legacy._verify_binding(output["cameras"], data_root, repo)),
            legacy._verify_binding(output["pose_csv"], data_root, repo),
            row.get("pose_image_scale", 1.0),
        )
    return output


def export_observation_only(
    row: dict,
    result_path: Path,
    manifest_path: Path,
    out: Path,
    data_root: Path,
    repo: Path,
    run_label: str,
) -> dict:
    """Export original video for an unfitted attempt, with no inferred geometry."""
    result = legacy._read_json(result_path)
    if result.get("key") != row["key"] or result.get("measurement"):
        raise ValueError("observation-only export requires the matching unfitted result")
    entry = {
        "key": row["key"],
        "held": True,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "declared_flights": row["declared_flights"],
        "result": record(result_path),
    }
    if not row.get("labels"):
        return entry
    label_path = legacy._verify_binding(row["labels"], data_root, repo)
    label = legacy._read_json(label_path)
    return _export_observation_document(
        row,
        label,
        result,
        entry,
        result_path,
        manifest_path,
        out,
        data_root,
        run_label,
        checked_export_origin(row),
    )


def _export_observation_document(
    row: dict,
    label: dict,
    result: dict,
    entry: dict,
    result_path: Path,
    manifest_path: Path,
    out: Path,
    data_root: Path,
    run_label: str,
    origin: dict,
) -> dict:
    """Render already-verified observations; callers own source and ancestry checks."""
    prefix, group = export_namespace(origin)
    attempt = label["attempt"]
    source_clip = attempt["clip"]
    images = [r for r in label["source_pack"]["images"] if r["clip"] == source_clip]
    lo, hi = min(int(r["frame"]) for r in images), max(int(r["frame"]) for r in images)
    start, end = attempt.get("first_contact_frame", lo), attempt.get("ending_frame", hi)
    lo, hi = native_context_bounds(label, source_clip, (lo, hi), start, end, data_root)
    timebase, native_time = legacy._native_timebase(label)
    key = row["key"]
    clip = f"{prefix}_{key}"
    frame_manifest, frame_sha = legacy._link_frames(
        clip, label, source_clip, lo, hi, out, data_root, native_time
    )
    empty = {"measurement": {"dense_flights": [], "native_projection": []}}
    players = legacy._sparse_report_players({}, {}, native_time)
    players["source"] = "no fitted output"
    overlay = legacy._video_overlay(label, empty, source_clip, lo, hi)
    overlay["fitted_marker"] = "No fitted output"
    doc = {
        "schema": legacy.SCHEMA,
        "extension_schema": "local_s6_point3d_v1",
        "output_status": "no_fitted_output",
        "speed_caveat": "No fitted speed or spin is available.",
        "match": label.get("match_id", key.split("_pt")[0]),
        "point": key,
        "clip": clip,
        "source_clip": source_clip,
        "tag": run_label,
        "fps": timebase["fps"],
        "native_timebase": timebase,
        "force_follow": True,
        "frames_base": ".",
        "frames_dir": "frames",
        "frames_pattern": "f_%04d" + Path(frame_manifest["frames"][0]["file"]).suffix,
        "frames_available": True,
        "frame_range": [lo, hi],
        "review_context": {
            "kind": "local_s6_observations_only",
            "view_start_t": native_time(lo),
            "view_end_t": native_time(hi),
            "score_start_frame": start,
            "score_end_frame": end,
            "score_start_t": native_time(start),
            "score_end_t": native_time(end),
        },
        "court": legacy.COURT,
        "frames": legacy._candidate_frames(empty, lo, hi, native_time, "native_context"),
        "contacts": [],
        "bounces": [],
        "net_collisions": [],
        "players": players,
        "skeletons": {"near": [], "far": []},
        "labeled_events": [],
        "timeline_ticks": [],
        "video_overlay": overlay,
        "camera": None,
        "per_flight": None,
        "alternates": [],
        "family": {},
        "spin_present": False,
        "quality": {
            "accepted": None,
            "gate_accepted_overall": False,
            "visual_review_status": "no_fitted_output",
            "verdict": "No fitted output — original video available",
            "blocker_text": str(result.get("reason") or result.get("status") or "Unknown failure"),
            "footnote": "Original pictures and supplied observations only. This attempt remains in the failure denominator.",
        },
        "counts": {
            "frames": 0,
            "native_video_frames": frame_manifest["count"],
            "contacts": 0,
            "contacts_fit": 0,
            "bounces": 0,
            "bounces_in_court": 0,
            "net_collisions": 0,
        },
        "provenance": {
            **origin,
            "run_manifest": record(manifest_path),
            "result": record(result_path),
            "input_bindings": {"labels": row["labels"]},
            "frame_manifest_sha256": frame_sha,
            "exporter": record(Path(__file__)),
        },
    }
    filename = f"{prefix}_{key}.json"
    mark_automatic_observations(doc, origin)
    legacy._write_json(out / "data" / filename, doc)
    return {
        **entry,
        "point": key,
        "clip": clip,
        "match": doc["match"],
        "file": filename,
        "frames_available": True,
        "gate_accepted": False,
        "source_group": group,
        "n_frames": 0,
    }


COMPONENT_GROUP = "local_s6_contact_components"


def export_component_source(row, result_path, manifest_path, out, data_root, repo, run_label):
    from cv.pipeline import s6_component_orchestration as components

    result = legacy._read_json(result_path)
    manifest = legacy._read_json(manifest_path)
    components.validate_result(result, row=row, policy=manifest["policy"])
    children = []
    for record in result["component_results"]:
        child_manifest_path = legacy._verify_binding(record["manifest"], data_root, repo)
        child_manifest = legacy._read_json(child_manifest_path)
        child_result_path = legacy._verify_binding(record["result_binding"], data_root, repo)
        child = export_case(
            child_manifest["rows"][0],
            child_result_path,
            child_manifest_path,
            out,
            data_root,
            repo,
            run_label,
        )
        child.update(
            parent_source_key=row["key"],
            source_group=COMPONENT_GROUP,
            component_index=record["component_index"],
            local_to_original=record["local_to_original"],
            count_in_source_denominator=False,
        )
        for name in ("accepted_flights", "flight_count", "declared_flights"):
            if name in child:
                child["component_" + name] = child.pop(name)
        children.append(child)
    # Source video/roster and each component use separate geometry arrays. No
    # parent trajectory or interpolation connects the independent parameter sets.
    entry = export_observation_only(
        row, result_path, manifest_path, out, data_root, repo, run_label
    )
    if entry.get("file"):
        document_path = out / "data" / entry["file"]
        document = legacy._read_json(document_path)
        document.update(
            component_scenes=children,
            original_source_verdict=result["verdict"],
            original_source_plan=result["source_plan"],
            complete_original_source=False,
            continuity_between_components=False,
        )
        legacy._write_json(document_path, document)
    entry.update(
        held=False,
        component_entries=children,
        complete_original_source=False,
        accepted_flights=result["verdict"]["accepted_flight_count"],
        flight_count=result["source_plan"]["original_slot_count"],
        fitted_scene_flights=result["verdict"]["flight_count"],
        slot_count_semantics=result["source_plan"]["slot_count_semantics"],
        label=f"{row['key']} · source overview; independent contact components",
    )
    return entry


def export_case(
    row: dict,
    result_path: Path,
    manifest_path: Path,
    out: Path,
    data_root: Path,
    repo: Path,
    run_label: str,
) -> dict:
    origin = checked_export_origin(row)
    prefix, group = export_namespace(origin)
    result = legacy._read_json(result_path)
    key = row["key"]
    if result.get("key") != key:
        raise ValueError("result key differs from frozen row")
    result_binding = record(result_path)
    from cv.pipeline import s6_component_orchestration

    if result.get("schema") == s6_component_orchestration.SCHEMA:
        return export_component_source(
            row, result_path, manifest_path, out, data_root, repo, run_label
        )
    if not result.get("measurement"):
        return export_observation_only(
            row, result_path, manifest_path, out, data_root, repo, run_label
        )
    if origin.get("observation_origin") == "automatic" and (
        result.get("observation_origin") != "automatic"
        or result.get("automatic_provenance") != row["automatic_provenance"]
        or result.get("human_derived_inputs") is not False
        or result.get("upstream_automatic_inference_eligible") is not True
    ):
        raise ValueError("automatic stage result differs from validated observation origin")
    search_path = result_path.parent / "search/report.json"
    invocation_path = result_path.parent / "invocation.json"
    search = legacy._read_json(search_path)
    invocation = legacy._read_json(invocation_path)
    manifest = legacy._read_json(manifest_path)
    if result.get("contact_components", {}).get("status") == "not_applicable_source_scope":
        from cv.pipeline import s6_contact_prefix_scope

        source_packet = legacy._read_json(legacy._verify_binding(row["packet"], data_root, repo))
        attempt = source_packet["attempts"][0]
        declaration = attempt.get(
            "original_observation_scope", attempt.get("observation_scope", {})
        )
        if (
            manifest["policy"].get("contact_components") != "unresolved_ending"
            or declaration.get("schema") == s6_contact_prefix_scope.UNRESOLVED_INPUT_SCHEMA
        ):
            raise ValueError("component policy passthrough differs from original source scope")
        manifest = manifest | {"policy": manifest["policy"] | {"contact_components": "off"}}
    if invocation["inputs"] != row or invocation["policy"] != manifest["policy"]:
        raise ValueError("invocation differs from frozen run inputs or policy")
    net_policy = checked_net_policy(result, search, invocation, manifest)
    if search.get("configuration", {}).get("interior_contact_epochs", "off") != manifest[
        "policy"
    ].get("interior_contact_epochs", "off"):
        raise ValueError("contact timing search and run policies differ")
    executed_bindings = checked_input_bindings(
        row, search, invocation, result_path, data_root, repo
    )
    # A qualified prefix scope abstains from terminal/topology mechanisms. The
    # requested policy stays recorded; readers must see what actually ran. The
    # replay above already rebuilt this applicability from the original inputs.
    applicability = invocation.get("scope_applicability") or {}
    effective_policy = manifest["policy"] | {
        name: change["applied"] for name, change in applicability.get("changes", {}).items()
    }
    role_binding = checked_first_contact_role(result, search)
    if role_binding is not None:
        effective_policy |= {
            name: change["applied"]
            for name, change in invocation["preparation"]["first_contact_role"]["changes"].items()
        }
    checked_observation_operator(
        result,
        search,
        legacy._read_json(legacy._binding_path(executed_bindings["packet"], data_root, repo)),
    )
    checked_observation_scope(
        result,
        search,
        legacy._read_json(legacy._binding_path(executed_bindings["packet"], data_root, repo)),
        legacy._read_json(legacy._binding_path(executed_bindings["labels"], data_root, repo)),
    )
    from cv.experiments.connected_shooting import net_collision, passive_bounce

    passive_policy = checked_passive_bounce_policy(result, search, invocation, manifest)
    with (
        net_collision.physical_eligibility(net_policy == "physical_mesh_v1"),
        passive_bounce.using(passive_policy),
    ):
        original_context = original_candidate_context(
            row, search, executed_bindings, effective_policy, data_root, repo
        )
        optional_checked = False
        if search.get("optional_contact_selection") is not None:
            checked_optional_contact_source(
                search,
                legacy._read_json(
                    legacy._binding_path(executed_bindings["packet"], data_root, repo)
                ),
                legacy._read_json(
                    legacy._binding_path(executed_bindings["cameras"], data_root, repo)
                ),
                manifest["policy"],
                original_context,
                automatic=origin.get("observation_origin") == "automatic",
                data_root=data_root,
                repo=repo,
            )
            optional_checked = True
        candidate = checked_candidate(
            result,
            search,
            original_context=original_context,
            optional_source_checked=optional_checked,
        )
    if "initial_search" in result:
        selected_search = result["selected_by"]["selected_search"]
        if (
            legacy._verify_binding(selected_search, data_root, repo).resolve()
            != search_path.resolve()
        ):
            raise ValueError("refinement is bound to a different invocation search")
        if result["final_state_binding"]["selected_search"] != selected_search:
            raise ValueError("final state is bound to a different initial search")
    label_path = legacy._binding_path(row["labels"], data_root, repo)
    label = legacy._read_json(label_path)
    from cv.pipeline import s6_terminal_context

    s6_terminal_context.validate_result(result, label, search, effective_policy)
    source_clip = search["clip"]
    timebase, native_time = legacy._native_timebase(label)
    images = [r for r in label["source_pack"]["images"] if r["clip"] == source_clip]
    lo, hi = min(int(r["frame"]) for r in images), max(int(r["frame"]) for r in images)
    scope = competitive_context(
        result,
        native_time,
        lo,
        hi,
        strict_native_clock=(
            legacy._declared_native_clock(label) or legacy._explicit_native_pts(label)
        ),
    )
    lo, hi = native_context_bounds(
        label,
        source_clip,
        (lo, hi),
        scope["score_start_frame"],
        scope["score_end_frame"],
        data_root,
    )
    # Distinct frame namespace avoids replacing historical evidence for the same point.
    display_clip = f"{prefix}_{key}"
    frame_manifest, frame_sha = legacy._link_frames(
        display_clip, label, source_clip, lo, hi, out, data_root, native_time
    )
    verdict = result["verdict"]
    if len(candidate["measurement"]["dense_flights"]) != verdict["flight_count"]:
        raise ValueError("scored flight count differs from exported trajectory")
    component_scope = contact_component_scope(result)
    prefix_scope = contact_prefix_scope(result)
    prefix_receipt = result.get("preparation", {}).get("contact_prefix_scope") or {}
    if prefix_scope is not None and (
        prefix_receipt.get("status") != "qualified"
        or prefix_receipt.get("contract", prefix_scope) != prefix_scope
    ):
        raise ValueError("displayed contact prefix requires its qualified preparation receipt")
    per_flight = {**verdict, "rung": result["scorer"]["rung"]}
    frames = legacy._candidate_frames(candidate, lo, hi, native_time, "cold_s6_fitted_state")
    legacy._mark_frame_acceptance(frames, per_flight)
    context = competitive_context(
        result,
        native_time,
        lo,
        hi,
        strict_native_clock=(
            legacy._declared_native_clock(label) or legacy._explicit_native_pts(label)
        ),
    )
    mark_context_frames(frames, context)
    contacts = legacy._contacts(
        candidate,
        search,
        native_time,
        # k measured flights, k+1 original contacts: emit the last original
        # contact from measured incoming geometry, with no outgoing velocity.
        **(
            {
                "right_boundary": [
                    e
                    for e in component_scope["component"]["events"]
                    if e["event_type"] == "contact"
                ][-1]
            }
            if component_scope is not None
            # A ball-track terminal closes the component without a contact.
            and not component_scope.get("terminal_track_endpoint")
            else {}
            if component_scope is not None
            else {"right_boundary": prefix_scope["right_contact"]}
            if prefix_scope
            else {}
        ),
    )
    contacts, per_flight = display_contact_roles(result, contacts, per_flight)
    # Keep physical bounces distinct from labels rather than imply ordinal correspondence
    # when a rejected trajectory contains extra impacts.
    bounces = legacy._bounces(candidate, {**search, "events": []}, native_time)
    nets = legacy._net_collisions(candidate, native_time)
    camera, camera_receipt, _ = legacy._camera_summary(search, data_root, repo)
    players = legacy._sparse_report_players(search, {}, native_time)
    for side in ("near", "far"):
        players["names"][side] = next(
            (p.get("player") for p in search.get("player_states", []) if p.get("side") == side),
            None,
        )
    overlay = legacy._video_overlay(label, candidate, source_clip, lo, hi)
    overlay["fitted_marker"] = "yellow: stage exposure-model projection of the selected trajectory"
    accepted = verdict["accepted_flight_count"]
    count = verdict["flight_count"]
    status = f"Local S6: {accepted}/{count} flights pass gates; visual review pending"
    if component_scope is not None:
        status = (
            f"Local S6 component: {accepted}/{count} flights pass gates; original source incomplete"
        )
    if prefix_scope is not None:
        status = (
            f"Local S6 contact prefix: {accepted}/{count} retained flights pass gates; "
            f"{prefix_scope['unresolved_original_flight_count']} original source flights "
            "unresolved; no point ending; visual review pending"
        )
    match = label.get("match_id") or search["attempt_id"].split("__", 1)[-1].split("_pt", 1)[0]
    doc = {
        "schema": legacy.SCHEMA,
        "extension_schema": "local_s6_point3d_v1",
        "match": match,
        "point": key,
        "clip": display_clip,
        "source_clip": source_clip,
        "tag": run_label,
        "fps": timebase["fps"],
        "native_timebase": timebase,
        "force_follow": True,
        "frames_base": ".",
        "frames_dir": "frames",
        "frames_pattern": "f_%04d" + Path(frame_manifest["frames"][0]["file"]).suffix,
        "frames_available": True,
        "frame_range": [lo, hi],
        "review_context": context,
        **(
            {"contact_prefix_coverage": contact_prefix_coverage(prefix_scope, count)}
            if prefix_scope is not None
            else {}
        ),
        **({"contact_component_scope": component_scope} if component_scope is not None else {}),
        "court": legacy.COURT,
        "frames": frames,
        "contacts": contacts,
        "bounces": bounces,
        "net_collisions": nets,
        "players": players,
        "skeletons": {"near": [], "far": []},
        "labeled_events": search.get("events", []),
        "timeline_ticks": legacy._timeline_ticks(search, contacts, native_time),
        "video_overlay": overlay,
        "camera": camera,
        "per_flight": per_flight,
        "alternates": [],
        "family": {},
        "spin_present": False,
        "quality": {
            # Whole-point correctness stays unknown until visual review; the
            # per-flight column and gate field retain exact numerical verdicts.
            "accepted": None,
            "gate_accepted_overall": bool(verdict["complete_point"]),
            "visual_review_status": "unreviewed",
            "incorrect_accept_count": None,
            "verdict": status,
            "blocker_text": "; ".join(verdict.get("failure_counts", {})),
            "footnote": "Gate acceptance is not audited correctness. No independent airborne XYZ truth. "
            "Grey player positions are sparse supplied player states, not reconstructed skeletons. "
            "Grey trajectory continuation is passive context, outside competitive acceptance."
            + (
                ""
                if prefix_scope is None
                else " This is an original contact-to-contact prefix: the last contact closes "
                "the modeled scene, its outgoing flight was never modeled, and the original "
                "source stays incomplete."
            ),
            "trajectory_role": "cold_input_objective_selected",
        },
        "counts": {
            "frames": sum(x is not None for x in frames["x"]),
            "native_video_frames": frame_manifest["count"],
            "contacts": len(contacts),
            "contacts_fit": len(contacts),
            "bounces": len(bounces),
            "bounces_in_court": sum(bool(b["in_court"]) for b in bounces),
            "net_collisions": len(nets),
        },
        "confidence_status": "Fitted states at native exposures; confidence and spin remain uncalibrated.",
        "speed_caveat": "Speed is derived from the fitted physics, not independent measurement.",
        "provenance": {
            **origin,
            "runtime_model_calls": result.get("runtime_model_calls"),
            "cold_search": True,
            "net_collision_policy": net_policy,
            **({"passive_bounce_response": passive_policy} if passive_policy != "off" else {}),
            **(
                {
                    "contact_prefix_scope": {
                        name: prefix_receipt.get(name)
                        for name in (
                            "status",
                            "mode",
                            "observation_partition",
                            "observation_fallback",
                            "first_coverage_failure",
                        )
                    }
                    | {"sha256": value_sha256(prefix_receipt)},
                    "contact_prefix_applicability": result["preparation"].get(
                        "scope_applicability"
                    ),
                }
                if prefix_scope is not None
                else {}
            ),
            "run_manifest": record(manifest_path),
            "result": result_binding,
            "search_report": record(search_path),
            "invocation": record(invocation_path),
            "selected_by": result["selected_by"],
            "input_bindings": {
                name: row[name] for name in ("labels", "packet", "cameras", "pose_csv")
            },
            "executed_input_bindings": executed_bindings,
            "final_state_binding": result.get("final_state_binding"),
            "refinement": None
            if "refinement" not in result
            else {
                "status": result["refinement"]["status"],
                "applied_mechanisms": result["refinement"].get("applied_mechanisms", []),
                "policy": result["refinement"].get("policy"),
                "sha256": value_sha256(result["refinement"]),
            },
            **(
                {
                    "interior_contact_epoch_fit": result["evaluation_context"][
                        "interior_contact_epoch_fit"
                    ]
                }
                if result.get("evaluation_context", {}).get("interior_contact_epoch_fit")
                is not None
                else {}
            ),
            "net_response": result.get("net_response"),
            "ground_response": result.get("ground_response"),
            **(
                {"terminal_context_support": result["context"]}
                if result.get("evaluation_context", {}).get("terminal_context_support") is not None
                else {}
            ),
            "camera": camera_receipt,
            "frame_manifest_sha256": frame_sha,
            "measurement_sha256": hashlib.sha256(
                json.dumps(result["measurement"], sort_keys=True).encode()
            ).hexdigest(),
            "exporter": record(Path(__file__)),
        },
    }
    filename = f"{prefix}_{key}.json"
    mark_automatic_observations(doc, origin)
    legacy._write_json(out / "data" / filename, doc)
    return {
        "key": key,
        "point": key,
        "clip": display_clip,
        "match": match,
        "file": filename,
        "tag": run_label,
        "label": f"{key} · {status}",
        "source_group": group,
        "accepted": False,
        "gate_accepted": bool(verdict["complete_point"]),
        "accepted_flights": accepted,
        "flight_count": count,
        # Fixed original denominator; a prefix never shrinks or pads it.
        "declared_flights": row["declared_flights"],
        **(
            {
                "right_boundary_kind": "original_contact",
                "complete_original_source": False,
                "unresolved_original_flights": prefix_scope["unresolved_original_flight_count"],
            }
            if prefix_scope is not None
            else {}
        ),
        "n_frames": doc["counts"]["frames"],
        "n_contacts": len(contacts),
        "n_bounces": len(bounces),
        "frames_available": True,
        "visual_review_status": "unreviewed",
        "verdict": status,
    }


def export_run(
    run: Path,
    out: Path,
    cases: list[str] | None = None,
    merge: bool = False,
    run_label: str | None = None,
) -> dict:
    repo = repository_root()
    root = resolve_shared_root(repo, None)
    if out.resolve() == (root / "processed/review_queue_v1/portal/3d").resolve():
        raise ValueError("export to staging; coordinator reviews before deployment")
    if (run / "summary.json").is_file() and legacy._read_json(run / "summary.json").get(
        "schema"
    ) == "automatic_broadcast_shared_s6_v1":
        from cv.viz.export_automatic_s6 import export_run as export_automatic

        return export_automatic(run, out, cases, merge, run_label)
    manifest_path = run / "manifest.json"
    manifest = legacy._read_json(manifest_path)
    if manifest.get("schema") != "labeled_s6_observations_v1":
        raise ValueError("frozen labeled S6 observations manifest required")
    rows = manifest["rows"]
    requested = set(cases or [])
    if requested - {r["key"] for r in rows}:
        raise ValueError("unknown case")
    if any(Path(r["key"]).name != r["key"] or r["key"] in (".", "..") for r in rows):
        raise ValueError("unsafe case key")
    if out.exists() and not merge:
        raise ValueError("fresh staging directory required unless --merge is explicit")
    prior = (
        legacy._read_json(out / "data/index.json")
        if merge and (out / "data/index.json").exists()
        else {}
    )
    copy_static(out)
    entries, held = [], []
    for row in rows:
        if requested and row["key"] not in requested:
            continue
        result = run / "cases" / row["key"] / "result.json"
        if not result.exists():
            held.append(
                {
                    "key": row["key"],
                    "status": "pending",
                    "declared_flights": row["declared_flights"],
                }
            )
            continue
        entry = export_case(row, result, manifest_path, out, root, repo, run_label or run.name)
        (held if entry.get("held") or not entry.get("file") else entries).append(entry)
        # Component documents are bound views inside their original source, not
        # extra attempts in the picker or source denominator.
    replaced = {
        f"local_s6_{row['key']}.json" for row in rows if not requested or row["key"] in requested
    }
    selected_source_keys = {r["key"] for r in rows if not requested or r["key"] in requested}
    old = [
        p
        for p in prior.get("points", [])
        if p.get("file") not in replaced and p.get("parent_source_key") not in selected_source_keys
    ]
    replaced_keys = {row["key"] for row in rows if not requested or row["key"] in requested}
    prior_held = [
        r
        for r in prior.get("local_s6", {}).get("held_before_fitting", [])
        if r["key"] not in replaced_keys
        and r.get("parent_source_key") not in replaced_keys
        and r.get("source_group", GROUP) in (GROUP, COMPONENT_GROUP)
    ]
    held = [*prior_held, *held]
    groups = [g for g in prior.get("source_groups", []) if g["id"] != GROUP]
    if any(p.get("source_group") == COMPONENT_GROUP for p in [*entries, *old]):
        groups = [g for g in groups if g["id"] != COMPONENT_GROUP]
        groups.append({"id": COMPONENT_GROUP, "label": "Independent source contact components"})
    index = {
        **prior,
        "schema": legacy.INDEX_SCHEMA,
        "collection": "canonical_3d_viewer",
        "source_groups": [{"id": GROUP, "label": GROUP_LABEL}, *groups],
        "points": [*entries, *old],
        "count": len(entries) + len(old),
        "local_s6": {
            "run": record(manifest_path),
            "held_before_fitting": held,
            "exported_attempts": sum(p.get("source_group") == GROUP for p in [*entries, *old]),
            "requested_attempts": sum(p.get("source_group") == GROUP for p in [*entries, *old])
            + sum(r.get("source_group", GROUP) != COMPONENT_GROUP for r in held),
            "scope": run_label or run.name,
        },
    }
    legacy._write_json(out / "data/index.json", index, pretty=True)
    legacy._write_json(out / "local_s6_export.json", index["local_s6"], pretty=True)
    return index["local_s6"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--run-label", help="Explicit scope, e.g. obsolete-policy protocol smoke")
    args = parser.parse_args()
    print(
        json.dumps(export_run(args.run, args.out, args.case, args.merge, args.run_label), indent=2)
    )


if __name__ == "__main__":
    main()
