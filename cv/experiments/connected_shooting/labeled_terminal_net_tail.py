"""Opt-in original-interval net ending with a fully observed physical aftermath.

No ground epoch is supplied for the terminal interaction. Ground impacts inside
its uncertain interval/aftermath remain simulated latent events, with unchanged
passivity and native image checks. The net-present branch is an input-qualified
hypothesis, not promotion of an ambiguous annotation to an observed occurrence.
"""

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence
from cv.experiments.connected_shooting import observation_operator

from copy import deepcopy
import hashlib
import json

import numpy as np

KIND = "net_stop"
# Event simulation/root replay can differ by tens of nanoframes at a boundary.
# One microframe (about 20 ns at 50 fps) is numerical comparison tolerance;
# original intervals and raw residuals remain unchanged.
EVENT_TOLERANCE_FRAMES = 1e-6


def domain_admissible(slacks):
    values = np.asarray(slacks, float)
    return bool(np.isfinite(values).all() and np.all(values >= -EVENT_TOLERANCE_FRAMES))


def source_occurrence_support(event, ending, labels):
    """Distinguish source-supported occurrence from timing/order uncertainty."""
    if event.get("occurrence_status") in ("absent", "unsupported", "ambiguous"):
        return "none"
    if occurrence.predicted_membership(event):
        return "source_predicted"
    if event.get("status") == "labeled":
        return "source_exact"
    from cv.experiments.connected_shooting.labeled_event_occurrence import timing_admission

    if timing_admission(event) is not None:
        return "source_reviewed"
    if (
        ending.get("status") == "labeled"
        and "net" in ending.get("ending_kind", "")
        and ending.get("frame_interval") == event.get("frame_interval")
        and labels.get("attempt", {}).get("topology", {}).get("net_contact_timing_ambiguous")
        is True
        and ending.get("occurrence_status") not in ("absent", "unsupported", "ambiguous")
    ):
        return "source_ending"
    return "none"


def qualify(packet, labels):
    """Extend only the observation horizon; preserve all original event records."""
    from cv.experiments.connected_shooting.event_recovery import extend_attempt_window

    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one original attempt required")
    attempt = packet["attempts"][0]
    clip = attempt["point_clip"]
    records = [e for e in labels["events"]["records"] if e.get("clip", clip) == clip]
    contacts = [float(e["frame"]) for e in attempt["events"] if e["event_type"] == "contact"]
    if not contacts:
        raise ValueError("original contact required")
    end = float(attempt["owner_end_frame"])
    endings = [e for e in records if e["event_type"] == "ending" and float(e["frame"]) == end]
    nets = [
        e
        for e in records
        if e["event_type"] == "net_hit"
        and float(e["frame"]) == end
        and (e.get("status") in ("labeled", "ambiguous") or occurrence.predicted_membership(e))
        and e.get("occurrence_status") not in ("absent", "unsupported")
    ]
    if len(endings) != 1 or len(nets) != 1 or "net" not in endings[0].get("ending_kind", ""):
        raise ValueError("one original ending-coincident net interaction required")
    event, ending = nets[0], endings[0]
    low, high = map(float, event["frame_interval"])
    elow, ehigh = map(float, ending["frame_interval"])
    if (
        not np.isfinite([low, high, elow, ehigh, end]).all()
        or not max(contacts) < low <= end <= high
        or (low, high) != (elow, ehigh)
    ):
        raise ValueError("ordered original matching net/ending intervals required")
    if any(
        e["event_type"] == "bounce" and float(e["frame"]) > max(contacts) for e in attempt["events"]
    ):
        raise ValueError("existing supplied terminal-ground path remains authoritative")
    visible = [
        int(row["frame"])
        for record in labels["ball"]["records"]
        if record["clip"] == clip
        for row in record["frames"]
        if row["status"] == "visible"
    ]
    from cv.experiments.connected_shooting import real_exposure_replay as exposure

    duration = observation_operator.from_packet(packet)["exposure_duration_frames"]
    inventory = exposure.before_next_physical_event(
        {"postbounce_labeled_frames": sorted(set(f for f in visible if f > high))},
        labels,
        clip,
        end,
        duration,
        event_types=("contact",),
    )
    after = inventory["postbounce_labeled_frames"]
    if len(after) < 2:
        raise ValueError("two original post-interval visible exposures required")
    horizon = float(after[-1]) + observation_operator.support_span(duration)
    prepared = deepcopy(packet)
    active = extend_attempt_window(prepared["attempts"][0], labels["ball"]["records"], horizon)
    # Window helper changes owner_end_frame; keep the original competitive endpoint.
    active["owner_end_frame"] = attempt["owner_end_frame"]
    raw_hash = hashlib.sha256(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    hypothesis = {
        **deepcopy(event),
        "hypothesis_presence": "net_present",
        "exact_epoch_observed": False,
        "original_event_sha256": raw_hash,
    }
    # Earlier racket-to-racket flights may contain their own net interactions.
    # Only the final flight can conflict with this terminal interaction contract.
    existing_nets = [
        e
        for e in active["events"]
        if e["event_type"] == "net_hit" and float(e["frame"]) > max(contacts)
    ]
    if existing_nets and (
        len(existing_nets) != 1
        or float(existing_nets[0]["frame"]) != end
        or list(existing_nets[0]["frame_interval"]) != [low, high]
    ):
        raise ValueError("original terminal net interval conflicts with packet inventory")
    # Dispatch remains separate from the original observed event inventory.
    active["terminal_net_hypothesis_event"] = hypothesis
    contract = dict(
        kind=KIND,
        interval=[low, high],
        representative=end,
        observation_horizon=horizon,
        native_aftermath_frames=after,
        continuity_scope=inventory.get("next_contact_boundary"),
        original_net_event=deepcopy(event),
        original_ending=deepcopy(ending),
        original_event_sha256=raw_hash,
        hypothesis="net_present",
        occurrence_confirmed=source_occurrence_support(event, ending, labels)
        not in ("none", "source_predicted"),
        occurrence_support=source_occurrence_support(event, ending, labels),
        latent_ground_epochs_supplied=False,
        original_competitive_end_frame=end,
    )
    if contract["occurrence_support"] == "source_predicted":
        contract["predicted_occurrence_admitted"] = True
    active["terminal_net_tail"] = contract
    prepared["attempts"][0] = active
    return prepared, contract


def interval(scene):
    value = getattr(scene, "terminal_net_tail", None)
    return None if value is None else tuple(map(float, value["interval"]))


def event_slacks_frames(scene, flights, *, interior=0.0):
    """Original input-domain inequalities, independent of image or scoring gates."""
    bounds = interval(scene)
    if bounds is None:
        return np.empty(0)
    if not np.isfinite(interior) or interior < 0:
        raise ValueError("nonnegative finite numerical interval margin required")
    if len(flights) != len(scene.pixels):
        raise ValueError("complete original flight inventory required")
    hits = flights[-1].get("net_hits", [])
    if len(hits) != 1 or not np.isfinite(hits[0]["frame"]):
        return np.full(2, -1e6)
    epoch = float(hits[0]["frame"])
    return np.array([epoch - bounds[0] - interior, bounds[1] - epoch - interior])


def original_domain_slacks_frames(scene, flights):
    """Keep unsupplied competitive grounds outside the declared interaction window.

    Qualification requires zero supplied grounds after the final contact. A
    latent ground may occur anywhere in the original interaction/aftermath,
    including before the net; this does not invent a ground epoch or occurrence.
    One fixed extra inequality also covers zero or multiple passive impacts.
    """
    net_slacks = event_slacks_frames(scene, flights)
    bounds = interval(scene)
    if bounds is None:
        return net_slacks
    ground_frames = np.asarray([hit["frame"] for hit in flights[-1].get("bounces", [])], float)
    if not np.isfinite(ground_frames).all():
        ground_slack = -1e6
    else:
        ground_slack = float(ground_frames.min()) - bounds[0] if ground_frames.size else 1.0
    return np.r_[net_slacks, ground_slack]


def require_event_domain(scene, flights):
    """Strict unforced replay; missing or out-of-interval hits cannot be incumbents."""
    slacks = original_domain_slacks_frames(scene, flights)
    if not domain_admissible(slacks):
        raise ValueError(f"physical interaction outside original input interval: {slacks.tolist()}")
    return slacks


def mesh_response_slacks_m(scene, flights, *, include_supplied_final=False):
    """Require nonpenetration only for an unambiguous below-cord mesh hit.

    Within a ball radius plus 3 cm geometric uncertainty of the top edge,
    forward/tape responses remain unconstrained. Else the outgoing normal
    component must point back to the incoming side. Tangential and vertical
    components are unrestricted. Both alternatives are expressed in metres;
    the velocity term is its displacement over one native frame.
    """
    if interval(scene) is None:
        from cv.experiments.connected_shooting.observation_net_seed import single_final_net

        if not include_supplied_final or not single_final_net(scene):
            return np.empty(0)
    hits = flights[-1].get("net_hits", [])
    if len(hits) != 1:
        return np.array([-1e6])
    hit = hits[0]
    from cv.experiments.connected_shooting import model

    edge_margin = float(hit["x"][2]) + model.R_BALL + 0.03 - float(hit["tape_height_m"])
    incoming, outgoing = float(hit["v_in"][1]), float(hit["v_out"][1])
    normal_displacement = -np.sign(incoming) * outgoing / scene.fps
    return np.array([max(edge_margin, normal_displacement)])


def require_mesh_response(scene, flights, *, include_supplied_final=False):
    slacks = mesh_response_slacks_m(scene, flights, include_supplied_final=include_supplied_final)
    if not np.isfinite(slacks).all() or np.any(slacks < -1e-6):
        raise ValueError("below-cord net response passes through mesh")
    return slacks


def collision_residuals(scene, flight_index, hits, supplied_frame):
    """Keep physical residuals; qualified intervals do not create exact timestamps."""
    from cv.experiments.connected_shooting import net_collision

    residuals = net_collision.residuals(hits, supplied_frame)
    bounds = interval(scene)
    if bounds is not None and flight_index == len(scene.pixels) - 1 and len(hits) == 1:
        epoch = float(hits[0]["frame"])
        residuals[-1] = (min(epoch - bounds[0], 0.0) + max(epoch - bounds[1], 0.0)) / 0.25
    return residuals


def competitive_grounds(scene, flight_index, impacts):
    bounds = interval(scene)
    if bounds is None or flight_index != len(scene.pixels) - 1:
        return impacts
    return [hit for hit in impacts if float(hit["frame"]) < bounds[0] - EVENT_TOLERANCE_FRAMES]


def completion(scene, parameters, native, *, duration: float | None = 0.25):
    """Certify full physical response and native horizon, never a fake ground end."""
    from cv.experiments.connected_shooting import model

    bounds = interval(scene)
    if bounds is None:
        raise ValueError("original terminal interaction contract required")
    flights = model.chain(scene, parameters)
    fit = flights[-1]
    hits = fit.get("net_hits", [])
    checks = dict(
        one_physical_net=len(hits) == 1,
        native_tail_retained=max(native) + observation_operator.support_span(duration)
        <= float(scene.contact_frames[-1]) + 1e-8,
        original_aftermath_retained=set(
            scene.terminal_net_tail["native_aftermath_frames"]
        ).issubset(set(native)),
    )
    if len(hits) == 1:
        hit = hits[0]
        xyz = np.asarray(hit["x"])
        checks.update(
            net_inside_original_interval=domain_admissible(event_slacks_frames(scene, flights)),
            net_position_continuous=bool(hit["position_continuous"]),
            net_plane=abs(float(xyz[1]) - 11.885) <= 0.05,
            net_geometry=-0.915 <= xyz[0] <= 11.885
            and model.R_BALL - 1e-3 <= xyz[2] <= float(hit["tape_height_m"]) + model.R_BALL + 0.03,
        )
    valid = all(checks.values())
    return dict(
        status="completed" if valid else "held",
        reason=None if valid else "terminal_net_tail_not_supported",
        kind=KIND,
        checks=checks,
        valid=valid,
        end_frame=float(scene.contact_frames[-1]),
        end_xyz=np.asarray(fit["end_xyz"]).tolist(),
        parameters_changed=False,
        observations_discarded=0,
        postimpact_response_reconstructed=True,
        original_contract=scene.terminal_net_tail,
        net_epoch_bound_limited=bool(
            len(hits) == 1 and min(event_slacks_frames(scene, flights)) <= 1e-3
        ),
        exact_net_epoch_observed=False,
        latent_ground_frames=[
            float(hit["frame"])
            for hit in fit["bounces"]
            if float(hit["frame"]) >= bounds[0] - EVENT_TOLERANCE_FRAMES
        ],
    )


def terminal_slack(scene, parameters, native_last, *, cache=None, duration: float | None = 0.25):
    """Original interval feasibility, never proximity to the last video picture."""
    from cv.experiments.connected_shooting import model

    flights = model.chain(scene, parameters, simulation_cache=cache)
    slacks = original_domain_slacks_frames(scene, flights) + EVENT_TOLERANCE_FRAMES
    if len(flights[-1].get("net_hits", [])) != 1:
        return np.full(3, -1e6)
    horizon_slack = (
        1.0
        if scene.contact_frames[-1] >= native_last + observation_operator.support_span(duration)
        else -1e6
    )
    return np.r_[min(horizon_slack, slacks[-1]), slacks[:2]]


def mark_conditional(verdict, contract):
    """Keep conditional geometry passes distinct from unconditional stage yield."""
    if contract is None:
        return verdict
    support = contract.get(
        "occurrence_support",
        "source_exact" if contract["original_net_event"].get("status") == "labeled" else "none",
    )
    if support != "none":
        result = deepcopy(verdict)
        result["terminal_interaction_assumption"] = dict(
            net_present="model_predicted" if support == "source_predicted" else "source_supported",
            occurrence=(
                "model_predicted"
                if support == "source_predicted"
                else "source_supported_timing_abstained"
            ),
            occurrence_support=support,
            net_absent="not_evaluated",
            exact_epoch_observed=False,
            ground_net_order="not_observed",
            original_net_event_sha256=contract["original_event_sha256"],
        )
        return result
    result = deepcopy(verdict)
    result["terminal_interaction_assumption"] = dict(
        net_present="assumed",
        occurrence="unresolved",
        net_absent="not_evaluated",
        conditional_complete_point=bool(verdict["complete_point"]),
        conditional_accepted_flight_count=verdict["accepted_flight_count"],
        original_net_event_sha256=contract["original_event_sha256"],
    )
    final = result["flights"][-1]
    final["checks"]["terminal_event_occurrence_resolved"] = False
    final["failures"] = [*final["failures"], "terminal_event_occurrence_resolved"]
    final["accepted"] = False
    result["complete_point"] = False
    accepted = [i for i, f in enumerate(result["flights"]) if f["accepted"]]
    result["accepted_flight_count"] = len(accepted)
    result["accepted_flight_indices"] = accepted
    result["rejected_flight_indices"] = [
        i for i in range(len(result["flights"])) if i not in accepted
    ]
    result["partial_point"] = bool(accepted)
    result["failure_counts"] = {
        name: sum(name in f["failures"] for f in result["flights"])
        for name in sorted({name for f in result["flights"] for name in f["failures"]})
    }
    gaps = []
    for i, flight in enumerate(result["flights"]):
        if flight["accepted"]:
            continue
        if gaps and gaps[-1]["last_flight_index"] == i - 1:
            gaps[-1].update(last_flight_index=i, end_frame=flight.get("end_frame"))
            gaps[-1]["failures"] = sorted(set(gaps[-1]["failures"] + flight["failures"]))
        else:
            gaps.append(
                dict(
                    first_flight_index=i,
                    last_flight_index=i,
                    start_frame=flight.get("start_frame"),
                    end_frame=flight.get("end_frame"),
                    failures=list(flight["failures"]),
                )
            )
    result["gaps"] = gaps
    return result


def candidate_input_rank(candidate):
    """Prefer an original-domain eligible candidate, then its unchanged input cost.

    An ineligible search proposal can still seed feasibility recovery when no
    eligible proposal exists. It cannot displace a valid sibling or become the
    final stage source. Ordinary scenes have no new receipt and retain ordering.
    """
    admission = candidate["measurement"].get("input_event_domain")
    invalid = admission is not None and not admission["admissible"]
    return invalid, candidate["evidence"]["input_only_rank_score"]


def retain_input_domain_incumbent(candidate, incumbent):
    """Do not discard a valid coarse state for an out-of-domain refinement."""
    admission = incumbent["measurement"].get("input_event_domain")
    if admission is None or not admission["admissible"]:
        return candidate
    proposed = (
        dict(admissible=False, reason="refinement_failed_numerically")
        if candidate is None
        else candidate["measurement"].get("input_event_domain")
    )
    if proposed is not None and not proposed["admissible"]:
        retained = deepcopy(incumbent)
        retained["input_domain_refinement_retention"] = dict(
            rule="retain original-domain incumbent after ineligible refinement",
            rejected_proposal=deepcopy(proposed),
            original_stage=incumbent["stage"],
            parameters_changed=False,
            selection_reads_acceptance=False,
        )
        return retained
    return candidate
