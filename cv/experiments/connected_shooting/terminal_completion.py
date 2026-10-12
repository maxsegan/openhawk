"""Can a frozen connected trajectory end at a nearby physical ground impact?

Research only. No fitting, coordinate snapping, event insertion or image removal.
The caller supplies an ending kind/time and uncertainty, not a true impact position.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cv.experiments.connected_shooting import model, terminal_exposure_context
from cv.pipeline.trajectory_contract import TIME_NUMERICAL_TOLERANCE_FRAMES

ORIGINAL_CONTACT_SCHEMA = "connected_original_contact_boundary_v1"
ORIGINAL_CONTACT_REASON = "original_contact_right_boundary_not_a_physical_ending"


def original_contact_receipt(scene: model.Scene, parameters: np.ndarray) -> dict:
    """Report the unchanged endpoint of a contact-to-contact prefix.

    The last boundary is the supplied contact k. Nothing is completed, moved or
    classified: the receipt carries the frozen endpoint, its incoming state and
    the venue-envelope gate on that contact. A missing right actor is an absent
    soft prior; the envelope gate does not depend on one.
    """
    from cv.experiments.connected_shooting import event_constraints

    if not model.original_contact_boundary(scene):
        raise ValueError("original-contact receipt requires an original-contact right boundary")
    times = np.asarray(scene.contact_frames, float)
    supplied_end = float(times[-1])
    # Deliberately no ``end_frame``/``end_xyz``: consumers that read those keys
    # as a completed ground impact must not mistake contact k for one.
    receipt = {
        "schema": ORIGINAL_CONTACT_SCHEMA,
        "status": "not_applicable",
        "kind": model.ORIGINAL_CONTACT_TERMINATION_KIND,
        "required": False,
        "reason": ORIGINAL_CONTACT_REASON,
        "supplied_end_frame": supplied_end,
        "right_contact_frame": supplied_end,
        "frame_delta": 0.0,
        "endpoint_moved": False,
        "parameters_changed": False,
        "observations_discarded": 0,
        "physical_ending_classified": False,
        "ending_semantics": "unresolved",
        "terminal_ground_count": "not_asserted",
        "outgoing_velocity": None,
        "scope": (
            "explicit contact-to-contact prefix; endpoint is the supplied contact k, "
            "no ground completion, rebound, net or horizon grammar"
        ),
    }
    try:
        final = model.chain(
            scene,
            parameters,
            query_frames=tuple(np.asarray([a, b], float) for a, b in zip(times[:-1], times[1:])),
        )[-1]
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return {**receipt, "valid": False, "reason": "simulation_failed", "error": str(exc)}
    end_xyz = np.asarray(final["end_xyz"], float)
    incoming = np.asarray(final["velocities"][-1], float)
    finite = bool(np.isfinite(end_xyz).all() and np.isfinite(incoming).all())
    envelope = event_constraints.contact_xy_envelope_evidence(end_xyz) if finite else None
    return {
        **receipt,
        "right_contact_xyz_m": end_xyz.tolist(),
        "incoming_velocity_mps": incoming.tolist(),
        "right_contact_envelope": envelope,
        "valid": bool(finite and envelope["inside_venue_envelope"]),
    }


def _inside(frame: float, bounds) -> bool:
    """Closed declared window, with the float tolerance every neighbouring time check uses."""
    return bool(
        bounds[0] - TIME_NUMERICAL_TOLERANCE_FRAMES
        <= float(frame)
        <= bounds[1] + TIME_NUMERICAL_TOLERANCE_FRAMES
    )


def complete(
    scene: model.Scene,
    parameters: np.ndarray,
    kind: str,
    *,
    uncertainty_frames: float,
    last_observation_frame: float,
    passive_context_frames: float = 0.0,
    native_frames=None,
    live_contact_frames=(),
    duration: float | None = 0.25,
    window_tests_reported_ending: bool = True,
) -> dict:
    """Find the declared final-flight impact without changing the fitted state.

    ``window_tests_reported_ending`` (on) judges the declared timing window against
    the ending epoch this receipt reports. When the last native exposure falls inside
    the simulated ground dwell, that epoch is the exposure, not the instant the dwell
    began; an impact that starts a hair before the window opens while the ball is
    still on the ground at an exposure inside the window is therefore inside it
    (source02_short_0002_a1: impact 169.9976, exposure 170.0, window [170, 172]).
    No tolerance constant is introduced beyond the shared float tolerance. Off
    restores the earlier rule that tested only the dwell's first instant.

    ``passive_context_frames`` is off at zero, which is the promoted reference:
    an exposure recorded after the ground impact then holds the completion.
    Above zero the point is allowed to end at its impact with those exposures
    named as passive post-ending context, bounded by the annotators' own
    agreement on the epoch and refused outright where a labeled live contact
    sits inside the span.
    """
    scene.validate()
    if (
        not np.isfinite(uncertainty_frames)
        or not 0 < uncertainty_frames <= 1.0
        or not np.isfinite(last_observation_frame)
        or last_observation_frame != round(last_observation_frame)
        or last_observation_frame < max(f[-1] for f in scene.observation_frames)
        or last_observation_frame > scene.contact_frames[-1]
        or not np.isfinite(passive_context_frames)
        or not 0 <= passive_context_frames <= 1
    ):
        raise ValueError(
            "bounded timing uncertainty and full native observation inventory required"
        )
    contact_ending = model.original_contact_boundary(scene)
    if contact_ending or kind == model.ORIGINAL_CONTACT_TERMINATION_KIND:
        if not (contact_ending and kind == model.ORIGINAL_CONTACT_TERMINATION_KIND):
            raise ValueError(
                "original-contact completion kind and right boundary must be declared together"
            )
        return original_contact_receipt(scene, parameters)
    if kind == "net_stop" and scene.terminal_net_tail is not None:
        from cv.experiments.connected_shooting.labeled_terminal_net_tail import completion

        if native_frames is None:
            raise ValueError("full original native inventory required for net aftermath")
        return completion(scene, parameters, native_frames, duration=duration)
    if kind == "observed_horizon" and getattr(scene, "observed_horizon_tail", None) is not None:
        from cv.experiments.connected_shooting.observed_horizon_tail import completion

        if native_frames is None:
            raise ValueError("full original native inventory required for an observed tail")
        return completion(scene, parameters, native_frames, duration=duration)
    original_end = float(scene.contact_frames[-1])
    bounds = [original_end - uncertainty_frames, original_end + uncertainty_frames]
    receipt = {
        "schema": "connected_ground_completion_v2",
        "status": "held",
        "kind": kind,
        "supplied_end_frame": original_end,
        "bounds_frames": bounds,
        "last_observation_frame": float(last_observation_frame),
        "parameters_changed": False,
        "observations_discarded": 0,
        "passive_context_frames": float(passive_context_frames),
        "scope": "supplied ending evidence only; no independent event or point certificate",
    }
    if kind not in {"terminal_bounce", "second_bounce"}:
        return {**receipt, "reason": "unsupported_or_missing_ground_ending"}
    if scene.dynamics != "measured_240hz":
        return {**receipt, "reason": "unsupported_dynamics"}
    times = scene.contact_frames.copy()
    times[-1] = bounds[1]
    try:
        impacts = model.chain(replace(scene, contact_frames=times), parameters)[-1]["bounces"]
        ordinal = 2 if kind == "second_bounce" else 1
        receipt["required_impact_ordinal"] = ordinal
        receipt["simulated_impact_frames"] = [float(b["frame"]) for b in impacts]
        if len(impacts) < ordinal:
            return {**receipt, "reason": "required_impact_not_reached"}
        impact = impacts[ordinal - 1]
        end = float(impact["frame"])
        receipt["impact_frame"] = end
        receipt["impact_dwell_end_frame"] = end + float(impact["dwell_seconds"]) * scene.fps
        receipt["candidate_end_frame"] = end
        # The epoch reported below is the last exposure when it sits inside the
        # simulated dwell; judge the window on what is reported.
        reported = (
            max(end, float(last_observation_frame))
            if last_observation_frame <= receipt["impact_dwell_end_frame"]
            else end
        )
        if window_tests_reported_ending:
            in_window = _inside(end, bounds) or _inside(reported, bounds)
        else:
            in_window = bool(bounds[0] <= end <= bounds[1])
        receipt["window_test"] = {
            "reported_ending_frame": reported,
            "impact_instant_inside": _inside(end, bounds),
            "reported_ending_inside": _inside(reported, bounds),
            "tests_reported_ending": bool(window_tests_reported_ending),
            "float_tolerance_frames": TIME_NUMERICAL_TOLERANCE_FRAMES,
        }
        if not in_window:
            return {**receipt, "reason": "required_impact_outside_window"}
        if window_tests_reported_ending:
            others = sum(_inside(b["frame"], bounds) for b in impacts if b is not impact)
        else:
            others = sum(bounds[0] <= float(b["frame"]) <= bounds[1] for b in impacts) - 1
        if others != 0:
            return {**receipt, "reason": "ambiguous_impacts_in_window"}
        if end <= times[-2]:
            return {**receipt, "reason": "completion_would_discard_observations_or_flight"}
        if last_observation_frame > receipt["impact_dwell_end_frame"]:
            # The point already ended. Either the arm is off, in which case the
            # trailing exposure holds the completion as before, or the exposures
            # after the impact are named as passive context and kept.
            if not passive_context_frames:
                return {**receipt, "reason": "completion_would_discard_observations_or_flight"}
            passive = terminal_exposure_context.passive_context(
                receipt["impact_dwell_end_frame"],
                [float(last_observation_frame)] if native_frames is None else native_frames,
                live_contact_frames=live_contact_frames,
                allowance_frames=passive_context_frames,
            )
            receipt["passive_post_ending_context"] = passive
            if not passive["valid"]:
                return {**receipt, "reason": "passive_post_ending_context:" + passive["reason"]}
        else:
            # Impact has finite duration in this model. A native exposure inside
            # the already simulated ground dwell need not be discarded. Preserve
            # it without lengthening dwell, changing state or keeping outgoing flight.
            end = max(end, float(last_observation_frame))
        receipt["ending_phase"] = "impact_dwell" if end > impact["frame"] else "impact_instant"
        if not np.isfinite(impact["v_in"]).all() or impact["v_in"][2] >= 0:
            return {**receipt, "reason": "impact_not_descending"}
        times[-1] = end
        # Re-evaluate the existing path at its impact; no replacement XYZ state.
        # The query set is the flight boundaries alone, because a passive
        # exposure recorded after the ending sits outside the scored domain.
        final = model.chain(
            replace(scene, contact_frames=times),
            parameters,
            query_frames=tuple(np.asarray([a, b], float) for a, b in zip(times[:-1], times[1:])),
        )[-1]
        if abs(final["end_xyz"][2] - model.R_BALL) > 1e-6 or not np.allclose(
            final["end_xyz"], impact["x"], atol=1e-6, rtol=0
        ):
            return {**receipt, "reason": "impact_state_replay_mismatch"}
        return {
            **receipt,
            "status": "completed",
            "end_frame": end,
            "end_xyz": final["end_xyz"].tolist(),
            "frame_delta": end - original_end,
            "incoming_velocity": impact["v_in"].tolist(),
        }
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return {**receipt, "reason": "simulation_failed", "error": str(exc)}


def original_ground_event(
    scene: model.Scene, bounce_frames: tuple, events: list[dict]
) -> dict | None:
    """Find the unique supplied terminal ground; never derive its time from the fit."""
    if model.original_contact_boundary(scene):
        # The last supplied ground of a prefix is an interior bounce before
        # contact k, never a terminal ground event.
        return None
    if len(bounce_frames) == 0 or not len(bounce_frames[-1]):
        return None
    epoch = float(bounce_frames[-1][-1])
    matches = [
        event
        for event in events
        if event.get("event_type") == "bounce"
        and float(scene.contact_frames[-2]) < float(event["frame"])
        and abs(float(event["frame"]) - epoch) <= 1e-8
    ]
    return matches[0] if len(matches) == 1 else None


def complete_observed_rebound(
    scene: model.Scene,
    parameters: np.ndarray,
    kind: str,
    bounce_frames: tuple,
    rebound_frames: np.ndarray,
    ground_event: dict | None,
    *,
    uncertainty_frames: float,
) -> dict:
    """Check the original ground while retaining the entire observed rebound domain.

    Unlike ``complete``, the scene endpoint here is a simulation limit, not an
    impact observation. Neither that endpoint nor any native timestamp is moved.
    """
    from cv.experiments.connected_shooting import event_constraints

    receipt = dict(
        schema="connected_observed_rebound_completion_v1",
        status="held",
        kind=kind,
        parameters_changed=False,
        observations_discarded=0,
        simulation_end_frame=float(scene.contact_frames[-1]),
        scope="original supplied ground and explicit observed rebound; no correctness certificate",
    )
    if model.original_contact_boundary(scene):
        return {**receipt, "reason": "not_applicable_" + ORIGINAL_CONTACT_REASON}
    try:
        frames = event_constraints.validate_terminal_rebound(scene, bounce_frames, rebound_frames)
        if np.any(frames != np.rint(frames)):
            raise ValueError("native rebound epochs must be integer exposures")
        if not np.isfinite(uncertainty_frames) or not 0 < uncertainty_frames <= 1:
            raise ValueError("bounded ending timing uncertainty required")
        ordinal = 2 if kind == "second_bounce" else 1
        if kind not in {"terminal_bounce", "second_bounce"} or len(bounce_frames[-1]) != ordinal:
            return {**receipt, "reason": "unsupported_original_ground_topology"}
        if ground_event is None or ground_event.get("event_type") != "bounce":
            return {**receipt, "reason": "missing_original_ground_interval"}
        epoch = float(ground_event["frame"])
        interval = np.asarray(ground_event.get("frame_interval", []), float)
        if (
            interval.shape != (2,)
            or not np.isfinite(interval).all()
            or not np.isfinite(epoch)
            or not scene.contact_frames[-2] < interval[0] <= epoch <= interval[1]
            or interval[1] >= scene.contact_frames[-1]
            or abs(epoch - float(bounce_frames[-1][-1])) > 1e-8
            or not np.any(frames > interval[1])
        ):
            return {**receipt, "reason": "unsupported_original_ground_interval"}
        bounds = [
            max(float(interval[0]), epoch - uncertainty_frames),
            min(float(interval[1]), epoch + uncertainty_frames),
        ]
        receipt.update(
            supplied_end_frame=epoch,
            original_ground_interval=interval.tolist(),
            bounds_frames=bounds,
            required_impact_ordinal=ordinal,
            postbounce_labeled_frames=frames.tolist(),
        )
        impacts = model.chain(scene, parameters)[-1]["bounces"]
        receipt["simulated_impact_frames"] = [float(b["frame"]) for b in impacts]
        if len(impacts) < ordinal:
            return {**receipt, "reason": "required_impact_not_reached"}
        impact = impacts[ordinal - 1]
        end = float(impact["frame"])
        dwell_end = end + float(impact["dwell_seconds"]) * scene.fps
        receipt.update(impact_frame=end, impact_dwell_end_frame=dwell_end)
        # Same closed window as ``complete``, with the shared float tolerance only:
        # this path reports the impact instant itself, so there is no dwell exposure.
        if not _inside(end, bounds):
            return {**receipt, "reason": "required_impact_outside_original_window"}
        if sum(_inside(b["frame"], bounds) for b in impacts) != 1:
            return {**receipt, "reason": "ambiguous_impacts_in_window"}
        if any(float(b["frame"]) <= interval[1] for b in impacts[ordinal:]):
            return {**receipt, "reason": "extra_impact_before_original_ending_interval_closes"}
        # A native epoch inside the original bounce interval is not proof of
        # post-impact flight, even when it follows the representative label time.
        qualified = frames[frames > interval[1]]
        receipt["qualified_rebound_frames"] = qualified.tolist()
        if dwell_end > qualified[0]:
            return {**receipt, "reason": "rebound_observation_before_impact_dwell_ends"}
        xyz, velocity = np.asarray(impact["x"], float), np.asarray(impact["v_in"], float)
        if not np.isfinite(velocity).all() or velocity[2] >= 0:
            return {**receipt, "reason": "impact_not_descending"}
        if not np.isfinite(xyz).all() or abs(xyz[2] - model.R_BALL) > 1e-6:
            return {**receipt, "reason": "impact_state_not_ground"}
        return {
            **receipt,
            "status": "completed",
            "end_frame": end,
            "end_xyz": xyz.tolist(),
            "incoming_velocity": velocity.tolist(),
            "frame_delta": end - epoch,
            "observed_continuation_end_frame": float(frames[-1]),
            "later_simulated_ground_frames": [float(b["frame"]) for b in impacts[ordinal:]],
        }
    except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError) as exc:
        return {**receipt, "reason": "unsupported_rebound_context_or_simulation", "error": str(exc)}
