"""Input-only physical compatibility for connected ground-ending candidates.

No spatial truth, reviewed quality flag or uncertainty cutoff enters this check.
Supplied event/camera correctness, ball identity and contact reach remain outside
its scope. A compatible candidate is NOT an independently correct complete point.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cv.experiments.connected_shooting import model, terminal_completion, event_constraints
from cv.pipeline import camera_cal, net_cord_response, trajectory_contract


def crossings(frames: np.ndarray, xyz: np.ndarray) -> list[dict]:
    """Dense chord/plane intersections, including samples exactly on the plane."""
    output = []
    y = xyz[:, 1] - 11.885
    for i in range(len(frames) - 1):
        if y[i] == 0:
            fraction = 0.0
        elif y[i] * y[i + 1] < 0 or y[i + 1] == 0:
            fraction = -y[i] / (y[i + 1] - y[i])
        else:
            continue
        frame = float(frames[i] + fraction * (frames[i + 1] - frames[i]))
        if output and abs(output[-1]["frame"] - frame) < 1e-8:
            continue
        q = xyz[i] + fraction * (xyz[i + 1] - xyz[i])
        within = -0.914 <= q[0] <= 11.884
        band = float(camera_cal.net_height_at_x(float(q[0])))
        clearance = float(q[2] - model.R_BALL - band)
        output.append(
            dict(
                frame=frame,
                xyz=q.tolist(),
                within_net_width=bool(within),
                net_band_height_m=band,
                ball_centre_clearance_m=float(q[2] - band),
                ball_surface_clearance_m=clearance,
                penetration=bool(within and clearance < -1e-3),
            )
        )
    return output


def impact_passivity(impact: dict, end_frame: float, fps: float) -> dict:
    """Internal kinetic-energy consistency; unused terminal outgoing states excluded."""
    returned = (
        float(impact["frame"]) + float(impact.get("dwell_seconds", 0.0)) * fps < end_frame - 1e-6
    )
    result = dict(outgoing_flight_returned=returned, valid=not returned, kinetic_energy_ratio=None)
    if not returned:
        return result
    try:
        vectors = [np.asarray(impact[k], float) for k in ("v_in", "v_out", "w_in", "w_out")]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            return result
        ratio = float(model.rich_ball_physics.bounce_energy_ratio(impact))
        return {
            **result,
            "kinetic_energy_ratio": ratio,
            "valid": bool(np.isfinite(ratio) and 0 <= ratio <= 1.000001),
        }
    except (KeyError, TypeError, ValueError):
        return result


def evaluate(
    scene: model.Scene,
    parameters: np.ndarray,
    bounce_frames: tuple,
    termination_kind: str,
    native_frames: tuple,
    *,
    uncertainty_frames: float = 1.0,
    ending_uncertainty_frames: float = 1.0,
    terminal_rebound_frames: np.ndarray | None = None,
    terminal_ground_event: dict | None = None,
    duration: float | None = 0.25,
) -> dict:
    """Check a frozen state against explicit event times and original exposures only."""
    scene.validate()
    expected = event_constraints.validate(scene, bounce_frames, uncertainty_frames)
    n = len(scene.contact_frames) - 1
    contact_ending = model.original_contact_boundary(scene)
    if contact_ending != (termination_kind == model.ORIGINAL_CONTACT_TERMINATION_KIND):
        raise ValueError(
            "original-contact termination kind and right boundary must be declared together"
        )
    if contact_ending and terminal_rebound_frames is not None:
        raise ValueError("terminal rebound is not applicable to an original-contact boundary")
    if len(native_frames) != n:
        raise ValueError("complete native exposure inventory per flight required")
    native = tuple(np.asarray(f, float) for f in native_frames)
    for i, f in enumerate(native):
        if (
            f.ndim != 1
            or not len(f)
            or not np.isfinite(f).all()
            or np.any(f != np.rint(f))
            or np.any(np.diff(f) <= 0)
            or f[0] < scene.contact_frames[i]
            or f[-1] > scene.contact_frames[i + 1]
            # Half-open original membership: no native row of the last retained
            # flight may sit at or beyond supplied contact k.
            or (contact_ending and i == n - 1 and f[-1] >= scene.contact_frames[-1])
            or not set(scene.observation_frames[i]).issubset(f)
        ):
            raise ValueError("ordered native exposures must cover every fitting observation")
    result = dict(
        schema="connected_input_physical_compatibility_v1",
        compatible=False,
        scope=__doc__,
        complete_real_point_accepted=False,
        failures=[],
        parameters_changed=False,
        observations_discarded=0,
        native_exposures=sum(map(len, native)),
        bounce_uncertainty_frames=uncertainty_frames,
        ending_uncertainty_frames=ending_uncertainty_frames,
        physical_sampling_hz=240,
    )
    try:
        unresolved = scene.observed_horizon_tail is not None
        continued = (
            terminal_rebound_frames is not None
            and scene.terminal_net_tail is None
            and not unresolved
        )
        if continued:
            completion = terminal_completion.complete_observed_rebound(
                scene,
                parameters,
                termination_kind,
                expected,
                terminal_rebound_frames,
                terminal_ground_event,
                uncertainty_frames=ending_uncertainty_frames,
            )
        else:
            completion = terminal_completion.complete(
                scene,
                parameters,
                termination_kind,
                uncertainty_frames=ending_uncertainty_frames,
                last_observation_frame=float(native[-1][-1]),
                duration=duration,
                **(
                    {"native_frames": native[-1]}
                    if scene.terminal_net_tail is not None or unresolved
                    else {}
                ),
            )
        result["terminal_completion"] = completion
        supported_horizon = unresolved and completion.get("valid") is True
        # A contact-ending prefix never completes; its full domain is still
        # evaluated below and the endpoint gate is reported as a condition.
        if completion["status"] != "completed" and not supported_horizon and not contact_ending:
            return {**result, "failures": ["ground_completion:" + str(completion.get("reason"))]}
        times = scene.contact_frames.copy()
        if not continued and not contact_ending:
            times[-1] = completion["end_frame"]
        returned = replace(scene, contact_frames=times)
        queries = tuple(
            np.unique(
                np.r_[f, np.linspace(a, b, max(2, int(np.ceil((b - a) * 240 / scene.fps)) + 1))]
            )
            for a, b, f in zip(times[:-1], times[1:], native, strict=True)
        )
        fitted = model.chain(returned, parameters, query_frames=queries)
        # Numerical-only endpoint probe; it never changes the returned domain.
        probed_times = times.copy()
        probed_times[-1] += trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
        final_impacts = model.chain(replace(returned, contact_frames=probed_times), parameters)[-1][
            "bounces"
        ]
        rows = []
        for i, (fit, q, events) in enumerate(zip(fitted, queries, expected, strict=True)):
            impacts = final_impacts if i == n - 1 else fit["bounces"]
            observed = [
                b
                for b in impacts
                if b["frame"]
                <= fit["end_frame"] + trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
            ]
            # Post-ending grounds are simulated continuation, never replacements
            # for an original competitive ordinal. They still enter passivity.
            competitive = observed
            if scene.terminal_net_tail is not None:
                from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                    competitive_grounds,
                )

                competitive = competitive_grounds(scene, i, observed)
            elif unresolved:
                from cv.experiments.connected_shooting.observed_horizon_tail import (
                    competitive_grounds,
                )

                competitive = competitive_grounds(scene, i, observed, supplied_count=len(events))
            elif continued and i == n - 1:
                competitive = [
                    b
                    for j, b in enumerate(observed)
                    if j < len(events)
                    or float(b["frame"]) <= completion["original_ground_interval"][1]
                ]
            deltas = [float(b["frame"] - t) for b, t in zip(competitive, events)]
            net_hits = fit.get("net_hits", [])
            expected_net = (
                np.empty(0)
                if scene.net_hit_frames is None
                else np.asarray(scene.net_hit_frames[i], float)
            )
            net_checks = []
            for hit in net_hits:
                xyz = np.asarray(hit["x"], float)
                net_checks.append(
                    {
                        "frame": float(hit["frame"]),
                        "xyz": xyz.tolist(),
                        "plane_error_m": float(abs(xyz[1] - 11.885)),
                        "at_or_below_tape": bool(
                            xyz[2] <= float(hit["tape_height_m"]) + model.R_BALL + 0.03
                        ),
                        "in_tape_band": bool(net_cord_response.in_tape_band(xyz)),
                        "within_net_width": bool(-0.915 <= xyz[0] <= 11.885),
                        "position_continuous": bool(hit.get("position_continuous")),
                    }
                )
            rows.append(
                dict(
                    flight_index=i,
                    supplied_bounce_frames=events.tolist(),
                    modeled_bounce_frames=[float(b["frame"]) for b in observed],
                    bounce_time_deltas=deltas,
                    bounce_count_timing_met=len(competitive) == len(events)
                    and all(abs(d) <= uncertainty_frames for d in deltas),
                    net_crossings=crossings(q, fit["positions"]),
                    minimum_height_m=float(fit["positions"][:, 2].min()),
                    impact_passivity=[
                        impact_passivity(b, fit["end_frame"], scene.fps) for b in observed
                    ],
                    supplied_net_hit_frames=expected_net.tolist(),
                    modeled_net_hits=net_checks,
                    net_hit_transition_met=bool(
                        len(net_checks) == len(expected_net)
                        and all(
                            (
                                scene.terminal_net_tail["interval"][0]
                                <= row["frame"]
                                <= scene.terminal_net_tail["interval"][1]
                                if scene.terminal_net_tail is not None and i == n - 1
                                else abs(row["frame"] - expected_net[j]) <= uncertainty_frames
                            )
                            and row["plane_error_m"] <= 0.05
                            and (
                                row["in_tape_band"]
                                if net_cord_response.uses_tape_band(row["frame"])
                                else row["at_or_below_tape"]
                            )
                            and row["within_net_width"]
                            and row["position_continuous"]
                            for j, row in enumerate(net_checks)
                        )
                    ),
                )
            )
        endpoints = [
            dict(
                start_frame=f["start_frame"],
                end_frame=f["end_frame"],
                start_xyz=f["start_xyz"].tolist(),
                end_xyz=f["end_xyz"].tolist(),
                terminal_end=i == n - 1 and not contact_ending,
                **({"right_boundary_end": i == n - 1} if contact_ending else {}),
            )
            for i, f in enumerate(fitted)
        ]
        connection = trajectory_contract.trajectory_connection_report(endpoints)
        ending_endpoints = endpoints
        if continued:
            ending_endpoints = [
                *endpoints[:-1],
                {
                    **endpoints[-1],
                    "end_frame": completion["end_frame"],
                    "end_xyz": completion["end_xyz"],
                },
            ]
        if contact_ending:
            ending = {
                "valid": completion.get("valid") is True,
                "required": False,
                "kind": model.ORIGINAL_CONTACT_TERMINATION_KIND,
                "physical_ending_classified": False,
                "ending_semantics": "unresolved",
                "right_contact_frame": completion["right_contact_frame"],
                "right_contact_xyz_m": completion.get("right_contact_xyz_m"),
                "incoming_velocity_mps": completion.get("incoming_velocity_mps"),
                "outgoing_velocity_mps": None,
                "right_contact_envelope": completion.get("right_contact_envelope"),
            }
        elif unresolved:
            ending = {
                "valid": supported_horizon,
                "required": False,
                "kind": "observed_horizon",
                "physical_ending_classified": False,
            }
        else:
            ending = trajectory_contract.terminal_ground_endpoint_report(
                termination_kind, ending_endpoints
            )
        transition_frames = {
            round(float(frame), 4) for group in (scene.net_hit_frames or ()) for frame in group
        }
        if scene.terminal_net_tail is not None:
            transition_frames = {
                float(hit["frame"]) for fit in fitted for hit in fit.get("net_hits", [])
            }
        penetrations = [
            crossing
            for row in rows
            for crossing in row["net_crossings"]
            if crossing["penetration"]
            and crossing["ball_surface_clearance_m"] < -0.01
            and not any(
                abs(crossing["frame"] - frame) <= uncertainty_frames for frame in transition_frames
            )
        ]
        conditions = dict(
            structural_connections=connection["valid"],
            **(
                {"original_contact_endpoint_inside_envelope": ending["valid"]}
                if contact_ending
                else {"observed_horizon_supported": ending["valid"]}
                if unresolved
                else {"physical_ground_ending": ending["valid"]}
            ),
            bounce_count_timing=all(r["bounce_count_timing_met"] for r in rows),
            no_net_penetration=not penetrations,
            supplied_net_transitions_met=all(row["net_hit_transition_met"] for row in rows),
            no_underground_path=all(r["minimum_height_m"] >= model.R_BALL - 1e-3 for r in rows),
            passive_court_impacts=all(b["valid"] for r in rows for b in r["impact_passivity"]),
        )
        return {
            **result,
            "compatible": all(conditions.values()),
            "conditions": conditions,
            "failures": [k for k, v in conditions.items() if not v],
            "flights": rows,
            "connections": connection,
            "ending": ending,
            "physical_sample_counts": list(map(len, queries)),
            **(
                {
                    "right_boundary_kind": model.ORIGINAL_CONTACT_TERMINATION_KIND,
                    "complete_original_source": False,
                    "ending_semantics": "unresolved",
                }
                if contact_ending
                else {}
            ),
        }
    except (ValueError, FloatingPointError, OverflowError) as error:
        return {**result, "failures": ["simulation_failed"], "error": str(error)}
