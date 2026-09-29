"""Explicit bounce-window evidence for research fitting, not an event detector.

The simulator still detects every impact. No state, picture or event is snapped
to a label. These residuals guide optimization; unchanged physical audits decide
whether the resulting trajectory respects the supplied evidence.
"""

from __future__ import annotations

import numpy as np

TIME_SCALE_FRAMES = 0.25
MISSING_GROUND_SCALE_M = 0.05

# Court coordinates run from the near doubles sideline and baseline. The
# professional ITF run-back remains ranking evidence, but is not a hard wall:
# broadcast venues can have substantially more room and players use it.
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
PROFESSIONAL_RUNBACK_XY_M = (4.57, 8.23)
CONTACT_XY_ENVELOPE_M = (
    (-6.0, COURT_WIDTH_M + 6.0),
    (-12.0, COURT_LENGTH_M + 12.0),
)
PROFESSIONAL_CONTACT_XY_M = (
    (-PROFESSIONAL_RUNBACK_XY_M[0], COURT_WIDTH_M + PROFESSIONAL_RUNBACK_XY_M[0]),
    (-PROFESSIONAL_RUNBACK_XY_M[1], COURT_LENGTH_M + PROFESSIONAL_RUNBACK_XY_M[1]),
)
RUNBACK_SOFT_SCALE_M = 1.0


#: The widest bounce timing dead zone a fit may declare (owner, 2026-09-08:
#: bounces within two frames pass the eye test); one frame stays the default.
MAX_UNCERTAINTY_FRAMES = 2.0
#: Unmodelled net clearance scale for ordinary flights (optimization guidance).
NET_CLEARANCE_GUIDANCE_M = 0.01
#: ``serve_tape_clearance="hard"``: the serve flight's scale while a serve-origin
#: S6 stage runs (``serve_clearance``). None keeps the guidance scale everywhere.
SERVE_CLEARANCE_SCALE_M: float | None = None
HARD_SERVE_CLEARANCE_SCALE_M = 0.0005


class serve_clearance:
    """Hold the serve flight's net clearance at ``scale_m`` inside this block."""

    def __init__(self, scale_m: float | None):
        self.scale_m = scale_m

    def __enter__(self):
        global SERVE_CLEARANCE_SCALE_M
        self.previous = SERVE_CLEARANCE_SCALE_M
        SERVE_CLEARANCE_SCALE_M = self.scale_m
        return self

    def __exit__(self, *exc):
        global SERVE_CLEARANCE_SCALE_M
        SERVE_CLEARANCE_SCALE_M = self.previous
        return False


def contact_xy_envelope_evidence(contact_xyz) -> dict:
    """Return the outer venue gate and soft professional-run-back cost."""
    contact = np.asarray(contact_xyz, float)
    if contact.shape != (3,) or not np.isfinite(contact).all():
        raise ValueError("one finite contact XYZ required")
    outer = np.asarray(CONTACT_XY_ENVELOPE_M, float)
    professional = np.asarray(PROFESSIONAL_CONTACT_XY_M, float)
    xy = contact[:2]
    overrun = np.maximum(professional[:, 0] - xy, 0.0) + np.maximum(xy - professional[:, 1], 0.0)
    return {
        "inside_venue_envelope": bool(np.all((xy >= outer[:, 0]) & (xy <= outer[:, 1]))),
        "inside_professional_runback": bool(
            np.all((xy >= professional[:, 0]) & (xy <= professional[:, 1]))
        ),
        "professional_runback_overrun_xy_m": overrun.tolist(),
        "runback_soft_scale_m": RUNBACK_SOFT_SCALE_M,
        "runback_soft_penalty": float(np.sum((overrun / RUNBACK_SOFT_SCALE_M) ** 2)),
        "venue_envelope_xy_m": [list(row) for row in CONTACT_XY_ENVELOPE_M],
        "professional_runback_xy_m": [list(row) for row in PROFESSIONAL_CONTACT_XY_M],
        "hard_wall": False,
    }


def validate(scene, bounce_frames, uncertainty_frames):
    if not np.isfinite(uncertainty_frames) or not 0 <= uncertainty_frames <= MAX_UNCERTAINTY_FRAMES:
        raise ValueError("bounce uncertainty must be between zero and two native frames")
    if len(bounce_frames) != len(scene.contact_frames) - 1:
        raise ValueError("explicit bounce inventory per flight required")
    groups = tuple(np.asarray(group, float) for group in bounce_frames)
    for group, start, end in zip(
        groups, scene.contact_frames[:-1], scene.contact_frames[1:], strict=True
    ):
        if (
            group.ndim != 1
            or not np.isfinite(group).all()
            or np.any(np.diff(group) <= 0)
            or np.any(group <= start)
            or np.any(group > end)
        ):
            raise ValueError("ordered in-flight physical bounce times required")
    return groups


def mixed_loss(image_residual_count: int):
    """Soft-L1 images, quadratic evidence: do not downweight a wrong event as an outlier."""

    def loss(z):
        root = np.sqrt(1 + z)
        rho = np.array([2 * (root - 1), 1 / root, -0.5 / root**3])
        rho[0, image_residual_count:] = z[image_residual_count:]
        rho[1, image_residual_count:] = 1
        rho[2, image_residual_count:] = 0
        return rho

    return loss


def _refuse_terminal_guidance_on_contact_boundary(scene, name: str) -> None:
    """The low layer refuses: a contact-to-contact prefix has no terminal ending."""
    from cv.experiments.connected_shooting import model

    if model.original_contact_boundary(scene):
        raise ValueError(
            f"{name} is not applicable to an original-contact right boundary; "
            "the last flight ends at supplied contact k, not at a ground ending"
        )


def validate_terminal_observation(scene, bounce_frames, last_frame):
    """Require explicit ground-ending evidence and the full last native exposure."""
    _refuse_terminal_guidance_on_contact_boundary(scene, "terminal observation guidance")
    if (
        scene.dynamics != "measured_240hz"
        or bounce_frames is None
        or len(bounce_frames) != len(scene.contact_frames) - 1
        or len(bounce_frames[-1]) not in {1, 2}
        or abs(bounce_frames[-1][-1] - scene.contact_frames[-1]) > 1e-8
        or isinstance(last_frame, (bool, np.bool_))
        or not np.isfinite(last_frame)
        or last_frame != round(last_frame)
        or last_frame < max(f[-1] for f in scene.observation_frames)
        or last_frame > scene.contact_frames[-1]
    ):
        raise ValueError(
            "terminal observation requires explicit ground ending and full native coverage"
        )


def terminal_observation_penalty(flight, expected, last_frame, fps):
    """Penalize an ending that would omit a supplied exposure after physical dwell.

    The ordinary missing-impact residual still guides a missing terminal bounce.
    This adds only the no-discard condition that the post-fit completion already
    enforces; it changes neither the path, the dwell, nor the timing tolerance.
    """
    ordinal = len(expected)
    impact = flight["bounces"][ordinal - 1] if len(flight["bounces"]) >= ordinal else None
    dwell_end = None if impact is None else float(impact["frame"] + impact["dwell_seconds"] * fps)
    violation = 0.0 if dwell_end is None else max(float(last_frame - dwell_end), 0.0)
    return violation / TIME_SCALE_FRAMES, dict(
        last_native_frame=float(last_frame),
        required_impact_ordinal=ordinal,
        modeled_dwell_end_frame=dwell_end,
        native_coverage_violation_frames=violation,
        time_scale_frames=TIME_SCALE_FRAMES,
        missing_impact_guided_by_existing_ground_residual=impact is None,
        observations_discarded=0,
        scope="optimization guidance only; unchanged physical completion decides compatibility",
    )


def validate_terminal_rebound(scene, bounce_frames, rebound_frames):
    """Validate an explicit post-terminal-bounce image segment.

    The segment contains real labeled exposures already present in the scene;
    it does not create an event or move a timestamp.  Its first exposure is the
    structural witness that the measured bounce and dwell have completed.
    """
    _refuse_terminal_guidance_on_contact_boundary(scene, "terminal rebound guidance")
    frames = np.asarray(rebound_frames, float)
    if (
        scene.dynamics != "measured_240hz"
        or bounce_frames is None
        or len(bounce_frames) != len(scene.contact_frames) - 1
        or not len(bounce_frames[-1])
        or frames.ndim != 1
        or not len(frames)
        or not np.isfinite(frames).all()
        or np.any(np.diff(frames) <= 0)
        or np.any(frames <= float(bounce_frames[-1][-1]))
        or np.any(frames > float(scene.contact_frames[-1]))
        or not np.all(np.isin(frames, np.asarray(scene.observation_frames[-1], float)))
    ):
        raise ValueError(
            "terminal rebound requires measured dynamics and ordered labeled exposures "
            "after the last supplied bounce"
        )
    return frames


def terminal_rebound_penalty(flight, expected, rebound_frames, fps):
    """Require the modeled measured-surface rebound before its first picture."""
    ordinal = len(expected)
    impact = flight["bounces"][ordinal - 1] if len(flight["bounces"]) >= ordinal else None
    first = float(rebound_frames[0])
    dwell_end = None if impact is None else float(impact["frame"] + impact["dwell_seconds"] * fps)
    violation = 0.0 if dwell_end is None else max(dwell_end - first, 0.0)
    return violation / TIME_SCALE_FRAMES, {
        "status": "active",
        "labeled_frames": rebound_frames.tolist(),
        "labeled_frame_count": len(rebound_frames),
        "first_postbounce_frame": first,
        "supplied_terminal_bounce_frame": float(expected[-1]),
        "modeled_terminal_bounce_frame": None if impact is None else float(impact["frame"]),
        "modeled_dwell_end_frame": dwell_end,
        "rebound_order_violation_frames": violation,
        "time_scale_frames": TIME_SCALE_FRAMES,
        "surface_model": "measured_240hz selected hard/clay/hard-for-grass profile",
        "spin_carried_through_impact": True,
        "pictures_added": 0,
        "timestamps_changed": False,
        "missing_impact_guided_by_existing_ground_residual": impact is None,
    }


def terminal_rebound_slack(
    scene, parameters, bounce_frames, rebound_frames, *, simulation_cache=None
):
    """Hard feasibility for a terminal flight continued through its rebound."""
    from cv.experiments.connected_shooting import model

    frames = validate_terminal_rebound(scene, bounce_frames, rebound_frames)
    flights = model.chain(
        scene,
        parameters,
        **({"simulation_cache": simulation_cache} if simulation_cache is not None else {}),
    )
    expected = np.asarray(bounce_frames[-1], float)
    observed = flights[-1]["bounces"]
    if len(observed) < len(expected):
        raise ValueError("required terminal impact not reached in rebound segment")
    impact = observed[len(expected) - 1]
    impact_frame = float(impact["frame"])
    dwell_end = impact_frame + float(impact["dwell_seconds"]) * scene.fps
    supplied = float(expected[-1])
    return np.asarray(
        [
            float(frames[0]) - dwell_end,
            impact_frame - (supplied - MAX_UNCERTAINTY_FRAMES),
            supplied + MAX_UNCERTAINTY_FRAMES - impact_frame,
        ]
    )


def evaluate(
    scene,
    parameters,
    bounce_frames,
    uncertainty_frames,
    *,
    simulation_cache=None,
    net_clearance_scale_m=None,
    terminal_last_observation_frame=None,
    terminal_rebound_frames=None,
):
    from cv.experiments.connected_shooting import model

    if terminal_last_observation_frame is not None:
        validate_terminal_observation(scene, bounce_frames, terminal_last_observation_frame)
    rebound_frames = None
    if terminal_rebound_frames is not None:
        rebound_frames = validate_terminal_rebound(scene, bounce_frames, terminal_rebound_frames)
    queries = tuple(
        np.unique(np.r_[native, events])
        for native, events in zip(scene.observation_frames, bounce_frames, strict=True)
    )
    # Measured collision-free scenes use None as well as explicit empty groups.
    # Their physical objective must agree without changing topology metadata used
    # by preparation/initialization. An explicit scale already supplies this term.
    implicit_net_clearance = (
        scene.dynamics == "measured_240hz"
        and scene.net_hit_frames is None
        and net_clearance_scale_m is None
    )
    if scene.net_hit_frames is not None or implicit_net_clearance:
        from cv.experiments.connected_shooting import net_constraints

        queries = net_constraints.dense_queries(scene, bounce_frames)
    if net_clearance_scale_m is not None:
        from cv.experiments.connected_shooting import net_constraints

        net_constraints.validate(scene, net_clearance_scale_m, bounce_frames)
        queries = net_constraints.dense_queries(scene, bounce_frames)
    flights = model.chain(
        scene,
        parameters,
        query_frames=queries,
        **({"simulation_cache": simulation_cache} if simulation_cache is not None else {}),
    )
    residuals, evidence, native_flights = [], [], []
    for i, (flight, query, expected) in enumerate(
        zip(flights, queries, bounce_frames, strict=True)
    ):
        observed = flight["bounces"]
        if scene.terminal_net_tail is not None:
            from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                competitive_grounds,
            )

            observed = competitive_grounds(scene, i, observed)
        if getattr(scene, "observed_horizon_tail", None) is not None:
            # Unknown terminal ground count: do not let the extra-impact push
            # below drive a true latent landing past the observation horizon.
            # Every interior flight keeps its supplied-count constraint.
            from cv.experiments.connected_shooting.observed_horizon_tail import (
                competitive_grounds as tail_grounds,
            )

            observed = tail_grounds(scene, i, observed, supplied_count=len(expected))
        native_flights.append(
            {
                **flight,
                "positions": flight["positions"][
                    np.searchsorted(query, scene.observation_frames[i])
                ],
                "velocities": flight["velocities"][
                    np.searchsorted(query, scene.observation_frames[i])
                ],
            }
        )
        for j, frame in enumerate(expected):
            if j < len(observed):
                delta = observed[j]["frame"] - frame
                residuals.append(
                    np.sign(delta) * max(abs(delta) - uncertainty_frames, 0) / TIME_SCALE_FRAMES
                )
            else:
                height = flight["positions"][np.searchsorted(query, frame), 2]
                residuals.append(max(float(height) - model.R_BALL, 0) / MISSING_GROUND_SCALE_M)
        # First extra impact must move beyond the returned flight domain. Unlike
        # a count-only constant penalty, this supplies a timing gradient.
        extra = observed[len(expected) :]
        residuals.append(
            max(flight["end_frame"] - extra[0]["frame"], 0) / TIME_SCALE_FRAMES if extra else 0
        )
        evidence.append(
            {
                "flight_index": i,
                "expected_bounce_frames": expected.tolist(),
                "modeled_bounce_frames": [b["frame"] for b in observed],
                "count_agrees": len(expected) == len(observed),
            }
        )
        if net_clearance_scale_m is not None:
            residual, receipt = net_constraints.penalty(
                query, flight["positions"], net_clearance_scale_m
            )
            residuals.append(residual)
            evidence[-1]["net_clearance"] = receipt
        if terminal_last_observation_frame is not None and i == len(flights) - 1:
            residual, receipt = terminal_observation_penalty(
                flight, expected, terminal_last_observation_frame, scene.fps
            )
            residuals.append(residual)
            evidence[-1]["terminal_observation"] = receipt
        if rebound_frames is not None and i == len(flights) - 1:
            residual, receipt = terminal_rebound_penalty(
                flight, expected, rebound_frames, scene.fps
            )
            residuals.append(residual)
            evidence[-1]["terminal_rebound"] = receipt
        if scene.net_hit_frames is not None and len(scene.net_hit_frames[i]):
            from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                collision_residuals,
            )

            net_residuals = collision_residuals(
                scene, i, flight.get("net_hits", []), float(scene.net_hit_frames[i][0])
            )
            residuals.extend(net_residuals)
            evidence[-1]["net_collision"] = {
                "supplied_frame": float(scene.net_hit_frames[i][0]),
                "modeled": flight.get("net_hits", []),
                "residuals": net_residuals.tolist(),
                "position_snapped": False,
            }
        elif scene.net_hit_frames is not None or implicit_net_clearance:
            from cv.experiments.connected_shooting import net_constraints

            scale = (
                SERVE_CLEARANCE_SCALE_M
                if i == 0 and SERVE_CLEARANCE_SCALE_M is not None
                else NET_CLEARANCE_GUIDANCE_M
            )
            residual, receipt = net_constraints.penalty(query, flight["positions"], scale)
            residuals.append(residual)
            evidence[-1]["unmodeled_net_clearance"] = receipt
    return native_flights, np.asarray(residuals, float), evidence
