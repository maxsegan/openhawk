"""Score every flight of a jointly fitted point on its own evidence.

Evaluation only.  The whole-point fit is unchanged: one joint solve still
produces one connected parameter vector.  What changes here is *acceptance*.
The promoted arm asks one question of the whole point, so a twelve-contact
attempt is lost when a single flight fails.  This module asks the same
questions flight by flight -- that flight's own native observations, its own
bounce witness, the contact reach at both of its ends, its own physics and its
continuity into its neighbours -- and returns one verdict per flight.

A point is COMPLETE only when every flight is accepted and the whole-point
structural connection and physical ending hold. Ground completion is the
default; the explicit labeled terminal-net adapter supplies a separately
reviewed net-ending receipt. Otherwise accepted flights are emitted as valid
segments and rejected ones as explicit gaps.

Nothing here is automatic-inference eligible and no threshold below is a
product promotion.  ``Thresholds`` carries the declared relaxation ladder; its
default rung is exactly the promoted reference, so the base rung reproduces the
published acceptance rather than restating it more loosely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from cv.experiments.connected_shooting.labeled_terminal_net_tail import EVENT_TOLERANCE_FRAMES

from cv.experiments.connected_shooting import (
    event_constraints,
    model,
    observed_horizon_tail as horizon_tail,
    player_position,
    real_exposure_replay as exposure,
    terminal_completion,
)
from cv.experiments.connected_shooting.physical_compatibility import crossings, impact_passivity
from cv.pipeline import trajectory_contract
from cv.validation.flight_gate_audit import BOUNCE_GRADED_SIGMA_MULTIPLE
from cv.pipeline import net_cord_response as _net_cord


def _tape_band(xyz) -> bool:
    return _net_cord.in_tape_band(xyz)


def _weighted_post_contact_rms(rows: list[dict], net_frame: float) -> float | None:
    errors = []
    for row in rows:
        if row.get("split") != "training" or row.get("error_px") is None:
            continue
        error = float(row["error_px"])
        if float(row["frame"]) > net_frame:
            error = error / _net_cord.POST_CONTACT_WIDENING
        errors.append(error)
    return _rms(errors)


SCHEMA = "connected_per_flight_acceptance_v1"
BOUNCE_WITNESS_SIGMA_LIMIT_M = 0.30
BOUNCE_WITNESS_TIMING_LIMIT_FRAMES = 1.25
#: exit_partial_endings: an extra modelled impact this close to a picture-exit end is latent.
EXIT_LATENT_FRAMES = 1.0
GRASS_BOUNCE_WITNESS_TIMING_LIMIT_FRAMES = 0.85

#: Shared with the whole-point search. Only this outer venue envelope gates;
#: the smaller professional run-back contributes a soft selection penalty.
CONTACT_XY_ENVELOPE_M = event_constraints.CONTACT_XY_ENVELOPE_M


#: The widest event timing tolerance the ladder may declare.  The owner's eye
#: test accepts a bounce modelled up to two native frames from its label
#: (2026-09-08); one frame stays the promoted default until the pattern delta
#: and the synthetic audit clear a wider rung.
#: Two frames is the promoted timing window. 3.5 is the owner-approved cap from
#: 2026-09-24 (`owner_gate_loosening_20260924`); the production rung stays at 2
#: unless that policy overlays it. Ending-window uncertainty is not that decision
#: and stays inside two frames.
MAX_EVENT_UNCERTAINTY_FRAMES = 3.5
MAX_ENDING_UNCERTAINTY_FRAMES = 2.0
GRASS_BOUNCE_EPOCH_FRAMES = 0.85
WINDOW_TIME_SHIFTS_FRAMES = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class Thresholds:
    """One rung of the declared relaxation ladder.

    The defaults are the promoted reference.  ``bounce_circle_floor_m`` is the
    graded bounce circle's floor (the witness circle is
    ``max(floor, 2 sigma)``); ``directional_rms_limit_px`` is the event-side
    4/10-frame window limit; ``flight_reprojection_rms_limit_px`` caps a
    flight's own training reprojection RMS over its whole span, which the
    event-side windows do not cover; ``velocity_slack_mps`` is the largest
    unmodelled per-axis velocity change allowed at a flight junction or at one
    unobserved interior frame.
    """

    bounce_circle_floor_m: float = 0.20
    bounce_ray_limit_m: float = 0.9144
    serve_bounce_ray_limit_m: float | None = None
    directional_rms_limit_px: float = 16.0
    directional_time_shift_frames: float = 0.0
    flight_reprojection_rms_limit_px: float = 16.0
    velocity_slack_mps: float = 0.0
    bounce_uncertainty_frames: float = 1.0
    ending_uncertainty_frames: float = 1.0
    ending_passive_context_frames: float = 0.0
    net_clearance_sigma_m: float | None = None
    grass_bounce_epoch_frames: float = GRASS_BOUNCE_EPOCH_FRAMES
    #: Set only by the 2026-09-24 owner policy. It does not raise the ordinary
    #: 32 px window. A flight gets it only when directional is the only failure.
    directional_only_waiver_px: float | None = None

    def validate(self) -> None:
        values = (
            self.bounce_circle_floor_m,
            self.bounce_ray_limit_m,
            self.directional_rms_limit_px,
            self.directional_time_shift_frames,
            self.flight_reprojection_rms_limit_px,
            self.velocity_slack_mps,
            self.bounce_uncertainty_frames,
            self.ending_uncertainty_frames,
            self.ending_passive_context_frames,
            self.grass_bounce_epoch_frames,
        )
        if not all(np.isfinite(value) for value in values) or any(value < 0 for value in values):
            raise ValueError("finite non-negative acceptance thresholds required")
        if self.ending_passive_context_frames > 1:
            raise ValueError("passive post-ending context stays inside one native frame")
        if self.serve_bounce_ray_limit_m is not None and (
            not np.isfinite(self.serve_bounce_ray_limit_m)
            or not 0 < self.serve_bounce_ray_limit_m <= self.bounce_ray_limit_m
        ):
            raise ValueError(
                "a serve bounce ray limit tightens the general limit, never loosens it"
            )
        if self.net_clearance_sigma_m is not None and (
            not np.isfinite(self.net_clearance_sigma_m) or not 0 < self.net_clearance_sigma_m <= 0.5
        ):
            raise ValueError("a net clearance sigma is a bounded positive geometry uncertainty")
        if not 0 < self.bounce_uncertainty_frames <= MAX_EVENT_UNCERTAINTY_FRAMES:
            raise ValueError("bounce timing uncertainty stays within 3.5 native frames")
        if not 0 < self.ending_uncertainty_frames <= MAX_ENDING_UNCERTAINTY_FRAMES:
            raise ValueError("ending timing uncertainty stays within two native frames")
        if not 0 < self.grass_bounce_epoch_frames <= BOUNCE_WITNESS_TIMING_LIMIT_FRAMES:
            raise ValueError("grass bounce epoch stays within the hard-court epoch allowance")
        if self.directional_only_waiver_px is not None and (
            not np.isfinite(self.directional_only_waiver_px)
            or not self.directional_rms_limit_px < self.directional_only_waiver_px <= 48.0
        ):
            raise ValueError("a directional-only waiver stays inside the measured 48 px rung")
        if self.directional_time_shift_frames not in {0.0, 1.0}:
            raise ValueError("direction-window model time shift is off or one native frame")

    def as_dict(self) -> dict[str, float]:
        return {
            "bounce_circle_floor_m": float(self.bounce_circle_floor_m),
            "bounce_ray_limit_m": float(self.bounce_ray_limit_m),
            "serve_bounce_ray_limit_m": (
                None
                if self.serve_bounce_ray_limit_m is None
                else float(self.serve_bounce_ray_limit_m)
            ),
            "directional_rms_limit_px": float(self.directional_rms_limit_px),
            "directional_time_shift_frames": float(self.directional_time_shift_frames),
            "flight_reprojection_rms_limit_px": float(self.flight_reprojection_rms_limit_px),
            "velocity_slack_mps": float(self.velocity_slack_mps),
            "bounce_uncertainty_frames": float(self.bounce_uncertainty_frames),
            "ending_uncertainty_frames": float(self.ending_uncertainty_frames),
            "ending_passive_context_frames": float(self.ending_passive_context_frames),
            "net_clearance_sigma_m": (
                None if self.net_clearance_sigma_m is None else float(self.net_clearance_sigma_m)
            ),
            "grass_bounce_epoch_frames": float(self.grass_bounce_epoch_frames),
            "directional_only_waiver_px": (
                None
                if self.directional_only_waiver_px is None
                else float(self.directional_only_waiver_px)
            ),
        }


def net_ending_labeled(ending_kind: str | None) -> bool:
    """Does the labeled ending name the net as the physical event that ended it?

    Read from the free-text ending kind rather than the collapsed family, so a
    ``serve_fault_net`` -- a fault whose physical ending is still the net -- is
    a net ending here even though its scoring family is the fault.
    """
    return bool(ending_kind) and "net" in str(ending_kind).lower()


def net_crossing_verdict(centre_clearance_m: float, sigma_m: float) -> str:
    """The frozen contract's three-way reading of one modelled net crossing.

    The margin is the ball centre above the band, which is 0.914 m at the centre
    strap rising to 1.07 m at the posts.  The sigma carries the ball radius, the
    camera height uncertainty and the exposure phase together; it is a statement
    about what the geometry can resolve, not a widened allowance.
    """
    if centre_clearance_m > sigma_m:
        return "cleared"
    if centre_clearance_m < -sigma_m:
        return "into_net"
    return "net_contact"


def graded_circle_radius_m(sigma_m: float | None, floor_m: float) -> float:
    """The bounce circle at an explicit floor; the 2 sigma widening is unchanged."""
    if sigma_m is None or not np.isfinite(float(sigma_m)):
        return float(floor_m)
    return float(max(floor_m, BOUNCE_GRADED_SIGMA_MULTIPLE * float(sigma_m)))


def _rms(values: Sequence[float]) -> float | None:
    values = [float(v) for v in values]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


#: Role of the last retained flight of a contact-to-contact prefix. It ends at
#: supplied contact k and is scored as an interior flight, never as ``final``.
INTERIOR_CONTACT_FLIGHT_ROLE = "interior_contact_flight"
#: The first modelled flight of a source whose origin was never observed.
UNKNOWN_ORIGIN_FLIGHT_ROLE = "unknown_origin_flight"


def flight_role(
    index: int,
    count: int,
    *,
    right_boundary_kind: str = "supplied_end",
    first_contact_role: str = "unspecified",
) -> str:
    """Name the flight the way a rally reads, not by bare index.

    The serve keeps its role in every scene. Only the explicit right-boundary
    kind decides whether the last flight is the point's final flight or an
    interior flight that happens to be the last one retained. An unknown origin
    names its own first flight for what it is and claims no later stroke order.
    """
    if right_boundary_kind not in model.RIGHT_BOUNDARY_KINDS:
        raise ValueError("explicit supported right boundary kind required")
    if first_contact_role not in ("serve", "unspecified", "rally", "unknown"):
        raise ValueError("supported first contact role required")
    if first_contact_role in ("rally", "unknown"):
        return (
            INTERIOR_CONTACT_FLIGHT_ROLE
            if right_boundary_kind == "original_contact" and index == count - 1
            else "final"
            if index == count - 1
            else UNKNOWN_ORIGIN_FLIGHT_ROLE
            if first_contact_role == "unknown" and index == 0
            else "mid_rally"
        )
    if index == 0:
        return "serve"
    if index == count - 1:
        return (
            INTERIOR_CONTACT_FLIGHT_ROLE if right_boundary_kind == "original_contact" else "final"
        )
    if index == 1:
        return "return"
    return "mid_rally"


def _flight_of(frame: float, boundaries: np.ndarray) -> int | None:
    """Which flight owns a native exposure; boundaries are the contact times."""
    index = int(np.searchsorted(boundaries, float(frame), side="right")) - 1
    if index < 0 or index >= len(boundaries) - 1:
        return None
    return index


def directional_windows(
    native_projection: list[dict],
    events: list[dict],
    boundaries: np.ndarray,
    *,
    horizons: tuple[int, int] = (4, 10),
) -> list[dict]:
    """The promoted event-side windows, each split into the flights it covers.

    A window around a bounce lies inside one flight.  A window around a contact
    lies on one side of it, so its rows normally belong to a single flight too;
    when two contacts fall closer together than a horizon the window can reach
    across a junction, and that is recorded per row rather than mixed into one
    RMS that belongs to no flight.
    """
    training = [row for row in native_projection if row["split"] == "training"]
    windows = []
    for event in events:
        frame = float(event["frame"])
        for horizon_name, horizon in (("short", horizons[0]), ("medium", horizons[1])):
            for direction, (low, high) in (
                ("backward", (frame - horizon, frame)),
                ("forward", (frame, frame + horizon)),
            ):
                rows = [row for row in training if low <= float(row["frame"]) <= high]
                per_flight: dict[int, list[dict]] = {}
                for row in rows:
                    index = _flight_of(float(row["frame"]), boundaries)
                    if index is not None:
                        per_flight.setdefault(index, []).append(row)
                for index, flight_rows in sorted(per_flight.items()):
                    shift_residuals = []
                    for shift in WINDOW_TIME_SHIFTS_FRAMES:
                        errors = [
                            row.get("time_shift_error_px", {}).get(str(shift))
                            for row in flight_rows
                        ]
                        if shift == 0:
                            errors = [float(row["error_px"]) for row in flight_rows]
                        rms = (
                            _rms(errors)
                            if all(value is not None and np.isfinite(value) for value in errors)
                            else None
                        )
                        shift_residuals.append({"model_time_shift_frames": shift, "rms_px": rms})
                    available = [row for row in shift_residuals if row["rms_px"] is not None]
                    best = min(
                        available,
                        key=lambda row: (
                            row["rms_px"],
                            abs(row["model_time_shift_frames"]),
                            row["model_time_shift_frames"],
                        ),
                    )
                    windows.append(
                        {
                            "flight_index": index,
                            "event_type": event["event_type"],
                            "event_frame": frame,
                            "direction": direction,
                            "horizon": horizon_name,
                            "native_training_pictures": len(flight_rows),
                            "rms_px": next(
                                row["rms_px"]
                                for row in shift_residuals
                                if row["model_time_shift_frames"] == 0
                            ),
                            "zero_shift_rms_px": next(
                                row["rms_px"]
                                for row in shift_residuals
                                if row["model_time_shift_frames"] == 0
                            ),
                            "best_time_shift_frames": best["model_time_shift_frames"],
                            "best_shift_rms_px": best["rms_px"],
                            "time_shift_residuals": shift_residuals,
                            "window_spans_more_than_one_flight": len(per_flight) > 1,
                        }
                    )
    return windows


def _bounce_witness_rows(
    modeled: list[dict],
    completion_xyz: list[float] | None,
    supplied: np.ndarray,
    targets: list[dict],
) -> list[dict]:
    """Pair each supplied bounce of one flight with the fitted impact it claims."""
    actual: list[dict | None] = [row for row in modeled[: len(targets)]]
    if len(actual) < len(targets) and completion_xyz is not None:
        actual.append({"frame": None, "x": np.asarray(completion_xyz, float), "from_ending": True})
    actual += [None] * (len(targets) - len(actual))
    rows = []
    for supplied_frame, target, impact in zip(supplied, targets, actual, strict=True):
        witness = np.asarray(target["xyz_m"], float)
        row = {
            "supplied_frame": float(supplied_frame),
            # Owner-approved 2026-09-19 (change D1, in person: "We should accept bounces in
            # range, yes."). The supplied frame above is the midpoint of the label's own
            # frame range; carrying the range itself lets `bounce_count_timing` accept a
            # modelled impact that lands inside the range the bounce was observed over.
            # Absent unless the witness published it, so the midpoint rule stands alone.
            **(
                {"supplied_frame_interval": [float(v) for v in interval]}
                if (interval := target.get("supplied_frame_interval")) is not None
                else {}
            ),
            "witness_xyz_m": witness.tolist(),
            "uncertainty_sigma_m": target.get("uncertainty_sigma_m"),
            "witness_mode": target.get("witness_mode"),
            "ground_covariance_xy_m2": target.get("ground_covariance_xy_m2"),
            "impact_pixel_xy": target.get("impact_pixel_xy"),
            "impact_epoch_witness": target.get("impact_epoch_witness"),
            "incoming_wing": target.get("incoming_wing"),
            "outgoing_wing": target.get("outgoing_wing"),
            "construction_mode": target.get("construction_mode"),
            "construction": target.get("construction"),
            "native_frames": target.get("native_frames"),
            "camera_frames_at_impact": target.get("camera_frames_at_impact"),
            "wing_policy": target.get("wing_policy"),
            "legacy_subframe_witness": target.get("legacy_subframe_witness"),
            "modeled_frame": None,
            "modeled_xyz_m": None,
            "raw_distance_m": None,
            "frame_delta": None,
            "from_terminal_completion": bool(impact.get("from_ending")) if impact else False,
        }
        if impact is not None:
            position = np.asarray(impact["x"], float)
            row["modeled_xyz_m"] = position.tolist()
            row["raw_distance_m"] = float(np.linalg.norm(position[:2] - witness[:2]))
            if impact.get("frame") is not None:
                row["modeled_frame"] = float(impact["frame"])
                row["frame_delta"] = float(impact["frame"]) - float(supplied_frame)
            legacy = target.get("legacy_subframe_witness")
            if legacy is not None:
                legacy_position = np.asarray(legacy["xyz_m"], float)
                row["legacy_raw_distance_m"] = float(
                    np.linalg.norm(position[:2] - legacy_position[:2])
                )
        rows.append(row)
    return rows


def _modelled_frame_in_supplied_interval(entry: dict) -> bool:
    """Does the modelled impact land inside the label's own observed frame range?

    Owner-approved 2026-09-19 (change D1, in person: "We should accept bounces in range,
    yes.").  The midpoint test this is ORed with asks the modelled bounce to sit within
    `bounce_uncertainty_frames` of the CENTRE of the range the bounce was observed over,
    which refuses a bounce the label itself places inside that range whenever the range is
    wider than the allowance.  A modelled frame inside the supplied range is not a timing
    disagreement: it is the label's own answer.

    Strictly additive.  Nothing that passes the midpoint test can fail because of this, and
    a witness without a published range (the default, and every frozen receipt) is judged
    by the midpoint alone.  The shared numerical frame tolerance keeps an endpoint in.
    """
    interval = entry.get("supplied_frame_interval")
    modelled = entry.get("modeled_frame")
    if interval is None or modelled is None or len(interval) != 2:
        return False
    low, high = (float(interval[0]), float(interval[1]))
    if not (np.isfinite(low) and np.isfinite(high) and np.isfinite(float(modelled))):
        return False
    return (
        low - trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
        <= float(modelled)
        <= high + trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
    )


def _flight_queries(start: float, end: float, native: np.ndarray, fps: float) -> np.ndarray:
    """Keep native rows immutable; co-locate only numerical endpoint queries.

    Terminal completion can return an integer epoch a few ulps below its native
    picture. The caller already treats pictures within the contract's numerical
    tolerance as in-flight; the strict trajectory API needs the corresponding
    evaluation query to lie inside its domain. Materially later pictures remain
    passive context, and materially earlier pictures are still an error.
    """
    tolerance = trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
    frames = np.asarray(native, float)
    if np.any(frames < start - tolerance):
        raise ValueError("native flight picture precedes trajectory domain")
    inside = frames[frames <= end + tolerance]
    queries = np.clip(inside, start, end)
    dense = np.linspace(start, end, max(2, int(np.ceil((end - start) * 240 / fps)) + 1))
    return np.unique(np.r_[queries, dense])


def projection_scope_diagnostics(projections: list[dict], start: float, end: float) -> dict:
    """Separate flight and context residuals without changing legacy gate evidence.

    The final flight historically inherits later observations in its aggregate
    reprojection metric. Preserve that metric for comparable frozen scores, but
    expose the trajectory-domain split so a bad aftermath track cannot masquerade
    as an in-flight fitting error (or dilute one).
    """
    tolerance = trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
    groups: dict[str, dict[str, list[float]]] = {
        name: {} for name in ("before_flight", "in_flight", "after_flight")
    }
    for row in projections:
        frame = float(row["frame"])
        scope = (
            "before_flight"
            if frame < start - tolerance
            else "after_flight"
            if frame > end + tolerance
            else "in_flight"
        )
        groups[scope].setdefault(row["split"], []).append(float(row["error_px"]))
    return {
        "schema": "flight_projection_scope_diagnostics_v1",
        "gate_evidence_changed": False,
        "start_frame": float(start),
        "end_frame": float(end),
        "scopes": {
            scope: {
                split: {"pictures": len(errors), "rms_px": _rms(errors)}
                for split, errors in splits.items()
            }
            for scope, splits in groups.items()
        },
    }


def measure(
    scene: model.Scene,
    heldout: model.Scene,
    parameters: np.ndarray,
    bounces: tuple,
    native: tuple,
    axes: np.ndarray,
    targets: list[list[dict]],
    players: list[dict] | None,
    events: list[dict],
    *,
    termination_kind: str,
    duration: float | None = 0.25,
    ending_uncertainty_frames: float = 1.0,
    ending_passive_context_frames: float = 0.0,
    athlete: dict | None = None,
    measurement: dict | None = None,
    apply_terminal_completion: bool = True,
    terminal_rebound_frames: np.ndarray | None = None,
    flight_offset: int = 0,
    flight_total: int | None = None,
    preserve_observation_horizon: bool = False,
    right_contact_player: dict | None = None,
) -> dict:
    """Everything a flight-level verdict needs, with no threshold applied yet.

    ``measurement`` is the published whole-point measurement when it is already
    available; it is reused so the reprojection numbers here are the same
    numbers the promoted report published rather than a second computation.

    ``right_contact_player`` is the separately stored actor of supplied contact
    k on an original-contact scene. It is endpoint evidence only, never a launch
    state; when absent the venue-envelope gate on that contact still applies.
    """
    from cv.pipeline import s6_first_contact_role as contact_role

    scene.validate()
    parameters = np.asarray(parameters, float)
    n = len(scene.contact_frames) - 1
    contact_ending = model.original_contact_boundary(scene)
    if contact_ending != (termination_kind == model.ORIGINAL_CONTACT_TERMINATION_KIND):
        raise ValueError(
            "original-contact termination kind and right boundary must be declared together"
        )
    if contact_ending and terminal_rebound_frames is not None:
        raise ValueError("terminal rebound is not applicable to an original-contact boundary")
    if right_contact_player is not None and not contact_ending:
        raise ValueError("a right-contact actor belongs only to an original-contact boundary")
    if contact_ending and len(native[-1]) and float(np.max(native[-1])) >= scene.contact_frames[-1]:
        raise ValueError("native rows at or beyond supplied contact k belong to the dropped flight")
    if measurement is None:
        measurement = exposure.measure(
            scene,
            heldout,
            bounces,
            native,
            parameters,
            axes,
            duration,
            [],
            termination_kind=termination_kind,
            **(
                {
                    "terminal_rebound_frames": terminal_rebound_frames,
                    "terminal_ground_event": terminal_completion.original_ground_event(
                        scene, bounces, events
                    ),
                }
                if terminal_rebound_frames is not None
                else {}
            ),
        )
    training_rows = [row for row in measurement["native_projection"] if row["split"] == "training"]
    owner_pixels = np.asarray([row["owner"] for row in training_rows], float)
    for shift in WINDOW_TIME_SHIFTS_FRAMES:
        if shift == 0:
            continue
        shifted = exposure.time_shifted_prediction(
            scene,
            parameters,
            axes,
            duration,
            shift,
            termination_kind=termination_kind,
            **({"preserve_observation_horizon": True} if preserve_observation_horizon else {}),
        )
        errors = np.linalg.norm(owner_pixels - shifted, axis=1)
        for row, error in zip(training_rows, errors, strict=True):
            row.setdefault("time_shift_error_px", {})[str(shift)] = (
                None if not np.isfinite(error) else float(error)
            )
    # An unresolved tail has no supplied terminal impact, so the rebound
    # completion path -- which indexes the supplied terminal bounce -- is not
    # applicable to it. The observed-horizon receipt below replaces it.
    if (
        apply_terminal_completion
        and terminal_rebound_frames is not None
        and getattr(scene, "observed_horizon_tail", None) is None
    ):
        rebound_frames = event_constraints.validate_terminal_rebound(
            scene, bounces, terminal_rebound_frames
        )
        terminal_flight = model.chain(scene, parameters)[-1]
        ordinal = len(bounces[-1])
        impact = (
            terminal_flight["bounces"][ordinal - 1]
            if len(terminal_flight["bounces"]) >= ordinal
            else None
        )
        completion = (
            {
                "status": "missing_required_terminal_impact",
                "parameters_changed": False,
            }
            if impact is None
            else {
                "status": "completed",
                "end_frame": float(impact["frame"]),
                "end_xyz": np.asarray(impact["x"], float).tolist(),
                "parameters_changed": False,
                "source": "explicit measured terminal rebound segment",
                "postbounce_labeled_frames": rebound_frames.tolist(),
            }
        )
    elif contact_ending and apply_terminal_completion:
        # Terminal completion abstains on a contact-to-contact prefix: the
        # receipt reports the unchanged endpoint, its incoming state and the
        # contact envelope gate, and classifies no physical point ending.
        completion = terminal_completion.complete(
            scene,
            parameters,
            termination_kind,
            uncertainty_frames=ending_uncertainty_frames,
            last_observation_frame=float(native[-1][-1]),
            duration=duration,
        )
    else:
        completion = (
            terminal_completion.complete(
                scene,
                parameters,
                termination_kind,
                uncertainty_frames=ending_uncertainty_frames,
                last_observation_frame=float(native[-1][-1]),
                passive_context_frames=ending_passive_context_frames,
                native_frames=np.asarray(native[-1], float),
                duration=duration,
                live_contact_frames=[
                    float(row["frame"]) for row in events if row.get("event_type") == "contact"
                ],
            )
            if apply_terminal_completion
            else {"status": "not_a_terminal_flight_group", "parameters_changed": False}
        )
    times = scene.contact_frames.copy()
    ending_completed = completion["status"] == "completed"
    if ending_completed:
        times[-1] = completion["end_frame"]
    returned = replace(scene, contact_frames=times)
    # A point that ended at its ground impact can still have native exposures
    # after it.  They keep their original timestamps and their image residual
    # -- the whole-point projection already scored them -- but they are passive
    # context, so they never extend the scored flight domain.
    tolerance = trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
    passive_frames = tuple(
        np.asarray(f, float)[np.asarray(f, float) > b + tolerance]
        for b, f in zip(times[1:], native, strict=True)
    )
    queries = tuple(
        _flight_queries(a, b, f, scene.fps)
        for a, b, f in zip(times[:-1], times[1:], native, strict=True)
    )
    fitted = model.chain(returned, parameters, query_frames=queries)
    probed = times.copy()
    probed[-1] += trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    terminal_impact_census = None
    try:
        final_impacts = model.chain(
            replace(returned, contact_frames=probed),
            parameters,
            query_frames=tuple(np.asarray([a, b], float) for a, b in zip(probed[:-1], probed[1:])),
        )[-1]["bounces"]
    except BounceCapacityError as error:
        if not (
            times[-1] < error.frame <= probed[-1]
            and error.resolved_impacts == len(fitted[-1]["bounces"]) + 1
        ):
            raise
        # The retained trajectory was measured above. Its tolerance probe can
        # reach an additional impact just outside that domain; do not discard
        # earlier flights or pretend the retained impact list is a full census.
        final_impacts = fitted[-1]["bounces"]
        terminal_impact_census = {
            "status": "unavailable",
            "reason": "bounce_capacity_at_tolerance_probe",
            "retained_end_frame": float(times[-1]),
            "probe_end_frame": float(probed[-1]),
            "guard_impact_frame": error.frame,
            "guard_resolved_impacts": error.resolved_impacts,
            "retained_domain_impact_count": len(final_impacts),
            "tolerance_domain_impact_count": None,
            "complete": False,
        }

    total = n if flight_total is None else int(flight_total)
    boundaries = np.asarray(scene.contact_frames, float)
    windows = directional_windows(measurement["native_projection"], events, boundaries)
    per_flight_windows: dict[int, list[dict]] = {}
    for row in windows:
        per_flight_windows.setdefault(row["flight_index"], []).append(row)
    in_sample_check = "check_in_sample" in measurement["rms_px"]
    residuals: dict[str, dict[int, list[float]]] = {
        "training": {},
        "withheld": {},
        "check_in_sample": {},
    }
    projection_rows: dict[int, list[dict]] = {}
    for row in measurement["native_projection"]:
        index = _flight_of(float(row["frame"]), boundaries)
        if index is None:
            index = n - 1 if float(row["frame"]) >= boundaries[-1] else 0
        residuals[row["split"]].setdefault(index, []).append(float(row["error_px"]))
        projection_rows.setdefault(index, []).append(row)

    unscorable_checks = {}
    for row in measurement.get("unscorable_native_projection", []):
        if row.get("status") != "unscorable" or row.get("reason") != "no_training_wing_direction":
            raise ValueError("unsupported unscorable check projection record")
        index = _flight_of(float(row["frame"]), boundaries)
        if index is None:
            index = n - 1 if float(row["frame"]) >= boundaries[-1] else 0
        unscorable_checks.setdefault(index, []).append(dict(row))

    completion_xyz = completion.get("end_xyz") if ending_completed else None
    rows = []
    endpoints = []
    for i, fit in enumerate(fitted):
        impacts = final_impacts if i == n - 1 else fit["bounces"]
        observed = [
            b
            for b in impacts
            if b["frame"] <= fit["end_frame"] + trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
        ]
        expected_net = (
            np.empty(0)
            if scene.net_hit_frames is None
            else np.asarray(scene.net_hit_frames[i], float)
        )
        net_checks = []
        for hit in fit.get("net_hits", []):
            xyz = np.asarray(hit["x"], float)
            net_checks.append(
                {
                    "frame": float(hit["frame"]),
                    "plane_error_m": float(abs(xyz[1] - 11.885)),
                    "at_or_below_tape": bool(
                        xyz[2] <= float(hit["tape_height_m"]) + model.R_BALL + 0.03
                    ),
                    "in_tape_band": bool(_tape_band(xyz)),
                    "within_net_width": bool(-0.915 <= xyz[0] <= 11.885),
                    "position_continuous": bool(hit.get("position_continuous")),
                }
            )
        transition_frames = {round(float(f), 4) for f in expected_net}
        if scene.terminal_net_tail is not None and i == n - 1:
            transition_frames = {float(hit["frame"]) for hit in fit.get("net_hits", [])}
        net_crossings = [
            {
                **row,
                "at_a_supplied_net_transition": bool(
                    any(abs(row["frame"] - f) <= 1.0 for f in transition_frames)
                ),
            }
            for row in crossings(queries[i], fit["positions"])
        ]
        penetrations = [
            row
            for row in net_crossings
            if row["penetration"]
            and row["ball_surface_clearance_m"] < -0.01
            and not row["at_a_supplied_net_transition"]
        ]
        contact_rows = []
        for end_name, contact_index in (("start", i), ("end", i + 1)):
            right_boundary_end = contact_ending and end_name == "end" and i == n - 1
            terminal_end = (
                end_name == "end" and i + flight_offset == total - 1 and not contact_ending
            )
            if right_boundary_end:
                # Supplied contact k: the frozen endpoint of the last retained
                # flight, scored as a contact with incoming velocity only. The
                # right actor is separate endpoint evidence; without one the
                # venue-envelope gate still applies.
                contact = np.asarray(fit["end_xyz"], float)
                envelope = event_constraints.contact_xy_envelope_evidence(contact)
                incoming = np.asarray(fit["velocities"][-1], float)
                player = right_contact_player
                contact_rows.append(
                    {
                        "end": end_name,
                        "kind": "contact",
                        "right_boundary": model.ORIGINAL_CONTACT_TERMINATION_KIND,
                        # Supplied contact k of the whole retained scene, also
                        # when this is a one-flight sub-scene of it.
                        "contact_index": total,
                        "player": None if player is None else player.get("player"),
                        "player_evidence": (
                            "absent_soft_prior" if player is None else "right_contact_actor"
                        ),
                        "contact_xyz_m": contact.tolist(),
                        "incoming_velocity_mps": incoming.tolist(),
                        "outgoing_velocity_mps": None,
                        # Reported diagnostic only, exactly as at a launching
                        # contact: a closing actor whose court position is
                        # explicitly absent abstains instead of naming a
                        # distance, and keeps its side and stature evidence.
                        "root_to_contact_m": (
                            None
                            if player is None or player_position.root_xy(player) is None
                            else float(
                                np.linalg.norm(
                                    contact[:2] - np.asarray(player["court_centre_xy_m"], float)
                                )
                            )
                        ),
                        **(
                            {}
                            if player is None or player_position.root_xy(player) is not None
                            else {"player_position": player["player_position_evidence"]}
                        ),
                        "stature_m": None if player is None else player.get("stature_m"),
                        "zero_penalty_root_reach_m": None,
                        "inside_envelope": envelope["inside_venue_envelope"],
                        "runback_envelope": envelope,
                        "physical_ending_classified": False,
                    }
                )
                continue
            if terminal_end:
                contact_rows.append(
                    {
                        "end": end_name,
                        "kind": "observed_net_aftermath_end"
                        if scene.terminal_net_tail is not None
                        else "observation_horizon_end"
                        if getattr(scene, "observed_horizon_tail", None) is not None
                        else "ground_ending",
                    }
                )
                continue
            contact = np.asarray(
                fit["positions"][0] if end_name == "start" else fit["end_xyz"], float
            )
            envelope = event_constraints.contact_xy_envelope_evidence(contact)
            if players is None or contact_index >= len(players):
                # A synthetic control has no player roster; the contact is still
                # a contact and still has to sit inside the declared envelope.
                contact_rows.append(
                    {
                        "end": end_name,
                        "kind": "contact",
                        "contact_index": contact_index,
                        "player": None,
                        "contact_xyz_m": contact.tolist(),
                        "root_to_contact_m": None,
                        "stature_m": None,
                        "zero_penalty_root_reach_m": None,
                        "inside_envelope": envelope["inside_venue_envelope"],
                        "runback_envelope": envelope,
                    }
                )
                continue
            player = players[contact_index]
            athlete_row = (athlete or {}).get("contacts", [None] * len(players))[contact_index]
            contact_rows.append(
                {
                    "end": end_name,
                    "kind": "contact",
                    "contact_index": contact_index,
                    "player": player.get("player"),
                    "contact_xyz_m": contact.tolist(),
                    # Reported diagnostic only; an absent court position abstains
                    # here rather than naming a distance that was never measured.
                    "root_to_contact_m": (
                        None
                        if player_position.root_xy(player) is None
                        else float(
                            np.linalg.norm(
                                contact[:2] - np.asarray(player["court_centre_xy_m"], float)
                            )
                        )
                    ),
                    **(
                        {}
                        if player_position.root_xy(player) is not None
                        else {"player_position": player["player_position_evidence"]}
                    ),
                    "stature_m": player.get("stature_m"),
                    "zero_penalty_root_reach_m": (
                        None if athlete_row is None else athlete_row["zero_penalty_root_reach_m"]
                    ),
                    "inside_envelope": envelope["inside_venue_envelope"],
                    "runback_envelope": envelope,
                }
            )
        rows.append(
            {
                "flight_index": i + flight_offset,
                "role": flight_role(
                    i + flight_offset,
                    total,
                    right_boundary_kind=scene.right_boundary_kind,
                    first_contact_role=contact_role.bound_role(players),
                ),
                "start_frame": float(fit["start_frame"]),
                "end_frame": float(fit["end_frame"]),
                "start_xyz_m": np.asarray(fit["start_xyz"], float).tolist(),
                "end_xyz_m": np.asarray(fit["end_xyz"], float).tolist(),
                "native_exposures": int(len(native[i])),
                "training_pictures": int(len(scene.observation_frames[i])),
                "withheld_pictures": 0
                if in_sample_check
                else int(len(heldout.observation_frames[i])),
                "reprojection_rms_px": _rms(residuals["training"].get(i, [])),
                **(
                    {
                        "post_contact_widened_rms_px": _weighted_post_contact_rms(
                            projection_rows.get(i, []), float(expected_net[0])
                        )
                    }
                    if _net_cord.active_mode() == _net_cord.ADMISSIBLE_SET and len(expected_net)
                    else {}
                ),
                **(
                    {
                        "net_cord_response": {
                            **_net_cord.receipt(_net_cord.ADMISSIBLE_SET),
                            "v_out_mps": np.asarray(fit["net_hits"][0]["v_out"], float).tolist(),
                            "outgoing_velocity_mps": np.asarray(
                                fit["net_hits"][0]["v_out"], float
                            ).tolist(),
                            "v_in_mps": np.asarray(fit["net_hits"][0]["v_in"], float).tolist(),
                            "model": "admissible_net_response_v1",
                            "in_tape_band": bool(net_checks[0]["in_tape_band"]),
                            "covariates": _net_cord.contact_covariates(fit["net_hits"][0]["x"]),
                            "spin_law": "incoming_preserved_unconstrained",
                        }
                    }
                    if _net_cord.active_mode() == _net_cord.ADMISSIBLE_SET
                    and fit.get("net_hits")
                    and net_checks
                    else {}
                ),
                **(
                    {
                        "net_cord_response": _net_cord.evidence_witness(
                            _evidence_bound, fit["net_hits"][0] if fit.get("net_hits") else None
                        )
                    }
                    if (
                        _evidence_bound := _net_cord.matching_bound(
                            float(expected_net[0])
                            if len(expected_net)
                            else float((fit.get("net_hits") or [{"frame": -1e9}])[0]["frame"])
                        )
                    )
                    is not None
                    and _net_cord.active_mode() == _net_cord.EVIDENCE_BOUND
                    else {}
                ),
                "projection_scope_diagnostics": projection_scope_diagnostics(
                    projection_rows.get(i, []), fit["start_frame"], fit["end_frame"]
                ),
                "withheld_rms_px": None
                if i in unscorable_checks
                else _rms(residuals["withheld"].get(i, [])),
                **(
                    {"unscorable_check_directions": unscorable_checks[i]}
                    if i in unscorable_checks
                    else {}
                ),
                **(
                    {
                        "check_in_sample_pictures": int(len(heldout.observation_frames[i])),
                        "check_in_sample_rms_px": None
                        if i in unscorable_checks
                        else _rms(residuals["check_in_sample"].get(i, [])),
                        "independent_withheld_pictures": 0,
                    }
                    if in_sample_check
                    else {}
                ),
                "directional_windows": per_flight_windows.get(i, []),
                "maximum_window_rms_px": (
                    max(
                        (
                            w["rms_px"]
                            for w in per_flight_windows.get(i, [])
                            if w["rms_px"] is not None
                        ),
                        default=None,
                    )
                ),
                "bounce_witness": _bounce_witness_rows(
                    observed,
                    completion_xyz if i == n - 1 else None,
                    np.asarray(bounces[i], float),
                    targets[i],
                ),
                "supplied_bounce_count": int(len(bounces[i])),
                "modeled_bounce_count": int(len(observed)),
                **(
                    {
                        "terminal_impact_census": terminal_impact_census,
                        "modeled_bounce_count_scope": "retained_domain_only_incomplete_census",
                    }
                    if i == n - 1 and terminal_impact_census is not None
                    else {}
                ),
                **(
                    {
                        "latent_terminal_ground_count": sum(
                            float(b["frame"])
                            >= scene.terminal_net_tail["interval"][0] - EVENT_TOLERANCE_FRAMES
                            for b in observed
                        ),
                        "terminal_net_tail": completion,
                    }
                    if scene.terminal_net_tail is not None and i == n - 1
                    else {}
                ),
                **(
                    {
                        # Unknown supplied ground count: every modelled impact
                        # after the last supplied contact is a latent modelled
                        # landing, never a source ground witness.
                        "latent_terminal_ground_count": len(
                            horizon_tail.latent_grounds(scene, [{"bounces": observed}])
                        ),
                        "latent_terminal_ground_frames": horizon_tail.latent_grounds(
                            scene, [{"bounces": observed}]
                        ),
                        "latent_ground_origin": "model_latent_not_source_event",
                        "terminal_ground_count": "unknown",
                        "observed_horizon_tail": completion,
                    }
                    if getattr(scene, "observed_horizon_tail", None) is not None and i == n - 1
                    else {}
                ),
                **(
                    {
                        # exit_partial_endings: the picture, not the ground, ends this
                        # flight. An extra modelled impact within one frame of the exit
                        # cannot show as a bounce in the pictures, so it is latent.
                        "latent_terminal_ground_count": sum(
                            float(b["frame"])
                            >= float(fit["end_frame"])
                            - EXIT_LATENT_FRAMES
                            - trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
                            for b in observed[len(bounces[i]) :]
                        ),
                        "latent_ground_origin": "modelled_impact_at_picture_exit",
                    }
                    if i == n - 1
                    and isinstance(scene.supported_ending, dict)
                    and scene.supported_ending.get("partial_flight") is True
                    and scene.supported_ending.get("kind") in ("fov_exit", "last_visible_sample")
                    else {}
                ),
                "minimum_height_m": float(np.asarray(fit["positions"], float)[:, 2].min()),
                "net_crossings": net_crossings,
                "net_penetrations": penetrations,
                "post_ending_passive_exposure_frames": passive_frames[i].tolist(),
                "supplied_net_hit_frames": expected_net.tolist(),
                "modeled_net_hits": net_checks,
                "impact_passivity": [
                    impact_passivity(b, fit["end_frame"], scene.fps) for b in observed
                ],
                "contacts": contact_rows,
                "terminal": i + flight_offset == total - 1 and not contact_ending,
                **(
                    {
                        "supported_ending_kind": scene.supported_ending["kind"],
                        **(
                            {"partial_flight": True}
                            if scene.supported_ending.get("partial_flight") is True
                            else {}
                        ),
                        **(
                            {
                                "supported_ending_evidence": "labelled_boundary",
                                "start_xyz_m": np.asarray(fit["positions"][0], float).tolist(),
                                "start_velocity_mps": np.asarray(
                                    fit["velocities"][0], float
                                ).tolist(),
                                **(
                                    {
                                        "boundary_row_event": scene.supported_ending[
                                            "boundary_row_event"
                                        ],
                                        # Seconds from the fitted end to the row.
                                        "boundary_row_lead_s": (
                                            float(scene.supported_ending["boundary_frame"])
                                            - float(fit["end_frame"])
                                        )
                                        / float(scene.fps),
                                        "end_xyz_m": np.asarray(
                                            fit["positions"][-1], float
                                        ).tolist(),
                                        "end_velocity_mps": np.asarray(
                                            fit["velocities"][-1], float
                                        ).tolist(),
                                    }
                                    if scene.supported_ending.get("boundary_row_event")
                                    in ("bounce", "net_hit")
                                    else {}
                                ),
                            }
                            if scene.supported_ending.get("evidence") == "labelled_boundary"
                            else {}
                        ),
                    }
                    if i == n - 1 and isinstance(scene.supported_ending, dict)
                    else {}
                ),
                **(
                    {"right_boundary": model.ORIGINAL_CONTACT_TERMINATION_KIND}
                    if contact_ending and i == n - 1
                    else {}
                ),
                "velocity_slack": None,
            }
        )
        endpoints.append(
            {
                "start_frame": fit["start_frame"],
                "end_frame": fit["end_frame"],
                "start_xyz": np.asarray(fit["start_xyz"], float).tolist(),
                "end_xyz": np.asarray(fit["end_xyz"], float).tolist(),
                "terminal_end": i == n - 1 and not contact_ending,
            }
        )
    connection = trajectory_contract.trajectory_connection_report(endpoints)
    if contact_ending and apply_terminal_completion:
        ending = {
            "valid": completion.get("valid") is True,
            "required": False,
            "kind": model.ORIGINAL_CONTACT_TERMINATION_KIND,
            "physical_ending_classified": False,
            "ending_semantics": "unresolved",
            "reasons": []
            if completion.get("valid") is True
            else ["right_contact_outside_envelope"],
            "right_contact_frame": completion["right_contact_frame"],
            "right_contact_xyz_m": completion.get("right_contact_xyz_m"),
            "incoming_velocity_mps": completion.get("incoming_velocity_mps"),
            "outgoing_velocity_mps": None,
        }
    elif apply_terminal_completion:
        ending = trajectory_contract.terminal_ground_endpoint_report(termination_kind, endpoints)
    else:
        ending = {"valid": False, "reasons": ["not_a_terminal_flight_group"]}
    return {
        **(
            {contact_role.FIELD: players[0][contact_role.FIELD]}
            if not contact_role.serve_priors_applicable(players)
            else {}
        ),
        "schema": SCHEMA,
        "surface": scene.surface,
        "flight_count": n,
        "flights": rows,
        "terminal_completion": completion,
        "ending_completed": bool(ending_completed),
        "connections": connection,
        "ending": ending,
        "whole_point_rms_px": measurement["rms_px"],
        **(
            {"observation_partition": measurement["observation_partition"]}
            if "observation_partition" in measurement
            else {}
        ),
        "contact_frames": boundaries.tolist(),
        **(
            {
                "right_boundary_kind": model.ORIGINAL_CONTACT_TERMINATION_KIND,
                "complete_original_source": False,
                "ending_semantics": "unresolved",
                "right_contact_player": right_contact_player,
            }
            if contact_ending
            else {}
        ),
    }


#: A flight closed at a labelled boundary row (``boundary_endings``) must leave
#: its contact as a struck shot: toward the other half, this fast horizontally,
#: from a racket a player can reach. A dead ball tapped or rolled after play has
#: ended, or a monocular fit that slid its contact far up the viewing ray, fails.
BOUNDARY_CLOSE_MIN_SPEED_MPS = 5.0
BOUNDARY_CLOSE_MAX_CONTACT_HEIGHT_M = 4.0


def _struck_shot(row: dict) -> bool:
    from cv.pipeline.physics_knot_solver import NET_Y

    _x0, y0, z0 = (float(v) for v in row["start_xyz_m"][:3])
    vx, vy = (float(v) for v in row["start_velocity_mps"][:2])
    toward = vy > 0.0 if y0 < NET_Y else vy < 0.0
    return bool(
        toward
        and math.hypot(vx, vy) >= BOUNDARY_CLOSE_MIN_SPEED_MPS
        and z0 <= BOUNDARY_CLOSE_MAX_CONTACT_HEIGHT_M
    )


#: A flight closed just before an uncertain landing or net row must be arriving
#: there: at the row's time the ball is this low (bounce), or this close to the
#: net plane (net touch). A fit that ends mid-air far from that event placed the
#: ball at the wrong depth along the viewing ray.
BOUNDARY_ROW_BOUNCE_MAX_HEIGHT_M = 0.5
BOUNDARY_ROW_NET_MAX_DISTANCE_M = 2.0


def _reaches_row(row: dict) -> bool:
    from cv.pipeline.physics_knot_solver import NET_Y

    lead = max(0.0, float(row["boundary_row_lead_s"]))
    x, y, z = (float(v) for v in row["end_xyz_m"][:3])
    vx, vy, vz = (float(v) for v in row["end_velocity_mps"][:3])
    if row["boundary_row_event"] == "bounce":
        return bool(z + vz * lead - 4.905 * lead * lead <= BOUNDARY_ROW_BOUNCE_MAX_HEIGHT_M)
    return bool(abs(y + vy * lead - NET_Y) <= BOUNDARY_ROW_NET_MAX_DISTANCE_M)


def _physical_net_stop(row: dict) -> bool:
    """The ending state is on the mesh. A modelled hit is preferred; the endpoint qualifies too."""
    from cv.pipeline.physics_knot_solver import NET_Y, net_tape_height

    def on_mesh(xyz, tape: float) -> bool:
        point = np.asarray(xyz, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return False
        return (
            abs(float(point[1]) - NET_Y) <= 0.05
            and abs(float(point[0]) - 5.485) <= 6.4
            and model.R_BALL - 0.02 <= float(point[2]) <= tape + model.R_BALL + 0.03
        )

    hits = row.get("modeled_net_hits") or []
    if hits:
        return all(
            hit.get("plane_error_m", 1.0) <= 0.05
            and hit.get("within_net_width")
            and hit.get("at_or_below_tape")
            and hit.get("position_continuous")
            for hit in hits
        )
    tape = float(net_tape_height(float(np.asarray(row.get("end_xyz_m", [0.0]), float)[0])))
    return on_mesh(row.get("end_xyz_m"), tape)


def score(
    measured: dict,
    thresholds: Thresholds = Thresholds(),
    *,
    depth_bounds: tuple[float, float] | None = None,
    athlete_prior_mode: str = "stature_pose_soft",
    global_reach_cap_m: float = 1.75,
    extra_flight_checks: dict[int, dict[str, bool]] | None = None,
    ending_kind: str | None = None,
) -> dict:
    """Turn one flight-level measurement into one verdict per flight.

    Every check is the promoted whole-point check restricted to the evidence
    that belongs to this flight.  The point-level verdict is the conjunction,
    so ordinary ground-ending measurements retain the promoted checks. Explicit
    labeled terminal-net measurements substitute their declared ending gate.
    """
    thresholds.validate()
    from cv.pipeline import s6_first_contact_role as contact_role

    # A bound rally origin and a bound unknown origin both refuse the serve
    # depth gate; only the rally receipt also asserts a rally stroke.
    is_rally = not contact_role.serve_priors_applicable(
        [{contact_role.FIELD: measured.get(contact_role.FIELD)}]
    )
    flights = measured["flights"]
    count = len(flights)
    timing_limit_frames = (
        thresholds.grass_bounce_epoch_frames
        if measured.get("surface") == "grass"
        else BOUNCE_WITNESS_TIMING_LIMIT_FRAMES
    )
    serve_witness_covariance_resolved = all(
        entry.get("ground_covariance_xy_m2") is None
        or (
            entry.get("uncertainty_sigma_m") is not None
            and np.isfinite(float(entry["uncertainty_sigma_m"]))
            and float(entry["uncertainty_sigma_m"]) <= BOUNCE_WITNESS_SIGMA_LIMIT_M
        )
        for flight in flights
        if flight.get("role") == "serve"
        for entry in flight["bounce_witness"]
    )
    verdicts = []
    for row in flights:
        checks: dict[str, bool] = {}
        witness = []
        ray_limit = thresholds.bounce_ray_limit_m
        if row.get("role") == "serve" and thresholds.serve_bounce_ray_limit_m is not None:
            # The owner keeps a tighter landing-spot cap on serves even where
            # the timing is relaxed: a serve a yard from its witness is not
            # the serve that was played (2026-09-08).
            ray_limit = thresholds.serve_bounce_ray_limit_m
        for entry in row["bounce_witness"]:
            radius = graded_circle_radius_m(
                entry.get("uncertainty_sigma_m"), thresholds.bounce_circle_floor_m
            )
            raw = entry.get("raw_distance_m")
            gate = None if raw is None else max(0.0, float(raw) - radius)
            sigma = entry.get("uncertainty_sigma_m")
            covariance_required = entry.get("ground_covariance_xy_m2") is not None
            covariance_resolved = bool(
                not covariance_required
                or (
                    sigma is not None
                    and np.isfinite(float(sigma))
                    and float(sigma) <= BOUNCE_WITNESS_SIGMA_LIMIT_M
                )
            )
            # A broad covariance is an honest abstention, not permission to
            # erase the rung's original raw location allowance.  This retains
            # the graded-circle/ray gate while preventing an uncertain ellipse
            # from certifying a different court-depth zone.
            raw_limit = ray_limit + thresholds.bounce_circle_floor_m
            raw_location_supported = bool(raw is not None and float(raw) <= raw_limit)
            legacy = entry.get("legacy_subframe_witness")
            legacy_raw = entry.get("legacy_raw_distance_m")
            legacy_radius = graded_circle_radius_m(
                None if legacy is None else legacy.get("uncertainty_sigma_m"),
                thresholds.bounce_circle_floor_m,
            )
            legacy_gate = (
                None if legacy_raw is None else max(0.0, float(legacy_raw) - legacy_radius)
            )
            legacy_agrees = bool(legacy_gate is not None and legacy_gate <= ray_limit)
            timing = entry.get("impact_epoch_witness") or {}
            timing_delta = timing.get("delta_from_supplied_frames")
            timing_resolved = bool(
                not timing
                or (
                    timing_delta is not None
                    and np.isfinite(float(timing_delta))
                    and abs(float(timing_delta)) <= timing_limit_frames
                )
            )
            # Astra (2026-09-09): a one-sided wing extrapolated to the bounce
            # frame certified a ball that landed at the net foot as 1.3 m short
            # (uso2024qf_pt0001 flight 3).  Only a two-sided reversal may
            # certify a new accept; one-sided witnesses fall back to the
            # frozen construction.
            construction_mode = entry.get("construction_mode")
            # The frozen chord carries no wing metadata and keeps its own rule.
            two_sided = construction_mode in (None, "two_sided_reversal")
            reversal_agrees = bool(
                gate is not None
                and two_sided
                and gate <= ray_limit
                and covariance_resolved
                and serve_witness_covariance_resolved
                and raw_location_supported
                and timing_resolved
            )
            witness.append(
                {
                    **entry,
                    "graded_circle_radius_m": radius,
                    "gate_distance_m": gate,
                    "covariance_sigma_limit_m": BOUNCE_WITNESS_SIGMA_LIMIT_M,
                    "covariance_resolved": covariance_resolved,
                    "raw_distance_limit_m": raw_limit,
                    "raw_location_supported": raw_location_supported,
                    "serve_witness_covariance_resolved": serve_witness_covariance_resolved,
                    "timing_witness_limit_frames": timing_limit_frames,
                    "timing_witness_resolved": timing_resolved,
                    "legacy_graded_circle_radius_m": legacy_radius,
                    "legacy_gate_distance_m": legacy_gate,
                    "legacy_agrees": legacy_agrees,
                    "reversal_agrees": reversal_agrees,
                    "two_sided_reversal": two_sided,
                    "relies_on_reversal_witness": bool(reversal_agrees and not legacy_agrees),
                    "agrees": bool(legacy_agrees or reversal_agrees),
                }
            )
        checks["bounce_rays_agree"] = all(entry["agrees"] for entry in witness)
        checks["bounce_count_timing"] = bool(
            row["modeled_bounce_count"] - row.get("latent_terminal_ground_count", 0)
            == row["supplied_bounce_count"]
            and all(
                entry["frame_delta"] is not None
                and (
                    abs(entry["frame_delta"]) <= thresholds.bounce_uncertainty_frames
                    or _modelled_frame_in_supplied_interval(entry)
                )
                for entry in witness
                if not entry["from_terminal_completion"]
            )
        )
        gate_rms = row.get("post_contact_widened_rms_px", row["reprojection_rms_px"])
        checks["flight_reprojection_supported"] = bool(
            gate_rms is not None and gate_rms <= thresholds.flight_reprojection_rms_limit_px
        )
        selected_windows = []
        for window in row["directional_windows"]:
            allowed = [
                candidate
                for candidate in window["time_shift_residuals"]
                if candidate["rms_px"] is not None
                and abs(candidate["model_time_shift_frames"])
                <= thresholds.directional_time_shift_frames
            ]
            selected = min(
                allowed,
                key=lambda candidate: (
                    candidate["rms_px"],
                    abs(candidate["model_time_shift_frames"]),
                    candidate["model_time_shift_frames"],
                ),
            )
            selected_windows.append(
                {
                    **window,
                    "selected_time_shift_frames": selected["model_time_shift_frames"],
                    "selected_rms_px": selected["rms_px"],
                    "passes_only_with_time_shift": bool(
                        window["zero_shift_rms_px"] > thresholds.directional_rms_limit_px
                        and selected["rms_px"] <= thresholds.directional_rms_limit_px
                    ),
                }
            )
        maximum_window = max(
            (window["selected_rms_px"] for window in selected_windows), default=None
        )
        directional_limit = thresholds.directional_rms_limit_px
        if _net_cord.active_mode() == _net_cord.ADMISSIBLE_SET and row.get(
            "supplied_net_hit_frames"
        ):
            # Evidence-bound does not widen this gate. Widening is what made the
            # free search wander; the pixel set is applied before the fit instead.
            directional_limit = directional_limit * _net_cord.POST_CONTACT_WIDENING
        checks["directional_windows_supported"] = bool(
            maximum_window is not None and maximum_window <= directional_limit
        )
        checks["no_underground_path"] = bool(row["minimum_height_m"] >= model.R_BALL - 1e-3)
        sigma = thresholds.net_clearance_sigma_m
        net_rows: list[dict] = []
        if sigma is None:
            checks["no_net_penetration"] = not row["net_penetrations"]
        else:
            # The clearance margin replaces the binary penetration test. Below
            # minus one sigma the fitted path went through the mesh, which is
            # wrong whatever the label says; inside the band the geometry cannot
            # separate a cord clip from a clearance, so the labeled ending kind
            # decides which of the two neighbours the point is allowed to be.
            for entry in row.get("net_crossings", []):
                if not entry["within_net_width"] or entry["at_a_supplied_net_transition"]:
                    continue
                margin = float(entry["ball_centre_clearance_m"])
                net_rows.append(
                    {
                        "frame": entry["frame"],
                        "court_x_m": entry["xyz"][0],
                        "net_band_height_m": entry["net_band_height_m"],
                        "ball_centre_clearance_m": margin,
                        "sigma_m": float(sigma),
                        "verdict": net_crossing_verdict(margin, float(sigma)),
                    }
                )
            traversed = [entry for entry in net_rows if entry["verdict"] == "into_net"]
            cleared = [entry for entry in net_rows if entry["verdict"] == "cleared"]
            checks["no_net_mesh_traversal"] = not traversed
            if row["terminal"] and net_ending_labeled(ending_kind):
                # A net-ending label requires net contact on the flight that
                # ends the point: a modelled net hit, or a plane crossing the
                # geometry cannot call a clearance. A ball that sails clear
                # contradicts the label and is not accepted by omission.
                checks["net_ending_supported"] = bool(row["modeled_net_hits"] or not cleared)
        supplied_net = row["supplied_net_hit_frames"]
        checks["supplied_net_transitions_met"] = bool(
            len(row["modeled_net_hits"]) == len(supplied_net)
            and all(
                (
                    row["terminal_net_tail"]["original_contract"]["interval"][0]
                    - EVENT_TOLERANCE_FRAMES
                    <= hit["frame"]
                    <= row["terminal_net_tail"]["original_contract"]["interval"][1]
                    + EVENT_TOLERANCE_FRAMES
                    if row.get("terminal_net_tail")
                    else abs(hit["frame"] - supplied_net[j]) <= thresholds.bounce_uncertainty_frames
                )
                and hit["plane_error_m"] <= 0.05
                and (
                    hit["in_tape_band"]
                    if _net_cord.uses_tape_band(hit["frame"])
                    else hit["at_or_below_tape"]
                )
                and hit["within_net_width"]
                and hit["position_continuous"]
                for j, hit in enumerate(row["modeled_net_hits"])
            )
        )
        checks["passive_court_impacts"] = all(entry["valid"] for entry in row["impact_passivity"])
        contacts = [entry for entry in row["contacts"] if entry.get("kind") == "contact"]
        checks["contacts_inside_declared_search_envelope"] = all(
            entry["inside_envelope"] for entry in contacts
        )
        if row.get("right_boundary") == model.ORIGINAL_CONTACT_TERMINATION_KIND:
            # The explicit right contact is a contact, not an ending: its own
            # named gate makes the abstained ground completion visible.
            right = [
                entry
                for entry in contacts
                if entry.get("right_boundary") == model.ORIGINAL_CONTACT_TERMINATION_KIND
            ]
            checks["original_contact_endpoint_inside_envelope"] = bool(
                len(right) == 1 and right[0]["inside_envelope"]
            )
        if row["flight_index"] == 0 and not is_rally and depth_bounds is not None and contacts:
            depth = contacts[0]["contact_xyz_m"][1]
            checks["serve_depth_plausible"] = bool(depth_bounds[0] <= depth <= depth_bounds[1])
        if athlete_prior_mode == "global_hard_caps":
            checks["contact_reaches_plausible"] = all(
                entry["root_to_contact_m"] is not None
                and entry["root_to_contact_m"] <= global_reach_cap_m
                for entry in contacts
            )
        if row["terminal"]:
            net_ending = measured.get("terminal_net_impact")
            ending_kind_row = row.get("supported_ending_kind")
            if ending_kind_row == "net_stop":
                # The net is the ending. A ground impact is not required, and a
                # cord that continues is not given this kind.
                checks["physical_net_stop"] = _physical_net_stop(row)
            elif ending_kind_row in {
                "camera_cut",
                "held_camera",
                "point_end",
                "dead_ball",
                # Present only with exit_partial_endings on (a picture exit is partial).
                "fov_exit",
                "last_visible_sample",
            }:
                # Partial on purpose. Image, bounce and direction checks above
                # still apply; the endpoint type stays the boundary that stopped it.
                checks["partial_flight_labelled"] = row.get("partial_flight") is True
                if row.get("supported_ending_evidence") == "labelled_boundary":
                    checks["boundary_close_struck_shot"] = _struck_shot(row)
                    if row.get("boundary_row_event") in ("bounce", "net_hit"):
                        checks["boundary_close_reaches_row"] = _reaches_row(row)
            elif row.get("terminal_net_tail") is not None:
                checks["physical_net_interaction_with_observed_tail"] = (
                    row["terminal_net_tail"]["valid"] is True
                )
            elif row.get("observed_horizon_tail") is not None:
                # No physical ending is claimed and no ground count is asserted.
                # The existing image/passivity/continuity/direction/net checks
                # above still decide whether this partial flight is supported.
                checks["observed_horizon_tail_supported"] = (
                    row["observed_horizon_tail"]["valid"] is True
                )
            elif net_ending is not None:
                checks["physical_net_ending"] = net_ending["valid"] is True
            else:
                checks["physical_ground_ending"] = bool(
                    row.get("_ending_completed", measured["ending_completed"])
                    and row.get("_ending", measured["ending"])["valid"]
                )
        slack = row.get("velocity_slack") or {}
        gap = float(slack.get("junction_position_gap_m") or 0.0)
        tolerance = float(slack.get("junction_gap_tolerance_m") or 1e-3)
        checks["continuity_into_neighbours"] = bool(
            gap <= tolerance
            and float(slack.get("applied_velocity_change_mps") or 0.0)
            <= thresholds.velocity_slack_mps + 1e-9
        )
        checks.update((extra_flight_checks or {}).get(row["flight_index"], {}))
        # Owner 2026-09-24: waive the directional window only when it is the
        # flight's only failure. This does not raise the ordinary limit, so a
        # flight that also failed a bounce check stays failed.
        waiver = thresholds.directional_only_waiver_px
        if (
            waiver is not None
            and checks.get("directional_windows_supported") is False
            and maximum_window is not None
            and maximum_window <= waiver
            and [name for name, passed in checks.items() if not passed]
            == ["directional_windows_supported"]
        ):
            checks["directional_windows_supported"] = True
        if row.get("unscorable_check_directions"):
            checks["check_projection_directions_supported"] = False
        if "terminal_impact_census" in row:
            checks["terminal_impact_census_supported"] = False
            checks["bounce_count_timing"] = False
        failures = [name for name, passed in checks.items() if not passed]
        verdicts.append(
            {
                "flight_index": row["flight_index"],
                "role": row["role"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "accepted": not failures,
                "checks": checks,
                "failures": failures,
                **(
                    {"terminal_impact_census": row["terminal_impact_census"]}
                    if row.get("terminal_impact_census") is not None
                    else {}
                ),
                "bounce_witness": witness,
                "reprojection_rms_px": row["reprojection_rms_px"],
                **(
                    {"post_contact_widened_rms_px": row["post_contact_widened_rms_px"]}
                    if "post_contact_widened_rms_px" in row
                    else {}
                ),
                **(
                    {"net_cord_response": row["net_cord_response"]}
                    if row.get("net_cord_response")
                    else {}
                ),
                **(
                    {"projection_scope_diagnostics": row["projection_scope_diagnostics"]}
                    if "projection_scope_diagnostics" in row
                    else {}
                ),
                "withheld_rms_px": row["withheld_rms_px"],
                **(
                    {
                        "check_in_sample_rms_px": row["check_in_sample_rms_px"],
                        "check_in_sample_pictures": row["check_in_sample_pictures"],
                        "independent_withheld_pictures": 0,
                    }
                    if "check_in_sample_rms_px" in row
                    else {}
                ),
                "maximum_window_rms_px": maximum_window,
                "maximum_window_rms_px_zero_shift": row["maximum_window_rms_px"],
                "directional_windows": selected_windows,
                **(
                    {"unscorable_check_directions": row["unscorable_check_directions"]}
                    if row.get("unscorable_check_directions")
                    else {}
                ),
                "timing_only_window_count": sum(
                    window["passes_only_with_time_shift"] for window in selected_windows
                ),
                "location_error_after_time_shift": bool(
                    maximum_window is not None
                    and maximum_window > thresholds.directional_rms_limit_px
                ),
                "net_clearance": net_rows,
                "post_ending_passive_exposure_frames": row.get(
                    "post_ending_passive_exposure_frames", []
                ),
                "training_pictures": row["training_pictures"],
                "native_exposures": row["native_exposures"],
                "contacts": contacts,
                "velocity_slack": row.get("velocity_slack"),
            }
        )
    accepted = [row["flight_index"] for row in verdicts if row["accepted"]]
    rejected = [row["flight_index"] for row in verdicts if not row["accepted"]]
    gaps = []
    for row in verdicts:
        if row["accepted"]:
            continue
        index = row["flight_index"]
        if gaps and gaps[-1]["last_flight_index"] == index - 1:
            gaps[-1]["last_flight_index"] = index
            gaps[-1]["end_frame"] = row["end_frame"]
            gaps[-1]["failures"] = sorted(set(gaps[-1]["failures"] + row["failures"]))
        else:
            gaps.append(
                {
                    "first_flight_index": index,
                    "last_flight_index": index,
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "failures": list(row["failures"]),
                }
            )
    structure_valid = bool(measured["connections"]["valid"])
    complete = bool(len(accepted) == count and structure_valid)
    contact_ending = measured.get("right_boundary_kind") == model.ORIGINAL_CONTACT_TERMINATION_KIND
    return {
        "schema": SCHEMA,
        "thresholds": thresholds.as_dict(),
        "athlete_prior_mode": athlete_prior_mode,
        **(
            {
                # ``complete_point`` below is the retained-prefix geometry
                # verdict only. Source completeness is a separate, false, fact.
                "right_boundary_kind": model.ORIGINAL_CONTACT_TERMINATION_KIND,
                "complete_original_source": False,
                "ending_semantics": "unresolved",
                "physical_ending_classified": False,
            }
            if contact_ending
            else {}
        ),
        **(
            {"observation_partition": measured["observation_partition"]}
            if "observation_partition" in measured
            else {}
        ),
        "flight_count": count,
        "accepted_flight_count": len(accepted),
        "accepted_flight_indices": accepted,
        "rejected_flight_indices": rejected,
        "gaps": gaps,
        "complete_point": complete,
        "partial_point": bool(accepted and not complete),
        "structural_connections_valid": structure_valid,
        "flights": verdicts,
        "failure_counts": {
            name: sum(1 for row in verdicts if name in row["failures"])
            for name in sorted({name for row in verdicts for name in row["failures"]})
        },
        "labeled_ending_kind": ending_kind,
        "complete_real_point_accepted": False,
        "automatic_inference_eligible": False,
    }


#: A bounce sitting exactly on its gate limit costs the local solve the same as
#: one picture sitting exactly on the directional window limit.  This is a
#: declared weighting between two measured quantities, not a calibration.
BOUNCE_RESIDUAL_PX_PER_M = 16.0 / 0.9144


def _split_parameters(
    parameters: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the promoted connected parameter vector into its named blocks."""
    parameters = np.asarray(parameters, float)
    rebound = parameters[-2:]
    body = parameters[:-2]
    if body.shape != (3 + 6 * n,):
        raise ValueError("connected parameters with one explicit spin triple per flight required")
    return body[:3], body[3 : 3 + 3 * n].reshape(n, 3), body[3 + 3 * n :].reshape(n, 3), rebound


def _sub_scene(
    scene: model.Scene, index: int, boundaries: Sequence[float], spin_rows: np.ndarray
) -> model.Scene:
    """One flight of an existing scene, optionally split at an unobserved frame."""
    frames = np.asarray(scene.observation_frames[index], float)
    bounds = np.asarray(boundaries, float)
    groups = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mask = (frames >= a) & (frames < b)
        if b == bounds[-1]:
            mask |= frames == b
        groups.append(np.flatnonzero(mask))
    if any(not len(group) for group in groups):
        raise ValueError("every sub-flight needs at least one native picture")
    net = None
    if scene.net_hit_frames is not None:
        supplied = np.asarray(scene.net_hit_frames[index], float)
        net = tuple(
            supplied[(supplied > a) & (supplied < b)] for a, b in zip(bounds[:-1], bounds[1:])
        )
    tail = scene.observed_horizon_tail if index == len(scene.pixels) - 1 else None
    if tail is not None:
        low = max(float(bounds[-2]), float(tail["interval"][0]))
        tail = {
            **tail,
            "interval": [low, float(tail["observation_horizon"])],
            "native_tail_frames": [frame for frame in tail["native_tail_frames"] if frame > low],
        }
        if "supplied_ground_frames" in tail:
            tail["supplied_ground_frames"] = [
                frame for frame in tail["supplied_ground_frames"] if frame > float(bounds[-2])
            ]
            tail["supplied_ground_count"] = len(tail["supplied_ground_frames"])
    # Only the last flight of an original-contact scene ends at contact k; an
    # interior flight's sub-scene keeps the ordinary supplied-end grammar.
    right_boundary_kind = (
        scene.right_boundary_kind if index == len(scene.pixels) - 1 else "supplied_end"
    )
    sub = model.Scene(
        observed_horizon_tail=tail,
        right_boundary_kind=right_boundary_kind,
        contact_frames=bounds,
        observation_frames=tuple(frames[group] for group in groups),
        cameras=tuple(np.asarray(scene.cameras[index])[group] for group in groups),
        pixels=tuple(np.asarray(scene.pixels[index])[group] for group in groups),
        camera_distortion=(
            None
            if scene.camera_distortion is None
            else tuple(np.asarray(scene.camera_distortion[index])[group] for group in groups)
        ),
        spin_parameters=spin_rows,
        fps=scene.fps,
        surface=scene.surface,
        dynamics=scene.dynamics,
        bounce_profile=scene.bounce_profile,
        rebound_mode=scene.rebound_mode,
        net_hit_frames=net,
    )
    sub.validate()
    return sub, [group for group in groups]


def _interior_split_frame(scene: model.Scene, index: int) -> float | None:
    """The midpoint of this flight's widest unobserved stretch, or nothing."""
    frames = np.asarray(scene.observation_frames[index], float)
    if len(frames) < 2:
        return None
    gaps = np.diff(frames)
    widest = int(np.argmax(gaps))
    if gaps[widest] < 2:
        return None
    candidate = float(np.floor((frames[widest] + frames[widest + 1]) / 2.0))
    if candidate <= frames[0] or candidate >= frames[-1] or candidate in set(frames.tolist()):
        candidate = float(frames[widest]) + 1.0
    if candidate <= frames[0] or candidate >= frames[-1] or candidate in set(frames.tolist()):
        return None
    return candidate


def relax_flight(
    scene: model.Scene,
    heldout: model.Scene,
    parameters: np.ndarray,
    index: int,
    start_xyz: np.ndarray,
    axes_group: np.ndarray,
    bounces_group: np.ndarray,
    native_group: np.ndarray,
    targets_group: list[dict],
    players_pair: list[dict],
    events: list[dict],
    *,
    slack_mps: float,
    duration: float,
    termination_kind: str,
    ending_uncertainty_frames: float,
    ending_passive_context_frames: float,
    terminal: bool,
    flight_total: int,
    athlete: dict | None,
    split_frame: float | None,
    max_nfev: int,
    thresholds: Thresholds,
    depth_bounds: tuple[float, float] | None,
    athlete_prior_mode: str,
    right_contact_player: dict | None = None,
) -> dict | None:
    """Refit one flight to its own pictures inside a bounded velocity change.

    The flight keeps the joint fit's start position, so nothing here moves a
    contact.  Only the outgoing velocity moves, by at most ``slack_mps`` per
    axis, optionally with a second bounded change at one unobserved interior
    frame.  The displaced endpoint is reported as the junction gap it is.
    """
    n = len(scene.contact_frames) - 1
    _, velocities, spins, rebound = _split_parameters(parameters, n)
    bounds = [float(scene.contact_frames[index]), float(scene.contact_frames[index + 1])]
    if split_frame is not None:
        bounds = [bounds[0], float(split_frame), bounds[1]]
    pieces = len(bounds) - 1
    spin_rows = np.tile(spins[index], (pieces, 1))
    try:
        sub, groups = _sub_scene(scene, index, bounds, spin_rows)
        sub_heldout, _ = _sub_scene(heldout, index, bounds, spin_rows)
    except ValueError:
        return None
    target_pixels = np.concatenate(sub.pixels)
    sub_axes = np.concatenate([np.asarray(axes_group)[group] for group in groups])
    sub_native = tuple(
        np.asarray(native_group, float)[
            (np.asarray(native_group, float) >= a)
            & (
                (np.asarray(native_group, float) < b)
                | (np.asarray(native_group, float) == bounds[-1])
            )
        ]
        for a, b in zip(bounds[:-1], bounds[1:])
    )
    if any(not len(group) for group in sub_native):
        return None
    sub_bounces = tuple(
        np.asarray(bounces_group, float)[
            (np.asarray(bounces_group, float) > a) & (np.asarray(bounces_group, float) <= b)
        ]
        for a, b in zip(bounds[:-1], bounds[1:])
    )
    if any(len(group) > 2 for group in sub_bounces):
        return None
    sub_targets = []
    for group in sub_bounces:
        sub_targets.append(
            [
                next(
                    target
                    for target in targets_group
                    if abs(float(target["event_frame"]) - float(frame)) < 1e-6
                )
                for frame in group
            ]
        )
    kind = termination_kind if terminal else "terminal_bounce"

    def build(delta: np.ndarray) -> np.ndarray:
        outgoing = [np.asarray(velocities[index], float) + delta[:3]]
        if pieces == 2:
            first = model.chain(
                _sub_scene(scene, index, bounds[:2], spin_rows[:1])[0],
                np.r_[start_xyz, outgoing[0], spin_rows[0], rebound],
                query_frames=(np.asarray([bounds[0], bounds[1]], float),),
            )[0]
            outgoing.append(np.asarray(first["velocities"][-1], float) + delta[3:])
        return np.r_[start_xyz, np.concatenate(outgoing), spin_rows.ravel(), rebound]

    def residual(delta: np.ndarray) -> np.ndarray:
        try:
            candidate = build(np.asarray(delta, float))
            image = (
                exposure.prediction(sub, candidate, sub_axes, duration, termination_kind=kind)
                - target_pixels
            ).ravel()
            witness = []
            fitted = model.chain(sub, candidate)
            for piece, group in enumerate(sub_targets):
                impacts = fitted[piece]["bounces"]
                for j, target in enumerate(group):
                    if j >= len(impacts):
                        witness.append(BOUNCE_RESIDUAL_PX_PER_M * 10.0)
                        continue
                    radius = graded_circle_radius_m(
                        target.get("uncertainty_sigma_m"), thresholds.bounce_circle_floor_m
                    )
                    raw = float(
                        np.linalg.norm(
                            np.asarray(impacts[j]["x"], float)[:2]
                            - np.asarray(target["xyz_m"], float)[:2]
                        )
                    )
                    witness.append(BOUNCE_RESIDUAL_PX_PER_M * max(0.0, raw - radius))
            return np.r_[image, witness]
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(2 * len(target_pixels) + sum(map(len, sub_targets)), 1e6)

    width = 3 * pieces
    solved = least_squares(
        residual,
        np.zeros(width),
        bounds=(np.full(width, -slack_mps), np.full(width, slack_mps)),
        max_nfev=max_nfev,
        xtol=1e-8,
    )
    delta = np.asarray(solved.x, float)
    candidate = build(delta)
    try:
        measured = measure(
            sub,
            sub_heldout,
            candidate,
            sub_bounces,
            sub_native,
            sub_axes,
            sub_targets,
            players_pair,
            events,
            termination_kind=kind,
            duration=duration,
            ending_uncertainty_frames=ending_uncertainty_frames,
            ending_passive_context_frames=ending_passive_context_frames,
            athlete=athlete,
            apply_terminal_completion=terminal,
            flight_offset=index,
            flight_total=flight_total,
            **(
                {"right_contact_player": right_contact_player}
                if terminal and model.original_contact_boundary(scene)
                else {}
            ),
        )
    except (ValueError, FloatingPointError, OverflowError):
        return None
    row = measured["flights"][-1]
    # A flight split at an unobserved interior frame is still one flight of the
    # point; the split is an internal degree of freedom, not a new rally shot.
    contact_ending = model.original_contact_boundary(scene)
    row["flight_index"] = index
    from cv.pipeline import s6_first_contact_role as contact_role

    row["role"] = flight_role(
        index,
        flight_total,
        right_boundary_kind=scene.right_boundary_kind,
        first_contact_role=contact_role.bound_role(players_pair),
    )
    row["terminal"] = index == flight_total - 1 and not contact_ending
    row["start_xyz_m"] = np.asarray(start_xyz, float).tolist()
    row["start_frame"] = bounds[0]
    row["contacts"] = measured["flights"][0]["contacts"][:1] + row["contacts"][1:]
    row["velocity_slack"] = {
        "allowed_per_axis_mps": float(slack_mps),
        "junction_change_mps": np.round(delta[:3], 9).tolist(),
        "interior_split_frame": None if pieces == 1 else float(bounds[1]),
        "interior_change_mps": None if pieces == 1 else np.round(delta[3:], 9).tolist(),
        "applied_velocity_change_mps": float(np.max(np.abs(delta))) if len(delta) else 0.0,
        "solver_cost": float(solved.cost),
        "sub_flight_pieces": pieces,
    }
    row["interior_flights"] = [
        {
            "start_frame": entry["start_frame"],
            "end_frame": entry["end_frame"],
            "minimum_height_m": entry["minimum_height_m"],
        }
        for entry in measured["flights"]
    ]
    if pieces == 2:
        row["projection_scope_diagnostics"] = {
            "schema": "piecewise_flight_projection_scope_diagnostics_v1",
            "gate_evidence_changed": False,
            "pieces": [entry["projection_scope_diagnostics"] for entry in measured["flights"]],
        }
        # Sum before mutating the aliased final row's count below.
        if row.get("terminal_impact_census") is not None:
            row["terminal_impact_census"] = {
                **row["terminal_impact_census"],
                "probe_subflight_retained_impact_count": row["terminal_impact_census"][
                    "retained_domain_impact_count"
                ],
                "retained_domain_impact_count": sum(
                    entry["modeled_bounce_count"] for entry in measured["flights"]
                ),
            }
        for name in ("net_crossings", "net_penetrations", "impact_passivity", "modeled_net_hits"):
            row[name] = [item for entry in measured["flights"] for item in entry[name]]
        row["minimum_height_m"] = min(entry["minimum_height_m"] for entry in measured["flights"])
        row["bounce_witness"] = [
            item for entry in measured["flights"] for item in entry["bounce_witness"]
        ]
        row["supplied_bounce_count"] = sum(
            entry["supplied_bounce_count"] for entry in measured["flights"]
        )
        row["modeled_bounce_count"] = sum(
            entry["modeled_bounce_count"] for entry in measured["flights"]
        )
        row["supplied_net_hit_frames"] = [
            item for entry in measured["flights"] for item in entry["supplied_net_hit_frames"]
        ]
        errors = [
            value
            for entry in measured["flights"]
            for value in ([entry["reprojection_rms_px"]] if entry["reprojection_rms_px"] else [])
        ]
        row["reprojection_rms_px"] = _rms(errors) if errors else row["reprojection_rms_px"]
        row["maximum_window_rms_px"] = max(
            (
                entry["maximum_window_rms_px"]
                for entry in measured["flights"]
                if entry["maximum_window_rms_px"] is not None
            ),
            default=None,
        )
    row["_terminal_completion"] = measured["terminal_completion"]
    row["_ending"] = measured["ending"]
    row["_ending_completed"] = measured["ending_completed"]
    return row


def _apply_junction_geometry(rows: list[dict], fps: float, slack_mps: float) -> None:
    """Record what the allowed velocity change did to each junction, in metres.

    A velocity change of at most ``slack_mps`` per axis, applied at the start of
    a flight of duration ``T`` seconds, can move that flight's endpoint by at
    most ``sqrt(3) * slack * T``.  That bound is the junction tolerance; the
    measured gap is reported beside it so the displacement is never implicit.
    """
    for i, row in enumerate(rows):
        seconds = (float(row["end_frame"]) - float(row["start_frame"])) / float(fps)
        gap = 0.0
        if i + 1 < len(rows):
            gap = float(
                np.linalg.norm(
                    np.asarray(row["end_xyz_m"], float)
                    - np.asarray(rows[i + 1]["start_xyz_m"], float)
                )
            )
        slack = dict(row.get("velocity_slack") or {})
        slack.setdefault("allowed_per_axis_mps", float(slack_mps))
        slack.setdefault("applied_velocity_change_mps", 0.0)
        slack["flight_seconds"] = seconds
        slack["junction_position_gap_m"] = gap
        slack["junction_gap_tolerance_m"] = max(
            1e-3, float(slack["allowed_per_axis_mps"]) * np.sqrt(3.0) * seconds
        )
        row["velocity_slack"] = slack


def evaluate(
    scene: model.Scene,
    heldout: model.Scene,
    parameters: np.ndarray,
    bounces: tuple,
    native: tuple,
    axes: np.ndarray,
    targets: list[list[dict]],
    players: list[dict],
    events: list[dict],
    *,
    termination_kind: str,
    thresholds: Thresholds = Thresholds(),
    duration: float = 0.25,
    athlete: dict | None = None,
    depth_bounds: tuple[float, float] | None = None,
    athlete_prior_mode: str = "stature_pose_soft",
    measurement: dict | None = None,
    base: dict | None = None,
    max_nfev: int = 40,
    extra_flight_checks: dict[int, dict[str, bool]] | None = None,
    ending_kind: str | None = None,
) -> dict:
    """One connected fit in, one verdict per flight out, at one declared rung."""
    thresholds.validate()
    if base is None:
        base = measure(
            scene,
            heldout,
            parameters,
            bounces,
            native,
            axes,
            targets,
            players,
            events,
            termination_kind=termination_kind,
            duration=duration,
            ending_uncertainty_frames=thresholds.ending_uncertainty_frames,
            ending_passive_context_frames=thresholds.ending_passive_context_frames,
            athlete=athlete,
            measurement=measurement,
        )
    rows = [dict(row) for row in base["flights"]]
    relaxation = {"attempted": 0, "replaced": 0, "failed": 0, "rejected_after_relaxation": 0}
    if thresholds.velocity_slack_mps > 0:
        n = len(rows)
        offsets = np.r_[0, np.cumsum([len(g) for g in scene.observation_frames])]
        axes = np.asarray(axes, float)
        strict = Thresholds(**{**thresholds.as_dict(), "velocity_slack_mps": 0.0})
        _apply_junction_geometry(rows, scene.fps, 0.0)
        joint = score(
            {**base, "flights": rows},
            strict,
            depth_bounds=depth_bounds,
            athlete_prior_mode=athlete_prior_mode,
            extra_flight_checks=extra_flight_checks,
            ending_kind=ending_kind,
        )
        for index in range(n):
            # Relaxation is strictly additive: a flight the joint fit already
            # satisfies is never re-fitted, so an allowed velocity change can
            # only add accepted flights, never trade one away.
            if joint["flights"][index]["accepted"]:
                continue
            relaxation["attempted"] += 1
            group_axes = axes[offsets[index] : offsets[index + 1]]
            candidates = []
            for split in (None, _interior_split_frame(scene, index)):
                if split is None and candidates:
                    continue
                row = relax_flight(
                    scene,
                    heldout,
                    parameters,
                    index,
                    np.asarray(rows[index]["start_xyz_m"], float),
                    group_axes,
                    np.asarray(bounces[index], float),
                    np.asarray(native[index], float),
                    targets[index],
                    players,
                    events,
                    slack_mps=thresholds.velocity_slack_mps,
                    duration=duration,
                    termination_kind=termination_kind,
                    ending_uncertainty_frames=thresholds.ending_uncertainty_frames,
                    ending_passive_context_frames=thresholds.ending_passive_context_frames,
                    terminal=index == n - 1,
                    flight_total=n,
                    athlete=athlete,
                    split_frame=split,
                    max_nfev=max_nfev,
                    thresholds=thresholds,
                    depth_bounds=depth_bounds,
                    athlete_prior_mode=athlete_prior_mode,
                    right_contact_player=base.get("right_contact_player"),
                )
                if row is not None:
                    candidates.append(row)
            if not candidates:
                relaxation["failed"] += 1
                continue
            accepted = []
            for candidate in candidates:
                trial = [*rows[:index], candidate, *rows[index + 1 :]]
                _apply_junction_geometry(trial, scene.fps, thresholds.velocity_slack_mps)
                verdict = score(
                    {**base, "flights": trial},
                    thresholds,
                    depth_bounds=depth_bounds,
                    athlete_prior_mode=athlete_prior_mode,
                    extra_flight_checks=extra_flight_checks,
                    ending_kind=ending_kind,
                )
                if verdict["flights"][index]["accepted"]:
                    accepted.append(candidate)
            if not accepted:
                relaxation["rejected_after_relaxation"] += 1
                continue
            rows[index] = min(accepted, key=lambda row: row["velocity_slack"]["solver_cost"])
            relaxation["replaced"] += 1
    _apply_junction_geometry(rows, scene.fps, thresholds.velocity_slack_mps)
    relaxed_endpoints = [
        {
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "start_xyz": row["start_xyz_m"],
            "end_xyz": row["end_xyz_m"],
            "terminal_end": row["terminal"],
        }
        for row in rows
    ]
    measured = {
        **base,
        "flights": rows,
        "relaxed_connections": trajectory_contract.trajectory_connection_report(relaxed_endpoints),
        "relaxation": relaxation,
    }
    verdict = score(
        measured,
        thresholds,
        depth_bounds=depth_bounds,
        athlete_prior_mode=athlete_prior_mode,
        extra_flight_checks=extra_flight_checks,
        ending_kind=ending_kind,
    )
    verdict["maximum_junction_gap_m"] = max(
        (float(row["velocity_slack"]["junction_position_gap_m"]) for row in rows[:-1]),
        default=0.0,
    )
    verdict["relaxation"] = relaxation
    verdict["ending_completed"] = base["ending_completed"]
    verdict["terminal_completion_reason"] = base["terminal_completion"].get("reason")
    return verdict
