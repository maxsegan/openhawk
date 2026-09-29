"""Can forward-linked physical flights avoid post-fit contact seam repair?

Status: synthetic mechanism control only, not a production fitter. The first
position and each racket's outgoing velocity are variables; every subsequent
position is the previous flight's simulated endpoint, never a second variable.
Spin and contact times are explicit fixed inputs in this first control. It reuses
the existing four-substep drag/Magnus/bounce simulator, whose accuracy and impact
law remain separate validation questions. No net collision or ending certificate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from cv.pipeline.provenance import file_record, git_record
from cv.pipeline import rich_ball_physics
from cv.pipeline.rich_ball_physics import MAX_SIMULATED_BOUNCES, R_BALL, simulate_with_spin

if TYPE_CHECKING:
    from cv.experiments.connected_shooting.flight_cache import FlightCache
    from cv.experiments.connected_shooting.reach_constraints import ReachConstraints


#: What the last entry of ``contact_frames`` means.  ``supplied_end`` is the
#: existing grammar: the scene ends at a supplied physical ending or an explicit
#: tail contract and the last flight may be completed to a ground impact.
#: ``original_contact`` is a contact-to-contact prefix: the last boundary is the
#: supplied contact k itself, the last flight is an ordinary interior flight,
#: and no terminal completion, rebound, net or horizon grammar applies.
RIGHT_BOUNDARY_KINDS = ("supplied_end", "original_contact")
ORIGINAL_CONTACT_TERMINATION_KIND = "original_contact"


def original_contact_boundary(scene) -> bool:
    """Read the explicit right-boundary role, never the position of a flight."""
    kind = getattr(scene, "right_boundary_kind", "supplied_end")
    if kind not in RIGHT_BOUNDARY_KINDS:
        raise ValueError("explicit supported right boundary kind required")
    return kind == "original_contact"


@dataclass(frozen=True)
class Scene:
    contact_frames: np.ndarray
    observation_frames: tuple[np.ndarray, ...]
    cameras: tuple[np.ndarray, ...]
    pixels: tuple[np.ndarray, ...]
    spin_parameters: np.ndarray
    fps: float
    surface: str
    dynamics: str = "legacy_cross"
    bounce_profile: str = "nominal"
    rebound_mode: str = "fixed"
    camera_distortion: tuple[np.ndarray, ...] | None = None
    bounce_regime_override: tuple[int, int, str] | None = None
    net_hit_frames: tuple[np.ndarray, ...] | None = None
    parameterization: str = "single_shooting"
    terminal_net_tail: dict | None = None
    observed_horizon_tail: dict | None = None
    observation_partition: str = "fifth_frame_withheld"
    ground_settling: bool = False
    bounce_witness_observation_policy: str = (
        "legacy_fit_frames_including_existing_terminal_activation"
    )
    right_boundary_kind: str = "supplied_end"
    #: Net stop or a partial span (camera cut, held camera, point end, dead ball).
    #: Absent on every flight the supported-endings switch did not close.
    supported_ending: dict | None = None

    def validate(self, *, allow_empty_observations: bool = False) -> None:
        from cv.experiments.connected_shooting.observation_partition import (
            validate,
            validate_witness_policy,
        )

        if type(allow_empty_observations) is not bool:
            raise ValueError("empty observation permission must be explicit boolean")
        contact_ending = original_contact_boundary(self)
        if contact_ending and (
            self.terminal_net_tail is not None
            or self.observed_horizon_tail is not None
            or self.ground_settling
        ):
            raise ValueError(
                "an original-contact right boundary carries no terminal net, horizon or "
                "ground-settling grammar"
            )
        validate(self.observation_partition)
        validate_witness_policy(self.bounce_witness_observation_policy, self.observation_partition)
        if type(self.ground_settling) is not bool or (
            self.ground_settling and self.dynamics != "measured_240hz"
        ):
            raise ValueError("explicit measured-dynamics ground-settling policy required")
        if self.parameterization not in {"single_shooting", "shared_contact_states"}:
            raise ValueError("explicit supported shooting parameterization required")
        if self.dynamics not in {"legacy_cross", "measured_240hz"}:
            raise ValueError("explicit supported dynamics profile required")
        if self.rebound_mode not in {"fixed", "point_scales"} or (
            self.rebound_mode == "point_scales" and self.dynamics != "measured_240hz"
        ):
            raise ValueError("point rebound corrections require measured dynamics")
        from cv.experiments.connected_shooting.measured_dynamics import BOUNCE_PROFILES

        if self.bounce_profile not in BOUNCE_PROFILES or (
            self.dynamics != "measured_240hz" and self.bounce_profile != "nominal"
        ):
            raise ValueError("supported bounce profile requires measured dynamics")
        times = np.asarray(self.contact_frames, float)
        n = len(times) - 1
        override = self.bounce_regime_override
        if override is not None and (
            self.dynamics != "measured_240hz"
            or len(override) != 3
            or not isinstance(override[0], int)
            or not isinstance(override[1], int)
            or not 0 <= override[0] < n
            or override[1] < 0
            or (not self.ground_settling and override[1] >= MAX_SIMULATED_BOUNCES)
            or override[2] not in {"slide", "grip"}
        ):
            raise ValueError("one explicit measured flight/bounce/regime surrogate required")
        if times.ndim != 1 or n < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("finite strictly ordered physical contact boundaries required")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("positive native cadence required")
        if not (len(self.observation_frames) == len(self.cameras) == len(self.pixels) == n):
            raise ValueError("one observation/camera/pixel group per flight required")
        if self.camera_distortion is not None and (
            len(self.camera_distortion) != n
            or any(
                np.shape(row) != (len(frames), 3) or not np.isfinite(row).all()
                for row, frames in zip(self.camera_distortion, self.observation_frames, strict=True)
            )
        ):
            raise ValueError("finite native radial-camera groups required")
        if np.shape(self.spin_parameters) != (n, 3) or not np.isfinite(self.spin_parameters).all():
            raise ValueError("three explicit fixed spin parameters per flight required")
        if self.net_hit_frames is not None:
            if len(self.net_hit_frames) != n or self.dynamics != "measured_240hz":
                raise ValueError("one measured-dynamics net-hit group per flight required")
            for group, start, end in zip(self.net_hit_frames, times[:-1], times[1:], strict=True):
                values = np.asarray(group, float)
                if (
                    values.ndim != 1
                    or len(values) > 1
                    or not np.isfinite(values).all()
                    or np.any(values <= start)
                    or np.any(values >= end)
                ):
                    raise ValueError("at most one finite interior net transition per flight")
        if self.terminal_net_tail is not None:
            bounds = np.asarray(self.terminal_net_tail["interval"], float)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or not times[-2] < bounds[0] <= bounds[1] < times[-1]
                or self.terminal_net_tail.get("kind") != "net_stop"
                or self.terminal_net_tail.get("latent_ground_epochs_supplied") is not False
            ):
                raise ValueError(
                    "original terminal-net interval and separate observed horizon required"
                )
        ending = self.supported_ending
        if ending is not None and (
            not isinstance(ending, dict)
            or not isinstance(ending.get("kind"), str)
            or not ending["kind"]
            or ending.get("partial_flight") not in (None, True, False)
        ):
            raise ValueError("explicit supported ending receipt required")
        if self.observed_horizon_tail is not None:
            bounds = np.asarray(self.observed_horizon_tail["interval"], float)
            if (
                self.terminal_net_tail is not None
                or bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or not times[-2] <= bounds[0] <= bounds[1]
                or abs(bounds[1] - times[-1]) > 1e-6
                or self.observed_horizon_tail.get("kind") != "observed_horizon"
                or self.observed_horizon_tail.get("physical_ending") is not None
                or self.observed_horizon_tail.get("terminal_ground_count") != "unknown"
                or self.observed_horizon_tail.get("latent_ground_epochs_supplied") is not False
            ):
                raise ValueError(
                    "unresolved observed tail requires its own horizon and unknown ground count"
                )
        for i, (frames, P, xy) in enumerate(
            zip(self.observation_frames, self.cameras, self.pixels, strict=True)
        ):
            if (
                np.ndim(frames) != 1
                or (not len(frames) and not allow_empty_observations)
                or not np.isfinite(frames).all()
                or np.any(frames != np.rint(frames))
                or np.any(np.diff(frames) <= 0)
                or (len(frames) and min(frames) < times[i])
                or (len(frames) and max(frames) > times[i + 1])
                # The last retained flight keeps the original half-open interior
                # membership: a row at contact k belongs to the dropped flight k.
                or (contact_ending and i == n - 1 and len(frames) and max(frames) >= times[-1])
                or np.shape(P) != (len(frames), 3, 4)
                or not np.isfinite(P).all()
                or np.shape(xy) != (len(frames), 2)
                or not np.isfinite(xy).all()
            ):
                raise ValueError("unique in-flight native observations and finite cameras required")


def chain(
    scene: Scene,
    parameters: np.ndarray,
    *,
    query_frames: tuple[np.ndarray, ...] | None = None,
    simulation_cache: FlightCache | None = None,
) -> list[dict]:
    """Forward-link existing dynamics; a racket changes velocity, not position."""
    from cv.experiments.connected_shooting import ground_contact, passive_bounce

    n = len(scene.contact_frames) - 1
    shared_blocks = shared_parameter_slices(scene)
    shared_size = (
        shared_blocks["rebound_scales"].stop
        if scene.rebound_mode == "point_scales"
        else shared_blocks["spins"].stop
    )
    if scene.parameterization == "shared_contact_states" or np.shape(parameters) == (shared_size,):
        return shared_contact_chain(
            replace(scene, parameterization="shared_contact_states"),
            parameters,
            query_frames=query_frames,
            simulation_cache=simulation_cache,
        )
    parameters = np.asarray(parameters, float)
    rebound_scales = (1.0, 1.0)
    if scene.rebound_mode == "point_scales":
        if scene.dynamics != "measured_240hz" or parameters.shape not in {
            (5 + 3 * n,),
            (5 + 6 * n,),
        }:
            raise ValueError(
                "two explicit point rebound parameters required with measured dynamics"
            )
        rebound_scales, parameters = parameters[-2:], parameters[:-2]
    elif scene.rebound_mode != "fixed":
        raise ValueError("explicit supported rebound mode required")
    if parameters.shape not in {(3 + 3 * n,), (3 + 6 * n,)} or not np.isfinite(parameters).all():
        raise ValueError("one initial position and one outgoing velocity per flight required")
    spin = (
        parameters[3 + 3 * n :].reshape(n, 3)
        if len(parameters) == 3 + 6 * n
        else scene.spin_parameters
    )
    # Fractional queries sample an existing trajectory for evaluation; they are
    # never substituted for native pictures in Scene or in the image objective.
    queries = scene.observation_frames if query_frames is None else query_frames
    if len(queries) != n:
        raise ValueError("one query group per flight required")
    position = parameters[:3].copy()
    flights = []
    for i in range(n):
        start, end = scene.contact_frames[i : i + 2]
        requested = np.asarray(queries[i], float)
        if (
            requested.ndim != 1
            or not len(requested)
            or not np.isfinite(requested).all()
            or np.any(np.diff(requested) <= 0)
            or requested[0] < start
            or requested[-1] > end
        ):
            raise ValueError("ordered finite in-flight trajectory queries required")
        net_frames = (
            np.empty(0)
            if scene.net_hit_frames is None
            else np.asarray(scene.net_hit_frames[i], float)
        )
        query = np.unique(np.r_[start, requested, net_frames, end])
        theta = np.r_[position, parameters[3 + 3 * i : 6 + 3 * i], spin[i]]
        if scene.dynamics == "legacy_cross":
            if scene.bounce_profile != "nominal":
                raise ValueError("bounce perturbations require measured dynamics")
            simulator = simulate_with_spin
            kwargs = {}
        elif scene.dynamics == "measured_240hz":
            if len(net_frames):
                from cv.experiments.connected_shooting.net_collision import simulate

                kwargs = {"net_frame": float(net_frames[0])}
            else:
                from cv.experiments.connected_shooting.measured_dynamics import simulate

                kwargs = {}
            simulator = simulate
            kwargs.update(
                {"bounce_profile": scene.bounce_profile, "rebound_scales": rebound_scales}
            )
            if scene.ground_settling or ground_contact.active():
                kwargs["ground_settling"] = scene.ground_settling
            if scene.bounce_regime_override is not None and scene.bounce_regime_override[0] == i:
                kwargs["bounce_regime_override"] = scene.bounce_regime_override[1:]
        else:
            raise ValueError("explicit supported dynamics profile required")
        if simulation_cache is None or len(net_frames):
            simulated = simulator(theta, float(start), query, scene.fps, scene.surface, **kwargs)
        else:
            simulated = simulation_cache.simulate(
                i, simulator, theta, float(start), query, scene.fps, scene.surface, **kwargs
            )
        x, v, w, bounces, *transition_rows = simulated
        net_hits = transition_rows[0] if transition_rows else []
        impact_capacity = (
            passive_bounce.impact_capacity(MAX_SIMULATED_BOUNCES)
            if scene.dynamics == "measured_240hz"
            else MAX_SIMULATED_BOUNCES
        )
        # The measured settling integrator resolves impacts to the finite
        # horizon, even when their apex remains above the contact threshold.
        if not scene.ground_settling and len(bounces) >= impact_capacity:
            raise ValueError("bounce cap reached; ground-clamped continuation is unsupported")
        if not all(np.isfinite(value).all() for value in (x, v, w)):
            raise ValueError("nonfinite connected propagation")
        observed = np.searchsorted(query, requested)
        flights.append(
            {
                "start_frame": float(start),
                "end_frame": float(end),
                "start_xyz": position.copy(),
                "end_xyz": x[-1].copy(),
                "positions": x[observed],
                "velocities": v[observed],
                "bounces": bounces,
                "net_hits": net_hits,
            }
        )
        if (
            scene.ground_settling
            and not len(net_frames)
            and ground_contact.qualifies(theta[:3], theta[3:6])
        ):
            flights[-1]["initial_normal_contact"] = ground_contact.transition_record(theta[3:6]) | {
                "frame": float(start),
                "physical_bounce_invented": False,
            }
        position = x[-1].copy()
    return flights


def shared_parameter_slices(scene: Scene) -> dict[str, slice]:
    """Name the explicit-contact parameter blocks used by multiple shooting."""
    n = len(scene.contact_frames) - 1
    contact_stop = 3 * (n + 1)
    velocity_stop = contact_stop + 3 * n
    spin_stop = velocity_stop + 3 * n
    return {
        "contacts": slice(0, contact_stop),
        "velocities": slice(contact_stop, velocity_stop),
        "spins": slice(velocity_stop, spin_stop),
        "rebound_scales": slice(spin_stop, spin_stop + 2),
    }


def shared_contact_seed(scene: Scene, parameters: np.ndarray) -> np.ndarray:
    """Lift one connected single-shooting state into explicit shared contacts.

    Existing shared vectors pass through byte for byte. New contact states are
    the single-shooting flight endpoints, so enabling the arm starts from the
    exact trajectory the active initializer supplied and adds no seam.
    """
    n = len(scene.contact_frames) - 1
    values = np.asarray(parameters, float)
    blocks = shared_parameter_slices(scene)
    expected = (
        blocks["rebound_scales"].stop
        if scene.rebound_mode == "point_scales"
        else blocks["spins"].stop
    )
    if values.shape == (expected,):
        return values.copy()
    single_expected = 3 + 6 * n + (2 if scene.rebound_mode == "point_scales" else 0)
    if values.shape != (single_expected,) or not np.isfinite(values).all():
        raise ValueError("finite fitted-spin single or shared parameter vector required")
    single_scene = replace(scene, parameterization="single_shooting")
    flights = chain(single_scene, values)
    contacts = np.r_[flights[0]["start_xyz"], *(flight["end_xyz"] for flight in flights)]
    velocity_start = 3
    spin_start = velocity_start + 3 * n
    return np.r_[
        contacts,
        values[velocity_start:spin_start],
        values[spin_start : spin_start + 3 * n],
        *(values[-2:] if scene.rebound_mode == "point_scales" else ()),
    ]


def shared_to_single_parameters(scene: Scene, parameters: np.ndarray) -> np.ndarray:
    """Export a shared fit through the legacy acceptance parameter interface."""
    values = np.asarray(parameters, float)
    blocks = shared_parameter_slices(scene)
    expected = (
        blocks["rebound_scales"].stop
        if scene.rebound_mode == "point_scales"
        else blocks["spins"].stop
    )
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError("finite shared parameter vector required")
    contacts = values[blocks["contacts"]]
    return np.r_[
        contacts[:3],
        values[blocks["velocities"]],
        values[blocks["spins"]],
        *(values[blocks["rebound_scales"]] if scene.rebound_mode == "point_scales" else ()),
    ]


def shared_contact_chain(
    scene: Scene,
    parameters: np.ndarray,
    *,
    query_frames: tuple[np.ndarray, ...] | None = None,
    simulation_cache: FlightCache | None = None,
) -> list[dict]:
    """Propagate independent flights from exact shared contact parameters."""
    from cv.experiments.connected_shooting import ground_contact, passive_bounce

    scene.validate()
    n = len(scene.contact_frames) - 1
    blocks = shared_parameter_slices(scene)
    expected = (
        blocks["rebound_scales"].stop
        if scene.rebound_mode == "point_scales"
        else blocks["spins"].stop
    )
    values = np.asarray(parameters, float)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError("one shared XYZ per boundary and one velocity/spin per flight required")
    contacts = values[blocks["contacts"]].reshape(n + 1, 3)
    velocities = values[blocks["velocities"]].reshape(n, 3)
    spins = values[blocks["spins"]].reshape(n, 3)
    rebound_scales = (
        values[blocks["rebound_scales"]] if scene.rebound_mode == "point_scales" else np.ones(2)
    )
    queries = scene.observation_frames if query_frames is None else query_frames
    if len(queries) != n:
        raise ValueError("one query group per flight required")
    flights = []
    for i in range(n):
        start, end = scene.contact_frames[i : i + 2]
        requested = np.asarray(queries[i], float)
        if (
            requested.ndim != 1
            or not len(requested)
            or not np.isfinite(requested).all()
            or np.any(np.diff(requested) <= 0)
            or requested[0] < start
            or requested[-1] > end
        ):
            raise ValueError("ordered finite in-flight trajectory queries required")
        net_frames = (
            np.empty(0)
            if scene.net_hit_frames is None
            else np.asarray(scene.net_hit_frames[i], float)
        )
        query = np.unique(np.r_[start, requested, net_frames, end])
        theta = np.r_[contacts[i], velocities[i], spins[i]]
        if scene.dynamics == "legacy_cross":
            if scene.bounce_profile != "nominal":
                raise ValueError("bounce perturbations require measured dynamics")
            simulator = simulate_with_spin
            kwargs = {}
        elif scene.dynamics == "measured_240hz":
            if len(net_frames):
                from cv.experiments.connected_shooting.net_collision import simulate

                kwargs = {"net_frame": float(net_frames[0])}
            else:
                from cv.experiments.connected_shooting.measured_dynamics import simulate

                kwargs = {}
            simulator = simulate
            kwargs.update(
                {"bounce_profile": scene.bounce_profile, "rebound_scales": rebound_scales}
            )
            if scene.ground_settling or ground_contact.active():
                kwargs["ground_settling"] = scene.ground_settling
            if scene.bounce_regime_override is not None and scene.bounce_regime_override[0] == i:
                kwargs["bounce_regime_override"] = scene.bounce_regime_override[1:]
        else:
            raise ValueError("explicit supported dynamics profile required")
        if simulation_cache is None or len(net_frames):
            simulated = simulator(theta, float(start), query, scene.fps, scene.surface, **kwargs)
        else:
            simulated = simulation_cache.simulate(
                i, simulator, theta, float(start), query, scene.fps, scene.surface, **kwargs
            )
        x, v, w, bounces, *transition_rows = simulated
        impact_capacity = (
            passive_bounce.impact_capacity(MAX_SIMULATED_BOUNCES)
            if scene.dynamics == "measured_240hz"
            else MAX_SIMULATED_BOUNCES
        )
        # The measured settling integrator resolves impacts to the finite
        # horizon, even when their apex remains above the contact threshold.
        if not scene.ground_settling and len(bounces) >= impact_capacity:
            raise ValueError("bounce cap reached; ground-clamped continuation is unsupported")
        if not all(np.isfinite(value).all() for value in (x, v, w)):
            raise ValueError("nonfinite multiple-shooting propagation")
        observed = np.searchsorted(query, requested)
        flights.append(
            {
                "start_frame": float(start),
                "end_frame": float(end),
                "start_xyz": contacts[i].copy(),
                "end_xyz": x[-1].copy(),
                "shared_end_xyz": contacts[i + 1].copy(),
                "positions": x[observed],
                "velocities": v[observed],
                "bounces": bounces,
                "net_hits": transition_rows[0] if transition_rows else [],
            }
        )
        if (
            scene.ground_settling
            and not len(net_frames)
            and ground_contact.qualifies(theta[:3], theta[3:6])
        ):
            flights[-1]["initial_normal_contact"] = ground_contact.transition_record(theta[3:6]) | {
                "frame": float(start),
                "physical_bounce_invented": False,
            }
    return flights


def shared_contact_gaps(flights: list[dict]) -> np.ndarray:
    """Return physical endpoint minus the next exact shared state, one row per flight."""
    if not flights or any("shared_end_xyz" not in flight for flight in flights):
        raise ValueError("explicit shared-end states required")
    return np.asarray([flight["end_xyz"] - flight["shared_end_xyz"] for flight in flights], float)


def image_residual(scene: Scene, parameters: np.ndarray) -> np.ndarray:
    return projected_residual(scene, chain(scene, parameters))


def projected_residual(scene: Scene, flights: list[dict]) -> np.ndarray:
    from cv.experiments.connected_shooting.camera_geometry import project

    residuals = []
    for i, (flight, P, target) in enumerate(zip(flights, scene.cameras, scene.pixels, strict=True)):
        radial = None if scene.camera_distortion is None else scene.camera_distortion[i]
        residuals.append((project(P, flight["positions"], radial) - target).ravel())
    return np.concatenate(residuals)


def fit(
    scene: Scene,
    initial: np.ndarray,
    *,
    max_nfev: int = 60,
    optimize_spin: bool = False,
    oracle_positions: tuple[np.ndarray, ...] | None = None,
    bounce_frames: tuple[np.ndarray, ...] | None = None,
    bounce_uncertainty_frames: float = 1.0,
    contact_reach_constraints: ReachConstraints | None = None,
    recover_bounce_regimes: bool = False,
    bounce_regime_strategy: str = "perturb",
    rebound_prior_scale: float | None = None,
    warm_start_rebound: bool = False,
    retain_direct_rebound: bool = False,
    memoize_flights: bool = False,
    net_clearance_scale_m: float | None = None,
    terminal_last_observation_frame: float | None = None,
) -> dict:
    """Research fit; explicit oracle XYZ conditioning is never automatic inference."""
    scene.validate()
    if retain_direct_rebound and not warm_start_rebound:
        raise ValueError("direct rebound retention requires warm-start search")
    if warm_start_rebound and (
        scene.rebound_mode != "point_scales" or scene.bounce_regime_override is not None
    ):
        raise ValueError("rebound warm start requires unforced point-scale dynamics")
    if rebound_prior_scale is not None and (
        scene.rebound_mode != "point_scales"
        or not np.isfinite(rebound_prior_scale)
        or rebound_prior_scale <= 0
    ):
        raise ValueError("rebound prior requires point scales and a finite positive scale")
    if recover_bounce_regimes and (scene.dynamics != "measured_240hz" or not optimize_spin):
        raise ValueError("regime recovery requires measured dynamics and fitted spin")
    if bounce_regime_strategy not in {"perturb", "fixed_branch"} or (
        bounce_regime_strategy != "perturb" and not recover_bounce_regimes
    ):
        raise ValueError("explicit supported regime recovery strategy required")
    if recover_bounce_regimes and scene.bounce_regime_override is not None:
        raise ValueError("recovery must start from the unforced physical model")
    from cv.experiments.connected_shooting import event_constraints

    cache_options = {}
    evidence_options = {}
    if net_clearance_scale_m is not None:
        from cv.experiments.connected_shooting import net_constraints

        net_constraints.validate(scene, net_clearance_scale_m, bounce_frames)
        evidence_options["net_clearance_scale_m"] = net_clearance_scale_m
    if terminal_last_observation_frame is not None:
        event_constraints.validate_terminal_observation(
            scene, bounce_frames, terminal_last_observation_frame
        )
        evidence_options["terminal_last_observation_frame"] = terminal_last_observation_frame
    simulation_cache = None
    if memoize_flights:
        from cv.experiments.connected_shooting.flight_cache import FlightCache

        simulation_cache = FlightCache()
        cache_options["simulation_cache"] = simulation_cache

    if bounce_frames is not None:
        bounce_frames = event_constraints.validate(scene, bounce_frames, bounce_uncertainty_frames)
    if contact_reach_constraints is not None:
        contact_reach_constraints.validate(scene)
    n = len(scene.contact_frames) - 1
    lower = np.r_[[-10.0, -15.0, R_BALL], np.full(3 * n, -75.0)]
    upper = np.r_[[21.0, 40.0, 12.0], np.full(3 * n, 75.0)]
    if optimize_spin:
        lower, upper = np.r_[lower, np.full(3 * n, -6.0)], np.r_[upper, np.full(3 * n, 6.0)]
    if scene.rebound_mode == "point_scales":
        lower, upper = np.r_[lower, [0.8, 0.8]], np.r_[upper, [1.2, 1.2]]
    if oracle_positions is not None and (
        len(oracle_positions) != n
        or any(
            np.shape(x) != (len(f), 3) or not np.isfinite(x).all()
            for x, f in zip(oracle_positions, scene.observation_frames, strict=True)
        )
    ):
        raise ValueError("explicit finite oracle XYZ at each native observation required")
    seed = np.asarray(initial, float)
    if (
        seed.shape != lower.shape
        or not np.isfinite(seed).all()
        or np.any(seed <= lower)
        or np.any(seed >= upper)
    ):
        raise ValueError("explicit finite interior seed required")
    if warm_start_rebound and not np.array_equal(seed[-2:], [1.0, 1.0]):
        raise ValueError("rebound warm start requires unit initial corrections")

    def residual(parameters):
        if bounce_frames is not None:
            flights, physical, _ = event_constraints.evaluate(
                scene,
                parameters,
                bounce_frames,
                bounce_uncertainty_frames,
                **cache_options,
                **evidence_options,
            )
        else:
            flights, physical = chain(scene, parameters, **cache_options), np.empty(0)
        if rebound_prior_scale is not None:
            physical = np.r_[physical, (parameters[-2:] - 1.0) / rebound_prior_scale]
        # 1 cm per residual unit, stated in every oracle report. Robust loss
        # then transitions at 2 cm; no claim that these are image measurements.
        image = (
            projected_residual(scene, flights)
            if oracle_positions is None
            else np.concatenate(
                [
                    ((flight["positions"] - xyz) / 0.01).ravel()
                    for flight, xyz in zip(flights, oracle_positions, strict=True)
                ]
            )
        )
        if contact_reach_constraints is not None:
            return np.r_[image, physical, contact_reach_constraints.evaluate(flights)["residuals"]]
        return (
            image
            if bounce_frames is None and rebound_prior_scale is None
            else np.r_[image, physical]
        )

    def prior_receipt(parameters):
        return (
            None
            if rebound_prior_scale is None
            else {
                "center": [1.0, 1.0],
                "scale": rebound_prior_scale,
                "loss": "quadratic_not_robustified",
                "residuals": ((parameters[-2:] - 1.0) / rebound_prior_scale).tolist(),
                "cost": float(0.5 * np.sum(((parameters[-2:] - 1.0) / rebound_prior_scale) ** 2)),
                "scope": "experimental regularization, not calibrated physical uncertainty",
            }
        )

    if warm_start_rebound:
        from cv.experiments.connected_shooting import nested_rebound

        loss = event_constraints.mixed_loss(
            (2 if oracle_positions is None else 3) * sum(map(len, scene.observation_frames))
        )

        def solve_phase(phase_scene, parameters):
            return fit(
                phase_scene,
                parameters,
                max_nfev=max_nfev,
                optimize_spin=optimize_spin,
                oracle_positions=oracle_positions,
                bounce_frames=bounce_frames,
                bounce_uncertainty_frames=bounce_uncertainty_frames,
                contact_reach_constraints=contact_reach_constraints,
                recover_bounce_regimes=recover_bounce_regimes,
                bounce_regime_strategy=bounce_regime_strategy,
                rebound_prior_scale=rebound_prior_scale
                if phase_scene.rebound_mode == "point_scales"
                else None,
                **({"memoize_flights": True} if memoize_flights else {}),
                **evidence_options,
            )

        return nested_rebound.refine(
            scene,
            seed,
            solve_phase,
            lambda parameters: float(2 * np.sum(loss((residual(parameters) / 2) ** 2)[0])),
            prior_receipt,
            (lower, upper),
            retain_direct=retain_direct_rebound,
        )

    initial_error = residual(seed)
    calls = 0

    def objective(parameters):
        nonlocal calls
        calls += 1
        try:
            result = residual(parameters)
            return result if np.isfinite(result).all() else np.full_like(initial_error, 1e6)
        except (ValueError, FloatingPointError, OverflowError):
            return np.full_like(initial_error, 1e6)

    result = least_squares(
        objective,
        seed,
        bounds=(lower, upper),
        loss="soft_l1"
        if bounce_frames is None
        and contact_reach_constraints is None
        and rebound_prior_scale is None
        else event_constraints.mixed_loss(
            (2 if oracle_positions is None else 3) * sum(map(len, scene.observation_frames))
        ),
        f_scale=2.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    flights = chain(scene, result.x)
    fitted = {
        "status": "synthetic_mechanism_only_not_accepted_trajectory",
        "parameters": result.x,
        "conditioning": "oracle_native_xyz" if oracle_positions is not None else "native_pixels",
        "optimized_spin": optimize_spin,
        "rebound_scale_factors": result.x[-2:].tolist()
        if scene.rebound_mode == "point_scales"
        else [1.0, 1.0],
        "rebound_prior_evidence": prior_receipt(result.x),
        "flights": flights,
        "contact_reach_evidence": None
        if contact_reach_constraints is None
        else contact_reach_constraints.evaluate(flights),
        "bounce_window_evidence": None
        if bounce_frames is None
        else {
            "uncertainty_frames": bounce_uncertainty_frames,
            "time_scale_frames": event_constraints.TIME_SCALE_FRAMES,
            "missing_ground_scale_m": event_constraints.MISSING_GROUND_SCALE_M,
            "residual_loss": "quadratic_not_robustified",
            "final": event_constraints.evaluate(
                scene, result.x, bounce_frames, bounce_uncertainty_frames, **evidence_options
            )[2],
        },
        "initial_pixel_rms": float(np.sqrt(np.mean(image_residual(scene, seed) ** 2))),
        "final_pixel_rms": float(np.sqrt(np.mean(image_residual(scene, result.x) ** 2))),
        "optimizer_success": bool(result.success),
        "optimizer_evidence": {
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "function_evaluations": int(result.nfev),
            "jacobian_evaluations": int(result.njev),
        },
        "regime_recovery_evidence": None,
        "surrogate_bounce_override": scene.bounce_regime_override,
        "objective_calls": calls,
        **(
            {
                "flight_cache_evidence": {
                    **simulation_cache.receipt(),
                    "scope": "selected local optimizer only, not all recovery/search phases",
                }
            }
            if simulation_cache is not None
            else {}
        ),
        "junction_gaps_m": [
            float(np.linalg.norm(a["end_xyz"] - b["start_xyz"]))
            for a, b in zip(flights, flights[1:])
        ],
    }
    if recover_bounce_regimes:
        from cv.experiments.connected_shooting import regime_recovery

        def fit_candidate(parameters, override=None):
            return fit(
                scene if override is None else replace(scene, bounce_regime_override=override),
                parameters,
                max_nfev=max_nfev,
                optimize_spin=optimize_spin,
                oracle_positions=oracle_positions,
                bounce_frames=bounce_frames,
                bounce_uncertainty_frames=bounce_uncertainty_frames,
                contact_reach_constraints=contact_reach_constraints,
                rebound_prior_scale=rebound_prior_scale,
                **({"memoize_flights": True} if memoize_flights else {}),
                **evidence_options,
            )

        if bounce_regime_strategy == "fixed_branch":
            fitted = regime_recovery.refine_fixed_branches(scene, fitted, fit_candidate)
        else:
            fitted = regime_recovery.refine(scene, fitted, fit_candidate)
    return fitted


def control() -> tuple[Scene, np.ndarray]:
    """Two short airborne arcs, one camera, known simulator; not independent truth."""
    times = np.array([1.0, 13.0, 26.0])
    frames = (np.arange(1, 14, dtype=float), np.arange(14, 27, dtype=float))
    camera = np.array([[1000, 320, 0, 0], [0, 240, -1000, 10000], [0, 1, 0, 20]], float)
    cameras = tuple(np.repeat(camera[None], len(f), axis=0) for f in frames)
    truth = np.array([5.0, 8.0, 6.0, 2.0, 15.0, 3.0, -3.0, -12.0, 4.0])
    scene = Scene(
        times,
        frames,
        cameras,
        tuple(np.zeros((len(f), 2)) for f in frames),
        np.zeros((2, 3)),
        25.0,
        "hard",
    )
    values = image_residual(scene, truth)
    pixels = tuple(part.reshape(-1, 2) for part in np.split(values, [2 * len(frames[0])]))
    return Scene(times, frames, cameras, pixels, scene.spin_parameters, 25.0, "hard"), truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-output",
        type=Path,
        required=True,
        help="new JSON for the generated same-simulator two-flight control only",
    )
    args = parser.parse_args()
    if args.control_output.exists():
        raise FileExistsError(args.control_output)
    scene, truth = control()
    result = fit(scene, truth + np.array([0.2, -0.3, 0.1, 0.5, -0.4, 0.2, -0.3, 0.4, -0.2]))
    report = {k: v for k, v in result.items() if k not in {"parameters", "flights"}}
    report.update(
        {
            "schema": "connected_shooting_mechanism_control_v1",
            "code": git_record(Path(__file__).resolve().parents[3]),
            "implementation": file_record(Path(__file__)),
            "camera_projection": file_record(Path(__file__).with_name("camera_geometry.py")),
            "simulator": file_record(Path(rich_ball_physics.__file__)),
            "configuration": {
                "fps": scene.fps,
                "surface": scene.surface,
                "contact_frames": scene.contact_frames.tolist(),
                "observation_counts": [len(f) for f in scene.observation_frames],
                "fixed_spin_parameters": scene.spin_parameters.tolist(),
                "max_nfev": 60,
                "first_position_and_outgoing_velocities_only": True,
            },
            "initial_position_error_m": float(np.linalg.norm(result["parameters"][:3] - truth[:3])),
            "maximum_outgoing_velocity_error_mps": float(
                np.linalg.norm((result["parameters"][3:] - truth[3:]).reshape(-1, 3), axis=1).max()
            ),
            "scope": "same-simulator easy airborne control; no bounces, real video or independent accuracy",
        }
    )
    args.control_output.parent.mkdir(parents=True, exist_ok=True)
    args.control_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
