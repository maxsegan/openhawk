"""Research seed from post-bounce images when the incoming arc lacks observations.

The known bounce time is evaluation conditioning. Neither XYZ truth nor held-out
images enter this initializer. It supplies a seed, never a returned/certified path.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares, minimize

from cv.experiments.connected_shooting import camera_geometry, initialization, measured_dynamics
from cv.experiments.connected_shooting.model import R_BALL, Scene
from cv.pipeline.rich_ball_physics import spin_vector
from physics import bounce_reference


def finite_bootstrap(guess: np.ndarray, residual) -> tuple[np.ndarray, dict]:
    """Try fixed image-derived velocity/spin scales before abandoning a pole.

    These are initialization hypotheses, never altered observations or returned
    trajectories. The original incoming estimate may diverge under backward drag.
    """
    candidates, rows = [], []
    for scale in (1.0, 0.8, 0.6, 0.4, 0.2):
        for spin in (float(guess[5]), 0.0):
            trial = guess.copy()
            trial[2:5] *= scale
            trial[5] = spin
            value = np.asarray(residual(trial), float)
            finite = bool(np.isfinite(value).all() and np.max(np.abs(value)) < 1e6)
            cost = float(4 * np.sum(np.sqrt(1 + (value / 2) ** 2) - 1)) if finite else None
            rows.append(dict(velocity_scale=scale, spin=spin, finite=finite, image_cost=cost))
            if finite:
                candidates.append((cost, trial, len(rows) - 1))
    if not candidates:
        raise ValueError("no finite scaled post-bounce-derived incoming seed")
    _, selected, index = min(candidates, key=lambda row: row[0])
    return selected, dict(
        method="fixed velocity/spin bootstrap scales",
        candidates=rows,
        selected_index=index,
        observations_discarded=0,
    )


def spin_coordinates(x, v, w):
    axes = np.column_stack([spin_vector(np.r_[x, v, np.eye(3)[k]]) / 100 for k in range(3)])
    return np.linalg.solve(axes, w / 100)


def knot_positions(state, frames, bounce, scene):
    """Sample an impact knot, carrying its known rebound into later propagation."""
    from cv.experiments.connected_shooting import passive_bounce

    frames = np.asarray(frames, float)
    xyz = np.tile(np.r_[state[:2], R_BALL], (len(frames), 1))
    before = frames < bounce
    if np.any(before):
        xyz[before] = initialization.backward_arc(state, (bounce - frames[before]) / scene.fps)[0]
    outgoing_start = bounce + bounce_reference.DWELL_SECONDS * scene.fps
    after = frames > outgoing_start
    if np.any(after):
        x, v = np.r_[state[:2], R_BALL], state[2:5]
        incoming_spin = spin_vector(np.r_[x, v, state[5], 0.0, 0.0])
        # Additive: a bare surface name ignores the landing point (physics.surface_model).
        rebound = bounce_reference.court_bounce(v, incoming_spin, scene.surface, position=x)
        outgoing, _ = measured_dynamics.rebound_velocity(rebound, scene.bounce_profile)
        theta = np.r_[x, outgoing, spin_coordinates(x, outgoing, rebound.spin)]
        xyz[after] = measured_dynamics.simulate(
            theta,
            outgoing_start,
            frames[after],
            scene.fps,
            scene.surface,
            bounce_profile=scene.bounce_profile,
            **({"initial_successive_grounds": 1} if passive_bounce.active() != "off" else {}),
        )[0]
    return xyz


def seed_flight(
    scene: Scene, bounce: float, *, max_nfev: int = 60, recover_launch_feasibility: bool = False
) -> tuple[np.ndarray, dict]:
    """Fit one supplied bounce knot using original training images on either side."""
    scene.validate()
    if (
        len(scene.contact_frames) != 2
        or scene.dynamics != "measured_240hz"
        or not np.isfinite(bounce)
        or not scene.contact_frames[0] < bounce < scene.contact_frames[1]
        or max_nfev <= 0
    ):
        raise ValueError("one measured-dynamics flight with an interior bounce required")
    frames, cameras, pixels = scene.observation_frames[0], scene.cameras[0], scene.pixels[0]
    radial = None if scene.camera_distortion is None else scene.camera_distortion[0]
    post = frames > bounce + bounce_reference.DWELL_SECONDS * scene.fps
    if np.count_nonzero(post) < 4:
        raise ValueError("post-bounce initializer needs four outgoing training pictures")
    # Gravity-only outgoing arc with a known ground knot gives five linear unknowns:
    # impact x/y and outgoing vx/vy/vz. The dwell uses the same model constant.
    seconds = (frames[post] - bounce) / scene.fps - bounce_reference.DWELL_SECONDS
    undistorted = camera_geometry.undistort(pixels[post], None if radial is None else radial[post])
    rows, targets = [], []
    for dt, camera, pixel in zip(seconds, cameras[post], undistorted, strict=True):
        for axis in range(2):
            plane = camera[axis] - pixel[axis] * camera[2]
            norm = np.linalg.norm(plane[:3])
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError("degenerate post-bounce camera ray")
            plane /= norm
            rows.append(np.r_[plane[:2], dt * plane[:3]])
            targets.append(-plane[3] - plane[2] * (R_BALL - 4.905 * dt**2))
    gravity, _, rank, singular = np.linalg.lstsq(rows, targets, rcond=None)
    if rank != 5 or not np.isfinite(gravity).all() or gravity[4] <= 0:
        raise ValueError("unidentified or nonascending outgoing gravity seed")

    def velocity_residual(v):
        # Additive: a bare surface name ignores the landing point (physics.surface_model).
        rebound = bounce_reference.court_bounce(v, np.zeros(3), scene.surface, position=gravity[:2])
        return measured_dynamics.rebound_velocity(rebound, scene.bounce_profile)[0] - gravity[2:]

    inverse = least_squares(
        velocity_residual,
        np.clip(gravity[2:] * [1.3, 1.3, -1.3], [-74.9, -74.9, -59.9], [74.9, 74.9, -0.02]),
        bounds=([-75, -75, -60], [75, 75, -0.01]),
        max_nfev=max_nfev,
    )
    lower, upper = np.array([-10, -15, -75, -75, -60, -6.0]), np.array([21, 40, 75, 75, -0.01, 6.0])
    guess = np.clip(np.r_[gravity[:2], inverse.x, 2.0], lower + 1e-5, upper - 1e-5)
    elapsed = (bounce - scene.contact_frames[0]) / scene.fps
    calls = 0

    def residual(state):
        nonlocal calls
        calls += 1
        try:
            # The unobserved launch must still be a finite physical propagation.
            initialization.backward_arc(state, np.array([elapsed]))
            positions = knot_positions(state, frames, bounce, scene)
            value = (camera_geometry.project(cameras, positions, radial) - pixels).ravel()
            if not np.isfinite(value).all():
                raise ValueError("nonfinite knot projection")
            return value
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return np.full(2 * len(frames), 1e6)

    bootstrap_recovery = None
    if np.max(np.abs(residual(guess))) >= 1e6:
        if not recover_launch_feasibility:
            raise ValueError("no finite post-bounce-derived incoming seed")
        guess, bootstrap_recovery = finite_bootstrap(guess, residual)
    # Resolve translation before releasing spin through the piecewise impact
    # law. Spin is not independently observable from a short outgoing arc.
    translation = least_squares(
        lambda values: residual(np.r_[values, guess[5]]),
        guess[:5],
        bounds=(lower[:5], upper[:5]),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    result = least_squares(
        residual,
        np.r_[translation.x, guess[5]],
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    if np.max(np.abs(residual(result.x))) >= 1e6:
        raise ValueError("invalid post-bounce refinement plateau")
    x, v, w = (a[0] for a in initialization.backward_arc(result.x, np.array([elapsed])))
    raw = np.r_[x, v, spin_coordinates(x, v, w)]
    lo = np.r_[[-10, -15, R_BALL], np.full(3, -75), np.full(3, -6)]
    hi = np.r_[[21, 40, 12], np.full(3, 75), np.full(3, 6)]
    # Unlike ordinary pre-bounce seeding, do not conceal an unsupported inverse
    # solution by clipping its unobserved launch into bounds.
    recovery = None
    selected_state = result.x
    if np.any(raw <= lo) or np.any(raw >= hi):
        if not recover_launch_feasibility:
            raise ValueError("post-bounce-derived launch outside fitting bounds")

        def launch(state):
            x, v, w = (a[0] for a in initialization.backward_arc(state, np.array([elapsed])))
            return np.r_[x, v, spin_coordinates(x, v, w)]

        def inequalities(state):
            try:
                value = launch(state)
                return np.r_[value - lo - 1e-5, hi - value - 1e-5]
            except (ValueError, FloatingPointError, OverflowError):
                return np.full(18, -1e6)

        def cost(state):
            return float(4 * np.sum(np.sqrt(1 + (residual(state) / 2) ** 2) - 1))

        candidates, receipts = [], []
        for name, start in (
            ("released_spin", result.x),
            ("fixed_spin", np.r_[translation.x, guess[5]]),
        ):
            candidate = minimize(
                cost,
                start,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints={"type": "ineq", "fun": inequalities},
                options={"maxiter": 2 * max_nfev, "ftol": 1e-9},
            )
            feasible = bool(
                np.min(inequalities(candidate.x)) >= -1e-7
                and np.max(np.abs(residual(candidate.x))) < 1e6
            )
            receipts.append(
                dict(
                    start=name,
                    optimizer_success=bool(candidate.success),
                    message=str(candidate.message),
                    iterations=int(candidate.nit),
                    feasible=feasible,
                    cost=cost(candidate.x),
                )
            )
            if feasible:
                candidates.append((cost(candidate.x), candidate.x.copy(), name))
        if not candidates:
            raise ValueError("no launch-feasible post-bounce initialization found")
        _, selected_state, selected_name = min(candidates, key=lambda r: r[0])
        raw = launch(selected_state)
        if np.any(raw <= lo) or np.any(raw >= hi):
            raise ValueError("recovered post-bounce launch outside strict fitting bounds")
        recovery = dict(
            method="SLSQP launch-bound inequalities",
            starts=receipts,
            selected_start=selected_name,
            coordinates_clipped=False,
            bounds_lo=lo.tolist(),
            bounds_hi=hi.tolist(),
        )
    return raw, {
        "method": "post_bounce_measured_knot_seed",
        "training_frames": frames.tolist(),
        "outgoing_training_frames": frames[post].tolist(),
        "bounce_frame": float(bounce),
        "gravity_rank": int(rank),
        "gravity_condition_number": float(singular[0] / singular[-1]),
        "inverse_velocity_optimizer_success": bool(inverse.success),
        "translation_optimizer_success": bool(translation.success),
        "optimizer_success": bool(result.success),
        "objective_calls": calls,
        "training_pixel_rms": float(np.sqrt(np.mean(residual(selected_state) ** 2))),
        "bounce_state": selected_state.tolist(),
        **({"launch_feasibility_recovery": recovery} if recover_launch_feasibility else {}),
        **(
            {"finite_bootstrap_recovery": bootstrap_recovery}
            if bootstrap_recovery is not None
            else {}
        ),
        "uses_truth_xyz_or_velocity": False,
        "heldout_images_used": False,
        "returned_trajectory": False,
        "spin_limitation": "Short outgoing arcs do not independently certify incoming spin or hidden depth",
    }


def seed(
    scene: Scene,
    bounces: list[float],
    *,
    max_nfev: int = 60,
    recover_launch_feasibility: bool = False,
):
    """Explicit research fallback; existing sufficient-pre-bounce behavior is unchanged.

    A flight with no supplied bounce is a volley, and there is no post-bounce arc
    to seed it from; it keeps the ordinary image-physics seed.  Refusing the whole
    point because one of its flights is a volley loses the flights this fallback
    could have seeded, so only the volley flight is excluded.
    """
    scene.validate()
    if len(bounces) != len(scene.pixels) or any(
        b is not None and not np.isfinite(b) for b in bounces
    ):
        raise ValueError("one finite or absent supplied bounce per flight required")
    fallback = [
        b is not None and np.count_nonzero(fs < b) < 4
        for fs, b in zip(scene.observation_frames, bounces)
    ]
    if not any(fallback):
        return initialization.physics_backward_seed(scene, bounces, max_nfev=max_nfev)
    states, evidence = [], []
    for i, use_post in enumerate(fallback):
        local = replace(
            scene,
            contact_frames=scene.contact_frames[i : i + 2],
            observation_frames=(scene.observation_frames[i],),
            cameras=(scene.cameras[i],),
            pixels=(scene.pixels[i],),
            spin_parameters=scene.spin_parameters[i : i + 1],
            camera_distortion=None
            if scene.camera_distortion is None
            else (scene.camera_distortion[i],),
        )
        state, record = (
            seed_flight(
                local,
                bounces[i],
                max_nfev=max_nfev,
                recover_launch_feasibility=recover_launch_feasibility,
            )
            if use_post
            else initialization.physics_backward_seed(local, [bounces[i]], max_nfev=max_nfev)
        )
        states.append(state)
        evidence.append(
            {"flight_index": i, "post_bounce_fallback": bool(use_post), "evidence": record}
        )
    return np.r_[
        states[0][:3],
        np.concatenate([s[3:6] for s in states]),
        np.concatenate([s[6:9] for s in states]),
    ], {
        "method": "explicit_missing_incoming_arc_fallback",
        "flights": evidence,
        "uses_truth_xyz_or_velocity": False,
        "heldout_images_used": False,
    }
