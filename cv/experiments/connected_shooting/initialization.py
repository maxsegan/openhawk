"""Stage 6 seeds given explicit contact and first-bounce times.

``image_ballistic_seed`` and ``physics_backward_seed`` are image-only: they fit
pre-bounce pinhole equations under gravity-only motion and read no XYZ labels,
truth velocities, player locations or held-out images.  ``anchor_connected_seed``
and ``serve_contact_anchor`` are the experimental anchored seeds; they
additionally read the automatic striker court positions and the supplied bounce
ground-ray witnesses, which are the same explicit inputs the unchanged
acceptance gates are scored against.  Every one of these is an initialization
approximation, not a returned trajectory or physical certificate, and none reads
an owner XYZ or a held-out picture.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from physics import flight
from cv.pipeline.rich_ball_physics import spin_vector

from cv.experiments.connected_shooting import camera_geometry
from cv.experiments.connected_shooting.model import Scene, R_BALL


def image_ballistic_seed(
    scene: Scene,
    first_bounce_frames: list[float | None],
    *,
    ground_anchor: bool = False,
    first_contact_y_m: float | None = None,
    consensus: bool = False,
    bounce_ground_targets: list | None = None,
) -> tuple[np.ndarray, dict]:
    """Seed each flight from its own pre-bounce pictures.

    ``first_contact_y_m`` conditions the first state on an explicit depth
    hypothesis by eliminating Y before solving X, Z and velocity. It does not
    move picture epochs or infer a depth from evaluation success.

    ``bounce_ground_targets`` optionally supplies the measured court position of
    each flight's first bounce -- the same native ground ray the bounce witness
    already constructs.  It is used only where the pictures alone leave the
    gravity-only seed unidentified: a serve is short and often blurred, and a
    flight with two usable pre-bounce pictures otherwise loses the whole attempt
    before any fit runs.  The bounce epoch and the pictures are unchanged; this
    adds the two horizontal equations the ball's own observed impact already
    implies.
    """
    scene.validate()
    if first_contact_y_m is not None and (
        isinstance(first_contact_y_m, (bool, np.bool_))
        or not isinstance(first_contact_y_m, (int, float, np.number))
        or not np.isfinite(first_contact_y_m)
        or not -15.0 <= first_contact_y_m <= 40.0
        or not ground_anchor
        or not first_bounce_frames
        or first_bounce_frames[0] is None
    ):
        raise ValueError("fixed first depth requires a finite original ground-constrained seed")
    n = len(scene.contact_frames) - 1
    if len(first_bounce_frames) != n:
        raise ValueError("explicit bounce boundary per flight required")
    if bounce_ground_targets is not None and len(bounce_ground_targets) != n:
        raise ValueError("one optional bounce ground target per flight required")
    positions, velocities, evidence = [], [], []
    gravity = np.array([0.0, 0.0, -9.81])
    for i, (frames, cameras, pixels, bounce) in enumerate(
        zip(scene.observation_frames, scene.cameras, scene.pixels, first_bounce_frames, strict=True)
    ):
        if bounce is not None and (not np.isfinite(bounce) or bounce <= scene.contact_frames[i]):
            raise ValueError("invalid first-bounce boundary")
        stop = scene.contact_frames[i + 1] if bounce is None else bounce
        before = frames < stop
        anchored_flight = (
            ground_anchor and bounce is not None and bounce <= scene.contact_frames[i + 1]
        )
        if i == 0 and first_contact_y_m is not None and not anchored_flight:
            raise ValueError("fixed first depth requires a ground within the first flight")
        target_xy = None if bounce_ground_targets is None else bounce_ground_targets[i]
        if target_xy is not None:
            target_xy = np.asarray(target_xy, float)[:2]
            if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
                raise ValueError("finite two-dimensional bounce ground target required")
        anchored_target = anchored_flight and target_xy is not None
        # Six equations identify the unanchored six-parameter seed, so four
        # pictures are the honest minimum there.  An exact bounce time removes
        # one unknown, and three pictures then still over-determine the
        # remaining five.  A measured bounce position removes two more, so two
        # pictures suffice.  The rank check below is what actually decides.
        minimum_pictures = 2 if anchored_target else (3 if anchored_flight else 4)
        if sum(before) < minimum_pictures:
            word = {2: "two", 3: "three"}.get(minimum_pictures, "four")
            raise ValueError(f"image initializer needs {word} pre-bounce training pictures")
        t = (frames[before] - scene.contact_frames[i]) / scene.fps
        from cv.experiments.connected_shooting import camera_geometry

        radial = None if scene.camera_distortion is None else scene.camera_distortion[i][before]
        P, native_xy = cameras[before], pixels[before]
        xy = camera_geometry.undistort(native_xy, radial)
        rows, targets = [], []
        for dt, camera, pixel in zip(t, P, xy, strict=True):
            for axis in range(2):
                plane = camera[axis] - pixel[axis] * camera[2]
                norm = np.linalg.norm(plane[:3])
                if not np.isfinite(norm) or norm < 1e-12:
                    raise ValueError("degenerate camera ray equation")
                plane = plane / norm
                rows.append(np.r_[plane[:3], dt * plane[:3]])
                targets.append(-plane[3] - plane[:3] @ (0.5 * gravity * dt**2))
        picture_rows = len(rows)
        if anchored_target:
            # The ball is at the measured impact position when the supplied
            # bounce epoch arrives.  Gravity has no horizontal component, so the
            # gravity-only seed states this exactly.
            tb = (bounce - scene.contact_frames[i]) / scene.fps
            rows.append(np.array([1.0, 0.0, 0.0, tb, 0.0, 0.0]))
            targets.append(float(target_xy[0]))
            rows.append(np.array([0.0, 1.0, 0.0, 0.0, tb, 0.0]))
            targets.append(float(target_xy[1]))
        A, b = np.asarray(rows), np.asarray(targets)
        anchored = anchored_flight
        if anchored:
            # Exact known bounce time supplies a ground-plane constraint, not
            # an oracle bounce pixel/XYZ. Eliminate z0 algebraically so this
            # physical initialization condition cannot become a soft outlier.
            tb = (bounce - scene.contact_frames[i]) / scene.fps
            offset = np.array([0.0, 0.0, R_BALL + 4.905 * tb**2, 0.0, 0.0, 0.0])
            transform = np.zeros((6, 5))
            transform[0, 0] = transform[1, 1] = transform[3, 2] = transform[4, 3] = transform[
                5, 4
            ] = 1
            transform[2, 4] = -tb
            fixed_first_depth = i == 0 and first_contact_y_m is not None
            if fixed_first_depth:
                offset[1] = first_contact_y_m
                transform = np.delete(transform, 1, axis=1)
            free, _, rank, singular = np.linalg.lstsq(A @ transform, b - A @ offset, rcond=None)
            state = offset + transform @ free
            required_rank = 4 if fixed_first_depth else 5
        else:
            state, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
            required_rank = 6
        if rank != required_rank or not np.isfinite(state).all():
            raise ValueError("unidentified ballistic image seed")
        consensus_evidence = None
        if consensus and anchored:
            # Fixed deterministic image-only hypotheses. They initialize the
            # model; no observation is deleted from refinement or final fitting.
            anchors = np.unique(np.linspace(0, len(t) - 1, min(8, len(t))).round().astype(int))
            candidates = [(state, list(range(len(t))))]
            for indices in combinations(anchors, 4):
                rows4 = np.array([[2 * j, 2 * j + 1] for j in indices]).ravel()
                free, _, subset_rank, _ = np.linalg.lstsq(
                    (A @ transform)[rows4], (b - A @ offset)[rows4], rcond=None
                )
                if subset_rank == required_rank:
                    candidates.append((offset + transform @ free, list(map(int, indices))))
            ranked = []
            for candidate, indices in candidates:
                x = candidate[:3] + t[:, None] * candidate[3:] + 0.5 * t[:, None] ** 2 * gravity
                uvw = np.einsum("nij,nj->ni", P, np.c_[x, np.ones(len(x))])
                if not np.isfinite(uvw).all() or np.any(np.abs(uvw[:, 2]) < 1e-9):
                    continue
                residual = camera_geometry.distort(uvw[:, :2] / uvw[:, 2:], radial) - native_xy
                squared = np.sum(residual**2, axis=1)
                support = max(4, int(np.ceil(0.7 * len(squared))))
                cost = float(np.sum(np.sort(squared)[:support]))
                if np.isfinite(cost):
                    ranked.append((cost, candidate, indices))
            if not ranked:
                raise ValueError("no finite gravity consensus hypothesis")
            cost, state, indices = min(ranked, key=lambda item: item[0])
            consensus_evidence = {
                "hypotheses": len(candidates),
                "finite_hypotheses": len(ranked),
                "selected_native_frames": frames[before][indices].tolist(),
                "trimmed_squared_pixel_cost": cost,
                "ranking_support_fraction": 0.7,
                "deleted_observations": 0,
            }
        positions.append(state[:3])
        velocities.append(state[3:])
        evidence.append(
            {
                "flight_index": i,
                **(
                    {"fixed_first_contact_y_m": float(first_contact_y_m)}
                    if i == 0 and first_contact_y_m is not None
                    else {}
                ),
                "training_frames": frames[before].tolist(),
                "rank": int(rank),
                "ground_plane_at_known_bounce": bool(anchored),
                # Only stated when the caller offered measured impacts at all, so
                # a run that does not use them keeps its receipt byte for byte.
                **(
                    {}
                    if bounce_ground_targets is None
                    else {
                        "measured_bounce_position_rows": int(len(rows) - picture_rows),
                        "bounce_ground_target_xy_m": (
                            None if not anchored_target else target_xy.tolist()
                        ),
                    }
                ),
                "condition_number": float(singular[0] / singular[-1]),
                "local_start_xyz_seed": state[:3].tolist(),
                "local_velocity_seed": state[3:].tolist(),
                **({"consensus": consensus_evidence} if consensus else {}),
            }
        )
    raw = np.r_[positions[0], np.asarray(velocities).ravel(), scene.spin_parameters.ravel()]
    # Bounds constrain optimization, not evidence. Record every clipped entry.
    lower = np.r_[[-10.0, -15.0, R_BALL], np.full(3 * n, -75.0), np.full(3 * n, -6.0)]
    upper = np.r_[[21.0, 40.0, 12.0], np.full(3 * n, 75.0), np.full(3 * n, 6.0)]
    bounded = np.clip(raw, lower + 1e-5, upper - 1e-5)
    return bounded, {
        "method": "pre_bounce_native_pinhole_gravity_only",
        "flights": evidence,
        "clipped_parameter_indices": np.flatnonzero(raw != bounded).tolist(),
        "uses_truth_xyz_or_velocity": False,
        "bounce_times_are_explicit_inputs": True,
        "ground_anchor_enabled": ground_anchor,
    }


def backward_arc(state: np.ndarray, seconds_before: np.ndarray):
    """Free flight backwards from a descending ground impact, no bounce insertion.

    state = bounce x/y, incoming vx/vy/vz, incoming topspin / 100 rad/s.
    Positive queries mean seconds BEFORE the supplied physical bounce.
    """
    state, query = np.asarray(state, float), np.asarray(seconds_before, float)
    if (
        state.shape != (6,)
        or not np.isfinite(state).all()
        or state[4] >= 0
        or query.ndim != 1
        or not len(query)
        or not np.isfinite(query).all()
        or np.any(query < 0)
    ):
        raise ValueError("descending finite ground state and nonnegative backward times required")
    x = np.r_[state[:2], R_BALL]
    v = state[2:5].copy()
    w = spin_vector(np.r_[x, v, state[5], 0.0, 0.0])
    times, xs, vs, ws = [0.0], [x.copy()], [v.copy()], [w.copy()]
    dt = 1 / 240
    for step in range(int(np.ceil(query.max() / dt))):
        if np.linalg.norm(v) > 250 or np.linalg.norm(w) > 5000 or np.linalg.norm(x) > 1000:
            raise ValueError("backward flight diverged outside search bounds")
        x, v, w = flight.rk4_step(x, v, w, -dt)
        if not np.isfinite(np.r_[x, v, w]).all():
            raise ValueError("nonfinite backward flight")
        times.append((step + 1) * dt)
        xs.append(x.copy())
        vs.append(v.copy())
        ws.append(w.copy())
    return tuple(
        np.column_stack([np.interp(query, times, np.asarray(a)[:, k]) for k in range(3)])
        for a in (xs, vs, ws)
    )


def physics_backward_seed(
    scene: Scene,
    first_bounce_frames: list[float | None],
    *,
    max_nfev: int = 60,
    consensus: bool = False,
    include_bounce_exposure: bool = False,
) -> tuple[np.ndarray, dict]:
    """Image-only drag/Magnus seeds; independent arcs initialize a connected fit."""
    seed, record = image_ballistic_seed(
        scene, first_bounce_frames, ground_anchor=True, consensus=consensus
    )
    n = len(first_bounce_frames)
    record["method"] = (
        "consensus_pre_bounce_backward_drag_magnus_ground_knot"
        if consensus
        else "pre_bounce_backward_drag_magnus_ground_knot"
    )
    record["backward_refinement"] = []
    record["include_bounce_exposure"] = include_bounce_exposure
    for i, bounce in enumerate(first_bounce_frames):
        if bounce is None or bounce > scene.contact_frames[i + 1]:
            record["backward_refinement"].append(
                {"flight_index": i, "status": "no_in_scope_bounce_kept_ballistic_seed"}
            )
            continue
        f0 = scene.contact_frames[i]
        tb = (bounce - f0) / scene.fps
        frames = scene.observation_frames[i]
        before = frames <= bounce if include_bounce_exposure else frames < bounce
        q = (bounce - frames[before]) / scene.fps
        P, target = scene.cameras[i][before], scene.pixels[i][before]
        from cv.experiments.connected_shooting import camera_geometry

        radial = None if scene.camera_distortion is None else scene.camera_distortion[i][before]
        local = record["flights"][i]
        p0, v0 = np.asarray(local["local_start_xyz_seed"]), np.asarray(local["local_velocity_seed"])
        g = np.array([0.0, 0.0, -9.81])
        guess = np.r_[(p0 + v0 * tb + 0.5 * g * tb**2)[:2], v0 + g * tb, 2.0]
        lower, upper = (
            np.array([-10.0, -15.0, -75.0, -75.0, -60.0, -6.0]),
            np.array([21.0, 40.0, 75.0, 75.0, -0.01, 6.0]),
        )
        guess = np.clip(guess, lower + 1e-5, upper - 1e-5)
        calls = 0

        def residual(parameters):
            nonlocal calls
            calls += 1
            try:
                # Validate propagation through the actual contact as well as
                # visible pictures; an unobserved prefix must not conceal a
                # divergent starting state.
                xyz = backward_arc(parameters, np.r_[q, tb])[0][:-1]
                value = (camera_geometry.project(P, xyz, radial) - target).ravel()
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite backward projection")
                return value
            except (ValueError, FloatingPointError, OverflowError):
                return np.full(2 * len(target), 1e6)

        # An invalid initial state produces a constant penalty with zero
        # derivative. Do not call that plateau a converged initializer. Try a
        # fixed small set of image-derived velocity/spin scales first.
        candidates = [guess]
        for scale in (0.8, 0.6, 0.4, 0.2):
            for spin_guess in (2.0, 0.0):
                trial = guess.copy()
                trial[2:5] *= scale
                trial[5] = spin_guess
                candidates.append(trial)

        def robust_cost(p):
            squared = (residual(p) / 2) ** 2
            return float(np.sum(np.log1p(squared) if consensus else np.sqrt(1 + squared) - 1))

        scored = [(robust_cost(p), p) for p in candidates]
        feasible = [(cost, p) for cost, p in scored if np.max(np.abs(residual(p))) < 1e6]
        if not feasible:
            raise ValueError(f"flight {i}: no finite image-derived backward seed")
        # The consensus control refines every existing finite bootstrap, then
        # selects by image loss among launch-valid results. A low-cost but
        # underground/out-of-bounds local solution cannot become a clipped seed.
        starts = feasible if consensus else [min(feasible, key=lambda item: item[0])]
        refined = []
        for _, guess in starts:
            candidate = least_squares(
                residual,
                guess,
                bounds=(lower, upper),
                loss="cauchy" if consensus else "soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
            if consensus:
                try:
                    cx, cv, _ = (a[0] for a in backward_arc(candidate.x, np.array([tb])))
                except (ValueError, FloatingPointError, OverflowError):
                    continue
                if (
                    np.any(cx < [-10 + 1e-5, -15 + 1e-5, R_BALL + 1e-5])
                    or np.any(cx > [21 - 1e-5, 40 - 1e-5, 12 - 1e-5])
                    or np.any(np.abs(cv) > 75 - 1e-5)
                ):
                    continue
            refined.append((robust_cost(candidate.x), candidate))
        if not refined:
            raise ValueError(f"flight {i}: no launch-valid consensus backward seed")
        _, result = min(refined, key=lambda item: item[0])
        x, v, w = (a[0] for a in backward_arc(result.x, np.array([tb])))
        axes = np.column_stack([spin_vector(np.r_[x, v, np.eye(3)[k]]) / 100 for k in range(3)])
        spin = np.linalg.solve(axes, w / 100)
        raw_local = np.r_[x, v, spin]
        local_lower = np.r_[[-10.0, -15.0, R_BALL], np.full(3, -75.0), np.full(3, -6.0)]
        local_upper = np.r_[[21.0, 40.0, 12.0], np.full(3, 75.0), np.full(3, 6.0)]
        clipped = np.flatnonzero(
            (raw_local < local_lower + 1e-5) | (raw_local > local_upper - 1e-5)
        ).tolist()
        if consensus and clipped:
            raise ValueError(
                f"flight {i}: consensus backward seed violates launch bounds {clipped}"
            )
        if i == 0:
            seed[:3] = np.clip(
                x,
                [-10.0 + 1e-5, -15.0 + 1e-5, R_BALL + 1e-5],
                [21.0 - 1e-5, 40.0 - 1e-5, 12.0 - 1e-5],
            )
        seed[3 + 3 * i : 6 + 3 * i] = np.clip(v, -75.0 + 1e-5, 75.0 - 1e-5)
        seed[3 + 3 * n + 3 * i : 6 + 3 * n + 3 * i] = np.clip(spin, -6.0 + 1e-5, 6.0 - 1e-5)
        record["backward_refinement"].append(
            {
                "flight_index": i,
                "status": "image_fit_seed_only",
                "training_frames": frames[before].tolist(),
                "included_exact_bounce_exposure": bool(np.any(frames[before] == bounce)),
                "optimizer_success": bool(result.success),
                "objective_calls": calls,
                "finite_start_candidates": len(feasible),
                "training_pixel_rms": float(np.sqrt(np.mean(residual(result.x) ** 2))),
                "bounce_state": result.x.tolist(),
                "local_start_xyz_seed": x.tolist(),
                "local_velocity_seed": v.tolist(),
                "local_spin_seed": spin.tolist(),
                "backward_local_clipped_indices": clipped,
                "initialization_loss": "cauchy" if consensus else "soft_l1",
                "refined_start_candidates": len(starts),
                "eligible_refined_candidates": len(refined),
            }
        )
    return seed, record


def serve_contact_anchor(
    camera: np.ndarray,
    pixel: np.ndarray,
    stature_m: float | None,
    first_contact_y_m: float,
    *,
    height_band: tuple[float, float] = (1.5, 1.75),
    radial: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Put the serve contact on the server's own observation ray, not in the sky.

    The depth beam pins the first contact's court Y, so the ray already fixes X
    and Z there.  When that height falls outside the stature band the point is
    taken at the clipped height instead and only X follows the ray; Y stays
    pinned by the beam either way.  No label XYZ, no withheld picture and no
    acceptance verdict is read: this is the labeled contact pixel through the
    same camera the fit already uses.  Without a stature (explicitly unavailable
    athlete evidence) there is no band: the ray point at the beam depth is the
    anchor as observed, and the receipt says so.
    """
    from cv.experiments.connected_shooting import real_bidirectional_search as whole

    if radial is not None:
        pixel = camera_geometry.undistort(
            np.asarray(pixel, float)[None], np.asarray(radial, float)[None]
        )[0]
    at_depth = whole.ray_point_at_y(camera, pixel, float(first_contact_y_m))
    if stature_m is None:
        xyz = at_depth.copy()
        xyz[1] = float(first_contact_y_m)
        return xyz, {
            "method": "server_contact_ray_at_depth_branch_unbanded",
            "stature_m": None,
            "height_band_m": None,
            "ray_height_at_depth_m": float(at_depth[2]),
            "clipped_to_band": False,
            "contact_xyz_m": xyz.tolist(),
        }
    low, high = (value * float(stature_m) for value in height_band)
    if not np.isfinite([low, high]).all() or low <= 0 or high <= low:
        raise ValueError("positive increasing stature contact-height band required")
    height = float(np.clip(at_depth[2], low, high))
    xyz = at_depth.copy()
    receipt = {
        "method": "server_contact_ray_at_depth_branch",
        "stature_m": float(stature_m),
        "height_band_m": [low, high],
        "ray_height_at_depth_m": float(at_depth[2]),
        "clipped_to_band": bool(height != at_depth[2]),
    }
    if receipt["clipped_to_band"]:
        # Slide along the ray to the banded height and keep that X; the beam
        # rewrites Y immediately afterwards, so only X and Z matter here.
        try:
            from cv.pipeline.anthropometric_contacts import ray_at_z

            banded = np.asarray(
                ray_at_z(np.asarray(camera, float), np.asarray(pixel, float), height), float
            )
            xyz = np.r_[banded[0], at_depth[1], height]
        except (ValueError, FloatingPointError, OverflowError, TypeError):
            xyz = np.r_[at_depth[0], at_depth[1], height]
            receipt["method"] = "server_contact_ray_at_depth_branch_height_clipped_in_place"
    xyz[1] = float(first_contact_y_m)
    receipt["contact_xyz_m"] = xyz.tolist()
    return xyz, receipt


def toss_contact_seed(
    scene: Scene, pinhole_seed: np.ndarray, contact_estimate: dict
) -> tuple[np.ndarray, dict]:
    """Seed a serve at the toss contact and aim it through outgoing pictures.

    Once the toss fixes the starting position, the first native post-contact
    pictures identify the outgoing ballistic velocity without sliding the
    short serve arc along its camera ray.  This gravity-only state initializes
    the connected physical solve; all original pictures and final gates remain.
    """
    scene.validate()
    seed = np.asarray(pinhole_seed, float).copy()
    contact = np.asarray(contact_estimate.get("contact_xyz_m"), float)
    sigma = np.asarray(contact_estimate.get("contact_sigma_m"), float)
    if (
        contact_estimate.get("status") != "supported"
        or contact.shape != (3,)
        or sigma.shape != (3,)
        or not np.isfinite(np.r_[contact, sigma]).all()
        or np.any(sigma <= 0)
    ):
        raise ValueError("supported finite toss contact and positive sigma required")
    frames = np.asarray(scene.observation_frames[0], float)
    after = frames >= float(scene.contact_frames[0])
    indices = np.flatnonzero(after)[:6]
    if len(indices) < 2:
        raise ValueError("two post-contact native pictures required for serve direction")
    selected_frames = frames[indices]
    times = (selected_frames - float(scene.contact_frames[0])) / float(scene.fps)
    cameras = np.asarray(scene.cameras[0], float)[indices]
    native_pixels = np.asarray(scene.pixels[0], float)[indices]
    radial = None
    if scene.camera_distortion is not None:
        radial = np.asarray(scene.camera_distortion[0], float)[indices]
    pixels = camera_geometry.undistort(native_pixels, radial)
    gravity = np.array([0.0, 0.0, -9.81])
    rows, targets = [], []
    for dt, camera, pixel in zip(times, cameras, pixels, strict=True):
        for axis in range(2):
            plane = camera[axis] - pixel[axis] * camera[2]
            norm = np.linalg.norm(plane[:3])
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError("degenerate post-contact camera ray")
            plane = plane / norm
            rows.append(dt * plane[:3])
            targets.append(-plane[3] - plane[:3] @ (contact + 0.5 * gravity * dt**2))
    matrix, target = np.asarray(rows), np.asarray(targets)
    velocity, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != 3 or not np.isfinite(velocity).all():
        raise ValueError("post-contact pictures do not identify a serve direction")
    raw_velocity = velocity.copy()
    pinhole_speed = float(np.linalg.norm(seed[3:6]))
    direction_speed = float(np.linalg.norm(velocity))
    if not np.isfinite(pinhole_speed) or pinhole_speed <= 0 or direction_speed <= 0:
        raise ValueError("positive finite image-derived serve speeds required")
    velocity *= pinhole_speed / direction_speed
    raw = seed.copy()
    raw[:3] = contact
    raw[3:6] = velocity
    if raw.ndim != 1 or len(raw) < 6 or not np.isfinite(raw).all():
        raise ValueError("finite connected seed with a first velocity required")
    bounded = raw.copy()
    bounded[:3] = np.clip(
        raw[:3],
        [-10.0 + 1e-5, -15.0 + 1e-5, R_BALL + 1e-5],
        [21.0 - 1e-5, 40.0 - 1e-5, 12.0 - 1e-5],
    )
    bounded[3:6] = np.clip(raw[3:6], -75.0 + 1e-5, 75.0 - 1e-5)
    return bounded, {
        "method": "toss_contact_plus_postcontact_native_ballistic_direction",
        "contact_xyz_m": contact.tolist(),
        "contact_sigma_m": sigma.tolist(),
        "postcontact_frames": selected_frames.tolist(),
        "postcontact_velocity_seed_mps": velocity.tolist(),
        "postcontact_unscaled_velocity_mps": raw_velocity.tolist(),
        "velocity_scale_source": "pinhole_seed_speed_with_toss_contact_ray_direction",
        "postcontact_ray_equation_rank": int(rank),
        "postcontact_ray_condition_number": float(singular[0] / singular[-1]),
        "clipped_parameter_indices": np.flatnonzero(raw != bounded).tolist(),
        "uses_outgoing_serve_arc_for_contact_position": False,
        "uses_withheld_pixels": False,
    }


def serve_start_prior_seed(
    scene: Scene,
    pinhole_seed: np.ndarray,
    contact_estimate: dict,
    first_bounce_target: dict,
) -> tuple[np.ndarray, dict]:
    """Seed the prior contact and aim the serve at its observed bounce witness."""
    seeded, receipt = toss_contact_seed(scene, pinhole_seed, contact_estimate)
    contact = np.asarray(seeded[:3], float)
    target = np.asarray(first_bounce_target["xyz_m"], float)
    bounce_frame = float(first_bounce_target["event_frame"])
    duration = (bounce_frame - float(scene.contact_frames[0])) / float(scene.fps)
    if target.shape != (3,) or not np.isfinite(target).all() or duration <= 0:
        raise ValueError("finite future serve-bounce target required")
    gravity = np.array([0.0, 0.0, -9.81])
    velocity = (target - contact - 0.5 * gravity * duration**2) / duration
    raw = seeded.copy()
    raw[3:6] = velocity
    bounded = raw.copy()
    bounded[3:6] = np.clip(raw[3:6], -75.0 + 1e-5, 75.0 - 1e-5)
    return bounded, {
        **receipt,
        "method": "serve_location_prior_plus_observed_bounce_ballistic_direction",
        "bounce_event_frame": bounce_frame,
        "bounce_target_xyz_m": target.tolist(),
        "ballistic_bounce_velocity_seed_mps": velocity.tolist(),
        "postcontact_picture_direction_retained_as_diagnostic_mps": receipt[
            "postcontact_velocity_seed_mps"
        ],
        "bounce_target_is_existing_input_gate": True,
        "clipped_parameter_indices": np.flatnonzero(raw != bounded).tolist(),
    }


def outgoing_contact_seed(
    scene: Scene,
    pinhole_seed: np.ndarray,
    hypothesis: dict,
    *,
    maximum_observations: int = 6,
    first_bounce_target: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the first clean outgoing fronts with drag and extrapolate to contact.

    The hypothesis supplies only the contact start and sub-frame epoch.  Velocity
    and all three spin coordinates are fitted against three to six native outgoing
    pictures through the repository drag/Magnus model.  This is a restart and does
    not create a contact bound or residual in the whole-point solve.
    """
    scene.validate()
    seed = np.asarray(pinhole_seed, float).copy()
    contact = np.asarray(hypothesis["contact_xyz_m"], float)
    epoch = float(hypothesis["contact_epoch_frame"])
    frames = np.asarray(scene.observation_frames[0], float)
    indices = np.flatnonzero(frames >= epoch)[:maximum_observations]
    if not 3 <= len(indices) <= 6:
        raise ValueError("three to six clean post-contact pictures required")
    selected_frames = frames[indices]
    seconds = (selected_frames - epoch) / float(scene.fps)
    cameras = np.asarray(scene.cameras[0], float)[indices]
    native_pixels = np.asarray(scene.pixels[0], float)[indices]
    radial = None
    if scene.camera_distortion is not None:
        radial = np.asarray(scene.camera_distortion[0], float)[indices]

    gravity_seed = {
        "status": "supported",
        "contact_xyz_m": contact.tolist(),
        "contact_sigma_m": hypothesis["contact_sigma_m"],
    }
    directed, direction_receipt = toss_contact_seed(scene, seed, gravity_seed)
    n = len(scene.pixels)
    spin_slice = slice(3 + 3 * n, 6 + 3 * n)
    free_spin = len(seed) in {3 + 6 * n, 5 + 6 * n}
    initial_spin = seed[spin_slice] if free_spin else np.asarray(scene.spin_parameters[0], float)
    initial = np.r_[directed[3:6], initial_spin] if free_spin else directed[3:6]

    def positions(parameters: np.ndarray) -> np.ndarray:
        velocity = parameters[:3]
        spin_parameters = parameters[3:6] if free_spin else initial_spin
        angular_velocity = spin_vector(np.r_[contact, velocity, spin_parameters])
        return flight.sample_states(contact, velocity, angular_velocity, seconds)[0]

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            projected = camera_geometry.project(cameras, positions(parameters), radial)
            value = (projected - native_pixels).ravel()
            return value if np.isfinite(value).all() else np.full(2 * len(indices), 1e6)
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(2 * len(indices), 1e6)

    lower = np.r_[[-75.0] * 3, [-6.0] * 3] if free_spin else np.asarray([-75.0] * 3)
    upper = np.r_[[75.0] * 3, [6.0] * 3] if free_spin else np.asarray([75.0] * 3)
    solved = least_squares(
        residual,
        np.clip(initial, lower, upper),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=120,
    )
    fitted_positions = positions(solved.x)
    pixel_residual = residual(solved.x).reshape(-1, 2)
    fitted_output = seed.copy()
    fitted_output[:3] = contact
    fitted_output[3:6] = solved.x[:3]
    if free_spin:
        fitted_output[spin_slice] = solved.x[3:6]
    directed_output = directed.copy()
    directed_output[:3] = contact
    original_output = seed.copy()
    original_output[:3] = contact

    # A raw local outgoing fit can put the simulated first ground impact before
    # the labeled bounce epoch, after which the complete-point simulator reaches
    # its bounce cap before the next racket contact.  Retain its fitted lateral
    # velocity and offer deterministic vertical variants that reach the court at
    # the existing bounce time.  This is seed feasibility, not a bound or gate;
    # the whole solve remains free and sees the same bounce evidence as before.
    bounce_variants = []
    if first_bounce_target is not None:
        bounce_frame = float(first_bounce_target["event_frame"])
        bounce_xyz = np.asarray(first_bounce_target["xyz_m"], float)
        duration = (bounce_frame - epoch) / float(scene.fps)
        if duration > 0 and bounce_xyz.shape == (3,) and np.isfinite(bounce_xyz).all():
            ballistic_velocity = (
                bounce_xyz - contact - 0.5 * np.asarray([0.0, 0.0, -9.81]) * duration**2
            ) / duration
            for drag_weight in (0.75, 0.5, 0.25, 0.0):
                candidate = fitted_output.copy()
                candidate[3:6] = (
                    drag_weight * fitted_output[3:6] + (1.0 - drag_weight) * ballistic_velocity
                )
                bounce_variants.append((f"drag_bounce_blend_{drag_weight:.2f}", candidate))

    from cv.experiments.connected_shooting import model as connected_model

    starts = []
    blockers = []
    candidates = [
        ("drag_fit", fitted_output),
        ("gravity_direction", directed_output),
        ("original_velocity", original_output),
        *bounce_variants,
    ]
    for name, candidate in candidates:
        complete_point_supported = True
        try:
            connected_model.chain(scene, candidate)
        except (ValueError, FloatingPointError, OverflowError) as error:
            complete_point_supported = False
            blockers.append({"seed": name, "blocker": f"{type(error).__name__}: {error}"})
        candidate_velocity = candidate[3:6]
        candidate_spin = candidate[spin_slice] if free_spin else initial_spin
        candidate_parameters = (
            np.r_[candidate_velocity, candidate_spin] if free_spin else candidate_velocity
        )
        candidate_residual = residual(candidate_parameters).reshape(-1, 2)
        starts.append(
            (
                float(np.sqrt(np.mean(candidate_residual**2))),
                name,
                candidate,
                complete_point_supported,
            )
        )
    selected_rms, selected_name, output, selected_supported = min(starts, key=lambda row: row[0])
    return output, {
        "schema": "connected_outgoing_serve_backward_contact_seed_v1",
        "method": "drag_magnus_outgoing_fronts_extrapolated_backward_to_contact",
        "contact_epoch_frame": epoch,
        "contact_xyz_m": contact.tolist(),
        "outgoing_frames": selected_frames.tolist(),
        "outgoing_velocity_mps": output[3:6].tolist(),
        "outgoing_spin_parameters": (
            output[spin_slice].tolist() if free_spin else initial_spin.tolist()
        ),
        "spin_fitted": free_spin,
        "outgoing_xyz_m": fitted_positions.tolist(),
        "outgoing_pixel_rms": selected_rms,
        "drag_fit_pixel_rms": float(np.sqrt(np.mean(pixel_residual**2))),
        "selected_complete_point_seed": selected_name,
        "complete_point_supported_seed_count": sum(row[3] for row in starts),
        "selected_seed_complete_point_supported_before_structure_repair": selected_supported,
        "complete_point_seed_blockers": blockers,
        "first_bounce_target_used_for_seed_feasibility": first_bounce_target,
        "optimizer_success": bool(solved.success),
        "optimizer_status": int(solved.status),
        "optimizer_message": str(solved.message),
        "function_evaluations": int(solved.nfev),
        "gravity_direction_bootstrap": direction_receipt,
        "candidate_generator_only": True,
        "optimizer_bound": False,
        "optimizer_inequality": False,
        "uses_withheld_pixels": False,
    }


def _native_net_anchor_guesses(scene, start_xyz, spin, rebound_scales, expected_bounces):
    """Conditional native net rays seed an airborne approach, never a final constraint."""
    from cv.experiments.connected_shooting import labeled_net_height_chart as chart
    from cv.pipeline.physics_knot_solver import net_tape_height

    receipt = {"method": "conditional_native_net_ray", "trials": []}
    if scene.net_hit_frames is None or len(scene.net_hit_frames[0]) != 1:
        return [], receipt
    epoch = float(scene.net_hit_frames[0][0])
    if any(float(frame) <= epoch for frame in expected_bounces):
        receipt["unavailable"] = "first-net airborne chart does not own an earlier ground impact"
        return [], receipt
    required = np.unique([np.floor(epoch), np.ceil(epoch)])
    frames = np.asarray(scene.observation_frames[0])
    selected = np.isin(frames, required)
    if not np.array_equal(frames[selected], required):
        receipt["unavailable"] = (
            "missing original native exposure bracketing the supplied net epoch"
        )
        return [], receipt
    radial = None if scene.camera_distortion is None else scene.camera_distortion[0][selected]
    pixels = camera_geometry.undistort(scene.pixels[0][selected], radial)
    rays = []
    try:
        for frame, P, pixel in zip(
            frames[selected], scene.cameras[0][selected], pixels, strict=True
        ):
            equations = P[:2] - pixel[:, None] * P[2]
            xz = np.linalg.solve(equations[:, [0, 2]], -equations[:, 1] * 11.885 - equations[:, 3])
            rays.append({"frame": float(frame), "xyz_m": [float(xz[0]), 11.885, float(xz[1])]})
        position = np.mean([row["xyz_m"] for row in rays], axis=0)
        if not np.isfinite(position).all() or not (
            abs(position[0] - 5.485) <= 6.4
            and R_BALL <= position[2] <= net_tape_height(float(position[0])) + R_BALL
        ):
            raise ValueError("conditional original net rays lie outside the physical mesh")
    except (ValueError, np.linalg.LinAlgError) as error:
        receipt["unavailable"] = str(error)
        return [], receipt
    seconds = (epoch - scene.contact_frames[0]) / scene.fps
    velocity = (position - start_xyz) / seconds
    velocity[2] += 0.5 * 9.81 * seconds
    receipt.update(
        original_native_rays=rays,
        conditional_mean_xyz_m=position.tolist(),
        height_constraint_in_final_fit=False,
        uses_withheld_pixels=False,
    )
    guesses = []
    spin_guesses = [np.asarray(spin, float)]
    if np.any(spin_guesses[0] != 0):
        spin_guesses.append(np.zeros(3))
    for spin_guess in spin_guesses:
        try:
            theta, projection = chart.project_net_height(
                np.r_[start_xyz, velocity, spin_guess],
                scene.contact_frames[0],
                epoch,
                scene.fps,
                scene.surface,
                height_offset_m=float(position[2] - net_tape_height(float(position[0]))),
                rebound_scales=rebound_scales,
                bounce_profile=scene.bounce_profile,
            )
            guess = theta[3:]
            if np.any(np.abs(guess[:3]) > 75) or np.any(np.abs(guess[3:]) > 6):
                raise ValueError("net-ray seed exceeds original velocity or spin bounds")
            guesses.append(guess)
            receipt["trials"].append(
                {"status": "proposed", "guess": guess.tolist(), "projection": projection}
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            receipt["trials"].append({"status": "unsupported", "reason": str(error)})
    return guesses, receipt


GROUND_EPOCH_LAUNCH_LIMIT_MPS = 75.0


def ground_epoch_launch(
    start_xyz: np.ndarray,
    seconds: float,
    horizontal: np.ndarray,
    *,
    target_xy: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return the gravity-only launch that reaches the ground at a declared epoch.

    A connected walk hands each flight the previous flight's arrival.  When that
    arrival is a court-level state the flight is launchable only while ascending,
    and the flight's own declared ground epoch is what says how far it has to
    ascend.  With a supplied ground ray the whole velocity is the arc to that
    impact; without one the caller's horizontal estimate is kept and only the
    vertical component is rebuilt.  This moves no event time, no start position
    and no observation: the epoch and the state are the supplied ones, and an
    epoch, target or launch that is not usable returns ``None`` instead of being
    clipped into one.
    """
    start = np.asarray(start_xyz, float)
    seconds = float(seconds)
    if start.shape != (3,) or not np.isfinite(start).all():
        return None
    if not np.isfinite(seconds) or seconds <= 0.0:
        return None
    if target_xy is None:
        velocity = np.array(horizontal, float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            return None
    else:
        target = np.asarray(target_xy, float)
        if target.shape != (2,) or not np.isfinite(target).all():
            return None
        velocity = (np.r_[target, R_BALL] - start) / seconds
    velocity[2] = (R_BALL - start[2]) / seconds + 4.905 * seconds
    if np.max(np.abs(velocity)) > GROUND_EPOCH_LAUNCH_LIMIT_MPS:
        return None
    return velocity


def anchor_connected_seed(
    scene,
    pinhole_seed: np.ndarray,
    axes: np.ndarray,
    bounces,
    first_contact_y_m: float,
    *,
    first_contact_xyz_m: np.ndarray,
    contact_targets_xyz_m: list,
    bounce_targets: list,
    flat_spin: bool = False,
    max_nfev: int = 40,
    recover_terminal_net_seed: bool = False,
    defer_final_net_to_initializer: bool = False,
    ground_directed_seed: bool = False,
    ground_intervals: list | None = None,
    net_intervals: list | None = None,
    exposure_duration: float | None = 0.25,
    terminal_ground_anchor: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Walk the point once from its anchors instead of from free pinhole fits.

    Each flight's six velocity/spin parameters are solved against that flight's
    own training pixels, the supplied bounce ground rays at their
    projection-derived sigma, the supplied bounce epochs, and the next contact
    anchor at the labeled contact time.  The starting position of flight ``i+1``
    is the arrival of flight ``i``, so the walk stays connected.  This is an
    initializer; the unchanged whole-point solve and every unchanged gate follow.

    A flight without a next-contact anchor -- the terminal flight, or one whose
    next striker position is explicitly absent -- keeps only the disconnected
    pinhole velocity, so a court-level connected arrival can leave it with no
    supported launch and discard the whole walk.  Where that happens the flight
    retries once from ``ground_epoch_launch``, built from its own declared
    ground epoch and supplied ground ray.

    ``terminal_ground_anchor`` (``{"frame", "xy_m"}``) makes this walk a distinct
    restart for a final flight whose first bounce has a picture-only impact epoch:
    that flight starts only from the launch reaching the ground at that epoch and
    ground point, and its local seed solve times the first bounce to the epoch.
    """
    from dataclasses import replace as _replace

    from cv.experiments.connected_shooting import model as _model
    from cv.experiments.connected_shooting import real_exposure_replay as exposure

    n = len(scene.pixels)
    if len(contact_targets_xyz_m) != n or len(bounce_targets) != n:
        raise ValueError("one contact anchor and one bounce group per connected flight required")
    if ground_directed_seed and any(
        groups is not None and len(groups) != n for groups in (ground_intervals, net_intervals)
    ):
        raise ValueError("one original event-interval group per connected flight required")
    if type(defer_final_net_to_initializer) is not bool:
        raise ValueError("explicit final-net deferral policy required")
    if defer_final_net_to_initializer:
        from cv.experiments.connected_shooting.observation_net_seed import single_final_net

        if not single_final_net(scene):
            raise ValueError("final-net deferral requires a single supplied final net")
    parameters = np.asarray(pinhole_seed, float).copy()
    parameters[:3] = np.asarray(first_contact_xyz_m, float)
    parameters[1] = float(first_contact_y_m)
    if flat_spin:
        parameters[3 + 3 * n : 3 + 6 * n] = 0.0
    axis_groups = np.split(
        np.asarray(axes, float), np.cumsum([len(row) for row in scene.pixels])[:-1]
    )
    end_xyz = None
    receipt = []
    for index in range(n - int(defer_final_net_to_initializer)):
        subscene = _replace(
            scene,
            contact_frames=scene.contact_frames[index : index + 2],
            terminal_net_tail=scene.terminal_net_tail if index == n - 1 else None,
            observed_horizon_tail=scene.observed_horizon_tail if index == n - 1 else None,
            observation_frames=(scene.observation_frames[index],),
            cameras=(scene.cameras[index],),
            camera_distortion=(
                None if scene.camera_distortion is None else (scene.camera_distortion[index],)
            ),
            pixels=(scene.pixels[index],),
            spin_parameters=scene.spin_parameters[index : index + 1],
            net_hit_frames=(
                None if scene.net_hit_frames is None else (scene.net_hit_frames[index],)
            ),
        )
        # A retained contact prefix has no passive ground-ending image context.
        # Preserve the default path for ordinary supplied-ending scenes.
        prediction_options = (
            {"termination_kind": _model.ORIGINAL_CONTACT_TERMINATION_KIND}
            if _model.original_contact_boundary(subscene)
            else {}
        )
        start_xyz = parameters[:3].copy() if end_xyz is None else end_xyz.copy()
        target_xyz = contact_targets_xyz_m[index]
        target_xyz = None if target_xyz is None else np.asarray(target_xyz, float)
        velocity = parameters[3 + 3 * index : 6 + 3 * index]
        spin = parameters[3 + 3 * n + 3 * index : 6 + 3 * n + 3 * index]
        guesses = [np.r_[velocity, spin]]
        if target_xyz is not None:
            duration = (scene.contact_frames[index + 1] - scene.contact_frames[index]) / scene.fps
            ballistic = (target_xyz - start_xyz) / duration
            ballistic[2] += 4.905 * duration
            spin_guesses = [0.0, 0.0, 0.0] if flat_spin else [2.0, 0.0, 0.0]
            guesses.append(np.r_[ballistic, spin_guesses])
            guesses.append(np.r_[ballistic, [0.0, 0.0, 0.0]])
        expected_bounces = np.asarray(bounces[index], float)
        anchors = bounce_targets[index]
        terminal_anchor_receipt = None
        if terminal_ground_anchor is not None and index == n - 1:
            if target_xyz is not None or not len(expected_bounces):
                raise ValueError("terminal ground anchor requires a final flight with a bounce")
            anchor_frame = float(terminal_ground_anchor["frame"])
            seconds = (anchor_frame - float(subscene.contact_frames[0])) / float(scene.fps)
            launch = ground_epoch_launch(
                start_xyz,
                seconds,
                velocity,
                target_xy=np.asarray(terminal_ground_anchor["xy_m"], float),
            )
            if launch is None:
                raise ValueError("terminal ground anchor has no supported launch")
            guesses = [np.r_[launch, spin]]
            expected_bounces = expected_bounces.copy()
            expected_bounces[0] = anchor_frame
            terminal_anchor_receipt = {
                "frame": anchor_frame,
                "xy_m": [float(v) for v in terminal_ground_anchor["xy_m"]],
                "launch_velocity_mps": launch.tolist(),
            }
        net_seed_receipt = None
        original_guess_count = len(guesses)
        if recover_terminal_net_seed and target_xyz is None:
            extra_guesses, net_seed_receipt = _native_net_anchor_guesses(
                subscene, start_xyz, spin, parameters[-2:], expected_bounces
            )
            guesses.extend(extra_guesses)
        ground_seed_receipt = None
        if ground_directed_seed:
            from cv.experiments.connected_shooting import ground_directed_seed as ground_seed

            ground_seed_receipt = {
                "status": "unavailable",
                "reason": "missing_original_ground_support",
            }
            intervals = [] if ground_intervals is None else ground_intervals[index]
            original_nets = [] if net_intervals is None else net_intervals[index]
            declared_nets = [] if subscene.net_hit_frames is None else subscene.net_hit_frames[0]

            def mapped_intervals(frames, mapped):
                if len(frames) != len(mapped):
                    return False
                return all(
                    interval is not None
                    and np.asarray(interval).shape == (2,)
                    and np.isfinite(interval).all()
                    and interval[0] <= frame <= interval[1]
                    for frame, interval in zip(frames, mapped, strict=True)
                )

            net_mapping_valid = mapped_intervals(declared_nets, original_nets)
            if subscene.terminal_net_tail is not None:
                original_interval = subscene.terminal_net_tail.get("interval")
                net_mapping_valid = net_mapping_valid and any(
                    np.array_equal(bounds, original_interval) for bounds in original_nets
                )
            if not net_mapping_valid:
                ground_seed_receipt["reason"] = "missing_or_mismatched_original_net_intervals"
            elif not mapped_intervals(expected_bounces, intervals):
                ground_seed_receipt["reason"] = "missing_or_mismatched_original_ground_intervals"
            elif len(expected_bounces) and anchors and anchors[0] is not None:
                extra_guess, ground_seed_receipt = ground_seed.propose(
                    start_xyz,
                    spin,
                    float(subscene.contact_frames[0]),
                    intervals[0],
                    anchors[0][0],
                    float(subscene.fps),
                    net_intervals=original_nets,
                )
                if extra_guess is not None:
                    guesses.append(extra_guess)
        residual_length = (
            2 * len(subscene.pixels[0])
            + (0 if target_xyz is None else 3)
            + len(expected_bounces)
            + 1
            + 2 * sum(row is not None for row in anchors)
        )

        def residual(
            candidate,
            start_xyz=start_xyz,
            target_xyz=target_xyz,
            subscene=subscene,
            index=index,
            expected_bounces=expected_bounces,
            anchors=anchors,
            residual_length=residual_length,
            reject_unsupported=False,
        ):
            try:
                full = np.r_[start_xyz, candidate[:3], candidate[3:], parameters[-2:]]
                flight = _model.chain(subscene, full)[0]
                image = (
                    exposure.prediction(
                        subscene,
                        full,
                        axis_groups[index],
                        exposure_duration,
                        **prediction_options,
                    )
                    - subscene.pixels[0]
                ).ravel() / 8.0
                endpoint = (
                    np.empty(0) if target_xyz is None else (flight["end_xyz"] - target_xyz) / 0.30
                )
                observed = flight["bounces"]
                timing, ground = [], []
                for ordinal, frame in enumerate(expected_bounces):
                    timing.append(
                        20.0
                        if ordinal >= len(observed)
                        else (observed[ordinal]["frame"] - frame) / 0.25
                    )
                for ordinal, anchor in enumerate(anchors):
                    if anchor is None:
                        continue
                    if ordinal >= len(observed):
                        ground.extend((0.0, 0.0))
                        continue
                    xy, sigma = anchor
                    ground.extend((np.asarray(observed[ordinal]["x"], float)[:2] - xy) / sigma)
                extras = observed[len(expected_bounces) :]
                timing.append(
                    0.0 if not extras else (flight["end_frame"] - extras[0]["frame"]) / 0.25
                )
                return np.r_[image, endpoint, timing, ground]
            except (ValueError, FloatingPointError, OverflowError):
                if reject_unsupported:
                    raise
                return np.full(residual_length, 1e5)

        lower = np.r_[[-75.0] * 3, [-6.0] * 3]
        upper = np.r_[[75.0] * 3, [6.0] * 3]
        solved = []
        unsupported_candidates = []

        def attempt(guess_index: int, guess) -> None:
            # Preserve existing guess preparation; newly admitted net proposals
            # already satisfy the physical bounds and must not be clipped.
            prepared_guess = (
                np.clip(guess, lower + 0.1, upper - 0.1)
                if guess_index < original_guess_count
                else np.asarray(guess, float)
            )
            candidate = least_squares(
                residual,
                prepared_guess,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
            try:
                values = residual(candidate.x, reject_unsupported=True)
            except (ValueError, FloatingPointError, OverflowError) as error:
                unsupported_candidates.append({"guess_index": guess_index, "reason": str(error)})
                return
            cost = float(np.sum(np.sqrt(1.0 + (values / 2.0) ** 2) - 1.0))
            if np.isfinite(cost):
                candidate.anchor_guess_index = guess_index
                solved.append((cost, candidate))

        for guess_index, guess in enumerate(guesses):
            attempt(guess_index, guess)
        ground_epoch_receipt = None
        if not solved and target_xyz is None and len(expected_bounces):
            # Every flight that has a next contact also has an ascending start:
            # its next-contact ballistic guess carries ``+4.905 * duration``.  A
            # flight without that anchor keeps only the disconnected pinhole
            # velocity, so a court-level connected start leaves it with no
            # supported launch at all and the whole walk is discarded.  Rebuild
            # one start from the witness this flight does own -- its declared
            # ground epoch and, when supplied, that ground ray -- and retry the
            # same bounded local solve.  Reached only where the walk already
            # failed outright, so a walk that solves is untouched.
            seconds = float(expected_bounces[0] - subscene.contact_frames[0]) / float(scene.fps)
            anchor_xy = None if not anchors or anchors[0] is None else anchors[0][0]
            recovery = ground_epoch_launch(start_xyz, seconds, velocity, target_xy=anchor_xy)
            ground_epoch_receipt = {
                "status": "unavailable" if recovery is None else "retried",
                "declared_ground_frame": float(expected_bounces[0]),
                "seconds_to_declared_ground": seconds,
                "uses_supplied_ground_ray": anchor_xy is not None,
            }
            if recovery is not None:
                ground_epoch_receipt["launch_velocity_mps"] = recovery.tolist()
                guesses.append(np.r_[recovery, spin])
                attempt(len(guesses) - 1, guesses[-1])
                ground_epoch_receipt["supported"] = bool(solved)
        if not solved:
            raise ValueError(f"flight {index}: no finite anchored seed")
        _, selected = min(solved, key=lambda row: row[0])
        parameters[3 + 3 * index : 6 + 3 * index] = selected.x[:3]
        parameters[3 + 3 * n + 3 * index : 6 + 3 * n + 3 * index] = selected.x[3:]
        flight = _model.chain(
            subscene, np.r_[start_xyz, selected.x[:3], selected.x[3:], parameters[-2:]]
        )[0]
        end_xyz = np.asarray(flight["end_xyz"], float)
        image = (
            exposure.prediction(
                subscene,
                np.r_[start_xyz, selected.x[:3], selected.x[3:], parameters[-2:]],
                axis_groups[index],
                exposure_duration,
                **prediction_options,
            )
            - subscene.pixels[0]
        ).ravel()
        receipt.append(
            {
                "flight_index": index,
                "optimizer_success": bool(selected.success),
                "objective_evaluations": int(selected.nfev),
                "training_pixel_rms": float(np.sqrt(np.mean(image**2))),
                "start_xyz_m": start_xyz.tolist(),
                "end_xyz_m": end_xyz.tolist(),
                "next_contact_anchor_xyz_m": (None if target_xyz is None else target_xyz.tolist()),
                "bounce_anchors_used": sum(row is not None for row in anchors),
                "terminal_net_seed": net_seed_receipt,
                **({"ground_directed_seed": ground_seed_receipt} if ground_directed_seed else {}),
                **(
                    {}
                    if ground_epoch_receipt is None
                    else {"ground_epoch_launch": ground_epoch_receipt}
                ),
                "selected_guess_index": int(selected.anchor_guess_index),
                "unsupported_candidates": unsupported_candidates,
                **(
                    {}
                    if terminal_anchor_receipt is None
                    else {"terminal_ground_anchor": terminal_anchor_receipt}
                ),
            }
        )
    lower = np.r_[[-10.0, -15.0, R_BALL], np.full(3 * n, -75.0), np.full(3 * n, -6.0), [0.8, 0.8]]
    upper = np.r_[[21.0, 40.0, 12.0], np.full(3 * n, 75.0), np.full(3 * n, 6.0), [1.2, 1.2]]
    bounded = np.clip(parameters, lower + 1e-5, upper - 1e-5)
    return bounded, {
        "method": "anchored_contact_ray_and_striker_walk",
        **(
            {"pending_final_net_initializer": True, "complete_forward_support": False}
            if defer_final_net_to_initializer
            else {}
        ),
        "flat_spin_restart": bool(flat_spin),
        "uses_bounce_ray_targets": True,
        "uses_withheld_pixels": False,
        "next_contact_anchor_scale_m": 0.30,
        "training_image_scale_px": 8.0,
        "bounce_timing_seed_scale_frames": 0.25,
        "clipped_parameter_indices": np.flatnonzero(parameters != bounded).tolist(),
        "flights": receipt,
    }
