"""Original-ray initializer for a continuous terminal net response.

The original three-picture gravity route remains preferred. A shorter net-to-
ground wing may instead use two original pictures and an existing supplied ground
witness to generate a fixed-spin drag proposal. Both routes retain source epochs;
full support is required for the ground route and is not a useful-fit certificate.

``qualify`` answers the same structural question from the original inputs alone,
without any numerical work, so a caller can decide whether this optional
candidate family exists before it builds anything.
"""

from dataclasses import replace

import numpy as np

from cv.experiments.connected_shooting import camera_geometry, model
from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity


METHOD = "three_consecutive_native_rays_gravity_seed_v1"
TWO_METHOD = "two_consecutive_native_rays_gravity_seed_v1"
GROUND_METHOD = "two_native_rays_supplied_ground_drag_seed_v1"
GRAVITY_SEED_METHODS = (METHOD, TWO_METHOD)


def single_final_net(scene):
    groups = scene.net_hit_frames
    return bool(
        groups is not None
        and len(groups) == len(scene.pixels)
        and sum(len(v) for v in groups) == 1
        and len(groups[-1]) == 1
    )


def net_contract(scene, physical_events=()):
    """Original tail, or a final net flight with an observed ground ending.

    The first ground supplies seed geometry. A later second ground can supply
    the physical ending without making the earlier net response unavailable.
    Neither ground is moved or removed from the source flight.
    """
    if scene.terminal_net_tail is not None:
        return scene.terminal_net_tail
    if not single_final_net(scene):
        return None
    epoch = float(scene.net_hit_frames[-1][0])
    nets = [
        e
        for e in physical_events
        if e.get("event_type") == "net_hit" and float(e["frame"]) == epoch
    ]
    if len(nets) != 1 or "frame_interval" not in nets[0]:
        return None
    high = float(nets[0]["frame_interval"][1])
    grounds = [
        e
        for e in physical_events
        if e.get("event_type") == "bounce"
        and high < float(e.get("frame_interval", [e["frame"]])[0])
        and float(e["frame"]) <= float(scene.contact_frames[-1])
    ]
    if not grounds:
        return None
    ground = min(grounds, key=lambda e: float(e["frame"]))
    endings = [
        e
        for e in physical_events
        if e.get("event_type") == "ending"
        and float(e["frame"]) == float(ground["frame"])
        and e.get("frame_interval") == ground.get("frame_interval")
    ]
    contract = dict(
        interval=list(nets[0]["frame_interval"]),
        representative=epoch,
        ground_interval=list(ground["frame_interval"]),
        ground_representative=float(ground["frame"]),
    )
    if endings:
        return contract
    # An in-court first bounce does not end the point. Qualify the original
    # second-bounce ending while retaining the first bounce as the seed target.
    ordered = sorted(grounds, key=lambda e: float(e["frame"]))
    if len(ordered) < 2:
        return None
    second = ordered[1]
    bounds = np.asarray(second.get("frame_interval", []), float)
    first_bounds = np.asarray(ground.get("frame_interval", []), float)
    if (
        bounds.shape != (2,)
        or first_bounds.shape != (2,)
        or not np.isfinite(np.r_[first_bounds, bounds]).all()
        or not first_bounds[0] <= float(ground["frame"]) <= first_bounds[1] < bounds[0]
        or not bounds[0] <= float(second["frame"]) <= bounds[1] <= scene.contact_frames[-1]
    ):
        return None
    endings = [
        e
        for e in physical_events
        if e.get("event_type") == "ending"
        and float(e["frame"]) == float(second["frame"])
        and e.get("frame_interval") == second.get("frame_interval")
    ]
    if len(endings) != 1:
        return None
    for event in physical_events:
        if event.get("event_type") != "contact":
            continue
        event_bounds = event.get("frame_interval", [event["frame"], event["frame"]])
        if float(event_bounds[1]) >= high and float(event_bounds[0]) <= bounds[1]:
            return None
    return contract | dict(
        ending_ground_interval=bounds.tolist(),
        ending_ground_representative=float(second["frame"]),
        original_post_net_ground_frames=[float(ground["frame"]), float(second["frame"])],
    )


def ground_support(scene, physical_events, ground_targets, contract):
    """Bind the first post-net ground to its existing original witness by epoch."""
    high = float(contract["interval"][1])
    grounds = sorted(
        (
            e
            for e in physical_events
            if e.get("event_type") == "bounce"
            and float(e["frame"]) > high
            and float(e["frame"]) <= float(scene.contact_frames[-1])
        ),
        key=lambda e: float(e["frame"]),
    )
    if not grounds or not ground_targets or len(ground_targets) != len(scene.pixels):
        return None
    ground = grounds[0]
    interval = np.asarray(ground.get("frame_interval", []), float)
    if (
        interval.shape != (2,)
        or not np.isfinite(interval).all()
        or not high < interval[0] <= float(ground["frame"]) <= interval[1]
    ):
        return None
    for event in physical_events:
        if event.get("event_type") not in ("contact", "net_hit"):
            continue
        if event["event_type"] == "net_hit" and float(event["frame"]) == float(
            contract["representative"]
        ):
            continue
        bounds = event.get("frame_interval", [event["frame"], event["frame"]])
        if float(event["frame"]) >= high and float(bounds[0]) <= interval[1]:
            return None
    targets = [
        t
        for t in ground_targets[-1]
        if t is not None and float(t.get("event_frame", -np.inf)) == float(ground["frame"])
    ]
    if len(targets) != 1:
        return None
    target = targets[0]
    xyz = np.asarray(target.get("xyz_m", []), float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or not target.get("native_frames"):
        return None
    return dict(event_frame=float(ground["frame"]), interval=interval.tolist(), witness=target)


def native_source(scene, heldout):
    """Exact union for initialization; already active check rows are not duplicated.

    The union is taken only when the fitted scene already owns every native
    picture.  Under a withheld partition the check scene is this run's declared
    evaluation split, not a further supplied input: merging it would seed the
    search from the very pictures the run scores after selection, so the seed
    keeps the fitted rows alone.  The source metadata agreement below is checked
    either way, so an inconsistent check copy still refuses the family.
    """
    from cv.experiments.connected_shooting import observation_partition as partition

    fields = ("observation_frames", "pixels", "cameras", "camera_distortion")
    if (scene.camera_distortion is None) != (heldout.camera_distortion is None):
        raise ValueError("native camera distortion metadata differs")
    if not np.array_equal(scene.contact_frames, heldout.contact_frames):
        raise ValueError("native source contact scope differs")
    if not partition.all_native(scene):
        return scene
    active = fields[:-1] if scene.camera_distortion is None else fields
    merged = {name: [] for name in active}
    for i in range(len(scene.pixels)):
        rows = {}
        for source in (scene, heldout):
            for j, frame in enumerate(source.observation_frames[i]):
                values = {name: getattr(source, name)[i][j] for name in active}
                if frame in rows and not all(
                    np.array_equal(values[k], rows[frame][k]) for k in active
                ):
                    raise ValueError("overlapping native source row metadata differs")
                rows[frame] = values
        for name in active:
            merged[name].append(np.asarray([rows[f][name] for f in sorted(rows)]))
    return replace(scene, **{name: tuple(groups) for name, groups in merged.items()})


def _intervening_event(contract, physical_events, low, last_frame):
    """Return the refusal reason when another declared event covers the wing."""
    for event in physical_events:
        if event.get("event_type") not in ("bounce", "contact", "net_hit"):
            continue
        interval = event.get("frame_interval", (event["frame"], event["frame"]))
        if event["event_type"] == "net_hit" and float(event["frame"]) == float(
            contract["representative"]
        ):
            continue
        if float(event["frame"]) < low:
            continue
        if float(interval[0]) <= last_frame:
            return "post-net seed wing intersects another physical event interval"
    return None


def _finite(value):
    return value is not None and np.isfinite(np.asarray(value, float)).all()


def qualify(scene, *, physical_events=(), ground_targets=None):
    """Report whether the original inputs alone support this seed family.

    Only declared scene inputs are read: no integration, root finding or least
    squares runs here, and ``initialize`` is not called.  A structurally absent
    candidate is ``unavailable`` with a reason; nothing else is caught.
    """

    def refuse(reason, **known):
        return dict(status="unavailable", reason=reason, method=METHOD, **known)

    contract = net_contract(scene, physical_events)
    if contract is None or scene.dynamics != "measured_240hz":
        return refuse("declared terminal net tail with measured dynamics required")
    if not isinstance(contract, dict) or not {"interval", "representative"} <= set(contract):
        return refuse("declared terminal net tail lacks its interval and representative")
    interval = np.asarray(contract["interval"], float).ravel()
    if interval.shape != (2,) or not _finite(interval) or interval[0] > interval[1]:
        return refuse("ordered finite declared net interval required")
    low, high = (float(value) for value in interval)
    if not _finite(contract["representative"]):
        return refuse("finite declared net representative frame required")
    if not low <= float(contract["representative"]) <= high:
        return refuse("declared net representative must belong to its original interval")
    known = dict(supplied_net_interval=[low, high])
    if not single_final_net(scene):
        return refuse("single terminal net required", **known)
    if float(scene.net_hit_frames[-1][0]) != float(contract["representative"]):
        return refuse("net interval must bind the supplied final net", **known)
    frames, cameras, pixels = (
        np.asarray(scene.observation_frames[-1], float),
        np.asarray(scene.cameras[-1], float),
        np.asarray(scene.pixels[-1], float),
    )
    lens = None if scene.camera_distortion is None else np.asarray(scene.camera_distortion[-1])
    rows = len(frames)
    if (
        frames.ndim != 1
        or cameras.shape[:1] != (rows,)
        or cameras.shape[1:] != (3, 4)
        or pixels.shape != (rows, 2)
        or (lens is not None and (lens.shape != (rows, 3)))
    ):
        return refuse("consistent original frame, camera and pixel rows required", **known)
    if not (_finite(frames) and _finite(scene.contact_frames) and _finite(scene.fps)):
        return refuse("finite original frames and declared epochs required", **known)
    known["full_original_horizon"] = float(scene.contact_frames[-1])
    if not (frames <= high).any():
        return refuse("original incoming observations required", **known)
    eligible = np.flatnonzero((frames >= low) & (frames <= high))
    if not len(eligible):
        return refuse("original net-interval native ray required", **known)
    net_index = min(
        eligible,
        key=lambda i: (abs(frames[i] - float(contract["representative"])), frames[i]),
    )
    known.update(net_index=int(net_index), net_frame=float(frames[net_index]))
    # Earliest consecutive original post-interval pictures. A later convenient
    # triplet would silently interpolate across a missing picture. Three rays
    # are preferred; two consecutive rays already identify velocity if rank
    # and condition pass. That two-ray route is a seed, not acceptance.
    wing = np.flatnonzero(frames > high)[:3]
    three_rows = len(wing) == 3 and np.all(np.diff(frames[wing]) == 1)
    two_rows = len(wing) >= 2 and np.all(np.diff(frames[wing[:2]]) == 1)
    if three_rows:
        known.update(original_indices=wing.tolist(), original_frames=frames[wing].tolist())
        reason = _intervening_event(contract, physical_events, low, frames[wing][-1])
        method = METHOD
    elif two_rows:
        wing = wing[:2]
        known.update(original_indices=wing.tolist(), original_frames=frames[wing].tolist())
        reason = _intervening_event(contract, physical_events, low, frames[wing][-1])
        method = TWO_METHOD
    else:
        reason = "three consecutive original post-net pictures required"
        method = METHOD
    ground = None
    if reason is not None:
        ground = ground_support(scene, physical_events, ground_targets, contract)
        two = wing[:2]
        if (
            ground is None
            or len(two) != 2
            or not np.all(np.diff(frames[two]) == 1)
            or not np.all(frames[two] < ground["interval"][0])
            or (len(wing) == 3 and frames[wing[2]] < ground["interval"][0])
            or _intervening_event(contract, physical_events, low, frames[two][-1]) is not None
        ):
            return refuse(reason, **known)
        wing, method = two, GROUND_METHOD
    known.update(original_indices=wing.tolist(), original_frames=frames[wing].tolist())
    used = np.r_[net_index, wing]
    if not (
        _finite(cameras[used]) and _finite(pixels[used]) and (lens is None or _finite(lens[used]))
    ):
        return refuse("finite cameras, pixels and declared lens rows required", **known)
    return dict(status="supported", reason=None, method=method, ground_support=ground, **known)


def initialize(
    scene,
    parameters,
    *,
    physical_events=(),
    ground_targets=None,
    speed_scale_mps=15.0,
    deadline_check=None,
):
    """Return a response and evidence, or refuse insufficient/ill-conditioned rays."""

    def check_deadline():
        if deadline_check is not None:
            deadline_check()

    if (
        isinstance(speed_scale_mps, bool)
        or not np.isfinite(speed_scale_mps)
        or speed_scale_mps <= 0
    ):
        raise ValueError("positive finite net seed speed scale required")
    contract = net_contract(scene, physical_events)
    if contract is None or scene.dynamics != "measured_240hz":
        raise ValueError("declared terminal net tail with measured dynamics required")
    qualification = qualify(scene, physical_events=physical_events, ground_targets=ground_targets)
    if qualification["status"] != "supported":
        raise ValueError(qualification["reason"])
    low, high = map(float, contract["interval"])
    if not single_final_net(scene):
        raise ValueError("single terminal net required")
    # Propagate the incoming seed through the original interval, without asking
    # the fixed response to explain the whole passive aftermath first.
    masks = [np.ones(len(f), dtype=bool) for f in scene.observation_frames]
    masks[-1] = scene.observation_frames[-1] <= high
    if not masks[-1].any():
        raise ValueError("original incoming observations required")
    contacts = scene.contact_frames.copy()
    contacts[-1] = high
    incoming = replace(
        scene,
        contact_frames=contacts,
        terminal_net_tail=None,
        observation_frames=tuple(
            v[m] for v, m in zip(scene.observation_frames, masks, strict=True)
        ),
        cameras=tuple(v[m] for v, m in zip(scene.cameras, masks, strict=True)),
        pixels=tuple(v[m] for v, m in zip(scene.pixels, masks, strict=True)),
        camera_distortion=None
        if scene.camera_distortion is None
        else tuple(v[m] for v, m in zip(scene.camera_distortion, masks, strict=True)),
    )
    from cv.experiments.connected_shooting.labeled_preparation_physical_seed import (
        solve_flight,
    )

    from cv.experiments.connected_shooting.agent_whole_point_search import terminal_seed_start

    parameters = np.asarray(parameters, float).copy()
    # The final launch is determined by preceding flights alone. The unseeded
    # final extrapolation can exhaust the bounce cap before its native net ray
    # has initialized velocity. Full incoming and aftermath replay still follow.
    check_deadline()
    start_xyz = terminal_seed_start(incoming, parameters)
    check_deadline()
    count = len(scene.pixels)
    last = count - 1
    vi = 3 + 3 * last
    si = 3 + 3 * count + 3 * last
    has_spin = len(parameters) - (2 if scene.rebound_mode == "point_scales" else 0) == 3 + 6 * count
    if not has_spin:
        raise ValueError("explicit incoming spin seed required")
    eligible = np.flatnonzero(
        (scene.observation_frames[-1] >= low) & (scene.observation_frames[-1] <= high)
    )
    if not len(eligible):
        raise ValueError("original net-interval native ray required")
    net_index = min(
        eligible,
        key=lambda i: (
            abs(scene.observation_frames[-1][i] - contract["representative"]),
            scene.observation_frames[-1][i],
        ),
    )
    lens = (
        None
        if scene.camera_distortion is None
        else scene.camera_distortion[-1][net_index : net_index + 1]
    )
    from cv.pipeline.physics_knot_solver import net_tape_height

    from scipy.optimize import least_squares

    def xyz_from_net(q):
        return np.array([q[0], 11.885, model.R_BALL + q[1] * net_tape_height(q[0])])

    def ray_residual(q):
        check_deadline()
        return (
            camera_geometry.project(
                scene.cameras[-1][net_index : net_index + 1], xyz_from_net(q)[None], lens
            )[0]
            - scene.pixels[-1][net_index]
        )

    net_fit = least_squares(
        ray_residual,
        [5.485, 0.5],
        bounds=([-0.915, 1e-5], [11.885, 1.0 - 1e-5]),
        max_nfev=60,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    net_target = xyz_from_net(net_fit.x)
    native_net_residual = ray_residual(net_fit.x)
    if not net_fit.success or not np.isfinite(native_net_residual).all():
        raise ValueError("native net-ray constrained initialization failed")
    arguments = (
        start_xyz,
        parameters[vi : vi + 3],
        net_target,
        scene.contact_frames[-2],
        contract["representative"],
        scene.fps,
        scene.surface,
        0,
    )
    try:
        velocity, _, projection = solve_flight(*arguments)
    except (ValueError, FloatingPointError, OverflowError) as error:
        projection = {"usable": False, "error": f"{type(error).__name__}: {error}"}
    check_deadline()
    if not projection["usable"]:
        # The incoming source velocity can already be on a bounced branch.
        # Retry the same required airborne endpoint solve from its ordinary
        # ballistic arc; source topology/target/tolerance do not change.
        refused = projection
        seconds = (arguments[4] - arguments[3]) / arguments[5]
        ballistic = (net_target - arguments[0]) / seconds
        ballistic[2] += 0.5 * 9.81 * seconds
        velocity, _, projection = solve_flight(arguments[0], ballistic, *arguments[2:])
        check_deadline()
        projection["airborne_initialization_retry"] = {
            "original_refusal": refused,
            "initial_velocity_mps": ballistic.tolist(),
            "original_target_and_zero_ground_requirement_retained": True,
        }
    if not projection["usable"]:
        raise ValueError("native-net incoming seed solve unsupported")
    projection["native_frame"] = float(scene.observation_frames[-1][net_index])
    projection["zero_spin_initializer_only"] = True
    projection["native_net_residual_px"] = native_net_residual.tolist()
    projection["net_state_bound_active"] = bool(net_fit.x[1] > 0.999)
    parameters[vi : vi + 3] = velocity
    parameters[si : si + 3] = 0.0
    chain = model.chain(incoming, parameters)
    hits = chain[-1]["net_hits"]
    if len(hits) != 1:
        raise ValueError("incoming seed has no continuous physical net intersection")
    hit = hits[0]
    if qualification["method"] == GROUND_METHOD:
        return _ground_response(
            scene, parameters, hit, projection, qualification, check_deadline, contract
        )
    rows = np.asarray(qualification["original_indices"], int)
    frames = scene.observation_frames[-1][rows]
    if qualification["method"] == TWO_METHOD:
        if len(rows) != 2 or not np.all(np.diff(frames) == 1):
            raise ValueError("two consecutive original post-net pictures required")
    elif qualification["method"] != METHOD or len(rows) != 3 or not np.all(np.diff(frames) == 1):
        raise ValueError("three consecutive original post-net pictures required")
    reason = _intervening_event(contract, physical_events, low, frames[-1])
    if reason is not None:
        raise ValueError(reason)
    cameras = scene.cameras[-1][rows]
    radial = None if scene.camera_distortion is None else scene.camera_distortion[-1][rows]
    pixels = camera_geometry.undistort(scene.pixels[-1][rows], radial)
    dt = (frames - float(hit["frame"])) / scene.fps
    origins = np.repeat(np.asarray(hit["x"])[None], len(rows), axis=0)
    origins[:, 2] -= 0.5 * 9.81 * dt**2
    equations = cameras[:, :2] - pixels[:, :, None] * cameras[:, 2:3]
    design = (equations[:, :, :3] * dt[:, None, None]).reshape(-1, 3)
    target = -np.einsum("nij,nj->ni", equations, np.c_[origins, np.ones(len(rows))]).ravel()
    velocity, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if rank != 3 or not np.isfinite(condition) or condition > 1e8:
        raise ValueError("post-net native rays do not identify an outgoing velocity")

    def outgoing_residual(v):
        check_deadline()
        prediction = camera_geometry.project(cameras, origins + dt[:, None] * v, radial)
        # Broad, direction-free numerical regularization: three very close rays
        # can fit depth with implausibly large speed. This is initialization only.
        return np.r_[(prediction - scene.pixels[-1][rows]).ravel(), v / speed_scale_mps]

    outgoing_fit = least_squares(
        outgoing_residual,
        np.asarray(hit["v_out"]),
        bounds=(-75.0, 75.0),
        max_nfev=60,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    velocity = outgoing_fit.x
    if not outgoing_fit.success:
        raise ValueError("regularized original-ray seed did not converge")
    try:
        response = FreeNetVelocity(velocity)
    except ValueError as exc:
        raise ValueError(
            f"outgoing seed unsupported: velocity={velocity.tolist()}, net={hit['x'].tolist()}, epoch={hit['frame']}, condition={condition}"
        ) from exc
    prediction = camera_geometry.project(cameras, origins + dt[:, None] * velocity, radial)
    receipt = dict(
        method=qualification["method"],
        incoming_chart=projection,
        **({"net_before_ground_contract": contract} if scene.terminal_net_tail is None else {}),
        original_frames=frames.tolist(),
        original_pixels=scene.pixels[-1][rows].tolist(),
        net_xyz_m=np.asarray(hit["x"]).tolist(),
        net_frame=float(hit["frame"]),
        supplied_net_interval=[low, high],
        incoming_speed_mps=float(np.linalg.norm(hit["v_in"])),
        outgoing_speed_mps=float(np.linalg.norm(velocity)),
        translational_energy_ratio=float(
            np.dot(velocity, velocity) / np.dot(hit["v_in"], hit["v_in"])
        ),
        condition_number=condition,
        outgoing_regularization_mps=float(speed_scale_mps),
        outgoing_nfev=outgoing_fit.nfev,
        native_seed_rms_px=float(
            np.sqrt(np.mean(np.sum((prediction - scene.pixels[-1][rows]) ** 2, axis=1)))
        ),
        response=dict(outgoing_velocity_mps=velocity.tolist()),
        full_original_horizon=float(scene.contact_frames[-1]),
        ground_epoch_supplied=False,
        selection_reads_acceptance=False,
        approximation="short gravity-only initializer; an unreported early bounce remains possible",
    )
    return parameters, response, receipt


def _ground_response(scene, parameters, hit, projection, qualification, check_deadline, contract):
    from cv.experiments.connected_shooting import ground_directed_seed
    from cv.experiments.connected_shooting.labeled_passive_tape import using_response

    ground = qualification["ground_support"]
    check_deadline()
    guess, shooting = ground_directed_seed.propose(
        np.asarray(hit["x"]),
        np.zeros(3),
        float(hit["frame"]),
        ground["interval"],
        ground["witness"]["xyz_m"][:2],
        scene.fps,
    )
    if guess is None:
        raise ValueError(f"supplied-ground outgoing seed unsupported: {shooting}")
    rows = np.asarray(qualification["original_indices"], int)
    frames = scene.observation_frames[-1][rows]
    cameras = scene.cameras[-1][rows]
    radial = None if scene.camera_distortion is None else scene.camera_distortion[-1][rows]
    seconds = (frames - float(hit["frame"])) / scene.fps

    def predicted(velocity):
        check_deadline()
        xyz = np.asarray(
            [
                ground_directed_seed.free_arc(
                    np.asarray(hit["x"]), velocity, np.zeros(3), float(t)
                )[0][-1]
                for t in seconds
            ]
        )
        return camera_geometry.project(cameras, xyz, radial)

    velocity = guess[:3]
    epsilon = 1e-4
    jacobian = np.column_stack(
        [
            (
                (
                    predicted(velocity + np.eye(3)[j] * epsilon)
                    - predicted(velocity - np.eye(3)[j] * epsilon)
                )
                / (2 * epsilon)
            ).ravel()
            for j in range(3)
        ]
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if rank != 3 or not np.isfinite(condition) or condition > 1e8:
        raise ValueError("post-net native rays do not identify an outgoing velocity")
    response = FreeNetVelocity(velocity)
    # Complete support uses the caller's declared physical policy. This may
    # still refuse a settling tail; initialization never toggles that setting.
    with using_response(response):
        flights = model.chain(scene, parameters)
    last = flights[-1]
    if len(last["net_hits"]) != 1 or not last["bounces"]:
        raise ValueError("supplied net and ground require complete physical forwarding")
    first_ground = float(last["bounces"][0]["frame"])
    low, high = ground["interval"]
    if not low - 1e-7 <= first_ground <= high + 1e-7:
        raise ValueError("outgoing seed does not reach the supplied first-ground interval")
    from cv.experiments.connected_shooting.labeled_terminal_net_tail import mesh_response_slacks_m

    receipt = dict(
        method=GROUND_METHOD,
        incoming_chart=projection,
        **({"net_before_ground_contract": contract} if scene.terminal_net_tail is None else {}),
        original_frames=frames.tolist(),
        original_pixels=scene.pixels[-1][rows].tolist(),
        net_xyz_m=np.asarray(hit["x"]).tolist(),
        net_frame=float(hit["frame"]),
        supplied_net_interval=qualification["supplied_net_interval"],
        ground_support=ground,
        ground_shooting=shooting,
        ground_native_evidence_may_be_correlated=True,
        condition_number=condition,
        native_velocity_jacobian_rank=rank,
        native_seed_rms_px=float(
            np.sqrt(np.mean(np.sum((predicted(velocity) - scene.pixels[-1][rows]) ** 2, axis=1)))
        ),
        response=dict(outgoing_velocity_mps=velocity.tolist()),
        full_original_horizon=float(scene.contact_frames[-1]),
        complete_forward_support=True,
        ground_settling=bool(scene.ground_settling),
        initial_mesh_slacks_m=mesh_response_slacks_m(
            scene, flights, include_supplied_final=True
        ).tolist(),
        ground_epoch_supplied=True,
        selection_reads_acceptance=False,
        approximation="fixed native net chart and zero-spin ground proposal; uncertain correlated witness; not a useful-fit certificate",
    )
    return parameters, response, receipt


def require_ground_ending_binding(scene, receipt, *, bounces=None):
    """A non-tail replay must retain its original net-before-ground source scope."""
    contract = receipt.get("net_before_ground_contract") if isinstance(receipt, dict) else None
    if not single_final_net(scene) or not isinstance(contract, dict):
        raise ValueError(
            "prebuilt net response replay requires a single-shooting net tail or original ground-ending binding"
        )
    interval = np.asarray(contract.get("interval", []), float)
    ground = np.asarray(contract.get("ground_interval", []), float)
    net_frame = float(scene.net_hit_frames[-1][0])
    if (
        interval.shape != (2,)
        or ground.shape != (2,)
        or not np.isfinite(np.r_[interval, ground]).all()
        or contract.get("representative") != net_frame
        or not interval[0] <= net_frame <= interval[1] < ground[0]
        or not ground[0] <= contract.get("ground_representative", -np.inf) <= ground[1]
        or ground[1] > scene.contact_frames[-1]
    ):
        raise ValueError("invalid original ground-ending binding")
    if bounces is not None and (
        len(bounces) != len(scene.pixels)
        or len(bounces[-1]) == 0
        or float(bounces[-1][0]) != float(contract["ground_representative"])
    ):
        raise ValueError("original ground-ending binding differs from supplied bounce inventory")
    if "ending_ground_interval" in contract:
        ending = np.asarray(contract["ending_ground_interval"], float)
        original = contract.get("original_post_net_ground_frames")
        if (
            ending.shape != (2,)
            or not np.isfinite(ending).all()
            or not ground[1]
            < ending[0]
            <= contract.get("ending_ground_representative", -np.inf)
            <= ending[1]
            <= scene.contact_frames[-1]
            or original
            != [contract["ground_representative"], contract["ending_ground_representative"]]
        ):
            raise ValueError("invalid original second-ground ending binding")
        if bounces is not None and (
            len(bounces[-1]) < 2 or list(map(float, bounces[-1][:2])) != original
        ):
            raise ValueError("original second-ground ending differs from supplied bounce inventory")
    return contract


def _compose(outer, inner):
    from contextlib import ExitStack

    class _both:
        def __enter__(self):
            self._stack = ExitStack()
            self._stack.__enter__()
            self._stack.enter_context(outer)
            self._stack.enter_context(inner)
            return self

        def __exit__(self, *exc):
            return self._stack.__exit__(*exc)

    return _both()


def response_context(fit):
    """Replay only a response bound to this numerical candidate's initializer."""
    from contextlib import nullcontext
    from cv.experiments.connected_shooting.labeled_passive_tape import using_response
    from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
    from cv.pipeline import net_cord_response as cord

    evidence = fit.get("net_cord_evidence") if isinstance(fit, dict) else None
    if isinstance(evidence, list) and evidence and not fit.get("_evidence_context_open"):
        bounds = cord.bounds_from_records(evidence)
        h_tol = next((bound.h_tol_m for bound in bounds if bound.admitted), None)
        if h_tol is None and bounds:
            h_tol = bounds[0].h_tol_m
        if any(bound.admitted for bound in bounds):
            response = fit.get("net_response") if isinstance(fit.get("net_response"), dict) else {}
            velocity = response.get("outgoing_velocity_mps") or response.get("v_out_mps")
            locked = nullcontext()
            if velocity is not None:
                locked = using_response(response_from_record({"outgoing_velocity_mps": velocity}))
            return _compose(cord.using_evidence(bounds, h_tol=h_tol), locked)
        inner = {key: value for key, value in fit.items() if key != "net_cord_evidence"}
        inner["_evidence_context_open"] = True
        return _compose(cord.using_evidence(bounds, h_tol=h_tol), response_context(inner))
    response = fit.get("net_response")
    receipt = fit.get("net_response_initialization")
    if response is None and receipt is None:
        return nullcontext()
    if cord.active_mode() == cord.ADMISSIBLE_SET and isinstance(response, dict):
        from cv.experiments.connected_shooting.admissible_net_response import (
            from_record as admissible_from_record,
            using_admissible,
        )

        record = response if "outgoing_velocity_mps" in response else {
            "outgoing_velocity_mps": response.get("v_out_mps")
        }
        if record.get("outgoing_velocity_mps") is not None:
            return using_admissible(admissible_from_record(record))
    if (
        not isinstance(receipt, dict)
        or receipt.get("method") not in (*GRAVITY_SEED_METHODS, GROUND_METHOD)
        or receipt.get("selection_reads_acceptance") is not False
        or receipt.get("ground_epoch_supplied") is not (receipt.get("method") == GROUND_METHOD)
        or (receipt.get("method") == GROUND_METHOD and not receipt.get("complete_forward_support"))
        or response != receipt.get("response")
    ):
        raise ValueError("numerical response lacks its original-observation initializer binding")
    return using_response(response_from_record(response))
