"""Stage 6 backward validation: oracle XYZ capacity versus exact native image lifting.

Development-only known-truth experiment, never an automatic inference entrypoint.
Both arms receive exact physical boundaries, synthetic cameras and favorable
truth-derived initialization. XYZ additionally receives training-frame 3D truth.
Neither arm receives scoring-frame XYZ in its objective. All source points stay
in the denominator, including unsupported and timed-out cases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import signal
import time

import numpy as np

from cv.experiments.connected_shooting import model, event_constraints
from cv.experiments.connected_shooting.measured_dynamics import BOUNCE_PROFILES
from cv.pipeline.provenance import file_record, git_record
from cv.validation import owner_spatial_audit, s6_point_bench
from cv.validation.owner_spatial_audit import CONTRACT
from cv.validation.s6_point_bench import truth_flight_paths


def sample(path: dict, frames: np.ndarray) -> np.ndarray:
    if frames[0] < path["frames"][0] or frames[-1] > path["frames"][-1]:
        raise ValueError("truth must cover every requested time; no extrapolation")
    return np.column_stack(
        [np.interp(frames, path["frames"], path["positions"][:, axis]) for axis in range(3)]
    )


def noisy_pixels(scene: model.Scene, point_id: str, noise_seed: int | None = None) -> model.Scene:
    """Independent training-pixel noise; None preserves the original draw exactly.

    An explicit replicate namespaces the point-specific seed. Scheduling, other
    points and held-out images do not consume this random stream.
    """
    import hashlib

    if noise_seed is not None and (type(noise_seed) is not int or not 0 <= noise_seed < 2**32):
        raise ValueError("noise seed must be an integer in [0, 2**32)")
    identity = point_id if noise_seed is None else f"noise_replicate_v1:{noise_seed}:{point_id}"
    digits = 8 if noise_seed is None else 16
    seed_id = int(hashlib.sha256(identity.encode()).hexdigest()[:digits], 16)
    rng = np.random.default_rng(seed_id)
    return replace(scene, pixels=tuple(p + rng.normal(0, 2, p.shape) for p in scene.pixels))


def prepare(point: dict, camera_path: Path) -> tuple:
    if point.get("trajectory_truth", {}).get("schema") != "simulation_trajectory_samples_v1":
        raise ValueError("recorded exact truth required; approximate truth cannot substitute")
    paths = truth_flight_paths(point, scored_only=True)
    boundaries = np.r_[[p["frames"][0] for p in paths], paths[-1]["frames"][-1]]
    if any(a["frames"][-1] != b["frames"][0] for a, b in zip(paths, paths[1:])):
        raise ValueError("truth flight boundaries are not connected")
    with np.load(camera_path, allow_pickle=False) as source:
        selected = source["clips"].astype(str) == point["clip"]
        frames = source["frames"][selected]
        matrices = source["P"][selected]
    if len(np.unique(frames)) != len(frames):
        raise ValueError("duplicate oracle camera exposures")
    camera = dict(zip(frames.tolist(), matrices, strict=True))
    groups, cameras, pixels, xyz_groups = [], [], [], []
    test_frames, test_cameras, test_pixels = [], [], []
    velocities = []
    for index, path in enumerate(paths):
        # Use only pictures the synthetic generator declared visible. Fractional
        # contact times and dense hidden 3D are evaluation/conditioning, not pictures.
        visible = np.asarray(point["frames"], float)
        before_end = (
            visible <= path["frames"][-1]
            if index == len(paths) - 1
            else visible < path["frames"][-1]
        )
        # An integer contact exposure belongs to its outgoing flight once,
        # matching the human adapter. The final ground horizon stays inclusive.
        visible = visible[(visible >= path["frames"][0]) & before_end]
        if np.any(visible != np.rint(visible)) or np.any(np.diff(visible) <= 0):
            raise ValueError("invalid native visible frame inventory")
        train = visible[(visible.astype(int) % 3) != 0]
        test = visible[(visible.astype(int) % 3) == 0]
        if len(train) < 4 or not len(test):
            raise ValueError("insufficient training/held-out native pictures")
        for f, fg, pg, xyg in [
            (train, groups, cameras, pixels),
            (test, test_frames, test_cameras, test_pixels),
        ]:
            P = np.stack([camera[int(frame)] for frame in f])
            xyz = sample(path, f)
            q = np.einsum("nij,nj->ni", P, np.c_[xyz, np.ones(len(xyz))])
            if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-9):
                raise ValueError("invalid oracle projection")
            fg.append(f)
            pg.append(P)
            xyg.append(q[:, :2] / q[:, 2:])
        xyz_groups.append(sample(path, train))
        # Deliberately privileged starting velocity, estimated from the recorded
        # outgoing interpolant, not claimed available to automatic inference.
        t = (path["frames"][:5] - path["frames"][0]) / point["fps"]
        v = np.polynomial.polynomial.polyfit(t, path["positions"][:5], 2)[1]
        velocities.append(v + np.array([0.3, -0.4, 0.2]))
    scene = model.Scene(
        boundaries,
        tuple(groups),
        tuple(cameras),
        tuple(pixels),
        np.tile([2.0, 0.0, 0.0], (len(paths), 1)),
        float(point["fps"]),
        point["surface"],
    )
    scene.validate()
    seed = np.r_[
        paths[0]["positions"][0] + [0.2, -0.3, 0.1],
        np.asarray(velocities).ravel(),
        scene.spin_parameters.ravel(),
    ]
    heldout = replace(
        scene,
        observation_frames=tuple(test_frames),
        cameras=tuple(test_cameras),
        pixels=tuple(test_pixels),
    )
    heldout.validate()
    return scene, seed, tuple(xyz_groups), paths, heldout


def measure(scene: model.Scene, result: dict, paths: list[dict], heldout: model.Scene) -> dict:
    dense = model.chain(scene, result["parameters"], query_frames=tuple(p["frames"] for p in paths))
    rows = []
    for fitted, truth in zip(dense, paths, strict=True):
        errors = np.linalg.norm(fitted["positions"] - truth["positions"], axis=1)
        rows.append(
            {
                "flight_index": truth["flight_index"],
                "samples": len(errors),
                "trajectory_rms_m": float(np.sqrt(np.mean(errors**2))),
                "trajectory_peak_m": float(errors.max()),
                "start_error_m": float(errors[0]),
                "end_error_m": float(errors[-1]),
                "minimum_height_m": float(fitted["positions"][:, 2].min()),
                "modeled_bounce_frames": [float(b["frame"]) for b in fitted["bounces"]],
                "bounce_profile_evidence": [
                    {
                        k: v
                        for k, v in b.items()
                        if k.startswith(
                            ("requested_", "applied_", "coefficient_", "bounce_profile")
                        )
                    }
                    for b in fitted["bounces"]
                    if "bounce_profile" in b
                ],
            }
        )
    spatial = all(
        r["trajectory_rms_m"] <= CONTRACT["trajectory_rms_tolerance_m"]
        and r["trajectory_peak_m"] <= CONTRACT["trajectory_peak_ceiling_m"]
        and r["start_error_m"] <= CONTRACT["contact_tolerance_m"]
        and r["end_error_m"]
        <= (
            CONTRACT["termination_tolerance_m"]
            if i == len(rows) - 1
            else CONTRACT["contact_tolerance_m"]
        )
        for i, r in enumerate(rows)
    )
    return {
        "flights": rows,
        "trajectory_and_endpoint_tolerances_met": spatial,
        "heldout_pixel_rms": float(
            np.sqrt(np.mean(model.image_residual(heldout, result["parameters"]) ** 2))
        ),
        "junction_gaps_m": result["junction_gaps_m"],
        "terminal_height_m": float(dense[-1]["end_xyz"][2]),
        "optimizer_success": result["optimizer_success"],
        "optimizer_evidence": result.get("optimizer_evidence"),
        "regime_recovery_evidence": result.get("regime_recovery_evidence"),
        "rebound_prior_evidence": result.get("rebound_prior_evidence"),
        "nested_rebound_evidence": result.get("nested_rebound_evidence"),
        "objective_calls_complete": result.get("objective_calls_complete", True),
        "objective_calls": result["objective_calls"],
        "training_pixel_rms": result["final_pixel_rms"],
        "parameters": result["parameters"].tolist(),
        "rebound_scale_factors": result["parameters"][-2:].tolist()
        if scene.rebound_mode == "point_scales"
        else [1.0, 1.0],
        "complete_point_accepted": False,
        "qualification_note": "No net/reach/event/start/end acceptance certificate; spatial diagnostic only",
    }


def configured_scene(scene: model.Scene, configuration: dict) -> model.Scene:
    """Use the same explicit dynamics configuration for fitting and frozen replay."""
    result = replace(
        scene,
        dynamics=configuration.get("dynamics", "legacy_cross"),
        bounce_profile=configuration.get("bounce_profile", "nominal"),
        rebound_mode=configuration.get("rebound_mode", "fixed"),
    )
    result.validate()
    return result


def in_scope_bounce_evidence(point: dict, scene: model.Scene) -> tuple[tuple, list]:
    groups, excluded = [], []
    for i, flight in enumerate(point["flights"]):
        frames = np.array([float(b["frame"]) for b in flight.get("bounces", [])])
        if not np.isfinite(frames).all():
            raise ValueError("nonfinite source bounce evidence")
        inside = (frames > scene.contact_frames[i]) & (frames <= scene.contact_frames[i + 1])
        groups.append(frames[inside])
        excluded.extend(
            {"flight_index": i, "frame": float(f), "reason": "outside_modeled_flight"}
            for f in frames[~inside]
        )
    return tuple(groups), excluded


def conditioning_bounces(point: dict, scene: model.Scene, policy: str) -> tuple[dict, list]:
    """Perturb supplied interior event times only; never pictures or scoring truth."""
    if policy not in {"exact", "nearest_native_interior"}:
        raise ValueError("unsupported bounce-time conditioning")
    flights, receipt = [], []
    for i, flight in enumerate(point["flights"]):
        bounces = []
        start, end = scene.contact_frames[i : i + 2]
        for bounce in flight.get("bounces", []):
            original = float(bounce["frame"])
            supplied = original
            reason = "exact_policy_or_boundary_or_outside_flight"
            if not np.isfinite(original):
                raise ValueError("nonfinite source bounce evidence")
            if policy == "nearest_native_interior" and start < original < end:
                # Half-up is explicit; neither a new exposure nor a timebase repair.
                supplied = float(np.floor(original + 0.5))
                reason = "interior_time_rounded_half_up"
                if not start < supplied < end:
                    raise ValueError("rounded interior bounce crosses a contact/ending boundary")
            bounces.append({**bounce, "frame": supplied})
            receipt.append(
                {
                    "flight_index": i,
                    "original_frame": original,
                    "supplied_frame": supplied,
                    "delta_frames": supplied - original,
                    "reason": reason,
                    "training_exposure_available": bool(
                        np.any(scene.observation_frames[i] == supplied)
                    ),
                }
            )
        if any(a["frame"] >= b["frame"] for a, b in zip(bounces, bounces[1:])):
            raise ValueError("supplied bounce times collide or reverse")
        flights.append({**flight, "bounces": bounces})
    return {**point, "flights": flights}, receipt


def run_point(
    point: dict,
    cameras_root: str,
    arm: str,
    max_nfev: int,
    timeout: int,
    dynamics: str = "legacy_cross",
    initialization: str = "truth_perturbed",
    bounce_profile: str = "nominal",
    rebound_mode: str = "fixed",
    fit_bounce_windows: bool = False,
    include_bounce_exposure: bool = False,
    bounce_time_conditioning: str = "exact",
    training_visibility: str = "all",
    bounce_uncertainty_frames: float = 1.0,
    recover_bounce_regimes: bool = False,
    bounce_regime_strategy: str = "perturb",
    rebound_prior_scale: float | None = None,
    warm_start_rebound: bool = False,
    retain_direct_rebound: bool = False,
    noise_seed: int | None = None,
) -> dict:
    started = time.monotonic()
    record = {
        "point": point["point"],
        "broadcast": point["match_id"],
        "arm": arm,
        "expected_flights": len(point["flights"]),
        "status": "failed",
        "bounce_time_conditioning": bounce_time_conditioning,
        "bounce_uncertainty_frames": bounce_uncertainty_frames,
    }
    if noise_seed is not None:
        record["noise_seed"] = noise_seed

    def timed_out(*_):
        raise TimeoutError("point fitting time limit")

    prior = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(timeout)
    try:
        camera_path = Path(cameras_root) / point["match_id"] / "camera_P_per_frame_v1.npz"
        scene, seed, xyz, paths, heldout = prepare(point, camera_path)
        configuration = {
            "dynamics": dynamics,
            "bounce_profile": bounce_profile,
            "rebound_mode": rebound_mode,
        }
        scene, heldout = (
            configured_scene(scene, configuration),
            configured_scene(heldout, configuration),
        )
        scene, masks, record["training_visibility_evidence"] = apply_training_visibility(
            scene, point, training_visibility
        )
        xyz = tuple(x[m] for x, m in zip(xyz, masks, strict=True))
        supplied_point, record["bounce_time_conditioning_evidence"] = conditioning_bounces(
            point, scene, bounce_time_conditioning
        )
        if arm == "pixels_noise2":
            scene = noisy_pixels(scene, point["point"], noise_seed)
        if initialization in {
            "image_ballistic",
            "image_ballistic_ground",
            "image_physics_backward",
            "image_physics_consensus",
            "image_physics_postbounce_fallback",
        }:
            from cv.experiments.connected_shooting.initialization import (
                image_ballistic_seed,
                physics_backward_seed,
            )

            bounce_frames = [
                float(f["bounces"][0]["frame"]) if f.get("bounces") else None
                for f in supplied_point["flights"]
            ]
            if initialization == "image_physics_postbounce_fallback":
                from cv.experiments.connected_shooting import postbounce_initialization

                seed, seed_record = postbounce_initialization.seed(scene, bounce_frames)
            elif initialization in {"image_physics_backward", "image_physics_consensus"}:
                seed, seed_record = physics_backward_seed(
                    scene,
                    bounce_frames,
                    consensus=initialization == "image_physics_consensus",
                    include_bounce_exposure=include_bounce_exposure,
                )
            else:
                seed, seed_record = image_ballistic_seed(
                    scene, bounce_frames, ground_anchor=initialization == "image_ballistic_ground"
                )
            record["initialization_evidence"] = seed_record
        elif initialization != "truth_perturbed":
            raise ValueError("unsupported initialization")
        record["initialization"] = initialization
        if rebound_mode == "point_scales":
            seed = np.r_[seed, [1.0, 1.0]]
        record["applied_initial_parameters"] = seed.tolist()
        bounce_evidence = None
        if fit_bounce_windows:
            bounce_evidence, record["excluded_bounce_evidence"] = in_scope_bounce_evidence(
                supplied_point, scene
            )
        result = model.fit(
            scene,
            seed,
            max_nfev=max_nfev,
            optimize_spin=True,
            oracle_positions=xyz if arm == "xyz" else None,
            bounce_frames=bounce_evidence,
            bounce_uncertainty_frames=bounce_uncertainty_frames,
            recover_bounce_regimes=recover_bounce_regimes,
            bounce_regime_strategy=bounce_regime_strategy,
            rebound_prior_scale=rebound_prior_scale,
            warm_start_rebound=warm_start_rebound,
            retain_direct_rebound=retain_direct_rebound,
        )
        record["bounce_window_evidence"] = result["bounce_window_evidence"]
        record.update(measure(scene, result, paths, heldout))
        record["status"] = "measured"
    except (ValueError, KeyError, FileNotFoundError, TimeoutError, FloatingPointError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["complete_point_accepted"] = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)
    record["wall_seconds"] = time.monotonic() - started
    return record


def apply_training_visibility(scene: model.Scene, point: dict, policy: str) -> tuple:
    """Mask declared training evidence only; held-out pictures/dense truth stay untouched."""
    if policy not in {"all", "first_flight_postbounce_only"}:
        raise ValueError("explicit supported training visibility required")
    masks = [np.ones(len(f), dtype=bool) for f in scene.observation_frames]
    if policy != "all":
        bounces = point["flights"][0].get("bounces", [])
        if not bounces:
            raise ValueError("first bounce required for visibility control")
        bounce = float(bounces[0]["frame"])
        if (
            not np.isfinite(bounce)
            or not scene.contact_frames[0] < bounce <= scene.contact_frames[1]
        ):
            raise ValueError("in-scope first bounce required for visibility control")
        masks[0] = scene.observation_frames[0] > bounce
        if np.count_nonzero(masks[0]) < 4:
            raise ValueError("fewer than four post-bounce training exposures; point remains held")
    evidence = {
        "policy": policy,
        "removed_training_frames": [
            f[~m].tolist() for f, m in zip(scene.observation_frames, masks, strict=True)
        ],
        "heldout_images_changed": False,
        "scoring_truth_changed": False,
    }
    if policy == "all":
        return scene, tuple(masks), evidence
    result = replace(
        scene,
        observation_frames=tuple(
            f[m] for f, m in zip(scene.observation_frames, masks, strict=True)
        ),
        cameras=tuple(p[m] for p, m in zip(scene.cameras, masks, strict=True)),
        pixels=tuple(p[m] for p, m in zip(scene.pixels, masks, strict=True)),
        camera_distortion=None
        if scene.camera_distortion is None
        else tuple(d[m] for d, m in zip(scene.camera_distortion, masks, strict=True)),
    )
    result.validate()
    return result, tuple(masks), evidence


def summarize(rows: list[dict]) -> dict:
    return {
        "points": len(rows),
        "expected_flights": sum(r["expected_flights"] for r in rows),
        "measured_points": sum(r["status"] == "measured" for r in rows),
        "trajectory_and_endpoint_tolerances_met": sum(
            r.get("trajectory_and_endpoint_tolerances_met", False) for r in rows
        ),
        "complete_points_accepted": 0,
        "optimizer_successes": sum(r.get("optimizer_success", False) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--cameras-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory; refuses overwrite"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["xyz", "pixels", "pixels_noise2"],
        default=["xyz", "pixels", "pixels_noise2"],
    )
    parser.add_argument(
        "--limit", type=int, help="explicit development prefix, not a selected success subset"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--point-timeout", type=int, default=300)
    parser.add_argument(
        "--noise-seed",
        type=int,
        help="explicit nonnegative 32-bit training-noise replicate; omit to preserve the original point-specific draw",
    )
    parser.add_argument(
        "--dynamics", choices=["legacy_cross", "measured_240hz"], default="legacy_cross"
    )
    parser.add_argument("--bounce-profile", choices=list(BOUNCE_PROFILES), default="nominal")
    parser.add_argument("--rebound-mode", choices=["fixed", "point_scales"], default="fixed")
    parser.add_argument(
        "--warm-start-rebound",
        action="store_true",
        help="solve fixed coefficients first, then extend and retain the lower-cost candidate; research only",
    )
    parser.add_argument(
        "--retain-direct-rebound",
        action="store_true",
        help="retain direct and warm-start corrections after fixed fit under the shared point timeout; research only",
    )
    parser.add_argument(
        "--rebound-prior-scale",
        type=float,
        help="positive quadratic regularization scale around unit point corrections; not calibrated uncertainty",
    )
    parser.add_argument("--fit-bounce-windows", action="store_true")
    parser.add_argument(
        "--recover-bounce-regimes",
        action="store_true",
        help="at most two objective-selected launch-spin retries at a slide/grip boundary; research only",
    )
    parser.add_argument(
        "--bounce-regime-strategy",
        choices=["perturb", "fixed_branch"],
        default="perturb",
        help="fixed_branch enumerates formula extensions but only selects exact unforced physical replays",
    )
    parser.add_argument(
        "--bounce-uncertainty-frames",
        type=float,
        default=1.0,
        help="explicit objective dead-zone half-width in [0, 1]; zero uses supplied timing precision, not exact enforcement",
    )
    parser.add_argument("--include-bounce-exposure", action="store_true")
    parser.add_argument(
        "--training-visibility",
        choices=["all", "first_flight_postbounce_only"],
        default="all",
        help="synthetic missing incoming-arc control; held-out exposures and dense truth unchanged",
    )
    parser.add_argument(
        "--bounce-time-conditioning",
        choices=["exact", "nearest_native_interior"],
        default="exact",
        help="explicit synthetic event-time uncertainty; original pictures and scoring truth unchanged",
    )
    parser.add_argument(
        "--initialization",
        choices=[
            "truth_perturbed",
            "image_ballistic",
            "image_ballistic_ground",
            "image_physics_backward",
            "image_physics_consensus",
            "image_physics_postbounce_fallback",
        ],
        default="truth_perturbed",
    )
    args = parser.parse_args()
    if args.noise_seed is not None and (
        not 0 <= args.noise_seed < 2**32 or "pixels_noise2" not in args.arms
    ):
        parser.error("noise seed requires the noisy arm and an integer in [0, 2**32)")
    if min(args.jobs, args.max_nfev, args.point_timeout, args.limit or 1) <= 0:
        parser.error("positive limits required")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms")
    if (
        not np.isfinite(args.bounce_uncertainty_frames)
        or not 0 <= args.bounce_uncertainty_frames <= 1
    ):
        parser.error("bounce uncertainty must be finite and within [0, 1] frames")
    if args.bounce_uncertainty_frames != 1.0 and not args.fit_bounce_windows:
        parser.error("nondefault bounce uncertainty requires explicit bounce windows")
    if args.bounce_profile != "nominal" and args.dynamics != "measured_240hz":
        parser.error("bounce perturbations require measured dynamics")
    if args.rebound_mode == "point_scales" and args.dynamics != "measured_240hz":
        parser.error("point rebound corrections require measured dynamics")
    if args.warm_start_rebound and args.rebound_mode != "point_scales":
        parser.error("rebound warm start requires point-scale mode")
    if args.retain_direct_rebound and not args.warm_start_rebound:
        parser.error("direct rebound retention requires warm-start search")
    if args.rebound_prior_scale is not None and (
        args.rebound_mode != "point_scales"
        or not np.isfinite(args.rebound_prior_scale)
        or args.rebound_prior_scale <= 0
    ):
        parser.error("rebound prior requires point scales and a finite positive scale")
    if args.recover_bounce_regimes and args.dynamics != "measured_240hz":
        parser.error("bounce regime recovery requires measured dynamics")
    if args.bounce_regime_strategy != "perturb" and not args.recover_bounce_regimes:
        parser.error("explicit regime strategy requires recovery flag")
    if (
        args.initialization == "image_physics_postbounce_fallback"
        and args.dynamics != "measured_240hz"
    ):
        parser.error("post-bounce fallback requires measured dynamics")
    if args.include_bounce_exposure and args.initialization not in {
        "image_physics_backward",
        "image_physics_consensus",
    }:
        parser.error("bounce exposure requires backward initialization")
    if args.bounce_time_conditioning != "exact" and (
        not args.fit_bounce_windows
        or args.initialization not in {"image_physics_backward", "image_physics_consensus"}
    ):
        parser.error("rounded bounce control requires explicit windows and backward initialization")
    truth_record = file_record(args.truth)
    truth = json.loads(args.truth.read_text())
    points = truth["points"][: args.limit]
    if not points or len({p["point"] for p in points}) != len(points):
        raise ValueError("nonempty unique fixed point inventory required")
    sources = [
        args.truth,
        Path(s6_point_bench.__file__),
        Path(owner_spatial_audit.__file__),
        Path(__file__),
        Path(model.__file__),
        Path(__file__).with_name("camera_geometry.py"),
        Path(event_constraints.__file__),
        Path(model.rich_ball_physics.__file__),
        Path(model.rich_ball_physics.pe.__file__),
        Path(model.rich_ball_physics.flight.__file__),
        Path(model.rich_ball_physics.impact.__file__),
    ]
    sources.extend(
        args.cameras_root / m / "camera_P_per_frame_v1.npz"
        for m in sorted({p["match_id"] for p in points})
    )
    from cv.experiments.connected_shooting import measured_dynamics
    from cv.experiments.connected_shooting import initialization

    sources.extend(
        [Path(measured_dynamics.__file__), Path(measured_dynamics.bounce_reference.__file__)]
    )
    sources.append(Path(initialization.__file__))
    if args.recover_bounce_regimes:
        sources.append(Path(__file__).with_name("regime_recovery.py"))
    if args.warm_start_rebound:
        sources.append(Path(__file__).with_name("nested_rebound.py"))
        if not args.recover_bounce_regimes:
            sources.append(Path(__file__).with_name("regime_recovery.py"))
    if args.initialization == "image_physics_postbounce_fallback":
        sources.append(Path(__file__).with_name("postbounce_initialization.py"))
    records = [file_record(p) for p in sources]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "s6_backward_oracle_connected_v1",
        "status": "running",
        "code": git_record(Path(__file__).resolve().parents[3]),
        "inputs": records,
        "input_bindings": [
            {"resolved_path": str(p.resolve()), "record": r}
            for p, r in zip(sources, records, strict=True)
        ],
        "truth": truth_record,
        "point_inventory": [p["point"] for p in points],
        "configuration": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "noise_seed" or v is not None
        },
        "conditioning": {
            "all_arms": "exact contact/end times; known synthetic cameras; fitted spin",
            "bounce_times": args.bounce_time_conditioning,
            "bounce_windows_in_objective": {
                "uncertainty_frames": args.bounce_uncertainty_frames,
                "time_scale_frames": event_constraints.TIME_SCALE_FRAMES,
                "missing_ground_scale_m": event_constraints.MISSING_GROUND_SCALE_M,
                "loss": "quadratic evidence; soft-L1 image or XYZ observations",
                "scope": "only bounces within modeled flights; future/dead-ball bounces excluded explicitly",
            }
            if args.fit_bounce_windows
            else None,
            "bounce_model": {
                "profile": args.bounce_profile,
                "rebound_mode": args.rebound_mode,
                "fitted_point_corrections": {
                    "order": ["restitution", "horizontal_retention"],
                    "bounds": [0.8, 1.2],
                    "initial": [1.0, 1.0],
                    "prior_residual": None
                    if args.rebound_prior_scale is None
                    else {
                        "formula": "(point_scales - 1) / scale",
                        "scale": args.rebound_prior_scale,
                        "loss": "quadratic_not_robustified",
                        "scope": "experimental regularization, not calibrated uncertainty",
                    },
                }
                if args.rebound_mode == "point_scales"
                else None,
                "restitution_multiplier": BOUNCE_PROFILES[args.bounce_profile][0],
                "horizontal_retention_multiplier": BOUNCE_PROFILES[args.bounce_profile][1],
                "coefficient_clip": [0.05, 1.0],
                "spin_and_dwell": "unchanged nominal law; not a coupled impact uncertainty model",
                "truth_and_observations": "unchanged across profiles",
            },
            "initialization": "truth-derived perturbed XYZ/velocity"
            if args.initialization == "truth_perturbed"
            else "native training pixels/cameras and declared supplied first-bounce times; no XYZ/velocity truth",
            "xyz": "3D truth at visible native training exposures; robust residual units 0.01 m",
            "pixels": "exact known-camera projections at visible native training exposures",
            "pixels_noise2": "same with deterministic independent Gaussian 2 native px standard deviation",
            "heldout": "every native frame divisible by three excluded from all objectives",
            "scoring": "dense recorded 240 Hz interpolant including hidden arcs; never fit target for image arms",
        },
        "scope": "opened synthetic development, oracle-conditioned; no real-video or complete-point yield claim",
        "working_spatial_contract": CONTRACT,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                run_point,
                p,
                str(args.cameras_root),
                arm,
                args.max_nfev,
                args.point_timeout,
                args.dynamics,
                args.initialization,
                args.bounce_profile,
                args.rebound_mode,
                args.fit_bounce_windows,
                args.include_bounce_exposure,
                args.bounce_time_conditioning,
                args.training_visibility,
                args.bounce_uncertainty_frames,
                args.recover_bounce_regimes,
                args.bounce_regime_strategy,
                args.rebound_prior_scale,
                args.warm_start_rebound,
                args.retain_direct_rebound,
                args.noise_seed,
            )
            for p in points
            for arm in args.arms
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with (args.output / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in [
                            "point",
                            "arm",
                            "status",
                            "trajectory_and_endpoint_tolerances_met",
                            "wall_seconds",
                            "error",
                        ]
                    }
                ),
                flush=True,
            )
    if records != [file_record(p) for p in sources]:
        raise ValueError("inputs changed during oracle evaluation")
    manifest.update(
        status="complete",
        wall_seconds=time.monotonic() - started,
        summaries={arm: summarize([r for r in rows if r["arm"] == arm]) for arm in args.arms},
        by_broadcast={
            m: {
                arm: summarize([r for r in rows if r["arm"] == arm and r["broadcast"] == m])
                for arm in args.arms
            }
            for m in sorted({p["match_id"] for p in points})
        },
    )
    (args.output / "report.json").write_text(
        json.dumps({**manifest, "results": rows}, indent=2) + "\n"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summaries"], indent=2))


if __name__ == "__main__":
    main()
