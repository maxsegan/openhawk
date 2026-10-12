"""Can conditional pose wrists resolve S6 launch depth without losing physical coverage?

Research only. Constrain contact to the union of explicit wrist-centred spheres;
the radius is an unverified diagnostic allowance, not a certified racket/pose bound.
Reuse the frozen image objective and terminal checks. No automatic promotion.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import minimize

from cv.experiments.connected_shooting import (
    camera_geometry,
    event_constraints,
    exposure_support,
    flight_cache,
    model,
    physical_compatibility,
    terminal_feasibility as terminal,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import paths, provenance
from cv.validation import s6_sparse_owner_replay as replay


def wrist_distance(position, wrists):
    position, wrists = np.asarray(position, float), np.asarray(wrists, float)
    if (
        position.shape != (3,)
        or wrists.ndim != 2
        or wrists.shape[1:] != (3,)
        or len(wrists) == 0
        or not np.isfinite(position).all()
        or not np.isfinite(wrists).all()
    ):
        raise ValueError("finite position and nonempty wrist XYZ hypotheses required")
    return float(np.linalg.norm(wrists - position, axis=1).min())


def hypotheses_at_contact(hypotheses: list[dict], frame: float) -> list[dict]:
    """Pair time with each pose condition; interpolate latent XYZ, never pictures.

    Linear interpolation between adjacent native epochs is an explicit diagnostic
    motion assumption. It does not establish metric pose accuracy or racket reach.
    Missing endpoints reject the test instead of silently dropping a condition.
    """
    if isinstance(frame, (bool, np.bool_)) or not np.isfinite(frame):
        raise ValueError("finite contact epoch required")
    low, high = math.floor(frame), math.ceil(frame)
    groups = {}
    for row in hypotheses:
        key = (row["condition_index"], row["arm"], row["hand"])
        f = row["frame"]
        if isinstance(f, (bool, np.bool_)) or not np.isfinite(f) or int(f) != f:
            raise ValueError("native integer wrist epochs required")
        group = groups.setdefault(key, {})
        if f in group:
            raise ValueError("duplicate wrist epoch within a pose condition")
        wrist_distance(np.zeros(3), [row["xyz"]])
        group[f] = row
    if not groups:
        raise ValueError("nonempty coherent wrist conditions required")
    selected = []
    for (condition, arm, hand), group in groups.items():
        if low not in group or high not in group:
            raise ValueError("each wrist condition must bracket the contact epoch")
        a, b = group[low], group[high]
        for field in ("height_m", "ground_ankle_height_m"):
            if a[field] != b[field]:
                raise ValueError("wrist interpolation cannot mix body/ground conditions")
        sources = [low] if low == high else [low, high]
        weights = [1.0] if low == high else [high - frame, frame - low]
        xyz = sum(w * np.asarray(group[f]["xyz"], float) for f, w in zip(sources, weights))
        selected.append(
            dict(
                condition_index=condition,
                arm=arm,
                hand=hand,
                frame=float(frame),
                xyz=xyz.tolist(),
                height_m=a["height_m"],
                ground_ankle_height_m=a["ground_ankle_height_m"],
                source_frames=sources,
                source_weights=weights,
                latent_motion_assumption="native_epoch" if low == high else "linear_xyz",
            )
        )
    return selected


def validate_context(attempt, cameras, context, packet):
    cfg = context["configuration"]
    frame = attempt["events"][0]["frame"]
    if (
        cfg["clip"] != attempt["point_clip"]
        or Path(cfg["frames_root"]).parent.name != attempt["match_id"]
        or cfg["fps"] != attempt["fps"]
        or packet["contact_frame"] != frame
        or packet["pose_metric_accuracy_certified"] is not False
        or packet["automatic_inference_eligible"] is not False
    ):
        raise ValueError("matching diagnostic pose/ball clip, match, cadence and epoch required")
    # The same physical coordinate system and native images must underlie both inputs.
    camera_table = {r["frame"]: r for r in cameras["cameras"]}
    for row in context["cameras"]:
        if row["frame"] not in camera_table:
            continue
        other = camera_table[row["frame"]]
        if (
            row["status"] != "supported"
            or other["status"] != "supported"
            or not np.allclose(row["P"], other["P"], rtol=0, atol=1e-8)
        ):
            raise ValueError("pose and ball cameras differ")
    for f in (frame - 1, frame, frame + 1):
        names = {f"f_{int(f):04d}.jpg", f"{attempt['point_clip']}_{int(f):04d}.jpg"}
        pose_images = [r for r in context["inputs"] if Path(r["path"]).name in names]
        ball_images = [r for r in cameras["inputs"] if Path(r["path"]).name in names]
        identities = {(r["sha256"], r["bytes"]) for r in pose_images + ball_images}
        if not pose_images or not ball_images or len(identities) != 1:
            raise ValueError("identical native contact-context image bytes required")
    for row in packet["hypotheses"]:
        if (
            row["frame"] not in (frame - 1, frame, frame + 1)
            or row["hand"] not in ("left_wrist", "right_wrist")
            or row["arm"] == "archived_mean"
        ):
            raise ValueError("unsupported wrist hypothesis selection")
    wrist_distance(np.zeros(3), [r["xyz"] for r in packet["hypotheses"]])


def refine(
    scene,
    initial,
    bounces,
    last,
    wrists,
    radius,
    *,
    maxiter=200,
    guide=True,
    exposure_half_width_frames=0.0,
):
    scene.validate()
    groups = event_constraints.validate(scene, bounces, 1)
    event_constraints.validate_terminal_observation(scene, groups, last)
    initial = np.asarray(initial, float).copy()
    n = len(groups)
    if (
        scene.rebound_mode != "point_scales"
        or scene.bounce_regime_override is not None
        or initial.shape != (3 + 6 * n + 2,)
        or not np.isfinite(initial).all()
        or isinstance(radius, (bool, np.bool_))
        or not np.isfinite(radius)
        or radius <= 0
        or maxiter <= 0
        or isinstance(exposure_half_width_frames, (bool, np.bool_))
        or not np.isfinite(exposure_half_width_frames)
        or not 0 <= exposure_half_width_frames <= 0.5
    ):
        raise ValueError("finite unforced point-scale seed, positive radius and limits required")
    wrist_distance(initial[:3], wrists)
    # Exactly the terminal-feasibility physical search bounds and parameter scaling.
    scale = np.r_[[10] * 3, [30] * (3 * n), [3] * (3 * n), [1, 1]]
    lower = np.r_[[-10, -15, model.R_BALL], [-75] * (3 * n), [-6] * (3 * n), [0.8, 0.8]]
    upper = np.r_[[21, 40, 12], [75] * (3 * n), [6] * (3 * n), [1.2, 1.2]]
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("seed violates unchanged physical search bounds")
    cache, calls = FlightCache(), 0

    def cost(q):
        nonlocal calls
        calls += 1
        try:
            value = terminal.objective(
                scene,
                q * scale,
                groups,
                last,
                cache=cache,
                terminal_guidance=guide,
                exposure_half_width_frames=exposure_half_width_frames,
            )
            return value if np.isfinite(value) else 1e12
        except (ValueError, FloatingPointError, OverflowError):
            return 1e12

    def constraint(q):
        try:
            p = q * scale
            return np.r_[
                terminal.terminal_slack(scene, p, groups, last, cache=cache)
                - terminal.NUMERICAL_INTERIOR_FRAMES,
                radius - wrist_distance(p[:3], wrists),
            ]
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(4, -1e6)

    start = time.monotonic()
    initial_cost = cost(initial / scale)
    solved = minimize(
        cost,
        initial / scale,
        method="SLSQP",
        bounds=list(zip(lower / scale, upper / scale)),
        constraints=[dict(type="ineq", fun=constraint)],
        options=dict(maxiter=maxiter, ftol=1e-8),
    )
    parameters = solved.x * scale
    return dict(
        parameters=parameters,
        initial_parameters=initial,
        initial_cost=initial_cost,
        final_cost=cost(solved.x),
        wrist_distance_m=wrist_distance(parameters[:3], wrists),
        constraint_slack=constraint(solved.x),
        optimizer=dict(
            success=bool(solved.success),
            status=int(solved.status),
            message=str(solved.message),
            iterations=int(solved.nit),
            objective_calls=calls,
        ),
        wall_seconds=time.monotonic() - start,
        observations_discarded=0,
        acceptance_checks_changed=False,
        complete_point_accepted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "cameras", "baseline", "wrists", "context", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--exposure-half-width-frames",
        type=float,
        default=0.0,
        help="Opt-in symmetric temporal image-support hypothesis in [0,0.5], not measured "
        "shutter timing or an acceptance threshold. Native observations stay unchanged.",
    )
    parser.add_argument(
        "--wrist-time-mode",
        choices=("all_context", "contact_epoch"),
        default="all_context",
        help="Legacy three-frame union, or time-paired coherent wrist hypotheses; "
        "fractional epochs explicitly assume linear latent XYZ motion.",
    )
    parser.add_argument("--replay-current-code", action="store_true")
    parser.add_argument("--legacy-terminal-guidance", choices=("enabled", "disabled"))
    parser.add_argument("--interior-contact-index", type=int, default=1)
    parser.add_argument(
        "--interior-contact-offset-frames",
        type=float,
        default=0.0,
        help="Explicit fixed interior timing hypothesis within +/-1 native frame; "
        "changes flight ownership, never owner labels or train/withheld membership.",
    )
    args = parser.parse_args()
    if args.output.exists() or min(args.maxiter, args.timeout) <= 0:
        raise ValueError("new output and positive limits required")
    input_paths = [args.packet, args.cameras, args.baseline, args.wrists, args.context]
    packet, cameras, baseline, wrists, context = [json.loads(p.read_text()) for p in input_paths]
    if (
        baseline.get("schema") != "s6_sparse_owner_replay_v1"
        or baseline.get("status") != "fit_measured_not_quality_accepted"
        or wrists.get("schema") != "s6_pose_wrist_hypotheses_v1"
        or context.get("schema") != "s6_player_context_v1"
    ):
        raise ValueError("explicit frozen replay, diagnostic wrists and pose context required")
    cfg = baseline["configuration"]
    if (
        cfg["net_clearance_scale_m"] is not None
        or cfg["rebound_prior_scale"] != 0.02
        or cfg["bounce_uncertainty_frames"] != 1
        or baseline["fit"]["contact_reach_evidence"] is not None
    ):
        raise ValueError("unchanged original image/bounce/point-prior objective required")
    if any(provenance.file_record(p) not in baseline["inputs"] for p in input_paths[:2]):
        raise ValueError("baseline must bind exact packet and cameras")
    if provenance.file_record(args.context) not in wrists["inputs"]:
        raise ValueError("wrist hypotheses must bind this pose context")
    attempts = [a for a in packet["attempts"] if a["attempt_id"] == baseline["attempt_id"]]
    if len(attempts) != 1:
        raise ValueError("one matching attempt required")
    validate_context(attempts[0], cameras, context, wrists)
    files = input_paths + [
        Path(__file__),
        Path(terminal.__file__),
        Path(flight_cache.__file__),
        *[
            Path(m.__file__)
            for m in (
                camera_geometry,
                event_constraints,
                exposure_support,
                model,
                physical_compatibility,
                replay,
                paths,
                provenance,
            )
        ],
    ]
    for document in (baseline, wrists, context, packet, cameras):
        files += [
            terminal.replay_dependency(r, args.replay_current_code) for r in document["inputs"]
        ]
    files = list(dict.fromkeys(p.resolve() for p in files))
    records = [provenance.file_record(p) for p in files]
    offset = cfg.get(
        "first_contact_offset_frames",
        baseline["fit"]["flights"][0]["start_frame"] - attempts[0]["events"][0]["frame"],
    )
    scene, withheld, bounces, native = replay.prepare(
        attempts[0], cameras, first_contact_offset_frames=offset
    )
    initial = np.array(baseline["fit"]["parameters"])
    last = float(native[-1][-1])
    guide, guide_source = terminal.resolve_terminal_guidance(cfg, args.legacy_terminal_guidance)
    cost = terminal.objective(scene, initial, bounces, last, terminal_guidance=guide)
    if not np.isclose(cost, baseline["fit"]["optimizer_evidence"]["cost"], rtol=1e-9, atol=1e-8):
        raise ValueError("frozen baseline objective does not reproduce")
    terminal.verify_baseline_geometry(scene, initial, baseline["fit"]["flights"])
    if args.interior_contact_offset_frames != 0:
        scene, withheld, bounces, native = replay.prepare(
            attempts[0],
            cameras,
            first_contact_offset_frames=offset,
            interior_contact_offsets_frames={
                args.interior_contact_index: args.interior_contact_offset_frames
            },
        )
    selected = (
        hypotheses_at_contact(wrists["hypotheses"], scene.contact_frames[0])
        if args.wrist_time_mode == "contact_epoch"
        else wrists["hypotheses"]
    )
    result = dict(
        schema="s6_pose_contact_feasibility_v1",
        scope=__doc__,
        inputs=records,
        code=provenance.git_record(paths.REPO_ROOT),
        attempt_id=baseline["attempt_id"],
        configuration=dict(
            radius_m=args.radius_m,
            maxiter=args.maxiter,
            timeout=args.timeout,
            wrist_time_mode=args.wrist_time_mode,
            resolved_first_contact_offset_frames=offset,
            first_contact_offset_source="baseline_configuration"
            if "first_contact_offset_frames" in cfg
            else "stored_flight_epoch_minus_owner_event",
            resolved_soft_terminal_guidance=guide,
            terminal_guidance_source=guide_source,
            replay_current_code=args.replay_current_code,
            baseline_objective_and_geometry_reproduced=True,
            baseline_reproduced_cost=cost,
            interior_contact_index=args.interior_contact_index,
            interior_contact_offset_frames=args.interior_contact_offset_frames,
            resolved_contact_frames=scene.contact_frames.tolist(),
            exposure_half_width_frames=args.exposure_half_width_frames,
            exposure_support_is_calibrated=False,
        ),
        status="held",
        human_derived=True,
        pose_metric_accuracy_certified=False,
        automatic_inference_eligible=False,
        complete_point_accepted=False,
        hypothesis_count=len(selected),
        selected_wrist_hypotheses=selected,
    )
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")

    def deadline(_signum, _frame):
        raise TimeoutError("whole pose-contact refinement deadline")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout)
    try:
        result["refinement"] = fitted = refine(
            scene,
            initial,
            bounces,
            last,
            np.array([r["xyz"] for r in selected]),
            args.radius_m,
            maxiter=args.maxiter,
            guide=guide,
            exposure_half_width_frames=args.exposure_half_width_frames,
        )
        parameters = fitted["parameters"]
        nearest = int(
            np.argmin(
                np.linalg.norm(np.array([r["xyz"] for r in selected]) - parameters[:3], axis=1)
            )
        )
        result["nearest_wrist_hypothesis"] = selected[nearest]
        result["status"] = "measured_not_accepted"
        result["physical_compatibility"] = physical_compatibility.evaluate(
            scene,
            parameters,
            bounces,
            "terminal_bounce" if len(bounces[-1]) == 1 else "second_bounce",
            native,
        )
        projections = []
        for split, s in (("training", scene), ("withheld", withheld)):
            for i, flight in enumerate(model.chain(s, parameters)):
                predicted = camera_geometry.project(s.cameras[i], flight["positions"])
                for f, xy, q in zip(s.observation_frames[i], s.pixels[i], predicted, strict=True):
                    projections.append(
                        dict(
                            frame=int(f),
                            flight=i,
                            split=split,
                            owner_xy=xy,
                            fitted_xy=q,
                            native_error_px=float(np.linalg.norm(xy - q)),
                        )
                    )
        result["native_projection"] = sorted(projections, key=lambda r: r["frame"])
        result["euclidean_pixel_rms"] = {
            s: float(
                np.sqrt(
                    np.mean([p["native_error_px"] ** 2 for p in projections if p["split"] == s])
                )
            )
            for s in ("training", "withheld")
        }
        if args.exposure_half_width_frames > 0:
            support_rows = []
            for split, s in (("training", scene), ("withheld", withheld)):
                _, rows = exposure_support.evaluate(s, parameters, args.exposure_half_width_frames)
                support_rows.extend({**r, "split": split} for r in rows)
            result["exposure_support"] = sorted(support_rows, key=lambda r: r["frame"])
            result["exposure_support_rms_px"] = {
                split: float(
                    np.sqrt(
                        np.mean(
                            [
                                r["support_error_px"] ** 2
                                for r in support_rows
                                if r["split"] == split
                            ]
                        )
                    )
                )
                for split in ("training", "withheld")
            }
        queries = tuple(
            np.unique(np.r_[np.arange(a, b, scene.fps / 240), b])
            for a, b in zip(scene.contact_frames, scene.contact_frames[1:])
        )
        result["dense_flights"] = [
            {**f, "query_frames": q}
            for f, q in zip(
                model.chain(scene, parameters, query_frames=queries), queries, strict=True
            )
        ]
        replay.plot(result, args.output)
    except (ValueError, TimeoutError, FloatingPointError, OverflowError) as exc:
        result["error"] = str(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if records != [provenance.file_record(p) for p in files]:
        raise ValueError("bound inputs changed")
    (args.output / "report.json").write_text(
        json.dumps(result, default=replay.default, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        k: v
        for k, v in result.items()
        if k
        not in (
            "inputs",
            "native_projection",
            "dense_flights",
            "selected_wrist_hypotheses",
            "exposure_support",
        )
    }
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Conditional pose reach — not accepted 3D</h1>'
        + ('<img src="trajectory.png" style="max-width:100%">' if "dense_flights" in result else "")
        + "<pre>"
        + html.escape(json.dumps(summary, default=replay.default, indent=2))
        + "</pre>"
    )
    print(json.dumps(summary, default=replay.default), flush=True)


if __name__ == "__main__":
    main()
