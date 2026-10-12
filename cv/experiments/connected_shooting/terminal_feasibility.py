"""Constrained terminal-coverage refinement of a frozen owner-conditioned fit.

Question: can the same image objective meet the existing no-discard ending
contract when coverage is a hard constraint? Research only, never acceptance.
No observations, physical constants, event times or acceptance checks change.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import html
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import minimize

from cv.experiments.connected_shooting import (
    model,
    event_constraints,
    physical_compatibility,
    camera_geometry,
    exposure_support,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import paths, provenance
from cv.validation import s6_owner_ground_camera as ground, s6_sparse_owner_replay as replay

NUMERICAL_INTERIOR_FRAMES = 1e-5
#: Explicit not-applicable slack for an original-contact right boundary. Every
#: caller subtracts the numerical interior and requires the vector to be
#: non-negative; a fixed unit vector keeps those callers feasible without ever
#: probing the simulator for a last bounce.
ORIGINAL_CONTACT_SLACK_FRAMES = (1.0, 1.0, 1.0)


def resolve_terminal_guidance(configuration: dict, legacy_choice: str | None) -> tuple[bool, str]:
    """Require an explicit choice for old reports without this objective metadata."""
    if "terminal_observation_constraint" in configuration:
        value = configuration["terminal_observation_constraint"]
        if not isinstance(value, bool) or legacy_choice is not None:
            raise ValueError("Recorded boolean terminal guidance cannot be overridden")
        return value, "baseline_configuration"
    if legacy_choice not in {"enabled", "disabled"}:
        raise ValueError("Legacy baseline requires explicit --legacy-terminal-guidance")
    return legacy_choice == "enabled", "explicit_legacy_cli_choice_objective_reproduction_required"


def replay_dependency(record: dict, current_code: bool) -> Path:
    """Separate explicitly replayed Python producers from immutable data bindings."""
    if current_code and record["path_base"] == "repository" and record["path"].endswith(".py"):
        path = (paths.REPO_ROOT / record["path"]).resolve()
        if not path.is_relative_to(paths.REPO_ROOT.resolve()) or not path.is_file():
            raise ValueError("Repository Python dependency escaped its root or is missing")
        return path
    return ground.resolve_record(record)


def verify_baseline_geometry(scene, parameters, baseline_flights) -> None:
    """Current solver must reproduce every stored baseline sample and endpoint."""
    flights = model.chain(scene, parameters)
    if len(flights) != len(baseline_flights):
        raise ValueError("Baseline flight inventory does not reproduce")
    for actual, expected in zip(flights, baseline_flights, strict=True):
        for key in ("start_frame", "end_frame", "start_xyz", "end_xyz", "positions"):
            a, b = np.asarray(actual[key]), np.asarray(expected[key])
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-9, rtol=1e-9):
                raise ValueError(f"Baseline geometry does not reproduce: {key}")


def objective(
    scene,
    parameters,
    bounces,
    last,
    *,
    cache=None,
    terminal_guidance=True,
    exposure_half_width_frames=0.0,
):
    flights, physical, _ = event_constraints.evaluate(
        scene,
        parameters,
        bounces,
        1,
        simulation_cache=cache,
        **({"terminal_last_observation_frame": last} if terminal_guidance else {}),
    )
    image_residual = (
        exposure_support.evaluate(
            scene, parameters, exposure_half_width_frames, simulation_cache=cache
        )[0]
        if exposure_half_width_frames != 0
        else model.projected_residual(scene, flights)
    )
    residual = np.r_[image_residual, physical, (parameters[-2:] - 1) / 0.02]
    loss = event_constraints.mixed_loss(2 * sum(map(len, scene.observation_frames)))
    return float(2 * np.sum(loss((residual / 2) ** 2)[0]))


def _net_stop_slack(scene, parameters, *, cache=None):
    """The flight ends on the net. A missing hit is infeasible; a cord is not this path.

    Three inequalities, same length as the ground slack: plane error in metres,
    mesh height in metres, and the hit time against the net event in frames.
    Without a discrete hit the endpoint itself has to lie on the mesh. That is
    the ball stopping at the net plane, not a free post-tape velocity.
    """
    from cv.pipeline.physics_knot_solver import NET_Y, net_tape_height

    flights = model.chain(scene, parameters, simulation_cache=cache)
    last = flights[-1]
    hits = last.get("net_hits") or []
    target = None
    ending = getattr(scene, "supported_ending", None)
    if isinstance(ending, dict) and ending.get("net_frame") is not None:
        target = float(ending["net_frame"])
    if hits:
        hit = hits[0]
        xyz = np.asarray(hit["x"], float)
        when = float(hit["frame"])
        tape = float(hit.get("tape_height_m", net_tape_height(float(xyz[0]))))
    else:
        xyz = np.asarray(last["end_xyz"], float)
        when = float(last["end_frame"])
        tape = float(net_tape_height(float(xyz[0]))) if np.isfinite(xyz[0]) else 0.914
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(when):
        return np.full(3, -1e6)
    half_width = 6.4
    centre = 5.485
    width_slack = half_width - abs(float(xyz[0]) - centre)
    height_slack = min(
        float(xyz[2]) - (model.R_BALL - 0.02),
        tape + model.R_BALL + 0.03 - float(xyz[2]),
    )
    plane_slack = 0.05 - abs(float(xyz[1]) - NET_Y)
    time_slack = 1.0 if target is None else 2.0 - abs(when - target)
    return np.array([min(plane_slack, width_slack), height_slack, time_slack], float)


def terminal_slack(scene, parameters, bounces, last, *, cache=None, duration: float | None = 0.25):
    if model.original_contact_boundary(scene):
        # Not applicable: the scene ends at supplied contact k. No last bounce
        # is sought and the endpoint is never pushed toward the ground.
        return np.asarray(ORIGINAL_CONTACT_SLACK_FRAMES, float)
    if scene.terminal_net_tail is not None:
        from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
            terminal_slack as net_slack,
        )

        return net_slack(scene, parameters, last, cache=cache, duration=duration)
    if getattr(scene, "observed_horizon_tail", None) is not None:
        from cv.experiments.connected_shooting.observed_horizon_tail import (
            terminal_slack as horizon_slack,
        )

        return horizon_slack(scene, parameters, last, cache=cache, duration=duration)
    kind = None
    ending = getattr(scene, "supported_ending", None)
    if isinstance(ending, dict):
        kind = ending.get("kind")
    # An ending with no required bounce used to index impacts[-1] and raise
    # IndexError when the probe had no impact. A net stop is constrained to the
    # net plane. A partial span has no ground ending to seek. Any other empty
    # bounce group is a clean refusal, not an index into an empty impact list,
    # so an ordinary flight is not made feasible by the absence of a bounce.
    if kind == "net_stop":
        return _net_stop_slack(scene, parameters, cache=cache)
    if kind in {"camera_cut", "held_camera", "point_end", "dead_ball"} or (
        kind in {"fov_exit", "last_visible_sample"} and ending.get("partial_flight") is True
    ):
        return np.asarray(ORIGINAL_CONTACT_SLACK_FRAMES, float)
    if len(bounces[-1]) == 0:
        raise ValueError("terminal slack has no bounce to index")
    times = scene.contact_frames.copy()
    times[-1] += 1
    impacts = model.chain(replace(scene, contact_frames=times), parameters, simulation_cache=cache)[
        -1
    ]["bounces"]
    if len(impacts) < len(bounces[-1]):
        raise ValueError("required terminal impact not reached within probe")
    impact = impacts[len(bounces[-1]) - 1]
    t, end = impact["frame"], scene.contact_frames[-1]
    return np.array([t + impact["dwell_seconds"] * scene.fps - last, t - (end - 1), end + 1 - t])


def refine(
    scene, initial, bounces, last, *, maxiter=60, terminal_guidance=True, first_contact_y_m=None
):
    scene.validate()
    groups = event_constraints.validate(scene, bounces, 1)
    event_constraints.validate_terminal_observation(scene, groups, last)
    n = len(groups)
    initial = np.array(initial, float, copy=True)
    if (
        scene.rebound_mode != "point_scales"
        or scene.bounce_regime_override is not None
        or initial.shape != (3 + 6 * n + 2,)
        or not np.isfinite(initial).all()
        or maxiter <= 0
    ):
        raise ValueError(
            "finite unforced point-scale/fitted-spin seed and positive iteration limit required"
        )
    scale = np.r_[[10] * 3, [30] * (3 * n), [3] * (3 * n), [1, 1]]
    lower = np.r_[[-10, -15, model.R_BALL], [-75] * (3 * n), [-6] * (3 * n), [0.8, 0.8]]
    upper = np.r_[[21, 40, 12], [75] * (3 * n), [6] * (3 * n), [1.2, 1.2]]
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("seed violates unchanged physical search bounds")
    supplied_initial = initial.copy()
    if first_contact_y_m is not None:
        if (
            isinstance(first_contact_y_m, (bool, np.bool_))
            or not np.isfinite(first_contact_y_m)
            or not lower[1] <= first_contact_y_m <= upper[1]
        ):
            raise ValueError(
                "finite first-contact court-Y hypothesis inside original bounds required"
            )
        lower[1] = upper[1] = initial[1] = first_contact_y_m
    cache = FlightCache()
    initial_slack = terminal_slack(scene, initial, groups, last, cache=cache)
    initial_cost = objective(
        scene, initial, groups, last, cache=cache, terminal_guidance=terminal_guidance
    )
    calls = 0

    def cost(q):
        nonlocal calls
        calls += 1
        try:
            value = objective(
                scene, q * scale, groups, last, cache=cache, terminal_guidance=terminal_guidance
            )
            return value if np.isfinite(value) else 1e12
        except (ValueError, FloatingPointError, OverflowError):
            return 1e12

    def constraints(q):
        try:
            return (
                terminal_slack(scene, q * scale, groups, last, cache=cache)
                - NUMERICAL_INTERIOR_FRAMES
            )
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(3, -1e6)

    start = time.monotonic()
    solved = minimize(
        cost,
        initial / scale,
        method="SLSQP",
        bounds=list(zip(lower / scale, upper / scale)),
        constraints=[dict(type="ineq", fun=constraints)],
        options=dict(maxiter=maxiter, ftol=1e-8),
    )
    parameters = solved.x * scale
    return dict(
        parameters=parameters,
        initial_parameters=initial,
        supplied_initial_parameters=supplied_initial,
        first_contact_y_hypothesis_m=first_contact_y_m,
        depth_hypothesis_is_measurement=False,
        initial_cost=initial_cost,
        final_cost=objective(scene, parameters, groups, last, terminal_guidance=terminal_guidance),
        initial_terminal_slack_frames=initial_slack,
        terminal_slack_frames=terminal_slack(scene, parameters, groups, last),
        numerical_interior_frames=NUMERICAL_INTERIOR_FRAMES,
        optimizer=dict(
            method="SLSQP",
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
    for name in ("packet", "cameras", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--replay-current-code",
        action="store_true",
        help="Explicitly replay historical repository Python producers using current code; "
        "data bytes, original geometry and objective must still reproduce.",
    )
    parser.add_argument(
        "--legacy-terminal-guidance",
        choices=("enabled", "disabled"),
        help="Required only for old baselines missing soft-objective guidance metadata; "
        "must reproduce their frozen objective. Does not disable the hard ending constraint.",
    )
    parser.add_argument(
        "--first-contact-y-m",
        type=float,
        help="explicit fixed court-Y depth hypothesis, not a measured contact or automatic prior",
    )
    args = parser.parse_args()
    if args.output.exists() or min(args.maxiter, args.timeout) <= 0:
        raise ValueError("new output and positive limits required")
    baseline = json.loads(args.baseline.read_text())
    if (
        baseline.get("schema") != "s6_sparse_owner_replay_v1"
        or baseline.get("status") != "fit_measured_not_quality_accepted"
    ):
        raise ValueError("frozen measured owner replay required")
    cfg = baseline["configuration"]
    if (
        cfg["net_clearance_scale_m"] is not None
        or cfg["rebound_prior_scale"] != 0.02
        or cfg["bounce_uncertainty_frames"] != 1
        or baseline["fit"]["contact_reach_evidence"] is not None
    ):
        raise ValueError(
            "this control requires the original image/bounce/point-prior objective only"
        )
    for p in (args.packet, args.cameras):
        if provenance.file_record(p) not in baseline["inputs"]:
            raise ValueError("baseline must bind these exact input bytes")
    modules = [
        model,
        event_constraints,
        physical_compatibility,
        camera_geometry,
        ground,
        replay,
        paths,
        provenance,
    ]
    files = [
        args.baseline,
        args.packet,
        args.cameras,
        Path(__file__),
        *[Path(m.__file__) for m in modules],
    ]
    files += [replay_dependency(r, args.replay_current_code) for r in baseline["inputs"]]
    files = list(dict.fromkeys(p.resolve() for p in files))
    records = [provenance.file_record(p) for p in files]
    packet, cameras = (json.loads(p.read_text()) for p in (args.packet, args.cameras))
    attempts = [a for a in packet["attempts"] if a["attempt_id"] == baseline["attempt_id"]]
    if len(attempts) != 1:
        raise ValueError("one matching complete attempt required")
    # Old artifacts already store the actual fitted epoch even when the optional
    # timing-hypothesis configuration field did not yet exist. Recover from that
    # recorded geometry, not an assumed zero offset or a changed owner label.
    offset = cfg.get(
        "first_contact_offset_frames",
        baseline["fit"]["flights"][0]["start_frame"] - attempts[0]["events"][0]["frame"],
    )
    scene, withheld, bounces, native = replay.prepare(
        attempts[0], cameras, first_contact_offset_frames=offset
    )
    last = float(native[-1][-1])
    initial = np.array(baseline["fit"]["parameters"])
    guide, guide_source = resolve_terminal_guidance(cfg, args.legacy_terminal_guidance)
    initial_cost = objective(scene, initial, bounces, last, terminal_guidance=guide)
    if not np.isclose(
        initial_cost, baseline["fit"]["optimizer_evidence"]["cost"], rtol=1e-9, atol=1e-8
    ):
        raise ValueError("refinement objective does not reproduce the frozen baseline")
    verify_baseline_geometry(scene, initial, baseline["fit"]["flights"])
    args.output.mkdir(parents=True)
    result = dict(
        schema="connected_terminal_feasibility_v1",
        scope=__doc__,
        human_derived=True,
        inputs=records,
        code=provenance.git_record(paths.REPO_ROOT),
        status="held",
        complete_point_accepted=False,
        configuration=dict(
            maxiter=args.maxiter,
            timeout_seconds=args.timeout,
            baseline=cfg,
            numerical_interior_frames=NUMERICAL_INTERIOR_FRAMES,
            first_contact_y_hypothesis_m=args.first_contact_y_m,
            resolved_soft_terminal_guidance=guide,
            terminal_guidance_source=guide_source,
            replay_current_code=args.replay_current_code,
            baseline_objective_and_geometry_reproduced=True,
            resolved_first_contact_offset_frames=offset,
            first_contact_offset_source="baseline_configuration"
            if "first_contact_offset_frames" in cfg
            else "stored_flight_epoch_minus_owner_event",
        ),
    )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")

    def deadline(_signum, _frame):
        raise TimeoutError("whole terminal refinement deadline")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout)
    try:
        fitted = refine(
            scene,
            initial,
            bounces,
            last,
            maxiter=args.maxiter,
            terminal_guidance=guide,
            first_contact_y_m=args.first_contact_y_m,
        )
        signal.alarm(0)
        result.update(status="measured_not_accepted", refinement=fitted)
        parameters = fitted["parameters"]
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
        raise ValueError("refinement inputs changed")
    (args.output / "report.json").write_text(
        json.dumps(result, default=replay.default, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        k: result[k]
        for k in ("status", "error", "refinement", "euclidean_pixel_rms", "physical_compatibility")
        if k in result
    }
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Terminal feasibility — not accepted 3D</h1>'
        + ('<img src="trajectory.png" style="max-width:100%">' if "dense_flights" in result else "")
        + "<pre>"
        + html.escape(json.dumps(summary, default=replay.default, indent=2))
        + "</pre>"
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "error", "euclidean_pixel_rms") if k in result}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
