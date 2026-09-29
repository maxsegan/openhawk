"""Opened matched joint toss fit with an optional qualified cross-attempt serve prior.

Fit original native toss and outgoing images at one shared contact, profile only
inside its original interval, and retain original flight physics and scoring.
No shared default, new labels, inferred XYZ targets or material-law edits.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import (
    athlete_priors,
    full_native_continuation as full,
    joint_toss_residual as toss,
    labeled_context_census as census,
    labeled_context_witness_scope as scope,
    net_constraints,
    labeled_player_serve_prior as player_prior,
)
from cv.pipeline import provenance


def future_ground_guidance(flight, end, fps):
    """Continuous horizon proxy, not a claim that a modeled impact exists."""
    height = max(float(flight["end_xyz"][2]) - 0.0325, 0.0)
    vz = float(flight["velocities"][-1, 2])
    seconds = (vz + np.sqrt(vz * vz + 2 * 9.81 * height)) / 9.81
    return float(end + fps * seconds)


def fit(
    context,
    source,
    observations,
    feet,
    release_interval,
    duration,
    *,
    epoch,
    maxiter=80,
    seconds=300,
    initializer=None,
    contact_prior=None,
    prior_weight=0.0,
):
    began = time.monotonic()
    ctx = full.profile.profile_context(context, epoch)
    scene, axes, activated = full.merge_scene(
        ctx["scene"], ctx["heldout"], ctx["bounces"], ctx["axes"]
    )
    n = len(scene.pixels)
    indices = np.r_[np.arange(3), np.arange(3, 12), np.arange(3 + 3 * n, 12 + 3 * n)]
    scale = np.r_[[1.0] * 3, [30.0] * 9, [3.0] * 9]
    reference = full.model.chain(scene, source)
    obs = deepcopy(observations)
    low_epoch = next(e for e in ctx["events"] if e["event_type"] == "contact")["frame_interval"][0]
    obs["rows"] = [r for r in obs["rows"] if r["frame"] < low_epoch]
    obs["contact_frame"] = float(epoch)
    raw = source.copy()
    incoming = np.array([0.0, 0.0, -2.0])
    if initializer is not None:
        raw[:3] = initializer["toss"]["shared_contact_xyz_m"]
        incoming = np.asarray(initializer["toss"]["incoming_contact_velocity_mps"])
        # Preserve a useful rough first endpoint when changing contact depth.
        raw[3:6] += (source[:3] - raw[:3]) / ((scene.contact_frames[1] - epoch) / scene.fps)
    q0 = np.r_[raw[indices] / scale, incoming, np.mean(release_interval)]
    depth_low, depth_high = map(float, ctx["depth_bounds"])
    lo = np.r_[[0.0, depth_low, 2.0], [-2.5] * 9, [-2.0] * 9, [-12.0] * 3, release_interval[0]]
    hi = np.r_[[10.97, depth_high, 4.0], [2.5] * 9, [2.0] * 9, [12.0] * 3, release_interval[1]]
    q0 = np.clip(q0, lo + 1e-8, hi - 1e-8)
    target = np.concatenate(scene.pixels)
    queries = net_constraints.dense_queries(scene, ctx["bounces"])
    expected = [
        [e for e in ctx["events"] if e["event_type"] == "bounce" and a < e["frame"] <= b]
        for a, b in zip(scene.contact_frames[:-1], scene.contact_frames[1:], strict=True)
    ]
    best = None
    calls = invalid = 0

    def evaluate(q):
        p = source.copy()
        p[indices] = q[:21] * scale
        flights = full.model.chain(scene, p, query_frames=queries)
        image = (
            full.exposure.prediction(
                scene, p, axes, duration, termination_kind=ctx["termination_kind"]
            )
            - target
        )
        tau = (epoch - q[-1]) / scene.fps
        tr, ts, receipt = toss.evaluate(p[:3], np.r_[q[21:24], tau], obs, feet, scene.fps)
        ground = []
        net = []
        for i, (flight, events) in enumerate(zip(flights, expected, strict=True)):
            a, b = scene.contact_frames[i : i + 2]
            for j, event in enumerate(events):
                actual = (
                    flight["bounces"][j]["frame"]
                    if j < len(flight["bounces"])
                    else future_ground_guidance(flight, b, scene.fps)
                )
                low, high = event["frame_interval"]
                ground.append((min(actual - low, 0) + max(actual - high, 0)) / 0.35)
            ground.append(
                max(b - flight["bounces"][len(events)]["frame"], 0)
                if len(flight["bounces"]) > len(events)
                else 0.0
            )
            declared = scene.net_hit_frames is not None and len(scene.net_hit_frames[i])
            net.append(
                0.0
                if declared
                else net_constraints.penalty(queries[i], flight["positions"], 0.025)[0]
            )
        contacts = np.asarray([f["start_xyz"] for f in flights])
        athlete = athlete_priors.optimization_residuals(contacts, ctx["players"])
        stay = np.concatenate(
            [(flights[i]["end_xyz"] - reference[i]["end_xyz"]) / 0.35 for i in range(3)]
        )
        # Every toss/front row retains its declared uncertainty. The separate
        # image block is pixel-scaled; 4 gives toss rows comparable pixel weight.
        residual = np.r_[
            image.ravel(),
            4 * tr,
            8 * np.minimum(ts, 0),
            ground,
            net,
            athlete * 4,
            (
                []
                if contact_prior is None
                else player_prior.optimization_residuals(
                    contact_prior, p[:3], feet["side"], feet["court_xy_m"], prior_weight
                )
            ),
            stay,
            (p[indices[12:]] - source[indices[12:]]) / 6,
        ]
        return residual, p, receipt, flights, image

    length = len(evaluate(q0)[0])

    def residual(q):
        nonlocal best, calls, invalid
        if time.monotonic() - began > seconds:
            raise TimeoutError("joint serve block wall budget")
        calls += 1
        try:
            r, p, tr, flights, image = evaluate(q)
        except ValueError:
            invalid += 1
            return np.full(length, 1e4)
        cost = float(r @ r)
        if best is None or cost < best["cost"]:
            best = dict(
                cost=cost,
                parameters=p,
                shared_toss=tr,
                release_epoch=float(q[-1]),
                rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
                calls=calls,
            )
        return r

    def jac(q):
        columns = []
        for j in range(len(q)):
            left, right = q.copy(), q.copy()
            left[j] = max(lo[j], q[j] - 1e-5)
            right[j] = min(hi[j], q[j] + 1e-5)
            columns.append((residual(right) - residual(left)) / (right[j] - left[j]))
        return np.column_stack(columns)

    try:
        solved = least_squares(
            residual,
            q0,
            jac=jac,
            bounds=(lo, hi),
            max_nfev=maxiter,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        termination = dict(
            success=bool(solved.success), message=solved.message, nfev=solved.nfev, njev=solved.njev
        )
    except TimeoutError as error:
        termination = dict(success=False, message=str(error))
    if best is None:
        raise ValueError("no physical joint serve candidate")
    return ctx, dict(
        best=best,
        termination=termination,
        seconds=time.monotonic() - began,
        calls=calls,
        invalid_calls=invalid,
        contact_epoch=epoch,
        activated_frames=activated,
        fitting_scene=asdict(scene),
        policy=dict(
            changed_launch_blocks=[0, 1, 2],
            depth_bounds_m=[depth_low, depth_high],
            contact_prior=contact_prior,
            prior_weight=prior_weight,
            original_ground_law=True,
            shared_contact_exact=True,
            prior_withheld_consumed=True,
            release_interval_frames=release_interval,
            original_toss_checks=True,
            original_exposure_timestamps=True,
            source_end_soft_sigma_m=0.35,
            ground_missing_proxy="vertical-gravity future horizon guidance; never scored as impact",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--toss-cameras", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--prior-records", type=Path, required=True)
    parser.add_argument("--player-id", required=True)
    parser.add_argument("--service-side", choices=("deuce", "ad"), required=True)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    args = parser.parse_args()
    prior_doc = json.loads(args.prior_records.read_text())
    contact_prior = player_prior.fit_prior(
        prior_doc["qualified_records"], args.player_id, args.service_side, args.key
    )
    args.output.mkdir(parents=True, exist_ok=True)
    row = next(r for r in json.loads(args.inventory.read_text())["rows"] if r["key"] == args.key)
    ext = next(r for r in json.loads(args.toss_cameras.read_text())["rows"] if r["key"] == args.key)
    rung = "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win32_shift1f"
    c, p, _, _, _, t, d, loaded = full.load_case(dict(item=row["item"], rung=rung))
    before, _ = census.score(
        c, p, t, d, row["item"]["metadata"]["ending_kind"], scorer=scope.LOCAL_SCORE
    )
    initializer = None
    if args.preflight:
        pre = next(
            r for r in json.loads(args.preflight.read_text())["rows"] if r["key"] == args.key
        )
        initializer = min(
            (
                r
                for r in pre["trials"]
                if r["mode"] == "free_contact_diagnostic" and r["contact_epoch"] == args.epoch
            ),
            key=lambda r: r["cost"],
        )
    sources = [
        *loaded["sources"],
        *[
            provenance.file_record(Path(x))
            for x in [
                __file__,
                args.inventory,
                args.toss_cameras,
                toss.__file__,
                args.prior_records,
                player_prior.__file__,
            ]
        ],
    ]
    if args.preflight:
        sources.append(provenance.file_record(args.preflight))
    report = dict(
        key=args.key,
        before=before,
        sources=sources,
        automatic_inference_eligible=False,
        opened_development=True,
    )
    (args.output / "start.json").write_text(
        json.dumps(full.profile.jsonable(report), indent=2) + "\n"
    )
    ctx, result = fit(
        c,
        p,
        ext["toss_observations"],
        ext["server_anchor"],
        row["release"]["frame_interval"],
        d,
        epoch=args.epoch,
        maxiter=args.maxiter,
        seconds=args.seconds,
        initializer=initializer,
        contact_prior=contact_prior,
        prior_weight=args.prior_weight,
    )
    after, measurement = census.score(
        ctx,
        result["best"]["parameters"],
        t,
        d,
        row["item"]["metadata"]["ending_kind"],
        scorer=scope.LOCAL_SCORE,
    )
    report.update(
        fit=result,
        after=after,
        measurement=measurement,
        actual_contact_epochs=ctx["scene"].contact_frames,
        evaluation_scene=asdict(ctx["scene"]),
        evaluation_heldout=asdict(ctx["heldout"]),
        events=ctx["events"],
    )
    (args.output / "report.json").write_text(
        json.dumps(full.profile.jsonable(report), indent=2) + "\n"
    )
    print(
        args.key,
        args.epoch,
        result["termination"],
        [(f["flight_index"], f["failures"]) for f in after["flights"]],
        result["best"]["shared_toss"]["image_rms_px"],
        flush=True,
    )


if __name__ == "__main__":
    main()
