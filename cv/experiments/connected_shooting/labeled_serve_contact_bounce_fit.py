"""Joint serve fit with latent contact and first-ground epochs inside original intervals.

Bounded, isolated research fitter (API only, no CLI). It adapts the frozen joint
toss/outgoing objective of ``labeled_player_prior_fit.fit`` block for block and
weight for weight, and changes only its coordinates:

* the first outgoing Vz (single-shooting index 5) becomes a bounded first-ground
  epoch inside the original bounce interval; the physical Vz is recovered by the
  existing ``FirstImpactChart`` root, so every export is a single-shooting vector
  whose unchanged ``model.chain`` reproduces that impact time;
* the shared contact epoch may be a bounded coordinate inside the original
  contact interval, applied through the existing ``profile_context`` so native
  exposure frames, cameras and axes never move.

Disclosed choices: both epoch coordinates are centred on their intervals; the
source-end soft stay compares against the fixed source vector replayed under each
evaluated contact epoch; the chart root bracket is seeded by the warm/source Vz.
No gates, labels, XYZ priors or learned player priors are added; scoring stays
external. Every candidate is retained only if its physical objective is finite.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import (
    athlete_priors,
    full_native_continuation as full,
    joint_toss_residual as toss,
    net_constraints,
)
from cv.experiments.connected_shooting.labeled_interval_block_fit import FirstImpactChart
from cv.experiments.connected_shooting.labeled_player_prior_fit import future_ground_guidance

FIRST_VZ = 5
EXPORT_TOLERANCE_FRAMES = 1e-7
BUNDLE_CACHE_ENTRIES = 48


class FirstFlightProjector:
    """Map a latent first-ground epoch to the physical first outgoing Vz.

    The chart runs on a shared-contact view of the same scene whose flight-0
    contact is simply ``p[:3]``; all current velocities/spins and the source
    rebound scalars enter its exact cache key. Only ``p[FIRST_VZ]`` changes.
    """

    def __init__(self, scene, bounce_interval, reference_vz):
        if scene.rebound_mode != "point_scales" or scene.dynamics != "measured_240hz":
            raise ValueError("measured point-scale single-shooting layout required")
        self.n = len(scene.contact_frames) - 1
        self.shared = replace(scene, parameterization="shared_contact_states")
        self.chart = FirstImpactChart(self.shared, 0, bounce_interval)
        self.slices = full.model.shared_parameter_slices(self.shared)
        self.reference_vz = float(reference_vz)

    def project(self, parameters, epoch):
        p = np.asarray(parameters, float)
        n, s = self.n, self.slices
        if p.shape != (5 + 6 * n,) or not np.isfinite(p).all():
            raise ValueError("finite 5+6n single-shooting parameters required")
        shared = np.zeros(s["rebound_scales"].stop)
        shared[:3] = p[:3]
        shared[s["velocities"]] = p[3 : 3 + 3 * n]
        shared[s["spins"]] = p[3 + 3 * n : 3 + 6 * n]
        shared[s["rebound_scales"]] = p[-2:]
        shared[self.chart.vz] = self.reference_vz
        projected, receipt = self.chart.project(shared, epoch)
        out = p.copy()
        out[FIRST_VZ] = projected[self.chart.vz]
        return out, receipt


def _interval(values, name):
    low, high = map(float, values)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f"finite nonzero original {name} interval required")
    return low, high


def _inside(inner, outer):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def fit(
    context,
    source,
    observations,
    feet,
    release_interval,
    duration,
    *,
    contact_interval,
    bounce_interval,
    initial_parameters=None,
    initial_toss=None,
    fixed_contact_epoch=None,
    maxiter=80,
    seconds=300,
):
    """Return ``(context profiled at the best contact epoch, result)``."""
    began = time.monotonic()
    source = np.asarray(source, float)
    fps = float(context["scene"].fps)
    n = len(context["scene"].pixels)
    if n < 3:
        raise ValueError("three original launch blocks required")
    contact_event = next(e for e in context["events"] if e["event_type"] == "contact")
    label = _interval(contact_event["frame_interval"], "contact")
    contact_interval = _interval(contact_interval, "contact")
    bounce_interval = _interval(bounce_interval, "bounce")
    if not _inside(contact_interval, label):
        raise ValueError("contact interval must lie inside the original label interval")
    bounce_event = next(
        (
            e
            for e in context["events"]
            if e["event_type"] == "bounce"
            and label[1] < e["frame"] <= context["scene"].contact_frames[1]
        ),
        None,
    )
    if bounce_event is None or not _inside(
        bounce_interval, _interval(bounce_event["frame_interval"], "bounce")
    ):
        raise ValueError("bounce interval must lie inside the original first-flight bounce label")
    free_contact = fixed_contact_epoch is None
    indices = np.r_[np.arange(3), np.arange(3, 12), np.arange(3 + 3 * n, 12 + 3 * n)]
    scale = np.r_[[1.0] * 3, [30.0] * 9, [3.0] * 9]
    scale[FIRST_VZ] = 1.0
    depth_low, depth_high = map(float, context["depth_bounds"])
    contact_centre = float(np.mean(contact_interval))
    bounce_centre = float(np.mean(bounce_interval))
    obs = deepcopy(observations)
    obs["rows"] = [r for r in obs["rows"] if r["frame"] < label[0]]

    raw = source.copy()
    if initial_parameters is not None:
        raw[indices] = np.asarray(initial_parameters, float)[indices]
    incoming = np.array([0.0, 0.0, -2.0])
    if free_contact:
        contact0 = (
            float(initial_toss["shared_contact_frame"])
            if initial_toss is not None
            else float(context["scene"].contact_frames[0])
        )
        contact0 = float(np.clip(contact0, *contact_interval))
    else:
        contact0 = float(fixed_contact_epoch)
    release0 = float(np.mean(release_interval))
    if initial_toss is not None:
        incoming = np.asarray(initial_toss["incoming_contact_velocity_mps"], float)
        release0 = contact0 - float(initial_toss["release_seconds_before_contact"]) * fps
    release0 = float(np.clip(release0, *release_interval))
    try:
        warm = full.model.chain(full.profile.profile_context(context, contact0)["scene"], raw)
        bounce0 = float(warm[0]["bounces"][0]["frame"]) if warm[0]["bounces"] else bounce_centre
    except ValueError:
        bounce0 = bounce_centre
    bounce0 = float(np.clip(bounce0, *bounce_interval))

    q0 = np.r_[raw[indices] / scale, incoming, release0]
    q0[FIRST_VZ] = bounce0 - bounce_centre
    half_bounce = (bounce_interval[1] - bounce_interval[0]) / 2
    lo = np.r_[[0.0, depth_low, 2.0], [-2.5] * 9, [-2.0] * 9, [-12.0] * 3, release_interval[0]]
    hi = np.r_[[10.97, depth_high, 4.0], [2.5] * 9, [2.0] * 9, [12.0] * 3, release_interval[1]]
    lo[FIRST_VZ], hi[FIRST_VZ] = -half_bounce, half_bounce
    if free_contact:
        half_contact = (contact_interval[1] - contact_interval[0]) / 2
        q0 = np.r_[q0, contact0 - contact_centre]
        lo = np.r_[lo, -half_contact]
        hi = np.r_[hi, half_contact]
    q0 = np.clip(q0, lo + 1e-8, hi - 1e-8)
    reference_vz = float(raw[FIRST_VZ])
    bundles = {}

    def bundle(epoch):
        """Everything the objective needs at one actual contact epoch."""
        key = float(epoch)
        if key not in bundles:
            if len(bundles) >= BUNDLE_CACHE_ENTRIES:
                bundles.clear()
            ctx = full.profile.profile_context(context, key)
            scene, axes, activated = full.merge_scene(
                ctx["scene"], ctx["heldout"], ctx["bounces"], ctx["axes"]
            )
            bundles[key] = dict(
                ctx=ctx,
                scene=scene,
                axes=axes,
                activated=activated,
                projector=FirstFlightProjector(scene, bounce_interval, reference_vz),
                reference=full.model.chain(scene, source),
                target=np.concatenate(scene.pixels),
                queries=net_constraints.dense_queries(scene, ctx["bounces"]),
                expected=[
                    [
                        e
                        for e in ctx["events"]
                        if e["event_type"] == "bounce" and a < e["frame"] <= b
                    ]
                    for a, b in zip(
                        scene.contact_frames[:-1], scene.contact_frames[1:], strict=True
                    )
                ],
            )
        return bundles[key]

    def evaluate(q):
        epoch = contact_centre + q[25] if free_contact else contact0
        b = bundle(epoch)
        scene, queries, expected = b["scene"], b["queries"], b["expected"]
        p = source.copy()
        p[indices] = q[:21] * scale
        p, receipt = b["projector"].project(p, bounce_centre + q[FIRST_VZ])
        flights = full.model.chain(scene, p, query_frames=queries)
        image = (
            full.exposure.prediction(
                scene, p, b["axes"], duration, termination_kind=b["ctx"]["termination_kind"]
            )
            - b["target"]
        )
        tau = (epoch - q[24]) / fps
        tr, ts, record = toss.evaluate(
            p[:3], np.r_[q[21:24], tau], dict(obs, contact_frame=float(epoch)), feet, fps
        )
        ground = []
        net = []
        for i, (flight, events) in enumerate(zip(flights, expected, strict=True)):
            a, bb = scene.contact_frames[i : i + 2]
            for j, event in enumerate(events):
                actual = (
                    flight["bounces"][j]["frame"]
                    if j < len(flight["bounces"])
                    else future_ground_guidance(flight, bb, fps)
                )
                low, high = event["frame_interval"]
                ground.append((min(actual - low, 0) + max(actual - high, 0)) / 0.35)
            ground.append(
                max(bb - flight["bounces"][len(events)]["frame"], 0)
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
        athlete = athlete_priors.optimization_residuals(contacts, b["ctx"]["players"])
        stay = np.concatenate(
            [(flights[i]["end_xyz"] - b["reference"][i]["end_xyz"]) / 0.35 for i in range(3)]
        )
        residual = np.r_[
            image.ravel(),
            4 * tr,
            8 * np.minimum(ts, 0),
            ground,
            net,
            athlete * 4,
            stay,
            (p[indices[12:]] - source[indices[12:]]) / 6,
        ]
        return residual, p, record, receipt, image, float(epoch)

    best = None
    calls = invalid = 0
    length = len(evaluate(q0)[0])

    def residual(q):
        nonlocal best, calls, invalid
        if time.monotonic() - began > seconds:
            raise TimeoutError("joint serve contact/bounce wall budget")
        calls += 1
        try:
            r, p, record, receipt, image, epoch = evaluate(q)
        except ValueError:
            invalid += 1
            return np.full(length, 1e4)
        cost = float(r @ r)
        if not np.isfinite(cost):
            invalid += 1
            return np.full(length, 1e4)
        if best is None or cost < best["cost"]:
            best = dict(
                cost=cost,
                parameters=p,
                shared_toss=record,
                release_epoch=float(q[24]),
                rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
                contact_epoch=epoch,
                bounce_epoch=float(receipt["epoch"]),
                bounce_receipt=receipt,
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
        raise ValueError(
            f"no physical joint serve candidate ({calls} calls, {invalid} invalid): {termination}"
        )
    chosen = bundle(best["contact_epoch"])
    ctx = chosen["ctx"]
    exported = full.model.chain(ctx["scene"], best["parameters"])
    first = exported[0]["bounces"][0]["frame"] if exported[0]["bounces"] else None
    mismatch = None if first is None else abs(float(first) - best["bounce_epoch"])
    export = dict(
        original_chain_first_bounce_frame=first,
        latent_bounce_epoch=best["bounce_epoch"],
        mismatch_frames=mismatch,
        tolerance_frames=EXPORT_TOLERANCE_FRAMES,
        reproduced=mismatch is not None and mismatch <= EXPORT_TOLERANCE_FRAMES,
    )
    if not export["reproduced"]:
        raise ValueError(f"single-shooting export did not reproduce the latent impact: {export}")
    return ctx, dict(
        best=best,
        termination=termination,
        seconds=time.monotonic() - began,
        calls=calls,
        invalid_calls=invalid,
        contact_epoch=best["contact_epoch"],
        bounce_epoch=best["bounce_epoch"],
        export=export,
        activated_frames=chosen["activated"],
        fitting_scene=asdict(chosen["scene"]),
        policy=dict(
            changed_launch_blocks=[0, 1, 2],
            depth_bounds_m=[depth_low, depth_high],
            contact_interval_frames=list(contact_interval),
            bounce_interval_frames=list(bounce_interval),
            fixed_contact_epoch=fixed_contact_epoch,
            release_interval_frames=list(map(float, release_interval)),
            epoch_coordinates="interval-centred offsets; both bounded by original intervals",
            first_vz_coordinate="latent first-ground epoch; physical Vz from FirstImpactChart root",
            source_end_reference="source vector replayed under each evaluated contact epoch",
            root_bracket_seed_vz=reference_vz,
            toss_rows_fixed_before_frame=label[0],
            native_exposure_frames_unchanged=True,
            original_ground_law=True,
            shared_contact_exact=True,
            original_toss_checks=True,
            prior_withheld_consumed=True,
            source_end_soft_sigma_m=0.35,
            contact_prior=None,
            prior_weight=0.0,
            ground_missing_proxy="vertical-gravity future horizon guidance; never scored as impact",
            export="physical single-shooting vector; chain reproduces latent impact",
        ),
    )
