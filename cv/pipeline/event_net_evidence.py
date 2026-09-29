"""Optional native motion/net-envelope evidence for existing decoder nodes.

This is an explicitly uncalibrated bounded prior, never a new network prediction,
ground-truth event, physical ending, or competitive-phase certificate. Missing
observations/qualified same-frame geometry contribute zero evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline.camera_cal import net_source_is_tape_top
from cv.pipeline.event_grammar_decoder import MAX_TIME_CORRECTION, PROPOSAL_PRIOR_CAP, build_nodes
from cv.pipeline.event_impulse_support import CONFIG as WING_CONFIG, TrackSnapshot
from cv.pipeline.event_model_v2_features import COURT_WIDTH_M, NET_Y_M
from cv.pipeline.provenance import file_record

REGISTERED_SOURCES = frozenset({"registered", "registered_interpolated", "line_model_registered"})

CONFIG = {
    "prior_cap_nats": PROPOSAL_PRIOR_CAP,
    "native_ball_scale_px": 6.0,
    "excluded_event_radius_frames": 1,
    "wing_configuration": dict(WING_CONFIG),
    "calibration": "uncalibrated_bounded_native_net_evidence",
    "camera_policy": "direct_calibration_and_explicit_frame_track_registration; unknown_families_abstain",
    "candidate_policy": "existing_classifier_net_local_maxima_only",
    "certifies_competitive_phase": False,
    "certifies_physical_ending": False,
}


@dataclass
class NetSnapshot:
    track: TrackSnapshot
    cameras: dict[tuple[str, int], dict]

    @classmethod
    def load(cls, root: Path, match_ids: list[str]) -> NetSnapshot:
        track = TrackSnapshot.load(root, match_ids)
        cameras = {}
        for match in sorted(set(match_ids)):
            path = root / match / "camera_P_per_frame_v1.npz"
            cadence = root / match / "audit_frames_native_1080.coordinates.json"
            if not path.exists() or not cadence.exists():
                continue
            track.paths.extend([path, cadence])
            track.records.extend([file_record(path), file_record(cadence)])
            sizes = json.loads(cadence.read_text())
            if any(
                sizes.get(k) != {"width": 1920, "height": 1080}
                for k in ("image_size", "artifact_size")
            ):
                raise ValueError("net evidence requires declared native camera pixels")
            with np.load(path, allow_pickle=False) as archive:
                data = {name: archive[name] for name in archive.files}
                required = {
                    "clips",
                    "frames",
                    "P",
                    "reliable",
                    "source",
                    "net_residual_px",
                    "ground_residual_px",
                    "frame_scope",
                    "fallback_ancestry",
                }
                if not required <= set(data):
                    continue
                if data["frames"].dtype.kind not in "iu" or data["reliable"].dtype.kind != "b":
                    raise ValueError("net camera native epochs/reliability invalid")
                for i, (clip, frame) in enumerate(zip(data["clips"], data["frames"], strict=True)):
                    key = (f"{match}__{clip}", int(frame))
                    if key in cameras:
                        raise ValueError("duplicate same-frame net camera")
                    source = str(data["source"][i])
                    # camera_artifacts expands direct net-calibrated anchors using
                    # court_topology_runner's explicit registration state. Its
                    # observed cord travels through the same warp (not a repeated
                    # point-static pixel). Unknown/bundle/median families abstain
                    # until their own coordinate/lens contract is implemented.
                    calibration, _, registration = source.partition("+")
                    ground = float(data["ground_residual_px"][i])
                    reliable = (
                        bool(data["reliable"][i])
                        and calibration == "direct"
                        and registration in REGISTERED_SOURCES
                        and str(data["frame_scope"][i]) == "frame_track"
                        and json.loads(str(data["fallback_ancestry"][i])) == []
                        and math.isfinite(ground)
                        and 0 <= ground <= 4.0
                    )
                    projection = np.asarray(data["P"][i], float)
                    residual = float(data["net_residual_px"][i])
                    reliable &= projection.shape == (3, 4) and bool(np.isfinite(projection).all())
                    reliable &= math.isfinite(residual) and 0 <= residual <= 8.0
                    camera = dict(reliable=reliable, source=source, net_residual_px=residual)
                    if reliable:
                        x = np.linspace(0, COURT_WIDTH_M, 9)
                        # Net sags from 1.07m at the ends to .914m at the strap.
                        z = (
                            0.914
                            + (1.07 - 0.914) * ((x - COURT_WIDTH_M / 2) / (COURT_WIDTH_M / 2)) ** 2
                        )
                        world = np.column_stack(
                            (
                                np.r_[x, x[::-1]],
                                np.full(18, NET_Y_M),
                                np.r_[z, np.zeros(9)],
                                np.ones(18),
                            )
                        )
                        projected = world @ projection.T
                        if np.any(projected[:, 2] <= 0):
                            camera["reliable"] = False
                        else:
                            polygon = projected[:, :2] / projected[:, 2:3]
                            if "net_cord_valid" in data and bool(data["net_cord_valid"][i]):
                                cord = np.asarray(data["net_cord_xy"][i], float)
                                cord_source = (
                                    str(data["net_cord_source"][i])
                                    if "net_cord_source" in data
                                    else "unknown"
                                )
                                if (
                                    cord.shape == (9, 2)
                                    and np.isfinite(cord).all()
                                    and net_source_is_tape_top(cord_source)
                                ):
                                    polygon[:9] = cord
                                    camera["top_source"] = (
                                        cord_source + ":registered_to_native_frame"
                                    )
                                else:
                                    camera["reliable"] = False
                                    camera["qualification_failure"] = (
                                        "unknown_or_invalid_observed_net_cord"
                                    )
                            camera.setdefault(
                                "top_source", "same_frame_camera_projected_sagging_tape"
                            )
                            camera["polygon"] = polygon
                    cameras[key] = camera
        track.assert_unchanged()
        return cls(track, cameras)


def certificate(
    track: dict[int, np.ndarray], camera: dict | None, epoch: float, fps: float
) -> dict:
    def held(reason: str, **details) -> dict:
        return dict(supported=False, reason=reason, prior_nats=0.0, **details)

    if not math.isfinite(epoch) or not math.isfinite(fps) or fps <= 0:
        return held("invalid_native_epoch_or_cadence")
    if camera is None or not camera.get("reliable"):
        return held("qualified_same_frame_net_geometry_unavailable")
    center = round(epoch)
    count = max(WING_CONFIG["minimum_wing_frames"], math.ceil(WING_CONFIG["wing_seconds"] * fps))
    radius = CONFIG["excluded_event_radius_frames"]
    spans = [
        list(range(center - radius - count, center - radius)),
        list(range(center + radius + 1, center + radius + count + 1)),
    ]
    coefficients, residuals = [], []
    for samples in spans:
        if any(frame not in track for frame in samples):
            return held("insufficient_native_observed_wing", wing_frames=spans)
        xy = np.asarray([track[f] for f in samples], float)
        if not np.isfinite(xy).all():
            return held("nonfinite_native_observation")
        times = np.asarray(samples, float) - epoch
        coefficient = np.polynomial.polynomial.polyfit(times, xy, 2)
        predicted = np.polynomial.polynomial.polyval(times, coefficient).T
        coefficients.append(coefficient)
        residuals.append(float(np.sqrt(np.mean(np.sum((predicted - xy) ** 2, axis=1)))))
    delta = coefficients[0] - coefficients[1]
    squared = sum(
        np.polynomial.polynomial.polymul(delta[:, axis], delta[:, axis]) for axis in range(2)
    )
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(squared))
    time_radius = WING_CONFIG["event_time_radius_frames"]
    times = [
        -time_radius,
        time_radius,
        *[
            float(r.real)
            for r in roots
            if abs(r.imag) < 1e-8 and -time_radius <= r.real <= time_radius
        ],
    ]
    offset = min(times, key=lambda t: np.polynomial.polynomial.polyval(t, squared))
    join = float(np.linalg.norm(np.polynomial.polynomial.polyval(offset, delta)))
    velocity = float(np.linalg.norm(delta[1] + 2 * offset * delta[2]) * fps)
    pixel = np.mean([np.polynomial.polynomial.polyval(offset, c) for c in coefficients], axis=0)
    polygon = np.asarray(camera["polygon"], np.float32)
    distance = max(0.0, -cv2.pointPolygonTest(polygon, tuple(map(float, pixel)), True))
    sigma = math.hypot(CONFIG["native_ball_scale_px"], camera["net_residual_px"])
    details = dict(
        wing_frames=spans,
        wing_rms_native_px=residuals,
        join_native_px=join,
        velocity_change_native_px_per_second=velocity,
        join_epoch=epoch + offset,
        join_pixel_native=pixel.tolist(),
        envelope_distance_native_px=distance,
        envelope_scale_native_px=sigma,
        camera_source=camera["source"],
        net_top_source=camera.get("top_source"),
        model_epoch_unchanged=epoch,
        localization_status="interval_support_only",
    )
    if (
        max(residuals) > WING_CONFIG["maximum_wing_rms_native_px"]
        or join > WING_CONFIG["maximum_wing_join_native_px"]
        or velocity < WING_CONFIG["minimum_velocity_change_native_px_per_second"]
    ):
        return held("native_wing_motion_not_supported", **details)
    if distance > sigma:
        return held("motion_change_outside_physical_net_envelope", **details)
    # One bounded evidence contribution, never a calibrated event probability.
    # A boundary-uncertain envelope receives smoothly smaller influence.
    prior = CONFIG["prior_cap_nats"] * math.exp(-0.5 * (distance / sigma) ** 2)
    return dict(
        supported=True, reason="native_motion_at_qualified_net", prior_nats=prior, **details
    )


def build_evidence(
    prediction: dict, grammar, fps: dict[str, float], snapshot: NetSnapshot
) -> tuple[np.ndarray, list[dict]]:
    """Return priors for the original classifier row space, plus every net-node audit."""
    clips, frames = np.asarray(prediction["clips"]), np.asarray(prediction["frames"])
    prior = np.zeros((len(frames), 3), dtype=float)
    report = []
    for clip in sorted(set(clips.tolist())):
        positions = np.flatnonzero(clips == clip)
        nodes = build_nodes(
            prediction["probabilities"][positions],
            frames[positions],
            prediction["court_y"][positions],
            config=grammar,
            clip=clip,
        )
        for node in nodes:
            if node.event_type != "net_hit":
                continue
            row = int(positions[node.index])
            shift = float(
                np.clip(prediction["time_offset"][row], -MAX_TIME_CORRECTION, MAX_TIME_CORRECTION)
            )
            epoch = float(frames[row]) + shift
            camera = snapshot.cameras.get((clip, round(epoch)))
            observed = bool(prediction["track_observed"][row])
            evidence = (
                certificate(
                    snapshot.track.tracks.get(clip, {}),
                    camera,
                    epoch,
                    float(fps[str(prediction["broadcasts"][row])]),
                )
                if observed
                else dict(supported=False, reason="candidate_not_observed", prior_nats=0.0)
            )
            prior[row, 2] = evidence["prior_nats"]
            report.append(dict(clip=clip, row=row, candidate_frame=int(frames[row]), **evidence))
    snapshot.track.assert_unchanged()
    return prior, report


def retain_original_decisions(emissions: list[dict], baseline: list[dict]) -> list[dict]:
    """Keep the unconditioned decoder distinct from the evidence-conditioned one."""

    def key(r):
        return r["clip"], r["event_type"], r["candidate_frame"]

    original = {key(r): r for r in baseline}
    result = []
    for row in emissions:
        old = original.get(key(row))
        fields = (
            "abstain",
            "path_marginal",
            "acceptance_marginal",
            "decision_threshold",
            "on_best_path",
        )
        result.append(
            {
                **row,
                "unconditioned_decoder": {
                    "emitted": old is not None,
                    **({k: old.get(k) for k in fields} if old else {}),
                },
                "decoder_evidence_scope": CONFIG["calibration"],
            }
        )
    return result


def condition_emissions(
    prediction, store, grammar, fps, threshold, *, emission_mode, correct_frames, baseline, snapshot
):
    """Shared runtime and exact CPU replay boundary; no model or S6 calls."""
    from cv.pipeline import event_time_distribution as timing
    from cv.pipeline.event_grammar_decoder import decode, emission_rows

    prior, audit = build_evidence(prediction, grammar, fps, snapshot)
    events = decode(
        prediction["probabilities"],
        store.clips(),
        store.frames(),
        store.court_y(),
        config=grammar,
        emission_mode=emission_mode,
        evidence_log_prior=prior,
    )
    rows = emission_rows(
        events,
        store,
        prediction["probabilities"],
        prediction["offset"],
        prediction["xy"],
        store.centres(),
        fps,
        threshold,
        time_offset=prediction["time_offset"],
        time_distribution=timing.arrays(prediction, rows=len(store.frames())),
        correct_frames=correct_frames,
    )
    return retain_original_decisions(rows, baseline), audit
