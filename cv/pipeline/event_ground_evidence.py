"""Optional bounded native-motion evidence for existing bounce decoder nodes.

Ground intersection is a hypothesis, not measured airborne depth. The prior is
uncalibrated and correlated with classifier inputs; it cannot certify an ending.
Unknown camera families, radial optics and missing actor context abstain.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from cv.pipeline.event_grammar_decoder import MAX_TIME_CORRECTION, PROPOSAL_PRIOR_CAP, build_nodes
from cv.pipeline.event_impulse_support import CONFIG as WING_CONFIG, TrackSnapshot
from cv.pipeline.event_model_v2_features import COURT_LENGTH_M, COURT_WIDTH_M
from cv.pipeline.provenance import file_record
from cv.pipeline.resolution import coordinate_manifest_path, coordinate_manifest_errors

CONFIG = {
    "prior_cap_nats": PROPOSAL_PRIOR_CAP,
    "wing_configuration": dict(WING_CONFIG),
    "excluded_event_radius_frames": 1,
    "minimum_smooth_rms_native_px": 3.0,
    "minimum_smooth_to_wing_rms": 2.0,
    "maximum_cross_up_impulse_ratio": 0.5,
    "minimum_player_confidence": 0.5,
    "player_exclusion_body_heights": 0.5,
    "runoff_margin_m": 10.0,
    "maximum_ground_residual_px": 4.0,
    "calibration": "uncalibrated_bounded_native_ground_evidence",
    "candidate_policy": "existing_classifier_bounce_local_maxima_only",
    "camera_policy": "direct_registered_native_pinhole_only",
    "player_cadence_policy": "declared_native_source_fps_equals_classifier_fps",
    "certifies_physical_ending": False,
    "certifies_competitive_phase": False,
}


def native_player_artifact(folder: Path) -> Path | None:
    """Find the original native actor artifact without assuming a 25fps source.

    Multiple cadence exports are ambiguous; do not choose one by filename order.
    The coordinate sidecar and certificate still validate actual source cadence.
    """
    candidates = sorted(folder.glob("player_boxes_*_native_sided_v1.csv"))
    if len(candidates) > 1:
        raise ValueError("multiple native player artifacts; explicit source binding required")
    return candidates[0] if candidates else None


@dataclass
class GroundSnapshot:
    track: TrackSnapshot
    cameras: dict
    players: dict

    @classmethod
    def load(cls, root: Path, match_ids: list[str], *, track=None):
        track = track if track is not None else TrackSnapshot.load(root, match_ids)
        cameras, players = {}, {}
        for match in sorted(set(match_ids)):
            folder = root / match
            path = folder / "camera_P_per_frame_v1.npz"
            cadence = folder / "audit_frames_native_1080.coordinates.json"
            if path.exists() and cadence.exists():
                track.paths.extend([path, cadence])
                track.records.extend([file_record(path), file_record(cadence)])
                sizes = json.loads(cadence.read_text())
                if any(
                    sizes.get(k) != {"width": 1920, "height": 1080}
                    for k in ("image_size", "artifact_size")
                ):
                    raise ValueError("ground evidence requires declared native camera pixels")
                with np.load(path, allow_pickle=False) as archive:
                    z = {key: archive[key] for key in archive.files}
                    required = {
                        "clips",
                        "frames",
                        "P",
                        "reliable",
                        "source",
                        "ground_residual_px",
                        "frame_scope",
                        "fallback_ancestry",
                    }
                    if required <= set(z):
                        if z["frames"].dtype.kind not in "iu" or z["reliable"].dtype.kind != "b":
                            raise ValueError("ground camera native epochs/reliability invalid")
                        for i, (clip, frame) in enumerate(
                            zip(z["clips"], z["frames"], strict=True)
                        ):
                            key = (f"{match}__{clip}", int(frame))
                            if key in cameras:
                                raise ValueError("duplicate same-frame ground camera")
                            source = str(z["source"][i])
                            residual = float(z["ground_residual_px"][i])
                            projection = np.asarray(z["P"][i], float)
                            calibration, _, registration = source.partition("+")
                            # Do not silently interpret a radial or partially described
                            # camera as pinhole. Its ground/Jacobian path is not supported here.
                            lens_declared = any(k in z for k in ("k1", "dist_center", "lens_model"))
                            reliable = (
                                bool(z["reliable"][i])
                                and calibration == "direct"
                                and registration
                                in {
                                    "registered",
                                    "registered_interpolated",
                                    "line_model_registered",
                                }
                                and str(z["frame_scope"][i]) == "frame_track"
                                and json.loads(str(z["fallback_ancestry"][i])) == []
                                and math.isfinite(residual)
                                and 0 <= residual <= CONFIG["maximum_ground_residual_px"]
                                and projection.shape == (3, 4)
                                and np.isfinite(projection).all()
                                and not lens_declared
                            )
                            cameras[key] = dict(
                                reliable=bool(reliable),
                                P=projection,
                                source=source,
                                ground_residual_px=residual,
                                lens_status="unsupported_explicit_lens"
                                if lens_declared
                                else "native_pinhole_producer",
                            )
            path = native_player_artifact(folder)
            if path is None:
                continue
            sidecar = coordinate_manifest_path(path)
            if not path.exists() or not sidecar.exists():
                continue
            track.paths.extend([path, sidecar])
            track.records.extend([file_record(path), file_record(sidecar)])
            metadata = json.loads(sidecar.read_text())
            if (
                coordinate_manifest_errors(metadata)
                or metadata.get("box_coordinate_space") != "native"
                or metadata.get("image_size") != {"width": 1920, "height": 1080}
                or metadata.get("artifact_size") != {"width": 1920, "height": 1080}
                or metadata.get("coordinate_columns", {}).get("native_1920x1080")
                != ["x0_native", "y0_native", "x1_native", "y1_native"]
            ):
                raise ValueError("ground evidence requires explicit native player box columns")
            declared_fps = metadata.get("source_fps")
            if type(declared_fps) not in (int, float):
                declared_fps = None
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    frame = int(Path(row["frame"]).stem.removeprefix("f_"))
                    key = (f"{match}__{row['clip']}", frame)
                    players.setdefault(key, []).append(
                        dict(
                            side=row["side"],
                            confidence=float(row["conf"]),
                            source_fps=declared_fps,
                            box=[
                                float(row[c])
                                for c in ("x0_native", "y0_native", "x1_native", "y1_native")
                            ],
                        )
                    )
        track.assert_unchanged()
        return cls(track, cameras, players)


def certificate(track, camera, players, epoch, fps, *, actor_attenuation=False):
    def held(reason, **details):
        return dict(supported=False, reason=reason, prior_nats=0.0, **details)

    if not math.isfinite(epoch) or not math.isfinite(fps) or fps <= 0:
        return held("invalid_native_epoch_or_cadence")
    if camera is None or not camera.get("reliable"):
        return held("qualified_same_frame_ground_geometry_unavailable")
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
    delta = coefficients[1] - coefficients[0]
    squared = np.polynomial.polynomial.polyadd(
        *[np.polynomial.polynomial.polymul(delta[:, axis], delta[:, axis]) for axis in range(2)]
    )
    derivative = np.polynomial.polynomial.polyder(squared)
    derivative = np.polynomial.polynomial.polytrim(
        derivative, tol=1e-12 * max(1.0, float(np.max(np.abs(derivative))))
    )
    roots = np.polynomial.polynomial.polyroots(derivative)
    radius = WING_CONFIG["event_time_radius_frames"]
    offsets = [
        -radius,
        radius,
        *[float(r.real) for r in roots if abs(r.imag) < 1e-8 and -radius <= r.real <= radius],
    ]
    offset = min(offsets, key=lambda t: np.polynomial.polynomial.polyval(t, squared))
    join = float(np.linalg.norm(np.polynomial.polynomial.polyval(offset, delta)))
    pixel = np.mean([np.polynomial.polynomial.polyval(offset, c) for c in coefficients], axis=0)
    impulse = (delta[1] + 2 * offset * delta[2]) * fps
    all_frames = sum(spans, [])
    times = np.asarray(all_frames, float) - epoch
    xy = np.asarray([track[f] for f in all_frames])
    smooth = np.polynomial.polynomial.polyfit(times, xy, 2)
    smooth_rms = float(
        np.sqrt(
            np.mean(np.sum((np.polynomial.polynomial.polyval(times, smooth).T - xy) ** 2, axis=1))
        )
    )
    details = dict(
        wing_frames=spans,
        wing_rms_native_px=residuals,
        single_smooth_rms_native_px=smooth_rms,
        join_native_px=join,
        join_epoch=epoch + offset,
        join_pixel_native=pixel.tolist(),
        impulse_native_px_per_second=impulse.tolist(),
        model_epoch_unchanged=epoch,
        localization_status="interval_support_only",
        camera_source=camera["source"],
    )
    if (
        max(residuals) > WING_CONFIG["maximum_wing_rms_native_px"]
        or join > WING_CONFIG["maximum_wing_join_native_px"]
    ):
        return held("native_wing_motion_not_supported", **details)
    if smooth_rms < max(
        CONFIG["minimum_smooth_rms_native_px"],
        CONFIG["minimum_smooth_to_wing_rms"] * max(residuals),
    ):
        return held("smooth_continuation_explains_observations", **details)
    projection = np.asarray(camera["P"])
    try:
        ground = np.linalg.solve(projection[:, [0, 1, 3]], np.r_[pixel, 1.0])
    except np.linalg.LinAlgError:
        return held("ground_projection_singular", **details)
    if abs(ground[2]) < 1e-10:
        return held("ground_ray_near_horizon", **details)
    ground /= ground[2]
    projected = projection @ np.array([*ground[:2], 0.0, 1.0])
    if projected[2] <= 0:
        return held("ground_ray_behind_camera", **details)
    margin = CONFIG["runoff_margin_m"]
    if not (
        -margin <= ground[0] <= COURT_WIDTH_M + margin
        and -margin <= ground[1] <= COURT_LENGTH_M + margin
    ):
        return held("ground_hypothesis_outside_court_and_runoff", **details)
    vertical = (projection[:2, 2] - pixel * projection[2, 2]) / projected[2]
    if np.linalg.norm(vertical) < 1e-6:
        return held("ground_vertical_projection_unconditioned", **details)
    vertical /= np.linalg.norm(vertical)
    upward = float(impulse @ vertical)
    cross = float(np.linalg.norm(impulse - upward * vertical))
    details.update(
        hypothetical_ground_m=ground[:2].tolist(),
        upward_impulse_native_px_per_second=upward,
        cross_vertical_impulse_native_px_per_second=cross,
    )
    if (
        upward < WING_CONFIG["minimum_velocity_change_native_px_per_second"]
        or cross > CONFIG["maximum_cross_up_impulse_ratio"] * upward
    ):
        return held("impulse_not_supported_as_upward_ground_response", **details)
    if players and any(
        type(row.get("source_fps")) not in (int, float)
        or not math.isfinite(row["source_fps"])
        or not math.isclose(row["source_fps"], fps, rel_tol=0.0, abs_tol=1e-6)
        for row in players
    ):
        return held("native_player_cadence_unavailable_or_mismatch", **details)
    qualified = []
    for row in players or []:
        box = np.asarray(row["box"], float)
        if (
            row["side"] in {"near", "far"}
            and np.isfinite(box).all()
            and math.isfinite(row["confidence"])
            and row["confidence"] >= CONFIG["minimum_player_confidence"]
            and box[2] > box[0]
            and box[3] > box[1]
        ):
            qualified.append(row)
    if len(qualified) != 2 or {r["side"] for r in qualified} != {"near", "far"}:
        return held("two_unambiguous_native_player_boxes_unavailable", **details)
    actor_distances = []
    for row in qualified:
        box = np.asarray(row["box"], float)
        distance = float(
            np.linalg.norm(np.maximum(np.maximum(box[:2] - pixel, pixel - box[2:]), 0))
        )
        body_height = box[3] - box[1]
        actor_distances.append(dict(side=row["side"], distance_body_heights=distance / body_height))
    details["actor_distances"] = actor_distances
    overlaps_actor = (
        min(r["distance_body_heights"] for r in actor_distances)
        <= CONFIG["player_exclusion_body_heights"]
    )
    if overlaps_actor and not actor_attenuation:
        return held("native_motion_overlaps_actor_context", **details)
    # Bounded geometric consistency, not calibrated likelihood. Ground fit error
    # shares image units with native wing error, so enlarge that uncertainty scale.
    sigma = math.hypot(WING_CONFIG["maximum_wing_rms_native_px"], camera["ground_residual_px"])
    penalty = (
        (join / WING_CONFIG["maximum_wing_join_native_px"]) ** 2
        + (max(residuals) / sigma) ** 2
        + (cross / upward) ** 2
    )
    prior = CONFIG["prior_cap_nats"] * math.exp(-0.5 * penalty)
    if actor_attenuation:
        # Proximity makes a vertical impulse less specific to ground impact. This
        # fixed discount changes evidence strength, not the observed motion or
        # calibrated classifier probability; missing actor context still holds.
        factor = 0.5 if overlaps_actor else 1.0
        details.update(actor_prior_factor=factor, actor_policy="qualified_proximity_half_prior_v1")
        prior *= factor
    return dict(
        supported=True,
        reason=(
            "native_upward_ground_impulse_actor_discounted"
            if overlaps_actor
            else "native_upward_ground_impulse_away_from_actors"
        ),
        prior_nats=prior,
        **details,
    )


def build_evidence(prediction, grammar, fps, snapshot, *, actor_attenuation=False):
    clips, frames = np.asarray(prediction["clips"]), np.asarray(prediction["frames"])
    prior = np.zeros((len(frames), 3), float)
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
            if node.event_type != "bounce":
                continue
            row = int(positions[node.index])
            epoch = float(frames[row]) + float(
                np.clip(prediction["time_offset"][row], -MAX_TIME_CORRECTION, MAX_TIME_CORRECTION)
            )
            key = (clip, round(epoch))
            evidence = (
                certificate(
                    snapshot.track.tracks.get(clip, {}),
                    snapshot.cameras.get(key),
                    snapshot.players.get(key),
                    epoch,
                    float(fps[str(prediction["broadcasts"][row])]),
                    actor_attenuation=actor_attenuation,
                )
                if bool(prediction["track_observed"][row])
                else dict(supported=False, reason="candidate_not_observed", prior_nats=0.0)
            )
            prior[row, 1] = evidence["prior_nats"]
            report.append(
                dict(
                    clip=clip,
                    row=row,
                    candidate_frame=int(frames[row]),
                    event_type="bounce",
                    original_model_epoch=epoch,
                    original_classifier_probabilities=dict(
                        zip(
                            ("none", "contact", "bounce", "net_hit"),
                            map(float, prediction["probabilities"][row]),
                            strict=True,
                        )
                    ),
                    **evidence,
                )
            )
    snapshot.track.assert_unchanged()
    return prior, report


def condition_emissions(
    prediction,
    store,
    grammar,
    fps,
    threshold,
    *,
    emission_mode,
    correct_frames,
    baseline,
    snapshot,
    net_snapshot=None,
    actor_attenuation=False,
):
    """One decode with optional independent class priors; retain raw decisions."""
    from cv.pipeline import event_net_evidence as net
    from cv.pipeline import event_time_distribution as timing
    from cv.pipeline.event_grammar_decoder import decode, emission_rows

    prior, audit = build_evidence(
        prediction,
        grammar,
        fps,
        snapshot,
        **({"actor_attenuation": True} if actor_attenuation else {}),
    )
    if net_snapshot is not None:
        net_prior, net_audit = net.build_evidence(prediction, grammar, fps, net_snapshot)
        prior += net_prior
        audit += [dict(event_type="net_hit", **row) for row in net_audit]
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
    rows = net.retain_original_decisions(rows, baseline)
    scope = (
        "uncalibrated_bounded_native_net_and_ground_evidence"
        if net_snapshot is not None
        else CONFIG["calibration"]
    )
    return [dict(row, decoder_evidence_scope=scope) for row in rows], audit
