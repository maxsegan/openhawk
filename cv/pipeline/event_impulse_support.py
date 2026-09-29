"""Optional two-wing image support for tracking-held contact/bounce events.

Disabled unless explicitly requested by an event consumer. This bounded positional
and velocity-impulse check does not certify event truth, 3D geometry or physical
endings, and cannot override camera holds or model abstentions. Fixed thresholds
were tested on opened development evidence; composed default promotion is pending.

``native_streak_fallback`` adds one further, separately disabled way for a single wing
to satisfy the existing per-sample localization allowance, using streak extents
measured from the original pictures by ``event_wing_streak``. It is OR-ed with the
original test and never AND-ed, so with the flag off -- the default -- every output on
this path is byte-identical to the path without it.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cv.pipeline.provenance import file_record
from cv.pipeline.resolution import (
    coordinate_columns_and_size,
    coordinate_manifest_errors,
    coordinate_manifest_path,
)


@dataclass
class TrackSnapshot:
    """Explicit run-scoped native observations, checked again before publishing."""

    tracks: dict[str, dict[int, np.ndarray]]
    paths: list[Path]
    records: list[dict]
    excluded_derived_frames: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def load(
        cls, root: Path, match_ids: list[str], *, extra_paths: tuple[Path, ...] = ()
    ) -> TrackSnapshot:
        tracks = defaultdict(dict)
        excluded = defaultdict(list)
        seen = defaultdict(set)
        paths = list(extra_paths)
        records = [file_record(path) for path in paths]
        for match_id in sorted(set(match_ids)):
            path = root / match_id / "ball_track_joint_native1080_arc_augmented_v2.csv"
            sidecar = coordinate_manifest_path(path)
            paths.extend([path, sidecar])
            records.extend([file_record(path), file_record(sidecar)])
            coordinates = json.loads(sidecar.read_text())
            if coordinate_manifest_errors(coordinates):
                raise ValueError("invalid impulse track coordinate manifest")
            # The composed tracker can copy an interpolated guide row while
            # labeling it coarse_lock. Preserve that ancestry when it is bound
            # by the standard coordinate sidecar and the copied pixel is exact.
            guide_rows = {}
            guide_name = str(coordinates.get("source", "")).split(" + ", 1)[0]
            guide = Path(guide_name)
            if not guide.is_absolute():
                guide = path.parent / guide
            if guide != path and guide.suffix == ".csv":
                if not guide.is_file():
                    raise ValueError("declared impulse guide source is unavailable")
                from cv.pipeline.resolution import native_point_reader

                guide_sidecar = coordinate_manifest_path(guide)
                paths.extend([guide, guide_sidecar])
                records.extend([file_record(guide), file_record(guide_sidecar)])
                with guide.open() as handle:
                    reader = csv.DictReader(handle)
                    native, _, _ = native_point_reader(guide, reader.fieldnames or ())
                    for row in reader:
                        if "interpolated" in row.get("sources", ""):
                            guide_frame = int(Path(row["frame"]).stem.removeprefix("f_"))
                            guide_rows[(row["clip"], guide_frame)] = native(row)
            with path.open() as handle:
                reader = csv.DictReader(handle)
                columns, _ = coordinate_columns_and_size(
                    reader.fieldnames or [],
                    coordinates,
                    native_columns=("x_native", "y_native"),
                    legacy_columns=("x", "y"),
                )
                if columns != ("x_native", "y_native"):
                    raise ValueError("impulse support requires explicit native track columns")
                for row in reader:
                    key = f"{match_id}__{row['clip']}"
                    frame = int(Path(row["frame"]).stem.removeprefix("f_"))
                    if frame in seen[key]:
                        raise ValueError("duplicate impulse track exposure")
                    seen[key].add(frame)
                    xy = np.asarray([float(row["x_native"]), float(row["y_native"])])
                    source = row.get("sources", "")
                    guide_xy = guide_rows.get((row["clip"], frame))
                    derived = "interpolated" in source or (
                        "coarse_lock" in source
                        and guide_xy is not None
                        and np.allclose(xy, guide_xy, rtol=0, atol=1e-7)
                    )
                    if derived:
                        excluded[key].append(frame)
                        continue
                    tracks[key][frame] = xy
        snapshot = cls(dict(tracks), paths, records, dict(excluded))
        snapshot.assert_unchanged()
        return snapshot

    def assert_unchanged(self) -> None:
        for path, before in zip(self.paths, self.records, strict=True):
            if file_record(path)["sha256"] != before["sha256"]:
                raise ValueError("impulse support input changed during inference")


def recover_events(
    rows: list[dict],
    tracking_gate: dict,
    snapshot: TrackSnapshot,
    *,
    native_wing_sampling: str = "consecutive",
    native_streak_fallback: bool = False,
    pictures=None,
) -> list[dict]:
    """Recover only independently supported event claims, never held track frames."""
    arcs = {f"{r['match_id']}__{r['clip']}": r.get("arcs", []) for r in tracking_gate["rows"]}
    output = []
    for source in rows:
        clip = source["clip"]
        prefix = f"{source['match_id']}__"
        key = clip if clip.startswith(prefix) else prefix + clip
        evidence = certificate(
            source,
            arcs.get(key, []),
            snapshot.tracks.get(key, {}),
            native_wing_sampling=native_wing_sampling,
            native_streak_fallback=native_streak_fallback,
            pictures=pictures,
        )
        row = {
            **source,
            "impulse_support": evidence,
            "pre_impulse_abstain": source["abstain"],
            "pre_impulse_gate_held": source.get("gate_held"),
            "pre_impulse_point_gate_failure_reasons": list(
                source.get("point_gate_failure_reasons", [])
            ),
        }
        if evidence["supported"]:
            row.update(abstain=False, gate_held=False, point_gate_failure_reasons=[])
            row["event_identity_support"] = {
                "status": "supported_by_model_and_observed_impulse",
                "original_model_abstain": source["model_abstain"],
                "localization_status": evidence["localization_status"],
                "certifies_physical_ending": False,
            }
            original = source.get("tracking_arc_gate", {})
            row["pre_impulse_tracking_arc_gate"] = dict(original)
            row["tracking_arc_gate"] = {
                **original,
                "decision": "supported_impulse_event_only",
                "track_frames_restored": False,
            }
        output.append(row)
    snapshot.assert_unchanged()
    return output


ALLOWED_REASONS = {
    "too_few_arc_observations",
    "excessive_arc_innovation",
    "excessive_arc_innovation_covariance",
}
CONFIG = {
    "maximum_impulse_seconds": 0.12,
    "wing_seconds": 0.20,
    "minimum_wing_frames": 5,
    "maximum_wing_rms_native_px": 3.0,
    "maximum_wing_join_native_px": 12.0,
    "maximum_event_pixel_native_px": 12.0,
    "minimum_velocity_change_native_px_per_second": 50.0,
    "event_time_radius_frames": 1.0,
    "identity_admission": "original_decoder_decision",
    "event_types": ["contact", "bounce"],
}


def model_identity_admission(event: dict) -> dict | None:
    """Read the decoder's admission score, without recalibrating a raw class head.

    Current second-bounce support states use acceptance_marginal, which can be
    lower than the full-lattice confidence. Historical decoder rows used their
    path marginal directly. Both retain the original operating threshold.
    """
    if event.get("model_abstain") is not False:
        return None
    field = next(
        (key for key in ("acceptance_marginal", "path_marginal", "confidence") if key in event),
        None,
    )
    try:
        marginal = float(event[field]) if field is not None else float("nan")
        threshold = float(event["decision_threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= threshold <= marginal <= 1):
        return None
    return dict(
        policy="original_decoder_decision",
        score_field=field,
        score=marginal,
        decision_threshold=threshold,
        original_model_abstain=False,
        raw_class_probability_rethresholded=False,
    )


STREAK_FALLBACK_SCHEMA = "event_native_streak_fallback_v1"


def streak_fallback_configuration(
    enabled: bool, *, supported_impulse_events: bool = True
) -> dict | None:
    """Declare the optional measured-streak wing fallback, or nothing when it is off.

    The fallback is an extra way for *one* wing to satisfy the existing per-sample
    localization allowance; it is not a new event test.  Everything else in
    :func:`certificate` -- the original decoder admission, the tracking-only hold, the
    short fully observed impulse, the two retained contiguous wings, and the join,
    event-pixel, velocity-change and timing-window conditions -- is unchanged, and the
    extra evidence certifies neither competitive flight nor a physical ending.
    """
    if not enabled:
        return None
    if not supported_impulse_events:
        raise ValueError("native streak fallback requires supported impulse events")
    from cv.pipeline import event_wing_streak as streak

    return {
        "schema": STREAK_FALLBACK_SCHEMA,
        "flag": "native_streak_fallback",
        "default": False,
        "applies_to": "exactly one wing whose rms exceeds maximum_wing_rms_native_px",
        "other_wing": "untouched",
        "per_sample_admission": (
            "original residual within the allowance, or measured native streak support "
            "bound to this source/frame/observed pixel/config, measured in pictures "
            "verified against the clip's original extraction receipt, with the "
            "cross-track residual within the allowance and the fitted point inside the "
            "measurement"
        ),
        "requires_original_extraction_receipt": True,
        "allowance_native_px": CONFIG["maximum_wing_rms_native_px"],
        "measurement_schema": streak.SCHEMA,
        "measurement_config_digest": streak.configuration_digest(),
        "combines_with_original_test": "or_never_and",
        "reclassifies_event": False,
        "changes_event_time": False,
        "certifies_competitive_flight": False,
        "certifies_physical_ending": False,
    }


def _measured_step(samples: list[int], observed: np.ndarray, index: int):
    """Per-frame displacement from the wing's own measured rows, never a fit."""
    if len(samples) < 2:
        return None
    first, second = (index, index + 1) if index + 1 < len(samples) else (index - 1, index)
    span = float(samples[second] - samples[first])
    if span <= 0:
        return None
    return (observed[second] - observed[first]) / span


def _support_binds(
    support,
    *,
    match_id: str,
    clip: str,
    frame: int,
    point,
    digest: str,
    fps: float | None = None,
    motion=None,
) -> bool:
    """Does this measurement belong to exactly the sample that asked for it?

    Both the binding written by the picture source and the measurement's own recorded
    source are checked, on exact floats.  Support measured at frame ``n`` therefore
    cannot be read as support for ``n+1``, for a different clip or source, for a
    neighbouring pixel, for a different declared cadence or measured local motion, or
    for a different measurement configuration.  Available support must additionally
    declare that its pictures were verified against the original extraction receipt.
    """
    if not isinstance(support, dict):
        return False
    binding = support.get("binding")
    expected = [float(value) for value in np.ravel(point)[:2]]
    motion = None if motion is None else [float(value) for value in np.ravel(motion)[:2]]
    # The schema name is spelled out rather than imported: event_native_pictures binds
    # its measurements onto this module's TrackSnapshot, so it imports this one.
    if not isinstance(binding, dict) or (
        binding.get("schema") != "event_native_picture_binding_v1"
        or binding.get("match_id") != match_id
        or binding.get("clip") != clip
        or binding.get("frame") != frame
        or binding.get("config_digest") != digest
        or binding.get("observed_point_native") != expected
        or (fps is not None and binding.get("requested_fps") != float(fps))
        or binding.get("observed_motion_native_px") != motion
    ):
        return False
    if not support.get("available"):
        return True
    source = support.get("source")
    return (
        # Available support must name a verified original extraction: pixels that are
        # not bound to the receipt that produced them are not evidence about this event.
        binding.get("extraction_verified") is True
        and support.get("config_digest") == digest
        and isinstance(source, dict)
        and (source.get("current") or {}).get("frame_index") == frame
        and source.get("observed_point_native") == expected
        and source.get("observed_motion_native_px") == motion
        and [record.get("frame_index") for record in source.get("neighbours", [])]
        == [frame - 1, frame + 1]
    )


def _native_streak_wing(
    event: dict,
    exceeded: list[int],
    spans: list[list[int]],
    coefficients: list[np.ndarray],
    residuals: list[float],
    track: dict[int, np.ndarray],
    frame: float,
    fps: float,
    pictures,
) -> dict:
    """May the one wing outside the allowance still be a locally consistent wing?

    A fast ball is smeared over tens of native pixels in one exposure and the detector
    row sits somewhere along that smear, so a quadratic through the row centres cannot
    be closer than the smear allows.  Each sample of the offending wing must therefore
    either sit inside the original allowance, or have its fitted point fall inside a
    streak measured *from the original pictures* at that exact sample, with the residual
    across the wing's own direction of travel still inside the original allowance.  A
    single unsupported sample leaves the original hold in place.
    """
    from cv.pipeline import event_wing_streak as streak

    allowance = CONFIG["maximum_wing_rms_native_px"]
    match_id = str(event.get("match_id") or "")
    clip = str(event.get("clip") or "")
    if match_id:
        clip = clip.removeprefix(f"{match_id}__")
    digest = streak.configuration_digest()
    report = {
        "schema": STREAK_FALLBACK_SCHEMA,
        "consulted": True,
        "qualified": False,
        "wing": None,
        "wing_rms_native_px": list(residuals),
        "allowance_native_px": allowance,
        "measurement_config_digest": digest,
        "samples": [],
        "measured_support_consulted": 0,
        "measured_support_passed": 0,
        "certifies_competitive_flight": False,
        "certifies_physical_ending": False,
        "reclassifies_event": False,
    }

    def reject(reason: str, entry: dict | None = None) -> dict:
        if entry is not None:
            entry["status"] = "abstained"
            entry["reason"] = reason
            report["samples"].append(entry)
        report["reason"] = reason
        return report

    if len(exceeded) != 1:
        return reject("more_than_one_wing_outside_original_allowance")
    if pictures is None or not match_id or not clip:
        return reject("native_picture_source_unavailable")
    wing = exceeded[0]
    report["wing"] = wing
    samples = spans[wing]
    coefficient = coefficients[wing]
    observed = np.asarray([track[t] for t in samples], dtype=float)
    offsets = np.asarray(samples, dtype=float) - frame
    fitted = np.polynomial.polynomial.polyval(offsets, coefficient).T
    for index, sample in enumerate(samples):
        delta = fitted[index] - observed[index]
        entry = {
            "frame": int(sample),
            "original_residual_native_px": float(np.linalg.norm(delta)),
        }
        if entry["original_residual_native_px"] <= allowance:
            entry["status"] = "within_original_allowance"
            report["samples"].append(entry)
            continue
        # The streak extends along the ball's direction of travel, so only the residual
        # across that direction is localization error.  It keeps the original allowance;
        # no measurement may widen it, and a long streak cannot excuse a sideways miss.
        velocity = coefficient[1] + 2 * offsets[index] * coefficient[2]
        speed = float(np.linalg.norm(velocity))
        if not math.isfinite(speed) or speed < streak.CONFIG["minimum_observed_motion_native_px"]:
            return reject("wing_velocity_too_small_to_define_a_direction", entry)
        normal = np.asarray([-velocity[1], velocity[0]], dtype=float) / speed
        entry["cross_track_residual_native_px"] = abs(float(np.dot(delta, normal)))
        if entry["cross_track_residual_native_px"] > allowance:
            return reject("cross_track_residual_exceeds_allowance", entry)
        report["measured_support_consulted"] += 1
        # Measured local motion from this wing's own rows, so the measurement can reject
        # a component whose axis disagrees with where the ball was actually going.
        motion = _measured_step(samples, observed, index)
        request = dict(
            match_id=match_id,
            clip=clip,
            frame=int(sample),
            point=observed[index],
            digest=digest,
            fps=fps,
            motion=motion,
        )
        support = pictures.measure(
            match_id,
            clip,
            int(sample),
            observed[index],
            fps=fps,
            observed_motion_native_px=motion,
        )
        if not _support_binds(support, **request):
            return reject("measured_support_not_bound_to_this_sample", entry)
        if not support.get("available"):
            return reject(str(support.get("reason")), entry)
        inside = streak.within_measured_support(fitted[index], support)
        entry["measured_length_native_px"] = inside.get("measured_length_native_px")
        if not inside.get("inside"):
            return reject(str(inside.get("reason")), entry)
        entry["status"] = "supported_by_measured_streak"
        entry["support_key"] = pictures.support_key(
            match_id,
            clip,
            int(sample),
            observed[index],
            digest,
            fps=fps,
            observed_motion_native_px=motion,
        )
        report["measured_support_passed"] += 1
        report["samples"].append(entry)
    report["qualified"] = True
    report["reason"] = "wing_samples_within_allowance_or_measured_streak"
    return report


def sampling_configuration(mode: str) -> dict:
    if mode not in ("consecutive", "available_native"):
        raise ValueError("unsupported native event wing sampling")
    return (
        CONFIG
        if mode == "consecutive"
        else {
            **CONFIG,
            "native_wing_sampling": mode,
            "duration_origin": "adjacent retained arc edge; closed native timestamp interval",
            "derived_observations": "excluded",
        }
    )


def certificate(
    event: dict,
    arcs: list[dict],
    track: dict[int, np.ndarray],
    *,
    native_wing_sampling: str = "consecutive",
    native_streak_fallback: bool = False,
    pictures=None,
) -> dict:
    sampling_configuration(native_wing_sampling)

    def held(reason, **extra):
        return {"supported": False, "reason": reason, **extra}

    if (
        event.get("model_abstain") is not False
        or event.get("gate_held") is not True
        or event.get("point_gate_verdict") != "retain"
        or event.get("point_gate_failure_reasons") != ["tracking_arc_abstained"]
        or event.get("event_type") not in CONFIG["event_types"]
    ):
        return held("not_exclusively_tracking_held_physical_event")
    identity = model_identity_admission(event)
    if identity is None:
        return held("original_decoder_admission_missing_or_inconsistent")
    frame = float(event["frame"])
    location = event.get("location") or {}
    fps = float(location.get("fps", 0))
    probability = float(event.get("probability", 0))
    pixel = np.asarray([location.get("image_x"), location.get("image_y")], dtype=float)
    if (
        not math.isfinite(frame)
        or not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
        or location.get("image_coordinate_space") != "native_1920x1080"
        or not np.isfinite(pixel).all()
    ):
        return held("missing_native_event_evidence")
    ordered = sorted(arcs, key=lambda arc: arc["start_frame"])
    matching = [
        i for i, arc in enumerate(ordered) if arc["start_frame"] <= frame <= arc["end_frame"]
    ]
    if len(matching) != 1:
        return held("missing_or_overlapping_arc")
    index = matching[0]
    impulse = ordered[index]
    length = impulse["end_frame"] - impulse["start_frame"] + 1
    if (
        impulse.get("regime") != "impulse"
        or impulse.get("decision") != "hold"
        or not 1 <= length <= max(1, math.floor(CONFIG["maximum_impulse_seconds"] * fps))
        or not impulse.get("failure_reasons")
        or not set(impulse["failure_reasons"]) <= ALLOWED_REASONS
        or not 0.94 <= impulse.get("candidate_support_rate", 0) <= 1.0
        or impulse.get("coverage_rate", 0) != 1.0
    ):
        return held("impulse_not_fully_observed_or_has_other_failure")
    if index == 0 or index + 1 == len(ordered):
        return held("missing_adjacent_wing")
    left, right = ordered[index - 1], ordered[index + 1]
    if (
        any(
            arc.get("decision") != "retain" or arc.get("regime") != "ballistic"
            for arc in (left, right)
        )
        or left["end_frame"] + 1 != impulse["start_frame"]
        or right["start_frame"] != impulse["end_frame"] + 1
    ):
        return held("adjacent_wing_not_retained_or_not_contiguous")
    count = max(CONFIG["minimum_wing_frames"], math.ceil(CONFIG["wing_seconds"] * fps))
    spans = [
        list(range(int(left["end_frame"]) - count + 1, int(left["end_frame"]) + 1)),
        list(range(int(right["start_frame"]), int(right["start_frame"]) + count)),
    ]
    if native_wing_sampling == "available_native":
        # A missing detector row is not an observation. Use the nearest actual
        # samples in the same retained arc and the existing duration, preserving
        # their original offsets in the quadratic fit below. Never cross the
        # impulse or borrow samples from a different arc.
        if left["end_frame"] not in track or right["start_frame"] not in track:
            return held("missing_native_wing_junction")
        radius = CONFIG["wing_seconds"] * fps
        before = [
            t
            for t in track
            if max(left["start_frame"], left["end_frame"] - radius) <= t <= left["end_frame"]
        ]
        after = [
            t
            for t in track
            if right["start_frame"] <= t <= min(right["end_frame"], right["start_frame"] + radius)
        ]
        spans = [sorted(before)[-count:], sorted(after)[:count]]
    coefficients, residuals = [], []
    for arc, samples in zip((left, right), spans, strict=True):
        if (
            len(samples) < count
            or samples[0] < arc["start_frame"]
            or samples[-1] > arc["end_frame"]
            or any(t not in track for t in samples)
        ):
            return held("insufficient_observed_wing")
        xy = np.asarray([track[t] for t in samples], dtype=float)
        if xy.shape != (count, 2) or not np.isfinite(xy).all():
            return held("invalid_wing_coordinates")
        times = np.asarray(samples, dtype=float) - frame
        coefficient = np.polynomial.polynomial.polyfit(times, xy, 2)
        predicted = np.polynomial.polynomial.polyval(times, coefficient).T
        residuals.append(float(np.sqrt(np.mean(np.sum((predicted - xy) ** 2, axis=1)))))
        coefficients.append(coefficient)
    streak_report = (
        {
            "schema": STREAK_FALLBACK_SCHEMA,
            "consulted": False,
            "qualified": True,
            "reason": "both_wings_within_original_allowance",
            "wing": None,
            "wing_rms_native_px": list(residuals),
            "allowance_native_px": CONFIG["maximum_wing_rms_native_px"],
            "samples": [],
            "measured_support_consulted": 0,
            "measured_support_passed": 0,
            "certifies_competitive_flight": False,
            "certifies_physical_ending": False,
        }
        if native_streak_fallback
        else None
    )
    exceeded = [
        wing for wing, value in enumerate(residuals) if value > CONFIG["maximum_wing_rms_native_px"]
    ]
    if exceeded:
        if not native_streak_fallback:
            return held("wing_not_locally_consistent")
        streak_report = _native_streak_wing(
            event, exceeded, spans, coefficients, residuals, track, frame, fps, pictures
        )
        if not streak_report["qualified"]:
            return held("wing_not_locally_consistent", native_streak_fallback=streak_report)
    # Search only a +/-1-frame event-time uncertainty interval; the emitted time
    # stays immutable. Cubic stationarity gives exact minima for quadratic wings.
    difference = coefficients[0] - coefficients[1]
    squared = np.polynomial.polynomial.polyadd(
        np.polynomial.polynomial.polymul(difference[:, 0], difference[:, 0]),
        np.polynomial.polynomial.polymul(difference[:, 1], difference[:, 1]),
    )
    predicted_epoch = float(location.get("frame_subpixel", frame))
    supplied = event.get("frame_interval")
    interval = (
        list(map(float, supplied))
        if supplied is not None
        else [
            predicted_epoch - CONFIG["event_time_radius_frames"],
            predicted_epoch + CONFIG["event_time_radius_frames"],
        ]
    )
    if (
        len(interval) != 2
        or not np.isfinite([predicted_epoch, *interval]).all()
        or not interval[0] <= predicted_epoch <= interval[1]
    ):
        return held("invalid_original_prediction_interval")
    lower = max(interval[0] - frame, impulse["start_frame"] - 1 - frame)
    upper = min(interval[1] - frame, impulse["end_frame"] + 1 - frame)
    if lower > upper:
        return held("prediction_interval_does_not_overlap_impulse")
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(squared))
    candidates = [
        lower,
        upper,
        *[float(t.real) for t in roots if abs(t.imag) < 1e-8 and lower <= t.real <= upper],
    ]
    offset = min(candidates, key=lambda t: np.polynomial.polynomial.polyval(t, squared))
    join = float(np.linalg.norm(np.polynomial.polynomial.polyval(offset, difference)))
    velocity_change = float(np.linalg.norm(difference[1] + 2 * offset * difference[2]) * fps)
    owner = coefficients[0 if offset >= 0 else 1]
    rounded_pixel_error = float(np.linalg.norm(owner[0] - pixel))
    # The XY and timing heads do not assert that the event pixel was observed
    # at the rounded emission frame. Check association anywhere in the original
    # interval, on the incoming/outgoing wing's own side of the supported join.
    pixel_trials = []
    for coefficient, lo, hi in ((coefficients[0], lower, offset), (coefficients[1], offset, upper)):
        delta = coefficient.copy()
        delta[0] -= pixel
        norm = np.polynomial.polynomial.polyadd(
            np.polynomial.polynomial.polymul(delta[:, 0], delta[:, 0]),
            np.polynomial.polynomial.polymul(delta[:, 1], delta[:, 1]),
        )
        roots_xy = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(norm))
        epochs = [
            lo,
            hi,
            *[float(t.real) for t in roots_xy if abs(t.imag) < 1e-8 and lo <= t.real <= hi],
        ]
        pixel_trials.extend(
            (
                float(np.linalg.norm(np.polynomial.polynomial.polyval(t, coefficient) - pixel)),
                float(t),
            )
            for t in epochs
        )
    pixel_error, pixel_offset = min(pixel_trials)
    pixel_wing = coefficients[0 if pixel_offset <= offset else 1]
    pixel_speed = float(np.linalg.norm(pixel_wing[1] + 2 * pixel_offset * pixel_wing[2]))
    supported = (
        join <= CONFIG["maximum_wing_join_native_px"]
        and pixel_error <= CONFIG["maximum_event_pixel_native_px"]
        and velocity_change >= CONFIG["minimum_velocity_change_native_px_per_second"]
    )
    return {
        "supported": supported,
        "reason": "two_wing_support" if supported else "wing_join_pixel_or_impulse_disagreement",
        "impulse_arc_id": impulse.get("arc_id"),
        "wing_arc_ids": [left.get("arc_id"), right.get("arc_id")],
        "wing_frames": spans,
        "wing_rms_native_px": residuals,
        "minimum_join_native_px": join,
        "join_offset_frames": float(offset),
        "join_argmin_on_boundary": bool(abs(offset - lower) < 1e-8 or abs(offset - upper) < 1e-8),
        "event_pixel_error_native_px": pixel_error,
        "rounded_event_pixel_error_native_px": rounded_pixel_error,
        "pixel_support_offset_frames": pixel_offset,
        "pixel_support_delta_from_predicted_epoch_frames": pixel_offset - (predicted_epoch - frame),
        "pixel_support_wing_speed_native_px_per_frame": pixel_speed,
        "localization_status": "interval_supported_only",
        "prediction_interval_frames": interval,
        "prediction_interval_origin": "producer_supplied"
        if supplied is not None
        else "one_frame_prediction_search_radius_not_calibrated_confidence",
        "predicted_epoch_unchanged": predicted_epoch,
        "certifies_physical_ending": False,
        "velocity_change_native_px_per_second": velocity_change,
        "event_frame_unchanged": frame,
        "model_identity_admission": identity,
        # Declared only when the fallback is enabled, so a default run's emission rows
        # carry no extra diagnostics and no extra keys.
        **({"native_streak_fallback": streak_report} if native_streak_fallback else {}),
    }


#: Serve contacts have a toss instead of an incoming wing, so ``certificate`` holds
#: them at ``insufficient_observed_wing`` even when the decoder is past its own
#: threshold. This is the outgoing-only alternative. It does not run unless a
#: consumer asks, and it never invents a contact: the row is already a decoder
#: emission. Measured on the three held serves (14 px, 57 px, 61 px from the
#: server box; the other player was >= 377 px away; outgoing quadratic RMS
#: 2.4-5.6 px). The allowances sit above those three and below the other player.
SERVE_RELEASE_KEY = "serve_outgoing_release"
SERVE_RELEASE_SCHEMA = "serve_outgoing_wing_release_v1"
SERVE_BOX_MARGIN_NATIVE_PX = 80.0
SERVE_WING_RMS_NATIVE_PX = 8.0
SERVE_MINIMUM_SPEED_NATIVE_PX_PER_FRAME = 2.0


def _box_distance(box: tuple[float, float, float, float], pixel: np.ndarray) -> float:
    x0, y0, x1, y1 = box
    x, y = float(pixel[0]), float(pixel[1])
    dx = 0.0 if x0 <= x <= x1 else min(abs(x - x0), abs(x - x1))
    dy = 0.0 if y0 <= y <= y1 else min(abs(y - y0), abs(y - y1))
    return float(math.hypot(dx, dy))


def serve_outgoing_support(
    event: dict,
    arcs: list[dict],
    track: dict[int, np.ndarray],
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    *,
    fps: float,
) -> dict:
    """Support for one tracking-held contact read as a serve.

    ``boxes`` are the player boxes at the contact frame, ``(side, (x0, y0, x1, y1))``
    in native pixels. The caller decides that this row is the point's first
    model-accepted contact; this function does not look at the rest of the stream.
    """

    def held(reason: str, **extra) -> dict:
        return {"schema": SERVE_RELEASE_SCHEMA, "supported": False, "reason": reason, **extra}

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("serve release needs a positive fps")
    if (
        event.get("model_abstain") is not False
        or event.get("gate_held") is not True
        or event.get("point_gate_verdict") != "retain"
        or event.get("point_gate_failure_reasons") != ["tracking_arc_abstained"]
        or event.get("event_type") != "contact"
    ):
        return held("not_exclusively_tracking_held_contact")
    identity = model_identity_admission(event)
    if identity is None:
        return held("original_decoder_admission_missing_or_inconsistent")
    location = event.get("location") or {}
    pixel = np.asarray([location.get("image_x"), location.get("image_y")], dtype=float)
    if location.get("image_coordinate_space") != "native_1920x1080" or not np.isfinite(pixel).all():
        return held("missing_native_event_pixel")
    frame = float(event["frame"])
    if not math.isfinite(frame):
        return held("missing_native_event_pixel")
    limit = frame + max(2.0, math.floor(CONFIG["maximum_impulse_seconds"] * fps) + 1.0)
    outgoing = [
        arc
        for arc in arcs
        if arc.get("decision") == "retain"
        and arc.get("regime") == "ballistic"
        and frame < float(arc["start_frame"]) <= limit
    ]
    if not outgoing:
        return held("no_outgoing_flight_arc")
    arc = min(outgoing, key=lambda item: float(item["start_frame"]))
    count = max(CONFIG["minimum_wing_frames"], math.ceil(CONFIG["wing_seconds"] * fps))
    samples = []
    cursor = int(math.ceil(float(arc["start_frame"])))
    end = int(math.floor(float(arc["end_frame"])))
    while cursor <= end and len(samples) < count:
        if cursor in track:
            samples.append(cursor)
        cursor += 1
    if len(samples) < CONFIG["minimum_wing_frames"]:
        return held("insufficient_outgoing_wing", outgoing_samples=len(samples))
    xy = np.asarray([track[t] for t in samples], dtype=float)
    times = np.asarray(samples, dtype=float) - frame
    coefficient = np.polynomial.polynomial.polyfit(times, xy, 2)
    predicted = np.polynomial.polynomial.polyval(times, coefficient).T
    rms = float(np.sqrt(np.mean(np.sum((predicted - xy) ** 2, axis=1))))
    speed = float(np.linalg.norm(coefficient[1]))
    if rms > SERVE_WING_RMS_NATIVE_PX or speed < SERVE_MINIMUM_SPEED_NATIVE_PX_PER_FRAME:
        return held(
            "outgoing_wing_not_flight_shaped",
            wing_rms_native_px=rms,
            wing_speed_native_px_per_frame=speed,
        )
    nearest = None
    for side, box in boxes:
        distance = _box_distance(box, pixel)
        if nearest is None or distance < nearest[0]:
            nearest = (distance, side)
    if nearest is None or nearest[0] > SERVE_BOX_MARGIN_NATIVE_PX:
        return held(
            "server_box_not_near_ball",
            box_distance_native_px=None if nearest is None else nearest[0],
        )
    return {
        "schema": SERVE_RELEASE_SCHEMA,
        "supported": True,
        "reason": "outgoing_wing_and_server_box",
        "model_identity_admission": identity,
        "outgoing_arc_id": arc.get("arc_id"),
        "outgoing_samples": samples,
        "wing_rms_native_px": rms,
        "wing_speed_native_px_per_frame": speed,
        "server_side": nearest[1],
        "box_distance_native_px": nearest[0],
        "box_margin_native_px": SERVE_BOX_MARGIN_NATIVE_PX,
        "certifies_physical_ending": False,
        "incoming_wing_required": False,
    }


def release_serve_contacts(
    rows: list[dict],
    *,
    match_id: str,
    clip: str,
    arcs: list[dict],
    track: dict[int, np.ndarray],
    boxes_by_frame: dict[int, list[tuple[str, tuple[float, float, float, float]]]],
    fps: float,
) -> tuple[list[dict], dict]:
    """Release the earliest model-accepted contact of this clip when it is a serve.

    Later tracking holds stay held. A row the live-shot trim has already re-held
    is not exclusively a tracking hold, so a close-up cannot take this path.
    """

    census = {"schema": SERVE_RELEASE_SCHEMA, "released": 0, "reason": None}

    def same(row: dict) -> bool:
        if row.get("match_id") != match_id:
            return False
        name = str(row.get("clip", ""))
        return name == clip or name.endswith("__" + clip)

    contacts = [
        row
        for row in rows
        if same(row) and row.get("event_type") == "contact" and row.get("model_abstain") is False
    ]
    if not contacts:
        census["reason"] = "no_model_accepted_contact"
        return rows, census
    first = min(contacts, key=lambda row: float(row["frame"]))
    output = []
    for source in rows:
        if source is not first:
            output.append(source)
            continue
        frame = int(round(float(source["frame"])))
        evidence = serve_outgoing_support(
            source,
            arcs,
            track,
            boxes_by_frame.get(frame, []) + boxes_by_frame.get(frame - 1, []),
            fps=fps,
        )
        if not evidence["supported"]:
            census["reason"] = evidence["reason"]
            output.append(source)
            continue
        output.append(
            {
                **source,
                "abstain": False,
                "gate_held": False,
                "point_gate_verdict": "retain",
                "point_gate_failure_reasons": [],
                "serve_outgoing_release": evidence,
            }
        )
        census["released"] = 1
        census["reason"] = evidence["reason"]
        census["frame"] = source.get("frame")
    return output, census
