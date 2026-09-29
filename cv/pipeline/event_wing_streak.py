"""Independently measured per-frame native streak support for one observed ball pixel.

Why this exists
---------------
``event_impulse_support`` certifies a tracking-held contact/bounce from two quadratic
wings and holds the claim when either wing's fit residual exceeds 3 native px.  On a
genuinely fast ball that test can hold a true contact: the ball is *visibly* smeared
over tens of native pixels in each exposure, the detector row sits somewhere along
that smear, and a quadratic through the row centres cannot be closer than the smear
allows.  The residual is then large because the *observation* has an extent, not
because the wing is inconsistent.

This module measures that extent directly from the original pictures.  It answers one
question and nothing else:

    *Does the current native picture contain a single compact moving component at this
    observed pixel, and if so, what is its measured axis, extent and transverse width?*

What it deliberately does not do
--------------------------------
* **No corrected ball position and no event time are emitted.**  ``centroid_native``
  and ``endpoints_native`` describe the *measured component*, not a better estimate of
  where the ball was; nothing here may be consumed as an observation row or an epoch.
* **No tolerance is derived from a residual or a fit.**  The extent is measured from
  pixels.  A caller must not feed a residual, an RMS or a fitted trajectory in.
* **No labels, reference fits or per-source tuning.**  The thresholds below come from
  ball geometry at 1920x1080 and from 8-bit compression noise, and are the same for
  every source.  There is no threshold on a source number or a frame number.
* **No camera model and no optical flow.**  Background motion is *detected* and
  abstained on: the patch rim must be static, the two neighbours must agree about what
  lies behind the component, and the component's surround must be a single background
  rather than a seam between two.  A panning camera is never compensated and its
  difference residual is never reinterpreted as ball blur.  A pan is only *invisible*
  here when it disturbs no rim pixel, leaves the two neighbours agreeing, and shifts a
  structure whose surround is uniform; that combination is not defended against and is
  the first thing a native replay should look for.
* **No new model.**  Segmentation is three-frame differencing plus connected
  components, both already ordinary in this repository.
* **No whiteness rule.**  A ball is yellow on some sources and reads white or cyan on
  others, and a white court line is not a ball.  Admission uses *luminance contrast of
  either polarity* against the component's own local surround, never "it is white".

Relationship to ``subframe_timing.measure_streak``
--------------------------------------------------
That function unions *every* moving pixel in the patch and measures the union's
principal-axis spread.  Next to a player it therefore measures the player.  Nothing
here reuses it: the component must be connected, anchored on the observed pixel,
unambiguous, unclipped, and of a single compact transverse width.

Availability contract
---------------------
``available`` false means *no extra tolerance is available* for this frame.  It is
never a rejection of the event: the caller keeps its original 3 px path unchanged and
must OR this evidence in, never AND it.  See :func:`within_measured_support`.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from cv.pipeline.resolution import FrameSize, meets_native_tracking_minimum, points_inside_image

SCHEMA = "event_wing_streak_v1"

# The fixed per-row native localization allowance the existing two-wing test already
# grants (the same 3 px as ``event_impulse_support.CONFIG['maximum_wing_rms_native_px']``).
# It enters here as a geometric margin in pixels; it is not an RMS statistic, not a
# fitted residual and not a calibrated sigma.  The duplication is deliberate: this
# module must not start tracking a residual threshold if that one is ever retuned.
LOCALIZATION_ALLOWANCE_NATIVE_PX = 3.0

CONFIG = {
    # --- exposure identity -------------------------------------------------------
    "required_neighbour_count": 2,
    # Scene-stationarity assumption, not a tuned value: background sampled from
    # exposures further away than this is not contemporaneous with the current one.
    "maximum_neighbour_separation_seconds": 0.25,
    # --- patch admissibility -----------------------------------------------------
    "required_native_px_per_patch_px": 1.0,
    "minimum_patch_side_native_px": 24,
    "patch_rim_native_px": 2,
    # --- motion support ----------------------------------------------------------
    # 8-bit levels.  JPEG/compression noise on flat broadcast regions is ~3-4 levels,
    # so 10 is a few times noise; the robust term lifts it on textured neighbourhoods.
    "minimum_motion_levels": 10.0,
    "motion_noise_multiple": 6.0,
    "elevated_motion_multiple": 1.5,
    "minimum_length_stability_fraction": 0.5,
    "maximum_rim_motion_fraction": 0.05,
    "minimum_rim_motion_pixels": 8,
    "maximum_neighbour_disagreement_levels": 10.0,
    # --- association -------------------------------------------------------------
    "maximum_anchor_offset_native_px": LOCALIZATION_ALLOWANCE_NATIVE_PX,
    # --- ball geometry at 1920x1080 ----------------------------------------------
    # A ball is 6.7 cm.  On the elevated baseline broadcast view a near-court ball is
    # ~9 native px across and a far-court ball ~4 px, and at 25 fps a fast ball smears
    # over 25-45 px.  So a ball component covers ~10 px^2 (far, slow) to ~900 px^2
    # (near, fast), with a transverse width under ~22 px and a length under ~130 px.
    # A ball is also never a one-pixel line, which is what a court-line or banner edge
    # ghost looks like.  These are the same geometry-derived bounds the native streak
    # candidate diagnosis uses; none of them was fitted to a source.
    "minimum_component_pixels": 10,
    "maximum_component_pixels": 900,
    "minimum_transverse_width_native_px": 2.0,
    "maximum_transverse_width_native_px": 22.0,
    "maximum_length_native_px": 130.0,
    "minimum_fill_fraction": 0.35,
    "maximum_width_variation_ratio": 2.5,
    "minimum_single_run_slice_fraction": 0.9,
    # Tolerates diagonal rasterization inside one axis slice; two runs separated by
    # more than this are two objects, not one width.
    "maximum_transverse_gap_native_px": 1.5,
    # --- current-frame appearance ------------------------------------------------
    "contrast_ring_native_px": 3,
    "minimum_contrast_ring_pixels": 10,
    "minimum_contrast_levels": 10.0,
    "minimum_polarity_fraction": 0.75,
    # The surround must be one background: nearly all of the ring within the same 8-bit
    # noise quantum of its own median.  A ring that is part one background and part
    # another is a seam, and a seam displaced by a camera move imitates an object.
    "surround_same_background_band_levels": 10.0,
    "minimum_uniform_surround_fraction": 0.9,
    # --- optional direction compatibility ----------------------------------------
    "minimum_directional_elongation": 1.5,
    "minimum_observed_motion_native_px": 2.0,
    # 30 degrees.  A streak is direction-symmetric, so the test is on |cos|.
    "minimum_axis_alignment_cosine": 0.866,
    # --- contract ----------------------------------------------------------------
    "emits_corrected_ball_position": False,
    "emits_event_time": False,
    "colour_admission": "luminance_contrast_either_polarity_no_whiteness_rule",
    "camera_motion_policy": "detect_and_abstain_never_compensate",
}

REASONS = (
    "measured_streak_support",
    "requires_exactly_two_native_neighbours",
    "unsupported_or_inconsistent_patch_images",
    "patch_intensities_outside_8bit_range",
    "patch_smaller_than_measurable_neighbourhood",
    "patch_not_at_native_scale",
    "source_below_native_tracking_resolution",
    "patch_outside_declared_native_image",
    "duplicate_or_nonunique_native_exposures",
    "nonmonotonic_native_timestamps",
    "neighbours_do_not_straddle_current_exposure",
    "neighbour_exposures_not_contemporaneous",
    "duplicate_native_pictures",
    "observed_point_outside_native_image",
    "observed_point_outside_patch",
    "unsupported_background_motion_at_patch_rim",
    "no_moving_component_at_observed_point",
    "component_clipped_by_patch_edge",
    "component_outside_ball_area_geometry",
    "component_width_outside_ball_geometry",
    "component_longer_than_ball_streak_geometry",
    "component_not_a_single_compact_width",
    "ambiguous_competing_component",
    "neighbour_backgrounds_disagree_behind_component",
    "insufficient_local_background_for_contrast",
    "insufficient_current_frame_contrast",
    "mixed_current_frame_contrast_polarity",
    "component_surround_not_uniform_enough_to_isolate_it",
    "measured_extent_unstable_under_threshold",
    "axis_incompatible_with_observed_motion",
)

# Abstentions that say "the background here is not one static plain background", which
# is the only answer this module has to camera motion and to moving background objects.
# Neither is compensated and neither may be reinterpreted as ball blur.  The three are
# independent: the rim witnesses a global move, the neighbours must agree about what is
# behind the component, and the component's own surround must be a single background.
UNSUPPORTED_BACKGROUND_REASONS = (
    "unsupported_background_motion_at_patch_rim",
    "neighbour_backgrounds_disagree_behind_component",
    "component_surround_not_uniform_enough_to_isolate_it",
)


@dataclass(frozen=True)
class NativeFrame:
    """One original native picture with its own exposure identity.

    ``image`` is native RGB (H, W, 3) or grayscale (H, W) at native scale, cut at the
    same crop coordinates as every other frame in the call.  ``frame_index`` and
    ``native_time_seconds`` are the source's own, and are recorded verbatim.
    """

    image: np.ndarray
    frame_index: int
    native_time_seconds: float


@dataclass(frozen=True)
class PatchGeometry:
    """Where the shared crop sits in the original native picture.

    ``native_px_per_patch_px`` must be exactly 1: an upsampled patch would inflate
    every measured extent below, and this module cannot detect that from pixels alone,
    so the caller declares it and a declaration other than 1 abstains.
    """

    origin_native: tuple[int, int]
    image_size: FrameSize
    native_px_per_patch_px: float = 1.0


def _luminance(image) -> np.ndarray | None:
    """8-bit luma of an RGB or grayscale native patch, as float.

    RGB channel order is the caller's declared convention; BT.601 weights are applied
    in that order.  Colour is used for nothing else -- see ``colour_admission``.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(float)
    if array.ndim == 3 and array.shape[2] == 3:
        weights = np.asarray([0.299, 0.587, 0.114])
        return array.astype(float) @ weights
    return None


def _runs(values: np.ndarray, tolerance: float) -> int:
    """Number of maximal groups in ``values`` separated by gaps above ``tolerance``."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return 0
    return 1 + int((np.diff(ordered) > tolerance).sum())


def _slice_widths(projection: np.ndarray, transverse: np.ndarray) -> tuple[list[float], int, int]:
    """Per-slice transverse extent along the major axis, and how many slices are one run."""
    slices = np.rint(projection).astype(int)
    widths, single = [], 0
    for index in np.unique(slices):
        band = transverse[slices == index]
        widths.append(float(band.max() - band.min()) + 1.0)
        single += int(_runs(band, CONFIG["maximum_transverse_gap_native_px"]) == 1)
    return widths, single, len(widths)


def _frame_record(frame: NativeFrame) -> dict:
    return {
        "frame_index": int(frame.frame_index),
        "native_time_seconds": float(frame.native_time_seconds),
    }


def configuration_digest() -> str:
    """Identity of the declared source-image measurement settings."""
    return hashlib.sha256(
        json.dumps(CONFIG, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def measure_support(
    current: NativeFrame,
    neighbours: Sequence[NativeFrame],
    observed_point_native,
    geometry: PatchGeometry,
    *,
    observed_motion_native_px=None,
) -> dict:
    """Measure the moving component at ``observed_point_native`` in ``current``.

    ``neighbours`` are two unique other native exposures cut at the same crop
    coordinates; they supply the background and nothing else.  ``observed_point_native``
    is a *measured* ball pixel in native coordinates -- a detector/tracker row, never an
    interpolated, fitted or label-derived point.  ``observed_motion_native_px`` is an
    optional measured local displacement ``(dx, dy)`` in native px per frame, taken from
    the same measured rows; when it is supplied and the component is elongated enough to
    have a determined axis, the component's axis must agree with it.  Supplying it is the
    only way a long component on the wrong axis is rejected, so callers that can compute
    it from measured rows should.

    Returns a record that always carries ``available``, ``reason`` and ``source``.  When
    ``available`` is true, ``measurement`` holds the measured axis, endpoints, centroid,
    length and transverse width in native coordinates, together with the threshold
    sensitivity of the extent.  When it is false there is no measured support and the
    caller keeps its original behaviour unchanged.
    """
    record = {
        "schema": SCHEMA,
        "config_digest": configuration_digest(),
        "available": False,
        "reason": "unsupported_or_inconsistent_patch_images",
        "source": {
            "current": _frame_record(current),
            "neighbours": [_frame_record(frame) for frame in neighbours],
            "patch_origin_native": [int(value) for value in geometry.origin_native],
            "native_image_size": [geometry.image_size.width, geometry.image_size.height],
            "native_px_per_patch_px": float(geometry.native_px_per_patch_px),
            "observed_point_native": [
                float(value) for value in np.ravel(observed_point_native)[:2]
            ],
            "observed_motion_native_px": (
                None
                if observed_motion_native_px is None
                else [float(value) for value in np.ravel(observed_motion_native_px)[:2]]
            ),
        },
        "measurement": None,
        "diagnostics": {},
    }

    def held(reason: str, **diagnostics) -> dict:
        record["reason"] = reason
        record["diagnostics"].update(diagnostics)
        return record

    frames = [current, *neighbours]
    if len(neighbours) != CONFIG["required_neighbour_count"]:
        return held("requires_exactly_two_native_neighbours")

    luminance = [_luminance(frame.image) for frame in frames]
    if any(plane is None for plane in luminance):
        return held("unsupported_or_inconsistent_patch_images")
    if any(plane.shape != luminance[0].shape for plane in luminance):
        return held("unsupported_or_inconsistent_patch_images")
    if any(
        not np.isfinite(plane).all() or plane.min() < 0 or plane.max() > 255 for plane in luminance
    ):
        return held("patch_intensities_outside_8bit_range")
    height, width = luminance[0].shape
    if min(height, width) < CONFIG["minimum_patch_side_native_px"]:
        return held("patch_smaller_than_measurable_neighbourhood", patch_shape=[height, width])

    if float(geometry.native_px_per_patch_px) != CONFIG["required_native_px_per_patch_px"]:
        return held("patch_not_at_native_scale")
    if not meets_native_tracking_minimum(geometry.image_size):
        return held("source_below_native_tracking_resolution")
    origin_x, origin_y = (int(value) for value in geometry.origin_native)
    if (
        origin_x < 0
        or origin_y < 0
        or origin_x + width > geometry.image_size.width
        or origin_y + height > geometry.image_size.height
    ):
        return held("patch_outside_declared_native_image")

    indices = [int(frame.frame_index) for frame in frames]
    times = [float(frame.native_time_seconds) for frame in frames]
    if len(set(indices)) != len(indices):
        return held("duplicate_or_nonunique_native_exposures", frame_indices=indices)
    if not np.isfinite(times).all():
        return held("nonmonotonic_native_timestamps", native_times=times)
    order = np.argsort(indices)
    ordered_times = [times[position] for position in order]
    if any(b <= a for a, b in zip(ordered_times, ordered_times[1:])):
        return held("nonmonotonic_native_timestamps", native_times=times)
    if not min(indices[1:]) < indices[0] < max(indices[1:]):
        return held("neighbours_do_not_straddle_current_exposure")
    separation = max(abs(time - times[0]) for time in times[1:])
    if separation > CONFIG["maximum_neighbour_separation_seconds"]:
        return held("neighbour_exposures_not_contemporaneous", separation_seconds=separation)
    for first in range(len(frames)):
        for second in range(first + 1, len(frames)):
            if np.array_equal(luminance[first], luminance[second]):
                return held("duplicate_native_pictures")

    point = np.asarray(np.ravel(observed_point_native)[:2], dtype=float)
    if not points_inside_image(point, geometry.image_size).all():
        return held("observed_point_outside_native_image")
    patch_point = point - np.asarray([origin_x, origin_y], dtype=float)
    if not (0.0 <= patch_point[0] < width and 0.0 <= patch_point[1] < height):
        return held("observed_point_outside_patch")

    # Three-frame support: a pixel is moving only if it differs from *both* neighbours.
    # A ball that has not left its own position by the neighbour exposure produces no
    # motion here and abstains, which is the conservative direction.
    motion = np.minimum(np.abs(luminance[0] - luminance[1]), np.abs(luminance[0] - luminance[2]))
    flat = motion.reshape(-1)
    median = float(np.median(flat))
    deviation = float(np.median(np.abs(flat - median)))
    threshold = max(
        CONFIG["minimum_motion_levels"], median + CONFIG["motion_noise_multiple"] * deviation
    )
    mask = (motion >= threshold).astype(np.uint8)
    record["diagnostics"]["motion_threshold_levels"] = threshold

    # The rim is this patch's background sample.  A panning camera, a cut or a global
    # exposure change moves it; none of those is modelled here, so they abstain.
    rim = CONFIG["patch_rim_native_px"]
    border = np.zeros_like(mask, dtype=bool)
    border[:rim, :] = border[-rim:, :] = border[:, :rim] = border[:, -rim:] = True
    rim_moving = int(mask[border].sum())
    rim_total = int(border.sum())
    record["diagnostics"]["rim_motion_fraction"] = rim_moving / max(rim_total, 1)
    if rim_moving >= max(
        CONFIG["minimum_rim_motion_pixels"], CONFIG["maximum_rim_motion_fraction"] * rim_total
    ):
        return held("unsupported_background_motion_at_patch_rim", rim_moving_pixels=rim_moving)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return held("no_moving_component_at_observed_point")
    rows, columns = np.nonzero(mask)
    centres = np.stack([columns + 0.5, rows + 0.5], axis=1)
    distances = np.linalg.norm(centres - patch_point, axis=1)
    tags = labels[rows, columns]
    nearest = {int(tag): float(distances[tags == tag].min()) for tag in np.unique(tags)}
    anchor = min(nearest, key=lambda tag: nearest[tag])
    offset = nearest[anchor]
    record["diagnostics"]["anchor_offset_native_px"] = offset
    if offset > CONFIG["maximum_anchor_offset_native_px"]:
        return held("no_moving_component_at_observed_point", anchor_offset_native_px=offset)

    selected = tags == anchor
    pixels = centres[selected]
    area = int(pixels.shape[0])
    record["diagnostics"]["component_pixels"] = area
    left = int(stats[anchor, cv2.CC_STAT_LEFT])
    top = int(stats[anchor, cv2.CC_STAT_TOP])
    if (
        left == 0
        or top == 0
        or left + int(stats[anchor, cv2.CC_STAT_WIDTH]) >= width
        or top + int(stats[anchor, cv2.CC_STAT_HEIGHT]) >= height
    ):
        # The patch edge truncates the object, so its extent is a lower bound only.
        # Whether that edge is also the native picture edge is recorded, not excused.
        return held(
            "component_clipped_by_patch_edge",
            clipped_at_native_image_edge=bool(
                (left == 0 and origin_x == 0)
                or (top == 0 and origin_y == 0)
                or (
                    left + int(stats[anchor, cv2.CC_STAT_WIDTH]) >= width
                    and origin_x + width >= geometry.image_size.width
                )
                or (
                    top + int(stats[anchor, cv2.CC_STAT_HEIGHT]) >= height
                    and origin_y + height >= geometry.image_size.height
                )
            ),
        )
    if not CONFIG["minimum_component_pixels"] <= area <= CONFIG["maximum_component_pixels"]:
        return held("component_outside_ball_area_geometry")

    centroid = pixels.mean(axis=0)
    centred = pixels - centroid
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    axis = basis[0] / float(np.linalg.norm(basis[0]))
    normal = np.asarray([-axis[1], axis[0]])
    projection = centred @ axis
    transverse = centred @ normal
    length = float(projection.max() - projection.min()) + 1.0
    widths, single_run, slice_count = _slice_widths(projection, transverse)
    widest = float(max(widths))
    typical = float(np.median(widths))
    fill = area / max(length * widest, 1e-9)
    record["diagnostics"].update(
        length_native_px=length,
        transverse_width_native_px=typical,
        maximum_transverse_width_native_px=widest,
        fill_fraction=fill,
    )
    if (
        not (
            CONFIG["minimum_transverse_width_native_px"]
            <= typical
            <= CONFIG["maximum_transverse_width_native_px"]
        )
        or widest > CONFIG["maximum_transverse_width_native_px"]
    ):
        return held("component_width_outside_ball_geometry")
    if length > CONFIG["maximum_length_native_px"]:
        return held("component_longer_than_ball_streak_geometry")
    # A ball streak has one width all the way along itself.  A limb, a racket arc or a
    # ball merged into either does not: its slices are wider, uneven, or split in two.
    if (
        fill < CONFIG["minimum_fill_fraction"]
        or widest > CONFIG["maximum_width_variation_ratio"] * max(typical, 1e-9)
        or single_run < CONFIG["minimum_single_run_slice_fraction"] * max(slice_count, 1)
    ):
        return held(
            "component_not_a_single_compact_width",
            single_run_slice_fraction=single_run / max(slice_count, 1),
        )

    # Another component that could itself be the ball, close enough that this pixel
    # cannot be attributed to one of them, makes the measurement ambiguous.  A player
    # or racket blob is not a competitor: it fails the same ball geometry above.
    competing = []
    radius = length + CONFIG["maximum_anchor_offset_native_px"]
    for tag, distance in nearest.items():
        if tag == anchor or distance > radius:
            continue
        rival = int((tags == tag).sum())
        if not CONFIG["minimum_component_pixels"] <= rival <= CONFIG["maximum_component_pixels"]:
            continue
        rival_pixels = centres[tags == tag]
        rival_centred = rival_pixels - rival_pixels.mean(axis=0)
        _, _, rival_basis = np.linalg.svd(rival_centred, full_matrices=False)
        rival_axis = rival_basis[0] / float(np.linalg.norm(rival_basis[0]))
        rival_projection = rival_centred @ rival_axis
        rival_widths, _, _ = _slice_widths(
            rival_projection, rival_centred @ np.asarray([-rival_axis[1], rival_axis[0]])
        )
        rival_length = float(rival_projection.max() - rival_projection.min()) + 1.0
        if (
            rival_length <= CONFIG["maximum_length_native_px"]
            and max(rival_widths) <= CONFIG["maximum_transverse_width_native_px"]
            and float(np.median(rival_widths)) >= CONFIG["minimum_transverse_width_native_px"]
        ):
            competing.append({"pixels": rival, "distance_native_px": distance})
    if competing:
        return held("ambiguous_competing_component", competing_components=competing)

    # The two neighbours must agree about what lies behind the component.  They do when
    # the background is static; they do not under a pan, a zoom or a cut, and that
    # disagreement must never be read as ball blur.
    disagreement = float(
        np.median(np.abs(luminance[1] - luminance[2])[rows[selected], columns[selected]])
    )
    record["diagnostics"]["neighbour_background_disagreement_levels"] = disagreement
    if disagreement > CONFIG["maximum_neighbour_disagreement_levels"]:
        return held("neighbour_backgrounds_disagree_behind_component")

    # Current-frame appearance: the component must stand out in the picture it was
    # measured in, against its own immediate surround, with one consistent polarity.
    # Either polarity is admissible; nothing is accepted for being white or yellow.
    span = 2 * CONFIG["contrast_ring_native_px"] + 1
    component_mask = np.zeros_like(mask, dtype=np.uint8)
    component_mask[rows[selected], columns[selected]] = 1
    grown = cv2.dilate(component_mask, np.ones((span, span), np.uint8))
    ring = (grown > 0) & (mask == 0)
    if int(ring.sum()) < CONFIG["minimum_contrast_ring_pixels"]:
        return held("insufficient_local_background_for_contrast")
    background = float(np.median(luminance[0][ring]))
    values = luminance[0][rows[selected], columns[selected]]
    contrast = float(values.mean() - background)
    polarity = float(np.mean(np.sign(values - background) == math.copysign(1.0, contrast)))
    deviation_from_background = np.abs(luminance[0][ring] - background)
    uniform = float(
        np.mean(deviation_from_background <= CONFIG["surround_same_background_band_levels"])
    )
    record["diagnostics"].update(
        current_frame_contrast_levels=contrast,
        contrast_polarity_fraction=polarity,
        uniform_surround_fraction=uniform,
    )
    if abs(contrast) < CONFIG["minimum_contrast_levels"]:
        return held("insufficient_current_frame_contrast")
    if polarity < CONFIG["minimum_polarity_fraction"]:
        return held("mixed_current_frame_contrast_polarity")
    # The surround must be one background, not two.  A static edge or line displaced by a
    # camera move leaves a moving sliver of pure difference that has a ball's area, width,
    # length and compactness, and whose neighbours agree about what is behind it; what it
    # cannot have is a uniform surround, because it is the seam between two backgrounds.
    # This also abstains on a ball crossing a painted line or a shadow boundary, which is
    # a coverage cost rather than a wrong measurement.
    if uniform < CONFIG["minimum_uniform_surround_fraction"]:
        return held("component_surround_not_uniform_enough_to_isolate_it")

    # How much of the measured length survives a strictly higher motion threshold.  This
    # is a sensitivity band on the measurement, not a calibrated uncertainty.
    elevated = motion[rows[selected], columns[selected]] >= (
        threshold * CONFIG["elevated_motion_multiple"]
    )
    if int(elevated.sum()) >= 2:
        kept = projection[elevated]
        elevated_length = float(kept.max() - kept.min()) + 1.0
    else:
        elevated_length = 0.0
    stability = elevated_length / max(length, 1e-9)
    record["diagnostics"].update(
        length_at_elevated_threshold_native_px=elevated_length, length_stability_fraction=stability
    )
    if stability < CONFIG["minimum_length_stability_fraction"]:
        return held("measured_extent_unstable_under_threshold")

    elongation = length / max(typical, 1e-9)
    determined = elongation >= CONFIG["minimum_directional_elongation"]
    alignment, gate = None, "not_supplied"
    if observed_motion_native_px is not None:
        motion_vector = np.asarray(np.ravel(observed_motion_native_px)[:2], dtype=float)
        speed = float(np.linalg.norm(motion_vector))
        if (
            not np.isfinite(motion_vector).all()
            or speed < CONFIG["minimum_observed_motion_native_px"]
        ):
            gate = "observed_motion_too_small_to_define_a_direction"
        elif not determined:
            gate = "component_isotropic_axis_not_determined"
        else:
            alignment = abs(float(np.dot(axis, motion_vector / speed)))
            gate = "applied"
            record["diagnostics"]["axis_alignment_cosine"] = alignment
            if alignment < CONFIG["minimum_axis_alignment_cosine"]:
                return held("axis_incompatible_with_observed_motion")

    record["diagnostics"]["length_threshold_sensitivity_native_px"] = max(
        0.0, length - elevated_length
    )
    origin = np.asarray([origin_x, origin_y], dtype=float)
    centre_native = centroid + origin
    start = centre_native + (float(projection.min()) - 0.5) * axis
    end = centre_native + (float(projection.max()) + 0.5) * axis
    record.update(
        available=True,
        reason="measured_streak_support",
        measurement={
            "axis_native": [float(axis[0]), float(axis[1])],
            "axis_determined": bool(determined),
            "endpoints_native": [
                [float(start[0]), float(start[1])],
                [float(end[0]), float(end[1])],
            ],
            "centroid_native": [float(centre_native[0]), float(centre_native[1])],
            "length_native_px": length,
            "transverse_width_native_px": typical,
            "maximum_transverse_width_native_px": widest,
            "component_pixels": area,
            "fill_fraction": fill,
            "elongation": elongation,
            "anchor_offset_native_px": offset,
            "motion_threshold_levels": threshold,
            "length_at_elevated_threshold_native_px": elevated_length,
            "length_stability_fraction": stability,
            "current_frame_contrast_levels": contrast,
            "contrast_polarity": "brighter" if contrast >= 0 else "darker",
            "contrast_polarity_fraction": polarity,
            "neighbour_background_disagreement_levels": disagreement,
            "axis_alignment_cosine": alignment,
            "direction_gate": gate,
            "measured_from": "native_pictures_only",
            "is_corrected_ball_position": False,
            "is_event_time": False,
        },
    )
    return record


def within_measured_support(
    point_native,
    support: dict,
    *,
    allowance_native_px: float = LOCALIZATION_ALLOWANCE_NATIVE_PX,
) -> dict:
    """Is a predicted image point inside the measured streak, widened by the 3 px allowance?

    Erode the measured extent by its transverse diameter to obtain the centre's
    possible path, then apply a fixed-radius capsule around that segment. Ball
    diameter is not extra localization uncertainty. A round component reduces
    to a radius-3-pixel disk; long streaks grant no transverse widening.

    This is a geometric extent interval, not a calibrated sigma and not a fitted
    allowance.  ``inside`` false is not a rejection of anything: a caller must OR this
    with its original 3 px test, never AND it, so that an unavailable or failing
    measurement leaves the original path exactly as it was.
    """
    result = {
        "schema": SCHEMA,
        "inside": False,
        "reason": "support_unavailable",
        "allowance_native_px": float(allowance_native_px),
        "basis": "diameter_eroded_centre_path_plus_fixed_localization_capsule",
        "combine_with_original_test": "or_never_and",
    }
    measurement = support.get("measurement") if support else None
    if not support or not support.get("available") or not measurement:
        return result
    point = np.asarray(np.ravel(point_native)[:2], dtype=float)
    start, end = (np.asarray(value, dtype=float) for value in measurement["endpoints_native"])
    span = end - start
    extent = float(np.linalg.norm(span))
    if not np.isfinite(point).all() or not np.isfinite(span).all() or extent <= 0:
        result["reason"] = "degenerate_measured_extent"
        return result
    axis = span / extent
    normal = np.asarray([-axis[1], axis[0]])
    offset = point - 0.5 * (start + end)
    longitudinal = abs(float(np.dot(offset, axis)))
    cross = abs(float(np.dot(offset, normal)))
    width = float(measurement["transverse_width_native_px"])
    allowance = float(allowance_native_px)
    if not math.isfinite(width) or width <= 0 or not math.isfinite(allowance) or allowance < 0:
        result["reason"] = "invalid_measured_width_or_allowance"
        return result
    half_path = 0.5 * max(0.0, extent - width)
    longitudinal_limit = half_path + allowance
    cross_limit = allowance
    distance = math.hypot(max(0.0, longitudinal - half_path), cross)
    inside = distance <= allowance
    result.update(
        inside=bool(inside),
        reason="inside_measured_extent" if inside else "outside_measured_extent",
        longitudinal_native_px=longitudinal,
        transverse_native_px=cross,
        longitudinal_limit_native_px=longitudinal_limit,
        transverse_limit_native_px=cross_limit,
        measured_length_native_px=extent,
        centre_path_length_native_px=2 * half_path,
        distance_to_centre_path_native_px=distance,
        measured_transverse_width_native_px=float(measurement["transverse_width_native_px"]),
    )
    return result


__all__ = [
    "CONFIG",
    "LOCALIZATION_ALLOWANCE_NATIVE_PX",
    "REASONS",
    "SCHEMA",
    "UNSUPPORTED_BACKGROUND_REASONS",
    "NativeFrame",
    "PatchGeometry",
    "measure_support",
    "configuration_digest",
    "within_measured_support",
]
