"""Label-free dense temporal features for the S5 event models.

Coordinate contract.  The ball track's pixel columns are read through the
coordinate sidecar's declared column map, so every channel is built from the
frame size those numbers are actually in:

* a sidecar that declares ``native_1920x1080`` columns is read there, at the
  sidecar's authoritative ``artifact_size``;
* a sidecar that declares ``legacy_960x540`` columns is read there, at the
  sidecar's ``legacy_artifact_size``;
* a sidecar with no column map keeps the historical rule, ``x``/``y`` at
  ``artifact_size``.

This is the same resolution :func:`cv.pipeline.event_crops.track_coordinate_space`
performs for the crop cutter.  Until 2026-09-04 this module instead read ``x``/``y``
unconditionally while taking the frame size from ``artifact_size``, so on a
dual-column cohort (``cohort_root_v2``/``v3``) half-resolution pixels were pushed
through a native-resolution homography and normalised by a native frame width.
``court_x``/``court_y`` were therefore not court fractions at all -- on
``cohort_root_v3`` ``court_y`` ran from -610 to +320 with a median of 1.489 and
98.22% of rows above 0.5 -- and ``image_x``/``image_y`` and their derivatives were
half their true value.  ``docs/wk1/net_line.md`` traced the defect;
``docs/wk1/WK3_REPORT`` for package ``eventnative`` fixes it.

* :func:`track_pixel_space` resolves the columns and the frame size they live in;
* :func:`track_column_size` returns just that frame size;
* :func:`feature_court_xy` reproduces the builder's projection for any pixel;
* :func:`net_geometry_in_feature_frame` pushes the net's *own* pixels through
  that same projection, so the net line a downstream consumer needs is derived
  from the court geometry rather than assumed.

With the contract honoured the net's ground line lands on ``NET_Y_M /
COURT_LENGTH_M`` = 0.5 exactly, which
``test_event_model_v2_features.py`` asserts.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv.pipeline import resolution
from cv.pipeline.automatic_ball_track import (
    LEGACY_EVENT_TRACK,
    LEGACY_TRACK_NAME,
    CurrentTrackBinding,
    bind_current_track,
    validate_event_track,
)
from cv.pipeline.automatic_ball_track import track_name as event_track_name

RADIUS = 12
WINDOW_SIZE = 2 * RADIUS + 1
#: The default (``event_track="legacy"``) ball track. ``event_track="s4_current"``
#: selects the composed S4 track instead; see :func:`bind_broadcast_track`.
TRACK_NAME = LEGACY_TRACK_NAME
LEGACY_FEATURE_SCHEMA = "event_model_v2_corrected_physical_windows_v2"
FEATURE_SCHEMA = "event_model_s5_court_nan_windows_v3"
DEPRECATED_WARPED_FEATURE_SCHEMA = "event_model_v2_automatic_windows_v1"
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = COURT_LENGTH_M / 2.0
COURT_MISSING_POLICIES = ("nan", "zero")
BASE_FEATURE_NAMES = (
    "image_x",
    "image_y",
    "image_vx",
    "image_vy",
    "image_ax",
    "image_ay",
    "direction_change",
    "track_presence",
    "court_x",
    "court_y",
    "court_vx",
    "court_vy",
    "court_ax",
    "court_ay",
    "court_speed",
    "net_proximity",
)
CAMERA_ELEVATION_FEATURE_NAME = "camera_elevation_proxy"
PHYSICAL_FEATURE_NAMES = (
    "distance_camera_interaction",
    "court_surface_signed_margin",
)
LEGACY_PHYSICAL_FEATURE_NAMES = (
    "distance_camera_interaction",
    CAMERA_ELEVATION_FEATURE_NAME,
    "court_surface_signed_margin",
)
# Court channels whose value is undefined when the point has no homography.
COURT_FEATURE_NAMES = (
    "court_x",
    "court_y",
    "court_vx",
    "court_vy",
    "court_ax",
    "court_ay",
    "court_speed",
    "net_proximity",
    "distance_camera_interaction",
    "court_surface_signed_margin",
)


def feature_names(*, include_camera_elevation: bool = False) -> tuple[str, ...]:
    """Return the channel schema for the requested physical-feature set."""

    if include_camera_elevation:
        return (*BASE_FEATURE_NAMES, *LEGACY_PHYSICAL_FEATURE_NAMES)
    return (*BASE_FEATURE_NAMES, *PHYSICAL_FEATURE_NAMES)


FEATURE_NAMES = feature_names()
LEGACY_FEATURE_NAMES = feature_names(include_camera_elevation=True)


@dataclass(frozen=True)
class AutomaticDataset:
    windows: np.ndarray
    clips: np.ndarray
    broadcasts: np.ndarray
    frames: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    # The point has no court homography at all.
    court_geometry_missing: np.ndarray | None = None
    # The window centre frame has no court coordinate, whether because the point
    # has no homography or because the ball was not detected on that frame.
    court_coordinate_missing: np.ndarray | None = None

    def _mask(self, values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.zeros(len(self.frames), dtype=bool)
        return np.asarray(values, dtype=bool)

    def missing_court_geometry(self) -> np.ndarray:
        return self._mask(self.court_geometry_missing)

    def missing_court_coordinate(self) -> np.ndarray:
        return self._mask(self.court_coordinate_missing)

    def subset(self, mask: np.ndarray) -> AutomaticDataset:
        return AutomaticDataset(
            windows=self.windows[mask],
            clips=self.clips[mask],
            broadcasts=self.broadcasts[mask],
            frames=self.frames[mask],
            feature_names=self.feature_names,
            court_geometry_missing=(
                None if self.court_geometry_missing is None else self.court_geometry_missing[mask]
            ),
            court_coordinate_missing=(
                None
                if self.court_coordinate_missing is None
                else self.court_coordinate_missing[mask]
            ),
        )


def _frame_number(value: str) -> int:
    return int(Path(value).stem.removeprefix("f_"))


def _point_number(clip: str) -> int:
    return int(clip.removeprefix("pt"))


def _supported_derivative(values: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(values, dtype=np.float32)
    supported = np.zeros(len(values), dtype=bool)
    for index in np.flatnonzero(present):
        before = index > 0 and present[index - 1]
        after = index + 1 < len(values) and present[index + 1]
        if before and after:
            output[index] = (values[index + 1] - values[index - 1]) / 2.0
            supported[index] = True
        elif before:
            output[index] = values[index] - values[index - 1]
            supported[index] = True
        elif after:
            output[index] = values[index + 1] - values[index]
            supported[index] = True
    return output, supported


def _project(
    homography: np.ndarray | None,
    x: float,
    y: float,
    *,
    image_size: resolution.FrameSize,
    artifact_size: resolution.FrameSize,
) -> tuple[float, float]:
    if homography is None:
        return math.nan, math.nan
    image_point = resolution.scale_points(
        np.asarray([[x, y]], dtype=float), artifact_size, image_size
    )[0]
    projected = homography @ np.asarray([*image_point, 1.0], dtype=float)
    if not np.isfinite(projected).all() or abs(projected[2]) < 1e-9:
        return math.nan, math.nan
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def frame_features(
    n_frames: int,
    observations: dict[int, tuple[float, float]],
    homographies: np.ndarray | dict[int, np.ndarray] | None,
    *,
    image_size: resolution.FrameSize,
    artifact_size: resolution.FrameSize,
    court_missing: str = "nan",
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame base channels and the court-geometry presence mask.

    ``court_missing="nan"`` leaves every court channel undefined on frames with
    no usable homography, which histogram gradient boosting consumes natively.
    ``court_missing="zero"`` reproduces the legacy silent zero fill.
    """

    if court_missing not in COURT_MISSING_POLICIES:
        raise ValueError(f"unsupported court_missing policy: {court_missing!r}")
    x = np.zeros(n_frames, dtype=np.float32)
    y = np.zeros(n_frames, dtype=np.float32)
    court_x = np.zeros(n_frames, dtype=np.float32)
    court_y = np.zeros(n_frames, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    court_present = np.zeros(n_frames, dtype=bool)
    for frame, (image_x, image_y) in observations.items():
        if frame < 0 or frame >= n_frames:
            continue
        present[frame] = True
        x[frame] = image_x / artifact_size.width
        y[frame] = image_y / artifact_size.height
        homography = homographies.get(frame) if isinstance(homographies, dict) else homographies
        projected_x, projected_y = _project(
            homography,
            image_x,
            image_y,
            image_size=image_size,
            artifact_size=artifact_size,
        )
        if math.isfinite(projected_x) and math.isfinite(projected_y):
            court_present[frame] = True
            court_x[frame] = projected_x / COURT_WIDTH_M
            court_y[frame] = projected_y / COURT_LENGTH_M

    vx, velocity_present = _supported_derivative(x, present)
    vy, _ = _supported_derivative(y, present)
    ax, _ = _supported_derivative(vx, velocity_present)
    ay, _ = _supported_derivative(vy, velocity_present)
    court_vx, court_velocity_present = _supported_derivative(court_x, court_present)
    court_vy, _ = _supported_derivative(court_y, court_present)
    court_ax, court_acceleration_present = _supported_derivative(court_vx, court_velocity_present)
    court_ay, _ = _supported_derivative(court_vy, court_velocity_present)
    direction_change = np.zeros(n_frames, dtype=np.float32)
    for index in range(1, n_frames - 1):
        if not (present[index - 1] and present[index] and present[index + 1]):
            continue
        incoming = np.asarray([x[index] - x[index - 1], y[index] - y[index - 1]])
        outgoing = np.asarray([x[index + 1] - x[index], y[index + 1] - y[index]])
        scale = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if scale > 1e-9:
            direction_change[index] = 1.0 - float(
                np.clip(np.dot(incoming, outgoing) / scale, -1.0, 1.0)
            )
    court_speed = np.hypot(court_vx, court_vy).astype(np.float32)
    net_proximity = np.zeros(n_frames, dtype=np.float32)
    net_proximity[court_present] = (
        np.abs(court_y[court_present] * COURT_LENGTH_M - NET_Y_M) / NET_Y_M
    )
    if court_missing == "nan":
        for values, support in (
            (court_x, court_present),
            (court_y, court_present),
            (net_proximity, court_present),
            (court_vx, court_velocity_present),
            (court_vy, court_velocity_present),
            (court_speed, court_velocity_present),
            (court_ax, court_acceleration_present),
            (court_ay, court_acceleration_present),
        ):
            values[~support] = np.nan
    return (
        np.column_stack(
            (
                x,
                y,
                vx,
                vy,
                ax,
                ay,
                direction_change,
                present.astype(np.float32),
                court_x,
                court_y,
                court_vx,
                court_vy,
                court_ax,
                court_ay,
                court_speed,
                net_proximity,
            )
        ).astype(np.float32),
        court_present,
    )


def centered_windows(features: np.ndarray, frames: np.ndarray) -> np.ndarray:
    padded = np.pad(features, ((RADIUS, RADIUS), (0, 0)))
    return np.stack(
        [padded[int(frame) : int(frame) + WINDOW_SIZE] for frame in frames], axis=0
    ).astype(np.float32)


def _add_physical_features(windows: np.ndarray, elevation: float | None = None) -> np.ndarray:
    """Append the physical channels; ``elevation`` adds the legacy feature 17."""

    image_y = windows[:, :, 1]
    court_x = windows[:, :, 8]
    court_y = windows[:, :, 9]
    distance_camera_interaction = (image_y * court_y)[:, :, None]
    court_surface_signed_margin = np.minimum.reduce(
        (court_x, 1.0 - court_x, court_y, 1.0 - court_y)
    )[:, :, None]
    extra = [distance_camera_interaction]
    if elevation is not None:
        extra.append(np.full((*windows.shape[:2], 1), elevation, dtype=np.float32))
    extra.append(court_surface_signed_margin)
    return np.concatenate((windows, *extra), axis=2).astype(np.float32)


# The pixel columns a sidecar without a column map is assumed to carry.
TRACK_PIXEL_COLUMNS = ("x", "y")
NATIVE_COLUMNS_KEY = "native_1920x1080"
LEGACY_COLUMNS_KEY = "legacy_960x540"
# Height of the net at the posts, which is the top edge the observed net-cord
# samples trace.  The tape sags to 0.914 m at the centre strap.
NET_TAPE_HEIGHT_M = 1.07
NET_SAMPLES = 9
CAMERA_PROJECTION_NAME = "camera_P_per_point.npz"
HOMOGRAPHY_NAME = "court_H_per_point.npz"
FRAME_HOMOGRAPHY_NAME = "court_H_per_frame_v1.npz"
COURT_GEOMETRY_MODES = ("point_static", "reliable_per_frame")
COURT_FRAME_MISSING_POLICIES = ("hold", "point_static")
NET_GEOMETRY_SCHEMA = "event_feature_net_geometry_v1"


@dataclass(frozen=True)
class TrackPixelSpace:
    """Which track columns the builder reads and the frame size they are in."""

    x_column: str
    y_column: str
    size: resolution.FrameSize
    space: str


def track_pixel_space(path: Path) -> TrackPixelSpace:
    """Resolve the track's pixel columns through the sidecar's column map.

    Native columns win, then declared legacy columns against
    ``legacy_artifact_size``, then the historical ``x``/``y`` at
    ``artifact_size`` for a sidecar that declares no column map.  This is the
    rule :func:`cv.pipeline.event_crops.track_coordinate_space` applies, so the
    features and the crops read the ball from the same place.
    """

    manifest = resolution.read_coordinate_manifest(path)
    if manifest is None:
        raise FileNotFoundError(
            f"event features require an explicit coordinate sidecar: "
            f"{resolution.coordinate_manifest_path(path)}"
        )
    columns = manifest.get("coordinate_columns") or {}
    native = tuple(columns.get(NATIVE_COLUMNS_KEY) or ())
    if native:
        if len(native) != 2:
            raise ValueError(f"{path} declares malformed native columns: {native}")
        return TrackPixelSpace(
            native[0],
            native[1],
            resolution.manifest_artifact_size(manifest),
            NATIVE_COLUMNS_KEY,
        )
    legacy = tuple(columns.get(LEGACY_COLUMNS_KEY) or ())
    if legacy:
        if len(legacy) != 2:
            raise ValueError(f"{path} declares malformed legacy columns: {legacy}")
        return TrackPixelSpace(
            legacy[0],
            legacy[1],
            resolution.manifest_artifact_size(manifest, legacy_columns=True),
            LEGACY_COLUMNS_KEY,
        )
    return TrackPixelSpace(
        *TRACK_PIXEL_COLUMNS,
        resolution.manifest_artifact_size(manifest),
        "artifact_size",
    )


def track_column_size(path: Path) -> resolution.FrameSize:
    """Return the frame size the columns the builder reads actually live in."""

    return track_pixel_space(path).size


@dataclass(frozen=True)
class BroadcastTrack:
    """One broadcast's bound ball evidence, whichever track the run selected.

    ``observations`` are pixels in ``artifact_size``; ``image_size`` is the frame
    the homography lives in.  Both the feature builder and the crop cutter take
    their centres from here, so the two readers cannot disagree about which row
    is an observation or where it is.
    """

    path: Path
    image_size: resolution.FrameSize
    artifact_size: resolution.FrameSize
    observations: dict[str, dict[int, tuple[float, float]]]
    space: dict
    binding: CurrentTrackBinding | None = None


def bound_track_space(
    match_root: Path, *, event_track: str = LEGACY_EVENT_TRACK
) -> tuple[resolution.FrameSize, resolution.FrameSize] | None:
    """``(image_size, artifact_size)`` of the selected track, without reading rows.

    Derived geometry -- the net line above all -- needs the frame the selected
    pixels live in, not the pixels themselves, so it must not pay for a parse of
    the whole artifact.  ``None`` means this broadcast has no legacy track; the
    current track fails closed instead, exactly as :func:`bind_broadcast_track`.
    """

    validate_event_track(event_track)
    match_root = Path(match_root)
    path = match_root / event_track_name(event_track)
    if event_track != LEGACY_EVENT_TRACK:
        if not path.is_file():
            raise FileNotFoundError(f"the selected current ball track is missing: {path}")
        image_size, _declared_artifact = _coordinate_sizes(path)
        # Every row the shared reader returns is scaled into native pixels.
        return image_size, resolution.NATIVE_SIZE
    if not path.exists():
        return None
    image_size, _declared_artifact = _coordinate_sizes(path)
    return image_size, track_pixel_space(path).size


def bind_broadcast_track(
    match_root: Path, *, event_track: str = LEGACY_EVENT_TRACK
) -> BroadcastTrack | None:
    """Bind one broadcast's selected ball track, or ``None`` when it has none.

    ``event_track="legacy"`` is the historical reader: every row of the
    availability track is an observation, best score per ``(clip, frame)``.

    ``event_track="s4_current"`` reads the composed S4 track through the shared
    S6 reader instead, which follows the track's declared guide ancestry.  Only
    measured rows become observations there; a guide-interpolated row and a row
    whose coarse claim the guide cannot confirm are carried as context, so they
    position a crop but are never presented as a detection.  That path fails
    closed -- an absent track or an absent, malformed, self-referential or
    untraceable guide raises rather than falling back to the legacy track.
    """

    validate_event_track(event_track)
    match_root = Path(match_root)
    if event_track != LEGACY_EVENT_TRACK:
        binding = bind_current_track(match_root)
        return BroadcastTrack(
            path=binding.track,
            image_size=binding.image_size,
            artifact_size=binding.artifact_size,
            observations=binding.observations,
            space={
                "columns": list(binding.columns),
                # The reader scales whatever columns the sidecar declares into
                # native pixels, so the rows are native whatever it read.
                "space": NATIVE_COLUMNS_KEY,
                "size": {
                    "width": binding.artifact_size.width,
                    "height": binding.artifact_size.height,
                },
                "declared_source_space": binding.declared_source_space,
                "event_track": binding.event_track,
            },
            binding=binding,
        )
    path = match_root / TRACK_NAME
    if not path.exists():
        return None
    space = track_pixel_space(path)
    image_size, _declared_artifact = _coordinate_sizes(path)
    return BroadcastTrack(
        path=path,
        image_size=image_size,
        # The pixels just read live in ``space.size``; that, not the sidecar's
        # authoritative artifact size, is the frame every channel is built from.
        artifact_size=space.size,
        observations=_load_track(path, space),
        space={
            "columns": [space.x_column, space.y_column],
            "space": space.space,
            "size": {"width": space.size.width, "height": space.size.height},
        },
    )


def feature_court_xy(
    homography: np.ndarray,
    points,
    *,
    image_size: resolution.FrameSize,
    artifact_size: resolution.FrameSize,
) -> np.ndarray:
    """Return ``(court_x, court_y)`` exactly as :func:`frame_features` computes it.

    ``points`` are in the artifact space the builder reads.  Rows that do not
    project are ``nan``.
    """

    values = np.atleast_2d(np.asarray(points, dtype=float))
    if not values.size:
        return np.zeros((0, 2), dtype=float)
    scaled = resolution.scale_points(values, artifact_size, image_size)
    homogeneous = np.column_stack((scaled, np.ones(len(scaled))))
    projected = homogeneous @ np.asarray(homography, dtype=float).T
    depth = projected[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        court = projected[:, :2] / depth[:, None]
    court[~np.isfinite(court).all(axis=1) | (np.abs(depth) < 1e-9)] = np.nan
    return court / np.asarray([COURT_WIDTH_M, COURT_LENGTH_M], dtype=float)


def _net_ground_pixels(homography: np.ndarray, samples: int = NET_SAMPLES) -> np.ndarray:
    """Native image pixels of the net's ground line, from the point homography."""

    inverse = np.linalg.inv(np.asarray(homography, dtype=float))
    world = np.column_stack(
        (
            np.linspace(0.0, COURT_WIDTH_M, samples),
            np.full(samples, NET_Y_M),
            np.ones(samples),
        )
    )
    projected = world @ inverse.T
    return projected[:, :2] / projected[:, 2:3]


def _net_tape_pixels(projection: np.ndarray, samples: int = NET_SAMPLES) -> np.ndarray:
    """Native image pixels of the net's tape top, from the camera matrix."""

    world = np.column_stack(
        (
            np.linspace(0.0, COURT_WIDTH_M, samples),
            np.full(samples, NET_Y_M),
            np.full(samples, NET_TAPE_HEIGHT_M),
            np.ones(samples),
        )
    )
    projected = world @ np.asarray(projection, dtype=float).T
    return projected[:, :2] / projected[:, 2:3]


def net_geometry_in_feature_frame(
    homography: np.ndarray,
    *,
    image_size: resolution.FrameSize,
    artifact_size: resolution.FrameSize,
    track_size: resolution.FrameSize,
    net_cord_pixels=None,
    camera_projection=None,
) -> dict | None:
    """Where the net's own pixels land in the ``court_y`` the builder reports.

    The net is a physical object at ``y = NET_Y_M`` of ``COURT_LENGTH_M``.  Its
    ground line comes from the point homography and its top edge from the
    observed net-cord samples of the camera bundle, or, when those are absent,
    from the camera matrix at ``NET_TAPE_HEIGHT_M``.  Both are native image
    pixels; they are converted into the space the track columns are in and then
    pushed through the builder's own projection, so the answer is in the same
    frame as the ``court_y`` feature whatever that frame happens to be.

    Returns ``net_line`` -- the net's top edge, which is the line a ball must be
    beyond to be on the far half -- and ``net_span``, the net's own projected
    thickness, which is the natural width of the dead band around it.
    """

    homography = np.asarray(homography, dtype=float)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        return None
    top_pixels = None
    source = None
    if net_cord_pixels is not None:
        candidate = np.atleast_2d(np.asarray(net_cord_pixels, dtype=float))
        if candidate.size and np.isfinite(candidate).all():
            top_pixels = candidate
            source = "observed_net_cord"
    if top_pixels is None and camera_projection is not None:
        projection = np.asarray(camera_projection, dtype=float)
        if np.isfinite(projection).all():
            top_pixels = _net_tape_pixels(projection)
            source = "camera_projection_tape_top"
    if top_pixels is None:
        return None
    base_pixels = _net_ground_pixels(homography)

    def _court_y(native_pixels: np.ndarray) -> np.ndarray:
        in_track_space = resolution.scale_points(native_pixels, image_size, track_size)
        return feature_court_xy(
            homography,
            in_track_space,
            image_size=image_size,
            artifact_size=artifact_size,
        )[:, 1]

    base = _court_y(base_pixels)
    top = _court_y(top_pixels)
    if not (np.isfinite(base).any() and np.isfinite(top).any()):
        return None
    net_line = float(np.nanmedian(top))
    net_span = float(net_line - np.nanmedian(base))
    return {
        "net_line": net_line,
        "net_span": net_span,
        "base_court_y": [float(np.nanmin(base)), float(np.nanmax(base))],
        "top_court_y": [float(np.nanmin(top)), float(np.nanmax(top))],
        "samples": int(len(top_pixels)),
        "source": source,
    }


def derive_net_lines(
    root: Path,
    *,
    track_name: str | None = None,
    event_track: str = LEGACY_EVENT_TRACK,
) -> dict[str, dict]:
    """Derive the net line of every point of a cohort root, keyed by global clip.

    Points whose own camera bundle cannot place the net inherit their
    broadcast's median net line and span, and are marked
    ``source="broadcast_median"``.

    The line itself comes from the point's homography and the net's own pixels.
    :func:`net_geometry_in_feature_frame` converts those native pixels into the
    track's declared space and projects them back out of it, so the declared
    space cancels and the returned line does not depend on which track was
    selected.  What the selection does decide is which artifact is opened for
    that space at all: a point whose selected track is absent has no line here.
    ``track_name`` still names an explicit artifact for callers that pin one; it
    defaults to the artifact ``event_track`` selects.
    """

    validate_event_track(event_track)
    root = Path(root)
    active = json.loads((root / "active_play_v1.json").read_text())
    output: dict[str, dict] = {}
    for broadcast in sorted({key.split("/", 1)[0] for key in active}):
        match_root = root / broadcast
        homography_path = match_root / HOMOGRAPHY_NAME
        if track_name is not None:
            track_path = match_root / track_name
            if not (track_path.exists() and homography_path.exists()):
                continue
            image_size, _declared_artifact = _coordinate_sizes(track_path)
            artifact_size = track_size = track_pixel_space(track_path).size
        else:
            if not homography_path.exists():
                continue
            sizes = bound_track_space(match_root, event_track=event_track)
            if sizes is None:
                continue
            image_size, artifact_size = sizes
            track_size = artifact_size
        homographies = _load_homographies(homography_path)
        cords: dict[int, np.ndarray] = {}
        projections: dict[int, np.ndarray] = {}
        camera_path = match_root / CAMERA_PROJECTION_NAME
        if camera_path.exists():
            with np.load(camera_path, allow_pickle=False) as camera:
                for position, point in enumerate(camera["pts"]):
                    projections[int(point)] = np.asarray(camera["P"][position], dtype=float)
                    if "net_cord_xy" in camera.files and bool(camera["net_cord_valid"][position]):
                        cords[int(point)] = np.asarray(camera["net_cord_xy"][position], dtype=float)
        derived: dict[int, dict] = {}
        for point, homography in sorted(homographies.items()):
            geometry = net_geometry_in_feature_frame(
                homography,
                image_size=image_size,
                artifact_size=artifact_size,
                track_size=track_size,
                net_cord_pixels=cords.get(point),
                camera_projection=projections.get(point),
            )
            if geometry is not None:
                derived[point] = geometry
        median_line = (
            float(np.median([item["net_line"] for item in derived.values()])) if derived else None
        )
        median_span = (
            float(np.median([item["net_span"] for item in derived.values()])) if derived else None
        )
        for key, record in active.items():
            match, clip = key.split("/", 1)
            if match != broadcast:
                continue
            del record
            point = _point_number(clip)
            geometry = derived.get(point)
            if geometry is None:
                if median_line is None:
                    continue
                geometry = {
                    "net_line": median_line,
                    "net_span": median_span,
                    "base_court_y": None,
                    "top_court_y": None,
                    "samples": 0,
                    "source": "broadcast_median",
                }
            output[f"{broadcast}__{clip}"] = {**geometry, "broadcast": broadcast, "point": point}
    return output


def _load_track(
    path: Path, space: TrackPixelSpace | None = None
) -> dict[str, dict[int, tuple[float, float]]]:
    """Read the track's pixel columns, best score per ``(clip, frame)``.

    ``space`` names the columns to read; it defaults to the sidecar's own
    resolution.  A sidecar that declares columns the track does not carry is
    refused rather than silently falling back, which is the failure mode
    ``docs/wk1/events_pass3.md`` traced in the crop cutter.
    """

    if space is None:
        space = track_pixel_space(path)
    output: dict[str, dict[int, tuple[float, float]]] = {}
    scores: dict[tuple[str, int], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for name in (space.x_column, space.y_column):
            if reader.fieldnames is not None and name not in reader.fieldnames:
                raise ValueError(
                    f"{path} has no column {name} declared by its coordinate "
                    f"sidecar as {space.space}: {reader.fieldnames}"
                )
        for row in reader:
            clip = row["clip"]
            frame = _frame_number(row["frame"])
            score = float(row.get("score") or 0.0)
            key = (clip, frame)
            if key in scores and scores[key] >= score:
                continue
            scores[key] = score
            output.setdefault(clip, {})[frame] = (
                float(row[space.x_column]),
                float(row[space.y_column]),
            )
    return output


def _coordinate_sizes(path: Path) -> tuple[resolution.FrameSize, resolution.FrameSize]:
    manifest = resolution.read_coordinate_manifest(path)
    if manifest is None:
        raise FileNotFoundError(
            f"event features require an explicit coordinate sidecar: "
            f"{resolution.coordinate_manifest_path(path)}"
        )
    if manifest.get("schema") != "tennis.coordinate-space.v1":
        raise ValueError(f"unsupported coordinate manifest: {manifest.get('schema')!r}")
    image = manifest["image_size"]
    artifact = manifest["artifact_size"]
    return (
        resolution.FrameSize(int(image["width"]), int(image["height"])),
        resolution.FrameSize(int(artifact["width"]), int(artifact["height"])),
    )


def _load_homographies(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {
            int(point): np.asarray(homography, dtype=float)
            for point, homography in zip(data["pts"], data["H"], strict=True)
            if np.isfinite(homography).all()
        }


def validate_court_transport(mode: str, missing: str) -> None:
    if mode not in COURT_GEOMETRY_MODES:
        raise ValueError(f"unsupported court_geometry mode: {mode!r}")
    if missing not in COURT_FRAME_MISSING_POLICIES:
        raise ValueError(f"unsupported court_frame_missing policy: {missing!r}")
    if mode == "point_static" and missing != "hold":
        raise ValueError("court_frame_missing requires reliable_per_frame geometry")


@dataclass(frozen=True)
class CourtTransport:
    """Explicit native-image ground transport; missing is distinct from static fallback."""

    points: dict[int, np.ndarray]
    frames: dict[str, dict[int, np.ndarray]]
    sources: dict[tuple[str, int], str]
    mode: str
    missing: str
    artifact: dict | None

    def matrix(self, clip: str, frame: int) -> np.ndarray | None:
        if self.mode == "point_static":
            return self.points.get(_point_number(clip))
        value = self.frames.get(clip, {}).get(frame)
        if value is not None:
            return value
        return self.points.get(_point_number(clip)) if self.missing == "point_static" else None

    def row_record(self, clip: str, frame: int) -> dict:
        accepted = frame in self.frames.get(clip, {})
        reason = self.sources.get(
            (clip, frame), "missing_frame" if self.artifact is not None else "missing_artifact"
        )
        fallback = (
            self.mode != "point_static" and not accepted and self.matrix(clip, frame) is not None
        )
        return {
            "mode": self.mode,
            "frame": int(frame),
            "frame_index_origin": 1,
            "homography_direction": "native_image_to_court_metres",
            "source": "point_static" if self.mode == "point_static" else reason,
            "reliable_per_frame": accepted,
            "static_fallback": fallback,
            "held": self.matrix(clip, frame) is None,
            "coordinate_role": "proposal_pixel_ground_plane_projection; not airborne XYZ",
        }

    def for_clip(self, clip: str, n_frames: int) -> np.ndarray | dict[int, np.ndarray] | None:
        if self.mode == "point_static":
            return self.points.get(_point_number(clip))
        return {
            frame: matrix
            for frame in range(n_frames)
            if (matrix := self.matrix(clip, frame)) is not None
        }


def load_court_transport(
    match_root: Path, *, mode: str = "point_static", missing: str = "hold"
) -> CourtTransport:
    """Read the declared automatic frame artifact only in explicitly enabled mode.

    Original native frame integers are used directly; no nearest-frame substitution,
    interpolation, human-legacy reliability inference or ambient opt-in occurs here.
    """
    from cv.pipeline.provenance import file_record

    validate_court_transport(mode, missing)
    points = _load_homographies(match_root / HOMOGRAPHY_NAME)
    frames: dict[str, dict[int, np.ndarray]] = {}
    sources: dict[tuple[str, int], str] = {}
    artifact = None
    path = match_root / FRAME_HOMOGRAPHY_NAME
    if mode == "reliable_per_frame" and path.exists():
        artifact = file_record(path)
        with np.load(path, allow_pickle=False) as data:
            epochs = data["frames"]
            if epochs.ndim != 1 or epochs.dtype.kind not in "iu" or np.any(epochs < 0):
                raise ValueError("court frame epochs must be nonnegative native integers")
            reliable = data.get("reliable", np.zeros(len(epochs), dtype=bool))
            if reliable.dtype != np.dtype(bool):
                raise ValueError("court frame reliability must be boolean")
            origin = data.get("source", np.full(len(epochs), "unspecified_source"))
            for clip, frame, matrix, accepted, source in zip(
                data["clips"], epochs, data["H"], reliable, origin, strict=True
            ):
                key = str(clip), int(frame)
                if key in sources:
                    raise ValueError(f"duplicate native court frame: {key}")
                matrix = np.asarray(matrix, dtype=float)
                valid = matrix.shape == (3, 3) and np.isfinite(matrix).all()
                valid = valid and abs(np.linalg.det(matrix)) > 1e-12
                if accepted and valid:
                    frames.setdefault(key[0], {})[key[1]] = matrix
                    sources[key] = str(source)
                else:
                    sources[key] = ("invalid_matrix:" if not valid else "unreliable:") + str(source)
        if file_record(path) != artifact:
            raise ValueError("court frame artifact changed during loading")
    return CourtTransport(points, frames, sources, mode, missing, artifact)


def _scope_frames(record: dict) -> np.ndarray:
    frames: set[int] = set()
    n_frames = int(record["n_frames"])
    for start, stop in record.get("event_spans", []):
        first = max(0, int(math.ceil(float(start))))
        last = min(n_frames - 1, int(math.floor(float(stop))))
        frames.update(range(first, last + 1))
    return np.asarray(sorted(frames), dtype=np.int32)


def _horizon_proxy(homography: np.ndarray, image_size: resolution.FrameSize) -> float:
    """Return the normalized court-plane horizon height at image center."""

    line = np.asarray(homography[2], dtype=float)
    if not np.isfinite(line).all() or abs(line[1]) < 1e-12:
        return math.nan
    return float(-(line[0] * image_size.width / 2.0 + line[2]) / line[1] / image_size.height)


def _camera_elevation_proxy(
    homographies: list[np.ndarray], image_size: resolution.FrameSize
) -> tuple[float, int]:
    values = [_horizon_proxy(homography, image_size) for homography in homographies]
    finite = [value for value in values if math.isfinite(value)]
    return (float(np.median(finite)) if finite else 0.0, len(finite))


def build_automatic_dataset(
    root: Path,
    *,
    include_camera_elevation: bool = False,
    court_missing: str = "nan",
    court_geometry: str = "point_static",
    court_frame_missing: str = "hold",
    event_track: str = LEGACY_EVENT_TRACK,
    live_shot_camera: bool = False,
) -> tuple[AutomaticDataset, dict]:
    """Build label-free 25-frame windows for every scoped event frame.

    ``include_camera_elevation`` adds the legacy per-broadcast constant channel
    (feature 17), which is a broadcast identifier and is off by default.
    ``court_missing`` selects the missing-court-geometry fill policy.
    ``event_track`` selects which ball track supplies the observations; see
    :func:`bind_broadcast_track`. ``live_shot_camera`` is the same switch as
    the S6 release: a clip the shot gate emptied is scoped to its wide shot.
    Off, which is the default, the producer's event spans are the whole scope.
    """

    if court_missing not in COURT_MISSING_POLICIES:
        raise ValueError(f"unsupported court_missing policy: {court_missing!r}")
    validate_court_transport(court_geometry, court_frame_missing)
    validate_event_track(event_track)
    active_path = root / "active_play_v1.json"
    active = json.loads(active_path.read_text())
    windows: list[np.ndarray] = []
    clips: list[str] = []
    broadcasts: list[str] = []
    frames: list[int] = []
    missing_geometry: list[np.ndarray] = []
    missing_coordinate: list[np.ndarray] = []
    track_cache: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
    homography_cache: dict[str, dict[int, np.ndarray]] = {}
    coordinate_cache: dict[str, tuple[resolution.FrameSize, resolution.FrameSize]] = {}
    pixel_spaces: dict[str, dict] = {}
    bindings: dict[str, CurrentTrackBinding] = {}
    camera_scalars: dict[str, dict] = {}
    transports: dict[str, CourtTransport] = {}
    transport_counts: dict[str, dict[str, int]] = {}

    available_broadcasts = sorted({point_key.split("/", 1)[0] for point_key in active})
    support_spans: dict[str, dict[str, list]] = {}
    live_scoped: list[str] = []
    if live_shot_camera:
        # Loaded here, not at import: camera support imports this module.
        from cv.pipeline.camera_frame_support import closed_runs
        from cv.pipeline.live_shot_camera import scope_event_spans

        for broadcast in available_broadcasts:
            support = load_court_transport(
                root / broadcast, mode="reliable_per_frame", missing="hold"
            )
            support_spans[broadcast] = {
                clip: closed_runs(frames) for clip, frames in support.frames.items()
            }
    for broadcast in available_broadcasts:
        match_root = root / broadcast
        bound = bind_broadcast_track(match_root, event_track=event_track)
        if bound is None:
            continue
        pixel_spaces[broadcast] = bound.space
        if bound.binding is not None:
            bindings[broadcast] = bound.binding
        track_cache[broadcast] = bound.observations
        transport = load_court_transport(
            match_root, mode=court_geometry, missing=court_frame_missing
        )
        transports[broadcast] = transport
        homography_cache[broadcast] = transport.points
        coordinate_cache[broadcast] = (bound.image_size, bound.artifact_size)

    if include_camera_elevation:
        for broadcast in sorted(track_cache):
            image_size, _ = coordinate_cache[broadcast]
            value, samples = _camera_elevation_proxy(
                list(homography_cache[broadcast].values()), image_size
            )
            camera_scalars[broadcast] = {
                "value": value,
                "samples": samples,
                "source": "court_H_per_point",
            }

    for point_key, record in sorted(active.items()):
        broadcast, clip = point_key.split("/", 1)
        if broadcast not in track_cache:
            continue
        scoped_record = record
        if live_shot_camera:
            spans = scope_event_spans(record, support_spans.get(broadcast, {}).get(clip, []))
            if spans and not record.get("event_spans"):
                live_scoped.append(f"{broadcast}/{clip}")
            scoped_record = {**record, "event_spans": spans}
        scoped_frames = _scope_frames(scoped_record)
        if not len(scoped_frames):
            continue
        observations = track_cache[broadcast].get(clip, {})
        homographies = transports[broadcast].for_clip(clip, int(record["n_frames"]))
        image_size, artifact_size = coordinate_cache[broadcast]
        base, court_present = frame_features(
            int(record["n_frames"]),
            observations,
            homographies,
            image_size=image_size,
            artifact_size=artifact_size,
            court_missing=court_missing,
        )
        local_windows = _add_physical_features(
            centered_windows(base, scoped_frames),
            float(camera_scalars[broadcast]["value"]) if include_camera_elevation else None,
        )
        windows.append(local_windows)
        missing_geometry.append(
            np.asarray([transports[broadcast].matrix(clip, int(f)) is None for f in scoped_frames])
        )
        if court_geometry != "point_static":
            counts = transport_counts.setdefault(broadcast, {})
            for frame in scoped_frames:
                row = transports[broadcast].row_record(clip, int(frame))
                status = (
                    "held"
                    if row["held"]
                    else "static_fallback"
                    if row["static_fallback"]
                    else "reliable_per_frame"
                )
                counts[status] = counts.get(status, 0) + 1
        missing_coordinate.append(~court_present[scoped_frames])
        global_clip = f"{broadcast}__{clip}"
        clips.extend([global_clip] * len(scoped_frames))
        broadcasts.extend([broadcast] * len(scoped_frames))
        frames.extend(scoped_frames.tolist())
    names = feature_names(include_camera_elevation=include_camera_elevation)
    dataset = AutomaticDataset(
        windows=np.concatenate(windows),
        clips=np.asarray(clips),
        broadcasts=np.asarray(broadcasts),
        frames=np.asarray(frames, dtype=np.int32),
        feature_names=names,
        court_geometry_missing=np.concatenate(missing_geometry),
        court_coordinate_missing=np.concatenate(missing_coordinate),
    )
    geometry_missing = dataset.missing_court_geometry()
    coordinate_missing = dataset.missing_court_coordinate()
    return dataset, {
        "schema": LEGACY_FEATURE_SCHEMA if include_camera_elevation else FEATURE_SCHEMA,
        "root": str(root.resolve()),
        "rows": len(dataset.frames),
        "clips": len(set(dataset.clips.tolist())),
        "broadcasts": len(set(dataset.broadcasts.tolist())),
        "window_radius": RADIUS,
        "window_size": WINDOW_SIZE,
        "feature_names": list(names),
        "court_projection": "track pixels read through the sidecar's declared column map, scaled from the frame size those columns live in to native homography pixels",
        "track_pixel_space": dict(sorted(pixel_spaces.items())),
        "physical_features": list(
            LEGACY_PHYSICAL_FEATURE_NAMES if include_camera_elevation else PHYSICAL_FEATURE_NAMES
        ),
        "camera_elevation_proxy": {
            "included": include_camera_elevation,
            "channels": camera_scalars,
            "note": (
                "one constant per broadcast; it is a broadcast identifier under a "
                "leave-one-broadcast-out split"
            ),
        },
        "court_geometry": {
            "missing_policy": court_missing,
            "missing_court_channels": list(COURT_FEATURE_NAMES),
            "no_homography_rows": int(geometry_missing.sum()),
            "no_homography_fraction": float(geometry_missing.mean()),
            "no_centre_frame_coordinate_rows": int(coordinate_missing.sum()),
            "no_centre_frame_coordinate_fraction": float(coordinate_missing.mean()),
            "per_broadcast_no_homography_fraction": {
                broadcast: float(geometry_missing[dataset.broadcasts == broadcast].mean())
                for broadcast in sorted(set(dataset.broadcasts.tolist()))
            },
            "per_broadcast_no_centre_frame_coordinate_fraction": {
                broadcast: float(coordinate_missing[dataset.broadcasts == broadcast].mean())
                for broadcast in sorted(set(dataset.broadcasts.tolist()))
            },
        },
        "warped_feature_path": {
            "schema": DEPRECATED_WARPED_FEATURE_SCHEMA,
            "status": "deprecated_rollback_only",
            "removal": "after one clean canonical rebenchmark cycle",
        },
        # Declared only when the switch is on, so a default feature manifest is unchanged.
        **(
            {"live_shot_camera_scope": live_scoped}
            if live_shot_camera
            else {}
        ),
        **(
            {
                "court_transport": {
                    "mode": court_geometry,
                    "missing_policy": court_frame_missing,
                    "artifact_name": FRAME_HOMOGRAPHY_NAME,
                    "per_broadcast": {
                        name: {
                            "artifact": transport.artifact,
                            "scoped_row_counts": transport_counts.get(name, {}),
                        }
                        for name, transport in sorted(transports.items())
                    },
                }
            }
            if court_geometry != "point_static"
            else {}
        ),
        # Default-legacy manifests keep their exact shape; a selected track
        # declares itself, its guide ancestry and the rows it refused to call
        # observations, mirroring the optional-transport block above.
        **(
            {
                "ball_track": {
                    "event_track": event_track,
                    "per_broadcast": {
                        name: binding.record() for name, binding in sorted(bindings.items())
                    },
                }
            }
            if event_track != LEGACY_EVENT_TRACK
            else {}
        ),
        "labels_or_reviewed_inputs": [],
    }


def save_dataset(path: Path, dataset: AutomaticDataset, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        windows=dataset.windows,
        clips=dataset.clips,
        broadcasts=dataset.broadcasts,
        frames=dataset.frames,
        feature_names=np.asarray(dataset.feature_names),
        court_geometry_missing=dataset.missing_court_geometry(),
        court_coordinate_missing=dataset.missing_court_coordinate(),
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def load_dataset(path: Path) -> tuple[AutomaticDataset, dict]:
    """Load a dataset written by :func:`save_dataset`."""

    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        return (
            AutomaticDataset(
                windows=data["windows"],
                clips=data["clips"],
                broadcasts=data["broadcasts"],
                frames=data["frames"],
                feature_names=tuple(str(name) for name in data["feature_names"]),
                court_geometry_missing=data["court_geometry_missing"],
                court_coordinate_missing=data["court_coordinate_missing"],
            ),
            manifest,
        )
