"""Resolution and coordinate-space contract for the CV pipeline.

Images are always read from the largest available source.  Existing artifacts may retain
their declared coordinate convention (historically 960x540), but conversion happens
explicitly at the image/artifact boundary. New and migrated interfaces use 1920x1080
coordinates. Nothing in this module infers coordinates from a directory name.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True, order=True)
class FrameSize:
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid frame size {self.width}x{self.height}")

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


NATIVE_SIZE = FrameSize(1920, 1080)
TRACKING_SIZE = NATIVE_SIZE
LEGACY_TRACKING_SIZE = FrameSize(960, 540)
LEGACY_BAD_SHOULD_UPDATE = "LEGACY_BAD_SHOULD_UPDATE"

# Compatibility alias for unmigrated modules. New code must use TRACKING_SIZE or
# NATIVE_SIZE and may expose 540-space values only as read-only mirrors.
CANONICAL_SIZE = LEGACY_TRACKING_SIZE
SUPPORTED_TEST_SIZES = (
    LEGACY_TRACKING_SIZE,
    FrameSize(1280, 720),
    NATIVE_SIZE,
)
PLAYER_BOXES_NATIVE_IDENTITY = "player_boxes.native.v1"
PLAYER_BOXES_NATIVE_SIDED_IDENTITY = "player_boxes.native_sided.v1"
PLAYER_POSE_NATIVE_IDENTITY = "player_pose.native.v1"
PLAYER_NATIVE_BOX_COLUMNS = ("x0_native", "y0_native", "x1_native", "y1_native")
TRACK_NATIVE_POINT_COLUMNS = ("x_native", "y_native")


def meets_native_tracking_minimum(size: FrameSize) -> bool:
    return size.width >= NATIVE_SIZE.width and size.height >= NATIVE_SIZE.height


def resolution_status(size: FrameSize) -> str:
    return (
        "native_1080_or_better" if meets_native_tracking_minimum(size) else LEGACY_BAD_SHOULD_UPDATE
    )


def _scale_matrix(source: FrameSize, target: FrameSize) -> np.ndarray:
    """Homogeneous image-coordinate transform from ``source`` to ``target``."""
    return np.diag(
        [
            target.width / source.width,
            target.height / source.height,
            1.0,
        ]
    )


def scale_points(points, source: FrameSize, target: FrameSize) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    scale = np.array([target.width / source.width, target.height / source.height])
    return values * scale


def scale_boxes(boxes, source: FrameSize, target: FrameSize) -> np.ndarray:
    values = np.asarray(boxes, dtype=float)
    scale = np.array(
        [
            target.width / source.width,
            target.height / source.height,
            target.width / source.width,
            target.height / source.height,
        ]
    )
    return values * scale


def points_inside_image(points, size: FrameSize) -> np.ndarray:
    """Finite point centres supported by the image, not physical court membership.

    Padding and extrapolated coordinates are not observed pixels. A clipped
    object's centre outside this domain needs a separate censored operator.
    """
    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (2,):
        raise ValueError("image point support requires x/y pairs")
    return (
        np.isfinite(values).all(axis=-1)
        & (values >= 0).all(axis=-1)
        & (values < np.array([size.width, size.height])).all(axis=-1)
    )


def normalized_points(points, size: FrameSize) -> np.ndarray:
    return scale_points(points, size, FrameSize(1, 1))


def image_to_world_homography(
    homography: np.ndarray, source: FrameSize, target: FrameSize
) -> np.ndarray:
    """Move an image->world homography from ``source`` pixels to ``target`` pixels."""
    return np.asarray(homography, float) @ _scale_matrix(target, source)


def world_to_image_projection(
    projection: np.ndarray, source: FrameSize, target: FrameSize
) -> np.ndarray:
    """Move a world->image projection from ``source`` pixels to ``target`` pixels."""
    return _scale_matrix(source, target) @ np.asarray(projection, float)


def pixel_length(value: float, source: FrameSize, target: FrameSize) -> float:
    """Scale an isotropic pixel radius/rate; exact for equal-aspect-ratio substrates."""
    factor = np.sqrt((target.width / source.width) * (target.height / source.height))
    return float(value * factor)


def first_frame_path(frames_dir: str | os.PathLike) -> Path:
    root = Path(frames_dir)
    for pattern in ("pt*/f_*.jpg", "pt*/f_*.png", "f_*.jpg", "f_*.png"):
        path = next(root.glob(pattern), None)
        if path is not None:
            return path
    raise FileNotFoundError(f"no frame images below {root}")


def frame_size_for_path(path: str | os.PathLike) -> FrameSize:
    image = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image dimensions: {path}")
    height, width = image.shape[:2]
    return FrameSize(width, height)


def frame_size_for_dir(frames_dir: str | os.PathLike) -> FrameSize:
    return frame_size_for_path(first_frame_path(frames_dir))


def select_highest_resolution_frame_dir(
    candidates: Iterable[str | os.PathLike],
) -> tuple[str, FrameSize]:
    """Return the readable candidate with the largest measured pixel area."""
    measured = []
    for candidate in candidates:
        path = os.path.abspath(os.fspath(candidate))
        if not os.path.isdir(path):
            continue
        try:
            measured.append((frame_size_for_dir(path), path))
        except (FileNotFoundError, ValueError):
            continue
    if not measured:
        raise FileNotFoundError("none of the candidate frame directories contains images")
    size, path = max(measured, key=lambda item: (item[0].area, item[0].width, item[0].height))
    return path, size


def frame_twin_candidates(frames_dir: str | os.PathLike) -> list[str]:
    """Known legacy/native sibling names, without assuming any candidate exists."""
    path = os.path.abspath(os.fspath(frames_dir))
    parent, name = os.path.dirname(path), os.path.basename(path)
    names = [name]
    if not name.endswith("_1080"):
        names.append(f"{name}_1080")
    without_variant = re.sub(r"_contact_v\d+$", "", name)
    if without_variant != name:
        names.extend((without_variant, f"{without_variant}_1080"))
    match = re.match(r"(rally_frames_\d+)", name)
    if match:
        names.append(f"{match.group(1)}_1080")
    return list(dict.fromkeys(os.path.join(parent, candidate) for candidate in names))


def write_coordinate_manifest(
    path: str | os.PathLike,
    *,
    image_size: FrameSize,
    artifact_size: FrameSize,
    source: str,
    extra: dict | None = None,
    subnative_flagged: bool = False,
    subnative_justification: str | None = None,
) -> None:
    payload = {
        "schema": "tennis.coordinate-space.v1",
        "image_size": asdict(image_size),
        "artifact_size": asdict(artifact_size),
        "source": source,
    }
    if subnative_flagged:
        payload["subnative_flagged"] = True
        payload["resolution_status"] = LEGACY_BAD_SHOULD_UPDATE
    if subnative_justification is not None:
        payload["subnative_justification"] = subnative_justification
    if extra:
        payload.update(extra)
    name = payload.get("artifact") or os.path.basename(os.fspath(path)).removesuffix(
        ".coordinates.json"
    )
    if (
        "native" in str(name).lower()
        and artifact_size != NATIVE_SIZE
        and payload.get("interface_space") != "native_1920x1080"
    ):
        raise ValueError(
            f"{name} claims native in its name but declares {artifact_size.label}. "
            "Emit the native interface (write_native_dual_coordinate_manifest) or drop "
            "'native' from the artifact name; a name is never coordinate evidence and this "
            "combination is what made consumers wrong by half a frame."
        )
    if artifact_size != NATIVE_SIZE:
        if payload.get("subnative_flagged") is not True:
            raise ValueError(
                f"new non-native coordinate artifact {artifact_size.label} requires "
                "subnative_flagged: true"
            )
        justification = payload.get("subnative_justification")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError(
                f"new non-native coordinate artifact {artifact_size.label} requires a "
                "subnative_justification"
            )
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def coordinate_manifest_path(artifact: str | os.PathLike) -> Path:
    path = Path(artifact)
    return path.with_suffix(path.suffix + ".coordinates.json")


def read_coordinate_manifest(artifact: str | os.PathLike) -> dict | None:
    path = coordinate_manifest_path(artifact)
    return json.loads(path.read_text()) if path.exists() else None


def resolve_player_boxes(
    match_dir: str | os.PathLike,
    *,
    sided: bool,
    required: bool = True,
) -> Path | None:
    """Resolve player boxes by coordinate-contract identity, never by FPS formatting."""
    root = Path(match_dir)
    identity = PLAYER_BOXES_NATIVE_SIDED_IDENTITY if sided else PLAYER_BOXES_NATIVE_IDENTITY
    candidates: set[Path] = set()
    for sidecar in root.glob("player_boxes*.csv.coordinates.json"):
        manifest = json.loads(sidecar.read_text())
        if manifest.get("artifact_identity") != identity:
            continue
        artifact = Path(os.fspath(sidecar).removesuffix(".coordinates.json"))
        if artifact.is_file():
            candidates.add(artifact)

    if not candidates:
        patterns = (
            ("player_boxes_native_sided_v1.csv", "player_boxes_*_native_sided_v1.csv")
            if sided
            else ("player_boxes_native_v1.csv", "player_boxes_*_native_v1.csv")
        )
        for pattern in patterns:
            for artifact in root.glob(pattern):
                if coordinate_manifest_path(artifact).is_file():
                    candidates.add(artifact)

    if len(candidates) > 1:
        rendered = ", ".join(path.name for path in sorted(candidates))
        raise ValueError(f"ambiguous {identity} artifacts in {root}: {rendered}")
    if candidates:
        artifact = candidates.pop()
        # Read-side validation only: the sidecar must establish a space.  The
        # subnative_flagged rule governs *writing* a new artifact, and applying it here
        # rejected every shipped cohort, which is worse than reading it honestly.
        declared_artifact_space(artifact, columns=("x0", "y0", "x1", "y1"))
        return artifact
    if required:
        raise FileNotFoundError(f"missing coordinate-contract artifact {identity} in {root}")
    return None


def manifest_artifact_size(manifest: dict, *, legacy_columns: bool = False) -> FrameSize:
    """Return the declared size for authoritative or deprecation-cycle columns."""
    key = "legacy_artifact_size" if legacy_columns else "artifact_size"
    size = manifest.get(key)
    if size is None and legacy_columns:
        size = manifest.get("artifact_size")
    if not isinstance(size, dict):
        raise ValueError(f"coordinate manifest lacks {key}")
    return FrameSize(int(size["width"]), int(size["height"]))


def coordinate_columns_and_size(
    fieldnames: Iterable[str],
    manifest: dict,
    *,
    native_columns: tuple[str, ...],
    legacy_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], FrameSize]:
    """Select a complete column set together with its declared coordinate space.

    Native suffixes mean the native-1080 interface, not the source image dimensions.
    A partial native schema is corrupt; an explicitly declared legacy mirror may be read
    when the entire native set is absent. Never give legacy mirrors the native sidecar size.
    """
    available = set(fieldnames)
    native_present = available.intersection(native_columns)
    if native_present:
        if not set(native_columns).issubset(available):
            raise ValueError("partial native coordinate columns")
        if manifest_artifact_size(manifest) != NATIVE_SIZE:
            raise ValueError("native columns require a native coordinate manifest")
        return native_columns, NATIVE_SIZE
    if not set(legacy_columns).issubset(available):
        raise ValueError("missing coordinate columns")
    if manifest.get("interface_space") == "native_1920x1080" and not manifest.get(
        "legacy_artifact_size"
    ):
        raise ValueError("missing native columns and no declared legacy coordinate space")
    return legacy_columns, manifest_artifact_size(manifest, legacy_columns=True)


class MissingCoordinateContract(ValueError):
    """Raised when an artifact carries pixel coordinates but declares no space."""


def declared_artifact_space(
    artifact: str | os.PathLike,
    *,
    columns: Iterable[str] = ("x", "y"),
) -> FrameSize:
    """Return the space ``columns`` of ``artifact`` are written in, from its sidecar only.

    Fail closed.  A filename is never coordinate evidence, and a reader that cannot find a
    declaration must not fall back to a guess.
    """
    manifest = read_coordinate_manifest(artifact)
    if manifest is None:
        raise MissingCoordinateContract(
            f"{coordinate_manifest_path(artifact)} is missing; "
            f"{os.fspath(artifact)} declares no coordinate space"
        )
    names = tuple(columns)
    if names and all(name.endswith("_native") for name in names):
        return NATIVE_SIZE
    return manifest_artifact_size(manifest, legacy_columns=True)


def require_artifact_space(
    artifact: str | os.PathLike,
    expected: FrameSize,
    *,
    columns: Iterable[str] = ("x", "y"),
    consumer: str,
) -> FrameSize:
    """Assert the space a consumer's pixel thresholds are calibrated in, from the sidecar.

    Consumers whose gates, radii and offsets are calibrated in one space must not silently
    read another.  Refusing is the correct answer for them: converting the rows would leave
    the thresholds behind and reproduce the half-frame defect in a new place.
    """
    declared = declared_artifact_space(artifact, columns=columns)
    if declared != expected:
        raise ValueError(
            f"{consumer} reads {'/'.join(columns)} calibrated in {expected.label} but "
            f"{os.fspath(artifact)} declares {declared.label}; migrate the consumer or the "
            "artifact rather than rescaling one of them here"
        )
    return declared


def coordinate_scale(
    artifact: str | os.PathLike,
    *,
    columns: Iterable[str] = ("x", "y"),
    target: FrameSize = NATIVE_SIZE,
) -> tuple[float, float]:
    """Multipliers taking ``columns`` of ``artifact`` into ``target`` pixels.

    This replaces every hard-coded ``2.0 *``.  When the producer migrates the artifact to
    another space the factor follows the sidecar instead of silently halving the frame.
    """
    source = declared_artifact_space(artifact, columns=columns)
    return target.width / source.width, target.height / source.height


def native_point_reader(
    artifact: str | os.PathLike,
    fieldnames: Iterable[str],
    *,
    native_columns: tuple[str, ...] = ("x_native", "y_native"),
    legacy_columns: tuple[str, ...] = ("x", "y"),
    target: FrameSize = NATIVE_SIZE,
):
    """Return ``(read, columns, source_size)`` for one artifact's declared point columns.

    ``read(row)`` yields ``target`` pixels.  Native columns win when the header carries them;
    otherwise the declared legacy space supplies the scale.  Nothing is inferred from the name.
    """
    manifest = read_coordinate_manifest(artifact)
    if manifest is None:
        raise MissingCoordinateContract(
            f"{coordinate_manifest_path(artifact)} is missing; "
            f"{os.fspath(artifact)} declares no coordinate space"
        )
    columns, source = coordinate_columns_and_size(
        fieldnames,
        manifest,
        native_columns=native_columns,
        legacy_columns=legacy_columns,
    )
    scale_x = target.width / source.width
    scale_y = target.height / source.height
    first, second = columns

    def read(row) -> tuple[float, float]:
        return float(row[first]) * scale_x, float(row[second]) * scale_y

    return read, columns, source


def native_box_reader(
    artifact: str | os.PathLike,
    fieldnames: Iterable[str],
    *,
    native_columns: tuple[str, ...] = ("x0_native", "y0_native", "x1_native", "y1_native"),
    legacy_columns: tuple[str, ...] = ("x0", "y0", "x1", "y1"),
    target: FrameSize = NATIVE_SIZE,
):
    """``native_point_reader`` for the four-column player-box schema."""
    manifest = read_coordinate_manifest(artifact)
    if manifest is None:
        raise MissingCoordinateContract(
            f"{coordinate_manifest_path(artifact)} is missing; "
            f"{os.fspath(artifact)} declares no coordinate space"
        )
    columns, source = coordinate_columns_and_size(
        fieldnames,
        manifest,
        native_columns=native_columns,
        legacy_columns=legacy_columns,
    )
    scale = (
        target.width / source.width,
        target.height / source.height,
        target.width / source.width,
        target.height / source.height,
    )

    def read(row) -> tuple[float, float, float, float]:
        return tuple(float(row[name]) * factor for name, factor in zip(columns, scale))

    return read, columns, source


def native_mirror_point(
    x: float,
    y: float,
    source_size: FrameSize = CANONICAL_SIZE,
) -> tuple[float, float]:
    """Convert a legacy point to the native-1080 interface convention."""
    point = scale_points((x, y), source_size, NATIVE_SIZE)
    return float(point[0]), float(point[1])


def write_native_dual_coordinate_manifest(
    path: str | os.PathLike,
    *,
    image_size: FrameSize,
    legacy_size: FrameSize,
    source: str,
    native_columns: Iterable[str],
    legacy_columns: Iterable[str],
    extra: dict | None = None,
) -> None:
    """Declare native authority while documenting temporary legacy mirror columns."""
    metadata = {
        "interface_space": "native_1920x1080",
        "legacy_artifact_size": asdict(legacy_size),
        "coordinate_columns": {
            "native_1920x1080": list(native_columns),
            f"legacy_{legacy_size.label}": list(legacy_columns),
        },
        "deprecation": {
            "legacy_columns_read_only": True,
            "resolution_status": LEGACY_BAD_SHOULD_UPDATE,
            "policy": "one_deprecation_cycle",
        },
    }
    if extra:
        metadata.update(extra)
    write_coordinate_manifest(
        path,
        image_size=image_size,
        artifact_size=NATIVE_SIZE,
        source=source,
        extra=metadata,
    )


def coordinate_manifest_errors(manifest: dict) -> list[str]:
    """Return Resolution Contract v2 declaration violations for one sidecar."""
    if manifest.get("schema") != "tennis.coordinate-space.v1":
        return []
    try:
        artifact_size = manifest_artifact_size(manifest)
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)]
    if artifact_size == NATIVE_SIZE:
        return []
    if manifest.get("subnative_flagged") is not True:
        return [f"non-native artifact space {artifact_size.label} lacks subnative_flagged: true"]
    if manifest.get("resolution_status") != LEGACY_BAD_SHOULD_UPDATE:
        return [
            f"non-native artifact space {artifact_size.label} lacks resolution_status: "
            f"{LEGACY_BAD_SHOULD_UPDATE}"
        ]
    justification = manifest.get("subnative_justification")
    if not isinstance(justification, str) or not justification.strip():
        return [f"non-native artifact space {artifact_size.label} lacks subnative_justification"]
    return []


def propagate_coordinate_manifest(
    sources: Iterable[str | os.PathLike],
    output: str | os.PathLike,
    *,
    extra: dict | None = None,
) -> bool:
    """Copy a shared coordinate contract through a coordinate-preserving transform."""
    manifests = [
        manifest for source in sources if (manifest := read_coordinate_manifest(source)) is not None
    ]
    if not manifests:
        return False
    contracts = {
        (
            manifest["image_size"]["width"],
            manifest["image_size"]["height"],
            manifest["artifact_size"]["width"],
            manifest["artifact_size"]["height"],
        )
        for manifest in manifests
    }
    if len(contracts) != 1:
        raise ValueError(f"coordinate manifest mismatch: {sorted(contracts)}")
    image_width, image_height, artifact_width, artifact_height = contracts.pop()
    first = manifests[0]
    preserved = {
        key: first[key]
        for key in (
            "interface_space",
            "legacy_artifact_size",
            "coordinate_columns",
            "deprecation",
            "subnative_flagged",
            "subnative_justification",
            "resolution_status",
        )
        if key in first
    }
    if extra:
        preserved.update(extra)
    write_coordinate_manifest(
        coordinate_manifest_path(output),
        image_size=FrameSize(image_width, image_height),
        artifact_size=FrameSize(artifact_width, artifact_height),
        source=" + ".join(os.fspath(source) for source in sources),
        extra=preserved,
    )
    return True
