"""Resolution Contract v2 writer for deprecation-cycle ball-track CSVs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

from cv.pipeline import resolution as res


TRACK_NATIVE_COLUMNS = ("x_native", "y_native")
TRACK_LEGACY_COLUMNS = ("x", "y")


def _shared_source_contract(sources: Iterable[Path]) -> tuple[res.FrameSize, res.FrameSize]:
    manifests = []
    for source in sources:
        manifest = res.read_coordinate_manifest(source)
        if manifest is None:
            raise ValueError(f"coordinate manifest missing for track source: {source}")
        manifests.append(manifest)
    if not manifests:
        raise ValueError("track artifact requires at least one coordinate-declared source")
    image_sizes = {
        res.FrameSize(
            int(manifest["image_size"]["width"]),
            int(manifest["image_size"]["height"]),
        )
        for manifest in manifests
    }
    legacy_sizes = {
        res.manifest_artifact_size(manifest, legacy_columns=True) for manifest in manifests
    }
    if len(image_sizes) != 1 or len(legacy_sizes) != 1:
        raise ValueError(
            "track source coordinate mismatch: "
            f"image={sorted(size.label for size in image_sizes)} "
            f"legacy={sorted(size.label for size in legacy_sizes)}"
        )
    return image_sizes.pop(), legacy_sizes.pop()


def rows_with_native_mirror(rows: list[dict], legacy_size: res.FrameSize) -> list[dict]:
    """Add authoritative native coordinates without changing legacy x/y values."""
    output = []
    for row in rows:
        x_native, y_native = res.native_mirror_point(
            float(row["x"]),
            float(row["y"]),
            legacy_size,
        )
        output.append(
            {
                **row,
                "x_native": x_native,
                "y_native": y_native,
            }
        )
    return output


def write_candidate_artifact(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
    *,
    image_size: res.FrameSize,
    legacy_size: res.FrameSize,
    source: str,
) -> None:
    """Write detector candidates in the dual schema with a native-authoritative sidecar.

    Detector peaks are produced in the tracker's 960x540 convention; the legacy ``x``/``y``
    columns keep that convention unchanged while ``x_native``/``y_native`` carry the native
    interface, so a ``native``-named candidate file never again declares a sub-native space.
    """
    mirrored = rows_with_native_mirror(rows, legacy_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *TRACK_NATIVE_COLUMNS])
        writer.writeheader()
        writer.writerows(mirrored)
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=image_size,
        legacy_size=legacy_size,
        source=source,
        native_columns=TRACK_NATIVE_COLUMNS,
        legacy_columns=TRACK_LEGACY_COLUMNS,
        extra={"artifact": path.name},
    )


def write_track_artifact(path: Path, rows: list[dict], sources: Iterable[Path]) -> None:
    """Write a dual-column track with a native-authoritative coordinate sidecar."""
    source_paths = list(sources)
    image_size, legacy_size = _shared_source_contract(source_paths)
    mirrored = rows_with_native_mirror(rows, legacy_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_fieldnames = list(rows[0]) if rows else ["clip", "frame", "x", "y", "track_id"]
    fieldnames = [*legacy_fieldnames, *TRACK_NATIVE_COLUMNS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mirrored)
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=image_size,
        legacy_size=legacy_size,
        source=" + ".join(os.fspath(source) for source in source_paths),
        native_columns=TRACK_NATIVE_COLUMNS,
        legacy_columns=TRACK_LEGACY_COLUMNS,
        extra={"artifact": path.name},
    )
