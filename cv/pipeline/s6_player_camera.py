"""Optional input-only player roots in the supplied native camera's coordinates.

Use original native tracker boxes when exported, otherwise the declared pose-box
space. A box bottom is a soft ground proxy, not a labeled foot or a calibrated
player truth. Unsupported target rows retain their pixels and original roots but
have no usable derived root; ordinary explicit player-row fallback may apply.

Sided player-box CSVs must record ``keep_unique_admissible_frames`` matching
``SelectionConfig`` (production default on). Stale unflagged boxes fail rather
than mix. Rollback: ``SelectionConfig.keep_unique_admissible_frames = False``.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from cv.experiments.connected_shooting import camera_geometry
from cv.experiments.connected_shooting.labeled_toss_player_anchor import ground_point
from cv.pipeline import provenance, resolution

POLICY = "native_box_bottom_same_camera_v1"
EXTRA_FIELDS = ("s6_original_court_x", "s6_original_court_y", "s6_root_status", "s6_root_proxy")
NATIVE_BOX_COLUMNS = ("x0_native", "y0_native", "x1_native", "y1_native")


def require_replay_inputs(report: dict, pose_csv: Path, cameras: Path) -> None:
    """A cached search with derived roots must replay that exact derived artifact."""
    receipt = report.get("s6_player_camera_coordinates", {})
    if not receipt.get("enabled"):
        return
    sidecar = pose_csv.with_name(pose_csv.name + ".coordinates.json")
    for name, path, expected in (
        ("pose CSV", pose_csv, receipt["derived_pose"]),
        ("coordinate sidecar", sidecar, receipt["derived_coordinate_sidecar"]),
        ("camera", cameras, receipt["sources"]["cameras"]),
    ):
        if not path.is_file() or provenance.file_sha256(path) != expected["sha256"]:
            raise ValueError(f"player camera preparation replay {name} binding changed")


def verify_derivation(inputs: dict[str, Path], row: dict, actual: Path, receipt: dict) -> None:
    """Export replays observation preparation, never trusting only a receipt flag."""
    require_replay_inputs({"s6_player_camera_coordinates": receipt}, actual, inputs["cameras"])
    packet, cameras = (json.loads(inputs[n].read_text()) for n in ("packet", "cameras"))
    with TemporaryDirectory(prefix="s6_player_camera_") as directory:
        _, replay = prepare(inputs, row, packet, cameras, Path(directory), "on")
        for name in ("derived_pose", "derived_coordinate_sidecar"):
            if replay[name]["sha256"] != receipt[name]["sha256"]:
                raise ValueError("player camera derivation does not replay original observations")
        # Output location changes in temporary replay; all source/proxy decisions must not.
        ignored = {"derived_pose", "derived_coordinate_sidecar"}
        if {k: v for k, v in receipt.items() if k not in ignored} != {
            k: v for k, v in replay.items() if k not in ignored
        }:
            raise ValueError("player camera preparation receipt does not replay")


def native_bottom(
    row: dict, pose_image_scale: float, *, native_box_columns: tuple[str, ...] | None = None
) -> tuple[np.ndarray, str]:
    """track_* from pose_player_crop is native, independently of pose image scale."""
    track = [row.get("track_" + k, "") for k in ("x0", "y0", "x1", "y1")]
    if any(v not in (None, "") for v in track):
        values, scale, proxy = track, 1.0, "original_tracker_box_native"
    elif native_box_columns is not None:
        values = [row.get(k, "") for k in native_box_columns]
        scale, proxy = 1.0, "declared_player_box_native"
    else:
        values = [row.get(k, "") for k in ("x0", "y0", "x1", "y1")]
        scale, proxy = pose_image_scale, "pose_box_fallback"
    try:
        box = np.asarray(values, float)
    except (TypeError, ValueError) as error:
        raise ValueError("incomplete player box") from error
    if not np.isfinite(box).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("finite player box and positive pose scale required")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("positive player box dimensions required")
    return scale * np.array([(box[0] + box[2]) / 2, box[3]]), proxy


def derive_row(
    row: dict,
    camera: dict,
    pose_image_scale: float,
    *,
    native_box_columns: tuple[str, ...] | None = None,
) -> dict:
    """No old court root, ball state, scorer or label participates in derivation."""
    if camera.get("status") != "supported" or camera.get("supported") is False:
        raise ValueError("same-frame camera is not explicitly supported")
    uv, proxy = native_bottom(row, pose_image_scale, native_box_columns=native_box_columns)
    P = np.asarray(camera.get("P"), float)
    radial = camera_geometry.radial_row(camera)
    xy = ground_point(P, uv, 0.0, radial)
    projected = camera_geometry.project(
        P[None], np.r_[xy, 0.0][None], None if radial is None else radial[None]
    )[0]
    error = float(np.max(np.abs(projected - uv)))
    if error > 1e-6:
        raise ValueError("ground proxy does not reproduce its native source pixel")
    return dict(
        court_xy_m=xy.tolist(),
        native_bottom_px=uv.tolist(),
        proxy=proxy,
        radial=None if radial is None else radial.tolist(),
        native_reprojection_max_px=error,
    )


def prepare(
    inputs: dict[str, Path], row: dict, packet: dict, cameras: dict, output: Path, mode: str
) -> tuple[dict[str, Path], dict]:
    """Write one derived CSV before both search and replay; OFF is a path no-op."""
    from cv.pipeline.player_side_association import require_keep_unique_admissible_frames

    require_keep_unique_admissible_frames(inputs["pose_csv"])
    if mode not in ("off", "on"):
        raise ValueError("player camera coordinates must be on or off")
    if mode == "off":
        return inputs, {"policy": POLICY, "enabled": False}
    clip = packet["attempts"][0]["point_clip"]
    if cameras.get("clip") != clip:
        raise ValueError("player preparation camera/attempt clip mismatch")
    scale = float(row.get("pose_image_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite pose image scale required")
    by_frame = {}
    for camera in cameras["cameras"]:
        frame = camera.get("frame")
        if not isinstance(frame, (int, float)) or not np.isfinite(frame) or int(frame) != frame:
            raise ValueError("integer native camera epochs required")
        by_frame.setdefault(int(frame), []).append(camera)
    source = inputs["pose_csv"]
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if not {"clip", "frame", "court_x", "court_y"} <= set(fields):
        raise ValueError("sided player CSV with court roots required")
    if set(EXTRA_FIELDS).intersection(fields):
        raise ValueError("player camera preparation requires original observed CSV")
    source_sidecar = source.with_name(source.name + ".coordinates.json")
    space = json.loads(source_sidecar.read_text()) if source_sidecar.is_file() else {}
    native_box_columns = None
    if set(NATIVE_BOX_COLUMNS).intersection(fields):
        native_box_columns, _ = resolution.coordinate_columns_and_size(
            fields,
            space,
            native_columns=NATIVE_BOX_COLUMNS,
            legacy_columns=("x0", "y0", "x1", "y1"),
        )
        # The shared reader validates native dimensions and complete headers.
        # An explicit column mapping must agree too; never reinterpret a
        # declared legacy mirror using the native artifact dimensions.
        declared = space.get("coordinate_columns")
        if declared is not None and (
            not isinstance(declared, dict)
            or declared.get("native_1920x1080") != list(NATIVE_BOX_COLUMNS)
        ):
            raise ValueError("native player box columns conflict with coordinate sidecar")
    if "image_size" in space and "artifact_size" in space:
        for axis in ("width", "height"):
            if not np.isclose(
                float(space["artifact_size"][axis]) * scale,
                float(space["image_size"][axis]),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("pose image scale conflicts with original coordinate sidecar")
    mirror_normalization = None
    if native_box_columns is not None:
        native_size = resolution.manifest_artifact_size(space)
        mirror_size = resolution.manifest_artifact_size(space, legacy_columns=True)
        if mirror_size != native_size:
            # Downstream player/serve readers consume x0..y1 in pose_image_scale.
            # The source's deprecated mirrors can instead be 540p while its
            # authoritative native boxes and pose domain are 1080p. Normalize
            # these aliases once, retaining the original values explicitly.
            mirrors = ("x0", "y0", "x1", "y1")
            archived = tuple(f"s6_original_box_{name}" for name in mirrors)
            if set(archived).intersection(fields):
                raise ValueError("source contains already-normalized player box mirrors")
            for record in rows:
                for name, native, archive in zip(
                    mirrors, native_box_columns, archived, strict=True
                ):
                    record[archive] = record[name]
                    record[name] = record[native]
            fields.extend(archived)
            mirror_normalization = {
                "source_columns": list(native_box_columns),
                "normalized_columns": list(mirrors),
                "archived_columns": list(archived),
                "original_mirror_size": {"width": mirror_size.width, "height": mirror_size.height},
                "normalized_mirror_size": {
                    "width": native_size.width,
                    "height": native_size.height,
                },
                "rows": len(rows),
                "native_observations_changed": False,
                "rule": "declared authoritative native columns replace deprecated box aliases",
            }
    applications = []
    for index, record in enumerate(rows):
        if record["clip"] != clip:
            continue
        evidence = {
            "row": index,
            "frame": record["frame"],
            "side": record.get("side"),
            "original_court_xy": [record["court_x"], record["court_y"]],
        }
        record.update(
            s6_original_court_x=record["court_x"],
            s6_original_court_y=record["court_y"],
            court_x="",
            court_y="",
            s6_root_status="unsupported",
            s6_root_proxy="",
        )
        try:
            match = re.fullmatch(r"f_(\d+)\.jpg", record["frame"])
            if match is None:
                raise ValueError("native source frame name required")
            matches = by_frame.get(int(match[1]), [])
            if len(matches) != 1:
                raise ValueError("one exact native camera row required")
            derived = derive_row(record, matches[0], scale, native_box_columns=native_box_columns)
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            evidence.update(status="unsupported", reason=str(error))
        else:
            record.update(
                court_x=repr(derived["court_xy_m"][0]),
                court_y=repr(derived["court_xy_m"][1]),
                s6_root_status="derived",
                s6_root_proxy=derived["proxy"],
            )
            evidence.update(status="derived", **derived)
        applications.append(evidence)
    if not applications:
        raise ValueError("no source player rows for this attempt")
    destination = output / "prepared_player_pose.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + list(EXTRA_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    sources = {name: provenance.file_record(inputs[name]) for name in ("pose_csv", "cameras")}
    sources["pose_coordinate_sidecar"] = (
        provenance.file_record(source_sidecar) if source_sidecar.is_file() else None
    )
    coordinate_path = destination.with_name(destination.name + ".coordinates.json")
    # Preserve missing coordinate declarations: do not invent dimensions and thereby
    # activate downstream priors which correctly abstained on absent metadata.
    coordinate = space | {
        "artifact": destination.name,
        "s6_player_camera_derivation": {
            "policy": POLICY,
            "sources": sources,
            "pose_image_scale": scale,
            "pixels_changed": False,
            "original_coordinate_space_declared": bool(space),
        },
    }
    if mirror_normalization is not None:
        coordinate["legacy_artifact_size"] = mirror_normalization["normalized_mirror_size"]
        coordinate["s6_box_mirror_normalization"] = mirror_normalization
    coordinate_path.write_text(json.dumps(coordinate, indent=2, allow_nan=False) + "\n")
    receipt = dict(
        policy=POLICY,
        enabled=True,
        sources=sources,
        clip=clip,
        pose_image_scale=scale,
        pixels_changed=False,
        native_clocks_changed=False,
        unsupported_policy="unusable XY; original preserved; explicit supported-row fallback only",
        counts=dict(Counter(item["status"] for item in applications)),
        applications=applications,
        derived_pose=provenance.file_record(destination),
        derived_coordinate_sidecar=provenance.file_record(coordinate_path),
    )
    if mirror_normalization is not None:
        receipt["box_mirror_normalization"] = mirror_normalization
    return inputs | {"pose_csv": destination}, receipt
