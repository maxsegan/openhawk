"""Audit frozen contacts against native box-bottom proxies, not certified grounded feet.

Research diagnostic only. Compare both sided players at floor/ceil native contact
exposures; do not interpolate pictures, select a player, refit, or accept a point.
The fit camera also projects the proxies, so this is not independent metric truth.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import camera_geometry, contact_reach, human_audit
from cv.pipeline.provenance import file_record, git_record


def plane_proxy(
    camera: np.ndarray, radial: np.ndarray | None, pixel: np.ndarray, height_m: float = 0.0
) -> np.ndarray:
    """Intersect a native box-bottom ray with an explicitly assumed horizontal plane."""
    camera, pixel = np.asarray(camera, float), np.asarray(pixel, float)
    if (
        camera.shape != (3, 4)
        or pixel.shape != (2,)
        or not np.isfinite(camera).all()
        or not np.isfinite(pixel).all()
        or np.any(pixel < 0)
        or np.any(pixel >= [1920, 1080])
        or not math.isfinite(height_m)
    ):
        raise ValueError("finite native camera and in-image root required")
    plane = camera[:, [0, 1, 3]].copy()
    plane[:, 2] += camera[:, 2] * height_m
    if np.linalg.matrix_rank(plane) != 3:
        raise ValueError("nondegenerate ground-plane camera required")
    undistorted = camera_geometry.undistort(
        pixel[None], None if radial is None else np.asarray(radial, float)[None]
    )[0]
    homogeneous = np.linalg.solve(plane, np.r_[undistorted, 1.0])
    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-9:
        raise ValueError("ground ray at infinity")
    return homogeneous[:2] / homogeneous[2]


def ground_proxy(camera, radial, pixel):
    return plane_proxy(camera, radial, pixel)


def neighboring_exposures(frame: float) -> list[int]:
    if not math.isfinite(frame) or frame < 1:
        raise ValueError("finite positive source frame required")
    return sorted({math.floor(frame), math.ceil(frame)})


def compare_contact(frame, xyz, clip, players, cameras) -> list[dict]:
    """Keep both native exposures and sides, including absent or held evidence."""
    xyz = np.asarray(xyz, float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("finite physical contact position required")
    output = []
    for exposure in neighboring_exposures(frame):
        geometry = cameras.get((clip, exposure))
        for side in ("near", "far"):
            rows = players.get((clip, exposure, side), [])
            row = rows[0] if len(rows) == 1 else None
            item = dict(
                native_frame=exposure,
                frame_offset_from_contact=exposure - frame,
                side=side,
                status="held",
                player_row=row,
                conflicting_player_rows=rows if len(rows) > 1 else [],
            )
            if len(rows) > 1:
                item["reason"] = "ambiguous_player_observations"
            elif row is None:
                item["reason"] = "missing_player_observation"
            elif geometry is None or not geometry["reliable"]:
                item["reason"] = "missing_or_held_metric_camera"
            else:
                try:
                    pixel = np.array([float(row[k]) for k in ("root_x_native", "root_y_native")])
                    proxy = ground_proxy(geometry["P"], geometry["radial"], pixel)
                    sensitivity = []
                    for height in (0.0, 0.25, 0.5, 0.75, 1.0):
                        alternative = plane_proxy(geometry["P"], geometry["radial"], pixel, height)
                        sensitivity.append(
                            dict(
                                assumed_box_bottom_height_m=height,
                                proxy_xy_m=alternative.tolist(),
                                horizontal_distance_m=float(np.linalg.norm(xyz[:2] - alternative)),
                            )
                        )
                    reach = []
                    for height in (0.0, 0.5, 1.0, 1.5):
                        try:
                            upper = np.r_[
                                plane_proxy(geometry["P"], geometry["radial"], pixel, height),
                                height,
                            ]
                            lower = np.r_[proxy, 0.0]
                            depths = (
                                np.asarray(geometry["P"])[2]
                                @ np.c_[np.array([lower, upper]), np.ones(2)].T
                            )
                            if (
                                not np.isfinite(depths).all()
                                or np.any(np.abs(depths) < 1e-9)
                                or depths[0] * depths[1] <= 0
                            ):
                                raise ValueError(
                                    "root-height interval crosses camera projection singularity"
                                )
                            reach.append(
                                {
                                    "status": "measured_conditional_bound",
                                    **contact_reach.minimum_reach(xyz, lower, upper),
                                }
                            )
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            reach.append(
                                {
                                    "status": "held",
                                    "assumed_root_height_interval_m": [0.0, height],
                                    "reason": str(exc),
                                }
                            )
                    item.update(
                        status="measured_proxy_only",
                        ground_proxy_xy_m=proxy.tolist(),
                        horizontal_distance_m=float(np.linalg.norm(xyz[:2] - proxy)),
                        ground_proxy_distance_3d_m=float(np.linalg.norm(xyz - np.r_[proxy, 0])),
                        height_sensitivity=sensitivity,
                        continuous_reach_sensitivity=reach,
                    )
                except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                    item.update(reason="invalid_proxy_geometry", error=str(exc))
            output.append(item)
    return output


def reach_table(evidence: list[dict]) -> str:
    """Show conditional reach minima without selecting a side or pose hypothesis."""
    rows = [
        "<table border='1' cellpadding='5'><caption>Minimum 3D root-to-contact reach (m), by assumed maximum box-bottom height</caption><tr><th>Exposure / side</th><th>0 m</th><th>0.5 m</th><th>1 m</th><th>1.5 m</th></tr>"
    ]
    for item in evidence:
        values = item.get("continuous_reach_sensitivity", [])
        cells = (
            [
                f"{row['minimum_root_to_contact_distance_m']:.3f}"
                if row["status"] == "measured_conditional_bound"
                else "held"
                for row in values
            ]
            if values
            else ["held"] * 4
        )
        rows.append(
            f"<tr><td>{item['native_frame']} / {html.escape(item['side'])}</td>"
            + "".join(f"<td>{value}</td>" for value in cells)
            + "</tr>"
        )
    return "".join([*rows, "</table>"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--player-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    captured = {}

    def bind(path):
        path = Path(path).resolve()
        record = file_record(path)
        if path in captured and captured[path] != record:
            raise ValueError("diagnostic input changed during reading")
        captured[path] = record
        return record

    def read_json(path):
        bind(path)
        return json.loads(path.read_text())

    for path in (
        Path(__file__),
        Path(camera_geometry.__file__),
        Path(contact_reach.__file__),
        Path(human_audit.__file__),
    ):
        bind(path)
    player_report = read_json(args.player_report)
    if (
        player_report["status"] != "complete"
        or player_report["schema"] != "archived_player_replay_v1"
    ):
        raise ValueError("completed explicit player replay required")
    track = args.player_report.parent / "player_tracks_native.csv"
    sidecar = track.with_name(track.name + ".coordinates.json")
    if any(bind(p) not in player_report["outputs"] for p in (track, sidecar)):
        raise ValueError("player output bytes differ from frozen replay")
    metadata = read_json(sidecar)
    if any(metadata[k] != {"width": 1920, "height": 1080} for k in ("image_size", "artifact_size")):
        raise ValueError("native player coordinate contract required")
    with track.open(newline="") as handle:
        players = {}
        for row in csv.DictReader(handle):
            key = row["clip"], int(row["frame"]), row["side"]
            if key[2] not in {"near", "far"}:
                raise ValueError("explicit near/far player observations required")
            players.setdefault(key, []).append(row)
    results = []
    bindings = []
    for report_path in args.report:
        report = read_json(report_path)
        if report["status"] != "complete" or report["schema"] != "owner_ground_attempt_replay_v1":
            raise ValueError("completed explicit ground replay required")
        bindings += report["inputs"]
        config = report["configuration"]
        source_root = Path(config["source_root"])
        source_match = source_root / config["match_id"]
        player_config = player_report["configuration"]
        if (
            Path(player_config["boxes"]).resolve().parent != source_match.resolve()
            or Path(player_config["frame_court"]).resolve().parent
            != Path(config["camera_report"]).resolve().parent
        ):
            raise ValueError("player source match/court differs from fitted evidence")
        manifest = read_json(source_root / "manifest.json")
        match = next(m for m in manifest["matches"] if m["id"] == config["match_id"])
        if metadata["fps"] != match["source_fps"] or player_config["fps"] != metadata["fps"]:
            raise ValueError("player and source cadence disagree")
        camera_path = Path(config["camera_report"]).parent / "camera_P_per_frame_v1.npz"
        bind(camera_path)
        camera_rows = {}
        with np.load(camera_path, allow_pickle=False) as camera:
            for i, (clip, frame) in enumerate(zip(camera["clips"], camera["frames"], strict=True)):
                key = str(clip), int(frame)
                if key in camera_rows:
                    raise ValueError("unique metric camera exposures required")
                camera_rows[key] = {
                    "P": camera["P"][i],
                    "reliable": bool(camera["reliable"][i]),
                    "radial": np.r_[camera["k1"][i], camera["dist_center"][i]],
                }
        for summary in report["results"]:
            path = report_path.parent / f"{summary['case']}_{summary['rebound_mode']}.json"
            if bind(path) not in report["outputs"]:
                raise ValueError("fit bytes differ from frozen report")
            fit = read_json(path)
            item = dict(
                report=str(report_path), case=fit["case"], status=fit["status"], contacts=[]
            )
            if fit["status"] == "measured_not_quality_accepted":
                for frame, flight in zip(
                    fit["physical_boundaries"][:-1], fit["dense_flights"], strict=True
                ):
                    evidence = compare_contact(
                        frame, flight["start_xyz"], fit["clip"], players, camera_rows
                    )
                    images = [
                        Path(config["source_root"])
                        / config["match_id"]
                        / "audit_frames_native_1080"
                        / fit["clip"]
                        / f"f_{f:04d}.jpg"
                        for f in neighboring_exposures(frame)
                    ]
                    for image_path in images:
                        bind(image_path)
                    item["contacts"].append(
                        dict(
                            frame=frame,
                            xyz_m=flight["start_xyz"],
                            evidence=evidence,
                            native_images=list(map(str, images)),
                        )
                    )
            results.append(item)
    if not all(human_audit.source_identity_matches(r) for r in bindings):
        raise ValueError("frozen replay source changed")
    sources, records = list(captured), list(captured.values())
    # This diagnostic has no gate: raw rows and both sides remain available for review.
    args.output.mkdir(parents=True)
    document = dict(
        schema="frozen_contact_player_proxy_audit_v2",
        status="complete",
        code=git_record(Path.cwd()),
        scope=__doc__,
        independent_metric_accuracy=False,
        optimization_performed=False,
        complete_points_accepted=0,
        assumed_box_bottom_heights_m=[0.0, 0.25, 0.5, 0.75, 1.0],
        height_sensitivity_scope="Uncalibrated diagnostic hypotheses, not measured foot heights, validated bounds, or a reach gate",
        continuous_reach_height_caps_m=[0.0, 0.5, 1.0, 1.5],
        continuous_reach_scope="Full-3D minimum over each assumed box-bottom ray segment; includes no root-pixel, camera, player-motion or contact-location error allowance. The minimizing height is not measured pose. No player or height is selected for inference.",
        player_ambiguities=[
            dict(clip=k[0], frame=k[1], side=k[2], rows=v) for k, v in players.items() if len(v) > 1
        ],
        inputs=[dict(resolved_path=str(p.resolve()), record=r) for p, r in zip(sources, records)],
        results=results,
    )
    body = [
        "<html><meta charset='utf-8'><h1>Frozen contact / player proxies</h1>",
        "<p>Both players, both neighboring native exposures. Box bottoms are not grounded feet; the shared camera is not independent metric truth. No fit or point acceptance is changed.</p>",
        "<p>Heights 0–1 m are sensitivity hypotheses, not measured jumps or validated reach bounds. No height or player is selected from fit agreement.</p>",
        "<p>Continuous reach uses full 3D distance over height intervals 0–0/0.5/1/1.5 m. These are conditional geometry bounds, not anatomical gates. Camera/pixel errors and movement between contact and native exposure are not bounded here.</p>",
    ]
    for result in results:
        body.append(
            f"<h2>{html.escape(result['case'])}</h2><p>{html.escape(result['report'])}: {html.escape(result['status'])}</p>"
        )
        for contact in result["contacts"]:
            body.append(
                f"<h3>Contact {contact['frame']}: {contact['xyz_m']}</h3>"
                + reach_table(contact["evidence"])
                + f"<details><summary>All source rows and hypotheses</summary><pre>{html.escape(json.dumps(contact['evidence'], indent=2))}</pre></details>"
            )
            for path in contact["native_images"]:
                link = html.escape(os.path.relpath(path, args.output), quote=True)
                body.append(
                    f"<a href='{link}'>Native source</a><img style='max-width:100%' src='{link}'>"
                )
    body.append("</html>")
    if records != [file_record(p) for p in sources]:
        raise ValueError("diagnostic input changed")
    (args.output / "index.html").write_text("\n".join(body) + "\n")
    (args.output / "report.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
