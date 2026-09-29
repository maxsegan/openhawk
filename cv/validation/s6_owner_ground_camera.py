"""Fit explicit owner-ground-conditioned S6 camera controls, not automatic cameras.

Question: can a rigid, centered square-pixel pinhole camera explain existing
manual court landmarks? Leave-one-landmark-out errors test planar consistency,
not airborne accuracy. No net, ball, event or 3D trajectory truth enters fitting.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import html
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cv.pipeline import camera_cal, paths, provenance


def projection(parameters: np.ndarray) -> np.ndarray:
    """Seven parameters: rotation vector, translation, log focal length."""
    focal = np.exp(parameters[6])
    intrinsic = np.array([[focal, 0, 960], [0, focal, 540], [0, 0, 1]])
    return intrinsic @ np.c_[Rotation.from_rotvec(parameters[:3]).as_matrix(), parameters[3:6]]


def project(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[xyz, np.ones(len(xyz))] @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= 1e-9):
        raise ValueError("points must lie strictly in front of the physical camera")
    return homogeneous[:, :2] / homogeneous[:, 2:]


def fit_ground(xyz: np.ndarray, pixels: np.ndarray) -> dict:
    xyz, pixels = np.asarray(xyz, float), np.asarray(pixels, float)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(xyz) < 6
        or pixels.shape != (len(xyz), 2)
        or not np.isfinite(xyz).all()
        or not np.isfinite(pixels).all()
        or np.any(xyz[:, 2] != 0)
        or len(np.unique(xyz, axis=0)) != len(xyz)
        or np.linalg.matrix_rank(xyz[:, :2] - xyz[:, :2].mean(axis=0)) != 2
    ):
        raise ValueError("six or more unique finite non-collinear ground landmarks required")
    homography, _ = cv2.findHomography(xyz[:, :2], pixels, method=0)
    if homography is None:
        raise ValueError("ground homography is degenerate")
    seed = camera_cal.h_to_projection(homography, w=1920, h=1080)
    if seed is None:
        raise ValueError("ground homography has no centered square-pixel initialization")
    _, _, rotation, translation, focal = seed
    initial = np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation, np.log(focal)]
    lower = np.r_[np.full(6, -np.inf), np.log(400)]
    upper = np.r_[np.full(6, np.inf), np.log(40000)]
    initial[-1] = np.clip(initial[-1], lower[-1] + 1e-8, upper[-1] - 1e-8)

    def residual(parameters):
        return (project(projection(parameters), xyz) - pixels).ravel()

    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=500,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    matrix = projection(solved.x)
    center = -np.linalg.solve(matrix[:, :3], matrix[:, 3])
    if not solved.success or not np.isfinite(center).all() or center[2] <= 0:
        raise ValueError("physical camera fit did not converge above the court")
    errors = np.linalg.norm(project(matrix, xyz) - pixels, axis=1)
    return dict(
        P=matrix.tolist(),
        parameters=solved.x.tolist(),
        camera_center_m=center.tolist(),
        focal_native_px=float(np.exp(solved.x[6])),
        native_rms_px=float(np.sqrt(np.mean(errors**2))),
        native_max_px=float(errors.max()),
        landmark_errors_px=errors.tolist(),
        evaluations=int(solved.nfev),
        focal_bound_active=bool(solved.active_mask[-1]),
    )


def evaluate(labels: list[dict]) -> dict:
    if len({r["landmark_id"] for r in labels}) != len(labels) or any(
        r["coordinate_convention"] != "itf_outside_edge_v1" for r in labels
    ):
        raise ValueError("unique outside-edge landmark IDs required")
    xyz = np.array([[float(r["court_x_m"]), float(r["court_y_m"]), 0] for r in labels])
    pixels = 2 * np.array(
        [[float(r["corrected_x540"]), float(r["corrected_y540"])] for r in labels]
    )
    fitted = fit_ground(xyz, pixels)
    omitted = []
    for i, label in enumerate(labels):
        mask = np.arange(len(labels)) != i
        try:
            control = fit_ground(xyz[mask], pixels[mask])
            error = float(
                np.linalg.norm(project(np.array(control["P"]), xyz[i : i + 1])[0] - pixels[i])
            )
            omitted.append(
                dict(landmark_id=label["landmark_id"], status="measured", native_error_px=error)
            )
        except ValueError as exc:
            omitted.append(dict(landmark_id=label["landmark_id"], status="held", reason=str(exc)))
    return dict(
        fit=fitted,
        ground_labels=labels,
        leave_one_landmark_out=omitted,
        temporal_camera_validated=False,
        airborne_metric_accuracy_certified=False,
        automatic_inference_eligible=False,
    )


def resolve_record(record: dict) -> Path:
    roots = {
        "repository": paths.REPO_ROOT,
        "TENNIS_DATA_ROOT": paths.data_root(),
        "TENNIS_TRACKER_ROOT": paths.tracker_root(),
    }
    if record["path_base"] not in roots:
        raise ValueError("configured source root required")
    root = roots[record["path_base"]].resolve()
    source = (root / record["path"]).resolve()
    if not source.is_relative_to(root) or provenance.file_record(source) != record:
        raise ValueError("source-bound packet bytes changed or escaped their root")
    return source


def overlay(row: dict, image: Path) -> str:
    """Existing source image, owner marks, and explicitly predicted geometry."""
    matrix = np.array(row["fit"]["P"])
    labels = row["ground_labels"]
    elements = []
    for label in labels:
        x, y = 2 * np.array([float(label["corrected_x540"]), float(label["corrected_y540"])])
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="8" fill="none" stroke="#00ff90" stroke-width="3"/>'
        )
    lines = [((0, 0, 0), (10.97, 0, 0)), ((0, 23.77, 0), (10.97, 23.77, 0))]
    lines += [((x, 0, 0), (x, 23.77, 0)) for x in (0, 1.37, 9.6, 10.97)]
    lines += [((1.37, y, 0), (9.6, y, 0)) for y in (5.485, 18.285)]
    for line in lines:
        p = project(matrix, np.array(line))
        elements.append(
            f'<polyline points="{p[0, 0]},{p[0, 1]} {p[1, 0]},{p[1, 1]}" fill="none" stroke="#ff6060" stroke-width="2"/>'
        )
    net_x = np.linspace(camera_cal.NET_POST_X[0], camera_cal.NET_POST_X[1], 101)
    net = project(
        matrix, np.array([[x, camera_cal.NET_Y, camera_cal.net_height_at_x(x)] for x in net_x])
    )
    points = " ".join(f"{x},{y}" for x, y in net)
    elements.append(
        f'<polyline points="{points}" fill="none" stroke="#ffd34e" stroke-width="3" stroke-dasharray="10 8"/>'
    )
    encoded = base64.b64encode(image.read_bytes()).decode()
    return (
        '<svg viewBox="0 0 1920 1080" style="width:100%"><image width="1920" height="1080" href="data:image/jpeg;base64,'
        + encoded
        + '"/>'
        + "".join(elements)
        + "</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new camera-control directory")
    files = [
        args.packet,
        Path(__file__),
        Path(camera_cal.__file__),
        Path(paths.__file__),
        Path(provenance.__file__),
    ]
    records = [provenance.file_record(p) for p in files]
    packet = json.loads(args.packet.read_text())
    if (
        packet.get("schema") != "s6_sparse_owner_input_packet_v1"
        or packet.get("human_derived") is not True
    ):
        raise ValueError("explicit owner input packet required")
    files.extend(resolve_record(r) for r in packet["inputs"])
    records.extend(packet["inputs"])
    rows, views, seen = [], [], set()
    for attempt in packet["attempts"]:
        groups = defaultdict(list)
        for label in attempt["court_ground_labels"]:
            groups[int(label["frame"])].append(label)
        for frame, labels in sorted(groups.items()):
            key = (attempt["match_id"], attempt["point_clip"], frame)
            if key in seen:
                continue
            seen.add(key)
            row = dict(match_id=key[0], clip=key[1], frame=frame, **evaluate(labels))
            source_suffix = f"/{key[0]}/audit_frames_native_1080/{key[1]}/f_{frame:04d}.jpg"
            images = [p for p in files if p.as_posix().endswith(source_suffix)]
            if len(set(images)) != 1:
                raise ValueError("one source-bound native court image required")
            rows.append(row)
            views.append(
                f"<h2>{html.escape(key[1])} · frame {frame}</h2>"
                + overlay(row, images[0])
                + "<pre>"
                + html.escape(
                    json.dumps({k: v for k, v in row.items() if k != "ground_labels"}, indent=2)
                )
                + "</pre>"
            )
    if not rows or records != [provenance.file_record(p) for p in files]:
        raise ValueError("empty control or changed input bytes")
    result = dict(
        schema="s6_owner_ground_camera_control_v1",
        human_derived=True,
        assumptions=[
            "native 1920x1080",
            "principal point fixed at (960,540)",
            "square pixels, zero skew, zero radial distortion",
            "rigid rotation and translation",
            "each labeled frame fitted independently",
            "all supplied ground landmarks retained with equal weight",
            "no camera is extended to other frames",
        ],
        scope=__doc__,
        inputs=records,
        code=provenance.git_record(paths.REPO_ROOT),
        cameras=rows,
        complete_real_points_accepted=0,
    )
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>S6 owner-ground camera control</title><body style="background:#17202a;color:white;max-width:1400px;margin:auto;font-family:system-ui"><h1>Owner-ground-conditioned camera control</h1><p>Evaluation only. Green: owner ground marks. Red: fitted court. Yellow dashed: predicted net, not a verified net label. Fits apply only at their labeled frame. Planar agreement does not certify airborne depth.</p>'
        + "".join(views)
        + "</body>"
    )
    print(
        json.dumps(
            {"cameras": len(rows), "native_rms_px": [r["fit"]["native_rms_px"] for r in rows]}
        )
    )


if __name__ == "__main__":
    main()
