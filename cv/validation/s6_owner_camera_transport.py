"""Transport one owner-ground camera to sparse S6 observation frames.

Evaluation only. Static-scene registration proposes pan/tilt/roll/zoom; every
output is refitted to a rigid square-pixel camera with the anchor center fixed.
No unsupported frame inherits a fallback camera. This does not certify depth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cv.pipeline import court_topology, paths, provenance
from cv.validation import s6_owner_ground_camera as ground

GROUND = np.array(
    [[x, y, 0] for x in (0, 1.37, 5.485, 9.6, 10.97) for y in (0, 5.485, 11.885, 18.285, 23.77)]
)


def fit_view(anchor: dict, target_h: np.ndarray) -> dict:
    """Project a registration proposal onto a fixed-center physical camera."""
    center = np.array(anchor["camera_center_m"])
    seed = np.array(anchor["parameters"])[[0, 1, 2, 6]]
    target_p = np.linalg.inv(target_h)
    q = np.c_[GROUND[:, :2], np.ones(len(GROUND))] @ target_p.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-9):
        raise ValueError("invalid target ground projection")
    pixels = q[:, :2] / q[:, 2:]

    def parameters(v):
        rotation = Rotation.from_rotvec(v[:3]).as_matrix()
        return np.r_[v[:3], -rotation @ center, v[3]]

    def residual(v):
        return (ground.project(ground.projection(parameters(v)), GROUND) - pixels).ravel()

    fit = least_squares(
        residual,
        seed,
        bounds=(np.r_[[-np.inf] * 3, np.log(400)], np.r_[[np.inf] * 3, np.log(40000)]),
        x_scale="jac",
        max_nfev=100,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not fit.success or fit.active_mask[-1]:
        raise ValueError("physical view fit failed or focal bound active")
    delta = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    # Fixed diagnostic tolerances, not calibrated spatial accuracy thresholds.
    rms, maximum = float(np.sqrt(np.mean(delta**2))), float(delta.max())
    if rms > 4 or maximum > 8:
        raise ValueError(
            f"registration conflicts with fixed-center camera: rms={rms:.3f}, max={maximum:.3f}px"
        )
    return dict(
        P=ground.projection(parameters(fit.x)).tolist(),
        parameters=parameters(fit.x).tolist(),
        native_warp_rms_px=rms,
        native_warp_max_px=maximum,
    )


@lru_cache(maxsize=1)
def reference_image(path: str):
    image = cv2.imread(path)
    if image is None or image.shape[:2] != (1080, 1920):
        raise ValueError("native reference image required")
    return image


def transport(job: tuple) -> dict:
    frame, source, reference, anchor, reference_frame = job
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    try:
        if frame == reference_frame:
            return dict(
                frame=frame,
                status="supported",
                source="owner_ground_anchor",
                P=anchor["P"],
                parameters=anchor["parameters"],
                native_warp_rms_px=0.0,
                native_warp_max_px=0.0,
            )
        target = cv2.imread(source)
        if target is None or target.shape[:2] != (1080, 1920):
            raise ValueError("native target image required")
        projection = np.array(anchor["P"])
        h = np.linalg.inv(projection[:, [0, 1, 3]])
        target_h, evidence = court_topology.transfer_court_homography(
            target, reference_image(reference), h
        )
        return dict(
            frame=frame,
            status="supported",
            source="registered_fixed_center_view",
            registration=evidence,
            **fit_view(anchor, target_h),
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return dict(frame=frame, status="held", reason=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.jobs <= 16:
        raise ValueError("new output directory and 1–16 CPU workers required")
    files = [
        args.packet,
        args.anchors,
        Path(__file__),
        Path(ground.__file__),
        Path(court_topology.__file__),
        Path(court_topology.court.__file__),
        Path(court_topology.res.__file__),
        Path(paths.__file__),
        Path(provenance.__file__),
    ]
    records = [provenance.file_record(p) for p in files]
    packet, anchors = (json.loads(p.read_text()) for p in (args.packet, args.anchors))
    if (
        packet.get("schema") != "s6_sparse_owner_input_packet_v1"
        or anchors.get("schema") != "s6_owner_ground_camera_control_v1"
        or anchors.get("human_derived") is not True
    ):
        raise ValueError("explicit source-bound owner packet and ground camera control required")
    if provenance.file_record(args.packet) not in anchors["inputs"]:
        raise ValueError("ground cameras must use the same owner packet")
    for document in (packet, anchors):
        files.extend(ground.resolve_record(r) for r in document["inputs"])
        records.extend(document["inputs"])
    chosen = [
        r
        for r in anchors["cameras"]
        if r["clip"] == args.clip and r["frame"] == args.reference_frame
    ]
    attempts = [a for a in packet["attempts"] if a["point_clip"] == args.clip]
    if len(chosen) != 1 or len(attempts) != 1 or chosen[0]["match_id"] != attempts[0]["match_id"]:
        raise ValueError("one explicit anchor and active attempt required")
    anchor, attempt = chosen[0], attempts[0]
    if anchor["fit"]["focal_bound_active"]:
        raise ValueError("anchor focal bound active")
    source_images = {
        int(p.stem.removeprefix("f_")): p
        for p in files
        if p.parent.name == args.clip and p.suffix == ".jpg"
    }
    frames = sorted({r["frame"] for r in attempt["owner_ball_labels"]} | {args.reference_frame})
    if not set(frames) <= set(source_images):
        raise ValueError("all owner exposures and anchor must be source-bound")
    jobs = [
        (
            f,
            str(source_images[f]),
            str(source_images[args.reference_frame]),
            anchor["fit"],
            args.reference_frame,
        )
        for f in frames
    ]
    args.output.mkdir(parents=True)
    rows = []
    with (
        ProcessPoolExecutor(max_workers=args.jobs) as pool,
        (args.output / "progress.jsonl").open("w") as handle,
    ):
        for row in pool.map(transport, jobs):
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if len(rows) % 25 == 0:
                print(f"{len(rows)}/{len(frames)} camera frames complete", flush=True)
    if records != [provenance.file_record(p) for p in files]:
        raise ValueError("transport inputs changed")
    result = dict(
        schema="s6_owner_camera_transport_v1",
        human_derived=True,
        match_id=attempt["match_id"],
        clip=args.clip,
        reference_frame=args.reference_frame,
        inputs=records,
        code=provenance.git_record(paths.REPO_ROOT),
        cameras=rows,
        supported=sum(r["status"] == "supported" for r in rows),
        total=len(rows),
        scope=__doc__,
        configuration=dict(
            jobs=args.jobs,
            maximum_warp_rms_px=4.0,
            maximum_warp_error_px=8.0,
            random_seed=0,
            fallback="none",
            interpolated_frames=0,
        ),
        airborne_metric_accuracy_certified=False,
        automatic_inference_eligible=False,
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    panels = []
    for frame in sorted({frames[0], frames[-1], args.reference_frame, *frames[::50]}):
        row = next(r for r in rows if r["frame"] == frame)
        if row["status"] == "supported":
            view = ground.overlay(dict(fit=row, ground_labels=[]), source_images[frame])
        else:
            view = "Held; no fallback camera."
        panels.append(f"<h2>Native frame {frame}</h2>" + view)
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>S6 owner camera transport</title><body style="max-width:1400px;margin:auto;font-family:system-ui"><h1>Human-conditioned camera transport</h1><p>Evaluation only. Red: fitted court. Yellow: predicted net, not human net truth. No interpolation or fallback.</p>'
        + "".join(panels)
        + "</body>"
    )
    print(json.dumps({"supported": result["supported"], "total": result["total"]}), flush=True)


if __name__ == "__main__":
    main()
