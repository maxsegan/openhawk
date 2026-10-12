"""Transport the calibrated 3D camera projection onto every per-frame court solve.

``court_H_per_frame_v1.npz`` fixes within-point pan/zoom and the net-tape ground-plane
bias, but a 3D flight fit needs a full projection matrix.  A broadcast camera's pan,
tilt, and zoom induce one image homography for every static 3D point.  Therefore, if
``P0`` is any net-height-calibrated projection from the same fixed camera and ``H0`` is
its image-to-court ground mapping, then for a per-frame image-to-court mapping ``Hf``::

    G(0 -> f) = inv(Hf) @ H0
    Pf         = G @ P0

The construction reproduces ``inv(Hf)`` exactly on z=0 while transporting the calibrated
vertical geometry rather than re-solving monocular height independently each frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_cal import draw_projection_audit, project  # noqa: E402
from run_manifest import StageRun  # noqa: E402


def ground_homography_from_P(P: np.ndarray) -> np.ndarray:
    """Image-to-court homography implied by a court-to-image projection matrix."""
    court_to_image = P[:, [0, 1, 3]]
    return np.linalg.inv(court_to_image)


def transport_projection(P0: np.ndarray, Hf: np.ndarray) -> np.ndarray:
    """Transport ``P0`` so its z=0 mapping equals the supplied image-to-court ``Hf``."""
    H0 = ground_homography_from_P(P0)
    image_warp = np.linalg.inv(Hf) @ H0
    Pf = image_warp @ P0
    scale = Pf[2, 3]
    return Pf / scale if abs(scale) > 1e-12 else Pf / np.linalg.norm(Pf)


def ground_reprojection_error(P: np.ndarray, Hf: np.ndarray) -> float:
    points = np.array([[0, 0, 0], [10.97, 0, 0], [0, 23.77, 0], [10.97, 23.77, 0],
                       [5.485, 11.885, 0]], dtype=float)
    via_p = project(P, points)
    via_h = cv2.perspectiveTransform(
        points[None, :, :2].astype(np.float32), np.linalg.inv(Hf)
    )[0]
    return float(np.max(np.linalg.norm(via_p - via_h, axis=1)))


def choose_reference(Ps: np.ndarray, source: np.ndarray | None) -> tuple[np.ndarray, str]:
    if source is not None:
        direct = np.flatnonzero(source.astype(str) == "direct")
        if len(direct):
            return Ps[int(direct[len(direct) // 2])], "transported_direct"
    return Ps[len(Ps) // 2], "transported_fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="processed match directory")
    parser.add_argument("--h-file", default="court_H_per_frame_v1.npz")
    parser.add_argument("--p-file", default="camera_P_per_point.npz")
    parser.add_argument("--output", default="camera_P_per_frame_v1.npz")
    parser.add_argument("--frames-dir", default="rally_frames_50_contact_v2")
    parser.add_argument("--audit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    stage = StageRun(args.out, "camera_P_per_frame", args, seed=args.seed)

    hdata = np.load(os.path.join(args.out, args.h_file))
    pdata = np.load(os.path.join(args.out, args.p_file))
    reference, source = choose_reference(
        pdata["P"], pdata["source"] if "source" in pdata.files else None
    )
    projections = np.stack([transport_projection(reference, H) for H in hdata["H"]])
    errors = np.asarray(
        [ground_reprojection_error(P, H) for P, H in zip(projections, hdata["H"])]
    )
    if float(errors.max()) > 1e-3:
        raise ValueError(f"transported projection ground error {errors.max():.6f}px")
    output_path = os.path.join(args.out, args.output)
    np.savez_compressed(
        output_path,
        clips=hdata["clips"],
        frames=hdata["frames"],
        P=projections,
        source=np.repeat(source, len(projections)),
        ground_error_px=errors,
    )

    audit_path = None
    if args.audit:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(projections), min(args.audit, len(projections)), replace=False)
        panels = []
        for index in chosen:
            clip = str(hdata["clips"][index])
            frame = int(hdata["frames"][index])
            path = os.path.join(args.out, args.frames_dir, clip, f"f_{frame:04d}.jpg")
            image = cv2.imread(path)
            if image is None:
                continue
            panel = draw_projection_audit(image, projections[index], f"{clip} f={frame} {source}")
            panels.append(cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA))
        if panels:
            blank = np.zeros_like(panels[0])
            while len(panels) % 3:
                panels.append(blank)
            montage = np.vstack(
                [np.hstack(panels[i : i + 3]) for i in range(0, len(panels), 3)]
            )
            audit_path = os.path.join(
                args.out, f"camera_P_per_frame_audit_n{args.audit}_seed{args.seed}.jpg"
            )
            cv2.imwrite(audit_path, montage)
    print(
        f"{len(projections)} frame projections / {len(set(hdata['clips'].tolist()))} clips "
        f"| max ground error={errors.max():.3g}px -> {output_path}"
    )
    stage.finish(
        outputs={
            "projections": len(projections),
            "clips": len(set(hdata["clips"].tolist())),
            "source": source,
            "max_ground_error_px": float(errors.max()),
            "output": output_path,
            "audit": audit_path,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
