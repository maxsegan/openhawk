"""Sampled per-frame court homography track (S2 hill-climb candidate).

Finding (2026-08-07, s1-s3 lane): the per-point homography's landmark-truth tail
(20/55 views over 3 px, worst 32.8 px) is not a solver limitation — re-solving on the
labeled frame lands every tested tail view at ~2-3 px. The camera moves within the
point, so one H per point is stale at distant frames.

This module solves the canonical topology on frames sampled every ``--stride`` across
each point's clip and emits ``court_H_frame_track_v1.npz`` per match: for every point,
the sampled frame indices, their homographies, and topology scores. Consumers pick the
nearest accepted sample to their frame. Low-scoring solves are dropped (the per-point H
remains the fallback), so the track fails open to current behavior, never worse.

    PYTHONPATH=. python -m cv.pipeline.court_topology_frame_track \
        --frames-root <corpus>/<match>/audit_frames_native_1080 --out <lane>/<match>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline.court_topology import solve_court_h_topology_native

MINIMUM_TOPOLOGY_SCORE = 0.9


def track_match(frames_root: Path, out_dir: Path, stride: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema": "court_h_frame_track_v1", "stride": stride, "points": {}}
    points, frame_lists, h_lists, score_lists = [], [], [], []
    for clip_dir in sorted(frames_root.glob("pt*")):
        frames = sorted(clip_dir.glob("f_*.jpg"))
        if not frames:
            continue
        point = int(clip_dir.name[2:])
        sampled = frames[::stride]
        if frames[-1] not in sampled:
            sampled.append(frames[-1])
        accepted_frames, accepted_h, accepted_scores = [], [], []
        failures = 0
        for frame_path in sampled:
            image = cv2.imread(str(frame_path))
            if image is None:
                failures += 1
                continue
            try:
                solution = solve_court_h_topology_native(image)
            except Exception:
                failures += 1
                continue
            if solution.topology_score < MINIMUM_TOPOLOGY_SCORE:
                failures += 1
                continue
            accepted_frames.append(int(frame_path.stem.removeprefix("f_")))
            accepted_h.append(solution.homography)
            accepted_scores.append(float(solution.topology_score))
        report["points"][clip_dir.name] = {
            "sampled": len(sampled),
            "accepted": len(accepted_frames),
            "failed_or_low_score": failures,
        }
        if accepted_frames:
            points.append(point)
            frame_lists.append(np.asarray(accepted_frames, dtype=np.int32))
            h_lists.append(np.stack(accepted_h))
            score_lists.append(np.asarray(accepted_scores, dtype=np.float64))
    np.savez(
        out_dir / "court_H_frame_track_v1.npz",
        pts=np.asarray(points, dtype=np.int32),
        frames=np.asarray(frame_lists, dtype=object),
        H=np.asarray(h_lists, dtype=object),
        scores=np.asarray(score_lists, dtype=object),
        allow_pickle=True,
    )
    (out_dir / "court_H_frame_track_v1.json").write_text(json.dumps(report, indent=1))
    return report


def nearest_H(npz_path: Path, point: int, frame: int) -> np.ndarray | None:
    """Nearest accepted sampled homography for (point, frame); None if untracked."""
    z = np.load(npz_path, allow_pickle=True)
    pts = [int(v) for v in z["pts"]]
    if point not in pts:
        return None
    i = pts.index(point)
    frames = z["frames"][i]
    if len(frames) == 0:
        return None
    return z["H"][i][int(np.argmin(np.abs(frames - frame)))]


def interpolate_track_H(frames: np.ndarray, homographies: np.ndarray, frame: int) -> np.ndarray:
    """Linearly interpolate normalized homographies between the two nearest samples.

    Valid for the small within-point pans this track exists to absorb; landmark truth
    scores interpolation ahead of nearest-sample (views <=3px: 49/55 vs 43/55).
    """
    if len(frames) == 1 or frame <= frames[0]:
        return homographies[0]
    if frame >= frames[-1]:
        return homographies[-1]
    j = int(np.searchsorted(frames, frame))
    a, b = j - 1, j
    t = (frame - frames[a]) / (frames[b] - frames[a])
    Ha = homographies[a] / homographies[a][2, 2]
    Hb = homographies[b] / homographies[b][2, 2]
    return (1.0 - t) * Ha + t * Hb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=25)
    args = parser.parse_args()
    report = track_match(args.frames_root, args.out, args.stride)
    accepted = sum(p["accepted"] for p in report["points"].values())
    sampled = sum(p["sampled"] for p in report["points"].values())
    print(f"{args.out.name}: accepted {accepted}/{sampled} sampled solves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
