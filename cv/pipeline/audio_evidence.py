"""Materialize the cadence-aware audio evidence cache used by current event proposals."""

from __future__ import annotations

import argparse
import os

from cv.pipeline.audio_features import (
    SAMPLE_RATE,
    evaluation_points,
    load_or_compute_audio_scores,
    load_windows,
)
from cv.pipeline.run_manifest import StageRun


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--point-map", default="point_video_map.csv")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--audio-cache", default="contact_audio_scores_16k_v1.npz")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    stage = StageRun(args.out, "audio_evidence", args, seed=args.seed)
    points = set(evaluation_points(args.out, args.point_map, args.frames_dir, ""))
    windows = load_windows(args.out, args.point_map, points)
    cache = args.audio_cache
    if not os.path.isabs(cache):
        cache = os.path.join(args.out, cache)
    _, report = load_or_compute_audio_scores(args.video, windows, args.sample_rate, cache)
    stage.finish(outputs={"audio_cache": report})
    print(
        f"audio evidence: {report['cache_hits']} hit / {report['cache_misses']} miss -> "
        f"{report['cache_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
