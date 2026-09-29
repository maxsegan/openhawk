"""Build a ground-truth chart stream for a real MCP match, and demo the scoring loop.

When the CV pipeline can emit a :class:`ChartShot` stream for a match that has HD video, we
score it against ``ground_truth_stream(match_id)``. Until then, the ``__main__`` demo scores a
*simulated* noisy CV chart (random dropped / relabeled / spurious shots) against a real match
so the end-to-end validation loop is exercised on real data.

    python3 cv/validation/ground_truth.py --match <match_id>
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "charting"))
sys.path.insert(0, os.path.dirname(__file__))
csv.field_size_limit(10_000_000)

from tennis_charting.notation import parse_point  # noqa: E402
from scoring import ChartShot, chart_stream, score  # noqa: E402

MCP_DIR = "data/raw/mcp"


def ground_truth_stream(match_id: str, mcp_dir: str = MCP_DIR):
    """Parse a match's points (in order) into a ground-truth ChartShot stream."""
    gender = "m" if "-M-" in match_id else "w"
    rows = []
    for path in glob.glob(os.path.join(mcp_dir, f"charting-{gender}-points-*.csv")):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["match_id"] == match_id:
                    rows.append(r)
    rows.sort(key=lambda r: int(r["Pt"]))
    points = [parse_point(r.get("1st", ""), r.get("2nd", "")) for r in rows]
    return chart_stream(points), len(rows)


def _first_match_id(mcp_dir: str) -> str:
    with open(glob.glob(os.path.join(mcp_dir, "charting-m-points-*.csv"))[0], newline="") as f:
        return next(csv.DictReader(f))["match_id"]


def _simulate_cv(gt, drop=0.10, relabel_dir=0.15, spurious=0.05, seed=0):
    """A stand-in noisy CV charter: drop, relabel-direction, and hallucinate shots."""
    import random
    from dataclasses import replace
    rng = random.Random(seed)
    out = []
    for s in gt:
        if s.kind == "shot" and rng.random() < drop:
            continue
        if s.kind == "shot" and rng.random() < relabel_dir:
            s = replace(s, direction=rng.choice([1, 2, 3]))
        out.append(s)
        if s.kind == "shot" and rng.random() < spurious:
            out.append(ChartShot("shot", "forehand", rng.choice([1, 2, 3]), None, "in_play", "server"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default=None, help="match_id (default: first men's match)")
    ap.add_argument("--mcp-dir", default=MCP_DIR)
    args = ap.parse_args()
    mid = args.match or _first_match_id(args.mcp_dir)
    gt, n_points = ground_truth_stream(mid, args.mcp_dir)
    print(f"match: {mid}\n  points={n_points}  gt shots={sum(s.kind=='shot' for s in gt)}")
    print("\nsimulated noisy-CV chart vs human ground truth:")
    pred = _simulate_cv(gt)
    print("  " + score(pred, gt).summary())
    print("  (perfect charter would score P=R=F1=1.000 and field_acc=1.000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
