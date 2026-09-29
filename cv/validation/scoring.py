"""Shot-for-shot scoring of a CV-produced chart against the MCP human chart.

This is the validation loop that makes auto-charting publishable: because every MCP match was
charted from video, matches with both a human chart and HD video let us score the CV charter
field-by-field against ground truth. This module is **video-agnostic** — it compares two
symbolic charts (streams of shots), so it is defined and tested before any vision code exists.

Approach: flatten each point to a token stream (shots + a POINT_BREAK marker), edit-distance
align predicted vs ground-truth (handling inserted / dropped shots and mis-segmented point
boundaries), then report shot detection precision/recall/F1 and per-field accuracy
(shot type / direction / depth / outcome) on the aligned shots.

Ground truth is produced by :mod:`tennis_charting.notation`; a CV pipeline must emit the same
:class:`ChartShot` stream.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "charting"))

from tennis_charting.notation import Point  # noqa: E402

POINT_BREAK = "<point>"


@dataclass(frozen=True)
class ChartShot:
    """A single charted shot, field-comparable across human and CV charts."""

    kind: str                 # 'shot' | 'serve' | POINT_BREAK
    shot_type: str | None = None
    direction: int | None = None
    depth: int | None = None
    outcome: str | None = None   # winner / forced_error / unforced_error / in_play / None
    role: str | None = None


def point_to_shots(point: Point) -> list[ChartShot]:
    """Flatten a parsed :class:`Point` into serve + rally ChartShots (no POINT_BREAK)."""
    out: list[ChartShot] = []
    for s in point.serves:
        out.append(ChartShot("serve", shot_type="serve", direction=None, role="server"))
    n = len(point.shots)
    for i, sh in enumerate(point.shots):
        outcome = "in_play"
        if i == n - 1:
            if sh.winner:
                outcome = "winner"
            elif sh.error_forced is True:
                outcome = "forced_error"
            elif sh.error_forced is False:
                outcome = "unforced_error"
        out.append(ChartShot("shot", sh.shot_type, sh.direction, sh.depth, outcome, sh.role.value))
    return out


def chart_stream(points: list[Point]) -> list[ChartShot]:
    """A whole match/segment as one token stream with POINT_BREAK markers between points."""
    stream: list[ChartShot] = []
    for i, p in enumerate(points):
        if i:
            stream.append(ChartShot(POINT_BREAK))
        stream.extend(point_to_shots(p))
    return stream


# --- alignment (edit distance with backtrace) --------------------------------------------

def _sub_cost(a: ChartShot, b: ChartShot) -> float:
    if a.kind != b.kind:
        return 2.0                      # never align a shot to a boundary
    if a.kind != "shot":
        return 0.0                      # serve<->serve, break<->break: free
    return 0.0 if a.shot_type == b.shot_type else 0.6  # prefer aligning same-type shots


def align(pred: list[ChartShot], gt: list[ChartShot]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch alignment. Returns (pred_idx, gt_idx) pairs; None = insertion/deletion."""
    n, m = len(pred), len(gt)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j - 1] + _sub_cost(pred[i - 1], gt[j - 1]),
                           dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + _sub_cost(pred[i - 1], gt[j - 1]):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1.0:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    out.reverse()
    return out


# --- scoring -----------------------------------------------------------------------------

@dataclass
class ChartScore:
    n_gt: int
    n_pred: int
    matched: int          # shots aligned shot<->shot
    insertions: int       # predicted shots with no gt match (false positives)
    deletions: int        # gt shots with no predicted match (misses)
    field_correct: dict   # field -> count correct among matched shots
    field_total: dict     # field -> count comparable among matched shots

    @property
    def precision(self) -> float:
        return self.matched / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def field_acc(self, field: str) -> float:
        t = self.field_total.get(field, 0)
        return self.field_correct.get(field, 0) / t if t else 0.0

    def summary(self) -> str:
        fields = " ".join(f"{f}={self.field_acc(f):.3f}"
                          for f in ("shot_type", "direction", "depth", "outcome"))
        return (f"shots gt={self.n_gt} pred={self.n_pred} | "
                f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} | {fields} "
                f"| ins={self.insertions} del={self.deletions}")


def score(pred: list[ChartShot], gt: list[ChartShot]) -> ChartScore:
    """Score a predicted chart stream against ground truth."""
    fields = ("shot_type", "direction", "depth", "outcome")
    fc = {f: 0 for f in fields}
    ft = {f: 0 for f in fields}
    n_gt = sum(1 for s in gt if s.kind == "shot")
    n_pred = sum(1 for s in pred if s.kind == "shot")
    matched = ins = dele = 0
    for pi, gj in align(pred, gt):
        if pi is not None and gj is not None:
            a, b = pred[pi], gt[gj]
            if a.kind == "shot" and b.kind == "shot":
                matched += 1
                for f in fields:
                    ft[f] += 1
                    if getattr(a, f) == getattr(b, f):
                        fc[f] += 1
        elif pi is not None and pred[pi].kind == "shot":
            ins += 1
        elif gj is not None and gt[gj].kind == "shot":
            dele += 1
    return ChartScore(n_gt, n_pred, matched, ins, dele, fc, ft)
