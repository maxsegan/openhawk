"""Validate the CV scoring harness by perturbing a ground-truth chart.

We can't test against real video yet, but we can prove the scorer reacts correctly to the
error modes a CV charter will make: dropped shots (recall), spurious shots (precision),
relabeled fields (per-field accuracy), and mis-segmented point boundaries. Runnable with
pytest or as a plain script.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "charting"))
sys.path.insert(0, os.path.dirname(__file__))

from tennis_charting.notation import parse_point  # noqa: E402
from scoring import ChartShot, chart_stream, score  # noqa: E402

# a small "match": four real MCP-notation points
POINTS = [parse_point(c) for c in [
    "4b37f1f3b2f3*",       # rally, forehand winner
    "6f28f1f1v2n@",        # unforced volley error
    "5*",                  # ace
    "4b3f1f3b1b3f3d#",     # forced error
]]
GT = chart_stream(POINTS)


def _shots_only(stream):
    return [s for s in stream if s.kind == "shot"]


def test_perfect_prediction_scores_one():
    s = score(list(GT), GT)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0
    for f in ("shot_type", "direction", "depth", "outcome"):
        assert s.field_acc(f) == 1.0, f


def test_dropped_shot_lowers_recall():
    # remove one shot from the prediction
    pred = [s for i, s in enumerate(GT) if i != 3]
    s = score(pred, GT)
    assert s.deletions == 1 and s.recall < 1.0 and s.precision == 1.0


def test_spurious_shot_lowers_precision():
    pred = list(GT)
    pred.insert(4, ChartShot("shot", "forehand", 2, None, "in_play", "server"))
    s = score(pred, GT)
    assert s.insertions == 1 and s.precision < 1.0 and s.recall == 1.0


def test_relabeled_direction_only_hits_direction_acc():
    # flip the direction of every shot but keep shot types -> dir acc drops, type acc stays 1
    pred = [replace(s, direction=(1 if s.direction != 1 else 3)) if s.kind == "shot" else s
            for s in GT]
    s = score(pred, GT)
    assert s.field_acc("shot_type") == 1.0
    assert s.field_acc("direction") < 1.0


def test_wrong_shot_type_still_aligns_by_position():
    pred = []
    for s in GT:
        if s.kind == "shot" and s.shot_type == "forehand":
            pred.append(replace(s, shot_type="backhand"))
        else:
            pred.append(s)
    s = score(pred, GT)
    # every shot still matched (alignment tolerates a substitution), but type acc falls
    assert s.matched == len(_shots_only(GT))
    assert s.field_acc("shot_type") < 1.0


def test_point_boundary_error_is_handled():
    # merge two points by dropping a POINT_BREAK: shots still align, no crash
    pred = [s for i, s in enumerate(GT)
            if not (s.kind == "<point>" and i == GT.index(next(x for x in GT if x.kind == "<point>")))]
    s = score(pred, GT)
    assert s.recall == 1.0  # no shots lost, only a boundary marker


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  ok  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
