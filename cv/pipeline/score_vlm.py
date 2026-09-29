"""Score-overlay reading v3: configured current VLM on change-gated scoreboard crops.

Replaces the digit-cell clustering + hand-label pipeline (score_cells.py). A local vLLM
server reads the whole scoreboard row as text, so there is nothing to hand-label and the
stylized broadcast fonts are handled for free. Sending all ~1fps crops would be wasteful:
score states persist for many seconds, so we

1. **Gate**: compute a small downscaled grayscale signature per crop, masked to the
   temporally-stable region (the overlay; masks out moving crowd/court). A crop is sent to
   the VLM only when its signature differs materially from the last SENT crop AND the
   scene is stable (signature ~= previous frame; skips camera pans/replays), plus an
   unconditional heartbeat send every --heartbeat seconds.
2. **Read**: ask the VLM for a flat left-to-right reading per row (name, score values,
   points-box value, serve row) as strict JSON; --concurrency requests in flight.
3. **Assign slots in python**: last value before the points box = current games, earlier
   values = completed sets; points is 0/15/30/40/AD or a small tiebreak number.
4. **Emit**: frames between sends inherit the last read; consecutive identical states are
   collapsed into runs -> score_runs.csv (same schema as score_cells.py; the old file is
   backed up once to score_runs_cells.csv). Raw reads go to vlm_reads.jsonl for debugging.

    .venv/bin/python cv/pipeline/score_vlm.py --out data/processed/rg2025f
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from cv.pipeline.vlm_policy import configured_vlm_model, reasoning_request_options

VLM_URL = os.environ.get("VLM_URL", "http://localhost:8399/v1/chat/completions")
VLM_REASONING_LEVEL = os.environ.get("VLM_REASONING_LEVEL", "none")
SIG_W, SIG_H = 96, 24
PTS_WORDS = {"15", "30", "40", "AD"}  # unambiguous points-box values
FIELDS = ["g1", "g2", "p1", "p2", "sets1", "sets2"]
DEFAULT_MAX_READ_FAILURE_RATE = 0.25

PROMPT = """This image may contain a tennis scoreboard overlay with two rows (one per player).
Read it and answer with STRICT JSON only, no other text, using this schema:
{"present": true/false,
 "rows": [
   {"name": "<player surname>", "values": ["<v1>", ...], "points": "<pts>" or null},
   {"name": "...", "values": [...], "points": ...}
 ],
 "serving_row": 1 or 2 or null}

Rules:
- "present": false if there is no scoreboard in the image (then rows=[] and serving_row=null).
- rows[0] must be the TOP row of the scoreboard, rows[1] the bottom row.
- "values": ALL score numbers in the row read left to right, EXCLUDING the player's name,
  country code (e.g. ITA/ESP), any seed or row number printed next to the name, and the
  points box. These are set/game scores, each a one- or two-digit number.
- "points": the value inside the points box at the far RIGHT end of the row (a separate,
  differently-colored box, usually white): one of 0, 15, 30, 40, AD, or a small tiebreak
  number. Use null ONLY if that box is empty or missing. If the points box shows a value,
  put it in "points" and do NOT repeat it in "values".
- "serving_row": which row (1=top, 2=bottom) has the serve indicator (a small ball icon,
  arrow, or // mark next to a name); null if none is visible.
Answer with JSON only."""


# ---------------- change gating -----------------------------------------------------------


def load_sig(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)


def build_mask(crops: list[str], n_sample: int = 256) -> np.ndarray:
    """Temporally-stable pixel mask: the overlay region is static, crowd/court churn."""
    sample = crops[:: max(1, len(crops) // n_sample)]
    sigs = [s for s in (load_sig(p) for p in sample) if s is not None]
    if not sigs:
        return np.ones((SIG_H, SIG_W), bool)
    std = np.stack(sigs).astype(np.float32).std(axis=0)
    mask = std <= np.percentile(std, 50)
    return mask if mask.sum() >= 100 else np.ones((SIG_H, SIG_W), bool)


def diff_frac(a: np.ndarray, b: np.ndarray, mask: np.ndarray, delta: int) -> float:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)) > delta
    return float(d[mask].mean())


def select_frames(
    crops: list[str], mask: np.ndarray, heartbeat: int, thresh: float, stable: float, delta: int
) -> list[int]:
    """Indices to send: signature changed vs last SENT crop while scene stable, or heartbeat."""
    selected: list[int] = []
    last_sig = prev_sig = None
    last_i = -(10**9)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, sig in enumerate(ex.map(load_sig, crops)):
            if sig is None:
                prev_sig = None
                continue
            send = False
            if i - last_i >= heartbeat or last_sig is None:
                send = True
            elif diff_frac(sig, last_sig, mask, delta) > thresh:
                # changed vs last sent; only send once the scene settles (not mid-pan)
                if prev_sig is None or diff_frac(sig, prev_sig, mask, delta) <= stable:
                    send = True
            if send:
                selected.append(i)
                last_sig, last_i = sig, i
            prev_sig = sig
    return selected


# ---------------- VLM query + parsing ------------------------------------------------------


def query_vlm(path: str, timeout: float = 180.0) -> dict | None:
    """One crop -> parsed JSON dict, or None. Retries once (nudged) on parse failure."""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": PROMPT},
    ]
    for attempt, temp in ((0, 0.0), (1, 0.2)):
        model = configured_vlm_model()
        body = {
            "model": model,
            "temperature": temp,
            "messages": [{"role": "user", "content": content}],
        }
        body.update(
            reasoning_request_options(
                model,
                VLM_REASONING_LEVEL,
                answer_tokens=300,
            )
        )
        try:
            req = urllib.request.Request(
                VLM_URL,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001 — server hiccup: treat as absent, don't kill the pool
            continue
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                obj = json.loads(m.group())
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return None


def _norm_token(v) -> str:
    t = str(v).strip().upper().rstrip(".")
    return "AD" if t in ("A", "AD") else t


def _raw_pts(row: dict) -> str:
    p = row.get("points")
    return "" if p in (None, "", "null", "NULL") else _norm_token(p)


def _parse_row(row: dict) -> tuple[list[str], str] | None:
    """-> (set/game digits left-to-right, points) or None if malformed."""
    vals = [_norm_token(v) for v in (row.get("values") or [])]
    vals = [v for v in vals if v == "AD" or v.isdigit()]  # drop ITA/ESP etc.
    p = _raw_pts(row)
    # a set/game score can never be 15/30/40/AD: a trailing one is the points value the
    # model misplaced (or duplicated) — claim it for the points slot
    if vals and vals[-1] in PTS_WORDS:
        p = vals.pop()
    elif p and p != "0" and vals and vals[-1] == p:
        # points duplicated into values.  Never claim this for "0": a games box showing 0
        # beside a points box showing 0 is the ordinary start of a game, and popping it
        # there loses the current-games column and with it the completed-set boundary.
        vals.pop()
    if any(v in PTS_WORDS for v in vals):
        return None
    if p and p != "AD":
        if not (p in PTS_WORDS or (p.isdigit() and 0 <= int(p) <= 30)):
            return None  # 0/15/30/40/AD or tiebreak int
    if not vals or any(not v.isdigit() or len(v) > 1 for v in vals):
        return None  # set/game scores are 0..9
    return vals, p


def norm_read(obj: dict | None) -> dict | None:
    """VLM JSON -> {g1,g2,p1,p2,sets1,sets2} state, or None (absent/unusable)."""
    if not obj or not obj.get("present"):
        return None
    rows = obj.get("rows") or []
    if len(rows) != 2 or not all(isinstance(r, dict) for r in rows):
        return None
    parsed = [_parse_row(r) for r in rows]
    if len(parsed) != 2 or parsed[0] is None or parsed[1] is None:
        return None
    (d1, p1), (d2, p2) = parsed
    # cross-row repair: one row's points ended up as a trailing digit in values (e.g. "0",
    # which is ambiguous within a single row) — detectable when the other row got points
    if len(d1) == len(d2) + 1 and p2 and not p1 and _plausible_pts(d1[-1], p2):
        p1 = d1.pop()
    elif len(d2) == len(d1) + 1 and p1 and not p2 and _plausible_pts(d2[-1], p1):
        p2 = d2.pop()
    if len(d1) != len(d2) or not d1:  # rows must have equal columns
        return None
    # a completed set is never 6-6, so a 6-6 column followed by a trailing column with no
    # points read = live tiebreak games + tiebreak points misplaced into the columns
    if p1 == p2 == "" and len(d1) >= 2 and d1[-2] == "6" and d2[-2] == "6":
        p1, p2 = d1.pop(), d2.pop()
    # tiebreak-looking points (bare digits other than 0) are only real at games 6-6.
    # Anywhere else the model mistook the highlighted current-games box for the points box
    # (common when the points box is empty): shift points back into the digit columns.
    if (_tb_pts(p1) or _tb_pts(p2)) and not (d1[-1] == "6" and d2[-1] == "6"):
        if not (p1.isdigit() and int(p1) <= 7 and p2.isdigit() and int(p2) <= 7):
            return None  # impossible state — reject
        d1.append(p1)
        d2.append(p2)
        p1 = p2 = ""
    return {
        "g1": d1[-1],
        "g2": d2[-1],
        "p1": p1,
        "p2": p2,
        "sets1": " ".join(d1[:-1]),
        "sets2": " ".join(d2[:-1]),
    }


def _tb_pts(p: str) -> bool:
    return p.isdigit() and p not in ("0", "15", "30", "40")


def _plausible_pts(v: str, other_p: str) -> bool:
    """Can v be this row's points given the opponent shows other_p? In a standard game the
    only bare-digit points value is 0; in a tiebreak (opponent numeric) any digit works."""
    return v == "0" if other_p in PTS_WORDS else v.isdigit()


def read_summary(reads_attempted: int, reads_ok: int) -> dict:
    failures = reads_attempted - reads_ok
    return {
        "reads_attempted": reads_attempted,
        "reads_ok": reads_ok,
        "read_failures": failures,
        "read_failure_rate": failures / max(1, reads_attempted),
    }


def write_stage_manifest(out_dir: str, summary: dict, *, status: str) -> None:
    path = os.path.join(out_dir, "score_vlm_manifest.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "schema": "score_vlm_stage_v1",
                "status": status,
                **summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )


# ---------------- driver -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="dir containing the scoreboard crops")
    ap.add_argument(
        "--crop-dir",
        default="auto",
        help="crop subdirectory; 'auto' prefers native score_crops_native/ over overlay_crops/",
    )
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--heartbeat", type=float, default=20.0, help="max seconds between sends")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--sig-thresh",
        type=float,
        default=0.02,
        help="fraction of masked signature pixels changed vs last sent to trigger",
    )
    ap.add_argument(
        "--sig-stable",
        type=float,
        default=0.01,
        help="max fraction changed vs previous frame to count as a stable scene",
    )
    ap.add_argument("--sig-delta", type=int, default=25, help="per-pixel gray delta")
    ap.add_argument(
        "--max-read-failure-rate",
        type=float,
        default=DEFAULT_MAX_READ_FAILURE_RATE,
        help="fail closed when more than this share of selected VLM reads fails",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="only report how many crops would be sent"
    )
    ap.add_argument(
        "--from-raw",
        action="store_true",
        help="reuse vlm_reads.jsonl (re-parse + re-emit runs, no VLM queries)",
    )
    args = ap.parse_args()
    if not 0.0 <= args.max_read_failure_rate <= 1.0:
        ap.error("--max-read-failure-rate must be between 0 and 1")

    crop_dir = args.crop_dir
    if crop_dir == "auto":
        native = os.path.join(args.out, "score_crops_native")
        crop_dir = (
            "score_crops_native" if glob.glob(os.path.join(native, "s_*.jpg")) else "overlay_crops"
        )
    crops = sorted(glob.glob(os.path.join(args.out, crop_dir, "s_*.jpg")))
    print(f"{len(crops)} crops from {crop_dir}")
    if not crops:
        return 1

    raw_path = os.path.join(args.out, "vlm_reads.jsonl")
    reads: dict[int, dict | None] = {}
    n_json = n_absent = 0
    if args.from_raw:
        raws = [json.loads(line) for line in open(raw_path)]
        selected = [d["i"] for d in raws]
        for d in raws:
            reads[d["i"]] = norm_read(d["raw"])
            n_json += d["raw"] is not None
            n_absent += bool(d["raw"]) and not d["raw"].get("present")
        print(f"reusing {len(selected)} raw reads from {raw_path}")
    else:
        mask = build_mask(crops)
        hb = max(1, int(round(args.heartbeat * args.fps)))
        selected = select_frames(crops, mask, hb, args.sig_thresh, args.sig_stable, args.sig_delta)
        print(
            f"gating: mask {int(mask.sum())}/{mask.size} px | sending "
            f"{len(selected)}/{len(crops)} crops ({len(selected) / len(crops):.1%})"
        )
        if args.dry_run:
            return 0
        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex, open(raw_path, "w") as rawf:
            for i, obj in zip(selected, ex.map(lambda j: query_vlm(crops[j]), selected)):
                reads[i] = norm_read(obj)
                n_json += obj is not None
                n_absent += bool(obj) and not obj.get("present")
                rawf.write(
                    json.dumps(
                        {"i": i, "file": os.path.basename(crops[i]), "read": reads[i], "raw": obj}
                    )
                    + "\n"
                )
                done += 1
                if done % 200 == 0:
                    print(f"  vlm {done}/{len(selected)}")
    summary = read_summary(len(selected), n_json)
    if summary["read_failure_rate"] > args.max_read_failure_rate:
        write_stage_manifest(args.out, summary, status="failed_read_threshold")
        raise RuntimeError(
            "score VLM read failure rate "
            f"{summary['read_failure_rate']:.1%} exceeds {args.max_read_failure_rate:.1%} "
            f"({summary['reads_ok']}/{summary['reads_attempted']} reads succeeded)"
        )
    n_state = sum(1 for i in selected if reads[i] is not None)
    print(
        f"reads: json {n_json}/{len(selected)} ({n_json / len(selected):.1%}) | "
        f"scoreboard-absent {n_absent} | usable states {n_state} "
        f"({n_state / len(selected):.1%}); raw -> {raw_path}"
    )

    # frames between sends inherit the last sent read (that's what the gate guarantees)
    runs, cur, start = [], None, None
    state: dict | None = None
    sel_set = set(selected)
    for i in range(len(crops)):
        if i in sel_set:
            state = reads[i]
        key = tuple(state[k] for k in FIELDS) if state else None
        if key != cur:
            if cur is not None:
                runs.append((start / args.fps, i / args.fps, *cur))
            cur, start = key, i
    if cur is not None:
        runs.append((start / args.fps, len(crops) / args.fps, *cur))

    out_csv = os.path.join(args.out, "score_runs.csv")
    backup = os.path.join(args.out, "score_runs_cells.csv")
    if os.path.exists(out_csv) and not os.path.exists(backup):
        os.rename(out_csv, backup)
        print(f"backed up old score_runs.csv -> {backup}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", *FIELDS])
        for row in runs:
            w.writerow([f"{row[0]:.1f}", f"{row[1]:.1f}", *row[2:]])

    # sanity: completed-set count should never decrease over time
    nsets = [len(r[6].split()) + len(r[7].split()) for r in runs]
    n_mono = sum(1 for a, b in zip(nsets, nsets[1:]) if b < a)
    print(f"runs={len(runs)} -> {out_csv} | set-count regressions: {n_mono}")
    write_stage_manifest(args.out, summary, status="completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
