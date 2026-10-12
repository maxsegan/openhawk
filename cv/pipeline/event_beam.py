"""Event beam: label-free contact candidates from independent time witnesses, re-fitted in S6.

The product step after the cascade refit (``product_runner`` step ``beam``; rollback
``--event-beam off``). Port of the ``cv/experiments/event_beam`` beam4p arm (guard level 2, no
evidence floor); development result on opened panels 0/A, C, D, E, F in
``cv/experiments/event_beam/FINAL_REPORT_event_beam.md`` section 6 (+38/-0 matched flights vs a
same-session control, 0 competitive wrong after post-validation, ~4.6x S6 fit CPU).

1. Per-match audio delay (``audio_delay_calibration``): anchors are the automatic model's
   confident contacts, plus image-space ball reversals at a player when those alone are not
   confident. Uncertain estimates keep a second lag mode and audio witnesses branch on both.
2. Candidates per attempt (``candidates``): audio onset peaks (robust z >= ``Z_MIN``, lag
   corrected), held S5 contact emissions, S5 contact-probability peaks, and the gap rule (a
   track gap at a player with the ball returning the other way; epoch at the loudest sound),
   merged within ``MERGE`` frames, each with its own producer interval.
3. Slots: the flights the incumbent fit did not accept and the track spans outside every
   flight. Per slot, candidates pass pre-fit gates (outgoing flight observable, image reach
   of a player, the split test beats one arc on the same observations, the bounce guard),
   ranked by family; each also offers its image break time.
4. Slot-parallel branches (the j-th candidate of every slot), each rebound on the first-fit
   packet with the cascade's settled decisions and refit. Per slot, the best branch that gains
   accepted flights in the slot window and loses none, whose new flights pass post-validation
   (``flight_validation``) and the contact guard (height, foot-point reach). The composition of
   slot winners is refit; the incumbent moves only to a fit that accepts more flights and
   loses none of the incumbent's (the baseline is always a branch). ``ROUNDS`` rounds.

No evaluation label is read. Every fit is an ordinary automatic S6 run on an edited packet
bound by its own automatic producer provenance (``event_cascade_rebind``).
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from cv.pipeline import audio_delay_calibration as cal
from cv.pipeline import flight_validation as fv

SCHEMA = "event_beam_v1"
DATA = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
AUDIO_NAME = "contact_audio_scores_16k_native_v1.npz"
EMISSIONS_NAME = "event_emissions.json"
PREDICTIONS_NAME = "event_emissions.predictions.npz"

# Witnesses (witnesses.py).
Z_MIN = 5.0
PEAK_DISTANCE = 12  # audio samples (120 ms)
S5_PEAK_MIN = 0.2
MIN_GAP, MAX_GAP, EDGE = 3, 45, 4
MERGE = 2.0
TOL = 2.0  # native frames: a candidate this close to an automatic contact is a duplicate
RETIME_NEAR = 13.0
CLEARANCE = 3.0
REVERSAL_W = 6
# Guards (guards.py).
FOOT_SIGMA = 0.1
NEAR_FRAMES = 3
REACH_M = 2.5
REACH_MARGIN_M = 1.5
Z_RANGE_M = (0.02, 3.7)
SPLIT_W = 12
SPLIT_SIDE = 3
SNAP_MIN = 2.0
BOUNCE_GUARD = 3.0
CROSS_S = 1.0
CROSS_RUN = 3
NET_Y_M, NET_TOP_M, COURT_W_M = 11.885, 0.914, 10.97
# Beam (beam.py).
BRANCHES = 4
ROUNDS = 2
# Optional pre-fit prune of a slot's ranked branches (``run(prune=...)``; default off). On the
# combined-v4 opened-panel beams no slot winner lacked a time witness (audio or an S5 peak) or
# was an audio-only onset below PRUNE_AUDIO_Z (0 of 23 winners; ~9% of branch fit wall).
# Pruning happens after the BRANCHES cut, so the surviving fits are a subset of the unpruned
# beam's, never replacements.
PRUNE_MODES = ("off", "time_witness")
PRUNE_AUDIO_Z = 8.0
MATCH_START, MATCH_END = 1.5, 4.0


def _resolve(path) -> Path:
    path = path["path"] if isinstance(path, dict) else path
    return Path(path) if str(path).startswith("/") else DATA / path


def _load(path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- match inputs


@dataclass
class Match:
    match_id: str
    traces: dict  # clip -> contact audio score (100 Hz)
    emissions: list
    probabilities: dict  # "<match>__<clip>" -> {frame: contact probability}


def load_match(upstream: Path, match_id: str) -> Match:
    """The match's automatic event emissions, per-frame contact probabilities and contact
    audio score (``broadcast_runner`` layout: ``upstream/event_emissions*``,
    ``upstream/<match>/contact_audio_scores_16k_native_v1.npz``); missing files are empty."""
    traces = {}
    audio = upstream / match_id / AUDIO_NAME
    if audio.is_file():
        z = np.load(audio)
        traces = {k[len("scores_") :]: z[k] for k in z.files if k.startswith("scores_")}
    emissions = _load(upstream / EMISSIONS_NAME) if (upstream / EMISSIONS_NAME).is_file() else []
    probabilities: dict = defaultdict(dict)
    pred = upstream / PREDICTIONS_NAME
    if pred.is_file():
        z = np.load(pred)
        index = 1
        manifest = pred.with_name("event_emissions.manifest.json")
        if manifest.is_file():
            doc = _load(manifest)
            for key in ("classes", "class_names", "labels"):
                names = doc.get(key) or (doc.get("model") or {}).get(key)
                if names and "contact" in names:
                    index = list(names).index("contact")
        for clip, frame, prob in zip(z["clips"], z["frames"], z["probabilities"]):
            probabilities[str(clip)][int(frame)] = float(prob[index])
    return Match(match_id, traces, emissions, dict(probabilities))


@dataclass
class Attempt:
    """One fitted attempt: its first-fit manifest, the cascade's settled decisions for it,
    the baseline result (refit if the cascade edited it, else first fit) and derived inputs."""

    key: str
    manifest: dict
    settled: dict
    baseline: Path  # result.json
    timeout_seconds: float
    packet: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    fps: float = 0.0
    clip: str = ""
    px: dict = field(default_factory=dict)
    boxes: dict = field(default_factory=dict)
    bounces: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    inventory: list | None = None

    @classmethod
    def build(cls, key, manifest, settled, baseline, timeout_seconds) -> Attempt:
        from cv.pipeline import event_cascade_propose as proposer

        row = manifest["rows"][0]
        a = cls(key, manifest, settled, Path(baseline), timeout_seconds)
        a.packet = _load(_resolve(row["packet"]))
        a.labels = _load(_resolve(row["labels"]))
        a.cameras = _load(_resolve(row["cameras"]))
        source = a.packet["attempts"][0]
        a.fps = float(source["fps"])
        a.clip = str(source.get("point_clip") or source.get("clip"))
        a.px = ball_pixels(a.labels)
        a.boxes = player_boxes(row)
        events = automatic_events(a.labels)
        a.bounces = [e["frame"] for e in events if e["type"] == "bounce"]
        a.contacts = [e["frame"] for e in events if e["type"] == "contact"]
        a.inventory = proposer.editable_inventory(a.packet)
        return a


def automatic_events(doc: dict) -> list[dict]:
    return [
        {"type": e["event_type"], "frame": float(e["frame"]), "status": e.get("status")}
        for e in (doc.get("events") or {}).get("records") or []
    ]


def ball_pixels(doc: dict) -> dict[int, np.ndarray]:
    out = {}
    for rec in (doc.get("ball") or {}).get("records") or []:
        for row in rec.get("frames") or []:
            if row.get("status") == "visible" and row.get("x1080") is not None:
                f = float(row["frame"])
                if f.is_integer():
                    out[int(f)] = np.asarray([row["x1080"], row["y1080"]], float)
    return out


def attempt_span(doc: dict) -> tuple[float, float]:
    frames = [
        float(r["frame"])
        for rec in (doc.get("ball") or {}).get("records") or []
        for r in rec.get("frames") or []
    ]
    return (min(frames), max(frames)) if frames else (0.0, 0.0)


def player_boxes(row: dict) -> dict[int, list]:
    spec = row.get("pose_csv")
    if not spec:
        return {}
    out = defaultdict(list)
    with _resolve(spec).open() as handle:
        for r in csv.DictReader(handle):
            if not r.get("x0_native"):
                continue
            frame = int(r["frame"].split("_")[1].split(".")[0])
            xyxy = [float(r[k]) for k in ("x0_native", "y0_native", "x1_native", "y1_native")]
            out[frame].append((r["side"], xyxy))
    return dict(out)


def player_distance(rows, xy) -> tuple[float, str | None]:
    """Image distance from the ball to the nearest player box, in box heights."""
    best, side = np.inf, None
    for s, (x0, y0, x1, y1) in rows:
        h = max(y1 - y0, 1.0)
        d = float(np.hypot(max(x0 - xy[0], 0.0, xy[0] - x1), max(y0 - xy[1], 0.0, xy[1] - y1)) / h)
        if d < best:
            best, side = d, s
    return best, side


def _velocity(px: dict, frames) -> np.ndarray | None:
    pts = [(f, px[f]) for f in frames if f in px]
    if len(pts) < 3:
        return None
    t = np.asarray([p[0] for p in pts], float)
    return np.polyfit(t, np.asarray([p[1] for p in pts]), 1)[0]


# --------------------------------------------------------------------------- calibration


def confident_contacts(match: Match, a: Attempt) -> list[dict]:
    key = f"{match.match_id}__{a.clip}"
    out = []
    for row in match.emissions:
        if row.get("clip") != key or row.get("event_type") != "contact":
            continue
        if row.get("abstain") or row.get("gate_held"):
            continue
        if float((row.get("class_probabilities") or {}).get("contact", 0.0)) < cal.CONFIDENT_P:
            continue
        f = float((row.get("location") or {}).get("frame_subpixel") or row["frame"])
        out.append({"clip": a.clip, "frame": f, "kind": "confident_contact"})
    return out


def reversals(a: Attempt, contacts: list[float]) -> list[dict]:
    """Image-vertical ball reversals within 1.5 box heights of a player, away from that player,
    not within 5 frames of an automatic bounce or 3 of a confident contact (hinge break time)."""
    out, last = [], -1e9
    for f in sorted(a.px):
        v0 = _velocity(a.px, range(f - REVERSAL_W, f + 1))
        v1 = _velocity(a.px, range(f + 1, f + REVERSAL_W + 2))
        if v0 is None or v1 is None or np.sign(v0[1]) == np.sign(v1[1]):
            continue
        if min(abs(v0[1]), abs(v1[1])) <= 1.0:
            continue
        d, side = player_distance(a.boxes.get(f, []), a.px[f])
        if d > 1.5 or side is None or not (v1[1] < 0 if side == "near" else v1[1] > 0):
            continue
        b = best_break(a.px, float(f))
        t = float(f) if b is None else float(b)
        if any(abs(t - x) <= 5 for x in a.bounces) or any(abs(t - c) <= 3 for c in contacts):
            continue
        if t - last <= 8:
            continue
        last = t
        out.append({"clip": a.clip, "frame": t, "kind": "reversal"})
    return out


def calibrate(match: Match, attempts: list[Attempt]) -> dict:
    """The match's audio delay (frames, audio minus video) with its confidence and lag modes."""
    fps, conf, rev, seen = {}, [], [], set()
    for a in attempts:
        if a.clip in seen:
            continue
        seen.add(a.clip)
        fps[a.clip] = a.fps
        c = confident_contacts(match, a)
        conf += c
        rev += reversals(a, [x["frame"] for x in c])
    traces = {k: v for k, v in match.traces.items() if k in fps}
    est = cal.calibrate(traces, fps, conf)
    est["anchor_set"] = "confident_contacts"
    if not est["confident"]:
        alt = cal.calibrate(traces, fps, conf + rev)
        if alt["confident"]:
            est = dict(alt, anchor_set="confident_contacts+reversals")
    return est


# --------------------------------------------------------------------------- witnesses


def _loudest(trace, fps: float, lag: float, lo: float, hi: float) -> float | None:
    a, b = cal.sample_of_frame(lo + lag, fps), cal.sample_of_frame(hi + lag, fps)
    a, b = max(a, 0), min(b, len(trace) - 1)
    if b <= a:
        return None
    return cal.frame_of_sample(a + int(np.argmax(trace[a : b + 1])), fps) - lag


def witnesses(match: Match, a: Attempt, lag: dict) -> list[dict]:
    """Every raw witness of one attempt: frame, interval, family, strength."""
    fps = a.fps
    lo_span, hi_span = attempt_span(a.labels)
    out = []
    trace = match.traces.get(a.clip)
    half = float(np.clip(lag["spread"], 1.0, 5.0))
    if trace is not None:
        z = cal.robust_z(np.asarray(trace, float))
        peaks, _ = find_peaks(z, height=Z_MIN, distance=PEAK_DISTANCE)
        for i in peaks:
            for mode, shift in enumerate(lag["modes"]):
                f = cal.frame_of_sample(float(i), fps) - shift
                if lo_span <= f <= hi_span:
                    out.append(
                        {
                            "family": "audio",
                            "frame": round(f, 2),
                            "interval": [round(f - half, 2), round(f + half, 2)],
                            "z": round(float(z[i]), 1),
                            "lag_mode": mode,
                        }
                    )
    key = f"{match.match_id}__{a.clip}"
    emitted = []
    for row in match.emissions:
        if row.get("clip") != key or row.get("event_type") != "contact":
            continue
        emitted.append(float(row["frame"]))
        if not (row.get("abstain") or row.get("gate_held")):
            continue
        p = row.get("class_probabilities") or {}
        if float(p.get("contact", 0)) < float(p.get("none", 1)) or float(p.get("contact", 0)) <= 0:
            continue
        f = float((row.get("location") or {}).get("frame_subpixel") or row["frame"])
        out.append(
            {
                "family": "s5_held",
                "frame": round(f, 2),
                "interval": [round(f - 1.5, 2), round(f + 1.5, 2)],
                "p": round(float(p["contact"]), 3),
            }
        )
    probs = match.probabilities.get(key, {})
    if probs:
        frames = np.array(sorted(probs))
        values = np.array([probs[f] for f in frames])
        idx, _ = find_peaks(np.concatenate([[0], values, [0]]), height=S5_PEAK_MIN, distance=4)
        for k in idx - 1:
            f = float(frames[k])
            if any(abs(f - e) <= 3 for e in emitted) or not lo_span <= f <= hi_span:
                continue
            out.append(
                {
                    "family": "s5_peak",
                    "frame": f,
                    "interval": [f - 2.0, f + 2.0],
                    "p": round(float(values[k]), 3),
                }
            )
    seen = sorted(a.px)
    for f0, f1 in zip(seen, seen[1:]):
        if not MIN_GAP <= f1 - f0 - 1 <= MAX_GAP:
            continue
        v0 = _velocity(a.px, range(f0 - EDGE - 2, f0 + 1))
        v1 = _velocity(a.px, range(f1, f1 + EDGE + 3))
        if v0 is None or v1 is None:
            continue
        if np.sign(v0[1]) == np.sign(v1[1]) or min(abs(v0[1]), abs(v1[1])) <= 1.0:
            continue
        (da, sa), (db, sb) = (
            player_distance(a.boxes.get(f0, []), a.px[f0]),
            player_distance(a.boxes.get(f1, []), a.px[f1]),
        )
        d, side = (da, sa) if da <= db else (db, sb)
        if d > 1.5:
            continue
        if side is not None and not (v0[1] > 0 if side == "near" else v0[1] < 0):
            continue
        if any(f0 - 5 <= x <= f1 + 5 for x in a.bounces):
            continue
        f = (f0 + f1) / 2
        if trace is not None:
            f = _loudest(trace, fps, lag["modes"][0], f0 - 2, f1 + 2) or f
        out.append(
            {
                "family": "gap",
                "frame": round(f, 2),
                "interval": [float(f0 - 1), float(f1 + 1)],
                "gap": [f0, f1],
                "side": side,
                "player_distance": round(d, 2),
            }
        )
    return out


def candidates(match: Match, a: Attempt, lag: dict) -> list[dict]:
    """Witnesses merged within ``MERGE`` frames; ``relation`` to the automatic contacts:
    ``duplicate`` (<= 2 frames), ``retime`` (<= 13 frames), else ``insert``."""
    raw = sorted(witnesses(match, a, lag), key=lambda w: w["frame"])
    merged: list[dict] = []
    for w in raw:
        if merged and w["frame"] - merged[-1]["frames"][-1] <= MERGE:
            merged[-1]["witnesses"].append(w)
            merged[-1]["frames"].append(w["frame"])
        else:
            merged.append({"witnesses": [w], "frames": [w["frame"]]})
    contacts = sorted(a.contacts)
    out = []
    for m in merged:
        ws = m["witnesses"]
        timed = [w for w in ws if w["family"] in ("audio", "gap")] or ws
        frame = float(np.mean([w["frame"] for w in timed]))
        near = min(contacts, key=lambda c: abs(c - frame), default=None)
        d = abs(near - frame) if near is not None else np.inf
        relation = "duplicate" if d <= TOL else "retime" if d <= RETIME_NEAR else "insert"
        out.append(
            {
                "frame": round(frame, 2),
                "interval": [
                    round(min(w["interval"][0] for w in ws), 2),
                    round(max(w["interval"][1] for w in ws), 2),
                ],
                "families": sorted({w["family"] for w in ws}),
                "audio_z": max((w["z"] for w in ws if w["family"] == "audio"), default=None),
                "relation": relation,
            }
        )
    return out


# --------------------------------------------------------------------------- guards


def _camera(cameras: dict, frame: int) -> np.ndarray | None:
    by_frame = cameras.setdefault("_by_frame", {int(c["frame"]): c for c in cameras["cameras"]})
    c = by_frame.get(frame)
    return None if c is None or not c.get("supported") else np.asarray(c["P"], float)


def _ray_at_ground(P: np.ndarray, pixel) -> np.ndarray:
    eq = P[:2] - np.asarray(pixel, float)[:, None] * P[2]
    return np.linalg.solve(eq[:, :2], -eq[:, 3])


def reach(cameras: dict, boxes: dict, frame: float, xyz) -> dict | None:
    """Nearest player's court-plane foot point to a fitted contact, horizontally, with a sigma
    (box bottom moved by ``FOOT_SIGMA`` box heights); None when unmeasured."""
    if xyz is None:
        return None
    centre = int(round(frame))
    for d in sorted(range(-NEAR_FRAMES, NEAR_FRAMES + 1), key=abs):
        k = centre + d
        P = _camera(cameras, k)
        if P is None or not boxes.get(k):
            continue
        feet = []
        for side, (x0, y0, x1, y1) in boxes[k]:
            h = max(y1 - y0, 1.0)
            bottom = np.array([(x0 + x1) / 2, y1])
            try:
                g = _ray_at_ground(P, bottom)
                lo = _ray_at_ground(P, bottom + [0, FOOT_SIGMA * h])
                hi = _ray_at_ground(P, bottom - [0, FOOT_SIGMA * h])
            except np.linalg.LinAlgError:
                continue
            feet.append((g, float(max(np.linalg.norm(lo - g), np.linalg.norm(hi - g)))))
        if not feet:
            continue
        xy = np.asarray(xyz, float)[:2]
        g, sigma = min(feet, key=lambda f: np.linalg.norm(xy - f[0]) - f[1])
        dist = float(np.linalg.norm(xy - g))
        return {"reach_m": dist, "sigma_m": sigma, "excess_m": max(0.0, dist - REACH_M - sigma)}
    return None


def contact_ok(r: dict | None, contact_z: float | None) -> bool:
    """Post-fit guard on a new flight's contact (unmeasured reach passes)."""
    if contact_z is not None and not Z_RANGE_M[0] <= contact_z <= Z_RANGE_M[1]:
        return False
    return r is None or r["excess_m"] <= REACH_MARGIN_M


def _hinge_sse(t: np.ndarray, y: np.ndarray, b: float | None) -> float:
    cols = [np.ones_like(t), t, t**2, t**3]
    if b is not None:
        h = np.maximum(t - b, 0.0)
        cols += [h, h**2]
    m = np.stack(cols, 1)
    coef = np.linalg.lstsq(m, y, rcond=None)[0]
    return float(((m @ coef - y) ** 2).sum())


def split_ok(px: dict, frame: float) -> bool:
    """Complexity penalty: a track break at ``frame`` must beat one arc on the same
    observations (BIC over +-``SPLIT_W`` frames); thin tracks abstain (pass)."""
    ks = [k for k in range(int(frame - SPLIT_W), int(frame + SPLIT_W) + 1) if k in px]
    if len(ks) < 2 * SPLIT_SIDE + 4:
        return True
    ys = np.array([px[k] for k in ks], float)
    t = np.array(ks, float) - frame
    if sum(t < 0) < SPLIT_SIDE or sum(t > 0) < SPLIT_SIDE:
        return True
    s1 = sum(_hinge_sse(t, ys[:, i], None) for i in range(2))
    s2 = sum(_hinge_sse(t, ys[:, i], 0.0) for i in range(2))
    n = 2 * len(ks)
    return n * np.log(max(s1, 1e-6) / max(s2, 1e-6)) - 4 * np.log(n) > 0


def best_break(px: dict, frame: float) -> float | None:
    """The hinge time minimising the split residual within +-``SPLIT_W`` frames."""
    ks = [k for k in range(int(frame - SPLIT_W), int(frame + SPLIT_W) + 1) if k in px]
    if len(ks) < 2 * SPLIT_SIDE + 4:
        return None
    ys = np.array([px[k] for k in ks], float)
    t = np.array(ks, float) - frame
    best = None
    for b in np.arange(-SPLIT_W + SPLIT_SIDE, SPLIT_W - SPLIT_SIDE + 0.5, 0.5):
        if sum(t < b) < SPLIT_SIDE or sum(t > b) < SPLIT_SIDE:
            continue
        s = sum(_hinge_sse(t, ys[:, i], b) for i in range(2))
        if best is None or s < best[0]:
            best = (s, b)
    return None if best is None else float(frame + best[1])


def _net_side(P: np.ndarray, xy) -> float | None:
    """+1 below the net's image line (near court), -1 above it."""
    ends = []
    for x in (0.0, COURT_W_M):
        h = P @ np.array([x, NET_Y_M, NET_TOP_M, 1.0])
        if abs(h[2]) < 1e-9:
            return None
        ends.append(h[:2] / h[2])
    (x0, y0), (x1, y1) = ends
    if abs(x1 - x0) < 1e-6:
        return None
    return 1.0 if xy[1] > y0 + (y1 - y0) * (xy[0] - x0) / (x1 - x0) else -1.0


def crosses_back(a: Attempt, frame: float) -> bool:
    """After ``frame`` the ball crosses the net image line (``CROSS_RUN`` visible frames)
    within ``CROSS_S``, before the next automatic contact."""
    fi = int(round(frame))
    near = sorted((k for k in range(fi - 4, fi + 5) if k in a.px), key=lambda k: abs(k - frame))
    side0 = None
    for k in near:
        P = _camera(a.cameras, k)
        if P is not None:
            side0 = _net_side(P, a.px[k])
            break
    if side0 is None:
        return False
    stop = fi + int(round(CROSS_S * a.fps))
    nxt = [c for c in a.contacts if c > frame + BOUNCE_GUARD]
    if nxt:
        stop = min(stop, int(min(nxt)))
    run = 0
    for k in range(fi + 1, stop + 1):
        P = _camera(a.cameras, k) if k in a.px else None
        if P is None:
            continue
        run = run + 1 if _net_side(P, a.px[k]) == -side0 else 0
        if run >= CROSS_RUN:
            return True
    return False


def bounce_ok(a: Attempt, frame: float) -> bool:
    """A candidate within ``BOUNCE_GUARD`` frames before an emitted bounce is that bounce; one
    just after it (a half volley) is kept when the ball then crosses the net."""
    near = [b for b in a.bounces if abs(frame - b) <= BOUNCE_GUARD]
    if not near:
        return True
    if any(b > frame for b in near):
        return False
    return crosses_back(a, frame)


def gate(c: dict, a: Attempt) -> bool:
    """An outgoing flight is observable and the ball is within image reach of a player."""
    fi = int(round(c["frame"]))
    if sum(1 for k in range(fi + 2, fi + 41) if k in a.px) < 4:
        return False
    near = [k for k in range(fi - 4, fi + 5) if k in a.px]
    if near:
        return min(player_distance(a.boxes.get(k, []), a.px[k])[0] for k in near) <= 1.5
    before = [k for k in range(fi - 20, fi) if k in a.px]
    after = [k for k in range(fi + 1, fi + 21) if k in a.px]
    ks = ([max(before)] if before else []) + ([min(after)] if after else [])
    return bool(ks) and min(player_distance(a.boxes.get(k, []), a.px[k])[0] for k in ks) <= 2.0


def rank(c: dict) -> float:
    f = set(c["families"])
    return (
        3.0 * ("gap" in f)
        + 2.0 * ("s5_held" in f)
        + 1.0 * ("s5_peak" in f)
        + min(c["audio_z"] or 0.0, 60.0) / 20.0
    )


def cand_id(c: dict) -> str:
    return f"{c['frame']:.2f}"


def with_snaps(ranked: list[dict], px: dict) -> list[dict]:
    """Each candidate followed by its image-break-time alternative."""
    out = []
    for c in ranked:
        out.append(c)
        b = best_break(px, c["frame"])
        if b is None or abs(b - c["frame"]) <= SNAP_MIN:
            continue
        if any(abs(b - o["frame"]) <= TOL for o in ranked + out):
            continue
        out.append(
            dict(
                c,
                frame=round(b, 2),
                interval=[round(b - 1.5, 2), round(b + 1.5, 2)],
                families=sorted(set(c["families"]) | {"break"}),
                snapped_from=c["frame"],
            )
        )
    return out


# --------------------------------------------------------------------------- flights


def flights_of(result: dict | None) -> list[dict] | None:
    if result is None:
        return None
    out = []
    for f in (result.get("verdict") or {}).get("flights") or []:
        contacts = f.get("contacts") or []
        start = next((c for c in contacts if c.get("end") == "start"), None)
        closing = next(
            (c for c in contacts if c.get("end") == "end" and c.get("kind") == "contact"), None
        )
        out.append(
            {
                "start": float(f["start_frame"]),
                "end": float(f["end_frame"]),
                "accepted": bool(f.get("accepted")),
                "rms": f.get("reprojection_rms_px"),
                "start_xyz": None
                if not start or start.get("kind") != "contact"
                else start.get("contact_xyz_m"),
                "end_xyz": None if not closing else closing.get("contact_xyz_m"),
            }
        )
    return out


def _same(f: dict, g: dict) -> bool:
    return abs(f["start"] - g["start"]) <= MATCH_START and abs(f["end"] - g["end"]) <= MATCH_END


def lost_flights(before: list[dict], after: list[dict], window=None) -> list[dict]:
    kept = [f for f in after if f["accepted"]]
    return [
        f
        for f in before
        if f["accepted"]
        and (window is None or window[0] <= f["start"] <= window[1])
        and not any(_same(f, g) for g in kept)
    ]


def slots(flights: list[dict] | None, span: tuple[float, float]) -> list[list[float]]:
    lo, hi = span
    if not flights:
        return [[lo, hi]]
    out = [[f["start"], f["end"]] for f in flights if not f["accepted"]]
    first, last = min(f["start"] for f in flights), max(f["end"] for f in flights)
    if first - lo > 5:
        out.append([lo, first])
    if hi - last > 5:
        out.append([last, hi])
    return sorted(out)


def validation_rows(a: Attempt, result: dict) -> list[dict]:
    cams = {"cameras": a.cameras["cameras"]}
    return fv.validate(result, lambda f, xyz: fv.foot_reach(cams, a.boxes, f, xyz))


def guard_ok(a: Attempt, base: list[dict], result: dict, frames: list[float]) -> bool:
    """Every new accepted flight passes post-validation, and every new flight opening or ending
    within 3 frames of a beam contact passes the contact guard."""
    for r in validation_rows(a, result):
        new = {"start": r["start_frame"], "end": r["end_frame"]}
        if not r["ok"] and not any(g["accepted"] and _same(new, g) for g in base):
            return False
    for f in flights_of(result) or []:
        if not f["accepted"] or any(g["accepted"] and _same(f, g) for g in base):
            continue
        for at, xyz in ((f["start"], f["start_xyz"]), (f["end"], f["end_xyz"])):
            if xyz is None or not any(abs(at - c) <= 3 for c in frames):
                continue
            if not contact_ok(reach(a.cameras, a.boxes, at, xyz), float(xyz[2])):
                return False
    return True


# --------------------------------------------------------------------------- edits


def contact_event(frame: float, interval: list[float], families: list[str]) -> dict:
    from cv.experiments.connected_shooting.auto_packet import automatic_physical_event

    lo, hi = min(interval[0], frame), max(interval[1], frame)
    event = automatic_physical_event(
        {
            "event_type": "contact",
            "frame": float(round(frame)),
            "frame_interval": [round(lo, 3), round(hi, 3)],
            "location": {"frame_subpixel": round(frame, 3), "source": "event_beam_time_witness"},
        }
    )
    event["note"] = "time-witness contact (" + "+".join(families) + ")"
    event["epoch_source"] = "event_beam_time_witness"
    event["time_witness_families"] = families
    return event


def edit_for(c: dict, a: Attempt, used_rows: set, taken: list[float]) -> dict | None:
    """Insert, or retime of an automatic contact 2-13 frames away (never a row the cascade
    settled); None if not editable."""
    from cv.pipeline.event_cascade_rebind import event_key

    f = c["frame"]
    if any(abs(f - t) <= CLEARANCE for t in taken):
        return None
    contacts = [e for e in a.inventory or [] if e.get("event_type") == "contact"]
    auto = min(contacts, key=lambda e: abs(float(e["frame"]) - f), default=None)
    d = abs(float(auto["frame"]) - f) if auto is not None else float("inf")
    if d <= TOL:
        return None
    event = contact_event(f, c["interval"], c["families"])
    decision = {"id": cand_id(c), "frame": f, "families": c["families"], "receipts": []}
    if d <= RETIME_NEAR:
        k = event_key(auto)
        if k in used_rows or k in a.settled:
            return None
        event["cascade_retimed_from"] = float(auto["frame"])
        return {
            "action": "retime",
            "key": list(k),
            "event": event,
            "decision": dict(decision, action="retime"),
        }
    others = [
        float(e["frame"]) for e in a.inventory or [] if e.get("event_type") in ("bounce", "net_hit")
    ]
    if any(abs(f - o) <= CLEARANCE for o in others):
        return None
    return {"action": "insert", "event": event, "decision": dict(decision, action="insert")}


def edits_for(a: Attempt, cands: list[dict]) -> list[dict]:
    edits, used, taken = [], set(), []
    for c in cands:
        e = edit_for(c, a, used, taken)
        if e is None:
            continue
        if e["action"] == "retime":
            used.add(tuple(e["key"]))
        taken.append(c["frame"])
        edits.append(e)
    return edits


def write_branch(a: Attempt, edits: list[dict], folder: Path) -> Path | None:
    """The edited packet (first-fit packet + the cascade's settled decisions + beam edits)
    and its manifest; None when the scope cannot take the edits."""
    from cv.pipeline import event_cascade_rebind as rebind

    try:
        packet, labels = rebind.rebind(a.packet, a.labels, a.cameras, a.settled, edits)
    except ValueError:
        return None
    return rebind.write_manifest(a.manifest, packet, labels, folder, a.settled, edits)


# --------------------------------------------------------------------------- the beam


def pruned(c: dict) -> bool:
    """``time_witness`` prune: no audio/S5-peak time witness, or a weak audio-only onset."""
    families = set(c["families"])
    if not families & {"audio", "s5_peak"}:
        return True
    return families == {"audio"} and (c["audio_z"] or 0.0) < PRUNE_AUDIO_Z


def _plan_attempt(
    match: Match,
    a: Attempt,
    st: dict,
    cands: list[dict],
    out: Path,
    r: int,
    prune: str = "off",
):
    """This round's branches for one attempt -> list of (name, manifest, folder)."""
    flights = flights_of(_result(st["incumbent"]))
    seen = sorted(a.px)
    span = (seen[0], seen[-1]) if seen else attempt_span(a.labels)
    accepted = [f for f in flights or [] if f["accepted"]]
    taken = [c["frame"] for c in st["cands"]]

    def usable(c):
        return (
            cand_id(c) not in st["tried"]
            and not any(abs(c["frame"] - t) <= CLEARANCE for t in taken)
            and not any(f["start"] + 3 < c["frame"] < f["end"] - 3 for f in accepted)
            and gate(c, a)
        )

    per_slot, dropped = [], []
    for lo, hi in slots(flights, span):
        ranked = [
            c
            for c in cands
            if c["relation"] != "duplicate"
            and (set(c["families"]) != {"audio"} or (c["audio_z"] or 0.0) >= Z_MIN)
            and lo - RETIME_NEAR <= c["frame"] <= hi + 2
            and usable(c)
        ]
        ranked.sort(key=rank, reverse=True)
        ranked = with_snaps([c for c in ranked if split_ok(a.px, c["frame"])], a.px)
        ranked = [c for c in ranked if usable(c) and bounce_ok(a, c["frame"])]
        ranked = ranked[:BRANCHES]
        if prune == "time_witness":
            dropped += [cand_id(c) for c in ranked if pruned(c)]
            ranked = [c for c in ranked if not pruned(c)]
        if ranked:
            per_slot.append({"slot": [lo, hi], "ranked": ranked})
    rec = {"slots": [], "branches": [], "cands": {}}
    if prune != "off":
        rec["pruned"] = dropped
    st["rounds"][str(r)] = rec
    rec["slots"] = [
        {"slot": s["slot"], "ranked": [cand_id(c) for c in s["ranked"]]} for s in per_slot
    ]
    rec["cands"] = {cand_id(c): c for s in per_slot for c in s["ranked"]}
    jobs = []
    for j in range(BRANCHES):
        chosen = [s["ranked"][j] for s in per_slot if len(s["ranked"]) > j]
        if not chosen:
            break
        edits = edits_for(a, st["cands"] + chosen)
        new = {e["decision"]["id"] for e in edits} - {cand_id(c) for c in st["cands"]}
        if not new:
            continue
        folder = out / a.key / f"r{r}" / f"b{j}"
        manifest = write_branch(a, edits, folder / "packet")
        branch = {
            "name": f"r{r}b{j}",
            "cands": sorted(new),
            "folder": str(folder),
            "edits_sha256": _edits_digest(edits),
        }
        if manifest is None:
            branch["error"] = "rebind_refused"
        else:
            jobs.append((a, manifest, folder))
        rec["branches"].append(branch)
        st["tried"] = sorted(set(st["tried"]) | new)
    return jobs


def _edits_digest(edits: list[dict]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(edits, sort_keys=True, default=str).encode()).hexdigest()


def _result(path) -> dict | None:
    return None if path is None or not Path(path).is_file() else _load(path)


def _fitted(folder) -> Path | None:
    path = Path(folder) / "stage" / "result.json"
    run = Path(folder) / "run.json"
    if not run.is_file() or _load(run).get("status") != "completed":
        return None
    return path if path.is_file() else None


def _compose_attempt(a: Attempt, st: dict, out: Path, r: int):
    rec = st["rounds"].get(str(r))
    if not rec:
        return []
    base = flights_of(_result(st["incumbent"])) or []
    results = {b["name"]: _fitted(b["folder"]) for b in rec["branches"] if "error" not in b}
    winners = []
    for slot in rec["slots"]:
        lo, hi = slot["slot"]
        window = (lo - RETIME_NEAR, hi + RETIME_NEAR)
        best = None
        for b in rec["branches"]:
            path = results.get(b["name"])
            mine = [c for c in b["cands"] if c in slot["ranked"]]
            if path is None or not mine:
                continue
            result = _load(path)
            fl = flights_of(result)
            before = sum(f["accepted"] and window[0] <= f["start"] <= window[1] for f in base)
            after = [f for f in fl if f["accepted"] and window[0] <= f["start"] <= window[1]]
            gain = len(after) - before
            if gain < 1 or lost_flights(base, fl, window):
                continue
            if not guard_ok(a, base, result, [rec["cands"][mine[0]]["frame"]]):
                continue
            rms = [f["rms"] for f in after if f["rms"] is not None]
            score = (gain, -(sum(rms) / len(rms)) if rms else 0.0)
            if best is None or score > best[0]:
                best = (score, mine[0], b["name"])
        slot["winner"] = None if best is None else {"cand": best[1], "branch": best[2]}
        if best is not None:
            winners.append(rec["cands"][best[1]])
    if not winners:
        return []
    cands = st["cands"] + winners
    edits = edits_for(a, cands)
    digest = _edits_digest(edits)
    for b in rec["branches"]:
        # A composition of one slot winner is often exactly its branch's edits (one slot, or
        # the other slots offered nothing): that fit is already done, and refitting the same
        # packet only reruns it.
        if b.get("edits_sha256") == digest and results.get(b["name"]) is not None:
            rec["composed"] = {
                "folder": b["folder"],
                "cands": [cand_id(c) for c in cands],
                "reused_branch": b["name"],
            }
            rec["composed_records"] = cands
            return []
    folder = out / a.key / f"r{r}" / "composed"
    manifest = write_branch(a, edits, folder / "packet")
    rec["composed"] = {"folder": str(folder), "cands": [cand_id(c) for c in cands]}
    if manifest is None:
        rec["composed"]["error"] = "rebind_refused"
        return []
    rec["composed_records"] = cands
    return [(a, manifest, folder)]


def _decide_attempt(a: Attempt, st: dict, r: int) -> None:
    rec = st["rounds"].get(str(r))
    if not rec:
        return
    base = flights_of(_result(st["incumbent"])) or []
    n0 = sum(f["accepted"] for f in base)
    options = []
    comp = rec.get("composed")
    if comp and "error" not in comp and _fitted(comp["folder"]):
        options.append((_fitted(comp["folder"]), 1, rec["composed_records"], "composed"))
    for b in rec["branches"]:
        path = None if "error" in b else _fitted(b["folder"])
        if path is not None:
            cands = st["cands"] + [rec["cands"][c] for c in b["cands"] if c in rec["cands"]]
            options.append((path, 0, cands, b["name"]))
    scored = []
    for path, pref, cands, label in options:
        result = _load(path)
        scored.append((sum(f["accepted"] for f in flights_of(result)), pref, path, cands, label))
    for n, _pref, path, cands, label in sorted(scored, key=lambda o: (o[0], o[1]), reverse=True):
        result = _load(path)
        if n <= n0 or lost_flights(base, flights_of(result)):
            continue
        if not guard_ok(a, base, result, [c["frame"] for c in cands]):
            continue
        st["incumbent"], st["cands"] = str(path), cands
        st["incumbent_manifest"] = str(Path(path).parents[1] / "packet" / "manifest.json")
        rec["decision"] = {"kept": label, "accepted_before": n0, "accepted_after": n}
        st["history"].append({"round": r, "incumbent": label})
        return
    rec["decision"] = {"kept": "incumbent", "accepted": n0}


def fit_one(a: Attempt, manifest: Path, folder: Path) -> dict:
    """One S6 fit of an edited manifest, cached in ``folder`` (resumable)."""
    from cv.pipeline.s6_broadcast_backend import run_stage

    run = folder / "run.json"
    if run.is_file():
        return _load(run)
    doc = _load(manifest)
    job = folder / "job.json"
    folder.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({"row": doc["rows"][0], "policy": doc["policy"]}) + "\n")
    stage = folder / "stage"
    if stage.exists():  # an interrupted fit starts afresh
        stage.rename(folder / f"stage.partial.{os.getpid()}")
    result = run_stage(job, stage, a.timeout_seconds)
    run.write_text(json.dumps(result, default=str) + "\n")
    return result


def shared_paint_cache(attempts: list[Attempt], shared: Path) -> Path:
    """One content-addressed court-paint cache for every beam fit of the match, seeded with
    the baselines' own entries: branches keep their attempt's cameras and native frames, so
    each would otherwise recompute the identical refit (a hit is exact; see
    ``s6_labeled_stage._paint_cameras``)."""
    from cv.pipeline.s6_labeled_stage import PAINT_CACHE_DIR

    shared.mkdir(parents=True, exist_ok=True)
    for a in attempts:
        source = a.baseline.parent / PAINT_CACHE_DIR
        for entry in sorted(source.glob("*.json")) if source.is_dir() else []:
            target = shared / entry.name
            if not target.exists():
                try:
                    os.link(entry, target)
                except OSError:
                    target.write_bytes(entry.read_bytes())
    return shared


def run(
    match: Match,
    attempts: list[Attempt],
    out: Path,
    fit_many,
    prune: str = "off",
    calibration: dict | None = None,
) -> dict:
    """The beam over one match. ``fit_many(list of (attempt, manifest, folder))`` runs the fits
    (any order, in parallel); state is kept in ``out/STATE.json`` so a rerun resumes. A caller
    that already set ``SHARED_PAINT_CACHE_ENV`` (a wider cache) keeps it. ``calibration``: the
    match's audio delay calibration from an earlier run of all its attempts (a reprocess of a
    few attempts would otherwise calibrate on those alone)."""
    from cv.pipeline.s6_labeled_stage import SHARED_PAINT_CACHE_ENV

    out.mkdir(parents=True, exist_ok=True)
    outer = os.environ.get(SHARED_PAINT_CACHE_ENV)
    shared = shared_paint_cache(attempts, Path(outer) if outer else out / "paint_cache")
    if outer:
        return _run(match, attempts, out, fit_many, prune, calibration)
    os.environ[SHARED_PAINT_CACHE_ENV] = str(shared)
    try:
        return _run(match, attempts, out, fit_many, prune, calibration)
    finally:
        os.environ.pop(SHARED_PAINT_CACHE_ENV, None)


def _run(
    match: Match,
    attempts: list[Attempt],
    out: Path,
    fit_many,
    prune: str,
    calibration: dict | None = None,
) -> dict:
    if prune not in PRUNE_MODES:
        raise ValueError(f"unknown beam prune mode {prune!r}")
    calibration = calibrate(match, attempts) if calibration is None else calibration
    (out / "CALIBRATION.json").write_text(json.dumps(calibration, indent=2) + "\n")
    lag = {"modes": calibration["modes"], "spread": calibration["spread"]}
    state_path = out / "STATE.json"
    state = _load(state_path) if state_path.is_file() else {}
    by_key = {}
    for a in attempts:
        by_key[a.key] = a
        st = state.setdefault(
            a.key,
            {
                "incumbent": str(a.baseline),
                "incumbent_manifest": None,
                "cands": [],
                "tried": [],
                "rounds": {},
                "history": [{"round": 0, "incumbent": "baseline"}],
            },
        )
        if a.inventory is None:
            st["skip"] = "uneditable_scope"
    cands = {k: candidates(match, a, lag) for k, a in by_key.items() if a.inventory is not None}

    def save():
        state_path.write_text(json.dumps(state, indent=1, default=str) + "\n")

    for r in range(1, ROUNDS + 1):
        jobs = []
        for key, a in by_key.items():
            st = state[key]
            if st.get("skip") or str(r) in st["rounds"]:
                jobs += [
                    (a, Path(b["folder"]) / "packet" / "manifest.json", Path(b["folder"]))
                    for b in (st["rounds"].get(str(r)) or {}).get("branches", [])
                    if "error" not in b
                ]
                continue
            jobs += _plan_attempt(match, a, st, cands[key], out, r, prune)
        save()
        fit_many(jobs)
        jobs = []
        for key, a in by_key.items():
            st = state[key]
            rec = st["rounds"].get(str(r))
            if st.get("skip") or not rec or "decision" in rec:
                continue
            jobs += _compose_attempt(a, st, out, r)
        save()
        fit_many(jobs)
        for key, a in by_key.items():
            st = state[key]
            rec = st["rounds"].get(str(r))
            if st.get("skip") or not rec or "decision" in rec:
                continue
            _decide_attempt(a, st, r)
        save()
    summary = {
        "schema": SCHEMA,
        "prune": prune,
        "calibration": calibration,
        "attempts": len(attempts),
        "candidates": sum(len(c) for c in cands.values()),
        "fits": sum(
            len(rec.get("branches", []))
            + ("composed" in rec and "reused_branch" not in rec["composed"])
            for st in state.values()
            for rec in st["rounds"].values()
        ),
        "attempts_changed": sum(
            st["incumbent"] != str(by_key[k].baseline) for k, st in state.items()
        ),
        "changed": {
            k: {
                "result": st["incumbent"],
                "manifest": st["incumbent_manifest"],
                "contacts": [
                    {"frame": c["frame"], "families": c["families"], "relation": c["relation"]}
                    for c in st["cands"]
                ],
            }
            for k, st in state.items()
            if st["incumbent"] != str(by_key[k].baseline)
        },
    }
    (out / "BEAM.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
