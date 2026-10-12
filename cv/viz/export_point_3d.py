"""export_point_3d.py — per-point 3D reconstruction exporter for the portal 3D viewer.

Reads the refined rich-ball-physics artifacts for a point (per-frame 3D ball state +
contacts + bounces + quality) plus player court positions, and writes a compact,
self-describing JSON (one file per point) that ``portal/3d/viewer.js`` renders as a
pan-around WebGL scene. This is the 3D successor to ``cv/viz/point_replay.py`` (the 2D
top-down renderer); it consumes the SAME court frame (RICH_CHART_SPEC: x across
[0, 10.97] m, y along [0, 23.77] m, z up).

WHAT IT READS (per artifact "tag" = the stem of a rich-physics triple):
  <tag>.csv.gz          per-frame ball state (rich_ball_physics.py schema):
                        frame, x, y, z, vx, vy, vz, spin_x/y/z, confidence,
                        ci95_x/y/z_m, velocity_ci95_ms, spin_ci95_rad_s, segment,
                        fit_rms_px, source
  <tag>_contacts.csv    per-contact records: timing interval (frame_lo/hi, t_lo/hi_s,
                        timing_source), side, phase, status ("fit" vs
                        "unsupported_dead_or_same_side"), x/y/z (+ci95), v_in/v_out,
                        speed_in/out, racket_normal_* + face_yaw/pitch (physics-implied
                        proxy, NOT observed pose), and honest CIs.
  <tag>_bounces.csv     per-bounce: segment, frame, x/y/z, v_in/v_out, regime, fit_rms_px
  <tag>_quality.json    the acceptance gate + per-clip stats (accepted flag carried through
                        so the viewer can flag diagnostic-candidate points honestly)

Player positions come from the Player Ledger CSV (cv/pipeline/player_ledger.py emit-ledger)
when one is supplied via --ledger, else from the tracked box artifact's court_x/court_y
(player_boxes_*_v2.csv) — exactly the fallback point_replay.py uses.

A single ``<tag>.csv.gz`` may contain several clips (the ``clip`` column distinguishes
them); per-point campaign tags (atscale_ptNNNN_iN, prune_loop_ptNNNN_iN) carry one clip.
Either way the exporter groups by clip and writes one JSON per clip, so the atscale
campaign's ~270 extra points are a single glob:

  python cv/viz/export_point_3d.py --match rg2025f --artifact 'atscale_pt*_i1' \
      --out data/processed/review_queue_v1/portal/3d/data

Honesty: every emitted physical quantity carries its CI and its per-frame confidence /
segment / RMS, so the viewer can distinguish fitted from low-confidence spans and render
missing frames as gaps (never smoothed over). Speeds inherit the spec's UNCALIBRATED flag.

Uses the standard library only (no cv2 / pandas) so it runs under the system interpreter.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict

_CLIP_RE = re.compile(r"^pt\d{3,4}$")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Court frame (camera_cal.py / court.py — regulation, RICH_CHART_SPEC convention).
COURT_W = 10.97
COURT_L = 23.77
NET_Y = COURT_L / 2.0            # 11.885
SERVICE_FROM_NET = 6.40
SINGLES_INSET = (10.97 - 8.23) / 2.0   # 1.37
NET_CENTER_H = 0.914             # net tape height at center
NET_POST_H = 1.07                # net height at the posts (height-validation anchor)

FPS = {"rg2025f": 50.0, "uso2025f": 59.94005994005994}

# Broadcast reference-monitor frames per match (served by review_serve.py at the funnel root
# via the review_queue_v1/<match> symlink, i.e. /<match>/<frames_dir>/<clip>/f_%04d.jpg).
# Artifact frame numbers are LOCAL clip frame indices and equal the f_%04d.jpg file numbers
# (verified: pt0006 ball frames 308-383, contacts f_0040.., dir f_0001..f_0450 -> direct map),
# so the viewer maps a playhead second t -> round(t*fps) -> f_<frame:04d>.jpg under this dir.
FRAMES_DIR = {"rg2025f": "rally_frames_50_1080_v2", "uso2025f": "rally_frames_60_1080"}
FRAMES_PATTERN = "f_%04d.jpg"

# Tracked-box court-position artifact per match (player_court_v2.py) — the ledger fallback.
BOXES_V2 = {
    "rg2025f": "player_boxes_50_full_v2.csv",
    "uso2025f": "player_boxes_59.9401_full_v2.csv",
}

SCHEMA = "point3d_v1"
INDEX_SCHEMA = "point3d_index_v1"

# Per-frame confidence below this (or RMS above RMS_LOW_PX) is flagged low-confidence so the
# viewer can fade/dash it. These only DERIVE a convenience label; raw conf/rms are always
# emitted too so the viewer can threshold differently.
CONF_LOW = 0.15
RMS_LOW_PX = 8.0


def _num(row: dict, key: str):
    """Parse a CSV cell to float, or None when blank/missing/non-numeric."""
    val = row.get(key)
    if val is None:
        return None
    val = val.strip()
    if val == "" or val.lower() in ("nan", "none"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _r(val, ndigits=4):
    """Round for compact JSON; pass through None."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return round(val, ndigits)


def _frame_int(raw: str) -> int:
    """'f_0181.jpg' or '181' or '181.5' -> int frame index."""
    raw = raw.strip()
    if raw.startswith("f_"):
        return int(raw[2:6])
    return int(round(float(raw)))


def _read_csv(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_paths(base: str, stem: str) -> dict:
    return {
        "ball": os.path.join(base, f"{stem}.csv.gz"),
        "contacts": os.path.join(base, f"{stem}_contacts.csv"),
        "bounces": os.path.join(base, f"{stem}_bounces.csv"),
        "quality": os.path.join(base, f"{stem}_quality.json"),
    }


def _norm3(a, b, c):
    parts = [v for v in (a, b, c) if v is not None]
    if len(parts) < 3:
        return None
    return math.sqrt(a * a + b * b + c * c)


# ---------------------------------------------------------------------------
# Player positions
# ---------------------------------------------------------------------------
def load_players(match: str, clip: str, ledger_path: str | None,
                 players_csv: str | None = None) -> dict:
    """Return {"near": {frame:[...], t:[...], x:[...], y:[...], conf:[...], track_id:[...]},
    "far": {...}, "names": {"near": name|None, "far": name|None}, "source": str}.

    Preference: an explicit ``players_csv`` (or the auto-discovered per-clip
    ``player_court_v3_<clip>.csv`` — feet-on-ground court positions, the 2026-07-24 fix) >
    a Player Ledger CSV > the tracked-box court_x/court_y artifact (player_court_v2). The v3
    source uses the SMOOTHED (airborne-held, speed-gated) court positions and also carries
    per-frame grounded/flags so the viewer can surface held/serve-stance frames."""
    fps = FPS.get(match, 50.0)
    sides = {"near": defaultdict(dict), "far": defaultdict(dict)}
    names = {"near": None, "far": None}
    source = "none"

    # v3 feet-on-ground positions: explicit path, else auto-discover the per-clip file
    v3_path = players_csv
    if not v3_path:
        cand = os.path.join(REPO, "data", "processed", match, f"player_court_v3_{clip}.csv")
        if os.path.exists(cand):
            v3_path = cand

    if v3_path and os.path.exists(v3_path):
        source = "player_court_v3"
        for row in _read_csv(v3_path):
            if row.get("clip") != clip:
                continue
            side = row.get("side")
            if side not in sides:
                continue
            f = _frame_int(row["frame"])
            cx = _num(row, "court_x_smooth")
            cy = _num(row, "court_y_smooth")
            if cx is None or cy is None:
                continue
            sides[side][f] = {
                "x": cx, "y": cy,
                "conf": _num(row, "conf"),
                "track_id": row.get("source", "") or None,
                "grounded": (row.get("grounded", "") or "").strip() in ("1", "True", "true"),
                "flags": (row.get("flags", "") or "").strip(),
            }
    elif ledger_path and os.path.exists(ledger_path):
        source = "player_ledger"
        for row in _read_csv(ledger_path):
            if row.get("clip") != clip:
                continue
            f = _frame_int(row["frame"])
            for side in ("near", "far"):
                cx = _num(row, f"{side}_court_x")
                cy = _num(row, f"{side}_court_y")
                if cx is None or cy is None:
                    continue
                sides[side][f] = {
                    "x": cx, "y": cy,
                    "conf": _num(row, f"{side}_conf"),
                    "track_id": row.get(f"{side}_track_id", "") or None,
                }
                nm = row.get(f"{side}_player", "") or None
                if nm and not names[side]:
                    names[side] = nm
    else:
        boxes = os.path.join(REPO, "data", "processed", match, BOXES_V2.get(match, ""))
        if os.path.exists(boxes):
            source = "player_boxes_v2"
            for row in _read_csv(boxes):
                if row.get("clip") != clip:
                    continue
                side = row.get("side")
                if side not in sides:
                    continue
                f = _frame_int(row["frame"])
                cx, cy = _num(row, "court_x"), _num(row, "court_y")
                if cx is None or cy is None:
                    continue
                sides[side][f] = {
                    "x": cx, "y": cy,
                    "conf": _num(row, "conf"),
                    "track_id": row.get("track_id", "") or None,
                }

    out = {"names": names, "source": source}
    for side in ("near", "far"):
        frames = sorted(sides[side])
        col = {
            "frame": frames,
            "t": [_r(f / fps, 4) for f in frames],
            "x": [_r(sides[side][f]["x"], 3) for f in frames],
            "y": [_r(sides[side][f]["y"], 3) for f in frames],
            "conf": [_r(sides[side][f]["conf"], 3) for f in frames],
            "track_id": [sides[side][f]["track_id"] for f in frames],
        }
        if source == "player_court_v3":  # additive per-frame provenance for the viewer
            col["grounded"] = [bool(sides[side][f].get("grounded", True)) for f in frames]
            col["flags"] = [sides[side][f].get("flags", "") for f in frames]
        out[side] = col
    return out


# ---------------------------------------------------------------------------
# Ball frames
# ---------------------------------------------------------------------------
def build_frames(rows: list[dict], match: str) -> dict:
    fps = FPS.get(match, 50.0)
    rows = sorted(rows, key=lambda r: _frame_int(r["frame"]))
    cols = {k: [] for k in (
        "frame", "t", "x", "y", "z", "vx", "vy", "vz", "speed",
        "spin_x", "spin_y", "spin_z", "spin_mag",
        "conf", "ci_xy", "ci_z", "spin_ci", "rms_px", "segment", "method",
    )}
    for row in rows:
        f = _frame_int(row["frame"])
        vx, vy, vz = _num(row, "vx"), _num(row, "vy"), _num(row, "vz")
        sx, sy, sz = _num(row, "spin_x"), _num(row, "spin_y"), _num(row, "spin_z")
        conf = _num(row, "confidence")
        rms = _num(row, "fit_rms_px")
        low = (conf is not None and conf < CONF_LOW) or (rms is not None and rms > RMS_LOW_PX)
        cols["frame"].append(f)
        cols["t"].append(_r(f / fps, 4))
        cols["x"].append(_r(_num(row, "x"), 3))
        cols["y"].append(_r(_num(row, "y"), 3))
        cols["z"].append(_r(_num(row, "z"), 3))
        cols["vx"].append(_r(vx, 3))
        cols["vy"].append(_r(vy, 3))
        cols["vz"].append(_r(vz, 3))
        cols["speed"].append(_r(_norm3(vx, vy, vz), 3))
        cols["spin_x"].append(_r(sx, 2))
        cols["spin_y"].append(_r(sy, 2))
        cols["spin_z"].append(_r(sz, 2))
        cols["spin_mag"].append(_r(_norm3(sx, sy, sz), 2))
        cols["conf"].append(_r(conf, 3))
        cols["ci_xy"].append(_r(_num(row, "ci95_x_m"), 3))
        cols["ci_z"].append(_r(_num(row, "ci95_z_m"), 3))
        cols["spin_ci"].append(_r(_num(row, "spin_ci95_rad_s"), 1))
        cols["rms_px"].append(_r(rms, 2))
        cols["segment"].append(row.get("segment", "") or None)
        cols["method"].append("low_conf" if low else "fit")
    return cols


def _spin_at_contact(frames: dict, cframe: float, window: int = 10) -> dict:
    """Outgoing spin for a contact: the first fitted ball-state frame at/after the contact
    (within `window` frames). Reported with its honest CI. None when unavailable."""
    if not frames["frame"]:
        return {"mag": None, "x": None, "y": None, "z": None, "ci": None, "frame": None}
    target = int(round(cframe))
    best = None
    for i, f in enumerate(frames["frame"]):
        if target <= f <= target + window:
            best = i
            break
    if best is None:  # fall back to nearest within +/- window
        for i, f in enumerate(frames["frame"]):
            if abs(f - target) <= window and (best is None or abs(f - target) < abs(frames["frame"][best] - target)):
                best = i
    if best is None:
        return {"mag": None, "x": None, "y": None, "z": None, "ci": None, "frame": None}
    return {
        "mag": frames["spin_mag"][best], "x": frames["spin_x"][best],
        "y": frames["spin_y"][best], "z": frames["spin_z"][best],
        "ci": frames["spin_ci"][best], "frame": frames["frame"][best],
    }


# ---------------------------------------------------------------------------
# Contacts / bounces
# ---------------------------------------------------------------------------
def build_contacts(rows: list[dict], frames: dict, match: str) -> list[dict]:
    fps = FPS.get(match, 50.0)
    out = []
    for row in rows:
        frame = _num(row, "frame")
        if frame is None:
            continue
        status = row.get("status", "") or "unknown"
        z = _num(row, "z")
        spin = _spin_at_contact(frames, frame)
        out.append({
            "index": int(_num(row, "contact_index") or len(out)),
            "frame": _r(frame, 2),
            "frame_lo": _num(row, "frame_lo"),
            "frame_hi": _num(row, "frame_hi"),
            "t": _r(frame / fps, 4),
            "t_lo": _r(_num(row, "t_lo_s"), 4),
            "t_hi": _r(_num(row, "t_hi_s"), 4),
            "timing_source": row.get("timing_source", "") or None,
            "timing_residual_px": _r(_num(row, "timing_residual_px"), 2),
            "side": row.get("side", "") or "unknown",
            "phase": row.get("phase", "") or "rally",
            "status": status,
            "fit": status == "fit",
            "x": _r(_num(row, "x"), 3),
            "y": _r(_num(row, "y"), 3),
            "z": _r(z, 3),
            "ci_x": _r(_num(row, "ci95_x_m"), 3),
            "ci_z": _r(_num(row, "ci95_z_m"), 3),
            "speed_in": _r(_num(row, "speed_in"), 2),
            "speed_out": _r(_num(row, "speed_out"), 2),
            "v_in": [_r(_num(row, "vx_in"), 3), _r(_num(row, "vy_in"), 3), _r(_num(row, "vz_in"), 3)],
            "v_out": [_r(_num(row, "vx_out"), 3), _r(_num(row, "vy_out"), 3), _r(_num(row, "vz_out"), 3)],
            "velocity_ci": _r(_num(row, "velocity_ci95_ms"), 3),
            # racket state: PHYSICS-IMPLIED proxy from the velocity impulse, not observed pose
            "racket_normal": [_r(_num(row, "racket_normal_x"), 3), _r(_num(row, "racket_normal_y"), 3),
                              _r(_num(row, "racket_normal_z"), 3)],
            "racket_normal_speed": _r(_num(row, "racket_normal_speed_ms"), 2),
            "racket_face_yaw_deg": _r(_num(row, "racket_face_yaw_deg"), 1),
            "racket_face_pitch_deg": _r(_num(row, "racket_face_pitch_deg"), 1),
            "racket_normal_ci_deg": _r(_num(row, "racket_normal_ci95_deg"), 1),
            "racket_speed_ci": _r(_num(row, "racket_speed_ci95_ms"), 2),
            "rms_in_px": _r(_num(row, "rms_in_px"), 2),
            "rms_out_px": _r(_num(row, "rms_out_px"), 2),
            "spin_out": spin,
        })
    out.sort(key=lambda c: (c["frame"] if c["frame"] is not None else 0))
    return out


def build_bounces(rows: list[dict], match: str) -> list[dict]:
    fps = FPS.get(match, 50.0)
    out = []
    for row in rows:
        frame = _num(row, "frame")
        if frame is None:
            continue
        x, y = _num(row, "x"), _num(row, "y")
        in_court = (
            x is not None and y is not None
            and -0.3 <= x <= COURT_W + 0.3 and -0.3 <= y <= COURT_L + 0.3
        )
        out.append({
            "segment": row.get("segment", "") or None,
            "frame": _r(frame, 2),
            "t": _r(frame / fps, 4),
            "x": _r(x, 3), "y": _r(y, 3), "z": _r(_num(row, "z"), 3),
            "v_in": [_r(_num(row, "vx_in"), 3), _r(_num(row, "vy_in"), 3), _r(_num(row, "vz_in"), 3)],
            "v_out": [_r(_num(row, "vx_out"), 3), _r(_num(row, "vy_out"), 3), _r(_num(row, "vz_out"), 3)],
            "regime": row.get("regime", "") or None,
            "rms_px": _r(_num(row, "fit_rms_px"), 2),
            "in_court": in_court,
        })
    out.sort(key=lambda b: (b["frame"] if b["frame"] is not None else 0))
    return out


def quality_for_clip(quality: dict | None, clip: str) -> dict:
    if not quality:
        return {"accepted": None, "tier1_labels_consumed": None, "clip": None}
    entry = None
    for c in quality.get("clips", []):
        if c.get("clip") == clip:
            entry = c
            break
    return {
        "accepted": entry.get("accepted") if entry else quality.get("accepted"),
        "gate_accepted_overall": quality.get("accepted"),
        "tier1_labels_consumed": quality.get("tier1_labels_consumed"),
        "clip": _clean_json(entry) if entry else None,
        "gate": _clean_json(quality.get("gate")),
    }


def _clean_json(obj):
    """Replace NaN/inf with None recursively so the JSON is strictly valid."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Per-clip export
# ---------------------------------------------------------------------------
def export_clip(match: str, clip: str, stem: str, ball_rows, contact_rows, bounce_rows,
                quality, ledger_path, players_csv=None) -> dict:
    frames = build_frames(ball_rows, match)
    contacts = build_contacts(contact_rows, frames, match)
    bounces = build_bounces(bounce_rows, match)
    players = load_players(match, clip, ledger_path, players_csv)

    fnums = frames["frame"]
    all_f = list(fnums)
    all_f += [int(round(c["frame"])) for c in contacts if c["frame"] is not None]
    all_f += [int(round(b["frame"])) for b in bounces if b["frame"] is not None]
    for side in ("near", "far"):
        all_f += players[side]["frame"]
    frame_lo = min(all_f) if all_f else 0
    frame_hi = max(all_f) if all_f else 0

    spin_present = any(v is not None and v != 0 for v in frames["spin_mag"])

    try:
        point = int(clip.replace("pt", ""))
    except ValueError:
        point = None

    # Broadcast frame directory for the synchronized reference-monitor panel. Emit the dir
    # + whether this clip actually has frames on disk, so the viewer shows the panel only
    # when it can honestly fill it (and hides it with a note otherwise).
    frames_dir = FRAMES_DIR.get(match)
    frames_available = False
    if frames_dir:
        clip_dir = os.path.join(REPO, "data", "processed", match, frames_dir, clip)
        if os.path.isdir(clip_dir):
            try:
                frames_available = any(n.endswith(".jpg") for n in os.listdir(clip_dir))
            except OSError:
                frames_available = False

    return {
        "schema": SCHEMA,
        "match": match,
        "clip": clip,
        "point": point,
        "tag": stem,
        "fps": FPS.get(match, 50.0),
        "force_follow": True,
        "trail": {
            "recent": 0.55,
            "ghost": 0.015,
            "fade": 0.25,
            "full": 0.95,
        },
        "frames_dir": frames_dir,
        "frames_pattern": FRAMES_PATTERN,
        "frames_available": frames_available,
        "court": {
            "width": COURT_W, "length": COURT_L, "net_y": NET_Y,
            "singles_inset": SINGLES_INSET, "service_from_net": SERVICE_FROM_NET,
            "net_center_h": NET_CENTER_H, "net_post_h": NET_POST_H,
        },
        "quality": quality_for_clip(quality, clip),
        "frame_range": [frame_lo, frame_hi],
        "counts": {
            "frames": len(fnums),
            "contacts": len(contacts),
            "contacts_fit": sum(1 for c in contacts if c["fit"]),
            "bounces": len(bounces),
            "bounces_in_court": sum(1 for b in bounces if b["in_court"]),
        },
        "spin_present": spin_present,
        "speed_caveat": "UNCALIBRATED (pre broadcast-radar cross-check; B2 rally v_out ~20% high)",
        "skeleton_caveat": "Player pose was not exported for this legacy reconstruction.",
        "frames": frames,
        "contacts": contacts,
        "bounces": bounces,
        "players": players,
        "skeletons": {"near": [], "far": []},
    }


def resolve_stems(base: str, patterns: list[str]) -> list[str]:
    """Expand --artifact globs to artifact stems (the part before .csv.gz)."""
    stems = []
    seen = set()
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(base, pat + ".csv.gz")))
        if not matches:  # allow passing a bare stem that exists
            direct = os.path.join(base, pat + ".csv.gz")
            if os.path.exists(direct):
                matches = [direct]
        for m in matches:
            stem = os.path.basename(m)[:-len(".csv.gz")]
            if stem not in seen:
                seen.add(stem)
                stems.append(stem)
    return stems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="rg2025f")
    ap.add_argument("--artifact", action="append", default=[],
                    help="artifact stem or glob (relative to data/processed/<match>/), "
                         "repeatable, e.g. 'atscale_pt*_i1' or 'knot_composed_v1'")
    ap.add_argument("--dev9", action="store_true",
                    help="convenience: the 9 dev9 points via prune_loop_ptNNNN_i1 + pt0092 via "
                         "knot_composed_v1 (the shipped initial set)")
    ap.add_argument("--ledger", default=None, help="optional player_ledger emit-ledger CSV")
    ap.add_argument("--players-csv", default=None,
                    help="explicit player_court_v3 CSV (else auto-discover "
                         "player_court_v3_<clip>.csv, which takes preference over the ledger/boxes)")
    ap.add_argument("--out", default=os.path.join(
        REPO, "data", "processed", "review_queue_v1", "portal", "3d", "data"))
    ap.add_argument("--no-index", action="store_true", help="skip index.json (re)write")
    args = ap.parse_args()

    base = os.path.join(REPO, "data", "processed", args.match)
    patterns = list(args.artifact)
    if args.dev9:
        patterns += [f"prune_loop_pt{p:04d}_i1" for p in (21, 23, 27, 117, 119, 121, 134, 164, 165)]
        patterns += ["knot_composed_v1"]  # pt0092
    if not patterns:
        ap.error("nothing to export: pass --artifact <glob> and/or --dev9")

    stems = resolve_stems(base, patterns)
    if not stems:
        print(f"no artifacts matched under {base} for: {patterns}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    written = []  # (clip, filename, summary)
    for stem in stems:
        paths = _artifact_paths(base, stem)
        if not os.path.exists(paths["ball"]):
            print(f"skip {stem}: no {os.path.basename(paths['ball'])}", file=sys.stderr)
            continue
        ball = _read_csv(paths["ball"])
        contacts = _read_csv(paths["contacts"])
        bounces = _read_csv(paths["bounces"])
        quality = None
        if os.path.exists(paths["quality"]):
            with open(paths["quality"]) as h:
                quality = json.load(h, parse_constant=lambda _c: None)

        by_clip = defaultdict(lambda: {"ball": [], "contacts": [], "bounces": []})
        n_bad = 0
        for kind, rowset in (("ball", ball), ("contacts", contacts), ("bounces", bounces)):
            for r in rowset:
                clip = (r.get("clip") or "").strip()
                if not _CLIP_RE.match(clip):  # drop upstream-malformed rows (shifted fields)
                    n_bad += 1
                    continue
                by_clip[clip][kind].append(r)
        if n_bad:
            print(f"   warn {stem}: dropped {n_bad} malformed row(s) with invalid clip id",
                  file=sys.stderr)

        for clip, groups in sorted(by_clip.items()):
            doc = export_clip(args.match, clip, stem, groups["ball"], groups["contacts"],
                              groups["bounces"], quality, args.ledger, args.players_csv)
            fname = f"{args.match}_{clip}.json"
            with open(os.path.join(args.out, fname), "w") as h:
                json.dump(doc, h, separators=(",", ":"))
            written.append((clip, fname, doc))
            print(f"-> {fname}  frames={doc['counts']['frames']} "
                  f"contacts={doc['counts']['contacts']}({doc['counts']['contacts_fit']} fit) "
                  f"bounces={doc['counts']['bounces']} spin={'y' if doc['spin_present'] else 'n'} "
                  f"accepted={doc['quality']['accepted']}")

    if not args.no_index and written:
        write_index(args.out, args.match)
    return 0


def write_index(out_dir: str, match: str) -> None:
    """(Re)build index.json from every *_pt*.json in the out dir — all matches, so
    incremental exports (dev9, atscale, other tournaments) accumulate into one picker
    index instead of the latest match clobbering the rest."""
    import datetime
    entries = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_pt*.json"))):
        with open(path) as h:
            doc = json.load(h)
        dur = None
        fr = doc.get("frame_range")
        if fr and doc.get("fps"):
            dur = _r((fr[1] - fr[0]) / doc["fps"], 2)
        qc = (doc.get("quality") or {}).get("clip") or {}
        entries.append({
            "clip": doc["clip"], "point": doc["point"], "match": doc["match"],
            "file": os.path.basename(path), "tag": doc["tag"],
            "accepted": doc["quality"]["accepted"],
            "n_frames": doc["counts"]["frames"],
            "n_contacts": doc["counts"]["contacts"],
            "n_contacts_fit": doc["counts"].get("contacts_fit"),
            "n_bounces": doc["counts"]["bounces"],
            "spin_present": doc["spin_present"], "duration_s": dur,
            "frames_available": doc.get("frames_available"),
            # quality signals so the picker can rank diagnostics best-first without loading
            # every point JSON (junction agreement in metres, fit RMS in px, bounce-in rate)
            "junction_med": _r(qc.get("junction_median_pass2"), 4),
            "rms_med": _r(qc.get("rms_median_px"), 2),
            "bounce_in_court_rate": _r(qc.get("bounce_in_court_rate"), 3),
        })
    entries.sort(key=lambda e: (e["match"], e["point"] is None, e["point"] or 0))
    index = {
        "schema": INDEX_SCHEMA,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "count": len(entries),
        "points": entries,
    }
    # atomic: a reader mid-rebuild must never see a truncated index
    final = os.path.join(out_dir, "index.json")
    tmp = final + ".tmp"
    with open(tmp, "w") as h:
        json.dump(index, h, separators=(",", ":"))
    os.replace(tmp, final)
    print(f"-> index.json ({len(entries)} points)")


if __name__ == "__main__":
    raise SystemExit(main())
