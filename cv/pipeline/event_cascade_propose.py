"""Paid event cascade PROPOSE/RETIME tier: ask for a contact the automatic events lack or mistime.

FREEZE v2 (``event_cascade``) can only confirm or reject an emitted row. This tier opens
label-free windows where the automatic structure says a contact is missing or mistimed
and asks a hosted model for the impact frame:

``locate``  a ±12-frame window inside a contact-to-contact gap longer than ``gap_s``. The
            stretch before a rally whose first automatic event is not a contact (``leading``)
            is not searched: that source declares its leading evidence, so a contact placed
            there contradicts the declaration and S6 refuses the whole attempt (measured:
            every leading insert lost the attempt or changed nothing). The stretch after the
            last automatic contact (``trailing``) is not searched: it is most of the cost and
            the detector's own scored stretch is the cheaper lever.
            Rule: one Gemini Flash draw on the ball-crop grid and one on the player crops;
            both ``no_contact`` settles none. Otherwise one Opus draw on the ball crops plus
            player crops (``ball_player``); a contact inside the window is proposed.
``retime``  an automatic contact that bounds a flight the first fit did not accept. One
            Opus draw on the FREEZE v2 ball-crop grid; a contact at least ``RETIME_MIN``
            frames from the automatic epoch moves it there.

Accepted answers become packet edits (``event_cascade_rebind``): a proposal is a new
supported automatic contact, a retime moves the row's epoch; both carry the model, and every
call's tokens and dollars are in ``propose/decisions.json`` and the spend ledger. Answers
come only from pictures; no evaluation label is read. Measured in
``cv/experiments/cascade_proposer/FINAL_REPORT_cascade_proposer.md``. The runner switch is
``--propose off|retime|retime_locate`` (default ``off``).

Evidence renderers (ball-track independent ``player`` crops) and the prompt were ported from
the recall probe (``cv/experiments/cascade_proposer``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from cv.pipeline import event_cascade_models as models
from cv.pipeline.event_cascade import Answer, answer_from_reply, reply_is_good

SCHEMA = "event_cascade_propose_v1"
DECISIONS_SCHEMA = "event_cascade_propose_decisions_v1"
MODES = ("locate", "retime")
#: ``off`` asks nothing; ``retime`` applies only retime edits; ``retime_locate`` both.
SWITCH_CHOICES = ("off", "retime", "retime_locate")
DEFAULT_SWITCH = "off"
CONTACT = "contact"
NONE_TYPES = ("no_contact", "no_event", "none")
OPUS_SIDE = 1280
#: A retime closer than this to the automatic epoch is not an edit.
RETIME_MIN = 2.0
#: A proposal this close to an existing physical row is not inserted.
INSERT_CLEARANCE = 2.0
HALF = 12
ACCEPT_MARGIN = 3.0
#: Seconds. See ``locate_centres``.
WINDOW_DEFAULTS = {"gap_s": 1.8, "margin_s": 0.4, "lead_s": 2.0, "trail_s": 3.0}
LOCATE_TRIGGERS = ("gap",)


@dataclass(frozen=True)
class ProposeTier:
    name: str
    model: str
    max_tokens: tuple[int, ...]
    reasoning: str | None


PROPOSE_TIERS = (
    ProposeTier("gemini", "google/gemini-3.8-flash", (8192, 16384), "low"),
    ProposeTier("opus", "anthropic/claude-opus-5.5", (8192, 16384), "low"),
)
PROPOSE_TIER_BY_NAME = {tier.name: tier for tier in PROPOSE_TIERS}
#: (tier, presentation) asked, in order, per mode.
LOCATE_SCREENS = (("gemini", "ball"), ("gemini", "player"))
LOCATE_SETTLER = ("opus", "ball_player")
RETIME_SETTLER = ("opus", "ball")


# --- prompt ------------------------------------------------------------------


def _rules() -> str:
    from cv.pipeline.event_cascade_frozen.fairlib import decision_rules

    return decision_rules()


IMAGES_TEXT = {
    "ball": """Each crop is 160 native pixels around an automatic ball track, magnified x8 with nearest-neighbour,
so one native pixel is an 8x8 block. The number in the top-left corner is the FRAME INDEX.
Nothing is drawn on the ball. Crops are centred on a tracker's estimate, which can drift or lose
the ball, so the ball may sit off-centre or leave the crop. The frames are a contiguous native
sequence, skipping only frames the clip does not have.
The CLEAN images come first. Use those.
You are then shown the SAME frames a second time, labelled FALLIBLE TRACKER HINT. Orange dots
are earlier automatic-track positions and cyan dots are later ones, faded with distance. The track
is often wrong: a dot is not the ball, and the absence of a dot is not the absence of a ball.
Judge the pixels.""",
    "player": """Each row is one native frame: the FAR player on the left, the NEAR player on the right.
Each side's crop is fixed for the whole window, cut around an automatic player track and padded
for racket reach, so the ball moves across a still background. The label in the top-left corner
of each tile is the FRAME INDEX and the side. Nothing is drawn on the ball. The ball is small
(a few pixels) and may be outside both crops. Frames are a contiguous native sequence, skipping
only frames the clip does not have. The player track may be wrong; judge the pixels.""",
    "full": """Each image is one whole broadcast frame, downscaled to 1280 px wide. The number in the
top-left corner is the FRAME INDEX. Only every second native frame is shown, so give the shown
frame closest to the impact. Nothing is drawn on the ball; it is small (a few pixels).""",
    "ball_player": """Two views of the same frames. First, BALL CROPS: 160 native pixels around an
automatic ball track, magnified x8, frame index top-left. The track can drift, lose the ball or
follow a ball kid, so the ball may be off-centre or absent. Then PLAYER CROPS: each row is one
native frame, the FAR player on the left and the NEAR player on the right, each side's crop fixed
for the window around an automatic player track and padded for racket reach; the tile label is
the FRAME INDEX and side. Nothing is drawn on the ball. Judge the pixels.""",
}


def preamble(mode: str, presentation: str = "ball") -> str:
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "locate":
        task = (
            "An automatic pass found no racket contact in this stretch of the rally, but the\n"
            "rally structure suggests a player may have hit the ball here. That suggestion is\n"
            "often wrong: the window may hold only a bounce, a ball in flight, a ball passing a\n"
            "player without being struck, or no visible ball. Answer no_contact freely."
        )
    else:
        task = (
            "An automatic pass placed a racket contact near the middle of this window, but its\n"
            "timing may be several frames off. Find the frame of the actual impact."
        )
    return f"""You are locating one tennis racket contact from native broadcast crops.

WHAT THE IMAGES ARE
{IMAGES_TEXT[presentation]}

THE TASK
{task}

WHAT TO DECIDE
  contact    - a racket (or body) strikes the ball inside the frames shown. Give its frame.
  no_contact - the frames show no racket strike (bounce only, flight only, a miss, no ball).
  abstain    - the ball or the strike cannot be resolved from these crops. Do not guess.

Give the printed frame index of the impact: the frame where the ball meets the strings. If the
instant falls between two frames, give the frame where the ball is closest to the racket.
A bounce is not a contact. A serve toss, trophy pose or swing without the ball at the racket is
not a contact. For no_contact or abstain, set frame to null.

{_rules()}

The JSON format above is for another task. For THIS task return exactly one JSON object and
nothing else:
{{"id":"<the id>","event_type":"contact|no_contact|abstain","frame":<int or null>,"confidence":"high|medium|low","evidence":"brief visible reason"}}
"""


def target_text(nom: dict, mode: str) -> str:
    frames = nom["frames"]
    if mode == "locate":
        return (
            f"Unlabeled window id {nom['id']}, frames {frames[0]} to {frames[-1]}. "
            "Is there a racket contact? Return the JSON object now."
        )
    return (
        f"Unlabeled window id {nom['id']}, frames {frames[0]} to {frames[-1]}. "
        f"The automatic contact is near frame {nom['nominated_frame']}; that time is not truth. "
        "Return the JSON object now."
    )


def build_parts(nom: dict, mode: str, presentation: str = "ball") -> list[dict]:
    """Instructions, the presentation's images in order, then the window sentence."""
    from cv.pipeline.event_cascade_frozen.run_label import ordered_images

    parts: list[dict] = [{"type": "text", "text": preamble(mode, presentation).rstrip() + "\n\n"}]
    parts.append({"type": "text", "text": "UNLABELED WINDOW images:\n"})
    images = ordered_images(nom, "trail") if presentation == "ball" else nom["images"]
    names = {"trail": "FALLIBLE TRACKER HINT", "clean": "CLEAN", "ball": "BALL CROPS"}
    for index, image in enumerate(images, start=1):
        frames = ",".join(str(frame) for frame in image["frames"])
        hint = names.get(image["kind"], image["kind"].upper())
        parts.append({"type": "text", "text": f"Image {index}: {hint} frames {frames}\n"})
        parts.append({"type": "image", "path": image["path"]})
    parts.append({"type": "text", "text": "\n" + target_text(nom, mode)})
    return parts


# --- answers -----------------------------------------------------------------


def normalise(answer: Answer) -> Answer:
    """``contact`` with a frame, ``none``, or unsettled (``abstain`` or unreadable)."""
    kind = (answer.event_type or "").strip().lower()
    if kind == CONTACT and answer.frame is not None:
        return answer
    if kind in NONE_TYPES:
        return Answer("none", None, answer.usd, answer.tokens, answer.confidence)
    return Answer(None, None, answer.usd, answer.tokens, answer.confidence)


def read(path: Path) -> Answer | None:
    if not reply_is_good(path):
        return None
    return normalise(answer_from_reply(json.loads(path.read_text())))


def _receipt(tier: str, presentation: str, answer: Answer, window: dict) -> dict:
    return {
        "tier": tier,
        "model": PROPOSE_TIER_BY_NAME[tier].model,
        "presentation": presentation,
        "window": window["id"],
        "usd": float(answer.usd),
        "tokens": dict(answer.tokens or {}),
        "answer": {"event_type": answer.event_type, "frame": answer.frame},
    }


def _in_window(answer: Answer, window: dict) -> bool:
    return answer.event_type == CONTACT and window["start"] <= float(answer.frame) <= window["end"]


def settle(window: dict, answer_for) -> dict:
    """The deployable rule for one window. ``answer_for(tier, presentation)`` returns a saved
    normalised Answer or None (owed). A contact outside the rendered window is not a contact."""
    receipts = []
    if window["mode"] == "locate":
        screens = []
        for tier, presentation in LOCATE_SCREENS:
            answer = answer_for(tier, presentation)
            if answer is None:
                return {"owed": (tier, presentation), "receipts": receipts}
            receipts.append(_receipt(tier, presentation, answer, window))
            screens.append(answer)
        if all(answer.event_type == "none" for answer in screens):
            return {"owed": None, "action": "none", "tier": "gemini", "receipts": receipts}
        tier, presentation = LOCATE_SETTLER
    else:
        tier, presentation = RETIME_SETTLER
    answer = answer_for(tier, presentation)
    if answer is None:
        return {"owed": (tier, presentation), "receipts": receipts}
    receipts.append(_receipt(tier, presentation, answer, window))
    if not _in_window(answer, window):
        action = "none" if answer.event_type == "none" else "unsettled"
        return {"owed": None, "action": action, "tier": tier, "receipts": receipts}
    frame = float(answer.frame)
    if window["mode"] == "retime" and abs(frame - float(window["event_frame"])) < RETIME_MIN:
        return {
            "owed": None,
            "action": "keep",
            "frame": frame,
            "tier": tier,
            "model": PROPOSE_TIER_BY_NAME[tier].model,
            "receipts": receipts,
        }
    return {
        "owed": None,
        "action": "insert" if window["mode"] == "locate" else "retime",
        "frame": frame,
        "tier": tier,
        "model": PROPOSE_TIER_BY_NAME[tier].model,
        "receipts": receipts,
    }


# --- windows -----------------------------------------------------------------


def _subtract(regions, spans, margin: float = ACCEPT_MARGIN):
    out = []
    for start, end in regions:
        pieces = [(start, end)]
        for left, right in spans:
            left, right = left + margin, right - margin
            nxt = []
            for a, b in pieces:
                if right <= a or left >= b:
                    nxt.append((a, b))
                    continue
                if left > a:
                    nxt.append((a, left))
                if right < b:
                    nxt.append((right, b))
            pieces = nxt
        out.extend(piece for piece in pieces if piece[1] - piece[0] >= 4)
    return out


def locate_centres(
    events: list[dict],
    *,
    fps: float,
    window: tuple[float, float],
    accepted: list[tuple[float, float]],
    params: dict | None = None,
    triggers: tuple[str, ...] = ("leading", "gap", "trailing"),
) -> list[dict]:
    """Label-free missing-contact windows of one attempt.

    ``events`` are the attempt's automatic physical events (any occurrence status),
    ``window`` its native frame range clipped to the last observed ball sample, and
    ``accepted`` the first fit's accepted flight spans, which are never searched.
    A contact-to-contact gap longer than ``gap_s`` is searched ``margin_s`` inside its
    bounding contacts; a rally whose first event is not a contact is searched ``lead_s``
    back from it; the stretch after the last contact up to ``trail_s`` when longer than
    ``gap_s``. Returns ``{"centre", "trigger"}`` rows, one per ±12-frame window.
    """
    chosen = dict(WINDOW_DEFAULTS, **(params or {}))
    ordered = sorted(
        (e for e in events if e.get("event_type") in ("contact", "bounce", "net_hit")),
        key=lambda e: float(e["frame"]),
    )
    if not ordered:
        return []
    contacts = [float(e["frame"]) for e in ordered if e["event_type"] == "contact"]
    gap, margin = chosen["gap_s"] * fps, chosen["margin_s"] * fps
    regions: list[tuple[float, float, str]] = []
    first = float(ordered[0]["frame"])
    if ordered[0]["event_type"] != "contact":
        regions.append((max(window[0], first - chosen["lead_s"] * fps), first - 3, "leading"))
    for left, right in zip(contacts, contacts[1:], strict=False):
        if right - left > gap:
            regions.append((left + margin, right - margin, "gap"))
    if contacts and window[1] - contacts[-1] > gap:
        end = min(window[1], contacts[-1] + chosen["trail_s"] * fps)
        regions.append((contacts[-1] + margin, end, "trailing"))
    rows = []
    span = 2 * HALF + 1
    for start, end, trigger in regions:
        if trigger not in triggers:
            continue
        for a, b in _subtract([(start, end)], accepted):
            count = max(1, math.ceil((b - a) / span - 0.01))
            step = (b - a) / count
            rows += [
                {"centre": int(round(a + step * (i + 0.5))), "trigger": trigger}
                for i in range(count)
            ]
    return rows


def retime_centres(events: list[dict], rejected: list[tuple[float, float]]) -> list[dict]:
    """Automatic contacts that bound a flight the first fit did not accept."""
    rows = []
    for event in events:
        if event.get("event_type") != "contact":
            continue
        frame = float(event["frame"])
        if any(abs(frame - a) < 1 or abs(frame - b) < 1 for a, b in rejected):
            rows.append({"centre": int(round(frame)), "trigger": "retime", "event_frame": frame})
    return rows


def attempt_windows(
    *,
    job: dict,
    plan: dict,
    events: list[dict],
    ball_frames: list[float],
    verdict: dict,
    fps: float,
    match: Path,
    frames_dir: str,
    track_csv: str,
) -> list[dict]:
    """Every locate (gap) and retime window of one fitted automatic attempt."""
    flights = verdict.get("flights") or []
    native = plan.get("original_native_window") or [1, 10**6]
    last_ball = max(ball_frames) if ball_frames else native[1]
    span = (float(native[0]), float(min(native[1], last_ball)))
    accepted = [(f["start_frame"], f["end_frame"]) for f in flights if f.get("accepted")]
    rejected = [(f["start_frame"], f["end_frame"]) for f in flights if not f.get("accepted")]
    rows = locate_centres(
        events, fps=fps, window=span, accepted=accepted, triggers=LOCATE_TRIGGERS
    ) + retime_centres(events, rejected)
    point = plan["clip"]
    out = []
    seen = set()
    for row in rows:
        key = (row["trigger"], row["centre"])
        if key in seen:
            continue
        seen.add(key)
        mode = "retime" if row["trigger"] == "retime" else "locate"
        out.append(
            {
                "id": f"{job['attempt']}__{row['trigger']}{row['centre']}",
                "attempt": job["attempt"],
                "panel": job["panel"],
                "match_id": plan["match_id"],
                "point": point,
                "clip": f"{plan['match_id']}__{point}",
                "mode": mode,
                "trigger": row["trigger"],
                "centre": row["centre"],
                "event_frame": row.get("event_frame"),
                "start": row["centre"] - HALF,
                "end": row["centre"] + HALF,
                "frame_dir": frames_dir,
                "track_csv": track_csv,
                "match_dir": str(match),
            }
        )
    return out


# --- evidence ----------------------------------------------------------------

PLAYER_TILE = 448
PLAYER_ROWS = 3
PLAYER_PAD = 0.9
PLAYER_MIN = 240
PLAYERS = "player_tracks_native_v1.csv"
SIDES = ("far", "near")


def _stamp(panel, text: str, scale: float = 1.0):
    import cv2

    width = int(24 + 22 * len(text) * scale)
    cv2.rectangle(panel, (0, 0), (width, int(46 * scale)), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (8, int(36 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1 * scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _write(path: Path, image) -> dict:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def window_frames(folder: Path, centre: int, stride: int = 1) -> list[int]:
    frames = range(centre - HALF, centre + HALF + 1)
    have = [f for f in frames if (folder / f"f_{f:04d}.jpg").is_file()]
    if stride == 1:
        return have
    # Keep the centre's parity so both halves are sampled alike.
    return [f for f in have if (f - centre) % stride == 0]


def player_boxes(match: Path, point: str) -> dict[tuple[int, str], tuple[float, ...]]:
    out = {}
    with (match / PLAYERS).open() as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != point or row["side"] not in SIDES:
                continue
            try:
                box = tuple(
                    float(row[k]) for k in ("x0_native", "y0_native", "x1_native", "y1_native")
                )
            except (TypeError, ValueError):
                continue
            out[(int(row["frame"]), row["side"])] = box
    return out


def side_crop(boxes: list[tuple[float, ...]], width: int, height: int) -> tuple[int, int, int]:
    """(left, top, side) of one fixed square crop covering a side's boxes plus racket reach."""
    import numpy as np

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    tall = float(np.median([b[3] - b[1] for b in boxes]))
    pad = PLAYER_PAD * tall
    side = int(max(PLAYER_MIN, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad))
    side = min(side, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.15 * tall
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side / 2, 0, height - side))
    return left, top, side


def render_player(window: dict, out_dir: Path, match: Path | None = None) -> dict:
    """Both players' fixed crops for every native frame of the window, three rows per image."""
    import cv2
    import numpy as np

    folder = Path(window["frame_dir"])
    match = Path(window.get("match_dir") or match or folder.parents[1])
    frames = window_frames(folder, int(window["centre"]))
    boxes = player_boxes(match, window["point"])
    images = {f: cv2.imread(str(folder / f"f_{f:04d}.jpg"), cv2.IMREAD_COLOR) for f in frames}
    height, width = images[frames[0]].shape[:2]
    crops = {}
    for side in SIDES:
        seen = [boxes[(f, side)] for f in frames if (f, side) in boxes]
        if not seen:
            # Widen to the nearest tracked frames rather than dropping the side.
            near = sorted(
                (k for k in boxes if k[1] == side), key=lambda k: abs(k[0] - window["centre"])
            )[:6]
            seen = [boxes[k] for k in near]
        crops[side] = side_crop(seen, width, height) if seen else None
    tiles = []
    for frame in frames:
        pair = []
        for side in SIDES:
            if crops[side] is None:
                tile = np.full((PLAYER_TILE, PLAYER_TILE, 3), 60, np.uint8)
                _stamp(tile, f"{frame} {side}: no player track")
            else:
                left, top, size = crops[side]
                crop = images[frame][top : top + size, left : left + size]
                tile = cv2.resize(crop, (PLAYER_TILE, PLAYER_TILE), interpolation=cv2.INTER_AREA)
                _stamp(tile, f"{frame} {side}")
            cv2.rectangle(tile, (0, 0), (PLAYER_TILE - 1, PLAYER_TILE - 1), (40, 40, 40), 2)
            pair.append(tile)
        tiles.append((frame, cv2.hconcat(pair)))
    record = {
        "clip": window["clip"],
        "id": window["id"],
        "frames": frames,
        "presentation": "player",
        "crops": {s: list(c) if c else None for s, c in crops.items()},
        "images": [],
    }
    for start in range(0, len(tiles), PLAYER_ROWS):
        chunk = tiles[start : start + PLAYER_ROWS]
        canvas = cv2.vconcat([t for _f, t in chunk])
        name = f"{window['id']}_f{chunk[0][0]:04d}_{chunk[-1][0]:04d}.jpg"
        meta = _write(out_dir / "player" / window["clip"] / name, canvas)
        meta.update(kind="player", frames=[f for f, _t in chunk])
        record["images"].append(meta)
    return record


def render_ball(window: dict, evidence: Path) -> dict:
    """The FREEZE v2 25-frame grids centred on ``window['centre']``."""
    return models.render(
        {
            "clip": window["clip"],
            "id": window["id"],
            "point": window["point"],
            "nominated_frame": int(window["centre"]),
            "event_type": "contact",
            "frame_dir": window["frame_dir"],
            "track_csv": window["track_csv"],
        },
        evidence,
    )


def render(window: dict, presentation: str, out_dir: Path) -> dict:
    """Render or reuse one window's evidence for ``presentation`` (ball, player, ball_player)."""
    if presentation == "ball":
        nom = render_ball(window, out_dir / "one" / "pergrid")
        return {**nom, "nominated_frame": int(window["centre"])}
    manifest = out_dir / "manifests" / presentation / window["clip"] / f"{window['id']}.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if all(Path(i["path"]).is_file() for i in saved["images"]):
            return saved
    if presentation == "player":
        record = render_player(window, out_dir)
    elif presentation == "ball_player":
        ball = render(window, "ball", out_dir)
        player = render(window, "player", out_dir)
        clean = [dict(i, kind="ball") for i in ball["images"] if i["kind"] == "clean"]
        record = {
            "clip": window["clip"],
            "id": window["id"],
            "frames": player["frames"],
            "presentation": "ball_player",
            "images": clean + player["images"],
        }
    else:
        raise ValueError(presentation)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record))
    return record


def opus_sized(nom: dict, evidence: Path) -> dict:
    """FREEZE v2 ball grids go to Opus at OPUS_SIDE px; player pictures are already sized."""
    grids = [image for image in nom["images"] if image["kind"] in ("clean", "trail", "ball")]
    if not grids:
        return nom
    kinds = {id(image): image["kind"] for image in grids}
    small = models.resized(
        {**nom, "images": [dict(image, kind="clean") for image in grids]}, OPUS_SIDE, evidence
    )
    resized = iter(small["images"])
    images = []
    for image in nom["images"]:
        if id(image) in kinds:
            images.append(dict(next(resized), kind=kinds[id(image)]))
        else:
            images.append(image)
    return {**nom, "images": images}


# --- calls -------------------------------------------------------------------


def reply_path(root: Path, tier: str, presentation: str, window: dict) -> Path:
    return root / "replies" / presentation / tier / window["clip"] / f"{window['id']}.json"


def call(
    window: dict,
    nom: dict,
    tier: str,
    presentation: str,
    *,
    root: Path,
    ledger: Path,
    cap_usd: float,
    path: Path | None = None,
) -> str:
    """One hosted draw. A reply that already parses is never bought twice."""
    from cv.pipeline.event_cascade_frozen import run_label

    out = path or reply_path(root, tier, presentation, window)
    if reply_is_good(out):
        return f"skip {window['id']} {tier} {presentation}"
    spec = PROPOSE_TIER_BY_NAME[tier]
    if tier == "opus":
        nom = opus_sized(nom, root / "evidence" / "one")
    parts = build_parts(nom, window["mode"], presentation)
    image_extra, body_extra = run_label.provider_kwargs(spec.model)
    last: dict = {}
    for cap in spec.max_tokens:
        with models.SPEND_LOCK:
            if models.spent(ledger) + 0.08 > cap_usd:
                raise SystemExit(f"propose spend cap ${cap_usd} reached")
        started = time.time()
        try:
            last = models._with_402_retry(
                lambda cap=cap: run_label.openrouter(
                    spec.model, parts, cap, spec.reasoning, image_extra, body_extra, None
                )
            )
        except Exception as error:  # noqa: BLE001 - recorded, then the next cap is tried
            detail = ""
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = error.read().decode(errors="replace")[:400]
                except Exception:  # noqa: BLE001
                    detail = ""
            last = {"raw": "", "finish_reason": "error", "usage": {}, "error": f"{error} {detail}"}
            time.sleep(5)
            continue
        verdict = run_label.parse_verdict(last.get("raw") or "")
        usage = last.get("usage") or {}
        usd = float(usage.get("cost") or 0)
        saved = {
            "schema": SCHEMA,
            "id": window["id"],
            "clip": window["clip"],
            "mode": window["mode"],
            "frames": nom["frames"],
            "model": spec.model,
            "tier": tier,
            "presentation": presentation,
            "images": [image["path"] for image in nom["images"]],
            "evidence_px": OPUS_SIDE if tier == "opus" else None,
            "finish_reason": last.get("finish_reason"),
            "max_tokens": cap,
            "verdict": verdict,
            "raw": last.get("raw") or "",
            "usage": usage,
            "usd": usd,
            "dt": round(time.time() - started, 1),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with models.SPEND_LOCK, ledger.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "cost": usd,
                        "model": spec.model,
                        "tier": tier,
                        "presentation": presentation,
                        "window": window["id"],
                        "prompt": usage.get("prompt_tokens"),
                        "completion": usage.get("completion_tokens"),
                    }
                )
                + "\n"
            )
        if verdict is None or last.get("finish_reason") in {"length", "max_tokens"}:
            out.with_suffix(".truncated.json").write_text(json.dumps(saved))
            continue
        out.write_text(json.dumps(saved))
        return (
            f"ok {window['id']} {tier} {presentation} "
            f"{verdict.get('event_type')} {verdict.get('frame')} ${usd:.4f}"
        )
    return f"FAIL {window['id']} {tier} {presentation} {last.get('error') or last.get('finish_reason')}"


# --- packet edits ------------------------------------------------------------


def track_position(track_csv: Path, point: str, frame: float) -> tuple[float, float] | None:
    """Observed native ball position at the rounded frame, if the automatic track has one."""
    target = int(round(frame))
    with Path(track_csv).open() as handle:
        for sample in csv.DictReader(handle):
            name = sample.get("clip") or ""
            if name != point and not name.endswith(point):
                continue
            stem = Path(sample["frame"]).stem
            index = int(stem[2:]) if stem.startswith("f_") else int(float(stem))
            if index != target or "interpolated" in (sample.get("sources") or ""):
                continue
            for xs, ys in (("x_native", "y_native"), ("x1080", "y1080")):
                if sample.get(xs) not in (None, "") and sample.get(ys) not in (None, ""):
                    return float(sample[xs]), float(sample[ys])
            return None
    return None


def cascade_fields(decision: dict) -> dict:
    return {
        "cascade_tier": 4,
        "cascade_source": f"propose_{decision['tier']}",
        "cascade_model": decision["model"],
        "cascade_window": decision["window"],
    }


def inserted_event(decision: dict, position: tuple[float, float] | None) -> dict:
    """A new supported automatic contact at the proposed frame."""
    from cv.experiments.connected_shooting.auto_packet import automatic_physical_event

    frame = float(round(float(decision["frame"])))
    location = {"frame_subpixel": frame, "source": "event_cascade_propose"}
    if position is not None:
        location.update(
            image_x=round(position[0], 2),
            image_y=round(position[1], 2),
            image_coordinate_space="native_1920x1080",
        )
    event = automatic_physical_event({"event_type": CONTACT, "frame": frame, "location": location})
    event["note"] = "cascade proposal of a contact the automatic events lack"
    event["epoch_source"] = "event_cascade_propose"
    return {**event, **cascade_fields(decision)}


def retimed_event(event: dict, decision: dict, position: tuple[float, float] | None) -> dict:
    """The same automatic contact at the model's impact frame, still a supported claim."""
    frame = float(round(float(decision["frame"])))
    row = {
        key: value
        for key, value in event.items()
        if key not in ("automatic_abstention", "original_prediction_search_interval")
    }
    location = dict(event.get("automatic_location") or {})
    for key in ("image_x", "image_y", "image_coordinate_space", "court_transport"):
        location.pop(key, None)
    location.update(frame_subpixel=frame, source="event_cascade_retime")
    if position is not None:
        location.update(
            image_x=round(position[0], 2),
            image_y=round(position[1], 2),
            image_coordinate_space="native_1920x1080",
        )
    row.update(
        frame=frame,
        frame_interval=[frame - 1.0, frame + 1.0],
        interval_origin="prediction_search_radius_one_native_frame_v1",
        status="predicted",
        occurrence_status="predicted",
        emitted_rounded_frame=frame,
        epoch_source="event_cascade_retime",
        automatic_location=location,
        note="cascade retime of an automatic contact",
        cascade_retimed_from=float(event["frame"]),
    )
    row.pop("prediction_search_native_window", None)
    return {**row, **cascade_fields(decision)}


def editable_inventory(packet: dict) -> list[dict] | None:
    """The physical rows a cascade edit may touch, or None for a scope it cannot edit."""
    from cv.pipeline import s6_contact_prefix_scope as prefix

    attempt = packet["attempts"][0]
    schema = (attempt.get("observation_scope") or {}).get("schema")
    if schema == prefix.SCHEMA:
        return attempt["original_physical_events"]
    if schema == prefix.UNRESOLVED_INPUT_SCHEMA:
        return attempt["events"]
    return None


def propose_edits(
    decisions: list[dict],
    inventory: list[dict],
    settled: dict,
    *,
    switch: str,
    track_csv: Path,
    point: str,
) -> tuple[list[dict], list[dict]]:
    """Packet edits for one attempt from its window decisions, and the ones dropped (why).

    ``settled`` are the attempt's FREEZE v2 confirms and rejects keyed like
    ``event_cascade_rebind.event_key``: a retime of a settled row is dropped (the cascade
    decision stands), a rejected row is not an obstacle to an insert. An insert from a
    trigger outside ``LOCATE_TRIGGERS`` (a stored older decision) is dropped. An insert within
    ``INSERT_CLEARANCE`` frames of a kept physical row, or of an earlier insert, is dropped.
    """
    from cv.pipeline.event_cascade_rebind import event_key

    if switch not in SWITCH_CHOICES:
        raise ValueError(f"propose switch must be one of {SWITCH_CHOICES}")
    wanted = {"off": (), "retime": ("retime",), "retime_locate": ("retime", "insert")}[switch]
    rows = {event_key(e): e for e in inventory if e.get("event_type") == CONTACT}
    edits, dropped = [], []
    moved: dict[tuple, float] = {}
    for decision in sorted(decisions, key=lambda d: (d["action"] != "retime", d["id"])):
        if decision["action"] not in wanted:
            continue
        if decision["action"] == "retime":
            key = next((k for k in rows if abs(k[1] - float(decision["event_frame"])) < 1e-4), None)
            if key is None:
                dropped.append({"id": decision["id"], "why": "retimed_row_not_in_inventory"})
                continue
            if key in settled:
                dropped.append({"id": decision["id"], "why": "cascade_settled_row"})
                continue
            if key in moved:
                dropped.append({"id": decision["id"], "why": "row_already_retimed"})
                continue
            moved[key] = float(round(decision["frame"]))
            position = track_position(track_csv, point, decision["frame"])
            edits.append(
                {
                    "action": "retime",
                    "key": list(key),
                    "event": retimed_event(rows[key], decision, position),
                    "decision": decision,
                }
            )
    kept = [
        moved.get(event_key(e), float(e["frame"]))
        for e in inventory
        if e.get("event_type") in ("contact", "bounce", "net_hit")
        and not (event_key(e) in settled and settled[event_key(e)]["action"] == "reject")
    ]
    for decision in sorted(decisions, key=lambda d: d["id"]):
        if decision["action"] != "insert" or "insert" not in wanted:
            continue
        if decision.get("trigger") not in LOCATE_TRIGGERS:
            dropped.append({"id": decision["id"], "why": "trigger_not_searched"})
            continue
        frame = float(round(decision["frame"]))
        if any(abs(frame - other) <= INSERT_CLEARANCE for other in kept):
            dropped.append({"id": decision["id"], "why": "near_existing_row"})
            continue
        kept.append(frame)
        position = track_position(track_csv, point, frame)
        edits.append(
            {"action": "insert", "event": inserted_event(decision, position), "decision": decision}
        )
    return edits, dropped


def edit_label(edit: dict) -> str:
    decision = edit["decision"]
    if edit["action"] == "retime":
        return f"retime:contact:{float(decision['event_frame'])}->{float(round(decision['frame']))}"
    return f"insert:contact:{float(round(decision['frame']))}"
