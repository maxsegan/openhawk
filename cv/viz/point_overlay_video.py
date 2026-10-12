"""Render native broadcast point overlays from automatic pipeline artifacts.

The renderer is intentionally a consumer only: it does not repair tracks, invent frames,
or change any acceptance decision.  It draws the evidence the automatic runner emitted.

Example::

    python -m cv.viz.point_overlay_video \
      --root "$TENNIS_DATA_ROOT/processed/wk3_e2e/qf_v2" \
      --match-id atp_2025_0425_qf_381_arthur_fils_stefanos_tsitsipas \
      --select-twelve --seed 20260902
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080
TRAIL_FRAMES = 10
EVENT_HOLD_FRAMES = 5
ACCEPTED_FLIGHT_STATUS = "provisional_valid"

# Pipeline court frame: doubles width x full length in metres.
COURT_WIDTH = 10.97
COURT_LENGTH = 23.77
SINGLES_INSET = (COURT_WIDTH - 8.23) / 2.0
SERVICE_FROM_NET = 6.40
NET_Y = COURT_LENGTH / 2.0
COURT_LINES_M = (
    ((0.0, 0.0), (COURT_WIDTH, 0.0)),
    ((0.0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),
    ((0.0, 0.0), (0.0, COURT_LENGTH)),
    ((COURT_WIDTH, 0.0), (COURT_WIDTH, COURT_LENGTH)),
    ((SINGLES_INSET, 0.0), (SINGLES_INSET, COURT_LENGTH)),
    ((COURT_WIDTH - SINGLES_INSET, 0.0), (COURT_WIDTH - SINGLES_INSET, COURT_LENGTH)),
    (
        (SINGLES_INSET, NET_Y - SERVICE_FROM_NET),
        (COURT_WIDTH - SINGLES_INSET, NET_Y - SERVICE_FROM_NET),
    ),
    (
        (SINGLES_INSET, NET_Y + SERVICE_FROM_NET),
        (COURT_WIDTH - SINGLES_INSET, NET_Y + SERVICE_FROM_NET),
    ),
    ((COURT_WIDTH / 2.0, NET_Y - SERVICE_FROM_NET), (COURT_WIDTH / 2.0, NET_Y + SERVICE_FROM_NET)),
    ((0.0, NET_Y), (COURT_WIDTH, NET_Y)),
)

# OpenCV BGR. Distinct hues make provenance readable without a legend lookup.
PROVENANCE_COLORS = {
    "crop": (224, 80, 224),
    "full": (40, 210, 255),
    "far-native": (255, 190, 40),
    "interpolated": (170, 170, 170),
}
EVENT_COLORS = {
    "contact": (70, 70, 255),
    "bounce": (40, 220, 255),
    "net_hit": (255, 90, 40),
    "point_end": (220, 220, 220),
}


def frame_number(value: str | int | float) -> int:
    """Return the integer frame number used by point-scoped artifacts."""
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    match = re.search(r"(\d+)(?:\.[^.]+)?$", Path(value).name)
    if match is None:
        raise ValueError(f"cannot parse frame number from {value!r}")
    return int(match.group(1))


def point_clip(point: str | int) -> str:
    if isinstance(point, int) or str(point).isdigit():
        return f"pt{int(point):04d}"
    value = str(point)
    match = re.search(r"pt(\d+)$", value)
    if match is None:
        raise ValueError(f"cannot parse point from {point!r}")
    return f"pt{int(match.group(1)):04d}"


def provenance_kind(sources: str) -> str:
    value = sources.lower().replace("_", "-")
    explicit = re.search(r"(?:^|\+)provenance:(crop|far-native|interpolated|coarse)(?:\+|$)", value)
    if explicit is not None:
        return "full" if explicit.group(1) == "coarse" else explicit.group(1)
    if "interpol" in value or "gap-fill" in value:
        return "interpolated"
    if "far-native" in value:
        return "far-native"
    if "crop" in value:
        return "crop"
    return "full"


def project(P: np.ndarray, xyz: Iterable[float], scale: float = 1.0) -> tuple[int, int] | None:
    homogeneous = np.asarray([*xyz, 1.0], dtype=float)
    image = np.asarray(P, dtype=float) @ homogeneous
    if not np.isfinite(image).all() or abs(float(image[2])) < 1e-9:
        return None
    uv = scale * image[:2] / image[2]
    if not np.isfinite(uv).all():
        return None
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def accepted_flight_counts(flight_ledger: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in flight_ledger.get("rows", []):
        if row.get("status") != ACCEPTED_FLIGHT_STATUS:
            continue
        clip = row.get("clip") or point_clip(row["point"])
        counts[point_clip(clip)] += 1
    return dict(counts)


def select_six_points(clips: Iterable[str], flight_ledger: dict[str, Any], seed: int) -> list[str]:
    """Choose three highest accepted-flight counts and three seeded random others."""
    available = sorted({point_clip(clip) for clip in clips})
    if len(available) < 6:
        raise ValueError(f"six overlays requested but only {len(available)} point clips exist")
    counts = accepted_flight_counts(flight_ledger)
    ranked = sorted(available, key=lambda clip: (-counts.get(clip, 0), clip))
    top = ranked[:3]
    remainder = [clip for clip in available if clip not in top]
    random_points = sorted(random.Random(seed).sample(remainder, 3))
    return top + random_points


def select_twelve_points(
    clips: Iterable[str],
    flight_ledger: dict[str, Any],
    active_play: dict[str, dict[str, Any]],
    match_id: str,
    seed: int,
) -> list[str]:
    """Choose six highest accepted-flight counts and six seeded in-play fitted controls."""
    available = sorted({point_clip(clip) for clip in clips})
    if len(available) < 12:
        raise ValueError(f"twelve overlays requested but only {len(available)} point clips exist")
    counts = accepted_flight_counts(flight_ledger)
    fitted = {
        point_clip(row.get("clip") or row["point"])
        for row in flight_ledger.get("rows", [])
        if row.get("status") != "unsolved"
    }
    ranked = sorted(available, key=lambda clip: (-counts.get(clip, 0), clip))
    top = ranked[:6]
    in_play = [
        clip
        for clip in available
        if clip not in top
        and clip in fitted
        and active_play.get(f"{match_id}/{clip}", {}).get("point_valid") is True
        and bool(active_play.get(f"{match_id}/{clip}", {}).get("active_spans"))
    ]
    if len(in_play) < 6:
        raise ValueError(f"six random in-play overlays requested but only {len(in_play)} exist")
    random_points = sorted(random.Random(seed).sample(in_play, 6))
    return top + random_points


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _clip_matches(value: Any, match_id: str, clip: str) -> bool:
    text = str(value or "")
    return text == clip or text == f"{match_id}__{clip}" or text.endswith(f"__{clip}")


def _artifact_rows_by_frame(
    rows: Iterable[dict[str, Any]], match_id: str, clip: str
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _clip_matches(row.get("clip"), match_id, clip):
            continue
        result[frame_number(row["frame"])].append(row)
    return dict(result)


def _point_row(ledger_rows: list[dict[str, str]], clip: str) -> dict[str, str]:
    number = int(clip[2:])
    return next((row for row in ledger_rows if int(row.get("pt", -1)) == number), {})


def _scoreboard_names(reads_path: Path, match_id: str) -> tuple[str, str]:
    """Return the modal ordered scoreboard names, expanded when the match is known."""
    pairs: Counter[tuple[str, str]] = Counter()
    try:
        lines = reads_path.read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            raw = json.loads(line).get("raw", {})
        except json.JSONDecodeError:
            continue
        rows = raw.get("rows", [])
        if len(rows) != 2:
            continue
        names = tuple(str(row.get("name", "")).strip().upper() for row in rows)
        if all(names):
            pairs[names] += 1

    full_names = {
        "ALCARAZ": "Carlos Alcaraz",
        "FILS": "Arthur Fils",
        "SINNER": "Jannik Sinner",
        "TSITSIPAS": "Stefanos Tsitsipas",
    }
    fallback = {
        "rg2025f": ("SINNER", "ALCARAZ"),
        "rg2025f_m_alcaraz_sinner": ("SINNER", "ALCARAZ"),
        "atp_2025_0425_qf_381_arthur_fils_stefanos_tsitsipas": ("TSITSIPAS", "FILS"),
    }
    ordered = pairs.most_common(1)[0][0] if pairs else fallback.get(match_id)
    if ordered is None:
        return "player 1", "player 2"
    return tuple(full_names.get(name, name.title()) for name in ordered)


def write_s1_point_table(match_dir: Path, match_id: str, output: Path) -> Path:
    """Write the automatic point ledger's score, server, and player identities."""
    ledger = _read_csv(match_dir / "automatic_point_ledger_v1.csv")
    names = _scoreboard_names(match_dir / "vlm_reads.jsonl", match_id)
    fields = [
        "match_id",
        "point",
        "rally_t_start",
        "rally_t_end",
        "set_number",
        "sets_won_1",
        "sets_won_2",
        "games_1",
        "games_2",
        "points_1",
        "points_2",
        "player_1_name",
        "player_2_name",
        "server",
        "server_name",
        "confidence",
        "outcome_hint",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ledger:
            try:
                server = int(row.get("server", ""))
            except ValueError:
                server = 0
            writer.writerow(
                {
                    "match_id": match_id,
                    "point": point_clip(row["pt"]),
                    **{field: row.get(field, "") for field in fields[2:11]},
                    "player_1_name": names[0],
                    "player_2_name": names[1],
                    "server": server or "",
                    "server_name": names[server - 1] if server in {1, 2} else "",
                    "confidence": row.get("confidence", ""),
                    "outcome_hint": row.get("outcome_hint", ""),
                }
            )
    return output


def _server_side(
    point_detail: dict[str, Any],
    events: list[dict[str, Any]],
    players: dict[int, list[dict[str, Any]]],
    track: dict[int, list[dict[str, Any]]],
) -> str | None:
    fits = point_detail.get("fits", [])
    if fits:
        first = min(fits, key=lambda fit: float(fit.get("start_frame", 1e12)))
        side = first.get("start_side")
        if side in {"near", "far"}:
            return str(side)
    contacts = sorted(
        (event for event in events if event.get("event_type") == "contact"),
        key=lambda event: float(event.get("frame", 1e12)),
    )
    if contacts:
        frame = frame_number(contacts[0]["frame"])
        candidates = players.get(frame, [])
        # The server is normally the deeper box. This is a fallback only; unresolved is preferable.
        if len(candidates) == 2:
            return max(
                candidates, key=lambda row: abs(float(row.get("court_y", 11.885)) - 11.885)
            ).get("side")

    # Held clips often have no accepted contact. Near the clip start, the tracked ball should be
    # beside the server during the toss/strike, so use its distance to the sided player boxes.
    shared_frames = sorted(set(track) & set(players))
    if shared_frames:
        last_frame = shared_frames[0] + 60
        distances: dict[str, float] = {"near": float("inf"), "far": float("inf")}
        for frame in (value for value in shared_frames if value <= last_frame):
            for ball in track[frame]:
                x = float(ball.get("x_native") or 2.0 * float(ball["x"]))
                y = float(ball.get("y_native") or 2.0 * float(ball["y"]))
                for player in players[frame]:
                    side = str(player.get("side", ""))
                    if side not in distances:
                        continue
                    x0 = float(player.get("x0_native") or 2.0 * float(player["x0"]))
                    y0 = float(player.get("y0_native") or 2.0 * float(player["y0"]))
                    x1 = float(player.get("x1_native") or 2.0 * float(player["x1"]))
                    y1 = float(player.get("y1_native") or 2.0 * float(player["y1"]))
                    dx = max(x0 - x, 0.0, x - x1)
                    dy = max(y0 - y, 0.0, y - y1)
                    distances[side] = min(distances[side], float(np.hypot(dx, dy)))
        side, distance = min(distances.items(), key=lambda item: item[1])
        if distance <= 160.0:
            return side
    return None


def _side_names(
    scoreboard_names: tuple[str, str],
    ledger_row: dict[str, str],
    point_detail: dict[str, Any],
    events: list[dict[str, Any]],
    players: dict[int, list[dict[str, Any]]],
    track: dict[int, list[dict[str, Any]]],
) -> dict[str, str]:
    first_name, second_name = scoreboard_names
    try:
        server_index = int(ledger_row.get("server", ""))
    except ValueError:
        server_index = 0
    side = _server_side(point_detail, events, players, track)
    if server_index not in {1, 2} or side not in {"near", "far"}:
        return {"near": "identity abstain", "far": "identity abstain"}
    server_name = (first_name, second_name)[server_index - 1]
    receiver_name = (second_name, first_name)[server_index - 1]
    other = "far" if side == "near" else "near"
    return {side: server_name, other: receiver_name}


def _arc_for_frame(arcs: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    for arc in arcs:
        if int(arc["start_frame"]) <= frame <= int(arc["end_frame"]):
            return arc
    return None


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.62,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_court(
    image: np.ndarray, P: np.ndarray | None, reliable: bool, camera_scale: float
) -> None:
    if P is None:
        return
    color = (80, 235, 80) if reliable else (90, 90, 255)
    for first, second in COURT_LINES_M:
        a = project(P, (*first, 0.0), camera_scale)
        b = project(P, (*second, 0.0), camera_scale)
        if a is not None and b is not None:
            cv2.line(image, a, b, color, 2, cv2.LINE_AA)


def draw_overlay_frame(
    image: np.ndarray,
    *,
    frame: int,
    clip: str,
    track: dict[int, list[dict[str, Any]]],
    players: dict[int, list[dict[str, Any]]],
    side_names: dict[str, str],
    events: dict[int, list[dict[str, Any]]],
    arcs: list[dict[str, Any]],
    camera: dict[int, tuple[np.ndarray, bool]],
    trajectories: dict[int, list[dict[str, Any]]],
    camera_scale: float,
) -> np.ndarray:
    canvas = image.copy()
    arc = _arc_for_frame(arcs, frame)
    if arc is not None:
        retained = arc.get("decision") == "retain"
        tint = np.full_like(canvas, (35, 120, 35) if retained else (30, 30, 150))
        canvas = cv2.addWeighted(canvas, 0.94, tint, 0.06, 0.0)
        label = f"arc {arc.get('arc_id', '?')}  {'RETAINED' if retained else 'HELD'}"
        _draw_text(canvas, label, (28, 68), (100, 255, 120) if retained else (100, 120, 255), 0.8)

    P, reliable = camera.get(frame, (None, False))
    draw_court(canvas, P, reliable, camera_scale)
    _draw_text(canvas, f"{clip}  frame {frame:04d}", (28, 35), scale=0.7)
    _draw_text(
        canvas,
        f"camera: {'reliable' if reliable else 'ABSTAIN / fallback'}",
        (28, 98),
        (110, 255, 120) if reliable else (110, 140, 255),
        0.58,
    )

    # Draw an observation-only ten-frame tail. Missing frames remain visibly missing.
    for trail_frame in range(max(1, frame - TRAIL_FRAMES + 1), frame + 1):
        for row in track.get(trail_frame, []):
            x = float(row.get("x_native") or 2.0 * float(row["x"]))
            y = float(row.get("y_native") or 2.0 * float(row["y"]))
            kind = provenance_kind(str(row.get("sources", "")))
            color = PROVENANCE_COLORS[kind]
            radius = 3 + max(0, trail_frame - frame + TRAIL_FRAMES) // 3
            cv2.circle(canvas, (round(x), round(y)), radius, color, -1, cv2.LINE_AA)
            if trail_frame == frame:
                cv2.circle(canvas, (round(x), round(y)), 11, color, 2, cv2.LINE_AA)

    for row in players.get(frame, []):
        side = str(row.get("side", "unknown"))
        x0 = round(float(row.get("x0_native") or 2.0 * float(row["x0"])))
        y0 = round(float(row.get("y0_native") or 2.0 * float(row["y0"])))
        x1 = round(float(row.get("x1_native") or 2.0 * float(row["x1"])))
        y1 = round(float(row.get("y1_native") or 2.0 * float(row["y1"])))
        color = (255, 170, 50) if side == "far" else (80, 220, 100)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
        _draw_text(
            canvas,
            f"{side} | {side_names.get(side, 'identity abstain')}",
            (x0, max(24, y0 - 9)),
            color,
        )

    # Every solved fit is visible; acceptance status remains explicit in color and text.
    if P is not None:
        for sample in trajectories.get(frame, []):
            xyz = sample["xyz"]
            accepted = sample.get("flight_status") == ACCEPTED_FLIGHT_STATUS
            fit_color = (255, 255, 255) if accepted else (30, 150, 255)
            ball = project(P, xyz, camera_scale)
            shadow = project(P, (xyz[0], xyz[1], 0.0), camera_scale)
            if ball is None or shadow is None:
                continue
            cv2.line(canvas, ball, shadow, fit_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, shadow, 8, (80, 80, 80), 2, cv2.LINE_AA)
            cv2.circle(canvas, ball, 8, fit_color, -1, cv2.LINE_AA)
            state = "ACCEPT" if accepted else "REJECT"
            _draw_text(
                canvas,
                f"3D {state} | z={float(xyz[2]):.2f} m",
                (ball[0] + 12, ball[1] - 8),
                fit_color,
                0.56,
            )

    lane_y = 132
    visible_events = [
        (event_frame, event)
        for event_frame in range(frame - EVENT_HOLD_FRAMES, frame + EVENT_HOLD_FRAMES + 1)
        for event in events.get(event_frame, [])
    ]
    for event_frame, event in visible_events:
        event_type = str(event.get("event_type", "event"))
        held = (
            bool(event.get("abstain"))
            or bool(event.get("gate_held"))
            or not bool(event.get("production_scope", True))
        )
        grammar = event.get("point_grammar") or {}
        held = held or grammar.get("verdict") not in {None, "in_play"}
        state = "ABSTAIN" if held else "EMIT"
        color = EVENT_COLORS.get(event_type, (230, 230, 230))
        _draw_text(
            canvas,
            f"{event_type} @ f{event_frame:04d} | {state}",
            (NATIVE_WIDTH - 430, lane_y),
            color,
            0.66,
        )
        lane_y += 30
        support = event.get("impulse_support") or event.get("research_impulse_support") or {}
        if support.get("supported") is True:
            _draw_text(
                canvas,
                "two-wing event support; track held",
                (NATIVE_WIDTH - 430, lane_y),
                color,
                0.56,
            )
            lane_y += 30
        location = event.get("location") or {}
        if location.get("image_x") is not None and location.get("image_y") is not None:
            centre = (round(float(location["image_x"])), round(float(location["image_y"])))
            cv2.drawMarker(canvas, centre, color, cv2.MARKER_DIAMOND, 28, 3, cv2.LINE_AA)
        else:
            event_track = track.get(event_frame, [])
            if event_track:
                row = event_track[0]
                centre = (
                    round(float(row.get("x_native") or 2.0 * float(row["x"]))),
                    round(float(row.get("y_native") or 2.0 * float(row["y"]))),
                )
                cv2.drawMarker(canvas, centre, color, cv2.MARKER_DIAMOND, 28, 3, cv2.LINE_AA)

    legend_y = NATIVE_HEIGHT - 28
    x = 26
    for kind, color in PROVENANCE_COLORS.items():
        cv2.circle(canvas, (x, legend_y - 7), 7, color, -1, cv2.LINE_AA)
        _draw_text(canvas, kind, (x + 13, legend_y), color, 0.52)
        x += 175
    return canvas


def _load_camera(match_dir: Path, clip: str) -> dict[int, tuple[np.ndarray, bool]]:
    path = match_dir / "camera_P_per_frame_v1.npz"
    if not path.is_file():
        return {}
    result: dict[int, tuple[np.ndarray, bool]] = {}
    with np.load(path, allow_pickle=True) as data:
        reliable = data["reliable"] if "reliable" in data else np.ones(len(data["frames"]), bool)
        for row_clip, frame, P, valid in zip(data["clips"], data["frames"], data["P"], reliable):
            if str(row_clip) == clip:
                result[int(frame)] = (np.asarray(P, dtype=float), bool(valid))
    return result


def _point_detail(reconstruction: dict[str, Any], match_id: str, clip: str) -> dict[str, Any]:
    for detail in reconstruction.get("points_detail", []):
        if detail.get("match_id") == match_id and detail.get("clip") == clip:
            return detail
    return {}


def _fitted_trajectories(
    detail: dict[str, Any], flight_ledger: dict[str, Any], match_id: str, clip: str
) -> dict[int, list[dict[str, Any]]]:
    statuses = {
        int(row["flight_index"]): str(row.get("status", "unknown"))
        for row in flight_ledger.get("rows", [])
        if row.get("match_id") == match_id
        and point_clip(row.get("clip") or row.get("point")) == clip
    }
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fit in detail.get("fits", []):
        flight_index = int(fit.get("flight_index", -1))
        status = statuses.get(flight_index, "not_in_flight_ledger")
        for sample in fit.get("trajectory", []):
            result[frame_number(sample["frame"])].append(
                {**sample, "flight_index": flight_index, "flight_status": status}
            )
    return dict(result)


def _events_for_point(
    emissions: list[dict[str, Any]],
    detail: dict[str, Any],
    match_id: str,
    clip: str,
    terminal_frame: int,
) -> dict[int, list[dict[str, Any]]]:
    rows = [row for row in emissions if _clip_matches(row.get("clip"), match_id, clip)]
    known = {(row.get("event_type"), frame_number(row["frame"])) for row in rows}
    selected = detail.get("joint_rally_refinement", {}).get("selected_events", [])
    for row in selected:
        key = (row.get("event_type"), frame_number(row["frame"]))
        if key in known:
            continue
        rows.append(
            {**row, "gate_held": detail.get("decision") != "retain", "production_scope": True}
        )
        known.add(key)
    # Current event artifacts do not carry a point_end row. Show that absence explicitly on the
    # final audit frame rather than pretending the model emitted an accepted event.
    if not any(row.get("event_type") == "point_end" for row in rows):
        rows.append(
            {
                "clip": f"{match_id}__{clip}",
                "match_id": match_id,
                "frame": terminal_frame,
                "event_type": "point_end",
                "gate_held": True,
                "production_scope": False,
            }
        )
    return _artifact_rows_by_frame(rows, match_id, clip)


def render_point(
    *,
    root: Path,
    match_id: str,
    clip: str,
    output_dir: Path,
    fps: float,
    camera_scale: float = 1.0,
) -> tuple[Path, Path]:
    clip = point_clip(clip)
    match_dir = root / match_id
    frames_dir = match_dir / "audit_frames_native_1080" / clip
    frame_paths = sorted(frames_dir.glob("f_*.jpg"), key=lambda path: frame_number(path.name))
    if not frame_paths:
        raise FileNotFoundError(f"no native audit frames under {frames_dir}")

    track_rows = _read_csv(match_dir / "ball_track_joint_native1080_arc_augmented_v2.csv")
    player_paths = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))
    player_rows = _read_csv(player_paths[0]) if player_paths else []
    gate = _load_json(root / "untouched_tracking_point_gate_v1.json", {})
    emissions = _load_json(root / "event_emissions.json", [])
    reconstruction = _load_json(root / "reconstruction_3d.json", {})
    flight_ledger = _load_json(root / "flight_ledger.json", {})
    ledger_rows = _read_csv(match_dir / "automatic_point_ledger_v1.csv")

    track = _artifact_rows_by_frame(track_rows, match_id, clip)
    players = _artifact_rows_by_frame(player_rows, match_id, clip)
    detail = _point_detail(reconstruction, match_id, clip)
    events = _events_for_point(
        emissions,
        detail,
        match_id,
        clip,
        frame_number(frame_paths[-1].name),
    )
    gate_row = next(
        (
            row
            for row in gate.get("rows", [])
            if row.get("match_id") == match_id and row.get("clip") == clip
        ),
        {},
    )
    arcs = gate_row.get("arcs", [])
    camera = _load_camera(match_dir, clip)
    trajectories = _fitted_trajectories(detail, flight_ledger, match_id, clip)
    names = _side_names(
        _scoreboard_names(match_dir / "vlm_reads.jsonl", match_id),
        _point_row(ledger_rows, clip),
        detail,
        [r for v in events.values() for r in v],
        players,
        track,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{clip}.mp4"
    sheet_path = output_dir / f"{clip}_contact_sheet.png"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (NATIVE_WIDTH, NATIVE_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")

    sheet_indices = set(np.linspace(0, len(frame_paths) - 1, 6, dtype=int).tolist())
    sheet_frames: list[np.ndarray] = []
    try:
        for index, path in enumerate(frame_paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read {path}")
            if image.shape[:2] != (NATIVE_HEIGHT, NATIVE_WIDTH):
                raise ValueError(f"{path} is {image.shape[1]}x{image.shape[0]}, expected 1920x1080")
            overlaid = draw_overlay_frame(
                image,
                frame=frame_number(path.name),
                clip=clip,
                track=track,
                players=players,
                side_names=names,
                events=events,
                arcs=arcs,
                camera=camera,
                trajectories=trajectories,
                camera_scale=camera_scale,
            )
            writer.write(overlaid)
            if index in sheet_indices:
                sheet_frames.append(cv2.resize(overlaid, (640, 360), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()

    while len(sheet_frames) < 6:
        sheet_frames.append(sheet_frames[-1].copy())
    sheet = np.vstack((np.hstack(sheet_frames[:3]), np.hstack(sheet_frames[3:6])))
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError(f"could not write {sheet_path}")
    return video_path, sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="runner output root")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--point", action="append", default=[], help="ptNNNN; repeatable")
    parser.add_argument("--select-six", action="store_true")
    parser.add_argument("--select-twelve", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, help="override source processing FPS")
    parser.add_argument("--camera-scale", type=float, default=1.0)
    args = parser.parse_args()

    match_dir = args.root / args.match_id
    source = _load_json(match_dir / "source_integrity.json", {})
    fps = args.fps or float(source.get("cadence", {}).get("processing_fps", 0.0))
    if fps <= 0:
        raise ValueError("FPS is absent from source_integrity.json; pass --fps")
    clips = [
        path.name for path in (match_dir / "audit_frames_native_1080").glob("pt*") if path.is_dir()
    ]
    flight_ledger = _load_json(args.root / "flight_ledger.json", {})
    requested = [point_clip(value) for value in args.point]
    if args.select_six:
        requested.extend(select_six_points(clips, flight_ledger, args.seed))
    if args.select_twelve:
        active_play = _load_json(args.root / "active_play_v1.json", {})
        requested.extend(
            select_twelve_points(clips, flight_ledger, active_play, args.match_id, args.seed)
        )
    requested = list(dict.fromkeys(requested))
    if not requested:
        parser.error("pass --point, --select-six, or --select-twelve")

    output_dir = args.output_dir or args.root / "overlays"
    selection = {
        "schema": "point_overlay_selection_v1",
        "match_id": args.match_id,
        "seed": args.seed if args.select_six or args.select_twelve else None,
        "selection_rule": (
            "six most accepted flights plus six seeded random automatic in-play points"
            if args.select_twelve
            else "three most accepted flights plus three seeded random points"
            if args.select_six
            else "explicit points"
        ),
        "points": requested,
        "accepted_flights": accepted_flight_counts(flight_ledger),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    table = write_s1_point_table(
        match_dir,
        args.match_id,
        args.root / "s1_point_score_serve_names.csv",
    )
    print(f"S1 table: {table}", flush=True)
    for clip in requested:
        video, sheet = render_point(
            root=args.root,
            match_id=args.match_id,
            clip=clip,
            output_dir=output_dir,
            fps=fps,
            camera_scale=args.camera_scale,
        )
        print(f"{clip}: {video} | {sheet}", flush=True)


if __name__ == "__main__":
    main()
