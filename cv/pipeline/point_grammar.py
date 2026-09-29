"""Attach lossless point-grammar metadata to automatic event emissions.

The classifier is label-free.  It fuses local event-chain, track-continuity,
play-phase, player-motion, and serve evidence.  The artifact always retains
every input emission; consumers select the default in-play view explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cv.pipeline.resolution import read_coordinate_manifest

REFERENCE_FPS = 25.0
CHAIN_GAP_REFERENCE_FRAMES = 80.0
PLAYER_STATIONARY_RADIUS = 20
PLAYER_STATIONARY_MAX_PX_PER_FRAME = 1.0
PLAYER_BOX_RADIUS = 5
PLAYER_BOX_MAX_PX_PER_FRAME = 0.5
SERVE_GUARD_FRAMES = 4.0
BOX_FRAME_TOLERANCE = 2
COURT_MIDLINE_Y_M = 11.885
METADATA_SCHEMA = "point_grammar_emission_metadata_v1"
LOCATION_SCHEMA = "s5_emission_location_v1"
POINT_END_SCHEMA = "point_grammar_point_end_v1"
# A proposal further away than the decoder's own suppression radius is not a
# witness for this emission, so no location is attached.
LOCATION_MAX_OFFSET_FRAMES = 4.0
BALL_TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ChainMembership:
    chain_id: str | None
    chain_length: int
    member: bool
    distance_reference_frames: float | None


def _frame_number(value: str) -> int:
    match = re.search(r"\d+", Path(str(value)).stem)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def _group(rows: Iterable[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["clip"])].append(row)
    return output


def _float(row: dict | None, field: str, default: float = math.nan) -> float:
    if row is None:
        return default
    try:
        value = float(row.get(field, default))
    except (TypeError, ValueError):
        return default
    return value


def _nearest(rows: list[dict], frame: float) -> dict | None:
    return min(rows, key=lambda row: abs(float(row["proposal_frame"]) - frame), default=None)


class AutomaticEvidence:
    """Automatic track, player, phase, and toss evidence for grammar metadata."""

    def __init__(self, root: Path, proposals: list[dict], active_play: dict[str, dict]) -> None:
        self.root = root
        self.proposals = _group(proposals)
        self.active_play = {
            f"{match_id}__{local_clip}": decision
            for key, decision in active_play.items()
            for match_id, local_clip in [key.split("/", 1)]
        }
        self._tracks: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
        self._boxes: dict[str, dict[str, dict[str, dict[int, tuple[float, ...]]]]] = {}
        self._scales: dict[str, tuple[float, float, dict | None]] = {}

    @staticmethod
    def split_clip(clip: str) -> tuple[str, str]:
        return tuple(clip.rsplit("__", 1))  # type: ignore[return-value]

    def native_scale(self, match_id: str) -> tuple[float, float, dict | None]:
        """Return the artifact-to-native scale for the match ball track."""

        if match_id not in self._scales:
            manifest = read_coordinate_manifest(self.root / match_id / BALL_TRACK_NAME)
            if manifest is None:
                self._scales[match_id] = (1.0, 1.0, None)
            else:
                image = manifest["image_size"]
                artifact = manifest["artifact_size"]
                self._scales[match_id] = (
                    float(image["width"]) / float(artifact["width"]),
                    float(image["height"]) / float(artifact["height"]),
                    {
                        "width": int(image["width"]),
                        "height": int(image["height"]),
                        "artifact_width": int(artifact["width"]),
                        "artifact_height": int(artifact["height"]),
                    },
                )
        return self._scales[match_id]

    def location(self, clip: str, frame: float, proposal: dict | None, fps: float) -> dict:
        """Native image location and sub-frame time from the matched proposal."""

        match_id, _ = self.split_clip(clip)
        scale_x, scale_y, space = self.native_scale(match_id)
        offset = float(proposal["proposal_frame"]) - frame if proposal is not None else math.nan
        if (
            proposal is None
            or not math.isfinite(offset)
            or abs(offset) > LOCATION_MAX_OFFSET_FRAMES
        ):
            return {
                "schema": LOCATION_SCHEMA,
                "source": (f"no_proposal_within_{LOCATION_MAX_OFFSET_FRAMES:g}_frames"),
                "image_x": None,
                "image_y": None,
                "image_coordinate_space": space,
                "court_x_m": None,
                "court_y_m": None,
                "frame_subpixel": None,
                "time_seconds": None,
                "fps": fps,
                "proposal_offset_frames": None if proposal is None else offset,
            }
        subpixel = float(proposal["proposal_frame"])
        image_x = _float(proposal, "img_x")
        image_y = _float(proposal, "img_y")
        court_x = _float(proposal, "court_x")
        court_y = _float(proposal, "court_y")
        return {
            "schema": LOCATION_SCHEMA,
            "source": "nearest_event_proposal",
            "image_x": float(image_x * scale_x) if math.isfinite(image_x) else None,
            "image_y": float(image_y * scale_y) if math.isfinite(image_y) else None,
            "image_coordinate_space": space,
            "court_x_m": float(court_x) if math.isfinite(court_x) else None,
            "court_y_m": float(court_y) if math.isfinite(court_y) else None,
            "frame_subpixel": subpixel,
            "time_seconds": subpixel / fps if fps else None,
            "fps": fps,
            "proposal_offset_frames": offset,
        }

    def fps(self, clip: str) -> float:
        decision = self.active_play.get(clip, {})
        if decision.get("fps") is not None:
            return float(decision["fps"])
        proposal = self.proposals.get(clip, [{}])[0]
        return float(proposal.get("source_fps") or REFERENCE_FPS)

    def tracks(self, match_id: str) -> dict[str, dict[int, tuple[float, float]]]:
        if match_id not in self._tracks:
            output: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
            path = self.root / match_id / "ball_track_joint_native1080_arc_augmented_v2.csv"
            if path.exists():
                import csv

                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        output[row["clip"]][_frame_number(row["frame"])] = (
                            float(row["x"]),
                            float(row["y"]),
                        )
            self._tracks[match_id] = output
        return self._tracks[match_id]

    def track(self, clip: str) -> dict[int, tuple[float, float]]:
        match_id, local_clip = self.split_clip(clip)
        return self.tracks(match_id).get(local_clip, {})

    def boxes(self, match_id: str) -> dict[str, dict[str, dict[int, tuple[float, ...]]]]:
        if match_id not in self._boxes:
            import csv

            output: dict[str, dict[str, dict[int, tuple[float, ...]]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            paths = sorted((self.root / match_id).glob("player_boxes_*_native_sided_v1.csv"))
            if paths:
                with paths[0].open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("side") not in {"near", "far"}:
                            continue
                        output[row["clip"]][row["side"]][_frame_number(row["frame"])] = tuple(
                            float(row[key]) for key in ("x0", "y0", "x1", "y1")
                        )
            self._boxes[match_id] = output
        return self._boxes[match_id]

    def player_motion(self, clip: str, frame: float, radius: int) -> tuple[float, float]:
        match_id, local_clip = self.split_clip(clip)
        frame_i = int(round(frame))
        speeds = []
        for side in ("near", "far"):
            side_boxes = self.boxes(match_id).get(local_clip, {}).get(side, {})
            steps = []
            for index in range(frame_i - radius, frame_i + radius):
                if index not in side_boxes or index + 1 not in side_boxes:
                    continue
                left = side_boxes[index]
                right = side_boxes[index + 1]
                left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
                right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
                steps.append(math.dist(left_center, right_center))
            speeds.append(float(np.quantile(steps, 0.75)) if len(steps) >= radius else math.inf)
        return speeds[0], speeds[1]

    def both_players_stationary(
        self, clip: str, frame: float, *, radius: int, maximum: float
    ) -> bool:
        return max(self.player_motion(clip, frame, radius)) <= maximum

    def ball_in_player_box(self, clip: str, frame: float) -> bool:
        match_id, local_clip = self.split_clip(clip)
        track = self.track(clip)
        if not track:
            return False
        frame_i = int(round(frame))
        ball_frame = min(track, key=lambda value: abs(value - frame_i))
        if abs(ball_frame - frame_i) > BOX_FRAME_TOLERANCE:
            return False
        ball_x, ball_y = track[ball_frame]
        for side in ("near", "far"):
            side_boxes = self.boxes(match_id).get(local_clip, {}).get(side, {})
            if not side_boxes:
                continue
            box_frame = min(side_boxes, key=lambda value: abs(value - frame_i))
            if abs(box_frame - frame_i) > BOX_FRAME_TOLERANCE:
                continue
            x0, y0, x1, y1 = side_boxes[box_frame]
            if x0 <= ball_x <= x1 and y0 <= ball_y <= y1:
                return True
        return False

    def serve_frames(self, clip: str) -> list[float]:
        return sorted(
            {
                float(row["proposal_frame"])
                for row in self.proposals.get(clip, [])
                if str(row.get("source_serve", "0")) == "1"
            }
        )

    def track_continuity(self, clip: str, frame: float, fps: float) -> dict:
        track = self.track(clip)
        center = round(frame)
        radius = max(1, round(0.40 * fps))
        pre = sum(index in track for index in range(center - radius, center))
        post = sum(index in track for index in range(center + 1, center + radius + 1))
        return {
            "pre_observations_400ms": pre,
            "post_observations_400ms": post,
            "continuous_through_emission": pre >= radius // 2 and post >= radius // 2,
        }

    def structure_span_id(self, clip: str, frame: float) -> str | None:
        decision = self.active_play.get(clip, {})
        spans = decision.get("event_spans", decision.get("active_spans", []))
        for index, (start, end) in enumerate(spans):
            if float(start) <= frame <= float(end):
                return f"{clip}::span_{index:02d}"
        return None


def toss_ascent_signature(proposal: dict | None) -> bool:
    observations = _float(proposal, "pre_400ms_observations", 0.0)
    displacement = _float(proposal, "pre_400ms_displacement_px", 0.0)
    straightness = _float(proposal, "pre_400ms_straightness", 0.0)
    post_displacement = _float(proposal, "post_400ms_displacement_px", 0.0)
    return bool(
        (observations >= 10.0 and displacement >= 30.0 and straightness >= 0.65)
        or post_displacement >= 180.0
    )


def _allowed_transition(left: str, right: str) -> bool:
    return (left, right) in {
        ("contact", "bounce"),
        ("contact", "net_hit"),
        ("bounce", "contact"),
        ("net_hit", "bounce"),
        ("net_hit", "contact"),
    }


def chain_memberships(rows: list[dict], fps: float) -> list[ChainMembership]:
    chains: list[list[int]] = []
    current: list[int] = []
    for index, row in enumerate(rows):
        if not current:
            current = [index]
            continue
        previous = rows[current[-1]]
        gap = (float(row["frame"]) - float(previous["frame"])) * REFERENCE_FPS / fps
        if gap <= CHAIN_GAP_REFERENCE_FRAMES and _allowed_transition(
            str(previous["event_type"]), str(row["event_type"])
        ):
            current.append(index)
        else:
            chains.append(current)
            current = [index]
    if current:
        chains.append(current)
    accepted = [
        chain
        for chain in chains
        if len(chain) >= 2
        and any(rows[index]["event_type"] == "contact" for index in chain)
        and any(rows[index]["event_type"] in {"bounce", "net_hit"} for index in chain)
    ]
    lookup = {
        index: (chain_index, chain)
        for chain_index, chain in enumerate(accepted, start=1)
        for index in chain
    }
    accepted_frames = [float(rows[index]["frame"]) for chain in accepted for index in chain]
    output = []
    for index, row in enumerate(rows):
        if index in lookup:
            chain_index, chain = lookup[index]
            output.append(ChainMembership(f"chain_{chain_index:02d}", len(chain), True, 0.0))
        else:
            distance = (
                min(abs(float(row["frame"]) - frame) for frame in accepted_frames)
                * REFERENCE_FPS
                / fps
                if accepted_frames
                else None
            )
            output.append(ChainMembership(None, 0, False, distance))
    return output


def _double_bounce_dead_time(
    row: dict,
    previous: dict | None,
    proposal: dict | None,
    previous_proposal: dict | None,
    fps: float,
    stationary_dead: bool,
) -> bool:
    if previous is None:
        return False
    gap = (float(row["frame"]) - float(previous["frame"])) * REFERENCE_FPS / fps
    court_y = _float(proposal, "court_y")
    previous_y = _float(previous_proposal, "court_y")
    same_side = (
        math.isfinite(court_y)
        and math.isfinite(previous_y)
        and (court_y < COURT_MIDLINE_Y_M) == (previous_y < COURT_MIDLINE_Y_M)
    )
    return bool(
        row["event_type"] == "bounce"
        and previous["event_type"] == "bounce"
        and 25.0 <= gap < 40.0
        and proposal is not None
        and proposal.get("excluded") == "pre_serve"
        and proposal.get("proposal_source") == "trajectory"
        and same_side
        and not stationary_dead
    )


def _failed_toss_contact(
    row: dict,
    previous: dict | None,
    following: dict | None,
    proposal: dict | None,
    fps: float,
) -> bool:
    previous_gap = (
        (float(row["frame"]) - float(previous["frame"])) * REFERENCE_FPS / fps
        if previous is not None
        else math.inf
    )
    following_gap = (
        (float(following["frame"]) - float(row["frame"])) * REFERENCE_FPS / fps
        if following is not None
        else math.inf
    )
    return bool(
        row["event_type"] == "contact"
        and previous_gap > CHAIN_GAP_REFERENCE_FRAMES
        and following_gap > CHAIN_GAP_REFERENCE_FRAMES
        and proposal is not None
        and proposal.get("excluded") == "between_serves"
        and not toss_ascent_signature(proposal)
    )


def _confidence(in_play: bool, reasons: list[str], chain: ChainMembership, track: dict) -> float:
    if not in_play:
        if "s3_phase_dead_plus_stationary_players" in reasons:
            return 0.90
        if "ball_in_stationary_player_box" in reasons:
            return 0.88
        if "pre_serve_unbridged_second_bounce" in reasons:
            return 0.85
        return 0.80
    if chain.member and track["continuous_through_emission"]:
        return 0.90
    if chain.member:
        return 0.80
    if track["continuous_through_emission"]:
        return 0.65
    return 0.50


def pre_serve_transfer_guard(
    proposal: dict | None,
    membership: ChainMembership,
    track: dict,
    toss: bool,
    reasons: list[str],
) -> bool:
    """Protect live chains from an over-broad pre-serve phase decision."""
    return bool(
        proposal is not None
        and proposal.get("excluded") == "pre_serve"
        and membership.member
        and track["continuous_through_emission"]
        and toss
        and "s3_phase_dead_plus_stationary_players" in reasons
    )


def pre_serve_double_bounce_guard(track: dict, toss: bool, reasons: list[str]) -> bool:
    """Protect a live toss/bounce chain from the narrow double-bounce rule."""
    return bool(
        reasons == ["pre_serve_unbridged_second_bounce"]
        and track["continuous_through_emission"]
        and toss
    )


def pre_serve_missing_serve_guard(
    proposal: dict | None,
    membership: ChainMembership,
    track: dict,
    serves: list[float],
    reasons: list[str],
) -> bool:
    """Fail open for a coherent live chain when the serve witness is unavailable."""
    return bool(
        reasons == ["s3_phase_dead_plus_stationary_players"]
        and proposal is not None
        and proposal.get("excluded") == "pre_serve"
        and membership.member
        and track["continuous_through_emission"]
        and not serves
    )


def annotate_emissions(
    emissions: list[dict], root: Path, proposals: list[dict], active_play: dict[str, dict]
) -> list[dict]:
    """Return every emission with S1-19 grammar metadata attached."""
    evidence = AutomaticEvidence(root, proposals, active_play)
    output = []
    for clip, clip_rows in sorted(_group(emissions).items()):
        clip_rows.sort(key=lambda row: (float(row["frame"]), str(row["event_type"])))
        fps = evidence.fps(clip)
        memberships = chain_memberships(clip_rows, fps)
        serves = evidence.serve_frames(clip)
        for index, (row, membership) in enumerate(zip(clip_rows, memberships, strict=True)):
            frame = float(row["frame"])
            proposal = _nearest(evidence.proposals.get(clip, []), frame)
            phase_dead = proposal is not None and str(proposal.get("in_play", "1")) == "0"
            stationary = evidence.both_players_stationary(
                clip,
                frame,
                radius=PLAYER_STATIONARY_RADIUS,
                maximum=PLAYER_STATIONARY_MAX_PX_PER_FRAME,
            )
            serve_guard = any(abs(frame - serve) <= SERVE_GUARD_FRAMES for serve in serves)
            in_stationary_box = evidence.ball_in_player_box(
                clip, frame
            ) and evidence.both_players_stationary(
                clip,
                frame,
                radius=PLAYER_BOX_RADIUS,
                maximum=PLAYER_BOX_MAX_PX_PER_FRAME,
            )
            previous = clip_rows[index - 1] if index else None
            following = clip_rows[index + 1] if index + 1 < len(clip_rows) else None
            previous_proposal = (
                _nearest(evidence.proposals.get(clip, []), float(previous["frame"]))
                if previous is not None
                else None
            )
            track = evidence.track_continuity(clip, frame, fps)
            toss = toss_ascent_signature(proposal)
            reasons = []
            if phase_dead and stationary and not serve_guard:
                reasons.append("s3_phase_dead_plus_stationary_players")
            if in_stationary_box:
                reasons.append("ball_in_stationary_player_box")
            if _double_bounce_dead_time(
                row, previous, proposal, previous_proposal, fps, stationary
            ):
                reasons.append("pre_serve_unbridged_second_bounce")
            if _failed_toss_contact(row, previous, following, proposal, fps):
                reasons.append("isolated_between_serves_contact_without_toss_ascent")

            guards = []
            if pre_serve_transfer_guard(proposal, membership, track, toss, reasons):
                reasons.remove("s3_phase_dead_plus_stationary_players")
                guards.append("pre_serve_chain_track_toss_protection")
            if pre_serve_double_bounce_guard(track, toss, reasons):
                reasons.remove("pre_serve_unbridged_second_bounce")
                guards.append("pre_serve_double_bounce_track_toss_protection")
            if pre_serve_missing_serve_guard(proposal, membership, track, serves, reasons):
                reasons.remove("s3_phase_dead_plus_stationary_players")
                guards.append("pre_serve_chain_track_missing_serve_evidence_protection")
            in_play = not reasons
            output.append(
                {
                    **row,
                    "location": evidence.location(clip, frame, proposal, fps),
                    "point_grammar": {
                        "schema": METADATA_SCHEMA,
                        "in_play": in_play,
                        "verdict": "in_play" if in_play else "dead_time",
                        "confidence": _confidence(in_play, reasons, membership, track),
                        "reasons": reasons,
                        "guards": guards,
                        "structure_span_id": evidence.structure_span_id(clip, frame),
                        "chain_evidence": {
                            "member": membership.member,
                            "chain_id": membership.chain_id,
                            "chain_length": membership.chain_length,
                            "distance_reference_frames": membership.distance_reference_frames,
                        },
                        "track_evidence": track,
                        "phase_evidence": {
                            "nearest_proposal_frame": (
                                float(proposal["proposal_frame"]) if proposal is not None else None
                            ),
                            "in_play": not phase_dead if proposal is not None else None,
                            "excluded_reason": (
                                proposal.get("excluded") if proposal is not None else None
                            ),
                            "stationary_dead_signal": stationary,
                            "ball_in_stationary_player_box": in_stationary_box,
                        },
                        "serve_evidence": {
                            "candidate_frames": serves,
                            "guarded": serve_guard,
                            "toss_ascent_signature": toss,
                        },
                        "truth_inputs_loaded": False,
                    },
                }
            )
    return sorted(output, key=lambda row: (row["clip"], float(row["frame"]), row["event_type"]))


def point_end_emissions(annotated: list[dict]) -> list[dict]:
    """Emit one ``point_end`` row per terminated in-play grammar chain.

    The grammar decides which emissions form a live rally chain, but its last
    member is not automatically a tennis ending.  Reuse the automatic
    ending-kind rule so a point end needs a first out bounce, a second in-court
    bounce, a terminal net hit, or an observed track exit.
    """

    output = []
    for clip, clip_rows in sorted(_group(annotated).items()):
        live = [
            row
            for row in clip_rows
            if row.get("abstain") is not True
            and row["point_grammar"]["in_play"] is True
            and row["point_grammar"]["chain_evidence"]["member"]
        ]
        chains: dict[str, list[dict]] = defaultdict(list)
        for row in live:
            chains[str(row["point_grammar"]["chain_evidence"]["chain_id"])].append(row)
        for chain_id, members in sorted(chains.items()):
            from cv.pipeline.event_grammar_decoder import _decoded_ending

            ending = _decoded_ending(members)
            if ending is None:
                continue
            terminal, ending_evidence = ending
            location = dict(terminal.get("location") or {})
            output.append(
                {
                    "clip": clip,
                    "match_id": terminal["match_id"],
                    "event_type": "point_end",
                    "frame": float(terminal["frame"]),
                    "confidence": float(terminal["point_grammar"]["confidence"]),
                    "probability": float(terminal["point_grammar"]["confidence"]),
                    "abstain": False,
                    "court_geometry_missing": bool(terminal.get("court_geometry_missing", False)),
                    "point_gate_verdict": terminal.get("point_gate_verdict"),
                    "point_gate_failure_reasons": list(
                        terminal.get("point_gate_failure_reasons", [])
                    ),
                    "gate_held": terminal.get("gate_held"),
                    "production_scope": terminal.get("production_scope"),
                    "location": location,
                    "point_end": {
                        "schema": POINT_END_SCHEMA,
                        "source": "point_grammar_terminal_chain_member",
                        "chain_id": chain_id,
                        "chain_length": len(members),
                        "terminal_event_type": terminal["event_type"],
                        "terminal_event_frame": float(terminal["frame"]),
                        "termination_kind": ending_evidence["termination_kind"],
                        "structure_span_id": terminal["point_grammar"]["structure_span_id"],
                    },
                    "point_grammar": {
                        **terminal["point_grammar"],
                        "verdict": "point_end",
                    },
                }
            )
    return sorted(output, key=lambda row: (row["clip"], float(row["frame"])))


def annotate_abstentions(
    rows: list[dict], root: Path, proposals: list[dict], active_play: dict[str, dict]
) -> list[dict]:
    """Attach location and a minimal metadata block to decoder abstentions.

    Abstentions are deliberately kept out of the chain grammar: running them
    through it would change the chain structure, and therefore the verdicts, of
    the accepted emissions they sit beside.
    """

    evidence = AutomaticEvidence(root, proposals, active_play)
    output = []
    for clip, clip_rows in sorted(_group(rows).items()):
        fps = evidence.fps(clip)
        clip_proposals = evidence.proposals.get(clip, [])
        for row in clip_rows:
            frame = float(row["frame"])
            output.append(
                {
                    **row,
                    "abstain": True,
                    "location": evidence.location(
                        clip, frame, _nearest(clip_proposals, frame), fps
                    ),
                    "point_grammar": {
                        "schema": METADATA_SCHEMA,
                        "in_play": False,
                        "verdict": "abstained",
                        "confidence": 0.0,
                        "reasons": ["below_operating_threshold_above_abstention_floor"],
                        "guards": [],
                        "grammar_evaluated": False,
                        "truth_inputs_loaded": False,
                    },
                }
            )
    return sorted(output, key=lambda row: (row["clip"], float(row["frame"]), row["event_type"]))


def select_consumer_emissions(
    rows: list[dict],
    *,
    include_dead_time: bool = False,
    include_abstained: bool = False,
) -> list[dict]:
    """Select the default in-play view, with an explicit lossless bypass.

    Abstentions are excluded from every consumer view unless asked for by name:
    they are recorded evidence, not claims.
    """
    if not include_abstained:
        rows = [row for row in rows if row.get("abstain") is not True]
    if include_dead_time:
        return list(rows)
    missing = [index for index, row in enumerate(rows) if "point_grammar" not in row]
    if missing:
        raise ValueError(f"point-grammar metadata missing from emissions: {missing[:10]}")
    return [row for row in rows if row["point_grammar"].get("in_play") is True]


def annotate_from_root(
    root: Path, emissions: list[dict], *, emit_point_end: bool = True
) -> tuple[list[dict], dict]:
    """Generate explicit automatic witnesses and annotate a lossless emission list."""
    from cv.pipeline.event_inference import apply_production_scope, generate_proposals

    manifest_path = root / "manifest.json"
    active_play_path = root / "active_play_v1.json"
    point_gate_path = root / "point_validity_gate_v1.json"
    active_play = json.loads(active_play_path.read_text())
    proposals = generate_proposals(json.loads(manifest_path.read_text()), root)
    scope = apply_production_scope(proposals, active_play_path, point_gate_path)
    accepted = [row for row in emissions if row.get("abstain") is not True]
    abstained = [row for row in emissions if row.get("abstain") is True]
    annotated = annotate_emissions(accepted, root, proposals, active_play)
    endings = point_end_emissions(annotated) if emit_point_end else []
    if abstained:
        abstained = annotate_abstentions(abstained, root, proposals, active_play)
    annotated = sorted(
        annotated + endings + abstained,
        key=lambda row: (row["clip"], float(row["frame"]), row["event_type"]),
    )
    return annotated, {
        "schema": METADATA_SCHEMA,
        "input_emissions": len(emissions),
        "annotated_emissions": len(annotated),
        "accepted_emissions": len(accepted),
        "abstained_emissions": len(abstained),
        "point_end_emissions": len(endings),
        "default_in_play_emissions": len(select_consumer_emissions(annotated)),
        "dead_time_emissions": sum(
            row["point_grammar"]["in_play"] is False and row.get("abstain") is not True
            for row in annotated
        ),
        "proposal_rows": len(proposals),
        "proposal_rows_sha256": _rows_sha256(proposals),
        "proposal_scope": scope,
        "automatic_inputs": {
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _file_sha256(manifest_path),
            },
            "active_play": {
                "path": str(active_play_path.resolve()),
                "sha256": _file_sha256(active_play_path),
            },
            "point_gate": {
                "path": str(point_gate_path.resolve()),
                "sha256": _file_sha256(point_gate_path),
            },
        },
        "lossless_bypass": "select_consumer_emissions(rows, include_dead_time=True)",
    }
