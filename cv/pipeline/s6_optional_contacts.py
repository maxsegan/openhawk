"""Source-witnessed contact alternatives and a bounded pre-gate S6 selector.

Optional engineering ablation. No ball coordinates, accepted event, ending or
reference denominator changes. Audio support is correlated with S5's audio CNN:
modalities qualify alternatives, never multiply independent probabilities.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from cv.pipeline import event_contact_witness as witness, provenance

SCHEMA = "s6_optional_contact_witness_v1"
# Additive final-gap scope. v1 documents stay byte-identical and readable: the
# scope is declared only when the shared policy asks for it.
SCHEMA_FINAL = "s6_optional_contact_witness_v2"
SCHEMAS = (SCHEMA, SCHEMA_FINAL)
FINAL_SCOPE = "final_contacts_v1"
INTERIOR = "interior"
FINAL = "final"
# Additive optional interior scope; see ``s6_witnessed_interior_contacts``. Its
# gaps and candidates are declared separately and never enter the legacy kinds.
INTERIOR_OPTIONAL = "optional_interior"
# How the attempt itself declares the end of the final gap. Only SUPPLIED_ENDING is an
# actual source-supplied physical ending; the two horizons are explicitly nonphysical
# and carry `ending_supplied=False`.
SUPPLIED_ENDING = "supplied_attempt_ending"
OBSERVED_HORIZON_ENDING = "original_observation_horizon"
SCOPE_HORIZON_ENDING = "observation_horizon_not_physical_event"
NO_ENDING = "no_declared_ending"
ENDING_SEMANTICS = (SUPPLIED_ENDING, OBSERVED_HORIZON_ENDING, SCOPE_HORIZON_ENDING, NO_ENDING)
MAX_ADDED_HYPOTHESES = 2
MAX_OCCURRENCE_LOG_ODDS = 1.0
PIXEL_SCALE = 3.0


def inconsistent_gaps(events: list[dict]) -> list[dict]:
    contacts = sorted((e for e in events if e["event_type"] == "contact"), key=lambda e: e["frame"])
    result = []
    for left, right in zip(contacts, contacts[1:]):
        bounces = [
            e
            for e in events
            if e["event_type"] == "bounce" and left["frame"] < e["frame"] < right["frame"]
        ]
        if len(bounces) > 1:
            result.append(
                {"interval": [left["frame"], right["frame"]], "bounces": deepcopy(bounces)}
            )
    return result


def declared_final_scope(document: dict) -> bool:
    """Whether a prepared witness declares the final-gap scope. v1 never does."""
    scope = document.get("scope")
    if scope is None:
        return False
    if scope != FINAL_SCOPE or document.get("schema") != SCHEMA_FINAL:
        raise ValueError("unsupported optional contact witness scope declaration")
    return True


def final_gap(events: list[dict], end_boundary: float | None) -> dict | None:
    """The one gap after the last accepted contact, up to the declared ending.

    ``end_boundary`` is the original observation horizon for an unresolved tail
    and the supplied ending epoch otherwise. Nothing here decides that a point
    ended; the boundary is whatever the attempt already declared.
    """
    if end_boundary is None:
        return None
    contacts = [e for e in events if e["event_type"] == "contact"]
    if not contacts:
        return None
    last = max(contacts, key=lambda e: float(e["frame"]))
    high = float(end_boundary)
    if not np.isfinite(high) or not float(last["frame"]) < high:
        return None
    return {
        "interval": [float(last["frame"]), high],
        "kind": FINAL,
        "bounces": deepcopy(
            [
                e
                for e in events
                if e["event_type"] == "bounce" and float(last["frame"]) < float(e["frame"]) <= high
            ]
        ),
        "last_contact_interval": [float(x) for x in _interval(last)],
    }


def _interval(event: dict) -> list[float]:
    return [float(x) for x in event.get("frame_interval", [event["frame"], event["frame"]])]


def scoped_gaps(document: dict, kind: str) -> list[int]:
    """Indexes of the declared gaps of one kind; v1 gaps are all interior."""
    return [
        index for index, gap in enumerate(document["gaps"]) if gap.get("kind", INTERIOR) == kind
    ]


def excluded_phase(row: dict) -> bool:
    declarations = [row.get("phase"), row.get("scope"), row.get("production_scope")]
    grammar = row.get("point_grammar") or {}
    declarations.extend([grammar.get("phase"), grammar.get("scope")])
    return grammar.get("in_play") is False or any(
        x
        in {
            "aftermath",
            "post_delivery_aftermath",
            "post_winner_aftermath",
            "collection",
            "afterplay",
            "dead_ball",
        }
        for x in declarations
        if isinstance(x, str)
    )


def _held_source_contact(row: dict, clip: str) -> tuple[float, float] | None:
    """The unchanged source-emitted held-contact occurrence rule, shared by scopes."""
    if row.get("clip") != clip or row.get("event_type") != "contact" or excluded_phase(row):
        return None
    if not (row.get("abstain") or row.get("gate_held")):
        return None
    probability = row.get("class_probabilities") or {}
    contact = float(probability.get("contact", 0))
    none = float(probability.get("none", 1))
    if not (0 < contact <= 1 and 0 <= none <= 1 and contact >= none):
        return None
    return contact, none


def _native_clock(pts: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    frames = np.array(sorted(pts), float)
    times = np.array([pts[int(f)] for f in frames])
    if (
        len(frames) < 2
        or np.any(np.diff(frames) != 1)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("complete ordered original native PTS required")
    return frames, times


def source_candidates(
    events: list[dict], emissions: list[dict], clip: str, pts: dict[int, float]
) -> list[dict]:
    from cv.experiments.connected_shooting.auto_packet import automatic_physical_event

    result = []
    frames, times = _native_clock(pts)
    for gap_index, gap in enumerate(inconsistent_gaps(events)):
        low, high = gap["interval"]
        for index, row in enumerate(emissions):
            held = _held_source_contact(row, clip)
            if held is None:
                continue
            contact, none = held
            event = automatic_physical_event(
                row, prediction_window=(int(frames[0]), int(frames[-1]))
            )
            a, b = event["frame_interval"]
            if not low < a <= event["frame"] <= b < high:
                continue
            # An added contact must genuinely separate the existing two ground
            # encounters, rather than treating a collection hit as a repair.
            if not any(e["frame"] < a for e in gap["bounces"]) or not any(
                e["frame"] > b for e in gap["bounces"]
            ):
                continue
            event.update(
                note="optional classifier contact; independently witnessed occurrence",
                occurrence_status="optional",
                status="predicted",
            )
            result.append(
                {
                    "id": f"emission_{index}",
                    "gap_index": gap_index,
                    "gap_interval": gap["interval"],
                    "event": event,
                    "source_emission_index": index,
                    "source_emission": row,
                    "source_pts_interval": np.interp([a, b], frames, times).tolist(),
                    "occurrence_log_odds": float(
                        np.clip(math.log(contact / max(none, 1e-9)), -1, 1)
                    ),
                }
            )
    return result


def _bounce_emission_rows(emissions: list[dict], clip: str) -> list[dict]:
    """Held ground encounters under the bounce arm's own unchanged source rule."""
    from cv.pipeline import s6_optional_bounces

    return [row for row in emissions if s6_optional_bounces.held_bounce_row(row, clip)]


def final_gap_candidates(
    events: list[dict],
    emissions: list[dict],
    clip: str,
    pts: dict[int, float],
    gap: dict,
    gap_index: int,
    *,
    cuts: set[int],
    tail_contract: dict | None,
    ending_semantics: str,
) -> tuple[list[dict], list[dict]]:
    """Source-held contacts inside the one final gap, with explicit exclusions.

    Occurrence remains the source emission's own held-contact rule and the
    generated timing interval is untouched. Everything added here is grammar:
    where the row lies relative to the accepted stream, whether the native view
    is continuous across it, and whether the original observed suffix still
    supports a later contact. A volley is expressible with no intervening
    ground, so an absent ground encounter is recorded, never a veto.

    ``ending_semantics`` is the attempt's own declaration from
    ``declared_end_boundary``. Only an actual supplied ending makes an accepted ground in
    this gap terminal; under either nonphysical horizon the source witnessed no ending, so
    a ground followed by a later contact is an ordinary return, not a dead ball.
    """
    from cv.experiments.connected_shooting.auto_packet import automatic_physical_event

    if ending_semantics not in ENDING_SEMANTICS:
        raise ValueError("explicit declared final-gap ending semantics required")
    supplied_ending = ending_semantics == SUPPLIED_ENDING
    frames, times = _native_clock(pts)
    low, high = float(gap["interval"][0]), float(gap["interval"][1])
    last_end = float(gap["last_contact_interval"][1])
    accepted = sorted(float(e["frame"]) for e in gap["bounces"])
    held_grounds = sorted(
        float(
            automatic_physical_event(row, prediction_window=(int(frames[0]), int(frames[-1])))[
                "frame"
            ]
        )
        for row in _bounce_emission_rows(emissions, clip)
    )
    candidates: list[dict] = []
    exclusions: list[dict] = []

    def refuse(index, row, event, reason, **extra):
        exclusions.append(
            {
                "id": f"emission_{index}",
                "gap_index": gap_index,
                "gap_kind": FINAL,
                "source_emission_index": index,
                "frame_interval": None if event is None else _interval(event),
                "reason": reason,
                **extra,
            }
        )

    for index, row in enumerate(emissions):
        held = _held_source_contact(row, clip)
        if held is None:
            continue
        contact, none = held
        event = automatic_physical_event(row, prediction_window=(int(frames[0]), int(frames[-1])))
        a, b = (float(x) for x in event["frame_interval"])
        if not low < a <= float(event["frame"]) <= b < high:
            continue
        if not last_end < a:
            refuse(index, row, event, "candidate_overlaps_the_last_accepted_contact")
            continue
        # An accepted terminal ground declared by an actual source-supplied ending must
        # stay ordered after the added contact. Under a nonphysical horizon there is no
        # declared ending for a ground to terminate, so the same ground only counts
        # toward the dead-ball rule below.
        #
        # Both horizons reach this line. ``observed_horizon_tail.qualify`` refuses an
        # attempt whose accepted stream carries any event after the last contact, so for
        # that ending ``accepted`` is empty as prepared and this branch cannot fire from
        # ordinary preparation; it is reachable there only through a rebased alternative,
        # which ``observed_horizon_tail.for_events`` does support (it recounts the
        # supplied grounds of the rebased suffix). The observation-scope horizon has no
        # such restriction: it declares one or two supplied grounds after the last
        # contact, and those are exactly the candidates this branch used to refuse.
        terminal = [f for f in accepted if f > b]
        if supplied_ending and accepted and not terminal:
            refuse(index, row, event, "accepted_terminal_ground_precedes_candidate")
            continue
        if sum(1 for f in accepted if f < a) >= 2:
            refuse(index, row, event, "dead_ball_before_candidate")
            continue
        if any(int(frame) in cuts for frame in range(int(math.floor(a)), int(math.ceil(b)) + 1)):
            refuse(index, row, event, "native_view_cut_inside_candidate_interval")
            continue
        preceding = (
            "accepted"
            if any(f < a for f in accepted)
            else "held"
            if any(low < f < a for f in held_grounds)
            else "none"
        )
        if tail_contract is not None:
            try:
                from cv.experiments.connected_shooting import observed_horizon_tail

                observed_horizon_tail.for_events(
                    tail_contract,
                    [*events, {**event, "event_type": "contact"}],
                )
            except (ValueError, KeyError, IndexError) as error:
                refuse(
                    index,
                    row,
                    event,
                    "no_original_native_suffix_support",
                    detail=f"{type(error).__name__}: {error}",
                )
                continue
        event.update(
            note="optional classifier contact; independently witnessed occurrence",
            occurrence_status="optional",
            status="predicted",
        )
        candidates.append(
            {
                "id": f"emission_{index}",
                "gap_index": gap_index,
                "gap_interval": gap["interval"],
                "gap_kind": FINAL,
                "preceding_ground": preceding,
                "event": event,
                "source_emission_index": index,
                "source_emission": row,
                "source_pts_interval": np.interp([a, b], frames, times).tolist(),
                "occurrence_log_odds": float(np.clip(math.log(contact / max(none, 1e-9)), -1, 1)),
            }
        )
    return candidates, exclusions


def pose_rows(path: Path, clip: str, pts: dict[int, float]) -> tuple[dict, dict]:
    """Consume declared native skeletons if present; boxes alone are not poses."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = {f"{joint}_{axis}_native" for joint in witness.JOINTS for axis in ("x", "y")}
        if not required <= columns:
            return {}, {"available": False, "reason": "bound_player_input_has_no_native_skeleton"}
        grouped = {}
        for row in reader:
            if row["clip"] != clip:
                continue
            f = int(row["frame"])
            if f not in pts:
                continue
            identity = (row.get("side", "unknown"), row.get("track_id", ""))
            if not identity[1]:
                continue
            try:
                item = {
                    "frame": f,
                    "source_pts_seconds": pts[f],
                    "track_id": identity[1],
                    "box_xyxy_native": [
                        float(row[x + "_native"]) for x in ("x0", "y0", "x1", "y1")
                    ],
                    "joints": {
                        joint: [
                            float(row[joint + "_x_native"]),
                            float(row[joint + "_y_native"]),
                            float(row[joint + "_confidence"]),
                        ]
                        for joint in witness.JOINTS
                    },
                }
            except (ValueError, KeyError):
                continue
            grouped.setdefault(identity, []).append(item)
    return grouped, {"available": bool(grouped), "tracks": len(grouped)}


def independent_pose_frame_receipt(inputs) -> Path | None:
    """Declare the original extraction input when pose pictures live elsewhere."""
    from cv.pipeline import resolution

    path = getattr(inputs, "optional_contact_pose", None)
    if path is None:
        return None
    manifest = resolution.read_coordinate_manifest(path)
    if not manifest or not manifest.get("source"):
        raise ValueError("independent contact pose requires a declared native picture source")
    source = Path(manifest["source"])
    if not source.is_absolute():
        source = path.parent / source
    if source.resolve() == inputs.frames_directory.parent.resolve():
        return None
    receipt = source / inputs.clip / "extraction_receipt.json"
    if not receipt.is_file():
        raise ValueError(
            "independent contact pose addresses a different native frame clock: missing extraction receipt"
        )
    return receipt


def _independent_pose_clock(inputs, pts: dict[int, float], images: list[dict] | None):
    """Authenticate copied pictures without interpreting local row numbers as new epochs.

    The adapter binds both receipts in ordinary automatic ancestry before this
    helper runs. Its current inventory is already verified; verify the pose's
    original extraction against the same processing video and source PTS too.
    Only a common native origin with identical overlapping pictures is eligible.
    """
    from types import SimpleNamespace

    from cv.pipeline.s6_automatic_observations import _native_inventory

    receipt_path = independent_pose_frame_receipt(inputs)
    if receipt_path is None:
        return pts, None
    if not images:
        raise ValueError("independent contact pose copy requires verified native inventory")
    receipt = json.loads(receipt_path.read_text())
    point = receipt.get("identity", {}).get("point", {})
    if (
        receipt.get("schema") != "player_frame_extraction_v1"
        or not {"pt", "t0", "t1"} <= point.keys()
    ):
        raise ValueError("independent contact pose requires an original extraction window")
    if point.get("pt") != int(inputs.clip[2:]):
        raise ValueError("independent contact pose extraction addresses a different clip")
    original_inputs = SimpleNamespace(
        extraction_receipt=receipt_path,
        frames_directory=receipt_path.parent,
        source_video=inputs.source_video,
        source_pts=inputs.source_pts,
        clip=inputs.clip,
    )
    original = _native_inventory(
        original_inputs,
        float(
            json.loads(inputs.extraction_receipt.read_text())["source_identity"]["configuration"][
                "fps"
            ]
        ),
        {"pt": point["pt"], "rally_t_start": point["t0"], "rally_t_end": point["t1"]},
    )
    if not original or original[0]["source_frame_index"] != images[0]["source_frame_index"]:
        raise ValueError("independent contact pose has shifted or reindexed native origin")
    current = {int(row["frame"]): row for row in images}
    if len(current) != len(images) or set(current) != set(pts):
        raise ValueError("independent contact pose requires the complete current native inventory")
    original_pts = {int(row["frame"]): float(row["native_pts_seconds"]) for row in original}
    overlap = set(original_pts) & set(current)
    if not overlap:
        raise ValueError("independent contact pose has no shared native picture coverage")
    for row in original:
        frame = int(row["frame"])
        if frame not in overlap:
            continue
        other = current[frame]
        if (
            row["source_frame_index"] != other["source_frame_index"]
            or row["native_pts_seconds"] != other["native_pts_seconds"]
            or row["native_pts_seconds"] != pts[frame]
            or row["source"]["sha256"] != other["source"]["sha256"]
        ):
            raise ValueError(
                "independent contact pose native picture or epoch differs from current input"
            )
    return original_pts, {
        "schema": "independent_pose_native_identity_v1",
        "original_extraction_receipt": provenance.file_record(receipt_path),
        "current_extraction_receipt": provenance.file_record(inputs.extraction_receipt),
        "source_pts": provenance.file_record(inputs.source_pts),
        "native_origin": original[0]["source_frame_index"],
        "original_picture_count": len(original),
        "current_picture_count": len(current),
        "shared_picture_count": len(overlap),
        "current_pictures_without_original_pose_coverage": len(set(current) - overlap),
        "original_pictures_outside_current_scope": len(set(original_pts) - overlap),
        "comparison": "same processing video, native frame index, exact PTS and picture SHA256",
    }


def independent_pose_rows(
    inputs, pts: dict[int, float], *, images: list[dict] | None = None
) -> tuple[dict, dict]:
    """Read separately bound native crop poses without replacing actor inputs.

    Ordinary crop output retains its unsuffixed COCO columns and source tracked
    box. Its frame names address the same native picture directory whose PTS
    inventory was already checked by automatic preparation.
    """
    from cv.pipeline import resolution
    from cv.pipeline.pose_player_crop import frame_number, native_box_for_row

    path = inputs.optional_contact_pose
    manifest = resolution.read_coordinate_manifest(path)
    if (
        manifest is None
        or manifest.get("artifact_identity") != resolution.PLAYER_POSE_NATIVE_IDENTITY
        or resolution.manifest_artifact_size(manifest) != resolution.NATIVE_SIZE
        or manifest.get("image_size")
        != {"width": resolution.NATIVE_SIZE.width, "height": resolution.NATIVE_SIZE.height}
    ):
        raise ValueError("independent contact pose requires declared native COCO coordinates")
    pose_pts, native_identity = _independent_pose_clock(inputs, pts, images)
    box_manifest = resolution.read_coordinate_manifest(inputs.players)
    if box_manifest is None:
        raise ValueError("original actor box coordinate contract required")
    box_size = resolution.manifest_artifact_size(box_manifest)
    originals = {}
    with inputs.players.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["clip"] != inputs.clip:
                continue
            key = (frame_number(row["frame"]), row.get("side"), row.get("track_id"))
            box = native_box_for_row(row, box_size, resolution.NATIVE_SIZE)
            originals.setdefault(key, []).append(box)
    grouped = {}
    seen = set()
    counts = {"rows": 0, "missing_joints": 0, "missing_identity": 0, "ambiguous_identity_rows": 0}
    if native_identity is not None:
        counts["outside_current_scope_rows"] = 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            f"{joint}_{axis}" for joint in witness.JOINTS for axis in ("x", "y", "confidence")
        }
        if not required <= set(reader.fieldnames or []):
            return {}, {"available": False, "reason": "independent_input_has_no_native_skeleton"}
        for row in reader:
            if row["clip"] != inputs.clip:
                continue
            counts["rows"] += 1
            frame = frame_number(row["frame"])
            if frame not in pose_pts:
                raise ValueError("independent contact pose frame is outside original native PTS")
            for field in ("native_pts_seconds", "source_pts_seconds"):
                if row.get(field) not in (None, "") and (
                    not math.isfinite(float(row[field]))
                    or abs(float(row[field]) - pose_pts[frame]) > 1e-6
                ):
                    raise ValueError("independent contact pose PTS differs from original frame")
            if frame not in pts:
                # Count authentic source-clock rows, not in-scope actor/joint support.
                counts["outside_current_scope_rows"] += 1
                continue
            side, track = row.get("side"), row.get("track_id")
            if not track or not side:
                counts["missing_identity"] += 1
                continue
            key = (frame, side, track)
            if key not in originals or side not in {"near", "far"}:
                raise ValueError(
                    "independent contact pose differs from original side/track identity"
                )
            try:
                tracked_box = [float(row[f"track_{axis}"]) for axis in ("x0", "y0", "x1", "y1")]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "independent pose must retain its original native tracked box"
                ) from error
            if not np.isfinite(tracked_box).all() or not any(
                np.allclose(tracked_box, box, atol=0.011, rtol=0) for box in originals[key]
            ):
                raise ValueError("independent contact pose tracked box differs from actor input")
            # The original tracker occasionally emits distinct boxes with the
            # same side/track/frame identity. Keep both source rows explicit,
            # but neither defines a unique observed motion sample. Omitting
            # this pose epoch also blocks derivatives across its neighbors.
            if len(originals[key]) != 1:
                counts["ambiguous_identity_rows"] += 1
                continue
            if key in seen:
                raise ValueError("duplicate independent pose for one native tracked player")
            seen.add(key)
            try:
                joints = {
                    joint: [float(row[f"{joint}_{axis}"]) for axis in ("x", "y", "confidence")]
                    for joint in witness.JOINTS
                }
            except (KeyError, ValueError):
                counts["missing_joints"] += 1
                continue
            # Keep low-confidence rows: the motion qualifier refuses their
            # local neighborhoods. Missing/non-finite measurements stay gaps.
            if not np.isfinite(list(joints.values())).all():
                counts["missing_joints"] += 1
                continue
            grouped.setdefault((side, track), []).append(
                {
                    "frame": frame,
                    "source_pts_seconds": pts[frame],
                    "track_id": track,
                    "box_xyxy_native": originals[key][0],
                    "joints": joints,
                }
            )
    return grouped, {
        "available": bool(grouped),
        "tracks": len(grouped),
        **counts,
        "source": "separate_automatic_native_tracked_crop_pose",
        "native_clock": "original verified picture directory and PTS inventory",
        "actor_players_replaced": False,
        **({"native_picture_identity": native_identity} if native_identity is not None else {}),
    }


def pose_view_barriers(path: Path | None, pts: dict[int, float]) -> tuple[set[int], set[int], dict]:
    """Map automatic source-time shots to native epochs; never infer an absent cut."""
    if path is None:
        return set(), set(pts), {"available": False, "reason": "missing_automatic_shot_intervals"}
    with path.open(newline="") as stream:
        shots = list(csv.DictReader(stream))
    intervals = []
    for row in shots:
        start, end = float(row["t_start"]), float(row["t_end"])
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError("automatic pose shot intervals require finite ordered source times")
        intervals.append((start, end, row["shot_index"]))
    assigned, unsupported = {}, set()
    for frame, epoch in pts.items():
        owners = [i for i, (start, end, _) in enumerate(intervals) if start <= epoch < end]
        if len(owners) == 1:
            assigned[frame] = owners[0]
        else:
            unsupported.add(frame)
    cuts = {
        frame
        for frame in assigned
        if frame - 1 in assigned and assigned[frame] != assigned[frame - 1]
    }
    return (
        cuts,
        unsupported,
        {
            "available": True,
            "source": provenance.file_record(path),
            "cut_frames": sorted(cuts),
            "uncovered_or_ambiguous_frames": sorted(unsupported),
            "interval_clock": "original_source_pts_seconds_half_open",
        },
    )


def declared_end_boundary(attempt: dict) -> tuple[float | None, dict | None, str]:
    """The attempt's own declared ending for the final gap; nothing is inferred.

    An unresolved tail ends at the original observed horizon it already declared. An
    observation-scoped attempt ends at its own nonphysical horizon, which the legacy
    ``owner_end_frame`` alias carries: it declares ``ending_supplied=False`` and
    ``owner_end_frame_semantics="observation_horizon_not_physical_event"``, so calling it
    a supplied ending would assert a physical ending the source never witnessed. Only an
    attempt that actually supplies an ending event reports ``supplied_attempt_ending``.
    """
    tail = attempt.get("observed_horizon_tail")
    if isinstance(tail, dict) and tail.get("observation_horizon") is not None:
        return float(tail["observation_horizon"]), tail, OBSERVED_HORIZON_ENDING
    end = attempt.get("owner_end_frame")
    if end is None:
        return None, None, NO_ENDING
    if (
        attempt.get("ending_supplied") is False
        or attempt.get("owner_end_frame_semantics") == SCOPE_HORIZON_ENDING
    ):
        return float(end), None, SCOPE_HORIZON_ENDING
    return float(end), None, SUPPLIED_ENDING


def build_witness(
    inputs,
    attempt: dict,
    images: list[dict],
    cameras: dict,
    *,
    optional_final_contacts: str = "off",
    optional_contact_timing: str = "off",
    optional_contact_composition: str = "off",
    optional_interior_contacts: str = "off",
    independent_event_proposals: str = "off",
) -> dict:
    if optional_final_contacts not in {"off", "on"}:
        raise ValueError("explicit optional final contact policy required")
    if optional_contact_timing not in {"off", "pmf_peaks"}:
        raise ValueError("explicit optional contact timing policy required")
    if optional_contact_composition not in {"off", "bounded_pairs"}:
        raise ValueError("explicit optional contact composition policy required")
    if optional_interior_contacts not in {"off", "on"}:
        raise ValueError("explicit optional interior contact policy required")
    if independent_event_proposals not in {"off", "on"}:
        raise ValueError("explicit independent event proposal policy required")
    scoped = optional_final_contacts == "on"
    pts = {int(row["frame"]): float(row["native_pts_seconds"]) for row in images}
    clip = f"{inputs.match_id}__{inputs.clip}"
    emissions = json.loads(inputs.events.read_text())
    candidates = source_candidates(attempt["events"], emissions, clip, pts)
    gaps = inconsistent_gaps(attempt["events"])
    final_exclusions: list[dict] = []
    ending = None
    if scoped:
        for candidate in candidates:
            candidate["gap_kind"] = INTERIOR
        boundary, tail_contract, ending_semantics = declared_end_boundary(attempt)
        gap = final_gap(attempt["events"], boundary)
        ending = {
            "end_boundary": boundary,
            "ending_semantics": ending_semantics,
            "final_gap_built": gap is not None,
        }
        if gap is not None:
            for row in gaps:
                row["kind"] = INTERIOR
            gap_index = len(gaps)
            gaps.append(gap)
            cuts, _unsupported, _view = pose_view_barriers(
                getattr(inputs, "optional_contact_views", None), pts
            )
            found, final_exclusions = final_gap_candidates(
                attempt["events"],
                emissions,
                clip,
                pts,
                gap,
                gap_index,
                cuts=cuts,
                tail_contract=tail_contract,
                ending_semantics=ending_semantics,
            )
            candidates.extend(found)
    interior_scope = None
    if optional_interior_contacts == "on":
        from cv.pipeline import s6_witnessed_interior_contacts as interior_module

        # Separate declaration and separate candidate collection: the legacy
        # gaps above are never reopened, so they can gain no new interval.
        interior_cuts, view_unsupported, interior_view = pose_view_barriers(
            getattr(inputs, "optional_contact_views", None), pts
        )
        # An absent view inventory makes the declared view unavailable, exactly as
        # the existing scopes treat it; a declared unsupported epoch still counts.
        # This new scope needs actual view and camera continuity, not absence
        # of a recorded refusal. Missing view metadata already marks all PTS
        # unavailable in pose_view_barriers. Missing camera epochs stay absent.
        supported_frames = {
            int(row["frame"])
            for row in cameras["cameras"]
            if row.get("status") == "supported" and int(row["frame"]) in pts
        } - set(view_unsupported)
        boxes, box_receipt = interior_module.actor_boxes(inputs.players, inputs.clip)
        points, observation_receipt = interior_module.observed_points(
            attempt.get("owner_ball_labels"), supported_frames
        )
        interior_gaps = interior_module.scope_gaps(attempt["events"])
        found, interior_exclusions = interior_module.candidates(
            attempt["events"],
            emissions,
            clip,
            pts,
            interior_gaps,
            len(gaps),
            points=points,
            boxes=boxes,
            cuts=interior_cuts,
            supported_frames=supported_frames,
        )
        gaps.extend(interior_gaps)
        candidates.extend(found)
        interior_scope = {
            "policy": "on",
            "schema": interior_module.SCHEMA,
            "scope": interior_module.SCOPE,
            "maximum_total_alternatives": interior_module.MAX_TOTAL_ALTERNATIVES,
            "gap_count": len(interior_gaps),
            "candidate_count": len(found),
            "exclusions": interior_exclusions,
            "observations": observation_receipt,
            "actor_boxes": box_receipt,
            "view_continuity": interior_view,
            "runtime_model_calls": 0,
        }
    report = {
        "schema": SCHEMA_FINAL if scoped else SCHEMA,
        "clip": clip,
        "policy": "source_witness_optional_contact_v1",
        "source_events": provenance.file_record(inputs.events),
        "source_video": provenance.file_record(inputs.source_video),
        "source_pose": provenance.file_record(inputs.players),
        "source_pts": provenance.file_record(inputs.source_pts),
        "source_attempt_events": deepcopy(attempt["events"]),
        "gaps": gaps,
        "candidates": [],
        "runtime_model_calls": 0,
        "native_timestamps_changed": False,
        "audio_pose_independent_probabilities": False,
    }
    if optional_contact_composition != "off":
        report["contact_composition"] = {"policy": optional_contact_composition}
    if interior_scope is not None:
        report["optional_interior_contacts"] = interior_scope
    if optional_contact_timing != "off":
        report["contact_timing"] = {
            "policy": optional_contact_timing,
            "schema": "s6_contact_timing_pmf_v1",
            "maximum_additional_hypotheses": 2,
        }
    if scoped:
        report["scope"] = FINAL_SCOPE
        report["final_gap"] = ending
        report["final_gap_exclusions"] = final_exclusions
    if independent_event_proposals == "on":
        from cv.pipeline import s6_independent_event_proposals as independent

        report[independent.KEY] = independent.prepare(inputs, attempt, images, cameras)
    if not candidates:
        report["status"] = (
            "independent_source_candidates"
            if report.get("independent_event_proposals", {}).get("candidates")
            else "no_source_candidate"
        )
        if interior_scope is not None:
            interior_scope["expansion"] = interior_module.expansion_receipt(report, [])
        return report
    peaks = []
    # Fixed gap windows are source-defined, not event/reference-selected clips.
    audio_receipts = []
    legacy_audio_receipts = []
    interior_audio = {}
    candidate_gaps = {candidate["gap_index"] for candidate in candidates}
    for gap_index, gap in enumerate(report["gaps"]):
        if gap.get("kind") == INTERIOR_OPTIONAL and gap_index not in candidate_gaps:
            continue
        low, high = np.interp(gap["interval"], sorted(pts), [pts[f] for f in sorted(pts)])
        try:
            audio = witness.decode_clocked_audio(
                inputs.source_video, max(0, low - 0.25), high + 0.25
            )
            decoded_peaks = witness.audio_peaks(audio)
            audio_receipt = audio.receipt
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            decoded_peaks = []
            audio_receipt = {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
        if gap.get("kind") == INTERIOR_OPTIONAL:
            audio_receipt = {**audio_receipt, "gap_index": gap_index, "gap_kind": INTERIOR_OPTIONAL}
        audio_receipts.append(audio_receipt)
        if gap.get("kind") == INTERIOR_OPTIONAL:
            # New padded windows may overlap legacy windows. Their peaks must
            # not duplicate an old transient and change its unique-audio rule.
            # Each new candidate uses its own original source-gap audio only.
            interior_audio[gap_index] = (decoded_peaks, [audio_receipt])
        else:
            peaks.extend(decoded_peaks)
            legacy_audio_receipts.append(audio_receipt)
    if getattr(inputs, "optional_contact_pose", None) is not None:
        groups, pose_receipt = independent_pose_rows(inputs, pts, images=images)
        report["source_pose"] = provenance.file_record(inputs.optional_contact_pose)
        report["source_pose_coordinates"] = provenance.file_record(
            inputs.optional_contact_pose.with_suffix(
                inputs.optional_contact_pose.suffix + ".coordinates.json"
            )
        )
        report["actor_players_unchanged"] = provenance.file_record(inputs.players)
    else:
        groups, pose_receipt = pose_rows(inputs.players, inputs.clip, pts)
    # Supported camera projections do not establish continuity through a video cut.
    cuts, unsupported, view_receipt = pose_view_barriers(
        getattr(inputs, "optional_contact_views", None), pts
    )
    # Missing pose-view metadata makes poses unavailable, not every PMF frame.
    # Apply actual declared view gaps/cuts and camera gaps to timing proposals.
    timing_unsupported = set(unsupported) if view_receipt.get("available") else set()
    camera_unsupported = {
        int(row["frame"]) for row in cameras["cameras"] if row.get("status") != "supported"
    }
    unsupported.update(camera_unsupported)
    timing_unsupported.update(camera_unsupported)
    motion = {
        key: witness.pose_motion([row for row in rows if row["frame"] not in unsupported], cuts)
        for key, rows in groups.items()
    }
    pose_receipt["view_continuity"] = view_receipt
    for candidate in candidates:
        candidate_peaks, candidate_audio_receipts = (
            interior_audio.get(candidate["gap_index"], ([], []))
            if candidate.get("gap_kind") == INTERIOR_OPTIONAL
            else (peaks, legacy_audio_receipts)
        )
        audio = (
            witness.audio_support(candidate_peaks, tuple(candidate["source_pts_interval"]))
            if candidate_peaks
            else {
                "available": bool(
                    candidate_audio_receipts
                    and any("command" in r for r in candidate_audio_receipts)
                ),
                "supported": False,
                "reason": "no_audio_peak",
            }
        )
        a, b = candidate["event"]["frame_interval"]
        support = []
        for (side, track), series in motion.items():
            for row in series:
                if a <= row["frame"] <= b and row["supported"] and row.get("score_z", 0) >= 0.5:
                    support.append({**row, "side": side, "track_id": track})
        pose = {"available": pose_receipt["available"], "supported": bool(support), "rows": support}
        if scoped and not support:
            # Absent local pose is an unavailable modality, never physical
            # evidence against the candidate. Audio alone may still support it.
            pose["reason"] = (
                "no_pose_rows_in_candidate_interval"
                if pose_receipt["available"]
                else pose_receipt.get("reason", "no_pose_input_available")
            )
        candidate.update(audio=audio, pose=pose, supported=audio["supported"] or pose["supported"])
        candidate["witness_support"] = max(
            float(audio.get("support", 0)),
            min(1.0, max([r["score_z"] for r in support], default=0) / 2),
        )
        if optional_contact_timing != "off" and candidate.get("gap_kind") != INTERIOR_OPTIONAL:
            from cv.pipeline import s6_contact_timing

            candidate["contact_timing"] = s6_contact_timing.prepare(
                candidate, attempt, gaps[candidate["gap_index"]], pts, cuts, timing_unsupported
            )
        report["candidates"].append(candidate)
    report.update(status="completed", audio_clocks=audio_receipts, pose_coverage=pose_receipt)
    if interior_scope is not None:
        legacy = _legacy_hypotheses(
            report, attempt["events"], observations=attempt.get("owner_ball_labels")
        )
        interior_scope["expansion"] = interior_module.expansion_receipt(report, legacy)
    return report


def _ranked(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            -r["witness_support"],
            -r["occurrence_log_odds"],
            r["event"]["frame"],
            r["id"],
        ),
    )[:MAX_ADDED_HYPOTHESES]


def hypotheses(
    document: dict, events: list[dict], *, observations: list[dict] | None = None
) -> list[dict]:
    """The existing alternatives, then the bounded optional-interior additions.

    The legacy helper below is unchanged and still produces the first branches
    in their exact original order and content. The optional interior scope, when
    the witness declares it, may only append capped singletons after them.
    """
    from cv.pipeline import s6_witnessed_interior_contacts

    legacy = _legacy_hypotheses(document, events, observations=observations)
    existing = s6_witnessed_interior_contacts.extend_hypotheses(document, events, legacy)
    from cv.pipeline import s6_independent_event_proposals as independent

    return independent.extend_hypotheses(document, events, existing)


def _legacy_hypotheses(
    document: dict, events: list[dict], *, observations: list[dict] | None = None
) -> list[dict]:
    if document.get("schema") not in SCHEMAS or document.get("source_attempt_events") != events:
        raise ValueError("optional contact evidence does not bind unchanged accepted events")
    from cv.pipeline import s6_contact_composition

    composition_policy = s6_contact_composition.policy(document)
    if composition_policy != "off" and observations is None:
        raise ValueError("contact composition requires original native ball observations")
    scoped = declared_final_scope(document)
    groups = {}
    for row in document["candidates"]:
        if row.get("supported"):
            groups.setdefault(row["gap_index"], []).append(row)
    if not groups:
        return []
    interior = scoped_gaps(document, INTERIOR)
    final = scoped_gaps(document, FINAL)
    if not scoped and final:
        raise ValueError("undeclared final optional contact scope in a v1 witness")
    choices = [_ranked(groups[gap]) for gap in interior if gap in groups]
    # Deterministic small beam, at most one addition per source gap. Every
    # *interior* gap must get support; otherwise no purported full topology
    # repair is emitted. A final-gap addition is optional by construction and
    # never demands that the interior repair exist, or the reverse.
    if len(choices) != len(interior):
        return []
    beam = [()]
    for choice in choices:
        beam = sorted(
            ((*a, b) for a in beam for b in choice),
            key=lambda rows: (
                -sum(r["witness_support"] for r in rows),
                -sum(r["occurrence_log_odds"] for r in rows),
                tuple(r["id"] for r in rows),
            ),
        )[:MAX_ADDED_HYPOTHESES]
    legacy_beam = beam
    beam, pair_receipt = s6_contact_composition.extend_interior_beam(
        document, events, beam, groups, interior, observations
    )
    paired_base = beam[1] if pair_receipt is not None and beam != legacy_beam else None
    final_rows = _ranked(groups[final[0]]) if final and final[0] in groups else []
    composition = None
    if scoped:
        if interior and final_rows and paired_base is not None:
            # Preserve both existing final-scope alternatives, then their paired
            # counterparts. Four branches only for this explicit interaction;
            # they still share the same invocation-wide wall-clock budget.
            base = legacy_beam[0]
            beam = [base, (*base, final_rows[0]), paired_base, (*paired_base, final_rows[0])]
            composition = [
                "interior_only",
                "interior_and_final",
                "paired_interior_only",
                "paired_interior_and_final",
            ]
        elif interior and final_rows:
            # One interior repair, then the same repair plus the best final
            # addition: the final gap never costs the interior beam its width.
            beam, composition = (
                [beam[0], (*beam[0], final_rows[0])],
                [
                    "interior_only",
                    "interior_and_final",
                ],
            )
        elif not interior:
            beam = [(row,) for row in final_rows]
            composition = ["final_only"] * len(beam)
        else:
            composition = ["interior_only"] * len(beam)
    if not beam or beam == [()]:
        return []

    def conditioned_event(candidate):
        event = deepcopy(candidate["event"])
        event["occurrence_status"] = "predicted"
        event["optional_topology_membership"] = {
            "candidate_id": candidate["id"],
            "conditioned_on_this_hypothesis": True,
            "accepted_source_stream_changed": False,
        }
        if scoped:
            event["optional_gap_kind"] = candidate.get("gap_kind", INTERIOR)
        return event

    result = [
        {
            "name": f"optional_contacts_{i + 1}",
            "events": sorted(
                [*deepcopy(events), *[conditioned_event(r) for r in rows]], key=lambda r: r["frame"]
            ),
            "added": [conditioned_event(r) for r in rows],
            "candidate_ids": [r["id"] for r in rows],
            "occurrence_log_odds": sum(r["occurrence_log_odds"] for r in rows),
            **({"beam_composition": composition[i]} if composition else {}),
        }
        for i, rows in enumerate(beam)
    ]
    from cv.pipeline import s6_contact_timing

    extended = s6_contact_timing.extend_hypotheses(result, document)
    if composition_policy == "off":
        return extended
    admitted = []
    refused_timing = []
    for hypothesis in extended:
        support = s6_contact_composition.admissibility(hypothesis, document, events, observations)
        if not support["eligible"]:
            # A source PMF peak can cross another optional contact even though
            # it stays inside the original gap. It cannot silently break a pair.
            if "contact_timing_alternative" not in hypothesis:
                raise ValueError("original paired contact hypothesis lost source support")
            refused_timing.append({"name": hypothesis["name"], "support": support})
            continue
        hypothesis["contact_composition"] = {
            "policy": composition_policy,
            "beam": pair_receipt,
            "support": support,
            "base_contact_alternatives": len(result),
            "invocation_wall_budget_unchanged": True,
        }
        admitted.append(hypothesis)
    for hypothesis in admitted:
        hypothesis["contact_composition"]["refused_timing_alternatives"] = refused_timing
    return admitted


def search_hypotheses(document: dict, attempt: dict) -> list[dict]:
    """Dispatch bound event alternatives; the contact-only API remains unchanged."""
    if document.get("schema") in SCHEMAS:
        from cv.pipeline import s6_contact_prefix_scope as prefix

        contract = attempt.get("observation_scope") or {}
        if (
            contract.get("schema") == prefix.SCHEMA
            and contract.get("mode") == prefix.TERMINAL_IDENTITY
        ):
            document, _ = prefix.prefix_local_witness(attempt, document)
        from cv.pipeline import s6_terminal_net_membership as membership

        alternatives = hypotheses(
            document,
            membership.source_events(attempt),
            observations=attempt.get("owner_ball_labels"),
        )
        if membership.membership_of(attempt) == membership.ABSENT:
            return [
                {**row, "events": membership.effective_events(attempt, row["events"])}
                for row in alternatives
            ]
        return alternatives
    from cv.pipeline import s6_optional_bounce_scope

    checked = s6_optional_bounce_scope.load_witness(attempt)
    if checked != document:
        raise ValueError("optional bounce search witness differs from original packet")
    return s6_optional_bounce_scope.eligible_hypotheses(checked)


def observation_count_receipt(original_count: int, activated_count: int) -> dict:
    """Keep the fixed complexity denominator separate from activated pixel rows.

    Rebound context adds observations, but the predeclared optional-topology
    complexity term uses the original source partition in every branch.
    ``training_observations`` is the historical score-replay denominator.
    """
    if (
        type(original_count) is not int
        or type(activated_count) is not int
        or original_count < 1
        or activated_count < original_count
    ):
        raise ValueError("positive original count and preserved activated inventory required")
    return {
        "training_observations": original_count,
        "original_partition_training_observations": original_count,
        "activated_training_observations": activated_count,
    }


def physical_rank(
    candidate: dict, added: int, occurrence: float, n: int, *, added_parameters: int | None = None
) -> float:
    if n < 1:
        raise ValueError("fixed original training observation count required")
    evidence = candidate["evidence"]
    measurement = candidate["measurement"]
    rms = float(measurement["rms_px"]["training"])
    geometry = float(evidence["input_geometry_penalty"])
    if not np.isfinite([rms, geometry]).all():
        return math.inf
    # Original source score is an engineering sum, not calibrated pixel/meter
    # likelihood. Preserve its components/scaling, plus explicit complexity.
    dimensions = 6 * added if added_parameters is None else added_parameters
    if type(dimensions) is not int or dimensions < 0:
        raise ValueError("nonnegative added continuous parameter count required")
    return (rms * rms + geometry) / (PIXEL_SCALE**2) + (
        dimensions * math.log(2 * n) - 2 * occurrence
    ) / (2 * n)


def best_completed(
    branch: dict,
    hypothesis: dict | None,
    n: int,
    *,
    input_admission=None,
    added_dimensions=None,
    holds: list | None = None,
) -> tuple[dict, float]:
    """Rank completed candidates on the declared input-only objective.

    ``added_dimensions`` is an optional per-candidate hook: it returns the extra
    continuous parameters *this* candidate spent beyond the hypothesis-level
    count, or ``None`` when that candidate's construction receipt cannot be
    substantiated. An unsubstantiated candidate is unrankable and is dropped with
    its reason appended to ``holds`` rather than silently scored at zero. With no
    hook the ranking is exactly the existing one.
    """
    rows = [*branch["coarse"], *branch["refined"]]
    count = 0 if hypothesis is None else len(hypothesis["added"])
    prior = 0 if hypothesis is None else hypothesis["occurrence_log_odds"]
    dimensions = None if hypothesis is None else hypothesis.get("added_parameter_count")
    ranked = []
    for i, row in enumerate(rows):
        extra = 0 if added_dimensions is None else added_dimensions(row)
        if extra is None:
            if holds is not None:
                holds.append(
                    {
                        "depth_hypothesis_m": float(row["depth_hypothesis_m"]),
                        "stage": row.get("stage"),
                        "reason": "candidate response provenance is unknown; not scored",
                    }
                )
            continue
        declared = (
            dimensions if extra == 0 else (6 * count if dimensions is None else dimensions) + extra
        )
        ranked.append(
            (
                physical_rank(row, count, prior, n, added_parameters=declared),
                float(row["depth_hypothesis_m"]),
                i,
                row,
            )
        )
    ranked = [
        row
        for row in ranked
        if math.isfinite(row[0])
        and (input_admission is None or input_admission(row[3])["admissible"])
    ]
    if not ranked:
        raise ValueError("no finite optional-contact candidate")
    selected = min(ranked, key=lambda row: row[:3])
    return selected[3], selected[0]


def validate_report_events(attempt: dict, report: dict) -> bool:
    selection = report.get("optional_contact_selection")
    # Only this witness family's winners are replayed here; a bounce selection
    # keeps its own reader rather than being validated by the contact inventory.
    if not selection or selection.get("policy") != "source_witness_optional_contact_v1":
        return False
    from cv.pipeline.s6_labeled_stage import resolve

    binding = selection["witness_record"]
    path = resolve(binding)
    if provenance.file_sha256(path) != binding["sha256"]:
        raise ValueError("optional contact witness changed before replay")
    document = json.loads(path.read_text())
    from cv.pipeline import s6_contact_prefix_scope as prefix

    contract = attempt.get("observation_scope") or {}
    if contract.get("schema") == prefix.SCHEMA and contract.get("mode") == prefix.TERMINAL_IDENTITY:
        _, receipt = prefix.prefix_local_witness(attempt, document)
        if report.get("configuration", {}).get("contact_prefix_interior_scope") != receipt:
            raise ValueError("prefix-local contact eligibility differs from its source witness")
    # Older union selections declare scope only in their shared source bindings.
    # New explicit declarations must also agree with the original bound witness.
    if (
        selection.get("joint_source_families") is None or "scope" in selection
    ) and declared_final_scope(document) != (selection.get("scope") == FINAL_SCOPE):
        raise ValueError("optional contact selection scope differs from its source witness")
    from cv.pipeline import s6_contact_composition

    if s6_contact_composition.policy(document) != selection.get(
        "optional_contact_composition", "off"
    ):
        raise ValueError("optional contact composition selection differs from its source witness")
    from cv.pipeline import s6_witnessed_interior_contacts

    if s6_witnessed_interior_contacts.policy(document) != selection.get(
        "optional_interior_contacts", "off"
    ):
        raise ValueError("optional interior contact selection differs from its source witness")
    alternatives = search_hypotheses(document, attempt)
    eligible = [{"name": "supplied", "events": attempt["events"]}, *alternatives]
    if not any(
        r["name"] == selection["selected_topology"] and r["events"] == report["events"]
        for r in eligible
    ):
        raise ValueError("search events are not a source-bound optional contact hypothesis")
    return True


def observation_signature(attempt: dict, cameras: dict, partition: str) -> tuple:
    # Terminal-context preparation may carry an original row before extending
    # the declared horizon, then append the same row again. The actual scene
    # keys observations by native frame; count that image equation once here
    # too, while refusing conflicting duplicates instead of choosing a winner.
    camera = {}
    for row in cameras["cameras"]:
        if row["status"] != "supported":
            continue
        frame = int(row["frame"])
        identity = json.dumps(row, sort_keys=True)
        if frame in camera and camera[frame] != identity:
            raise ValueError("conflicting camera identity for one native frame")
        camera[frame] = identity
    observations = {}
    for row in attempt["owner_ball_labels"]:
        if row["status"] != "visible":
            continue
        frame = int(row["frame"])
        if frame != row["frame"]:
            raise ValueError("integer native observation frame required")
        if (
            frame not in camera
            or not attempt["first_event_frame"] <= frame <= attempt["owner_end_frame"]
            or partition != "all_native"
            and frame % 5 == 0
        ):
            continue
        identity = (
            frame,
            float(row["x1080"]),
            float(row["y1080"]),
            camera[frame],
            row.get("native_pts_seconds"),
            row.get("source_pts_seconds"),
            row.get("source_image_sha256"),
            row.get("observation_semantics"),
        )
        if frame in observations and observations[frame] != identity:
            raise ValueError("conflicting observation identity for one native frame")
        observations[frame] = identity
    return tuple(observations[frame] for frame in sorted(observations))


def training_observation_signature(
    attempt: dict,
    cameras: dict,
    partition: str,
    rebound_inventory: dict,
    rebound_receipt: dict,
) -> tuple:
    """Common source identity after the existing declared rebound activation.

    This changes no scene or partition. The ordinary terminal rebound already
    activates supported post-bounce check exposures; its source inventory must
    account for exactly those rows before optional topologies are compared.
    """
    ordinary = observation_signature(attempt, cameras, partition)
    activated = rebound_receipt.get("withheld_rows_activated_frames", [])
    if rebound_receipt.get("withheld_rows_activated_count", 0) != len(activated):
        raise ValueError("rebound activation count differs from its native inventory")
    if "withheld_rows_activated_frames" not in rebound_receipt:
        return ordinary
    if rebound_receipt.get("original_partition_preserved") or partition == "all_native":
        if activated:
            raise ValueError("preserved training partition cannot activate rebound checks")
        return ordinary
    if (
        rebound_inventory.get("mode") != "on"
        or rebound_inventory.get("status") != "supported"
        or rebound_receipt.get("mode") != "on"
        or rebound_receipt.get("original_partition_preserved")
        or partition == "all_native"
    ):
        raise ValueError("training activation requires the existing supported rebound policy")
    source_frames = rebound_inventory["postbounce_labeled_frames"]
    for name, frames in (("source", source_frames), ("activated", activated)):
        if any(not math.isfinite(float(f)) or int(f) != f for f in frames):
            raise ValueError(f"integer native {name} rebound frames required")
        if len(set(frames)) != len(frames):
            raise ValueError(f"duplicate native {name} rebound frames")
    if rebound_receipt.get("withheld_rows_activated_count") != len(activated):
        raise ValueError("rebound activation count differs from its native inventory")
    if rebound_inventory.get("postbounce_labeled_frame_count") != len(source_frames):
        raise ValueError("source rebound count differs from its native inventory")
    bounce = float(rebound_receipt["supplied_terminal_bounce_frame"])
    if not math.isfinite(bounce) or any(float(f) <= bounce for f in source_frames):
        raise ValueError("source rebound exposures must follow the declared terminal bounce")
    if (
        "supplied_terminal_bounce_frame" in rebound_inventory
        and float(rebound_inventory["supplied_terminal_bounce_frame"]) != bounce
    ):
        raise ValueError("rebound activation changed the declared bounce epoch")
    all_native = {row[0]: row for row in observation_signature(attempt, cameras, "all_native")}
    expected = {int(f) for f in source_frames if int(f) in all_native and int(f) % 5 == 0}
    if set(activated) != expected:
        raise ValueError("rebound activation differs from supported original check exposures")
    common = {row[0]: row for row in ordinary}
    common.update({frame: all_native[frame] for frame in expected})
    return tuple(common[frame] for frame in sorted(common))


def cumulative_deadline(total_seconds: float, position: int, total_branches: int) -> float:
    """The shared progressive cumulative deadline for one enumerated branch.

    One clock, one total budget, fixed slices: branch ``position`` (0-based in
    the whole enumeration) may run until ``total * (position + 1) / total``. An
    early failure leaves its remaining time to the branches after it and nothing
    restarts the clock.
    """
    if not isinstance(position, int) or not isinstance(total_branches, int):
        raise ValueError("integer branch enumeration required")
    if total_branches < 1 or not 0 <= position < total_branches:
        raise ValueError("branch position must lie inside the declared enumeration")
    return total_seconds * (position + 1) / total_branches


def fit_alternatives(
    run_topology,
    proposals: list[dict],
    end_frame: float,
    budget,
    total_seconds: float,
    *,
    start_position: int = 1,
    total_branches: int | None = None,
    run_kwargs: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Actual numerical consumer: fixed source order, one shared cumulative budget.

    No fitted parameter is passed to another branch. The no-addition branch has
    already received slice zero, including any immediate topology refusal.
    ``start_position``/``total_branches`` place this arm inside a larger fixed
    enumeration, such as the two declared terminal-net memberships, without
    changing the single cumulative clock or the caller's outer timeout.
    """
    import time
    from cv.experiments.connected_shooting.search_budget import SearchDeadline

    total = len(proposals) + 1 if total_branches is None else total_branches
    extra = dict(run_kwargs or {})
    branches = []
    receipts = []
    for index, hypothesis in enumerate(proposals):
        budget.seconds = cumulative_deadline(total_seconds, start_position + index, total)
        budget.exhausted = False
        try:
            branch = run_topology(hypothesis["events"], end_frame, hypothesis["name"], **extra)
            branch["optional_hypothesis"] = hypothesis
            branches.append(branch)
            receipts.append({"name": hypothesis["name"], "status": "fitted"})
        except (
            ValueError,
            KeyError,
            IndexError,
            FloatingPointError,
            OverflowError,
            SearchDeadline,
        ) as error:
            receipts.append(
                {
                    "name": hypothesis["name"],
                    "status": "no_fit",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    budget.seconds = total_seconds
    budget.exhausted = time.monotonic() - budget.started >= total_seconds
    return branches, receipts
