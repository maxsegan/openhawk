"""Validate an explicit owner-page export for S6 evaluation, never automatic inference.

No labels are generated or repaired. Partial windows, complete windows, court
landmarks and fractional estimates remain distinct, source-bound evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from cv.pipeline.provenance import file_record, git_record

FRAME_STATUSES = {"visible", "occluded", "not_visible", "outside_frame", "ambiguous"}
WINDOW_STATUSES = {"correct", "repaired", "unusable", "ambiguous"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def validate_coordinate(row: dict) -> None:
    require(row.get("status") in FRAME_STATUSES, "invalid visibility status")
    x, y, radius = (row.get(k) for k in ("x1080", "y1080", "uncertainty_radius_px1080"))
    if row["status"] == "visible":
        require(all(finite_number(v) for v in (x, y, radius)), "visible native xy/radius required")
        require(0 <= x < 1920 and 0 <= y < 1080 and radius > 0, "invalid native xy/radius")
    else:
        require(x is None and y is None and radius is None, "abstention coordinates must be null")


def normalize(pack: dict, labels: dict) -> dict:
    """Validate against the immutable target inventory; preserve completeness declarations."""
    require(pack.get("schema") == "s6_owner_input_pack_v1", "unsupported owner pack")
    require(labels.get("schema") == "s6_owner_input_labels_v1", "unsupported owner export")
    for key in ("benchmark_id", "match_id", "native_size", "court_convention", "ball_convention"):
        require(key in pack and labels.get(key) == pack[key], f"pack/export mismatch: {key}")
    require(pack["native_size"] == [1920, 1080], "native 1920x1080 pack required")
    require(
        labels.get("annotation_origin") in {None, "owner", "agent"}, "invalid annotation origin"
    )
    require(
        (pack.get("annotation_origin") == "agent") == (labels.get("annotation_origin") == "agent"),
        "agent pack/export attribution mismatch",
    )
    agent = labels.get("annotation_origin") == "agent"
    if agent:
        provenance = labels.get("provenance", {})
        require(
            provenance.get("annotator") and provenance.get("owner_verified") is False,
            "agent annotator and unverified provenance required",
        )
        require(
            labels.get("extension_schema") == "tennis_ball_streak_endpoints_v1",
            "agent streak extension required",
        )
    cases = {c["id"]: c for c in pack["cases"]}
    require(len(cases) == len(pack["cases"]) and bool(cases), "duplicate or empty case inventory")
    images = {(r["clip"], r["frame"]): r for r in pack["images"]}
    require(len(images) == len(pack["images"]), "duplicate source image inventory")
    ball = labels.get("ball", {})
    require(
        ball.get("schema") == "tennis_ball_track_sequence_labels_v1", "invalid nested ball schema"
    )
    require(ball.get("benchmark_id") == pack["benchmark_id"], "nested ball benchmark mismatch")
    output = {
        k: []
        for k in ("complete_ball_windows", "partial_ball_windows", "court", "fractional_estimates")
    }
    seen = set()
    for source_key, records in (
        ("complete_ball_windows", ball.get("records")),
        ("partial_ball_windows", labels.get("ball_drafts")),
        ("court", labels.get("court")),
    ):
        require(isinstance(records, list), f"missing export section: {source_key}")
        for record in records:
            key = record.get("case_id")
            require(key in cases and key not in seen, "unknown or duplicate case")
            seen.add(key)
            case = cases[key]
            require(
                case["kind"] == ("court" if source_key == "court" else "ball"), "case kind mismatch"
            )
            require(record.get("clip") == case["clip"], "case clip mismatch")
            require(record.get("window_status") in WINDOW_STATUSES, "window status required")
            require(type(record.get("complete")) is bool, "explicit completeness required")
            if source_key != "court":
                require(
                    record["complete"] == (source_key == "complete_ball_windows"),
                    "partial window misclassified",
                )
            targets = {t["id"]: t for t in case["targets"]}
            require(len(targets) == len(case["targets"]), "duplicate pack target")
            rows = record.get("frames")
            require(isinstance(rows, list), "native labels must be a list")
            target_ids = set()
            bound = []
            for row in rows:
                target_id = row.get("target_id")
                require(
                    target_id in targets and target_id not in target_ids,
                    "unknown or duplicate target",
                )
                target_ids.add(target_id)
                frame = row.get("frame")
                require(
                    type(frame) is int and frame == targets[target_id]["frame"],
                    "native frame/target mismatch",
                )
                require(
                    row.get("evidence_kind") is None and row.get("source_frames") is None,
                    "fractional evidence cannot be native",
                )
                validate_coordinate(row)
                source = images.get((case["clip"], frame))
                require(source is not None, "native label lacks source image")
                if agent:
                    require(
                        row.get("source_image_sha256") == source["source"]["sha256"],
                        "agent label/source hash mismatch",
                    )
                    require(bool(row.get("note")), "agent label evidence note required")
                    if source_key != "court":
                        from cv.validation.ball_streak_reference import validate_streak

                        validate_streak(row)
                bound.append({**row, "source_image_sha256": source["source"]["sha256"]})
            if record["complete"]:
                require(target_ids == set(targets), "complete window has missing targets")
            if agent and source_key != "court":
                numbers = [r["frame"] for r in rows]
                require(
                    bool(numbers) and numbers == list(range(numbers[0], numbers[-1] + 1)),
                    "agent windows require contiguous unique native frames",
                )
            output[source_key].append({**record, "frames": bound})
    require(seen == set(cases), "missing cases must remain in the export denominator")
    fractional = labels.get("fractional_estimates", [])
    require(isinstance(fractional, list), "fractional estimates must be a list")
    seen_fractional = set()
    for row in fractional:
        case = cases.get(row.get("case_id"))
        require(
            case is not None and case["kind"] == "ball" and row.get("clip") == case["clip"],
            "fractional case mismatch",
        )
        frame = row.get("frame")
        require(finite_number(frame) and frame % 1 == 0.5, "explicit half-frame estimate required")
        source_frames = [int(frame - 0.5), int(frame + 0.5)]
        require(
            isinstance(row.get("source_frames"), list)
            and all(type(f) is int for f in row["source_frames"])
            and row["source_frames"] == source_frames
            and row.get("blend_weight") == 0.5,
            "fractional source pair mismatch",
        )
        require(
            row.get("evidence_kind") == "human_fractional_estimate_from_adjacent_native_frames",
            "fractional origin required",
        )
        require(
            all(f in case["frames"] and (case["clip"], f) in images for f in source_frames),
            "fractional source outside context",
        )
        key = (case["id"], frame)
        require(key not in seen_fractional, "duplicate fractional estimate")
        seen_fractional.add(key)
        validate_coordinate(row)
        output["fractional_estimates"].append(
            {
                **row,
                "source_image_sha256": [
                    images[(case["clip"], f)]["source"]["sha256"] for f in source_frames
                ],
            }
        )
    events = validate_agent_events(labels, images) if agent else None
    speed = validate_speed_evidence(labels.get("serve_speed_evidence"), images, events)
    return {
        "schema": "s6_owner_input_intake_v1",
        "benchmark_id": pack["benchmark_id"],
        "match_id": pack["match_id"],
        "native_size": pack["native_size"],
        "fps": pack["fps"],
        "scope": "explicit annotated S6 evaluation conditioning; forbidden for automatic inference",
        "annotation_origin": "agent" if agent else "owner",
        "annotator_provenance": labels.get("provenance") if agent else None,
        "human_derived": True,
        "provenance_limit": (
            "Associates the explicit export with the supplied immutable pack; "
            "v1 exports do not cryptographically authenticate the owner or label-time images. "
            "Intake is not a visual-correctness certificate."
        ),
        "ball_convention": pack["ball_convention"],
        "court_convention": pack["court_convention"],
        "window_quality_is_owner_judgment_not_3d_certification": True,
        **output,
        **({"events": events, "attempt": labels["attempt"]} if agent else {}),
        **({"serve_speed_evidence": speed} if speed is not None else {}),
        "counts": {
            "expected_cases": len(cases),
            "complete_ball_windows": len(output["complete_ball_windows"]),
            "partial_ball_windows": len(output["partial_ball_windows"]),
            "native_ball_labels": sum(
                len(r["frames"])
                for k in ("complete_ball_windows", "partial_ball_windows")
                for r in output[k]
            ),
            "court_landmarks": sum(len(r["frames"]) for r in output["court"]),
            "fractional_estimates": len(output["fractional_estimates"]),
        },
    }


def validate_speed_evidence(
    evidence: dict | None, images: dict, events: dict | None
) -> dict | None:
    """Preserve a separately attributed display reading, never a measured launch velocity."""
    if evidence is None:
        return None
    require(isinstance(evidence, dict), "invalid speed evidence")
    require(evidence.get("schema") == "tennis_serve_speed_graphic_evidence_v1", "speed schema")
    require(evidence.get("annotation_origin") == "agent", "speed attribution required")
    require(evidence.get("status") in {"visible", "ambiguous", "not_visible"}, "speed status")
    require(bool(evidence.get("note")), "speed evidence note required")
    value, unit = evidence.get("value"), evidence.get("unit")
    if evidence["status"] == "visible":
        require(finite_number(value) and value > 0 and unit in {"km/h", "mph"}, "speed value/unit")
        contacts = [r for r in (events or {}).get("records", []) if r["event_type"] == "contact"]
        require(
            bool(contacts)
            and evidence.get("contact_event_id") == min(contacts, key=lambda r: r["frame"])["id"],
            "speed must bind first contact",
        )
    else:
        require(value is None and unit is None, "speed abstention value/unit must be null")
    observations = evidence.get("observations")
    require(isinstance(observations, list) and bool(observations), "speed native evidence required")
    seen, current = set(), []
    for row in observations:
        key = (row.get("clip"), row.get("frame"))
        require(type(key[1]) is int and key in images and key not in seen, "speed source frame")
        seen.add(key)
        require(
            row.get("source_image_sha256") == images[key]["source"]["sha256"], "speed source hash"
        )
        require(
            row.get("role") in {"current_serve", "preceding_display", "unresolved"}, "speed role"
        )
        require(
            bool(row.get("display_text")) and bool(row.get("note")), "speed display transcription"
        )
        if row["role"] == "current_serve":
            require(evidence["status"] == "visible", "abstained speed cannot bind current serve")
            require(row.get("value") == value and row.get("unit") == unit, "speed reading mismatch")
            require(
                row["frame"] > min(contacts, key=lambda r: r["frame"])["frame"],
                "stale speed witness",
            )
            current.append(row["frame"])
        box = row.get("crop_native_xywh")
        require(
            isinstance(box, list)
            and len(box) == 4
            and all(type(v) is int for v in box)
            and 0 <= box[0] < 1920
            and 0 <= box[1] < 1080
            and box[2] > 0
            and box[3] > 0
            and box[0] + box[2] <= 1920
            and box[1] + box[3] <= 1080,
            "invalid native speed crop",
        )
        digest = row.get("native_crop_sha256", "")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "speed crop hash",
        )
        path = Path(row.get("crop_path", ""))
        require(
            row.get("crop_path_base") == "TENNIS_DATA_ROOT"
            and bool(row.get("crop_path"))
            and not path.is_absolute()
            and ".." not in path.parts,
            "speed crop path",
        )
    require(evidence.get("visible_frames") == sorted(current), "speed visible-frame inventory")
    require(evidence["status"] != "visible" or bool(current), "visible speed lacks current witness")
    return evidence


def verify_speed_crops(pack: dict, evidence: dict | None, pack_directory: Path) -> list[Path]:
    """Verify PNG bytes and exact native RGB pixels against the bound full source image."""
    if evidence is None:
        return []
    from scripts.shared_data import resolve_shared_root

    root = resolve_shared_root(Path(__file__).resolve().parents[2], None)
    images = {(r["clip"], r["frame"]): r for r in pack["images"]}
    paths = []
    for row in evidence["observations"]:
        path = root / row["crop_path"]
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == row["native_crop_sha256"],
            "speed crop bytes changed",
        )
        source = bound_image_path(pack, images[(row["clip"], row["frame"])], pack_directory)
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == row["source_image_sha256"],
            "speed source bytes changed",
        )
        left, top, width, height = row["crop_native_xywh"]
        with Image.open(source) as native, Image.open(path) as crop:
            expected = native.convert("RGB").crop((left, top, left + width, top + height))
            require(
                crop.size == expected.size and crop.convert("RGB").tobytes() == expected.tobytes(),
                "speed crop differs from native pixels",
            )
        paths.append(path)
    return paths


def validate_agent_events(labels: dict, images: dict) -> dict:
    """Keep event epochs/intervals separate from native ball observations."""
    events = labels.get("events", {})
    require(events.get("schema") == "tennis_attempt_event_labels_v1", "agent event schema required")
    require(type(events.get("complete")) is bool, "explicit event completeness required")
    records = events.get("records")
    require(isinstance(records, list), "event records required")
    seen = set()
    for row in records:
        require(row.get("id") and row["id"] not in seen, "duplicate or missing event id")
        seen.add(row["id"])
        require(
            row.get("event_type") in {"contact", "bounce", "net_hit", "ending", "abstain"},
            "invalid event type",
        )
        frame = row.get("frame")
        require(
            finite_number(frame) and frame * 2 == int(frame * 2),
            "native or half-frame event required",
        )
        interval = row.get("frame_interval")
        require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(finite_number(v) for v in interval)
            and interval[0] <= frame <= interval[1],
            "event interval must contain estimate",
        )
        require(
            row.get("status") in {"labeled", "ambiguous"} and bool(row.get("note")),
            "event status and evidence note required",
        )
        sources = row.get("source_frames")
        require(
            isinstance(sources, list)
            and len(sources) >= 2
            and all(type(f) is int and (row.get("clip"), f) in images for f in sources)
            and sources == sorted(set(sources))
            and min(sources) <= interval[0] <= interval[1] <= max(sources),
            "event interval lacks native neighbors",
        )
        require(
            row.get("source_image_sha256")
            == [images[(row["clip"], f)]["source"]["sha256"] for f in sources],
            "event source hash mismatch",
        )
    attempt = labels.get("attempt", {})
    require(
        attempt.get("clip")
        and attempt.get("coverage") in {"complete_attempt", "owner_overlap_only"},
        "attempt identity and coverage required",
    )
    if attempt["coverage"] == "complete_attempt":
        require(
            any(r["event_type"] == "contact" for r in records)
            and any(r["event_type"] == "ending" for r in records),
            "complete attempt requires contact and ending evidence",
        )
    return events


def bound_image_path(pack: dict, row: dict, pack_directory: Path) -> Path:
    """Resolve explicit shared-root agent images or existing published owner-pack images."""
    if row.get("path_base") == "TENNIS_DATA_ROOT":
        require(pack.get("annotation_origin") == "agent", "shared image paths require agent pack")
        from scripts.shared_data import resolve_shared_root

        root = resolve_shared_root(Path(__file__).resolve().parents[2], None)
        candidate = root / row["image_url"]
        require(
            not Path(row["image_url"]).is_absolute() and ".." not in Path(row["image_url"]).parts,
            "image outside shared data root",
        )
        return candidate.resolve()
    candidate = (pack_directory / row["image_url"]).resolve()
    require(candidate.is_relative_to(pack_directory.resolve()), "image outside published pack")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        help="published immutable manifest.json; agent labels may embed source_pack",
    )
    parser.add_argument("--labels", type=Path, required=True, help="explicit owner-downloaded JSON")
    parser.add_argument(
        "--output", type=Path, required=True, help="new evaluation-only JSON; no overwrite"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [*([args.pack] if args.pack else []), args.labels, Path(__file__)]
    records = [file_record(p) for p in sources]
    labels = json.loads(args.labels.read_text())
    pack = json.loads(args.pack.read_text()) if args.pack else labels.get("source_pack")
    require(isinstance(pack, dict), "explicit pack or embedded source_pack required")
    result = normalize(pack, labels)
    for row in pack["images"]:
        path = bound_image_path(pack, row, args.pack.parent if args.pack else args.labels.parent)
        record = file_record(path)
        require(record["sha256"] == row["source"]["sha256"], "published image differs from source")
        with Image.open(path) as picture:
            require(picture.size == (1920, 1080), "published image is not native 1080")
        sources.append(path)
        records.append(record)
    for path in verify_speed_crops(pack, result.get("serve_speed_evidence"), args.labels.parent):
        sources.append(path)
        records.append(file_record(path))
    require(records == [file_record(p) for p in sources], "inputs changed during intake")
    result.update(
        code=git_record(Path(__file__).resolve().parents[2]),
        input_bindings=[
            {"resolved_path": str(p.resolve()), "record": r}
            for p, r in zip(sources, records, strict=True)
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
