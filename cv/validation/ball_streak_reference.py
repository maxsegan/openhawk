"""Validate and render additive agent streak annotations; never an inference input.

Endpoints describe visible image extent, not latent ball centres or measured shutter times.
Use --labels, --benchmark and --output to reproduce native clean/endpoint review sheets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from cv.experiments.ball_track_labeler.contract import validate_labels
from cv.pipeline.paths import REPO_ROOT, data_root


def validate(payload: dict, benchmark: dict) -> None:
    validate_labels(payload, benchmark)
    if payload.get("extension_schema") != "tennis_ball_streak_endpoints_v1":
        raise ValueError("unsupported streak extension")
    provenance = payload["provenance"]
    if not provenance.get("annotator") or provenance.get("owner_verified") is not False:
        raise ValueError("agent reference must preserve unverified annotator provenance")
    for record in payload["records"]:
        frames = record["frames"]
        numbers = [row["frame"] for row in frames]
        if any(type(number) is not int for number in numbers) or numbers != list(
            range(numbers[0], numbers[-1] + 1)
        ):
            raise ValueError("require complete contiguous unique native frames")
        for row in frames:
            validate_streak(row)


def validate_streak(row: dict) -> None:
    """Validate the native-only extension in either sequence or owner-style exports."""
    streak = row["streak"]
    lead, tail = streak["leading"], streak["trailing"]
    expected = {
        "paired": (True, True),
        "partial": (True, False),
        "ambiguous": (False, False),
    }
    if expected.get(streak["status"]) != (lead is not None, tail is not None):
        raise ValueError("endpoint status disagrees with availability")
    if (row["status"] == "visible") != (lead is not None):
        raise ValueError("leading-edge point visibility mismatch")
    if streak.get("exposure_duration_seconds") is not None:
        raise ValueError("this reference supplies no shutter measurement")
    for endpoint in (lead, tail):
        if endpoint is None:
            continue
        x, y, radius = (endpoint[k] for k in ("x1080", "y1080", "uncertainty_radius_px1080"))
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (x, y, radius)):
            raise ValueError("finite numeric native endpoint required")
        if not (0 <= x < 1920 and 0 <= y < 1080 and radius > 0):
            raise ValueError("endpoint bounds or uncertainty invalid")
    if lead is not None and any(
        lead[k] != row[k] for k in ("x1080", "y1080", "uncertainty_radius_px1080")
    ):
        raise ValueError("point label must retain leading-edge convention")


def render(labels: Path, benchmark_path: Path, output: Path) -> dict:
    payload = json.loads(labels.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    validate(payload, benchmark)
    source_manifest = REPO_ROOT / payload["source_manifest"]
    if (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        != payload["source_manifest_sha256"]
    ):
        raise ValueError("source manifest changed")
    bindings = {
        Path(row["path"]).name: row
        for row in json.loads(source_manifest.read_text())["source_bindings"]
        if row["path_base"] == "TENNIS_DATA_ROOT"
    }
    cases = {case["case_id"]: case for case in benchmark["cases"]}
    panels = []
    receipts = []
    for record in payload["records"]:
        sources = {row["frame"]: row for row in cases[record["case_id"]]["frames"]}
        for row in record["frames"]:
            binding = bindings[sources[row["frame"]]["image_relpath"]]
            source = data_root() / binding["path"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != binding["sha256"]:
                raise ValueError(f"image changed: {source.name}")
            endpoints = [row["streak"][name] for name in ("leading", "trailing")]
            present = [p for p in endpoints if p is not None]
            focus = (
                [sum(p[k] for p in present) / len(present) for k in ("x1080", "y1080")]
                if present
                else row["inspection_focus_only"]
            )
            left, top = (
                max(0, min(1840, round(focus[0]) - 40)),
                max(0, min(1032, round(focus[1]) - 24)),
            )
            with Image.open(source) as native:
                if native.size != (1920, 1080):
                    raise ValueError("native 1920x1080 required")
                clean = (
                    native.convert("RGB")
                    .crop((left, top, left + 80, top + 48))
                    .resize((640, 384), Image.Resampling.NEAREST)
                )
            overlay = clean.copy()
            draw = ImageDraw.Draw(overlay)
            for endpoint, color, name in zip(endpoints, ("#ffff00", "#ff60ff"), ("front", "back")):
                if endpoint is None:
                    continue
                x, y = (endpoint["x1080"] - left) * 8, (endpoint["y1080"] - top) * 8
                radius = endpoint["uncertainty_radius_px1080"] * 8
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), outline=color, width=2
                )
                draw.text((x + radius + 3, y - 12), name, fill=color)
                draw.line((x - 4, y, x + 4, y), fill=color, width=1)
                draw.line((x, y - 4, x, y + 4), fill=color, width=1)
            panels.append((row, clean, overlay, left, top))
            receipts.append(
                {"frame": row["frame"], "source": binding, "crop_native": [left, top, 80, 48]}
            )
    output.mkdir(parents=True, exist_ok=False)
    for panel_index, name in ((1, "clean"), (2, "endpoints")):
        sheet = Image.new("RGB", (1920, 432 * ((len(panels) + 2) // 3)), "#151515")
        draw = ImageDraw.Draw(sheet)
        for index, panel in enumerate(panels):
            row, _, _, left, top = panel
            x, y = index % 3 * 640, index // 3 * 432
            draw.text(
                (x + 4, y + 3),
                f"f{row['frame']}  {row['streak']['status']}  origin {left},{top}  8x native",
                fill="white",
            )
            sheet.paste(panel[panel_index], (x, y + 48))
        sheet.save(output / f"{name}.png")
    report = {
        "schema": "tennis_ball_streak_review_v1",
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "frames": receipts,
        "owner_verified": False,
        "not_an_accuracy_score": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--benchmark", type=Path, help="Legacy sequence benchmark; omit for embedded agent pack"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner-labels", type=Path, help="Explicit post-freeze comparison only")
    parser.add_argument("--owner-case", help="Explicit case ID in owner export")
    parser.add_argument("--owner-events", type=Path, help="Explicit owner event CSV after freeze")
    parser.add_argument("--owner-events-sha256", help="Required immutable event CSV digest")
    args = parser.parse_args()
    if bool(args.owner_labels) != bool(args.owner_case):
        parser.error("--owner-labels and --owner-case must be supplied together")
    if bool(args.owner_events) != bool(args.owner_events_sha256):
        parser.error("--owner-events and --owner-events-sha256 must be supplied together")
    report = (
        render(args.labels, args.benchmark, args.output)
        if args.benchmark
        else render_attempt(args.labels, args.output)
    )
    if args.owner_labels:
        comparison = compare_owner(args.labels, args.owner_labels, args.owner_case)
        (args.output / "owner_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison))
    if args.owner_events:
        comparison = compare_owner_events(args.labels, args.owner_events, args.owner_events_sha256)
        (args.output / "event_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"frames": len(report["frames"]), "labels_sha256": report["labels_sha256"]}))


def compare_owner(labels: Path, owner_labels: Path, owner_case: str) -> dict:
    """Score front-position agreement only, never validate new trailing endpoints."""
    payload = json.loads(labels.read_text())
    if payload.get("annotation_status") != "frozen_agent_reference":
        raise ValueError("freeze agent labels before owner comparison")
    expected_hash = payload["provenance"]["owner_reference"]["sha256"]
    if hashlib.sha256(owner_labels.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("owner source changed")
    owner = json.loads(owner_labels.read_text())
    records = payload.get("records", payload.get("ball", {}).get("records", []))
    if len(records) != 1:
        raise ValueError("explicit single-case comparison required")
    owner_record = next(r for r in owner["ball"]["records"] if r["case_id"] == owner_case)
    if payload.get("annotation_origin") == "agent":
        if payload["match_id"] != owner["match_id"] or records[0]["clip"] != owner_record["clip"]:
            raise ValueError("owner comparison match/clip mismatch")
    truth = {r["frame"]: r for r in owner_record["frames"]}
    rows = []
    agent_rows = {r["frame"]: r for r in records[0]["frames"]}
    # Whole-attempt exports score every owner target, retaining missing predictions.
    targets = (
        truth if payload.get("annotation_origin") == "agent" else {f: truth[f] for f in agent_rows}
    )
    for frame, target in sorted(targets.items()):
        row = agent_rows.get(frame, {"frame": frame, "status": "missing"})
        error = (
            math.hypot(row["x1080"] - target["x1080"], row["y1080"] - target["y1080"])
            if row["status"] == target["status"] == "visible"
            else None
        )
        rows.append(
            {
                "frame": row["frame"],
                "agent_status": row["status"],
                "owner_status": target["status"],
                "leading_error_px": error,
            }
        )
    errors = [r["leading_error_px"] for r in rows if r["leading_error_px"] is not None]
    agent_visible = sum(r["agent_status"] == "visible" for r in rows)
    owner_visible = sum(r["owner_status"] == "visible" for r in rows)
    visibility_matches = sum(r["agent_status"] == r["owner_status"] for r in rows)
    return {
        "schema": "tennis_ball_streak_owner_comparison_v1",
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "owner_sha256": expected_hash,
        "owner_case": owner_case,
        "owner_blind": False,
        "trailing_endpoints_validated": False,
        "frames": rows,
        "denominator_frames": len(rows),
        "compared_fronts": len(errors),
        "mean_error_px": sum(errors) / len(errors) if errors else None,
        "median_error_px": percentile(errors, 0.5),
        "p90_error_px": percentile(errors, 0.9),
        "max_error_px": max(errors) if errors else None,
        "within_px": {str(t): sum(e <= t for e in errors) for t in (6, 12, 24)},
        "within_fraction_all_owner_frames": {
            str(t): sum(e <= t for e in errors) / len(rows) if rows else None for t in (6, 12, 24)
        },
        "visibility": {
            "agent_visible": agent_visible,
            "owner_visible": owner_visible,
            "true_visible": len(errors),
            "precision": len(errors) / agent_visible if agent_visible else None,
            "recall": len(errors) / owner_visible if owner_visible else None,
        },
        "emitted_coverage": agent_visible / len(rows) if rows else None,
        "exact_window_max_error_px": (
            max(errors) if errors and visibility_matches == len(rows) else None
        ),
        "complete_window_within_px": {
            str(t): visibility_matches == len(rows) and all(e <= t for e in errors)
            for t in (6, 12, 24)
        },
        "clean_frame_damage": None,
        "clean_frame_damage_limitation": "No frozen automatic baseline supplied for damage scoring.",
        "broadcasts": 1,
        "clustered_confidence_interval": None,
        "limitation": "One opened window; no transfer, trailing accuracy, or 3D certificate.",
    }


def percentile(values: list[float], quantile: float) -> float | None:
    """Linear empirical quantile, with no dependency on a numeric backend."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    return ordered[low] + (ordered[math.ceil(position)] - ordered[low]) * (position - low)


def compare_owner_events(labels: Path, owner_events: Path, expected_sha256: str) -> dict:
    """Match same-type events within two native frames, retaining misses and extras.

    Only the explicitly frozen attempt window is scored. Point endings remain separate
    from physical bounces; no coordinate, cadence or fractional-estimate conversion occurs.
    """
    payload = json.loads(labels.read_text())
    if payload.get("annotation_status") != "frozen_agent_reference":
        raise ValueError("freeze agent labels before owner comparison")
    if hashlib.sha256(owner_events.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("owner event source changed")
    attempt = payload["attempt"]
    low, high = attempt["native_window"]
    clip = payload["match_id"] + "__" + attempt["clip"]
    truth = []
    with owner_events.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != clip or not row["labeled_frame"]:
                continue
            kind = {"point_end": "ending"}.get(row["event_type"], row["event_type"])
            frame = float(row["labeled_frame"])
            if kind in {"contact", "bounce", "net_hit", "ending"} and low <= frame <= high:
                truth.append({"event_type": kind, "frame": frame, "source_id": row["seed_id"]})
    predictions = payload["events"]["records"]
    edges = sorted(
        (abs(t["frame"] - p["frame"]), i, j)
        for i, t in enumerate(truth)
        for j, p in enumerate(predictions)
        if t["event_type"] == p["event_type"]
        and p["status"] == "labeled"
        and abs(t["frame"] - p["frame"]) <= 2
    )
    matches, used = {}, set()
    for error, i, j in edges:
        if i not in matches and j not in used:
            matches[i] = (j, error)
            used.add(j)
    rows = []
    for i, target in enumerate(truth):
        match = matches.get(i)
        prediction = predictions[match[0]] if match else None
        rows.append(
            {
                **target,
                "agent_frame": prediction["frame"] if prediction else None,
                "absolute_error_frames": match[1] if match else None,
                "owner_in_agent_interval": bool(
                    prediction
                    and prediction["frame_interval"][0]
                    <= target["frame"]
                    <= prediction["frame_interval"][1]
                ),
            }
        )
    return {
        "schema": "tennis_attempt_event_owner_comparison_v1",
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "owner_events_sha256": expected_sha256,
        "clip": clip,
        "owner_blind": False,
        "matching_rule": "one-to-one same type, smallest absolute difference first, <=2 native frames",
        "denominator_events": len(truth),
        "matched_events": len(matches),
        "within_frames": {str(t): sum(m[1] <= t for m in matches.values()) for t in (0, 0.5, 1, 2)},
        "unmatched_agent_events": [p for j, p in enumerate(predictions) if j not in used],
        "events": rows,
    }


def render_attempt(labels: Path, output: Path) -> dict:
    """Render a clean-then-overlay HTML contact sheet with all native frame bindings.

    Paginated PNGs remain small enough to inspect without downsampling the zooms.
    The HTML sheet pairs clean and overlay views and links full native context.
    """
    from cv.validation.s6_owner_inputs import bound_image_path, normalize, verify_speed_crops

    payload = json.loads(labels.read_text())
    pack = payload["source_pack"]
    normalize(pack, payload)
    speed = payload.get("serve_speed_evidence")
    speed_paths = verify_speed_crops(pack, speed, labels.parent)
    sources = {(r["clip"], r["frame"]): r for r in pack["images"]}
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    html = [
        "<!doctype html><meta charset='utf-8'><title>Agent native label review</title>",
        "<style>body{background:#171717;color:white;font:16px sans-serif}img{max-width:100%}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}a{color:#8df}</style>",
        f"<h1>{payload['match_id']} / {payload['attempt']['clip']}</h1>",
        "<p>Clean first, then agent overlay. Yellow: leading tip; magenta: trailing tip. "
        "Rings show uncalibrated native uncertainty. Labels are agent evidence, not owner truth. "
        "Blends are viewing aids; native timestamps are unchanged.</p>",
    ]
    panels = []
    for record in payload["ball"]["records"]:
        for row in record["frames"]:
            binding = sources[(record["clip"], row["frame"])]
            path = bound_image_path(pack, binding, labels.parent)
            if hashlib.sha256(path.read_bytes()).hexdigest() != binding["source"]["sha256"]:
                raise ValueError("source image changed")
            with Image.open(path) as image:
                if image.size != (1920, 1080):
                    raise ValueError("native 1920x1080 required")
                native = image.convert("RGB")
            # Copy source bytes, not a re-encoded substitute exposure.
            native_name = f"f{row['frame']:04}.jpg"
            (output / native_name).write_bytes(path.read_bytes())
            focus = row.get("inspection_focus_only") or [row["x1080"], row["y1080"]]
            left = max(0, min(1840, round(focus[0]) - 40))
            top = max(0, min(1016, round(focus[1]) - 32))
            clean = native.crop((left, top, left + 80, top + 64)).resize(
                (320, 256), Image.Resampling.NEAREST
            )
            overlay = clean.copy()
            draw = ImageDraw.Draw(overlay)
            for key, color in (("leading", "yellow"), ("trailing", "magenta")):
                point = row["streak"][key]
                if point:
                    x, y = (point["x1080"] - left) * 4, (point["y1080"] - top) * 4
                    radius = point["uncertainty_radius_px1080"] * 4
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)
                    draw.line((x - 3, y, x + 3, y), fill=color)
                    draw.line((x, y - 3, x, y + 3), fill=color)
            panels.append((row, clean, overlay, left, top))
            receipts.append(
                {"frame": row["frame"], "source": binding, "crop_native": [left, top, 80, 64]}
            )
    for start in range(0, len(panels), 12):
        page = panels[start : start + 12]
        names = []
        for k, mode in ((1, "clean"), (2, "overlay")):
            sheet = Image.new("RGB", (960, 288 * math.ceil(len(page) / 3)), "#171717")
            draw = ImageDraw.Draw(sheet)
            for i, panel in enumerate(page):
                row, _, _, left, top = panel
                x, y = i % 3 * 320, i // 3 * 288
                draw.text(
                    (x + 3, y + 2),
                    f"f{row['frame']} {row['status']} origin {left},{top} 4x",
                    fill="white",
                )
                sheet.paste(panel[k], (x, y + 32))
            name = f"{mode}_{start // 12 + 1:02}.png"
            sheet.save(output / name)
            names.append(name)
        html.append(
            "<div class='pair'>"
            + "".join(f"<a href='{n}'><img src='{n}'></a>" for n in names)
            + "</div>"
        )
        html.append(
            "<p>Native context: "
            + " ".join(f"<a href='f{p[0]['frame']:04}.jpg'>{p[0]['frame']}</a>" for p in page)
            + "</p>"
        )
    for record in payload["court"]:
        frame = record["frames"][0]["frame"]
        path = output / f"f{frame:04}.jpg"
        if not path.exists():
            binding = sources[(record["clip"], frame)]
            source = bound_image_path(pack, binding, labels.parent)
            if hashlib.sha256(source.read_bytes()).hexdigest() != binding["source"]["sha256"]:
                raise ValueError("court source image changed")
            path.write_bytes(source.read_bytes())
        native = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(native)
        for row in record["frames"]:
            if row["status"] != "visible":
                continue
            x, y, r = row["x1080"], row["y1080"], row["uncertainty_radius_px1080"]
            draw.ellipse((x - r, y - r, x + r, y + r), outline="yellow", width=2)
            draw.text((x + 6, y + 6), row["target_id"], fill="yellow")
        name = f"court_{frame:04}.png"
        native.save(output / name)
        html.append(
            f"<h2>Court f{frame}</h2><div class='pair'><img src='f{frame:04}.jpg'><img src='{name}'></div>"
        )
    import html as html_module

    if speed is not None:
        html.append(
            "<h2>Serve-speed display evidence</h2><pre>"
            + html_module.escape(json.dumps(speed, indent=2))
            + "</pre>"
        )
        for row, path in zip(speed["observations"], speed_paths, strict=True):
            name = f"speed_{row['frame']:04}.png"
            (output / name).write_bytes(path.read_bytes())
            html.append(
                f"<p>Native f{row['frame']} crop, {html_module.escape(row['display_text'])}, {row['role']}</p><img src='{name}'>"
            )
    html.append(
        "<h2>Events and endings</h2><pre>"
        + html_module.escape(json.dumps(payload["events"], indent=2))
        + "</pre>"
    )
    (output / "contact_sheet.html").write_text("\n".join(html) + "\n")
    report = {
        "schema": "tennis_agent_attempt_review_v1",
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "frames": receipts,
        "serve_speed_evidence": speed,
        "owner_verified": False,
        "not_an_accuracy_score": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
