"""Read-only source/native triage of six one-flight-short labeled points.

Status: opened-development diagnosis. No fitting, relabeling or gate changes.
Run --composition REPORT --output NEW_DIRECTORY. The six-case inventory is fixed
before readback; all 89 attempts remain the cohort denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from cv.experiments.connected_shooting import agent_attempt_prepare
from cv.experiments.connected_shooting import full_native_continuation as full
from cv.experiments.connected_shooting import labeled_context_census as census
from cv.experiments.connected_shooting import labeled_context_witness_scope as scope
from cv.experiments.connected_shooting.labeled_preparation_recovery_review import (
    RUNG as PREPARATION_RUNG,
)
from cv.pipeline import provenance
from scripts.shared_data import repository_root, resolve_shared_root

RUNG = PREPARATION_RUNG + "_win32_shift1f"

KEYS = [
    "uso2020f_pt0006_a01",
    "ao2019f_pt0002_a01",
    "miami2025f_pt0004_a01",
    "miami2025f_pt0002_a01",
    "cincy2024f_pt0004_a01",
    "wim2025f_pt0002_a01",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = resolve_shared_root(repository_root(), None)
    base = json.loads(args.composition.read_text())
    reviewed = {x["key"]: x for x in base["attempts"]}
    inventory_path = root / "processed/ownerfix/adopted_perflight3/report.json"
    inventory = json.loads(inventory_path.read_text())
    items = {x["key"]: x for x in inventory["attempts"]}
    plan = dict(
        question="Which one-flight holds are path errors, biased witnesses, or execution/domain failures?",
        keys=KEYS,
        attempt_denominator=len(base["attempts"]),
        composition=provenance.file_record(args.composition),
        inventory=provenance.file_record(inventory_path),
        producer=provenance.file_record(__file__),
        no_fitting_or_label_changes=True,
        retained_rung=RUNG,
    )
    (args.output / "preregistered.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for key in KEYS:
        item = items[key]
        out = args.output / key
        out.mkdir()
        job = dict(item=item, rung=RUNG)
        context, p, before, _, search, threshold, duration, loaded = full.load_case(job)
        verdict, measurement = census.score(
            context,
            p,
            threshold,
            duration,
            item["metadata"]["ending_kind"],
            scorer=scope.LOCAL_SCORE,
        )
        assert verdict["accepted_flight_count"] == reviewed[key]["after"]["accepted_flights"]
        assert float(p[1]) == reviewed[key]["after"]["selected_depth_m"]
        failed = [f for f in verdict["flights"] if not f["accepted"]]
        assert len(failed) == 1
        held = failed[0]
        i = held["flight_index"]
        scene = context["scene"]
        physical = full.model.chain(scene, p)
        labels = json.loads(loaded["arguments"].labels.read_text())
        cameras = json.loads(loaded["arguments"].cameras.read_text())
        label_events = labels["events"]["records"]
        pictures = agent_attempt_prepare.source_images(labels)
        events = [
            e
            for e in label_events
            if scene.contact_frames[i] - 2 <= e["frame"] <= scene.contact_frames[i + 1] + 4
        ]
        ball = labels["ball"]["records"][0]["frames"]
        native_rows = (
            [x for x in measurement["native_projection"] if x["flight_index"] == i]
            if measurement["native_projection"]
            and "flight_index" in measurement["native_projection"][0]
            else [
                x
                for x in measurement["native_projection"]
                if scene.contact_frames[i] <= x["frame"] <= scene.contact_frames[i + 1]
            ]
        )
        event_frames = [e["frame"] for e in events]
        candidates = [x for x in native_rows if any(abs(x["frame"] - f) <= 2 for f in event_frames)]
        by_frame = {x["frame"]: x for x in candidates}
        candidates = [by_frame[f] for f in sorted(by_frame)]
        # Limit each event to its closest real exposure on either side, plus endpoints.
        chosen = {}
        for frame in event_frames:
            for side in [-1, 1]:
                options = [x for x in candidates if (x["frame"] - frame) * side >= 0]
                if options:
                    x = min(options, key=lambda x: abs(x["frame"] - frame))
                    chosen[x["frame"]] = x
        tiles = []
        native_checks = []
        for frame, x in sorted(chosen.items()):
            path = pictures.get((search["clip"], frame))
            if path is None:
                continue
            im = Image.open(path).convert("RGB")
            cx, cy = x["owner"]
            box = (round(cx) - 200, round(cy) - 140, round(cx) + 200, round(cy) + 140)
            tile = im.crop(box)
            draw = ImageDraw.Draw(tile)
            for xy, color in [(x["owner"], "#00ff66"), (x["predicted"], "#ff3355")]:
                a, z = xy[0] - box[0], xy[1] - box[1]
                draw.ellipse((a - 6, z - 6, a + 6, z + 6), outline=color, width=2)
            draw.rectangle((0, 0, 400, 25), fill="black")
            draw.text(
                (5, 7), f"native {frame}; green input/red fit; {x['error_px']:.1f}px", fill="white"
            )
            tiles.append(tile)
            native_checks.append(dict(projection=x, source=provenance.file_record(path)))
        for start in range(0, len(tiles), 6):
            canvas = Image.new("RGB", (800, 840), "#222222")
            for j, tile in enumerate(tiles[start : start + 6]):
                canvas.paste(tile, ((j % 2) * 400, (j // 2) * 280))
            canvas.save(out / f"native{start // 6 + 1}.jpg", quality=95)
        extended, extension = census.extended_context(context, labels, cameras, search)
        after_extension = None
        if extended is not None:
            v, _ = census.score(
                extended,
                p,
                threshold,
                duration,
                item["metadata"]["ending_kind"],
                scorer=scope.LOCAL_SCORE,
            )
            after_extension = dict(
                accepted_flights=v["accepted_flight_count"],
                held=[
                    dict(flight=f["flight_index"], failures=f["failures"])
                    for f in v["flights"]
                    if not f["accepted"]
                ],
                common_path_check=census.common_path_check(context, extended, p),
            )
        compact = dict(
            key=key,
            ending_kind=item["metadata"]["ending_kind"],
            source_summary=reviewed[key]["after"],
            source_reproduction=loaded["reproduction"],
            sources=loaded["sources"],
            flight_index=i,
            physical=dict(
                start_frame=float(scene.contact_frames[i]),
                end_frame=float(scene.contact_frames[i + 1]),
                start_xyz=physical[i]["start_xyz"],
                end_xyz=physical[i]["end_xyz"],
                bounces=physical[i]["bounces"],
                net_hits=physical[i]["net_hits"],
            ),
            events=events,
            held={k: v for k, v in held.items() if k not in ["directional_windows", "contacts"]},
            native_support=dict(
                training=scene.observation_frames[i],
                check=context["heldout"].observation_frames[i],
                label_frames=[
                    r
                    for r in ball
                    if scene.contact_frames[i] - 2 <= r["frame"] <= scene.contact_frames[i + 1] + 12
                ],
            ),
            native_review=native_checks,
            terminal_context_inventory=extension,
            fixed_vector_extended_score=after_extension,
        )
        (out / "diagnosis.json").write_text(
            json.dumps(full.profile.jsonable(compact), indent=2) + "\n"
        )
        results.append(
            dict(
                key=key,
                held_flight=i,
                accepted_flights=verdict["accepted_flight_count"],
                flight_count=verdict["flight_count"],
                failures=held["failures"],
                rms_px=held["reprojection_rms_px"],
                diagnosis=provenance.file_record(out / "diagnosis.json"),
                extension=after_extension,
            )
        )
        print(key, held["failures"], flush=True)
    (args.output / "inventory.json").write_text(
        json.dumps(
            dict(plan=plan, status="readback_complete_visual_review_pending", cases=results),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
