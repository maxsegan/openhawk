"""Native pictures for one audited flight of a per-flight acceptance run.

Evaluation only.  Every picture is a native 1080 source exposure named by the
label pack, drawn with the owner's labeled front, the fitted leading-edge
prediction and the fitted nominal centre at that same exposure, plus one
court-plane plate carrying the flight's bounce witness circle and the fitted
impact.  No picture is invented, resampled to a new cadence or re-timed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cv.experiments.connected_shooting import agent_attempt_prepare, model
from cv.pipeline import paths, provenance


def _candidate(search: dict, depth_m: float | None) -> dict:
    rows = search["refined_candidates"]
    if depth_m is None:
        return rows[0]
    return min(rows, key=lambda row: abs(float(row["depth_hypothesis_m"]) - float(depth_m)))


def flight_pictures(
    search: dict,
    per_flight: dict,
    labels: dict,
    rung: str,
    flight_index: int,
    output: Path,
    *,
    maximum_pictures: int = 8,
) -> list[str]:
    """Draw the audited flight's own exposures and its court-plane evidence."""
    entry = next(row for row in per_flight["rungs"] if row["rung"] == rung)
    verdict = entry["verdict"]
    flight = next(row for row in verdict["flights"] if row["flight_index"] == flight_index)
    candidate = _candidate(search, entry["selected_depth_m"])
    rows = [
        row
        for row in candidate["measurement"]["native_projection"]
        if flight["start_frame"] <= row["frame"] <= flight["end_frame"]
    ]
    sources = agent_attempt_prepare.source_images(labels)
    clip = search["clip"]
    available = [row for row in rows if (clip, row["frame"]) in sources]
    if not available:
        raise ValueError("this flight has no native source exposure to draw")
    step = max(1, len(available) // maximum_pictures)
    chosen = available[::step][:maximum_pictures]
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for row in chosen:
        image = Image.open(sources[(clip, row["frame"])]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for xy, colour, radius in (
            (row["owner"], "#00ff66", 9),
            (row["predicted"], "#ff3355", 7),
            (row["nominal_centre"], "#33bbff", 4),
        ):
            x, y = map(float, xy)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=3)
        draw.line(
            [tuple(map(float, row["owner"])), tuple(map(float, row["predicted"]))],
            fill="#ffcc00",
            width=2,
        )
        draw.text(
            (12, 12),
            "\n".join(
                [
                    f"{search['clip']} flight {flight_index} ({flight['role']}) native f{row['frame']}",
                    f"green owner front / red fitted front / blue fitted centre; miss {row['error_px']:.1f} px",
                    f"rejected by: {', '.join(flight['failures']) or 'nothing'}",
                    f"flight reprojection RMS {flight['reprojection_rms_px']:.1f} px, "
                    f"worst event window {flight['maximum_window_rms_px']:.1f} px"
                    if flight["reprojection_rms_px"] is not None
                    and flight["maximum_window_rms_px"] is not None
                    else "flight reprojection unavailable",
                ]
            ),
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        name = f"flight{flight_index:02d}_f{row['frame']:04d}.jpg"
        image.save(output / name, quality=92)
        artifacts.append(name)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dense = candidate["measurement"]["dense_flights"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for index, entry_flight in enumerate(dense):
        xyz = np.asarray(entry_flight["positions"], float)
        frames = np.linspace(entry_flight["start_frame"], entry_flight["end_frame"], len(xyz))
        accepted = (
            verdict["flights"][index]["accepted"] if index < len(verdict["flights"]) else None
        )
        colour = "#bbbbbb" if index != flight_index else ("#1f9d55" if accepted else "#cc2233")
        width = 1.0 if index != flight_index else 2.5
        axes[0].plot(xyz[:, 0], xyz[:, 1], color=colour, lw=width)
        axes[1].plot(frames, xyz[:, 2], color=colour, lw=width)
    for witness in flight["bounce_witness"]:
        target = np.asarray(witness["witness_xyz_m"], float)
        radius = float(witness["graded_circle_radius_m"])
        axes[0].add_patch(
            plt.Circle((target[0], target[1]), radius, fill=False, color="#00aaff", lw=1.5)
        )
        axes[0].scatter(target[0], target[1], marker="+", color="#00aaff", s=80)
        if witness["modeled_xyz_m"] is not None:
            fitted = np.asarray(witness["modeled_xyz_m"], float)
            axes[0].scatter(fitted[0], fitted[1], marker="x", color="#cc2233", s=80)
            axes[0].plot(
                [target[0], fitted[0]],
                [target[1], fitted[1]],
                color="#cc2233",
                ls=":",
                lw=1.2,
            )
    axes[0].plot([0, 10.97, 10.97, 0, 0], [0, 0, 23.77, 23.77, 0], color="black")
    axes[0].axhline(11.885, color="black", ls="--")
    axes[0].set(
        xlabel="court X (m)",
        ylabel="court Y (m)",
        title=f"court plane; flight {flight_index} in colour",
        aspect="equal",
    )
    axes[1].axhline(model.R_BALL, color="black", ls=":")
    axes[1].set(xlabel="native frame", ylabel="ball centre Z (m)", title="side elevation")
    figure.suptitle(
        f"{search['clip']} flight {flight_index} ({flight['role']}) at rung {rung}: "
        f"{', '.join(flight['failures']) or 'accepted'}"
    )
    figure.tight_layout()
    name = f"flight{flight_index:02d}_court_and_side.png"
    figure.savefig(output / name, dpi=130)
    plt.close(figure)
    artifacts.append(name)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("search-report", "per-flight", "labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rung", default="base")
    parser.add_argument("--flight-index", type=int, required=True)
    parser.add_argument("--maximum-pictures", type=int, default=8)
    args = parser.parse_args()
    search = json.loads(args.search_report.read_text())
    per_flight = json.loads(args.per_flight.read_text())
    labels = json.loads(args.labels.read_text())
    artifacts = flight_pictures(
        search,
        per_flight,
        labels,
        args.rung,
        args.flight_index,
        args.output,
        maximum_pictures=args.maximum_pictures,
    )
    (args.output / "pictures.json").write_text(
        json.dumps(
            {
                "schema": "connected_per_flight_pictures_v1",
                "clip": search["clip"],
                "rung": args.rung,
                "flight_index": args.flight_index,
                "artifacts": artifacts,
                "inputs": [
                    provenance.file_record(path)
                    for path in (args.search_report, args.per_flight, args.labels)
                ],
                "code": provenance.git_record(paths.REPO_ROOT),
                "automatic_inference_eligible": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"artifacts": artifacts}), flush=True)


if __name__ == "__main__":
    main()
