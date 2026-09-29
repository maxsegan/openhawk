"""Sanity-check an explicit S6 owner upload without repairing its labels.

Ground-only rigid cameras predict the separately labeled net tops. Outside-edge
and explicitly supplied service-line-center interpretations remain separate;
neither is silently chosen as truth. This is evaluation, not automatic inference.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from cv.pipeline import paths, provenance, camera_cal
from cv.validation import s6_owner_ground_camera as ground, s6_owner_inputs

GROUND = {
    "near_left_doubles": (0, 0, 0),
    "near_right_doubles": (10.97, 0, 0),
    "far_left_doubles": (0, 23.77, 0),
    "far_right_doubles": (10.97, 23.77, 0),
    "near_left_service": (1.37, 5.485, 0),
    "near_right_service": (9.6, 5.485, 0),
    "far_left_service": (1.37, 18.285, 0),
    "far_right_service": (9.6, 18.285, 0),
}
NET = {
    "net_center_top": (5.485, 11.885, 0.914),
    "net_left_singles_stick_top": (0.456, 11.885, 1.07),
    "net_right_singles_stick_top": (10.514, 11.885, 1.07),
}
LANDMARKS = GROUND | NET


def landmark_reprojection_errors(row: dict, projection: np.ndarray) -> dict:
    """Score a proposed camera only on the supplied visible court landmarks.

    This deliberately accepts incomplete landmark inventories.  It is used for
    leave-one-labeled-frame-out camera checks, where an absent singles support
    must stay absent rather than being fabricated for a convenient aggregate.
    """
    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("finite 3x4 projection required")
    labels = {item["target_id"]: item for item in row["frames"]}
    if len(labels) != len(row["frames"]) or not set(labels) <= set(LANDMARKS):
        raise ValueError("unique known court landmark ids required")
    measured = []
    for landmark, label in labels.items():
        if label["status"] != "visible":
            continue
        s6_owner_inputs.validate_coordinate(label)
        expected = ground.project(matrix, np.asarray([LANDMARKS[landmark]], float))[0]
        observed = np.asarray([label["x1080"], label["y1080"]], dtype=float)
        measured.append(
            dict(
                landmark=landmark,
                error_px=float(np.linalg.norm(expected - observed)),
                predicted_xy=expected.tolist(),
                labeled_xy=observed.tolist(),
            )
        )
    if not measured:
        raise ValueError("at least one visible supplied landmark required")
    errors = np.asarray([item["error_px"] for item in measured], dtype=float)
    return dict(
        visible_landmarks=len(measured),
        withheld_or_ambiguous_landmarks=len(labels) - len(measured),
        rms_px=float(np.sqrt(np.mean(errors**2))),
        median_px=float(np.median(errors)),
        maximum_px=float(errors.max()),
        landmarks=measured,
    )


def evaluate(row, *, service_line_width_m=0.0):
    if not np.isfinite(service_line_width_m) or not 0 <= service_line_width_m <= 0.05:
        raise ValueError("explicit service line width in [0, .05] m required")
    labels = {r["target_id"]: r for r in row["frames"]}
    if len(labels) != len(row["frames"]) or set(labels) != set(GROUND) | set(NET):
        raise ValueError("unique complete court and net inventory required for this diagnostic")
    if any(r["status"] != "visible" for r in labels.values()):
        raise ValueError(
            "this diagnostic requires visible supplied landmarks; do not invent hidden ones"
        )
    for r in labels.values():
        s6_owner_inputs.validate_coordinate(r)
    xyz = np.array(list(GROUND.values()), float)
    xyz[4:6, 1] += service_line_width_m / 2
    xyz[6:8, 1] -= service_line_width_m / 2
    pixels = np.array([[labels[k]["x1080"], labels[k]["y1080"]] for k in GROUND])
    fit = ground.fit_ground(xyz, pixels)
    matrix = np.array(fit["P"])
    net_pixels = np.array([[labels[k]["x1080"], labels[k]["y1080"]] for k in NET])
    predicted_net = ground.project(matrix, np.array(list(NET.values())))
    omitted = []
    for i, key in enumerate(GROUND):
        mask = np.arange(8) != i
        try:
            control = ground.fit_ground(xyz[mask], pixels[mask])
            error = float(
                np.linalg.norm(
                    ground.project(np.array(control["P"]), xyz[i : i + 1])[0] - pixels[i]
                )
            )
            omitted.append(dict(landmark=key, status="measured", error_px=error))
        except ValueError as e:
            omitted.append(dict(landmark=key, status="held", reason=str(e)))
    return dict(
        case_id=row["case_id"],
        owner_complete=row["complete"],
        owner_window_status=row["window_status"],
        owner_notes=row["notes"],
        service_line_width_m=service_line_width_m,
        interpretation="declared outside edges"
        if service_line_width_m == 0
        else "service Y at line center; X outside edge; assumed width, not measured",
        net_support="singles net supports at singles sideline plus .914 m; not doubles-net posts",
        fit=fit,
        leave_one_ground_landmark_out=omitted,
        withheld_net=[
            dict(
                landmark=k,
                error_px=float(np.linalg.norm(p - q)),
                predicted_xy=p.tolist(),
                owner_xy=q.tolist(),
            )
            for k, p, q in zip(NET, predicted_net, net_pixels)
        ],
        labels_changed=False,
        automatic_inference_eligible=False,
        airborne_accuracy_certified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-line-center-width-m", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.intake.read_text())
    if data.get("schema") != "s6_owner_input_intake_v1" or data.get("human_derived") is not True:
        raise ValueError("explicit owner intake required")
    files = [
        args.intake,
        Path(__file__),
        Path(ground.__file__),
        Path(camera_cal.__file__),
        Path(s6_owner_inputs.__file__),
        Path(provenance.__file__),
        Path(paths.__file__),
    ]
    for binding in data["input_bindings"]:
        files.append(ground.resolve_record(binding["record"]))
    records = [provenance.file_record(p) for p in files]
    widths = (
        [0.0]
        if args.service_line_center_width_m is None
        else [0.0, args.service_line_center_width_m]
    )
    rows = [evaluate(r, service_line_width_m=w) for r in data["court"] for w in widths]
    views = []
    originals = {r["case_id"]: r for r in data["court"]}
    for row in rows:
        original = originals[row["case_id"]]
        frame = original["frames"][0]["frame"]
        suffix = f"/images/{original['clip']}_{frame:04d}.jpg"
        images = set(p for p in files if p.as_posix().endswith(suffix))
        if len(images) != 1:
            raise ValueError("one source-bound native court image required")
        encoded = base64.b64encode(next(iter(images)).read_bytes()).decode()
        marks = []
        for label in original["frames"]:
            x, y = label["x1080"], label["y1080"]
            marks.append(
                f'<circle cx="{x}" cy="{y}" r="7" stroke="#00ff90" stroke-width="2" fill="none"/>'
            )
            marks.append(
                f'<text x="{x + 10}" y="{y - 10}" fill="#00ff90" stroke="#111" stroke-width=".4" font-size="16">{html.escape(label["target_id"])}</text>'
            )
        for label in row["withheld_net"]:
            x, y = label["predicted_xy"]
            marks.append(
                f'<path d="M{x - 6},{y}h12 M{x},{y - 6}v12" stroke="#ff6464" stroke-width="2"/>'
            )
        views.append(
            "<h2>"
            + html.escape(row["case_id"] + " · " + row["interpretation"])
            + '</h2><p>Green: unchanged owner labels. Red: net predictions from ground-only camera.</p><svg viewBox="0 0 1920 1080" style="width:100%"><image width="1920" height="1080" href="data:image/jpeg;base64,'
            + encoded
            + '"/>'
            + "".join(marks)
            + "</svg>"
        )
    result = dict(
        schema="s6_owner_court_sanity_v1",
        scope=__doc__,
        code=provenance.git_record(paths.REPO_ROOT),
        human_derived=True,
        inputs=records,
        cases=rows,
    )
    if records != [provenance.file_record(p) for p in files]:
        raise ValueError("inputs changed during audit")
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Owner court sanity</title><h1>Owner court sanity — not 3D certification</h1><pre>'
        + html.escape(json.dumps(rows, indent=2))
        + "</pre>"
        + "".join(views)
    )
    print(
        json.dumps(
            [
                dict(
                    case=r["case_id"],
                    width=r["service_line_width_m"],
                    ground_rms=r["fit"]["native_rms_px"],
                    net_errors=[x["error_px"] for x in r["withheld_net"]],
                )
                for r in rows
            ]
        )
    )


if __name__ == "__main__":
    main()
