"""Prepare frozen agent S6 labels for the connected research fitter.

Evaluation only. The label JSON embeds its immutable source pack. This adapter
validates that pack and keeps pre/post pictures as context.  The default retains
the anchor transport, which registers each native frame to a labeled ground
anchor.  The joint arm instead fits every visible court/net landmark of the
attempt to one rigid ITF court and fixed-centre rig with one shared radial k1,
optionally including the additive paint-centre landmark frames, and emits that
k1 and its distortion centre on every camera row; it predicts held-out clicks on
its own landmark frames better but loses on landmark frames it did not fit.
Agent events are retained with their original fractional epochs; no authoritative
event truth or automatic-pipeline input is substituted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable
import json
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import (
    camera_bundle,
    camera_cal,
    camera_project,
    court_topology,
    paths,
    provenance,
)
from cv.validation import (
    s6_owner_camera_transport as transport,
    s6_owner_court_audit as court,
    s6_owner_ground_camera as ground,
    s6_owner_inputs,
)

MINIMUM_COURT_SUPPORT = 0.80
TRANSPORT_REFERENCE_POLICIES = ("contact_anchor", "nearest_anchor")
CAMERA_MODELS = ("independent_anchor_transport", "joint_rigid_court_v1")


#: Agent label documents the fitter may open: the frozen reference export and a
#: frozen, separately attributed revision of it (an Astra review pass).
FROZEN_AGENT_STATUSES = frozenset({"frozen_agent_reference", "frozen_agent_revision"})


def transport_reference(
    anchors: list[dict], frame: int, first_event_frame: float, policy: str
) -> dict:
    """Choose a court-only registration reference without consulting ball residuals.

    A single contact-near reference keeps one physical camera parameterization
    across the continuous point while still transporting every native frame.
    The other labeled frames remain independent camera checks.  Nearest-anchor
    transport is retained only as the explicit regression diagnostic because it
    introduces parameter handoffs inside a connected trajectory.
    """
    if not anchors or policy not in TRANSPORT_REFERENCE_POLICIES:
        raise ValueError("nonempty anchors and a declared transport policy required")
    target = first_event_frame if policy == "contact_anchor" else frame
    return min(anchors, key=lambda row: (abs(row["frame"] - target), row["frame"]))


def transport_labeled_frame(job: tuple) -> dict:
    """Register one native image to a labeled court camera without fallback.

    The automatic registration proposal comes from the pipeline.  Its ordinary
    white-line witness is retained, but a saturated-clay witness is also
    evaluated: clay paint can be valid court evidence even when JPEG chroma
    makes the strict white mask reject it.  Both paths retain the same 0.80
    support threshold and fixed-centre physical camera fit.
    """
    frame, source, reference, anchor, reference_frame = job
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    try:
        if frame == reference_frame:
            return dict(
                frame=frame,
                status="supported",
                source="labeled_ground_anchor",
                P=anchor["P"],
                parameters=anchor["parameters"],
                native_warp_rms_px=0.0,
                native_warp_max_px=0.0,
                court_support_mask="labeled_anchor",
                court_support_score=1.0,
            )
        target = cv2.imread(source)
        reference_image = cv2.imread(reference)
        if (
            target is None
            or reference_image is None
            or target.shape[:2] != (1080, 1920)
            or reference_image.shape[:2] != (1080, 1920)
        ):
            raise ValueError("native target and reference images required")
        plane = np.linalg.inv(np.asarray(anchor["P"], float)[:, [0, 1, 3]])
        registered = court_topology.register_static_scene(target, reference_image)
        if registered is None:
            raise ValueError("static-scene registration failed")
        image_warp, evidence = registered
        candidate = plane @ image_warp
        if not court_topology._standard_broadcast_view(candidate, 1920, 1080):
            raise ValueError("registered court geometry is not a standard broadcast view")
        support = {
            name: float(court_topology.topology_score(candidate, mask))
            for name, mask in court_topology.court_line_masks(target)
        }
        support_mask, support_score = max(support.items(), key=lambda item: item[1])
        if support_score < MINIMUM_COURT_SUPPORT:
            raise ValueError(
                "registered court lacks target support "
                f"({support_score:.3f}; standard={support['standard']:.3f})"
            )
        return dict(
            frame=frame,
            status="supported",
            source="registered_fixed_center_labeled_view",
            registration={**evidence, "court_support": support},
            court_support_mask=support_mask,
            court_support_score=support_score,
            **transport.fit_view(anchor, candidate),
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return dict(frame=frame, status="held", reason=str(exc))


def leave_one_labeled_frame_out(
    anchors: list[dict], images: dict[tuple[str, int], Path], clip: str
) -> list[dict]:
    """Predict each landmark frame using only the other landmark cameras."""
    results = []
    for target in anchors:
        alternatives = [anchor for anchor in anchors if anchor["frame"] != target["frame"]]
        if not alternatives:
            results.append(
                dict(frame=target["frame"], status="held", reason="one landmark frame only")
            )
            continue
        reference = min(alternatives, key=lambda item: abs(item["frame"] - target["frame"]))
        camera = transport_labeled_frame(
            (
                target["frame"],
                str(images[(clip, target["frame"])]),
                str(images[(clip, reference["frame"])]),
                reference["fit"],
                reference["frame"],
            )
        )
        if camera["status"] == "supported":
            camera["landmark_reprojection"] = court.landmark_reprojection_errors(
                target["record"], np.asarray(camera["P"], float)
            )
        camera["held_out_anchor_frame"] = target["frame"]
        camera["source_anchor_frame"] = reference["frame"]
        results.append(camera)
    return results


def landmark_frame_motion(
    anchors: list[dict], images: dict[tuple[str, int], Path], clip: str
) -> list[dict]:
    """Measure actual landmark displacement between labeled frames.

    Raw displacement distinguishes a pan/zoom from an anchor issue; registered
    residual then checks that the same static-scene warp carries the landmarks.
    """
    rows = []
    for reference, target in zip(anchors, anchors[1:]):
        try:
            image_warp, evidence = court_topology.register_static_scene(
                cv2.imread(str(images[(clip, target["frame"])])),
                cv2.imread(str(images[(clip, reference["frame"])])),
            ) or (None, None)
            if image_warp is None:
                raise ValueError("static-scene registration failed")
            by_name = {record["target_id"]: record for record in reference["record"]["frames"]}
            common = [
                record
                for record in target["record"]["frames"]
                if record["status"] == "visible"
                and by_name[record["target_id"]]["status"] == "visible"
            ]
            target_xy = np.asarray([[record["x1080"], record["y1080"]] for record in common])
            reference_xy = np.asarray(
                [
                    [by_name[record["target_id"]]["x1080"], by_name[record["target_id"]]["y1080"]]
                    for record in common
                ]
            )
            registered_xy = cv2.perspectiveTransform(
                target_xy.reshape(1, -1, 2).astype(np.float32), image_warp
            )[0]
            raw = np.linalg.norm(target_xy - reference_xy, axis=1)
            residual = np.linalg.norm(registered_xy - reference_xy, axis=1)
            rows.append(
                dict(
                    reference_frame=reference["frame"],
                    target_frame=target["frame"],
                    status="measured",
                    landmarks=len(common),
                    raw_landmark_displacement_median_px=float(np.median(raw)),
                    raw_landmark_displacement_max_px=float(raw.max()),
                    registered_landmark_reprojection_rms_px=float(np.sqrt(np.mean(residual**2))),
                    registered_landmark_reprojection_max_px=float(residual.max()),
                    registration=evidence,
                )
            )
        except (ValueError, cv2.error, KeyError) as exc:
            rows.append(
                dict(
                    reference_frame=reference["frame"],
                    target_frame=target["frame"],
                    status="held",
                    reason=str(exc),
                )
            )
    return rows


def write_transport_overlay(
    image_path: Path, camera: dict, landmark_record: dict | None, output: Path
) -> None:
    """Write a native-size court overlay; no image pixels are synthesized."""
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (1080, 1920):
        raise ValueError("native image required for transport overlay")
    projection = np.asarray(camera["P"], float)
    k1 = float(camera.get("k1", 0.0))
    dist_center = np.asarray(camera.get("dist_center", [960.0, 540.0]), dtype=float)
    court_lines = (
        ((0, 0, 0), (10.97, 0, 0)),
        ((0, 23.77, 0), (10.97, 23.77, 0)),
        ((0, 0, 0), (0, 23.77, 0)),
        ((10.97, 0, 0), (10.97, 23.77, 0)),
        ((1.37, 5.485, 0), (9.6, 5.485, 0)),
        ((1.37, 18.285, 0), (9.6, 18.285, 0)),
        ((5.485, 5.485, 0), (5.485, 18.285, 0)),
        ((0, 11.885, 0), (10.97, 11.885, 0)),
    )
    for start, end in court_lines:
        pixels = (
            camera_project.project_distorted(
                projection, k1, dist_center, np.asarray([start, end], float)
            )
            .round()
            .astype(int)
        )
        cv2.line(image, tuple(pixels[0]), tuple(pixels[1]), (0, 0, 255), 2, cv2.LINE_AA)
    for point in court.NET.values():
        pixel = (
            camera_project.project_distorted(
                projection, k1, dist_center, np.asarray([point], float)
            )[0]
            .round()
            .astype(int)
        )
        cv2.drawMarker(image, tuple(pixel), (0, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    if landmark_record is not None:
        for label in landmark_record["frames"]:
            if label["status"] == "visible":
                cv2.circle(
                    image,
                    (round(label["x1080"]), round(label["y1080"])),
                    5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise ValueError("unable to write transport overlay")


def source_images(labels: dict) -> dict[tuple[str, int], Path]:
    rows = {}
    root = paths.data_root().resolve()
    for record in labels["source_pack"]["images"]:
        if record.get("path_base") != "TENNIS_DATA_ROOT":
            raise ValueError("agent source images must use TENNIS_DATA_ROOT")
        path = (root / record["image_url"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source image escaped TENNIS_DATA_ROOT")
        binding = provenance.file_record(path)
        if (binding["sha256"], binding["bytes"]) != (
            record["source"]["sha256"],
            record["source"]["bytes"],
        ):
            raise ValueError("frozen agent source-image binding changed")
        key = (record["clip"], int(record["frame"]))
        if key in rows:
            raise ValueError("duplicate source image")
        rows[key] = path
    return rows


def anchor_fit(record: dict, width_m: float) -> dict:
    labels = {row["target_id"]: row for row in record["frames"]}
    if any(labels[key]["status"] != "visible" for key in court.GROUND):
        raise ValueError("all eight ground anchors must be visible")
    xyz = np.array(list(court.GROUND.values()), float)
    xyz[4:6, 1] += width_m / 2
    xyz[6:8, 1] -= width_m / 2
    pixels = np.array([[labels[key]["x1080"], labels[key]["y1080"]] for key in court.GROUND])
    fit = ground.fit_ground(xyz, pixels)
    visible_net = [key for key in court.NET if labels.get(key, {}).get("status") == "visible"]
    net_errors = {}
    if visible_net:
        predicted = ground.project(
            np.asarray(fit["P"]), np.asarray([court.NET[key] for key in visible_net])
        )
        for key, estimate in zip(visible_net, predicted, strict=True):
            target = np.array([labels[key]["x1080"], labels[key]["y1080"]])
            net_errors[key] = float(np.linalg.norm(estimate - target))
    return dict(
        case_id=record["case_id"],
        frame=int(record["frames"][0]["frame"]),
        service_line_width_m=width_m,
        fit=fit,
        visible_net_errors_px=net_errors,
        hidden_net_supports=[key for key in court.NET if key not in visible_net],
        airborne_metric_accuracy_certified=False,
    )


ADDENDUM_SUFFIX = "_landmarks_v1.json"
ADDENDUM_MODES = ("none", "alternate", "all")
SHARED_K1_MODES = ("solved", "zero")
# Post and tape-end clicks follow padded silhouettes and an unresolved tape end, so
# they carry an unmodeled nuisance position and never enter the camera fit.
UNCERTIFIED_ADDENDUM_TARGETS = (
    "net_left_post_top",
    "net_left_post_base",
    "net_right_post_top",
    "net_right_post_base",
    "net_band_left_end",
    "net_band_right_end",
)


def landmark_addendum(labels_path: Path, mode: str) -> tuple[list[dict], dict | None]:
    """Load the additive paint-centre landmark frames bound to this label file.

    The addendum is a separate frozen agent reference in its own declared line
    convention. It is evaluation data of the same class as the parent labels, so it
    may condition this evaluation-only camera; ``alternate`` keeps every second
    additive frame so the rest stay a holdout for the emitted camera.
    """
    if mode not in ADDENDUM_MODES:
        raise ValueError(f"unknown landmark addendum mode {mode!r}")
    if mode == "none":
        return [], None
    binding = provenance.file_record(labels_path)
    found = []
    for path in sorted(labels_path.parent.glob(f"*{ADDENDUM_SUFFIX}")):
        document = json.loads(path.read_text())
        parent = document.get("parent_selection", {})
        if parent.get("sha256") == binding["sha256"]:
            found.append((document, path))
    if len(found) != 1:
        raise ValueError(f"exactly one additive landmark file required for {labels_path.name}")
    document, path = found[0]
    if (
        document.get("annotation_origin") != "agent"
        or document.get("annotation_status") not in FROZEN_AGENT_STATUSES
        or document.get("court_convention") != "painted_line_center_v1"
    ):
        raise ValueError("frozen paint-centre agent landmark addendum required")
    records = sorted(document["court"], key=lambda item: int(item["frames"][0]["frame"]))
    if mode == "alternate":
        records = records[::2]
    return records, {
        "path": str(path),
        "record": provenance.file_record(path),
        "convention": document["court_convention"],
        "consumed_frames": [int(item["frames"][0]["frame"]) for item in records],
        "available_frames": document["additional_frame_ids"],
        "mode": mode,
    }


def joint_label_fit(
    records: list[dict],
    anchors: list[dict],
    width_m: float,
    *,
    addendum_records: list[dict] | None = None,
    edge_offset_m: float = 0.025,
    shared_k1: str = "solved",
) -> camera_cal.JointCameraSolution:
    """Fit every visible landmark click in an attempt as an independent observation."""
    if shared_k1 not in SHARED_K1_MODES:
        raise ValueError(f"unknown shared k1 mode {shared_k1!r}")
    observations = []
    initial = {}
    anchors_by_frame = {int(anchor["frame"]): anchor for anchor in anchors}
    for record in records:
        frame = int(record["frames"][0]["frame"])
        if frame in anchors_by_frame:
            projection = np.asarray(anchors_by_frame[frame]["fit"]["P"], dtype=float)
        else:
            visible_ground = [
                label
                for label in record["frames"]
                if label["status"] == "visible" and label["target_id"] in court.GROUND
            ]
            try:
                partial = ground.fit_ground(
                    np.asarray(
                        [
                            camera_cal.landmark_world(label["target_id"], width_m)
                            for label in visible_ground
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        [[label["x1080"], label["y1080"]] for label in visible_ground],
                        dtype=float,
                    ),
                )
                projection = np.asarray(partial["P"], dtype=float)
            except ValueError:
                nearest = min(anchors, key=lambda row: abs(int(row["frame"]) - frame))
                projection = np.asarray(nearest["fit"]["P"], dtype=float)
        initial[frame] = projection
        for label in record["frames"]:
            if label["status"] != "visible":
                continue
            pixel = np.asarray([label["x1080"], label["y1080"]], dtype=float)
            world = camera_cal.landmark_world(label["target_id"], width_m)
            observations.append(
                camera_cal.CameraObservation(
                    frame=frame,
                    landmark=label["target_id"],
                    world=world,
                    pixel=pixel,
                    residual_transform=camera_cal.pixel_to_plane_residual_transform(
                        projection, pixel, float(world[2])
                    ),
                )
            )
    for record in addendum_records or []:
        frame = int(record["frames"][0]["frame"])
        if frame in initial:
            raise ValueError("the additive frames must be disjoint from the parent frames")
        clicks = [
            label
            for label in record["frames"]
            if label["status"] == "visible"
            and label["target_id"] not in UNCERTIFIED_ADDENDUM_TARGETS
            and label["target_id"] in camera_cal.COURT_LANDMARKS
        ]
        worlds = {
            label["target_id"]: camera_cal.landmark_world_convention(
                label["target_id"], "painted_line_center_v1", edge_offset_m=edge_offset_m
            )
            for label in clicks
        }
        ground_clicks = [label for label in clicks if worlds[label["target_id"]][2] == 0.0]
        if len(ground_clicks) < 4:
            continue
        try:
            projection = np.asarray(
                ground.fit_ground(
                    np.asarray([worlds[label["target_id"]] for label in ground_clicks], float),
                    np.asarray(
                        [[label["x1080"], label["y1080"]] for label in ground_clicks], float
                    ),
                )["P"],
                dtype=float,
            )
        except ValueError:
            nearest = min(anchors, key=lambda row: abs(int(row["frame"]) - frame))
            projection = np.asarray(nearest["fit"]["P"], dtype=float)
        initial[frame] = projection
        for label in clicks:
            pixel = np.asarray([label["x1080"], label["y1080"]], dtype=float)
            world = worlds[label["target_id"]]
            observations.append(
                camera_cal.CameraObservation(
                    frame=frame,
                    landmark=label["target_id"],
                    world=world,
                    pixel=pixel,
                    residual_transform=camera_cal.pixel_to_plane_residual_transform(
                        projection, pixel, float(world[2])
                    ),
                )
            )
    solution = camera_cal.fit_joint_camera(
        observations, initial, fixed_k1=0.0 if shared_k1 == "zero" else None
    )
    if not solution.success:
        raise ValueError("joint landmark camera did not converge")
    return solution


def _legacy_parameters(projection: np.ndarray, center: np.ndarray, focal: float) -> list[float]:
    """Retain the seven-value camera parameter field expected by old diagnostics."""
    _, rotation, _, *_ = cv2.decomposeProjectionMatrix(np.asarray(projection, dtype=float))
    rotation_vector, _ = cv2.Rodrigues(rotation)
    translation = -rotation @ np.asarray(center, dtype=float)
    return [*rotation_vector.reshape(3).tolist(), *translation.reshape(3).tolist(), np.log(focal)]


def transport_joint_frame(job: tuple) -> dict:
    """Register one frame, then constrain its view to the jointly fitted fixed rig."""
    frame, source, reference, reference_frame, reference_h, solution = job
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    try:
        if frame in solution.projections:
            projection = solution.projections[frame]
            pan, tilt, focal = solution.view_parameters[frame]
            return dict(
                frame=frame,
                status="supported",
                source="joint_labeled_rigid_court_view",
                P=projection.tolist(),
                parameters=_legacy_parameters(projection, solution.center, focal),
                pan_rad=pan,
                tilt_rad=tilt,
                focal_native_px=focal,
                k1=solution.k1,
                dist_center=solution.dist_center.tolist(),
                native_warp_rms_px=0.0,
                native_warp_max_px=0.0,
                court_support_mask="labeled_joint_fit",
                court_support_score=1.0,
            )
        target = cv2.imread(source)
        reference_image = cv2.imread(reference)
        if (
            target is None
            or reference_image is None
            or target.shape[:2] != (1080, 1920)
            or reference_image.shape[:2] != (1080, 1920)
        ):
            raise ValueError("native target and reference images required")
        registered = court_topology.register_static_scene(target, reference_image)
        if registered is None:
            raise ValueError("static-scene registration failed")
        image_warp, evidence = registered
        candidate = np.asarray(reference_h, dtype=float) @ image_warp
        if not court_topology._standard_broadcast_view(candidate, 1920, 1080):
            raise ValueError("registered court geometry is not a standard broadcast view")
        support = {
            name: float(court_topology.topology_score(candidate, mask))
            for name, mask in court_topology.court_line_masks(target)
        }
        support_mask, support_score = max(support.items(), key=lambda item: item[1])
        if support_score < MINIMUM_COURT_SUPPORT:
            raise ValueError(
                "registered court lacks target support "
                f"({support_score:.3f}; standard={support['standard']:.3f})"
            )
        initial = min(solution.view_parameters.items(), key=lambda item: abs(item[0] - int(frame)))[
            1
        ]
        projection, view, ground_rms = camera_bundle.fit_frame_view(
            np.linalg.inv(candidate), solution, (1920, 1080), initial
        )
        pan, tilt, focal = view
        if not np.isfinite(ground_rms) or ground_rms > 4.0:
            raise ValueError(f"registered fixed-rig ground residual {ground_rms:.3f}px")
        return dict(
            frame=frame,
            status="supported",
            source="registered_joint_fixed_center_view",
            registration={**evidence, "court_support": support},
            court_support_mask=support_mask,
            court_support_score=support_score,
            P=projection.tolist(),
            parameters=_legacy_parameters(projection, solution.center, focal),
            pan_rad=pan,
            tilt_rad=tilt,
            focal_native_px=focal,
            k1=solution.k1,
            dist_center=solution.dist_center.tolist(),
            native_warp_rms_px=float(ground_rms),
            native_warp_max_px=None,
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return dict(frame=frame, status="held", reason=str(exc))


def timing_uncertain_bounce(row: dict) -> bool:
    """Retain a declared bounce whose abstention concerns only its exact epoch.

    Generic ambiguous events remain excluded.  This label convention explicitly
    records a bounce and a bounded timing interval; its representative ``frame``
    is retained for initialization, never promoted to an exact-time annotation.
    """
    from cv.experiments.connected_shooting.labeled_event_occurrence import timing_admission

    return row.get("event_type") == "bounce" and timing_admission(row) is not None


def timing_uncertain_event(row: dict) -> bool:
    from cv.experiments.connected_shooting.labeled_event_occurrence import timing_admission

    return timing_admission(row) is not None


def attempt_record(labels: dict, intake: dict) -> tuple[dict, list[int], list[int]]:
    ball = intake["complete_ball_windows"]
    if len(ball) != 1 or not ball[0]["complete"]:
        raise ValueError("one complete frozen agent ball window required")
    events = [
        row
        for row in labels["events"]["records"]
        if row["status"] == "labeled" or timing_uncertain_event(row)
    ]
    ending_status = "labeled"
    if not any(row["event_type"] == "ending" for row in events):
        # The annotator recorded an ending but abstained on its exact status.
        # An abstention on certainty is not an absent ending: take it with its
        # own bracket and record that the epoch is uncertain.  This can only add
        # attempts, because an attempt with a labeled ending never reaches here.
        uncertain = [
            row
            for row in labels["events"]["records"]
            if row["event_type"] == "ending" and row["status"] != "labeled"
        ]
        if len(uncertain) == 1:
            events = [*events, uncertain[0]]
            ending_status = uncertain[0]["status"]
    contacts = [float(row["frame"]) for row in events if row["event_type"] == "contact"]
    endings = [float(row["frame"]) for row in events if row["event_type"] == "ending"]
    bounces = [row for row in events if row["event_type"] == "bounce"]
    if not contacts or len(endings) != 1:
        raise ValueError("ordered contact(s) and one ending required")
    # A contact after the ending is dead-ball play outside this attempt.  Drop it
    # rather than refusing the attempt; the modeled window already stops at the
    # ending, so no currently accepted attempt changes.
    dead_ball_contacts = [frame for frame in contacts if frame >= endings[0]]
    contacts = [frame for frame in contacts if frame < endings[0]]
    events = [
        row
        for row in events
        if not (row["event_type"] == "contact" and float(row["frame"]) >= endings[0])
    ]
    if not contacts:
        raise ValueError("ordered contact(s) and one later ending required")
    first, ending = contacts[0], endings[0]
    all_rows = ball[0]["frames"]
    in_attempt = [row for row in all_rows if first <= row["frame"] <= ending]
    context = [row["frame"] for row in all_rows if row["frame"] < first or row["frame"] > ending]
    required = list(range(int(np.ceil(first)), int(np.floor(ending)) + 1))
    if [row["frame"] for row in in_attempt] != required:
        raise ValueError("contiguous native in-attempt inventory required")
    physical_events = [
        dict(
            event_type=row["event_type"],
            frame=float(row["frame"]),
            frame_interval=row["frame_interval"],
            note=row["note"],
            annotation_origin="agent",
            **(
                {
                    "status": row["status"],
                    "timing_status": row["timing_status"],
                    "reason_class": row.get("reason_class"),
                    "exact_epoch_observed": False,
                    "occurrence_status": row.get("occurrence_status"),
                    "occurrence_evidence": row.get("occurrence_evidence"),
                }
                if timing_uncertain_event(row)
                else {}
            ),
        )
        for row in events
        if row["event_type"] in {"contact", "bounce", "net_hit"}
    ]
    from cv.experiments.connected_shooting.labeled_event_occurrence import terminal_interval_end

    interval_end = terminal_interval_end(events, ending, ball[0]["clip"])
    bounds = [*contacts, ending]
    interval_bounds = [*contacts, interval_end]
    for row in bounces:
        if timing_uncertain_bounce(row):
            low, high = map(float, row["frame_interval"])
            if (
                sum(a < low and high <= b for a, b in zip(interval_bounds, interval_bounds[1:]))
                != 1
            ):
                raise ValueError("timing-uncertain bounce interval must belong to one flight")
    bounces_by_flight = [
        [row for row in bounces if a < float(row["frame"]) <= b] for a, b in zip(bounds, bounds[1:])
    ]
    attempt = dict(
        attempt_id=labels["benchmark_id"],
        clip=f"{labels['match_id']}__{ball[0]['clip']}",
        point_clip=ball[0]["clip"],
        match_id=labels["match_id"],
        fps=float(labels["source_pack"]["fps"]),
        events=physical_events,
        first_event_frame=first,
        owner_end_frame=ending,
        contact_count=len(contacts),
        terminal_bounce_count=len(bounces_by_flight[-1]),
        # The current connected replay has exactly one bounce state per flight.
        # A second post-winner bounce is evidence to preserve, not a reason to
        # silently pretend that the whole attempt is replayable by that solver.
        structurally_ground_replayable=bool(all(len(rows) == 1 for rows in bounces_by_flight)),
        owner_ball_labels=in_attempt,
        labeled_native_frames=len(in_attempt),
        visible_native_frames=sum(row["status"] == "visible" for row in in_attempt),
        complete_native_label_inventory=True,
        ball_convention=labels["ball_convention"],
        owner_case_id=ball[0]["case_id"],
        owner_window_status=ball[0]["window_status"],
        owner_notes=ball[0]["notes"],
        fractional_estimates=[],
        agent_event_rows=events,
        context_native_frames=context,
        net_hit_modeled=any(row["event_type"] == "net_hit" for row in physical_events),
        annotation_origin="agent",
        ending_record_status=ending_status,
        dead_ball_contacts_after_ending=dead_ball_contacts,
    )
    return attempt, required, context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--service-line-width-m", type=float, default=0.05)
    parser.add_argument(
        "--transport-reference-policy",
        choices=TRANSPORT_REFERENCE_POLICIES,
        default="contact_anchor",
    )
    parser.add_argument(
        "--landmark-addendum",
        choices=ADDENDUM_MODES,
        default="all",
        help=(
            "additive paint-centre landmark frames the joint camera is also fitted to; "
            "'alternate' keeps every second additive frame as an emitted-camera holdout"
        ),
    )
    parser.add_argument(
        "--shared-k1",
        choices=SHARED_K1_MODES,
        default="solved",
        help="solve one radial k1 for the attempt, or hold it at zero",
    )
    parser.add_argument(
        "--edge-offset-m",
        type=float,
        default=0.025,
        help="half the painted line width the additive paint-centre clicks are moved back by",
    )
    parser.add_argument(
        "--camera-model",
        choices=CAMERA_MODELS,
        default="independent_anchor_transport",
        help=(
            "the joint rigid-court arm fits one optical centre and one shared radial k1 to "
            "every visible landmark click and emits them on each row. It predicts held-out "
            "clicks on its own landmark frames far better, but on the additive landmark "
            "frames it never fitted its emitted camera is 0.0729 m against the anchor "
            "transport's 0.0610 m of ground error, worse on 13 of 17 attempts, so it is "
            "not the default"
        ),
    )
    return parser


def main(
    *,
    anchor_fitter: Callable[[dict, float], dict] | None = None,
    frame_transport: Callable[[tuple], dict] | None = None,
    heldout_checker: Callable[[list[dict], dict, str], list[dict]] | None = None,
) -> None:
    """Prepare observations with explicit optional camera operators; defaults are unchanged."""
    anchor_fitter = anchor_fit if anchor_fitter is None else anchor_fitter
    frame_transport = transport_labeled_frame if frame_transport is None else frame_transport
    heldout_checker = leave_one_labeled_frame_out if heldout_checker is None else heldout_checker
    args = build_parser().parse_args()
    if args.output.exists() or not 1 <= args.jobs <= 8:
        raise ValueError("new output and 1-8 workers required")
    labels = json.loads(args.labels.read_text())
    if (
        labels.get("annotation_origin") != "agent"
        or labels.get("annotation_status") not in FROZEN_AGENT_STATUSES
    ):
        raise ValueError("frozen separately attributed agent labels required")
    intake = s6_owner_inputs.normalize(labels["source_pack"], labels)
    images = source_images(labels)
    attempt, frames, context_frames = attempt_record(labels, intake)
    anchors = []
    for record in labels["court"]:
        # The clay labels honestly abstain on the two singles posts, but retain all
        # eight ground controls and the centre strap.  A ground camera can use them.
        try:
            fitted = anchor_fitter(record, args.service_line_width_m)
        except ValueError:
            continue
        anchors.append(dict(**fitted, record=record))
    if not anchors:
        raise ValueError("at least one labeled frame with eight visible ground controls required")
    addendum_records, addendum_binding = (
        landmark_addendum(args.labels, args.landmark_addendum)
        if args.camera_model == "joint_rigid_court_v1"
        else ([], None)
    )
    joint_solution = (
        joint_label_fit(
            labels["court"],
            anchors,
            args.service_line_width_m,
            addendum_records=addendum_records,
            edge_offset_m=args.edge_offset_m,
            shared_k1=args.shared_k1,
        )
        if args.camera_model == "joint_rigid_court_v1"
        else None
    )
    anchor = min(anchors, key=lambda row: abs(row["frame"] - attempt["first_event_frame"]))
    all_frames = sorted({int(row["frame"]) for row in intake["complete_ball_windows"][0]["frames"]})
    visible = {row["frame"] for row in attempt["owner_ball_labels"] if row["status"] == "visible"}
    jobs = []
    job_anchors = []
    for frame in all_frames:
        local = transport_reference(
            anchors, frame, attempt["first_event_frame"], args.transport_reference_policy
        )
        if joint_solution is None:
            jobs.append(
                (
                    frame,
                    str(images[(attempt["point_clip"], frame)]),
                    str(images[(attempt["point_clip"], local["frame"])]),
                    local["fit"],
                    local["frame"],
                )
            )
        else:
            jobs.append(
                (
                    frame,
                    str(images[(attempt["point_clip"], frame)]),
                    str(images[(attempt["point_clip"], local["frame"])]),
                    local["frame"],
                    np.linalg.inv(joint_solution.projections[local["frame"]][:, [0, 1, 3]]),
                    joint_solution,
                )
            )
        job_anchors.append(local["frame"])
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        camera_rows = [
            dict(row, local_anchor_frame=anchor_frame)
            for row, anchor_frame in zip(
                pool.map(
                    frame_transport if joint_solution is None else transport_joint_frame,
                    jobs,
                ),
                job_anchors,
                strict=True,
            )
        ]
    args.output.mkdir(parents=True)
    held = [row for row in camera_rows if row["status"] != "supported"]
    by_frame = {row["frame"]: row for row in camera_rows}
    leave_one_out = heldout_checker(anchors, images, attempt["point_clip"])
    landmark_motion = landmark_frame_motion(anchors, images, attempt["point_clip"])
    landmark_records = {item["frames"][0]["frame"]: item for item in labels["court"]}
    selected_diagnostics = sorted(
        {
            *[item["frame"] for item in anchors],
            *[
                all_frames[round((len(all_frames) - 1) * quantile)]
                for quantile in np.linspace(0.1, 0.9, 6)
            ],
        }
    )
    for frame in selected_diagnostics:
        # A labeled court frame can sit outside the ball window and therefore
        # have no transported camera row.  It is still a valid anchor; it simply
        # has no overlay to draw.
        row = by_frame.get(frame)
        if row is not None and row["status"] == "supported":
            write_transport_overlay(
                images[(attempt["point_clip"], frame)],
                row,
                landmark_records.get(frame),
                args.output / "transport_overlays" / f"frame_{frame:04d}.jpg",
            )
    standard_support = [
        row.get("registration", {}).get("court_support", {}).get("standard")
        for row in camera_rows
        if row["status"] == "supported" and "registration" in row
    ]
    relaxed_support = [
        row.get("registration", {}).get("court_support", {}).get("relaxed_chroma")
        for row in camera_rows
        if row["status"] == "supported" and "registration" in row
    ]
    transport_classification = dict(
        pan_or_zoom_between_landmark_frames=any(
            item.get("raw_landmark_displacement_max_px", float("inf")) > 8
            for item in landmark_motion
            if item["status"] == "measured"
        ),
        wrong_anchor=any(
            item["status"] != "supported"
            or item.get("landmark_reprojection", {}).get("maximum_px", float("inf")) > 8
            for item in leave_one_out
        ),
        clay_line_contrast=bool(
            standard_support
            and relaxed_support
            and np.median(standard_support) < MINIMUM_COURT_SUPPORT
            and np.median(relaxed_support) >= MINIMUM_COURT_SUPPORT
        ),
        doubles_post_only_net=all(len(item["hidden_net_supports"]) == 2 for item in anchors),
        primary_failure=(
            "clay_line_contrast"
            if standard_support
            and relaxed_support
            and np.median(standard_support) < MINIMUM_COURT_SUPPORT
            and np.median(relaxed_support) >= MINIMUM_COURT_SUPPORT
            else None
        ),
        standard_support_median=float(np.median(standard_support)) if standard_support else None,
        relaxed_support_median=float(np.median(relaxed_support)) if relaxed_support else None,
    )
    qualification = dict(
        schema="s6_agent_attempt_camera_qualification_v2",
        status="qualified" if not held else "held_camera_transport",
        attempt_id=attempt["attempt_id"],
        labeled_frames=len(all_frames),
        supported_frames=len(all_frames) - len(held),
        held_frames=len(held),
        visible_observations=len(visible),
        visible_observations_supported=sum(
            by_frame[frame]["status"] == "supported" for frame in visible
        ),
        local_anchor_frames=sorted(set(job_anchors)),
        transport_reference_policy=args.transport_reference_policy,
        camera_model=args.camera_model,
        landmark_addendum=addendum_binding,
        shared_k1_mode=args.shared_k1,
        reference_frame=anchor["frame"]
        if args.transport_reference_policy == "contact_anchor"
        else None,
        held_rows=held,
        leave_one_labeled_frame_out=leave_one_out,
        landmark_frame_motion=landmark_motion,
        classification=transport_classification,
        diagnostic_overlay_frames=selected_diagnostics,
        complete_point_accepted=False,
        automatic_inference_eligible=False,
    )
    landmark_request = dict(
        schema="s6_landmark_review_request_v1",
        status="requested"
        if transport_classification["clay_line_contrast"]
        else "not_required_for_current_transport",
        attempt_id=attempt["attempt_id"],
        purpose=(
            "Independent Astra review of non-anchor clay frames.  Score these only after "
            "the current camera transport is frozen; visible singles supports must be "
            "clicked and absent/ambiguous supports must remain abstentions."
        ),
        requests=(
            [
                dict(
                    frame=frame,
                    landmarks=[*court.GROUND, *court.NET],
                    instructions="Native 1920x1080 coordinates; do not copy an anchor frame.",
                )
                for frame in (78, 145, 167)
                if frame in by_frame
            ]
            if transport_classification["clay_line_contrast"]
            else []
        ),
        automatic_inference_eligible=False,
    )
    (args.output / "landmark_review_request.json").write_text(
        json.dumps(landmark_request, indent=2, allow_nan=False) + "\n"
    )
    qualification["landmark_review_request"] = "landmark_review_request.json"
    (args.output / "qualification.json").write_text(
        json.dumps(qualification, indent=2, allow_nan=False) + "\n"
    )
    image_records = [provenance.file_record(path) for path in images.values()]
    module_records = [
        provenance.file_record(path)
        for path in (
            Path(__file__),
            Path(transport.__file__),
            Path(ground.__file__),
            Path(court.__file__),
            Path(camera_cal.__file__),
            Path(s6_owner_inputs.__file__),
        )
    ]
    packet = dict(
        schema="s6_sparse_owner_input_packet_v1",
        scope=__doc__,
        human_derived=True,
        annotation_origin="agent",
        external_label_binding=dict(
            resolved_path=str(args.labels.resolve()), record=provenance.file_record(args.labels)
        ),
        additive_landmark_binding=addendum_binding,
        inputs=[*image_records, *module_records],
        code=provenance.git_record(paths.REPO_ROOT),
        configuration=dict(
            jobs=args.jobs,
            service_line_width_m=args.service_line_width_m,
            transport_reference_policy=args.transport_reference_policy,
            camera_model=args.camera_model,
            landmark_addendum=addendum_binding,
            shared_k1_mode=args.shared_k1,
            edge_offset_m=args.edge_offset_m,
            context_frames=context_frames,
            context_is_image_only=True,
        ),
        attempts=[{**attempt, "camera_labeled_native_frames": all_frames}],
        input_case_inventory=[
            dict(case_id=ball["case_id"], selected=True) for ball in intake["complete_ball_windows"]
        ],
        scoring_point_denominator_established=False,
        automatic_inference_eligible=False,
    )
    packet_path = args.output / "packet.json"
    packet_path.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    cameras = dict(
        schema=(
            "s6_owner_camera_transport_v1" if joint_solution is None else "s6_owner_joint_camera_v1"
        ),
        scope=__doc__,
        human_derived=True,
        annotation_origin="agent",
        inputs=[*packet["inputs"], provenance.file_record(packet_path)],
        code=packet["code"],
        match_id=attempt["match_id"],
        clip=attempt["point_clip"],
        cameras=camera_rows,
        anchor=anchor,
        all_anchor_diagnostics=anchors,
        joint_rig=(
            None if joint_solution is None else camera_cal.joint_camera_manifest(joint_solution)
        ),
        landmark_addendum=addendum_binding,
        transport_reference_policy=args.transport_reference_policy,
        supported=len(camera_rows) - len(held),
        total=len(camera_rows),
        automatic_inference_eligible=False,
        airborne_metric_accuracy_certified=False,
    )
    (args.output / "cameras.json").write_text(json.dumps(cameras, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                attempt=attempt["attempt_id"],
                frames=len(frames),
                visible=len(visible),
                cameras=len(camera_rows),
                supported=len(camera_rows) - len(held),
                context=len(context_frames),
                anchor_frame=anchor["frame"],
                local_anchor_frames=sorted(set(job_anchors)),
                ground_rms_px=anchor["fit"]["native_rms_px"],
                net_errors_px=anchor["visible_net_errors_px"],
                joint_camera_center_m=(
                    None if joint_solution is None else joint_solution.center.tolist()
                ),
                shared_k1=None if joint_solution is None else joint_solution.k1,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
