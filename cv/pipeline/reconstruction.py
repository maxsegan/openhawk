"""Label-free 3D rally reconstruction from automatic pipeline artifacts.

Owner-truth loading and scoring live in ``cv.validation.v4_physics_lift``. This module accepts no
label path and has no validation dependency.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import multiprocessing as mp
import queue
import sys
import time
import traceback
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv.pipeline.camera_scope import window_scope  # noqa: E402
from cv.pipeline.contact_frame_refiner import (  # noqa: E402
    RefinedContact,
    refine as refine_contact_frame,
)
from cv.pipeline.anchor_first_fit import (  # noqa: E402
    configure_contact_adjacent_weighting,
    configure_contact_observation_sigma,
    configure_contact_observation_witness,
    configure_physical_spin_prior,
    configure_subframe_anchors,
    configure_subframe_contact_parts,
    configure_subframe_contacts,
    configure_subframe_bounce_witness,
    configure_subframe_plane_anchor_error,
    configure_whole_point_joint,
    configure_striker_witness,
    fit_anchor_first_shot,
    held_out_summary,
    refine_anchor_first_point,
)
from cv.pipeline.flight_ledger import classify_flight  # noqa: E402
from cv.pipeline.flight_anchors import (  # noqa: E402
    build_bounce_witnesses,
    build_point_anchors,
    build_time_priors,
    write_point_anchors,
)
from cv.pipeline.rich_ball_physics import (  # noqa: E402
    R_BALL,
    bounce_for_shot,
    bounce_energy_ratio,
    contact_image_observation,
    contact_image_residual,
    contact_ray_anchor,
    contact_ray_candidates,
    fit_fixed_start_terminal_segment,
    fit_shot,
    fuse_contacts,
    hard_bounce_recovery_score,
    jointly_refine_all,
    initialize_missing_shot,
    nearest,
    project_one,
    ray_at_y,
    recover_hard_bounce_shot,
    shoot_shot,
    spin_component_rpm,
    spin_vector,
)
from cv.pipeline.pipeline_evidence import (  # noqa: E402
    event_type_probability,
    select_physics_compatible_bounce,
    tracking_observation_weight,
)
from cv.pipeline.physics_interpretation import (  # noqa: E402
    bounce_options,
    branch_is_decisive,
    build_point_branches,
    refined_branch_score,
    shot_spin_options,
)
from cv.pipeline import resolution as res  # noqa: E402
from cv.pipeline.pose import load_native_pose_rows  # noqa: E402
from cv.pipeline.point_grammar import select_consumer_emissions  # noqa: E402
from cv.pipeline.track_heal import heal_track  # noqa: E402
from cv.pipeline.terminal_completion import complete_second_bounce, completion_window  # noqa: E402
from cv.pipeline.trajectory_contract import (  # noqa: E402
    terminal_boundary_kind,
    terminal_coverage_report,
    trajectory_connection_report,
)
from cv.pipeline.event_topology import (  # noqa: E402
    build_contact_topology_branches,
    score_contact_topology_branch,
    select_contact_topology_branch,
    shortlist_contact_topology_branches,
)

TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"
# Candidate flight scope, not an acceptance threshold. Slow slices and lobs in the
# owner corpus exceed 2.2 seconds and must reach the physics scorer.
MAX_SHOT_SECONDS = 3.5
MIN_TERMINAL_OBSERVATIONS = 10
MAX_TERMINAL_TRACK_GAP_SECONDS = 0.16
MAX_TERMINAL_REACQUISITION_GAP_SECONDS = 1.0
OUT_OF_FRAME_EDGE_PX = 80.0
MIN_TERMINAL_REACQUISITION_WEIGHT = 0.55
MIN_TERMINAL_BOUNCE_ANCHOR_PROBABILITY = 0.70
CONTACT_SIDE_BALL_WINDOW_SECONDS = 0.12
CONTACT_SIDE_BOX_WINDOW_SECONDS = 0.16
CONTACT_SIDE_MIN_CONFIDENCE = 0.50
APPARENT_PLAYER_HEIGHT_M = 1.85
S6_BOUNCE_MAX_TIMING_FRAMES = 2.0
S6_BOUNCE_MAX_COURT_DELTA_M = 1.0
S6_BOUNCE_MIN_PROBABILITY = 0.25
HARD_BOUNCE_FRAME_TOLERANCE = 1e-6
HARD_BOUNCE_POSITION_TOLERANCE_M = 1e-6
NET_COURT_Y_M = 11.885
NET_CENTER_HEIGHT_M = 0.914
NET_POST_HEIGHT_M = 1.07
NET_COLLISION_HEIGHT_TOLERANCE_M = 0.05
NET_COLLISION_SEARCH_FRAMES = 10


def select_reconstruction_bounce_anchor(
    candidates: list[dict],
    predicted_bounces: list[dict],
    *,
    fps: float,
) -> dict | None:
    """Choose a bounce node for bounded flight recovery.

    Prefer agreement with an unconstrained physical fit. When that fit cannot represent
    the discontinuity, admit exactly one high-confidence event hypothesis and rely on the
    bounce-node solver's absolute physical screen rather than silently dropping the flight.
    """
    compatible = select_physics_compatible_bounce(
        candidates,
        predicted_bounces,
        fps=fps,
    )
    if compatible is not None:
        return compatible
    high_confidence = [
        row
        for row in candidates
        if float(row.get("probability", 0.0)) >= MIN_TERMINAL_BOUNCE_ANCHOR_PROBABILITY
    ]
    if len(high_confidence) != 1:
        return None
    return high_confidence[0]


def validate_artifact_cadence(match_dir: Path, expected_fps: float) -> dict:
    """Fail closed when rendered frames and manifest timing use different cadences."""
    sidecar = match_dir / "audit_frames_native_1080.coordinates.json"
    if not sidecar.is_file():
        raise ValueError(f"missing frame coordinate sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text())
    artifact_fps = optional_float(payload.get("fps"))
    if artifact_fps is None:
        raise ValueError(f"frame coordinate sidecar lacks fps: {sidecar}")
    if not math.isclose(artifact_fps, expected_fps, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(
            "artifact cadence mismatch: "
            f"manifest={expected_fps:g} frames={artifact_fps:g} sidecar={sidecar}"
        )
    return {
        "manifest_fps": expected_fps,
        "artifact_fps": artifact_fps,
        "sidecar": str(sidecar),
        "matched": True,
    }


def upstream_hold_is_repairable(validity_row: dict | None) -> bool:
    """Allow S6 to reconsider tracking-only holds, never scope/camera/cadence failures."""
    if validity_row is None or validity_row.get("decision") == "retain":
        return False
    return set(validity_row.get("reasons", [])) == {"tracking_risk"}


def upstream_hold_has_retained_tracking_arc(
    validity_row: dict | None,
    tracking_row: dict | None,
) -> bool:
    """Allow retained arcs through a stale whole-point tracking hold."""
    if validity_row is None or validity_row.get("decision") == "retain" or tracking_row is None:
        return False
    hard_reasons = set(validity_row.get("reasons", [])) - {
        "tracking_risk",
        "tracking_arc_abstentions",
        "active_play_ambiguous",
    }
    return not hard_reasons and any(
        arc.get("decision") == "retain" for arc in tracking_row.get("arcs", [])
    )


def tracking_arc_fit_scope(
    tracking_row: dict | None,
    start_frame: float,
    end_frame: float,
    observed_frames: set[int],
) -> dict:
    """Select only retained-arc observations for one event-defined flight."""
    arcs = list((tracking_row or {}).get("arcs", []))
    if not arcs:
        return {
            "available": False,
            "decision": "legacy_point_scope",
            "retained_arc_ids": [],
            "held_arc_ids": [],
            "retained_frames": sorted(
                frame for frame in observed_frames if start_frame <= frame <= end_frame
            ),
        }
    overlapping = [
        arc
        for arc in arcs
        if float(arc["end_frame"]) >= start_frame and float(arc["start_frame"]) <= end_frame
    ]
    retained = [arc for arc in overlapping if arc.get("decision") == "retain"]
    retained_frames = sorted(
        frame
        for frame in observed_frames
        if start_frame <= frame <= end_frame
        and any(float(arc["start_frame"]) <= frame <= float(arc["end_frame"]) for arc in retained)
    )
    return {
        "available": True,
        "decision": "attempt" if retained_frames else "hold",
        "retained_arc_ids": [arc.get("arc_id") for arc in retained],
        "held_arc_ids": [arc.get("arc_id") for arc in overlapping if arc.get("decision") == "hold"],
        "retained_frames": retained_frames,
    }


def verify_hard_bounce_anchors(
    fits: dict,
    contacts: list[dict],
    anchors: list[dict],
) -> dict:
    """Verify that every hard event anchor is actually consumed by the final fit set."""
    required = [row for row in anchors if row.get("hard_geometry")]
    violations = []
    satisfied_flights = []
    for anchor in required:
        enclosing = [
            index
            for index in range(len(contacts) - 1)
            if contacts[index]["frame"] < anchor["frame"] < contacts[index + 1]["frame"]
        ]
        if len(enclosing) != 1:
            violations.append(
                {
                    "frame": float(anchor["frame"]),
                    "reason": "hard_bounce_has_no_unique_flight",
                    "flight_indices": enclosing,
                }
            )
            continue
        index = enclosing[0]
        fit = fits.get(index)
        if fit is None:
            violations.append(
                {
                    "frame": float(anchor["frame"]),
                    "reason": "hard_bounce_flight_unsolved",
                    "flight_index": index,
                }
            )
            continue
        consumed = getattr(fit, "_fixed_bounce_anchor", None)
        if consumed is None:
            violations.append(
                {
                    "frame": float(anchor["frame"]),
                    "reason": "hard_bounce_not_consumed",
                    "flight_index": index,
                }
            )
            continue
        if getattr(fit, "_initialization_only", False):
            violations.append(
                {
                    "frame": float(anchor["frame"]),
                    "reason": "hard_bounce_initializer_not_refined",
                    "flight_index": index,
                }
            )
            continue
        frame_delta = abs(float(consumed["frame"]) - float(anchor["frame"]))
        position_delta = float(
            np.linalg.norm(np.asarray(consumed["x"], float) - np.asarray(anchor["x"], float))
        )
        if (
            frame_delta > HARD_BOUNCE_FRAME_TOLERANCE
            or position_delta > HARD_BOUNCE_POSITION_TOLERANCE_M
        ):
            violations.append(
                {
                    "frame": float(anchor["frame"]),
                    "reason": "hard_bounce_anchor_substituted",
                    "flight_index": index,
                    "frame_delta": frame_delta,
                    "position_delta_m": position_delta,
                }
            )
            continue
        satisfied_flights.append(index)
    violating_flights = sorted(
        {int(row["flight_index"]) for row in violations if row.get("flight_index") is not None}
    )
    return {
        "required": len(required),
        "satisfied": len(required) - len(violations),
        "all_satisfied": not violations,
        "satisfied_flights": sorted(set(satisfied_flights)),
        "violating_flights": violating_flights,
        "violations": violations,
    }


def restrict_fits_to_attempts(raw_fits: dict, flight_attempts: list[dict]) -> list[int]:
    """Drop recovery artifacts that do not belong to the scored flight-attempt universe."""
    attempted = {int(row["flight_index"]) for row in flight_attempts}
    dropped = sorted(int(index) for index in raw_fits if int(index) not in attempted)
    for index in dropped:
        raw_fits.pop(index, None)
    return dropped


def joint_rally_refinement_report(
    boundary_events: list[dict],
    selected_contacts: list[dict],
    topology_report: dict,
    bounce_hypotheses: list[dict] | None = None,
) -> dict:
    """Describe S6 event repairs without mutating the upstream S5 artifact."""
    inserted = [dict(row) for row in topology_report.get("inserted", [])]
    omitted = [dict(row) for row in topology_report.get("omitted", [])]
    retimed = [dict(row) for row in topology_report.get("retimed", [])]
    baseline_bounces = [
        float(row["frame"]) for row in boundary_events if row["event_type"] == "bounce"
    ]
    inserted_bounces = []
    for row in bounce_hypotheses or []:
        if row.get("source") not in {
            "s5_leaky_bounce_hypothesis",
            "probabilistic_event_hypothesis",
        }:
            continue
        if float(row.get("probability", 0.0)) < S6_BOUNCE_MIN_PROBABILITY:
            continue
        if float(row.get("physics_timing_delta_frames", math.inf)) > S6_BOUNCE_MAX_TIMING_FRAMES:
            continue
        if float(row.get("physics_court_delta_m", math.inf)) > S6_BOUNCE_MAX_COURT_DELTA_M:
            continue
        if any(abs(float(row["frame"]) - frame) <= 3.0 for frame in baseline_bounces):
            continue
        inserted_bounces.append(
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in row.items()
            }
        )
    selected_events = [
        {
            "event_type": row["event_type"],
            "frame": float(row["frame"]),
            "probability": row.get("probability"),
            "origin": row.get("origin", "automatic_emission"),
        }
        for row in boundary_events
        if row["event_type"] != "contact"
    ]
    selected_events.extend(
        {
            "event_type": "contact",
            "frame": float(contact["frame"]),
            "probability": contact.get("row", {}).get("probability"),
            "origin": contact.get("row", {}).get(
                "origin",
                contact.get("source", "automatic_event_boundary"),
            ),
            "side": contact.get("side"),
        }
        for contact in selected_contacts
        if not contact.get("terminal")
    )
    selected_events.extend(
        {
            "event_type": "bounce",
            "frame": float(row["frame"]),
            "probability": row.get("probability"),
            "origin": "s6_physics_certified_bounce_insertion",
        }
        for row in inserted_bounces
    )
    selected_events.sort(key=lambda row: (row["frame"], row["event_type"]))
    changed = bool(inserted or omitted or retimed or inserted_bounces)
    modified_event_types = []
    if inserted or omitted or retimed:
        modified_event_types.append("contact")
    if inserted_bounces:
        modified_event_types.append("bounce")
    return {
        "schema": "s6_joint_rally_refinement_v1",
        "enabled": bool(topology_report.get("enabled")),
        "status": "selected_repair" if changed else "baseline_preserved",
        "selected_branch": topology_report.get("selected"),
        "selection_reason": topology_report.get("selection_reason"),
        "selection_margin": topology_report.get("margin"),
        "scoring_stage": topology_report.get("scoring_stage"),
        "upstream_artifacts_mutated": False,
        "input_event_count": len(boundary_events),
        "selected_event_count": len(selected_events),
        "modified_event_types": modified_event_types,
        "operations": {
            "inserted_contacts": inserted,
            "inserted_bounces": inserted_bounces,
            "omitted_contacts": omitted,
            "retimed_contacts": retimed,
        },
        "selected_events": selected_events,
    }


def topology_flight_key(start: dict, end: dict) -> tuple:
    """Identify a flight whose fitted result is reusable across topology branches."""
    return (
        round(float(start["frame"]), 6),
        round(float(end["frame"]), 6),
        start.get("side"),
        end.get("side"),
        start.get("phase"),
        end.get("phase"),
        start.get("span"),
        end.get("span"),
        bool(end.get("terminal")),
    )


def spatial_fit_scope(
    camera_scope_points: dict[str, dict],
    match_id: str,
    clip: str,
    start_frame: float,
    end_frame: float,
) -> dict:
    """Return the automatic spatial-scope decision for one proposed flight."""
    return window_scope(
        camera_scope_points.get(f"{match_id}/{clip}"),
        start_frame,
        end_frame,
    )


class UnsupportedCameraModelError(ValueError):
    """The pinhole reconstruction path cannot consume this lens model faithfully."""


class PointCamera:
    def __init__(
        self,
        match_dir: Path,
        clip: str,
        artifact_name: str = "camera_P_per_frame_v1.npz",
    ):
        projection = np.load(match_dir / artifact_name, allow_pickle=True)
        selected = projection["clips"] == clip
        selected_indices = np.flatnonzero(selected)
        # This fitter passes bare 3x4 matrices to its ray/project helpers. Until
        # native distortion is propagated through every residual and witness,
        # accepting nonzero lens coefficients would silently fit the wrong rays.
        radial_fields = {"k1", "dist_center"} & set(projection.files)
        if radial_fields:
            n = len(projection["frames"])
            if (
                radial_fields != {"k1", "dist_center"}
                or projection["k1"].shape != (n,)
                or projection["dist_center"].shape != (n, 2)
                or not np.isfinite(projection["k1"][selected]).all()
                or not np.isfinite(projection["dist_center"][selected]).all()
                or np.any(projection["k1"][selected] != 0)
            ):
                projection.close()
                raise UnsupportedCameraModelError(
                    "pinhole S6 requires absent or explicit zero radial distortion"
                )
        if "k2" in projection.files and (
            projection["k2"].shape != (len(projection["frames"]),)
            or not np.isfinite(projection["k2"][selected]).all()
            or np.any(projection["k2"][selected] != 0)
        ):
            projection.close()
            raise UnsupportedCameraModelError(
                "pinhole S6 does not support second-order radial distortion"
            )
        self.projections = dict(
            zip(
                projection["frames"][selected].astype(int),
                projection["P"][selected],
            )
        )
        defaults = {
            "reliable": False,
            "source": "legacy_missing_provenance",
            "ground_residual_px": float("nan"),
            "net_residual_px": float("nan"),
            "confidence": 0.0,
            "fallback_ancestry": "[]",
            "frame_scope": "unknown",
            "reference_frame": "",
        }
        self.quality = {}
        for index in selected_indices:
            frame = int(projection["frames"][index])
            self.quality[frame] = {
                field: (
                    projection[field][index].item()
                    if isinstance(projection[field][index], np.generic)
                    else projection[field][index]
                )
                if field in projection.files
                else default
                for field, default in defaults.items()
            }
        if not self.projections:
            raise ValueError(f"no camera projections for {clip}")
        homographies = np.load(match_dir / "court_H_per_point.npz")
        point = int(clip.removeprefix("pt"))
        index = list(homographies["pts"].astype(int)).index(point)
        self.homography = homographies["H"][index]
        self.homographies: dict[int, np.ndarray] = {}
        frame_homography_path = match_dir / "court_H_per_frame_v1.npz"
        if frame_homography_path.is_file():
            frame_homographies = np.load(frame_homography_path, allow_pickle=True)
            if {"clips", "frames", "H"}.issubset(frame_homographies.files):
                frame_selected = frame_homographies["clips"] == clip
                self.homographies = dict(
                    zip(
                        frame_homographies["frames"][frame_selected].astype(int),
                        frame_homographies["H"][frame_selected],
                    )
                )

    def p_at(self, frame: float) -> np.ndarray:
        rounded = int(round(frame))
        if rounded in self.projections:
            return self.projections[rounded]
        nearest = min(self.projections, key=lambda value: abs(value - rounded))
        return self.projections[nearest]

    def h_at(self, frame: float) -> np.ndarray:
        """Return registered frame geometry, falling back to the point solve."""
        if not self.homographies:
            return self.homography
        rounded = int(round(frame))
        if rounded in self.homographies:
            return self.homographies[rounded]
        nearest = min(self.homographies, key=lambda value: abs(value - rounded))
        return self.homographies[nearest]

    def quality_at(self, frame: float) -> dict:
        rounded = int(round(frame))
        if rounded in self.quality:
            return self.quality[rounded]
        nearest = min(self.quality, key=lambda value: abs(value - rounded))
        return self.quality[nearest]

    @staticmethod
    def _effective_reliable(row: dict) -> bool:
        # Recovery belongs to the camera producer, where registration evidence exists.
        # Anchor residuals and a source string cannot overrule its rejected frame here.
        return bool(row["reliable"])

    def quality_summary(self, active_spans: list[list[float]]) -> dict:
        active_frames = [
            frame
            for frame in sorted(self.quality)
            if any(start <= frame <= end for start, end in active_spans)
        ]
        if not active_frames:
            return {
                "accepted": False,
                "reason": "missing_active_scope",
                "frames": 0,
                "reliable_fraction": 0.0,
                "sources": [],
                "fallback_ancestry": [],
                "frame_scopes": [],
            }
        rows = [self.quality[frame] for frame in active_frames]
        raw_reliable_fraction = float(np.mean([bool(row["reliable"]) for row in rows]))
        effective_reliability = [self._effective_reliable(row) for row in rows]
        reliable_fraction = float(np.mean(effective_reliability))
        ground = np.asarray([float(row["ground_residual_px"]) for row in rows])
        net = np.asarray([float(row["net_residual_px"]) for row in rows])
        finite_ground = ground[np.isfinite(ground)]
        finite_net = net[np.isfinite(net)]
        sources = sorted({str(row["source"]) for row in rows})
        ancestry = sorted(
            {
                str(row["fallback_ancestry"])
                for row in rows
                if str(row["fallback_ancestry"]) not in {"", "[]"}
            }
        )
        scopes = sorted({str(row["frame_scope"]) for row in rows})
        accepted = (
            reliable_fraction >= 0.95
            and len(finite_ground) == len(rows)
            and float(np.max(finite_ground)) <= 4.0
            and len(finite_net) == len(rows)
            and float(np.max(finite_net)) <= 8.0
            and not ancestry
        )
        return {
            "accepted": bool(accepted),
            "reason": None if accepted else "camera_calibration_unreliable",
            "frames": len(rows),
            "reliable_fraction": reliable_fraction,
            "raw_reliable_fraction": raw_reliable_fraction,
            "lineage_recovered_frames": sum(
                effective and not bool(row["reliable"])
                for effective, row in zip(effective_reliability, rows, strict=True)
            ),
            "max_ground_residual_px": (
                float(np.max(finite_ground)) if len(finite_ground) else None
            ),
            "max_net_residual_px": float(np.max(finite_net)) if len(finite_net) else None,
            "minimum_confidence": float(min(float(row["confidence"]) for row in rows)),
            "sources": sources,
            "fallback_ancestry": ancestry,
            "frame_scopes": scopes,
            "reference_frames": sorted(
                {str(row["reference_frame"]) for row in rows if row["reference_frame"]}
            ),
        }


def frame_number(value: str) -> int:
    return int(Path(value).stem.rsplit("_", 1)[-1])


def load_track(match_dir: Path, clip: str) -> dict[int, np.ndarray]:
    """Native-1080 track points, scaled by the sidecar rather than by a fixed factor."""
    path = match_dir / TRACK_NAME
    output = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        read_point, _, _ = res.native_point_reader(path, reader.fieldnames or ())
        for row in reader:
            if row["clip"] != clip:
                continue
            output[frame_number(row["frame"])] = np.array(read_point(row))
    return output


def load_track_weights(match_dir: Path, clip: str) -> dict[int, float]:
    output = {}
    for row in csv.DictReader((match_dir / TRACK_NAME).open()):
        if row["clip"] != clip:
            continue
        output[frame_number(row["frame"])] = tracking_observation_weight(
            float(row["score"]),
            row.get("sources", ""),
            interpolated="interpolated" in row.get("sources", ""),
        )
    return output


def load_players(match_dir: Path, clip: str) -> tuple[dict, dict]:
    paths = sorted(glob.glob(str(match_dir / "player_boxes_*_native_sided_v1.csv")))
    if not paths:
        return {"near": {}, "far": {}}, {}
    players: dict[str, dict[int, np.ndarray]] = {"near": {}, "far": {}}
    boxes: dict[int, list[dict]] = defaultdict(list)
    path = Path(paths[0])
    with path.open() as handle:
        reader = csv.DictReader(handle)
        read_box, _, _ = res.native_box_reader(path, reader.fieldnames or ())
        for row in reader:
            if row["clip"] != clip:
                continue
            frame = frame_number(row["frame"])
            native = read_box(row)
            boxes[frame].append(
                {**row, **dict(zip(res.PLAYER_NATIVE_BOX_COLUMNS, native, strict=True))}
            )
            players[row["side"]][frame] = np.array([float(row["court_x"]), float(row["court_y"])])
    return players, boxes


def load_pose_rows(
    match_dir: Path,
    clip: str,
    artifact_name: str | None = None,
) -> dict[str, dict[int, dict]]:
    if not artifact_name:
        return {"near": {}, "far": {}}
    path = match_dir / artifact_name
    output: dict[str, dict[int, dict]] = {"near": {}, "far": {}}
    if not path.exists():
        return output
    for row in load_native_pose_rows(os.fspath(path)):
        if row["clip"] != clip or row.get("side") not in output:
            continue
        output[row["side"]][frame_number(row["frame"])] = row
    return output


@lru_cache(maxsize=128)
def _load_physical_motion_file(path: str) -> dict[str, dict[str, dict[int, dict]]]:
    output: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: {"near": {}, "far": {}})
    artifact = Path(path)
    if not artifact.is_file():
        return {}
    with artifact.open() as handle:
        for line in handle:
            row = json.loads(line)
            side = row.get("side")
            if side in {"near", "far"}:
                output[row["clip"]][side][int(row["frame"])] = row
    return dict(output)


def load_physical_motion(
    match_dir: Path,
    clip: str,
    artifact_name: str | None,
) -> dict[str, dict[int, dict]]:
    if not artifact_name:
        return {"near": {}, "far": {}}
    return _load_physical_motion_file(str(match_dir / artifact_name)).get(
        clip, {"near": {}, "far": {}}
    )


def optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_event_boundaries(
    path: Path, *, include_dead_time_emissions: bool = False
) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text())
    rows = payload.get("emissions", []) if isinstance(payload, dict) else payload
    rows = select_consumer_emissions(rows, include_dead_time=include_dead_time_emissions)
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("event_type") not in {"contact", "bounce", "net_hit", "point_end"}:
            continue
        # Event inference emits held-point hypotheses with explicit metadata.
        # Physical-claim reconstruction remains a precision consumer and filters
        # those rows without deleting them from the event artifact.
        if row.get("point_gate_verdict") == "hold" or row.get("gate_held") is True:
            continue
        clip = str(row["clip"])
        clip_key = clip if "__" in clip else f"{row['match_id']}__{clip}"
        event = {
            "clip": clip_key,
            "event_type": row["event_type"],
            "frame": float(row["frame"]),
            "origin": "automatic_emission",
            "probability": optional_float(
                row.get(
                    "score",
                    row.get("emission_probability", row.get("probability")),
                )
            ),
        }
        if row["event_type"] == "point_end" and isinstance(row.get("point_end"), dict):
            event["point_end"] = dict(row["point_end"])
        location = row.get("location")
        if isinstance(location, dict):
            event["location"] = dict(location)
            image_x = optional_float(location.get("image_x"))
            image_y = optional_float(location.get("image_y"))
            if image_x is not None and image_y is not None:
                event["x1080"] = image_x
                event["y1080"] = image_y
                event["observation_frame"] = float(row["frame"])
        output[clip_key].append(event)
    for values in output.values():
        values.sort(key=lambda row: row["frame"])
    return output


def load_event_hypotheses(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("hypotheses", payload) if isinstance(payload, dict) else payload
    rows = select_consumer_emissions(rows)
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("event_type") not in {"contact", "bounce", "net_hit"}:
            continue
        clip = str(row["clip"])
        clip_key = clip if "__" in clip else f"{row['match_id']}__{clip}"
        output[clip_key].append({**row, "clip": clip_key, "frame": float(row["frame"])})
    for values in output.values():
        values.sort(key=lambda row: (row["frame"], row["event_type"]))
    return output


def load_automatic_point_universe(
    audit_root: Path,
    manifest_path: Path,
) -> list[str]:
    """Return every automatically segmented point, including zero-event points."""
    active_paths = active_play_content_paths(audit_root)
    if not active_paths:
        raise FileNotFoundError(f"no active_play_v*.json under {audit_root}")
    active = json.loads(active_paths[-1].read_text())
    match_ids = {row["id"] for row in json.loads(manifest_path.read_text())["matches"]}
    return sorted(key.replace("/", "__", 1) for key in active if key.split("/", 1)[0] in match_ids)


def active_play_content_paths(audit_root: Path) -> list[Path]:
    """Return active-play content artifacts without provenance sidecars."""
    return sorted(
        path
        for path in audit_root.glob("active_play_v*.json")
        if not path.name.endswith(".provenance.json")
    )


def pose_slice_probability(
    pose_rows: dict[str, dict[int, dict]],
    *,
    side: str,
    frame: float,
    fps: float,
    ball_pixel: np.ndarray | None,
) -> float | None:
    """Weak high-to-low swing witness from the racket-side wrist path.

    Image y increases downward, so a positive pre/post wrist delta supports slice.
    This remains a soft branch likelihood because pose lacks racket identity and
    two-handed backhands can make the nearest-wrist assignment ambiguous.
    """
    if ball_pixel is None or side not in pose_rows or not pose_rows[side]:
        return None
    rows = pose_rows[side]
    center_frame = min(rows, key=lambda value: abs(value - frame))
    if abs(center_frame - frame) > 0.12 * fps:
        return None
    center = rows[center_frame]

    def wrist(row: dict, hand: str) -> np.ndarray | None:
        confidence = optional_float(row.get(f"{hand}_wrist_confidence"))
        x = optional_float(row.get(f"{hand}_wrist_x"))
        y = optional_float(row.get(f"{hand}_wrist_y"))
        if confidence is None or confidence < 0.20 or x is None or y is None:
            return None
        return np.array([x, y], float)

    hands = []
    for hand in ("left", "right"):
        point = wrist(center, hand)
        if point is not None:
            hands.append((float(np.linalg.norm(point - ball_pixel)), hand))
    if not hands:
        return None
    _, hand = min(hands)
    offset = max(1, int(round(0.12 * fps)))

    def nearest_row(target: int) -> dict | None:
        candidate = min(rows, key=lambda value: abs(value - target))
        return rows[candidate] if abs(candidate - target) <= 2 else None

    before = wrist(nearest_row(center_frame - offset) or {}, hand)
    after = wrist(nearest_row(center_frame + offset) or {}, hand)
    y0 = optional_float(center.get("y0"))
    y1 = optional_float(center.get("y1"))
    if before is None or after is None or y0 is None or y1 is None:
        return None
    height = max(y1 - y0, 1.0)
    downward_motion = float((after[1] - before[1]) / height)
    return float(1.0 / (1.0 + math.exp(-8.0 * downward_motion)))


def contact_pose_witness(
    pose_rows: dict[str, dict[int, dict]],
    physical_motion: dict[str, dict[int, dict]] | None = None,
    *,
    contact: dict,
    ball: dict[int, np.ndarray],
    players: dict[str, dict[int, np.ndarray]],
    camera,
    fps: float,
) -> dict | None:
    """Build independent 2D body-scale and wrist witnesses for racket contact."""
    side = contact.get("side")
    physical_motion = physical_motion or {"near": {}, "far": {}}
    if side not in {"near", "far"} or not (pose_rows.get(side) or physical_motion.get(side)):
        return None
    observed = nearest_track_point(ball, float(contact["frame"]), maximum_delta=3)
    player = nearest(players.get(side, {}), float(contact["frame"]))
    if observed is None or player is None:
        return None
    physical_candidates = []
    physical_rows = physical_motion.get(side, {})
    if physical_rows:
        physical_frame = min(physical_rows, key=lambda value: abs(value - float(contact["frame"])))
        document = physical_rows[physical_frame]
        if abs(physical_frame - float(contact["frame"])) <= max(2.0, 0.10 * fps):
            branches = document.get("branches", [])
            for branch_index, branch in enumerate(branches):
                diagnostics = branch.get("diagnostics", {})
                projection_quality = branch.get("body_prior_projection_quality") or {}
                physical_quality = float(diagnostics.get("quality") or 0.0)
                default_reliability = 0.10 if branch.get("body_prior_backend") else 1.0
                projection_reliability = float(
                    projection_quality.get("reliability", default_reliability)
                )
                branch_confidence = float(
                    np.clip(physical_quality / 0.18, 0.0, 1.0) * projection_reliability
                )
                if (
                    not document.get("camera", {}).get("reliable", True)
                    or branch_confidence < 0.10
                    or float(diagnostics.get("bone_rms_m") or 1.0) > 0.16
                    or float(diagnostics.get("root_error_m") or 1.0) > 0.70
                    or float(diagnostics.get("minimum_z_m", -1.0)) < -0.15
                ):
                    continue
                for racket in branch.get("racket_hypotheses", []):
                    head = np.asarray(racket["head_center_xyz"], dtype=float)
                    projected = project_one(camera.p_at(float(contact["frame"])), head)
                    distance = float(np.linalg.norm(projected - observed))
                    physical_candidates.append(
                        {
                            **racket,
                            "body_prior_backend": branch.get("body_prior_backend"),
                            "physical_branch_index": branch_index,
                            "physical_branch_confidence": branch_confidence,
                            "projected_head_pixel": projected.tolist(),
                            "ball_head_distance_px": distance,
                            "evidence_confidence": float(
                                branch_confidence * math.exp(-distance / 80.0)
                            ),
                        }
                    )
            physical_candidates.sort(
                key=lambda row: (
                    -row["evidence_confidence"],
                    row["ball_head_distance_px"],
                )
            )
    diverse_physical_candidates = []
    represented = set()
    for candidate in physical_candidates:
        identity = (candidate.get("body_prior_backend"), candidate.get("hand"))
        if identity in represented:
            continue
        represented.add(identity)
        diverse_physical_candidates.append(candidate)
    for candidate in physical_candidates:
        if len(diverse_physical_candidates) >= 6:
            break
        if candidate not in diverse_physical_candidates:
            diverse_physical_candidates.append(candidate)
    physical_candidates = diverse_physical_candidates[:6]
    rows = pose_rows.get(side, {})
    if not rows:
        if not physical_candidates:
            return None
        best = physical_candidates[0]
        return {
            "available": True,
            "frame": int(physical_frame),
            "confidence": round(float(best["evidence_confidence"]), 4),
            "racket_head_xyz": best["head_center_xyz"],
            "racket_head_hypotheses": physical_candidates,
            "source": "court_registered_physical_racket_branches",
        }
    frame = min(rows, key=lambda value: abs(value - float(contact["frame"])))
    if abs(frame - float(contact["frame"])) > max(2.0, 0.10 * fps):
        return None
    row = rows[frame]
    x0 = optional_float(row.get("x0"))
    x1 = optional_float(row.get("x1"))
    y0 = optional_float(row.get("y0"))
    y1 = optional_float(row.get("y1"))
    box_confidence = float(np.clip(optional_float(row.get("conf")) or 0.0, 0.0, 1.0))
    box_height = max(
        (y1 or 0.0) - (y0 or 0.0),
        1.0,
    )
    box_width = max((x1 or 0.0) - (x0 or 0.0), 1.0)
    body_scale = {}
    if y0 is not None and y1 is not None and y1 > y0:
        vertical_fraction = float(np.clip((y1 - observed[1]) / box_height, -0.25, 1.5))
        horizontal_excess = 0.0
        if x0 is not None and x1 is not None:
            horizontal_excess = max(0.0, x0 - observed[0], observed[0] - x1) / box_width
        apparent_confidence = float(box_confidence * math.exp(-2.0 * horizontal_excess))
        body_scale = {
            "apparent_ball_height_m": round(
                vertical_fraction * APPARENT_PLAYER_HEIGHT_M,
                4,
            ),
            "apparent_height_confidence": round(apparent_confidence, 4),
            "ball_vertical_box_fraction": round(vertical_fraction, 4),
            "ball_horizontal_box_excess": round(horizontal_excess, 4),
            "player_box": [x0, y0, x1, y1],
        }
    candidates = []
    projection = camera.p_at(float(contact["frame"]))
    for hand in ("left", "right"):
        confidence = optional_float(row.get(f"{hand}_wrist_confidence"))
        x = optional_float(row.get(f"{hand}_wrist_x"))
        y = optional_float(row.get(f"{hand}_wrist_y"))
        if confidence is None or confidence < 0.20 or x is None or y is None:
            continue
        pixel = np.asarray([x, y], dtype=float)
        normalized_distance = float(np.linalg.norm(pixel - observed) / box_height)
        if normalized_distance > 1.0:
            continue
        wrist_xyz = ray_at_y(projection, pixel, float(player[1]))
        if wrist_xyz is None or not np.all(np.isfinite(wrist_xyz)):
            continue
        if not -0.05 <= wrist_xyz[2] <= 3.25:
            continue
        witness_confidence = float(confidence * math.exp(-1.5 * normalized_distance))
        candidates.append(
            (
                -witness_confidence,
                normalized_distance,
                {
                    "available": True,
                    **body_scale,
                    "hand": hand,
                    "frame": int(frame),
                    "confidence": round(witness_confidence, 4),
                    "keypoint_confidence": round(float(confidence), 4),
                    "ball_wrist_distance_box_heights": round(normalized_distance, 4),
                    "wrist_pixel": pixel.tolist(),
                    "wrist_xyz": wrist_xyz.tolist(),
                    "source": "tracked_temporal_pose_wrist",
                },
            )
        )
    if candidates:
        result = min(candidates, key=lambda value: value[:2])[2]
        if physical_candidates:
            best = physical_candidates[0]
            result.update(
                {
                    "racket_head_xyz": best["head_center_xyz"],
                    "racket_head_hypotheses": physical_candidates,
                    "physical_racket_pixel_error": best["ball_head_distance_px"],
                    "physical_racket_confidence": best["evidence_confidence"],
                    "physical_racket_source": "court_registered_physical_racket_branches",
                }
            )
            result["confidence"] = round(
                max(float(result["confidence"]), float(best["evidence_confidence"])), 4
            )
        return result
    if body_scale:
        result = {
            "available": True,
            **body_scale,
            "frame": int(frame),
            "confidence": round(float(body_scale["apparent_height_confidence"]), 4),
            "source": "tracked_temporal_pose_box_height",
        }
        if physical_candidates:
            result.update(
                {
                    "racket_head_xyz": physical_candidates[0]["head_center_xyz"],
                    "racket_head_hypotheses": physical_candidates,
                }
            )
        return result
    return None


def bounded_contact_ray_search(candidates: list[dict], diagnostic: dict) -> list[dict]:
    """Keep exact-ray search broad only when body-scale evidence is weak."""
    geometry_candidates = sorted(
        candidates,
        key=lambda row: float(row.get("geometry_prior_score", row["prior_score"])),
    )[:4]
    if float(diagnostic.get("apparent_height_confidence") or 0.0) >= 0.45:
        search = candidates[:4] + geometry_candidates
        return list({tuple(np.round(row["point"], 6)): row for row in search}.values())
    ordered_by_depth = sorted(candidates, key=lambda row: float(row["point"][1]))
    stride = max(1, len(ordered_by_depth) // 9)
    search = candidates[:6] + geometry_candidates + ordered_by_depth[::stride][:10]
    unique_search = {}
    for candidate in search:
        unique_search[tuple(np.round(candidate["point"], 6))] = candidate
    return list(unique_search.values())


def nearest_track_point(
    track: dict[int, np.ndarray],
    frame: float,
    maximum_delta: int = 3,
) -> np.ndarray | None:
    if not track:
        return None
    nearest = min(track, key=lambda value: abs(value - frame))
    return track[nearest] if abs(nearest - frame) <= maximum_delta else None


def out_of_frame_gap(
    before: np.ndarray,
    after: np.ndarray,
    *,
    width: float = 1920.0,
    height: float = 1080.0,
    margin: float = OUT_OF_FRAME_EDGE_PX,
) -> str | None:
    """Classify a missing interval whose two visible ends meet an image edge."""
    first = np.asarray(before, float)
    second = np.asarray(after, float)
    if first[1] <= margin and second[1] <= margin:
        return "top"
    if first[0] <= margin and second[0] <= margin:
        return "left"
    if first[0] >= width - margin and second[0] >= width - margin:
        return "right"
    if first[1] >= height - margin and second[1] >= height - margin:
        return "bottom"
    return None


def append_terminal_contacts(
    contacts: list[dict],
    track: dict[int, np.ndarray],
    active_spans: list[list[float]],
    fps: float,
    observation_weights: dict[int, float] | None = None,
    boundary_events: list[dict] | None = None,
    bridge_out_of_frame_gaps: bool = False,
    forward_ground_completion: bool = False,
) -> list[dict]:
    """Close rallies at point_end, or at the observed active-span fallback.

    A bounce is not itself a generic cutoff.  In particular, an in-bounds first
    bounce keeps the terminal flight alive until its second-bounce ``point_end``.
    """
    output = [dict(contact) for contact in contacts]
    by_span: dict[int, list[dict]] = defaultdict(list)
    for contact in output:
        by_span[int(contact["span"])].append(contact)
    for span_index, span_contacts in by_span.items():
        last = max(span_contacts, key=lambda row: row["frame"])
        _, raw_end = active_spans[span_index]
        local_events = [
            row for row in (boundary_events or []) if last["frame"] < float(row["frame"]) <= raw_end
        ]
        point_ends = [row for row in local_events if terminal_boundary_kind(row) is not None]
        first_point_end = (
            min(point_ends, key=lambda row: float(row["frame"])) if point_ends else None
        )
        cutoff = float(first_point_end["frame"]) if first_point_end is not None else float(raw_end)
        bounces = sorted(
            (
                row
                for row in local_events
                if row.get("event_type") == "bounce" and float(row["frame"]) <= cutoff
            ),
            key=lambda row: float(row["frame"]),
        )
        cutoff_source = "point_end" if first_point_end is not None else "active_span_end"
        observed = sorted(frame for frame in track if last["frame"] < frame <= cutoff)
        if len(observed) < MIN_TERMINAL_OBSERVATIONS:
            continue
        runs = []
        contiguous = []
        previous = last["frame"]
        for frame in observed:
            if frame - previous > MAX_TERMINAL_TRACK_GAP_SECONDS * fps:
                if contiguous:
                    runs.append(contiguous)
                contiguous = []
            contiguous.append(frame)
            previous = frame
        if contiguous:
            runs.append(contiguous)
        maximum_end = cutoff
        weights = observation_weights or {}
        reliable_runs = []
        previous_end = last["frame"]
        for run in runs:
            raw_run_end = run[-1]
            run = [frame for frame in run if frame <= maximum_end]
            if len(run) < MIN_TERMINAL_OBSERVATIONS:
                previous_end = raw_run_end
                continue
            gap_seconds = (run[0] - previous_end) / fps
            median_weight = float(np.median([weights.get(frame, 1.0) for frame in run]))
            gap_edge = (
                out_of_frame_gap(track[int(previous_end)], track[int(run[0])])
                if bridge_out_of_frame_gaps and int(previous_end) in track and int(run[0]) in track
                else None
            )
            if reliable_runs and (
                (gap_seconds > MAX_TERMINAL_REACQUISITION_GAP_SECONDS and gap_edge is None)
                or median_weight < MIN_TERMINAL_REACQUISITION_WEIGHT
            ):
                break
            reliable_runs.append(
                {
                    "frames": run,
                    "gap_seconds": float(gap_seconds),
                    "median_weight": median_weight,
                    "out_of_frame_edge": gap_edge,
                }
            )
            previous_end = run[-1]
        if not reliable_runs:
            continue
        end_frame = (
            float(cutoff) if first_point_end is not None else float(reliable_runs[-1]["frames"][-1])
        )
        side = "far" if last["side"] == "near" else "near"
        output.append(
            {
                "frame": end_frame,
                "side": side,
                "phase": "terminal",
                "span": span_index,
                "source": (
                    "active_track_terminal_reacquired"
                    if len(reliable_runs) > 1
                    else "active_track_terminal"
                ),
                "terminal": True,
                "terminal_cutoff": {
                    "source": cutoff_source,
                    "frame": float(cutoff),
                    "point_end": first_point_end,
                    "termination_kind": (
                        (first_point_end.get("point_end") or {}).get("termination_kind")
                        if first_point_end is not None
                        else None
                    ),
                    "bounce_frames": [float(row["frame"]) for row in bounces],
                    "forward_completion_window": (
                        completion_window(first_point_end, raw_end)
                        if forward_ground_completion and first_point_end is not None
                        else None
                    ),
                },
                "reacquisition": {
                    "runs": len(reliable_runs),
                    "gaps_seconds": [row["gap_seconds"] for row in reliable_runs[1:]],
                    "median_weights": [row["median_weight"] for row in reliable_runs],
                    "out_of_frame_edges": [row["out_of_frame_edge"] for row in reliable_runs[1:]],
                },
                "row": {},
            }
        )
    return sorted(output, key=lambda row: row["frame"])


def terminal_residual_growth_frame(fit) -> float | None:
    """Return the first frame of a sustained post-flight residual tail."""
    frames = np.asarray(getattr(fit, "obs_frames", []), float)
    errors = np.asarray(getattr(fit, "_observation_errors_px", []), float)
    if len(frames) != len(errors) or len(errors) < 12:
        return None
    for index in range(8, len(errors) - 3):
        prefix = errors[:index]
        suffix = errors[index:]
        threshold = max(12.0, 3.0 * float(np.median(prefix)))
        if (
            float(np.median(suffix)) > threshold
            and float(np.mean(suffix > threshold)) >= 0.75
            and bool(np.all(suffix[:4] > threshold))
            and float(np.polyfit(np.arange(len(suffix)), suffix, 1)[0]) >= 0.0
        ):
            return float(frames[index - 1])
    return None


def _image_distance_to_box(point: np.ndarray, row: dict) -> tuple[float, float]:
    """``row`` carries the native box columns that ``load_players`` resolved from the sidecar."""
    x0, y0, x1, y1 = (float(row[key]) for key in res.PLAYER_NATIVE_BOX_COLUMNS)
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    return math.hypot(dx, dy), math.hypot(x1 - x0, y1 - y0)


def contact_side_evidence(
    frame: float,
    track: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
) -> dict:
    ball_window = max(1, int(math.ceil(CONTACT_SIDE_BALL_WINDOW_SECONDS * fps)))
    ball_frames = [candidate for candidate in track if abs(candidate - frame) <= ball_window]
    if not ball_frames:
        return {
            "side": "unknown",
            "confidence": 0.0,
            "reason": "no_ball_observation",
            "ball_frame": None,
        }
    ball_frame = min(ball_frames, key=lambda candidate: abs(candidate - frame))
    point = track[ball_frame]
    box_window = max(1, int(math.ceil(CONTACT_SIDE_BOX_WINDOW_SECONDS * fps)))
    nearest_by_side = {}
    for candidate_frame, rows in boxes.items():
        if abs(candidate_frame - ball_frame) > box_window:
            continue
        for row in rows:
            side = row.get("side")
            if side not in {"near", "far"}:
                continue
            candidate = (abs(candidate_frame - ball_frame), candidate_frame, row)
            if side not in nearest_by_side or candidate[:2] < nearest_by_side[side][:2]:
                nearest_by_side[side] = candidate
    if len(nearest_by_side) < 2:
        return {
            "side": "unknown",
            "confidence": 0.0,
            "reason": "missing_player_side",
            "ball_frame": ball_frame,
        }
    distances = {}
    scales = {}
    evidence_frames = {}
    for side, (_, candidate_frame, row) in nearest_by_side.items():
        distances[side], scales[side] = _image_distance_to_box(point, row)
        evidence_frames[side] = candidate_frame
    selected = min(distances, key=distances.get)
    other = "far" if selected == "near" else "near"
    gap = distances[other] - distances[selected]
    ambiguity_scale = 0.40 * max(scales.values())
    confidence = gap / max(gap + ambiguity_scale, 1e-9)
    if confidence < CONTACT_SIDE_MIN_CONFIDENCE:
        return {
            "side": "unknown",
            "confidence": round(float(confidence), 4),
            "reason": "players_image_overlap",
            "ball_frame": ball_frame,
            "evidence_frames": evidence_frames,
            "distances_px": distances,
        }
    return {
        "side": selected,
        "confidence": round(float(confidence), 4),
        "reason": "temporal_image_reach",
        "ball_frame": ball_frame,
        "evidence_frames": evidence_frames,
        "distances_px": distances,
    }


def contact_side(
    frame: float,
    track: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
) -> str:
    return str(contact_side_evidence(frame, track, boxes, fps)["side"])


def robust_polynomial(
    frames: np.ndarray,
    values: np.ndarray,
    degree: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) < degree + 3:
        return values.copy(), np.ones(len(frames), dtype=bool)
    center = float(np.mean(frames))
    scale = max(float(np.ptp(frames)) / 2.0, 1.0)
    time = (frames - center) / scale
    keep = np.ones(len(frames), dtype=bool)
    predicted = values.copy()
    for _ in range(4):
        if int(keep.sum()) < degree + 3:
            break
        coefficients = np.polyfit(time[keep], values[keep], degree)
        predicted = np.polyval(coefficients, time)
        residual = np.abs(values - predicted)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median))) * 1.4826
        threshold = max(6.0, median + 4.0 * max(mad, 0.5))
        updated = residual <= threshold
        if np.array_equal(updated, keep):
            break
        keep = updated
    return predicted, keep


def smooth_track(
    track: dict[int, np.ndarray],
    active_spans: list[list[float]],
    fps: float,
    *,
    bridge_out_of_frame_gaps: bool = False,
) -> tuple[dict[int, np.ndarray], dict]:
    output = {frame: point.copy() for frame, point in track.items()}
    repaired = filled = segments = 0
    observed_total = expected_total = 0
    for raw_start, raw_end in active_spans:
        start, end = int(math.floor(raw_start)), int(math.ceil(raw_end))
        expected_total += max(0, end - start + 1)
        observed = np.array(
            [frame for frame in sorted(track) if start <= frame <= end],
            dtype=float,
        )
        observed_total += len(observed)
        if len(observed) < 5:
            continue
        points = np.stack([track[int(frame)] for frame in observed])
        healed = heal_track(
            observed,
            points,
            tolerance_px=28.0,
            fps=fps,
        )
        for frame in observed[~healed.kept].astype(int):
            output.pop(frame, None)
        for frame, point in zip(healed.frames.astype(int), healed.points):
            output[frame] = point
        repaired += healed.n_removed
        filled += healed.n_filled
        segments += 1
    gaps = []
    in_frame_gaps = []
    out_of_frame_gaps = []
    teleport_steps = teleport_count = 0
    for raw_start, raw_end in active_spans:
        frames = np.array(
            [frame for frame in sorted(output) if raw_start <= frame <= raw_end],
            dtype=float,
        )
        if len(frames) < 2:
            continue
        local_gaps = np.diff(frames)
        gaps.extend(local_gaps.tolist())
        points = np.stack([output[int(frame)] for frame in frames])
        for gap_index, gap in enumerate(local_gaps):
            edge = (
                out_of_frame_gap(points[gap_index], points[gap_index + 1])
                if bridge_out_of_frame_gaps and gap > 1.5
                else None
            )
            record = {
                "start_frame": int(frames[gap_index]),
                "end_frame": int(frames[gap_index + 1]),
                "duration_seconds": float(gap / fps),
                "edge": edge,
            }
            (out_of_frame_gaps if edge is not None else in_frame_gaps).append(record)
        displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
        adjacent = local_gaps <= 2.0
        teleport_steps += int(adjacent.sum())
        teleport_count += int(
            np.sum(adjacent & (displacement / np.maximum(local_gaps, 1.0) * fps > 4500.0))
        )
    repair_rate = repaired / observed_total if observed_total else 1.0
    return output, {
        "segments": segments,
        "observations": observed_total,
        "expected_frames": expected_total,
        "coverage": observed_total / expected_total if expected_total else 0.0,
        "repaired_observations": repaired,
        "repair_rate": repair_rate,
        "filled_frames": filled,
        "maximum_gap_seconds": (
            max(row["duration_seconds"] for row in in_frame_gaps) if in_frame_gaps else 0.0
        ),
        "maximum_observed_gap_seconds": max(gaps) / fps if gaps else None,
        "long_gaps": sum(row["duration_seconds"] > 0.16 for row in in_frame_gaps),
        "out_of_frame_gaps": out_of_frame_gaps,
        "post_heal_teleport_rate": (teleport_count / teleport_steps if teleport_steps else 1.0),
    }


def court_point(camera: PointCamera, pixel: np.ndarray, frame: float | None = None) -> np.ndarray:
    transformed = cv2.perspectiveTransform(
        np.float32([[[pixel[0], pixel[1]]]]),
        camera.h_at(0.0 if frame is None else frame),
    )[0, 0]
    return np.asarray(transformed, float)


def net_collision_anchor(
    camera: PointCamera,
    track: dict[int, np.ndarray],
    frame: float,
) -> tuple[np.ndarray | None, dict]:
    """Intersect the observed ball ray with the physical net plane."""
    candidates = sorted(
        (
            candidate
            for candidate in track
            if abs(float(candidate) - float(frame)) <= NET_COLLISION_SEARCH_FRAMES
        ),
        key=lambda candidate: abs(float(candidate) - float(frame)),
    )
    if not candidates:
        return None, {"reason": "net_hit_has_no_ball_observation"}
    rejected = []
    for candidate_frame in candidates:
        pixel = np.asarray(track[candidate_frame], float)
        xyz = ray_at_y(camera.p_at(candidate_frame), pixel, NET_COURT_Y_M)
        if xyz is None or not np.all(np.isfinite(xyz)):
            continue
        x = float(xyz[0])
        net_height = NET_CENTER_HEIGHT_M + (NET_POST_HEIGHT_M - NET_CENTER_HEIGHT_M) * abs(
            x - 10.97 / 2.0
        ) / (10.97 / 2.0)
        maximum_ball_center = net_height + R_BALL + NET_COLLISION_HEIGHT_TOLERANCE_M
        if -0.25 <= x <= 11.22 and 0.0 <= float(xyz[2]) <= maximum_ball_center:
            return xyz, {
                "frame": float(candidate_frame),
                "emitted_frame": float(frame),
                "retiming_frames": float(candidate_frame) - float(frame),
                "pixel": pixel.tolist(),
                "xyz": xyz.tolist(),
                "source": "observed_ball_ray_physical_net_mesh_local_search",
                "net_height_m": net_height,
            }
        rejected.append(
            {
                "frame": float(candidate_frame),
                "xyz": xyz.tolist(),
                "net_height_m": net_height,
                "maximum_ball_center_height_m": maximum_ball_center,
            }
        )
    return None, {
        "reason": "net_hit_outside_physical_net",
        "emitted_frame": float(frame),
        "searched_frames": [float(value) for value in candidates],
        "nearest_rejected": rejected[0] if rejected else None,
    }


def match_bounces(
    predicted: list[dict],
    truth: list[dict],
    fps: float,
    maximum_timing_seconds: float = 0.20,
) -> dict:
    pairs = []
    for predicted_index, proposal in enumerate(predicted):
        for truth_index, target in enumerate(truth):
            pairs.append(
                (
                    abs(float(proposal["frame"]) - float(target["frame"])),
                    float(np.linalg.norm(proposal["x"][:2] - target["court_xy"])),
                    predicted_index,
                    truth_index,
                )
            )
    used_predicted = set()
    used_truth = set()
    matches = []
    for delta, _, predicted_index, truth_index in sorted(pairs):
        if delta > maximum_timing_seconds * fps:
            continue
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
        matches.append(
            {
                "timing_frames": delta,
                "timing_ms": 1000.0 * delta / fps,
                "court_error_m": float(
                    np.linalg.norm(
                        predicted[predicted_index]["x"][:2] - truth[truth_index]["court_xy"]
                    )
                ),
            }
        )
    return {
        "matches": matches,
        "false_positives": len(predicted) - len(matches),
        "false_negatives": len(truth) - len(matches),
    }


def _physical_trajectory_rows(
    fit,
    fps: float,
    surface: str,
    *,
    start_frame: float | None,
    end_frame: float | None,
    observed_spins: np.ndarray,
) -> list[dict]:
    """Sample the fitted physical state over the complete native-frame span.

    Observation gaps remain gaps in the image evidence and in ``observations``.  The
    reconstruction output is nevertheless a physical position per source frame, so it
    must propagate through those gaps rather than linearly connecting sparse stored
    observations downstream.
    """
    if start_frame is None or end_frame is None:
        return [
            {
                "frame": int(frame),
                "xyz": position.tolist(),
                "velocity": velocity.tolist(),
                "spin": angular_velocity.tolist(),
            }
            for frame, position, velocity, angular_velocity in zip(
                fit.obs_frames,
                fit.xs_obs,
                fit.vs_obs,
                observed_spins,
                strict=True,
            )
        ]
    first = int(math.ceil(float(start_frame)))
    last = int(math.floor(float(end_frame)))
    if last < first:
        return []
    observation_frames = np.asarray(fit.obs_frames, float)
    rows = []
    for frame in range(first, last + 1):
        position, velocity = fit.state(float(frame), fps, surface)
        nearest = int(np.argmin(np.abs(observation_frames - frame)))
        rows.append(
            {
                "frame": frame,
                "xyz": np.asarray(position, float).tolist(),
                "velocity": np.asarray(velocity, float).tolist(),
                "spin": np.asarray(observed_spins[nearest], float).tolist(),
            }
        )
    return rows


def compact_fit(
    fit,
    fps: float,
    surface: str,
    *,
    start_frame: float | None = None,
    end_frame: float | None = None,
) -> dict:
    spin = spin_vector(fit.theta)
    interpretation = getattr(fit, "_spin_interpretation", None)
    component_rpm = spin_component_rpm(fit.theta)
    prior_centers = (
        np.asarray(
            interpretation.get(
                "spin_center_components_rpm",
                [interpretation["spin_center_rpm"], 0.0, 0.0],
            ),
            float,
        )
        if interpretation is not None
        else None
    )
    prior_sigmas = (
        np.asarray(
            interpretation.get(
                "spin_sigma_components_rpm",
                [interpretation["spin_sigma_rpm"], 1200.0, 800.0],
            ),
            float,
        )
        if interpretation is not None
        else None
    )
    spin_deviation_components_z = (
        (component_rpm - prior_centers) / prior_sigmas if interpretation is not None else None
    )
    spin_deviation_z = (
        float(np.max(np.abs(spin_deviation_components_z)))
        if spin_deviation_components_z is not None
        else None
    )
    spins = getattr(
        fit,
        "_ws_obs",
        np.repeat(spin[None, :], len(fit.obs_frames), axis=0),
    )
    state_lengths = {
        len(fit.obs_frames),
        len(fit.xs_obs),
        len(fit.vs_obs),
        len(spins),
    }

    if len(state_lengths) != 1:
        raise ValueError(
            "trajectory state length mismatch: "
            f"frames={len(fit.obs_frames)} positions={len(fit.xs_obs)} "
            f"velocities={len(fit.vs_obs)} spins={len(spins)}"
        )
    physical_diagnostics = getattr(fit, "_physical_diagnostics", {})
    net_split = getattr(fit, "_net_split", None)
    compact = {
        "rms_px": float(fit.rms_px),
        "weighted_rms_px": float(getattr(fit, "_weighted_rms_px", fit.rms_px)),
        "observations": int(fit.n_obs),
        "theta": fit.theta.tolist(),
        "speed_kmh": float(np.linalg.norm(fit.theta[3:6]) * 3.6),
        "minimum_height_m": physical_diagnostics.get(
            "minimum_height_m",
            float(np.min(fit.xs_obs[:, 2])),
        ),
        "fixed_bounce_continuity_m": physical_diagnostics.get("continuity_m"),
        "bounce_speed_ratio": physical_diagnostics.get("speed_ratio"),
        "bounce_energy_ratio": max(
            (bounce_energy_ratio(row) for row in fit.bounces),
            default=None,
        ),
        "bounce_impact_velocity_slack_mps": physical_diagnostics.get("impact_velocity_slack_mps"),
        "spin_rpm": float(np.linalg.norm(spin) * 60.0 / (2.0 * np.pi)),
        "signed_spin_rpm": float(component_rpm[0]),
        "spin_components_rpm": {
            "topspin": float(component_rpm[0]),
            "sidespin": float(component_rpm[1]),
            "rifle": float(component_rpm[2]),
        },
        "spin_profile": (interpretation["profile"] if interpretation is not None else None),
        "spin_prior_center_rpm": (
            float(interpretation["spin_center_rpm"]) if interpretation is not None else None
        ),
        "spin_prior_sigma_rpm": (
            float(interpretation["spin_sigma_rpm"]) if interpretation is not None else None
        ),
        "spin_prior_center_components_rpm": (
            prior_centers.tolist() if prior_centers is not None else None
        ),
        "spin_prior_sigma_components_rpm": (
            prior_sigmas.tolist() if prior_sigmas is not None else None
        ),
        "spin_deviation_z": spin_deviation_z,
        "spin_deviation_components_z": (
            spin_deviation_components_z.tolist()
            if spin_deviation_components_z is not None
            else None
        ),
        "spin_deviation_flag": (spin_deviation_z > 2.5 if spin_deviation_z is not None else False),
        "start_xyz": fit.theta[:3].tolist(),
        "net_collision": (
            {
                "frame": float(net_split["frame"]),
                "xyz": np.asarray(net_split["xyz"], float).tolist(),
                "pre_velocity_ms": net_split["pre"]
                .state(float(net_split["frame"]), fps, surface)[1]
                .tolist(),
                "post_velocity_ms": net_split["post"]
                .state(float(net_split["frame"]), fps, surface)[1]
                .tolist(),
                "model": "independent_pre_post_net_impact",
            }
            if net_split is not None
            else None
        ),
        "trajectory": _physical_trajectory_rows(
            fit,
            fps,
            surface,
            start_frame=start_frame,
            end_frame=end_frame,
            observed_spins=spins,
        ),
        "bounces": [
            {
                "frame": float(row["frame"]),
                "x": row["x"].tolist(),
                "v_in": row["v_in"].tolist(),
                "v_out": row["v_out"].tolist(),
                "w_in": row["w_in"].tolist(),
                "w_out": row["w_out"].tolist(),
                "spin_in_rpm": float(np.linalg.norm(row["w_in"]) * 60.0 / (2.0 * np.pi)),
                "spin_out_rpm": float(np.linalg.norm(row["w_out"]) * 60.0 / (2.0 * np.pi)),
                "regime": row["regime"],
                "sampling_model": row.get("sampling_model", "legacy_instantaneous_impact"),
                "dwell_seconds": float(row.get("dwell_seconds", 0.0)),
                "termination_anchor": bool(row.get("termination_anchor", False)),
            }
            for row in fit.bounces
        ],
        "surface": surface,
        "fps": fps,
    }
    if getattr(fit, "_anchor_first", False):
        compact.update(held_out_summary(fit))
        compact.update(
            {
                "fit_method": getattr(fit, "_fit_objective", "anchor_first_height_on_ray_v1"),
                "shared_contact_loss": getattr(fit, "_shared_contact_loss", "legacy_unspecified"),
                "joint_parameter_seed_used": bool(
                    getattr(fit, "_joint_parameter_seed_used", False)
                ),
                "spin_identifiable": bool(getattr(fit, "_spin_identifiable", False)),
                "anchor_errors": getattr(fit, "_anchor_errors", []),
                "consistency_evidence": getattr(fit, "_consistency_evidence", {}),
                "net_constraint": getattr(fit, "_net_constraint", None),
                "optimizer_nfev": getattr(fit, "_optimizer_nfev", None),
                "junction_refined": bool(getattr(fit, "_junction_refined", False)),
                "shared_contacts": getattr(fit, "_shared_contacts", []),
                "shared_contact_trials": getattr(fit, "_shared_contact_trials", []),
                "held_out_frame_errors": [
                    {"frame": int(frame), "error_px": float(error)}
                    for frame, error in zip(
                        getattr(fit, "_held_out_frames", []),
                        getattr(fit, "_held_out_errors_px", []),
                        strict=True,
                    )
                ],
            }
        )
    return compact


def refit_exact_contact_rays(
    raw_fits: dict,
    contacts: list[dict],
    shared_positions: dict,
    initial_positions: dict,
    ball: dict[int, np.ndarray],
    players: dict,
    camera,
    fps: float,
    surface: str,
    max_nfev: int,
    bounce_hypotheses: list[dict],
    observation_weights: dict[int, float],
    pose_rows: dict[str, dict[int, dict]] | None = None,
    physical_motion: dict[str, dict[int, dict]] | None = None,
) -> tuple[dict, dict, dict]:
    """Try exact image-ray endpoints, adopting only bounded-reprojection flight refits."""
    output = dict(raw_fits)
    ray_positions = dict(shared_positions)
    contact_diagnostics = {}
    apply_arc_junction_observation_overrides(
        contacts,
        raw_fits,
        ball,
        camera,
        fps,
        surface,
    )
    for contact_index, contact in enumerate(contacts):
        seed = ray_positions.get(contact_index, initial_positions.get(contact_index))
        if seed is None:
            continue
        pose_witness = contact_pose_witness(
            pose_rows or {"near": {}, "far": {}},
            physical_motion,
            contact=contact,
            ball=ball,
            players=players,
            camera=camera,
            fps=fps,
        )
        anchor, diagnostic = contact_ray_anchor(
            contact,
            seed,
            ball,
            players,
            camera,
            observation_weights,
            pose_witness,
        )
        contact_diagnostics[contact_index] = diagnostic
        if anchor is not None:
            ray_positions[contact_index] = anchor
    refits = {}
    for index, initial in sorted(raw_fits.items()):
        start_anchor = ray_positions.get(index, initial_positions.get(index))
        end_anchor = ray_positions.get(index + 1, initial_positions.get(index + 1))
        if start_anchor is None or end_anchor is None:
            refits[index] = {
                "status": "missing_shared_anchor",
                "initial_rms_px": float(initial.rms_px),
                "candidate_rms_px": None,
            }
            continue
        candidate = shoot_shot(
            index,
            contacts,
            ball,
            camera,
            fps,
            surface,
            (start_anchor, end_anchor),
            initial,
            max_nfev,
            bounce_anchor=bounce_for_shot(
                bounce_hypotheses,
                contacts[index]["frame"],
                contacts[index + 1]["frame"],
            ),
        )
        if candidate is None:
            refits[index] = {
                "status": "no_boundary_solution",
                "initial_rms_px": float(initial.rms_px),
                "candidate_rms_px": None,
            }
            continue
        if candidate.rms_px > max(12.0, initial.rms_px + 2.0):
            refits[index] = {
                "status": "reprojection_regression",
                "initial_rms_px": float(initial.rms_px),
                "candidate_rms_px": float(candidate.rms_px),
            }
            continue
        output[index] = candidate
        refits[index] = {
            "status": "adopted",
            "initial_rms_px": float(initial.rms_px),
            "candidate_rms_px": float(candidate.rms_px),
        }

    # A monocular contact ray has many plausible depths. A single prior-selected point can
    # fit either adjacent flight while making the other impossible. Search a compact set of
    # exact-ray depths and require the incoming and outgoing boundary-value solves to agree.
    for contact_index in range(1, len(contacts) - 1):
        incoming_index = contact_index - 1
        outgoing_index = contact_index
        incoming_seed = raw_fits.get(incoming_index)
        outgoing_seed = raw_fits.get(outgoing_index)
        if incoming_seed is None or outgoing_seed is None:
            continue
        if hasattr(incoming_seed, "state") and hasattr(outgoing_seed, "theta"):
            incoming_endpoint = incoming_seed.state(
                contacts[contact_index]["frame"],
                fps,
                surface,
            )[0]
            outgoing_endpoint = np.asarray(outgoing_seed.theta[:3], float)
            _, incoming_endpoint_diagnostic = contact_image_residual(
                incoming_endpoint,
                contacts[contact_index],
                ball,
                camera,
                observation_weights,
            )
            _, outgoing_endpoint_diagnostic = contact_image_residual(
                outgoing_endpoint,
                contacts[contact_index],
                ball,
                camera,
                observation_weights,
            )
            endpoint_errors = [
                float(row["error_px"])
                for row in (incoming_endpoint_diagnostic, outgoing_endpoint_diagnostic)
                if row.get("error_px") is not None
            ]
            junction_gap = float(np.linalg.norm(incoming_endpoint - outgoing_endpoint))
            if junction_gap <= 0.20 and len(endpoint_errors) == 2 and max(endpoint_errors) <= 12.0:
                contact_diagnostics.setdefault(contact_index, {}).update(
                    {
                        "joint_status": "existing_two_flight_solution_consistent",
                        "existing_junction_gap_m": junction_gap,
                        "existing_max_endpoint_error_px": max(endpoint_errors),
                    }
                )
                continue
        pose_witness = contact_pose_witness(
            pose_rows or {"near": {}, "far": {}},
            physical_motion,
            contact=contacts[contact_index],
            ball=ball,
            players=players,
            camera=camera,
            fps=fps,
        )
        candidates, diagnostic = contact_ray_candidates(
            contacts[contact_index],
            ray_positions.get(contact_index, initial_positions.get(contact_index)),
            ball,
            players,
            camera,
            observation_weights,
            pose_witness,
        )
        if not candidates:
            continue
        diagnostic["joint_status"] = "two_flight_ray_search_started"
        contact_diagnostics[contact_index] = diagnostic
        # Keep the strongest prior samples while retaining coverage across the ray. This is
        # bounded and deterministic; it is not an unconstrained second global optimizer.
        search = bounded_contact_ray_search(candidates, diagnostic)
        diagnostic["joint_search_candidates"] = len(search)
        start_anchor = ray_positions.get(
            incoming_index,
            initial_positions.get(incoming_index),
        )
        end_anchor = ray_positions.get(
            contact_index + 1,
            initial_positions.get(contact_index + 1),
        )
        if start_anchor is None or end_anchor is None:
            diagnostic["joint_status"] = "missing_adjacent_ray_anchor"
            continue
        search_nfev = min(max_nfev, 20)
        best = None
        for candidate_row in search:
            anchor = candidate_row["point"]
            incoming = shoot_shot(
                incoming_index,
                contacts,
                ball,
                camera,
                fps,
                surface,
                (start_anchor, anchor),
                incoming_seed,
                search_nfev,
                bounce_anchor=bounce_for_shot(
                    bounce_hypotheses,
                    contacts[incoming_index]["frame"],
                    contacts[contact_index]["frame"],
                ),
            )
            if incoming is None:
                continue
            outgoing_bounce = bounce_for_shot(
                bounce_hypotheses,
                contacts[outgoing_index]["frame"],
                contacts[outgoing_index + 1]["frame"],
            )
            if contacts[outgoing_index + 1].get("terminal"):
                outgoing = fit_fixed_start_terminal_segment(
                    float(contacts[outgoing_index]["frame"]),
                    float(contacts[outgoing_index + 1]["frame"]),
                    anchor,
                    ball,
                    camera,
                    fps,
                    surface,
                    outgoing_seed,
                    search_nfev,
                    outgoing_bounce,
                )
                if outgoing is not None:
                    outgoing.start = outgoing_index
                    outgoing.end = outgoing_index + 1
            else:
                outgoing = shoot_shot(
                    outgoing_index,
                    contacts,
                    ball,
                    camera,
                    fps,
                    surface,
                    (anchor, end_anchor),
                    outgoing_seed,
                    search_nfev,
                    bounce_anchor=outgoing_bounce,
                )
            if outgoing is None:
                continue
            worst_rms = max(float(incoming.rms_px), float(outgoing.rms_px))
            score = (
                float(incoming.rms_px)
                + float(outgoing.rms_px)
                + 0.15 * float(candidate_row["prior_score"])
                + max(0.0, worst_rms - 8.0) * 2.0
            )
            if best is None or score < best[0]:
                best = (score, anchor, incoming, outgoing, candidate_row)
        if best is None:
            diagnostic["joint_status"] = "no_two_flight_ray_solution"
            contact_diagnostics[contact_index] = diagnostic
            continue
        _, anchor, incoming, outgoing, selected = best
        if search_nfev < max_nfev:
            refined_incoming = shoot_shot(
                incoming_index,
                contacts,
                ball,
                camera,
                fps,
                surface,
                (start_anchor, anchor),
                incoming_seed,
                max_nfev,
                bounce_anchor=bounce_for_shot(
                    bounce_hypotheses,
                    contacts[incoming_index]["frame"],
                    contacts[contact_index]["frame"],
                ),
            )
            outgoing_bounce = bounce_for_shot(
                bounce_hypotheses,
                contacts[outgoing_index]["frame"],
                contacts[outgoing_index + 1]["frame"],
            )
            if contacts[outgoing_index + 1].get("terminal"):
                refined_outgoing = fit_fixed_start_terminal_segment(
                    float(contacts[outgoing_index]["frame"]),
                    float(contacts[outgoing_index + 1]["frame"]),
                    anchor,
                    ball,
                    camera,
                    fps,
                    surface,
                    outgoing_seed,
                    max_nfev,
                    outgoing_bounce,
                )
                if refined_outgoing is not None:
                    refined_outgoing.start = outgoing_index
                    refined_outgoing.end = outgoing_index + 1
            else:
                refined_outgoing = shoot_shot(
                    outgoing_index,
                    contacts,
                    ball,
                    camera,
                    fps,
                    surface,
                    (anchor, end_anchor),
                    outgoing_seed,
                    max_nfev,
                    bounce_anchor=outgoing_bounce,
                )
            if refined_incoming is not None and refined_outgoing is not None:
                incoming, outgoing = refined_incoming, refined_outgoing
        existing_score = float(output[incoming_index].rms_px + output[outgoing_index].rms_px)
        candidate_score = float(incoming.rms_px + outgoing.rms_px)
        if max(incoming.rms_px, outgoing.rms_px) > 12.0 or candidate_score > existing_score + 3.0:
            diagnostic.update(
                {
                    "joint_status": "two_flight_ray_solution_rejected",
                    "candidate_combined_rms_px": candidate_score,
                    "existing_combined_rms_px": existing_score,
                }
            )
            contact_diagnostics[contact_index] = diagnostic
            continue
        output[incoming_index] = incoming
        output[outgoing_index] = outgoing
        ray_positions[contact_index] = anchor
        diagnostic.update(
            {
                "joint_status": "two_flight_ray_solution_adopted",
                "anchor_xyz": anchor.tolist(),
                "player_distance_m": selected["player_distance_m"],
                "wrist_distance_m": selected["wrist_distance_m"],
                "candidate_combined_rms_px": candidate_score,
                "existing_combined_rms_px": existing_score,
            }
        )
        contact_diagnostics[contact_index] = diagnostic
        refits[incoming_index]["joint_ray_status"] = "adopted_incoming"
        refits[outgoing_index]["joint_ray_status"] = "adopted_outgoing"
    return output, refits, contact_diagnostics


def apply_arc_junction_observation_overrides(
    contacts: list[dict],
    fits: dict,
    ball: dict[int, np.ndarray],
    camera,
    fps: float,
    surface: str,
) -> None:
    """Use a coherent two-arc image junction when the tracked contact pixel is an outlier."""
    for contact_index in range(1, len(contacts) - 1):
        incoming = fits.get(contact_index - 1)
        outgoing = fits.get(contact_index)
        if incoming is None or outgoing is None:
            continue
        frame = float(contacts[contact_index]["frame"])
        observed, source_frames = contact_image_observation(ball, frame)
        if observed is None:
            continue
        incoming_xyz, _ = incoming.state(frame, fps, surface)
        outgoing_xyz, _ = outgoing.state(frame, fps, surface)
        projection = camera.p_at(frame)
        incoming_pixel = project_one(projection, incoming_xyz)
        outgoing_pixel = project_one(projection, outgoing_xyz)
        separation = float(np.linalg.norm(incoming_pixel - outgoing_pixel))
        incoming_error = float(np.linalg.norm(incoming_pixel - observed))
        outgoing_error = float(np.linalg.norm(outgoing_pixel - observed))
        if separation > 8.0 or min(incoming_error, outgoing_error) <= 12.0:
            continue
        incoming_weight = 1.0 / max(float(incoming.rms_px), 1.0) ** 2
        outgoing_weight = 1.0 / max(float(outgoing.rms_px), 1.0) ** 2
        junction = np.average(
            np.stack([incoming_pixel, outgoing_pixel]),
            axis=0,
            weights=[incoming_weight, outgoing_weight],
        )
        contacts[contact_index]["image_observation_override"] = junction.tolist()
        contacts[contact_index]["image_observation_source_frames"] = list(source_frames)
        contacts[contact_index]["image_observation_confidence"] = 0.70
        contacts[contact_index]["image_observation_override_provenance"] = {
            "source": "coherent_incoming_outgoing_arc_junction",
            "incoming_pixel": incoming_pixel.tolist(),
            "outgoing_pixel": outgoing_pixel.tolist(),
            "raw_track_pixel": observed.tolist(),
            "arc_separation_px": separation,
            "incoming_raw_error_px": incoming_error,
            "outgoing_raw_error_px": outgoing_error,
        }


def point_gate(
    *,
    active_valid: bool,
    attempted: int,
    fits: list[dict],
    smoothing: dict,
    camera_valid: bool = True,
    unresolved_contact_sides: int = 0,
    fps: float = 25.0,
) -> tuple[str, list[str]]:
    reasons = []
    if not active_valid:
        reasons.append("active_play_invalid")
    if not camera_valid:
        reasons.append("camera_calibration_unreliable")
    if unresolved_contact_sides:
        reasons.append("player_side_unresolved")
    solved = len(fits)
    anchor_first_fits = bool(fits) and all(
        str(row.get("fit_method", "")).startswith("anchor_first_") for row in fits
    )
    if attempted == 0 or solved / attempted < 0.75:
        reasons.append("physics_solve_coverage")
    if anchor_first_fits:
        held_out_medians = [row.get("held_out_reprojection_median_px") for row in fits]
        held_out_p90 = [row.get("held_out_reprojection_p90_px") for row in fits]
        if (
            any(value is None for value in held_out_medians)
            or float(np.median(held_out_medians)) > 8.0
        ):
            reasons.append("physics_held_out_reprojection")
        if any(value is None for value in held_out_p90) or max(held_out_p90) > 12.0:
            reasons.append("physics_held_out_worst_arc")
        if any(row.get("anchor_satisfied") is not True for row in fits):
            reasons.append("physics_anchor_violation")
    else:
        if fits and float(np.median([row["rms_px"] for row in fits])) > 8.0:
            reasons.append("physics_reprojection")
        if fits and max(row["rms_px"] for row in fits) > 12.0:
            reasons.append("physics_worst_arc")
    if fits and any(not 5.0 <= row["speed_kmh"] <= 260.0 for row in fits):
        reasons.append("physics_speed")
    if fits and any(row.get("minimum_height_m", R_BALL) < 0.018 for row in fits):
        reasons.append("physics_underground")
    if fits and any(
        row.get("fixed_bounce_continuity_m") is not None and row["fixed_bounce_continuity_m"] > 0.12
        for row in fits
    ):
        reasons.append("physics_bounce_discontinuity")
    if fits and any(
        row.get("bounce_energy_ratio", row.get("bounce_speed_ratio")) is not None
        and row.get("bounce_energy_ratio", row.get("bounce_speed_ratio")) > 1.02
        for row in fits
    ):
        reasons.append("physics_bounce_energy_gain")
    if fits and any(
        row.get("bounce_impact_velocity_slack_mps") is not None
        and row["bounce_impact_velocity_slack_mps"] > 7.5
        for row in fits
    ):
        reasons.append("physics_bounce_impact_velocity_slack")
    if fits and any(
        max(
            row.get("start_player_distance_m") or 0.0,
            row.get("end_player_distance_m") or 0.0,
        )
        > 2.1
        for row in fits
    ):
        reasons.append("physics_contact_reach")
    if fits and any(
        max(
            row.get("start_contact_reprojection_px") or 0.0,
            row.get("end_contact_reprojection_px") or 0.0,
        )
        > 12.0
        for row in fits
    ):
        reasons.append("physics_contact_reprojection")
    if any(
        (row.get(f"{endpoint}_contact_reprojection") or {}).get("position_available") is False
        for row in fits
        for endpoint in ("start", "end")
    ):
        reasons.append("physics_contact_observation_uncovered")
    reasons.extend(trajectory_connection_report(fits)["reasons"])
    if smoothing["coverage"] < 0.65:
        reasons.append("tracking_coverage")
    if (
        smoothing["maximum_gap_seconds"] is None
        or smoothing["maximum_gap_seconds"] > 0.70
        or smoothing["long_gaps"] > 8
    ):
        reasons.append("tracking_long_gap")
    if smoothing["repair_rate"] > 0.20:
        reasons.append("excessive_track_repair")
    if smoothing["post_heal_teleport_rate"] > 0.02:
        reasons.append("tracking_teleport")
    return ("hold" if reasons else "retain"), reasons


def complete_point_gate(
    *,
    point_decision: str,
    attempted: int,
    solved: int,
    terminal_flights: int,
    all_shots_accepted: bool,
    terminal_coverage: dict | None = None,
) -> tuple[bool, list[str]]:
    """Separate exact-point eligibility from the partial-shot salvage gate."""
    reasons = []
    if point_decision != "retain":
        reasons.append("s6_point_gate_held")
    if attempted == 0 or solved != attempted:
        reasons.append("incomplete_shot_solve_coverage")
    elif not all_shots_accepted:
        reasons.append("constituent_shot_gate")
    if terminal_flights != 1:
        reasons.append("terminal_flight_coverage")
    if terminal_coverage is None:
        reasons.append("terminal_evidence_missing")
    elif terminal_coverage.get("valid") is not True:
        reasons.extend(terminal_coverage.get("reasons") or ["terminal_evidence_missing"])
    return not reasons, list(dict.fromkeys(reasons))


def finalize_complete_point_decision(
    point_decision: str,
    point_reasons: list[str],
    complete_candidate: bool,
    complete_reasons: list[str],
) -> tuple[str, list[str]]:
    """Make the exported point verdict honor its stricter composition contract."""
    if complete_candidate:
        return point_decision, list(point_reasons)
    return "hold", list(dict.fromkeys([*point_reasons, *complete_reasons]))


def apply_contact_frame_observation(
    contact: dict, refinement: RefinedContact, *, first_physical_frame: float
) -> None:
    """Keep a refined picture only when the modeled rally can own its exposure.

    The first contact has no incoming flight in this reconstruction. Its toss
    pictures must not replace a valid emitted observation from the served flight.
    Interior contacts may use incoming pictures; final piecewise reprojection
    still checks the actual solved owner after any fitted timing changes.
    """
    evidence = refinement.as_dict()
    evidence["observation_applied"] = False
    evidence["first_physical_frame"] = first_physical_frame
    contact["contact_frame_refinement"] = evidence
    if refinement.abstain or refinement.pixel is None:
        evidence["observation_rejection_reason"] = "refiner_abstained"
        return
    if not math.isfinite(first_physical_frame) or refinement.frame < first_physical_frame:
        evidence["observation_rejection_reason"] = "picture_precedes_modeled_rally"
        return
    contact.update(
        {
            "image_observation_override": list(refinement.pixel),
            "image_observation_frame": float(refinement.frame),
            "image_observation_source_frames": [int(refinement.frame)],
            "image_observation_confidence": float(np.clip(10.0 / refinement.sigma_px, 0.05, 1.0)),
            "image_observation_sigma_px": float(refinement.sigma_px),
            "image_observation_source": "contact_frame_refiner",
        }
    )
    evidence["observation_applied"] = True


def point_contact_reprojection(
    contact_index: int,
    contacts: list[dict],
    fits: dict[int, object],
    ball: dict[int, np.ndarray],
    camera,
    observation_weights: dict[int, float],
    fps: float,
    surface: str,
) -> dict:
    """Compare a contact observation with its owning piece of the point trajectory.

    An integer picture before a fractional racket impact belongs to the incoming
    flight; one after it belongs to the outgoing flight. Extrapolating the other
    flight across the impulse invents a different path and a spurious pixel error.
    Resolve the actual tracked picture before selecting its owning flight. A nearest
    observation keeps its own timestamp; a chord across a racket impulse is not an
    observed picture. Explicit pixel overrides retain their declared observation time.
    Endpoint continuity remains an independent hard check. Missing owned fits or
    invalid states abstain rather than falling back to the impact's position.
    """
    contact = contacts[contact_index]
    observation_frame = float(contact.get("image_observation_frame", contact["frame"]))
    requested_observation_frame = observation_frame
    if contact.get("image_observation_override") is None:
        _, source_frames = contact_image_observation(ball, observation_frame)
        if source_frames:
            observation_frame = float(
                min(source_frames, key=lambda frame: (abs(frame - observation_frame), frame))
            )
            # This is local sampling metadata, never a rewrite of an event/contact time.
            contact = {**contact, "image_observation_frame": observation_frame}
    evidence = {
        "sampling_convention": "piecewise_contact_observation_v2",
        "requested_observation_frame": requested_observation_frame,
        "observation_frame": observation_frame,
        "sampled_flight_index": None,
        "position_available": False,
        "available": False,
        "source_frames": [],
        "error_px": None,
    }
    owners = [
        index
        for index in (contact_index - 1, contact_index)
        if 0 <= index < len(contacts) - 1
        and float(contacts[index]["frame"])
        <= observation_frame
        <= float(contacts[index + 1]["frame"])
    ]
    # At the exact impact either connected endpoint is valid; use a solved owner.
    owner = next((index for index in owners if index in fits), None)
    if owner is None:
        return {**evidence, "reason": "observation_outside_solved_adjacent_flights"}
    evidence["sampled_flight_index"] = owner
    try:
        position = np.asarray(fits[owner].state(observation_frame, fps, surface)[0], float)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return {**evidence, "reason": "observation_state_failed"}
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return {**evidence, "reason": "observation_state_invalid"}
    _, residual = contact_image_residual(position, contact, ball, camera, observation_weights)
    return {**evidence, **residual, "position_available": True}


def junction_gaps(fits: list[dict]) -> list[float]:
    """Historical same-frame gap diagnostic; acceptance uses trajectory_connection_report."""
    ordered = sorted(fits, key=lambda row: row.get("start_frame", float("inf")))
    gaps = []
    for incoming, outgoing in zip(ordered, ordered[1:]):
        if incoming.get("end_frame") != outgoing.get("start_frame"):
            continue
        incoming_end = incoming.get("end_xyz")
        if incoming_end is None and incoming.get("trajectory"):
            incoming_end = incoming["trajectory"][-1]["xyz"]
        outgoing_start = outgoing.get("start_xyz")
        if incoming_end is None or outgoing_start is None:
            continue
        gaps.append(
            float(
                np.linalg.norm(np.asarray(incoming_end, float) - np.asarray(outgoing_start, float))
            )
        )
    return gaps


def junction_evidence(
    fits: list[dict],
    raw_fits: dict,
    contacts: list[dict],
    fps: float,
    surface: str,
) -> None:
    """Record each flight's disagreement with the flight it shares a contact with.

    The junction is one impact seen twice, so two flights that are each right agree about where
    the ball was and how fast it was going.  Both readings are label-free -- no truth, no owner
    click -- and they are the evidence ``docs/wk1/gate_search.md`` found missing in the frozen
    fit records: the position gap is the quantity the search's dominant weight already wanted,
    and the velocity disagreement is the one a wrong-depth pair cannot fake by meeting on the
    camera ray.  The gap is evaluated at the contact's own fitted time when a shared contact
    moved it, and at the emitted frame otherwise, which is where the pipeline reports it.
    """
    by_index = {int(row["flight_index"]): row for row in fits}
    for index, row in sorted(by_index.items()):
        evidence = row.setdefault("consistency_evidence", {})
        for name, neighbour_index, contact_index in (
            ("start", index - 1, index),
            ("end", index + 1, index + 1),
        ):
            evidence[f"junction_{name}_position_gap_m"] = None
            evidence[f"junction_{name}_velocity_gap_ms"] = None
            evidence[f"junction_{name}_speed_ratio"] = None
            if neighbour_index not in by_index or contact_index >= len(contacts):
                continue
            if contacts[contact_index].get("terminal"):
                continue
            here = raw_fits.get(index)
            there = raw_fits.get(neighbour_index)
            if here is None or there is None:
                continue
            frame = float(contacts[contact_index]["frame"])
            try:
                here_state = here.state(frame, fps, surface)
                there_state = there.state(frame, fps, surface)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            position_gap = float(
                np.linalg.norm(np.asarray(here_state[0], float) - np.asarray(there_state[0], float))
            )
            velocity_gap = float(
                np.linalg.norm(np.asarray(here_state[1], float) - np.asarray(there_state[1], float))
            )
            here_speed = float(np.linalg.norm(np.asarray(here_state[1], float)))
            there_speed = float(np.linalg.norm(np.asarray(there_state[1], float)))
            if not (math.isfinite(position_gap) and math.isfinite(velocity_gap)):
                continue
            evidence[f"junction_{name}_position_gap_m"] = position_gap
            evidence[f"junction_{name}_velocity_gap_ms"] = velocity_gap
            evidence[f"junction_{name}_speed_ratio"] = float(
                max(here_speed, there_speed) / max(min(here_speed, there_speed), 1e-6)
            )
        # The impact-time prior the shared contact ended up disagreeing with, in its own sigmas.
        evidence["contact_time_prior_residual_sigmas"] = None
        for record in getattr(raw_fits.get(index), "_shared_contacts", []) or []:
            contact_index = int(record.get("contact_index", -1))
            if not 0 <= contact_index < len(contacts):
                continue
            prior = contacts[contact_index].get("time_prior") or {}
            sigma = prior.get("sigma_frames")
            if not sigma:
                continue
            offset = float(record.get("time_offset_frames") or 0.0)
            evidence["contact_time_prior_residual_sigmas"] = float(
                (offset - float(prior.get("offset_frames") or 0.0)) / float(sigma)
            )


def apply_shared_contact_times(
    contacts: list[dict],
    flight_attempts: list[dict],
    fits: dict[int, object],
    camera: PointCamera,
    fps: float,
) -> None:
    """Carry an adopted joint contact time into downstream compaction metadata."""
    adopted: dict[int, float] = {}
    for fit in fits.values():
        for record in getattr(fit, "_shared_contacts", []):
            index = int(record["contact_index"])
            frame = float(record["frame"])
            if index in adopted and not math.isclose(adopted[index], frame, abs_tol=1e-9):
                raise ValueError(f"inconsistent shared contact time for contact {index}")
            adopted[index] = frame
    if not adopted:
        return
    for index, frame in adopted.items():
        contacts[index].setdefault("image_observation_frame", float(contacts[index]["frame"]))
        contacts[index]["frame"] = frame
    for attempt in flight_attempts:
        index = int(attempt["flight_index"])
        original_start = float(attempt["start_frame"])
        original_end = float(attempt["end_frame"])
        start = float(contacts[index]["frame"])
        end = float(contacts[index + 1]["frame"])
        attempt["start_frame"] = start
        attempt["end_frame"] = end
        attempt["duration_seconds"] = float((end - start) / fps)
        attempt["camera_start"] = camera.quality_at(start)
        attempt["camera_end"] = camera.quality_at(end)
        if start != original_start or end != original_end:
            attempt["shared_contact_time_adjustment"] = {
                "original_start_frame": original_start,
                "original_end_frame": original_end,
                "fitted_start_frame": start,
                "fitted_end_frame": end,
            }


def screening_gate(
    *,
    active_valid: bool,
    attempted: int,
    fits: list[dict],
    smoothing: dict,
    camera_valid: bool = True,
) -> tuple[str, list[str]]:
    reasons = []
    if not active_valid:
        reasons.append("active_play_invalid")
    if not camera_valid:
        reasons.append("camera_calibration_unreliable")
    if attempted == 0 or len(fits) / attempted < 0.75:
        reasons.append("physics_solve_coverage")
    if fits and float(np.median([row["rms_px"] for row in fits])) > 30.0:
        reasons.append("screen_reprojection")
    if fits and max(row["rms_px"] for row in fits) > 50.0:
        reasons.append("screen_worst_arc")
    if fits and any(not 5.0 <= row["speed_kmh"] <= 280.0 for row in fits):
        reasons.append("screen_speed")
    if smoothing["coverage"] < 0.65:
        reasons.append("tracking_coverage")
    if (
        smoothing["maximum_gap_seconds"] is None
        or smoothing["maximum_gap_seconds"] > 0.70
        or smoothing["long_gaps"] > 8
    ):
        reasons.append("tracking_long_gap")
    if smoothing["repair_rate"] > 0.20:
        reasons.append("excessive_track_repair")
    if smoothing["post_heal_teleport_rate"] > 0.02:
        reasons.append("tracking_teleport")
    return ("screen" if not reasons else "hold"), reasons


def _timeout_point_record(
    clip_key: str,
    *,
    event_boundary_source: str,
    event_consumer_mode: str,
    point_timeout_seconds: float,
) -> dict:
    """Emit a complete fail-closed record that the flight ledger can represent."""
    match_id, clip = clip_key.rsplit("__", 1)
    return {
        "match_id": match_id,
        "audit_root_name": None,
        "clip": clip,
        "point": clip_key,
        "surface": None,
        "fps": 0.0,
        "cadence_identity": "unknown_due_timeout",
        "decision": "hold",
        "reasons": ["timeout"],
        "point_gate": {"decision": "hold", "reasons": ["timeout"]},
        "screen_decision": "hold",
        "screen_reasons": ["timeout"],
        "active_play_valid": False,
        "active_spans": [],
        "camera_calibration": {"accepted": False, "reason": "timeout"},
        "attempted_shots": 1,
        "solved_shots": 0,
        "terminal_flights": 0,
        "complete_point_gate": {"accepted": False, "reasons": ["timeout"]},
        "flight_attempts": [
            {
                "flight_index": 0,
                "start_frame": None,
                "end_frame": None,
                "duration_seconds": None,
                "start_side": None,
                "end_side": None,
                "start_phase": None,
                "terminal_end": False,
                "intermediate_events": [],
                "solved": False,
                "skip_reason": "timeout",
            }
        ],
        "smoothing": {},
        "junction_gaps_m": [],
        "legacy_tracking_decision": None,
        "upstream_point_validity": None,
        "upstream_repair_override": {"applied": False, "reason": "timeout"},
        "event_boundary_source": event_boundary_source,
        "event_consumer_mode": event_consumer_mode,
        "camera_scope_source": None,
        "fits": [],
        "bounce_scores": [],
        "uncertainty": {"enabled": False, "accepted_bounce_hypotheses": 0},
        "interpretation_branches": {"enabled": False},
        "net_collision_topology": {},
        "unrefined_flight_initializers": {},
        "event_topology_branches": {"enabled": False},
        "joint_rally_refinement": {"enabled": False},
        "contact_reprojection": {},
        "owner_tracking_quality": None,
        "timeout_seconds": point_timeout_seconds,
    }


def _evaluate_point_worker(
    audit_root: Path,
    manifest_path: Path,
    clip_key: str,
    run_options: dict,
    math_threads: int,
    result_queue: mp.queues.Queue,
) -> None:
    """Evaluate one point in an isolated child and return its record to the parent."""
    try:
        with threadpool_limits(limits=math_threads):
            report = _run_reconstruction(
                audit_root,
                manifest_path,
                clips=[clip_key],
                workers=1,
                isolate_points=False,
                **run_options,
            )
        result_queue.put(("ok", report["points_detail"][0]))
    except BaseException as error:
        result_queue.put(("error", f"{type(error).__name__}: {error}\n{traceback.format_exc()}"))


def _robust_match_interval(values: list[float], floor: float) -> tuple[float, float, float] | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], float)
    if len(finite) < 4:
        return None
    center = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - center)))
    radius = max(floor, 3.0 * mad)
    return center, center - radius, center + radius


def apply_match_shared_priors(points: list[dict]) -> dict:
    """Estimate label-free broadcast priors and withhold only extreme outliers."""
    by_match: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_match[str(point["match_id"])].append(point)
    summaries = {}
    for match_id, match_points in by_match.items():
        heights: dict[str, list[float]] = defaultdict(list)
        serve_speeds: list[float] = []
        speed_losses: list[float] = []
        court_points: list[np.ndarray] = []
        fit_rows = []
        for point in match_points:
            attempts = {int(row["flight_index"]): row for row in point.get("flight_attempts", [])}
            for fit in point.get("fits", []):
                index = int(fit["flight_index"])
                attempt = attempts.get(index, {})
                start = np.asarray(fit.get("start_xyz", []), float)
                end = np.asarray(fit.get("end_xyz", []), float)
                if len(start) == 3:
                    heights[str(attempt.get("start_side", "unknown"))].append(float(start[2]))
                    court_points.append(start[:2])
                if len(end) == 3 and not attempt.get("terminal_end"):
                    heights[str(attempt.get("end_side", "unknown"))].append(float(end[2]))
                    court_points.append(end[:2])
                speed = float(fit.get("speed_kmh", np.nan))
                if attempt.get("start_phase") == "serve" and np.isfinite(speed):
                    serve_speeds.append(speed)
                trajectory = fit.get("trajectory", [])
                if len(trajectory) >= 2:
                    first = np.linalg.norm(np.asarray(trajectory[0]["velocity"], float))
                    last = np.linalg.norm(np.asarray(trajectory[-1]["velocity"], float))
                    duration = max(
                        (float(trajectory[-1]["frame"]) - float(trajectory[0]["frame"]))
                        / float(point["fps"]),
                        0.04,
                    )
                    if first > 1e-6 and last > 1e-6:
                        speed_losses.append(float(-np.log(last / first) / duration))
                fit_rows.append((point, fit, attempt))
        height_priors = {
            side: interval
            for side, values in heights.items()
            if (interval := _robust_match_interval(values, 0.45)) is not None
        }
        serve_prior = _robust_match_interval(serve_speeds, 35.0)
        aero_prior = _robust_match_interval(speed_losses, 0.35)
        camera_scale = None
        if len(court_points) >= 8:
            values = np.stack(court_points)
            x_span = float(np.percentile(values[:, 0], 95) - np.percentile(values[:, 0], 5))
            y_span = float(np.percentile(values[:, 1], 95) - np.percentile(values[:, 1], 5))
            camera_scale = float(
                np.clip(
                    np.median([10.97 / max(x_span, 1.0), 23.77 / max(y_span, 1.0)]),
                    0.8,
                    1.2,
                )
            )
        rejected = 0
        for point, fit, attempt in fit_rows:
            reasons = []
            for endpoint, side_key in (("start_xyz", "start_side"), ("end_xyz", "end_side")):
                xyz = fit.get(endpoint)
                prior = height_priors.get(str(attempt.get(side_key)))
                if (
                    xyz is not None
                    and prior is not None
                    and not prior[1] <= float(xyz[2]) <= prior[2]
                ):
                    reasons.append(f"{side_key}_height")
            if attempt.get("start_phase") == "serve" and serve_prior is not None:
                speed = float(fit.get("speed_kmh", np.nan))
                if not serve_prior[1] <= speed <= serve_prior[2]:
                    reasons.append("serve_speed")
            fit["match_shared_prior"] = {
                "accepted": not reasons,
                "violations": reasons,
                "camera_scale": camera_scale,
                "aero_speed_loss_per_second": aero_prior,
                "serve_speed_kmh": serve_prior,
                "contact_height_m": height_priors,
            }
            if reasons and fit.get("status") == "provisional_valid":
                fit["status"] = "recoverable"
                fit.setdefault("reasons", []).append("match_shared_prior_violation")
                rejected += 1
                point["decision"] = "hold"
                point.setdefault("reasons", []).append("match_shared_prior_violation")
                point["complete_point_gate"]["accepted"] = False
                point["complete_point_gate"].setdefault("reasons", []).append(
                    "match_shared_prior_violation"
                )
        summaries[match_id] = {
            "points": len(match_points),
            "fits": len(fit_rows),
            "camera_scale": camera_scale,
            "aero_speed_loss_per_second": aero_prior,
            "serve_speed_kmh": serve_prior,
            "contact_height_m": height_priors,
            "rejected_flights": rejected,
        }
    return {"schema": "match_shared_priors_v1", "matches": summaries}


def _run_reconstruction(
    audit_root: Path,
    manifest_path: Path,
    *,
    clips: list[str],
    automatic_boundaries: dict[str, list[dict]],
    automatic_hypotheses: dict[str, list[dict]] | None = None,
    max_nfev: int,
    workers: int,
    multi_start: bool = False,
    joint_refine: bool = False,
    terminal_flights: bool = False,
    uncertainty_hypotheses: Path | None = None,
    discrete_bounce_branches: bool = False,
    anchor_bounce_geometry: bool = False,
    anchor_first: bool = True,
    net_point_anchor: bool = True,
    downweight_contact_adjacent: bool = False,
    striker_witness_prior: bool = False,
    striker_witness_authority: bool = True,
    contact_observation_witness: bool = False,
    contact_observation_sigma: bool = False,
    physical_spin_prior: bool = False,
    subframe_anchors: bool = False,
    subframe_contacts: bool = False,
    subframe_time_priors: bool = False,
    subframe_contact_seam: bool = True,
    subframe_contact_advance: bool = True,
    subframe_contact_seam_skips_refit: bool = True,
    subframe_contact_seam_witnessed: bool = False,
    subframe_plane_anchor_error: bool = False,
    subframe_bounce_witness: bool = False,
    whole_point_joint: bool = False,
    shared_contact_fit: bool = True,
    match_shared_priors: bool = False,
    whole_point_branches: bool = False,
    exact_contact_reprojection: bool = False,
    event_topology_branches: bool = False,
    topology_sequence_branches: bool = False,
    topology_branch_width: int = 8,
    topology_branch_margin: float = 8.0,
    topology_post_joint_scoring: bool = False,
    topology_joint_width: int = 2,
    branch_width: int = 6,
    branch_margin: float = 8.0,
    event_boundary_source: str = "automatic",
    event_consumer_mode: str = "point_grammar_in_play_only",
    camera_scope: Path | None = None,
    pose_artifact_name: str | None = None,
    physical_motion_name: str | None = None,
    camera_artifact_name: str = "camera_P_per_frame_v1.npz",
    anchors_output_root: Path | None = None,
    point_timeout_seconds: float = 600.0,
    math_threads: int = 1,
    isolate_points: bool = False,
) -> dict:
    # Point workers may start through ``forkserver`` and therefore do not inherit
    # module globals configured by the CLI parent.  Apply the explicit run option
    # inside every process before constructing any fitter objective.
    configure_contact_adjacent_weighting(downweight_contact_adjacent)
    configure_striker_witness(prior=striker_witness_prior, authority=striker_witness_authority)
    configure_contact_observation_witness(contact_observation_witness)
    configure_contact_observation_sigma(contact_observation_sigma)
    configure_physical_spin_prior(physical_spin_prior)
    configure_subframe_anchors(subframe_anchors)
    configure_subframe_contacts(subframe_contacts)
    configure_subframe_contact_parts(
        seam=subframe_contact_seam,
        advance=subframe_contact_advance,
        seam_skips_refit=subframe_contact_seam_skips_refit,
        seam_witnessed=subframe_contact_seam_witnessed,
    )
    configure_subframe_plane_anchor_error(subframe_plane_anchor_error)
    configure_subframe_bounce_witness(subframe_bounce_witness)
    configure_whole_point_joint(whole_point_joint)
    manifest = json.loads(manifest_path.read_text())
    # Anchor-first owns its bounded whole-point contact pass. Do not compose it
    # with the legacy contact-first global/interpretation refiners.
    if anchor_first:
        joint_refine = False
        whole_point_branches = False
        discrete_bounce_branches = False
        anchor_bounce_geometry = False
        exact_contact_reprojection = False
    specs = {row["id"]: row for row in manifest["matches"]}
    hypotheses_by_point: dict[str, list[dict]] = defaultdict(list)
    if uncertainty_hypotheses is not None:
        hypothesis_scale = res.coordinate_scale(uncertainty_hypotheses, columns=("img_x", "img_y"))
        for row in json.loads(uncertainty_hypotheses.read_text()):
            hypotheses_by_point[f"{row['match_id']}__{row['clip']}"].append(
                {
                    **row,
                    "img_x_native": float(row["img_x"]) * hypothesis_scale[0],
                    "img_y_native": float(row["img_y"]) * hypothesis_scale[1],
                }
            )
    active_paths = active_play_content_paths(audit_root)
    if not active_paths:
        raise FileNotFoundError(f"no active_play_v*.json under {audit_root}")
    active = json.loads(active_paths[-1].read_text())
    tracking_gate = json.loads((audit_root / "untouched_tracking_point_gate_v1.json").read_text())
    tracking_rows_by_point = {
        f"{row['match_id']}__{row['clip']}": row for row in tracking_gate["rows"]
    }
    tracking_decisions = {key: row["decision"] for key, row in tracking_rows_by_point.items()}
    point_validity_path = audit_root / "point_validity_gate_v1.json"
    point_validity = {}
    if point_validity_path.is_file():
        validity_report = json.loads(point_validity_path.read_text())
        point_validity = {
            f"{row['match_id']}__{row['clip']}": row for row in validity_report["rows"]
        }
    camera_scope_points = {}
    if camera_scope is not None:
        scope_artifact = json.loads(camera_scope.read_text())
        if scope_artifact.get("labels_loaded") is not False:
            raise ValueError("camera scope must be an automatic label-free artifact")
        camera_scope_points = scope_artifact.get("points", {})

    def evaluate_point(clip_key: str) -> dict:
        match_id, clip = clip_key.split("__", 1)
        match_dir = audit_root / match_id
        spec = specs[match_id]
        fps = float(spec["source_fps"])
        cadence_identity = validate_artifact_cadence(match_dir, fps)
        surface = spec["surface"]
        active_row = active[f"{match_id}/{clip}"]
        validity_row = point_validity.get(clip_key)
        tracking_row = tracking_rows_by_point.get(clip_key)
        retained_arc_override = upstream_hold_has_retained_tracking_arc(
            validity_row,
            tracking_row,
        )
        repairable_upstream_hold = (
            event_topology_branches or joint_refine
        ) and upstream_hold_is_repairable(validity_row)
        if (
            validity_row is not None
            and validity_row["decision"] != "retain"
            and not repairable_upstream_hold
            and not retained_arc_override
        ):
            reason = "upstream_point_validity_gate"
            return {
                "match_id": match_id,
                "audit_root_name": audit_root.name,
                "clip": clip,
                "point": clip_key,
                "surface": surface,
                "fps": fps,
                "decision": "hold",
                "reasons": [reason, *validity_row.get("reasons", [])],
                "screen_decision": "hold",
                "screen_reasons": [reason, *validity_row.get("reasons", [])],
                "active_play_valid": bool(active_row["point_valid"]),
                "camera_calibration": {
                    "accepted": False,
                    "reason": reason,
                    "frames": 0,
                    "reliable_fraction": 0.0,
                    "sources": [],
                    "fallback_ancestry": [],
                    "frame_scopes": [],
                },
                "attempted_shots": 0,
                "solved_shots": 0,
                "terminal_flights": 0,
                "complete_point_gate": {
                    "accepted": False,
                    "reasons": [
                        "s6_point_gate_held",
                        "incomplete_shot_solve_coverage",
                        "terminal_flight_coverage",
                    ],
                },
                "flight_attempts": [],
                "smoothing": {},
                "junction_gaps_m": [],
                "legacy_tracking_decision": tracking_decisions.get(clip_key),
                "upstream_point_validity": validity_row,
                "event_boundary_source": event_boundary_source,
                "event_consumer_mode": event_consumer_mode,
                "fits": [],
                "bounce_scores": [],
                "uncertainty": {
                    "enabled": uncertainty_hypotheses is not None,
                    "weighted_observations": 0,
                    "accepted_bounce_hypotheses": 0,
                    "bounce_hypotheses": [],
                    "bounce_geometry_anchored": anchor_bounce_geometry,
                    "bounce_scoring_independent": not anchor_bounce_geometry,
                },
                "interpretation_branches": {
                    "enabled": whole_point_branches,
                    "decisive": False,
                    "reason": reason,
                },
                "owner_tracking_quality": None,
            }
        try:
            camera = PointCamera(match_dir, clip, camera_artifact_name)
        except ValueError as exc:
            if isinstance(exc, UnsupportedCameraModelError):
                reason = "unsupported_camera_model"
            elif str(exc).startswith("no camera projections for "):
                reason = "camera_projection_unavailable"
            else:
                raise
            return {
                "match_id": match_id,
                "audit_root_name": audit_root.name,
                "clip": clip,
                "point": clip_key,
                "surface": surface,
                "fps": fps,
                "decision": "hold",
                "reasons": [reason],
                "screen_decision": "hold",
                "screen_reasons": [reason],
                "active_play_valid": bool(active_row["point_valid"]),
                "camera_calibration": {
                    "accepted": False,
                    "reason": reason,
                    "detail": str(exc),
                    "frames": 0,
                    "reliable_fraction": 0.0,
                    "sources": [],
                    "fallback_ancestry": [],
                    "frame_scopes": [],
                },
                "attempted_shots": 0,
                "solved_shots": 0,
                "terminal_flights": 0,
                "flight_attempts": [],
                "smoothing": {},
                "junction_gaps_m": [],
                "legacy_tracking_decision": tracking_decisions.get(clip_key),
                "upstream_point_validity": validity_row,
                "event_boundary_source": event_boundary_source,
                "event_consumer_mode": event_consumer_mode,
                "fits": [],
                "bounce_scores": [],
                "uncertainty": {
                    "enabled": uncertainty_hypotheses is not None,
                    "weighted_observations": 0,
                    "accepted_bounce_hypotheses": 0,
                    "bounce_hypotheses": [],
                    "bounce_geometry_anchored": anchor_bounce_geometry,
                    "bounce_scoring_independent": not anchor_bounce_geometry,
                },
                "interpretation_branches": {
                    "enabled": whole_point_branches,
                    "decisive": False,
                    "reason": reason,
                },
                "owner_tracking_quality": None,
            }
        track = load_track(match_dir, clip)
        track_weights = load_track_weights(match_dir, clip)
        players, boxes = load_players(match_dir, clip)
        pose_rows = load_pose_rows(match_dir, clip, pose_artifact_name)
        physical_motion = load_physical_motion(match_dir, clip, physical_motion_name)
        active_spans = active_row["active_spans"]
        camera_calibration = camera.quality_summary(active_spans)
        if not camera_calibration["accepted"]:
            reason = "camera_calibration_unreliable"
            return {
                "match_id": match_id,
                "audit_root_name": audit_root.name,
                "clip": clip,
                "point": clip_key,
                "surface": surface,
                "fps": fps,
                "decision": "hold",
                "reasons": [reason],
                "screen_decision": "hold",
                "screen_reasons": [reason],
                "active_play_valid": bool(active_row["point_valid"]),
                "camera_calibration": camera_calibration,
                "attempted_shots": 0,
                "solved_shots": 0,
                "terminal_flights": 0,
                "flight_attempts": [],
                "smoothing": {},
                "junction_gaps_m": [],
                "legacy_tracking_decision": tracking_decisions.get(clip_key),
                "upstream_point_validity": validity_row,
                "event_boundary_source": event_boundary_source,
                "event_consumer_mode": event_consumer_mode,
                "fits": [],
                "bounce_scores": [],
                "uncertainty": {
                    "enabled": uncertainty_hypotheses is not None,
                    "weighted_observations": 0,
                    "accepted_bounce_hypotheses": 0,
                    "bounce_hypotheses": [],
                    "bounce_geometry_anchored": anchor_bounce_geometry,
                    "bounce_scoring_independent": not anchor_bounce_geometry,
                },
                "interpretation_branches": {
                    "enabled": whole_point_branches,
                    "decisive": False,
                    "reason": reason,
                },
                "owner_tracking_quality": None,
            }

        def active_span_index(frame: float) -> int | None:
            for span_index, (start, end) in enumerate(active_spans):
                if start <= frame <= end:
                    return span_index
            return None

        boundary_events = [
            row
            for row in automatic_boundaries.get(clip_key, [])
            if active_span_index(row["frame"]) is not None
        ]
        contacts = []
        for row in boundary_events:
            if row["event_type"] != "contact":
                continue
            side_evidence = contact_side_evidence(row["frame"], track, boxes, fps)
            contact = {
                "frame": row["frame"],
                "event_frame": row["frame"],
                "side": side_evidence["side"],
                "side_evidence": side_evidence,
                "phase": "rally",
                "span": active_span_index(row["frame"]),
                "source": "automatic_event_boundary",
                "row": row,
            }
            x1080 = optional_float(row.get("x1080"))
            y1080 = optional_float(row.get("y1080"))
            if x1080 is not None and y1080 is not None:
                contact.update(
                    {
                        "image_observation_override": [x1080, y1080],
                        "image_observation_frame": float(
                            row.get("observation_frame", row["frame"])
                        ),
                        "image_observation_source_frames": [],
                        "image_observation_confidence": float(
                            np.clip(row.get("probability") or 1.0, 0.05, 1.0)
                        ),
                        "image_observation_source": "contact_emission_native_xy",
                    }
                )
            contacts.append(contact)
        seen_spans = set()
        for contact in contacts:
            if contact["span"] not in seen_spans:
                contact["phase"] = "serve"
                seen_spans.add(contact["span"])
        smoothed, smoothing = smooth_track(
            track,
            active_spans,
            fps,
            bridge_out_of_frame_gaps=terminal_flights,
        )
        # One sub-frame time prior per impact emission, built once from the point's whole
        # composed track and reused by every anchor build below -- the topology branches call
        # ``build_point_anchors`` per flight attempt, and the prior does not depend on which
        # attempt is being tried.
        impact_time_priors = (
            build_time_priors(boundary_events, smoothed, camera, fps)
            if subframe_time_priors
            else {}
        )
        # The court-plane corner under every bounce emission.  This is built on every run, arm
        # or no arm, because it is the label-free evidence the gate lane asked the fitter to
        # emit; only ``--subframe-bounce-witness`` lets it into the objective.
        bounce_court_witnesses = build_bounce_witnesses(boundary_events, smoothed, camera, fps)
        for contact in contacts:
            prior = impact_time_priors.get(("contact", float(contact["frame"])))
            if prior is not None:
                contact["time_prior"] = prior
        observation_weights = {
            frame: (
                track_weights.get(frame, 0.20)
                if frame in track
                else tracking_observation_weight(
                    0.0,
                    (),
                    interpolated=True,
                )
            )
            for frame in smoothed
        }
        if shared_contact_fit:
            pose_frames: dict[int, list[dict]] = defaultdict(list)
            for side_rows in pose_rows.values():
                for frame, row in side_rows.items():
                    pose_frames[int(frame)].append(row)
            first_physical_frames = {
                span: min(c["frame"] for c in contacts if c["span"] == span)
                for span in {c["span"] for c in contacts}
            }
            for contact in contacts:
                refined_contact = refine_contact_frame(
                    smoothed,
                    float(contact.get("event_frame", contact["frame"])),
                    pose_frames,
                )
                apply_contact_frame_observation(
                    contact,
                    refined_contact,
                    first_physical_frame=float(first_physical_frames[contact["span"]]),
                )
                pose_witness = contact_pose_witness(
                    pose_rows,
                    physical_motion,
                    contact=contact,
                    ball=smoothed,
                    players=players,
                    camera=camera,
                    fps=fps,
                )
                if pose_witness is not None:
                    contact["pose_witness"] = pose_witness
        automatic_bounce_candidates = []
        for row in (automatic_hypotheses or {}).get(clip_key, []):
            if row.get("event_type") != "bounce" or float(row.get("probability", 0.0)) < 0.10:
                continue
            if active_span_index(float(row["frame"])) is None:
                continue
            point = nearest_track_point(smoothed, float(row["frame"]))
            if point is None:
                continue
            court_xy = court_point(camera, point, float(row["frame"]))
            automatic_bounce_candidates.append(
                {
                    "frame": float(row["frame"]),
                    "x": np.r_[court_xy, R_BALL],
                    "court_xy": court_xy,
                    "probability": float(row["probability"]),
                    "source": "s5_leaky_bounce_hypothesis",
                }
            )
        if terminal_flights:
            contacts = append_terminal_contacts(
                contacts,
                smoothed,
                active_spans,
                fps,
                observation_weights,
                boundary_events,
                bridge_out_of_frame_gaps=True,
                forward_ground_completion=subframe_anchors,
            )

        def initial_fit_sequence(
            sequence: list[dict],
            *,
            fit_max_nfev: int,
            fit_multi_start: bool,
            reusable_fits: dict[tuple, object | None] | None = None,
        ) -> tuple[dict, list[dict]]:
            sequence_fits = {}
            sequence_attempts = []
            for index in range(len(sequence) - 1):
                if sequence[index]["span"] != sequence[index + 1]["span"]:
                    continue
                if (
                    not sequence[index + 1].get("terminal")
                    and sequence[index + 1]["frame"] - sequence[index]["frame"]
                    > MAX_SHOT_SECONDS * fps
                ):
                    continue
                start = sequence[index]
                end = sequence[index + 1]
                sequence_attempts.append(
                    {
                        "flight_index": index,
                        "start_frame": float(start["frame"]),
                        "end_frame": float(end["frame"]),
                        "duration_seconds": float((end["frame"] - start["frame"]) / fps),
                        "start_side": start["side"],
                        "end_side": end["side"],
                        "start_phase": start["phase"],
                        "start_side_evidence": start.get("side_evidence"),
                        "end_side_evidence": end.get("side_evidence"),
                        "span": int(start["span"]),
                        "terminal_end": bool(end.get("terminal")),
                        "camera_start": camera.quality_at(start["frame"]),
                        "camera_end": camera.quality_at(end["frame"]),
                        "start_boundary": {
                            "source": start["source"],
                            "probability": start["row"].get("probability"),
                            "origin": start["row"].get("origin"),
                        },
                        "end_boundary": {
                            "source": end["source"],
                            "probability": end["row"].get("probability"),
                            "origin": end["row"].get("origin"),
                        },
                        "intermediate_events": [
                            {
                                "event_type": event["event_type"],
                                "frame": float(event["frame"]),
                                "probability": event.get("probability"),
                                "origin": event.get("origin"),
                            }
                            for event in boundary_events
                            if start["frame"] < event["frame"] < end["frame"]
                        ],
                        "solved": False,
                    }
                )
                arc_scope = tracking_arc_fit_scope(
                    tracking_row,
                    float(start["frame"]),
                    float(end["frame"]),
                    set(smoothed),
                )
                sequence_attempts[-1]["tracking_arc_scope"] = {
                    **arc_scope,
                    "retained_frame_count": len(arc_scope["retained_frames"]),
                }
                if arc_scope["available"] and arc_scope["decision"] == "hold":
                    sequence_attempts[-1]["skip_reason"] = "tracking_arc_abstained"
                    continue
                flight_track = (
                    {frame: smoothed[frame] for frame in arc_scope["retained_frames"]}
                    if arc_scope["available"]
                    else smoothed
                )
                scope_decision = spatial_fit_scope(
                    camera_scope_points,
                    match_id,
                    clip,
                    float(start["frame"]),
                    float(end["frame"]),
                )
                sequence_attempts[-1]["camera_scope"] = scope_decision
                if scope_decision["decision"] == "withhold_spatial":
                    sequence_attempts[-1]["skip_reason"] = "nonstandard_camera_scope"
                    continue
                reuse_key = topology_flight_key(start, end)
                if reusable_fits is not None and reuse_key in reusable_fits:
                    reused = reusable_fits[reuse_key]
                    if reused is not None:
                        sequence_fits[index] = reused
                    sequence_attempts[-1]["reused_topology_fit"] = True
                    continue
                if anchor_first:
                    flight_anchor_row = build_point_anchors(
                        clip_key,
                        [sequence_attempts[-1]],
                        boundary_events,
                        flight_track,
                        camera,
                        time_priors=impact_time_priors,
                        bounce_witnesses=bounce_court_witnesses,
                    )["flights"][0]
                    fit = fit_anchor_first_shot(
                        index,
                        sequence,
                        flight_track,
                        players,
                        camera,
                        fps,
                        surface,
                        fit_max_nfev,
                        flight_anchor_row["anchors"],
                        observation_weights=observation_weights,
                        net_point_anchor=net_point_anchor,
                    )
                    sequence_attempts[-1]["anchor_first"] = {
                        "anchor_types": [row["type"] for row in flight_anchor_row["anchors"]],
                        "checkerboard_fit": "even_frames",
                        "checkerboard_score": "odd_frames",
                    }
                    growth_frame = (
                        terminal_residual_growth_frame(fit) if end.get("terminal") else None
                    )
                    if (
                        fit is not None
                        and growth_frame is not None
                        and growth_frame > float(start["frame"]) + MIN_TERMINAL_OBSERVATIONS
                        and growth_frame < float(end["frame"])
                    ):
                        trimmed_end = {
                            **end,
                            "frame": growth_frame,
                            "source": "post_fit_residual_growth",
                            "terminal_cutoff": {
                                **end.get("terminal_cutoff", {}),
                                "residual_growth_frame": growth_frame,
                            },
                        }
                        sequence[index + 1] = trimmed_end
                        end = trimmed_end
                        sequence_attempts[-1].update(
                            {
                                "end_frame": growth_frame,
                                "duration_seconds": float((growth_frame - start["frame"]) / fps),
                                "terminal_residual_growth_cut": growth_frame,
                            }
                        )
                        trimmed_track = {
                            frame: pixel
                            for frame, pixel in flight_track.items()
                            if frame <= growth_frame
                        }
                        trimmed_anchors = build_point_anchors(
                            clip_key,
                            [sequence_attempts[-1]],
                            boundary_events,
                            trimmed_track,
                            camera,
                            time_priors=impact_time_priors,
                            bounce_witnesses=bounce_court_witnesses,
                        )["flights"][0]
                        trimmed_fit = fit_anchor_first_shot(
                            index,
                            sequence,
                            trimmed_track,
                            players,
                            camera,
                            fps,
                            surface,
                            fit_max_nfev,
                            trimmed_anchors["anchors"],
                            observation_weights=observation_weights,
                            net_point_anchor=net_point_anchor,
                        )
                        if trimmed_fit is not None:
                            fit = trimmed_fit
                            sequence_attempts[-1]["anchor_first"]["anchor_types"] = [
                                row["type"] for row in trimmed_anchors["anchors"]
                            ]
                else:
                    fit = fit_shot(
                        index,
                        sequence,
                        flight_track,
                        players,
                        camera,
                        fps,
                        surface,
                        fit_max_nfev,
                        multi_start=fit_multi_start,
                        observation_weights=observation_weights,
                    )
                local_bounces = [
                    row
                    for row in automatic_bounce_candidates
                    if start["frame"] < row["frame"] < end["frame"]
                ]
                selected_bounce = (
                    None
                    if anchor_first
                    else select_reconstruction_bounce_anchor(
                        local_bounces,
                        fit.bounces if fit is not None else [],
                        fps=fps,
                    )
                )
                if selected_bounce is not None:
                    recovered = recover_hard_bounce_shot(
                        index,
                        sequence,
                        flight_track,
                        players,
                        camera,
                        fps,
                        surface,
                        selected_bounce,
                        fit_max_nfev,
                    )
                    if recovered is not None and (
                        fit is None
                        or hard_bounce_recovery_score(
                            recovered,
                            index,
                            sequence,
                            flight_track,
                            players,
                            camera,
                            fps,
                            surface,
                        )
                        <= hard_bounce_recovery_score(
                            fit,
                            index,
                            sequence,
                            flight_track,
                            players,
                            camera,
                            fps,
                            surface,
                        )
                    ):
                        fit = recovered
                        sequence_attempts[-1]["bounce_anchor"] = {
                            key: (value.tolist() if isinstance(value, np.ndarray) else value)
                            for key, value in selected_bounce.items()
                            if key != "x"
                        }
                if fit is not None:
                    sequence_fits[index] = fit
            return sequence_fits, sequence_attempts

        topology_report = {
            "enabled": event_topology_branches,
            "sequence_branches": topology_sequence_branches,
            "decisive": False,
            "margin": None,
            "selected": "contacts_baseline",
            "selection_reason": "disabled",
            "scoring_stage": "disabled",
            "inserted": [],
            "omitted": [],
            "retimed": [],
            "alternatives": [],
        }
        selected_topology_fits = None
        selected_topology_attempts = None
        if event_topology_branches and smoothed:

            def topology_score(
                candidate: dict, candidate_fits: dict, candidate_attempts: list
            ) -> dict:
                endpoint_errors = []
                for index, fit in candidate_fits.items():
                    start = candidate["contacts"][index]
                    end = candidate["contacts"][index + 1]
                    _, start_diagnostic = contact_image_residual(
                        fit.theta[:3], start, smoothed, camera, observation_weights
                    )
                    if start_diagnostic["error_px"] is not None:
                        endpoint_errors.append(start_diagnostic["error_px"])
                    if not end.get("terminal"):
                        end_xyz = fit.state(end["frame"], fps, surface)[0]
                        _, end_diagnostic = contact_image_residual(
                            end_xyz, end, smoothed, camera, observation_weights
                        )
                        if end_diagnostic["error_px"] is not None:
                            endpoint_errors.append(end_diagnostic["error_px"])
                return score_contact_topology_branch(
                    candidate,
                    attempted=len(candidate_attempts),
                    solved=len(candidate_fits),
                    weighted_rms_px=[
                        float(getattr(fit, "_weighted_rms_px", fit.rms_px))
                        for fit in candidate_fits.values()
                    ],
                    endpoint_errors_px=endpoint_errors,
                )

            def jointly_refine_topology_candidate(candidate: dict, initial_fits: dict) -> dict:
                terminal = {
                    index: fit
                    for index, fit in initial_fits.items()
                    if candidate["contacts"][index + 1].get("terminal")
                }
                core = {index: fit for index, fit in initial_fits.items() if index not in terminal}
                if not core:
                    return dict(initial_fits)
                initial_positions, _ = fuse_contacts(
                    candidate["contacts"],
                    core,
                    players,
                    smoothed,
                    camera,
                    fps,
                    surface,
                )
                try:
                    refined_positions, refined = jointly_refine_all(
                        candidate["contacts"],
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                        core,
                        initial_positions,
                        min(max_nfev, 30),
                        [],
                        observation_weights=observation_weights,
                    )
                except (ValueError, FloatingPointError):
                    return {}
                for index, initial in terminal.items():
                    fixed_start = refined_positions.get(
                        index,
                        initial_positions.get(index, initial.theta[:3]),
                    )
                    terminal_fit = fit_shot(
                        index,
                        candidate["contacts"],
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                        min(max_nfev, 30),
                        fixed_start_anchor=fixed_start,
                        seed=initial.theta,
                        multi_start=False,
                        observation_weights=observation_weights,
                    )
                    if terminal_fit is not None:
                        refined[index] = terminal_fit
                return refined

            evaluated_topologies = []
            topology_candidates = build_contact_topology_branches(
                contacts,
                smoothed,
                fps,
                boxes=boxes,
                observed_frames=set(track),
                max_branches=topology_branch_width,
                include_sequences=topology_sequence_branches,
                span_ranges=active_spans,
                event_hypotheses=(automatic_hypotheses or {}).get(clip_key, []),
            )
            if terminal_flights:
                for candidate in topology_candidates:
                    candidate["contacts"] = append_terminal_contacts(
                        candidate["contacts"],
                        smoothed,
                        active_spans,
                        fps,
                        observation_weights,
                        boundary_events,
                        bridge_out_of_frame_gaps=True,
                    )
            topology_candidates.sort(key=lambda row: row["branch_id"] != "contacts_baseline")
            baseline_fit_cache = {}
            for candidate in topology_candidates:
                candidate_fits, candidate_attempts = initial_fit_sequence(
                    candidate["contacts"],
                    fit_max_nfev=min(max_nfev, 12),
                    fit_multi_start=False,
                    reusable_fits=baseline_fit_cache,
                )
                if candidate["branch_id"] == "contacts_baseline":
                    for attempt in candidate_attempts:
                        index = attempt["flight_index"]
                        start = candidate["contacts"][index]
                        end = candidate["contacts"][index + 1]
                        baseline_fit_cache[topology_flight_key(start, end)] = candidate_fits.get(
                            index
                        )
                score = topology_score(candidate, candidate_fits, candidate_attempts)
                evaluated_topologies.append(
                    {
                        **candidate,
                        "score": score,
                        "_initial_fits": candidate_fits,
                        "_attempts": candidate_attempts,
                        "reused_initial_flights": sum(
                            bool(row.get("reused_topology_fit")) for row in candidate_attempts
                        ),
                    }
                )
            selection_topologies = evaluated_topologies
            post_joint_scores = {}
            post_joint_fits = {}
            topology_scoring_stage = "initial_flight_fit"
            if topology_post_joint_scoring:
                shortlisted = shortlist_contact_topology_branches(
                    evaluated_topologies,
                    width=topology_joint_width,
                )
                if len(shortlisted) > 1:
                    selection_topologies = []
                    for candidate in shortlisted:
                        refined_fits = jointly_refine_topology_candidate(
                            candidate,
                            candidate["_initial_fits"],
                        )
                        post_joint_fits[candidate["branch_id"]] = refined_fits
                        score = topology_score(candidate, refined_fits, candidate["_attempts"])
                        post_joint_scores[candidate["branch_id"]] = score
                        selection_topologies.append({**candidate, "score": score})
                    topology_scoring_stage = "post_joint_shortlist"
                else:
                    selection_topologies = shortlisted
                    topology_scoring_stage = "initial_no_viable_alternative"
            (
                selected_topology,
                decisive,
                topology_margin,
                topology_selection_reason,
            ) = select_contact_topology_branch(
                selection_topologies,
                minimum_margin=topology_branch_margin,
            )
            contacts = selected_topology["contacts"]
            selected_topology_fits = post_joint_fits.get(
                selected_topology["branch_id"], selected_topology["_initial_fits"]
            )
            selected_topology_attempts = selected_topology["_attempts"]
            topology_report = {
                "enabled": True,
                "decisive": decisive,
                "margin": topology_margin,
                "selected": selected_topology["branch_id"],
                "selection_reason": topology_selection_reason,
                "scoring_stage": topology_scoring_stage,
                "shortlist_width": topology_joint_width if topology_post_joint_scoring else None,
                "selected_initial_fit_reused": max_nfev <= 30 and not multi_start,
                "inserted": selected_topology["inserted"],
                "omitted": selected_topology["omitted"],
                "retimed": selected_topology["retimed"],
                "alternatives": [
                    {
                        "branch_id": row["branch_id"],
                        "omitted": row["omitted"],
                        "inserted": row["inserted"],
                        "retimed": row["retimed"],
                        "reused_initial_flights": row["reused_initial_flights"],
                        "initial_score": row["score"],
                        "post_joint_score": post_joint_scores.get(row["branch_id"]),
                        "score": post_joint_scores.get(row["branch_id"], row["score"]),
                        "score_stage": (
                            "post_joint"
                            if row["branch_id"] in post_joint_scores
                            else "initial_flight_fit"
                        ),
                    }
                    for row in sorted(
                        evaluated_topologies,
                        key=lambda row: post_joint_scores.get(row["branch_id"], row["score"])[
                            "total"
                        ],
                    )
                ],
            }

        if (
            selected_topology_fits is not None
            and selected_topology_attempts is not None
            and max_nfev <= 30
            and not multi_start
        ):
            raw_fits = dict(selected_topology_fits)
            flight_attempts = list(selected_topology_attempts)
        else:
            raw_fits, flight_attempts = initial_fit_sequence(
                contacts,
                fit_max_nfev=max_nfev,
                fit_multi_start=multi_start,
            )
        anchor_first_contexts = {
            index: context
            for index, fit in raw_fits.items()
            if (context := getattr(fit, "_anchor_first_context", None)) is not None
        }
        anchors_artifact = build_point_anchors(
            clip_key,
            flight_attempts,
            boundary_events,
            smoothed,
            camera,
            time_priors=impact_time_priors,
            bounce_witnesses=bounce_court_witnesses,
        )
        anchors_path = None
        if anchors_output_root is not None:
            anchors_path = write_point_anchors(
                anchors_output_root,
                match_id,
                clip,
                anchors_artifact,
            )
        initial_positions, _ = fuse_contacts(
            contacts,
            raw_fits,
            players,
            smoothed,
            camera,
            fps,
            surface,
        )
        refined_positions = dict(initial_positions)
        terminal_indices = {index for index in raw_fits if contacts[index + 1].get("terminal")}
        bounce_hypotheses = []
        hard_bounce_requirements = []
        probabilistic_bounce_candidates = list(automatic_bounce_candidates)
        joint_refine_fallbacks = []
        branch_report = {
            "enabled": whole_point_branches,
            "decisive": None,
            "margin": None,
            "selected": None,
            "alternatives": [],
        }
        if joint_refine and raw_fits:
            terminal_initial = {index: raw_fits[index] for index in terminal_indices}
            joint_terminal_indices = {
                index
                for index in terminal_indices
                if any(
                    contacts[index]["frame"] < row["frame"] < contacts[index + 1]["frame"]
                    and float(row.get("probability") or 1.0)
                    >= MIN_TERMINAL_BOUNCE_ANCHOR_PROBABILITY
                    for row in [*boundary_events, *automatic_bounce_candidates]
                    if row.get("event_type", "bounce") == "bounce"
                )
            }
            core_fits = {
                index: fit
                for index, fit in raw_fits.items()
                if index not in terminal_indices or index in joint_terminal_indices
            }
            initial_positions, _ = fuse_contacts(
                contacts,
                core_fits,
                players,
                smoothed,
                camera,
                fps,
                surface,
            )
            refined_positions = dict(initial_positions)
            if anchor_bounce_geometry:
                for event in boundary_events:
                    if event["event_type"] != "bounce":
                        continue
                    point = nearest_track_point(smoothed, event["frame"])
                    if point is None:
                        continue
                    court_xy = court_point(camera, point)
                    bounce_hypotheses.append(
                        {
                            **event,
                            "x": np.r_[court_xy, R_BALL],
                            "court_xy": court_xy,
                            "probability": 1.0,
                            "sigma_frame": 0.20,
                            "sigma_xy_m": 0.05,
                            "hard_geometry": True,
                            "source": "ground_plane_bounce_geometry",
                        }
                    )
                hard_bounce_requirements = [
                    dict(row) for row in bounce_hypotheses if row.get("hard_geometry")
                ]
                for anchor in hard_bounce_requirements:
                    enclosing = [
                        index
                        for index in range(len(contacts) - 1)
                        if contacts[index]["frame"] < anchor["frame"] < contacts[index + 1]["frame"]
                    ]
                    if len(enclosing) != 1:
                        continue
                    index = enclosing[0]
                    if index in core_fits or index in terminal_initial:
                        continue
                    recovered = recover_hard_bounce_shot(
                        index,
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                        anchor,
                        max_nfev,
                    )
                    if recovered is None:
                        continue
                    raw_fits[index] = recovered
                    if contacts[index + 1].get("terminal"):
                        terminal_initial[index] = recovered
                    else:
                        core_fits[index] = recovered
                for index in range(len(contacts) - 1):
                    if contacts[index + 1].get("terminal") or index in core_fits:
                        continue
                    initializer = initialize_missing_shot(
                        index,
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                    )
                    if initializer is not None:
                        core_fits[index] = initializer
                        raw_fits[index] = initializer
                if core_fits:
                    initial_positions, _ = fuse_contacts(
                        contacts,
                        core_fits,
                        players,
                        smoothed,
                        camera,
                        fps,
                        surface,
                    )
            if uncertainty_hypotheses is not None:
                for row in hypotheses_by_point.get(clip_key, []):
                    probability = event_type_probability(row, "bounce")
                    if probability < 0.10:
                        continue
                    image = np.array([float(row["img_x_native"]), float(row["img_y_native"])])
                    probabilistic_bounce_candidates.append(
                        {
                            "frame": float(row["frame"]),
                            "x": np.r_[court_point(camera, image), R_BALL],
                            "court_xy": court_point(camera, image),
                            "probability": probability,
                            "source": "probabilistic_event_hypothesis",
                        }
                    )
            if probabilistic_bounce_candidates:
                for index, fit in sorted(core_fits.items()):
                    start = contacts[index]["frame"]
                    end = contacts[index + 1]["frame"]
                    local = [
                        row for row in probabilistic_bounce_candidates if start < row["frame"] < end
                    ]
                    selected = select_physics_compatible_bounce(
                        local,
                        fit.bounces,
                        fps=fps,
                    )
                    if selected is not None:
                        bounce_hypotheses.append(selected)
            if discrete_bounce_branches:
                for index, initial in sorted(core_fits.items()):
                    anchor = bounce_for_shot(
                        bounce_hypotheses,
                        contacts[index]["frame"],
                        contacts[index + 1]["frame"],
                    )
                    if anchor is None:
                        continue
                    branched = shoot_shot(
                        index,
                        contacts,
                        smoothed,
                        camera,
                        fps,
                        surface,
                        (
                            initial_positions[index],
                            initial_positions[index + 1],
                        ),
                        initial,
                        max_nfev,
                        bounce_anchor=anchor,
                    )
                    if branched is not None:
                        core_fits[index] = branched
            if whole_point_branches and core_fits:
                gender = "women" if "_w_" in match_id else "men"
                per_shot_options = {}
                for index, fit in sorted(core_fits.items()):
                    start = contacts[index]["frame"]
                    end = contacts[index + 1]["frame"]
                    hard_anchor = next(
                        (
                            row
                            for row in bounce_hypotheses
                            if row.get("hard_geometry") and start < row["frame"] < end
                        ),
                        None,
                    )
                    local_candidates = [
                        row for row in probabilistic_bounce_candidates if start < row["frame"] < end
                    ]
                    fitted_components_rpm = spin_component_rpm(fit.theta)
                    per_shot_options[index] = {
                        "start_side": contacts[index]["side"],
                        "end_side": contacts[index + 1]["side"],
                        "spin": shot_spin_options(
                            gender=gender,
                            phase=contacts[index]["phase"],
                            speed_kmh=float(np.linalg.norm(fit.theta[3:6]) * 3.6),
                            fitted_signed_rpm=float(fitted_components_rpm[0]),
                            fitted_sidespin_rpm=float(fitted_components_rpm[1]),
                            fitted_rifle_rpm=float(fitted_components_rpm[2]),
                            apex_m=float(np.max(fit.xs_obs[:, 2])),
                            duration_s=(end - start) / fps,
                            pose_slice_probability=pose_slice_probability(
                                pose_rows,
                                side=contacts[index]["side"],
                                frame=start,
                                fps=fps,
                                ball_pixel=nearest_track_point(
                                    smoothed,
                                    start,
                                ),
                            ),
                        ),
                        "bounce": bounce_options(
                            local_candidates,
                            start_side=contacts[index]["side"],
                            end_side=contacts[index + 1]["side"],
                            hard_anchor=hard_anchor,
                        ),
                    }
                branch_candidates = build_point_branches(
                    per_shot_options,
                    beam_width=branch_width,
                )
                refined_branches = []
                for interpretation in branch_candidates:
                    selected_anchors = [
                        shot["bounce_anchor"]
                        for shot in interpretation["shots"].values()
                        if shot["bounce_anchor"] is not None
                    ]
                    try:
                        branch_positions, branch_fits = jointly_refine_all(
                            contacts,
                            smoothed,
                            players,
                            camera,
                            fps,
                            surface,
                            core_fits,
                            initial_positions,
                            max_nfev,
                            selected_anchors,
                            observation_weights=observation_weights,
                            spin_priors=interpretation["shots"],
                        )
                    except (ValueError, FloatingPointError):
                        continue
                    if len(branch_fits) != len(core_fits):
                        continue
                    branch_gaps = []
                    for first, second in zip(
                        sorted(branch_fits),
                        sorted(branch_fits)[1:],
                    ):
                        if second != first + 1:
                            continue
                        end_state = branch_fits[first].state(
                            contacts[first + 1]["frame"],
                            fps,
                            surface,
                        )[0]
                        branch_gaps.append(
                            float(np.linalg.norm(end_state - branch_fits[second].theta[:3]))
                        )
                    score = refined_branch_score(
                        fits=branch_fits,
                        branch=interpretation,
                        junction_gaps_m=branch_gaps,
                    )
                    refined_branches.append(
                        {
                            "interpretation": interpretation,
                            "positions": branch_positions,
                            "fits": branch_fits,
                            "score": score,
                        }
                    )
                refined_branches.sort(key=lambda row: row["score"]["total"])
                decisive, margin = branch_is_decisive(
                    refined_branches,
                    minimum_margin=branch_margin,
                )
                branch_report.update(
                    {
                        "decisive": decisive,
                        "margin": margin,
                        "selected": (
                            refined_branches[0]["interpretation"]["branch_id"]
                            if refined_branches
                            else None
                        ),
                        "alternatives": [
                            {
                                "branch_id": row["interpretation"]["branch_id"],
                                "score": row["score"],
                                "shots": {
                                    str(index): {
                                        key: value
                                        for key, value in shot.items()
                                        if key != "bounce_anchor"
                                    }
                                    for index, shot in row["interpretation"]["shots"].items()
                                },
                            }
                            for row in refined_branches
                        ],
                    }
                )
                if refined_branches:
                    refined_positions = refined_branches[0]["positions"]
                    raw_fits = refined_branches[0]["fits"]
                    selected_shots = refined_branches[0]["interpretation"]["shots"]
                    for index, fit in raw_fits.items():
                        if index in selected_shots:
                            object.__setattr__(
                                fit,
                                "_spin_interpretation",
                                selected_shots[index],
                            )
                    bounce_hypotheses = [
                        shot["bounce_anchor"]
                        for shot in refined_branches[0]["interpretation"]["shots"].values()
                        if shot["bounce_anchor"] is not None
                    ]
                else:
                    try:
                        refined_positions, raw_fits = jointly_refine_all(
                            contacts,
                            smoothed,
                            players,
                            camera,
                            fps,
                            surface,
                            core_fits,
                            initial_positions,
                            max_nfev,
                            hard_bounce_requirements,
                            observation_weights=observation_weights,
                        )
                    except (ValueError, FloatingPointError):
                        refined_positions, raw_fits = initial_positions, {}
                    joint_refine_fallbacks.append(
                        {
                            "reason": "whole_point_branch_set_incomplete",
                            "detail": "retained only independently valid hard-anchor refinements",
                            "retained_flights": len(raw_fits),
                        }
                    )
            else:
                try:
                    refined_positions, raw_fits = jointly_refine_all(
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                        core_fits,
                        initial_positions,
                        max_nfev,
                        bounce_hypotheses,
                        observation_weights=observation_weights,
                    )
                except (ValueError, FloatingPointError) as error:
                    joint_refine_fallbacks.append(
                        {
                            "reason": "invalid_bounce_anchor_set",
                            "detail": str(error),
                            "rejected_bounces": len(bounce_hypotheses),
                        }
                    )
                    bounce_hypotheses = []
                    try:
                        refined_positions, raw_fits = jointly_refine_all(
                            contacts,
                            smoothed,
                            players,
                            camera,
                            fps,
                            surface,
                            core_fits,
                            initial_positions,
                            max_nfev,
                            [],
                            observation_weights=observation_weights,
                        )
                    except (ValueError, FloatingPointError) as fallback_error:
                        joint_refine_fallbacks.append(
                            {
                                "reason": "joint_refine_failed",
                                "detail": str(fallback_error),
                            }
                        )
                        refined_positions, raw_fits = initial_positions, core_fits
            for index, initial in core_fits.items():
                if index not in raw_fits and (
                    getattr(initial, "_bounce_node_split", None) is not None
                    or getattr(initial, "_ballistic_bounce_node", None) is not None
                ):
                    raw_fits[index] = initial
                    joint_refine_fallbacks.append(
                        {
                            "reason": "bounded_bounce_node_retained",
                            "flight_index": index,
                            "detail": (
                                "global refinement rejected the flight; retained the independently "
                                "screened exact-node fit"
                            ),
                        }
                    )
            for index, initial in terminal_initial.items():
                joint_terminal = raw_fits.get(index) if index in joint_terminal_indices else None
                fixed_start = refined_positions.get(
                    index,
                    initial_positions[index],
                )
                hard_anchor = bounce_for_shot(
                    hard_bounce_requirements,
                    contacts[index]["frame"],
                    contacts[index + 1]["frame"],
                )
                if hard_anchor is not None:
                    terminal = recover_hard_bounce_shot(
                        index,
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                        hard_anchor,
                        max_nfev,
                        allow_initialization_only=False,
                        fixed_start_anchor=fixed_start,
                    )
                    if terminal is None and any(
                        row.get("event_type") == "net_hit"
                        and contacts[index]["frame"] < row["frame"] < contacts[index + 1]["frame"]
                        for row in boundary_events
                    ):
                        terminal = recover_hard_bounce_shot(
                            index,
                            contacts,
                            smoothed,
                            players,
                            camera,
                            fps,
                            surface,
                            hard_anchor,
                            max_nfev,
                            fixed_start_anchor=fixed_start,
                        )
                else:
                    terminal = fit_shot(
                        index,
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                        max_nfev,
                        fixed_start_anchor=fixed_start,
                        seed=initial.theta,
                        multi_start=False,
                        observation_weights=observation_weights,
                    )
                if terminal is not None and joint_terminal is not None:
                    if hard_bounce_recovery_score(
                        joint_terminal,
                        index,
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                    ) <= hard_bounce_recovery_score(
                        terminal,
                        index,
                        contacts,
                        smoothed,
                        players,
                        camera,
                        fps,
                        surface,
                    ):
                        terminal = joint_terminal
                if terminal is not None:
                    raw_fits[index] = terminal
            endpoint_refits = {}
            contact_anchor_diagnostics = {}
        else:
            endpoint_refits = {}
            contact_anchor_diagnostics = {}
        for anchor in hard_bounce_requirements:
            enclosing = [
                index
                for index in range(len(contacts) - 1)
                if contacts[index]["frame"] < anchor["frame"] < contacts[index + 1]["frame"]
            ]
            if len(enclosing) != 1:
                continue
            index = enclosing[0]
            recovered = recover_hard_bounce_shot(
                index,
                contacts,
                smoothed,
                players,
                camera,
                fps,
                surface,
                anchor,
                max_nfev,
                fixed_start_anchor=refined_positions.get(index),
                fixed_end_anchor=(
                    None
                    if contacts[index + 1].get("terminal")
                    else refined_positions.get(index + 1)
                ),
            )
            if recovered is None or getattr(recovered, "_initialization_only", False):
                continue
            existing = raw_fits.get(index)
            if existing is not None:
                continue
            raw_fits[index] = recovered
            joint_refine_fallbacks.append(
                {
                    "reason": "hard_bounce_endpoint_aware_branch_selected",
                    "flight_index": index,
                    "detail": "selected exact-node branch after endpoint-aware comparison",
                }
            )
        for anchor in hard_bounce_requirements:
            enclosing = [
                index
                for index in range(len(contacts) - 1)
                if contacts[index]["frame"] < anchor["frame"] < contacts[index + 1]["frame"]
            ]
            if len(enclosing) != 1 or enclosing[0] in raw_fits:
                continue
            index = enclosing[0]
            recovered = recover_hard_bounce_shot(
                index,
                contacts,
                smoothed,
                players,
                camera,
                fps,
                surface,
                anchor,
                max_nfev,
                fixed_start_anchor=refined_positions.get(index),
                fixed_end_anchor=(
                    None
                    if contacts[index + 1].get("terminal")
                    else refined_positions.get(index + 1)
                ),
            )
            if recovered is not None and not getattr(recovered, "_initialization_only", False):
                raw_fits[index] = recovered
                joint_refine_fallbacks.append(
                    {
                        "reason": "post_joint_hard_bounce_recovery",
                        "flight_index": index,
                        "detail": "recovered after global refinement discarded the initial fit",
                    }
                )
        contact_ray_fit_diagnostics = []
        if anchor_first and shared_contact_fit and raw_fits:
            for index, fit in raw_fits.items():
                context = anchor_first_contexts.get(index)
                if context is not None and getattr(fit, "_anchor_first_context", None) is None:
                    object.__setattr__(fit, "_anchor_first_context", context)
            raw_fits = refine_anchor_first_point(raw_fits, contact_ray_fit_diagnostics)
            apply_shared_contact_times(contacts, flight_attempts, raw_fits, camera, fps)
            refined_positions, _ = fuse_contacts(
                contacts,
                raw_fits,
                players,
                smoothed,
                camera,
                fps,
                surface,
            )
        if exact_contact_reprojection:
            raw_fits, endpoint_refits, contact_anchor_diagnostics = refit_exact_contact_rays(
                raw_fits,
                contacts,
                refined_positions,
                initial_positions,
                smoothed,
                players,
                camera,
                fps,
                surface,
                max_nfev,
                bounce_hypotheses,
                observation_weights,
                pose_rows,
                physical_motion,
            )
        net_collision_integrity = {
            "required": 0,
            "satisfied": 0,
            "all_satisfied": True,
            "nodes": [],
            "violations": [],
            "role": "post_fit_annotation_only",
        }
        for attempt in flight_attempts:
            index = int(attempt["flight_index"])
            net_events = [
                row
                for row in attempt.get("intermediate_events", [])
                if row.get("event_type") == "net_hit"
            ]
            if not net_events:
                continue
            net_collision_integrity["required"] += len(net_events)
            # A net-hit emission is neither an observed crossing time nor a depth
            # witness.  Preserve it for downstream review, but never split, refit,
            # reject, or otherwise alter the physical flight with it.
            net_collision_integrity["satisfied"] += len(net_events)
            net_collision_integrity["nodes"].append(
                {
                    "flight_index": index,
                    "frames": [float(row["frame"]) for row in net_events],
                    "fit_available": index in raw_fits,
                    "role": "event_annotation_only",
                }
            )
        net_collision_integrity["all_satisfied"] = not net_collision_integrity["violations"]
        unrefined_initializers = sorted(
            index for index, fit in raw_fits.items() if getattr(fit, "_initialization_only", False)
        )
        for index in unrefined_initializers:
            raw_fits.pop(index, None)
        unscored_recovery_fits = restrict_fits_to_attempts(raw_fits, flight_attempts)
        joint_refine_fallbacks.extend(
            {
                "reason": "recovery_outside_flight_attempt_scope",
                "flight_index": index,
                "detail": "discarded before scoring because no canonical attempt exists",
            }
            for index in unscored_recovery_fits
        )
        hard_bounce_integrity = verify_hard_bounce_anchors(
            raw_fits,
            contacts,
            hard_bounce_requirements,
        )
        for index in hard_bounce_integrity["violating_flights"]:
            raw_fits.pop(index, None)
        fits = []
        attempts_by_index = {row["flight_index"]: row for row in flight_attempts}
        for violation in hard_bounce_integrity["violations"]:
            index = violation.get("flight_index")
            if index in attempts_by_index:
                attempts_by_index[index]["skip_reason"] = violation["reason"]
                attempts_by_index[index]["hard_bounce_violation"] = violation
        for violation in net_collision_integrity["violations"]:
            index = violation.get("flight_index")
            if index in attempts_by_index:
                attempts_by_index[index]["skip_reason"] = violation["reason"]
                attempts_by_index[index]["net_collision_violation"] = violation
        for index in unrefined_initializers:
            if index in attempts_by_index:
                attempts_by_index[index]["skip_reason"] = "joint_initializer_not_refined"
        for index, fit in sorted(raw_fits.items()):
            final_frame = float(contacts[index + 1]["frame"])
            terminal_completion = None
            if subframe_anchors and contacts[index + 1].get("terminal"):
                terminal_completion = complete_second_bounce(fit, contacts[index + 1], fps, surface)
                if terminal_completion["status"] == "completed":
                    candidate_end = float(terminal_completion["end_frame"])
                    if camera.quality_at(candidate_end).get("reliable") is True:
                        final_frame = candidate_end
                    else:
                        terminal_completion = {
                            **terminal_completion,
                            "status": "abstain",
                            "reason": "terminal_camera_unreliable",
                        }
            compact = compact_fit(
                fit,
                fps,
                surface,
                start_frame=contacts[index]["frame"],
                end_frame=final_frame,
            )
            compact["flight_index"] = index
            compact["start_frame"] = contacts[index]["frame"]
            compact["end_frame"] = final_frame
            if terminal_completion is not None:
                compact["terminal_completion"] = terminal_completion
                if terminal_completion["status"] == "completed":
                    compact["bounces"] = [
                        row
                        for row in compact["bounces"]
                        if not row.get("termination_anchor")
                        and not row.get("regime", "").startswith("terminal_measured_")
                    ] + [terminal_completion["bounce"]]
                    if (
                        not compact["trajectory"]
                        or abs(float(compact["trajectory"][-1]["frame"]) - final_frame) > 1e-8
                    ):
                        compact["trajectory"].append(
                            {
                                "frame": final_frame,
                                "xyz": terminal_completion["end_xyz"],
                                "velocity": terminal_completion["incoming_velocity"],
                                "spin": terminal_completion["bounce"]["w_in"],
                                "sample_kind": "physical_terminal_event",
                                "source_picture": False,
                            }
                        )
                    attempt = attempts_by_index[index]
                    attempt["terminal_completion"] = terminal_completion
                    attempt["end_frame"] = final_frame
                    attempt["duration_seconds"] = (final_frame - compact["start_frame"]) / fps
                    attempt["camera_end"] = camera.quality_at(final_frame)
            compact["observation_coverage"] = float(
                compact["observations"]
                / max(1.0, compact["end_frame"] - compact["start_frame"] + 1.0)
            )
            compact["terminal_end"] = bool(contacts[index + 1].get("terminal"))
            compact["camera_start"] = camera.quality_at(contacts[index]["frame"])
            compact["camera_end"] = camera.quality_at(final_frame)
            for held_out in compact.get("held_out_frame_errors", []):
                quality = camera.quality_at(held_out["frame"])
                held_out["camera_reliable"] = bool(quality.get("reliable"))
                held_out["camera_source"] = str(quality.get("source", "unknown"))
                held_out["camera_frame_scope"] = str(quality.get("frame_scope", "unknown"))
            compact["end_xyz"] = fit.state(
                final_frame,
                fps,
                surface,
            )[0].tolist()
            # ``theta`` is the state at the frame this flight was fitted from.  When a shared
            # contact moved the boundary to its own sub-frame time, the start of the flight is
            # that instant and not the emitted frame, and the junction is measured between the
            # two flights there.
            fitted_start_frame = float(getattr(fit, "_f0", contacts[index]["frame"]))
            if abs(float(contacts[index]["frame"]) - fitted_start_frame) > 1e-9:
                compact["start_xyz"] = fit.state(
                    contacts[index]["frame"],
                    fps,
                    surface,
                )[0].tolist()
            start_contact_reprojection = point_contact_reprojection(
                index,
                contacts,
                raw_fits,
                smoothed,
                camera,
                observation_weights,
                fps,
                surface,
            )
            end_contact_reprojection = point_contact_reprojection(
                index + 1,
                contacts,
                raw_fits,
                smoothed,
                camera,
                observation_weights,
                fps,
                surface,
            )
            compact["start_contact_reprojection_px"] = start_contact_reprojection["error_px"]
            compact["end_contact_reprojection_px"] = end_contact_reprojection["error_px"]
            compact["start_contact_reprojection"] = start_contact_reprojection
            compact["end_contact_reprojection"] = end_contact_reprojection
            compact["start_contact_anchor"] = contact_anchor_diagnostics.get(index)
            compact["end_contact_anchor"] = contact_anchor_diagnostics.get(index + 1)
            for prefix, contact_index, xyz_key in (
                ("start", index, "start_xyz"),
                ("end", index + 1, "end_xyz"),
            ):
                diagnostic = contact_anchor_diagnostics.get(contact_index, {})
                apparent_height = optional_float(diagnostic.get("apparent_height_m"))
                apparent_confidence = optional_float(diagnostic.get("apparent_height_confidence"))
                if (
                    apparent_height is not None
                    and apparent_confidence is not None
                    and apparent_confidence >= 0.45
                    and compact.get(xyz_key) is not None
                ):
                    compact[f"{prefix}_contact_apparent_height_m"] = apparent_height
                    compact[f"{prefix}_contact_apparent_height_confidence"] = apparent_confidence
                    compact[f"{prefix}_contact_apparent_height_error_m"] = abs(
                        float(compact[xyz_key][2]) - apparent_height
                    )
            start_player = nearest(
                players.get(contacts[index]["side"], {}),
                contacts[index]["frame"],
            )
            end_player = nearest(
                players.get(contacts[index + 1]["side"], {}),
                contacts[index + 1]["frame"],
            )
            compact["start_player_distance_m"] = (
                float(np.linalg.norm(np.asarray(compact["start_xyz"][:2]) - start_player))
                if start_player is not None and not contacts[index].get("terminal")
                else None
            )
            compact["end_player_distance_m"] = (
                float(np.linalg.norm(np.asarray(compact["end_xyz"][:2]) - end_player))
                if end_player is not None and not contacts[index + 1].get("terminal")
                else None
            )
            fits.append(compact)
            attempts_by_index[index]["solved"] = True
            attempts_by_index[index]["fit_index"] = len(fits) - 1
            attempts_by_index[index]["endpoint_refit"] = endpoint_refits.get(
                index,
                {"status": "not_requested"},
            )
        attempted = len(flight_attempts)
        decision, reasons = point_gate(
            active_valid=bool(active_row["point_valid"]),
            attempted=attempted,
            fits=fits,
            smoothing=smoothing,
            camera_valid=bool(camera_calibration["accepted"]),
            unresolved_contact_sides=sum(
                contact["side"] == "unknown" for contact in contacts if not contact.get("terminal")
            ),
            fps=fps,
        )
        if whole_point_branches and not branch_report["decisive"]:
            decision = "hold"
            reasons = [*reasons, "physics_branch_ambiguous"]
        if not hard_bounce_integrity["all_satisfied"]:
            decision = "hold"
            reasons = [*reasons, "hard_bounce_anchor_violation"]
        if unrefined_initializers:
            decision = "hold"
            reasons = [*reasons, "joint_initializer_not_refined"]
        junction_evidence(fits, raw_fits, contacts, fps, surface)
        gaps = junction_gaps(fits)
        terminal_flight_count = sum(fit["terminal_end"] for fit in fits)
        fits_by_flight = {int(fit["flight_index"]): fit for fit in fits}
        flight_gate_context = {
            "active_play_valid": bool(active_row["point_valid"]),
            "smoothing": smoothing,
            "interpretation_branches": branch_report,
        }
        all_shots_accepted = len(fits) == attempted and all(
            classify_flight(
                flight_gate_context,
                attempt,
                fits_by_flight.get(int(attempt["flight_index"])),
            )[0]
            == "provisional_valid"
            for attempt in flight_attempts
        )
        terminal_coverage = terminal_coverage_report(contacts, fits)
        complete_candidate, complete_reasons = complete_point_gate(
            point_decision=decision,
            attempted=attempted,
            solved=len(fits),
            terminal_flights=terminal_flight_count,
            all_shots_accepted=all_shots_accepted,
            terminal_coverage=terminal_coverage,
        )
        point_gate_decision = decision
        point_gate_reasons = list(reasons)
        decision, reasons = finalize_complete_point_decision(
            decision,
            reasons,
            complete_candidate,
            complete_reasons,
        )
        screen_decision, screen_reasons = screening_gate(
            active_valid=bool(active_row["point_valid"]),
            attempted=attempted,
            fits=fits,
            smoothing=smoothing,
            camera_valid=bool(camera_calibration["accepted"]),
        )
        return {
            "match_id": match_id,
            "audit_root_name": audit_root.name,
            "clip": clip,
            "point": clip_key,
            "surface": surface,
            "fps": fps,
            "cadence_identity": cadence_identity,
            "contact_frame_refinements": [
                {
                    "event_frame": contact.get("event_frame", contact["frame"]),
                    "physical_frame": contact["frame"],
                    "observation_frame": contact.get("image_observation_frame"),
                    "observation_source": contact.get("image_observation_source"),
                    "observation_pixel": contact.get("image_observation_override"),
                    "refinement": contact["contact_frame_refinement"],
                }
                for contact in contacts
                if "contact_frame_refinement" in contact
            ],
            "decision": decision,
            "reasons": reasons,
            "point_gate": {
                "decision": point_gate_decision,
                "reasons": point_gate_reasons,
            },
            "screen_decision": screen_decision,
            "screen_reasons": screen_reasons,
            "active_play_valid": bool(active_row["point_valid"]),
            "active_spans": [list(map(float, span)) for span in active_spans],
            "camera_calibration": camera_calibration,
            "attempted_shots": attempted,
            "solved_shots": len(fits),
            "terminal_flights": terminal_flight_count,
            "terminal_coverage": terminal_coverage,
            "complete_point_gate": {
                "accepted": complete_candidate,
                "reasons": complete_reasons,
                "contract": (
                    "all attempted shots solved and individually accepted, exactly one terminal "
                    "flight covering an explicit physical ending, and the S6 point gate "
                    "retained the reconstruction"
                ),
            },
            "flight_attempts": flight_attempts,
            "flight_anchors": {
                "schema": anchors_artifact["schema"],
                "path": str(anchors_path) if anchors_path is not None else None,
                "flights": anchors_artifact["flights"],
            },
            "smoothing": smoothing,
            "junction_gaps_m": gaps,
            "trajectory_connections": trajectory_connection_report(fits),
            "legacy_tracking_decision": tracking_decisions.get(clip_key),
            "upstream_point_validity": validity_row,
            "upstream_repair_override": {
                "applied": bool(repairable_upstream_hold or retained_arc_override),
                "reason": (
                    "retained_tracking_arcs"
                    if retained_arc_override
                    else "tracking_risk_only"
                    if repairable_upstream_hold
                    else None
                ),
                "final_gate_authority": "s6_point_gate",
            },
            "tracking_arc_composition": {
                "available": bool((tracking_row or {}).get("arcs")),
                "retained_arcs": sum(
                    arc.get("decision") == "retain" for arc in (tracking_row or {}).get("arcs", [])
                ),
                "held_arcs": sum(
                    arc.get("decision") == "hold" for arc in (tracking_row or {}).get("arcs", [])
                ),
                "attempted_flights": sum(
                    attempt.get("tracking_arc_scope", {}).get("decision") == "attempt"
                    for attempt in flight_attempts
                ),
                "abstained_flights": sum(
                    attempt.get("skip_reason") == "tracking_arc_abstained"
                    for attempt in flight_attempts
                ),
            },
            "event_boundary_source": event_boundary_source,
            "event_consumer_mode": event_consumer_mode,
            "camera_scope_source": str(camera_scope) if camera_scope is not None else None,
            "fits": fits,
            "bounce_scores": [],
            "uncertainty": {
                "enabled": uncertainty_hypotheses is not None,
                "weighted_observations": len(observation_weights),
                "accepted_bounce_hypotheses": (
                    hard_bounce_integrity["satisfied"]
                    if joint_refine
                    and (uncertainty_hypotheses is not None or anchor_bounce_geometry)
                    else 0
                ),
                "bounce_hypotheses": (
                    [
                        {
                            key: (value.tolist() if isinstance(value, np.ndarray) else value)
                            for key, value in row.items()
                        }
                        for row in bounce_hypotheses
                    ]
                    if joint_refine
                    and (uncertainty_hypotheses is not None or anchor_bounce_geometry)
                    else []
                ),
                "bounce_geometry_requested": anchor_bounce_geometry,
                "bounce_geometry_anchored": (
                    bool(hard_bounce_requirements) and hard_bounce_integrity["all_satisfied"]
                ),
                "hard_bounce_integrity": hard_bounce_integrity,
                "bounce_scoring_independent": not anchor_bounce_geometry,
                "fallbacks": joint_refine_fallbacks,
            },
            "interpretation_branches": branch_report,
            "shared_contact_fit": {
                "enabled": bool(anchor_first and shared_contact_fit),
                "adopted_contacts": len(
                    {
                        int(shared["contact_index"])
                        for fit in raw_fits.values()
                        for shared in getattr(fit, "_shared_contacts", [])
                    }
                ),
                "contact_ray_diagnostics": contact_ray_fit_diagnostics,
            },
            "net_treatment": "post_fit_plausibility_only",
            "net_collision_topology": net_collision_integrity,
            "unrefined_flight_initializers": unrefined_initializers,
            "event_topology_branches": topology_report,
            "joint_rally_refinement": joint_rally_refinement_report(
                boundary_events,
                contacts,
                topology_report,
                bounce_hypotheses,
            ),
            "contact_reprojection": {
                "soft_residual_enabled": bool(joint_refine),
                "exact_ray_branch_enabled": exact_contact_reprojection,
            },
            "owner_tracking_quality": None,
        }

    if not isolate_points:
        evaluated = [evaluate_point(clip_key) for clip_key in clips]
        shared_report = apply_match_shared_priors(evaluated) if match_shared_priors else None
        report = summarize(evaluated, owner_truth_loaded=False)
        report["match_shared_priors"] = shared_report
        return report

    # scipy's finite-difference Jacobian repeatedly enters Python, so threads
    # contend on the GIL.  Process isolation also makes a point timeout fail
    # closed without sacrificing completed neighbouring points.
    run_options = {
        "automatic_boundaries": automatic_boundaries,
        "automatic_hypotheses": automatic_hypotheses,
        "max_nfev": max_nfev,
        "multi_start": multi_start,
        "joint_refine": joint_refine,
        "terminal_flights": terminal_flights,
        "uncertainty_hypotheses": uncertainty_hypotheses,
        "discrete_bounce_branches": discrete_bounce_branches,
        "anchor_bounce_geometry": anchor_bounce_geometry,
        "anchor_first": anchor_first,
        "net_point_anchor": net_point_anchor,
        "downweight_contact_adjacent": downweight_contact_adjacent,
        "striker_witness_prior": striker_witness_prior,
        "striker_witness_authority": striker_witness_authority,
        "contact_observation_witness": contact_observation_witness,
        "contact_observation_sigma": contact_observation_sigma,
        "physical_spin_prior": physical_spin_prior,
        "subframe_anchors": subframe_anchors,
        "subframe_contacts": subframe_contacts,
        "subframe_time_priors": subframe_time_priors,
        "subframe_contact_seam": subframe_contact_seam,
        "subframe_contact_advance": subframe_contact_advance,
        "subframe_contact_seam_skips_refit": subframe_contact_seam_skips_refit,
        "subframe_contact_seam_witnessed": subframe_contact_seam_witnessed,
        "subframe_plane_anchor_error": subframe_plane_anchor_error,
        "subframe_bounce_witness": subframe_bounce_witness,
        "whole_point_joint": whole_point_joint,
        "shared_contact_fit": shared_contact_fit,
        "whole_point_branches": whole_point_branches,
        "exact_contact_reprojection": exact_contact_reprojection,
        "event_topology_branches": event_topology_branches,
        "topology_sequence_branches": topology_sequence_branches,
        "topology_branch_width": topology_branch_width,
        "topology_branch_margin": topology_branch_margin,
        "topology_post_joint_scoring": topology_post_joint_scoring,
        "topology_joint_width": topology_joint_width,
        "branch_width": branch_width,
        "branch_margin": branch_margin,
        "event_boundary_source": event_boundary_source,
        "event_consumer_mode": event_consumer_mode,
        "camera_scope": camera_scope,
        "pose_artifact_name": pose_artifact_name,
        "physical_motion_name": physical_motion_name,
        "camera_artifact_name": camera_artifact_name,
        "anchors_output_root": anchors_output_root,
    }
    points: list[dict | None] = [None] * len(clips)
    context = mp.get_context()
    pending = iter(enumerate(clips))
    active: dict[int, tuple[int, str, mp.Process, mp.queues.Queue, float]] = {}

    def start_available_workers() -> None:
        while len(active) < workers:
            try:
                index, clip_key = next(pending)
            except StopIteration:
                return
            result_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=_evaluate_point_worker,
                args=(
                    audit_root,
                    manifest_path,
                    clip_key,
                    run_options,
                    math_threads,
                    result_queue,
                ),
            )
            process.start()
            active[process.pid] = (index, clip_key, process, result_queue, time.monotonic())

    start_available_workers()
    while active:
        for process_id, (index, clip_key, process, result_queue, started) in list(active.items()):
            try:
                worker_result = result_queue.get_nowait()
            except queue.Empty:
                worker_result = None
            if worker_result is not None:
                del active[process_id]
                status, payload = worker_result
                # Reading first releases Queue feeder backpressure for large,
                # anchor-rich point records; the child can now finish cleanly.
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()
                    process.join()
                if status != "ok":
                    raise RuntimeError(f"point worker failed for {clip_key}: {payload}")
                points[index] = payload
                result_queue.close()
                start_available_workers()
                continue
            if process.is_alive():
                if time.monotonic() - started < point_timeout_seconds:
                    continue
                del active[process_id]
                process.terminate()
                process.join()
                points[index] = _timeout_point_record(
                    clip_key,
                    event_boundary_source=event_boundary_source,
                    event_consumer_mode=event_consumer_mode,
                    point_timeout_seconds=point_timeout_seconds,
                )
            else:
                del active[process_id]
                process.join()
                try:
                    status, payload = result_queue.get(timeout=1.0)
                except queue.Empty as error:
                    raise RuntimeError(
                        f"point worker exited without a result: {clip_key}"
                    ) from error
                if status != "ok":
                    raise RuntimeError(f"point worker failed for {clip_key}: {payload}")
                points[index] = payload
            result_queue.close()
            start_available_workers()
        if active:
            time.sleep(0.02)
    evaluated = [point for point in points if point is not None]
    shared_report = apply_match_shared_priors(evaluated) if match_shared_priors else None
    report = summarize(evaluated, owner_truth_loaded=False)
    report["match_shared_priors"] = shared_report
    return report


def reconstruct(
    audit_root: Path,
    manifest_path: Path,
    event_boundaries: Path,
    *,
    max_nfev: int,
    workers: int,
    point_keys: set[str] | None = None,
    multi_start: bool = False,
    joint_refine: bool = False,
    terminal_flights: bool = False,
    uncertainty_hypotheses: Path | None = None,
    discrete_bounce_branches: bool = False,
    anchor_bounce_geometry: bool = False,
    anchor_first: bool = True,
    net_point_anchor: bool = True,
    downweight_contact_adjacent: bool = False,
    striker_witness_prior: bool = False,
    striker_witness_authority: bool = True,
    contact_observation_witness: bool = False,
    contact_observation_sigma: bool = False,
    physical_spin_prior: bool = False,
    subframe_anchors: bool = False,
    subframe_contacts: bool = False,
    subframe_time_priors: bool = False,
    subframe_contact_seam: bool = True,
    subframe_contact_advance: bool = True,
    subframe_contact_seam_skips_refit: bool = True,
    subframe_contact_seam_witnessed: bool = False,
    subframe_plane_anchor_error: bool = False,
    subframe_bounce_witness: bool = False,
    whole_point_joint: bool = False,
    shared_contact_fit: bool = True,
    match_shared_priors: bool = False,
    whole_point_branches: bool = False,
    exact_contact_reprojection: bool = False,
    event_topology_branches: bool = False,
    topology_sequence_branches: bool = False,
    topology_branch_width: int = 8,
    topology_branch_margin: float = 8.0,
    topology_post_joint_scoring: bool = False,
    topology_joint_width: int = 2,
    branch_width: int = 6,
    branch_margin: float = 8.0,
    camera_scope: Path | None = None,
    include_dead_time_emissions: bool = False,
    event_hypotheses: Path | None = None,
    pose_artifact_name: str | None = None,
    physical_motion_name: str | None = None,
    camera_artifact_name: str = "camera_P_per_frame_v1.npz",
    anchors_output_root: Path | None = None,
    point_timeout_seconds: float = 600.0,
    math_threads: int = 1,
) -> dict:
    """Run label-blind 3D reconstruction from automatic pipeline artifacts."""
    automatic_boundaries = load_event_boundaries(
        event_boundaries,
        include_dead_time_emissions=include_dead_time_emissions,
    )
    automatic_hypotheses = load_event_hypotheses(event_hypotheses)
    clips = load_automatic_point_universe(audit_root, manifest_path)
    if point_keys:
        unknown = point_keys - set(clips)
        if unknown:
            raise ValueError(f"unknown automatic point keys: {sorted(unknown)}")
        clips = [clip_key for clip_key in clips if clip_key in point_keys]
    return _run_reconstruction(
        audit_root,
        manifest_path,
        clips=clips,
        automatic_boundaries=automatic_boundaries,
        automatic_hypotheses=automatic_hypotheses,
        max_nfev=max_nfev,
        workers=workers,
        multi_start=multi_start,
        joint_refine=joint_refine,
        terminal_flights=terminal_flights,
        uncertainty_hypotheses=uncertainty_hypotheses,
        discrete_bounce_branches=discrete_bounce_branches,
        anchor_bounce_geometry=anchor_bounce_geometry,
        anchor_first=anchor_first,
        net_point_anchor=net_point_anchor,
        downweight_contact_adjacent=downweight_contact_adjacent,
        striker_witness_prior=striker_witness_prior,
        striker_witness_authority=striker_witness_authority,
        contact_observation_witness=contact_observation_witness,
        contact_observation_sigma=contact_observation_sigma,
        physical_spin_prior=physical_spin_prior,
        subframe_anchors=subframe_anchors,
        subframe_contacts=subframe_contacts,
        subframe_time_priors=subframe_time_priors,
        subframe_contact_seam=subframe_contact_seam,
        subframe_contact_advance=subframe_contact_advance,
        subframe_contact_seam_skips_refit=subframe_contact_seam_skips_refit,
        subframe_contact_seam_witnessed=subframe_contact_seam_witnessed,
        subframe_plane_anchor_error=subframe_plane_anchor_error,
        subframe_bounce_witness=subframe_bounce_witness,
        whole_point_joint=whole_point_joint,
        shared_contact_fit=shared_contact_fit,
        match_shared_priors=match_shared_priors,
        whole_point_branches=whole_point_branches,
        exact_contact_reprojection=exact_contact_reprojection,
        event_topology_branches=event_topology_branches,
        topology_sequence_branches=topology_sequence_branches,
        topology_branch_width=topology_branch_width,
        topology_branch_margin=topology_branch_margin,
        topology_post_joint_scoring=topology_post_joint_scoring,
        topology_joint_width=topology_joint_width,
        branch_width=branch_width,
        branch_margin=branch_margin,
        event_boundary_source=str(event_boundaries),
        event_consumer_mode=(
            "lossless_include_dead_time"
            if include_dead_time_emissions
            else "point_grammar_in_play_only"
        ),
        camera_scope=camera_scope,
        pose_artifact_name=pose_artifact_name,
        physical_motion_name=physical_motion_name,
        camera_artifact_name=camera_artifact_name,
        anchors_output_root=anchors_output_root,
        point_timeout_seconds=point_timeout_seconds,
        math_threads=math_threads,
        isolate_points=True,
    )


def summarize(points: list[dict], *, owner_truth_loaded: bool = True) -> dict:
    retained = [row for row in points if row["decision"] == "retain"]
    screened = [row for row in points if row["screen_decision"] == "screen"]
    fits = [fit for row in retained for fit in row["fits"]]
    bounce_matches = [
        match for row in retained for score in row["bounce_scores"] for match in score["matches"]
    ]
    owner_good = [row for row in points if row["owner_tracking_quality"] == "good"]
    retained_good = [row for row in retained if row["owner_tracking_quality"] == "good"]
    uncertainty_enabled = any(
        row.get("uncertainty", {}).get("enabled")
        or row.get("uncertainty", {}).get("bounce_geometry_anchored")
        for row in points
    )
    accepted_bounce_hypotheses = sum(
        int(row.get("uncertainty", {}).get("accepted_bounce_hypotheses", 0)) for row in points
    )
    report = {
        "schema": "v4_physics_lift_v1",
        "status": ("offline_truth_scoring" if owner_truth_loaded else "raw_video_inference"),
        "evaluation": {"owner_truth_loaded": owner_truth_loaded},
        "event_boundary_source": sorted({row["event_boundary_source"] for row in points}),
        "event_consumer_mode": sorted(
            {row.get("event_consumer_mode", "point_grammar_in_play_only") for row in points}
        ),
        "physics": {
            "flight": (
                "gravity + aerodynamic drag + Magnus + aerodynamic spin decay; "
                "court impact updates both velocity and spin"
            ),
            "bounce": "surface-specific coefficient of restitution and friction",
            "spin": (
                "three-axis fitted topspin/sidespin/rifle spin with serve-family "
                "branches; report as diagnostic because monocular spin remains "
                "weakly identified"
            ),
        },
        "uncertainty": {
            "enabled": uncertainty_enabled,
            "contract": (
                "confidence-weighted 2D observations plus optional probabilistic "
                "event anchors accepted only with an independent physics witness"
            ),
            "accepted_bounce_hypotheses": accepted_bounce_hypotheses,
            "final_gate_unchanged": True,
        },
        "interpretation_branches": {
            "enabled": any(row.get("interpretation_branches", {}).get("enabled") for row in points),
            "decisive_points": sum(
                row.get("interpretation_branches", {}).get("decisive") is True for row in points
            ),
            "ambiguous_points": sum(
                row.get("interpretation_branches", {}).get("enabled")
                and row.get("interpretation_branches", {}).get("decisive") is not True
                for row in points
            ),
        },
        "event_topology_branches": {
            "enabled": any(row.get("event_topology_branches", {}).get("enabled") for row in points),
            "decisive_points": sum(
                row.get("event_topology_branches", {}).get("decisive") is True for row in points
            ),
            "changed_points": sum(
                row.get("event_topology_branches", {}).get("selected")
                not in {None, "contacts_baseline"}
                for row in points
            ),
        },
        "joint_rally_refinement": {
            "enabled": any(row.get("joint_rally_refinement", {}).get("enabled") for row in points),
            "changed_points": sum(
                row.get("joint_rally_refinement", {}).get("status") == "selected_repair"
                for row in points
            ),
            "inserted_contacts": sum(
                len(
                    row.get("joint_rally_refinement", {})
                    .get("operations", {})
                    .get("inserted_contacts", [])
                )
                for row in points
            ),
            "inserted_bounces": sum(
                len(
                    row.get("joint_rally_refinement", {})
                    .get("operations", {})
                    .get("inserted_bounces", [])
                )
                for row in points
            ),
            "omitted_contacts": sum(
                len(
                    row.get("joint_rally_refinement", {})
                    .get("operations", {})
                    .get("omitted_contacts", [])
                )
                for row in points
            ),
            "retimed_contacts": sum(
                len(
                    row.get("joint_rally_refinement", {})
                    .get("operations", {})
                    .get("retimed_contacts", [])
                )
                for row in points
            ),
            "contract": (
                "S6 may select bounded derived event repairs while preserving immutable "
                "S5 emissions and reporting every operation."
            ),
        },
        "points": len(points),
        "retained_points": len(retained),
        "screened_points": len(screened),
        "screened_owner_good": sum(row["owner_tracking_quality"] == "good" for row in screened),
        "held_points": len(points) - len(retained),
        "owner_good_points": len(owner_good),
        "retained_owner_good": len(retained_good),
        "retained_owner_good_precision": (len(retained_good) / len(retained) if retained else None),
        "retained_owner_good_recall": (
            len(retained_good) / len(owner_good) if owner_good else None
        ),
        "retained_shots": len(fits),
        "complete_point_candidates": sum(
            row.get("complete_point_gate", {}).get("accepted") is True for row in points
        ),
        "median_reprojection_px": (
            float(np.median([row["rms_px"] for row in fits])) if fits else None
        ),
        "median_speed_kmh": (
            float(np.median([row["speed_kmh"] for row in fits])) if fits else None
        ),
        "median_spin_rpm": (float(np.median([row["spin_rpm"] for row in fits])) if fits else None),
        "bounce_matches": len(bounce_matches),
        "median_bounce_timing_frames": (
            float(np.median([row["timing_frames"] for row in bounce_matches]))
            if bounce_matches
            else None
        ),
        "median_bounce_court_error_m": (
            float(np.median([row["court_error_m"] for row in bounce_matches]))
            if bounce_matches
            else None
        ),
        "points_detail": points,
    }
    if not owner_truth_loaded:
        for key in (
            "owner_good_points",
            "retained_owner_good",
            "retained_owner_good_precision",
            "retained_owner_good_recall",
            "screened_owner_good",
        ):
            report.pop(key, None)
    return report
