"""Canonical per-point court calibration from automatic line-topology witnesses.

One anchor frame carries the point's court geometry, as before. Each clip is then
registered frame by frame onto that anchor with ``court_topology.transfer_court_homography``
so the camera follows within-point pans instead of freezing on one frame, and
``court_H_per_frame_v1.npz`` carries the resulting per-frame homography, its reliability
and its provenance. Registration fails closed: a frame with no supporting registration
keeps the anchor homography and is marked unreliable, so the artifact is never worse than
the previous point-static behaviour. ``--anchor-mode best_of_n`` scores all ten candidate
frames instead of keeping the first that solves; it is not the default (see
``select_anchor``). When those established candidates all fail, five additional
quantiles are tried before the point abstains.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import court
from cv.pipeline.court_far_baseline_refinement import refine_far_baseline
from cv.pipeline.court_topology import (
    PIPELINE_SURFACE_WITNESS_POLICY,
    SURFACE_WITNESS_OFF,
    REGISTRATION_MASK_POLICIES,
    INTERPOLATED_CAMERA_POLICIES,
    SURFACE_WITNESS_POLICIES,
    assess_court_image_support,
    EDGE_CONVENTIONS,
    NO_TOPOLOGY_SOLUTION,
    _standard_broadcast_view,
    court_line_masks,
    court_surface_consistency,
    edge_convention_world_transform,
    solve_court_h_topology_native,
    topology_score,
    transfer_court_homography,
)
from cv.pipeline.court_topology_frame_track import interpolate_track_H

ANCHOR_QUANTILES = (0.30, 0.70, 0.50, 0.25, 0.90)
FALLBACK_ANCHOR_QUANTILES = (0.10, 0.20, 0.40, 0.60, 0.80)
MATCH_PRIOR_QUANTILES = tuple(float(value) for value in np.linspace(0.10, 0.90, 9))
MATCH_PRIOR_MAXIMUM_REFERENCES = 24
MATCH_PRIOR_MINIMUM_TOPOLOGY_SCORE = 0.92
MATCH_PRIOR_MINIMUM_SUPPORT_FRACTION = 0.55
# The solve scores a candidate by distance from the painted-line mask, so it settles on the
# centre of the paint, and that is what the emitted homography means. Every label set and
# benchmark CSV here *declares* ``itf_outside_edge_v1``, half a published ITF line width
# away: 5 cm at a baseline, 2.5 cm elsewhere. Moving the emitted homography onto that
# declared convention (``court_topology.to_edge_convention``, published widths only) halves
# the near-baseline bias against the agent landmark set but regresses the owner
# ``court_geometry_v1`` benchmark, whose clicks sit nearer the paint centre than its own
# declared convention. The two truth sets therefore disagree about the convention, so the
# artifact keeps the frame it is actually solved in and the transform stays opt-in until the
# owner settles which one the artifact should declare. See SCOREBOARD.md.
ARTIFACT_EDGE_CONVENTION = "paint_centre"
ANCHOR_MODES = ("first_success", "best_of_n", "first_metric_success")
DEFAULT_ANCHOR_MODE = "first_success"
DEFAULT_REGISTRATION_STRIDE = 5
# A registered frame further than this many frames from an accepted registration keeps
# the anchor homography and is reported unreliable.
MAXIMUM_RELIABLE_SAMPLE_GAP = 2


@dataclass(frozen=True)
class AnchorSolution:
    """The frame chosen to carry the point's court geometry."""

    index: int
    frame: str
    homography: np.ndarray
    source: str
    topology_score: float
    surface_score: float
    player_score: float | None
    refinement_evidence: dict
    far_evidence: dict
    metric_evidence: dict | None = None
    acceptance_route: str = "strict_surface"
    illumination_surface_score: float | None = None

    @property
    def rank(self) -> float:
        # Deliberately the single-median surface score even for an illumination-split
        # acceptance: that route is only reached after every ordinary acceptance has failed
        # for every candidate frame, so ranking it low can never displace an ordinary anchor.
        return (
            0.70 * self.topology_score
            + 0.25 * self.surface_score
            + 0.05 * (self.player_score if self.player_score is not None else 1.0)
        )


def _solve_anchor_frame(
    index: int,
    frame_path: Path,
    *,
    proposal_pool: int = 48,
    surface_witness_policy: str = SURFACE_WITNESS_OFF,
) -> AnchorSolution | dict:
    image = cv2.imread(str(frame_path))
    if image is None:
        return {"frame": frame_path.name, "error": "unreadable_frame"}
    try:
        solution = solve_court_h_topology_native(
            image,
            surface_witness_policy=surface_witness_policy,
            **({"proposal_pool": proposal_pool} if proposal_pool != 48 else {}),
        )
        far_refinement = refine_far_baseline(image, solution.homography)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return {"frame": frame_path.name, "error": str(exc)}
    far_evidence = far_refinement.evidence
    return AnchorSolution(
        index=index,
        frame=frame_path.name,
        homography=far_refinement.homography,
        source=(
            f"graded_far_refine:{solution.source}" if far_evidence["accepted"] else solution.source
        ),
        topology_score=float(solution.topology_score),
        surface_score=float(solution.surface_score),
        player_score=solution.player_score,
        refinement_evidence=solution.refinement_evidence,
        far_evidence=far_evidence,
        acceptance_route=solution.acceptance_route,
        illumination_surface_score=(
            None
            if solution.illumination_surface_score is None
            else float(solution.illumination_surface_score)
        ),
    )


def qualify_metric_anchor(
    anchor: AnchorSolution,
    frame_path: Path,
    *,
    edge_convention: str = ARTIFACT_EDGE_CONVENTION,
) -> dict:
    """Measure the candidate's own native net with ordinary camera defaults."""
    from cv.pipeline.camera_cal import NET_SUPPORT, direct_projection_from_net, measure_point_net

    image = cv2.imread(str(frame_path))
    if image is None:
        return {"reliable": False, "reason": "unreadable_frame"}
    height, width = image.shape[:2]
    convention, _ = edge_convention_world_transform(edge_convention)
    try:
        ground_to_image = np.linalg.inv(convention @ anchor.homography)
        _, reference, observed, focal, source = measure_point_net(
            (0, ground_to_image, str(frame_path), width, height, NET_SUPPORT)
        )
        _, evidence = direct_projection_from_net(
            ground_to_image, observed, focal, w=width, h=height, support=NET_SUPPORT
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return {"reliable": False, "reason": "calibration_error", "error": str(exc)}
    return {
        **evidence,
        "reference_frame": reference,
        "net_observation_source": source,
        "tape_measurement": "hough_top",
        "net_support": NET_SUPPORT,
        "edge_convention": edge_convention,
    }


def _anchor_pass(
    frames: list[Path],
    indices,
    attempts: list[dict],
    mode: str,
    edge_convention: str,
    *,
    proposal_pool: int = 48,
    surface_witness_policy: str = SURFACE_WITNESS_OFF,
    failures: list[int] | None = None,
) -> AnchorSolution | None:
    """Walk candidate frames in the source-defined order under one solver setting."""
    settings = {}
    if proposal_pool != 48:
        settings["proposal_pool"] = proposal_pool
    if surface_witness_policy != SURFACE_WITNESS_OFF:
        settings["surface_witness_policy"] = surface_witness_policy
    best: AnchorSolution | None = None
    for index in indices:
        outcome = _solve_anchor_frame(index, frames[index], **settings)
        if isinstance(outcome, dict):
            attempts.append({**outcome, **settings})
            if failures is not None and outcome.get("error") == NO_TOPOLOGY_SOLUTION:
                failures.append(index)
            continue
        if mode == "first_metric_success":
            metric = qualify_metric_anchor(outcome, frames[index], edge_convention=edge_convention)
            if not metric["reliable"]:
                attempts.append(
                    {
                        "frame": outcome.frame,
                        "error": "metric_calibration_rejected",
                        "metric": metric,
                        **settings,
                    }
                )
                continue
            outcome = replace(outcome, metric_evidence=metric)
        if best is None or outcome.rank > best.rank:
            best = outcome
        if mode in {"first_success", "first_metric_success"}:
            break
    return best


def select_anchor(
    frames: list[Path],
    mode: str = DEFAULT_ANCHOR_MODE,
    *,
    edge_convention: str = ARTIFACT_EDGE_CONVENTION,
    surface_witness_policy: str = PIPELINE_SURFACE_WITNESS_POLICY,
) -> tuple[AnchorSolution | None, list[dict]]:
    """Choose the frame that carries the point's court geometry.

    ``first_success`` walks the quantiles in order and keeps the first frame that solves;
    ``best_of_n`` solves all ten candidates and keeps the strongest witness.
    ``first_metric_success`` is opt-in: continue the same source-defined order after
    ordinary direct net-camera qualification rejects a topology candidate. It uses the
    camera producer's default singles-stick/hough-top convention, with no fallback promotion.
    ``best_of_n`` scored worse on the untouched transfer cohort (view medians at most 3 px 27/33 against
    31/33) because that truth was clicked on the frame ``first_success`` selects, so it is
    not the default.

    ``surface_witness_policy="illumination_split"`` adds one further pass over the same
    candidate frames in the same order, and only after both established passes have left the
    point with no anchor at all. A point that currently gets an anchor therefore gets exactly
    the same anchor, from the same frame, whatever this policy says.
    """
    if mode not in ANCHOR_MODES:
        raise ValueError(f"unknown anchor mode {mode!r}")
    if surface_witness_policy not in SURFACE_WITNESS_POLICIES:
        raise ValueError(f"unknown surface witness policy {surface_witness_policy!r}")
    indices = dict.fromkeys(
        round((len(frames) - 1) * quantile)
        for quantile in (*ANCHOR_QUANTILES, *FALLBACK_ANCHOR_QUANTILES)
    )
    attempts: list[dict] = []
    retry_indices: list[int] = []
    # Exhaust the original anchor policy before expanding any rejected proposal beam.
    # This prevents an expanded early frame preempting a later original success.
    best = _anchor_pass(frames, indices, attempts, mode, edge_convention, failures=retry_indices)
    if best is None:
        best = _anchor_pass(
            frames, retry_indices, attempts, mode, edge_convention, proposal_pool=384
        )
    # The illumination witness only ever answers NO_TOPOLOGY_SOLUTION -- a court the surface
    # witness rejected under a sun/shadow split.  ``retry_indices`` holds exactly those frames,
    # so gating on it keeps the extra pass off every other way an anchor can fail: an unreadable
    # frame, a non-broadcast view, or a topology that solved and then failed metric
    # qualification.  Those would otherwise pay for a full second pass that cannot help them.
    if best is None and retry_indices and surface_witness_policy != SURFACE_WITNESS_OFF:
        best = _anchor_pass(
            frames,
            retry_indices,
            attempts,
            mode,
            edge_convention,
            surface_witness_policy=surface_witness_policy,
        )
    return best, attempts


def _maximum_topology_score(image: np.ndarray, homography: np.ndarray) -> float:
    observation_mask = court.line_mask_observation_mask(image.shape, top_fraction=0.12)
    return max(
        topology_score(homography, mask, observation_mask=observation_mask)
        for _, mask in court_line_masks(image)
    )


def _maximum_mask_score(masks: tuple[tuple[str, np.ndarray], ...], homography: np.ndarray) -> float:
    return max(topology_score(homography, mask) for _, mask in masks)


def recover_anchor_from_match_prior(
    frames: list[Path],
    point: int,
    references: list[tuple[int, np.ndarray]],
) -> AnchorSolution | None:
    """Recover a static standard view from automatic match geometry and repeated lines.

    A match homography is only a proposal.  Acceptance is based on a stricter version of
    the ordinary target topology witness: at least 0.92 on a nine-frame temporal median
    and on a majority of the sampled real frames.  This rejects cuts, replay cameras and
    aerials even when one frame happens to alias a few painted lines.
    """
    if not frames or not references:
        return None
    indices = list(
        dict.fromkeys(round((len(frames) - 1) * quantile) for quantile in MATCH_PRIOR_QUANTILES)
    )
    samples = [(index, cv2.imread(os.fspath(frames[index]))) for index in indices]
    samples = [(index, image) for index, image in samples if image is not None]
    if len(samples) < 3:
        return None
    shapes = {image.shape for _, image in samples}
    if len(shapes) != 1:
        return None
    consensus = np.median(np.stack([image for _, image in samples]), axis=0).astype(np.uint8)
    consensus_masks = court_line_masks(consensus)
    sample_masks = [(index, image, court_line_masks(image)) for index, image in samples]
    width, height = consensus.shape[1], consensus.shape[0]
    candidates = []
    for reference_point, homography in sorted(references, key=lambda row: abs(row[0] - point))[
        :MATCH_PRIOR_MAXIMUM_REFERENCES
    ]:
        candidate = np.asarray(homography, dtype=float)
        if not np.isfinite(candidate).all() or not _standard_broadcast_view(
            candidate, width, height
        ):
            continue
        consensus_score = _maximum_mask_score(consensus_masks, candidate)
        if consensus_score < MATCH_PRIOR_MINIMUM_TOPOLOGY_SCORE:
            continue
        frame_scores = [
            (index, _maximum_mask_score(masks, candidate)) for index, _, masks in sample_masks
        ]
        minimum_support = max(
            3, math.ceil(MATCH_PRIOR_MINIMUM_SUPPORT_FRACTION * len(frame_scores))
        )
        supported = [row for row in frame_scores if row[1] >= MATCH_PRIOR_MINIMUM_TOPOLOGY_SCORE]
        if len(supported) < minimum_support:
            continue
        best_index, best_score = max(frame_scores, key=lambda row: row[1])
        candidates.append(
            (
                min(consensus_score, float(np.median([score for _, score in supported]))),
                -abs(reference_point - point),
                reference_point,
                best_index,
                best_score,
                consensus_score,
                len(supported),
                candidate,
            )
        )
    if not candidates:
        return None
    (
        _,
        _,
        reference_point,
        best_index,
        best_score,
        consensus_score,
        support_frames,
        homography,
    ) = max(candidates, key=lambda row: (row[0], row[1]))
    selected_image = next(image for index, image in samples if index == best_index)
    return AnchorSolution(
        index=best_index,
        frame=frames[best_index].name,
        homography=homography,
        source=f"match_prior_consensus:pt{reference_point:04d}",
        topology_score=float(best_score),
        surface_score=float(court_surface_consistency(selected_image, homography)),
        player_score=None,
        refinement_evidence={
            "automatic_match_prior": True,
            "reference_point": reference_point,
            "consensus_topology_score": float(consensus_score),
            "support_frames": support_frames,
            "sampled_frames": len(samples),
            "minimum_topology_score": MATCH_PRIOR_MINIMUM_TOPOLOGY_SCORE,
        },
        far_evidence={"accepted": False, "reason": "match_prior_consensus"},
    )


def register_clip(
    frames: list[Path],
    anchor: AnchorSolution,
    stride: int,
    *,
    registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
) -> tuple[dict[int, np.ndarray], dict[int, bool], dict[int, str], dict]:
    """Register samples and interpolate only across gaps with no rejected sample."""
    if registration_mask not in REGISTRATION_MASK_POLICIES:
        raise ValueError(f"unknown court registration mask {registration_mask!r}")
    if interpolated_camera_policy not in INTERPOLATED_CAMERA_POLICIES:
        raise ValueError(f"unknown interpolated camera policy {interpolated_camera_policy!r}")
    sample_indices = sorted(
        {0, len(frames) - 1, anchor.index, *range(0, len(frames), max(1, stride))}
    )
    anchor_image = cv2.imread(str(frames[anchor.index]))
    accepted_indices: list[int] = []
    accepted_homographies: list[np.ndarray] = []
    accepted_sources: list[str] = []
    inlier_ratios: list[float] = []
    target_scores: list[float] = []
    failures = 0
    failed_indices: list[int] = []
    # Per-sample witnesses. A registered frame's own line support is the only label-free
    # evidence that its camera still fits that picture, so it is retained rather than
    # collapsed into a median.
    sample_evidence: dict[int, dict] = {}
    for index in sample_indices:
        if index == anchor.index:
            accepted_indices.append(index)
            accepted_homographies.append(anchor.homography)
            accepted_sources.append("registered")
            inlier_ratios.append(1.0)
            target_scores.append(anchor.topology_score)
            continue
        image = cv2.imread(str(frames[index]))
        if image is None:
            failures += 1
            failed_indices.append(index)
            sample_evidence[index] = {"status": "held", "reason": "unreadable_frame"}
            continue
        try:
            homography, evidence = transfer_court_homography(
                image,
                anchor_image,
                anchor.homography,
                **({"registration_mask": registration_mask} if registration_mask != "off" else {}),
            )
        except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
            failures += 1
            failed_indices.append(index)
            sample_evidence[index] = {"status": "held", "reason": str(exc)}
            continue
        accepted_indices.append(index)
        accepted_homographies.append(homography)
        # Feature selection does not change the registered-camera lineage contract.
        # The source mask is recorded in sample evidence and the producer policy.
        accepted_sources.append("registered")
        inlier_ratios.append(float(evidence["inlier_ratio"]))
        target_scores.append(float(evidence["target_topology_score"]))
        sample_evidence[index] = {
            "status": "registered",
            "inlier_ratio": round(float(evidence["inlier_ratio"]), 4),
            "target_topology_score": round(float(evidence["target_topology_score"]), 4),
            "target_support_mask": evidence.get("target_support_mask", "standard"),
            **(
                {
                    key: evidence[key]
                    for key in (
                        "registration_feature_mask",
                        "registration_mask_top_fraction",
                        "original_transfer_failure",
                        "matches",
                        "inliers",
                    )
                    if key in evidence
                }
                if evidence.get("registration_feature_mask")
                else {}
            ),
        }
    # Texture registration is brittle on compression, sun/shadow changes and players.
    # Propose the current track (or the static rig view when only the anchor exists), but
    # accept a failed sample only when its own line mask clears the stricter 0.92 witness.
    line_model_recoveries = 0
    for index in failed_indices:
        image = cv2.imread(str(frames[index]))
        if image is None:
            continue
        accepted_order = np.argsort(accepted_indices)
        registered = np.asarray(accepted_indices, dtype=np.int32)[accepted_order]
        registered_h = np.stack([accepted_homographies[i] for i in accepted_order])
        candidate = (
            interpolate_track_H(registered, registered_h, index)
            if len(registered) > 1
            else anchor.homography
        )
        if not _standard_broadcast_view(candidate, image.shape[1], image.shape[0]):
            continue
        score = _maximum_topology_score(image, candidate)
        if score < MATCH_PRIOR_MINIMUM_TOPOLOGY_SCORE:
            continue
        accepted_indices.append(index)
        accepted_homographies.append(candidate)
        accepted_sources.append("line_model_registered")
        inlier_ratios.append(0.0)
        target_scores.append(score)
        sample_evidence[index] = {
            "status": "line_model_registered",
            "inlier_ratio": 0.0,
            "target_topology_score": round(float(score), 4),
            "target_support_mask": "maximum_of_court_line_masks",
            "held_reason": sample_evidence.get(index, {}).get("reason"),
        }
        line_model_recoveries += 1
    order = np.argsort(accepted_indices)
    sampled = np.asarray(accepted_indices, dtype=np.int32)[order]
    sampled_homographies = np.stack([accepted_homographies[i] for i in order])
    sampled_sources = np.asarray(accepted_sources, dtype=str)[order]
    unsupported = sorted(set(failed_indices) - set(accepted_indices))
    # A failed sample is counter-evidence, not an unobserved gap. A close-up or
    # unreadable picture between valid court views must not become "reliable"
    # merely because the neighbouring transforms can be interpolated numerically.
    blocked_intervals = [
        (int(left), int(right))
        for left, right in zip(sampled, sampled[1:])
        if any(left < failed < right for failed in unsupported)
    ]
    per_frame: dict[int, np.ndarray] = {}
    reliable: dict[int, bool] = {}
    source: dict[int, str] = {}
    for index in range(len(frames)):
        gap = int(np.min(np.abs(sampled - index)))
        blocked_gap = any(left < index < right for left, right in blocked_intervals)
        if gap == 0:
            nearest = int(np.argmin(np.abs(sampled - index)))
            per_frame[index] = sampled_homographies[nearest]
            reliable[index] = True
            source[index] = str(sampled_sources[nearest])
        elif (
            sampled[0] < index < sampled[-1]
            and gap <= max(1, stride) * MAXIMUM_RELIABLE_SAMPLE_GAP
            and len(sampled) > 1
            and not blocked_gap
        ):
            per_frame[index] = interpolate_track_H(sampled, sampled_homographies, index)
            reliable[index] = True
            source[index] = "registered_interpolated"
        else:
            per_frame[index] = anchor.homography
            reliable[index] = False
            source[index] = (
                "registration_gap_abstention" if blocked_gap else "anchor_static_fallback"
            )
    interpolation_validation = None
    if interpolated_camera_policy == "qualify_retry":
        interpolation_validation = qualify_interpolated_cameras(
            frames, anchor, anchor_image, per_frame, reliable, source
        )
    evidence = {
        "registration_mask": registration_mask,
        "interpolated_camera_policy": interpolated_camera_policy,
        **(
            {"interpolation_validation": interpolation_validation}
            if interpolation_validation is not None
            else {}
        ),
        "stride": stride,
        "sampled": len(sample_indices),
        "registered": len(sampled),
        "registration_failures": failures,
        "line_model_recoveries": line_model_recoveries,
        "unsupported_sample_frames": [frames[index].name for index in unsupported],
        "blocked_interpolation_intervals": len(blocked_intervals),
        "reliable_frames": int(sum(reliable.values())),
        "frames": len(frames),
        "median_inlier_ratio": round(float(np.median(inlier_ratios)), 4),
        "median_target_topology_score": round(float(np.median(target_scores)), 4),
        "samples": {int(index): row for index, row in sorted(sample_evidence.items())},
    }
    return per_frame, reliable, source, evidence


def qualify_interpolated_cameras(
    frames: list[Path],
    anchor: AnchorSolution,
    anchor_image: np.ndarray,
    per_frame: dict[int, np.ndarray],
    reliable: dict[int, bool],
    source: dict[int, str],
) -> dict:
    """Qualify each original interpolation independently on its native picture.

    Retries use the original point anchor and do not become interpolation anchors.
    Failed source witnesses must not retain the recoverable interpolated lineage.
    """
    rows = {}
    counts = {"checked": 0, "passed": 0, "retried": 0, "recovered": 0, "held": 0}
    for index in range(len(frames)):
        if source[index] != "registered_interpolated" or not reliable[index]:
            continue
        counts["checked"] += 1
        row = {"frame": frames[index].name, "origin": "interpolated", "anchor_frame": anchor.frame}
        image = cv2.imread(str(frames[index]))
        if image is None:
            witness = {"accepted": False, "reason": "unreadable_frame"}
        else:
            try:
                witness = assess_court_image_support(image, per_frame[index])
            except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
                witness = {"accepted": False, "reason": str(exc)}
        row["interpolation_witness"] = witness
        if witness["accepted"]:
            row["status"] = "passed_interpolation"
            counts["passed"] += 1
        else:
            # A held own-picture witness must survive camera expansion. Keeping the
            # registered_interpolated tag would allow legacy algebraic rehabilitation.
            reliable[index] = False
            source[index] = "interpolation_evidence_hold"
            row["status"] = "held"
            if image is not None and anchor_image is not None:
                counts["retried"] += 1
                try:
                    homography, evidence = transfer_court_homography(
                        image,
                        anchor_image,
                        anchor.homography,
                        registration_mask="court_observation",
                    )
                except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
                    row["retry"] = {"accepted": False, "reason": str(exc)}
                else:
                    per_frame[index] = homography
                    reliable[index] = True
                    source[index] = "registered"
                    row.update(status="registered_retry", origin="interpolation_retry")
                    row["retry"] = {"accepted": True, **evidence}
                    counts["recovered"] += 1
            if not reliable[index]:
                counts["held"] += 1
        rows[index] = row
    return {**counts, "retry_policy": "same_original_anchor_court_observation", "frames": rows}


def _calibrated_result(
    clip: Path,
    point: int,
    anchor: AnchorSolution,
    *,
    stride: int,
    frame_track: bool,
    anchor_mode: str,
    attempts: list[dict],
    fallback_ancestry: list,
    edge_convention: str = ARTIFACT_EDGE_CONVENTION,
    registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
) -> tuple[int, dict, np.ndarray, dict]:
    frames = sorted(clip.glob("f_*.jpg"))
    convention, convention_evidence = edge_convention_world_transform(edge_convention)
    row = {
        "point": point,
        "status": "direct",
        "frame": anchor.frame,
        "source": anchor.source,
        "topology_score": anchor.topology_score,
        "surface_score": anchor.surface_score,
        "acceptance_route": anchor.acceptance_route,
        "illumination_surface_score": anchor.illumination_surface_score,
        "player_score": anchor.player_score,
        "visible_edge_refinement": anchor.refinement_evidence,
        "graded_band_climb": anchor.far_evidence,
        "anchor_mode": anchor_mode,
        "anchor_failures": attempts,
        "frame_scope": [anchor.frame],
        "fallback_ancestry": fallback_ancestry,
        "edge_convention": convention_evidence,
    }
    if anchor.metric_evidence is not None:
        row["metric_anchor_qualification"] = anchor.metric_evidence
    track: dict = {}
    if frame_track:
        per_frame, reliable, source, evidence = register_clip(
            frames,
            anchor,
            stride,
            registration_mask=registration_mask,
            interpolated_camera_policy=interpolated_camera_policy,
        )
        row["frame_scope"] = "frame_track"
        row["frame_track"] = evidence
        track = {
            "frames": np.asarray(
                [int(path.stem.removeprefix("f_")) for path in frames], dtype=np.int32
            ),
            "H": np.stack([convention @ per_frame[index] for index in range(len(frames))]),
            "reliable": np.asarray([reliable[index] for index in range(len(frames))], dtype=bool),
            "source": np.asarray([source[index] for index in range(len(frames))], dtype=str),
        }
    return point, row, convention @ anchor.homography, track


def calibrate_clip(
    clip: Path,
    *,
    stride: int = DEFAULT_REGISTRATION_STRIDE,
    frame_track: bool = True,
    anchor_mode: str = DEFAULT_ANCHOR_MODE,
    edge_convention: str = ARTIFACT_EDGE_CONVENTION,
    registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
    surface_witness_policy: str = PIPELINE_SURFACE_WITNESS_POLICY,
) -> tuple[int, dict, np.ndarray, dict] | None:
    frames = sorted(clip.glob("f_*.jpg"))
    if not frames:
        return None
    point = int(clip.name[2:])
    anchor, attempts = select_anchor(
        frames,
        anchor_mode,
        surface_witness_policy=surface_witness_policy,
        **({"edge_convention": edge_convention} if anchor_mode == "first_metric_success" else {}),
    )
    if anchor is None:
        return (
            point,
            {
                "point": point,
                "status": "abstained",
                "attempts": attempts,
                "fallback_ancestry": [],
            },
            np.full((3, 3), np.nan, dtype=float),
            {},
        )
    return _calibrated_result(
        clip,
        point,
        anchor,
        stride=stride,
        frame_track=frame_track,
        anchor_mode=anchor_mode,
        attempts=attempts,
        fallback_ancestry=[],
        edge_convention=edge_convention,
        registration_mask=registration_mask,
        interpolated_camera_policy=interpolated_camera_policy,
    )


def _limit_worker_threads() -> None:
    """One OpenCV thread per pool worker.

    Each worker already has a whole clip to itself, so OpenCV's default of one thread per
    core multiplies the pool by the core count and the machine thrashes.
    """
    cv2.setNumThreads(1)


def calibrate_points(
    out_dir: Path,
    frames_dir: str,
    jobs: int = 4,
    *,
    stride: int = DEFAULT_REGISTRATION_STRIDE,
    frame_track: bool = True,
    anchor_mode: str = DEFAULT_ANCHOR_MODE,
    edge_convention: str = ARTIFACT_EDGE_CONVENTION,
    registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
    surface_witness_policy: str = PIPELINE_SURFACE_WITNESS_POLICY,
) -> dict:
    clips = sorted((out_dir / frames_dir).glob("pt*"))
    with ProcessPoolExecutor(max_workers=max(1, jobs), initializer=_limit_worker_threads) as pool:
        futures = [
            pool.submit(
                calibrate_clip,
                clip,
                stride=stride,
                frame_track=frame_track,
                anchor_mode=anchor_mode,
                edge_convention=edge_convention,
                registration_mask=registration_mask,
                interpolated_camera_policy=interpolated_camera_policy,
                surface_witness_policy=surface_witness_policy,
            )
            for clip in clips
        ]
        results = []
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                results.append(result)
            if completed == len(futures) or completed % 10 == 0:
                print(f"court clips {completed}/{len(futures)}", flush=True)
    results.sort(key=lambda result: result[0])
    references = [
        (point, homography)
        for point, row, homography, _ in results
        if row["status"] == "direct" and np.isfinite(homography).all()
    ]
    recovered_results = []
    match_prior_recovered = 0
    for point, row, homography, track in results:
        if row["status"] != "abstained":
            recovered_results.append((point, row, homography, track))
            continue
        clip = out_dir / frames_dir / f"pt{point:04d}"
        frames = sorted(clip.glob("f_*.jpg"))
        recovered = recover_anchor_from_match_prior(frames, point, references)
        if recovered is None:
            recovered_results.append((point, row, homography, track))
            continue
        if anchor_mode == "first_metric_success":
            metric = qualify_metric_anchor(
                recovered, frames[recovered.index], edge_convention=edge_convention
            )
            if not metric["reliable"]:
                row.setdefault("attempts", []).append(
                    {
                        "frame": recovered.frame,
                        "error": "match_prior_metric_rejected",
                        "metric": metric,
                    }
                )
                recovered_results.append((point, row, homography, track))
                continue
            recovered = replace(recovered, metric_evidence=metric)
        reference_point = int(recovered.refinement_evidence["reference_point"])
        recovered_results.append(
            _calibrated_result(
                clip,
                point,
                recovered,
                stride=stride,
                frame_track=frame_track,
                anchor_mode=anchor_mode,
                edge_convention=edge_convention,
                registration_mask=registration_mask,
                interpolated_camera_policy=interpolated_camera_policy,
                attempts=list(row.get("attempts", [])),
                fallback_ancestry=[
                    {
                        "source": "automatic_match_prior_consensus",
                        "reference_point": reference_point,
                    }
                ],
            )
        )
        match_prior_recovered += 1
    results = sorted(recovered_results, key=lambda result: result[0])
    rows = [result[1] for result in results]
    homographies = {point: homography for point, _, homography, _ in results}
    tracks = {point: track for point, _, _, track in results if track}

    ordered = sorted(homographies)
    output = out_dir / "court_H_per_point.npz"
    np.savez_compressed(
        output,
        pts=np.asarray(ordered, dtype=np.int32),
        H=np.stack([homographies[point] for point in ordered]),
        source=np.asarray(
            [
                next(row for row in rows if row["point"] == point).get("source", "abstained")
                for point in ordered
            ]
        ),
        topology_score=np.asarray(
            [
                next(row for row in rows if row["point"] == point).get("topology_score", np.nan)
                for point in ordered
            ],
            dtype=float,
        ),
        surface_score=np.asarray(
            [
                next(row for row in rows if row["point"] == point).get("surface_score", np.nan)
                for point in ordered
            ],
            dtype=float,
        ),
    )
    frame_track_path = write_frame_track(out_dir, tracks)
    evidence_path = out_dir / "court_topology_evidence_v1.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "court_topology_evidence_v1",
                "automatic": True,
                "points": len(rows),
                "direct": sum(row["status"] == "direct" for row in rows),
                "abstained": sum(row["status"] == "abstained" for row in rows),
                "match_prior_recovered": match_prior_recovered,
                "frame_track": bool(tracks),
                "registration_mask": registration_mask,
                "interpolated_camera_policy": interpolated_camera_policy,
                "surface_witness_policy": surface_witness_policy,
                "illumination_split_recovered": sum(
                    row.get("acceptance_route") == "strong_topology_illumination_split"
                    for row in rows
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    direct = [row for row in rows if row["status"] == "direct"]
    if direct:
        row = direct[len(direct) // 2]
        frame = cv2.imread(os.fspath(out_dir / frames_dir / f"pt{row['point']:04d}" / row["frame"]))
        court.draw_validation(
            frame,
            homographies[row["point"]],
            os.fspath(out_dir / "court_validation.jpg"),
        )
    return {
        "points": len(rows),
        "direct": len(direct),
        "abstained": len(rows) - len(direct),
        "match_prior_recovered": match_prior_recovered,
        "npz": os.fspath(output),
        "frame_track_npz": os.fspath(frame_track_path) if frame_track_path else None,
        "evidence": os.fspath(evidence_path),
    }


def write_frame_track(out_dir: Path, tracks: dict[int, dict]) -> Path | None:
    """Flatten per-point frame tracks into one per-frame court homography artifact."""
    if not tracks:
        return None
    clips, frames, matrices, reliable, sources = [], [], [], [], []
    for point in sorted(tracks):
        track = tracks[point]
        clip = f"pt{point:04d}"
        for index in range(len(track["frames"])):
            clips.append(clip)
            frames.append(int(track["frames"][index]))
            matrices.append(track["H"][index])
            reliable.append(bool(track["reliable"][index]))
            sources.append(str(track["source"][index]))
    output = out_dir / "court_H_per_frame_v1.npz"
    np.savez_compressed(
        output,
        clips=np.asarray(clips),
        frames=np.asarray(frames, dtype=np.int32),
        H=np.stack(matrices),
        reliable=np.asarray(reliable, dtype=bool),
        source=np.asarray(sources, dtype=str),
    )
    return output


def load_frame_track(path: Path) -> dict[str, dict[int, tuple[np.ndarray, bool, str]]]:
    """Read ``court_H_per_frame_v1.npz`` into {clip: {frame: (H, reliable, source)}}."""
    track: dict[str, dict[int, tuple[np.ndarray, bool, str]]] = {}
    with np.load(path) as data:
        for clip, frame, homography, reliable, source in zip(
            data["clips"],
            data["frames"],
            data["H"],
            data["reliable"],
            data["source"],
            strict=True,
        ):
            track.setdefault(str(clip), {})[int(frame)] = (
                np.asarray(homography, dtype=float),
                bool(reliable),
                str(source),
            )
    return track


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-dir", default="rally_frames")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--registration-stride", type=int, default=DEFAULT_REGISTRATION_STRIDE)
    parser.add_argument("--anchor-mode", choices=ANCHOR_MODES, default=DEFAULT_ANCHOR_MODE)
    parser.add_argument(
        "--edge-convention",
        choices=EDGE_CONVENTIONS,
        default=ARTIFACT_EDGE_CONVENTION,
        help="painted-line convention the emitted homographies are expressed in",
    )
    parser.add_argument(
        "--no-frame-track",
        action="store_true",
        help="keep the legacy one-homography-per-point behaviour",
    )
    parser.add_argument(
        "--court-registration-mask",
        choices=REGISTRATION_MASK_POLICIES,
        default="off",
        help="exclude existing court observation margins from registration features, always or only after rejection",
    )
    parser.add_argument(
        "--interpolated-camera-policy",
        choices=INTERPOLATED_CAMERA_POLICIES,
        default="off",
        help="qualify each interpolation on its native picture and retry unsupported rows from the original anchor (default: off)",
    )
    parser.add_argument(
        "--court-surface-witness",
        choices=SURFACE_WITNESS_POLICIES,
        default=PIPELINE_SURFACE_WITNESS_POLICY,
        help="reconsider a strong-topology candidate on a two-lightness-cluster surface witness, only for a point the established passes left without an anchor (default: off)",
    )
    args = parser.parse_args()
    report = calibrate_points(
        args.out,
        args.frames_dir,
        args.jobs,
        stride=args.registration_stride,
        frame_track=not args.no_frame_track,
        anchor_mode=args.anchor_mode,
        edge_convention=args.edge_convention,
        registration_mask=args.court_registration_mask,
        interpolated_camera_policy=args.interpolated_camera_policy,
        surface_witness_policy=args.court_surface_witness,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
