"""Search the first-contact depth of a frozen multi-flight agent attempt.

Evaluation only. This generalizes the opened single-flight search without
changing automatic inference: it holds each coarse first-contact court-Y
hypothesis fixed, refits the entire connected point, scores short/medium event
windows in both directions, and refines only candidates that pass player,
bounce, net/ground, and serve-region evidence. Fractional event epochs are
profiled within their declared domains. Native net seeds disclose any reserved
check-picture rays consumed before selection.
"""

from __future__ import annotations

import argparse
import csv
import copy
from collections.abc import Callable
from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import (
    native_seed_check_pixels,
    agent_attempt_prepare,
    agent_single_flight_search as single,
    athlete_priors,
    candidate_attempts,
    camera_geometry,
    ending_witnesses,
    event_constraints,
    event_recovery,
    initialization,
    interior_contact_epochs,
    model,
    player_position,
    player_state_fallback,
    real_bidirectional_search as whole,
    real_exposure_replay as exposure,
    search_reporting,
    search_budget,
    serve_reach_cylinder,
    serve_speed_witness,
    toss_witness,
)
from cv.pipeline import paths, provenance, serve_location_prior as serve_location
from cv.pipeline import s6_first_contact_role as contact_role
from cv.pipeline import s6_rally_origin_cue as rally_cue
from cv.validation.flight_gate_audit import (
    bounce_graded_radius_m,
    transverse_metres_per_pixel,
)
from cv.validation import s6_sparse_owner_replay as replay
from cv.validation import playstyle_pattern


BOUNCE_WITNESS_MODES = ("native_ray_average", "subframe_graded_circle")
BOUNCE_WING_SIZE = 4
BOUNCE_WING_MINIMUM = 2
BOUNCE_WING_RADIUS_FRAMES = 12
BOUNCE_LABEL_SIGMA_FLOOR_PX = 2.0
#: Owner-approved 2026-09-19 (change M, in person: "increasing ball labeling uncertainty
#: also makes sense - agreed").  A human click declares its own uncertainty radius; an
#: automatic detector row declares none at all -- `automatic_ball_track` deliberately sets
#: `uncertainty_radius_px1080` to None, because its innovation spread is not an observation
#: noise -- so `radii_px` is EMPTY on every automatic-ball arm and each automatic wing row
#: was handed the 2.0 px human-click floor above.  Denser automatic wings then made the
#: two-wing bounce witness MORE certain than the labelled arm on the same flight, shrinking
#: the graded circle and flipping the 0.75 m bounce-ray gate closed.
#:
#: This is the measured replacement, not a guess.  Pairing every AL automatic row with the
#: LL human row on the same native frame across the whole fresh36 development panel
#: (10,659 in-track frames of 12,145 paired; `loss_scoping_20260919/fixes/automatic_sigma_v1/`)
#: the automatic track's own localization error against the human click is median 1.51 px
#: and one sigma (the 68.27th percentile of the measured distance) 3.12 px.  Frames beyond
#: 25 px are a wrong-object lock rather than mislocalization -- a different failure, and the
#: jump filter's job -- so they are excluded from the sigma and left to that filter.
#: Rounded up to 3.2 px: still below the 4.0 px the human arm declares at the same
#: percentile, so this does not over-claim uncertainty either.
AUTOMATIC_BALL_SIGMA_FLOOR_PX = 3.2
ATHLETE_PRIOR_MODES = ("global_hard_caps", "stature_pose_soft")
# ``required``: the stature arm needs a roster order and a stature for every
# labeled contact player (the original labeled contract).  ``optional``: supplied
# evidence is used identically; with no roster at all the stature-scaled terms
# abstain explicitly.  Statures without an order are malformed under both.
ATHLETE_EVIDENCE_POLICIES = ("required", "optional")
# ``off``: an evidence-qualified recovery that cannot find any sided row inside
# its bounded radius refuses the attempt, exactly as before.  ``abstain``: it may
# instead return a state with an explicitly absent court position, keeping the
# independently established side and the supplied name/stature, so that
# root-dependent soft terms abstain rather than the whole point being lost.
MISSING_PLAYER_POSITION_POLICIES = ("off", "abstain")
CONTACT_XY_ENVELOPE_M = event_constraints.CONTACT_XY_ENVELOPE_M
REFERENCE_COARSE_ITERATIONS = 100


def abstained_toss_observations(
    error: Exception, contact_frame: float, contact_interval: tuple[float, float]
) -> dict:
    """Turn an unusable optional toss document into an explicit abstention."""
    return {
        "clip": None,
        "contact_frame": float(contact_frame),
        "contact_frame_interval": [float(value) for value in contact_interval],
        "rows": [],
        "labeled_fronts": 0,
        "automatic_fallbacks": 0,
        "status": "abstained",
        "abstention_reason": f"{type(error).__name__}: {error}",
        "minimum_observations": toss_witness.CONTACT_CONSTRAINT_CONFIG.minimum_observations,
        "native_timestamps_changed": False,
        "visible_labels_replaced": 0,
    }


def contact_player_feet_fallback(player: dict, error: Exception) -> dict:
    """Use the already-associated sided player box when feet history is absent.

    With the position explicitly absent there is no box to stand in for feet, so
    the court position abstains as ``None`` rather than naming a place.
    """
    root = player_position.root_xy(player)
    return {
        "side": player["side"],
        "track_id": None,
        "court_xy_m": None if root is None else list(map(float, root)),
        **({} if root is not None else {"court_position": player["player_position_evidence"]}),
        "history_frames": [int(player["frame"])],
        "history_count": 1,
        "history_status": "contact_player_state_fallback",
        "history_span_frames": 0,
        "association_frame": int(player["frame"]),
        "court_xy_median_absolute_deviation": [0.0, 0.0],
        "association_distance_px": player.get("image_association_distance_px"),
        "image_coordinate_scale": player.get("image_coordinate_scale", 1.0),
        "fallback_reason": f"{type(error).__name__}: {error}",
        "interpretation": (
            "automatic sided contact player box used because a pre-contact feet history "
            "was unavailable; soft prior input, never an acceptance precondition"
        ),
    }


def serve_ending_consistency(
    ending_kind: str | None,
    striker_end: str,
    modeled_bounce_xyz_m: list[float] | np.ndarray | None,
    witness_bounce_xyz_m: list[float] | np.ndarray | None,
) -> dict:
    """Require a labeled serve fault/out to be out in model and witness."""
    normalized = (ending_kind or "").strip().lower().replace("-", "_")
    applicable = normalized.startswith("serve_") and any(
        token in normalized for token in ("fault", "out")
    )

    def classify(xyz):
        if xyz is None:
            return {"class": "missing", "status": "abstained"}
        values = np.asarray(xyz, float)
        if values.shape != (3,) or not np.isfinite(values).all():
            return {"class": "missing", "status": "abstained"}
        return playstyle_pattern.serve_placement(values[0], values[1], striker_end)[2]

    modeled = classify(modeled_bounce_xyz_m)
    witness = classify(witness_bounce_xyz_m)
    passed = not applicable or (modeled["class"] == "fault" and witness["class"] == "fault")

    def serializable(xyz):
        return None if xyz is None else np.asarray(xyz, float).tolist()

    return {
        "applicable": applicable,
        "ending_kind": ending_kind,
        "striker_end": striker_end,
        "modeled_bounce_xyz_m": serializable(modeled_bounce_xyz_m),
        "witness_bounce_xyz_m": serializable(witness_bounce_xyz_m),
        "modeled_call": modeled,
        "witness_call": witness,
        "passed": bool(passed),
        "contract": "labeled serve fault/out requires both modeled and witnessed fault",
    }


def first_contact_epoch_prior(
    event: dict,
    scene: model.Scene,
    toss_observations: dict,
    *,
    initial_frame: float | None = None,
    stay_inside_labeled_bracket: bool = False,
) -> dict:
    """Build the soft labeled-epoch prior and its evidence-supported search interval."""
    target = float(event["frame"])
    low, high = map(float, event.get("frame_interval", [target, target]))
    postcontact = np.asarray(scene.observation_frames[0], float)
    early_pictures = postcontact[(target <= postcontact) & (postcontact <= target + 2.0)]
    support = []
    if toss_observations.get("status") == "supported" and toss_observations.get("rows"):
        support.append("precontact_toss_arc")
    if len(early_pictures) >= 2:
        support.append("first_two_postcontact_pictures")
    lower = low if stay_inside_labeled_bracket else (min(low, target - 1.0) if support else low)
    upper = min(high, float(postcontact[0]))
    if not np.isfinite([target, low, high, lower, upper]).all() or lower > target or target > upper:
        raise ValueError(
            "labeled first-contact bracket must contain its epoch and precede pictures"
        )
    sigma = max((high - low) / 2.0, 0.25)
    return {
        "target_frame": target,
        "initial_frame": (
            max(lower, target - 0.75)
            if initial_frame is None and support
            else target
            if initial_frame is None
            else float(np.clip(initial_frame, lower, upper))
        ),
        "sigma_frames": sigma,
        "bounds_frames": [lower, upper],
        "labeled_bracket_frames": [low, high],
        "early_extension_support": support,
        "maximum_early_shift_frames": 1.0 if support else target - lower,
        "soft_prior": True,
        "native_exposure_times_changed": False,
        "stays_inside_labeled_bracket": stay_inside_labeled_bracket,
    }


def serve_start_location_prior(
    labels: dict,
    player: dict,
    first_bounce_target: dict,
    serve_number: int | None,
    toss_estimate: dict | None,
    *,
    artifact_path: Path | None = None,
) -> tuple[dict, dict]:
    """Resolve the external mixture, optional toss intersection and its 3-sigma box."""
    server_end = str(player["side"])
    receiver_end = playstyle_pattern.receiving_end(server_end)
    bounce_x = float(first_bounce_target["xyz_m"][0])
    serve_side = playstyle_pattern.court_side(bounce_x, receiver_end)["class"]
    if serve_side not in {"deuce", "ad"}:
        raise ValueError("serve bounce witness must resolve a deuce/ad service box")
    prior = serve_location.serve_location_prior(
        player.get("player"),
        serve_side,
        serve_number,
        player.get("stature_m"),
        end=server_end,
        match_id=labels.get("match_id"),
        artifact_path=artifact_path,
    )
    if prior.get("status") != "supported":
        raise ValueError("external serve-location prior has no supported fallback")
    try:
        combined = (
            prior
            if toss_estimate is None
            else serve_location.intersect_toss_estimate(prior, toss_estimate)
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        combined = {
            **prior,
            "toss_intersection": "abstained_invalid_toss_estimate",
            "toss_intersection_blocker": f"{type(error).__name__}: {error}",
        }

    def mode_box(document: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        component = document["components"][document["mode_component_index"]]
        mean = np.asarray(component["mean_xyz_m"], float)
        covariance = np.asarray(component["covariance_xyz_m2"], float)
        sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if mean.shape != (3,) or sigma.shape != (3,) or np.any(sigma <= 0):
            raise ValueError("serve-location mode requires a finite positive diagonal sigma")
        low, high = mean - 3.0 * sigma, mean + 3.0 * sigma

        # The owner-specified standing lane is part of the prior's support: at most
        # 1.0 m behind and 0.6 m inside the baseline, and only the service-box half.
        if server_end == "near":
            low[1], high[1] = max(low[1], -1.0), min(high[1], 0.6)
        else:
            low[1], high[1] = max(low[1], 23.17), min(high[1], 24.77)
        centre = playstyle_pattern.CENTRE_X_M
        if (server_end == "near" and serve_side == "deuce") or (
            server_end == "far" and serve_side == "ad"
        ):
            low[0] = max(low[0], centre)
        else:
            high[0] = min(high[0], centre)
        return mean, sigma, low, high

    mean, sigma, low, high = mode_box(combined)
    if np.any(low >= high) and combined is not prior:
        combined = {
            **prior,
            "toss_intersection": "abstained_toss_three_sigma_misses_service_lane",
        }
        mean, sigma, low, high = mode_box(combined)
    if np.any(low >= high):
        raise ValueError("external serve-location prior misses the legal service lane")
    return combined, {
        "x_interval_m": [float(low[0]), float(high[0])],
        "y_interval_m": [float(low[1]), float(high[1])],
        "z_interval_m": [float(low[2]), float(high[2])],
        "target_xyz_m": mean.tolist(),
        "sigma_m": sigma.tolist(),
        "bound_kind": "three_sigma_parameter_box_intersected_with_service_lane",
        "optimizer_inequality_added": False,
        "server_end": server_end,
        "serve_side": serve_side,
        "baseline_inward_support_m": [-1.0, 0.6],
    }


def load_contact_history(path: Path | None, attempt_id: str) -> tuple[list[dict], dict]:
    """Load only earlier accepted contacts explicitly supplied for this attempt."""
    if path is None:
        return [], {"status": "abstained", "reason": "no_contact_history_artifact"}
    document = json.loads(path.read_text())
    if document.get("schema") != "connected_serve_contact_history_v1":
        raise ValueError("unsupported serve-contact history schema")
    rows = document.get("attempts", {}).get(attempt_id, [])
    if not isinstance(rows, list):
        raise ValueError("serve-contact history attempt entry must be a list")
    return rows, {
        "status": "supported" if rows else "abstained",
        "reason": None if rows else "no_earlier_accepted_contact_for_attempt",
        "source": provenance.file_record(path),
        "row_count": len(rows),
        "causal_order_asserted": bool(document.get("causal_order_asserted")),
    }


def input_domain_rank(candidate):
    from cv.experiments.connected_shooting.labeled_terminal_net_tail import candidate_input_rank

    return candidate_input_rank(candidate)


def select_reference_restart(primary: dict | None, reference: dict) -> dict:
    """Keep the lower input-only ranked branch without reading held-out evidence."""
    candidates = (
        (("active_arm", primary), ("inequalities_in_reference", reference))
        if primary is not None
        else (("inequalities_in_reference", reference),)
    )
    selected_name, selected = min(candidates, key=lambda item: input_domain_rank(item[1]))
    selected["reference_restart"] = {
        "selected": selected_name,
        "selection_rule": (
            "lowest all-native objective-input rank; no independent pixel check"
            if "observation_partition" in selected["measurement"]
            else "lowest input-only rank score; no withheld pixel is read"
        ),
        "reference_configuration": {
            "refine_inequalities": "in",
            "anchor_residuals": "absent",
            "seed_restarts": "off",
            "coarse_iterations": REFERENCE_COARSE_ITERATIONS,
        },
        "candidates": {
            name: {
                "training_rms_px": row["measurement"]["rms_px"]["training"],
                "input_only_rank_score": row["evidence"]["input_only_rank_score"],
                "survived": row["evidence"]["survived"],
            }
            for name, row in candidates
        },
    }
    if "input_event_domain" in selected["measurement"]:
        selected["reference_restart"]["selection_rule"] = (
            "original input event domain eligibility, then lowest input-only rank score"
        )
    return selected


def stage_refine_options(options: dict, stage: str) -> dict:
    """Apply second-stage-only solver arms without mutating the caller's options."""
    stage_options = dict(options)
    if stage == "coarse":
        stage_options.pop("shared_contact_states", None)
        stage_options.pop("local_flight_refits", None)
    return stage_options


def second_stage_reference(document: dict, attempt_id: str) -> tuple[list[dict], dict, list[dict]]:
    """Validate and copy the promoted rank-one input for shared-state refinement."""
    if document.get("schema") != "s6_agent_whole_point_search_v2":
        raise ValueError("second-stage reference must be a whole-point search report")
    if document.get("attempt_id") != attempt_id:
        raise ValueError("second-stage reference attempt does not match the prepared packet")
    arm = document.get("configuration", {}).get("solver_experimental_arm", {})
    if (
        arm.get("refine_inequalities") != "terminal_only"
        or arm.get("anchor_residuals") != "present"
        or arm.get("seed_restarts") != "three"
        or document.get("configuration", {}).get("coarse_iterations") != 150
    ):
        raise ValueError("second-stage reference must be the promoted terminal-only arm")
    rank_one = document.get("selected") or document.get("diagnostic_candidate")
    coarse = document.get("coarse_candidates")
    refined = document.get("refined_candidates")
    if rank_one is None or not coarse or not refined:
        raise ValueError("second-stage reference requires rank-one, coarse and refined families")
    return copy.deepcopy(coarse), copy.deepcopy(rank_one), copy.deepcopy(refined)


def contact_family_widths(candidates: list[dict]) -> dict:
    """Report boundary-state spread across depth candidates, never hide a depth slide."""
    rows = []
    for candidate in candidates:
        flights = candidate["measurement"]["dense_flights"]
        rows.append(
            np.asarray(
                [flight["start_xyz"] for flight in flights] + [flights[-1]["end_xyz"]], float
            )
        )
    if not rows:
        return {"candidate_count": 0, "contacts": []}
    if len({row.shape for row in rows}) != 1:
        raise ValueError("one boundary-state inventory required across a depth family")
    values = np.asarray(rows)
    contacts = []
    for index in range(values.shape[1]):
        points = values[:, index]
        pairwise = points[:, None, :] - points[None, :, :]
        contacts.append(
            {
                "boundary_index": index,
                "maximum_pairwise_width_m": float(np.linalg.norm(pairwise, axis=2).max()),
                "coordinate_span_m": np.ptp(points, axis=0).tolist(),
            }
        )
    return {"candidate_count": len(rows), "contacts": contacts}


def observation_accounting(rows: list[dict]) -> dict:
    """Make abstentions explicit without turning them into physical events."""
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    counts: dict[str, int] = {}
    for row in ordered:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    runs = []
    active = []
    for row in ordered:
        if row["status"] != "visible":
            if active and int(row["frame"]) != int(active[-1]["frame"]) + 1:
                runs.append(active)
                active = []
            active.append(row)
        elif active:
            runs.append(active)
            active = []
    if active:
        runs.append(active)
    return {
        "frame_count": len(ordered),
        "status_counts": counts,
        "abstention_runs": [
            {
                "start_frame": int(run[0]["frame"]),
                "end_frame": int(run[-1]["frame"]),
                "statuses": sorted({str(row["status"]) for row in run}),
            }
            for run in runs
        ],
        "gap_policy": (
            "non-visible observations are omitted from the image objective; the connected "
            "flight propagates continuously through them and they never imply termination"
        ),
    }


def scoped_branch_events(
    events: list[dict], end_frame: float, *, horizon_contract: dict | None = None
) -> tuple[list[dict], dict]:
    """Apply the branch's physical-event boundary before terminal grammar.

    A qualified unresolved tail owns its observed horizon. Its source packet
    may retain an earlier original ending for ancestry; that earlier epoch must
    not discard a newly qualified ground before the scene is constructed.
    Ordinary physical endings and present-net branches retain their boundary.
    """
    if horizon_contract is not None:
        end_frame = float(horizon_contract["observation_horizon"])
    scope = {
        "modeled_branch_end_frame": float(end_frame),
        "excluded_events": [
            copy.deepcopy(row) for row in events if float(row["frame"]) > end_frame
        ],
        "rule": "event representative at or before declared modeled branch horizon",
        "original_event_records_preserved": True,
    }
    return [row for row in events if float(row["frame"]) <= end_frame], scope


#: Ball-track endings where the picture, not the ground, stops the flight.
PICTURE_EXIT_ENDING_KINDS = frozenset({"fov_exit", "last_visible_sample"})


def prepare_attempt(
    attempt: dict,
    cameras_document: dict,
    surface: str,
    events: list[dict] | None = None,
    end_frame: float | None = None,
    *,
    observation_fallback: bool = False,
    fallback_receipt: list[dict] | None = None,
    observation_partition: str = "fifth_frame_withheld",
    ground_settling: bool = False,
    bounce_witness_observation_policy: str = (
        "legacy_fit_frames_including_existing_terminal_activation"
    ),
    exit_partial_endings: bool = False,
) -> tuple:
    """Build a connected scene while preserving a two-bounce terminal flight.

    The older validation adapter intentionally rejects this grammar.  This
    experiment-local adapter keeps the same train/check ownership but permits
    exactly two ordered bounces only in the final flight.

    ``events``/``end_frame`` let event recovery build the same scene from a
    candidate topology instead of the supplied one.  Omitting both keeps the
    supplied topology, which is what every reference arm uses.

    ``exit_partial_endings`` declares a ball-track picture exit or last visible
    sample as a partial flight, like a camera cut: the flight is not required to
    end at its last supplied bounce and no ground ending is claimed.
    """
    from cv.experiments.connected_shooting import observation_partition as partition

    partition.validate(observation_partition)
    events = attempt["events"] if events is None else events
    net_tail = attempt.get("terminal_net_tail")
    horizon_tail = attempt.get("observed_horizon_tail")
    if horizon_tail is not None:
        from cv.experiments.connected_shooting.observed_horizon_tail import for_events

        horizon_tail = for_events(horizon_tail, events)
    end_frame = attempt["owner_end_frame"] if end_frame is None else end_frame
    from cv.experiments.connected_shooting import source_flight_coverage as coverage
    from cv.pipeline import s6_contact_prefix_runtime as contact_prefix

    from cv.pipeline import s6_component_scope

    component_ending = s6_component_scope.active(attempt)
    terminal_track = s6_component_scope.track_terminal(attempt)
    # A ball-track terminal is still a component, but its right boundary is the
    # ball's own ending rather than the next contact. Horizon, recovery and
    # net-tail mechanisms stay off.
    contact_ending = (contact_prefix.active(attempt) or component_ending) and terminal_track is None
    if component_ending:
        s6_component_scope.validate(attempt)
        if end_frame != attempt["owner_end_frame"] or events != attempt["events"]:
            raise ValueError("component retains original events and contact endpoint")
        if net_tail is not None or horizon_tail is not None or ground_settling:
            raise ValueError("component contact boundary cannot consume terminal features")
    elif contact_ending:
        from cv.pipeline import s6_contact_prefix_scope

        mode = contact_prefix.bound_mode(attempt)
        # The source recomputation of a terminal-identity prefix needs the source
        # observation rows, which this per-branch scene builder does not carry; it
        # runs once per invocation in ``run_topology`` below, in the shared stage
        # and in the exporter.  Coverage keeps its unchanged per-branch replay.
        contract = s6_contact_prefix_scope.validate(
            attempt,
            cameras_document if mode == s6_contact_prefix_scope.COVERAGE else None,
            observation_partition=observation_partition,
        )
        if float(end_frame) != float(attempt["owner_end_frame"]) or (
            events != attempt["events"] and mode != s6_contact_prefix_scope.TERMINAL_IDENTITY
        ):
            raise ValueError("contact prefix retains its original events and right boundary")
        if events != attempt["events"]:
            s6_contact_prefix_scope.checked_interior_events(contract, attempt["events"], events)
        if net_tail is not None or horizon_tail is not None or ground_settling:
            raise ValueError("contact right boundary cannot consume terminal features")
    camera_rows = cameras_document["cameras"]
    cameras, labels = coverage.visible_inputs(
        attempt,
        cameras_document,
        observation_fallback=observation_fallback,
        fallback_receipt=fallback_receipt,
    )
    contacts = [float(row["frame"]) for row in events if row["event_type"] == "contact"]
    if not contacts:
        raise ValueError("connected scene requires at least one contact boundary")
    bounds = np.asarray(contacts if contact_ending else [*contacts, float(end_frame)])
    # The joint rigid-court label camera ships one shared radial k1 and its
    # distortion centre on every camera row. Consume it when every row declares
    # it, so the fitter projects through the same camera the packet was solved
    # with; a packet without it stays an explicit pinhole scene.
    radial = {}
    if all("k1" in row and "dist_center" in row for row in camera_rows):
        for frame, row in cameras.items():
            center = np.asarray(row["dist_center"], dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(row["k1"]):
                raise ValueError("finite shared k1 and distortion centre required")
            radial[frame] = np.asarray([float(row["k1"]), center[0], center[1]], dtype=float)
    elif any("k1" in row or "dist_center" in row for row in camera_rows):
        raise ValueError("a radial camera packet must declare k1 on every row")

    train, heldout, native, bounces = [], [], [], []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        frames, fit_frames, check_frames = coverage.frame_partition(
            labels,
            start,
            end,
            inclusive_end=index == len(bounds) - 2 and not contact_ending,
            observation_partition=observation_partition,
        )
        group = np.asarray(
            [
                float(row["frame"])
                for row in events
                if row["event_type"] == "bounce" and start < float(row["frame"]) <= end
            ]
        )
        terminal_allowed = (
            {0} if net_tail is not None else {0, 1, 2} if horizon_tail is not None else {1, 2}
        )
        if index == len(bounds) - 2 and terminal_track is not None:
            from cv.pipeline import s6_contact_components as _terminal_components

            allowed = set(_terminal_components.terminal_bounce_allowance(terminal_track["kind"]))
        elif index == len(bounds) - 2 and not contact_ending:
            allowed = terminal_allowed
        else:
            allowed = {0, 1}
        if len(group) not in allowed:
            raise ValueError(
                "zero/one bounce before a volley contact, or one/two terminal bounces required"
            )
        native.append(frames)
        bounces.append(group)
        for selected, output, minimum in (
            (fit_frames, train, 4),
            (check_frames, heldout, 1),
        ):
            if len(selected) < minimum:
                raise ValueError(f"flight {index} lacks fixed train/check coverage")
            output.append(
                (
                    selected,
                    np.asarray([cameras[int(frame)]["P"] for frame in selected]),
                    np.asarray(
                        [
                            [labels[int(frame)]["x1080"], labels[int(frame)]["y1080"]]
                            for frame in selected
                        ]
                    ),
                    np.asarray([radial[int(frame)] for frame in selected]) if radial else None,
                )
            )
    inside = [
        frame
        for frame in labels
        if bounds[0] <= frame and (frame < bounds[-1] or not contact_ending and frame == bounds[-1])
    ]
    if sum(map(len, native)) != len(inside):
        raise ValueError("every visible in-window observation must belong to one flight")
    outside_window = sorted(frame for frame in labels if frame not in set(inside))

    def make_scene(rows) -> model.Scene:
        scene = model.Scene(
            contact_frames=bounds,
            observation_frames=tuple(row[0] for row in rows),
            cameras=tuple(row[1] for row in rows),
            pixels=tuple(row[2] for row in rows),
            camera_distortion=tuple(row[3] for row in rows) if radial else None,
            spin_parameters=np.tile([2.0, 0.0, 0.0], (len(rows), 1)),
            fps=float(attempt["fps"]),
            surface=surface,
            dynamics="measured_240hz",
            rebound_mode="point_scales",
            right_boundary_kind="original_contact" if contact_ending else "supplied_end",
            terminal_net_tail=net_tail,
            observed_horizon_tail=horizon_tail,
            observation_partition=observation_partition,
            bounce_witness_observation_policy=bounce_witness_observation_policy,
            ground_settling=ground_settling,
            supported_ending=(
                terminal_track
                if terminal_track is not None
                and terminal_track.get("kind")
                in {"net_stop", "camera_cut", "held_camera", "point_end", "dead_ball"}
                else {**terminal_track, "partial_flight": True}
                if exit_partial_endings
                and terminal_track is not None
                and terminal_track.get("kind") in PICTURE_EXIT_ENDING_KINDS
                else None
            ),
        )
        scene.validate()
        return scene

    return (
        make_scene(train),
        make_scene(heldout),
        tuple(bounces),
        tuple(native),
        outside_window,
    )


def seed_ground_targets(
    scene: model.Scene,
    bounce_events: list[dict],
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    radii_px: dict[int, float] | None,
    *,
    mode: str,
    observation_fallback: bool,
    camera_distortion: dict[int, np.ndarray] | None = None,
    terminal_rebound_frames=None,
) -> list[np.ndarray] | None:
    """One measured court impact per flight for the image seed, where one exists.

    Seeding only.  A flight with no supplied bounce, or a bounce whose bracket
    has no usable native ground ray, contributes ``None`` and is seeded from its
    pictures exactly as before.  The gate that scores a fitted bounce against
    its ray is untouched and still built separately.
    """
    from cv.experiments.connected_shooting.observation_partition import witness_frames

    if not observation_fallback:
        return None
    rows: list[np.ndarray | None] = []
    for flight_index, (start, end) in enumerate(
        zip(scene.contact_frames, scene.contact_frames[1:])
    ):
        group = [row for row in bounce_events if start < float(row["frame"]) <= end]
        target = None
        if group:
            try:
                witness = event_ground_target(
                    group[0],
                    cameras,
                    labels,
                    radii_px,
                    mode=mode,
                    observation_fallback=True,
                    eligible_frames=witness_frames(scene, flight_index, terminal_rebound_frames),
                    camera_distortion=camera_distortion,
                )
                target = np.asarray(witness["xyz_m"], float)[:2]
            except (ValueError, np.linalg.LinAlgError):
                target = None
        rows.append(target)
    return rows


def pre_net_seed_mask(frames: np.ndarray, lower: float, duration: float | None) -> np.ndarray:
    """Centers at an uncertain net boundary are not proven incoming samples.

    Preserve the historical closed-front seed rule for exact labeled parity;
    a center has no forward exposure and must strictly precede the event.
    """
    return frames < lower if duration is None else frames + duration <= lower


def terminal_seed_start(scene, parameters):
    """Read the final launch from its prefix, before simulating the provisional tail.

    The pre-net image initializer has no post-net response yet. Its terminal
    extrapolation can exceed the physical bounce capacity even when every
    preceding flight and the final launch are finite. Only the preceding chain
    determines this start; the full tail remains mandatory in the later solve.
    """
    n = len(scene.pixels)
    p = np.asarray(parameters, float)
    scales = 2 if scene.rebound_mode == "point_scales" else 0
    if p.shape != (3 + 6 * n + scales,) or not np.isfinite(p).all():
        raise ValueError("finite explicit velocity/spin seed required")
    if scene.parameterization != "single_shooting" or scene.rebound_mode not in {
        "fixed",
        "point_scales",
    }:
        raise ValueError("terminal start requires a connected velocity/spin seed")
    if n == 1:
        return p[:3].copy()
    count = n - 1
    fields = {
        name: getattr(scene, name)[:count]
        for name in ("observation_frames", "cameras", "pixels", "spin_parameters")
    }
    for name in ("camera_distortion", "net_hit_frames"):
        value = getattr(scene, name)
        fields[name] = None if value is None else value[:count]
    override = scene.bounce_regime_override
    prefix = replace(
        scene,
        contact_frames=scene.contact_frames[: count + 1],
        terminal_net_tail=None,
        observed_horizon_tail=None,
        bounce_regime_override=override if override is not None and override[0] < count else None,
        **fields,
    )
    values = np.r_[
        p[: 3 + 3 * count],
        p[3 + 3 * n : 3 + 3 * n + 3 * count],
        p[-scales:] if scales else np.empty(0),
    ]
    return model.chain(prefix, values)[-1]["end_xyz"].copy()


GROUND_LAUNCH_DEATH = "descending initial ground state has no supported flight"


def baseline_seed(
    document: dict | None,
    scene: model.Scene,
    bounces: tuple,
    *,
    bounce_ground_targets: list | None = None,
    observation_fallback: bool = False,
    fallback_receipt: list[dict] | None = None,
    terminal_rebound: bool = False,
    exposure_duration: float | None = 0.25,
) -> tuple[np.ndarray, dict]:
    if document is not None:
        candidates = [
            document.get("fit", {}).get("parameters"),
            (document.get("selected") or {})
            .get("measurement", {})
            .get("fit", {})
            .get("parameters"),
            document.get("results", {}).get("front_d025", {}).get("fit", {}).get("parameters"),
        ]
        values = next((row for row in candidates if row is not None), None)
        if values is None:
            raise ValueError("baseline document has no connected parameter vector")
        return np.asarray(values, float), {"method": "explicit_baseline_connected_parameters"}
    if scene.terminal_net_tail is not None:
        from cv.experiments.connected_shooting.labeled_net_epoch_chart import project_net_velocity

        low, _ = scene.terminal_net_tail["interval"]
        masks = [np.ones(len(f), bool) for f in scene.observation_frames]
        masks[-1] = pre_net_seed_mask(scene.observation_frames[-1], low, exposure_duration)
        seed_scene = replace(
            scene,
            terminal_net_tail=None,
            net_hit_frames=None,
            observation_frames=tuple(
                f[m] for f, m in zip(scene.observation_frames, masks, strict=True)
            ),
            cameras=tuple(f[m] for f, m in zip(scene.cameras, masks, strict=True)),
            pixels=tuple(f[m] for f, m in zip(scene.pixels, masks, strict=True)),
            camera_distortion=None
            if scene.camera_distortion is None
            else tuple(f[m] for f, m in zip(scene.camera_distortion, masks, strict=True)),
        )
        seeded, evidence = baseline_seed(
            None,
            seed_scene,
            bounces,
            bounce_ground_targets=bounce_ground_targets,
            observation_fallback=observation_fallback,
            fallback_receipt=fallback_receipt,
            exposure_duration=exposure_duration,
        )
        count = len(scene.pixels)
        v, w = 3 + 3 * (count - 1), 3 + 3 * count + 3 * (count - 1)
        receipt = {
            "incoming_original_frames": seed_scene.observation_frames[-1].tolist(),
            "tail_excluded_from_seed_only": True,
            "all_tail_retained_in_objective": True,
        }
        try:
            start = terminal_seed_start(seed_scene, seeded)
        except ValueError as error:
            # The net-stop launch is the prefix chain's end. An automatic image
            # seed can already be skimming the ground off court, so the flight
            # before the net hits the bounce cap before any candidate exists.
            # That is seed death: the component partition still fits the point.
            # A bare bounce cap, and any other terminal kind, still propagate.
            from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

            if type(error) is ValueError and str(error) == GROUND_LAUNCH_DEATH:
                # A prefix launch already on the ground and descending (a short
                # automatic flight between close contacts) only denies the
                # net-plane refinement. Like a refused projection, keep the
                # seed; the search judges it with every other candidate.
                receipt.update(status="prefix_launch_refused_seed_retained", reason=str(error))
                return seeded, {**evidence, "terminal_net_tail_seed": receipt}
            tail = scene.terminal_net_tail or {}
            if type(error) is not BounceCapacityError or tail.get("kind") != "net_stop":
                raise
            raise ValueError(
                "net-stop seed measured dynamics bounce cap reached "
                f"at frame {float(error.frame):.6f} after {int(error.resolved_impacts)} impacts"
            ) from error
        theta = np.r_[start, seeded[v : v + 3], seeded[w : w + 3]]
        try:
            projected, projection = project_net_velocity(
                theta,
                scene.contact_frames[-2],
                scene.terminal_net_tail["representative"],
                scene.fps,
                scene.surface,
                rebound_scales=seeded[-2:],
            )
            seeded[v + 1] = projected[4]
            receipt.update(status="plane_projected", projection=projection)
        except (ValueError, FloatingPointError, OverflowError) as error:
            receipt.update(status="plane_projection_refused_seed_retained", reason=str(error))
        return seeded, {**evidence, "terminal_net_tail_seed": receipt}
    first_bounces = [None if not len(group) else float(group[0]) for group in bounces]
    try:
        seed, evidence = initialization.image_ballistic_seed(
            scene,
            first_bounces,
            ground_anchor=True,
            consensus=True,
            bounce_ground_targets=bounce_ground_targets,
        )
    except ValueError as error:
        # A flight whose incoming arc is barely pictured is still bounded by the
        # arc that leaves its bounce.  The existing post-bounce fallback seeds
        # exactly that case; it needs a bounce in every flight, so a volley
        # topology still fails closed with the original reason.
        if not observation_fallback or "pre-bounce training pictures" not in str(error):
            raise
        from cv.experiments.connected_shooting import postbounce_initialization

        # Net transitions remain part of the optimized scene.  The one-flight
        # post-bounce initializer cannot carry a point-wide tuple of net-hit
        # groups into each temporary flight, so omit that metadata only from
        # this terminal-rebound bootstrap copy.
        seed_scene = (
            replace(scene, net_hit_frames=None)
            if terminal_rebound and scene.net_hit_frames is not None
            else scene
        )
        seed, evidence = postbounce_initialization.seed(seed_scene, first_bounces)
        evidence = {**evidence, "refused_image_ballistic_seed": str(error)}
        if seed_scene is not scene:
            evidence["terminal_rebound_seed_net_hits_omitted"] = True
        if fallback_receipt is not None:
            fallback_receipt.append(
                {
                    "fallback": "post_bounce_seed_for_an_unpictured_incoming_arc",
                    "refused_with": f"{type(error).__name__}: {error}",
                    "method": evidence["method"],
                }
            )
    return np.r_[seed, [1.0, 1.0]], evidence


def training_directions(
    scene,
    bounces,
    reference: dict,
    *,
    observation_fallback: bool = False,
    exposure_duration: float | None = 0.25,
    fallback_receipt: list[dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """Resolve training axes, using the incoming wing at an exact terminal impact.

    A compact terminal-impact exposure can lack a separately labeled trailing
    tip and belongs to an otherwise empty post-impact wing. In that one case,
    carry the closest available pre-impact training axis forward. This is an
    explicitly attributed observation-model sensitivity, not a new picture or
    a shifted timestamp.
    """
    frames = np.concatenate(scene.observation_frames)
    if exposure_duration is None:
        axes = np.tile([1.0, 0.0], (len(frames), 1))
        return axes, dict(
            frames=frames.tolist(),
            axes=axes.tolist(),
            sources=["unused_for_nominal_center"] * len(frames),
        )
    front = np.concatenate(scene.pixels)
    tails = {}
    for record in reference["records"]:
        for row in record["frames"]:
            streak = row.get("streak", {})
            if streak.get("status") == "paired":
                tails[row["frame"]] = [
                    streak["trailing"]["x1080"],
                    streak["trailing"]["y1080"],
                ]
    back = np.asarray([tails.get(int(frame), [np.nan, np.nan]) for frame in frames])
    visible = exposure.streak_axis_capacity.in_image(front)
    boundaries = np.unique(np.r_[scene.contact_frames, *bounces])
    if scene.terminal_net_tail is not None:
        boundaries = np.unique(np.r_[boundaries, scene.terminal_net_tail["representative"]])
    wings = np.searchsorted(boundaries, frames, side="right")
    axes, available, sources = exposure.streak_axis_capacity.observed_axes(
        front,
        back,
        frames,
        visible,
        exposure.streak_axis_capacity.in_image(back),
        wings,
    )
    sources = list(sources)
    terminal = float(np.max(boundaries))
    for index in np.flatnonzero(~available):
        if scene.right_boundary_kind == "original_contact" or float(frames[index]) != terminal:
            continue
        incoming_wing = np.searchsorted(boundaries, terminal, side="left")
        choices = np.flatnonzero(available & (wings == incoming_wing))
        if len(choices):
            nearest = choices[np.argmin(np.abs(frames[choices] - terminal))]
            if abs(float(frames[nearest]) - terminal) <= 3:
                axes[index] = axes[nearest]
                available[index] = True
                sources[index] = "terminal_incoming_training_axis"
    if not available.all() and observation_fallback:
        # A front whose trailing tip is unavailable has no measured axis.  The
        # terminal rule above already substitutes the nearest incoming training
        # axis; extend exactly that plug-in to any exposure inside the same
        # flight wing.  This is a correlated plug-in direction, not a measured
        # one, and it is recorded per frame.
        substituted = []
        for index in np.flatnonzero(~available):
            wing = wings[index]
            choices = np.flatnonzero(available & (wings == wing))
            source = "in_wing_nearest_training_axis"
            if not len(choices):
                # No measured direction anywhere in this flight; the closest
                # measured direction in the point is the only plug-in available.
                choices = np.flatnonzero(available)
                source = "nearest_training_axis_outside_the_flight"
            if not len(choices):
                continue
            nearest = choices[np.argmin(np.abs(frames[choices] - frames[index]))]
            gap = abs(float(frames[nearest]) - float(frames[index]))
            axes[index] = axes[nearest]
            available[index] = True
            sources[index] = source
            substituted.append({"frame": float(frames[index]), "gap_frames": gap, "source": source})
        if substituted and fallback_receipt is not None:
            fallback_receipt.append(
                {
                    "fallback": "front_direction_plug_in",
                    "substitutions": substituted,
                    "note": (
                        "a correlated plug-in direction with its frame gap recorded, not a "
                        "measured trailing tip"
                    ),
                }
            )
    if not available.all():
        raise ValueError(f"unresolved original front directions at {frames[~available].tolist()}")
    return axes, dict(
        frames=frames.tolist(),
        sources=sources,
        axes=axes.tolist(),
        note=(
            "Correlated plug-in axes; no calibrated angular likelihood or truth direction. "
            "An exact compact terminal-impact front may use the closest <=3-frame incoming "
            "training axis when its trailing tip is explicitly unavailable."
        ),
    )


def _camera_at_epoch(cameras: dict[int, np.ndarray], epoch: float) -> tuple[np.ndarray, list[int]]:
    """Interpolate camera transport at a native epoch without moving an exposure."""
    frames = sorted(cameras)
    if not frames:
        raise ValueError("bounce witness has no supported camera")
    below = max((frame for frame in frames if frame <= epoch), default=None)
    above = min((frame for frame in frames if frame >= epoch), default=None)
    if below is None or above is None or below == above:
        frame = above if below is None else below
        return np.asarray(cameras[frame], float), [int(frame)]
    fraction = (float(epoch) - below) / (above - below)
    camera = (1.0 - fraction) * cameras[below] + fraction * cameras[above]
    return np.asarray(camera, float), [int(below), int(above)]


def _fit_pixel_wing(
    frames: np.ndarray,
    pixels: np.ndarray,
    epoch: float,
    sigmas_px: np.ndarray,
) -> dict:
    """Fit one local image-motion wing, retaining its residual noise."""
    relative = np.asarray(frames, float) - float(epoch)
    design = np.c_[np.ones(len(relative)), relative]
    weights = 1.0 / np.square(np.asarray(sigmas_px, float))
    normal = design.T @ (weights[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (weights[:, None] * np.asarray(pixels, float)))
    residuals = np.asarray(pixels, float) - design @ beta
    residual_rms = float(np.sqrt(np.mean(np.square(residuals))))
    return {
        "intercept": beta[0],
        "velocity_px_per_frame": beta[1],
        "residual_rms_px": residual_rms,
        "effective_sigmas_px": np.sqrt(np.square(sigmas_px) + residual_rms**2),
    }


def _reversal_solution(
    event_frame: float,
    incoming_frames: np.ndarray,
    incoming_pixels: np.ndarray,
    incoming_sigmas: np.ndarray,
    outgoing_frames: np.ndarray,
    outgoing_pixels: np.ndarray,
    outgoing_sigmas: np.ndarray,
    *,
    one_sided: str | None = None,
) -> tuple[float, np.ndarray, dict | None, dict | None]:
    """Return a common impact epoch/pixel from two independent local wings."""
    incoming = (
        None
        if not len(incoming_frames)
        else _fit_pixel_wing(incoming_frames, incoming_pixels, event_frame, incoming_sigmas)
    )
    outgoing = (
        None
        if not len(outgoing_frames)
        else _fit_pixel_wing(outgoing_frames, outgoing_pixels, event_frame, outgoing_sigmas)
    )
    if one_sided is not None:
        selected = incoming if one_sided == "incoming" else outgoing
        if selected is None:
            raise ValueError("requested one-sided bounce wing is unavailable")
        return float(event_frame), np.asarray(selected["intercept"], float), incoming, outgoing
    if incoming is None or outgoing is None:
        raise ValueError("two-sided bounce witness requires both motion wings")
    relative_velocity = np.asarray(incoming["velocity_px_per_frame"], float) - np.asarray(
        outgoing["velocity_px_per_frame"], float
    )
    denominator = float(relative_velocity @ relative_velocity)
    if denominator <= 1e-8:
        raise np.linalg.LinAlgError("bounce motion wings do not resolve a reversal")
    separation = np.asarray(incoming["intercept"], float) - np.asarray(outgoing["intercept"], float)
    offset = -float(separation @ relative_velocity) / denominator
    impact_epoch = float(event_frame + offset)
    incoming_pixel = np.asarray(incoming["intercept"], float) + offset * np.asarray(
        incoming["velocity_px_per_frame"], float
    )
    outgoing_pixel = np.asarray(outgoing["intercept"], float) + offset * np.asarray(
        outgoing["velocity_px_per_frame"], float
    )
    return impact_epoch, 0.5 * (incoming_pixel + outgoing_pixel), incoming, outgoing


def reversal_impact_witness(
    event_frame: float,
    frame_interval: tuple[float, float],
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    radii_px: dict[int, float] | None,
    available_frames: list[int],
    *,
    camera_distortion: dict[int, np.ndarray] | None = None,
    sigma_floor_px: float = BOUNCE_LABEL_SIGMA_FLOOR_PX,
) -> dict:
    """Estimate a bounce from separate image-motion wings and propagate covariance.

    The last/first four visible fronts within twelve native frames are fit as
    independent constant-velocity streaks.  Their closest shared-time approach
    supplies both an impact pixel and a timing witness.  A sole usable wing is
    evaluated at the supplied epoch, with extrapolation leverage represented in
    the finite-difference covariance rather than collapsed to a point.

    ``sigma_floor_px`` defaults to the human-click floor, which reproduces every
    existing witness byte for byte.  A caller reading automatic ball rows passes
    ``AUTOMATIC_BALL_SIGMA_FLOOR_PX`` instead (owner-approved 2026-09-19, change M).
    """
    incoming_frames = np.asarray(
        [frame for frame in available_frames if frame < event_frame][-BOUNCE_WING_SIZE:],
        float,
    )
    outgoing_frames = np.asarray(
        [frame for frame in available_frames if frame > event_frame][:BOUNCE_WING_SIZE],
        float,
    )
    incoming_ok = len(incoming_frames) >= BOUNCE_WING_MINIMUM
    outgoing_ok = len(outgoing_frames) >= BOUNCE_WING_MINIMUM
    if not incoming_ok and not outgoing_ok:
        raise ValueError("bounce witness needs at least two fronts on one temporal wing")
    if not incoming_ok:
        incoming_frames = np.asarray([], float)
    if not outgoing_ok:
        outgoing_frames = np.asarray([], float)
    incoming_pixels = np.asarray([labels[int(frame)] for frame in incoming_frames], float)
    outgoing_pixels = np.asarray([labels[int(frame)] for frame in outgoing_frames], float)

    def sigmas(frames: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                max(
                    sigma_floor_px,
                    float((radii_px or {}).get(int(frame), sigma_floor_px)),
                )
                for frame in frames
            ],
            float,
        )

    incoming_sigmas = sigmas(incoming_frames)
    outgoing_sigmas = sigmas(outgoing_frames)
    one_sided = None
    if not incoming_ok:
        one_sided = "outgoing"
    elif not outgoing_ok:
        one_sided = "incoming"

    def solve(flat_pixels: np.ndarray, forced_side: str | None = one_sided) -> np.ndarray:
        split = incoming_pixels.size
        before = flat_pixels[:split].reshape(incoming_pixels.shape)
        after = flat_pixels[split:].reshape(outgoing_pixels.shape)
        epoch, pixel, _, _ = _reversal_solution(
            event_frame,
            incoming_frames,
            before,
            incoming_sigmas,
            outgoing_frames,
            after,
            outgoing_sigmas,
            one_sided=forced_side,
        )
        camera, camera_frames = _camera_at_epoch(cameras, epoch)
        radial = camera_geometry.radial_at_epoch(camera_distortion, epoch, camera_frames)
        ground = whole.ground_point(camera, pixel, radial)
        return np.r_[epoch, pixel, ground[:2]]

    pixels = np.r_[incoming_pixels.ravel(), outgoing_pixels.ravel()]
    try:
        centre = solve(pixels)
        if one_sided is None and abs(float(centre[0]) - event_frame) > BOUNCE_WING_RADIUS_FRAMES:
            raise np.linalg.LinAlgError("two-sided intersection exceeds the local wing radius")
    except np.linalg.LinAlgError:
        # Parallel/noisy wings do not manufacture an intersection.  Keep the
        # nearer supported wing and make its extrapolation uncertainty explicit.
        incoming_gap = abs(float(incoming_frames[-1]) - event_frame)
        outgoing_gap = (
            abs(float(outgoing_frames[0]) - event_frame) if len(outgoing_frames) else np.inf
        )
        one_sided = "incoming" if incoming_gap <= outgoing_gap else "outgoing"
        centre = solve(pixels, one_sided)

    _, _, incoming_fit, outgoing_fit = _reversal_solution(
        event_frame,
        incoming_frames,
        incoming_pixels,
        incoming_sigmas,
        outgoing_frames,
        outgoing_pixels,
        outgoing_sigmas,
        one_sided=one_sided,
    )
    effective = np.r_[
        (
            np.repeat(incoming_fit["effective_sigmas_px"], 2)
            if incoming_fit is not None
            else np.asarray([], float)
        ),
        (
            np.repeat(outgoing_fit["effective_sigmas_px"], 2)
            if outgoing_fit is not None
            else np.asarray([], float)
        ),
    ]
    jacobian = np.empty((len(centre), len(pixels)), float)
    step_px = 1e-3
    for index in range(len(pixels)):
        shifted = pixels.copy()
        shifted[index] += step_px
        jacobian[:, index] = (solve(shifted, one_sided) - centre) / step_px
    covariance = (jacobian * np.square(effective)[None, :]) @ jacobian.T
    if one_sided is not None:
        interval_sigma = max(0.25, abs(float(frame_interval[1]) - float(frame_interval[0])) / 2.0)
        time_step = 1e-3

        def one_side_at(epoch: float) -> np.ndarray:
            selected = incoming_fit if one_sided == "incoming" else outgoing_fit
            pixel = np.asarray(selected["intercept"], float) + (epoch - event_frame) * np.asarray(
                selected["velocity_px_per_frame"], float
            )
            camera, camera_frames = _camera_at_epoch(cameras, epoch)
            radial = camera_geometry.radial_at_epoch(camera_distortion, epoch, camera_frames)
            ground = whole.ground_point(camera, pixel, radial)
            return np.r_[epoch, pixel, ground[:2]]

        time_jacobian = (one_side_at(event_frame + time_step) - centre) / time_step
        covariance += np.outer(time_jacobian, time_jacobian) * interval_sigma**2
    covariance = 0.5 * (covariance + covariance.T)
    ground_covariance = covariance[3:5, 3:5]
    sigma_m = float(np.sqrt(max(0.0, np.linalg.eigvalsh(ground_covariance)[-1])))
    camera, camera_frames = _camera_at_epoch(cameras, float(centre[0]))
    del camera
    incoming_at_impact = (
        None
        if incoming_fit is None
        else np.asarray(incoming_fit["intercept"], float)
        + (float(centre[0]) - event_frame)
        * np.asarray(incoming_fit["velocity_px_per_frame"], float)
    )
    outgoing_at_impact = (
        None
        if outgoing_fit is None
        else np.asarray(outgoing_fit["intercept"], float)
        + (float(centre[0]) - event_frame)
        * np.asarray(outgoing_fit["velocity_px_per_frame"], float)
    )
    used_frames = sorted({*map(int, incoming_frames), *map(int, outgoing_frames)})
    return {
        "xyz_m": [float(centre[3]), float(centre[4]), whole.BALL_RADIUS_M],
        "uncertainty_sigma_m": sigma_m,
        "ground_covariance_xy_m2": ground_covariance.tolist(),
        "impact_pixel_xy": centre[1:3].tolist(),
        "impact_epoch_witness": {
            "frame": float(centre[0]),
            "sigma_frames": float(np.sqrt(max(0.0, covariance[0, 0]))),
            "supplied_frame": float(event_frame),
            "delta_from_supplied_frames": float(centre[0] - event_frame),
            "source": "wing_intersection" if one_sided is None else "supplied_epoch",
        },
        "incoming_wing": (
            None
            if incoming_fit is None
            else {
                "frames": incoming_frames.tolist(),
                "velocity_px_per_frame": np.asarray(
                    incoming_fit["velocity_px_per_frame"], float
                ).tolist(),
                "residual_rms_px": float(incoming_fit["residual_rms_px"]),
                "pixel_at_impact": incoming_at_impact.tolist(),
            }
        ),
        "outgoing_wing": (
            None
            if outgoing_fit is None
            else {
                "frames": outgoing_frames.tolist(),
                "velocity_px_per_frame": np.asarray(
                    outgoing_fit["velocity_px_per_frame"], float
                ).tolist(),
                "residual_rms_px": float(outgoing_fit["residual_rms_px"]),
                "pixel_at_impact": outgoing_at_impact.tolist(),
            }
        ),
        "construction_mode": (
            "two_sided_reversal" if one_sided is None else f"one_sided_{one_sided}"
        ),
        "construction": (
            "separate incoming/outgoing image-motion wings intersected at a shared latent "
            "impact epoch; label and fit-residual covariance propagated through the "
            "ball-centre ground intersection"
            if one_sided is None
            else f"{one_sided} image-motion wing extrapolated to the supplied impact epoch; "
            "label, fit-residual and timing covariance propagated through the ball-centre "
            "ground intersection"
        ),
        "native_frames": used_frames,
        "camera_frames_at_impact": camera_frames,
        "wing_policy": {
            "maximum_fronts_per_side": BOUNCE_WING_SIZE,
            "minimum_fronts_per_side": BOUNCE_WING_MINIMUM,
            "maximum_radius_frames": BOUNCE_WING_RADIUS_FRAMES,
            "label_sigma_floor_px": BOUNCE_LABEL_SIGMA_FLOOR_PX,
            # Absent unless a caller overrode the floor, so an unchanged witness stays
            # byte for byte.  Present, it says which rows this witness actually believed.
            **(
                {}
                if sigma_floor_px == BOUNCE_LABEL_SIGMA_FLOOR_PX
                else {"applied_sigma_floor_px": float(sigma_floor_px)}
            ),
        },
    }


def event_ground_target(
    event: dict,
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    radii_px: dict[int, float] | None = None,
    *,
    mode: str = "native_ray_average",
    observation_fallback: bool = False,
    fallback_radius_frames: int = BOUNCE_WING_RADIUS_FRAMES,
    fallback_receipt: list[dict] | None = None,
    eligible_frames: np.ndarray | list[float] | tuple[float, ...] | None = None,
    camera_distortion: dict[int, np.ndarray] | None = None,
    sigma_floor_px: float = BOUNCE_LABEL_SIGMA_FLOOR_PX,
    publish_frame_interval: bool = False,
) -> dict:
    """Construct an explicit centre-height bounce witness at its supplied epoch.

    ``sigma_floor_px`` (owner-approved change M, 2026-09-19) is the per-row pixel
    uncertainty floor for the two-wing reversal witness; the default is the human-click
    floor and reproduces every existing witness byte for byte.

    ``publish_frame_interval`` (owner-approved change D1, 2026-09-19: "We should accept
    bounces in range, yes.") carries the supplied bounce's own labelled frame range into
    the witness, so acceptance can judge a modelled impact against the range the event was
    actually observed over instead of only against that range's midpoint.  Default off
    leaves the witness dict unchanged, and an absent range keeps the midpoint rule alone.
    """
    if mode not in BOUNCE_WITNESS_MODES:
        raise ValueError("supported bounce-witness mode required")
    low, high = event["frame_interval"]
    eligible = None if eligible_frames is None else {int(frame) for frame in eligible_frames}

    def usable_frame(frame: int) -> bool:
        return (eligible is None or frame in eligible) and frame in cameras and frame in labels

    frames = sorted(
        {
            frame
            for frame in (
                int(np.floor(low)),
                int(np.ceil(low)),
                int(np.floor(high)),
                int(np.ceil(high)),
            )
            if usable_frame(frame)
        }
    )
    construction_note = "supplied bounce bracket endpoints"
    if not frames and observation_fallback:
        # The supplied bracket has no visible front at any of its endpoint
        # exposures.  Rather than abstain on the whole attempt, take the nearest
        # visible fronts that *bound* the epoch, one on each side, inside a
        # bounded window.  A pair that brackets the impact interpolates to it; a
        # pair on the same side only extrapolates, so it is the last resort.
        # The epoch itself never moves and no picture is invented.
        epoch = float(event["frame"])
        usable = [
            frame
            for frame in labels
            if usable_frame(frame) and abs(frame - epoch) <= fallback_radius_frames
        ]
        below = max((frame for frame in usable if frame < epoch), default=None)
        above = min((frame for frame in usable if frame > epoch), default=None)
        if below is not None and above is not None:
            frames = [below, above]
        else:
            frames = sorted(sorted(usable, key=lambda frame: (abs(frame - epoch), frame))[:2])
        if frames and fallback_receipt is not None:
            fallback_receipt.append(
                {
                    "fallback": "bounce_ground_ray_widened_bracket",
                    "event_frame": epoch,
                    "supplied_interval": [float(low), float(high)],
                    "frames_used": frames,
                }
            )
        construction_note = "nearest visible fronts within the bounded fallback radius"
    if not frames:
        raise ValueError("bounce interval has no visible native ground ray")
    rays = np.asarray(
        [
            whole.ground_point(
                cameras[frame],
                labels[frame],
                camera_geometry.radial_at_epoch(camera_distortion, frame, [frame]),
            )
            for frame in frames
        ]
    )
    legacy_target = np.mean(rays, axis=0)
    legacy_subframe_target = legacy_target.copy()
    legacy_subframe_sigma_m = None
    event_frame = float(event["frame"])
    legacy_lower = int(np.floor(event_frame))
    legacy_upper = int(np.ceil(event_frame))
    legacy_available = all(usable_frame(frame) for frame in (legacy_lower, legacy_upper))
    if not legacy_available and len(frames) == 2 and frames[0] < event_frame < frames[1]:
        legacy_lower, legacy_upper = frames
        legacy_available = True
    if legacy_available:
        fraction = (
            (event_frame - legacy_lower) / (legacy_upper - legacy_lower)
            if legacy_upper != legacy_lower
            else 0.0
        )
        legacy_camera = (1.0 - fraction) * cameras[legacy_lower] + fraction * cameras[legacy_upper]
        legacy_pixel = (1.0 - fraction) * labels[legacy_lower] + fraction * labels[legacy_upper]
        legacy_radial = camera_geometry.radial_at_epoch(
            camera_distortion,
            event_frame,
            [legacy_lower] if legacy_lower == legacy_upper else [legacy_lower, legacy_upper],
        )
        legacy_subframe_target = whole.ground_point(legacy_camera, legacy_pixel, legacy_radial)
        legacy_radius_px = None
        if radii_px is not None and legacy_lower in radii_px and legacy_upper in radii_px:
            legacy_radius_px = (1.0 - fraction) * radii_px[legacy_lower] + fraction * radii_px[
                legacy_upper
            ]
    else:
        legacy_frame = min(frames, key=lambda frame: (abs(frame - event_frame), frame))
        legacy_camera = cameras[legacy_frame]
        legacy_radial = camera_geometry.radial_at_epoch(
            camera_distortion, legacy_frame, [legacy_frame]
        )
        legacy_radius_px = None if radii_px is None else radii_px.get(legacy_frame)
    legacy_metres_per_pixel = transverse_metres_per_pixel(
        legacy_camera, legacy_subframe_target, radial=legacy_radial
    )
    if legacy_radius_px is not None and legacy_metres_per_pixel is not None:
        legacy_subframe_sigma_m = float(legacy_radius_px * legacy_metres_per_pixel)
    target = legacy_target
    sigma_m = None
    graded_radius_m = 0.0
    construction = "mean of native endpoint rays at ball-centre height"
    subframe_available = False
    estimate = None
    if mode == "subframe_graded_circle":
        wing_frames = sorted(
            frame
            for frame in labels
            if usable_frame(frame) and abs(float(frame) - event_frame) <= fallback_radius_frames
        )
        has_fittable_wing = (
            sum(frame < event_frame for frame in wing_frames) >= BOUNCE_WING_MINIMUM
            or sum(frame > event_frame for frame in wing_frames) >= BOUNCE_WING_MINIMUM
        )
        if has_fittable_wing:
            estimate = reversal_impact_witness(
                event_frame,
                tuple(map(float, event["frame_interval"])),
                cameras,
                labels,
                radii_px,
                wing_frames,
                camera_distortion=camera_distortion,
                sigma_floor_px=sigma_floor_px,
            )
            target = np.asarray(estimate["xyz_m"], float)
            sigma_m = float(estimate["uncertainty_sigma_m"])
            subframe_available = estimate["construction_mode"] == "two_sided_reversal"
            construction = estimate["construction"]
        else:
            # One picture on each side cannot resolve either motion wing. Keep
            # the frozen chord only: this preserves old plate/readback behavior
            # but exposes no reversal metadata that could create a new accept.
            target = legacy_subframe_target
            sigma_m = legacy_subframe_sigma_m
            construction = "frozen two-front chord; neither motion wing has two pictures"
            subframe_available = bool(legacy_available)
        graded_radius_m = bounce_graded_radius_m(sigma_m)
        if estimate is not None:
            frames = estimate["native_frames"]
    return dict(
        event_frame=float(event["frame"]),
        # Owner-approved 2026-09-19 (change D1). The epoch above is the range's midpoint;
        # this is the range itself, which is what the label actually established.
        **(
            {"supplied_frame_interval": [float(low), float(high)]} if publish_frame_interval else {}
        ),
        native_frames=frames,
        xyz_m=target.tolist(),
        legacy_native_ray_average_xyz_m=legacy_target.tolist(),
        target_shift_from_legacy_m=float(np.linalg.norm(target[:2] - legacy_target[:2])),
        bracket_radius_m=float(np.max(np.linalg.norm(rays[:, :2] - legacy_target[:2], axis=1))),
        uncertainty_sigma_m=sigma_m,
        graded_circle_radius_m=graded_radius_m,
        witness_mode=mode,
        subframe_available=subframe_available,
        construction=construction,
        bracket_source=construction_note,
        legacy_subframe_witness={
            "xyz_m": legacy_subframe_target.tolist(),
            "uncertainty_sigma_m": legacy_subframe_sigma_m,
            "native_frames": [int(legacy_lower), int(legacy_upper)]
            if legacy_available
            else [int(legacy_frame)],
            "available": bool(legacy_available),
        },
        **(
            {
                key: value
                for key, value in estimate.items()
                if key not in {"xyz_m", "uncertainty_sigma_m", "native_frames", "construction"}
            }
            if estimate is not None
            else {}
        ),
    )


def alternating_player_states(
    pose_csv: Path,
    clip: str,
    contact_events: list[dict],
    labels: dict[int, np.ndarray],
    *,
    image_coordinate_scale: float = 1.0,
    player_names: list[str] | None = None,
    player_statures_m: dict[str, float] | None = None,
    observation_fallback: bool = False,
    fallback_receipt: list[dict] | None = None,
    missing_player_position: str = "off",
    serve_side_association: str = "centre",
) -> list[dict]:
    """Associate contacts to players under the alternating-hitter tennis grammar.

    ``serve_side_association`` chooses the first (unsided) hitter only; see
    ``serve_side``. Every later side still alternates from it.

    Under ``missing_player_position='abstain'``, an already sided interior actor
    may explicitly lack a root after the bounded source fallback is exhausted.
    This abstains from position priors, not contact topology or physical bounds.
    An unresolved first hitter still requires independent association evidence.
    """
    if missing_player_position not in MISSING_PLAYER_POSITION_POLICIES:
        raise ValueError("supported missing-player-position policy required")
    players = []
    first_side = None
    serve_association = (
        {}
        if serve_side_association == "centre"
        else {"serve_side_association": serve_side_association}
    )
    for index, event in enumerate(contact_events):
        frame = int(np.ceil(float(event["frame"])))
        required_side = None
        if first_side is not None:
            other_side = "far" if first_side == "near" else "near"
            required_side = first_side if index % 2 == 0 else other_side
        player_name = None if player_names is None else player_names[index]
        stature_m = None if player_name is None else player_statures_m[player_name]
        try:
            state = single.server_state(
                pose_csv,
                clip,
                frame,
                contact_association_pixel(
                    event,
                    labels,
                    observation_fallback=observation_fallback,
                    fallback_receipt=fallback_receipt,
                ),
                required_side=required_side,
                image_coordinate_scale=image_coordinate_scale,
                player_name=player_name,
                stature_m=stature_m,
                **serve_association,
            )
        except ValueError as error:
            # An absent detector row is not evidence that the player was absent.
            # The court position it would have carried enters the fit as a soft
            # reach term only, so a neighbouring sided row of the same automatic
            # artifact answers the same question with a declared, widened sigma.
            state = None
            if observation_fallback and required_side is None:
                # The first contact also fixes which end served, so retry the
                # ordinary association at the nearest picture that has rows
                # rather than substituting a side this artifact never stated.
                for offset in range(1, player_state_fallback.DEFAULT_SEARCH_RADIUS_FRAMES + 1):
                    for candidate in (frame - offset, frame + offset):
                        try:
                            state = single.server_state(
                                pose_csv,
                                clip,
                                candidate,
                                contact_association_pixel(
                                    event,
                                    labels,
                                    observation_fallback=True,
                                    fallback_receipt=None,
                                ),
                                image_coordinate_scale=image_coordinate_scale,
                                player_name=player_name,
                                stature_m=stature_m,
                                **serve_association,
                            )
                        except ValueError:
                            continue
                        state["substitution"] = {
                            "reason": "automatic player row absent at this contact picture",
                            "source_frame": candidate,
                            "frame_distance": offset,
                            "declared_sigma_m": (
                                player_state_fallback.BOXES_ONLY_SIGMA_M
                                + player_state_fallback.PLAYER_SPEED_M_PER_FRAME * offset
                            ),
                            "sigma_basis": "nearest picture with any sided automatic row",
                        }
                        break
                    if state is not None:
                        break
            if observation_fallback and state is None and required_side is not None:
                state = player_state_fallback.contact_state(
                    pose_csv,
                    clip,
                    frame,
                    required_side,
                    player_name=player_name,
                    stature_m=stature_m,
                    image_coordinate_scale=image_coordinate_scale,
                )
                missing_row = str(error) in {
                    "automatic player pose is absent at first post-contact picture",
                    f"automatic {required_side}-side player pose is absent at contact",
                }
                if state is None and missing_row and missing_player_position == "abstain":
                    state = player_position.absent_contact_state(
                        frame,
                        required_side,
                        player_name=player_name,
                        stature_m=stature_m,
                        image_coordinate_scale=image_coordinate_scale,
                        reason=(
                            f"no {required_side}-side automatic row within "
                            f"{player_state_fallback.DEFAULT_SEARCH_RADIUS_FRAMES} frames "
                            "of this contact; side follows the associated first hitter "
                            "and unchanged alternating contact topology"
                        ),
                        interpretation=(
                            "associated interior actor with explicitly absent court root; "
                            "root-dependent soft evidence abstains, contact remains"
                        ),
                    )
                    if fallback_receipt is not None:
                        fallback_receipt.append(
                            dict(
                                fallback="associated_contact_position_absent",
                                contact_index=index,
                                contact_frame=float(event["frame"]),
                                required_side=required_side,
                                evidence=state["player_position_evidence"],
                            )
                        )
            if state is None:
                raise
            if fallback_receipt is not None and not player_position.declared_absent(state):
                fallback_receipt.append(
                    {
                        "fallback": "contact_player_state_from_nearest_sided_row",
                        "contact_frame": float(event["frame"]),
                        "required_side": required_side,
                        "refused_with": f"{type(error).__name__}: {error}",
                        **state["substitution"],
                    }
                )
        first_side = state["side"] if first_side is None else first_side
        players.append(state)
    return players


def player_ledger_positions(path: Path, clip: str) -> dict[tuple[int, str], dict[str, float]]:
    """Index one ``tennis_player_state_v1`` sidecar by native frame and side.

    Blank court coordinates are the ledger's declared abstention and are skipped,
    so a frame the ledger will not stand behind keeps the sided-box position.
    """
    import csv as _csv

    rows: dict[tuple[int, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in _csv.DictReader(handle):
            if row.get("clip") != clip or row.get("side") not in {"near", "far"}:
                continue
            if not row.get("court_x") or not row.get("court_y"):
                continue
            rows[(int(float(row["frame"])), row["side"])] = {
                "court_x": float(row["court_x"]),
                "court_y": float(row["court_y"]),
                "court_sigma_m": float(row["court_sigma_m"]) if row.get("court_sigma_m") else None,
                "position_source": row.get("position_source"),
            }
    return rows


def apply_player_ledger(
    players: list[dict],
    contact_events: list[dict],
    ledger: dict[tuple[int, str], dict[str, float]],
) -> list[dict]:
    """Take each contact's court position from the ledger where it speaks.

    This substitutes the *source* of one already-modeled quantity, the striking
    player's court centre.  It does not change how the reach term is scored, and
    the sided-box position stays wherever the ledger abstains.
    """
    receipt = {"substituted": [], "abstained": []}
    for event, state in zip(contact_events, players, strict=True):
        frame = int(np.ceil(float(event["frame"])))
        if not player_position.root_available(state):
            receipt["abstained"].append(
                {
                    "frame": frame,
                    "side": state["side"],
                    "reason": "player position explicitly absent",
                }
            )
            continue
        row = ledger.get((frame, state["side"]))
        if row is None:
            receipt["abstained"].append({"frame": frame, "side": state["side"]})
            continue
        state["court_centre_xy_m_sided_box"] = list(state["court_centre_xy_m"])
        state["court_centre_xy_m"] = np.asarray([row["court_x"], row["court_y"]], float)
        state["court_position_source"] = "tennis_player_state_v1"
        state["court_position_sigma_m"] = row["court_sigma_m"]
        receipt["substituted"].append(
            {
                "frame": frame,
                "side": state["side"],
                "moved_m": float(
                    np.linalg.norm(
                        np.asarray(state["court_centre_xy_m"], float)
                        - np.asarray(state["court_centre_xy_m_sided_box"], float)
                    )
                ),
            }
        )
    return receipt


def _normalised_name(value: str) -> str:
    import unicodedata

    folded = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in folded.lower() if ch.isalpha())


def match_roster_name(labeled: str, roster: list[str]) -> str:
    """Map a labeled hitter name ("Jasmine Paolini", "Paolini") onto one roster name.

    A roster entry matches when its normalised form is contained in the labeled
    name or vice versa.  Zero or more than one match fails closed: a striker
    identity that cannot be tied to the roster must never be guessed.
    """
    wanted = _normalised_name(labeled)
    hits = []
    for name in dict.fromkeys(roster):
        key = _normalised_name(name)
        if key and wanted and (key in wanted or wanted in key):
            hits.append(name)
    if len(hits) != 1:
        raise ValueError(
            f"labeled hitter {labeled!r} matches {hits or 'no'} roster name(s) in {roster}"
        )
    return hits[0]


def labeled_contact_players(
    labels: dict, contact_events: list[dict], expected_cycle: list[str]
) -> list[str]:
    """Join additive player identities without changing any event epoch.

    The label documents write the striker under ``hitter`` (the owner's and the
    agents' schema) or, in a few older files, ``player``.  Where the labels name
    the striker, the labeled order is authoritative and replaces the alternating
    configured cycle (two consecutive contacts by one player are legal tennis);
    every name must map onto exactly one roster entry or the attempt fails
    closed.  Only when no contact is labeled does the configured cycle apply.
    """
    labeled = [
        row for row in labels.get("events", {}).get("records", []) if row["event_type"] == "contact"
    ]
    names = []
    for event in contact_events:
        matches = [row for row in labeled if float(row["frame"]) == float(event["frame"])]
        if len(matches) != 1:
            raise ValueError("athlete prior requires one labeled event per connected contact")
        row = matches[0]
        names.append(row.get("player") if row.get("player") is not None else row.get("hitter"))
    if any(name is not None for name in names):
        if any(name is None for name in names):
            raise ValueError("labeled hitter identities must cover every connected contact or none")
        return [match_roster_name(str(name), list(expected_cycle)) for name in names]
    if len(expected_cycle) not in {1, 2}:
        raise ValueError("one/two-player explicit topology cycle required")
    return [expected_cycle[index % len(expected_cycle)] for index in range(len(names))]


def parse_player_statures(values: list[str]) -> dict[str, float]:
    """Parse explicit ``PLAYER=METRES`` statures; malformed values fail closed."""
    statures: dict[str, float] = {}
    for value in values:
        try:
            name, raw = value.rsplit("=", 1)
            stature = float(raw)
        except (ValueError, TypeError) as error:
            raise ValueError("player stature must be PLAYER=METRES") from error
        if not name or name in statures or not np.isfinite(stature) or stature <= 0:
            raise ValueError("unique player names and positive finite statures required")
        statures[name] = stature
    return statures


def resolve_athlete_evidence(
    athlete_prior_mode: str,
    athlete_evidence: str,
    player_order: list[str],
    player_statures: list[str],
    labels: dict,
    contact_events: list[dict],
) -> tuple[list[str] | None, dict[str, float], dict]:
    """Decide once whether the stature arm has evidence, is explicitly without, or is malformed.

    Supplied evidence resolves exactly as it always did.  Under the ``optional``
    policy an empty roster is a declared absence, not a failure, and every
    stature-scaled term abstains.  Statures without a roster order, or a roster
    without statures for its contact players, remain malformed under both policies.
    """
    if athlete_evidence not in ATHLETE_EVIDENCE_POLICIES:
        raise ValueError("supported athlete-evidence policy required")
    statures = parse_player_statures(player_statures)
    receipt = {"policy": athlete_evidence, "status": "not_applicable", "player_order": []}
    if athlete_prior_mode != "stature_pose_soft":
        return None, statures, receipt
    if player_order:
        if len(set(player_order)) != len(player_order):
            raise ValueError("athlete arm requires a unique explicit player-order cycle")
        player_names = labeled_contact_players(labels, contact_events, list(player_order))
        # Every contact player needs a stature.  A configured roster entry
        # for the opponent who never strikes the ball in this attempt --
        # a fault, or any one-sided attempt -- is unused evidence, not a
        # missing input, and it must not refuse the attempt.
        if not set(player_names) <= set(statures):
            raise ValueError("athlete arm requires exact statures for all labeled contact players")
        return (
            player_names,
            statures,
            receipt | {"status": "supplied", "player_order": list(player_order)},
        )
    if athlete_evidence != "optional":
        raise ValueError("athlete arm requires a unique explicit player-order cycle")
    if statures:
        raise ValueError("player statures without a player order are malformed athlete evidence")
    return None, statures, receipt | {"status": "unavailable"}


def missing_player_position_configuration(
    policy: str, players: list[dict], *, right_contact_player: dict | None = None
) -> dict:
    """Declare the policy and, only under it, the absences it actually produced.

    With the policy off the configuration keeps its original shape, so existing
    reports and replays are byte-identical.
    """
    players = player_position.contact_states(players, right_contact_player)
    if policy not in MISSING_PLAYER_POSITION_POLICIES:
        raise ValueError("supported missing-player-position policy required")
    if policy == "off":
        if any(player_position.declared_absent(row) for row in players):
            raise ValueError("an absent contact position requires the explicit shared policy")
        return {}
    return {
        "missing_player_position": policy,
        "player_position_receipt": player_position.state_receipt(players),
    }


def athlete_evidence_configuration(policy: str, receipt: dict) -> dict:
    """The one report configuration shape every producer writes and the replay reads."""
    if policy not in ATHLETE_EVIDENCE_POLICIES or receipt.get("policy") != policy:
        raise ValueError("athlete-evidence receipt must carry the declared policy")
    return {
        "athlete_evidence": policy,
        "athlete_evidence_status": receipt["status"],
        "athlete_evidence_receipt": receipt,
    }


def mark_athlete_evidence(players: list[dict], receipt: dict) -> list[dict]:
    """Declare the absence explicitly on every player state the fit will read."""
    if receipt.get("status") != "unavailable":
        return players
    for state in players:
        if state.get("stature_m") is not None or state.get("player") is not None:
            raise ValueError("unavailable athlete evidence cannot carry a name or stature")
        state["athlete_evidence"] = athlete_priors.unavailable_evidence()
    return players


def fixed_depth_beam(
    depth_hypotheses_m: np.ndarray, depth_bounds_m: tuple[float, float]
) -> tuple[np.ndarray, list[dict]]:
    """Exclude depth branches that cannot pass the fixed serve-region gate.

    The optimizer pins the first contact's court-Y coordinate to the branch
    depth. Therefore a branch outside ``depth_bounds_m`` cannot move into the
    accepted region at either optimization budget. Record every excluded
    branch rather than spending a coarse and refined solve on a known failure.
    """
    depths = np.asarray(depth_hypotheses_m, float)
    low, high = map(float, depth_bounds_m)
    if (
        depths.ndim != 1
        or not len(depths)
        or not np.isfinite(depths).all()
        or len(set(depths.tolist())) != len(depths)
        or not np.isfinite([low, high]).all()
        or low > high
    ):
        raise ValueError("finite unique depths and ordered finite bounds required")
    included = (low <= depths) & (depths <= high)
    if not included.any():
        raise ValueError("fixed serve-region beam excluded every depth hypothesis")
    excluded = [
        {
            "depth_hypothesis_m": float(depth),
            "reason": "fixed_first_contact_y_outside_serve_region",
            "serve_depth_bounds_m": [low, high],
            "refit_cannot_change_fixed_depth": True,
        }
        for depth in depths[~included]
    ]
    return depths[included], excluded


ANCHOR_CONTACT_HEIGHT_M = 1.15


def native_seed_source(scene, heldout, bounces, axes, physical_events, ground_targets=None):
    """Optional source preparation must not invalidate ordinary seed families.

    Only in-memory observation merging/qualification is isolated here. Runtime,
    deadline, I/O and provenance failures are not caught.
    """
    from cv.experiments.connected_shooting import (
        observation_net_seed,
    )

    try:
        native = scene
        if observation_net_seed.net_contract(scene, physical_events) is not None:
            native = observation_net_seed.native_source(scene, heldout)
        return native, observation_net_seed.qualify(
            native, physical_events=physical_events, ground_targets=ground_targets
        )
    except (ValueError, KeyError, TypeError) as error:
        return scene, dict(
            status="unavailable",
            reason=f"optional native source metadata: {type(error).__name__}: {error}",
            method=observation_net_seed.METHOD,
        )


FINAL_BOUNCE_ANCHOR_MAX_DELTA_FRAMES = 1.0
FINAL_BOUNCE_ANCHOR_SIGMA_FRAMES = 1.0


def final_bounce_anchor_plan(scene, bounces, targets) -> dict:
    """Picture-agreed epochs for the bounces of a final flight (``--final-bounce-anchor``).

    A bounce qualifies when its own two-sided wing witness (incoming and outgoing
    image paths meeting, ``impact_epoch_witness``) is resolved and lies within one
    native frame of the supplied bounce.  In a labelled-event arm the supplied
    bounce is the label; in an automatic arm it is the automatic bounce, so the
    rule applies only where the automatic bounce agrees with the pictures.
    Qualifying bounces get a soft timing prior at the witness epoch; when the
    first one qualifies the search also adds one restart launched to reach the
    ground at that epoch and at its ground-ray point.  Every gate is unchanged.
    """
    from cv.experiments.connected_shooting import model as _model

    if _model.original_contact_boundary(scene):
        return {"status": "not_applicable", "reason": "last flight ends at an original contact"}
    last = len(scene.pixels) - 1
    if not len(bounces[last]):
        return {"status": "not_applicable", "reason": "final flight has no supplied bounce"}
    rows, priors = [], []
    for ordinal, (frame, target) in enumerate(zip(bounces[last], targets[last], strict=True)):
        witness = target.get("impact_epoch_witness") or {}
        epoch = witness.get("frame")
        delta = witness.get("delta_from_supplied_frames")
        reason = None
        if witness.get("source") != "wing_intersection" or target.get("construction_mode") not in (
            None,
            "two_sided_reversal",
        ):
            reason = "witness_not_two_sided"
        elif epoch is None or delta is None or not np.isfinite([epoch, delta]).all():
            reason = "witness_epoch_unresolved"
        elif abs(float(delta)) > FINAL_BOUNCE_ANCHOR_MAX_DELTA_FRAMES:
            reason = "witness_disagrees_with_supplied_bounce"
        elif not scene.contact_frames[last] < float(epoch) <= scene.contact_frames[last + 1]:
            reason = "witness_epoch_outside_flight"
        rows.append(
            {
                "ordinal": ordinal,
                "supplied_frame": float(frame),
                "witness_frame": None if epoch is None else float(epoch),
                "delta_from_supplied_frames": None if delta is None else float(delta),
                "qualified": reason is None,
                **({} if reason is None else {"reason": reason}),
            }
        )
        if reason is None:
            priors.append(
                dict(
                    flight_index=last,
                    ordinal=ordinal,
                    frame=float(epoch),
                    sigma_frames=FINAL_BOUNCE_ANCHOR_SIGMA_FRAMES,
                )
            )
    seed = None
    if rows[0]["qualified"]:
        seed = {
            "frame": rows[0]["witness_frame"],
            "xy_m": [float(v) for v in np.asarray(targets[last][0]["xyz_m"], float)[:2]],
        }
    return {
        "status": "applied" if priors else "abstained",
        "flight_index": last,
        "max_delta_frames": FINAL_BOUNCE_ANCHOR_MAX_DELTA_FRAMES,
        "bounces": rows,
        "epoch_priors": priors,
        "seed": seed,
    }


def native_seed_families(restarts: bool, availability: dict) -> tuple:
    """Fixed input-qualified queue; ordinary families remain, no fit/gate access.

    This shares the caller's cumulative budget. Additional work can reduce later
    family/depth effort; configured families are not a promise they all finish.
    """
    families = [("pinhole", None, False)]
    if availability.get("status") == "supported":
        families.append(("native_net", None, True))
    if restarts:
        families.extend((("anchor", False, False), ("anchor_flat_spin", True, False)))
    return tuple(families)


def structure_aware_connected_seed(
    scene: model.Scene,
    initial: np.ndarray,
    players: list[dict],
    axes: np.ndarray,
    bounces: tuple[np.ndarray, ...],
    first_contact_y_m: float,
    *,
    max_nfev: int = 40,
    deadline_check=None,
    exposure_duration: float | None = 0.25,
    native_net_initialization: dict | None = None,
    event_ground_targets: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Repair a long-point seed using training pixels and alternating player states.

    Independent pinhole fits estimate useful outgoing velocities but their local
    contact origins are not connected.  Starting at the depth branch, this
    initializer walks the point once.  Each six-parameter velocity/spin solve
    sees only its flight's training pixels and softly aims the next contact at
    the automatically localized next hitter.  It is an initializer, not an
    acceptance condition; the unchanged whole-point solve and gates follow it.

    ``event_ground_targets`` optionally supplies, per flight, the measured court
    position of that flight's supplied first bounce -- the same seed witness the
    pinhole initializer already consumes.  With it, a flight whose source events
    declare exactly one ground also starts one extra local solve from a
    gravity-only arc that reaches that position at the supplied bounce epoch.
    The extra start is a guess only: the per-flight residual, its selection by
    the walk's own cost, and every existing guess are unchanged, so with the
    argument omitted the walk is byte-for-byte the promoted one.
    """
    n = len(scene.pixels)
    if n < 2 or len(players) != n:
        raise ValueError("one alternating automatic player state per connected flight required")
    if event_ground_targets is not None and len(event_ground_targets) != n:
        raise ValueError("one optional event ground target per connected flight required")
    prediction_options = (
        {"termination_kind": model.ORIGINAL_CONTACT_TERMINATION_KIND}
        if model.original_contact_boundary(scene)
        else {}
    )
    parameters = np.asarray(initial, float).copy()
    parameters[1] = first_contact_y_m
    end_xyz = None
    receipt = []
    native_response = None
    native_receipt = None
    axis_groups = np.split(
        np.asarray(axes, float), np.cumsum([len(row) for row in scene.pixels])[:-1]
    )
    for index in range(n):
        subscene = replace(
            scene,
            contact_frames=scene.contact_frames[index : index + 2],
            terminal_net_tail=scene.terminal_net_tail if index == n - 1 else None,
            observed_horizon_tail=scene.observed_horizon_tail if index == n - 1 else None,
            observation_frames=(scene.observation_frames[index],),
            cameras=(scene.cameras[index],),
            camera_distortion=(
                None if scene.camera_distortion is None else (scene.camera_distortion[index],)
            ),
            pixels=(scene.pixels[index],),
            spin_parameters=scene.spin_parameters[index : index + 1],
            net_hit_frames=(
                None if scene.net_hit_frames is None else (scene.net_hit_frames[index],)
            ),
        )
        start_xyz = parameters[:3].copy() if end_xyz is None else end_xyz.copy()
        if index == n - 1 and native_net_initialization is not None:
            from cv.experiments.connected_shooting import observation_net_seed
            from cv.experiments.connected_shooting.labeled_passive_tape import using_response

            if (
                observation_net_seed.net_contract(scene, native_net_initialization["events"])
                is None
            ):
                raise ValueError("native terminal initialization requires original net contract")
            if deadline_check is not None:
                deadline_check()
            parameters, native_response, native_receipt = observation_net_seed.initialize(
                native_net_initialization["scene"],
                parameters,
                physical_events=native_net_initialization["events"],
                ground_targets=native_net_initialization.get("ground_targets"),
                speed_scale_mps=native_net_initialization["speed_scale_mps"],
                **(
                    {"deadline_check": deadline_check}
                    if native_net_initialization.get("candidate_family")
                    and deadline_check is not None
                    else {}
                ),
            )
            native_receipt = native_seed_check_pixels.annotate(
                native_receipt, native_net_initialization.get("reserved_check_frames")
            )
            if deadline_check is not None:
                deadline_check()
            with using_response(native_response):
                final = model.chain(scene, parameters)[-1]
                full_image = exposure.prediction(
                    scene, parameters, axes, exposure_duration, **prediction_options
                )
            last_count = len(subscene.pixels[0])
            image = full_image[-last_count:] - subscene.pixels[0]
            receipt.append(
                dict(
                    flight_index=index,
                    method="original_native_net_response_after_connected_prefix",
                    training_pixel_rms=float(np.sqrt(np.mean(image**2))),
                    start_xyz_m=start_xyz.tolist(),
                    end_xyz_m=final["end_xyz"].tolist(),
                    automatic_next_player_seed_xyz_m=None,
                    full_original_tail_replayed=True,
                    original_prefix_parameters_unchanged=True,
                    initialization=native_receipt,
                )
            )
            break
        target_xyz = (
            None
            if index == n - 1 or player_position.root_xy(players[index + 1]) is None
            else np.r_[players[index + 1]["court_centre_xy_m"], 1.2]
        )
        velocity = parameters[3 + 3 * index : 6 + 3 * index]
        spin = parameters[3 + 3 * n + 3 * index : 6 + 3 * n + 3 * index]
        guesses = [np.r_[velocity, spin]]
        if target_xyz is not None:
            duration = (scene.contact_frames[index + 1] - scene.contact_frames[index]) / scene.fps
            ballistic = (target_xyz - start_xyz) / duration
            ballistic[2] += 4.905 * duration
            guesses.extend(
                np.r_[ballistic, spin_guess] for spin_guess in ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            )

        expected_bounces = np.asarray(bounces[index], float)
        ground_guess = None
        if event_ground_targets is not None and len(expected_bounces) == 1:
            target_xy = event_ground_targets[index]
            if target_xy is not None:
                target_xy = np.asarray(target_xy, float)[:2]
                if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
                    raise ValueError("finite two-dimensional event ground target required")
                # Gravity-only arc from the connected start to the measured
                # impact at the supplied bounce epoch; drag and spin are left
                # to the same bounded local solve every other guess receives.
                seconds = (float(expected_bounces[0]) - scene.contact_frames[index]) / scene.fps
                ground_guess = (np.r_[target_xy, model.R_BALL] - start_xyz) / seconds
                ground_guess[2] += 4.905 * seconds
                guesses.append(np.r_[ground_guess, spin])
        residual_length = (
            2 * len(subscene.pixels[0])
            + (0 if target_xyz is None else 3)
            + len(expected_bounces)
            + 1
        )

        def residual(candidate):
            if deadline_check is not None:
                deadline_check()
            try:
                flight = model.chain(
                    subscene,
                    np.r_[start_xyz, candidate[:3], candidate[3:], parameters[-2:]],
                )[0]
                image = (
                    exposure.prediction(
                        subscene,
                        np.r_[start_xyz, candidate[:3], candidate[3:], parameters[-2:]],
                        axis_groups[index],
                        exposure_duration,
                        **prediction_options,
                    )
                    - subscene.pixels[0]
                ).ravel() / 8.0
                endpoint = (
                    np.empty(0) if target_xyz is None else (flight["end_xyz"] - target_xyz) / 0.10
                )
                observed_bounces = flight["bounces"]
                timing = []
                for bounce_index, frame in enumerate(expected_bounces):
                    timing.append(
                        20.0
                        if bounce_index >= len(observed_bounces)
                        else (observed_bounces[bounce_index]["frame"] - frame) / 0.25
                    )
                extras = observed_bounces[len(expected_bounces) :]
                timing.append(
                    0.0 if not extras else (flight["end_frame"] - extras[0]["frame"]) / 0.25
                )
                return np.r_[image, endpoint, timing]
            except (ValueError, FloatingPointError, OverflowError):
                return np.full(residual_length, 1e5)

        lower = np.r_[[-75.0] * 3, [-6.0] * 3]
        upper = np.r_[[75.0] * 3, [6.0] * 3]
        solved = []
        for guess_index, guess in enumerate(guesses):
            candidate = least_squares(
                residual,
                np.clip(guess, lower + 0.1, upper - 0.1),
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
            values = residual(candidate.x)
            cost = float(np.sum(np.sqrt(1.0 + (values / 2.0) ** 2) - 1.0))
            if np.isfinite(cost):
                candidate.structured_guess_index = guess_index
                solved.append((cost, candidate))
        if not solved:
            raise ValueError(f"flight {index}: no finite structure-aware seed")
        _, selected = min(solved, key=lambda row: row[0])

        def walked(values):
            return model.chain(
                subscene,
                np.r_[start_xyz, values[:3], values[3:], parameters[-2:]],
            )[0]

        ground_epoch_receipt = None
        try:
            flight = walked(selected.x)
        except (ValueError, FloatingPointError, OverflowError) as unsupported:
            # The walk hands this flight the previous flight's arrival, and the
            # per-flight residual reports an unsupported launch as its sentinel
            # instead of raising, so an unsupported selection only surfaces here.
            # Every flight with a next contact also owns an ascending start from
            # its next-contact ballistic guess; a flight without that anchor
            # keeps only the disconnected pinhole velocity and a court-level
            # start leaves it no supported launch.  Rebuild one start from this
            # flight's declared ground epoch and retry the same bounded local
            # solve.  Reached only where the walk already raised outright.
            recovery = None
            if target_xyz is None and len(expected_bounces):
                seconds = (float(expected_bounces[0]) - float(subscene.contact_frames[0])) / float(
                    scene.fps
                )
                target_xy = (
                    None
                    if event_ground_targets is None or event_ground_targets[index] is None
                    else np.asarray(event_ground_targets[index], float)[:2]
                )
                recovery = initialization.ground_epoch_launch(
                    start_xyz, seconds, velocity, target_xy=target_xy
                )
            if recovery is None:
                raise
            guesses.append(np.r_[recovery, spin])
            candidate = least_squares(
                residual,
                np.r_[recovery, spin],
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
            flight = walked(candidate.x)
            candidate.structured_guess_index = len(guesses) - 1
            selected = candidate
            ground_epoch_receipt = {
                "status": "retried",
                "declared_ground_frame": float(expected_bounces[0]),
                "seconds_to_declared_ground": seconds,
                "uses_supplied_ground_ray": target_xy is not None,
                "launch_velocity_mps": np.asarray(recovery, float).tolist(),
                "unsupported_reason": str(unsupported),
            }
        parameters[3 + 3 * index : 6 + 3 * index] = selected.x[:3]
        parameters[3 + 3 * n + 3 * index : 6 + 3 * n + 3 * index] = selected.x[3:]
        end_xyz = np.asarray(flight["end_xyz"], float)
        image = (
            exposure.prediction(
                subscene,
                np.r_[start_xyz, selected.x[:3], selected.x[3:], parameters[-2:]],
                axis_groups[index],
                exposure_duration,
                **prediction_options,
            )
            - subscene.pixels[0]
        ).ravel()
        receipt.append(
            {
                "flight_index": index,
                "optimizer_success": bool(selected.success),
                "objective_evaluations": int(selected.nfev),
                "training_pixel_rms": float(np.sqrt(np.mean(image**2))),
                "start_xyz_m": start_xyz.tolist(),
                "end_xyz_m": end_xyz.tolist(),
                "automatic_next_player_seed_xyz_m": (
                    None if target_xyz is None else target_xyz.tolist()
                ),
                **(
                    {}
                    if ground_epoch_receipt is None
                    else {"ground_epoch_launch": ground_epoch_receipt}
                ),
                # Stated only when the caller offered event ground targets at
                # all, so the promoted walk keeps its receipt byte for byte.
                **(
                    {}
                    if event_ground_targets is None
                    else {
                        "event_ground_guess": (
                            None
                            if ground_guess is None
                            else {
                                "guess_index": len(guesses) - 1,
                                "bounce_frame": float(expected_bounces[0]),
                                "target_xy_m": np.asarray(event_ground_targets[index], float)[
                                    :2
                                ].tolist(),
                                "gravity_only_velocity_mps": ground_guess.tolist(),
                            }
                        ),
                        "selected_guess_index": int(selected.structured_guess_index),
                        "selected_event_ground_guess": bool(
                            ground_guess is not None
                            and selected.structured_guess_index == len(guesses) - 1
                        ),
                    }
                ),
            }
        )
    if native_response is None:
        predicted = exposure.prediction(
            scene, parameters, axes, exposure_duration, **prediction_options
        )
    else:
        with using_response(native_response):
            predicted = exposure.prediction(
                scene, parameters, axes, exposure_duration, **prediction_options
            )
    global_residual = (predicted - np.concatenate(scene.pixels)).ravel()
    return parameters, {
        "method": "sequential_training_image_plus_automatic_alternating_player_seed",
        **(
            {"native_net_response_initialization": native_receipt}
            if native_receipt is not None
            else {}
        ),
        "trigger": "five_or_more_connected_flights",
        "nominal_next_contact_height_m": 1.2,
        "training_image_scale_px": 8.0,
        "automatic_player_endpoint_scale_m": 0.10,
        "bounce_timing_seed_scale_frames": 0.25,
        "uses_withheld_pixels": (
            False
            if native_receipt is None
            else native_receipt["check_pixel_usage"]["uses_reserved_check_pixels"]
        ),
        "uses_bounce_ray_targets": event_ground_targets is not None,
        **(
            {}
            if event_ground_targets is None
            else {
                "event_ground_targets_role": (
                    "one extra gravity-only start per flight with exactly one supplied "
                    "bounce and a measured impact position; residual, guess selection "
                    "and every existing start unchanged"
                ),
                "event_ground_guess_flights": [
                    row["flight_index"] for row in receipt if row.get("event_ground_guess")
                ],
                "event_ground_guess_selected_flights": [
                    row["flight_index"] for row in receipt if row.get("selected_event_ground_guess")
                ],
            }
        ),
        "flights": receipt,
        "connected_training_centre_rms_px": float(np.sqrt(np.mean(global_residual**2))),
    }


def contact_association_pixel(
    event: dict,
    labels: dict[int, np.ndarray],
    *,
    observation_fallback: bool = False,
    fallback_radius_frames: int = 8,
    fallback_receipt: list[dict] | None = None,
) -> np.ndarray:
    """Use the same-frame front or the nearest visible bracket neighbor."""
    frame = int(np.ceil(float(event["frame"])))
    if frame in labels:
        return labels[frame]
    low, high = event.get("frame_interval", [event["frame"], event["frame"]])
    choices = [
        candidate
        for candidate in labels
        if int(np.floor(low)) - 1 <= candidate <= int(np.ceil(high)) + 1
    ]
    if not choices and observation_fallback:
        # Player association only needs a pixel near the contact to pick the
        # struck side.  Widen the neighbourhood instead of abstaining on the
        # whole attempt; the event epoch is untouched.
        choices = [
            candidate
            for candidate in labels
            if abs(candidate - float(event["frame"])) <= fallback_radius_frames
        ]
        if not choices and labels:
            # Every front near this contact abstained.  The pixel is only used
            # to pick which player struck the ball and to associate the server's
            # track, so the nearest front anywhere in the attempt still answers
            # that question far better than losing the attempt does.  Its frame
            # distance is recorded so a bad association is visible.
            choices = list(labels)
        if choices and fallback_receipt is not None:
            fallback_receipt.append(
                {
                    "fallback": "contact_association_widened_window",
                    "event_frame": float(event["frame"]),
                    "radius_frames": fallback_radius_frames,
                    "nearest_front_frame_distance": min(
                        abs(candidate - float(event["frame"])) for candidate in choices
                    ),
                }
            )
    if not choices:
        raise ValueError("contact bracket has no nearby visible ball front for player association")
    nearest = min(choices, key=lambda candidate: (abs(candidate - event["frame"]), candidate))
    return labels[nearest]


def launch_and_right_players(
    pose_csv, clip, events, labels, *, right_boundary_kind="supplied_end", **kwargs
):
    """Keep the incoming-only right actor separate from outgoing contact states.

    Missing final pose is absent soft evidence. All launch associations retain
    their existing requirements and the right contact's venue gate still applies.
    """
    if right_boundary_kind != "original_contact":
        return alternating_player_states(pose_csv, clip, events, labels, **kwargs), None
    try:
        states = alternating_player_states(pose_csv, clip, events, labels, **kwargs)
        return states[:-1], states[-1]
    except ValueError as error:
        launch_kwargs = dict(kwargs)
        names = launch_kwargs.get("player_names")
        if names is not None:
            launch_kwargs["player_names"] = names[:-1]
        states = alternating_player_states(pose_csv, clip, events[:-1], labels, **launch_kwargs)
        receipt = kwargs.get("fallback_receipt")
        if receipt is not None:
            receipt.append(
                {
                    "fallback": "right_contact_position_unavailable",
                    "reason": str(error),
                    "original_contact_preserved": True,
                    "invented_player_position": False,
                }
            )
        return states, None


def closing_anchor_abstention(right_contact_player: dict | None) -> str:
    """Say why an original-contact prefix ends untied, inventing no court root.

    Either no actor was associated with the closing contact at all, or the one
    that was carries the explicit absent-position declaration; both are absent
    soft evidence and neither is allowed to become a tie.
    """
    if right_contact_player is None:
        return "no automatic player state at the closing contact"
    if player_position.declared_absent(right_contact_player):
        return right_contact_player["player_position_evidence"]["reason"]
    raise ValueError("a closing actor with a present court root carries its tie")


def candidate_evidence(
    measurement: dict,
    players: list[dict],
    bounce_targets: list[list[dict]],
    depth_bounds: tuple[float, float],
    events: list[dict],
    *,
    athlete_prior_mode: str = "global_hard_caps",
    serve_reach: serve_reach_cylinder.ServeReachCylinder | None = None,
    serve_ending_kind: str | None = None,
    enforce_serve_ending_consistency: bool = False,
    same_player_serve_regularization: dict | None = None,
) -> dict:
    is_rally = not contact_role.serve_priors_applicable(players)
    if is_rally and (
        serve_reach is not None
        or enforce_serve_ending_consistency
        or same_player_serve_regularization
    ):
        raise ValueError("rally evidence cannot consume serve-only constraints")
    if athlete_prior_mode not in ATHLETE_PRIOR_MODES:
        raise ValueError("supported athlete-prior mode required")
    flat_targets = bool(bounce_targets and isinstance(bounce_targets[0], dict))
    if flat_targets:
        bounce_targets = [[row] for row in bounce_targets]
    contacts = [np.asarray(row, float) for row in measurement["contact_xyz"]]
    if len(contacts) != len(players):
        raise ValueError("one coarse player state per connected contact required")
    # An absent court root abstains from this player's reach diagnostic and its
    # soft geometry penalty; it never relaxes a bound or reports a fake distance.
    player_distances = [
        None
        if player_position.root_xy(player) is None
        else float(np.linalg.norm(contact[:2] - player["court_centre_xy_m"]))
        for contact, player in zip(contacts, players, strict=True)
    ]
    rooted_distances = [value for value in player_distances if value is not None]
    modeled_bounces: list[list[np.ndarray | None]] = []
    bounce_errors: list[list[float | None]] = []
    bounce_gate_errors: list[list[float | None]] = []
    completion_xyz = measurement["physical"].get("terminal_completion", {}).get("end_xyz")
    for index, (flight, targets) in enumerate(
        zip(measurement["dense_flights"], bounce_targets, strict=True)
    ):
        actual = [np.asarray(row["x"], float) for row in flight["bounces"][: len(targets)]]
        if index == len(measurement["dense_flights"]) - 1 and len(actual) < len(targets):
            actual.append(None if completion_xyz is None else np.asarray(completion_xyz, float))
        actual += [None] * (len(targets) - len(actual))
        modeled_bounces.append(actual)
        bounce_errors.append(
            [
                None
                if row is None
                else float(np.linalg.norm(row[:2] - np.asarray(target["xyz_m"])[:2]))
                for row, target in zip(actual, targets, strict=True)
            ]
        )
        bounce_gate_errors.append(
            [
                None
                if value is None
                else max(0.0, value - float(target.get("graded_circle_radius_m", 0.0)))
                for value, target in zip(bounce_errors[-1], targets, strict=True)
            ]
        )
    direction = whole.directional_support(
        measurement["native_projection"], events, whole.SearchConfig()
    )
    if len(rooted_distances) != len(player_distances) and athlete_prior_mode == "global_hard_caps":
        # The hard-cap mode has no abstention contract: an absent root must hold,
        # never pass a reach cap it was never measured against.
        raise ValueError("global hard athlete caps require a present court root at every contact")
    legacy_checks = dict(
        **({} if is_rally else {"serve_height_plausible": bool(2.4 <= contacts[0][2] <= 3.1)}),
        all_player_reaches_plausible=bool(all(value <= 1.75 for value in rooted_distances)),
    )
    athlete = None
    if athlete_prior_mode == "stature_pose_soft":
        athlete = athlete_priors.evaluate(np.asarray(contacts), players)
    envelope_evidence = [event_constraints.contact_xy_envelope_evidence(row) for row in contacts]
    inside_envelope = bool(all(row["inside_venue_envelope"] for row in envelope_evidence))
    reach = None if serve_reach is None else serve_reach.evaluate(contacts[0])
    ending_consistency = None
    if enforce_serve_ending_consistency:
        ending_consistency = serve_ending_consistency(
            serve_ending_kind,
            players[0]["side"],
            None if not modeled_bounces or not modeled_bounces[0] else modeled_bounces[0][0],
            (
                None
                if not bounce_targets or not bounce_targets[0]
                else bounce_targets[0][0].get("xyz_m")
            ),
        )
    checks = dict(
        connected_input_physics=bool(measurement["physical"]["compatible"]),
        **(
            {}
            if is_rally
            else {
                "serve_depth_plausible": bool(depth_bounds[0] <= contacts[0][1] <= depth_bounds[1])
            }
        ),
        **({} if reach is None else {"serve_contact_inside_reach_cylinder": bool(reach["inside"])}),
        **(legacy_checks if athlete_prior_mode == "global_hard_caps" else {}),
        contacts_inside_declared_search_envelope=inside_envelope,
        all_bounce_rays_agree=bool(
            all(
                value is not None and value <= 0.9144
                for group in bounce_gate_errors
                for value in group
            )
        ),
        bidirectional_windows_supported=bool(
            direction["maximum_window_rms_px"] is not None
            and direction["maximum_window_rms_px"] <= 16
        ),
        **(
            {"serve_ending_bounce_consistent": ending_consistency["passed"]}
            if ending_consistency is not None
            else {}
        ),
    )
    deaths = [name for name, passed in checks.items() if not passed]
    geometry_penalty = sum(value**2 for value in rooted_distances) + sum(
        1e6 if value is None else value**2 for group in bounce_gate_errors for value in group
    )
    geometry_penalty += sum(row["runback_soft_penalty"] for row in envelope_evidence)
    if athlete is not None:
        geometry_penalty += athlete["selector_penalty"]
    regularization = serve_reach_cylinder.evaluate_same_player_regularization(
        contacts[0], same_player_serve_regularization or {"status": "abstained"}
    )
    geometry_penalty += regularization["selector_penalty"]
    return dict(
        contact_xyz_m=[row.tolist() for row in contacts],
        player_distances_m=player_distances,
        bounce_xyz_m=[
            [None if row is None else row.tolist() for row in group] for group in modeled_bounces
        ],
        bounce_horizontal_errors_m=[group[0] for group in bounce_errors]
        if flat_targets
        else bounce_errors,
        bounce_gate_errors_m=[group[0] for group in bounce_gate_errors]
        if flat_targets
        else bounce_gate_errors,
        directional_support=direction,
        checks=checks,
        legacy_global_cap_diagnostics=legacy_checks
        | {
            "player_reaches_measured": len(rooted_distances),
            "player_reaches_absent_position": len(player_distances) - len(rooted_distances),
        },
        player_position=player_position.state_receipt(players),
        athlete_prior_mode=athlete_prior_mode,
        athlete_soft_priors=athlete,
        serve_reach_cylinder=reach,
        same_player_serve_regularization=regularization,
        contact_runback_evidence=envelope_evidence,
        **({} if ending_consistency is None else {"serve_ending_consistency": ending_consistency}),
        contact_xy_search_envelope_m=[list(row) for row in CONTACT_XY_ENVELOPE_M],
        death_reasons=deaths,
        survived=not deaths,
        input_geometry_penalty=float(geometry_penalty),
        input_only_rank_score=float(measurement["rms_px"]["training"] ** 2 + geometry_penalty),
        **native_seed_check_pixels.selection_fields([{"measurement": measurement}]),
    )


def render(report: dict, labels: dict, output: Path) -> list[str]:
    sources = agent_attempt_prepare.source_images(labels)
    clip = report["clip"]
    diagnostic = (
        report.get("selected_arms", {}).get("combined_toss_and_serve_prior")
        or report.get("selected_arms", {}).get("toss_witness")
        or report["selected"]
        or report["diagnostic_candidate"]
    )
    artifacts = []
    table = {row["frame"]: row for row in diagnostic["measurement"]["native_projection"]}
    source_frames = sorted(frame for source_clip, frame in sources if source_clip == clip)
    for event in report["events"]:
        desired = int(np.ceil(float(event["frame"])))
        low, high = event.get("frame_interval", [event["frame"], event["frame"]])
        choices = [
            frame
            for frame in source_frames
            if int(np.floor(low)) - 1 <= frame <= int(np.ceil(high)) + 1
        ]
        if not choices:
            continue
        frame = min(choices, key=lambda value: (abs(value - desired), value))
        image = Image.open(sources[(clip, frame)]).convert("RGB")
        draw = ImageDraw.Draw(image)
        row = table.get(frame)
        if row is not None:
            for xy, color, radius in (
                (row["owner"], "#00ff66", 8),
                (row["predicted"], "#ff3355", 6),
                (row["nominal_centre"], "#33bbff", 4),
            ):
                x, y = map(float, xy)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=color,
                    width=3,
                )
        evidence_kind = (
            "fitted overlay" if row is not None else "native context; no fit observation"
        )
        draw.text(
            (12, 12),
            f"{event['event_type']} event f{event['frame']} / native f{frame}: {evidence_kind}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        name = f"native_overlay_f{frame:04d}_{event['event_type']}.jpg"
        image.save(output / name, quality=92)
        artifacts.append(name)

    toss = diagnostic["evidence"].get("toss_witness")
    toss_supported = toss is not None and bool(toss["native_projection"])
    if toss_supported:
        toss_rows = toss["native_projection"]
        chosen = {row["frame"] for row in toss_rows[:: max(1, len(toss_rows) // 3)]}
        chosen.add(toss_rows[-1]["frame"])
        for row in toss_rows:
            if row["frame"] not in chosen:
                continue
            image = Image.open(sources[(clip, row["frame"])]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for xy, color, radius in (
                (row["observed_front"], "#00ff66", 8),
                (row["predicted_ball_centre"], "#ffcc00", 6),
            ):
                x, y = map(float, xy)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), outline=color, width=3
                )
            draw.text(
                (12, 12),
                f"toss f{row['frame']} green=front yellow=joined ballistic centre",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            name = f"native_toss_overlay_f{row['frame']:04d}.jpg"
            image.save(output / name, quality=92)
            artifacts.append(name)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for flight in diagnostic["measurement"]["dense_flights"]:
        xyz = np.asarray(flight["positions"])
        frames = np.linspace(flight["start_frame"], flight["end_frame"], len(xyz))
        axes[0].plot(xyz[:, 0], xyz[:, 1])
        axes[1].plot(frames, xyz[:, 2])
    if toss_supported:
        toss_frames = np.linspace(toss["release_frame"], toss["contact_frame"], 100)
        toss_times = (toss_frames - toss["contact_frame"]) / report["configuration"]["fps"]
        contact = np.asarray(toss["shared_contact_xyz_m"])
        velocity = np.asarray(toss["incoming_contact_velocity_mps"])
        toss_xyz = (
            contact
            + toss_times[:, None] * velocity
            + 0.5 * toss_times[:, None] ** 2 * toss_witness.GRAVITY
        )
        origin = np.asarray(toss["release_xyz_m"])
        axes[0].plot(toss_xyz[:, 0], toss_xyz[:, 1], color="#aa3377", lw=2, label="toss")
        axes[0].scatter(origin[0], origin[1], marker="^", color="#aa3377", label="release")
        axes[1].plot(toss_frames, toss_xyz[:, 2], color="#aa3377", lw=2, label="toss")
        axes[1].scatter(toss["release_frame"], origin[2], marker="^", color="#aa3377")
    axes[0].plot([0, 10.97, 10.97, 0, 0], [0, 0, 23.77, 23.77, 0], color="black")
    axes[0].axhline(11.885, color="black", ls="--")
    axes[0].set(xlabel="court X (m)", ylabel="court Y (m)", title="court plane", aspect="equal")
    height_interval = report["configuration"]["serve_height_zero_penalty_interval_m"]
    axes[1].axhspan(*height_interval, color="green", alpha=0.12)
    axes[1].set(xlabel="native frame", ylabel="ball centre Z (m)", title="side elevation")
    if toss_supported:
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "court_and_side.png", dpi=150)
    plt.close(fig)
    artifacts.append("court_and_side.png")

    coarse = report["coarse_candidates"]
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = [row["depth_hypothesis_m"] for row in coarse]
    ax.plot(
        depths, [row["measurement"]["rms_px"]["training"] for row in coarse], "o-", label="training"
    )
    check_split = (
        "check_in_sample"
        if report["configuration"].get("observation_partition") == "all_native"
        else "withheld"
    )
    ax.plot(
        depths,
        [row["measurement"]["rms_px"][check_split] for row in coarse],
        "s--",
        label="fitted check subset"
        if check_split == "check_in_sample"
        else "withheld (not selected on)",
    )
    for row in coarse:
        if row["evidence"]["survived"]:
            ax.axvline(row["depth_hypothesis_m"], color="green", alpha=0.2)
        if row["evidence"].get("toss_witness", {}).get("status") == "supported":
            ax.scatter(
                row["depth_hypothesis_m"],
                row["measurement"]["rms_px"]["training"],
                marker="*",
                s=110,
                color="#aa3377",
            )
    ax.set(
        xlabel="fixed serve-contact court Y (m)",
        ylabel="native RMS (px)",
        title="coarse depth family",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "depth_family.png", dpi=150)
    plt.close(fig)
    artifacts.append("depth_family.png")

    context = report["context_frames"]
    chosen = context[:3] + context[-3:]
    canvas = Image.new("RGB", (1440, 540), "black")
    for index, frame in enumerate(chosen):
        image = Image.open(sources[(clip, frame)]).convert("RGB")
        image.thumbnail((480, 270))
        ImageDraw.Draw(image).text(
            (10, 10),
            f"native f{frame}: image context only",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        canvas.paste(image, ((index % 3) * 480, (index // 3) * 270))
    canvas.save(output / "before_after_context.jpg", quality=92)
    artifacts.append("before_after_context.jpg")
    return artifacts


def dense_front_rows(document: dict) -> dict[int, dict]:
    """Index one every-frame dense revision by native frame.

    Both shipped dense schemas carry the same native leading-tip convention as
    the base export.  Only rows whose front is explicitly visible are indexed;
    an abstention stays an abstention.
    """
    records = document.get("records") or document.get("ball", {}).get("records", [])
    rows: dict[int, dict] = {}
    for record in records:
        for row in record.get("frames", []):
            front = row.get("front") or row
            if front.get("status") != "visible":
                continue
            if front.get("x1080") is None or front.get("y1080") is None:
                continue
            rows[int(row["frame"])] = row
    return rows


def supplement_label_streaks(labels: dict, document: dict) -> list[int]:
    """Fill the label record's missing trailing tips from a dense revision.

    The direction witness reads paired streaks off the single frozen ball record,
    and the toss witness requires exactly that one record.  So the dense rows are
    merged into it rather than appended as a second record, and a base row that
    already carries a paired streak is never overwritten.
    """
    records = document.get("records") or document.get("ball", {}).get("records", [])
    paired = {
        int(row["frame"]): row["streak"]
        for record in records
        for row in record.get("frames", [])
        if (row.get("streak") or {}).get("status") == "paired"
    }
    filled: list[int] = []
    for record in labels.get("ball", {}).get("records", []):
        for row in record.get("frames", []):
            frame = int(row["frame"])
            if (row.get("streak") or {}).get("status") == "paired" or frame not in paired:
                continue
            row["streak"] = paired[frame]
            filled.append(frame)
    return sorted(filled)


def supplement_with_dense_labels(attempt: dict, document: dict) -> dict:
    """Fill base abstentions from an additive dense revision; never overwrite.

    The dense packages declare exactly this merge policy.  Only a native frame
    the base export already inventories is filled, so no picture, timestamp or
    attempt boundary is invented.
    """
    dense = dense_front_rows(document)
    filled = []
    for row in attempt["owner_ball_labels"]:
        frame = int(row["frame"])
        if row["status"] == "visible" or frame not in dense:
            continue
        source = dense[frame]
        front = source.get("front") or source
        row["status"] = "visible"
        row["x1080"] = float(front["x1080"])
        row["y1080"] = float(front["y1080"])
        row["uncertainty_radius_px1080"] = (
            None
            if front.get("uncertainty_radius_px1080") is None
            else float(front["uncertainty_radius_px1080"])
        )
        row["annotation_origin"] = "agent_dense_revision"
        filled.append(frame)
    return {
        "schema": document.get("schema"),
        "frames_filled": filled,
        "filled_count": len(filled),
        "dense_visible_rows": len(dense),
        "merge_policy": "fill base abstentions only; base visible rows are never overwritten",
    }


def optional_input_domain_admission(
    branch: dict, operator: dict, *, include_seeded_final_net=False
) -> Callable[[dict], dict] | None:
    """Replay each topology's timing against its original events and all native rows."""
    from cv.experiments.connected_shooting.labeled_common_source import input_domain_admission

    return input_domain_admission(
        {name: branch[name] for name in ("scene", "heldout", "events", "native")}
        | {"observation_operator": operator}
        | ({"bounces": branch["bounces"]} if include_seeded_final_net else {}),
        include_seeded_final_net=include_seeded_final_net,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "packet", "cameras", "pose-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--search-seconds",
        type=float,
        help="cooperative numerical deadline; return input-ranked completed candidates",
    )
    parser.add_argument(
        "--search-incumbent",
        choices=("off", "on"),
        default="off",
        help=(
            "with --search-seconds: when the deadline lands inside a whole-point solve, "
            "return that solve's best finite same-invocation iterate under the input "
            "objective, measure it like any other candidate and mark it as a timeout "
            "iterate rather than a converged optimum; already completed candidates are "
            "retained either way"
        ),
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="write numerical report without optional image gallery",
    )
    parser.add_argument(
        "--point-context",
        type=Path,
        help="explicit tennis_point_context_prior_v1 receipt; soft ending selection only",
    )
    parser.add_argument("--coarse-iterations", type=int, default=20)
    parser.add_argument("--refine-iterations", type=int, default=80)
    parser.add_argument("--serve-speed", type=Path)
    parser.add_argument(
        "--serve-location-prior", type=Path, help="Explicit trained serve-prior artifact"
    )
    parser.add_argument("--serve-number", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--serve-number-uncertain",
        action="store_true",
        help="query the serve-location mixture across first and second serves",
    )
    parser.add_argument("--serve-number-evidence", type=Path, required=True)
    parser.add_argument("--automatic-track", type=Path)
    parser.add_argument("--automatic-track-scale", type=float, default=1.0)
    parser.add_argument("--surface", choices=("hard", "clay", "grass"), default="hard")
    parser.add_argument("--depth-hypothesis-m", type=float, action="append")
    parser.add_argument("--pose-image-scale", type=float, default=1.0)
    parser.add_argument(
        "--athlete-prior-mode",
        choices=ATHLETE_PRIOR_MODES,
        default="global_hard_caps",
    )
    parser.add_argument(
        "--missing-player-position",
        choices=MISSING_PLAYER_POSITION_POLICIES,
        default="off",
        help="allow an evidence-qualified recovery to declare a contact position absent",
    )
    parser.add_argument(
        "--player-stature",
        action="append",
        default=[],
        metavar="PLAYER=METRES",
        help="Explicit labeled-player stature for the stature_pose_soft arm",
    )
    parser.add_argument(
        "--athlete-evidence",
        choices=ATHLETE_EVIDENCE_POLICIES,
        default="required",
        help=(
            "required: roster order and statures must be supplied for the stature arm; "
            "optional: absent roster/statures make stature-scaled terms abstain explicitly"
        ),
    )
    parser.add_argument("--first-contact-role", choices=contact_role.ROLES, default="unspecified")
    parser.add_argument(
        "--athlete-root-reach-loss", choices=athlete_priors.ROOT_REACH_LOSSES, default="quadratic"
    )
    parser.add_argument(
        "--player-order",
        action="append",
        default=[],
        metavar="PLAYER",
        help="Explicit server/opponent order from the frozen topology",
    )
    parser.add_argument(
        "--bounce-witness-mode",
        choices=BOUNCE_WITNESS_MODES,
        default="subframe_graded_circle",
    )
    parser.add_argument(
        "--fit-ground-witness",
        choices=("off", "interval_ballistic_center"),
        default="off",
        help="optional center-only seed/anchor witness; acceptance witness is unchanged",
    )
    parser.add_argument(
        "--event-recovery",
        choices=("off", "on"),
        default="off",
        help="branch on input-only event hypotheses when topology is missing or wrong",
    )
    parser.add_argument("--max-topology-branches", type=int, default=6)
    parser.add_argument(
        "--player-ledger",
        type=Path,
        help="tennis_player_state_v1 sidecar supplying each contact's court position",
    )
    parser.add_argument(
        "--dense-labels",
        type=Path,
        help="additive every-frame dense revision used to fill base front abstentions",
    )
    parser.add_argument(
        "--toss-labels",
        type=Path,
        help=(
            "explicit toss-only development witness; accepts toss_v1 or an older dense_v1 "
            "document without supplementing the outgoing ball observations"
        ),
    )
    parser.add_argument(
        "--serve-toss-constraint",
        choices=("off", "on"),
        default="off",
        help=(
            "experimental arm: infer the serve contact from the labeled ballistic toss and "
            "server feet, seed the outgoing serve there, add its sigma-weighted residual, "
            "and replace the fixed 0.5 m depth branch with continuous three-sigma bounds"
        ),
    )
    parser.add_argument(
        "--interior-contact-epochs",
        choices=("off", "source_interval"),
        default="off",
        help="Fit interior contact times inside source intervals and intact native observation gaps; original event records stay fixed.",
    )
    parser.add_argument(
        "--serve-contact-epoch",
        choices=("off", "on"),
        default="off",
        help=(
            "fit the first contact continuously inside its labeled bracket, extending as early "
            "as one frame when the toss arc or first two outgoing pictures support the search"
        ),
    )
    parser.add_argument(
        "--serve-start-prior",
        choices=("off", "on"),
        default="off",
        help=(
            "seed and softly regularize serve contact from the external player/side/serve "
            "mixture, intersecting it with a supported toss estimate when available"
        ),
    )
    parser.add_argument(
        "--serve-contact-hypotheses",
        choices=("off", "on"),
        default="off",
        help=(
            "add two or three seed-only serve starts from the toss/outgoing intersection, "
            "external player prior and grounded feet; never add contact bounds"
        ),
    )
    parser.add_argument(
        "--serve-contact-history",
        type=Path,
        help="causal earlier accepted contacts for ranking-only same-player regularization",
    )
    parser.add_argument(
        "--serve-ending-consistency",
        choices=("off", "on"),
        default="off",
        help="require a labeled serve fault/out to land out in model and bounce witness",
    )
    parser.add_argument("--ending-kind", help="explicit labeled ending call for the serve gate")
    parser.add_argument(
        "--serve-reach-cut",
        choices=("off", "on"),
        default="off",
        help=(
            "cut the serve contact's image ray with a hard reach cylinder around the "
            "server's sided box: the fitted first contact is bounded inside the cylinder "
            "at every depth branch and a contact outside it is rejected"
        ),
    )
    parser.add_argument("--serve-reach-margin-statures", type=float, default=0.10)
    parser.add_argument("--serve-reach-radius-statures", type=float, default=1.35)
    parser.add_argument(
        "--server-pose-association",
        choices=("off", "nearest_detected"),
        default="off",
        help=(
            "off keeps the eight-frame pose lookup. nearest_detected, only when that "
            "lookup finds nothing, uses the nearest detected pose within the wide "
            "pre-contact window instead of refusing the component"
        ),
    )
    parser.add_argument(
        "--serve-side-association",
        choices=("centre", "serve_reach"),
        default="centre",
        help=(
            "centre takes the box nearest the serve contact pixel. serve_reach takes "
            "the box whose serving reach holds the ball (serve_side), which fixes "
            "the serving end of the depth beam and every alternating side after it"
        ),
    )
    parser.add_argument(
        "--observation-fallback",
        choices=("off", "on"),
        default="off",
        help=(
            "fail soft on observation preconditions: omit visible fronts with no supported "
            "camera, widen a bounce/contact bracket to nearby visible fronts, and plug in the "
            "nearest in-wing training axis. Every substitution is recorded per frame."
        ),
    )
    parser.add_argument(
        "--refine-inequalities",
        choices=("in", "out", "terminal_only"),
        default="in",
        help=(
            "experimental arm: keep the terminal-slack and 15.5 px direction-window SLSQP "
            "inequalities inside the refine solve, take both out, or keep only the terminal "
            "slack and let the unchanged acceptance gates judge the direction windows"
        ),
    )
    parser.add_argument(
        "--anchor-residuals",
        choices=("absent", "present"),
        default="absent",
        help=(
            "experimental arm: add the supplied bounce ground-ray targets (projection-derived "
            "sigma) and the automatic striker court position as soft residuals in the refine "
            "objective; the same witnesses are still scored by the unchanged gates"
        ),
    )
    parser.add_argument(
        "--exit-partial-endings",
        choices=("off", "on"),
        default="off",
        help=(
            "a final flight that ends at a ball-track picture exit or last visible sample is a "
            "partial flight (like a camera cut): it need not end at its last supplied bounce"
        ),
    )
    parser.add_argument(
        "--final-bounce-anchor",
        choices=("off", "on"),
        default="off",
        help=(
            "final flights whose bounce has a two-sided picture witness within one frame of the "
            "supplied bounce get a 1-frame timing prior at the witness epoch and, with seed "
            "restarts, one more restart launched to that epoch and ground point; gates unchanged"
        ),
    )
    parser.add_argument(
        "--seed-restarts",
        choices=("off", "three"),
        default="off",
        help=(
            "experimental arm: solve every coarse depth branch from three seeds -- the "
            "reference pinhole seed, an anchored walk from the server's contact ray and the "
            "striker court positions, and the same walk with flat spin -- and keep the best"
        ),
    )
    parser.add_argument(
        "--depth-conditioned-seed",
        choices=("off", "on"),
        default="off",
        help="Re-solve the coarse first-flight gravity state at each existing serve depth; same observations and search budget",
    )
    parser.add_argument(
        "--bounce-interval-timing",
        choices=("off", "on"),
        default="off",
        help=(
            "accept a modelled bounce that lands inside the label's own observed frame "
            "range, in addition to the existing midpoint allowance (owner-approved "
            "2026-09-19); strictly additive, nothing that passes today can fail"
        ),
    )
    parser.add_argument(
        "--automatic-ball-witness-sigma",
        choices=("off", "on"),
        default="off",
        help=(
            "give automatic ball rows their own measured pixel uncertainty in the two-wing "
            "bounce witness instead of the human-click floor (owner-approved 2026-09-19); "
            "inert on a labelled ball arm, which already declares its own radii"
        ),
    )
    parser.add_argument(
        "--net-cord-response",
        choices=("tape_clip", "admissible_set", "evidence_bound"),
        default="tape_clip",
        help=(
            "post-contact net-cord velocity: the VR tape clip, a searched admissible "
            "set (owner-directed 2026-09-20), or an evidence-bounded clip that "
            "abstains when the outgoing pixels are missing or ambiguous. Absent "
            "from the policy is tape_clip and byte-identical"
        ),
    )
    parser.add_argument(
        "--net-cord-tape-band-m",
        type=float,
        default=0.05,
        help=(
            "tape-band half-width beyond ball radius for admissible-set net contact "
            "(default 0.05 m; owner: very permissive of fit/noise, so 0.20 m is the "
            "wide arm). Unused when net-cord-response is tape_clip"
        ),
    )
    parser.add_argument(
        "--event-ground-seed",
        choices=("off", "on"),
        default="off",
        help=(
            "experimental arm: in the coarse structure-aware walk, also start each flight "
            "with exactly one supplied bounce from a gravity-only arc that reaches the "
            "seed's own native ground-ray impact at the supplied bounce epoch; an extra "
            "start only, judged by the walk's unchanged cost"
        ),
    )
    parser.add_argument(
        "--observation-partition",
        choices=("fifth_frame_withheld", "all_native"),
        default="fifth_frame_withheld",
        help="fit every native observation or preserve the historical fifth-frame check split",
    )
    parser.add_argument(
        "--terminal-net-seed",
        choices=("off", "on"),
        default="off",
        help="add an original-native net-ray airborne seed to terminal anchored walks",
    )
    parser.add_argument(
        "--reference-restart",
        choices=("off", "on"),
        default="off",
        help=(
            "experimental safety arm: carry the prior inequalities-in, anchor-free solve "
            "through every depth branch and select it against the active solve by the same "
            "input-only rank score"
        ),
    )
    parser.add_argument("--observation-scope", choices=("off", "on"), default="off")
    parser.add_argument(
        "--terminal-context-ownership",
        choices=("off", "on"),
        default="off",
        help="Bound native aftermath ownership and preserve original train/check rows",
    )
    parser.add_argument(
        "--terminal-rebound",
        choices=("off", "on"),
        default="off",
        help=(
            "experimental terminal-flight arm: activate every labeled picture after the last "
            "supplied bounce in the fit and require the measured bounce+dwell before that segment"
        ),
    )
    parser.add_argument(
        "--shooting-parameterization",
        choices=("single", "shared_contacts"),
        default="single",
        help=(
            "experimental arm: fit one explicit XYZ state per flight boundary, shared exactly "
            "by the incoming target and outgoing flight start, with sparse local Jacobian blocks"
        ),
    )
    parser.add_argument(
        "--local-flight-refits",
        choices=("off", "on"),
        default="off",
        help=(
            "shared-contact arm only: refit each flight's velocity/spin block from the joint "
            "optimum and incoming seed while holding every contact and neighbour bit fixed"
        ),
    )
    parser.add_argument(
        "--shared-state-second-stage-reference",
        type=Path,
        default=None,
        help=(
            "explicit promoted whole-point report whose rank-one candidate seeds the shared-state "
            "second stage and remains an unchanged fallback"
        ),
    )
    parser.add_argument(
        "--bounce-bracket-frames",
        type=float,
        default=1.0,
        help="bounce timing dead zone inside the fit, native frames (reference 1.0)",
    )
    parser.add_argument(
        "--anchor-player-sigma-m",
        type=float,
        default=exposure.ANCHOR_PLAYER_SIGMA_M,
        help="soft tie from each contact to its automatic striker court position",
    )
    parser.add_argument("--contact-components", choices=("off", "unresolved_ending"), default="off")
    parser.add_argument(
        "--contact-prefix-scope",
        choices=("off", "coverage", "terminal_identity", "unresolved_ending"),
        default="off",
    )
    parser.add_argument("--terminal-net-tail", choices=("off", "on"), default="off")
    parser.add_argument("--observed-horizon-tail", choices=("off", "on"), default="off")
    parser.add_argument(
        "--terminal-net-membership",
        choices=("off", "predicted"),
        default="off",
        help=(
            "treat an automatically predicted terminal net as uncertain membership: fit the "
            "original net-present scope and the net-absent observed-horizon scope over the "
            "same original rows, cameras and epochs, and select one whole branch by the "
            "existing input-only score"
        ),
    )
    parser.add_argument("--ground-settling", choices=("off", "on"), default="off")
    parser.add_argument(
        "--passive-bounce-response",
        choices=("off", "coupled_slip"),
        default="off",
        help=(
            "Coupled translation/spin after successive grounds with 32-impact capacity; "
            "first ground, normal law and settling threshold unchanged"
        ),
    )
    parser.add_argument("--observation-net-seed", choices=("off", "on", "candidate"), default="off")
    parser.add_argument("--net-seed-speed-scale-mps", type=float, default=15.0)
    parser.add_argument("--optional-contact-witness", type=Path)
    parser.add_argument("--optional-final-contacts", choices=("off", "on"), default="off")
    parser.add_argument(
        "--optional-contact-composition", choices=("off", "bounded_pairs"), default="off"
    )
    parser.add_argument(
        "--optional-interior-contacts",
        choices=("off", "on"),
        default="off",
        help=(
            "Open the remaining accepted-contact gaps, which the existing required repair "
            "never owns, to witnessed source-held contacts; existing branches unchanged"
        ),
    )
    parser.add_argument(
        "--optional-contact-timing",
        choices=("off", "pmf_peaks"),
        default="off",
        help="Source-PMF preparation epochs for existing optional contacts; original hypotheses retained",
    )
    parser.add_argument("--optional-bounce-witness", type=Path)
    parser.add_argument(
        "--optional-conditional-grounds",
        choices=("off", "on"),
        default="off",
        help=(
            "Compose bounded joint branches whose held source ground is re-qualified under "
            "each existing contact hypothesis' partition; original inventories unchanged"
        ),
    )
    args = parser.parse_args()
    from cv.experiments.connected_shooting import passive_bounce

    with passive_bounce.using(args.passive_bounce_response):
        return _run_search(args)


def explicit_observation_scope(args, *, contact_ending: bool, ball_track_terminal: bool) -> None:
    """A supplied scope must name its ending and must not rewrite topology.

    A contact boundary is explicit by itself. A ball-track terminal is too: the
    component stops at the ball's own second bounce, first out bounce, image-border
    exit, or wall, and that ending does not turn the terminal-rebound mechanism on.
    Every other supplied scope still requires terminal rebound. Event recovery
    stays off in all of these cases.
    """
    if (
        args.observation_scope != "on"
        or (not contact_ending and not ball_track_terminal and args.terminal_rebound != "on")
        or args.event_recovery != "off"
    ):
        raise ValueError(
            "observation scope requires explicit scope/rebound policy and unchanged supplied topology"
        )


def _run_search(args):
    from cv.experiments.connected_shooting import observation_operator

    budget = search_budget.Budget(args.search_seconds, args.output / "completed_candidates.json")
    native_seed_usages: list[dict] = []
    if args.search_incumbent == "on" and args.search_seconds is None:
        raise ValueError("search incumbent retention requires --search-seconds")
    budget.retain_incumbent = args.search_incumbent == "on"
    # Off: exactly the previous keyword set.  On: the whole-point solve may also
    # hand back its best finite iterate when the deadline lands inside it.
    deadline_arguments = (
        {}
        if args.search_seconds is None
        else {
            "deadline_check": budget.check,
            **({"retain_deadline_incumbent": True} if budget.retain_incumbent else {}),
        }
    )
    if args.search_seconds is not None and args.event_recovery != "off":
        raise ValueError(
            "deadline mode currently requires one supplied topology; event recovery selection is separate"
        )
    if args.serve_contact_hypotheses == "on" and (
        args.serve_contact_epoch != "on"
        or args.serve_start_prior != "off"
        or args.serve_toss_constraint != "off"
    ):
        raise ValueError(
            "serve contact hypotheses require fitted contact epoch and cannot be composed "
            "with the old toss/prior constraint arms"
        )
    if args.local_flight_refits == "on" and args.shooting_parameterization != "shared_contacts":
        raise ValueError("local flight refits require explicit shared contact states")
    if args.shared_state_second_stage_reference is not None and (
        args.shooting_parameterization != "shared_contacts" or args.event_recovery != "off"
    ):
        raise ValueError(
            "shared-state second-stage reuse requires shared contacts and fixed topology"
        )
    if args.serve_toss_constraint == "on" and not (args.toss_labels or args.dense_labels):
        raise ValueError("serve toss constraint requires explicit toss or dense labels")
    serve_reach_config = serve_reach_cylinder.ServeReachConfig(
        margin_statures=args.serve_reach_margin_statures,
        radius_statures=args.serve_reach_radius_statures,
    )
    if args.serve_reach_cut == "on":
        serve_reach_config.validate()
        if args.athlete_prior_mode != "stature_pose_soft":
            raise ValueError("the serve reach cut requires the stature-scaled athlete arm")
    observation_fallback = args.observation_fallback == "on"
    fallback_receipt: list[dict] = []
    if args.max_topology_branches < 1:
        raise ValueError("at least one topology branch required")
    if args.output.exists() or min(args.coarse_iterations, args.refine_iterations) < 1:
        raise ValueError("new output and positive solve budgets required")
    labels, packet, cameras_document = [
        json.loads(path.read_text()) for path in (args.labels, args.packet, args.cameras)
    ]
    operator = observation_operator.from_packet(packet)
    exposure_duration = operator["exposure_duration_frames"]
    fit_witness_by_topology = {}
    if args.fit_ground_witness != "off":
        from cv.experiments.connected_shooting import fit_ground_witness_center
    point_context_document = (
        None if args.point_context is None else json.loads(args.point_context.read_text())
    )
    baseline = None if args.baseline is None else json.loads(args.baseline.read_text())
    role_receipt = contact_role.validate(
        packet["attempts"][0], labels, requested=args.first_contact_role
    )
    # A bound rally origin and a bound unknown origin both refuse the
    # serve-specific priors; only the rally receipt also asserts a rally stroke.
    origin_role = contact_role.effective_role(role_receipt)
    nonserve_origin = role_receipt is not None
    is_rally = origin_role == "rally"
    origin_reason = None
    if nonserve_origin:
        contact_role.check_search_options(args)
        origin_reason = contact_role.abstention_reason(origin_role)
    speed_document = (
        {
            "reading": {
                "abstained": True,
                "abstention_reason": origin_reason,
                "visible_refresh_observed": False,
            }
        }
        if nonserve_origin
        else json.loads(args.serve_speed.read_text())
        if args.serve_speed
        else serve_speed_witness.from_label_document(labels)
    )
    speed_reading = speed_document["reading"]
    attempt = packet["attempts"][0]
    if args.depth_conditioned_seed == "on" and args.shared_state_second_stage_reference is not None:
        raise ValueError(
            "depth-conditioned seed requires a cold search, not a saved stage reference"
        )
    shared_stage_reference = (
        None
        if args.shared_state_second_stage_reference is None
        else second_stage_reference(
            json.loads(args.shared_state_second_stage_reference.read_text()), attempt["attempt_id"]
        )
    )
    dense_receipt = None
    dense_document = None
    if args.dense_labels is not None:
        dense_document = json.loads(args.dense_labels.read_text())
        dense_receipt = {
            **supplement_with_dense_labels(attempt, dense_document),
            "source": provenance.file_record(args.dense_labels),
        }
        dense_receipt["trailing_tips_filled"] = supplement_label_streaks(labels, dense_document)
    toss_document = None if args.toss_labels is None else json.loads(args.toss_labels.read_text())
    camera_lookup = {
        int(row["frame"]): np.asarray(row["P"], float)
        for row in cameras_document["cameras"]
        if row.get("supported", True) and "P" in row
    }
    camera_distortion_lookup = camera_geometry.camera_radial_map(cameras_document)
    if (
        args.event_recovery == "on"
        and camera_distortion_lookup is not None
        and any(row[0] != 0 for row in camera_distortion_lookup.values())
    ):
        raise ValueError("optional event-recovery ray proposals do not yet support radial cameras")
    label_lookup = {
        int(row["frame"]): np.asarray([row["x1080"], row["y1080"]], float)
        for row in attempt["owner_ball_labels"]
        if row["status"] == "visible"
    }
    radius_lookup = {
        int(row["frame"]): float(row["uncertainty_radius_px1080"])
        for row in attempt["owner_ball_labels"]
        if row["status"] == "visible" and row.get("uncertainty_radius_px1080") is not None
    }
    # Owner-approved 2026-09-19 (change M). The ball stream declares its own origin; the
    # rows themselves carry `annotation_origin` too, but the arm is one stream, so the
    # stream declaration is the honest granularity and it never reads a label to decide.
    # A labelled arm keeps the human-click floor: it already declares real radii per row.
    automatic_ball_rows = attempt.get("stream_origins", {}).get("ball") == "automatic"
    bounce_witness_sigma_floor_px = (
        AUTOMATIC_BALL_SIGMA_FLOOR_PX
        if args.automatic_ball_witness_sigma == "on" and automatic_ball_rows
        else BOUNCE_LABEL_SIGMA_FLOOR_PX
    )
    publish_bounce_frame_interval = args.bounce_interval_timing == "on"
    # Built ONCE and splatted into EVERY configuration receipt this search can write.
    # `per_flight_rescore.build_context` rebuilds the bounce witness for the replay that
    # produces the verdict, and it decides how by reading these keys back off the report. A
    # report written without them therefore replays the OLD witness and silently discards
    # both gate changes -- which is exactly what happened in the first gate wave, where the
    # search used the new witness and the verdict did not. Absent when off, so an unchanged
    # receipt stays byte for byte. Owner sign-off 2026-09-19.
    from cv.pipeline import net_cord_response as _net_cord

    net_cord_response_receipt = _net_cord.receipt(
        args.net_cord_response, h_tol_m=args.net_cord_tape_band_m
    )
    owner_approved_gate_receipt = {
        # Read back by per_flight_rescore so the verdict replay builds the same scene.
        **({} if args.exit_partial_endings == "off" else {"exit_partial_endings": "on"}),
        **(
            {}
            if args.bounce_interval_timing == "off"
            else {
                "bounce_interval_timing": args.bounce_interval_timing,
                "bounce_interval_timing_policy": (
                    "a modelled bounce inside the label's own observed frame range passes "
                    "bounce_count_timing, ORed with the existing midpoint allowance; "
                    "owner sign-off 2026-09-19"
                ),
            }
        ),
        **(
            {}
            if args.automatic_ball_witness_sigma == "off"
            else {
                "automatic_ball_witness_sigma": args.automatic_ball_witness_sigma,
                "automatic_ball_witness_sigma_applied": automatic_ball_rows,
                "automatic_ball_witness_sigma_px": bounce_witness_sigma_floor_px,
                "automatic_ball_witness_sigma_policy": (
                    "automatic ball rows carry no measured localization radius, so the "
                    "two-wing bounce witness floored them at the 2.0 px human-click sigma; "
                    "this is the measured in-track one-sigma of the automatic track against "
                    "the human click on the fresh36 development panel. Position is unchanged "
                    "(uniform weights); only the covariance grows. Owner sign-off 2026-09-19"
                ),
            }
        ),
    }

    from cv.experiments.connected_shooting import observation_scope

    scope_contract = observation_scope.validate(attempt, labels)
    from cv.pipeline import s6_contact_prefix_runtime as contact_prefix

    from cv.pipeline import s6_component_scope

    component_ending = s6_component_scope.active(attempt)
    terminal_track = s6_component_scope.track_terminal(attempt)
    contact_ending = (contact_prefix.active(attempt) or component_ending) and terminal_track is None
    if not component_ending and contact_role.leading_physical_prefix(attempt["events"]):
        # Whole-point numerics own contact-to-ending flights only. Original
        # evidence before the first contact belongs to an unobserved flight,
        # so it must reach a component, never a silently widened first flight.
        raise ValueError("original leading physical evidence requires the contact-component policy")
    prefix_interior = None
    if component_ending:
        if args.contact_components != "unresolved_ending":
            raise ValueError("contact component requires explicit shared policy")
        s6_component_scope.validate_inputs(
            attempt, labels, cameras_document, args.observation_partition
        )
        if (
            any(
                getattr(args, k) != "off"
                for k in (
                    "terminal_rebound",
                    "terminal_net_tail",
                    "observed_horizon_tail",
                    "terminal_net_membership",
                    "event_recovery",
                    "ground_settling",
                    "terminal_context_ownership",
                    "serve_ending_consistency",
                    "terminal_net_seed",
                    "observation_net_seed",
                    "contact_prefix_scope",
                )
            )
            or args.optional_contact_witness is not None
            or args.optional_bounce_witness is not None
        ):
            raise ValueError("contact component cannot use terminal or topology mechanisms")
    elif contact_ending:
        from cv.pipeline import s6_contact_prefix_scope

        prefix_mode = contact_prefix.bound_mode(attempt)
        if args.contact_prefix_scope != prefix_mode:
            raise ValueError("contact prefix requires explicit shared policy")
        s6_contact_prefix_scope.validate(
            attempt,
            cameras_document,
            observation_partition=args.observation_partition,
            labels=labels,
        )
        # A contact strictly inside the retained prefix needs no ending, so the
        # terminal-identity mode alone admits the prepared contact witness; every
        # terminal family, the bounce witness and the joint families stay off.
        scope_local = prefix_mode == s6_contact_prefix_scope.TERMINAL_IDENTITY
        if any(
            (
                args.terminal_rebound != "off",
                args.terminal_net_tail != "off",
                args.observed_horizon_tail != "off",
                args.terminal_net_membership != "off",
                args.event_recovery != "off",
                args.ground_settling != "off",
                args.terminal_context_ownership != "off",
                args.serve_ending_consistency != "off",
                args.terminal_net_seed != "off",
                args.observation_net_seed != "off",
                args.optional_contact_witness is not None and not scope_local,
                args.optional_bounce_witness is not None,
            )
        ):
            raise ValueError("contact prefix cannot use terminal or topology-changing mechanisms")
        if scope_local and any(
            (
                args.optional_final_contacts != "off",
                args.optional_contact_timing != "off",
                args.optional_conditional_grounds != "off",
            )
        ):
            raise ValueError("a prefix-local interior scope admits contact-only interior families")
    if scope_contract is not None:
        # ``terminal_track`` is set only for a component whose right boundary is
        # one of the four ball-track endings. It is an explicit scope; rebound
        # stays off, which the component branch above already requires.
        explicit_observation_scope(
            args,
            contact_ending=contact_ending,
            ball_track_terminal=terminal_track is not None,
        )
    from cv.pipeline import s6_optional_contacts as optional

    optional_document = None
    optional_hypotheses = []
    optional_selection = None
    optional_signature = None
    optional_training_signature = None
    optional_union = None
    optional_composed = False
    optional_family_receipts = []
    optional_binding_receipt = None
    optional_witness_path = args.optional_contact_witness or args.optional_bounce_witness
    if args.optional_final_contacts == "on" and args.optional_contact_witness is None:
        raise ValueError("optional final contacts require the optional contact witness")
    if args.optional_contact_composition != "off" and args.optional_contact_witness is None:
        raise ValueError("optional contact composition requires its prepared witness")
    if args.optional_interior_contacts != "off" and args.optional_contact_witness is None:
        raise ValueError("optional interior contacts require the optional contact witness")
    if args.optional_contact_timing != "off" and args.optional_contact_witness is None:
        raise ValueError("optional contact timing requires its prepared witness")
    optional_bounce_mode = args.optional_bounce_witness is not None
    optional_joint = args.optional_contact_witness is not None and optional_bounce_mode
    if args.optional_conditional_grounds != "off" and not optional_joint:
        raise ValueError("optional conditional grounds require both prepared source witnesses")
    original_search_seconds = args.search_seconds
    if optional_witness_path is not None:
        if args.event_recovery != "off" or args.search_seconds is None:
            raise ValueError("optional events require a fixed budget and ordinary recovery OFF")
        if optional_joint:
            # Union of the two existing beams: each arm still builds its own
            # alternatives from its own witness, and both share one budget below.
            from cv.pipeline import s6_optional_event_union as optional_union

            optional_documents = {
                "contact": json.loads(args.optional_contact_witness.read_text()),
                "bounce": json.loads(args.optional_bounce_witness.read_text()),
            }
            # v2, and the declared final-contact scope, additionally compose a bounded
            # joint family from these same two witnesses, which owns its own receipt
            # file below. A composing run without that file would hide the branches a
            # composed winner was actually chosen from.
            optional_composed = (
                optional_union.composes(optional_documents)
                or args.optional_conditional_grounds == "on"
            )
            optional_document = optional_documents["bounce"]
        else:
            optional_document = json.loads(optional_witness_path.read_text())
            if contact_ending:
                from cv.pipeline import s6_contact_prefix_scope

                # Requalify the existing source witness against the retained
                # prefix before any branch is built: a candidate that leaves the
                # prefix is recorded and unsupported, and nothing is imported.
                _, prefix_interior = s6_contact_prefix_scope.prefix_local_witness(
                    attempt, optional_document
                )
        if args.optional_contact_witness is not None:
            contact_document = (
                optional_documents["contact"] if optional_joint else optional_document
            )
            from cv.pipeline import s6_contact_composition, s6_contact_timing

            if s6_contact_composition.policy(contact_document) != args.optional_contact_composition:
                raise ValueError(
                    "optional contact composition witness differs from declared policy"
                )
            if s6_contact_timing.policy(contact_document) != args.optional_contact_timing:
                raise ValueError("optional contact timing witness differs from declared policy")
            from cv.pipeline import s6_witnessed_interior_contacts

            if (
                s6_witnessed_interior_contacts.policy(contact_document)
                != args.optional_interior_contacts
            ):
                raise ValueError("optional interior contact witness differs from declared policy")
            if optional.declared_final_scope(contact_document) != (
                args.optional_final_contacts == "on"
            ):
                raise ValueError("optional contact witness scope differs from the declared policy")
    numerical_failures_by_topology = {}
    net_tail_contract = attempt.get("terminal_net_tail") if args.terminal_net_tail == "on" else None
    if attempt.get("terminal_net_tail") is not None and net_tail_contract is None:
        raise ValueError("terminal net-tail inputs require the explicit shared policy")
    horizon_tail_contract = (
        attempt.get("observed_horizon_tail") if args.observed_horizon_tail == "on" else None
    )
    if attempt.get("observed_horizon_tail") is not None and horizon_tail_contract is None:
        raise ValueError("observed-horizon tail inputs require the explicit shared policy")

    def check_tail_contracts(net_contract: dict | None, horizon_contract: dict | None) -> None:
        """One shared eligibility rule for every declared branch scope."""
        if horizon_contract is not None:
            from cv.experiments.connected_shooting import (
                observed_horizon_tail as horizon_tail_module,
            )

            if (
                net_contract is not None
                or horizon_contract["kind"] != horizon_tail_module.KIND
                or horizon_contract["physical_ending"] is not None
                or horizon_contract["terminal_ground_count"] != "unknown"
            ):
                raise ValueError(
                    "observed-horizon tail requires its own unresolved single-kind contract"
                )
        if net_contract is not None:
            from cv.experiments.connected_shooting import (
                labeled_terminal_net_tail as net_tail,
                net_collision,
            )

            # Rebuild qualification from the original invocation packet is checked by
            # stage/export; search binds and records the complete source interval
            # without editing it.
            if (
                net_contract["kind"] != net_tail.KIND
                or net_collision.active_policy() != "physical_mesh_v1"
            ):
                raise ValueError(
                    "terminal net tail requires original interval and physical mesh eligibility"
                )

    check_tail_contracts(net_tail_contract, horizon_tail_contract)
    from cv.pipeline import s6_terminal_net_membership as net_membership

    membership_policy = net_membership.policy(args.terminal_net_membership)
    if attempt.get(net_membership.RECEIPT_KEY) is not None and membership_policy == "off":
        raise ValueError("terminal net membership inputs require the explicit shared policy")
    membership_names: list[str | None] = [None]
    if membership_policy != "off":
        if attempt.get(net_membership.RECEIPT_KEY) is None or net_tail_contract is None:
            raise ValueError(
                "terminal net membership requires its qualified original net contract and receipt"
            )
        if horizon_tail_contract is not None:
            raise ValueError("terminal net membership owns the declared absent-branch horizon")
        if args.event_recovery != "off" or args.search_seconds is None:
            raise ValueError(
                "terminal net membership requires a fixed budget and ordinary recovery OFF"
            )
        membership_names = list(net_membership.MEMBERSHIPS)
    membership_active = len(membership_names) > 1

    def build_optional(branch_attempt: dict) -> dict:
        """The ordinary optional beams, rebuilt under one declared branch scope.

        Every arm keeps its own witness, source order and limits. Only the
        accepted event partition and the declared ending change, to exactly the
        branch this membership declares, so an existing source ground candidate
        requalifies from its original emission rather than a new inventory.
        """
        if optional_witness_path is None:
            return {
                "documents": None,
                "document": None,
                "hypotheses": [],
                "family_receipts": [],
                "binding": None,
            }
        if optional_joint:
            documents = {
                **optional_documents,
                "bounce": net_membership.bounce_witness(
                    branch_attempt, optional_documents["bounce"]
                ),
            }
            binding = optional_union.check_source_binding(
                documents, branch_attempt, args.optional_conditional_grounds
            )
            rows, receipts = optional_union.hypotheses(
                documents, branch_attempt, args.optional_conditional_grounds
            )
            return {
                "documents": documents,
                "document": documents["bounce"],
                "hypotheses": rows,
                "family_receipts": receipts,
                "binding": binding,
            }
        document = (
            net_membership.bounce_witness(branch_attempt, optional_document)
            if optional_bounce_mode
            else optional_document
        )
        return {
            "documents": None,
            "document": document,
            "hypotheses": optional.search_hypotheses(document, branch_attempt),
            "family_receipts": [],
            "binding": None,
        }

    membership_state: dict[str | None, dict] = {}
    for name in membership_names:
        branch_attempt = (
            attempt if name is None else net_membership.effective_attempt(attempt, name)
        )
        branch_net = net_tail_contract if name is None else branch_attempt.get("terminal_net_tail")
        branch_horizon = (
            horizon_tail_contract if name is None else branch_attempt.get("observed_horizon_tail")
        )
        if name is not None:
            check_tail_contracts(branch_net, branch_horizon)
        membership_state[name] = {
            "attempt": branch_attempt,
            "net_tail": branch_net,
            "horizon_tail": branch_horizon,
            "absent_nets": (
                []
                if name in (None, net_membership.PRESENT)
                else [attempt[net_membership.RECEIPT_KEY]["original_net_event"]]
            ),
            **build_optional(branch_attempt),
        }
    first = membership_state[membership_names[0]]
    optional_hypotheses = first["hypotheses"]
    optional_document = first["document"]
    optional_family_receipts = first["family_receipts"]
    optional_binding_receipt = first["binding"]

    def fit_topology(
        events: list[dict],
        end_frame: float,
        topology_name: str,
        *,
        membership: str | None = None,
    ) -> dict:
        """Fit one candidate topology end to end and return its whole receipt.

        The supplied topology and every recovered alternative go through exactly
        this function, so a recovered branch is never scored by a different rule
        than the reference arm. ``membership`` selects one declared terminal-net
        branch; every scope this solve reads comes from that branch's validated
        effective attempt, never from a captured global contract.
        """
        nonlocal optional_signature, optional_training_signature
        branch_state = membership_state[membership]
        attempt = branch_state["attempt"]
        net_tail_contract = branch_state["net_tail"]
        horizon_tail_contract = branch_state["horizon_tail"]
        optional_hypotheses = branch_state["hypotheses"]
        if scope_contract is not None and (
            (
                events != attempt["events"]
                and not any(events == h["events"] for h in optional_hypotheses)
            )
            or end_frame
            != scope_contract.get("modeled_horizon", scope_contract.get("observation_horizon"))
        ):
            raise ValueError(
                "observation-scoped solve cannot drop/change physical events or horizon"
            )
        if net_tail_contract is not None and not any(
            e["event_type"] == "net_hit"
            and float(e["frame"]) == net_tail_contract["representative"]
            for e in events
        ):
            events = sorted(
                [*events, copy.deepcopy(attempt["terminal_net_hypothesis_event"])],
                key=lambda e: float(e["frame"]),
            )
        # Packet context can retain later physical events whose contacts are
        # outside this competitive topology. Only the declared branch horizon
        # (including qualified preparation extensions) belongs to this solve.
        events, topology_scope = scoped_branch_events(
            events, end_frame, horizon_contract=horizon_tail_contract
        )
        end_frame = topology_scope["modeled_branch_end_frame"]
        if not any(row["event_type"] == "contact" for row in events):
            raise ValueError("connected topology requires an original contact")
        terminal = [
            row
            for row in events
            if row["event_type"] == "bounce"
            and float(row["frame"])
            > max(float(item["frame"]) for item in events if item["event_type"] == "contact")
        ]
        if terminal_track is not None:
            from cv.pipeline import s6_contact_components as _terminal_components

            if len(terminal) not in _terminal_components.terminal_bounce_allowance(
                terminal_track["kind"]
            ):
                raise ValueError("ball-track terminal bounce count does not match its endpoint")
            termination_kind = terminal_track["kind"]
        elif not any(row["event_type"] == "contact" for row in events) or len(terminal) not in (
            {0}
            if net_tail_contract is not None or contact_ending
            else {0, 1, 2}
            if horizon_tail_contract is not None
            else {1, 2}
        ):
            raise ValueError(
                "connected topology ending at its first or second terminal bounce required"
            )
        else:
            termination_kind = (
                "original_contact"
                if contact_ending
                else "net_stop"
                if net_tail_contract is not None
                else "observed_horizon"
                if horizon_tail_contract is not None
                else "second_bounce"
                if len(terminal) == 2
                else "terminal_bounce"
            )
        # A recovered topology may end later than the supplied one.  The fronts
        # between the two epochs are already in the frozen label document, so
        # this branch carries them; the supplied branch is untouched.
        branch_attempt = (
            copy.deepcopy(attempt)
            if contact_ending
            else event_recovery.extend_attempt_window(attempt, labels["ball"]["records"], end_frame)
        )
        rebound_inventory = {"mode": "off"}
        modeled_end_frame = float(end_frame)
        if net_tail_contract is not None:
            modeled_end_frame = float(net_tail_contract["observation_horizon"])
            branch_attempt = event_recovery.extend_attempt_window(
                branch_attempt, labels["ball"]["records"], modeled_end_frame
            )
        if horizon_tail_contract is not None:
            # No supplied terminal bounce exists, so nothing may index
            # ``terminal[-1]`` or require a physical ending on this branch.
            modeled_end_frame = float(horizon_tail_contract["observation_horizon"])
            branch_attempt = event_recovery.extend_attempt_window(
                branch_attempt, labels["ball"]["records"], modeled_end_frame
            )
        if (
            args.terminal_rebound == "on"
            and net_tail_contract is None
            and horizon_tail_contract is not None
        ):
            rebound_inventory = {
                "mode": "off",
                "status": "unsupported",
                "reason": "unresolved_horizon_is_not_a_physical_ending",
            }
        elif args.terminal_rebound == "on" and net_tail_contract is None:
            rebound_inventory = exposure.terminal_rebound_inventory(
                branch_attempt,
                labels["ball"]["records"],
                float(terminal[-1]["frame"]),
                labels=labels,
                duration_frames=exposure_duration,
            )
            if args.terminal_context_ownership == "on":
                from cv.pipeline import s6_terminal_context

                rebound_inventory = s6_terminal_context.limit_inventory(
                    rebound_inventory,
                    labels,
                    dict(
                        clip=branch_attempt["point_clip"],
                        events=events,
                        configuration=dict(
                            **owner_approved_gate_receipt,
                            **net_cord_response_receipt,
                            **(
                                {
                                    contact_role.FIELD: role_receipt,
                                    "first_contact_parameterization": "ordinary_contact_free_xyz",
                                    "serve_evidence_applicable": False,
                                }
                                if nonserve_origin
                                else {}
                            ),
                            fps=branch_attempt["fps"],
                            termination_kind=termination_kind,
                            exposure_duration_frames=exposure_duration,
                        ),
                    ),
                )
            if rebound_inventory["status"] == "supported":
                modeled_end_frame = max(
                    modeled_end_frame,
                    float(rebound_inventory["last_postbounce_labeled_frame"]),
                )
                branch_attempt = event_recovery.extend_attempt_window(
                    branch_attempt, labels["ball"]["records"], modeled_end_frame
                )
        if optional_hypotheses or membership_active:
            # The same full-frame source signature bounds every branch, including a
            # membership run with no ordinary optional hypotheses: neither branch
            # may broaden or shorten the original pixels, epochs or cameras.
            signature = optional.observation_signature(
                branch_attempt, cameras_document, args.observation_partition
            )
            if optional_signature is None:
                optional_signature = signature
            elif signature != optional_signature:
                raise ValueError("optional topology changed source epoch/pixel/camera domain")
        from cv.experiments.connected_shooting.observation_partition import witness_policy

        scene, heldout, bounces, native, outside_window = prepare_attempt(
            branch_attempt,
            cameras_document,
            args.surface,
            events,
            modeled_end_frame,
            observation_partition=args.observation_partition,
            bounce_witness_observation_policy=witness_policy(args.observation_partition),
            ground_settling=args.ground_settling == "on",
            observation_fallback=observation_fallback,
            fallback_receipt=fallback_receipt,
            exit_partial_endings=args.exit_partial_endings == "on",
        )
        net_groups = tuple(
            np.asarray(
                [
                    event["frame"]
                    for event in events
                    if event["event_type"] == "net_hit" and a < event["frame"] < b
                ],
                float,
            )
            for a, b in zip(scene.contact_frames[:-1], scene.contact_frames[1:], strict=True)
        )
        if any(len(group) for group in net_groups):
            scene = replace(scene, net_hit_frames=net_groups)
            heldout = replace(heldout, net_hit_frames=net_groups)
            scene.validate()
            heldout.validate()
        reference = dict(
            annotation_status="frozen_agent_reference", records=labels["ball"]["records"]
        )
        axes, axis_receipt = training_directions(
            scene,
            bounces,
            reference,
            observation_fallback=observation_fallback,
            exposure_duration=exposure_duration,
            fallback_receipt=fallback_receipt,
        )
        terminal_rebound_frames = None
        terminal_rebound_receipt = rebound_inventory
        if (
            args.terminal_rebound == "on"
            and net_tail_contract is None
            and horizon_tail_contract is None
            and rebound_inventory.get("status") == "supported"
        ):
            segment = exposure.terminal_rebound_segment
            if args.terminal_context_ownership == "on":
                from cv.pipeline import s6_terminal_context

                segment = s6_terminal_context.preserved_segment
            scene, axes, terminal_rebound_frames, terminal_rebound_receipt = segment(
                scene, heldout, bounces, axes
            )

            terminal_rebound_receipt = {
                **rebound_inventory,
                **terminal_rebound_receipt,
                "modeled_scene_end_frame": modeled_end_frame,
                **(
                    {"observation_horizon": float(end_frame), "semantic_ending_frame": None}
                    if scope_contract is not None
                    else {"semantic_ending_frame": float(end_frame)}
                ),
            }
        if optional_hypotheses or membership_active:
            training_signature = optional.training_observation_signature(
                branch_attempt,
                cameras_document,
                args.observation_partition,
                rebound_inventory,
                terminal_rebound_receipt,
            )
            if optional_training_signature is None:
                optional_training_signature = training_signature
            elif training_signature != optional_training_signature:
                raise ValueError(
                    "optional topology changed declared training observation ownership"
                )
        contact_events = [row for row in events if row["event_type"] == "contact"]
        bounce_events = [row for row in events if row["event_type"] == "bounce"]
        # The seed's own bounce witness.  A flight whose pre-bounce pictures
        # cannot identify a gravity-only arc on their own is still bounded by
        # the ball's observed impact, and losing the attempt before any fit is
        # worse than seeding it from that measured position.
        seed_targets = seed_ground_targets(
            scene,
            bounce_events,
            camera_lookup,
            label_lookup,
            radius_lookup,
            camera_distortion=camera_distortion_lookup,
            terminal_rebound_frames=terminal_rebound_frames,
            mode=args.bounce_witness_mode,
            observation_fallback=observation_fallback,
        )
        fit_witness_receipt = None
        if args.fit_ground_witness != "off":
            fit_witness_receipt = fit_ground_witness_center.source_groups(
                scene,
                bounce_events,
                events + labels.get("events", {}).get("records", []),
                camera_lookup,
                label_lookup,
                radius_lookup,
                operator=operator,
                camera_distortion=camera_distortion_lookup,
                terminal_rebound_frames=terminal_rebound_frames,
            )
            # Preserve source-only receipts even if baseline initialization fails
            # before any candidate or final report can be produced.
            fit_witness_by_topology[f"{membership}:{topology_name}"] = fit_witness_receipt
            search_budget.atomic_json(
                args.output / "fit_ground_witness.json", fit_witness_by_topology
            )
            seed_targets = fit_ground_witness_center.fitting_seed_targets(
                seed_targets, fit_witness_receipt
            )
        initial, initialization_evidence = baseline_seed(
            baseline,
            scene,
            bounces,
            bounce_ground_targets=seed_targets,
            observation_fallback=observation_fallback,
            fallback_receipt=fallback_receipt,
            terminal_rebound=terminal_rebound_frames is not None,
            exposure_duration=exposure_duration,
        )
        if fit_witness_receipt is not None:
            initialization_evidence = {
                **initialization_evidence,
                "fit_ground_witness": fit_witness_receipt,
            }
        player_names, statures, athlete_evidence = resolve_athlete_evidence(
            args.athlete_prior_mode,
            args.athlete_evidence,
            list(args.player_order),
            list(args.player_stature),
            labels,
            contact_events,
        )
        # A bound rally or unknown origin is not a serve; keep its association. The
        # search report records a non-default choice so the replay reproduces it.
        serve_association_kwargs = (
            {"serve_side_association": args.serve_side_association}
            if not nonserve_origin and args.serve_side_association != "centre"
            else {}
        )
        players, right_contact_player = launch_and_right_players(
            args.pose_csv,
            attempt["point_clip"],
            contact_events,
            label_lookup,
            right_boundary_kind=scene.right_boundary_kind,
            image_coordinate_scale=args.pose_image_scale,
            player_names=player_names,
            player_statures_m=statures,
            observation_fallback=observation_fallback,
            fallback_receipt=fallback_receipt,
            missing_player_position=args.missing_player_position,
            **serve_association_kwargs,
        )
        players = contact_role.apply_players(
            mark_athlete_evidence(players, athlete_evidence), role_receipt
        )
        if right_contact_player is not None:
            right_contact_player = mark_athlete_evidence([right_contact_player], athlete_evidence)[
                0
            ]
        athlete_priors.configure_root_reach(players, args.athlete_root_reach_loss)
        ledger_receipt = None
        if args.player_ledger is not None:
            ledger_receipt = {
                "source": provenance.file_record(args.player_ledger),
                **apply_player_ledger(
                    players,
                    contact_events[:-1] if contact_ending else contact_events,
                    player_ledger_positions(args.player_ledger, attempt["point_clip"]),
                ),
            }
        origin_depth_cue = None
        if nonserve_origin:
            # The rally depth cue needs a demonstrated rally origin. An unknown
            # origin gets the same widest venue envelope and no cue at all.
            if is_rally:
                with args.pose_csv.open() as stream:
                    cue_pose_rows = list(csv.DictReader(stream))
                origin_depth_cue = rally_cue.prepare(
                    players, events, labels, cameras_document, cue_pose_rows, args.pose_image_scale
                )
            low, high = contact_role.limits(players[0])
            depths = np.asarray([float(np.clip(initial[1], low, high))])
        else:
            low, high, depths = single.limits(players[0]["side"])
        if args.depth_hypothesis_m:
            depths = np.asarray(args.depth_hypothesis_m, float)
            if not np.isfinite(depths).all() or len(set(depths.tolist())) != len(depths):
                raise ValueError("finite unique explicit depth hypotheses required")
        all_depths = depths.copy()
        excluded_depths = []
        if args.serve_toss_constraint == "off" and args.serve_start_prior == "off":
            depths, excluded_depths = fixed_depth_beam(depths, (low, high))
        from cv.experiments.connected_shooting.observation_partition import witness_frames

        targets = [
            [
                event_ground_target(
                    row,
                    camera_lookup,
                    label_lookup,
                    radius_lookup,
                    camera_distortion=camera_distortion_lookup,
                    mode=args.bounce_witness_mode,
                    observation_fallback=observation_fallback,
                    fallback_receipt=fallback_receipt,
                    eligible_frames=witness_frames(scene, flight_index, terminal_rebound_frames),
                    sigma_floor_px=bounce_witness_sigma_floor_px,
                    publish_frame_interval=publish_bounce_frame_interval,
                )
                for row in bounce_events
                if start < float(row["frame"]) <= end
            ]
            for flight_index, (start, end) in enumerate(
                zip(scene.contact_frames, scene.contact_frames[1:])
            )
        ]
        if [len(group) for group in targets] != [len(group) for group in bounces]:
            raise ValueError("every supplied bounce requires one native ground-ray target")
        # Acceptance targets remain original all the way through candidate_evidence
        # and report replay. Only seed and anchor consumers receive this copy.
        fit_anchor_targets = (
            targets
            if fit_witness_receipt is None
            else fit_ground_witness_center.fitting_anchor_targets(targets, fit_witness_receipt)
        )
        contact_interval = tuple(map(float, contact_events[0]["frame_interval"]))
        toss_source = toss_document or dense_document
        if nonserve_origin:
            toss_observations = {"status": "not_applicable", "reason": origin_reason}
        elif args.serve_toss_constraint == "off" and not (
            (args.serve_start_prior == "on" or args.serve_contact_hypotheses == "on")
            and toss_source is not None
        ):
            toss_observations = toss_witness.precontact_observations(
                labels,
                cameras_document,
                contact_frame=float(scene.contact_frames[0]),
                automatic_track=args.automatic_track,
                automatic_track_scale=args.automatic_track_scale,
            )
        else:
            try:
                toss_observations = toss_witness.contact_constraint_observations(
                    toss_source,
                    cameras_document,
                    clip=attempt["point_clip"],
                    contact_frame=float(scene.contact_frames[0]),
                    contact_frame_interval=contact_interval,
                )
            except (ValueError, FloatingPointError, OverflowError) as error:
                if args.serve_start_prior != "on" and args.serve_contact_hypotheses != "on":
                    raise
                toss_observations = abstained_toss_observations(
                    error, float(scene.contact_frames[0]), contact_interval
                )
                toss_observations["clip"] = attempt["point_clip"]
                fallback_receipt.append(
                    {
                        "fallback": "optional_toss_document_abstained",
                        "reason": toss_observations["abstention_reason"],
                    }
                )
        serve_contact_pixel = contact_association_pixel(
            contact_events[0],
            label_lookup,
            observation_fallback=observation_fallback,
            fallback_receipt=fallback_receipt,
        )
        if nonserve_origin:
            toss_feet = {"side": players[0]["side"], "status": "not_applicable"}
        else:
            try:
                toss_feet = toss_witness.server_feet(
                    args.pose_csv,
                    attempt["point_clip"],
                    float(scene.contact_frames[0]),
                    serve_contact_pixel,
                    image_coordinate_scale=args.pose_image_scale,
                    observation_fallback=observation_fallback,
                    nearest_detected_frame=args.server_pose_association == "nearest_detected",
                    fallback_receipt=fallback_receipt,
                    **(
                        {"serve_side_association": args.serve_side_association}
                        if args.serve_side_association != "centre"
                        else {}
                    ),
                )
            except ValueError as error:
                if args.serve_start_prior != "on" and args.serve_contact_hypotheses != "on":
                    raise
                toss_feet = contact_player_feet_fallback(players[0], error)
                fallback_receipt.append(
                    {
                        "fallback": "contact_player_state_for_server_feet",
                        "reason": toss_feet["fallback_reason"],
                    }
                )
        toss_contact_estimate = None
        toss_contact_bounds = None
        location_prior = None
        location_prior_bounds = None
        contact_hypotheses = []
        contact_intersection = None
        contact_hypothesis_blocker = None
        contact_hypothesis_restart_failures = []
        contact_history, contact_history_receipt = load_contact_history(
            args.serve_contact_history, attempt["attempt_id"]
        )
        same_player_regularization = None
        if args.serve_toss_constraint == "on":
            toss_feet = toss_witness.labeled_server_anchor(
                toss_document or dense_document, cameras_document, toss_feet
            )
            server_bound = serve_reach_cylinder.build(players[0], serve_reach_config)
            centre = np.asarray(server_bound.centre_xy_m, float)
            server_low = np.asarray(
                [
                    centre[0] - server_bound.radius_m,
                    max(low, centre[1] - server_bound.radius_m),
                    server_bound.height_interval_m[0],
                ]
            )
            server_high = np.asarray(
                [
                    centre[0] + server_bound.radius_m,
                    min(high, centre[1] + server_bound.radius_m),
                    server_bound.height_interval_m[1],
                ]
            )
            toss_contact_estimate = toss_witness.fit_contact(
                toss_observations,
                toss_feet,
                float(attempt["fps"]),
                contact_frame_interval=tuple(toss_observations["contact_frame_interval"]),
                contact_bounds=(server_low, server_high),
                config=toss_witness.CONTACT_CONSTRAINT_CONFIG,
            )
            if toss_contact_estimate["status"] != "supported":
                raise ValueError(
                    "toss contact estimate held: "
                    + ", ".join(
                        toss_contact_estimate.get("failures")
                        or [toss_contact_estimate.get("abstention_reason", "unknown")]
                    )
                )
            initial, toss_seed = initialization.toss_contact_seed(
                scene, initial, toss_contact_estimate
            )
            initialization_evidence = {
                **initialization_evidence,
                "serve_toss_constraint": toss_seed,
            }
            mean = np.asarray(toss_contact_estimate["contact_xyz_m"], float)
            sigma = np.asarray(toss_contact_estimate["contact_sigma_m"], float)
            bound_low = np.maximum(
                mean - 3 * sigma,
                server_low,
            )
            bound_high = np.minimum(
                mean + 3 * sigma,
                server_high,
            )
            if np.any(bound_low > bound_high):
                raise ValueError("toss three-sigma box does not intersect the server reach bound")
            toss_contact_bounds = {
                "x_interval_m": [float(bound_low[0]), float(bound_high[0])],
                "y_interval_m": [float(bound_low[1]), float(bound_high[1])],
                "z_interval_m": [float(bound_low[2]), float(bound_high[2])],
                "toss_three_sigma_box_m": [
                    (mean - 3 * sigma).tolist(),
                    (mean + 3 * sigma).tolist(),
                ],
                "server_reach_bounding_box_m": [
                    [
                        float(centre[0] - server_bound.radius_m),
                        float(centre[1] - server_bound.radius_m),
                        float(server_bound.height_interval_m[0]),
                    ],
                    [
                        float(centre[0] + server_bound.radius_m),
                        float(centre[1] + server_bound.radius_m),
                        float(server_bound.height_interval_m[1]),
                    ],
                ],
                "bound_kind": "optimizer_parameter_bounds_not_inequality",
                "continuous_first_contact_depth": True,
            }
            all_depths = depths = np.asarray([float(mean[1])])
        elif args.serve_start_prior == "on":
            # Toss is evidence, never a precondition. A held or missing fit
            # simply leaves the external player/side/serve mixture untouched.
            if toss_source is not None and toss_observations.get("status") == "supported":
                try:
                    toss_contact_estimate = toss_witness.fit_contact(
                        toss_observations,
                        toss_feet,
                        float(attempt["fps"]),
                        contact_frame_interval=tuple(toss_observations["contact_frame_interval"]),
                        config=toss_witness.CONTACT_CONSTRAINT_CONFIG,
                    )
                except (ValueError, FloatingPointError, OverflowError) as error:
                    toss_contact_estimate = {
                        "schema": "connected_toss_contact_estimate_v1",
                        "status": "abstained",
                        "abstention_reason": f"{type(error).__name__}: {error}",
                        "contact_xyz_m": None,
                    }
            location_prior, location_prior_bounds = serve_start_location_prior(
                labels,
                players[0],
                targets[0][0],
                None if args.serve_number_uncertain else args.serve_number,
                toss_contact_estimate,
                artifact_path=args.serve_location_prior,
            )
            contact_estimate = {
                "status": "supported",
                "contact_xyz_m": location_prior_bounds["target_xyz_m"],
                "contact_sigma_m": location_prior_bounds["sigma_m"],
            }
            initial, prior_seed = initialization.serve_start_prior_seed(
                scene, initial, contact_estimate, targets[0][0]
            )
            initialization_evidence = {
                **initialization_evidence,
                "serve_start_location_prior": {
                    **prior_seed,
                    "source": "external_mixture_intersected_with_toss_when_supported",
                },
            }
            all_depths = depths = np.asarray([float(location_prior_bounds["target_xyz_m"][1])])
        elif args.serve_contact_hypotheses == "on":
            try:
                location_prior, location_metadata = serve_start_location_prior(
                    labels,
                    players[0],
                    targets[0][0],
                    None if args.serve_number_uncertain else args.serve_number,
                    None,
                    artifact_path=args.serve_location_prior,
                )
                # The old consumer's three-sigma box is deliberately discarded.  It is
                # retained only as metadata proving which side/end cell was queried.
                location_prior_bounds = location_metadata
                outgoing_frames = np.asarray(scene.observation_frames[0], float)
                outgoing_keep = np.flatnonzero(outgoing_frames >= contact_interval[0])[:6]
                contact_intersection = toss_witness.contact_curve_intersection(
                    toss_observations,
                    outgoing_frames[outgoing_keep],
                    np.asarray(scene.pixels[0], float)[outgoing_keep],
                    np.asarray(scene.cameras[0], float)[outgoing_keep],
                    (
                        contact_interval[0],
                        min(contact_interval[1], float(scene.observation_frames[0][0])),
                    ),
                )
                contact_ray = serve_reach_cylinder.image_ray(
                    np.asarray(contact_intersection["contact_camera"], float),
                    camera_geometry.radial_at_epoch(
                        camera_distortion_lookup,
                        contact_intersection["contact_camera_source_frame"],
                        [int(contact_intersection["contact_camera_source_frame"])],
                    ),
                    np.asarray(contact_intersection["contact_pixel_native"], float),
                )
                # The hypotheses scale their feet-to-contact prior by stature; with
                # explicitly unavailable athlete evidence they abstain, while the
                # location prior, toss intersection and regularization stay in place.
                generated_hypotheses = []
                if players[0]["stature_m"] is None:
                    contact_hypothesis_blocker = (
                        "abstained: serve contact hypotheses need the server's stature and "
                        "athlete evidence is explicitly unavailable"
                    )
                else:
                    generated_hypotheses = serve_reach_cylinder.contact_hypotheses(
                        contact_ray,
                        location_prior,
                        toss_feet,
                        float(players[0]["stature_m"]),
                        count=2,
                    )
                for hypothesis_index, hypothesis in enumerate(generated_hypotheses):
                    hypothesis["contact_epoch_frame"] = contact_intersection["contact_epoch_frame"]
                    try:
                        seeded, seed_receipt = initialization.outgoing_contact_seed(
                            scene,
                            initial,
                            hypothesis,
                            first_bounce_target=targets[0][0],
                        )
                    except (ValueError, KeyError, FloatingPointError, OverflowError) as error:
                        contact_hypothesis_restart_failures.append(
                            {
                                "hypothesis_index": hypothesis_index,
                                "phase": "outgoing_seed",
                                "contact_xyz_m": hypothesis.get("contact_xyz_m"),
                                "blocker": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue
                    hypothesis["seed_parameters"] = seeded.tolist()
                    hypothesis["outgoing_drag_fit"] = seed_receipt
                    ray_direction = np.asarray(contact_ray["per_metre_of_height"], float)
                    outgoing_direction = np.asarray(seed_receipt["outgoing_velocity_mps"], float)
                    cosine = abs(ray_direction @ outgoing_direction) / (
                        np.linalg.norm(ray_direction) * np.linalg.norm(outgoing_direction)
                    )
                    angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
                    hypothesis["intersection_geometry"] = {
                        "toss_ray_vs_backward_serve_path_angle_degrees": angle,
                        "depth_condition": (
                            "well_conditioned"
                            if angle >= 20.0
                            else "weak"
                            if angle >= 8.0
                            else "ill_conditioned"
                        ),
                        "ray_depth_sigma_m": hypothesis["ray_parameter_sigma"],
                        "image_intersection_separation_px": contact_intersection[
                            "intersection_separation_px"
                        ],
                    }
                    contact_hypotheses.append(hypothesis)
                same_player_regularization = serve_reach_cylinder.same_player_regularization(
                    contact_history,
                    players[0].get("player"),
                    players[0]["side"],
                    None if args.serve_number_uncertain else args.serve_number,
                    location_prior,
                )
            except (ValueError, KeyError, IndexError, FloatingPointError, OverflowError) as error:
                contact_hypotheses = []
                contact_intersection = None
                contact_hypothesis_blocker = f"{type(error).__name__}: {error}"
        epoch_prior = (
            None
            if args.serve_contact_epoch == "off"
            else first_contact_epoch_prior(
                contact_events[0],
                scene,
                toss_observations,
                initial_frame=(
                    None
                    if contact_intersection is None
                    else contact_intersection["contact_epoch_frame"]
                ),
                stay_inside_labeled_bracket=args.serve_contact_hypotheses == "on",
            )
        )
        cylinder = None
        ray_cut = None
        # An explicitly requested hard actor cut still requires its measured root.
        # Missing-position abstention changes soft evidence, not this hard policy.
        if args.serve_reach_cut == "on":
            cylinder = serve_reach_cylinder.build(players[0], serve_reach_config)
            serve_frame = int(np.ceil(float(scene.contact_frames[0])))
            if serve_frame in camera_lookup:
                ray_cut = serve_reach_cylinder.ray_cut(
                    serve_reach_cylinder.image_ray(
                        camera_lookup[serve_frame],
                        camera_geometry.radial_at_epoch(
                            camera_distortion_lookup, serve_frame, [serve_frame]
                        ),
                        serve_contact_pixel,
                    ),
                    cylinder,
                    depths,
                )
                ray_cut["native_frame"] = serve_frame
                ray_cut["contact_pixel_native"] = serve_contact_pixel.tolist()

        # Experimental arms.  Both default to the promoted reference behaviour;
        # with the flags off `refine_arm` is empty and the solve is unchanged.
        anchor_targets = None
        closing_anchor_receipt = None
        if args.anchor_residuals == "present":
            # An original-contact prefix ends at a racket strike, so its final
            # endpoint gets the same soft horizontal tie the launching contacts
            # already carry.  A supplied-end scene keeps exactly its old plan,
            # and a closing actor whose court position is explicitly absent
            # abstains from the tie instead of reading a missing root.
            closing_anchor = (
                exposure.closing_contact_anchor(
                    scene,
                    right_contact_player,
                    player_sigma_m=args.anchor_player_sigma_m,
                )
                if contact_ending
                else None
            )
            # ``anchor_plan`` already accepts ``None`` where an arm has no player
            # row; an absent position simply omits that flight's root anchor.
            anchor_targets = exposure.anchor_plan(
                fit_anchor_targets,
                exposure.launch_contact_targets(players),
                player_sigma_m=args.anchor_player_sigma_m,
                closing_contact=closing_anchor,
            )
            if contact_ending:
                closing_anchor_receipt = (
                    {
                        "status": "untied",
                        "reason": closing_anchor_abstention(right_contact_player),
                    }
                    if closing_anchor is None
                    else {"status": "tied", **closing_anchor["source"]}
                )
        refine_arm = {}
        if args.refine_inequalities == "out":
            refine_arm["inequality_constraints"] = False
        elif args.refine_inequalities == "terminal_only":
            refine_arm["directional_inequalities"] = False
        if anchor_targets is not None:
            refine_arm["anchor_targets"] = anchor_targets
        if args.shooting_parameterization == "shared_contacts":
            refine_arm["shared_contact_states"] = True
            if args.local_flight_refits == "on":
                refine_arm["local_flight_refits"] = True
        if args.bounce_bracket_frames != 1.0:
            refine_arm["bounce_bracket_frames"] = float(args.bounce_bracket_frames)
        if terminal_rebound_frames is not None:
            refine_arm["terminal_rebound_frames"] = terminal_rebound_frames
        final_anchor = (
            final_bounce_anchor_plan(scene, bounces, fit_anchor_targets)
            if args.final_bounce_anchor == "on"
            else None
        )
        if final_anchor is not None and final_anchor.get("epoch_priors"):
            refine_arm["bounce_epoch_priors"] = final_anchor["epoch_priors"]
        final_anchor_seed = None if final_anchor is None else final_anchor.get("seed")
        reference_refine_options = (
            {}
            if terminal_rebound_frames is None
            else {"terminal_rebound_frames": terminal_rebound_frames}
        )
        # The anchor restarts need the same explicit witnesses the gates use:
        # the striker's automatic court position one contact ahead, and the
        # supplied bounce ground rays with their projection-derived sigma.
        anchor_contact_targets = [
            None
            if index == len(players) - 1 or player_position.root_xy(players[index + 1]) is None
            else np.r_[players[index + 1]["court_centre_xy_m"], ANCHOR_CONTACT_HEIGHT_M]
            for index in range(len(players))
        ]
        closing_root = (
            None if right_contact_player is None else player_position.root_xy(right_contact_player)
        )
        if closing_root is not None:
            anchor_contact_targets[-1] = np.r_[closing_root, ANCHOR_CONTACT_HEIGHT_M]
        anchor_bounce_targets = None if anchor_targets is None else anchor_targets["bounce"]
        # The server's contact ray meets each depth plane at a different point,
        # so the anchor is resolved per depth branch and cached per branch.
        serve_anchors: dict[float, tuple[np.ndarray, dict]] = {}

        native_candidate_options = None
        native_availability = {"status": "disabled", "reason": "candidate mode off"}
        if args.observation_net_seed == "candidate":
            native_scene, native_availability = native_seed_source(
                scene, heldout, bounces, axes, labels["events"]["records"], targets
            )
            if native_availability["status"] == "supported":
                native_candidate_options = dict(
                    scene=native_scene,
                    events=labels["events"]["records"],
                    speed_scale_mps=args.net_seed_speed_scale_mps,
                    candidate_family=True,
                    ground_targets=targets,
                    reserved_check_frames=heldout.observation_frames[-1],
                )

        defer_final_net = False
        if args.observation_net_seed == "on" and scene.terminal_net_tail is None:
            _, availability = native_seed_source(
                scene, heldout, bounces, axes, labels["events"]["records"], targets
            )
            defer_final_net = availability["status"] == "supported"

        def anchor_seed(
            depth: float,
            seed: np.ndarray,
            flat_spin: bool,
            terminal_ground_anchor: dict | None = None,
        ):
            if depth not in serve_anchors:
                contact_frame = int(np.ceil(float(scene.contact_frames[0])))
                if contact_frame not in camera_lookup:
                    contact_frame = min(camera_lookup, key=lambda f: (abs(f - contact_frame), f))
                serve_anchors[depth] = initialization.serve_contact_anchor(
                    camera_lookup[contact_frame],
                    contact_association_pixel(
                        contact_events[0],
                        label_lookup,
                        observation_fallback=observation_fallback,
                        fallback_receipt=fallback_receipt,
                    ),
                    None if nonserve_origin else players[0]["stature_m"],
                    depth,
                    radial=camera_geometry.radial_at_epoch(
                        camera_distortion_lookup, contact_frame, [contact_frame]
                    ),
                )
            anchored = serve_anchors[depth][0]
            return initialization.anchor_connected_seed(
                scene,
                seed,
                axes,
                bounces,
                depth,
                first_contact_xyz_m=anchored,
                contact_targets_xyz_m=anchor_contact_targets,
                bounce_targets=(
                    anchor_bounce_targets
                    if anchor_bounce_targets is not None
                    else [[None] * len(group) for group in bounces]
                ),
                flat_spin=flat_spin,
                recover_terminal_net_seed=args.terminal_net_seed == "on",
                defer_final_net_to_initializer=defer_final_net,
                exposure_duration=exposure_duration,
                **(
                    {}
                    if terminal_ground_anchor is None
                    else {"terminal_ground_anchor": terminal_ground_anchor}
                ),
            )

        def solve_at(
            seed: np.ndarray,
            depth: float,
            iterations: int,
            slice_: dict | None,
            *,
            solver_options: dict | None = None,
            free_first_contact: bool = False,
            seed_fit: dict | None = None,
            native_seed_options: dict | None = None,
            native_seed_bound: dict | None = None,
        ) -> tuple:
            """One solve at a pinned depth, optionally cut by the reach cylinder."""

            def intersect(first, second):
                if first is None:
                    return second
                if second is None:
                    return first
                interval = [max(first[0], second[0]), min(first[1], second[1])]
                if interval[0] > interval[1]:
                    raise ValueError("serve-contact parameter bounds do not intersect")
                return interval

            continuous_contact_bounds = toss_contact_bounds or (
                location_prior_bounds if args.serve_start_prior == "on" else None
            )
            if free_first_contact or nonserve_origin:
                contact_arguments = (
                    {"first_contact_depth_cue": origin_depth_cue} if is_rally else {}
                )
            elif continuous_contact_bounds is None:
                contact_arguments = {
                    "first_contact_y_m": depth,
                    "first_contact_x_bounds_m": None if slice_ is None else slice_["x_interval_m"],
                    "first_contact_z_bounds_m": None if slice_ is None else slice_["z_interval_m"],
                }
            else:
                contact_target = (
                    (
                        toss_contact_estimate["contact_xyz_m"],
                        toss_contact_estimate["contact_sigma_m"],
                    )
                    if toss_contact_bounds is not None
                    else (
                        location_prior_bounds["target_xyz_m"],
                        location_prior_bounds["sigma_m"],
                    )
                )
                contact_arguments = {
                    "first_contact_x_bounds_m": intersect(
                        continuous_contact_bounds["x_interval_m"],
                        None if slice_ is None else slice_["x_interval_m"],
                    ),
                    "first_contact_y_bounds_m": continuous_contact_bounds["y_interval_m"],
                    "first_contact_z_bounds_m": intersect(
                        continuous_contact_bounds["z_interval_m"],
                        None if slice_ is None else slice_["z_interval_m"],
                    ),
                    "first_contact_target": contact_target,
                }
            active_options = refine_arm if solver_options is None else solver_options
            solver_scene = replace(
                scene,
                parameterization=(
                    "shared_contact_states"
                    if active_options.get("shared_contact_states", False)
                    else "single_shooting"
                ),
            )
            seed_options = {}
            if args.observation_net_seed == "on":
                seed_scene, availability = native_seed_source(
                    solver_scene, heldout, bounces, axes, labels["events"]["records"], targets
                )
                if (
                    solver_scene.terminal_net_tail is not None
                    or availability["status"] == "supported"
                ):
                    seed_options["observation_net_seed"] = dict(
                        scene=seed_scene,
                        events=labels["events"]["records"],
                        ground_targets=targets,
                        speed_scale_mps=args.net_seed_speed_scale_mps,
                        reserved_check_frames=heldout.observation_frames[-1],
                    )
            if args.observation_net_seed == "candidate":
                # A coarse/cut/refined incumbent already owns its response.
                # Never estimate it again from a moved state or reset its law.
                bound = native_seed_bound
                if seed_fit is not None and (
                    seed_fit.get("net_response") is not None
                    or "net_response_initialization" in seed_fit
                ):
                    bound = {
                        name: seed_fit.get(name)
                        for name in ("net_response", "net_response_initialization")
                    }
                if bound is not None:
                    seed_options["net_seed_candidate"] = bound
                elif native_seed_options is not None:
                    seed_options["observation_net_seed"] = native_seed_options
            timing_plan = None
            seeded_epoch_prior = epoch_prior
            if args.interior_contact_epochs != "off":
                timing_plan = interior_contact_epochs.seed_plan(
                    solver_scene,
                    interior_contact_epochs.prepare(
                        solver_scene, events, native, exposure_duration
                    ),
                    seed_fit,
                )
                if seed_fit is not None and epoch_prior is not None:
                    seeded_epoch_prior = dict(
                        epoch_prior,
                        initial_frame=seed_fit["first_contact_epoch_fit"]["fitted_frame"],
                    )
            from contextlib import nullcontext as _nullcontext
            from cv.pipeline import net_cord_response as _cord

            _evidence_bounds = None
            if args.net_cord_response == _cord.EVIDENCE_BOUND:
                from cv.experiments.connected_shooting.net_cord_evidence import bounds_from_scene

                _tips = {}
                _ball_records = []
                if isinstance(reference, dict):
                    _ball_records = reference.get("records") or []
                elif isinstance(labels, dict):
                    _ball_records = (labels.get("ball") or {}).get("records") or []
                for _record in _ball_records:
                    for _row in _record.get("frames") or []:
                        _streak = _row.get("streak") or {}
                        if _streak.get("status") == "paired":
                            _tips[int(_row["frame"])] = [
                                _streak["trailing"]["x1080"],
                                _streak["trailing"]["y1080"],
                            ]
                _evidence_bounds = bounds_from_scene(
                    solver_scene,
                    bounces,
                    seed,
                    _tips,
                    h_tol=args.net_cord_tape_band_m,
                )
            if args.net_cord_response == _cord.ADMISSIBLE_SET:
                _mode_cm = _cord.using_mode(_cord.ADMISSIBLE_SET, h_tol=args.net_cord_tape_band_m)
            elif _evidence_bounds:
                _mode_cm = _cord.using_evidence(_evidence_bounds, h_tol=args.net_cord_tape_band_m)
            else:
                _mode_cm = _nullcontext()
            with _mode_cm:
                solved = candidate_attempts.numerical_call(
                    "refine",
                    exposure.refine,
                    solver_scene,
                    seed,
                    bounces,
                    native,
                    axes,
                    exposure_duration,
                    iterations,
                    **contact_arguments,
                    first_contact_epoch_prior=seeded_epoch_prior,
                    interior_contact_epoch_plan=timing_plan,
                    termination_kind=termination_kind,
                    directional_event_frames=[row["frame"] for row in attempt["events"]],
                    directional_rms_limit_px=16.0,
                    **active_options,
                    **deadline_arguments,
                    **seed_options,
                )
                if args.net_cord_response == _cord.ADMISSIBLE_SET:
                    from cv.experiments.connected_shooting.admissible_net_response import (
                        search_from_chain,
                    )

                    bound = search_from_chain(
                        solver_scene, np.asarray(solved["parameters"], float), solver_scene.fps
                    )
                    if bound is not None:
                        solved = {**solved, "net_response": bound}
                elif _evidence_bounds:
                    from cv.experiments.connected_shooting.net_cord_evidence import (
                        attach_constrained_response,
                    )

                    solved = attach_constrained_response(solver_scene, solved, _evidence_bounds)
            if active_options.get("shared_contact_states", False) and (
                solved["shared_contact_states"]["maximum_endpoint_gap_m"]
                > exposure.SHARED_CONTACT_CERTIFICATE_M
            ):
                raise ValueError("shared contact endpoint exceeds the numerical certificate")
            fitted_scene = solver_scene
            fitted_heldout = replace(heldout, parameterization=solver_scene.parameterization)

            fitted_scene = interior_contact_epochs.fitted_scene(solver_scene, solved)
            fitted_heldout = interior_contact_epochs.fitted_scene(fitted_heldout, solved)
            from cv.experiments.connected_shooting import observation_net_seed

            with _mode_cm, observation_net_seed.response_context(solved):
                measurement = candidate_attempts.numerical_call(
                    "measure",
                    exposure.measure,
                    fitted_scene,
                    fitted_heldout,
                    bounces,
                    native,
                    np.asarray(solved["parameters"]),
                    axes,
                    exposure_duration,
                    [reference],
                    termination_kind=termination_kind,
                )
                measurement["fit"] = solved
                proof = candidate_attempts.numerical_call(
                    "candidate_evidence",
                    candidate_evidence,
                    measurement,
                    players,
                    targets,
                    (low, high),
                    events,
                    athlete_prior_mode=args.athlete_prior_mode,
                    serve_reach=cylinder,
                    serve_ending_kind=args.ending_kind,
                    enforce_serve_ending_consistency=args.serve_ending_consistency == "on",
                    same_player_serve_regularization=same_player_regularization,
                )
                if solved.get("first_contact_epoch_fit") is not None:
                    proof["first_contact_epoch_fit"] = solved["first_contact_epoch_fit"]
                if solved.get(interior_contact_epochs.FIELD) is not None:
                    proof[interior_contact_epochs.FIELD] = solved[interior_contact_epochs.FIELD]
                if active_options.get("shared_contact_states", False):
                    explicit = np.asarray(solved["parameters"], float)
                    solved["shared_contact_states"]["explicit_parameter_vector"] = explicit.tolist()
                    solved["shared_contact_states"]["exported_parameterization"] = (
                        "equivalent single-shooting vector for unchanged acceptance interface"
                    )
                    solved["parameters"] = candidate_attempts.numerical_call(
                        "shared_to_single",
                        model.shared_to_single_parameters,
                        solver_scene,
                        explicit,
                    ).tolist()
                if (
                    fitted_scene.terminal_net_tail is not None
                    or solved.get("net_response") is not None
                ):
                    from cv.experiments.connected_shooting import labeled_terminal_net_tail as tail

                    physical = model.chain(fitted_scene, np.asarray(solved["parameters"], float))
                    slacks = tail.original_domain_slacks_frames(fitted_scene, physical)
                    if solved.get("net_response") is not None:
                        tail.require_mesh_response(
                            fitted_scene, physical, include_supplied_final=True
                        )
                    measurement["input_event_domain"] = dict(
                        admissible=tail.domain_admissible(slacks),
                        slacks_frames=slacks.tolist(),
                        physical_replay=True,
                        selection_reads_acceptance=False,
                    )
            return measurement, proof

        def solve(
            depth: float,
            seed: np.ndarray,
            iterations: int,
            stage: str,
            *,
            solver_options: dict | None = None,
            free_first_contact: bool = False,
            seed_fit: dict | None = None,
            native_candidate: bool = False,
        ) -> dict:
            budget.check()
            began = time.monotonic()
            structured_seed = None
            native_bound = None
            if native_candidate and native_candidate_options is None:
                raise candidate_attempts.CandidateFailure(
                    "native seed family lacks source qualification"
                )
            if (
                args.serve_toss_constraint == "off"
                and stage == "coarse"
                and len(scene.pixels) >= 5
                and low <= depth <= high
            ):
                native_options = {}
                if args.observation_net_seed == "on":
                    native_scene, availability = native_seed_source(
                        scene, heldout, bounces, axes, labels["events"]["records"], targets
                    )
                    if scene.terminal_net_tail is not None or availability["status"] == "supported":
                        native_options["native_net_initialization"] = dict(
                            scene=native_scene,
                            events=labels["events"]["records"],
                            ground_targets=targets,
                            speed_scale_mps=args.net_seed_speed_scale_mps,
                            reserved_check_frames=heldout.observation_frames[-1],
                        )
                if native_candidate:
                    native_options["native_net_initialization"] = native_candidate_options
                seed, structured_seed = candidate_attempts.numerical_call(
                    "structured_seed",
                    structure_aware_connected_seed,
                    scene,
                    seed,
                    players,
                    axes,
                    bounces,
                    depth,
                    exposure_duration=exposure_duration,
                    **native_options,
                    **({"deadline_check": budget.check} if args.search_seconds is not None else {}),
                    # Source-only: the seed's own bounce witnesses (supplied bounce
                    # epochs and native ground rays), already built for the
                    # pinhole seed; never a saved fit or a selected winner.
                    **(
                        {"event_ground_targets": seed_targets}
                        if args.event_ground_seed == "on" and seed_targets is not None
                        else {}
                    ),
                )
            if native_candidate and structured_seed is not None:
                native_receipt = structured_seed.get("native_net_response_initialization")
                if native_receipt is None:
                    raise candidate_attempts.CandidateFailure(
                        "structured native seed omitted its response binding"
                    )
                native_bound = dict(
                    net_response=native_receipt["response"],
                    net_response_initialization=native_receipt,
                )
            stage_options = stage_refine_options(
                refine_arm if solver_options is None else solver_options, stage
            )
            # Explicit shared states are the second-stage fit.  Coarse branches
            # remain the promoted single-shooting search, both to preserve its
            # depth ranking and to avoid repeating the larger sparse solve for
            # every seed restart.  Local block refits likewise belong only to
            # the higher-budget refined branch.
            measurement, proof = solve_at(
                seed,
                depth,
                iterations,
                None,
                solver_options=stage_options,
                free_first_contact=free_first_contact,
                seed_fit=seed_fit,
                **(
                    {
                        "native_seed_options": native_candidate_options,
                        "native_seed_bound": native_bound,
                    }
                    if native_candidate
                    else {}
                ),
            )
            # The cut is a re-solve, not a re-seeded search.  A branch whose free
            # fit already puts the serve contact inside the cylinder is left
            # exactly as the reference arm left it, so the cut cannot move an
            # attempt that never needed it; only a branch that fails the cylinder
            # is solved again with the bound active, seeded from its own free
            # optimum.  Projecting a seed onto the bound before the free solve was
            # measured and rejected: it walked good branches into a worse local
            # optimum (USO2025 pt0003 coarse -0.5 m went 3.8 -> 142 px).
            #
            # Only the refined stage is cut.  Coarse fitting is search, not
            # acceptance, and it seeds the refined solve: cutting a coarse branch
            # moves the refined seed of branches whose own free fit satisfies the
            # cylinder, which was measured to change three accepted attempts'
            # selected depth and family width (miami2025 pt0001, rg2024f pt0002,
            # uso2020f pt0001) without gaining anything.  Cutting the refined
            # stage alone reproduces every branch the cut does not bind.
            if (
                cylinder is not None
                and stage == "refined"
                and not proof["checks"]["serve_contact_inside_reach_cylinder"]
                and budget.exhausted
            ):
                # The deadline ended inside this branch's free solve and a retained
                # iterate came back; a cut re-solve cannot start.  Keep the free
                # fit explicitly uncut so the unchanged cylinder gate judges it.
                proof["serve_reach_resolved_under_cut"] = False
                proof["serve_reach_cut_stage"] = stage
                proof["serve_reach_cut_skipped"] = "search_deadline_exhausted"
            elif (
                cylinder is not None
                and stage == "refined"
                and not proof["checks"]["serve_contact_inside_reach_cylinder"]
            ):
                free_measurement, free_proof = measurement, proof
                slice_ = candidate_attempts.numerical_call(
                    "serve_reach_slice", cylinder.depth_slice, depth
                )
                # The cut re-solve is seeded twice and the search keeps the better
                # by its own input-only rank score.  Seeding only from the free
                # optimum was measured and is not enough: a serve the free fit put
                # at the toss apex drags a long point's whole velocity chain with
                # it, and clipping that optimum leaves the solver in the same bad
                # basin (rg2024f pt0001 lost 8 accepted flights, beijing2024f
                # pt0003 four).  The branch's own pre-refine seed is the second
                # start.  This is search, not acceptance: both starts are scored
                # by the same published checks.
                trials = []
                for name, start, start_fit in (
                    (
                        "free_refined_optimum",
                        free_measurement["fit"]["parameters"],
                        free_measurement["fit"],
                    ),
                    ("branch_seed", seed, seed_fit),
                ):
                    trial = trials_log.run(
                        dict(stage=stage, depth_m=float(depth), seed=name, phase="serve_reach_cut"),
                        solve_at,
                        np.asarray(start, float),
                        depth,
                        iterations,
                        slice_,
                        solver_options=stage_options,
                        seed_fit=start_fit,
                    )
                    if trial is not None:
                        trial_measurement, trial_proof = trial
                        trials.append((name, trial_measurement, trial_proof))
                if not trials:
                    raise candidate_attempts.CandidateFailure(
                        "all serve-reach cut starts failed numerically"
                    )
                chosen, measurement, proof = min(
                    trials,
                    key=lambda row: input_domain_rank(dict(measurement=row[1], evidence=row[2])),
                )
                proof["serve_reach_depth_slice"] = slice_
                proof["serve_reach_resolved_under_cut"] = True
                proof["serve_reach_cut_seed"] = chosen
                proof["serve_reach_cut_seed_scores"] = {
                    row[0]: row[2]["input_only_rank_score"] for row in trials
                }
                proof["serve_reach_free_fit"] = {
                    "contact_xyz_m": free_proof["contact_xyz_m"][0],
                    "serve_reach_cylinder": free_proof["serve_reach_cylinder"],
                    "checks": free_proof["checks"],
                    "rms_px": free_measurement["rms_px"],
                    "input_only_rank_score": free_proof["input_only_rank_score"],
                }
            elif cylinder is not None:
                proof["serve_reach_resolved_under_cut"] = False
                proof["serve_reach_cut_stage"] = stage
            if structured_seed is not None:
                measurement["fit"]["structure_aware_initialization"] = structured_seed
            fitted_contact_frame = float(
                proof.get("first_contact_epoch_fit", {}).get(
                    "fitted_frame", scene.contact_frames[0]
                )
            )
            if nonserve_origin:
                proof["toss_witness"] = dict(
                    status="not_applicable",
                    reason=origin_reason,
                    selector_penalty=0.0,
                    native_projection=[],
                )
                proof["serve_contact_prior"] = dict(
                    status="not_applicable", reason=origin_reason, selector_penalty=0.0
                )
            else:
                fitted_toss_observations = candidate_attempts.numerical_call(
                    "toss_epoch",
                    toss_witness.observations_at_contact_epoch,
                    toss_observations,
                    fitted_contact_frame,
                )
                proof["toss_witness"] = candidate_attempts.numerical_call(
                    "toss_witness",
                    toss_witness.fit,
                    np.asarray(proof["contact_xyz_m"][0]),
                    fitted_toss_observations,
                    toss_feet,
                    float(attempt["fps"]),
                )
                proof["serve_contact_prior"] = candidate_attempts.numerical_call(
                    "serve_contact_prior",
                    toss_witness.serve_contact_prior,
                    float(proof["contact_xyz_m"][0][1]),
                    toss_feet["side"],
                    args.serve_number,
                )
            candidate = dict(
                depth_hypothesis_m=float(depth),
                stage=stage,
                measurement=measurement,
                evidence=proof,
                wall_seconds=time.monotonic() - began,
            )
            proof["survived_before_serve_speed"] = proof["survived"]
            if not nonserve_origin:
                candidate_attempts.numerical_call(
                    "serve_speed_witness",
                    serve_speed_witness.add_to_candidate,
                    candidate,
                    speed_reading,
                )
            candidate_attempts.validate_completed(candidate, len(scene.pixels))
            usage = native_seed_check_pixels.fit_usage(measurement["fit"])
            if usage is not None:
                native_seed_usages.append(usage)
            candidate["topology_name"] = topology_name
            if membership_active:
                candidate["membership"] = membership
            if args.observation_net_seed == "candidate":
                candidate["native_seed_family"] = (
                    "native_net"
                    if measurement["fit"].get("net_response") is not None
                    else "ordinary"
                )
            budget.record(candidate, len(scene.pixels))
            return candidate

        def fit(
            depth: float,
            seed: np.ndarray,
            iterations: int,
            stage: str,
            *,
            free_first_contact: bool = False,
            seed_fit: dict | None = None,
        ) -> dict:
            """Solve one depth branch, optionally from three restarts.

            With `--seed-restarts off` this is exactly the reference single
            solve from the pinhole seed.  With `three` the same branch is also
            started from the anchored walk and from the anchored walk with flat
            spin, and the best of the three by the search's own input-only rank
            score is kept.  Selection reads no withheld pixel and no gate that
            the reference arm does not already read.
            """
            extra_native = (
                args.observation_net_seed == "candidate" and native_candidate_options is not None
            )
            if (
                free_first_contact
                or stage != "coarse"
                or (args.seed_restarts == "off" and not extra_native)
            ):
                return solve(
                    depth,
                    seed,
                    iterations,
                    stage,
                    free_first_contact=free_first_contact,
                    seed_fit=seed_fit,
                )
            attempts_made = []
            families = (
                native_seed_families(
                    args.seed_restarts != "off" and args.serve_contact_hypotheses != "on",
                    native_availability,
                )
                if args.observation_net_seed == "candidate"
                else (
                    ("pinhole", None, False),
                    ("anchor", False, False),
                    ("anchor_flat_spin", True, False),
                )
            )
            if final_anchor_seed is not None:
                families = (*families, ("final_bounce_anchor", False, False))
            for label, flat, is_native in families:
                failure_count = len(trials_log.failures)
                seed_receipt = None
                started = seed
                if flat is not None:
                    prepared = trials_log.run(
                        dict(stage=stage, depth_m=float(depth), seed=label),
                        candidate_attempts.numerical_call,
                        "anchor_seed",
                        anchor_seed,
                        depth,
                        seed,
                        flat,
                        *((final_anchor_seed,) if label == "final_bounce_anchor" else ()),
                    )
                    if prepared is None:
                        attempts_made.append(
                            (label, {"blocker": trials_log.failures[-1]["reason"]}, None)
                        )
                        continue
                    started, seed_receipt = prepared
                row = trials_log.run(
                    dict(stage=stage, depth_m=float(depth), seed=label),
                    solve,
                    depth,
                    started,
                    iterations,
                    stage,
                    **({"native_candidate": True} if is_native else {}),
                )
                if row is None:
                    assert len(trials_log.failures) > failure_count
                    attempts_made.append(
                        (label, {"blocker": trials_log.failures[-1]["reason"]}, None)
                    )
                    continue
                if seed_receipt is not None:
                    row["measurement"]["fit"]["anchor_initialization"] = seed_receipt
                attempts_made.append((label, seed_receipt, row))
            scored = [(label, row) for label, _, row in attempts_made if row is not None]
            if not scored:
                raise candidate_attempts.CandidateFailure(
                    "all configured seeds failed numerically", aggregate=True
                )
            best_label, best = min(scored, key=lambda item: input_domain_rank(item[1]))
            best["seed_restarts"] = {
                **(
                    {
                        "native_source_qualification": native_availability,
                        "configured_families": [row[0] for row in families],
                        "budget": "existing cumulative search budget",
                    }
                    if args.observation_net_seed == "candidate"
                    else {}
                ),
                "selected": best_label,
                "selection_rule": (
                    "lowest all-native objective-input rank; no independent pixel check"
                    if args.observation_partition == "all_native"
                    else "lowest input-only rank score; no withheld pixel is read"
                ),
                "serve_contact_anchor": serve_anchors.get(depth, (None, None))[1],
                "restarts": [
                    {
                        "seed": label,
                        "blocked": receipt.get("blocker") if row is None else None,
                        "training_rms_px": None
                        if row is None
                        else row["measurement"]["rms_px"]["training"],
                        "input_only_rank_score": None
                        if row is None
                        else row["evidence"]["input_only_rank_score"],
                    }
                    for label, receipt, row in attempts_made
                ],
            }
            if "input_event_domain" in best["measurement"]:
                best["seed_restarts"]["selection_rule"] = (
                    "original input event domain eligibility, then lowest input-only rank score"
                )
            return best

        budget.context = dict(
            **({"membership": membership} if membership_active else {}),
            events=events,
            supplied_events=attempt["events"],
            modeled_event_scope=topology_scope,
            attempt_id=attempt["attempt_id"],
            clip=attempt["point_clip"],
            configuration=dict(
                **owner_approved_gate_receipt,
                **net_cord_response_receipt,
                **(
                    {
                        contact_role.FIELD: role_receipt,
                        "first_contact_parameterization": "ordinary_contact_free_xyz",
                        "serve_evidence_applicable": False,
                        **({rally_cue.FIELD: origin_depth_cue} if is_rally else {}),
                    }
                    if nonserve_origin
                    else {}
                ),
                **(
                    {
                        "serve_location_prior_model": provenance.file_record(
                            args.serve_location_prior
                        )
                    }
                    if args.serve_location_prior is not None
                    else {}
                ),
                **(
                    {"native_seed_availability": native_availability}
                    if args.observation_net_seed == "candidate"
                    else {}
                ),
                surface=args.surface,
                exposure_duration_frames=exposure_duration,
                ball_observation_operator=operator,
                termination_kind=termination_kind,
                terminal_net_tail=net_tail_contract,
                observed_horizon_tail=horizon_tail_contract,
                ground_settling=args.ground_settling,
                **(
                    {"passive_bounce_response": args.passive_bounce_response}
                    if args.passive_bounce_response != "off"
                    else {}
                ),
                **(
                    {
                        "observation_net_seed": args.observation_net_seed,
                        "net_seed_speed_scale_mps": args.net_seed_speed_scale_mps,
                    }
                    if args.observation_net_seed != "off"
                    else {}
                ),
                end_frame=float(end_frame),
                terminal_rebound=terminal_rebound_receipt,
                terminal_net_seed=args.terminal_net_seed,
                observation_partition=args.observation_partition,
                player_statures_m=statures,
                athlete_prior_mode=args.athlete_prior_mode,
                **athlete_evidence_configuration(args.athlete_evidence, athlete_evidence),
                **athlete_priors.root_reach_configuration(args.athlete_root_reach_loss),
                contact_xy_search_envelope_m=CONTACT_XY_ENVELOPE_M,
                bounce_witness_mode=args.bounce_witness_mode,
                **(
                    {"fit_ground_witness": args.fit_ground_witness}
                    if args.fit_ground_witness != "off"
                    else {}
                ),
                bounce_witness_observation_policy=scene.bounce_witness_observation_policy,
                serve_reach_cut=args.serve_reach_cut,
                **missing_player_position_configuration(
                    args.missing_player_position, players, right_contact_player=right_contact_player
                ),
                **serve_association_kwargs,
                serve_reach_margin_statures=args.serve_reach_margin_statures,
                serve_reach_radius_statures=args.serve_reach_radius_statures,
                labeled_ending_kind=args.ending_kind,
            ),
            initialization=initialization_evidence,
            bounce_ground_ray_targets=targets,
            inputs=[
                provenance.file_record(p)
                for p in (args.labels, args.packet, args.cameras, args.pose_csv)
            ],
            code=provenance.git_record(paths.REPO_ROOT),
            implementation_files=[
                provenance.file_record(Path(module.__file__))
                for module in (search_budget, exposure, initialization, model)
                + ((fit_ground_witness_center,) if fit_witness_receipt is not None else ())
            ],
        )
        trials_log = candidate_attempts.Attempts()
        coarse = []
        refined = []
        shortlist = []
        shared_stage_blocker = None
        reference_coarse = {}
        depth_seed_receipts = []
        try:
            if shared_stage_reference is None:
                for depth in depths:
                    branch_initial = initial
                    if args.depth_conditioned_seed == "on" and not nonserve_origin:
                        from cv.experiments.connected_shooting import depth_conditioned_seed

                        budget.check()
                        branch_initial, seed_receipt = depth_conditioned_seed.condition(
                            scene,
                            initial,
                            bounces,
                            float(depth),
                            bounce_ground_targets=seed_targets,
                        )
                        depth_seed_receipts.append(seed_receipt)
                    elif args.depth_conditioned_seed == "on":
                        depth_seed_receipts.append(
                            dict(
                                status="not_applicable",
                                reason="original rally contact",
                                depth_m=float(depth),
                            )
                        )
                    # Replace the two generic anchor restarts with the witness restarts
                    # in this arm.  The ordinary active seed and explicit reference
                    # restart remain, while the branch budget stays comparable.
                    row = trials_log.run(
                        dict(stage="coarse", depth_m=float(depth), seed="active"),
                        solve
                        if args.serve_contact_hypotheses == "on"
                        and args.observation_net_seed != "candidate"
                        else fit,
                        float(depth),
                        branch_initial,
                        args.coarse_iterations,
                        "coarse",
                    )
                    if args.reference_restart == "on":
                        reference = trials_log.run(
                            dict(stage="coarse", depth_m=float(depth), seed="reference"),
                            solve,
                            float(depth),
                            branch_initial,
                            REFERENCE_COARSE_ITERATIONS,
                            "coarse",
                            solver_options=reference_refine_options,
                        )
                        if reference is not None:
                            reference_coarse[float(depth)] = reference
                            row = select_reference_restart(row, reference)
                    if row is None:
                        continue
                    coarse.append(row)
                    print(
                        depth,
                        row["evidence"]["survived"],
                        row["measurement"]["rms_px"],
                        row["evidence"]["death_reasons"],
                        flush=True,
                    )
                for hypothesis_index, hypothesis in enumerate(contact_hypotheses):
                    try:
                        row = solve(
                            float(hypothesis["contact_xyz_m"][1]),
                            np.asarray(hypothesis["seed_parameters"], float),
                            args.coarse_iterations,
                            "coarse",
                            free_first_contact=True,
                        )
                    except (
                        ValueError,
                        KeyError,
                        IndexError,
                        FloatingPointError,
                        OverflowError,
                    ) as error:
                        contact_hypothesis_restart_failures.append(
                            {
                                "hypothesis_index": hypothesis_index,
                                "phase": "coarse_solve",
                                "contact_xyz_m": hypothesis.get("contact_xyz_m"),
                                "blocker": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue
                    row["contact_hypothesis_restart"] = {
                        key: value for key, value in hypothesis.items() if key != "seed_parameters"
                    }
                    coarse.append(row)
                shortlist = list(coarse)
            else:
                coarse, promoted_rank_one, promoted_family = shared_stage_reference
                shortlist = [promoted_rank_one]
                depths = np.asarray([promoted_rank_one["depth_hypothesis_m"]], float)
            coarse_survivors = [row for row in coarse if row["evidence"]["survived"]]
            # Coarse fitting is search, not acceptance, so every branch that survives the
            # fixed depth beam still receives the higher-budget solve and every final gate
            # again.  Reusing a coarse survivor unrefined was measured and rejected: it
            # keeps the family sets but changes the selected branch's fitted parameters,
            # which are published numbers.  Branches the beam excluded cannot change their
            # pinned first-contact Y at any budget and are recorded above instead.
            refined = []
            shared_stage_blocker = None
            for row in shortlist:
                depth = row["depth_hypothesis_m"]
                free_first_contact = "contact_hypothesis_restart" in row
                try:
                    candidate = fit(
                        depth,
                        np.asarray(row["measurement"]["fit"]["parameters"]),
                        args.refine_iterations,
                        "refined",
                        free_first_contact=free_first_contact,
                        seed_fit=row["measurement"]["fit"],
                    )
                except candidate_attempts.CandidateFailure as error:
                    trials_log.failures.append(
                        dict(
                            stage="refined", depth_m=float(depth), seed="active", reason=str(error)
                        )
                    )
                    if free_first_contact:
                        contact_hypothesis_restart_failures.append(
                            {
                                "phase": "refined_solve",
                                "contact_xyz_m": row["evidence"].get("contact_xyz_m", [None])[0],
                                "blocker": f"{type(error).__name__}: {error}",
                            }
                        )
                    candidate = None
                    if shared_stage_reference is not None:
                        shared_stage_blocker = f"{type(error).__name__}: {error}"
                        for fallback in promoted_family:
                            fallback["second_stage_restart"] = {
                                "source": "explicit promoted terminal-only refined family",
                                "unchanged_fallback": True,
                                "shared_fit_blocker": shared_stage_blocker,
                            }
                        refined.extend(promoted_family)
                        continue
                except (
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError,
                    OverflowError,
                ) as error:
                    if free_first_contact:
                        contact_hypothesis_restart_failures.append(
                            {
                                "phase": "refined_solve",
                                "contact_xyz_m": row["evidence"].get("contact_xyz_m", [None])[0],
                                "blocker": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue
                    if shared_stage_reference is None:
                        raise
                    shared_stage_blocker = f"{type(error).__name__}: {error}"
                    for fallback in promoted_family:
                        fallback["second_stage_restart"] = {
                            "source": "explicit promoted terminal-only refined family",
                            "unchanged_fallback": True,
                            "shared_fit_blocker": shared_stage_blocker,
                        }
                    refined.extend(promoted_family)
                    continue
                if shared_stage_reference is not None:
                    candidate["second_stage_restart"] = {
                        "source": "explicit promoted terminal-only rank-one candidate",
                        "selection_rule": "retain both; unchanged gates rank surviving candidates",
                    }
                    for fallback in promoted_family:
                        fallback["second_stage_restart"] = {
                            "source": "explicit promoted terminal-only refined family",
                            "unchanged_fallback": True,
                        }
                    refined.extend([candidate, *promoted_family])
                    continue
                if (
                    args.reference_restart == "on"
                    and not free_first_contact
                    and depth in reference_coarse
                ):
                    reference = trials_log.run(
                        dict(stage="refined", depth_m=float(depth), seed="reference"),
                        solve,
                        depth,
                        np.asarray(reference_coarse[depth]["measurement"]["fit"]["parameters"]),
                        args.refine_iterations,
                        "refined",
                        solver_options=reference_refine_options,
                        seed_fit=reference_coarse[depth]["measurement"]["fit"],
                    )
                    if reference is not None:
                        candidate = select_reference_restart(candidate, reference)
                from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                    retain_input_domain_incumbent,
                )

                candidate = retain_input_domain_incumbent(candidate, row)
                if candidate is None:
                    continue
                if free_first_contact:
                    candidate["contact_hypothesis_restart"] = row["contact_hypothesis_restart"]
                refined.append(candidate)
        except search_budget.SearchDeadline:
            budget.persist()
            failure_scope = f"{membership}:{topology_name}" if membership_active else topology_name
            numerical_failures_by_topology[failure_scope] = trials_log.failures
            search_budget.atomic_json(
                args.output / "numerical_candidate_failures.json",
                {"topologies": numerical_failures_by_topology},
            )
            # Every entry completed a full solve, measurement and input score.
            # No partially updated optimizer vector and no acceptance winner.
            completed = candidate_attempts.completed_at_deadline(
                budget.completed, topology_name, membership=membership
            )
            coarse = [row for row in completed if row["stage"] == "coarse"]
            refined = [row for row in completed if row["stage"] == "refined"]
            shortlist = list(coarse)
            coarse_survivors = [row for row in coarse if row["evidence"]["survived"]]
        failure_scope = f"{membership}:{topology_name}" if membership_active else topology_name
        numerical_failures_by_topology[failure_scope] = trials_log.failures
        search_budget.atomic_json(
            args.output / "numerical_candidate_failures.json",
            {"topologies": numerical_failures_by_topology},
        )
        if not refined and not coarse:
            raise candidate_attempts.CandidateFailure(
                f"no completed candidate after {len(trials_log.failures)} numerical failures"
            )
        eligible = [row for row in refined if row["evidence"]["survived"]]
        selected = min(
            eligible,
            key=lambda row: (row["evidence"]["input_only_rank_score"], row["depth_hypothesis_m"]),
            default=None,
        )
        diagnostic = min(
            refined or coarse,
            key=lambda row: (row["evidence"]["input_only_rank_score"], row["depth_hypothesis_m"]),
        )
        family_required_checks = (
            ("connected_input_physics",)
            if nonserve_origin
            else ("connected_input_physics", "serve_depth_plausible")
            if args.athlete_prior_mode == "stature_pose_soft"
            else ("connected_input_physics", "serve_depth_plausible", "serve_height_plausible")
        )
        families = {
            "baseline": search_reporting.family(
                refined,
                arm="baseline_connected_legal",
                eligible=lambda row: row["evidence"]["survived"],
                rank_score=lambda row: row["evidence"]["input_only_rank_score"],
                required_checks=family_required_checks,
            ),
            "toss_witness": search_reporting.family(
                refined,
                arm="real_toss_witness",
                eligible=lambda row: row["evidence"]["survived"],
                rank_score=lambda row: (
                    row["evidence"]["input_only_rank_score"]
                    + row["evidence"]["toss_witness"]["selector_penalty"]
                ),
                required_checks=family_required_checks,
            ),
            "serve_contact_prior": search_reporting.family(
                refined,
                arm=f"soft_serve_contact_prior_serve_{args.serve_number}",
                eligible=lambda row: row["evidence"]["survived"],
                rank_score=lambda row: (
                    row["evidence"]["input_only_rank_score"]
                    + row["evidence"]["serve_contact_prior"]["selector_penalty"]
                ),
                required_checks=family_required_checks,
            ),
            "combined_toss_and_serve_prior": search_reporting.family(
                refined,
                arm=f"real_toss_plus_soft_serve_{args.serve_number}_prior",
                eligible=lambda row: row["evidence"]["survived"],
                rank_score=lambda row: (
                    row["evidence"]["input_only_rank_score"]
                    + row["evidence"]["toss_witness"]["selector_penalty"]
                    + row["evidence"]["serve_contact_prior"]["selector_penalty"]
                ),
                required_checks=family_required_checks,
            ),
        }
        family_widths = {
            "all_refined_depths": contact_family_widths(refined),
            "surviving_refined_depths": contact_family_widths(
                [row for row in refined if row["evidence"]["survived"]]
            ),
            "interpretation": (
                "maximum pairwise XYZ spread at each exact flight boundary; a wide family is "
                "reported as depth ambiguity, never treated as continuity slack"
            ),
        }
        selected_arms = {}
        for name, family in families.items():
            depth = family["selected_depth_y_m"]
            rows_at_depth = [row for row in refined if row["depth_hypothesis_m"] == depth]
            arm_score = {
                "baseline": lambda row: row["evidence"]["input_only_rank_score"],
                "toss_witness": lambda row: (
                    row["evidence"]["input_only_rank_score"]
                    + row["evidence"]["toss_witness"]["selector_penalty"]
                ),
                "serve_contact_prior": lambda row: (
                    row["evidence"]["input_only_rank_score"]
                    + row["evidence"]["serve_contact_prior"]["selector_penalty"]
                ),
                "combined_toss_and_serve_prior": lambda row: (
                    row["evidence"]["input_only_rank_score"]
                    + row["evidence"]["toss_witness"]["selector_penalty"]
                    + row["evidence"]["serve_contact_prior"]["selector_penalty"]
                ),
            }[name]
            selected_arms[name] = (
                min(
                    (row for row in rows_at_depth if row["evidence"]["survived"]),
                    key=arm_score,
                    default=None,
                )
                if shared_stage_reference is not None
                else next(iter(rows_at_depth), None)
            )
        if args.serve_contact_hypotheses == "on":
            # This arm's contract is exactly the ordinary input-only rank.  Do not
            # reintroduce the legacy toss/prior selector after the hypotheses have
            # already served their sole purpose as restarts.
            selected_arms = {name: selected for name in families}
        return dict(
            topology_name=topology_name,
            membership=membership,
            terminal_net_tail=net_tail_contract,
            observed_horizon_tail=horizon_tail_contract,
            modeled_event_scope=topology_scope,
            events=events,
            observations_outside_modeled_window=outside_window,
            bounces=bounces,
            end_frame=float(end_frame),
            termination_kind=termination_kind,
            terminal_bounce_count=len(terminal),
            terminal_rebound=terminal_rebound_receipt,
            scene=scene,
            heldout=heldout,
            native=native,
            players=players,
            serve_association=serve_association_kwargs,
            right_contact_player=right_contact_player,
            closing_contact_anchor=closing_anchor_receipt,
            final_bounce_anchor=final_anchor,
            original_rally_origin_depth_cue=origin_depth_cue,
            statures=statures,
            athlete_evidence=athlete_evidence,
            player_ledger=ledger_receipt,
            depth_bounds=(low, high),
            all_depths=all_depths,
            fitted_depths=depths,
            excluded_depths=excluded_depths,
            targets=targets,
            axis_receipt=axis_receipt,
            initialization_evidence=initialization_evidence,
            depth_seed_receipts=depth_seed_receipts,
            toss_observations=toss_observations,
            toss_server_feet=toss_feet,
            toss_contact_estimate=toss_contact_estimate,
            toss_contact_bounds=toss_contact_bounds,
            serve_location_prior=location_prior,
            serve_location_prior_bounds=location_prior_bounds,
            serve_contact_hypotheses=contact_hypotheses,
            serve_contact_intersection=contact_intersection,
            same_player_serve_regularization=same_player_regularization,
            serve_contact_history_receipt=contact_history_receipt,
            serve_contact_hypothesis_blocker=contact_hypothesis_blocker,
            serve_contact_hypothesis_restart_failures=contact_hypothesis_restart_failures,
            first_contact_epoch_prior=epoch_prior,
            serve_reach_cylinder=None if cylinder is None else cylinder.record(),
            serve_reach_ray_cut=ray_cut,
            coarse=coarse,
            coarse_survivors=coarse_survivors,
            shortlist=shortlist,
            refined=refined,
            selected=selected,
            selected_arms=selected_arms,
            families=families,
            family_widths=family_widths,
            numerical_candidate_failures=trials_log.failures,
            shared_state_second_stage_blocker=shared_stage_blocker,
            diagnostic=diagnostic,
        )

    def run_topology(
        events: list[dict],
        end_frame: float,
        topology_name: str,
        *,
        membership: str | None = None,
    ) -> dict:
        """Solve one branch under its own declared net scope.

        A membership that declares a source net absent narrows the shared
        bounded-net preparation helper for the whole of its solve, so no branch
        inherits a captured global net it does not own.
        """
        from cv.pipeline import s6_preparation_policy as preparation_policy

        with preparation_policy.declared_absent_nets(membership_state[membership]["absent_nets"]):
            return fit_topology(events, end_frame, topology_name, membership=membership)

    args.output.mkdir(parents=True)
    supplied_topology = [
        *attempt["events"],
        {
            "event_type": "ending",
            "frame": float(attempt["owner_end_frame"]),
            "frame_interval": [
                float(attempt["owner_end_frame"]),
                float(attempt["owner_end_frame"]),
            ],
            "note": "supplied attempt ending",
            "annotation_origin": attempt.get("annotation_origin", "agent"),
        },
    ]
    if not attempt.get("ending_supplied", True):
        supplied_topology = list(attempt["events"])
    # Fixed enumeration: net present, then net absent, each retaining its own
    # no-addition/contact/bounce/joint family order. One cumulative clock covers
    # the whole union; an early failure leaves its remaining time to the branches
    # after it and no fitted parameter crosses a branch.
    membership_branch_counts = {
        name: 1 + len(membership_state[name]["hypotheses"]) for name in membership_names
    }
    total_branch_count = sum(membership_branch_counts.values())
    outcome, supplied_blocker = None, None
    membership_no_addition: dict[str | None, dict] = {}
    membership_blockers: dict[str | None, str | None] = {}
    membership_attempted: dict[str | None, bool] = {}
    membership_fitted: dict[str | None, list] = {}
    membership_receipts: dict[str | None, list] = {}
    position = 0
    for name in membership_names:
        state = membership_state[name]
        branch_attempt = state["attempt"]
        branch_document = state["document"]
        branch_hypotheses = state["hypotheses"]
        branch_outcome, branch_blocker = None, None
        attempted = False
        if branch_hypotheses or membership_active:
            budget.seconds = optional.cumulative_deadline(
                original_search_seconds, position, total_branch_count
            )
        if membership_active:
            budget.exhausted = False
        try:
            if optional_bounce_mode and branch_document.get("null_hold") is not None:
                raise ValueError(f"original no-addition scope held: {branch_document['null_hold']}")
            if not branch_attempt.get("ending_supplied", True) and scope_contract is None:
                raise ValueError("supplied topology has no ending event")
            attempted = True
            branch_outcome = run_topology(
                list(branch_attempt["events"]),
                float(branch_attempt["owner_end_frame"]),
                "supplied",
                membership=name,
            )
        except search_budget.SearchDeadline as error:
            if not branch_hypotheses and not membership_active:
                raise
            # This branch received only its own slice. Source-qualified alternatives
            # and the remaining declared membership still share the original clock.
            branch_blocker = f"{type(error).__name__}: {error}"
        except (ValueError, KeyError, IndexError, FloatingPointError, OverflowError) as error:
            branch_blocker = f"{type(error).__name__}: {error}"
            if args.event_recovery == "off" and not branch_hypotheses and not membership_active:
                raise
        position += 1
        membership_no_addition[name] = branch_outcome
        membership_blockers[name] = branch_blocker
        membership_attempted[name] = attempted
        if branch_hypotheses:
            fitted, receipts = optional.fit_alternatives(
                run_topology,
                branch_hypotheses,
                float(branch_attempt["owner_end_frame"]),
                budget,
                original_search_seconds,
                start_position=position,
                total_branches=total_branch_count,
                run_kwargs={"membership": name},
            )
        else:
            fitted, receipts = [], []
        position += len(branch_hypotheses)
        membership_fitted[name] = fitted
        membership_receipts[name] = receipts
    outcome = membership_no_addition[membership_names[0]]
    supplied_blocker = membership_blockers[membership_names[0]]
    recovery = dict(
        mode=args.event_recovery,
        supplied_topology_blocker=supplied_blocker,
        proposals=[],
        hypotheses=[],
        branches=[],
        recovered_events=[],
        demoted_events=[],
        surviving_alternatives=[],
        selection_uses_withheld_pixels=native_seed_check_pixels.selection_from_usages(
            native_seed_usages
        )["selector_uses_withheld_pixels"],
        selection_uses_evaluation_labels=False,
    )
    branches = [row for row in membership_no_addition.values() if row is not None]
    if args.event_recovery == "on":
        proposal_players = [] if outcome is None else outcome["players"]
        if not proposal_players:
            try:
                proposal_players = alternating_player_states(
                    args.pose_csv,
                    attempt["point_clip"],
                    [row for row in attempt["events"] if row["event_type"] == "contact"],
                    label_lookup,
                    image_coordinate_scale=args.pose_image_scale,
                )
            except (ValueError, KeyError):
                proposal_players = []
        window = (
            min(label_lookup, default=0.0),
            max(label_lookup, default=0.0),
        )
        proposals = event_recovery.propose_from_observations(
            camera_lookup, label_lookup, proposal_players, supplied_topology, window
        )
        if outcome is not None:
            selected_measurement = (outcome["selected"] or outcome["diagnostic"])["measurement"]
            proposals = [
                *event_recovery.propose(
                    selected_measurement,
                    camera_lookup,
                    label_lookup,
                    proposal_players,
                    float(window[1]),
                ),
                *proposals,
            ]
        if not any(row["event_type"] == "ending" for row in supplied_topology):
            proposals = [*event_recovery.ending_candidates(supplied_topology, window), *proposals]
        # A supplied event is only demotable when the physics did not explain the
        # point: either the supplied topology could not be built at all, or it
        # built and no branch of it survived the final gates.  Events the
        # observations cannot witness at all are named first, so a bounded beam
        # spends its budget on the event that actually blocked the fit rather
        # than walking the supplied topology in name order.
        demotable: list[dict] = []
        if outcome is None or outcome["selected"] is None:
            unwitnessable = event_recovery.unwitnessable_supplied_events(
                attempt["events"], camera_lookup, label_lookup
            )
            named = {(row["event_type"], float(row["frame"])) for row in unwitnessable}
            demotable = [
                *unwitnessable,
                *[
                    row
                    for row in attempt["events"]
                    if (row["event_type"], float(row["frame"])) not in named
                ],
            ]
        recovery["unwitnessable_supplied_events"] = [
            {
                "event_type": row["event_type"],
                "frame": row["frame"],
                "demotion_reason": row["demotion_reason"],
            }
            for row in demotable
            if row.get("demotion_reason")
        ]
        # A point whose modeled window ends at the net has no terminal ground
        # bounce, so the connected grammar refuses it before any fit runs.  The
        # legal readings of that ending are structural, not residual-driven, so
        # they are enumerated from the labels and the native fronts directly.
        window_labels = {
            int(row["frame"]): np.asarray([row["x1080"], row["y1080"]], float)
            for record in labels["ball"]["records"]
            if record["clip"] == attempt["point_clip"]
            for row in record["frames"]
            if row["status"] == "visible"
        }
        extensions = event_recovery.net_termination_hypotheses(
            supplied_topology,
            [
                row
                for row in attempt.get("agent_event_rows", [])
                if row["event_type"] == "bounce"
                and float(row["frame"]) >= float(attempt["owner_end_frame"])
            ],
            window_labels,
            uncertain_events=[
                row
                for row in labels.get("events", {}).get("records", [])
                if row.get("status") != "labeled"
            ],
        )
        recovery["net_termination_readings"] = [row.as_dict() for row in extensions]
        hypotheses = event_recovery.topology_hypotheses(
            supplied_topology,
            proposals,
            max_branches=args.max_topology_branches,
            demotable=demotable,
            extensions=extensions,
        )
        recovery["proposals"] = [row.as_dict() for row in proposals]
        recovery["hypotheses"] = [row.as_dict() for row in hypotheses]
        for hypothesis in hypotheses:
            if hypothesis.name == "supplied" and outcome is not None:
                branch = dict(outcome)
                branch["hypothesis"] = hypothesis
                recovery["branches"].append(
                    {**hypothesis.as_dict(), "status": "fitted", "reused_supplied_fit": True}
                )
                branches[0] = branch
                continue
            physical = [row for row in hypothesis.events if row["event_type"] != "ending"]
            ending = [row for row in hypothesis.events if row["event_type"] == "ending"]
            if not ending:
                recovery["branches"].append(
                    {**hypothesis.as_dict(), "status": "no_ending_in_branch"}
                )
                continue
            try:
                branch = run_topology(physical, float(ending[0]["frame"]), hypothesis.name)
            except search_budget.SearchDeadline as error:
                recovery["branches"].append(
                    {
                        **hypothesis.as_dict(),
                        "status": "search_deadline_no_completed_candidate",
                        "blocker": str(error),
                    }
                )
                # Retain completed earlier topologies; no budget remains for another.
                break
            except (
                ValueError,
                KeyError,
                IndexError,
                FloatingPointError,
                OverflowError,
            ) as error:
                recovery["branches"].append(
                    {
                        **hypothesis.as_dict(),
                        "status": "branch_failed",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            branch["hypothesis"] = hypothesis
            family = branch["families"]["combined_toss_and_serve_prior"]
            recovery["branches"].append(
                {
                    **hypothesis.as_dict(),
                    "status": "fitted",
                    "family_count": family["count"],
                    "reconstructed": bool(family["reconstructed_on_family_basis"]),
                    "input_only_rank_score": (
                        None
                        if branch["selected"] is None
                        else branch["selected"]["evidence"]["input_only_rank_score"]
                    ),
                }
            )
            branches.append(branch)

    supplied_reconstructs = bool(
        outcome is not None
        and outcome["families"]["combined_toss_and_serve_prior"]["reconstructed_on_family_basis"]
    )
    optional_ranked = bool(optional_hypotheses) or membership_active
    if optional_ranked:
        optional_branches = []
        for name in membership_names:
            optional_branches.append(
                {
                    "name": "supplied",
                    "status": "fitted" if membership_no_addition[name] else "topology_failed",
                    "reason": membership_blockers[name],
                    **(
                        {"numerically_attempted": membership_attempted[name]}
                        if optional_bounce_mode
                        else {}
                    ),
                    **({"membership": name} if membership_active else {}),
                }
            )
            optional_branches.extend(
                {**receipt, **({"membership": name} if membership_active else {})}
                for receipt in membership_receipts[name]
            )
            branches.extend(membership_fitted[name])
        if optional_joint:
            for receipt in optional_branches:
                receipt["source_family"] = optional_union.family_of(receipt["name"])
        ranked = []
        for branch in branches:
            flat = sorted(float(f) for frames in branch["scene"].observation_frames for f in frames)
            if flat != sorted(float(row[0]) for row in optional_training_signature):
                raise ValueError(
                    "optional topology changed training/contact epoch observation ownership"
                )
            admission = optional_input_domain_admission(
                branch, operator, include_seeded_final_net=args.observation_net_seed != "off"
            )
            hypothesis = branch.get("optional_hypothesis")
            # Actual added continuous parameters on the common membership reference.
            # Zero whenever both branches solve the same flight inventory, so the
            # existing optional conventions and scores are untouched by default.
            offset = net_membership.added_parameter_count(
                membership_state[branch["membership"]]["attempt"], 0
            )
            branch["membership_parameter_offset"] = offset
            if offset:
                base = hypothesis or {"added": [], "occurrence_log_odds": 0}
                declared = base.get("added_parameter_count")
                hypothesis = {
                    **base,
                    "added_parameter_count": (
                        (6 * len(base["added"]) if declared is None else declared) + offset
                    ),
                }
            if membership_active:
                prior = net_membership.occurrence_prior(
                    membership_state[branch["membership"]]["attempt"]
                )
                base = hypothesis or {"added": [], "occurrence_log_odds": 0}
                hypothesis = {
                    **base,
                    "occurrence_log_odds": base["occurrence_log_odds"] + prior["log_odds"],
                }
            # Candidate-specific complexity: the actual native-estimated net
            # response parameters this candidate's own construction receipt
            # declares, never a family constant. A net-absent branch has no net,
            # so an inherited response there is a contract failure, not a score.
            rows = [*branch["coarse"], *branch["refined"]]
            dimensions_of = None
            if membership_active:
                for row in rows:
                    net_membership.candidate_dimensions(branch["membership"], row)

                def dimensions_of(row, name=branch["membership"]):
                    return net_membership.candidate_dimensions(name, row)

            holds: list = []
            try:
                candidate, score = optional.best_completed(
                    branch,
                    hypothesis,
                    len(optional_signature),
                    input_admission=admission,
                    added_dimensions=dimensions_of,
                    holds=holds,
                )
            except ValueError as error:
                for receipt in optional_branches:
                    if (
                        receipt["name"] == branch["topology_name"]
                        and receipt.get("membership", branch["membership"]) == branch["membership"]
                    ):
                        receipt.update(
                            status="input_domain_refused",
                            reason=str(error),
                            **({"unrankable_candidates": holds} if holds else {}),
                        )
                continue
            branch["optional_selected"] = candidate
            branch["optional_score"] = score
            if membership_active:
                branch["membership_response_dimensions"] = net_membership.candidate_dimensions(
                    branch["membership"], candidate
                )
                for receipt in optional_branches:
                    if (
                        receipt["name"] == branch["topology_name"]
                        and receipt.get("membership") == branch["membership"]
                    ):
                        receipt["response_dimensions"] = branch["membership_response_dimensions"]
                        if holds:
                            receipt["unrankable_candidates"] = holds
            ranked.append(
                (
                    score,
                    (net_membership.index_of(branch["membership"]), branch["topology_name"]),
                    branch,
                )
            )
        if membership_active:
            # The declared complexity offset must match the flight inventory the
            # branches actually solved; a branch that silently changed dimensions
            # cannot be compared on the recorded common reference.
            base_dimensions = {
                row[2]["membership"]: 5 + 6 * len(row[2]["scene"].pixels)
                for row in ranked
                if row[2].get("optional_hypothesis") is None
            }
            if len(base_dimensions) > 1:
                reference = min(base_dimensions.values())
                for name, dimension in base_dimensions.items():
                    if dimension - reference != net_membership.added_parameter_count(
                        membership_state[name]["attempt"], 0
                    ):
                        raise ValueError(
                            "membership parameter complexity differs from its declared reference"
                        )
        if ranked:
            winner = min(ranked, key=lambda row: row[:2])[2]
            winner_hypothesis = winner.get("optional_hypothesis", {})
            added = len(winner_hypothesis.get("added", []))
            # Every family receipt and source binding belongs to the branch scope
            # the winner was actually enumerated and fitted under.
            winner_state = membership_state[winner["membership"]]
            optional_family_receipts = winner_state["family_receipts"]
            optional_binding_receipt = winner_state["binding"]
            if optional_joint:
                family_fields = optional_union.selection_fields(
                    winner["topology_name"],
                    winner_hypothesis,
                    optional_family_receipts,
                    optional_binding_receipt,
                    {
                        family: provenance.file_record(path)
                        for family, path in (
                            ("contact", args.optional_contact_witness),
                            ("bounce", args.optional_bounce_witness),
                        )
                    },
                )
            elif optional_bounce_mode:
                family_fields = {
                    "policy": "source_witness_optional_bounce_v1",
                    "witness_record": provenance.file_record(optional_witness_path),
                    "added_contact_count": 0,
                    "added_bounce_count": added,
                    "added_parameter_count": 0,
                    "legacy_selection_field": "optional_contact_selection",
                }
            elif optional_witness_path is None:
                # A membership run with no ordinary optional witness still selects
                # through the same input-only score and the same receipt shape.
                family_fields = {
                    "policy": net_membership.SELECTION_POLICY,
                    "added_contact_count": added,
                    "added_bounce_count": 0,
                    "added_parameter_count": winner["membership_parameter_offset"],
                }
            else:
                family_fields = {
                    "policy": "source_witness_optional_contact_v1",
                    "witness_record": provenance.file_record(optional_witness_path),
                    "added_contact_count": added,
                    **(
                        {"scope": optional.FINAL_SCOPE}
                        if args.optional_final_contacts == "on"
                        else {}
                    ),
                }
            if args.optional_contact_composition != "off":
                family_fields["optional_contact_composition"] = args.optional_contact_composition
            if args.optional_interior_contacts != "off":
                family_fields["optional_interior_contacts"] = args.optional_interior_contacts
            if args.optional_conditional_grounds != "off":
                family_fields["optional_conditional_grounds"] = args.optional_conditional_grounds
            optional_selection = {
                "selected_topology": winner["topology_name"],
                "input_domain_selection": "original_physical_domain_before_rank_v1",
                **optional.observation_count_receipt(
                    len(optional_signature), len(optional_training_signature)
                ),
                "branches": optional_branches,
                "source_ranked_hypotheses": membership_state[winner["membership"]]["hypotheses"],
                "selection_uses_gates": False,
                "selection_uses_reference": False,
                "score": winner["optional_score"],
                "total_search_seconds": original_search_seconds,
                **family_fields,
                "occurrence_log_odds": winner_hypothesis.get("occurrence_log_odds", 0),
                "candidate_scores": [
                    {
                        "topology": b["topology_name"],
                        "score": b["optional_score"],
                        **({"membership": b["membership"]} if membership_active else {}),
                    }
                    for _, _, b in ranked
                ],
                **(
                    {
                        **net_membership.selection_fields(
                            membership_state[winner["membership"]]["attempt"]
                        ),
                        "terminal_net_membership_parameter_offset": winner[
                            "membership_parameter_offset"
                        ],
                        # The winning candidate's own response complexity, so a
                        # replay reproduces exactly the ranking that was run.
                        "terminal_net_membership_response_dimensions": winner[
                            "membership_response_dimensions"
                        ],
                        "membership_branch_counts": membership_branch_counts,
                        "membership_enumeration": list(membership_names),
                    }
                    if membership_active
                    else {}
                ),
            }

        # One selection, but each existing single-arm reader keeps its own file
        # and sees only its own branches beside the shared no-addition receipt.
        optional_receipts = (
            []
            if optional_witness_path is None
            else (
                [
                    (
                        optional_union.RECEIPT_FILES[family],
                        {
                            "original_no_addition_retained": True,
                            "branches": [
                                row
                                for row in optional_branches
                                if row["source_family"] in (family, optional_union.SUPPLIED)
                            ],
                            "selection": optional_selection,
                            "gates_used_for_selection": False,
                            "fixed_horizon": attempt["owner_end_frame"],
                            "source_family": family,
                            "joint_source_families": list(optional_union.FAMILIES),
                            "joint_branches": optional_branches,
                            "optional_family_receipts": optional_family_receipts,
                            "source_bindings": optional_binding_receipt,
                        },
                    )
                    for family in (
                        optional_union.SELECTION_FAMILIES
                        if optional_composed
                        else optional_union.FAMILIES
                    )
                ]
                if optional_joint
                else [
                    (
                        "optional_bounce_search.json"
                        if optional_bounce_mode
                        else "optional_contact_search.json",
                        {
                            "original_no_addition_retained": True,
                            "branches": optional_branches,
                            "selection": optional_selection,
                            "gates_used_for_selection": False,
                            "fixed_horizon": attempt["owner_end_frame"],
                        },
                    )
                ]
            )
        )
        for name, document in optional_receipts:
            search_budget.atomic_json(args.output / name, document)
        if membership_active:
            # A membership run owns its own ordinary search receipt, written
            # before any terminal error: the two declared branches, the ordinary
            # arms enumerated inside each and how each one ended. An
            # all-branches-failed run therefore persists its failure inventory
            # rather than only a generic error.
            search_budget.atomic_json(
                args.output / net_membership.RECEIPT_FILE,
                {
                    "original_no_addition_retained": True,
                    "policy": membership_policy,
                    "enumeration": list(net_membership.MEMBERSHIPS),
                    "branch_counts": membership_branch_counts,
                    "branches": optional_branches,
                    "selection": optional_selection,
                    "gates_used_for_selection": False,
                    "fixed_horizon": attempt["owner_end_frame"],
                    "activated_training_observations": (
                        None
                        if optional_training_signature is None
                        else len(optional_training_signature)
                    ),
                    "training_observations": (
                        None if optional_signature is None else len(optional_signature)
                    ),
                    "original_partition_training_observations": (
                        None if optional_signature is None else len(optional_signature)
                    ),
                    "receipt": net_membership.derivation_fields(attempt),
                },
            )

    if optional_ranked:
        if not ranked:
            raise ValueError(
                "no finite optional-contact candidate in original physical input domain"
            )
        branches = [branch for _, _, branch in ranked]

    for branch in branches:
        branch["ending_context"] = ending_witnesses.score(
            point_context_document, branch["events"], branch["termination_kind"]
        )
    context_available = any(
        branch["ending_context"]["witness"].get("available") for branch in branches
    )

    def branch_score(row: dict) -> tuple:
        """Rank branches with bounded point context beside the fit's own score.

        Without usable context the historical additive-recovery ordering is exact.
        With context, a working supplied branch remains preferred by its fit and
        grammar score but is not a hard gate against a better supported ending.
        """
        family = row["families"]["combined_toss_and_serve_prior"]
        penalty = getattr(row.get("hypothesis"), "grammar_penalty", 0.0)
        context_penalty = float(row["ending_context"]["penalty"])
        best = row["selected"] or row["diagnostic"]
        return (
            supplied_reconstructs and not context_available and row["topology_name"] != "supplied",
            not family["reconstructed_on_family_basis"],
            best["evidence"]["input_only_rank_score"] + penalty + context_penalty,
            row["topology_name"],
        )

    if not branches:
        # Name why every branch failed, not only the supplied one: a recovery
        # arm that enumerates alternatives is useless if its refusals are
        # invisible to the loss analysis.
        failures = "; ".join(
            f"{row.get('name', 'unnamed')}: {row.get('reason', row.get('status'))}"
            for row in recovery["branches"]
            if row.get("status") != "fitted"
        )
        raise ValueError(
            f"no fittable topology branch: {supplied_blocker}"
            + (f" | branches: {failures}" if failures else "")
        )
    chosen = (
        min(
            branches,
            key=lambda b: (
                b["optional_score"],
                net_membership.index_of(b["membership"]),
                b["topology_name"],
            ),
        )
        if optional_selection is not None
        else min(branches, key=branch_score)
    )
    if optional_selection is not None:
        chosen["selected"] = chosen["optional_selected"]
        chosen["diagnostic"] = chosen["optional_selected"]
    if args.event_recovery == "on":
        hypothesis = chosen.get("hypothesis")
        recovery["recovered_events"] = [] if hypothesis is None else hypothesis.added
        recovery["demoted_events"] = [] if hypothesis is None else hypothesis.demoted
        recovery["selected_topology"] = chosen["topology_name"]
        recovery["surviving_alternatives"] = [
            {
                "topology_name": row["topology_name"],
                "family_count": row["families"]["combined_toss_and_serve_prior"]["count"],
                "grammar_penalty": getattr(row.get("hypothesis"), "grammar_penalty", 0.0),
                "input_only_rank_score": (row["selected"] or row["diagnostic"])["evidence"][
                    "input_only_rank_score"
                ],
                "ending_context": row["ending_context"],
                "added_events": getattr(row.get("hypothesis"), "added", []),
                "demoted_events": getattr(row.get("hypothesis"), "demoted", []),
            }
            for row in sorted(branches, key=branch_score)
            if row["families"]["combined_toss_and_serve_prior"]["reconstructed_on_family_basis"]
        ]
    termination_kind = chosen["termination_kind"]
    scene = chosen["scene"]
    players = chosen["players"]
    right_contact_player = chosen["right_contact_player"]
    statures = chosen["statures"]
    athlete_evidence = chosen["athlete_evidence"]
    low, high = chosen["depth_bounds"]
    all_depths = chosen["all_depths"]
    depths = chosen["fitted_depths"]
    excluded_depths = chosen["excluded_depths"]
    targets = chosen["targets"]
    axis_receipt = chosen["axis_receipt"]
    initialization_evidence = chosen["initialization_evidence"]
    toss_observations = chosen["toss_observations"]
    toss_feet = chosen["toss_server_feet"]
    toss_contact_estimate = chosen["toss_contact_estimate"]
    toss_contact_bounds = chosen["toss_contact_bounds"]
    location_prior = chosen["serve_location_prior"]
    location_prior_bounds = chosen["serve_location_prior_bounds"]
    contact_hypotheses = chosen["serve_contact_hypotheses"]
    contact_intersection = chosen["serve_contact_intersection"]
    same_player_regularization = chosen["same_player_serve_regularization"]
    contact_history_receipt = chosen["serve_contact_history_receipt"]
    contact_hypothesis_blocker = chosen["serve_contact_hypothesis_blocker"]
    contact_hypothesis_restart_failures = chosen["serve_contact_hypothesis_restart_failures"]
    epoch_prior = chosen["first_contact_epoch_prior"]
    coarse = chosen["coarse"]
    coarse_survivors = chosen["coarse_survivors"]
    shortlist = chosen["shortlist"]
    refined = chosen["refined"]
    selected = chosen["selected"]
    selected_arms = chosen["selected_arms"]
    families = chosen["families"]
    family_widths = chosen["family_widths"]
    diagnostic = chosen["diagnostic"]
    terminal_rebound_receipt = chosen["terminal_rebound"]

    report = dict(
        schema="s6_agent_whole_point_search_v2",
        scope=__doc__,
        status="selected_not_independently_xyz_certified"
        if selected
        else "held_no_surviving_candidate",
        attempt_id=attempt["attempt_id"],
        clip=attempt["point_clip"],
        events=chosen["events"],
        modeled_event_scope=chosen["modeled_event_scope"],
        supplied_events=attempt["events"],
        event_recovery=recovery,
        **({"optional_contact_selection": optional_selection} if optional_selection else {}),
        ending_context_selection={
            "context_available": context_available,
            "selected_topology": chosen["topology_name"],
            "selected": chosen["ending_context"],
            "branches": [
                {
                    "topology_name": row["topology_name"],
                    "ending_context": row["ending_context"],
                    "fit_input_only_rank_score": (row["selected"] or row["diagnostic"])["evidence"][
                        "input_only_rank_score"
                    ],
                    "grammar_penalty": getattr(row.get("hypothesis"), "grammar_penalty", 0.0),
                }
                for row in sorted(branches, key=branch_score)
            ],
            "hard_gate": False,
        },
        configuration=dict(
            # Read back by per_flight_rescore so the verdict replay builds the same scene.
            **({} if args.exit_partial_endings == "off" else {"exit_partial_endings": "on"}),
            **(
                {
                    contact_role.FIELD: role_receipt,
                    "first_contact_parameterization": "ordinary_contact_free_xyz",
                    "serve_evidence_applicable": False,
                    **(
                        {rally_cue.FIELD: chosen["original_rally_origin_depth_cue"]}
                        if is_rally
                        else {}
                    ),
                }
                if nonserve_origin
                else {}
            ),
            **(
                {"serve_location_prior_model": provenance.file_record(args.serve_location_prior)}
                if args.serve_location_prior is not None
                else {}
            ),
            **({"observation_scope": scope_contract} if scope_contract is not None else {}),
            **(
                {
                    ("contact_components" if component_ending else "contact_prefix_scope"): (
                        args.contact_components if component_ending else args.contact_prefix_scope
                    ),
                    "right_boundary_kind": (
                        "supplied_end" if terminal_track is not None else "original_contact"
                    ),
                    **(
                        {"endpoint_semantics": terminal_track["kind"]}
                        if terminal_track is not None
                        else {}
                    ),
                    **(
                        {"partial_flight": True}
                        if terminal_track is not None
                        and terminal_track.get("partial_flight") is True
                        else {}
                    ),
                    **(
                        {}
                        if prefix_interior is None
                        else {"contact_prefix_interior_scope": prefix_interior}
                    ),
                    **(
                        {}
                        if chosen.get("closing_contact_anchor") is None
                        else {"closing_contact_anchor": chosen["closing_contact_anchor"]}
                    ),
                }
                if contact_ending or terminal_track is not None
                else {}
            ),
            coarse_iterations=args.coarse_iterations,
            refine_iterations=args.refine_iterations,
            observation_partition=args.observation_partition,
            exposure_duration_frames=exposure_duration,
            ball_observation_operator=operator,
            exposure_duration_measured=False,
            surface=args.surface,
            termination_kind=termination_kind,
            # Branch-local: the selected membership owns the tail scope every
            # replay, refinement and export reconstruction reads back.
            terminal_net_tail=chosen["terminal_net_tail"],
            observed_horizon_tail=chosen["observed_horizon_tail"],
            **(
                net_membership.selection_fields(membership_state[chosen["membership"]]["attempt"])
                if membership_active
                else {}
            ),
            ground_settling=args.ground_settling,
            **(
                {"passive_bounce_response": args.passive_bounce_response}
                if args.passive_bounce_response != "off"
                else {}
            ),
            **(
                {
                    "observation_net_seed": args.observation_net_seed,
                    "net_seed_speed_scale_mps": args.net_seed_speed_scale_mps,
                }
                if args.observation_net_seed != "off"
                else {}
            ),
            terminal_bounce_count=chosen["terminal_bounce_count"],
            terminal_rebound=terminal_rebound_receipt,
            **(
                {"terminal_context_ownership": "on"}
                if args.terminal_context_ownership == "on"
                else {}
            ),
            interior_contact_epochs=args.interior_contact_epochs,
            fps=scene.fps,
            depth_values_m=all_depths.tolist(),
            fitted_depth_values_m=depths.tolist(),
            depth_bounds_m=[low, high],
            short_horizon_frames=4,
            medium_horizon_frames=10,
            directional_window_objective="fixed 16 px RMS feasibility constraints on every 4/10-frame event-side window",
            directional_rms_limit_px=16.0,
            player_reach_m=1.75,
            athlete_prior_mode=args.athlete_prior_mode,
            observation_fallback=args.observation_fallback,
            observation_fallback_receipt=fallback_receipt,
            dense_label_supplement=dense_receipt,
            player_ledger=chosen.get("player_ledger"),
            shared_state_second_stage_blocker=chosen.get("shared_state_second_stage_blocker"),
            player_statures_m=statures,
            **athlete_evidence_configuration(args.athlete_evidence, athlete_evidence),
            **athlete_priors.root_reach_configuration(args.athlete_root_reach_loss),
            serve_height_zero_penalty_interval_m=(
                None
                if nonserve_origin
                else [2.4, 3.1]
                if args.athlete_prior_mode == "global_hard_caps"
                else None
                if players[0]["stature_m"] is None
                else [
                    athlete_priors.AthletePriorConfig().serve_height_low_statures
                    * players[0]["stature_m"],
                    athlete_priors.AthletePriorConfig().serve_height_high_statures
                    * players[0]["stature_m"],
                ]
            ),
            serve_reach_cut=args.serve_reach_cut,
            **missing_player_position_configuration(
                args.missing_player_position, players, right_contact_player=right_contact_player
            ),
            **chosen.get("serve_association", {}),
            serve_reach_margin_statures=args.serve_reach_margin_statures,
            serve_reach_radius_statures=args.serve_reach_radius_statures,
            serve_reach_cylinder=chosen["serve_reach_cylinder"],
            serve_reach_ray_cut=chosen["serve_reach_ray_cut"],
            contact_xy_search_envelope_m=[list(row) for row in CONTACT_XY_ENVELOPE_M],
            player_image_coordinate_scale=args.pose_image_scale,
            bounce_tolerance_m=0.9144,
            bounce_witness_mode=args.bounce_witness_mode,
            **(
                {"fit_ground_witness": args.fit_ground_witness}
                if args.fit_ground_witness != "off"
                else {}
            ),
            bounce_witness_observation_policy=scene.bounce_witness_observation_policy,
            **(
                {}
                if args.serve_ending_consistency == "off"
                else {
                    "serve_ending_consistency": args.serve_ending_consistency,
                    "labeled_ending_kind": args.ending_kind,
                }
            ),
            bounce_gate_distance=(
                "max(0, raw centre-height ray distance - graded circle radius)"
                if args.bounce_witness_mode == "subframe_graded_circle"
                else "raw centre-height ray distance"
            ),
            **(
                {}
                if args.serve_toss_constraint == "off"
                else {
                    "serve_toss_constraint": {
                        "mode": "on",
                        "source_schema": toss_observations["source_schema"],
                        "continuous_first_contact_depth": True,
                        "fixed_half_metre_depth_beam": False,
                        "soft_residual": True,
                        "hard_bound": "three sigma intersected with server reach parameter box",
                        "optimizer_inequality_added": False,
                    }
                }
            ),
            **(
                {}
                if args.serve_contact_epoch == "off"
                else {
                    "serve_contact_epoch": {
                        "mode": "on",
                        **epoch_prior,
                        "optimizer_variable": True,
                        "optimizer_inequality_added": False,
                    }
                }
            ),
            **(
                {}
                if args.serve_start_prior == "off"
                else {
                    "serve_start_prior": {
                        "mode": "on",
                        "continuous_first_contact_xyz": True,
                        "fixed_half_metre_depth_beam": False,
                        "soft_residual": True,
                        "hard_bound": location_prior_bounds["bound_kind"],
                        "optimizer_inequality_added": False,
                    }
                }
            ),
            **(
                {}
                if args.serve_contact_hypotheses == "off"
                else {
                    "serve_contact_hypotheses": {
                        "mode": "on",
                        "candidate_count": len(contact_hypotheses),
                        "candidate_generator_only": True,
                        "contact_parameter_bound_added": False,
                        "contact_optimizer_inequality_added": False,
                        "reference_depth_restarts_retained": True,
                        "selection_rule": "unchanged lowest input-only rank score",
                        "contact_epoch_initialized_from_intersection": bool(
                            contact_intersection is not None
                        ),
                        "same_player_regularization": same_player_regularization,
                        "blocker": contact_hypothesis_blocker,
                        "restart_failures": contact_hypothesis_restart_failures,
                    }
                }
            ),
            # Recorded only when an explicit arm flag moved something, so the
            # promoted reference configuration keeps its published shape.
            **(
                {}
                if args.refine_inequalities == "in"
                and args.anchor_residuals == "absent"
                and args.seed_restarts == "off"
                and args.reference_restart == "off"
                and args.terminal_rebound == "off"
                and args.shooting_parameterization == "single"
                and args.local_flight_refits == "off"
                and args.final_bounce_anchor == "off"
                else {
                    "solver_experimental_arm": {
                        "refine_inequalities": args.refine_inequalities,
                        "anchor_residuals": args.anchor_residuals,
                        "anchor_player_sigma_m": args.anchor_player_sigma_m,
                        "seed_restarts": args.seed_restarts,
                        **(
                            {}
                            if args.final_bounce_anchor == "off"
                            else {
                                "final_bounce_anchor": args.final_bounce_anchor,
                                "final_bounce_anchor_plan": chosen["final_bounce_anchor"],
                            }
                        ),
                        **owner_approved_gate_receipt,
                        **net_cord_response_receipt,
                        **(
                            {}
                            if args.event_ground_seed == "off"
                            else {
                                "event_ground_seed": args.event_ground_seed,
                                "event_ground_seed_policy": (
                                    "coarse structure-aware walk gains one gravity-only start "
                                    "per one-bounce flight aimed at the seed's own native "
                                    "ground-ray impact at the supplied epoch; walk cost, "
                                    "solver, budget and selection unchanged"
                                ),
                            }
                        ),
                        **(
                            {}
                            if args.reference_restart == "off"
                            else {"reference_restart": args.reference_restart}
                        ),
                        **(
                            {}
                            if args.terminal_rebound == "off"
                            else {
                                "terminal_rebound": args.terminal_rebound,
                                "terminal_rebound_policy": (
                                    "source-owned native aftermath; original train/check partition retained; "
                                    "measured surface model; spin carried"
                                    if args.terminal_context_ownership == "on"
                                    else "all labeled frames after the final supplied bounce; "
                                    "measured surface model; spin carried"
                                ),
                            }
                        ),
                        **(
                            {}
                            if args.local_flight_refits == "off"
                            else {"local_flight_refits": args.local_flight_refits}
                        ),
                        **(
                            {}
                            if args.shooting_parameterization == "single"
                            else {"shooting_parameterization": args.shooting_parameterization}
                        ),
                        **(
                            {}
                            if args.shared_state_second_stage_reference is None
                            else {
                                "shared_state_second_stage": "promoted_rank_one_plus_unchanged_fallback"
                            }
                        ),
                        "anchor_contact_height_m": ANCHOR_CONTACT_HEIGHT_M,
                        "serve_anchor_stature_band": [1.5, 1.75],
                        "seeds_unchanged": args.seed_restarts == "off",
                        "depth_branches_unchanged": True,
                        "dynamics_unchanged": True,
                        "acceptance_gates_unchanged": True,
                    }
                }
            ),
        ),
        inputs=[
            provenance.file_record(path)
            for path in (
                args.labels,
                args.packet,
                args.cameras,
                args.pose_csv,
                Path(__file__),
                Path(single.__file__),
                Path(athlete_priors.__file__),
                Path(serve_reach_cylinder.__file__),
                Path(search_reporting.__file__),
                *(
                    [Path(fit_ground_witness_center.__file__)]
                    if args.fit_ground_witness != "off"
                    else []
                ),
                args.serve_number_evidence,
                *([args.baseline] if args.baseline is not None else []),
                *([args.automatic_track] if args.automatic_track is not None else []),
                *([args.serve_speed] if args.serve_speed is not None else []),
                *([args.dense_labels] if args.dense_labels is not None else []),
                *([args.toss_labels] if args.toss_labels is not None else []),
                *([args.point_context] if args.point_context is not None else []),
                *([args.serve_contact_history] if args.serve_contact_history is not None else []),
                *(
                    [args.shared_state_second_stage_reference]
                    if args.shared_state_second_stage_reference is not None
                    else []
                ),
            )
        ],
        code=provenance.git_record(paths.REPO_ROOT),
        human_derived=True,
        automatic_inference_eligible=False,
        observation_model=dict(
            duration_frames=exposure_duration, operator=operator, axes=axis_receipt
        ),
        **(
            {"depth_conditioned_seed": {"policy": "on", "branches": chosen["depth_seed_receipts"]}}
            if args.depth_conditioned_seed == "on"
            else {}
        ),
        initialization=initialization_evidence,
        player_states=players,
        **({"right_contact_player": chosen["right_contact_player"]} if contact_ending else {}),
        ball_observation_accounting=observation_accounting(attempt["owner_ball_labels"]),
        toss_observations=toss_observations,
        toss_server_feet=toss_feet,
        **(
            {}
            if args.serve_toss_constraint == "off"
            else {
                "toss_contact_estimate": toss_contact_estimate,
                "toss_contact_parameter_bounds": toss_contact_bounds,
            }
        ),
        **(
            {}
            if args.serve_start_prior == "off"
            else {
                "serve_location_prior": location_prior,
                "serve_location_parameter_bounds": location_prior_bounds,
                "serve_location_toss_estimate": toss_contact_estimate,
            }
        ),
        **(
            {}
            if args.serve_contact_hypotheses == "off"
            else {
                "serve_contact_intersection": contact_intersection,
                "serve_contact_hypotheses": [
                    {key: value for key, value in row.items() if key != "seed_parameters"}
                    for row in contact_hypotheses
                ],
                "serve_contact_hypothesis_blocker": contact_hypothesis_blocker,
                "serve_contact_hypothesis_restart_failures": contact_hypothesis_restart_failures,
                "same_player_serve_regularization": same_player_regularization,
                "serve_contact_history_receipt": contact_history_receipt,
                "serve_location_prior": location_prior,
                "serve_location_prior_metadata": location_prior_bounds,
            }
        ),
        serve_number=dict(status="not_applicable", reason=origin_reason)
        if nonserve_origin
        else dict(
            value=args.serve_number,
            evidence=provenance.file_record(args.serve_number_evidence),
            interpretation="explicit point-ledger/score-state attempt order; not inferred from pixels here",
        ),
        bounce_ground_ray_targets=targets,
        numerical_candidate_failures=chosen["numerical_candidate_failures"],
        coarse_candidates=coarse,
        coarse_candidate_count=len(coarse),
        fixed_depth_beam_exclusions=excluded_depths,
        coarse_survivor_count_before_serve_speed=sum(
            row["evidence"]["survived_before_serve_speed"] for row in coarse
        ),
        coarse_survivor_count=len(coarse_survivors),
        coarse_shortlist_count=len(shortlist),
        coarse_shortlist_rule=(
            "refine every branch inside the fixed serve-depth beam at the higher budget and "
            "reapply every final gate; never fit a branch whose pinned first-contact Y is "
            "outside the serve region, because no budget can move it"
        ),
        refined_candidates=refined,
        selected=selected,
        selected_arms=selected_arms,
        families=families,
        **(
            {}
            if args.shooting_parameterization == "single"
            else {"contact_family_widths": family_widths}
        ),
        diagnostic_candidate=diagnostic,
        **native_seed_check_pixels.selection_from_usages(native_seed_usages),
        heldout_scored_after_selection=args.observation_partition == "fifth_frame_withheld",
        fitted_fifth_frame_check=args.observation_partition == "all_native",
        context_frames=attempt["context_native_frames"],
        context_is_image_only=True,
        native_event_times_changed=False,
        input_only_reconstructed=bool(
            families["combined_toss_and_serve_prior"]["reconstructed_on_family_basis"]
        ),
        complete_point_accepted=False,
        independent_xyz_truth_available=False,
        serve_speed_graphic=speed_document,
    )
    completed, budget_receipt = candidate_attempts.completed_output(
        chosen, budget.completed, budget.receipt()
    )
    if (
        args.search_seconds is not None
        or budget_receipt["retained_completed_after_numerical_failures"]
    ):
        report["completed_candidates"] = completed
        report["search_budget"] = {
            **budget_receipt,
            "implementation_files": budget.context["implementation_files"],
        }
    if args.passive_bounce_response != "off":
        report["passive_bounce_response"] = args.passive_bounce_response
    search_budget.atomic_json(args.output / "report.json", report)
    if not args.skip_render:
        report["artifacts"] = render(report, labels, args.output)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=replay.default, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "coarse_survivors": len(coarse_survivors),
                "coarse_shortlist": len(shortlist),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
