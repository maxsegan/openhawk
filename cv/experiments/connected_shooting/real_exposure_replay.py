"""Replay a whole owner-labeled broadcast attempt with centre/front observation models.

Evaluation only. Fixed duration hypotheses are sensitivity conditions, not measured
camera metadata. Preserve every native label and the frame%5 held-out split. Derive
training directions only from training fronts and separately attributed same-frame
agent trailing tips; otherwise use a <=3frame same-event-wing training secant.
No owner/agent XYZ, new labels, timing edits or production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from copy import deepcopy
from cv.experiments.connected_shooting import observation_operator
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq, least_squares, minimize
from scipy.sparse import lil_matrix

from cv.experiments.connected_shooting.agent_attempt_prepare import FROZEN_AGENT_STATUSES
from cv.experiments.connected_shooting import (
    leading_edge_capacity as leading,
    physical_compatibility,
    player_position,
    search_budget,
    streak_axis_capacity,
    terminal_feasibility as terminal,
)
from cv.pipeline import paths, provenance
from cv.validation import s6_sparse_owner_replay as replay


ARMS = {"centre": None, "front_d025": 0.25, "front_d050": 0.5, "front_d075": 0.75}
BASELINE_SCHEMA = "connected_terminal_feasibility_v1"
DIRECTIONAL_INTERIOR_MARGIN_PX = 0.5
TERMINAL_INTERIOR_MARGIN_FRAMES = 0.01
# Experimental-arm defaults.  With ``inequality_constraints`` true and
# ``anchor_targets`` None the solve below is the promoted reference solve and
# reproduces it exactly; both knobs are set only by an explicit arm flag.
ANCHOR_BOUNCE_SIGMA_FLOOR_M = 0.20
ANCHOR_PLAYER_SIGMA_M = 0.65
SHARED_CONTACT_CONTINUITY_SCALE_M = 1e-4
SHARED_CONTACT_CERTIFICATE_M = 1e-3


def shared_jacobian_sparsity(scene, image_rows: int, residual_rows: int):
    """Block-sparse finite-difference structure for explicit-contact shooting."""
    blocks = leading.model.shared_parameter_slices(scene)
    width = blocks["rebound_scales"].stop
    pattern = lil_matrix((residual_rows, width), dtype=int)
    offset = 0
    for index, frames in enumerate(scene.observation_frames):
        rows = slice(offset, offset + 2 * len(frames))
        contact = slice(3 * index, 3 * index + 3)
        velocity = slice(
            blocks["velocities"].start + 3 * index, blocks["velocities"].start + 3 * index + 3
        )
        spin = slice(blocks["spins"].start + 3 * index, blocks["spins"].start + 3 * index + 3)
        pattern[rows, contact] = 1
        pattern[rows, velocity] = 1
        pattern[rows, spin] = 1
        pattern[rows, blocks["rebound_scales"]] = 1
        offset += 2 * len(frames)
    if offset != image_rows:
        raise ValueError("image residual rows must match the native flight blocks")
    # Event, anchor, continuity and terminal rows are few compared with the
    # image block. Keep them conservative: the simulator's discrete impact
    # topology can couple their row identity even though each value is local.
    pattern[image_rows:, :] = 1
    return pattern.tocsr()


def anchor_plan(
    bounce_targets,
    contact_targets,
    *,
    player_sigma_m=ANCHOR_PLAYER_SIGMA_M,
    closing_contact=None,
):
    """Validate the metric anchors an experimental arm adds to the image objective.

    ``bounce_targets`` is one group per flight of supplied ground-ray witnesses
    (``xyz_m`` and the projection-derived ``uncertainty_sigma_m``/graded radius);
    ``contact_targets`` is one automatic striker court position per flight, or
    None where the arm has no player row.  ``closing_contact`` is the optional
    ``closing_contact_anchor`` row of an original-contact prefix: the striker the
    last flight ends at, which the launching ties above cannot express.  Nothing
    here reads a withheld pixel, an owner XYZ or an acceptance verdict: these are
    the same explicit inputs the unchanged gates are scored against afterwards.
    """
    if not np.isfinite(player_sigma_m) or player_sigma_m <= 0:
        raise ValueError("positive finite player tie sigma required")
    bounces, contacts = [], []
    for group in bounce_targets:
        rows = []
        for target in group:
            if target is None:
                rows.append(None)
                continue
            xy = np.asarray(target["xyz_m"], float)[:2]
            sigma = target.get("graded_circle_radius_m")
            if sigma is None or not np.isfinite(sigma) or sigma <= 0:
                sigma = target.get("uncertainty_sigma_m")
                sigma = None if sigma is None else 2.0 * float(sigma)
            sigma = ANCHOR_BOUNCE_SIGMA_FLOOR_M if sigma is None else float(sigma)
            sigma = max(float(sigma), ANCHOR_BOUNCE_SIGMA_FLOOR_M)
            if not np.isfinite(xy).all():
                raise ValueError("finite bounce ground-ray anchor required")
            rows.append((xy, sigma))
        bounces.append(rows)
    for target in contact_targets:
        if target is None:
            contacts.append(None)
            continue
        xy = np.asarray(target, float)[:2]
        if not np.isfinite(xy).all():
            raise ValueError("finite contact court anchor required")
        contacts.append((xy, float(player_sigma_m)))
    if len(bounces) != len(contacts):
        raise ValueError("one bounce group and one contact anchor per connected flight required")
    plan = {"bounce": bounces, "contact": contacts, "player_sigma_m": float(player_sigma_m)}
    if closing_contact is None:
        return plan
    xy = np.asarray(closing_contact["court_xy_m"], float)
    sigma = float(closing_contact["sigma_m"])
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("finite closing-contact court anchor required")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite closing-contact tie sigma required")
    source = closing_contact.get("source")
    if not isinstance(source, dict) or source.get("contact_index") is None:
        raise ValueError("explicit closing-contact source metadata required")
    if int(source["contact_index"]) != len(contacts):
        raise ValueError("the closing anchor belongs to the contact after the last flight")
    plan["closing_contact"] = {"court_xy_m": xy, "sigma_m": sigma, "source": source}
    return plan


def launch_contact_targets(players):
    """The launching court roots ``anchor_plan`` ties, abstaining where absent.

    One entry per flight: the validated court root of that flight's striker, or
    None where the state declares its position explicitly absent, which
    ``anchor_plan`` already reads as "no root anchor for this flight".  Every
    stage builds the list here so a continuation cannot read a missing root as
    a position, or pass an unvalidated one.
    """
    return [player_position.root_xy(player) for player in players]


def closing_contact_anchor(scene, right_contact_player, *, player_sigma_m=ANCHOR_PLAYER_SIGMA_M):
    """Describe the soft court tie of the contact an original-contact prefix ends at.

    ``right_contact_player`` is the already-associated actor of that closing
    contact, kept apart from the launching states ``anchor_plan``'s
    ``contact_targets`` carry.  Only an ``original_contact`` right boundary owns
    one, so a supplied-end scene raises here instead of quietly gaining a soft
    endpoint tie next to its hard ending.  A missing actor is absent evidence and
    returns None; nothing here chooses a nearest actor, a side or a pixel of its
    own.  An associated actor whose court position is explicitly absent is the
    same absent evidence and also returns None, rather than reading a missing
    root.  The actor's own declared substitution sigma widens the configured
    player sigma and never tightens it.
    """
    if not leading.model.original_contact_boundary(scene):
        raise ValueError("only an original-contact right boundary carries a closing anchor")
    if not np.isfinite(player_sigma_m) or player_sigma_m <= 0:
        raise ValueError("positive finite player tie sigma required")
    if right_contact_player is None:
        return None
    index = len(scene.pixels)
    if len(scene.contact_frames) != index + 1:
        raise ValueError("one contact boundary per flight plus the closing contact required")
    xy = player_position.root_xy(right_contact_player)
    if xy is None:
        return None
    substitution = right_contact_player.get("substitution")
    declared = None if substitution is None else substitution.get("declared_sigma_m")
    sigma = float(player_sigma_m)
    if declared is not None:
        if not np.isfinite(declared) or float(declared) <= 0:
            raise ValueError("a declared substitution sigma must be positive and finite")
        sigma = max(sigma, float(declared))
    frame = right_contact_player.get("frame")
    return {
        "court_xy_m": xy,
        "sigma_m": sigma,
        "source": {
            "contact_index": index,
            "contact_epoch_frame": float(scene.contact_frames[-1]),
            "actor_frame": None if frame is None else int(frame),
            "actor_frame_kind": (
                "direct_native_row" if substitution is None else "nearest_sided_substitute"
            ),
            "substitution_source_frame": (
                None if substitution is None else substitution.get("source_frame")
            ),
            "declared_substitution_sigma_m": None if declared is None else float(declared),
            "configured_player_sigma_m": float(player_sigma_m),
            "root_observation": right_contact_player.get("root_observation"),
        },
    }


def context_closing_anchor(context, *, player_sigma_m=ANCHOR_PLAYER_SIGMA_M):
    """Rebuild a stored case's closing anchor so every stage ties the same target.

    A supplied-end context has no closing contact and returns None, so the
    continuation stages below keep their existing plan unchanged.
    """
    if not leading.model.original_contact_boundary(context["scene"]):
        if context.get("right_contact_player") is not None:
            raise ValueError("a closing actor requires an original-contact boundary")
        return None
    return closing_contact_anchor(
        context["scene"],
        context.get("right_contact_player"),
        player_sigma_m=player_sigma_m,
    )


def directions(scene, bounces, references):
    frames, front = np.concatenate(scene.observation_frames), np.concatenate(scene.pixels)
    tails = {}
    for document in references:
        if document.get("annotation_status") not in FROZEN_AGENT_STATUSES:
            raise ValueError("frozen separately attributed agent reference required")
        for record in document["records"]:
            for row in record["frames"]:
                tip = row.get("streak", {})
                if tip.get("status") == "paired":
                    f = row["frame"]
                    if f in tails:
                        raise ValueError("duplicate trailing-tip source")
                    tails[f] = [tip["trailing"]["x1080"], tip["trailing"]["y1080"]]
    back = np.array([tails.get(f, [np.nan, np.nan]) for f in frames])
    visible = streak_axis_capacity.in_image(front)
    boundaries = np.unique(np.r_[scene.contact_frames, *bounces])
    if scene.terminal_net_tail is not None:
        boundaries = np.unique(np.r_[boundaries, scene.terminal_net_tail["representative"]])
    wings = np.searchsorted(boundaries, frames, side="right")
    axes, available, sources = streak_axis_capacity.observed_axes(
        front, back, frames, visible, streak_axis_capacity.in_image(back), wings
    )
    if not available.all():
        raise ValueError(f"unresolved original front directions at {frames[~available].tolist()}")
    return axes, dict(
        frames=frames.tolist(),
        sources=sources,
        axes=axes.tolist(),
        note="Correlated plug-in axes; no calibrated angular likelihood or truth direction.",
    )


def prediction(
    scene, parameters, axes, duration, cache=None, *, termination_kind="terminal_bounce"
):
    if duration is None:
        return leading.image_prediction(scene, parameters, axes, False, cache)
    imaging = (
        scene
        if termination_kind
        in ("terminal_net_impact", "net_stop", "observed_horizon", "original_contact")
        else leading.context.imaging_scene(scene, 1.0, termination_kind)
    )
    rows = leading.swept.fitted_curves(imaging, parameters, duration, 0.0, cache=cache)
    return leading.swept.predict(imaging, rows, axes, 1.5)[:, 1]


def time_shifted_prediction(
    scene,
    parameters,
    axes,
    duration,
    time_shift_frames,
    *,
    termination_kind="terminal_bounce",
    preserve_observation_horizon=False,
):
    """Shifted projection; a terminal extension that hits the bounce cap is unsupported.

    The one-frame passive ground-ending extension can drive a settling (often
    net-crossing) final flight into the simulated bounce cap even though the
    fitted domain itself propagates. That extension cannot support a shift, so
    the probe falls back to the unextended fitted domain, where the extension
    rows leave the domain and are NaN. Errors inside the domain still raise.
    """
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    arguments = (scene, parameters, axes, duration, time_shift_frames)
    try:
        return _time_shifted_prediction(
            *arguments,
            termination_kind=termination_kind,
            preserve_observation_horizon=preserve_observation_horizon,
        )
    except BounceCapacityError:
        extended = not preserve_observation_horizon and termination_kind not in (
            "terminal_net_impact",
            "net_stop",
            "observed_horizon",
            "original_contact",
        )
        if not extended:
            raise
    return _time_shifted_prediction(
        *arguments, termination_kind=termination_kind, preserve_observation_horizon=True
    )


def _time_shifted_prediction(
    scene,
    parameters,
    axes,
    duration,
    time_shift_frames,
    *,
    termination_kind="terminal_bounce",
    preserve_observation_horizon=False,
):
    """Project the unchanged fitted path at globally shifted model times.

    Cameras, owner pixels, exposure duration and native timestamps remain at
    their original rows. Only the time at which the already-fitted connected
    path is sampled moves. A shifted exposure may cross a contact or bounce;
    propagation therefore uses the whole path rather than refitting one flight.
    Rows that leave the fitted point domain are NaN and cannot support a shift.
    """
    scene.validate()
    # A terminal net-impact fit supplies no post-impact response. Unsupported
    # shifted exposures remain NaN instead of propagating through the mesh.
    if type(preserve_observation_horizon) is not bool:
        raise ValueError("explicit boolean observation-horizon policy required")
    if not preserve_observation_horizon and termination_kind not in (
        "terminal_net_impact",
        "net_stop",
        "observed_horizon",
        "original_contact",
    ):
        scene = leading.context.imaging_scene(scene, 1.0, termination_kind)
    shift = float(time_shift_frames)
    if not np.isfinite(shift) or abs(shift) > 1.0:
        raise ValueError("finite model time shift within one native frame required")
    if duration is not None and (not np.isfinite(duration) or not 0 < duration <= 1):
        raise ValueError("positive finite full-exposure duration required")
    exposure_duration = 0.0 if duration is None else float(duration)
    axes = np.asarray(axes, float)
    rows = [
        {
            "flight": flight,
            "index": index,
            "bounds": np.asarray([frame + shift, frame + shift + exposure_duration], float),
        }
        for flight, frames in enumerate(scene.observation_frames)
        for index, frame in enumerate(frames)
    ]
    if axes.shape != (len(rows), 2):
        raise ValueError("one explicit image axis per original observation required")
    start, end = map(float, np.asarray(scene.contact_frames)[[0, -1]])
    active_indices = [
        index
        for index, row in enumerate(rows)
        if row["bounds"][0] >= start and row["bounds"][1] <= end
    ]
    times = (
        np.unique(
            np.concatenate(
                [
                    np.r_[
                        np.linspace(*rows[index]["bounds"], 9),
                        np.asarray(scene.contact_frames)[
                            (np.asarray(scene.contact_frames) >= rows[index]["bounds"][0])
                            & (np.asarray(scene.contact_frames) <= rows[index]["bounds"][1])
                        ],
                    ]
                    for index in active_indices
                ]
            )
        )
        if active_indices
        else np.empty(0)
    )

    def propagate(query):
        groups = tuple(
            np.unique(np.r_[a, query[(query >= a) & (query <= b)], b])
            for a, b in zip(scene.contact_frames[:-1], scene.contact_frames[1:], strict=True)
        )
        flights = leading.model.chain(scene, parameters, query_frames=groups)
        positions = {
            float(t): xyz
            for group, flight in zip(groups, flights, strict=True)
            for t, xyz in zip(group, flight["positions"], strict=True)
        }
        return flights, positions

    flights, positions = propagate(times)
    knots = np.asarray(
        [
            value
            for flight in flights
            for bounce in flight["bounces"]
            for value in (
                bounce["frame"],
                bounce["frame"] + bounce.get("dwell_seconds", 0) * scene.fps,
            )
        ],
        float,
    )
    knots = knots[(knots >= start) & (knots <= end)]
    if len(knots):
        _, positions = propagate(np.unique(np.r_[times, knots]))
    output = np.full((len(rows), 2), np.nan)
    for row_index in active_indices:
        row, axis = rows[row_index], axes[row_index]
        low, high = row["bounds"]
        sample_frames = np.unique(
            np.r_[
                np.linspace(low, high, 9),
                np.asarray(scene.contact_frames)[
                    (np.asarray(scene.contact_frames) >= low)
                    & (np.asarray(scene.contact_frames) <= high)
                ],
                knots[(knots >= low) & (knots <= high)],
            ]
        )
        xyz = np.asarray([positions[float(frame)] for frame in sample_frames])
        camera = scene.cameras[row["flight"]][row["index"]]
        radial = (
            None
            if scene.camera_distortion is None
            else scene.camera_distortion[row["flight"]][row["index"]]
        )
        if duration is None:
            output[row_index] = leading.camera_geometry.project(
                camera[None], xyz, None if radial is None else radial[None]
            )[0]
        else:
            tips, _ = leading.swept.directional_tips(
                xyz,
                camera,
                axis,
                blur_radius_px=1.5,
                radial=radial,
            )
            output[row_index] = tips[1]
    return output


def terminal_rebound_segment(scene, heldout, bounces, axes):
    """Activate every labeled post-bounce exposure in the terminal fit.

    The ordinary split already fits non-fifth frames.  This adds only the
    previously withheld fifth-frame rows, leaving native timestamps, cameras,
    pixels and the measured surface model untouched.  The returned frame list
    lets the event objective require that the modeled bounce+dwell precede the
    first rebound picture.
    """
    if scene.dynamics != "measured_240hz" or not len(bounces[-1]):
        raise ValueError("terminal rebound needs a supplied measured-dynamics bounce")
    # Context extension can be called after an all-native fit. Remove only
    # exact already-fitted rows from the merge copy, never duplicate exposures.
    from cv.experiments.connected_shooting.labeled_preparation_net_followup import fit_check_copy

    heldout, _ = fit_check_copy(scene, heldout)
    bounce = float(bounces[-1][-1])
    train_frames = np.asarray(scene.observation_frames[-1], float)
    held_frames = np.asarray(heldout.observation_frames[-1], float)
    train_indices = np.flatnonzero(train_frames > bounce)
    activated_indices = np.flatnonzero(held_frames > bounce)
    if not len(train_indices) and not len(activated_indices):
        raise ValueError("terminal rebound has no labeled post-bounce exposure")

    axis_groups = np.split(
        np.asarray(axes, float), np.cumsum([len(row) for row in scene.observation_frames])[:-1]
    )
    added_axes = []
    for frame in held_frames[activated_indices]:
        nearest = int(np.argmin(np.abs(train_frames - frame)))
        added_axes.append(axis_groups[-1][nearest])
    added_axes = np.asarray(added_axes, float).reshape(-1, 2)
    combined_frames = np.r_[train_frames, held_frames[activated_indices]]
    order = np.argsort(combined_frames)

    def replace_last(groups, extra):
        rows = list(groups)
        rows[-1] = np.concatenate([np.asarray(rows[-1]), np.asarray(extra)])[order]
        return tuple(rows)

    distortion = scene.camera_distortion
    held_distortion = heldout.camera_distortion
    if distortion is not None:
        distortion = replace_last(distortion, np.asarray(held_distortion[-1])[activated_indices])
    active_scene = replace(
        scene,
        observation_frames=replace_last(scene.observation_frames, held_frames[activated_indices]),
        cameras=replace_last(scene.cameras, np.asarray(heldout.cameras[-1])[activated_indices]),
        pixels=replace_last(scene.pixels, np.asarray(heldout.pixels[-1])[activated_indices]),
        camera_distortion=distortion,
    )
    active_scene.validate()
    axis_groups[-1] = np.concatenate([axis_groups[-1], added_axes])[order]
    active_axes = np.concatenate(axis_groups)
    rebound_frames = combined_frames[order]
    rebound_frames = rebound_frames[rebound_frames > bounce]
    return (
        active_scene,
        active_axes,
        rebound_frames,
        {
            "mode": "on",
            "supplied_terminal_bounce_frame": bounce,
            "postbounce_labeled_frame_count": int(len(rebound_frames)),
            "postbounce_labeled_frames": rebound_frames.tolist(),
            "already_training_count": int(len(train_indices)),
            "withheld_rows_activated_count": int(len(activated_indices)),
            "withheld_rows_activated_frames": held_frames[activated_indices].tolist(),
            "measured_surface_bounce": True,
            "spin_carried_through_impact": True,
            "native_timestamps_changed": False,
            "pictures_invented": 0,
            "heldout_independence_for_activated_rows": False,
        },
    )


def next_physical_event(labels, clip, after_frame, *, event_types=("contact", "bounce", "net_hit")):
    """Return the earliest possible subsequent raw impact, including uncertain presence."""
    from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

    events = [
        event
        for event in labels["events"]["records"]
        if event.get("clip", clip) == clip
        and event["event_type"] in event_types
        and event.get("occurrence_status") != "absent"
        and (
            event.get("status", "labeled") in ("labeled", "ambiguous")
            or occurrence.predicted_membership(event)
        )
        and float(event["frame"]) > float(after_frame)
    ]
    return (
        min(events, key=lambda event: float(event.get("frame_interval", [event["frame"]])[0]))
        if events
        else None
    )


def before_next_physical_event(
    receipt,
    labels,
    clip,
    terminal_bounce,
    duration_frames,
    *,
    event_types=("contact", "bounce", "net_hit"),
):
    """Bound single-ending rebound evidence before another original physical impact.

    A later court, net or racket impact needs separate event topology. Its native
    pictures remain in the labels; they are not evidence for this uninterrupted
    rebound segment. Bounds come from original intervals, never fitted timing.
    """
    event = next_physical_event(labels, clip, terminal_bounce, event_types=event_types)
    if event is None:
        return receipt
    boundary = float(event.get("frame_interval", [event["frame"]])[0])
    frames = [
        f
        for f in receipt["postbounce_labeled_frames"]
        if f + observation_operator.support_span(duration_frames) < boundary
    ]
    details = {
        "exposure_duration_frames": duration_frames,
        "excluded_frames": [f for f in receipt["postbounce_labeled_frames"] if f not in frames],
        "rule": "native exposure close strictly before original next-impact interval",
    }
    result = {
        **receipt,
        "status": "supported"
        if frames
        else "no_rebound_exposure_before_next_"
        + ("contact" if event["event_type"] == "contact" else "physical_event"),
        "postbounce_labeled_frames": frames,
        "postbounce_labeled_frame_count": len(frames),
        "last_postbounce_labeled_frame": max(frames) if frames else None,
        "next_physical_event_boundary": {"original_event": deepcopy(event), **details},
    }
    if boundary <= terminal_bounce:
        result["status"] = "source_event_order_uncertain"
        result["source_order_conflict"] = {
            "reference_frame": float(terminal_bounce),
            "next_interval_start": boundary,
            "reason": "next raw impact interval overlaps the declared continuation start",
        }
    if event["event_type"] == "contact":
        # Preserve the receipt consumed by the published Rome3 independent check.
        result["next_contact_boundary"] = {"original_contact": deepcopy(event), **details}
    return result


def terminal_rebound_inventory(
    attempt, ball_records, terminal_bounce_frame, *, labels=None, duration_frames=0.25
):
    """Inventory frozen visible context that can extend the terminal flight."""
    window = [row for row in ball_records if row["clip"] == attempt["point_clip"]]
    if len(window) != 1:
        raise ValueError("one matching frozen ball window required for terminal rebound")
    frames = [
        int(row["frame"])
        for row in window[0]["frames"]
        if row["status"] == "visible" and float(row["frame"]) > terminal_bounce_frame
    ]
    receipt = {
        "mode": "on",
        "status": "supported" if frames else "no_labeled_postbounce_frames",
        "postbounce_labeled_frame_count": len(frames),
        "postbounce_labeled_frames": frames,
        "last_postbounce_labeled_frame": None if not frames else max(frames),
        "source": "frozen native label window outside the former terminal endpoint",
    }

    receipt["barrier_inventory_source"] = (
        "raw_label_event_records" if labels is not None else "attempt_event_fallback"
    )
    if labels is None:
        labels = {"events": {"records": attempt.get("events", [])}}
    receipt["barrier_event_records_sha256"] = hashlib.sha256(
        json.dumps(labels["events"]["records"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return before_next_physical_event(
        receipt, labels, attempt["point_clip"], terminal_bounce_frame, duration_frames
    )


def refine(
    scene,
    initial,
    bounces,
    native,
    axes,
    duration,
    maxiter,
    *,
    first_contact_y_m=None,
    first_contact_x_bounds_m=None,
    first_contact_y_bounds_m=None,
    first_contact_z_bounds_m=None,
    first_contact_target=None,
    first_contact_depth_cue=None,
    first_contact_epoch_prior=None,
    interior_contact_epoch_plan=None,
    termination_kind="terminal_bounce",
    directional_event_frames=None,
    directional_horizons=(4, 10),
    directional_rms_limit_px=None,
    directional_interior_margin_px=DIRECTIONAL_INTERIOR_MARGIN_PX,
    parameter_scaling="global",
    optimizer_ftol=1e-8,
    inequality_constraints=True,
    directional_inequalities=True,
    anchor_targets=None,
    bounce_epoch_priors=None,
    shared_contact_states=False,
    local_flight_refits=False,
    bounce_bracket_frames=1.0,
    terminal_rebound_frames=None,
    deadline_check=None,
    retain_deadline_incumbent=False,
    observation_net_seed=None,
    net_seed_candidate=None,
    _net_mesh_response=False,
):
    call_arguments = locals().copy()
    from cv.experiments.connected_shooting.admissible_net_response import widen_image_residual
    from cv.pipeline import s6_rally_origin_cue as rally_cue

    # Validate before entering numerical callbacks; never turn a malformed cue into a zero.
    rally_cue.residuals(np.asarray(initial)[:3], first_contact_depth_cue)
    seed_call = call_arguments.copy() if observation_net_seed is not None else None
    candidate_call = call_arguments.copy() if net_seed_candidate is not None else None
    if observation_net_seed is not None and net_seed_candidate is not None:
        raise ValueError("a prebuilt net response candidate excludes its own initialization")
    n = len(scene.pixels)
    from cv.experiments.connected_shooting.observation_net_seed import single_final_net

    if _net_mesh_response and (not single_final_net(scene) or shared_contact_states):
        raise ValueError("net response domain requires a single-shooting terminal tail")
    if (
        first_contact_epoch_prior is not None or interior_contact_epoch_plan is not None
    ) and shared_contact_states:
        raise ValueError("contact-epoch fitting currently requires single shooting")
    if retain_deadline_incumbent and shared_contact_states:
        raise ValueError("deadline incumbent retention currently requires single shooting")
    # The cooperative deadline is consulted through this one local slot.  When a
    # retained incumbent replaces the aborted optimizer result, the slot is
    # cleared for the remainder of this invocation only, so the ordinary final
    # replay below can finish; the caller's budget object is untouched and the
    # next solve still raises.
    deadline = [deadline_check]

    def check_deadline():
        if deadline[0] is not None:
            deadline[0]()

    incumbent = None
    # A copy: this routine pins and projects the initializer, and the depth beam
    # hands the same seed array to every branch.
    initial = np.array(initial, float)
    if shared_contact_states:
        if scene.parameterization != "shared_contact_states":
            raise ValueError("shared-state refinement requires a shared-state scene")
        initial = leading.model.shared_contact_seed(scene, initial)
        global_scale = np.r_[[10] * (3 * (n + 1)), [30] * (3 * n), [3] * (3 * n), [1, 1]]
    else:
        if scene.parameterization != "single_shooting":
            raise ValueError("single-shooting refinement requires its declared scene")
        global_scale = np.r_[[10] * 3, [30] * (3 * n), [3] * (3 * n), [1, 1]]
    if parameter_scaling == "global":
        scale = global_scale
    elif parameter_scaling == "accepted_candidate_local":
        floor = (
            np.r_[[1] * (3 * (n + 1)), [10] * (3 * n), [1] * (3 * n), [0.1, 0.1]]
            if shared_contact_states
            else np.r_[[1] * 3, [10] * (3 * n), [1] * (3 * n), [0.1, 0.1]]
        )
        scale = np.maximum(np.abs(initial), floor)
    else:
        raise ValueError("supported explicit parameter scaling required")
    if shared_contact_states:
        lo = np.r_[
            np.tile([-10, -15, leading.model.R_BALL], n + 1),
            [-75] * (3 * n),
            [-6] * (3 * n),
            [0.8, 0.8],
        ]
        hi = np.r_[
            np.tile([21, 40, 12], n + 1),
            [75] * (3 * n),
            [6] * (3 * n),
            [1.2, 1.2],
        ]
        # The last boundary is the declared ground ending. Its position is an
        # explicit shared state and its height is exactly one ball radius.
        if len(bounces[-1]) and abs(float(bounces[-1][-1]) - scene.contact_frames[-1]) <= 1e-8:
            lo[3 * n + 2] = hi[3 * n + 2] = leading.model.R_BALL
            initial[3 * n + 2] = leading.model.R_BALL
    else:
        lo = np.r_[[-10, -15, leading.model.R_BALL], [-75] * (3 * n), [-6] * (3 * n), [0.8, 0.8]]
        hi = np.r_[[21, 40, 12], [75] * (3 * n), [6] * (3 * n), [1.2, 1.2]]
    if (
        initial.shape != scale.shape
        or not np.isfinite(initial).all()
        or np.any(initial < lo)
        or np.any(initial > hi)
    ):
        raise ValueError("finite point-scale initializer within unchanged bounds required")
    if first_contact_y_m is not None:
        if (
            isinstance(first_contact_y_m, (bool, np.bool_))
            or not np.isfinite(first_contact_y_m)
            or not lo[1] <= first_contact_y_m <= hi[1]
        ):
            raise ValueError("finite first-contact court-Y hypothesis inside bounds required")
        initial[1] = first_contact_y_m
        lo[1] = hi[1] = first_contact_y_m
    if first_contact_y_m is not None and first_contact_y_bounds_m is not None:
        raise ValueError("fixed and continuous first-contact court-Y are mutually exclusive")
    # A supplied first-contact interval narrows the published bound and never
    # widens it, so every restart below inherits the same cut.  The initializer
    # is projected onto the narrowed bound rather than rejected: an initializer
    # outside the cut is exactly the case the cut exists to move.
    for index, interval in (
        (0, first_contact_x_bounds_m),
        (1, first_contact_y_bounds_m),
        (2, first_contact_z_bounds_m),
    ):
        if interval is None:
            continue
        low, high = (float(value) for value in interval)
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError("ordered finite first-contact interval required")
        lo[index], hi[index] = max(lo[index], low), min(hi[index], high)
        if lo[index] > hi[index]:
            raise ValueError("first-contact interval lies outside the published search bound")
        initial[index] = float(np.clip(initial[index], lo[index], hi[index]))
    if net_seed_candidate is not None:
        from cv.experiments.connected_shooting import observation_net_seed as net_seed

        # A candidate built elsewhere is replayed as it stands: no initializer
        # runs here, and only its own binding admits its response.
        if shared_contact_states or not single_final_net(scene):
            raise ValueError("prebuilt net response replay requires a single-shooting net tail")
        if not isinstance(net_seed_candidate, dict):
            raise ValueError("prebuilt net response candidate must carry its response record")
        bound = {
            key: net_seed_candidate.get(key)
            for key in ("net_response", "net_response_initialization")
        }
        if any(value is None for value in bound.values()):
            raise ValueError("prebuilt net response candidate must carry its response record")
        if scene.terminal_net_tail is None:
            net_seed.require_ground_ending_binding(
                scene, bound["net_response_initialization"], bounces=bounces
            )
        context = net_seed.response_context(bound)
        candidate_call.update(initial=initial, net_seed_candidate=None, _net_mesh_response=True)
        with context:
            result = refine(**candidate_call)
        return result | bound
    if observation_net_seed is not None:
        from cv.experiments.connected_shooting import observation_net_seed as net_seed

        if shared_contact_states:
            raise ValueError("observed net response initialization requires single shooting")
        seed_scene = observation_net_seed["scene"]
        if not np.array_equal(seed_scene.contact_frames, scene.contact_frames):
            raise ValueError("net seed native scene differs from original contact scope")
        seeded, response, receipt = net_seed.initialize(
            seed_scene,
            initial,
            physical_events=observation_net_seed["events"],
            ground_targets=observation_net_seed.get("ground_targets"),
            speed_scale_mps=observation_net_seed["speed_scale_mps"],
            **(
                {"deadline_check": deadline_check}
                if observation_net_seed.get("candidate_family") and deadline_check is not None
                else {}
            ),
        )
        from cv.experiments.connected_shooting import native_seed_check_pixels

        receipt = native_seed_check_pixels.annotate(
            receipt, observation_net_seed.get("reserved_check_frames")
        )
        seed_call.update(initial=seeded, observation_net_seed=None, _net_mesh_response=True)
        bound = dict(net_response=receipt["response"], net_response_initialization=receipt)
        with net_seed.response_context(bound):
            result = refine(**seed_call)
        return result | bound
    contact_target = None
    if first_contact_target is not None:
        contact_xyz, contact_sigma = (np.asarray(value, float) for value in first_contact_target)
        if (
            contact_xyz.shape != (3,)
            or contact_sigma.shape != (3,)
            or not np.isfinite(np.r_[contact_xyz, contact_sigma]).all()
            or np.any(contact_sigma <= 0)
        ):
            raise ValueError("finite toss-contact XYZ and positive per-axis sigma required")
        contact_target = (contact_xyz, contact_sigma)
    epoch_prior = None
    if first_contact_epoch_prior is not None:
        target_epoch = float(first_contact_epoch_prior["target_frame"])
        epoch_sigma = float(first_contact_epoch_prior["sigma_frames"])
        epoch_low, epoch_high = map(float, first_contact_epoch_prior["bounds_frames"])
        if (
            not np.isfinite([target_epoch, epoch_sigma, epoch_low, epoch_high]).all()
            or epoch_sigma <= 0
            or epoch_low > target_epoch
            or target_epoch > epoch_high
            or epoch_high > float(scene.observation_frames[0][0])
            or epoch_high >= float(scene.contact_frames[1])
        ):
            raise ValueError("finite ordered first-contact epoch prior before its flight required")
        epoch_prior = (target_epoch, epoch_sigma, epoch_low, epoch_high)
        initial_epoch = float(first_contact_epoch_prior.get("initial_frame", target_epoch))
        if not epoch_low <= initial_epoch <= epoch_high:
            raise ValueError("first-contact epoch initializer must lie inside its bounds")
        initial = np.r_[initial, initial_epoch]
        scale = np.r_[scale, 1.0]
        lo = np.r_[lo, epoch_low]
        hi = np.r_[hi, epoch_high]
    from cv.experiments.connected_shooting import interior_contact_epochs

    interior_rows = (
        []
        if interior_contact_epoch_plan is None
        else interior_contact_epochs.validate_plan(scene, interior_contact_epoch_plan)
    )
    for row in interior_rows:
        initial = np.r_[initial, interior_contact_epochs.initial_frame(row)]
        scale = np.r_[scale, 1.0]
        lo = np.r_[lo, row["bounds_frames"][0]]
        hi = np.r_[hi, row["bounds_frames"][1]]
    timing_count = int(epoch_prior is not None) + len(interior_rows)
    target = np.concatenate(scene.pixels)
    event_frames = np.asarray(
        [] if directional_event_frames is None else directional_event_frames, float
    )
    observation_frames = np.concatenate(scene.observation_frames)
    if (
        event_frames.ndim != 1
        or not np.isfinite(event_frames).all()
        or any(not isinstance(value, int) or value < 1 for value in directional_horizons)
    ):
        raise ValueError("finite directional events and positive integer horizons required")

    directional_masks = []
    if directional_rms_limit_px is not None and (
        not np.isfinite(directional_rms_limit_px) or directional_rms_limit_px <= 0
    ):
        raise ValueError("positive finite directional RMS limit required")
    if (
        not np.isfinite(directional_interior_margin_px)
        or directional_interior_margin_px < 0
        or directional_rms_limit_px is not None
        and directional_interior_margin_px >= directional_rms_limit_px
    ):
        raise ValueError("finite directional interior margin inside the final limit required")
    if not np.isfinite(optimizer_ftol) or optimizer_ftol <= 0:
        raise ValueError("positive finite optimizer tolerance required")
    if directional_rms_limit_px is not None:
        for event_frame in event_frames:
            for horizon in directional_horizons:
                for low, high in (
                    (event_frame - horizon, event_frame),
                    (event_frame, event_frame + horizon),
                ):
                    mask = (low <= observation_frames) & (observation_frames <= high)
                    if np.any(mask):
                        directional_masks.append(mask)

    if terminal_rebound_frames is not None:
        terminal_rebound_frames = leading.event_constraints.validate_terminal_rebound(
            scene, bounces, terminal_rebound_frames
        )

    loss = leading.event_constraints.mixed_loss(2 * len(target))
    cache = leading.flight_cache.FlightCache()
    last = float(native[-1][-1])
    if anchor_targets is not None:
        if len(anchor_targets["bounce"]) != n:
            raise ValueError("one anchor group per connected flight required")
        if anchor_targets.get("closing_contact") is not None and not (
            leading.model.original_contact_boundary(scene)
        ):
            raise ValueError("a closing-contact anchor requires an original-contact boundary")
    epoch_priors = validate_bounce_epoch_priors(scene, bounces, bounce_epoch_priors)

    def active_state(q):
        """Decode the optimizer vector and, only for the opt-in arm, its epoch."""
        values = np.asarray(q, float) * scale
        if not timing_count:
            return values, scene, float(scene.contact_frames[0])
        physical = values[:-timing_count]
        contact_frames = np.asarray(scene.contact_frames, float).copy()
        fitted_times = values[-timing_count:]
        offset = 0
        if epoch_prior is not None:
            contact_frames[0] = fitted_times[0]
            offset = 1
        for row, epoch in zip(interior_rows, fitted_times[offset:], strict=True):
            contact_frames[row["contact_index"]] = epoch
        return physical, replace(scene, contact_frames=contact_frames), float(contact_frames[0])

    def anchor_residual(flights):
        """Metric pull toward the supplied bounce rays and striker court positions.

        Fixed length: a flight that has not yet produced its expected impact
        contributes zeros here, because the existing bounce-timing residual is
        what supplies that gradient.  These entries sit after the image block, so
        the mixed loss keeps them quadratic and cannot discount them as outliers.
        An original-contact prefix adds two more: the same horizontal tie for the
        closing striker the last flight ends at.  The endpoint height, the
        endpoint epoch and every acceptance gate stay where they were.
        """
        values = []
        for flight, group, contact in zip(
            flights, anchor_targets["bounce"], anchor_targets["contact"], strict=True
        ):
            observed = flight["bounces"]
            for index, row in enumerate(group):
                if row is None:
                    continue
                if index < len(observed):
                    xy, sigma = row
                    values.extend(((np.asarray(observed[index]["x"], float)[:2] - xy) / sigma))
                else:
                    values.extend((0.0, 0.0))
            if contact is not None:
                xy, sigma = contact
                values.extend(((np.asarray(flight["start_xyz"], float)[:2] - xy) / sigma))
        closing = anchor_targets.get("closing_contact")
        if closing is not None:
            values.extend(
                (np.asarray(flights[-1]["end_xyz"], float)[:2] - closing["court_xy_m"])
                / closing["sigma_m"]
            )
        return np.asarray(values, float)

    def objective_residual(q):
        check_deadline()
        try:
            p, active_scene, epoch = active_state(q)
            image = (
                prediction(
                    active_scene, p, axes, duration, cache, termination_kind=termination_kind
                )
                - target
            )
            image = widen_image_residual(image, active_scene)
            flights, physical, _ = leading.event_constraints.evaluate(
                active_scene,
                p,
                bounces,
                bounce_bracket_frames,
                simulation_cache=cache,
                terminal_rebound_frames=terminal_rebound_frames,
            )
            residual = np.r_[image.ravel(), physical, (p[-2:] - 1) / 0.02]
            if contact_target is not None:
                residual = np.r_[residual, (p[:3] - contact_target[0]) / contact_target[1]]
            if anchor_targets is not None:
                residual = np.r_[residual, anchor_residual(flights)]
            if epoch_priors:
                residual = np.r_[residual, bounce_epoch_prior_residuals(flights, epoch_priors)]
            if shared_contact_states:
                residual = np.r_[
                    residual,
                    leading.model.shared_contact_gaps(flights).ravel()
                    / SHARED_CONTACT_CONTINUITY_SCALE_M,
                ]
            if epoch_prior is not None:
                residual = np.r_[residual, (epoch - epoch_prior[0]) / epoch_prior[1]]
            if interior_rows:
                residual = np.r_[
                    residual,
                    [
                        (active_scene.contact_frames[row["contact_index"]] - row["nominal_frame"])
                        / row["sigma_frames"]
                        for row in interior_rows
                    ],
                ]
            if first_contact_depth_cue is not None:
                residual = np.r_[residual, rally_cue.residuals(p[:3], first_contact_depth_cue)]
            return residual if np.isfinite(residual).all() else None
        except (ValueError, FloatingPointError, OverflowError):
            return None

    def mesh_slack_for(active_scene, parameters):
        if not _net_mesh_response:
            return np.empty(0)
        from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
            mesh_response_slacks_m,
        )

        try:
            flights = leading.model.chain(active_scene, parameters, simulation_cache=cache)
            values = mesh_response_slacks_m(active_scene, flights, include_supplied_final=True)
            return values if np.isfinite(values).all() else np.array([-1e6])
        except (ValueError, FloatingPointError, OverflowError):
            return np.array([-1e6])

    def mesh_slack(q):
        p, active_scene, _ = active_state(q)
        return mesh_slack_for(active_scene, p)

    def cost(q):
        residual = objective_residual(q)
        if residual is None:
            return 1e12
        value = 2 * np.sum(loss((residual / 2) ** 2)[0])
        value = float(value) if np.isfinite(value) else 1e12
        if incumbent is not None and np.all(mesh_slack(q) >= -1e-6):
            incumbent.offer(q, value)
        return value

    def slack(q):
        check_deadline()
        if not len(bounces[-1]) and not inequality_constraints:
            return np.zeros(3)
        try:
            p, active_scene, _ = active_state(q)
            if terminal_rebound_frames is not None:
                return (
                    leading.event_constraints.terminal_rebound_slack(
                        active_scene,
                        p,
                        bounces,
                        terminal_rebound_frames,
                        simulation_cache=cache,
                    )
                    - terminal.NUMERICAL_INTERIOR_FRAMES
                )
            values = (
                terminal.terminal_slack(
                    active_scene, p, bounces, last, cache=cache, duration=duration
                )
                - terminal.NUMERICAL_INTERIOR_FRAMES
            )
            if directional_masks:
                values[0] -= TERMINAL_INTERIOR_MARGIN_FRAMES
            return values
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(3, -1e6)

    def directional_slack(q):
        check_deadline()
        try:
            p, active_scene, _ = active_state(q)
            image = (
                prediction(
                    active_scene,
                    p,
                    axes,
                    duration,
                    cache,
                    termination_kind=termination_kind,
                )
                - target
            )
            image = widen_image_residual(image, active_scene)
            return np.asarray(
                [
                    (directional_rms_limit_px - directional_interior_margin_px) ** 2
                    - float(np.mean(np.sum(image[mask] ** 2, axis=1)))
                    for mask in directional_masks
                ]
            )
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(len(directional_masks), -1e6)

    before = cost(initial / scale)
    if before >= 1e12:
        raise ValueError("initializer cannot supply complete observation support")
    constraints = [dict(type="ineq", fun=slack)]
    if directional_masks:
        constraints.append(dict(type="ineq", fun=directional_slack))
    # The `inequalities out` arm hands SLSQP the same cost, bounds, scaling and
    # budget without the terminal/direction inequalities.  `terminal_only` keeps
    # the terminal-slack physics condition, which a seed rarely violates, and
    # drops only the acceptance gate written into the optimizer as a per-window
    # 15.5 px inequality.  Every gate is still evaluated below and judged,
    # unchanged, at acceptance.
    if not inequality_constraints:
        constraints = []
    elif not directional_inequalities:
        constraints = [dict(type="ineq", fun=slack)]
    if _net_mesh_response:
        constraints.append(dict(type="ineq", fun=mesh_slack))
    shared_optimizer = None
    local_refit_receipts = []
    deadline_incumbent = None
    if shared_contact_states:
        q0, qlo, qhi = initial / scale, lo / scale, hi / scale
        fixed = qlo == qhi
        free = np.flatnonzero(~fixed)

        def expand(q):
            full = q0.copy()
            full[free] = q
            return full

        base_residual = objective_residual(q0)
        if base_residual is None:
            raise ValueError("shared-state initializer cannot supply residual support")

        def sparse_residual(q):
            full = expand(q)
            residual = objective_residual(full)
            if residual is None:
                residual = np.full_like(base_residual, 1e6)
            terminal_residual = (
                np.minimum(slack(full), 0.0) / 0.01 if inequality_constraints else np.empty(0)
            )
            directional_residual = (
                np.sqrt(np.maximum(-directional_slack(full), 0.0))
                if inequality_constraints and directional_inequalities and directional_masks
                else np.empty(0)
            )
            return np.r_[residual, terminal_residual, directional_residual]

        initial_sparse_residual = sparse_residual(q0[free])
        sparsity = shared_jacobian_sparsity(scene, 2 * len(target), len(initial_sparse_residual))[
            :, free
        ]
        result = least_squares(
            sparse_residual,
            q0[free],
            bounds=(qlo[free], qhi[free]),
            loss=leading.event_constraints.mixed_loss(2 * len(target)),
            f_scale=2.0,
            x_scale="jac",
            jac_sparsity=sparsity,
            max_nfev=maxiter,
            ftol=optimizer_ftol,
            xtol=optimizer_ftol,
            gtol=optimizer_ftol,
        )
        fitted_q = expand(result.x)
        if local_flight_refits:
            blocks = leading.model.shared_parameter_slices(scene)

            def sparse_cost(full):
                residual = sparse_residual(full[free])
                return float(2 * np.sum(loss((residual / 2) ** 2)[0]))

            for flight_index in range(n):
                local_indices = np.r_[
                    np.arange(
                        blocks["velocities"].start + 3 * flight_index,
                        blocks["velocities"].start + 3 * flight_index + 3,
                    ),
                    np.arange(
                        blocks["spins"].start + 3 * flight_index,
                        blocks["spins"].start + 3 * flight_index + 3,
                    ),
                ]
                local_free = np.asarray(
                    [np.flatnonzero(free == index)[0] for index in local_indices], int
                )
                fixed_full = fitted_q.copy()

                def expand_local(local):
                    full = fixed_full.copy()
                    full[local_indices] = local
                    return full

                def local_residual(local):
                    full = expand_local(local)
                    residual = objective_residual(full)
                    if residual is None:
                        residual = np.full_like(base_residual, 1e6)
                    terminal_residual = (
                        np.minimum(slack(full), 0.0) / 0.01
                        if inequality_constraints
                        else np.empty(0)
                    )
                    directional_residual = (
                        np.sqrt(np.maximum(-directional_slack(full), 0.0))
                        if inequality_constraints and directional_inequalities and directional_masks
                        else np.empty(0)
                    )
                    return np.r_[residual, terminal_residual, directional_residual]

                starts = (
                    ("joint_optimum", fixed_full[local_indices]),
                    (
                        "incoming_seed",
                        q0[local_indices],
                    ),
                )
                trials = []
                for name, start in starts:
                    local = least_squares(
                        local_residual,
                        start,
                        bounds=(qlo[local_indices], qhi[local_indices]),
                        loss=leading.event_constraints.mixed_loss(2 * len(target)),
                        f_scale=2.0,
                        x_scale="jac",
                        jac_sparsity=sparsity[:, local_free],
                        max_nfev=min(maxiter, 60),
                        ftol=optimizer_ftol,
                        xtol=optimizer_ftol,
                        gtol=optimizer_ftol,
                    )
                    candidate = expand_local(local.x)
                    gaps = leading.model.shared_contact_gaps(
                        leading.model.chain(scene, candidate * scale)
                    )
                    trials.append(
                        {
                            "name": name,
                            "parameters": candidate,
                            "cost": sparse_cost(candidate),
                            "maximum_endpoint_gap_m": float(np.linalg.norm(gaps, axis=1).max()),
                            "function_evaluations": int(local.nfev),
                        }
                    )
                eligible = [
                    row
                    for row in trials
                    if row["maximum_endpoint_gap_m"] <= SHARED_CONTACT_CERTIFICATE_M
                ]
                chosen = min(eligible, key=lambda row: row["cost"], default=None)
                before_local = sparse_cost(fitted_q)
                if chosen is not None and chosen["cost"] < before_local:
                    previous = fitted_q.copy()
                    fitted_q = chosen["parameters"]
                    outside = np.ones(len(fitted_q), bool)
                    outside[local_indices] = False
                    unchanged = bool(np.array_equal(previous[outside], fitted_q[outside]))
                else:
                    unchanged = True
                local_refit_receipts.append(
                    {
                        "flight_index": flight_index,
                        "selected": None
                        if chosen is None or chosen["cost"] >= before_local
                        else chosen["name"],
                        "cost_before": before_local,
                        "cost_after": sparse_cost(fitted_q),
                        "nonlocal_parameters_bit_unchanged": unchanged,
                        "trials": [
                            {key: value for key, value in row.items() if key != "parameters"}
                            for row in trials
                        ],
                    }
                )
        fitted = SimpleNamespace(
            x=fitted_q,
            success=result.success,
            status=result.status,
            message=result.message,
            nit=result.nfev,
        )
        shared_optimizer = {
            "method": "scipy_least_squares_trf",
            "function_evaluations": int(result.nfev),
            "jacobian_evaluations": int(result.njev),
            "optimality": float(result.optimality),
            "jacobian_shape": list(sparsity.shape),
            "jacobian_nonzero_fraction": float(sparsity.nnz / np.prod(sparsity.shape)),
            "fixed_parameter_indices": np.flatnonzero(fixed).tolist(),
        }
    else:
        if retain_deadline_incumbent:
            incumbent = search_budget.Incumbent(before, lo / scale, hi / scale)
        try:
            fitted = minimize(
                cost,
                initial / scale,
                method="SLSQP",
                bounds=list(zip(lo / scale, hi / scale)),
                constraints=constraints,
                options=dict(maxiter=maxiter, ftol=optimizer_ftol),
            )
        except search_budget.SearchDeadline:
            if incumbent is None or not incumbent.improved:
                raise
            deadline_incumbent = incumbent.receipt()
            deadline_incumbent["deadline_message"] = (
                "cooperative search deadline exhausted inside the optimizer"
            )
            deadline[0] = None
            fitted = SimpleNamespace(
                x=incumbent.vector,
                success=False,
                status=search_budget.SEARCH_DEADLINE_INCUMBENT_STATUS,
                message=(
                    "search deadline exhausted; best finite same-invocation iterate "
                    "returned, not a converged optimum"
                ),
                nit=0,
            )
    fitted_values = fitted.x * scale
    p, fitted_scene, fitted_epoch = active_state(fitted.x)
    physical_lo, physical_hi, physical_scale = (
        (lo, hi, scale)
        if not timing_count
        else (lo[:-timing_count], hi[:-timing_count], scale[:-timing_count])
    )

    def optimizer_coordinates(parameters):
        values = np.asarray(parameters, float) / physical_scale
        return values if not timing_count else np.r_[values, fitted_values[-timing_count:]]

    # A candidate can satisfy every declared final gate while SLSQP reports a
    # line-search failure against the deliberately tighter 15.5 px / terminal
    # interior targets.  In that case run a separate local-scaled feasibility
    # problem at the actual final limits.  Its zero-at-start displacement
    # objective asks only whether the accepted point is genuinely feasible; it
    # does not relabel a failed optimum or loosen a gate.
    feasibility = None
    final_floor = (
        np.r_[[1] * (3 * (n + 1)), [10] * (3 * n), [1] * (3 * n), [0.1, 0.1]]
        if shared_contact_states
        else np.r_[[1] * 3, [10] * (3 * n), [1] * (3 * n), [0.1, 0.1]]
    )
    final_scale = np.maximum(np.abs(p), final_floor)
    final_start = p / final_scale

    def final_terminal_slack(q):
        if not len(bounces[-1]) and not inequality_constraints:
            return np.zeros(3)
        try:
            if terminal_rebound_frames is not None:
                return (
                    leading.event_constraints.terminal_rebound_slack(
                        fitted_scene,
                        q * final_scale,
                        bounces,
                        terminal_rebound_frames,
                        simulation_cache=cache,
                    )
                    - terminal.NUMERICAL_INTERIOR_FRAMES
                )
            return (
                terminal.terminal_slack(
                    fitted_scene, q * final_scale, bounces, last, cache=cache, duration=duration
                )
                - terminal.NUMERICAL_INTERIOR_FRAMES
            )
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(3, -1e6)

    def final_directional_slack(q):
        try:
            image = (
                prediction(
                    fitted_scene,
                    q * final_scale,
                    axes,
                    duration,
                    cache,
                    termination_kind=termination_kind,
                )
                - target
            )
            image = widen_image_residual(image, fitted_scene)
            return np.asarray(
                [
                    directional_rms_limit_px**2 - float(np.mean(np.sum(image[mask] ** 2, axis=1)))
                    for mask in directional_masks
                ]
            )
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(len(directional_masks), -1e6)

    def final_mesh_slack(q):
        return mesh_slack_for(fitted_scene, q * final_scale)

    final_terminal_at_primary = final_terminal_slack(final_start)
    final_directional_at_primary = final_directional_slack(final_start)
    primary_final_feasible = bool(
        # The restart may begin inside the published terminal bound while
        # missing only the stricter numerical interior.  The certificate below
        # must still solve final_terminal_slack >= 0 cleanly.
        np.all(final_terminal_at_primary + terminal.NUMERICAL_INTERIOR_FRAMES >= 0)
        and (not len(final_directional_at_primary) or np.all(final_directional_at_primary >= 0))
        and np.all(final_mesh_slack(final_start) >= -1e-6)
    )
    # A deadline return permits replay and measurement only. The ordinary
    # feasibility restorations below perform additional numerical search, so
    # they must not restart optimization after the shared deadline has expired.
    if deadline_incumbent is None and not fitted.success and primary_final_feasible:
        if np.any(final_terminal_at_primary < 0):
            # A candidate can land a few numerical microframes outside the
            # optimizer interior while remaining inside the published ending
            # bound.  Correct only the final outgoing vertical velocity with a
            # bracketed scalar root, then recheck every final constraint.
            parameter_index = (
                leading.model.shared_parameter_slices(scene)["velocities"].start + 3 * (n - 1) + 2
                if shared_contact_states
                else 3 + 3 * (n - 1) + 2
            )
            target_slack = 1e-8

            def scalar_terminal_slack(delta):
                candidate = p.copy()
                candidate[parameter_index] += delta
                return float(
                    final_terminal_slack(candidate / final_scale)[
                        int(np.argmin(final_terminal_at_primary))
                    ]
                    - target_slack
                )

            base_value = scalar_terminal_slack(0.0)
            bracket = None
            for magnitude in np.logspace(-8, -1, 8):
                for delta in (-float(magnitude), float(magnitude)):
                    candidate_value = p[parameter_index] + delta
                    if (
                        not physical_lo[parameter_index]
                        <= candidate_value
                        <= physical_hi[parameter_index]
                    ):
                        continue
                    value = scalar_terminal_slack(delta)
                    if np.isfinite(value) and value * base_value <= 0:
                        bracket = (min(0.0, delta), max(0.0, delta))
                        break
                if bracket is not None:
                    break
            if bracket is not None:
                delta, root = brentq(
                    scalar_terminal_slack,
                    *bracket,
                    xtol=1e-12,
                    full_output=True,
                )
                delta = float(delta)
                candidate = p.copy()
                candidate[parameter_index] += delta
                candidate_terminal = final_terminal_slack(candidate / final_scale)
                candidate_directional = final_directional_slack(candidate / final_scale)
                success = bool(
                    np.all(candidate_terminal >= 0)
                    and (not len(candidate_directional) or np.all(candidate_directional >= 0))
                    and np.all(final_mesh_slack(candidate / final_scale) >= -1e-6)
                )
                feasibility = {
                    "method": "brentq_final_outgoing_vertical_velocity",
                    "success": success,
                    "status": 0 if success else 1,
                    "message": (
                        "Bracketed scalar terminal-interior root converged"
                        if success
                        else "Scalar root violated another final constraint"
                    ),
                    "iterations": int(root.iterations),
                    "function_evaluations": int(root.function_calls),
                    "parameter_index": parameter_index,
                    "primary_terminal_slack_at_published_limits": (
                        final_terminal_at_primary + terminal.NUMERICAL_INTERIOR_FRAMES
                    ).tolist(),
                    "terminal_slack_at_final_limits": candidate_terminal.tolist(),
                    "directional_slack_px2_at_final_limit": candidate_directional.tolist(),
                    "maximum_physical_parameter_change": abs(delta),
                }
                if success:
                    p = candidate

    if (
        deadline_incumbent is None
        and not fitted.success
        and primary_final_feasible
        and feasibility is None
    ):
        final_constraints = [dict(type="ineq", fun=final_terminal_slack)]
        if directional_masks:
            final_constraints.append(dict(type="ineq", fun=final_directional_slack))
        if _net_mesh_response:
            final_constraints.append(dict(type="ineq", fun=final_mesh_slack))
        certified = minimize(
            lambda q: float(np.sum((q - final_start) ** 2)),
            final_start,
            method="SLSQP",
            bounds=list(zip(physical_lo / final_scale, physical_hi / final_scale)),
            constraints=final_constraints,
            options=dict(maxiter=maxiter, ftol=1e-12),
        )
        certified_terminal = final_terminal_slack(certified.x)
        certified_directional = final_directional_slack(certified.x)
        certified_mesh = bool(np.all(final_mesh_slack(certified.x) >= -1e-6))
        feasibility = {
            "method": "SLSQP_local_parameter_scaling_zero_displacement",
            "success": bool(certified.success and certified_mesh),
            "status": int(certified.status),
            "message": str(certified.message),
            "iterations": int(certified.nit),
            "primary_terminal_slack_at_published_limits": (
                final_terminal_at_primary + terminal.NUMERICAL_INTERIOR_FRAMES
            ).tolist(),
            "terminal_slack_at_final_limits": certified_terminal.tolist(),
            "directional_slack_px2_at_final_limit": certified_directional.tolist(),
            "maximum_physical_parameter_change": float(
                np.max(np.abs(certified.x * final_scale - p))
            ),
        }
        if certified.success and certified_mesh:
            p = certified.x * final_scale
    if _net_mesh_response and np.any(mesh_slack_for(fitted_scene, p) < -1e-6):
        raise ValueError("seeded net candidate passes through below-cord mesh")
    resolved_message = (
        (
            "Final-gate feasibility certified with bracketed scalar restart"
            if feasibility["method"] == "brentq_final_outgoing_vertical_velocity"
            else "Final-gate feasibility certified with local parameter scaling"
        )
        if feasibility and feasibility["success"]
        else str(fitted.message)
    )
    if deadline_incumbent is not None:
        # The certificate speaks to final-gate feasibility of the returned point,
        # exactly as for any other unconverged optimizer exit; the top-level
        # message must still say the point is a deadline iterate.
        resolved_message = f"search deadline incumbent, not a converged optimum; {resolved_message}"
    interior_receipt = None
    if interior_contact_epoch_plan is not None:
        from copy import deepcopy

        interior_receipt = deepcopy(interior_contact_epoch_plan)
        interior_receipt["fitted_contact_frames"] = fitted_scene.contact_frames.tolist()
        for row in interior_receipt["contacts"]:
            fitted_time = float(fitted_scene.contact_frames[row["contact_index"]])
            row.update(
                initial_frame=interior_contact_epochs.initial_frame(row),
                fitted_frame=fitted_time,
                shift_frames=fitted_time - row["nominal_frame"],
                soft_prior_residual=0.0
                if row["status"] == "frozen"
                else (fitted_time - row["nominal_frame"]) / row["sigma_frames"],
            )
    return dict(
        parameters=p.tolist(),
        **(
            {rally_cue.FIELD: first_contact_depth_cue}
            if first_contact_depth_cue is not None
            else {}
        ),
        **(
            {interior_contact_epochs.FIELD: interior_receipt}
            if interior_receipt is not None
            else {}
        ),
        success=bool(fitted.success or feasibility and feasibility["success"]),
        status=0 if feasibility and feasibility["success"] else int(fitted.status),
        message=resolved_message,
        iterations=int(fitted.nit) + (0 if feasibility is None else int(feasibility["iterations"])),
        initial_cost=before,
        final_cost=cost(optimizer_coordinates(p)),
        terminal_slack=slack(optimizer_coordinates(p)).tolist(),
        **(
            {}
            if epoch_prior is None
            else {
                "first_contact_epoch_fit": {
                    "nominal_frame": epoch_prior[0],
                    "fitted_frame": fitted_epoch,
                    "shift_frames": fitted_epoch - epoch_prior[0],
                    "bounds_frames": [epoch_prior[2], epoch_prior[3]],
                    "soft_prior_sigma_frames": epoch_prior[1],
                    "soft_prior_residual": (fitted_epoch - epoch_prior[0]) / epoch_prior[1],
                    "native_exposure_times_changed": False,
                }
            }
        ),
        first_contact_y_hypothesis_m=first_contact_y_m,
        first_contact_x_bounds_m=None
        if first_contact_x_bounds_m is None
        else [float(lo[0]), float(hi[0])],
        **(
            {}
            if first_contact_y_bounds_m is None
            else {"first_contact_y_bounds_m": [float(lo[1]), float(hi[1])]}
        ),
        first_contact_z_bounds_m=None
        if first_contact_z_bounds_m is None
        else [float(lo[2]), float(hi[2])],
        directional_window_balancing=bool(len(event_frames)),
        directional_event_frames=event_frames.tolist(),
        directional_horizons=list(directional_horizons),
        directional_rms_limit_px=directional_rms_limit_px,
        directional_interior_margin_px=directional_interior_margin_px if directional_masks else 0.0,
        terminal_interior_margin_frames=TERMINAL_INTERIOR_MARGIN_FRAMES
        if directional_masks
        else 0.0,
        directional_slack_px2=(
            directional_slack(optimizer_coordinates(p)).tolist() if directional_masks else []
        ),
        parameter_scaling=parameter_scaling,
        optimizer_ftol=optimizer_ftol,
        primary_optimizer={
            "success": bool(fitted.success),
            "status": int(fitted.status),
            "message": str(fitted.message),
            "iterations": int(fitted.nit),
            **({} if shared_optimizer is None else shared_optimizer),
            **(
                {}
                if deadline_incumbent is None
                else {"returned_iterate": deadline_incumbent["returned_iterate"]}
            ),
        },
        **({} if deadline_incumbent is None else {"search_deadline_incumbent": deadline_incumbent}),
        final_gate_feasibility=feasibility,
        **(
            {}
            if not epoch_priors
            else {
                "bounce_epoch_priors": bounce_epoch_prior_receipt(
                    leading.event_constraints.evaluate(
                        fitted_scene, p, bounces, bounce_bracket_frames, simulation_cache=cache
                    )[0],
                    epoch_priors,
                )
            }
        ),
        **(
            {}
            if terminal_rebound_frames is None
            else {
                "terminal_rebound_segment": leading.event_constraints.evaluate(
                    fitted_scene,
                    p,
                    bounces,
                    bounce_bracket_frames,
                    simulation_cache=cache,
                    terminal_rebound_frames=terminal_rebound_frames,
                )[2][-1]["terminal_rebound"]
            }
        ),
        **(
            {}
            if not shared_contact_states
            else {
                "shared_contact_states": {
                    "enabled": True,
                    "contact_count": n + 1,
                    "exact_shared_start_parameter": True,
                    "continuity_scale_m": SHARED_CONTACT_CONTINUITY_SCALE_M,
                    "certificate_limit_m": SHARED_CONTACT_CERTIFICATE_M,
                    "endpoint_minus_shared_state_m": leading.model.shared_contact_gaps(
                        leading.model.chain(scene, p)
                    ).tolist(),
                    "maximum_endpoint_gap_m": float(
                        np.linalg.norm(
                            leading.model.shared_contact_gaps(leading.model.chain(scene, p)), axis=1
                        ).max()
                    ),
                    "soft_player_tie_m": ANCHOR_PLAYER_SIGMA_M,
                    "velocity_shared_across_racket": False,
                    "local_flight_refits": local_refit_receipts,
                    "rejected_flight_policy": (
                        "per-flight acceptance may drop a rejected flight as a named gap; "
                        "local refits never move shared contacts or neighbouring flight parameters"
                    ),
                }
            }
        ),
        **(
            {}
            if contact_target is None
            else {
                "toss_contact_constraint": {
                    "target_xyz_m": contact_target[0].tolist(),
                    "sigma_m": contact_target[1].tolist(),
                    "residual_sigma": ((p[:3] - contact_target[0]) / contact_target[1]).tolist(),
                    "soft_residual": True,
                    "hard_bound_kind": "parameter_bounds_not_optimizer_inequality",
                }
            }
        ),
        # Present only when an explicit arm flag moved something.  With both
        # knobs at their defaults this key is absent and the receipt is the
        # promoted reference receipt.
        **(
            {}
            if inequality_constraints and directional_inequalities and anchor_targets is None
            else {
                "experimental_arm": arm_receipt(
                    fitted_scene,
                    p,
                    bounces,
                    cache,
                    inequality_constraints,
                    anchor_targets,
                    directional_inequalities=directional_inequalities,
                )
            }
        ),
    )


def validate_bounce_epoch_priors(scene, bounces, priors) -> list[dict]:
    """Soft timing priors on named supplied bounces, e.g. a picture-only impact epoch.

    Each row names one supplied bounce (flight, ordinal) and a finite epoch with a
    positive sigma in native frames.  The epoch must stay inside that flight.
    """
    if not priors:
        return []
    rows = []
    for row in priors:
        flight, ordinal = int(row["flight_index"]), int(row["ordinal"])
        frame, sigma = float(row["frame"]), float(row["sigma_frames"])
        if not 0 <= flight < len(scene.pixels) or not 0 <= ordinal < len(bounces[flight]):
            raise ValueError("a bounce epoch prior must name a supplied bounce")
        if not np.isfinite(frame) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("a bounce epoch prior needs a finite epoch and positive sigma")
        if not scene.contact_frames[flight] < frame <= scene.contact_frames[flight + 1]:
            raise ValueError("a bounce epoch prior must lie inside its flight")
        rows.append(dict(flight_index=flight, ordinal=ordinal, frame=frame, sigma_frames=sigma))
    return rows


def bounce_epoch_prior_residuals(flights, priors) -> np.ndarray:
    """Fixed length: a missing modelled impact contributes zero, as the anchors do;
    the existing missing-ground residual supplies that gradient."""
    values = []
    for row in priors:
        observed = flights[row["flight_index"]]["bounces"]
        values.append(
            0.0
            if row["ordinal"] >= len(observed)
            else (float(observed[row["ordinal"]]["frame"]) - row["frame"]) / row["sigma_frames"]
        )
    return np.asarray(values, float)


def bounce_epoch_prior_receipt(flights, priors) -> list[dict]:
    rows = []
    for row in priors:
        observed = flights[row["flight_index"]]["bounces"]
        modeled = (
            None if row["ordinal"] >= len(observed) else float(observed[row["ordinal"]]["frame"])
        )
        rows.append(
            {
                **row,
                "modeled_frame": modeled,
                "residual_sigma": None
                if modeled is None
                else (modeled - row["frame"]) / row["sigma_frames"],
            }
        )
    return rows


def arm_receipt(
    scene,
    parameters,
    bounces,
    cache,
    inequality_constraints,
    anchor_targets,
    *,
    directional_inequalities=True,
):
    """Name the arm and score its anchors at the fitted point, in metres."""
    receipt = {
        "inequality_constraints": (
            ("in" if directional_inequalities else "terminal_only")
            if inequality_constraints
            else "out"
        ),
        "anchor_residuals": "absent" if anchor_targets is None else "present",
        "acceptance_gates_unchanged": True,
    }
    if anchor_targets is None:
        return receipt
    flights, _, _ = leading.event_constraints.evaluate(
        scene, parameters, bounces, 1, simulation_cache=cache
    )
    bounce_errors, bounce_sigmas, contact_errors = [], [], []
    for flight, group, contact in zip(
        flights, anchor_targets["bounce"], anchor_targets["contact"], strict=True
    ):
        observed = flight["bounces"]
        row = []
        for index, target in enumerate(group):
            if target is None or index >= len(observed):
                row.append(None)
                continue
            xy, sigma = target
            row.append(float(np.linalg.norm(np.asarray(observed[index]["x"], float)[:2] - xy)))
            bounce_sigmas.append(float(sigma))
        bounce_errors.append(row)
        contact_errors.append(
            None
            if contact is None
            else float(np.linalg.norm(np.asarray(flight["start_xyz"], float)[:2] - contact[0]))
        )
    closing = anchor_targets.get("closing_contact")
    receipt.update(
        bounce_anchor_sigma_m=bounce_sigmas,
        bounce_anchor_errors_m=bounce_errors,
        contact_anchor_errors_m=contact_errors,
        player_tie_sigma_m=anchor_targets["player_sigma_m"],
        # Present only for an original-contact prefix that carries a closing
        # actor.  Every other arm keeps its published receipt shape.
        **(
            {}
            if closing is None
            else {
                "closing_contact_anchor": {
                    "court_xy_m": [float(value) for value in closing["court_xy_m"]],
                    "sigma_m": float(closing["sigma_m"]),
                    "error_m": float(
                        np.linalg.norm(
                            np.asarray(flights[-1]["end_xyz"], float)[:2] - closing["court_xy_m"]
                        )
                    ),
                    "residuals": [
                        float(value)
                        for value in (
                            np.asarray(flights[-1]["end_xyz"], float)[:2] - closing["court_xy_m"]
                        )
                        / closing["sigma_m"]
                    ],
                    "source": closing["source"],
                }
            }
        ),
        bounce_sigma_rule=(
            "projection-derived graded circle radius max(0.20 m, 2 sigma) from the same "
            "supplied ground-ray witness the unchanged gate is scored against"
        ),
        uses_withheld_pixels=False,
    )
    return receipt


def measure(
    scene,
    heldout,
    bounces,
    native,
    parameters,
    axes,
    duration,
    references,
    *,
    termination_kind="terminal_bounce",
    terminal_rebound_frames: np.ndarray | None = None,
    terminal_ground_event: dict | None = None,
    missing_check_direction: str = "raise",
):
    """Score native pictures, optionally retaining unsupported checks as abstentions.

    Strict replay remains the default. The shared census opts into separate
    unscorable exposure-front rows; no blur axis, front coordinate or residual
    is imputed. Nominal-center projection needs no motion direction.
    """
    from cv.experiments.connected_shooting import observation_partition as partition

    if missing_check_direction not in {"raise", "abstain"}:
        raise ValueError("explicit raise or abstain missing-check-direction policy required")
    check_split = partition.check_split(scene)
    rows = []
    unscorable = []
    # Withheld directions use only withheld fronts and their same-frame agent tail;
    # if too sparse for a <=3frame secant, use a training-derived local wing axis.
    # A withheld label must not determine another training observation or the fit.
    train_frames = np.concatenate(scene.observation_frames)
    boundaries = np.unique(np.r_[scene.contact_frames, *bounces])
    if scene.terminal_net_tail is not None:
        boundaries = np.unique(np.r_[boundaries, scene.terminal_net_tail["representative"]])
    train_wings = np.searchsorted(boundaries, train_frames, side="right")
    check_frames = np.concatenate(heldout.observation_frames)
    check_axes = []
    for frame in check_frames:
        indices = np.flatnonzero(train_wings == np.searchsorted(boundaries, frame, side="right"))
        if not len(indices):
            # Terminal frame has no following wing; its incoming side is explicit.
            indices = np.flatnonzero(train_wings == np.searchsorted(boundaries, frame, side="left"))
        if not len(indices):
            if missing_check_direction == "raise":
                raise ValueError("no training-wing direction for withheld exposure")
            check_axes.append(None)
            continue
        j = indices[np.argmin(abs(train_frames[indices] - frame))]
        check_axes.append(axes[j])
    for split, source, direction in (
        ("training", scene, axes),
        (check_split, heldout, check_axes),
    ):
        supported = np.array([axis is not None for axis in direction], bool)
        if duration is None and missing_check_direction == "abstain":
            # A center is directly computable from the path and camera. Missing
            # motion evidence must not create an exposure-front requirement.
            direction = None if not np.all(supported) else np.asarray(direction)
            supported[:] = True
        else:
            direction = np.asarray(direction) if np.all(supported) else direction
        if np.all(supported):
            # Leave every supported historical calculation unchanged.
            predicted = prediction(
                source, parameters, direction, duration, termination_kind=termination_kind
            )
            centres = prediction(
                source, parameters, direction, None, termination_kind=termination_kind
            )
            predictions = dict(enumerate(predicted))
        else:
            # Centres need no blur direction and remain diagnostic geometry.
            # A missing axis never receives a fabricated front or residual.
            centres = prediction(source, parameters, None, None, termination_kind=termination_kind)
            indices = np.flatnonzero(supported)
            if duration is None:
                predictions = {int(i): centres[i] for i in indices}
            elif len(indices):
                imaging = (
                    source
                    if termination_kind
                    in ("terminal_net_impact", "net_stop", "observed_horizon", "original_contact")
                    else leading.context.imaging_scene(source, 1.0, termination_kind)
                )
                curves = leading.swept.fitted_curves(imaging, parameters, duration, 0.0)
                selected = [curves[i] for i in indices]
                selected_axes = np.asarray([direction[i] for i in indices], float)
                estimates = leading.swept.predict(imaging, selected, selected_axes, 1.5)[:, 1]
                predictions = dict(zip(map(int, indices), estimates, strict=True))
            else:
                predictions = {}
        for i, (frame, owner, centre) in enumerate(
            zip(
                np.concatenate(source.observation_frames),
                np.concatenate(source.pixels),
                centres,
                strict=True,
            )
        ):
            if not supported[i]:
                unscorable.append(
                    dict(
                        frame=int(frame),
                        split=split,
                        owner=owner.tolist(),
                        nominal_centre=centre.tolist(),
                        predicted=None,
                        error_px=None,
                        status="unscorable",
                        reason="no_training_wing_direction",
                    )
                )
                continue
            estimate = predictions[i]
            rows.append(
                dict(
                    frame=int(frame),
                    split=split,
                    owner=owner.tolist(),
                    predicted=estimate.tolist(),
                    nominal_centre=centre.tolist(),
                    error_px=float(np.linalg.norm(owner - estimate)),
                )
            )
    dense = leading.model.chain(
        scene,
        parameters,
        query_frames=tuple(
            np.linspace(a, b, max(2, int((b - a) / scene.fps * 240) + 1))
            for a, b in zip(scene.contact_frames, scene.contact_frames[1:])
        ),
    )
    return dict(
        native_projection=sorted(rows, key=lambda r: r["frame"]),
        **(
            {
                "unscorable_native_projection": unscorable,
                "check_direction_policy": "explicit_abstention_v1",
            }
            if unscorable
            else {}
        ),
        rms_px={
            split: None
            if any(r["split"] == split for r in unscorable)
            else float(np.sqrt(np.mean([r["error_px"] ** 2 for r in rows if r["split"] == split])))
            for split in ("training", check_split)
        }
        | ({"withheld": None} if partition.all_native(scene) else {}),
        **(
            {
                "observation_partition": partition.receipt(scene, heldout),
                "common_original_fit_rms_px": float(
                    np.sqrt(
                        np.mean(
                            [
                                r["error_px"] ** 2
                                for r in rows
                                if r["split"] == "training" and r["frame"] % 5 != 0
                            ]
                        )
                    )
                )
                if any(r["split"] == "training" and r["frame"] % 5 != 0 for r in rows)
                else None,
            }
            if partition.all_native(scene)
            else {}
        ),
        physical=physical_compatibility.evaluate(
            scene,
            parameters,
            bounces,
            termination_kind,
            native,
            duration=duration,
            **(
                {
                    "terminal_rebound_frames": terminal_rebound_frames,
                    "terminal_ground_event": terminal_ground_event,
                }
                if terminal_rebound_frames is not None
                else {}
            ),
        ),
        dense_flights=dense,
        contact_xyz=[f["positions"][0].tolist() for f in dense],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "cameras", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--streak-reference", type=Path, nargs="+", required=True)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists() or min(args.maxiter, args.timeout) < 1:
        raise ValueError("new output and positive budgets required")
    packet, cameras, baseline = [
        json.loads(p.read_text()) for p in (args.packet, args.cameras, args.baseline)
    ]
    if len(packet["attempts"]) != 1 or baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError("single explicit owner attempt and terminal-feasibility baseline required")
    if any(
        provenance.file_record(p) not in baseline["inputs"] for p in (args.packet, args.cameras)
    ):
        raise ValueError("baseline packet/camera identity mismatch")
    scene, heldout, bounces, native = replay.prepare(packet["attempts"][0], cameras)
    initial = np.asarray(baseline["refinement"]["parameters"])
    references = [json.loads(p.read_text()) for p in args.streak_reference]
    axes, receipt = directions(scene, bounces, references)
    files = [args.packet, args.cameras, args.baseline, *args.streak_reference, Path(__file__)]
    changes = []
    for document in (packet, cameras, baseline):
        for record in document["inputs"]:
            p = terminal.replay_dependency(record, True)
            current = provenance.file_record(p)
            if current != record:
                changes.append(dict(historical=record, current=current))
            files.append(p)
    for m in (
        leading,
        leading.swept,
        leading.context,
        leading.model,
        leading.event_constraints,
        leading.flight_cache,
        leading.camera_geometry,
        physical_compatibility,
        terminal,
        replay,
        streak_axis_capacity,
    ):
        files.append(Path(m.__file__))
    bindings = [provenance.file_record(p) for p in dict.fromkeys(files)]
    # Replay physical compatibility and every dense baseline path before reuse.
    check = physical_compatibility.evaluate(scene, initial, bounces, "terminal_bounce", native)
    if check["compatible"] != baseline["physical_compatibility"]["compatible"]:
        raise ValueError("baseline physical result changed")
    report = dict(
        schema="s6_real_exposure_replay_v1",
        status="running",
        scope=__doc__,
        inputs=bindings,
        historical_producer_changes=changes,
        code=provenance.git_record(paths.REPO_ROOT),
        attempt_id=packet["attempts"][0]["attempt_id"],
        axes=receipt,
        results={},
        configuration=dict(
            maxiter=args.maxiter,
            timeout=args.timeout,
            arms=ARMS,
            blur_radius_px=1.5,
            open_offset_frames=0,
            contact_frames=scene.contact_frames.tolist(),
            train_frames=np.concatenate(scene.observation_frames).tolist(),
            withheld_frames=np.concatenate(heldout.observation_frames).tolist(),
        ),
        human_derived=True,
        automatic_inference_eligible=False,
        complete_point_accepted=False,
    )
    args.output.mkdir(parents=True)

    def write():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=replay.default, allow_nan=False) + "\n"
        )

    write()

    def deadline(*_):
        raise TimeoutError("bounded real exposure fit")

    for arm, duration in ARMS.items():
        start = time.monotonic()
        previous = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.timeout)
        try:
            fit = refine(scene, initial, bounces, native, axes, duration, args.maxiter)
            measurement = measure(
                scene,
                heldout,
                bounces,
                native,
                np.asarray(fit["parameters"]),
                axes,
                duration,
                references,
            )
            row = dict(status="measured", fit=fit, **measurement)
        except (ValueError, FloatingPointError, OverflowError, TimeoutError) as error:
            row = dict(status="held", reason=str(error))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        row["wall_seconds"] = time.monotonic() - start
        report["results"][arm] = row
        write()
        print(arm, row["status"], row.get("rms_px"), flush=True)
    for record in bindings:
        terminal.replay_dependency(record, False)
    report["status"] = "complete"
    write()


if __name__ == "__main__":
    main()
