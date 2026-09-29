"""Re-score one already-fitted connected attempt flight by flight.

Evaluation only, and deliberately not a refit: this reads the frozen
whole-point search report for one attempt, rebuilds exactly the scene, axes,
players and bounce witnesses that search used, and asks the per-flight
acceptance question of the same fitted parameters.  The published whole-point
checks are recomputed first and compared field for field, so a rescoring run
that has drifted from the report it reads fails closed instead of publishing a
number that belongs to a different scene.

The relaxation ladder is development-only.  Its base rung is the promoted
reference, so ``complete_point`` at the base rung is the promoted acceptance.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import time

import numpy as np

from cv.experiments.connected_shooting import (
    agent_whole_point_search as whole,
    per_flight_acceptance as acceptance,
    athlete_priors,
    real_exposure_replay as exposure,
    serve_reach_cylinder,
)
from cv.pipeline import paths, provenance
from cv.validation import s6_sparse_owner_replay as replay


SCHEMA = "connected_per_flight_rescore_v1"

#: The two-annotator half-frame agreement on a physical event epoch, which is
#: the whole allowance for passive post-ending context.
PASSIVE_ENDING_FRAMES = 0.5

#: ``playstyle_pattern_v1``'s frozen net geometry sigma, carrying the cord
#: height, the ball radius, the camera height uncertainty and exposure phase.
NET_CLEARANCE_SIGMA_M = 0.10


def serve_ending_extra_check(
    base: dict, ending_kind: str | None, striker_end: str
) -> tuple[dict, dict]:
    """Translate the first flight's modeled/witnessed bounce into one gate."""
    witness_rows = base["flights"][0].get("bounce_witness") or []
    row = witness_rows[0] if witness_rows else {}
    receipt = whole.serve_ending_consistency(
        ending_kind,
        striker_end,
        row.get("modeled_xyz_m"),
        row.get("witness_xyz_m"),
    )
    return {0: {"serve_ending_bounce_consistent": receipt["passed"]}}, receipt


def ladder() -> list[dict]:
    """The declared relaxation ladder, one entry per rung.

    Knobs move one at a time from the promoted base, then together.  Nothing
    here is promoted by being measured; the synthetic false-accept audit at the
    same rung decides that.
    """
    rungs = [{"name": "base", "thresholds": acceptance.Thresholds(), "family": "base"}]
    for value in (0.30, 0.40):
        rungs.append(
            {
                "name": f"bounce_circle_{value:.2f}m",
                "family": "bounce_circle",
                "thresholds": acceptance.Thresholds(bounce_circle_floor_m=value),
            }
        )
    for value in (24.0, 32.0):
        rungs.append(
            {
                "name": f"direction_window_{int(value)}px",
                "family": "direction_window",
                "thresholds": acceptance.Thresholds(directional_rms_limit_px=value),
            }
        )
    for scale in (1.5, 2.0):
        rungs.append(
            {
                "name": f"reprojection_rms_x{scale:g}",
                "family": "reprojection_rms",
                "thresholds": acceptance.Thresholds(flight_reprojection_rms_limit_px=16.0 * scale),
            }
        )
    # The two image-side knobs move the same fitted picture residual, so their
    # joint rung is measured too rather than inferred from the two alone.
    rungs.append(
        {
            "name": "direction_window_32px_reprojection_x2",
            "family": "image_residual_combined",
            "thresholds": acceptance.Thresholds(
                directional_rms_limit_px=32.0, flight_reprojection_rms_limit_px=32.0
            ),
        }
    )
    for value in (1.0, 2.0, 4.0):
        rungs.append(
            {
                "name": f"velocity_slack_{value:g}mps",
                "family": "velocity_slack",
                "thresholds": acceptance.Thresholds(velocity_slack_mps=value),
            }
        )
    # B2/B3: the terminal semantics and the net check.  ``PASSIVE_ENDING_FRAMES``
    # is the annotators' own half-frame agreement on a physical event epoch and
    # ``NET_CLEARANCE_SIGMA_M`` is the frozen contract's geometry sigma; neither
    # is a tuned number and neither moves the fit.
    rungs.append(
        {
            "name": "passive_post_ending_context",
            "family": "terminal_semantics",
            "thresholds": acceptance.Thresholds(
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES
            ),
        }
    )
    rungs.append(
        {
            "name": "passive_post_ending_context_32px_x2",
            "family": "terminal_semantics",
            "thresholds": acceptance.Thresholds(
                directional_rms_limit_px=32.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
            ),
        }
    )
    rungs.append(
        {
            "name": "net_clearance_sigma",
            "family": "net_clearance",
            "thresholds": acceptance.Thresholds(net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M),
        }
    )
    rungs.append(
        {
            "name": "terminal_semantics_and_net",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
            ),
        }
    )
    rungs.append(
        {
            "name": "terminal_semantics_and_net_32px_x2",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                directional_rms_limit_px=32.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
            ),
        }
    )
    rungs.append(
        {
            "name": "bounce_timing_2f",
            "family": "event_timing",
            "thresholds": acceptance.Thresholds(bounce_uncertainty_frames=2.0),
        }
    )
    rungs.append(
        {
            "name": "terminal_semantics_and_net_32px_x2_bounce2f",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                directional_rms_limit_px=32.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                bounce_uncertainty_frames=2.0,
            ),
        }
    )
    rungs.append(
        {
            "name": "terminal_semantics_and_net_32px_x2_events2f",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                directional_rms_limit_px=32.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                bounce_uncertainty_frames=2.0,
                ending_uncertainty_frames=2.0,
            ),
        }
    )
    for limit in (0.30, 0.50):
        rungs.append(
            {
                "name": f"terminal_semantics_and_net_32px_x2_bounce2f_serve{int(limit * 100)}cm",
                "family": "terminal_semantics_and_net",
                "thresholds": acceptance.Thresholds(
                    directional_rms_limit_px=32.0,
                    flight_reprojection_rms_limit_px=32.0,
                    ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                    net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                    bounce_uncertainty_frames=2.0,
                    serve_bounce_ray_limit_m=limit,
                ),
            }
        )
    for limit in (0.60, 0.75):
        # Timing and landing slack should not both sit at their widest: the
        # two-frame rung with a tighter general bounce ray gate.
        rungs.append(
            {
                "name": f"terminal_semantics_and_net_32px_x2_bounce2f_ray{int(limit * 100)}cm",
                "family": "terminal_semantics_and_net",
                "thresholds": acceptance.Thresholds(
                    bounce_ray_limit_m=limit,
                    directional_rms_limit_px=32.0,
                    flight_reprojection_rms_limit_px=32.0,
                    ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                    net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                    bounce_uncertainty_frames=2.0,
                ),
            }
        )
    rungs.append(
        {
            # Owner 2026-09-09: "50 cm serve-only landing gate seems fairly
            # generous for a serve; 18 inch radius sounds about right."
            "name": "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                bounce_ray_limit_m=0.75,
                serve_bounce_ray_limit_m=0.50,
                directional_rms_limit_px=32.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                bounce_uncertainty_frames=2.0,
            ),
        }
    )
    rungs.append(
        {
            # Astra (2026-09-09 evening): a return 0.75 m short of its two-sided
            # witness passed the 0.75 m ray gate; measure a 0.60 m gate with the
            # serve cap.
            "name": "terminal_semantics_and_net_32px_x2_bounce2f_ray60cm_serve50cm",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                bounce_ray_limit_m=0.60,
                serve_bounce_ray_limit_m=0.50,
                directional_rms_limit_px=32.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                bounce_uncertainty_frames=2.0,
            ),
        }
    )
    for window in (40.0, 48.0):
        # Owner 2026-09-09: hardatp2026r128_pt0001 "looks totally fine" and
        # fails only a 33.8 px event-side window; measure the window alone.
        rungs.append(
            {
                "name": f"terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win{int(window)}",
                "family": "terminal_semantics_and_net",
                "thresholds": acceptance.Thresholds(
                    bounce_ray_limit_m=0.75,
                    serve_bounce_ray_limit_m=0.50,
                    directional_rms_limit_px=window,
                    flight_reprojection_rms_limit_px=32.0,
                    ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                    net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                    bounce_uncertainty_frames=2.0,
                ),
            }
        )
    rungs.append(
        {
            "name": "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win32_shift1f",
            "family": "terminal_semantics_and_net",
            "thresholds": acceptance.Thresholds(
                bounce_ray_limit_m=0.75,
                serve_bounce_ray_limit_m=0.50,
                directional_rms_limit_px=32.0,
                directional_time_shift_frames=1.0,
                flight_reprojection_rms_limit_px=32.0,
                ending_passive_context_frames=PASSIVE_ENDING_FRAMES,
                net_clearance_sigma_m=NET_CLEARANCE_SIGMA_M,
                bounce_uncertainty_frames=2.0,
            ),
        }
    )
    for level, (circle, window, scale, slack) in enumerate(
        ((0.30, 24.0, 1.5, 1.0), (0.40, 32.0, 2.0, 2.0), (0.40, 32.0, 2.0, 4.0)), start=1
    ):
        rungs.append(
            {
                "name": f"combined_L{level}",
                "family": "combined",
                "thresholds": acceptance.Thresholds(
                    bounce_circle_floor_m=circle,
                    directional_rms_limit_px=window,
                    flight_reprojection_rms_limit_px=16.0 * scale,
                    velocity_slack_mps=slack,
                ),
            }
        )
    return rungs


#: Fields the REBUILT witness is expected to lack because the replay attaches them itself
#: afterwards, or because they exist only on the frozen record. Everything else the frozen
#: witness declares must survive into the rebuild, or the verdict is scored on a witness the
#: search never used. Keep this list short and justified; each entry is a hole in the check.
_WITNESS_FIELDS_ADDED_AFTER_FREEZE = frozenset(
    {
        # The replay attaches this to the rebuilt target a few lines below, from the frozen one.
        "legacy_subframe_witness",
        # Written into the frozen record by the search's own reporting, not by the witness.
        "subframe_available",
    }
)


def build_context(args: argparse.Namespace, report: dict) -> dict:
    """Rebuild the frozen search's scene without refitting anything in it."""
    from cv.experiments.connected_shooting import passive_bounce, player_position

    position_policy = player_position.replay_policy(args, report["configuration"])

    recorded = report.get("configuration", {}).get("passive_bounce_response", "off")
    if passive_bounce.validate(recorded) != passive_bounce.active():
        raise ValueError("replay requires the recorded passive bounce response context")
    labels, packet, cameras_document = [
        json.loads(path.read_text()) for path in (args.labels, args.packet, args.cameras)
    ]
    from cv.experiments.connected_shooting import observation_operator

    operator = observation_operator.from_report(report, packet)
    attempt = packet["attempts"][0]
    fallback_receipt: list[dict] = []
    declared_contact_envelope = (report.get("configuration") or {}).get(
        "contact_xy_search_envelope_m"
    )
    observation_fallback = args.observation_fallback == "on"
    if args.dense_labels is not None:
        dense_document = json.loads(args.dense_labels.read_text())
        whole.supplement_with_dense_labels(attempt, dense_document)
        whole.supplement_label_streaks(labels, dense_document)
    # Match whole-point search ordering exactly: bounce witnesses are built
    # from the packet's competitive window, while a terminal-rebound arm may
    # extend only the fitted image context.  Taking these lookups after that
    # extension can silently turn rebound pictures into a different bounce
    # witness during rescore.
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
    from cv.pipeline import s6_terminal_net_membership as net_membership

    # Replay resolves the same validated effective attempt the search solved,
    # from the original packet plus the report's own membership receipt. The
    # existing scope/tail equality checks below then run unchanged against it.
    attempt = net_membership.replay_attempt(attempt, report.get("configuration", {}))
    from cv.experiments.connected_shooting import observation_scope

    observation_scope.validate(attempt, labels, report)
    from cv.pipeline import s6_first_contact_role

    role_receipt = s6_first_contact_role.validate(attempt, labels, report)
    events = list(report["events"])
    physical_events = [row for row in events if row["event_type"] != "ending"]
    end_frame = float(report["configuration"].get("end_frame", 0.0)) or float(
        attempt["owner_end_frame"]
    )
    ending = [row for row in events if row["event_type"] == "ending"]
    if ending:
        end_frame = float(ending[0]["frame"])
    rebound_receipt = report["configuration"].get("terminal_rebound") or {"mode": "off"}
    terminal_rebound_on = (
        rebound_receipt.get("mode") == "on" and rebound_receipt.get("status") == "supported"
    )
    ownership_on = report["configuration"].get("terminal_context_ownership", "off") == "on"
    if ownership_on:
        from cv.pipeline import s6_terminal_context

        s6_terminal_context.require_initial_ownership(labels, report)
    if terminal_rebound_on:
        end_frame = float(rebound_receipt["modeled_scene_end_frame"])
        attempt = whole.event_recovery.extend_attempt_window(
            attempt, labels["ball"]["records"], end_frame
        )
    net_tail = report["configuration"].get("terminal_net_tail")
    if net_tail is not None:
        if attempt.get("terminal_net_tail") != net_tail:
            raise ValueError("terminal net-tail source contract differs from search")
        end_frame = float(net_tail["observation_horizon"])
        attempt = whole.event_recovery.extend_attempt_window(
            attempt, labels["ball"]["records"], end_frame
        )
    horizon_tail = report["configuration"].get("observed_horizon_tail")
    if horizon_tail is not None:
        if attempt.get("observed_horizon_tail") != horizon_tail:
            raise ValueError("observed horizon source contract differs from search")
        end_frame = float(horizon_tail["observation_horizon"])
        attempt = whole.event_recovery.extend_attempt_window(
            attempt, labels["ball"]["records"], end_frame
        )
    camera_lookup = {
        int(row["frame"]): np.asarray(row["P"], float)
        for row in cameras_document["cameras"]
        if row.get("supported", True) and "P" in row
    }
    scene, heldout, bounces, native, _ = whole.prepare_attempt(
        attempt,
        cameras_document,
        report["configuration"]["surface"],
        physical_events,
        end_frame,
        observation_partition=report["configuration"].get(
            "observation_partition", "fifth_frame_withheld"
        ),
        ground_settling=report["configuration"].get("ground_settling", "off") == "on",
        bounce_witness_observation_policy=report["configuration"].get(
            "bounce_witness_observation_policy",
            "legacy_fit_frames_including_existing_terminal_activation",
        ),
        observation_fallback=observation_fallback,
        fallback_receipt=fallback_receipt,
        exit_partial_endings=report["configuration"].get("exit_partial_endings") == "on",
    )
    net_groups = tuple(
        np.asarray(
            [
                event["frame"]
                for event in physical_events
                if event["event_type"] == "net_hit" and a < event["frame"] < b
            ],
            float,
        )
        for a, b in zip(scene.contact_frames[:-1], scene.contact_frames[1:], strict=True)
    )
    if any(len(group) for group in net_groups):
        from dataclasses import replace as _replace

        scene = _replace(scene, net_hit_frames=net_groups)
        heldout = _replace(heldout, net_hit_frames=net_groups)
        scene.validate()
        heldout.validate()
    reference = dict(annotation_status="frozen_agent_reference", records=labels["ball"]["records"])
    axes, _ = whole.training_directions(
        scene,
        bounces,
        reference,
        observation_fallback=observation_fallback,
        exposure_duration=operator["exposure_duration_frames"],
        fallback_receipt=fallback_receipt,
    )
    terminal_rebound_frames = None
    if terminal_rebound_on:
        segment = exposure.terminal_rebound_segment
        if ownership_on:
            from cv.pipeline import s6_terminal_context

            segment = s6_terminal_context.preserved_segment
        scene, axes, terminal_rebound_frames, rebuilt_rebound = segment(
            scene, heldout, bounces, axes
        )
        if (
            rebuilt_rebound["postbounce_labeled_frames"]
            != rebound_receipt["postbounce_labeled_frames"]
        ):
            raise ValueError("terminal rebound context does not reproduce the search report")
    contact_events = [row for row in physical_events if row["event_type"] == "contact"]
    bounce_events = [row for row in physical_events if row["event_type"] == "bounce"]
    statures = dict(report["configuration"].get("player_statures_m") or {})
    athlete_evidence = replay_athlete_evidence(args, report)
    player_names = None
    if args.athlete_prior_mode == "stature_pose_soft" and statures:
        player_names = whole.labeled_contact_players(
            labels, contact_events, list(args.player_order)
        )
    players, right_contact_player = whole.launch_and_right_players(
        args.pose_csv,
        attempt["point_clip"],
        contact_events,
        label_lookup,
        image_coordinate_scale=args.pose_image_scale,
        right_boundary_kind=scene.right_boundary_kind,
        player_names=player_names,
        player_statures_m=statures,
        observation_fallback=observation_fallback,
        fallback_receipt=fallback_receipt,
        missing_player_position=position_policy,
        **(
            {"serve_side_association": report["configuration"]["serve_side_association"]}
            if "serve_side_association" in report["configuration"]
            else {}
        ),
    )
    player_position.require_replayed_states(
        report["configuration"], players, right_contact_player=right_contact_player
    )
    if args.athlete_prior_mode == "stature_pose_soft" and athlete_evidence["status"] == (
        "unavailable"
    ):
        whole.mark_athlete_evidence(players, athlete_evidence)
    s6_first_contact_role.apply_players(players, role_receipt)
    athlete_priors.configure_root_reach(players, athlete_priors.replay_root_reach(args, report))
    if args.player_ledger is not None:
        whole.apply_player_ledger(
            players,
            contact_events[:-1]
            if scene.right_boundary_kind == "original_contact"
            else contact_events,
            whole.player_ledger_positions(args.player_ledger, attempt["point_clip"]),
        )
    from cv.pipeline import s6_rally_origin_cue

    s6_rally_origin_cue.verify_report(
        report | {"player_states": players},
        labels,
        cameras_document,
        args.pose_csv,
        args.pose_image_scale,
    )
    from cv.experiments.connected_shooting.observation_partition import witness_frames

    # Read both owner-approved 2026-09-19 gate changes off the FROZEN WITNESSES the search
    # published, not off a receipt written beside them.
    #
    # This replay is what produces the verdict, so if it rebuilds the witness differently from
    # the search the gate change is silently discarded -- which is exactly what voided the
    # first gate wave: the receipt was keyed to `report["configuration"]`, and the several
    # modules that can write a report each build that dict themselves, so one of them wrote a
    # report carrying the new targets and none of the keys. The targets are self-describing and
    # every writer emits them, so ask them instead and there is no fourth writer to miss.
    frozen_targets = [
        target
        for group in report["bounce_ground_ray_targets"]
        for target in (group if isinstance(group, list) else [group])
    ]
    rescore_publish_frame_interval = any(
        target.get("supplied_frame_interval") is not None for target in frozen_targets
    )
    applied_floors = {
        float((target.get("wing_policy") or {}).get("applied_sigma_floor_px"))
        for target in frozen_targets
        if (target.get("wing_policy") or {}).get("applied_sigma_floor_px") is not None
    }
    if len(applied_floors) > 1:
        raise ValueError("one search cannot have used two bounce-witness sigma floors")
    rescore_sigma_floor_px = (
        applied_floors.pop() if applied_floors else whole.BOUNCE_LABEL_SIGMA_FLOOR_PX
    )
    targets = [
        [
            whole.event_ground_target(
                row,
                camera_lookup,
                label_lookup,
                radius_lookup,
                camera_distortion=whole.camera_geometry.camera_radial_map(cameras_document),
                mode=report["configuration"]["bounce_witness_mode"],
                observation_fallback=observation_fallback,
                fallback_receipt=fallback_receipt,
                eligible_frames=witness_frames(scene, flight_index, terminal_rebound_frames),
                # Rebuild the witness the search actually published; see above.
                sigma_floor_px=rescore_sigma_floor_px,
                publish_frame_interval=rescore_publish_frame_interval,
            )
            for row in bounce_events
            if start < float(row["frame"]) <= end
        ]
        for flight_index, (start, end) in enumerate(
            zip(scene.contact_frames, scene.contact_frames[1:])
        )
    ]
    # The search report is the immutable source for its own published witness.
    # Reconstructing that old chord after changing witness code would defeat the
    # fail-closed reproduction check and could silently lose accepted members.
    legacy_targets = report["bounce_ground_ray_targets"]
    if [len(group) for group in targets] != [len(group) for group in legacy_targets]:
        raise ValueError("new and frozen bounce witnesses must preserve event topology")
    # Topology equality alone let both owner-approved gate changes through silently: the
    # frozen witness carried `supplied_frame_interval` and `wing_policy.applied_sigma_floor_px`
    # and the rebuilt one did not, the counts matched, and the verdict was scored on the
    # pre-sign-off witness with nothing raised. Compare the FIELDS the two witnesses declare,
    # which is generic over mechanisms nobody has written yet and, unlike comparing values,
    # cannot false-positive on ordinary numerical drift between two builds of the same witness.
    for new_group, legacy_group in zip(targets, legacy_targets, strict=True):
        for rebuilt, frozen in zip(new_group, legacy_group, strict=True):
            missing = set(frozen) - set(rebuilt) - _WITNESS_FIELDS_ADDED_AFTER_FREEZE
            if missing:
                raise ValueError(
                    "the replayed bounce witness dropped fields the search published: "
                    f"{sorted(missing)}; the verdict would be scored on a different witness "
                    "than the search used"
                )
            frozen_wing = (frozen.get("wing_policy") or {}).keys()
            rebuilt_wing = (rebuilt.get("wing_policy") or {}).keys()
            if set(frozen_wing) - set(rebuilt_wing):
                raise ValueError(
                    "the replayed bounce witness dropped wing policy the search published: "
                    f"{sorted(set(frozen_wing) - set(rebuilt_wing))}"
                )
    for new_group, legacy_group in zip(targets, legacy_targets, strict=True):
        for target, legacy in zip(new_group, legacy_group, strict=True):
            target["legacy_subframe_witness"] = {
                "xyz_m": legacy["xyz_m"],
                "uncertainty_sigma_m": legacy.get("uncertainty_sigma_m"),
                "native_frames": legacy.get("native_frames", []),
                "available": bool(legacy.get("subframe_available")),
            }
    low, high = s6_first_contact_role.limits(players[0])
    # The search may have cut the serve contact with a hard reach cylinder. That
    # cylinder is part of the published whole-point checks, so it is rebuilt from
    # the same sided box here or the reproduction receipt fails closed below.
    cylinder = None
    if report["configuration"].get("serve_reach_cut") == "on":
        cylinder = serve_reach_cylinder.build(
            players[0],
            serve_reach_cylinder.ServeReachConfig(
                margin_statures=float(report["configuration"]["serve_reach_margin_statures"]),
                radius_statures=float(report["configuration"]["serve_reach_radius_statures"]),
            ),
        )
    return {
        "serve_reach": cylinder,
        "declared_contact_envelope": declared_contact_envelope,
        "scene": scene,
        "heldout": heldout,
        "bounces": bounces,
        "native": native,
        "axes": axes,
        "targets": targets,
        "legacy_targets": legacy_targets,
        "players": players,
        "events": physical_events,
        "depth_bounds": (low, high),
        "termination_kind": report["configuration"]["termination_kind"],
        "terminal_rebound_frames": terminal_rebound_frames,
        "right_contact_player": right_contact_player,
        "search_ending_kind": report["configuration"].get("labeled_ending_kind"),
        "search_serve_ending_consistency": (
            report["configuration"].get("serve_ending_consistency") == "on"
        ),
        "witness_surface": args.witness_surface or scene.surface,
        "attempt": attempt,
        "observation_fallback_receipt": fallback_receipt,
        "observation_operator": operator,
    }


def replay_athlete_evidence(args: argparse.Namespace, report: dict) -> dict:
    """The replay must name the same athlete-evidence policy the search declared.

    A cached search fitted under the original ``required`` contract replays
    only under it; an ``optional`` search records whether evidence was supplied
    or explicitly unavailable, and the replay reproduces exactly that state.
    """
    configuration = report.get("configuration", {})
    declared = configuration.get("athlete_evidence", "required")
    requested = getattr(args, "athlete_evidence", "required")
    if declared not in whole.ATHLETE_EVIDENCE_POLICIES:
        raise ValueError("cached search declares an unsupported athlete-evidence policy")
    if requested != declared:
        raise ValueError("replay athlete-evidence policy must match the cached search")
    statures = dict(configuration.get("player_statures_m") or {})
    status = configuration.get("athlete_evidence_status")
    if declared == "required":
        if status not in (None, "supplied", "not_applicable"):
            raise ValueError("required athlete evidence cannot replay an unavailable search")
        return {"policy": declared, "status": "supplied" if statures else "not_applicable"}
    expected = "supplied" if statures else "unavailable"
    if status != expected:
        raise ValueError("cached athlete-evidence status disagrees with its recorded statures")
    return {"policy": declared, "status": status}


def reproduction_receipt(context: dict, candidate: dict, athlete_prior_mode: str) -> dict:
    """Recompute the published whole-point checks; drift must fail closed."""
    measurement = candidate["measurement"]
    published = candidate["evidence"]["checks"]
    # The serve reach cylinder is an optional argument of the search's own
    # evidence builder.  A sweep fitted without the cut has no cylinder to
    # rebuild, so the keyword is omitted rather than passed as ``None``; a
    # sweep fitted *with* the cut against a build that cannot take it fails
    # closed here instead of silently dropping the gate from the receipt.
    reach: dict = {}
    if context["serve_reach"] is not None:
        if "serve_reach" not in inspect.signature(whole.candidate_evidence).parameters:
            raise ValueError(
                "this search build cannot rebuild the serve reach cylinder the report declares"
            )
        reach = {"serve_reach": context["serve_reach"]}
    recomputed = whole.candidate_evidence(
        measurement,
        context["players"],
        context["legacy_targets"],
        context["depth_bounds"],
        context["events"],
        athlete_prior_mode=athlete_prior_mode,
        serve_ending_kind=context.get("search_ending_kind"),
        enforce_serve_ending_consistency=context.get("search_serve_ending_consistency", False),
        **reach,
    )["checks"]
    shared = sorted(set(published) & set(recomputed))
    # The contact envelope is a declared rule, not scene evidence.  A report
    # fitted under an older, narrower envelope (recorded in its configuration)
    # reproduces its scene exactly while that one check may legitimately flip
    # under the current venue envelope; it is then judged by the acceptance
    # gate, not by this identity check.
    declared = context.get("declared_contact_envelope")
    current = [list(row) for row in whole.CONTACT_XY_ENVELOPE_M]
    envelope_changed = declared is not None and [list(row) for row in declared] != current
    if envelope_changed:
        shared = [name for name in shared if name != "contacts_inside_declared_search_envelope"]
    return {
        "published_checks": published,
        "recomputed_checks": recomputed,
        "declared_contact_envelope_m": declared,
        "current_contact_envelope_m": current,
        "envelope_rule_changed": envelope_changed,
        # ``serve_speed_scored`` is attached to the published candidate after
        # ``candidate_evidence`` returns and never gated a branch, so it is the
        # one key this comparison may not own; everything else must match.
        "published_only_checks": sorted(set(published) - set(recomputed)),
        "identical": all(published[name] == recomputed[name] for name in shared)
        and sorted(set(published) - set(recomputed)) in ([], ["serve_speed_scored"])
        and not set(recomputed) - set(published),
    }


def context_at_candidate_epoch(context: dict, candidate: dict) -> dict:
    """Replay fitted contact epochs without changing original observations or events."""
    from cv.experiments.connected_shooting import interior_contact_epochs

    fit = candidate.get("measurement", {}).get("fit", {})
    evidence = candidate.get("evidence", {})
    # Legacy candidates store only the first-contact fit in evidence.
    if interior_contact_epochs.FIELD not in fit and interior_contact_epochs.FIELD not in evidence:
        original = evidence.get("first_contact_epoch_fit")
        return interior_contact_epochs.apply_context(
            context, {} if original is None else {"first_contact_epoch_fit": original}
        )
    if interior_contact_epochs.FIELD in fit or interior_contact_epochs.FIELD in evidence:
        if fit.get(interior_contact_epochs.FIELD) != evidence.get(interior_contact_epochs.FIELD):
            raise ValueError("candidate timing evidence differs from its numerical fit")
    resolved = dict(fit)
    if (
        "first_contact_epoch_fit" not in resolved
        and evidence.get("first_contact_epoch_fit") is not None
    ):
        resolved["first_contact_epoch_fit"] = evidence["first_contact_epoch_fit"]
    return interior_contact_epochs.apply_context(context, resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "packet", "cameras", "pose-csv", "search-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pose-image-scale", type=float, default=1.0)
    parser.add_argument(
        "--athlete-prior-mode", choices=whole.ATHLETE_PRIOR_MODES, default="stature_pose_soft"
    )
    parser.add_argument("--player-order", action="append", default=[])
    parser.add_argument(
        "--athlete-evidence", choices=whole.ATHLETE_EVIDENCE_POLICIES, default="required"
    )
    parser.add_argument(
        "--athlete-root-reach-loss", choices=athlete_priors.ROOT_REACH_LOSSES, default="quadratic"
    )
    parser.add_argument("--observation-fallback", choices=("off", "on"), default="off")
    parser.add_argument("--player-ledger", type=Path)
    parser.add_argument("--dense-labels", type=Path)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument(
        "--witness-surface",
        choices=("hard", "clay", "grass"),
        help=(
            "actual broadcast surface for bounce-witness confidence; defaults "
            "to the fitted dynamics surface"
        ),
    )
    parser.add_argument(
        "--ending-kind",
        help=(
            "the label document's free-text ending kind, which gates the net "
            "clearance verdict; without it a net ending cannot be required"
        ),
    )
    parser.add_argument(
        "--serve-ending-consistency",
        choices=("off", "on"),
        default="off",
        help="gate a labeled serve fault/out on modeled and witnessed service-box calls",
    )
    args = parser.parse_args()
    report = json.loads(args.search_report.read_text())
    from cv.experiments.connected_shooting import passive_bounce

    with passive_bounce.using(
        report.get("configuration", {}).get("passive_bounce_response", "off")
    ):
        return _run_rescore(args, report)


def _run_rescore(args, report):
    began = time.monotonic()
    context = build_context(args, report)
    candidates = report["refined_candidates"]
    if not candidates:
        raise ValueError("a refined candidate is required to re-score a fitted attempt")
    receipts = [
        reproduction_receipt(context, candidate, args.athlete_prior_mode)
        for candidate in candidates
    ]
    if not all(row["identical"] for row in receipts):
        details = []
        for row in receipts:
            published = row["published_checks"]
            recomputed = row["recomputed_checks"]
            changed = sorted(
                name
                for name in set(published) & set(recomputed)
                if published[name] != recomputed[name]
            )
            if changed:
                details.append("changed " + ", ".join(changed))
            extra = sorted(set(recomputed) - set(published))
            if extra:
                details.append("recomputed only " + ", ".join(extra))
            if row["published_only_checks"] not in ([], ["serve_speed_scored"]):
                details.append("published only " + ", ".join(row["published_only_checks"]))
        raise ValueError(
            "re-scoring rebuilt a different scene than the report it reads: "
            + ("; ".join(sorted(set(details))) or "unnamed fields")
        )
    athletes = [
        athlete_priors.evaluate(
            np.asarray(candidate["evidence"]["contact_xyz_m"], float), context["players"]
        )
        if args.athlete_prior_mode == "stature_pose_soft"
        else None
        for candidate in candidates
    ]
    # The terminal semantics change what the ending measurement is, not what the
    # fit is, so one base measurement is built per distinct passive-context
    # allowance and every rung reads the one its thresholds declare.
    passive_values = sorted({rung["thresholds"].ending_passive_context_frames for rung in ladder()})
    candidate_contexts = [
        context_at_candidate_epoch(context, candidate) for candidate in candidates
    ]
    from cv.experiments.connected_shooting.admissible_net_response import replay_from_verdict

    bases_by_passive: dict[float, list[dict]] = {}
    with replay_from_verdict(report):
        for passive in passive_values:
            bases_by_passive[passive] = [
                acceptance.measure(
                    candidate_context["scene"],
                    candidate_context["heldout"],
                    np.asarray(candidate["measurement"]["fit"]["parameters"], float),
                    candidate_context["bounces"],
                    candidate_context["native"],
                    candidate_context["axes"],
                    candidate_context["targets"],
                    candidate_context["players"],
                    candidate_context["events"],
                    termination_kind=candidate_context["termination_kind"],
                    duration=context["observation_operator"]["exposure_duration_frames"],
                    ending_passive_context_frames=passive,
                    athlete=athlete,
                    measurement=candidate["measurement"],
                    terminal_rebound_frames=candidate_context["terminal_rebound_frames"],
                )
                for candidate, athlete, candidate_context in zip(
                    candidates, athletes, candidate_contexts, strict=True
                )
            ]
            for base in bases_by_passive[passive]:
                base["surface"] = context["witness_surface"]
    # ``serve_speed_scored`` is recorded in the published checks but is not a
    # gate: the search freezes ``survived`` before the speed witness is
    # attached, so an unavailable radar graphic never rejected a branch.  It
    # stays a reported witness here for the same reason.
    extra: list[dict[int, dict[str, bool]]] = [{} for _ in candidates]
    ending_receipts: list[dict | None] = [None for _ in candidates]
    if args.serve_ending_consistency == "on":
        representative = bases_by_passive[passive_values[0]]
        pairs = [
            serve_ending_extra_check(base, args.ending_kind, context["players"][0]["side"])
            for base in representative
        ]
        extra = [pair[0] for pair in pairs]
        ending_receipts = [pair[1] for pair in pairs]
    rungs = []
    for rung in ladder():
        thresholds = rung["thresholds"]
        bases = bases_by_passive[thresholds.ending_passive_context_frames]
        scored = [
            acceptance.score(
                base,
                acceptance.Thresholds(**{**thresholds.as_dict(), "velocity_slack_mps": 0.0}),
                depth_bounds=context["depth_bounds"],
                athlete_prior_mode=args.athlete_prior_mode,
                extra_flight_checks=extra[index],
                ending_kind=args.ending_kind,
            )
            for index, base in enumerate(bases)
        ]
        order = sorted(
            range(len(candidates)),
            key=lambda i: (
                -scored[i]["accepted_flight_count"],
                candidates[i]["evidence"]["input_only_rank_score"],
                candidates[i]["depth_hypothesis_m"],
            ),
        )
        chosen = order[0]
        verdict = scored[chosen]
        if thresholds.velocity_slack_mps > 0:
            candidate_context = candidate_contexts[chosen]
            verdict = acceptance.evaluate(
                candidate_context["scene"],
                candidate_context["heldout"],
                np.asarray(candidates[chosen]["measurement"]["fit"]["parameters"], float),
                candidate_context["bounces"],
                candidate_context["native"],
                candidate_context["axes"],
                candidate_context["targets"],
                candidate_context["players"],
                candidate_context["events"],
                termination_kind=candidate_context["termination_kind"],
                duration=context["observation_operator"]["exposure_duration_frames"],
                thresholds=thresholds,
                athlete=athletes[chosen],
                depth_bounds=context["depth_bounds"],
                athlete_prior_mode=args.athlete_prior_mode,
                base=bases[chosen],
                max_nfev=args.max_nfev,
                extra_flight_checks=extra[chosen],
                ending_kind=args.ending_kind,
            )
        else:
            verdict["ending_completed"] = bases[chosen]["ending_completed"]
            verdict["terminal_completion_reason"] = bases[chosen]["terminal_completion"].get(
                "reason"
            )
            verdict["maximum_junction_gap_m"] = 0.0
        rungs.append(
            {
                "rung": rung["name"],
                "family": rung["family"],
                "thresholds": thresholds.as_dict(),
                "selected_depth_m": candidates[chosen]["depth_hypothesis_m"],
                **(
                    {}
                    if args.serve_ending_consistency == "off"
                    else {"serve_ending_consistency": ending_receipts[chosen]}
                ),
                "verdict": verdict,
            }
        )
    payload = {
        "schema": SCHEMA,
        "scope": __doc__,
        "attempt_id": report["attempt_id"],
        "clip": report["clip"],
        "search_report": provenance.file_record(args.search_report),
        "whole_point_reconstructed": bool(report["input_only_reconstructed"]),
        "flight_count": len(context["scene"].contact_frames) - 1,
        "labeled_contact_count": len(
            [row for row in context["events"] if row["event_type"] == "contact"]
        ),
        "labeled_ending_kind": args.ending_kind,
        "witness_surface": context["witness_surface"],
        **({} if args.serve_ending_consistency == "off" else {"serve_ending_consistency": "on"}),
        "reproduction_receipts": receipts,
        "candidate_depths_m": [row["depth_hypothesis_m"] for row in candidates],
        "rungs": rungs,
        "inputs": [
            provenance.file_record(path)
            for path in (
                args.search_report,
                args.labels,
                args.packet,
                args.cameras,
                Path(acceptance.__file__),
                Path(__file__),
            )
        ],
        "code": provenance.git_record(paths.REPO_ROOT),
        "human_derived": True,
        "automatic_inference_eligible": False,
        "wall_seconds": time.monotonic() - began,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, default=replay.default, allow_nan=False) + "\n"
    )
    base_rung = next(row for row in rungs if row["rung"] == "base")
    print(
        json.dumps(
            {
                "attempt": report["attempt_id"],
                "flights": payload["flight_count"],
                "base_accepted_flights": base_rung["verdict"]["accepted_flight_count"],
                "base_complete": base_rung["verdict"]["complete_point"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
