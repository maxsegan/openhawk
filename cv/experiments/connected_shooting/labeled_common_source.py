"""Load explicit original labeled inputs and an input-ranked cached search stage.

No cohort composition or acceptance-selected depth is a source. This shared
post-search adapter does not generate labels, change physics or fit parameters.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import copy
import json
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import labeled_serve_recipe as recipe
from cv.experiments.connected_shooting import labeled_contact_association as association
from cv.pipeline import provenance, s6_refinement_input_context
from cv.experiments.connected_shooting import observation_net_seed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("search-report", "labels", "packet", "cameras", "pose-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pose-image-scale", type=float, default=1.0)
    parser.add_argument("--player-order", action="append", default=[])
    parser.add_argument("--athlete-evidence", choices=["required", "optional"], default="required")
    parser.add_argument(
        "--athlete-root-reach-loss", choices=["quadratic", "cauchy"], default="quadratic"
    )
    parser.add_argument("--observation-fallback", choices=["off", "on"], default="off")
    parser.add_argument("--witness-surface", choices=["hard", "clay", "grass"])
    parser.add_argument("--association-policy", choices=association.POLICIES, default="original")
    parser.add_argument(
        "--source-stage",
        choices=["input-ranked-refined", "input-ranked-completed"],
        default="input-ranked-refined",
    )


def input_domain_admission(
    context: dict, *, include_seeded_final_net=False
) -> Callable[[dict], dict] | None:
    """Replay original event-domain eligibility, never cached acceptance bits."""
    admission = None
    if getattr(context["scene"], "terminal_net_tail", None) is not None or (
        include_seeded_final_net and observation_net_seed.single_final_net(context["scene"])
    ):
        from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
            original_domain_slacks_frames,
            domain_admissible,
        )

        def admission(candidate):
            fit = candidate["measurement"]["fit"]
            if (
                context["scene"].terminal_net_tail is None
                and fit.get("net_response") is None
                and fit.get("net_response_initialization") is None
            ):
                return dict(
                    admissible=True,
                    physical_replay=False,
                    reason="candidate has no optional net response",
                )
            try:
                candidate_context = recipe.rescore.context_at_candidate_epoch(context, candidate)
                parameters = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
                scene = candidate_context["scene"]
                if scene.terminal_net_tail is None:
                    observation_net_seed.require_ground_ending_binding(
                        scene,
                        fit.get("net_response_initialization"),
                        bounces=candidate_context.get("bounces"),
                    )
                # Reintegrate the actual candidate under the active original law;
                # cached acceptance bits and fitted diagnostic measurements are unused.
                with observation_net_seed.response_context(candidate["measurement"]["fit"]):
                    flights = recipe.full.model.chain(scene, parameters)
                slacks = original_domain_slacks_frames(scene, flights)
                mesh = {}
                mesh_admissible = True
                if candidate["measurement"]["fit"].get("net_response") is not None:
                    from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                        mesh_response_slacks_m,
                    )

                    mesh_slacks = mesh_response_slacks_m(
                        scene, flights, include_supplied_final=include_seeded_final_net
                    )
                    mesh_admissible = bool(
                        np.isfinite(mesh_slacks).all() and np.all(mesh_slacks >= -1e-6)
                    )
                    mesh = dict(mesh_response_slacks_m=mesh_slacks.tolist())
                return dict(
                    admissible=domain_admissible(slacks) and mesh_admissible,
                    **mesh,
                    slacks_frames=slacks.tolist(),
                    physical_replay=True,
                )
            except (ValueError, FloatingPointError, OverflowError) as error:
                return dict(admissible=False, physical_replay=False, reason=str(error))

    return admission


def load(arguments: argparse.Namespace) -> dict:
    """Reproduce one finite cached candidate using only its recorded input rank."""
    args = copy(arguments)
    args.athlete_prior_mode = "stature_pose_soft"
    # Strict by default: the cached search must declare the same policy, and an
    # ``optional`` replay reproduces exactly the supplied/unavailable state it fitted.
    args.athlete_evidence = getattr(arguments, "athlete_evidence", "required")
    args.athlete_root_reach_loss = getattr(arguments, "athlete_root_reach_loss", "quadratic")
    args.player_order = list(getattr(arguments, "player_order", []) or [])
    args.dense_labels = args.player_ledger = None
    paths = [args.search_report, args.labels, args.packet, args.cameras, args.pose_csv]
    sources = [provenance.file_record(p) for p in paths]
    search, labels, packet, cameras = [json.loads(p.read_text()) for p in paths[:4]]
    from cv.pipeline import s6_player_camera

    s6_player_camera.require_replay_inputs(search, args.pose_csv, args.cameras)
    from cv.experiments.connected_shooting import net_collision

    net_collision.require_policy(search.get("net_collision_policy", "timing_window_v1"))
    from cv.experiments.connected_shooting import observation_operator

    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one explicitly separated original packet attempt required")
    operator = observation_operator.from_report(search, packet)
    policy = getattr(args, "association_policy", "original")
    declared = search.get("experimental_contact_association")
    if declared is not None and declared["policy"] != policy:
        raise ValueError("explicit association policy must match the cached search")
    with association.context(labels, cameras, policy) as association_receipt:
        context = recipe.rescore.build_context(args, search)
    if policy != "original":
        sources.extend(
            provenance.file_record(Path(module.__file__))
            for module in (association, association.launch, association.first_player)
        )
    selected_events = s6_refinement_input_context.admit_selected_events(
        packet["attempts"][0], search
    )
    admission = input_domain_admission(
        context,
        include_seeded_final_net=search.get("configuration", {}).get("observation_net_seed", "off")
        != "off",
    )

    candidate, selection = recipe.select_candidate(
        search,
        len(context["scene"].pixels),
        args.source_stage,
        None,
        **({"input_admission": admission} if admission is not None else {}),
    )
    context = recipe.rescore.context_at_candidate_epoch(context, candidate)
    context = recipe.full.profile.profile_context(
        context, float(context["scene"].contact_frames[0])
    )
    initial_fit = candidate["measurement"]["fit"]
    with observation_net_seed.response_context(initial_fit):
        reproduction = recipe.rescore.reproduction_receipt(
            context, candidate, args.athlete_prior_mode
        )
        if not reproduction["identical"]:
            raise ValueError("input-ranked original source candidate did not reproduce")
        parameters = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
        projection_replay = replay_projection(context, parameters, search, candidate["measurement"])
    threshold = next(
        row["thresholds"]
        for row in recipe.rescore.ladder()
        if row["name"] == recipe.isolated_cli.RUNG
    )
    return dict(
        context=context,
        **({"selected_event_admission": selected_events} if selected_events is not None else {}),
        parameters=parameters,
        **(
            {"initial_search_fit": initial_fit}
            if initial_fit.get("net_response") is not None
            else {}
        ),
        search=search,
        labels=labels,
        packet=packet,
        cameras=cameras,
        selection=selection,
        reproduction=reproduction,
        projection_replay=projection_replay,
        threshold=threshold,
        duration=operator["exposure_duration_frames"],
        observation_operator=operator,
        ending="unresolved"
        if context["scene"].right_boundary_kind == "original_contact"
        else labels["attempt"]["ending_kind"],
        sources=sources,
        association_preparation=association_receipt,
        physics=dict(
            source_surface=search["configuration"]["surface"],
            witness_surface=args.witness_surface,
            source_surface_changed=False,
        ),
    )


def replay_projection(context: dict, parameters, search: dict, measurement: dict) -> dict:
    """Recompute native fronts: cached check agreement alone is not path replay."""
    checks, consumed = recipe.followup.fit_check_copy(context["scene"], context["heldout"])
    active, axes, _ = recipe.full.merge_scene(
        context["scene"], checks, context["bounces"], context["axes"]
    )
    predicted = recipe.full.exposure.prediction(
        active,
        parameters,
        axes,
        search["configuration"]["exposure_duration_frames"],
        termination_kind=context["termination_kind"],
    )
    expected = {}
    for row in measurement["native_projection"]:
        frame = float(row["frame"])
        if frame in expected and any(
            not np.array_equal(row[field], expected[frame][field])
            for field in ("owner", "predicted")
        ):
            raise ValueError("cached duplicate native projections disagree")
        expected[frame] = row
    frames = np.concatenate(active.observation_frames)
    if set(map(float, frames)) != set(expected):
        raise ValueError("original cached native frame inventory differs from source replay")
    target = np.asarray([expected[float(f)]["predicted"] for f in frames], float)
    observed = np.asarray([expected[float(f)]["owner"] for f in frames], float)
    if not np.array_equal(np.concatenate(active.pixels), observed):
        raise ValueError("original cached native pixels differ from supplied source")
    maximum = float(np.max(np.linalg.norm(predicted - target, axis=1)))
    if not np.isfinite(maximum) or maximum > 1e-4:
        raise ValueError(f"original source projection replay differs by {maximum} px")
    return dict(
        native_frames=len(frames),
        maximum_projection_delta_px=maximum,
        original_pixels_identical=True,
        duplicate_check_frames_consumed_once=consumed,
        tolerance_px=1e-4,
    )
