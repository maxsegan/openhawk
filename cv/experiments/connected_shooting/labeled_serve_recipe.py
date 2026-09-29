"""Explicit labeled serve recipe: unchanged isolated path or joint non-net prefix.

Status: opened-development adapter, not automatic inference or a whole-cohort
recipe. Inputs name an ordinary search stage, labels, packet, cameras and players.
A reused candidate is distinct from input-ranked selection within that stage;
neither is a from-video initialization. Prefix mode adjusts one following launch.

Example: python -m cv.experiments.connected_shooting.labeled_serve_recipe
--mechanism prefix --source-stage input-ranked-refined --search-report ...
--labels ... --packet ... --cameras ... --pose-csv ... --output ...
"""

from __future__ import annotations

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from cv.experiments.connected_shooting import (
    full_native_continuation as full,
    labeled_context_census as census,
    labeled_context_witness_scope as scope,
    labeled_free_toss_support as free_toss,
    labeled_isolated_serve as isolated_cli,
    labeled_prefix_joint_impact as prefix,
    labeled_preparation_net_followup as followup,
    per_flight_acceptance as acceptance,
    per_flight_rescore as rescore,
)
from cv.pipeline import provenance


def select_candidate(
    search: dict, n: int, stage: str, index: int | None, *, input_admission=None
) -> tuple[dict, dict]:
    """Select by input score or explicit index; never read acceptance/selected rows."""
    optional_selection = search.get("optional_contact_selection")
    if optional_selection is not None:
        from cv.pipeline import s6_optional_contacts

        if index is not None:
            raise ValueError("optional contacts cannot accept a caller-selected fitted candidate")
        branch = {
            "coarse": search.get("coarse_candidates", []),
            "refined": search.get("refined_candidates", []),
        }
        hypothesis = {
            "added": [None] * optional_selection["added_contact_count"],
            "occurrence_log_odds": optional_selection["occurrence_log_odds"],
        }
        from cv.pipeline import s6_optional_event_union as union
        from cv.pipeline import s6_terminal_net_membership as net_membership

        policy = optional_selection.get("policy")
        if policy == net_membership.SELECTION_POLICY:
            # A membership run with no ordinary optional witness adds no event;
            # only the declared common-reference complexity offset applies.
            offset = optional_selection.get("added_parameter_count")
            if (
                type(offset) is not int
                or offset < 0
                or optional_selection["added_contact_count"] != 0
                or optional_selection.get("added_bounce_count") != 0
            ):
                raise ValueError("membership selection must declare its added parameter complexity")
            hypothesis.update(added=[], added_parameter_count=offset)
        elif policy == "source_witness_optional_bounce_v1":
            count = optional_selection.get("added_bounce_count")
            if (
                type(count) is not int
                or count < 0
                or optional_selection["added_contact_count"] != 0
                or optional_selection.get("added_parameter_count") != 0
            ):
                raise ValueError(
                    "optional bounce selection must preserve contact parameter dimensions"
                )
            hypothesis.update(added=[None] * count, added_parameter_count=0)
        elif policy == union.POLICIES[union.JOINT]:
            # Composed winner: the contacts carry the continuous dimensions and
            # the bounces add none, exactly as each single arm already declares.
            contacts = optional_selection["added_contact_count"]
            bounces = optional_selection.get("added_bounce_count")
            if (
                type(contacts) is not int
                or type(bounces) is not int
                or contacts < 0
                or bounces < 0
                or optional_selection.get("source_family") != union.JOINT
                or optional_selection.get("added_parameter_count") != 6 * contacts
            ):
                raise ValueError(
                    "composed optional selection must preserve contact parameter dimensions"
                )
            hypothesis.update(
                added=[None] * (contacts + bounces), added_parameter_count=6 * contacts
            )
        membership_offset = optional_selection.get("terminal_net_membership_parameter_offset", 0)
        if membership_offset and policy != net_membership.SELECTION_POLICY:
            # The winning branch was ranked on the common membership reference;
            # reproduce exactly that complexity, never a different one.
            declared = hypothesis.get("added_parameter_count")
            hypothesis["added_parameter_count"] = (
                6 * len(hypothesis["added"]) if declared is None else declared
            ) + membership_offset
        membership_block = optional_selection.get(net_membership.RECEIPT_KEY)
        dimensions_of = None
        if membership_block is not None:
            # Replay the same candidate-specific complexity the search ranked
            # on: each candidate's own declared response construction receipt.
            selected_membership = membership_block["selected_membership"]
            hypothesis["occurrence_log_odds"] += net_membership.replay_occurrence_prior(
                membership_block
            )

            def dimensions_of(row, name=selected_membership):
                return net_membership.candidate_dimensions(name, row)

        chosen, score = s6_optional_contacts.best_completed(
            branch,
            hypothesis,
            optional_selection["training_observations"],
            input_admission=(
                input_admission
                if optional_selection.get("input_domain_selection")
                == "original_physical_domain_before_rank_v1"
                else None
            ),
            added_dimensions=dimensions_of,
        )
        if membership_block is not None and net_membership.candidate_dimensions(
            selected_membership, chosen
        ) != optional_selection.get("terminal_net_membership_response_dimensions"):
            raise ValueError(
                "selected membership candidate response complexity differs from its record"
            )
        parameters = np.asarray(chosen["measurement"]["fit"]["parameters"], float)
        if parameters.shape != (5 + 6 * n,) or not np.isfinite(parameters).all():
            raise ValueError("optional contact candidate vector does not match selected topology")
        if input_admission is not None and not input_admission(chosen)["admissible"]:
            raise ValueError("selected optional candidate violates original input event domain")
        if not np.isclose(score, optional_selection["score"], rtol=1e-12, atol=1e-12):
            raise ValueError("optional contact selection does not reproduce its physical score")
        return chosen, {
            "source_stage": "input-ranked-optional-contact",
            "rule": "bounded physical engineering objective",
            "gate_used": False,
            "source_rank": score,
            "candidate_count": len(branch["coarse"]) + len(branch["refined"]),
        }
    candidates = search.get("refined_candidates", [])
    if stage == "input-ranked-completed":
        from cv.experiments.connected_shooting.candidate_attempts import completed_source_allowed

        if not completed_source_allowed(search):
            raise ValueError(
                "completed-candidate selection requires an exhausted declared deadline or numerical-failure fallback"
            )
        candidates = search.get("completed_candidates", [])
    domain_receipts = []
    if stage == "explicit-refined":
        if index is None or not 0 <= index < len(candidates):
            raise ValueError("explicit-refined requires an in-range candidate-index")
        chosen = candidates[index]
        if input_admission is not None and not input_admission(chosen)["admissible"]:
            raise ValueError("explicit candidate violates the original input event domain")
        rule = "explicit caller-supplied candidate index; development source reuse"
    elif stage in ("input-ranked-refined", "input-ranked-completed"):
        if index is not None:
            raise ValueError("candidate-index cannot override input-ranked selection")
        valid = []
        for i, candidate in enumerate(candidates):
            score = candidate.get("evidence", {}).get("input_only_rank_score")
            p = np.asarray(candidate.get("measurement", {}).get("fit", {}).get("parameters", []))
            if (
                score is not None
                and np.isfinite(score)
                and p.shape == (5 + 6 * n,)
                and np.isfinite(p).all()
            ):
                if input_admission is not None:
                    admission = input_admission(candidate)
                    domain_receipts.append({"candidate_index": i, **admission})
                    if not admission["admissible"]:
                        continue
                valid.append((float(score), float(candidate["depth_hypothesis_m"]), i))
        if not valid:
            raise ValueError("no finite full-point candidate with recorded input-only rank")
        _, _, index = min(valid)
        chosen = candidates[index]
        rule = "lowest recorded input_only_rank_score, depth, original index"
        if input_admission is not None:
            rule += "; original input event domain required before ranking"
    else:
        raise ValueError("unsupported source stage")
    p = np.asarray(chosen["measurement"]["fit"]["parameters"], float)
    if p.shape != (5 + 6 * n,) or not np.isfinite(p).all():
        raise ValueError("candidate vector does not match the original scene topology")
    return chosen, dict(
        source_stage=stage,
        selected_index=index,
        rule=rule,
        gate_used=False,
        source_rank=chosen.get("evidence", {}).get("input_only_rank_score"),
        candidate_count=len(candidates),
        from_scratch_initialization=False,
        **({"input_event_domain": domain_receipts} if input_admission is not None else {}),
    )


def prefix_serve_evidence(labels: dict, packet: dict, search: dict, qualified: bool) -> dict:
    """Require matching original contacts; an explicit non-serve cannot be overridden."""
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one explicitly separated original attempt required")
    ending = str(labels["attempt"].get("ending_kind", "")).lower()
    if "net" in ending.replace("-", "_").split("_"):
        raise ValueError("explicit net ending requires the separate net-aware recipe")
    attempt = packet["attempts"][0]
    clip = attempt["point_clip"]
    groups = [
        [
            e
            for e in labels["events"]["records"]
            if e.get("clip", clip) == clip
            and e["event_type"] == "contact"
            and (
                e.get("status", "labeled") in ("labeled", "ambiguous")
                or occurrence.predicted_membership(e)
            )
        ],
        [e for e in attempt["events"] if e["event_type"] == "contact"],
        [e for e in search["events"] if e["event_type"] == "contact"],
    ]
    groups = [sorted(g, key=lambda e: e["frame"]) for g in groups]
    if len(groups[0]) < 2 or any(len(g) != len(groups[0]) for g in groups):
        raise ValueError("matching original multi-contact label/packet/search topology required")
    epochs = [[float(e["frame"]) for e in g] for g in groups]
    if (
        epochs[0] != epochs[1]
        or epochs[0] != epochs[2]
        or float(labels["attempt"]["first_contact_frame"]) != epochs[0][0]
    ):
        raise ValueError("original contact epochs disagree between supplied sources")
    first = groups[0][0]
    if not occurrence.resolved_membership(first):
        raise ValueError("ambiguous first-contact membership requires separate input preparation")
    roles = [
        str(e[k]).lower()
        for g in groups
        for e in g[:1]
        for k in ("stroke", "role", "shot_type")
        if e.get(k) is not None
    ]
    if any(role != "serve" for role in roles):
        raise ValueError("explicit original non-serve role cannot receive serve fitting")
    typed = bool(roles) or first.get("serve_number") in (1, 2)
    if not typed and not qualified:
        raise ValueError("original typed serve or explicit qualified-first-contact-serve required")
    return dict(
        role="serve",
        origin="original_typed_contact" if typed else "explicit_qualified_flag",
        supplied_role_fields=roles,
        contact_count=len(groups[0]),
        contact_epoch=epochs[0][0],
        human_derived=True,
    )


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(full.profile.jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def delegate_isolated(arguments: list[str], stage: str, index: int | None) -> None:
    """Call the existing CLI with identical numerical arguments and defaults."""
    if stage != "input-ranked-refined" or index is not None:
        raise ValueError("isolated path preserves original input-ranked source selection")
    previous = sys.argv
    try:
        sys.argv = [isolated_cli.__file__, "--experiment", "isolated-serve-joint", *arguments]
        isolated_cli.main()
    finally:
        sys.argv = previous
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    args, _ = output_parser.parse_known_args(arguments)
    save(
        args.output / "adapter.json",
        dict(
            mechanism="isolated",
            source_stage=stage,
            numerics="unchanged delegated labeled_isolated_serve.main",
            automatic_inference_eligible=False,
            sources=[
                provenance.file_record(Path(__file__)),
                provenance.file_record(Path(isolated_cli.__file__)),
            ],
        ),
    )


def prefix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint non-net serve prefix with one following launch"
    )
    for name in ("search-report", "labels", "packet", "cameras", "pose-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pose-image-scale", type=float, default=1.0)
    parser.add_argument("--player-order", action="append", default=[])
    parser.add_argument("--observation-fallback", choices=["off", "on"], default="off")
    parser.add_argument("--witness-surface", choices=["hard", "clay", "grass"])
    parser.add_argument("--terminal-context", choices=["source", "observed"], default="source")
    parser.add_argument("--qualified-first-contact-serve", action="store_true")
    parser.add_argument("--free-toss-support", type=Path)
    parser.add_argument(
        "--score-policy", choices=["ordinary", "isolated-serve-scope"], default="ordinary"
    )
    parser.add_argument(
        "--incoming-initializer",
        choices=["linear_clipped", "bounded_front"],
        default="bounded_front",
    )
    parser.add_argument("--toss-weight", type=float, default=4.0)
    parser.add_argument(
        "--free-first-rebound",
        action="store_true",
        help="fit only the serve rebound scales; initialize vertical scale at unclipped0.95",
    )
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def run_prefix(args: argparse.Namespace, stage: str, index: int | None) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    sources = []
    start = dict(
        configuration={
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "mechanism": "prefix",
            "source_stage": stage,
            "candidate_index": index,
            "following_launches": 1,
        },
        sources=sources,
        human_derived=True,
        automatic_inference_eligible=False,
    )
    save(args.output / "start.json", start)
    phase = "input_preflight"
    try:
        paths = [
            args.search_report,
            args.labels,
            args.packet,
            args.cameras,
            args.pose_csv,
            Path(__file__),
            Path(prefix.__file__),
            Path(followup.__file__),
            Path(rescore.__file__),
            Path(census.__file__),
            Path(acceptance.__file__),
            Path(scope.__file__),
            Path(full.model.__file__),
            Path(full.exposure.__file__),
            Path(prefix.toss_front.__file__),
        ]
        if args.incoming_initializer == "bounded_front":
            from cv.experiments.connected_shooting import labeled_toss_velocity_initializer

            paths.append(Path(labeled_toss_velocity_initializer.__file__))
        if args.free_toss_support is not None:
            paths.extend([args.free_toss_support, Path(free_toss.__file__)])
        sources.extend(provenance.file_record(p) for p in paths)
        save(args.output / "start.json", start)
        search, labels, packet, cameras = [
            json.loads(p.read_text())
            for p in [args.search_report, args.labels, args.packet, args.cameras]
        ]
        role = prefix_serve_evidence(labels, packet, search, args.qualified_first_contact_serve)
        args.athlete_prior_mode = "stature_pose_soft"
        args.dense_labels = args.player_ledger = None
        context = rescore.build_context(args, search)
        candidate, selection = select_candidate(search, len(context["scene"].pixels), stage, index)
        context = rescore.context_at_candidate_epoch(context, candidate)
        # The builder changes only scene epochs. Update the representative event,
        # using the existing profile helper; original intervals/pictures remain.
        context = full.profile.profile_context(context, float(context["scene"].contact_frames[0]))
        context["prefix_serve_evidence"] = role
        reproduction = rescore.reproduction_receipt(context, candidate, args.athlete_prior_mode)
        if not reproduction["identical"]:
            raise ValueError("ordinary source candidate checks did not reproduce")
        clip = packet["attempts"][0]["point_clip"]
        observations = isolated_cli.bind_toss(search, labels, cameras, clip)
        toss_input = dict(mode="original_cached_precontact")
        if args.free_toss_support is not None:
            support = json.loads(args.free_toss_support.read_text())
            observations = free_toss.observations(labels, cameras, clip, support)
            toss_input.update(mode="explicit_original_free_toss_span", support=support)
        observations = prefix.toss_front.enrich(observations, labels, clip)
        source = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
        threshold = next(
            r["thresholds"] for r in rescore.ladder() if r["name"] == isolated_cli.RUNG
        )
        duration = search["configuration"]["exposure_duration_frames"]
        ending = labels["attempt"]["ending_kind"]
        scorer = acceptance.score if args.score_policy == "ordinary" else scope.LOCAL_SCORE
        source_before, source_measurement = census.score(
            context, source, threshold, duration, ending, scorer=scorer
        )
        context_receipt = dict(mode="source")
        if args.terminal_context == "observed" and context.get("terminal_rebound_frames") is None:
            context, context_receipt = census.extended_context(context, labels, cameras, search)
            if context is None:
                raise ValueError(f"observed terminal context unsupported: {context_receipt}")
            # Compare the same source vector and fitted vector on the same
            # explicitly activated observations, retaining source-only evidence.
            before, original_measurement = census.score(
                context, source, threshold, duration, ending, scorer=scorer
            )
        else:
            before, original_measurement = source_before, source_measurement
        eligibility = prefix.qualify(context)
        incoming = prefix._prefix_rows(observations, eligibility["contact"][0], duration)
        preflight = dict(
            status="passed",
            eligibility=eligibility,
            source_reproduction=reproduction,
            seed_selection=selection,
            context=context_receipt,
            toss_input=toss_input,
            incoming_frames=[r["frame"] for r in incoming],
            original_parameters=source,
            source_before=source_before,
            source_measurement=source_measurement,
            before=before,
            original_measurement=original_measurement,
        )
        save(args.output / "preflight.json", preflight)
        if args.preflight_only:
            result = dict(
                **start,
                status="preflight_only_no_fit",
                **{k: v for k, v in preflight.items() if k != "status"},
            )
        else:
            phase = "fit"
            # Extended ending context can expose the same native row in both
            # splits. Consume it once in fitting; retain both original splits
            # for the before/after evaluation.
            fit_checks, consumed = followup.fit_check_copy(context["scene"], context["heldout"])
            _, fit = prefix.fit(
                context | {"heldout": fit_checks},
                source,
                observations,
                toss_weight=args.toss_weight,
                duration=duration,
                max_nfev=args.max_nfev,
                seconds=args.seconds,
                following_launches=1,
                incoming_initializer=args.incoming_initializer,
                free_first_rebound=args.free_first_rebound,
                first_rebound_initial_scales=(
                    [0.95, float(source[-1])] if args.free_first_rebound else None
                ),
            )
            save(args.output / "fit.json", fit)
            phase = "original_rescore"
            active = full.profile.profile_context(context, fit["contact_epoch"])
            with prefix.using_first_rebound(
                fit["contact_epoch"], fit.get("first_flight_rebound_scales")
            ):
                after, measurement = census.score(
                    active, fit["full_vector"], threshold, duration, ending, scorer=scorer
                )
            result = dict(
                **start,
                status="measured_requires_native_review",
                attempt_id=packet["attempts"][0]["attempt_id"],
                clip=clip,
                seed_selection=selection,
                serve_role_evidence=role,
                source_reproduction=reproduction,
                context_receipt=context_receipt,
                toss_input=toss_input,
                source_before=source_before,
                source_measurement=source_measurement,
                before=before,
                original_measurement=original_measurement,
                after=after,
                measurement=measurement,
                fit=fit,
                fit_only_duplicate_check_rows_removed=consumed,
                original_parameters=source,
                evaluation_scene=asdict(active["scene"]),
                evaluation_heldout=asdict(active["heldout"]),
                events=active["events"],
                original_targets=active["targets"],
                terminal_rebound_frames=active.get("terminal_rebound_frames"),
                selection_scope="Input objective within explicitly declared post-source stage; scored afterward; no cohort composition.",
                independent_xyz_truth_available=False,
            )
    except Exception as error:
        result = dict(
            **start,
            status=(
                "ineligible"
                if phase == "input_preflight"
                and isinstance(error, ValueError)
                and not isinstance(error, json.JSONDecodeError)
                else "execution_failed"
            ),
            stage=phase,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    for record in sources:
        base = (
            full.paths.data_root()
            if record["path_base"] == "TENNIS_DATA_ROOT"
            else full.paths.REPO_ROOT
        )
        if provenance.file_sha256(base / record["path"]) != record["sha256"]:
            result.update(status="source_mutation_detected", mutated_source=record)
    save(args.output / "report.json", result)
    print(
        result["status"], result.get("error", result.get("after", {}).get("accepted_flight_count"))
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--mechanism", choices=["isolated", "prefix"], required=True)
    parser.add_argument(
        "--source-stage", choices=["input-ranked-refined", "explicit-refined"], required=True
    )
    parser.add_argument("--candidate-index", type=int)
    arguments = sys.argv[1:] if argv is None else argv
    if arguments in (["--help"], ["-h"]):
        parser.print_help()
        print("\nPrefix inputs/options:")
        prefix_parser().print_help()
        print("\nIsolated mode forwards all remaining flags unchanged to labeled_isolated_serve.")
        return
    args, remaining = parser.parse_known_args(arguments)
    if args.mechanism == "isolated":
        delegate_isolated(remaining, args.source_stage, args.candidate_index)
    else:
        run_prefix(prefix_parser().parse_args(remaining), args.source_stage, args.candidate_index)


if __name__ == "__main__":
    main()
