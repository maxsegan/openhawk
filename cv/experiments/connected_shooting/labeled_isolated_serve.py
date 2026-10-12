"""Run an explicit isolated-serve joint-toss experiment from ordinary labeled inputs.

Status: experimental, not the automatic pipeline. Requires the same packet,
labels, cameras and pose file used by a supplied ordinary whole-point search.
Selects its numerical seed by recorded input rank, never by a reviewed cohort row.
One reusable mechanism, not the complete reviewed development recipe.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

import argparse
from dataclasses import asdict
import json
import inspect
from pathlib import Path
import traceback

import numpy as np

from cv.experiments.connected_shooting import (
    full_native_continuation as full,
    labeled_context_census as census,
    labeled_context_witness_scope as scope,
    labeled_isolated_serve_fit as isolated,
    per_flight_rescore as rescore,
)
from cv.pipeline import provenance

RUNG = "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win32_shift1f"


def select_seed(search: dict) -> tuple[dict, dict]:
    """Use one same-objective input rank among finite existing refined seeds."""
    candidates = []
    for i, row in enumerate(search.get("refined_candidates", [])):
        score = row.get("evidence", {}).get("input_only_rank_score")
        parameters = row.get("measurement", {}).get("fit", {}).get("parameters")
        if score is None or parameters is None:
            continue
        p = np.asarray(parameters, float)
        if p.shape != (11,) or not np.isfinite(p).all() or not np.isfinite(score):
            continue
        candidates.append((float(score), float(row["depth_hypothesis_m"]), i, row))
    if not candidates:
        raise ValueError("no finite one-flight refined seed with recorded input-only rank")
    score, depth, index, chosen = min(candidates, key=lambda r: r[:3])
    return chosen, dict(
        rule="lowest recorded input_only_rank_score, depth, original index",
        selected_index=index,
        source_rank=score,
        source_depth_m=depth,
        candidate_count=len(candidates),
        gate_used=False,
    )


def serve_evidence(labels: dict, packet: dict, search: dict, *, qualified: bool = False) -> dict:
    """Require original one-contact attempt semantics, not a truncated search shape."""
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one explicitly separated packet attempt required")
    attempt = packet["attempts"][0]
    clip = attempt["point_clip"]
    contacts = [
        e
        for e in labels["events"]["records"]
        if e["event_type"] == "contact"
        and e.get("clip", clip) == clip
        and (
            e.get("status", "labeled") in ("labeled", "ambiguous")
            or occurrence.predicted_membership(e)
        )
    ]
    packet_contacts = [e for e in attempt["events"] if e["event_type"] == "contact"]
    search_contacts = [e for e in search["events"] if e["event_type"] == "contact"]
    if len(contacts) != 1 or len(packet_contacts) != 1 or len(search_contacts) != 1:
        raise ValueError("original label, packet and search must each contain exactly one contact")
    event = contacts[0]
    if not occurrence.resolved_membership(event):
        raise ValueError("ambiguous original contact membership requires separate qualification")
    epoch = float(event["frame"])
    if (
        any(float(e["frame"]) != epoch for e in [packet_contacts[0], search_contacts[0]])
        or float(labels["attempt"]["first_contact_frame"]) != epoch
    ):
        raise ValueError("original contact epoch disagrees between label, packet and search")
    roles = [
        str(event[k]).lower() for k in ("stroke", "shot_type", "role") if event.get(k) is not None
    ]
    if any(role != "serve" for role in roles):
        raise ValueError("original contact is explicitly not a serve")
    ending = str(labels["attempt"].get("ending_kind", ""))
    typed = (
        bool(roles)
        or event.get("serve_number") in (1, 2)
        or ending.startswith("serve_")
        or ending in ("unreturned_serve", "ace")
    )
    if not typed and not qualified:
        raise ValueError(
            "serve role not typed; explicit qualified-first-contact-serve input required"
        )
    return dict(
        role="serve",
        origin="original_contact_or_serve_ending" if typed else "explicit_qualified_flag",
        contact_epoch=epoch,
        label_contacts=1,
        packet_contacts=1,
        search_contacts=1,
        supplied_role_fields=roles,
        ending_kind=ending,
        human_derived=True,
    )


def bind_toss(search: dict, labels: dict, cameras: dict, clip: str) -> dict:
    """Check cached toss against explicitly supplied original native pixels/cameras."""
    observation = search.get("toss_observations", {})
    pixels = {
        float(r["frame"]): r
        for group in labels["ball"]["records"]
        if group["clip"] == clip
        for r in group["frames"]
        if r["status"] == "visible"
    }
    camera = {float(r["frame"]): r for r in cameras["cameras"]}
    for row in observation.get("rows", []):
        f = float(row["frame"])
        original, view = pixels.get(f), camera.get(f)
        if (
            original is None
            or view is None
            or view.get("supported", True) is False
            or view.get("status") != "supported"
        ):
            raise ValueError("cached toss lacks supported original label/camera")
        if not np.array_equal(
            row["pixel"], [original["x1080"], original["y1080"]]
        ) or not np.array_equal(row["camera"], view.get("P")):
            raise ValueError("cached toss differs from original native label/camera")
        from cv.experiments.connected_shooting import camera_geometry

        expected = camera_geometry.radial_row(view)
        supplied = camera_geometry.radial_row(row)
        if expected is not None and expected[0] == 0 and supplied is None:
            continue  # Explicit zero lens is the same historical pinhole observation.
        if not ((expected is None and supplied is None) or np.array_equal(expected, supplied)):
            raise ValueError("radial toss metadata differs from the frozen camera record")
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["isolated-serve-joint"], required=True)
    for name in ["search-report", "labels", "packet", "cameras", "pose-csv", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pose-image-scale", type=float, default=1.0)
    parser.add_argument("--player-order", action="append", default=[])
    parser.add_argument("--observation-fallback", choices=["off", "on"], default="off")
    parser.add_argument("--witness-surface", choices=["hard", "clay", "grass"])
    parser.add_argument("--terminal-context", choices=["source", "observed"], default="source")
    parser.add_argument("--contact-mode", choices=["fixed", "continuous"], default="continuous")
    parser.add_argument("--free-rebound-scales", action="store_true")
    parser.add_argument(
        "--contact-height-bounds",
        type=float,
        nargs=2,
        default=[2.0, 4.0],
        help="Explicit numerical search envelope, not an inferred player height",
    )
    parser.add_argument(
        "--qualified-first-contact-serve",
        action="store_true",
        help="Explicit reviewed serve-role input only when original typed role is absent; cannot override non-serve labels",
    )
    parser.add_argument(
        "--free-toss-support",
        type=Path,
        help="Explicit JSON with frame_interval, rationale and annotation_origin for native-reviewed free toss; original labels stay unchanged",
    )
    parser.add_argument(
        "--toss-horizontal-sigma-mps",
        type=float,
        help="Optional positive soft velocity preference scale; not an empirical limit or release observation",
    )
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--seconds", type=float, default=300.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = []
    start = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sources=sources,
        human_derived=True,
        automatic_inference_eligible=False,
    )
    (args.output / "start.json").write_text(json.dumps(start, indent=2) + "\n")
    stage = "input_preflight"
    try:
        sources.extend(
            [
                provenance.file_record(p)
                for p in [
                    args.search_report,
                    args.labels,
                    args.packet,
                    args.cameras,
                    args.pose_csv,
                    Path(__file__),
                    Path(isolated.__file__),
                    Path(inspect.getfile(getattr(isolated.fit, "func", isolated.fit))),
                    Path(inspect.getfile(isolated.FirstFlightProjector)),
                ]
            ]
        )
        if args.free_toss_support is not None:
            from cv.experiments.connected_shooting import labeled_free_toss_support as free_toss

            sources.extend(
                provenance.file_record(p)
                for p in [args.free_toss_support, Path(free_toss.__file__)]
            )
        if args.toss_horizontal_sigma_mps is not None:
            if (
                not np.isfinite(args.toss_horizontal_sigma_mps)
                or args.toss_horizontal_sigma_mps <= 0
            ):
                raise ValueError("positive finite soft horizontal velocity scale required")
        (args.output / "start.json").write_text(json.dumps(start, indent=2) + "\n")
        search, labels, packet, cameras = [
            json.loads(p.read_text())
            for p in [args.search_report, args.labels, args.packet, args.cameras]
        ]
        if len(packet["attempts"]) != 1:
            raise ValueError("one explicitly separated packet attempt required")
        attempt = packet["attempts"][0]
        role = serve_evidence(labels, packet, search, qualified=args.qualified_first_contact_serve)
        candidate, seed_receipt = select_seed(search)
        observations = bind_toss(search, labels, cameras, attempt["point_clip"])
        toss_input = dict(
            mode="original_cached_precontact",
            original_cached_rows=len(observations.get("rows", [])),
        )
        if args.free_toss_support is not None:
            support = json.loads(args.free_toss_support.read_text())
            observations = free_toss.observations(labels, cameras, attempt["point_clip"], support)
            toss_input.update(
                mode="explicit_original_free_toss_span",
                support=support,
                selected_rows=len(observations["rows"]),
                abstained=observations["abstained"],
            )

        args.athlete_prior_mode = "stature_pose_soft"
        args.dense_labels = args.player_ledger = None
        context = rescore.build_context(args, search)
        context["isolated_serve_evidence"] = role
        original_receipt = rescore.reproduction_receipt(context, candidate, args.athlete_prior_mode)
        if not original_receipt["identical"]:
            raise ValueError("ordinary source candidate did not reproduce")
        source = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
        threshold = next(r["thresholds"] for r in rescore.ladder() if r["name"] == RUNG)
        duration = search["configuration"]["exposure_duration_frames"]
        ending = labels["attempt"]["ending_kind"]
        before, original_measurement = census.score(
            context, source, threshold, duration, ending, scorer=scope.LOCAL_SCORE
        )
        context_receipt = dict(mode="source")
        if args.terminal_context == "observed" and context.get("terminal_rebound_frames") is None:
            context, context_receipt = census.extended_context(context, labels, cameras, search)
            if context is None:
                raise ValueError(
                    f"original observed terminal context unsupported: {context_receipt}"
                )
        eligibility = isolated.qualify(context, observations, duration)
        (args.output / "preflight.json").write_text(
            json.dumps(
                full.profile.jsonable(
                    dict(
                        status="passed",
                        eligibility=eligibility,
                        source_reproduction=original_receipt,
                        seed_selection=seed_receipt,
                        context=context_receipt,
                        toss_input=toss_input,
                    )
                ),
                indent=2,
            )
            + "\n"
        )
        stage = "fit"
        fit_options = {}
        if args.toss_horizontal_sigma_mps is not None:
            fit_options["toss_horizontal_sigma_mps"] = args.toss_horizontal_sigma_mps
        active, fit = isolated.fit(
            context,
            source,
            observations,
            duration,
            fixed_contact_epoch=float(context["scene"].contact_frames[0])
            if args.contact_mode == "fixed"
            else None,
            free_rebound_scales=args.free_rebound_scales,
            height_bounds=tuple(args.contact_height_bounds),
            maxiter=args.max_nfev,
            seconds=args.seconds,
            **fit_options,
        )
        stage = "original_rescore"
        after, measurement = census.score(
            active, fit["best"]["parameters"], threshold, duration, ending, scorer=scope.LOCAL_SCORE
        )
        result = dict(
            **start,
            status="measured_requires_native_review",
            attempt_id=attempt["attempt_id"],
            clip=attempt["point_clip"],
            seed_selection=seed_receipt,
            serve_role_evidence=role,
            source_reproduction=original_receipt,
            context_receipt=context_receipt,
            toss_input=toss_input,
            before=before,
            original_measurement=original_measurement,
            after=after,
            measurement=measurement,
            fit=fit,
            original_parameters=source,
            evaluation_scene=asdict(active["scene"]),
            evaluation_heldout=asdict(active["heldout"]),
            events=active["events"],
            original_targets=active["targets"],
            terminal_rebound_frames=active.get("terminal_rebound_frames"),
            selection_scope="Lowest objective iterate from this one fixed mechanism; acceptance scored afterward; no cohort composition.",
            independent_xyz_truth_available=False,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = dict(
            **start,
            status="ineligible" if stage == "input_preflight" else "execution_failed",
            stage=stage,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    for record in sources:
        p = (
            full.paths.data_root()
            if record["path_base"] == "TENNIS_DATA_ROOT"
            else full.paths.REPO_ROOT
        ) / record["path"]
        if provenance.file_sha256(p) != record["sha256"]:
            result.update(status="source_mutation_detected", mutated_source=record)
    (args.output / "report.json").write_text(
        json.dumps(full.profile.jsonable(result), indent=2) + "\n"
    )
    print(
        result["status"], result.get("error", result.get("after", {}).get("accepted_flight_count"))
    )


if __name__ == "__main__":
    main()
