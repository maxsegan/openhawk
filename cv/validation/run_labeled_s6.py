"""Measure a cold locally automatic S6 procedure on frozen labeled observations.

Usage: uv run python -m cv.validation.run_labeled_s6 --manifest FILE --output DIR
Every selected row stays in the denominator, including missing preparations,
solver failures and timeouts. Output must be new; cached ball fits cannot enter.
This first baseline reuses frozen observed camera/2D/event packets and has no
runtime labeling/LLM agent, per-point solver policy or manual fitted seed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path

from cv.pipeline import provenance, s6_labeled_stage as stage
from cv.pipeline.s6_attempt_execution import execute, failure  # noqa: F401  (re-exported)


def parent_case_summary(rows: list[dict], results: list[dict], cohort: dict) -> dict:
    """Keep source-case yield comparable after explicitly splitting serve attempts."""
    mapping = cohort.get("child_parent_map", {})
    if not mapping:
        return {}
    all_keys = {row["key"] for row in cohort["rows"]}
    if set(mapping) - all_keys:
        raise ValueError("parent mapping names an unknown child attempt")
    selected = {row["key"] for row in rows}
    parents = {mapping.get(key, key) for key in selected}
    groups = {}
    for row in cohort["rows"]:
        parent = mapping.get(row["key"], row["key"])
        if parent in parents:
            groups.setdefault(parent, []).append(row["key"])
    lookup = {row["key"]: row for row in results}
    records = []
    for parent, children in groups.items():
        records.append(
            {
                "parent_key": parent,
                "children": children,
                "all_children_selected": set(children) <= selected,
                "all_children_finished": all(key in lookup for key in children),
                "complete_gate_pass": set(children) <= selected
                and all(
                    lookup.get(key, {}).get("verdict", {}).get("complete_point", False)
                    for key in children
                ),
                "has_passing_flight": any(
                    lookup.get(key, {}).get("verdict", {}).get("accepted_flight_count", 0) > 0
                    for key in children
                ),
            }
        )
    return {
        "parent_case_denominator": len(records),
        "gate_accepted_complete_parent_cases": sum(r["complete_gate_pass"] for r in records),
        "parent_cases_with_accepted_flights": sum(r["has_passing_flight"] for r in records),
        "parent_case_rows": records,
        "parent_completion_rule": "every declared child attempt must pass; missing children fail",
    }


def summarize(rows: list[dict], results: list[dict], *, cohort: dict | None = None) -> dict:
    lookup = {row["key"]: row for row in results}
    counts = Counter(row["status"] for row in results)
    verdicts = [lookup.get(row["key"], {}).get("verdict", {}) for row in rows]
    return {
        "schema": "labeled_s6_cold_summary_v1",
        "attempt_denominator": len(rows),
        "declared_flight_denominator": sum(row["declared_flights"] for row in rows),
        "finished_attempts": len(results),
        "pending_attempts": len(rows) - len(results),
        "gate_accepted_complete_attempts": sum(bool(v.get("complete_point")) for v in verdicts),
        "gate_accepted_flights": sum(v.get("accepted_flight_count", 0) for v in verdicts),
        "fitted_scene_flights": sum(v.get("flight_count", 0) for v in verdicts),
        "attempts_with_accepted_flights": sum(
            v.get("accepted_flight_count", 0) > 0 for v in verdicts
        ),
        "statuses": dict(counts),
        "incorrect_accept_count": None,
        "correct_complete_yield": None,
        "scope": "cold shared S6 on fixed prepared labeled observations; gate acceptance, native audit pending",
        "rows": [
            {
                "key": row["key"],
                "status": lookup.get(row["key"], {}).get("status", "pending"),
                "verdict": {
                    k: v
                    for k, v in lookup.get(row["key"], {}).get("verdict", {}).items()
                    if k
                    in (
                        "complete_point",
                        "accepted_flight_count",
                        "flight_count",
                        "accepted_flight_indices",
                        "gaps",
                        "failure_counts",
                    )
                },
                "reason": lookup.get(row["key"], {}).get("reason"),
            }
            for row in rows
        ],
        **parent_case_summary(rows, results, cohort or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--coarse-iterations", type=int, default=150)
    parser.add_argument("--refine-iterations", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--search-seconds",
        type=float,
        help="cooperative search deadline; must reserve at least30s for replay/scoring",
    )
    parser.add_argument(
        "--search-incumbent",
        choices=["off", "on"],
        default="off",
        help="retain a solve's best finite iterate when the search deadline lands inside it",
    )
    parser.add_argument(
        "--observation-partition",
        choices=["fifth_frame_withheld", "all_native"],
        default="fifth_frame_withheld",
    )
    parser.add_argument("--terminal-net-seed", choices=["off", "on"], default="off")
    parser.add_argument(
        "--fit-ground-witness", choices=["off", "interval_ballistic_center"], default="off"
    )
    parser.add_argument("--terminal-ground-sparse-wings", choices=["off", "on"], default="off")
    parser.add_argument("--interior-sparse-wings", choices=["off", "on"], default="off")
    parser.add_argument("--interior-short-blocks", choices=["off", "on"], default="off")
    parser.add_argument("--interior-ground-horizontal", choices=["off", "retention"], default="off")
    parser.add_argument("--preparation", choices=["off", "on"], default="off")
    parser.add_argument("--refinement", choices=["off", "on"], default="off")
    parser.add_argument(
        "--contact-prefix-scope",
        choices=["off", "coverage", "terminal_identity", "unresolved_ending"],
        default="off",
    )
    parser.add_argument("--contact-components", choices=["off", "unresolved_ending"], default="off")
    parser.add_argument(
        "--whole-point-seed-fallback",
        choices=["off", "on"],
        default="off",
        help="on: numerical whole-point seed death degrades to the component partition; "
        "absent/off: the attempt still dies",
    )
    parser.add_argument(
        "--failed-component-split-fallback",
        choices=["off", "on"],
        default="off",
        help="on: a joined multi-flight component that dies or rejects is refit as "
        "one-flight children; absent/off: the joined failure stands",
    )
    parser.add_argument(
        "--component-search-budget",
        choices=["off", "exhausted_retry"],
        default="off",
        help="exhausted_retry: a component that exhausts its search is retried once "
        "with one extra equal search share; absent/off: the 900 s share stands",
    )
    parser.add_argument("--observation-scope", choices=["off", "on"], default="off")
    parser.add_argument("--following-bounce-intervals", choices=["off", "on"], default="off")
    parser.add_argument("--joint-toss-requalification", choices=["off", "on"], default="off")
    parser.add_argument("--earlier-toss-support", choices=["off", "on"], default="off")
    parser.add_argument("--prefix-local-boundary", choices=["off", "on"], default="off")
    parser.add_argument("--prefix-source-incumbent", choices=["off", "on"], default="off")
    parser.add_argument(
        "--prefix-pixel-loss",
        choices=["off", "soft_l1"],
        default="off",
        help="serve-prefix native coordinates only; physical/toss evidence and raw gates unchanged",
    )
    parser.add_argument(
        "--prefix-following-ground-timing",
        choices=["off", "prediction_hinge"],
        default="off",
        help="Soften only generated automatic search intervals of admitted one-ground, no-net "
        "movable following flights inside the serve prefix; human or producer supplied "
        "intervals, source events and every gate stay unchanged",
    )
    parser.add_argument(
        "--prefix-net-revisit",
        choices=["off", "on"],
        default="off",
        help="After the net stage, revisit the serve prefix under the fitted net response "
        "with declared-net chart coordinates, then one bounded net follow-on",
    )
    parser.add_argument("--terminal-impact-intervals", choices=["off", "on"], default="off")
    parser.add_argument(
        "--player-camera-coordinates",
        choices=["off", "on"],
        default="off",
        help="Derive soft player ground roots from native boxes and the supplied same-frame camera",
    )
    parser.add_argument(
        "--independent-toss-horizontal-prior",
        choices=["off", "on"],
        default="off",
        help="Apply the existing soft horizontal toss-speed preference to qualified incoming observations, independently of player stature or body anchoring",
    )
    parser.add_argument(
        "--toss-player-prior",
        choices=["off", "on"],
        default="off",
        help="Shared soft same-camera player-relative toss conditioning plus the soft "
        "horizontal incoming-velocity residual in both serve paths; exact OFF when unsupported",
    )
    parser.add_argument(
        "--terminal-ground-coupling",
        choices=("off", "on"),
        default="off",
        help="Optional connected final-ground pair with native-qualified horizontal response",
    )
    parser.add_argument(
        "--terminal-net-coupling",
        choices=["off", "on"],
        default="off",
        help="Joint preceding-ground normal and terminal-net contact refinement after the interior sweep; common 240-second phase",
    )
    parser.add_argument(
        "--terminal-ground-normal",
        choices=["off", "on"],
        default="off",
        help="Terminal stage frees the final flight's first-ground normal restitution through "
        "the persistent ground registry and supports one or two original terminal grounds",
    )
    parser.add_argument(
        "--interior-ground-normal",
        choices=["off", "on", "sweep", "sweep_revisit"],
        default="off",
        help="off, first block, chronological sweep, or sweep with one failed-neighbor revisit (same1800s budget)",
    )
    parser.add_argument(
        "--interior-restoration-geometry-only",
        choices=["off", "on"],
        default="off",
        help="Skip image work only for infeasible interior restoration trials; full objective still required for every feasible trial",
    )
    parser.add_argument(
        "--interior-block-seconds",
        type=float,
        default=180.0,
        help="Shared wall budget per interior block (same maxiter, sweep and outer budgets); "
        "the sweep still allocates min(block, remaining / remaining blocks)",
    )
    parser.add_argument("--net-physical-eligibility", choices=["off", "on"], default="off")
    parser.add_argument("--net-ground-normal", choices=["off", "on"], default="off")
    parser.add_argument(
        "--net-ground-horizontal", choices=["off", "retention", "heading"], default="off"
    )
    parser.add_argument("--net-first-ground-candidates", choices=["off", "on"], default="off")
    parser.add_argument("--net-first-contact-toss", choices=["off", "on"], default="off")
    parser.add_argument(
        "--net-mesh-height",
        choices=["off", "on"],
        default="off",
        help="Fit observed net contact across the physical mesh with continuous incoming roots",
    )
    parser.add_argument(
        "--first-contact-role",
        choices=["unspecified", "serve", "rally"],
        default="unspecified",
        help="Common supplied cohort role evidence, never a serve-ordinal claim",
    )
    parser.add_argument(
        "--athlete-root-reach-loss", choices=("quadratic", "cauchy"), default="quadratic"
    )
    parser.add_argument(
        "--athlete-evidence",
        choices=["required", "optional"],
        default="required",
        help="required: rows must supply player_order/statures; optional: absent roster "
        "evidence makes stature-scaled soft terms abstain explicitly (no invented heights)",
    )
    parser.add_argument("--observation-net-seed", choices=["off", "on", "candidate"], default="off")
    parser.add_argument("--net-seed-speed-scale-mps", type=float, default=15.0)
    parser.add_argument("--terminal-net-tail", choices=["off", "on"], default="off")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != stage.SCHEMA:
        raise ValueError(f"expected {stage.SCHEMA}")
    if min(args.workers, args.coarse_iterations, args.refine_iterations, args.timeout_seconds) <= 0:
        raise ValueError("positive workers and shared solve budgets required")
    if args.search_seconds is not None and not 0 < args.search_seconds <= args.timeout_seconds - 30:
        raise ValueError("search deadline must leave at least30s for replay/scoring")
    if not math.isfinite(args.interior_block_seconds) or args.interior_block_seconds <= 0:
        raise ValueError("positive finite shared interior block budget required")
    rows = manifest["rows"]
    keys = [row["key"] for row in rows]
    if any(not key or Path(key).name != key or key in (".", "..") for key in keys):
        raise ValueError("safe attempt keys required")
    if len(keys) != len(set(keys)) or set(args.case) - set(keys):
        raise ValueError("duplicate/unknown attempt key")
    if args.case:
        rows = [row for row in rows if row["key"] in args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    policy = {
        **stage.NUMERICAL_POLICY,
        **(
            {
                "observation_net_seed": args.observation_net_seed,
                "net_seed_speed_scale_mps": args.net_seed_speed_scale_mps,
            }
            if args.observation_net_seed != "off"
            else {}
        ),
        "coarse_iterations": args.coarse_iterations,
        "refine_iterations": args.refine_iterations,
        "preparation": args.preparation,
        "refinement": args.refinement,
        "contact_prefix_scope": args.contact_prefix_scope,
        **(
            {"contact_components": args.contact_components}
            if args.contact_components != "off"
            else {}
        ),
        **(
            {"whole_point_seed_fallback": args.whole_point_seed_fallback}
            if args.whole_point_seed_fallback != "off"
            else {}
        ),
        **(
            {"failed_component_split_fallback": args.failed_component_split_fallback}
            if args.failed_component_split_fallback != "off"
            else {}
        ),
        **(
            {"component_search_budget": args.component_search_budget}
            if args.component_search_budget != "off"
            else {}
        ),
        "observation_scope": args.observation_scope,
        "following_bounce_intervals": args.following_bounce_intervals,
        "joint_toss_requalification": args.joint_toss_requalification,
        "earlier_toss_support": args.earlier_toss_support,
        "net_physical_eligibility": args.net_physical_eligibility,
        "net_ground_normal": args.net_ground_normal,
        "net_ground_horizontal": args.net_ground_horizontal,
        "net_mesh_height": args.net_mesh_height,
        "net_first_ground_candidates": args.net_first_ground_candidates,
        "net_first_contact_toss": args.net_first_contact_toss,
        "prefix_local_boundary": args.prefix_local_boundary,
        "prefix_source_incumbent": args.prefix_source_incumbent,
        "prefix_pixel_loss": args.prefix_pixel_loss,
        "prefix_following_ground_timing": args.prefix_following_ground_timing,
        "prefix_net_revisit": args.prefix_net_revisit,
        "terminal_impact_intervals": args.terminal_impact_intervals,
        "terminal_ground_normal": args.terminal_ground_normal,
        "terminal_net_coupling": args.terminal_net_coupling,
        "terminal_ground_coupling": args.terminal_ground_coupling,
        "terminal_net_tail": args.terminal_net_tail,
        "interior_ground_normal": args.interior_ground_normal,
        "terminal_net_seed": args.terminal_net_seed,
        "terminal_ground_sparse_wings": args.terminal_ground_sparse_wings,
        "interior_sparse_wings": args.interior_sparse_wings,
        "interior_short_blocks": args.interior_short_blocks,
        "interior_ground_horizontal": args.interior_ground_horizontal,
        "interior_restoration_geometry_only": args.interior_restoration_geometry_only,
        "interior_block_seconds": args.interior_block_seconds,
        "first_contact_role": args.first_contact_role,
        "observation_partition": args.observation_partition,
        "search_incumbent": args.search_incumbent,
        **(
            {"fit_ground_witness": args.fit_ground_witness}
            if args.fit_ground_witness != "off"
            else {}
        ),
        "toss_player_prior": args.toss_player_prior,
        "independent_toss_horizontal_prior": args.independent_toss_horizontal_prior,
        "player_camera_coordinates": args.player_camera_coordinates,
        "athlete_evidence": args.athlete_evidence,
        **(
            {"athlete_root_reach_loss": args.athlete_root_reach_loss}
            if args.athlete_root_reach_loss != "quadratic"
            else {}
        ),
        **({"search_seconds": args.search_seconds} if args.search_seconds is not None else {}),
    }
    stage.save(
        args.output / "manifest.json",
        {
            **manifest,
            "rows": rows,
            "policy": policy,
            "subset_run": bool(args.case),
            "manifest_source": provenance.file_record(args.manifest),
            "code": provenance.git_record(stage.paths.REPO_ROOT),
            "implementation_files": [
                provenance.file_record(Path(p)) for p in (__file__, stage.__file__)
            ],
            "workers": args.workers,
            "timeout_seconds": args.timeout_seconds,
        },
    )
    results = []
    stage.save(args.output / "summary.json", summarize(rows, results, cohort=manifest))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(execute, row, args.output, policy, args.timeout_seconds): row
            for row in rows
        }
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            stage.save(args.output / "summary.json", summarize(rows, results, cohort=manifest))
            print(
                result["key"],
                result["status"],
                result["verdict"].get("accepted_flight_count", 0),
                flush=True,
            )


if __name__ == "__main__":
    main()
