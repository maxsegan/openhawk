"""Run label-blind whole-point 3D reconstruction from pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv.pipeline import anchor_first_fit  # noqa: E402
from cv.pipeline.reconstruction import reconstruct  # noqa: E402


def json_default(value: object) -> int:
    """Normalize NumPy integer diagnostics without hiding other JSON defects."""
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def joint_refine_enabled(
    *,
    requested: bool,
    whole_point_branches: bool,
    event_topology_branches: bool = False,
) -> bool:
    return requested or whole_point_branches or event_topology_branches


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit-root", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--event-boundaries", type=Path, required=True)
    result.add_argument(
        "--event-hypotheses",
        type=Path,
        help="Leaky probability-bearing S5 lattice used only by joint S6 refinement.",
    )
    result.add_argument(
        "--include-dead-time-emissions",
        action="store_true",
        help="Bypass the default point-grammar in-play filter without deleting metadata.",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--uncertainty-hypotheses", type=Path)
    result.add_argument(
        "--pose-artifact-name",
        help="Explicit automatic 2D pose artifact basename within each match directory.",
    )
    result.add_argument(
        "--physical-motion-name",
        help="Explicit automatic physical-motion artifact basename within each match directory.",
    )
    result.add_argument(
        "--camera-artifact-name",
        default="camera_P_per_frame_v1.npz",
        help="Automatic per-frame camera artifact basename within each match directory.",
    )
    result.add_argument(
        "--anchors-output-root",
        type=Path,
        help="Write one automatic anchors_v1.json below MATCH/POINT for each evaluated point.",
    )
    result.add_argument(
        "--camera-scope",
        type=Path,
        help="Optional automatic split timing/spatial camera-scope artifact.",
    )
    result.add_argument("--point", action="append", default=[])
    result.add_argument("--max-nfev", type=int, default=60)
    result.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    result.add_argument(
        "--math-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads per point worker. Keep at one for point-level parallelism.",
    )
    result.add_argument(
        "--point-timeout-seconds",
        type=float,
        default=600.0,
        help="Fail closed when one point exceeds this wall-clock budget.",
    )
    result.add_argument("--multi-start", action="store_true")
    result.add_argument("--joint-refine", action="store_true")
    result.add_argument("--terminal-flights", action="store_true")
    result.add_argument("--discrete-bounce-branches", action="store_true")
    result.add_argument("--anchor-bounce-geometry", action="store_true")
    fit_mode = result.add_mutually_exclusive_group()
    fit_mode.add_argument(
        "--anchor-first",
        dest="anchor_first",
        action="store_true",
        default=True,
        help="Use the default direct-reprojection fitter with odd-frame validation.",
    )
    fit_mode.add_argument(
        "--contact-first",
        dest="anchor_first",
        action="store_false",
        help="Use the legacy contact-first fitter for controlled comparisons.",
    )
    net_mode = result.add_mutually_exclusive_group()
    net_mode.add_argument(
        "--net-plane-constraint",
        dest="net_point_anchor",
        action="store_false",
        help="Deprecated compatibility flag; the net is always post-fit plausibility only.",
    )
    net_mode.add_argument(
        "--net-anchor-point",
        dest="net_point_anchor",
        action="store_true",
        help="Deprecated compatibility flag; the net is never a fitted anchor.",
    )
    result.set_defaults(net_point_anchor=True)
    result.add_argument(
        "--legacy-height-on-ray",
        action="store_true",
        help="Deprecated compatibility flag; ignored because that objective fitted the net.",
    )
    result.add_argument(
        "--striker-witness-prior",
        action="store_true",
        help="Put the tracked striker's court position into the flight objective as a reach and "
        "front-of-body band instead of the single 2.1 m hinge.",
    )
    striker_authority_mode = result.add_mutually_exclusive_group()
    striker_authority_mode.add_argument(
        "--striker-witness-authority",
        dest="striker_witness_authority",
        action="store_true",
        help="Let the striker witness decide which side of a bad contact seam is believed when "
        "the automatic contact pixel disagrees with the better-fitted endpoint.",
    )
    striker_authority_mode.add_argument(
        "--no-striker-witness-authority",
        dest="striker_witness_authority",
        action="store_false",
        help="Decide a bad contact seam on the automatic contact pixel alone.",
    )
    result.set_defaults(striker_witness_authority=True)
    contact_observation_mode = result.add_mutually_exclusive_group()
    contact_observation_mode.add_argument(
        "--contact-observation-witness",
        dest="contact_observation_witness",
        action="store_true",
        help="Fit a flight's contact row against the contact's own image witness at that "
        "witness's own error bar, instead of the tracked ball interpolated across the impact.",
    )
    contact_observation_mode.add_argument(
        "--no-contact-observation-witness",
        dest="contact_observation_witness",
        action="store_false",
        help="Interpolate the tracked ball at the contact frame and fit it at full weight.",
    )
    result.set_defaults(contact_observation_witness=False)
    contact_sigma_mode = result.add_mutually_exclusive_group()
    contact_sigma_mode.add_argument(
        "--contact-observation-sigma",
        dest="contact_observation_sigma",
        action="store_true",
        help="Keep the interpolated contact pixel but weight it by what a chord across a racket "
        "impact is worth, instead of at the same weight as a tracked observation.",
    )
    contact_sigma_mode.add_argument(
        "--no-contact-observation-sigma",
        dest="contact_observation_sigma",
        action="store_false",
        help="Weight the interpolated contact pixel as a tracked observation.",
    )
    result.set_defaults(contact_observation_sigma=False)
    spin_prior_mode = result.add_mutually_exclusive_group()
    spin_prior_mode.add_argument(
        "--physical-spin-prior",
        dest="physical_spin_prior",
        action="store_true",
        help="Charge a fitted spin against the measured incoming topspin of the surface in "
        "velocity-aligned components instead of against a ball with no spin at all.",
    )
    spin_prior_mode.add_argument(
        "--no-physical-spin-prior",
        dest="physical_spin_prior",
        action="store_false",
        help="Charge a fitted spin against zero spin.",
    )
    result.set_defaults(physical_spin_prior=False)
    subframe_anchor_mode = result.add_mutually_exclusive_group()
    subframe_anchor_mode.add_argument(
        "--subframe-anchors",
        dest="subframe_anchors",
        action="store_true",
        help="Read a bounce emission as a weighted observation -- the impact is on the court "
        "plane, its horizontal position and sub-frame time are fitted, and the emitted pixel is "
        "a ray observation of the ball on the emitted frame -- instead of pinning the flight "
        "through the ray/plane intersection of an integer-frame pixel.",
    )
    subframe_anchor_mode.add_argument(
        "--no-subframe-anchors",
        dest="subframe_anchors",
        action="store_false",
        help="Pin the flight through the bounce emission's ray/plane intersection exactly.",
    )
    result.set_defaults(subframe_anchors=False)
    subframe_contact_mode = result.add_mutually_exclusive_group()
    subframe_contact_mode.add_argument(
        "--subframe-contacts",
        dest="subframe_contacts",
        action="store_true",
        help="Judge a seam at the sub-frame time the two arcs meet, and carry the shared "
        "contact off the emitted pixel's ray to that instant, instead of asking two flights to "
        "agree at the integer frame the contact was emitted on.",
    )
    subframe_contact_mode.add_argument(
        "--no-subframe-contacts",
        dest="subframe_contacts",
        action="store_false",
        help="Judge a seam at the emitted contact frame.",
    )
    result.set_defaults(subframe_contacts=False)
    # The two separable halves of the arm above, plus whether an adopted seam stands the soft
    # reconciliation down.  All three are on inside --subframe-contacts, so that flag alone is
    # exactly what docs/wk1/point_fit7.md measured; turning one off attributes its cost.
    time_prior_mode = result.add_mutually_exclusive_group()
    time_prior_mode.add_argument(
        "--subframe-time-priors",
        dest="subframe_time_priors",
        action="store_true",
        help="Centre every free impact-time variable -- bounce, contact and terminal -- on "
        "cv.pipeline.subframe_timing's combined witness at that witness's own width, instead of "
        "on the emitted frame with the quantisation as the width.  Falls back to the emitted "
        "frame wherever the witnesses abstain.",
    )
    time_prior_mode.add_argument(
        "--no-subframe-time-priors",
        dest="subframe_time_priors",
        action="store_false",
        help="Centre every free impact time on the emitted frame.",
    )
    result.set_defaults(subframe_time_priors=False)
    seam_mode = result.add_mutually_exclusive_group()
    seam_mode.add_argument(
        "--subframe-contact-seam",
        dest="subframe_contact_seam",
        action="store_true",
        help="Record the sub-frame instant two adjacent arcs meet at as the contact's time.",
    )
    seam_mode.add_argument(
        "--no-subframe-contact-seam",
        dest="subframe_contact_seam",
        action="store_false",
        help="Leave the seam on the emitted frame; --subframe-contacts then only advances the "
        "soft pass's shared contact off the emitted pixel's ray.",
    )
    result.set_defaults(subframe_contact_seam=True)
    advance_mode = result.add_mutually_exclusive_group()
    advance_mode.add_argument(
        "--subframe-contact-advance",
        dest="subframe_contact_advance",
        action="store_true",
        help="Carry the soft pass's shared contact off the emitted pixel's ray to the fitted "
        "instant.",
    )
    advance_mode.add_argument(
        "--no-subframe-contact-advance",
        dest="subframe_contact_advance",
        action="store_false",
        help="Pin the soft pass's shared contact to the emitted pixel's ray.",
    )
    result.set_defaults(subframe_contact_advance=True)
    seam_refit_mode = result.add_mutually_exclusive_group()
    seam_refit_mode.add_argument(
        "--subframe-contact-seam-skips-refit",
        dest="subframe_contact_seam_skips_refit",
        action="store_true",
        help="An adopted seam time also stands the soft one-sided reconciliation down at that "
        "contact.",
    )
    seam_refit_mode.add_argument(
        "--no-subframe-contact-seam-skips-refit",
        dest="subframe_contact_seam_skips_refit",
        action="store_false",
        help="Record the seam time and still run the soft one-sided reconciliation there.",
    )
    result.set_defaults(subframe_contact_seam_skips_refit=True)
    seam_witness_mode = result.add_mutually_exclusive_group()
    seam_witness_mode.add_argument(
        "--subframe-contact-seam-witnessed",
        dest="subframe_contact_seam_witnessed",
        action="store_true",
        help="Adopt a seam time only where the contact's own sub-frame timing witness agrees "
        "with it (needs --subframe-time-priors), so two arcs that meet along the camera ray "
        "are not read as evidence about when the impact was.",
    )
    seam_witness_mode.add_argument(
        "--no-subframe-contact-seam-witnessed",
        dest="subframe_contact_seam_witnessed",
        action="store_false",
        help="Adopt every seam the two arcs close inside the emitted frame.",
    )
    result.set_defaults(subframe_contact_seam_witnessed=False)
    plane_anchor_error_mode = result.add_mutually_exclusive_group()
    plane_anchor_error_mode.add_argument(
        "--subframe-plane-anchor-error",
        dest="subframe_plane_anchor_error",
        action="store_true",
        help="Report a sub-frame impact anchor's error as the metric assertion it makes -- the "
        "impact is on the court plane -- instead of how far the fitted ball at the emitted frame "
        "is from the emission's ray, so the acceptance gate reads the same kind of number as it "
        "does for the shipped knot.",
    )
    plane_anchor_error_mode.add_argument(
        "--no-subframe-plane-anchor-error",
        dest="subframe_plane_anchor_error",
        action="store_false",
        help="Report the emission ray miss as the anchor's error.",
    )
    result.set_defaults(subframe_plane_anchor_error=False)
    bounce_witness_mode = result.add_mutually_exclusive_group()
    bounce_witness_mode.add_argument(
        "--subframe-bounce-witness",
        dest="subframe_bounce_witness",
        action="store_true",
        help="Observe the fitted bounce against the track's own sub-frame court-plane corner in "
        "the owner's shape: completely free inside max(0.20 m, 2 sigma) of it, a small penalty "
        "growing with distance outside, never a hard pin.  Needs --subframe-anchors.",
    )
    bounce_witness_mode.add_argument(
        "--no-subframe-bounce-witness",
        dest="subframe_bounce_witness",
        action="store_false",
        help="Leave the fitted bounce position to the emitted pixel and the court plane alone.",
    )
    result.set_defaults(subframe_bounce_witness=False)
    whole_point_joint_mode = result.add_mutually_exclusive_group()
    whole_point_joint_mode.add_argument(
        "--whole-point-joint",
        dest="whole_point_joint",
        action="store_true",
        help="Solve every flight of a point and every shared contact state in one least-squares "
        "call, seeded from the per-flight fits, for points with at most five flights.  The "
        "alternating one-sided-then-soft reconciliation stays as the fallback.",
    )
    whole_point_joint_mode.add_argument(
        "--no-whole-point-joint",
        dest="whole_point_joint",
        action="store_false",
        help="Reconcile adjacent flights with the alternating scheme only.",
    )
    result.set_defaults(whole_point_joint=False)
    contact_observation_mode = result.add_mutually_exclusive_group()
    contact_observation_mode.add_argument(
        "--downweight-contact-adjacent",
        dest="downweight_contact_adjacent",
        action="store_true",
        help="Give the two fitted observations nearest each contact 0.1x weight; held-out "
        "checkerboard frames remain unweighted.",
    )
    contact_observation_mode.add_argument(
        "--full-contact-adjacent-weight",
        dest="downweight_contact_adjacent",
        action="store_false",
        help="Keep contact-adjacent fitted observations at full weight.",
    )
    result.set_defaults(downweight_contact_adjacent=False)
    contact_mode = result.add_mutually_exclusive_group()
    contact_mode.add_argument(
        "--shared-contact-fit",
        dest="shared_contact_fit",
        action="store_true",
        help="Use the default guarded soft shared-contact refinement.",
    )
    contact_mode.add_argument(
        "--independent-contacts",
        dest="shared_contact_fit",
        action="store_false",
        help="Keep independently fitted anchor-first contact endpoints for comparison.",
    )
    result.set_defaults(shared_contact_fit=True)
    result.add_argument("--whole-point-branches", action="store_true")
    result.add_argument(
        "--match-shared-priors",
        action="store_true",
        help="Estimate label-free broadcast priors across all first-pass point fits and withhold outliers.",
    )
    result.add_argument(
        "--exact-contact-reprojection",
        action="store_true",
        help="Try exact event-ray contact anchors and retain only safe boundary refits.",
    )
    result.add_argument(
        "--event-topology-branches",
        action="store_true",
        help="Try guarded contact insertion, omission, and half-frame retiming branches.",
    )
    result.add_argument(
        "--joint-rally-refinement",
        action="store_true",
        help=(
            "Enable the bounded S6 repair lattice: robust contact retiming, single and "
            "sequence insertion/omission branches, and post-joint physics selection."
        ),
    )
    result.add_argument("--topology-branch-width", type=int, default=8)
    result.add_argument("--topology-branch-margin", type=float, default=8.0)
    result.add_argument(
        "--topology-post-joint-scoring",
        action="store_true",
        help="Rescore a bounded topology shortlist after whole-point joint refinement.",
    )
    result.add_argument("--topology-joint-width", type=int, default=3)
    result.add_argument("--branch-width", type=int, default=6)
    result.add_argument("--branch-margin", type=float, default=8.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.math_threads < 1:
        raise ValueError("--math-threads must be at least 1")
    if args.point_timeout_seconds <= 0:
        raise ValueError("--point-timeout-seconds must be positive")
    if args.topology_joint_width < 1:
        raise ValueError("--topology-joint-width must be at least 1")
    anchor_first_fit.configure_contact_adjacent_weighting(args.downweight_contact_adjacent)
    anchor_first_fit.configure_striker_witness(
        prior=args.striker_witness_prior,
        authority=args.striker_witness_authority,
    )
    anchor_first_fit.configure_contact_observation_witness(args.contact_observation_witness)
    anchor_first_fit.configure_contact_observation_sigma(args.contact_observation_sigma)
    anchor_first_fit.configure_physical_spin_prior(args.physical_spin_prior)
    anchor_first_fit.configure_subframe_anchors(args.subframe_anchors)
    anchor_first_fit.configure_subframe_contacts(args.subframe_contacts)
    anchor_first_fit.configure_subframe_contact_parts(
        seam=args.subframe_contact_seam,
        advance=args.subframe_contact_advance,
        seam_skips_refit=args.subframe_contact_seam_skips_refit,
        seam_witnessed=args.subframe_contact_seam_witnessed,
    )
    anchor_first_fit.configure_subframe_plane_anchor_error(args.subframe_plane_anchor_error)
    anchor_first_fit.configure_subframe_bounce_witness(args.subframe_bounce_witness)
    anchor_first_fit.configure_whole_point_joint(args.whole_point_joint)

    anchor_first_fit.configure_legacy_height_on_ray(args.legacy_height_on_ray)
    topology_enabled = args.event_topology_branches or args.joint_rally_refinement
    with threadpool_limits(limits=args.math_threads):
        report = reconstruct(
            args.audit_root,
            args.manifest,
            args.event_boundaries,
            max_nfev=args.max_nfev,
            workers=args.workers,
            point_keys=set(args.point) or None,
            multi_start=args.multi_start,
            joint_refine=joint_refine_enabled(
                requested=args.joint_refine,
                whole_point_branches=args.whole_point_branches,
                event_topology_branches=topology_enabled,
            ),
            terminal_flights=args.terminal_flights,
            uncertainty_hypotheses=args.uncertainty_hypotheses,
            discrete_bounce_branches=(args.discrete_bounce_branches or args.joint_rally_refinement),
            anchor_bounce_geometry=args.anchor_bounce_geometry,
            anchor_first=args.anchor_first,
            net_point_anchor=args.net_point_anchor,
            downweight_contact_adjacent=args.downweight_contact_adjacent,
            striker_witness_prior=args.striker_witness_prior,
            striker_witness_authority=args.striker_witness_authority,
            contact_observation_witness=args.contact_observation_witness,
            contact_observation_sigma=args.contact_observation_sigma,
            physical_spin_prior=args.physical_spin_prior,
            subframe_anchors=args.subframe_anchors,
            subframe_contacts=args.subframe_contacts,
            subframe_time_priors=args.subframe_time_priors,
            subframe_contact_seam=args.subframe_contact_seam,
            subframe_contact_advance=args.subframe_contact_advance,
            subframe_contact_seam_skips_refit=args.subframe_contact_seam_skips_refit,
            subframe_contact_seam_witnessed=args.subframe_contact_seam_witnessed,
            subframe_plane_anchor_error=args.subframe_plane_anchor_error,
            subframe_bounce_witness=args.subframe_bounce_witness,
            whole_point_joint=args.whole_point_joint,
            shared_contact_fit=args.shared_contact_fit,
            match_shared_priors=args.match_shared_priors,
            whole_point_branches=args.whole_point_branches,
            exact_contact_reprojection=args.exact_contact_reprojection,
            event_topology_branches=topology_enabled,
            topology_sequence_branches=args.joint_rally_refinement,
            topology_branch_width=args.topology_branch_width,
            topology_branch_margin=args.topology_branch_margin,
            topology_post_joint_scoring=(
                args.topology_post_joint_scoring or args.joint_rally_refinement
            ),
            topology_joint_width=args.topology_joint_width,
            branch_width=args.branch_width,
            branch_margin=args.branch_margin,
            camera_scope=args.camera_scope,
            include_dead_time_emissions=args.include_dead_time_emissions,
            event_hypotheses=args.event_hypotheses,
            pose_artifact_name=args.pose_artifact_name,
            physical_motion_name=args.physical_motion_name,
            camera_artifact_name=args.camera_artifact_name,
            anchors_output_root=args.anchors_output_root,
            point_timeout_seconds=args.point_timeout_seconds,
            math_threads=args.math_threads,
        )
    report["execution"] = {
        "point_workers": args.workers,
        "math_threads_per_worker": args.math_threads,
        "point_timeout_seconds": args.point_timeout_seconds,
        "downweight_contact_adjacent": args.downweight_contact_adjacent,
        "contact_adjacent_observation_weight": (
            anchor_first_fit.CONTACT_ADJACENT_WEIGHT if args.downweight_contact_adjacent else 1.0
        ),
        "legacy_height_on_ray_requested_ignored": args.legacy_height_on_ray,
        "net_treatment": "post_fit_plausibility_only",
        "match_shared_priors": args.match_shared_priors,
        "contact_observation_witness": args.contact_observation_witness,
        "contact_observation_sigma": args.contact_observation_sigma,
        "physical_spin_prior": args.physical_spin_prior,
        "subframe_anchors": args.subframe_anchors,
        "subframe_contacts": args.subframe_contacts,
        "subframe_time_priors": args.subframe_time_priors,
        "subframe_contact_seam": args.subframe_contact_seam,
        "subframe_contact_advance": args.subframe_contact_advance,
        "subframe_contact_seam_skips_refit": args.subframe_contact_seam_skips_refit,
        "subframe_contact_seam_witnessed": args.subframe_contact_seam_witnessed,
        "subframe_plane_anchor_error": args.subframe_plane_anchor_error,
        "subframe_bounce_witness": args.subframe_bounce_witness,
        "whole_point_joint": args.whole_point_joint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "points_detail"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
