"""In-memory, evidence-routed refinement of this invocation's S6 search result.

Call ``refine(bundle)`` immediately after the same cold invocation constructs its
input-ranked search bundle. This module has no file, cached-fit or case-key API.
It reuses bounded local numerical helpers: joint serve/toss first, then a flexible
terminal net. Unsupported inputs retain the last coherent state; execution
failures are reported separately. Acceptance is measured only after selection,
never used to route, pick a candidate, or roll back a valid numerical result.

This is supplied-input S6, not automatic upstream labeling. The caller owns the
cold-invocation provenance and serialization of the returned evaluation context.
Numerical response context managers are worker-local; use separate processes,
not concurrent refinement threads in the same Python process.
"""

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from cv.experiments.connected_shooting import (
    labeled_context_census as census,
    labeled_context_witness_scope as scope,
    labeled_isolated_serve as isolated_cli,
    labeled_interior_ground_response as interior_ground,
    labeled_interior_normal as interior,
    labeled_interior_sweep as interior_sweep,
    labeled_net_recipe as net,
    labeled_net_normal_response as normal_ground,
    labeled_prefix_joint_impact as prefix,
    labeled_prefix_later_net as later,
    labeled_prefix_net_chart as net_chart,
    labeled_serve_recipe as serve,
    labeled_terminal_ground_response as terminal_response,
    labeled_terminal_ground_coupling as terminal_ground_coupling,
    labeled_terminal_net_coupling as terminal_coupling,
    labeled_terminal_impact as terminal_impact,
    labeled_toss_admission as toss_admission,
    labeled_toss_player_anchor as player_anchor,
)
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
from cv.experiments.connected_shooting.labeled_passive_tape import using_response


@dataclass(frozen=True)
class Policy:
    """One fixed configuration for an invocation/cohort, never a per-key policy."""

    serve_max_nfev: int = 80
    net_max_nfev: int = 120
    seconds_per_stage: float = 300.0
    toss_weight: float = 4.0
    following_bounce_intervals: bool = False
    joint_toss_requalification: bool = False
    earlier_toss_support: bool = False
    net_ground_normal: bool = False
    net_mesh_height: bool = False
    net_first_ground_candidates: bool = False
    net_first_contact_toss: bool = False
    net_ground_horizontal: str = "off"
    prefix_local_boundary: bool = False
    prefix_source_incumbent: bool = False
    prefix_pixel_loss: str = "off"
    prefix_following_ground_timing: str = "off"
    prefix_net_revisit: bool = False
    prefix_net_revisit_seconds: float = 180.0
    terminal_impact_intervals: bool = False
    terminal_ground_normal: bool = False
    terminal_net_coupling: bool = False
    terminal_ground_coupling: bool = False
    terminal_ground_coupling_seconds: float = 120.0
    terminal_ground_coupling_maxiter: int = 60
    terminal_coupling_seconds: float = 240.0
    terminal_coupling_maxiter: int = 120
    interior_ground_normal: bool = False
    terminal_ground_sparse_wings: bool = False
    interior_sparse_wings: bool = False
    interior_short_blocks: bool = False
    interior_ground_horizontal: bool = False
    interior_restoration_geometry_only: bool = False
    interior_maxiter: int = 60
    interior_seconds: float = 180.0
    interior_schedule: str = "first"
    interior_sweep_seconds: float = 1800.0
    serve_ground_normal: bool = False
    toss_player_prior: bool = False
    independent_toss_horizontal_prior: bool = False
    toss_horizontal_sigma_mps: float = 2.0

    def validate(self) -> None:
        if self.prefix_pixel_loss not in ("off", "soft_l1"):
            raise ValueError("prefix_pixel_loss must be off or soft_l1")
        if self.prefix_following_ground_timing not in ("off", "prediction_hinge"):
            raise ValueError("prefix_following_ground_timing must be off or prediction_hinge")
        if type(self.serve_ground_normal) is not bool:
            raise ValueError("serve_ground_normal must be an explicit boolean")
        if type(self.independent_toss_horizontal_prior) is not bool:
            raise ValueError("independent_toss_horizontal_prior must be an explicit boolean")
        if type(self.toss_player_prior) is not bool:
            raise ValueError("toss_player_prior must be an explicit boolean")
        if (
            type(self.toss_horizontal_sigma_mps) not in (int, float)
            or not np.isfinite(self.toss_horizontal_sigma_mps)
            or self.toss_horizontal_sigma_mps <= 0
        ):
            raise ValueError("positive finite common soft horizontal toss velocity scale required")
        if type(self.interior_ground_normal) is not bool:
            raise ValueError("interior_ground_normal must be an explicit boolean")
        if type(self.interior_restoration_geometry_only) is not bool:
            raise ValueError("interior_restoration_geometry_only must be an explicit boolean")
        if self.interior_restoration_geometry_only and not self.interior_ground_normal:
            raise ValueError("geometry-only restoration requires interior normal refinement")
        if type(self.terminal_ground_sparse_wings) is not bool:
            raise ValueError("terminal_ground_sparse_wings must be an explicit boolean")
        if self.terminal_ground_sparse_wings and not self.terminal_ground_coupling:
            raise ValueError("terminal sparse wings require terminal ground coupling")
        if type(self.interior_sparse_wings) is not bool:
            raise ValueError("interior_sparse_wings must be an explicit boolean")
        if self.interior_sparse_wings and not self.interior_ground_normal:
            raise ValueError("sparse ground wings require interior normal refinement")
        if type(self.interior_ground_horizontal) is not bool:
            raise ValueError("interior_ground_horizontal must be an explicit boolean")
        if self.interior_ground_horizontal and not self.interior_ground_normal:
            raise ValueError("horizontal ground retention requires interior normal refinement")
        if type(self.interior_short_blocks) is not bool:
            raise ValueError("interior_short_blocks must be an explicit boolean")
        if self.interior_short_blocks and not self.interior_ground_normal:
            raise ValueError("short interior blocks require normal refinement enabled")
        if self.interior_schedule not in ("first", "sweep", "sweep_revisit"):
            raise ValueError("explicit first, sweep or sweep_revisit interior schedule required")
        if self.interior_schedule in ("sweep", "sweep_revisit") and not self.interior_ground_normal:
            raise ValueError("interior sweep requires normal refinement enabled")
        if not np.isfinite(self.interior_sweep_seconds) or self.interior_sweep_seconds <= 0:
            raise ValueError("positive common interior sweep budget required")
        if (
            type(self.interior_maxiter) is not int
            or self.interior_maxiter < 1
            or not np.isfinite(self.interior_seconds)
            or self.interior_seconds <= 0
        ):
            raise ValueError("positive shared interior budgets required")
        if type(self.earlier_toss_support) is not bool:
            raise ValueError("earlier_toss_support must be an explicit boolean")
        if type(self.prefix_source_incumbent) is not bool:
            raise ValueError("prefix_source_incumbent must be an explicit boolean")
        if type(self.prefix_net_revisit) is not bool:
            raise ValueError("prefix_net_revisit must be an explicit boolean")
        if (
            type(self.prefix_net_revisit_seconds) not in (int, float)
            or not np.isfinite(self.prefix_net_revisit_seconds)
            or self.prefix_net_revisit_seconds <= 0
        ):
            raise ValueError("positive finite common prefix/net revisit budget required")
        if self.net_ground_horizontal not in ("off", "retention", "heading"):
            raise ValueError("explicit supported net_ground_horizontal mode required")
        if self.net_ground_horizontal != "off" and not self.net_ground_normal:
            raise ValueError("net_ground_horizontal requires direct normal response enabled")
        if type(self.terminal_impact_intervals) is not bool:
            raise ValueError("terminal_impact_intervals must be an explicit boolean")
        if type(self.terminal_ground_normal) is not bool:
            raise ValueError("terminal_ground_normal must be an explicit boolean")
        if type(self.terminal_net_coupling) is not bool:
            raise ValueError("terminal_net_coupling must be an explicit boolean")
        if type(self.terminal_ground_coupling) is not bool:
            raise ValueError("terminal_ground_coupling must be an explicit boolean")
        if (
            type(self.terminal_ground_coupling_seconds) not in (int, float)
            or not np.isfinite(self.terminal_ground_coupling_seconds)
            or self.terminal_ground_coupling_seconds <= 0
            or type(self.terminal_ground_coupling_maxiter) is not int
            or self.terminal_ground_coupling_maxiter < 1
        ):
            raise ValueError("positive terminal ground coupling budgets required")
        if (
            type(self.terminal_coupling_seconds) not in (int, float)
            or not np.isfinite(self.terminal_coupling_seconds)
            or self.terminal_coupling_seconds <= 0
            or type(self.terminal_coupling_maxiter) is not int
            or self.terminal_coupling_maxiter < 1
        ):
            raise ValueError("positive terminal coupling budgets required")
        if self.terminal_ground_normal and not self.terminal_impact_intervals:
            raise ValueError("terminal_ground_normal requires terminal_impact_intervals enabled")
        if type(self.net_first_contact_toss) is not bool:
            raise ValueError("net_first_contact_toss must be an explicit boolean")
        if type(self.net_first_ground_candidates) is not bool:
            raise ValueError("net_first_ground_candidates must be an explicit boolean")
        if type(self.net_mesh_height) is not bool:
            raise ValueError("net_mesh_height must be an explicit boolean")
        if type(self.prefix_local_boundary) is not bool:
            raise ValueError("prefix_local_boundary must be an explicit boolean")
        if type(self.net_ground_normal) is not bool:
            raise ValueError("net_ground_normal must be an explicit boolean")
        if type(self.joint_toss_requalification) is not bool:
            raise ValueError("joint_toss_requalification must be an explicit boolean")
        if type(self.following_bounce_intervals) is not bool:
            raise ValueError("following_bounce_intervals must be an explicit boolean")
        for value in (self.serve_max_nfev, self.net_max_nfev):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer stage budgets required")
        if not np.isfinite([self.seconds_per_stage, self.toss_weight]).all() or (
            self.seconds_per_stage <= 0 or self.toss_weight <= 0
        ):
            raise ValueError("positive finite wall budget and toss weight required")


def _serve_role(bundle: dict) -> dict:
    """Validate original contact membership before recognizing supplied serve evidence."""
    from cv.experiments.connected_shooting import observation_scope

    labels, packet, search = bundle["labels"], bundle["packet"], bundle["search"]
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one explicitly separated original attempt required")
    attempt = packet["attempts"][0]
    from cv.pipeline import s6_first_contact_role

    if s6_first_contact_role.validate(attempt, labels, search) is not None:
        raise ValueError("a bound original non-serve origin cannot receive serve refinement")
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
            and float(e["frame"]) <= observation_scope.horizon(labels["attempt"])
        ],
        [e for e in attempt["events"] if e["event_type"] == "contact"],
        [e for e in search["events"] if e["event_type"] == "contact"],
    ]
    groups = [sorted(group, key=lambda e: e["frame"]) for group in groups]
    epochs = [[float(e["frame"]) for e in group] for group in groups]
    if not epochs[0] or epochs[0] != epochs[1]:
        raise ValueError("original label, packet and search contact topology must agree")
    topology_admission = None
    if epochs[0] != epochs[2]:
        from cv.pipeline import s6_refinement_input_context

        topology_admission = s6_refinement_input_context.require_selected_contacts(bundle)
    first = groups[0][0]
    if (
        not occurrence.resolved_membership(first)
        or float(labels["attempt"]["first_contact_frame"]) != epochs[0][0]
    ):
        raise ValueError("original first contact must have resolved membership")
    roles = [
        str(e[k]).lower()
        for group in groups
        for e in group[:1]
        for k in ("stroke", "shot_type", "role")
        if e.get(k) is not None
    ]
    if any(role != "serve" for role in roles):
        raise ValueError("an explicit non-serve cannot receive serve refinement")
    ending = str(bundle["ending"]).lower()
    typed = (
        bool(roles)
        or first.get("serve_number") in (1, 2)
        or ending.startswith("serve_")
        or ending in ("ace", "unreturned_serve")
    )
    supplied = bundle.get("serve_role_evidence", {})
    if not typed and not (
        supplied.get("role") == "serve" and supplied.get("origin") == "supplied_observation"
    ):
        raise ValueError("original typed serve or supplied observation role evidence required")
    return dict(
        role="serve",
        origin="original_semantics" if typed else "supplied_observation",
        original_contact_epoch=epochs[0][0],
        contact_count=len(epochs[0]),
        **(
            {
                "selected_contact_count": len(epochs[2]),
                "selected_event_admission": topology_admission,
            }
            if topology_admission
            else {}
        ),
    )


def _score(
    context: dict, parameters, bundle: dict, response: dict | None, ground_response=None
) -> tuple[dict, dict]:
    with (
        interior_ground.using_response(ground_response, context["scene"]),
        using_response(response_from_record(response)) if response is not None else nullcontext(),
    ):
        if context.get("terminal_context_support") is not None:
            from cv.pipeline import s6_terminal_context

            # Admission remains atomic after every proposed refinement, including
            # sparse tails whose unqueried dense physical interval can fail.
            s6_terminal_context.physical_replay(context, parameters, bundle["duration"])
        if getattr(context["scene"], "terminal_net_tail", None) is not None:
            from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                require_event_domain,
                require_mesh_response,
            )

            # Check original input-domain membership before any scoring, under the
            # same response law that will be exported. Failure preserves the prior
            # domain-valid incumbent through the ordinary stage fallback.
            flights = serve.full.model.chain(context["scene"], parameters)
            require_event_domain(context["scene"], flights)
            require_mesh_response(context["scene"], flights)
        elif (
            response is not None
            and bundle.get("initial_search_fit", {})
            .get("net_response_initialization", {})
            .get("net_before_ground_contract")
            is not None
        ):
            from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                require_mesh_response,
            )

            from cv.experiments.connected_shooting.observation_net_seed import (
                require_ground_ending_binding,
            )

            require_ground_ending_binding(
                context["scene"],
                bundle["initial_search_fit"]["net_response_initialization"],
                bounces=context.get("bounces"),
            )
            flights = serve.full.model.chain(context["scene"], parameters)
            require_mesh_response(context["scene"], flights, include_supplied_final=True)
        return census.score(
            context,
            parameters,
            bundle["threshold"],
            bundle["duration"],
            bundle["ending"],
            scorer=scope.LOCAL_SCORE,
        )


def _validate_state(reference: dict, context: dict, parameters) -> np.ndarray:
    """Reject representation changes; the fitter may change a contact within its interval."""
    if context.get("terminal_context_support") != reference.get("terminal_context_support"):
        raise ValueError("refinement changed original terminal ownership")
    from cv.experiments.connected_shooting.interior_contact_epochs import FIELD

    if reference.get(FIELD) != context.get(FIELD):
        raise ValueError("refinement changed the original interior contact timing receipt")
    n = len(reference["scene"].pixels)
    p = np.asarray(parameters, float)
    if p.shape != (5 + 6 * n,) or not np.isfinite(p).all():
        raise ValueError("refinement must export a finite full single-shooting vector")
    for name in ("scene", "heldout"):
        before, after = reference[name], context[name]
        after.validate()
        contacts = sorted(
            (event for event in reference["events"] if event["event_type"] == "contact"),
            key=lambda event: event["frame"],
        )
        lo, hi = contacts[0]["frame_interval"]
        if not float(lo) <= float(after.contact_frames[0]) <= float(hi):
            raise ValueError("refinement moved contact outside its original interval")
        for field in (
            "fps",
            "surface",
            "dynamics",
            "bounce_profile",
            "rebound_mode",
            "parameterization",
            "bounce_regime_override",
            "terminal_net_tail",
            "observed_horizon_tail",
        ):
            if getattr(before, field) != getattr(after, field):
                raise ValueError(f"refinement changed original {name} {field}")
        if not np.array_equal(before.contact_frames[1:], after.contact_frames[1:]):
            raise ValueError("refinement changed a later supplied physical epoch")
        for field in (
            "observation_frames",
            "cameras",
            "pixels",
            "camera_distortion",
            "net_hit_frames",
        ):
            left, right = getattr(before, field), getattr(after, field)
            if (left is None) != (right is None) or (
                left is not None
                and (
                    len(left) != len(right)
                    or any(not np.array_equal(a, b) for a, b in zip(left, right, strict=True))
                )
            ):
                raise ValueError(f"refinement changed original {name} {field}")
    return p.copy()


def _prepare_serve(
    bundle: dict,
    context: dict,
    *,
    joint_toss_requalification: bool = False,
    earlier_toss_support: bool = False,
    toss_player_prior: bool = False,
    independent_toss_horizontal_prior: bool = False,
    net_first_contact_toss: bool = False,
) -> tuple[dict, dict, str]:
    role = _serve_role(bundle)
    clip = bundle["packet"]["attempts"][0]["point_clip"]
    observations = isolated_cli.bind_toss(
        bundle["search"], bundle["labels"], bundle["cameras"], clip
    )
    first_contact = next(e for e in context["events"] if e["event_type"] == "contact")
    observations, admission = toss_admission.admit(
        observations,
        bundle["labels"],
        bundle["cameras"],
        clip=clip,
        original_contact_interval=tuple(first_contact["frame_interval"]),
        exposure_frames=bundle["duration"],
        support=bundle["packet"]["attempts"][0].get("free_toss_support"),
        enabled=earlier_toss_support,
    )
    observations = prefix.toss_front.enrich(observations, bundle["labels"], clip)
    observations = observations | {"admission_receipt": admission}
    n = len(context["scene"].pixels)
    if n == 1 and context["scene"].right_boundary_kind == "original_contact":
        raise ValueError("single-flight contact prefix is not an isolated ground-ending serve")
    if n == 1 and net_first_contact_toss:
        active = context | {"net_serve_evidence": role}
        eligibility = net.epoch.qualify_first_contact_toss(
            active,
            observations,
            bundle["duration"],
            joint_toss_requalification=joint_toss_requalification,
        )
        rows = (
            eligibility["rows"]
            if (toss_player_prior or independent_toss_horizontal_prior)
            else None
        )
        mechanism = "net_serve_toss"
    elif n == 1:
        active = context | {"isolated_serve_evidence": role}
        eligibility = isolated_cli.isolated.qualify(active, observations, bundle["duration"])
        rows = (
            eligibility["rows"]
            if (toss_player_prior or independent_toss_horizontal_prior)
            else None
        )
        mechanism = "isolated_serve"
    else:
        active = context | {"prefix_serve_evidence": role}
        has_nets = active["scene"].net_hit_frames is not None and any(
            len(g) for g in active["scene"].net_hit_frames
        )
        eligibility = prefix.qualify(active, preserve_later_nets=bool(has_nets))
        rows = prefix._prefix_rows(
            observations,
            eligibility["contact"][0],
            bundle["duration"],
            joint_toss_requalification=joint_toss_requalification,
        )
        mechanism = "serve_prefix"
    if toss_player_prior:
        # Input-only anchor from the same qualified rows both serve paths consume; an
        # abstention is recorded and leaves the objective exactly OFF.
        players = active.get("players") or []
        observations = observations | {
            "player_anchor": player_anchor.prepare(
                rows,
                bundle.get("cameras"),
                bundle.get("observed_pose_rows"),
                float(bundle.get("pose_image_scale", 1.0)),
                players[0] if players else None,
                clip=clip,
                pose_record=bundle.get("observed_pose_record"),
                pose_space=bundle.get("observed_pose_space"),
            )
        }
    if independent_toss_horizontal_prior:
        observations = observations | {
            "toss_qualification": dict(
                status="qualified",
                mechanism=mechanism,
                frames=[float(row["frame"]) for row in rows],
                source="existing source-only serve qualifier; unchanged native incoming rows",
            )
        }
    return active, observations, mechanism


def _horizontal_prior_receipt(observations: dict, policy: Policy, mechanism: str | None) -> dict:
    """A physical preference needs qualified toss evidence, not an inferred body height."""
    anchor = observations.get("player_anchor") if policy.toss_player_prior else None
    anchored = anchor is not None and anchor.get("status") == "supported"
    qualification = observations.get("toss_qualification") or {}
    qualified = (
        qualification.get("status") == "qualified"
        and qualification.get("mechanism") == mechanism
        and bool(qualification.get("frames"))
    )
    active = mechanism == "isolated_serve" or policy.toss_weight > 0
    independent = policy.independent_toss_horizontal_prior and qualified and active
    origin = (
        "both"
        if anchored and independent
        else "anchor"
        if anchored
        else "independent"
        if independent
        else "off"
    )
    return dict(
        status="active" if anchored or independent else "not_applicable",
        horizontal_prior_source=origin,
        sigma_mps=policy.toss_horizontal_sigma_mps if anchored or independent else None,
        independent_status="active" if independent else "not_applicable",
        reason=None
        if independent
        else "inactive toss objective"
        if qualified and not active
        else "no source-qualified toss for this route",
        qualification=qualification if policy.independent_toss_horizontal_prior else None,
        body_anchor_required=False,
        meaning="soft zero-centred court-horizontal velocity preference; not measured velocity or player stature",
        upstream_search_rank="unchanged",
    )


def _player_terms(observations: dict, policy: Policy, mechanism: str | None = None) -> dict:
    """Keep the legacy anchor pair; optionally admit independent source-qualified velocity."""
    anchor = observations.get("player_anchor") if policy.toss_player_prior else None
    supported = anchor is not None and anchor.get("status") == "supported"
    return dict(
        player_anchor=anchor if supported else None,
        toss_horizontal_sigma_mps=_horizontal_prior_receipt(observations, policy, mechanism)[
            "sigma_mps"
        ],
    )


def _fit_serve(
    context,
    parameters,
    observations,
    mechanism,
    bundle,
    policy,
    *,
    net_response=None,
    seconds=None,
):
    """Fit the serve mechanism; with ``net_response`` run the response-aware declared-net revisit."""
    if seconds is None:
        seconds = policy.seconds_per_stage
    player_terms = _player_terms(observations, policy, mechanism)
    if mechanism == "isolated_serve":
        if net_response is not None:
            raise ValueError("isolated serve has no following flight to revisit")
        active, result = isolated_cli.isolated.fit(
            context,
            parameters,
            observations,
            bundle["duration"],
            maxiter=policy.serve_max_nfev,
            free_rebound_scales=True,
            seconds=seconds,
            **player_terms,
        )
        if policy.prefix_pixel_loss != "off":
            result["prefix_pixel_loss"] = dict(
                mode=policy.prefix_pixel_loss,
                status="not_applicable",
                reason="isolated serve has no multi-flight prefix",
                upstream_search_rank="unchanged",
            )
        if policy.prefix_following_ground_timing != "off":
            result["prefix_following_ground_timing"] = dict(
                mode=policy.prefix_following_ground_timing,
                status="not_applicable",
                reason="isolated serve has no following flight whose ground timing could soften",
                upstream_search_rank="unchanged",
            )
        if policy.serve_ground_normal:
            result["serve_ground_normal_admission"] = dict(
                status="unsupported_topology",
                reason="direct first normal currently requires a joint serve/return prefix",
                legacy_behavior_retained=True,
            )
        if policy.independent_toss_horizontal_prior:
            result["independent_toss_horizontal_prior"] = _horizontal_prior_receipt(
                observations, policy, mechanism
            )
        return active, result["best"]["parameters"], result
    checks, _ = net.original.followup.fit_check_copy(context["scene"], context["heldout"])
    has_nets = context["scene"].net_hit_frames is not None and any(
        len(g) for g in context["scene"].net_hit_frames
    )
    direct_normal = (
        policy.serve_ground_normal
        and not has_nets
        and not any(e["event_type"] == "net_hit" for e in context["events"])
        and context["scene"].terminal_net_tail is None
    )
    common = dict(
        **({"direct_first_ground_normal": True} if direct_normal else {}),
        following_launches=1,
        toss_weight=policy.toss_weight,
        duration=bundle["duration"],
        max_nfev=policy.serve_max_nfev,
        seconds=seconds,
        incoming_initializer="bounded_front",
        following_bounce_intervals=policy.following_bounce_intervals,
        joint_toss_requalification=policy.joint_toss_requalification,
        local_boundary=policy.prefix_local_boundary,
        source_incumbent=policy.prefix_source_incumbent,
        **({"pixel_loss": policy.prefix_pixel_loss} if policy.prefix_pixel_loss != "off" else {}),
        **(
            {"following_ground_timing": policy.prefix_following_ground_timing}
            if policy.prefix_following_ground_timing != "off"
            else {}
        ),
        **player_terms,
    )
    if net_response is None:
        _, result = prefix.fit(
            context | {"heldout": checks},
            parameters,
            observations,
            preserve_later_nets=bool(has_nets),
            **common,
        )
    else:
        from cv.experiments.connected_shooting import observation_net_seed

        # Same objective, observations and priors as the original prefix stage; the
        # same-invocation net response is active through every root, residual,
        # cache, source-incumbent comparison and the exported chain.
        _, result = later.fit(
            context | {"heldout": checks},
            parameters,
            observations,
            net_response=net_response,
            declared_net_chart=True,
            **(
                {"terminal_mesh_response_domain": True}
                if bundle.get("initial_search_fit") is not None
                and (
                    context["scene"].terminal_net_tail is not None
                    or (
                        observation_net_seed.single_final_net(context["scene"])
                        and bundle["initial_search_fit"].get("net_response") is not None
                        and bundle["initial_search_fit"].get("net_response_initialization")
                        is not None
                    )
                )
                else {}
            ),
            **common,
        )
    if policy.serve_ground_normal and not direct_normal:
        result["serve_ground_normal_admission"] = dict(
            status="unsupported_topology",
            reason="direct first normal currently requires original ground-only topology",
            legacy_behavior_retained=True,
        )
    active = serve.full.profile.profile_context(context, result["contact_epoch"])
    if policy.independent_toss_horizontal_prior:
        result["independent_toss_horizontal_prior"] = _horizontal_prior_receipt(
            observations, policy, mechanism
        )
    return active, result["full_vector"], result


REVISIT_MOVABLE_FLIGHTS = 2  # serve flight plus one following launch, as in the prefix stage


def _qualify_revisit(
    context: dict,
    mechanism: str | None,
    response: dict | None,
    *,
    terminal_tail_retry: bool = False,
) -> dict:
    """Topology-only applicability of the response-aware declared-net prefix revisit."""
    if mechanism != "serve_prefix":
        raise ValueError("response-aware revisit requires an applied joint serve prefix")
    if response is None:
        raise ValueError("no same-invocation net response to carry into the prefix revisit")
    later.qualify_later_nets(context)
    inventory = net_chart.specs(context, REVISIT_MOVABLE_FLIGHTS)
    if terminal_tail_retry:
        if context["scene"].terminal_net_tail is None or len(context["scene"].pixels) != 2:
            raise ValueError("response-aware terminal-tail retry requires two original flights")
        return dict(
            movable_flights=REVISIT_MOVABLE_FLIGHTS,
            inventory=inventory,
            active_flights=[1],
            response_origin="same_invocation_terminal_net",
            terminal_tail_retry=True,
            coordinate_mode="ordinary following launch with original mesh/event constraints",
            airborne_net_chart=False,
            latent_ground_epoch_supplied=False,
        )
    supported = [row for row in inventory if row["supported"]]
    if not supported:
        reasons = "; ".join(f"flight {row['flight']}: {row['reason']}" for row in inventory)
        raise ValueError(f"no movable following flight with one whole declared net ({reasons})")
    return dict(
        movable_flights=REVISIT_MOVABLE_FLIGHTS,
        inventory=inventory,
        active_flights=[row["flight"] for row in supported],
        response_origin="same_invocation_terminal_net",
    )


def _fit_net(context, parameters, bundle, policy, prepared_net, *, seconds, source_response=None):
    """The terminal-net stage from the current state under the common budgets."""
    checks, consumed = net.original.followup.fit_check_copy(context["scene"], context["heldout"])
    fit_context = context | {"heldout": checks}
    toss_kwargs = {}
    toss_receipt = {"status": "disabled"}
    if policy.net_first_contact_toss:
        try:
            fit_context, observations, _ = _prepare_serve(
                bundle,
                fit_context,
                joint_toss_requalification=policy.joint_toss_requalification,
                earlier_toss_support=policy.earlier_toss_support,
                toss_player_prior=policy.toss_player_prior,
                independent_toss_horizontal_prior=policy.independent_toss_horizontal_prior,
                net_first_contact_toss=True,
            )
            if len(fit_context["scene"].pixels) != 1:
                raise ValueError("net first-contact toss is a single-flight mechanism")
            toss_kwargs = dict(
                first_contact_toss=observations,
                joint_toss_requalification=policy.joint_toss_requalification,
                initial_net_response=source_response,
                toss_weight=policy.toss_weight,
                **_player_terms(observations, policy, "net_serve_toss"),
            )
            toss_receipt = {
                "status": "qualified",
                "admission": observations.get("admission_receipt"),
                **(
                    {
                        "independent_toss_horizontal_prior": _horizontal_prior_receipt(
                            observations, policy, "net_serve_toss"
                        )
                    }
                    if policy.independent_toss_horizontal_prior
                    else {}
                ),
            }
        except (ValueError, KeyError) as error:
            toss_receipt = {"status": "unsupported_input", "reason": str(error)}
    normal_prior = None
    normal_receipt = {"status": "disabled"}
    if policy.net_ground_normal:
        try:
            normal_prior = normal_ground.prior_from_observations(fit_context, parameters)
            normal_receipt = {"status": "qualified", "prior": normal_prior}
        except ValueError as error:
            normal_receipt = {"status": "unsupported_input", "reason": str(error)}
    if prepared_net["eligibility"].get("terminal_tail"):
        if policy.independent_toss_horizontal_prior and policy.net_first_contact_toss:
            toss_kwargs = {}
            toss_receipt = dict(
                status="not_applicable",
                reason="terminal tail response fit consumes no first-contact toss",
                independent_toss_horizontal_prior=dict(
                    status="not_applicable",
                    reason="terminal tail response fit has no incoming toss velocity",
                ),
            )
        from cv.experiments.connected_shooting import (
            labeled_terminal_tail_response as tail_response,
        )

        details = tail_response.fit(
            fit_context,
            parameters,
            bundle["duration"],
            maxiter=policy.net_max_nfev,
            seconds=seconds,
        )
    else:
        with net.continuous_ground(fit_context):
            ground_seed = {}
            if len(prepared_net["eligibility"]["ground_events"]) == 2:
                first_ground_frame = prepared_net["eligibility"]["ground_events"][0]["frame"]
                ground_seed = {
                    "first_ground_seed": next(
                        target
                        for target in fit_context["targets"][-1]
                        if target["event_frame"] == first_ground_frame
                    )
                }
            from cv.experiments.connected_shooting import labeled_net_seed_family

            details = labeled_net_seed_family.fit(
                fit_context,
                parameters,
                bundle["duration"],
                maxiter=policy.net_max_nfev,
                seconds=seconds,
                free_net_velocity=True,
                enabled=policy.net_first_ground_candidates,
                **({"mesh_height": True} if policy.net_mesh_height or toss_kwargs else {}),
                **toss_kwargs,
                quadratic_images=True,
                jacobian_step=1e-6,
                **ground_seed,
                **(
                    {
                        "first_ground_normal_prior": normal_prior,
                        **(
                            {"first_ground_horizontal_mode": policy.net_ground_horizontal}
                            if policy.net_ground_horizontal != "off"
                            else {}
                        ),
                    }
                    if normal_prior is not None
                    else {}
                ),
            )
    active = context
    if toss_kwargs:
        active = serve.full.profile.profile_context(context, details["best"]["contact_epoch"])
    candidate = _validate_state(context, active, details["best"]["parameters"])
    law_record = deepcopy(details["best"]["response"])
    with using_response(response_from_record(law_record)):
        serve.full.model.chain(active["scene"], candidate)
    candidate_verdict, candidate_measurement = _score(active, candidate, bundle, law_record)
    return dict(
        context=active,
        parameters=candidate,
        response=law_record,
        verdict=candidate_verdict,
        measurement=candidate_measurement,
        receipt=dict(
            initial_source=np.asarray(parameters, float).tolist(),
            fit=details,
            consumed_check_rows=consumed,
            first_ground_normal=normal_receipt,
            **({"first_contact_toss": toss_receipt} if policy.net_first_contact_toss else {}),
        ),
    )


def _refine(bundle: dict, policy: Policy = Policy()) -> dict:
    """Refine an in-memory bundle from this invocation; return one matching final state.

    Required fields match labeled_common_source.load's returned bundle, with
    ``context`` replaced by the caller's actual baseline evaluation context
    after its ordinary optional rebound-context preparation. This precondition
    makes an unsupported no-op identical to the baseline assessment.

    An audited common supplied-input role contract may be passed as
    ``serve_role_evidence={"role": "serve", "origin": "supplied_observation"}``;
    it cannot override an explicit non-serve or an unadmitted contact-topology mismatch.
    No file is opened, no candidate is loaded/selected here, and no acceptance field in
    the input bundle is consulted. Callers must not supply an externally fitted
    response: the baseline search uses its original physical net law.
    """
    policy.validate()
    if bundle.get("net_response") is not None or bundle.get("ground_response") is not None:
        raise ValueError("same-invocation baseline search must use its original net response")
    original = bundle["context"]
    parameters = _validate_state(original, original, bundle["parameters"])
    context, response, ground_response = (
        original,
        deepcopy(bundle.get("initial_search_fit", {}).get("net_response")),
        None,
    )
    verdict, measurement = _score(context, parameters, bundle, response)
    stages, applied = [], []
    prepared_net, observations = None, {}
    prepared_serve_mechanism = None
    prepared_serve_role = None
    try:
        # Freeze net-bounded witnesses/context before any first-contact profiling.
        # Preparing again after serve fitting could reset the newly fitted epoch.
        if original["scene"].right_boundary_kind == "original_contact":
            raise ValueError("terminal-net refinement is not applicable at an original contact")
        prepared_net = net.prepare(bundle)
        proposed = prepared_net["context"]
        stages.append(
            dict(
                stage="net_support",
                status="qualified",
                context=prepared_net["context_receipt"],
                targets=prepared_net["target_receipts"],
            )
        )
    except (ValueError, KeyError) as error:
        proposed = original
        stages.append(dict(stage="net_support", status="unsupported_input", reason=str(error)))
    except Exception as error:
        proposed = original
        stages.append(
            dict(
                stage="net_support",
                status="execution_failed",
                reason=f"{type(error).__name__}: {error}",
            )
        )
    if prepared_net is None and proposed.get("terminal_rebound_frames") is None:
        try:
            extended, receipt = census.extended_context(
                proposed, bundle["labels"], bundle["cameras"], bundle["search"]
            )
            if extended is not None:
                proposed = extended
            stages.append(
                dict(
                    stage="serve_context",
                    status="prepared" if extended is not None else "source_context",
                    receipt=receipt,
                )
            )
        except (ValueError, KeyError) as error:
            stages.append(
                dict(
                    stage="serve_context",
                    status="unsupported_input",
                    reason=str(error),
                    retained="original native context; no aftermath extension",
                )
            )
        except Exception as error:
            stages.append(
                dict(
                    stage="serve_context",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                    retained="original native context; no aftermath extension",
                )
            )
    # Optional terminal observations are not a prerequisite for fitting the
    # already supported serve/return prefix. Its own qualification remains
    # strict, including original event intervals and native observation support.
    try:
        serve_context, observations, mechanism = _prepare_serve(
            bundle,
            proposed,
            joint_toss_requalification=policy.joint_toss_requalification,
            earlier_toss_support=policy.earlier_toss_support,
            toss_player_prior=policy.toss_player_prior,
            independent_toss_horizontal_prior=policy.independent_toss_horizontal_prior,
        )
    except (ValueError, KeyError) as error:
        stages.append(dict(stage="serve", status="unsupported_input", reason=str(error)))
    except Exception as error:
        stages.append(
            dict(
                stage="serve", status="execution_failed", reason=f"{type(error).__name__}: {error}"
            )
        )
    else:
        prepared_serve_mechanism = mechanism
        prepared_serve_role = deepcopy(serve_context.get("prefix_serve_evidence"))
        try:
            active, candidate, details = _fit_serve(
                serve_context,
                parameters,
                observations,
                mechanism,
                bundle,
                policy,
                **({"net_response": response} if response is not None else {}),
            )
            candidate = _validate_state(serve_context, active, candidate)
            # Ensure the complete physical chain is representable before adopting it.
            candidate_ground = interior_ground.normalize(
                details.get("ground_response"), active["scene"]
            )
            with interior_ground.using_response(candidate_ground, active["scene"]):
                serve.full.model.chain(active["scene"], candidate)
            candidate_verdict, candidate_measurement = _score(
                active,
                candidate,
                bundle,
                response,
                **({"ground_response": candidate_ground} if candidate_ground is not None else {}),
            )
            context, parameters, ground_response = active, candidate, candidate_ground
            verdict, measurement = candidate_verdict, candidate_measurement
            applied.append(mechanism)
            stages.append(
                dict(
                    stage="serve",
                    status="applied",
                    mechanism=mechanism,
                    fit=details,
                    toss_admission=observations.get("admission_receipt"),
                    toss_player_anchor=observations.get("player_anchor"),
                )
            )
        except Exception as error:
            stages.append(
                dict(
                    stage="serve",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                    toss_admission=observations.get("admission_receipt"),
                    toss_player_anchor=observations.get("player_anchor"),
                )
            )
    mechanism = applied[0] if applied else None
    if prepared_net is not None:
        try:
            active = context if applied else prepared_net["context"]
            fitted = _fit_net(
                active,
                parameters,
                bundle,
                policy,
                prepared_net,
                seconds=policy.seconds_per_stage,
                **({"source_response": response} if policy.net_first_contact_toss else {}),
            )
            context, parameters, response = (
                fitted.get("context", active),
                fitted["parameters"],
                fitted["response"],
            )
            verdict, measurement = fitted["verdict"], fitted["measurement"]
            applied.append("terminal_net")
            stages.append(dict(stage="net", status="applied", **fitted["receipt"]))
        except Exception as error:
            stages.append(
                dict(
                    stage="net",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
    if policy.prefix_net_revisit:
        # Fixed order: prefix -> net -> response-aware declared-net prefix revisit -> one
        # net follow-on. Both run under one declared common budget from the current
        # state; a failed follow-on keeps the latest valid prefix and its response.
        revisit_applied = False
        terminal_tail_retry = bool(
            bundle.get("initial_search_fit") is not None
            and context["scene"].terminal_net_tail is not None
            and len(context["scene"].pixels) == 2
            and prepared_serve_mechanism == "serve_prefix"
        )
        revisit_mechanism = prepared_serve_mechanism if terminal_tail_retry else mechanism
        revisit_context = (
            context | {"prefix_serve_evidence": prepared_serve_role}
            if terminal_tail_retry and prepared_serve_role is not None
            else context
        )
        try:
            qualification = _qualify_revisit(
                context,
                revisit_mechanism,
                response if "terminal_net" in applied else None,
                **({"terminal_tail_retry": True} if terminal_tail_retry else {}),
            )
        except (ValueError, KeyError) as error:
            stages.append(
                dict(stage="prefix_net_revisit", status="unsupported_input", reason=str(error))
            )
        except Exception as error:
            stages.append(
                dict(
                    stage="prefix_net_revisit",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
        else:
            try:
                with interior_ground.using_response(ground_response, context["scene"]):
                    active, candidate, details = _fit_serve(
                        revisit_context,
                        parameters,
                        observations,
                        revisit_mechanism,
                        bundle,
                        policy,
                        net_response=response,
                        seconds=policy.prefix_net_revisit_seconds,
                    )
                    if details["initial_source"] != np.asarray(parameters, float).tolist():
                        raise ValueError("prefix revisit did not start from the current state")
                    if details["retained_later_net_response"] != response:
                        raise ValueError(
                            "prefix revisit did not retain its same-invocation net response"
                        )
                    candidate = _validate_state(context, active, candidate)
                    with using_response(response_from_record(response)):
                        serve.full.model.chain(active["scene"], candidate)
                    candidate_verdict, candidate_measurement = _score(
                        active,
                        candidate,
                        bundle,
                        response,
                        **(
                            {"ground_response": ground_response}
                            if ground_response is not None
                            else {}
                        ),
                    )
                context, parameters = active, candidate
                verdict, measurement = candidate_verdict, candidate_measurement
                applied.append("serve_prefix_net_revisit")
                revisit_applied = True
                stages.append(
                    dict(
                        stage="prefix_net_revisit",
                        status="applied",
                        mechanism="serve_prefix_net_revisit",
                        qualification=qualification,
                        carried_net_response=deepcopy(response),
                        carried_ground_response=deepcopy(ground_response),
                        budget=dict(
                            max_nfev=policy.serve_max_nfev,
                            seconds=policy.prefix_net_revisit_seconds,
                        ),
                        fit=details,
                        toss_admission=observations.get("admission_receipt"),
                        toss_player_anchor=observations.get("player_anchor"),
                    )
                )
            except Exception as error:
                stages.append(
                    dict(
                        stage="prefix_net_revisit",
                        status="execution_failed",
                        qualification=qualification,
                        reason=f"{type(error).__name__}: {error}",
                    )
                )
        if revisit_applied:
            try:
                fitted = _fit_net(
                    context,
                    parameters,
                    bundle,
                    policy,
                    prepared_net,
                    seconds=policy.prefix_net_revisit_seconds,
                )
                parameters, response = fitted["parameters"], fitted["response"]
                verdict, measurement = fitted["verdict"], fitted["measurement"]
                applied.append("terminal_net_follow_on")
                stages.append(
                    dict(
                        stage="net_follow_on",
                        status="applied",
                        budget=dict(
                            max_nfev=policy.net_max_nfev,
                            seconds=policy.prefix_net_revisit_seconds,
                        ),
                        **fitted["receipt"],
                    )
                )
            except Exception as error:
                stages.append(
                    dict(
                        stage="net_follow_on",
                        status="execution_failed",
                        retained="latest valid revisited prefix and its carried net response",
                        reason=f"{type(error).__name__}: {error}",
                    )
                )
    if policy.interior_ground_normal:
        try:
            block_options = {"allow_pairs": True} if policy.interior_short_blocks else {}
            if policy.interior_sparse_wings:
                block_options["allow_sparse_wings"] = True
            inventory = interior.inventory(context, bundle["duration"], **block_options)
            if not any(row["eligible"] for row in inventory):
                stages.append(
                    dict(
                        stage="interior_ground",
                        status="unsupported_input",
                        inventory=inventory,
                        reason="no original eligible ground block"
                        if policy.interior_short_blocks
                        else "no original eligible three-ground block",
                    )
                )
            else:
                fit = interior.fit
                fit_budget = dict(seconds=policy.interior_seconds)
                if policy.interior_schedule in ("sweep", "sweep_revisit"):
                    fit = interior_sweep.fit_sweep
                    fit_budget = dict(
                        seconds=policy.interior_sweep_seconds,
                        block_seconds=policy.interior_seconds,
                        **(
                            {"revisit_failed": True}
                            if policy.interior_schedule == "sweep_revisit"
                            else {}
                        ),
                    )
                if policy.interior_ground_horizontal:
                    fit_budget["horizontal_retention"] = True
                if policy.interior_restoration_geometry_only:
                    fit_budget["restoration_geometry_only"] = True
                active, details = fit(
                    context,
                    parameters,
                    bundle["duration"],
                    ground_response=ground_response,
                    net_response=response,
                    labels=bundle.get("labels"),
                    cameras=bundle.get("cameras"),
                    pose_rows=bundle.get("observed_pose_rows"),
                    pose_image_scale=bundle.get("pose_image_scale", 1.0),
                    maxiter=policy.interior_maxiter,
                    **block_options,
                    **fit_budget,
                )
                candidate = _validate_state(context, active, details["full_vector"])
                if (
                    details["initial_source"] != parameters.tolist()
                    or details["initial_ground_response"] != ground_response
                    or details["retained_net_response"] != response
                ):
                    raise ValueError(
                        "interior fit did not retain its same-invocation source/response"
                    )
                candidate_ground = interior_ground.normalize(
                    details["ground_response"], active["scene"]
                )
                if details["status"] == "source_retained":
                    if (
                        not np.array_equal(candidate, parameters)
                        or candidate_ground != ground_response
                    ):
                        raise ValueError("interior fallback changed the incumbent")
                    stages.append(
                        dict(stage="interior_ground", status="source_retained", fit=details)
                    )
                elif details["status"] == "refined":
                    candidate_verdict, candidate_measurement = _score(
                        active, candidate, bundle, response, candidate_ground
                    )
                    context, parameters, ground_response = active, candidate, candidate_ground
                    verdict, measurement = candidate_verdict, candidate_measurement
                    applied.append("interior_ground_normal")
                    stages.append(dict(stage="interior_ground", status="applied", fit=details))
                else:
                    raise ValueError("unsupported interior fit status")
        except Exception as error:
            stages.append(
                dict(
                    stage="interior_ground",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
    if policy.terminal_net_coupling:
        try:
            terminal_coupling.qualify(context, bundle["duration"])
            normal_ground.prior_from_observations(context, parameters)
        except (ValueError, KeyError) as error:
            stages.append(
                dict(stage="terminal_net_coupling", status="unsupported_input", reason=str(error))
            )
        except Exception as error:
            stages.append(
                dict(
                    stage="terminal_net_coupling",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
        else:
            try:
                active, details = terminal_coupling.fit(
                    context,
                    parameters,
                    bundle["duration"],
                    ground_response=ground_response,
                    net_response=response,
                    enabled=True,
                    seconds=policy.terminal_coupling_seconds,
                    maxiter=policy.terminal_coupling_maxiter,
                )
                candidate = _validate_state(context, active, details["full_vector"])
                if active is not context or (
                    details["initial_source"] != parameters.tolist()
                    or details["initial_ground_response"] != ground_response
                    or details["initial_net_response"] != response
                    or details["gate_selection"] is not False
                    or details["external_fitted_inputs_used"] is not False
                ):
                    raise ValueError(
                        "terminal coupling changed its current context/source identity"
                    )
                candidate_ground = interior_ground.normalize(
                    details["ground_response"], active["scene"]
                )
                candidate_response = details["net_response"]
                if details["status"] == "source_retained":
                    if (
                        not np.array_equal(candidate, parameters)
                        or candidate_ground != ground_response
                        or candidate_response != response
                    ):
                        raise ValueError("terminal coupling fallback changed its incumbent")
                    failed = bool(details.get("reason")) or not details.get("fit", {}).get(
                        "feasible_calls", 0
                    )
                    stages.append(
                        dict(
                            stage="terminal_net_coupling",
                            status="execution_failed" if failed else "source_retained",
                            reason=details.get("reason")
                            or ("no feasible coupled trial" if failed else None),
                            fit=details,
                        )
                    )
                elif details["status"] == "improved":
                    candidate_verdict, candidate_measurement = _score(
                        active, candidate, bundle, candidate_response, candidate_ground
                    )
                    parameters, ground_response, response = (
                        candidate,
                        candidate_ground,
                        candidate_response,
                    )
                    verdict, measurement = candidate_verdict, candidate_measurement
                    applied.append("terminal_net_contact_coupling")
                    stages.append(
                        dict(stage="terminal_net_coupling", status="applied", fit=details)
                    )
                else:
                    raise ValueError("unsupported terminal coupling status")
            except Exception as error:
                stages.append(
                    dict(
                        stage="terminal_net_coupling",
                        status="execution_failed",
                        reason=f"{type(error).__name__}: {error}",
                    )
                )
    if policy.terminal_impact_intervals:
        adapter = terminal_response if policy.terminal_ground_normal else terminal_impact
        try:
            adapter.qualify(context, bundle["duration"])
        except (ValueError, KeyError) as error:
            stages.append(
                dict(stage="terminal_ground", status="unsupported_input", reason=str(error))
            )
        except Exception as error:
            stages.append(
                dict(
                    stage="terminal_ground",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
        else:
            try:
                if policy.terminal_ground_normal:
                    # Same stage slot and budgets; the final flight's first-ground normal
                    # moves through the persistent registry, which scoring then carries.
                    active, details = terminal_response.fit(
                        context,
                        parameters,
                        bundle["duration"],
                        ground_response=ground_response,
                        net_response=response,
                        free_normal=True,
                        max_nfev=policy.serve_max_nfev,
                        seconds=policy.seconds_per_stage,
                    )
                    candidate = _validate_state(context, active, details["full_vector"])
                    if (
                        details["initial_source"] != parameters.tolist()
                        or details["initial_ground_response"] != ground_response
                        or details["retained_net_response"] != response
                    ):
                        raise ValueError(
                            "terminal fit did not retain its same-invocation source/response"
                        )
                    candidate_ground = interior_ground.normalize(
                        details["ground_response"], active["scene"]
                    )
                    if details["status"] == "source_retained":
                        if (
                            not np.array_equal(candidate, parameters)
                            or candidate_ground != ground_response
                        ):
                            raise ValueError("terminal fallback changed the incumbent")
                        stages.append(
                            dict(
                                stage="terminal_ground",
                                status="source_retained"
                                if details.get("feasible_output", True)
                                else "execution_failed",
                                reason=details.get("reason"),
                                fit=details,
                            )
                        )
                    elif details["status"] == "refined":
                        candidate_verdict, candidate_measurement = _score(
                            active, candidate, bundle, response, candidate_ground
                        )
                        context, parameters, ground_response = active, candidate, candidate_ground
                        verdict, measurement = candidate_verdict, candidate_measurement
                        applied.append("terminal_ground_normal")
                        stages.append(
                            dict(
                                stage="terminal_ground",
                                status="applied",
                                mechanism="terminal_ground_normal",
                                fit=details,
                            )
                        )
                    else:
                        raise ValueError("unsupported terminal fit status")
                else:
                    with interior_ground.using_response(ground_response, context["scene"]):
                        active, details = terminal_impact.fit(
                            context,
                            parameters,
                            bundle["duration"],
                            max_nfev=policy.serve_max_nfev,
                            seconds=policy.seconds_per_stage,
                            net_response=response,
                        )
                    candidate = _validate_state(context, active, details["full_vector"])
                    candidate_verdict, candidate_measurement = _score(
                        active,
                        candidate,
                        bundle,
                        response,
                        **(
                            {"ground_response": ground_response}
                            if ground_response is not None
                            else {}
                        ),
                    )
                    if ground_response is not None:
                        details = details | {"retained_ground_response": deepcopy(ground_response)}
                    context, parameters = active, candidate
                    verdict, measurement = candidate_verdict, candidate_measurement
                    applied.append("terminal_impact_interval")
                    stages.append(dict(stage="terminal_ground", status="applied", fit=details))
            except Exception as error:
                stages.append(
                    dict(
                        stage="terminal_ground",
                        status="execution_failed",
                        reason=f"{type(error).__name__}: {error}",
                    )
                )
    if policy.terminal_ground_coupling:
        sparse_options = {"allow_sparse_wings": True} if policy.terminal_ground_sparse_wings else {}
        try:
            terminal_ground_coupling.qualify(
                context,
                bundle["duration"],
                second_normal=True,
                horizontal_retention=True,
                **sparse_options,
            )
        except (ValueError, KeyError) as error:
            stages.append(
                dict(
                    stage="terminal_ground_coupling", status="unsupported_input", reason=str(error)
                )
            )
        except Exception as error:
            stages.append(
                dict(
                    stage="terminal_ground_coupling",
                    status="execution_failed",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
        else:
            try:
                active, details = terminal_ground_coupling.fit(
                    context,
                    parameters,
                    bundle["duration"],
                    ground_response=ground_response,
                    net_response=response,
                    enabled=True,
                    second_normal=True,
                    horizontal_retention=True,
                    **sparse_options,
                    seconds=policy.terminal_ground_coupling_seconds,
                    maxiter=policy.terminal_ground_coupling_maxiter,
                )
                candidate = _validate_state(context, active, details["full_vector"])
                if active is not context or (
                    details["initial_source"] != parameters.tolist()
                    or details["initial_ground_response"] != ground_response
                    or details["retained_net_response"] != response
                    or details["gate_selection"] is not False
                    or details["external_fitted_inputs_used"] is not False
                ):
                    raise ValueError("terminal ground coupling changed its current source identity")
                candidate_ground = interior_ground.normalize(
                    details["ground_response"], active["scene"]
                )
                if details["status"] == "source_retained":
                    if (
                        not np.array_equal(candidate, parameters)
                        or candidate_ground != ground_response
                    ):
                        raise ValueError("terminal ground coupling fallback changed its incumbent")
                    failed_fit = bool(details.get("reason")) or not details.get("counters", {}).get(
                        "feasible", 0
                    )
                    stages.append(
                        dict(
                            stage="terminal_ground_coupling",
                            status="execution_failed" if failed_fit else "source_retained",
                            reason=details.get("reason")
                            or ("no feasible coupled trial" if failed_fit else None),
                            fit=details,
                        )
                    )
                elif details["status"] == "improved":
                    if details.get("replay", {}).get("status") != "verified":
                        raise ValueError("terminal ground coupling requires full-state replay")
                    candidate_verdict, candidate_measurement = _score(
                        active, candidate, bundle, response, candidate_ground
                    )
                    parameters, ground_response = candidate, candidate_ground
                    verdict, measurement = candidate_verdict, candidate_measurement
                    applied.append("terminal_ground_contact_coupling")
                    stages.append(
                        dict(stage="terminal_ground_coupling", status="applied", fit=details)
                    )
                else:
                    raise ValueError("unsupported terminal ground coupling status")
            except Exception as error:
                stages.append(
                    dict(
                        stage="terminal_ground_coupling",
                        status="execution_failed",
                        reason=f"{type(error).__name__}: {error}",
                    )
                )
    failed = any(row["status"] == "execution_failed" for row in stages)
    return dict(
        status=("partially_refined" if failed else "refined")
        if applied
        else ("execution_failed_source_retained" if failed else "unsupported_input_noop"),
        context=context,
        parameters=parameters,
        measurement=measurement,
        verdict=verdict,
        net_response=response,
        ground_response=ground_response,
        applied_mechanisms=applied,
        stages=stages,
        policy=asdict(policy),
        selection="fixed evidence-qualified stages; each numerical helper selects by its input objective; no gate-based rollback",
        external_fitted_inputs_used=False,
        caller_must_attest_same_invocation_search=True,
        runtime_model_calls=0,
        independent_xyz_truth_available=False,
    )


def refine(bundle: dict, policy: Policy = Policy()) -> dict:
    """Carry the input-ranked search response through the same shared refinement."""
    from cv.experiments.connected_shooting import observation_net_seed

    with observation_net_seed.response_context(bundle.get("initial_search_fit", {})):
        return _refine(bundle, policy)
