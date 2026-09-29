"""Chronological interior refinement with an optional failed-block revisit.

Each block receives this invocation's latest vector and response registry.
Original observations and topology select the first pass. The optional revisit
uses numerical failure plus later overlapping success, within the same deadline;
gates and reviewed trajectory quality never select a window or fitted result.
Requires the interior fitter's internal ``_first`` original-index selector.
"""

from copy import deepcopy
import time

import numpy as np

from cv.experiments.connected_shooting import (
    labeled_interior_ground_response as ground,
    labeled_interior_normal as normal,
)

SWEEP_SECONDS = 1800.0
BLOCK_SECONDS = 180.0
MAXITER_PER_PHASE = 60


def fit_sweep(
    context,
    source,
    duration,
    *,
    ground_response=None,
    net_response=None,
    labels=None,
    cameras=None,
    pose_rows=None,
    pose_image_scale=1.0,
    seconds=SWEEP_SECONDS,
    block_seconds=BLOCK_SECONDS,
    maxiter=MAXITER_PER_PHASE,
    allow_pairs=False,
    allow_sparse_wings=False,
    revisit_failed=False,
    restoration_geometry_only=False,
    horizontal_retention=False,
):
    """Visit each eligible original block once, retaining the latest incumbent.

    Each block gets min(block_seconds, remaining_seconds / remaining_blocks).
    The deadline includes inventory and wrapper work; numerical fits share their
    allocation between preparation, restoration and objective optimization.
    Completed earlier improvements survive a later numerical refusal/timeout.
    Contract violations propagate rather than masquerading as numerical failure.
    """
    started = time.monotonic()
    if (
        not np.isfinite([seconds, block_seconds]).all()
        or min(seconds, block_seconds) <= 0
        or type(maxiter) is not int
        or maxiter < 1
    ):
        raise ValueError("positive common sweep, block and iteration budgets required")
    if not np.isfinite(pose_image_scale) or pose_image_scale <= 0:
        raise ValueError("positive pose image scale required")
    inventory_options = {"allow_pairs": True} if allow_pairs else {}
    if type(allow_sparse_wings) is not bool:
        raise ValueError("allow_sparse_wings must be an explicit boolean")
    if allow_sparse_wings:
        inventory_options["allow_sparse_wings"] = True
    if type(horizontal_retention) is not bool:
        raise ValueError("horizontal_retention must be an explicit boolean")
    if type(restoration_geometry_only) is not bool:
        raise ValueError("restoration_geometry_only must be an explicit boolean")
    if type(allow_pairs) is not bool:
        raise ValueError("allow_pairs must be an explicit boolean")
    if type(revisit_failed) is not bool:
        raise ValueError("revisit_failed must be an explicit boolean")
    inventory = deepcopy(normal.inventory(context, duration, **inventory_options))
    firsts = [row["first"] for row in inventory if row["eligible"]]
    if firsts != sorted(set(firsts)):
        raise ValueError("unique chronological original interior inventory required")
    current = np.asarray(source, float).copy()
    if current.shape != (5 + 6 * len(context["scene"].pixels),) or not np.isfinite(current).all():
        raise ValueError("finite full single-shooting source vector required")
    routes = ground.normalize(deepcopy(ground_response), context["scene"])
    original_net = deepcopy(net_response)
    details = dict(
        status="source_retained",
        initial_source=current.tolist(),
        initial_ground_response=deepcopy(routes),
        retained_net_response=deepcopy(original_net),
        inventory=inventory,
        eligible_firsts=firsts,
        attempted_firsts=[],
        refined_firsts=[],
        deferred_eligible_firsts=[],
        blocks=[],
        gate_selection=False,
        external_fitted_inputs_used=False,
        policy=dict(
            selection=(
                "one chronological pass over original triples and isolated pairs"
                if allow_pairs
                else "one chronological pass over original eligible triples"
            ),
            seconds=float(seconds),
            block_seconds=float(block_seconds),
            allocation="min(block_seconds, remaining_seconds / remaining_blocks)",
            maxiter_per_phase=maxiter,
            restoration_geometry_only=restoration_geometry_only,
            **({"horizontal_retention": True} if horizontal_retention else {}),
            boundary_or_event_changes=False,
            repeated_block_retries=False,
            model_calls=0,
        ),
    )
    deadline = started + float(seconds)

    failed_inputs = {}

    def visit(first, allocation, *, revisit=False):
        nonlocal current, routes
        target = details["revisit"] if revisit else details
        block = dict(first=first, seconds_allocated=allocation)
        if revisit_failed and not revisit:
            failed_inputs[first] = (current.copy(), deepcopy(routes))
        target["attempted_firsts"].append(first)
        try:
            active, result = normal.fit(
                context,
                current.copy(),
                duration,
                _first=first,
                ground_response=deepcopy(routes),
                net_response=deepcopy(original_net),
                labels=labels,
                cameras=cameras,
                pose_rows=pose_rows,
                pose_image_scale=pose_image_scale,
                seconds=allocation,
                maxiter=maxiter,
                **inventory_options,
                **({"restoration_geometry_only": True} if restoration_geometry_only else {}),
                **({"horizontal_retention": True} if horizontal_retention else {}),
            )
        except (*normal.NUMERICAL_ERRORS, TimeoutError) as error:
            block.update(status="execution_failed", reason=f"{type(error).__name__}: {error}")
            target["blocks"].append(block)
            return
        # The single-block fitter owns numerical replay and objective selection.
        # Its unchanged-context contract preserves original global flight indices.
        if active is not context:
            raise RuntimeError("interior block replaced the original observed context")
        if (
            result["initial_source"] != current.tolist()
            or result["initial_ground_response"] != routes
            or result["retained_net_response"] != original_net
            or result["selected_first"] != first
        ):
            raise RuntimeError(
                "interior block lost its current source, responses or original index"
            )
        candidate = np.asarray(result["full_vector"], float)
        if candidate.shape != current.shape or not np.isfinite(candidate).all():
            raise RuntimeError("interior block returned an invalid complete vector")
        candidate_routes = ground.normalize(result["ground_response"], context["scene"])
        status = result["status"]
        if status == "source_retained":
            if not np.array_equal(candidate, current) or candidate_routes != routes:
                raise RuntimeError("interior block fallback changed the latest incumbent")
        elif status == "refined":
            current = candidate.copy()
            routes = deepcopy(candidate_routes)
            target["refined_firsts"].append(first)
            details["status"] = "refined"
        else:
            raise RuntimeError("unsupported interior block result status")
        block.update(status=status, fit=result)
        target["blocks"].append(block)

    for index, first in enumerate(firsts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            details["deferred_eligible_firsts"] = firsts[index:]
            break
        allocation = min(float(block_seconds), remaining / (len(firsts) - index))
        visit(first, allocation)

    if revisit_failed:
        # A later overlapping success changes the failed block's physical
        # boundary problem. Reconsider it once, using only remaining common time.
        original_blocks = list(details["blocks"])
        flights = {row["first"]: set(row["flights"]) for row in inventory if row["eligible"]}
        candidates = []
        for index, block in enumerate(original_blocks):
            fit = block.get("fit", {})
            failed = block["status"] == "execution_failed" or (
                block["status"] == "source_retained"
                and fit.get("feasibility", {}).get("feasible_trials") == 0
            )
            neighbors = [
                later["first"]
                for later in original_blocks[index + 1 :]
                if later["status"] == "refined"
                and flights[block["first"]].intersection(flights[later["first"]])
            ]
            if failed and neighbors:
                candidates.append(dict(first=block["first"], later_refined_neighbors=neighbors))
        details["revisit"] = dict(
            policy="one_failed_block_revisit_after_later_overlapping_success",
            candidates=candidates,
            attempted_firsts=[],
            refined_firsts=[],
            blocks=[],
            skipped=[],
            deferred_firsts=[],
            shared_original_seconds=float(seconds),
            additional_budget_seconds=0,
            scheduling_uses_gates=False,
        )
        details["policy"]["repeated_block_retries"] = True
        for index, candidate in enumerate(candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                details["revisit"]["deferred_firsts"] = [r["first"] for r in candidates[index:]]
                break
            first = candidate["first"]
            old_parameters, old_routes = failed_inputs[first]
            if np.array_equal(current, old_parameters) and routes == old_routes:
                details["revisit"]["skipped"].append(
                    dict(first=first, reason="same physical source and response registry")
                )
                continue
            allocation = min(float(block_seconds), remaining / (len(candidates) - index))
            visit(first, allocation, revisit=True)
        details["revisit"]["completed"] = not details["revisit"]["deferred_firsts"]
    details.update(
        full_vector=current.tolist(),
        ground_response=deepcopy(routes),
        response_routes=deepcopy(routes["routes"] if routes else []),
        wall_seconds=time.monotonic() - started,
        execution_failed_firsts=[
            b["first"] for b in details["blocks"] if b["status"] == "execution_failed"
        ],
        sweep_completed=not details["deferred_eligible_firsts"],
    )
    return context, details
