"""Optional additional original-ground initializer for terminal-net refinement.

Both candidates fit the same in-invocation source, observations and objective.
The existing passive/source family remains available. This is not the upstream
three-ray ``observation_net_seed`` search family, and consumes no cached fits.
"""

from copy import deepcopy
import time

import numpy as np

from cv.experiments.connected_shooting import labeled_net_epoch_fit as epoch
from cv.experiments.connected_shooting import labeled_net_recipe as recipe


def qualify(context: dict) -> dict:
    """Reuse the existing occurrence contract and bind one native ground target."""
    if "scene" not in context:
        return dict(status="unavailable", reason="original scene required")
    if getattr(context["scene"], "terminal_net_tail", None) is not None:
        return dict(status="unavailable", reason="unobserved ground in terminal net tail")
    if getattr(context["scene"], "right_boundary_kind", None) == "original_contact":
        return dict(status="unavailable", reason="original contact is not a terminal ground")
    try:
        eligible = recipe.qualify(context)
    except (ValueError, KeyError, TypeError) as error:
        return dict(status="unavailable", reason=str(error))
    if len(eligible["ground_events"]) != 1:
        return dict(status="unavailable", reason="existing two-ground ballistic family retained")
    event = eligible["ground_events"][0]
    try:
        targets = [t for t in context["targets"][-1] if t.get("event_frame") == event["frame"]]
    except (KeyError, IndexError, TypeError, AttributeError):
        return dict(status="unavailable", reason="original terminal ground targets required")
    if len(targets) != 1:
        return dict(status="unavailable", reason="unique original first-ground target required")
    target = targets[0]
    try:
        xyz = np.asarray(target["xyz_m"], float)
        frames = np.asarray(target["native_frames"], float)
        available = np.unique(
            np.r_[
                context["scene"].observation_frames[-1], context["heldout"].observation_frames[-1]
            ]
        )
        net_upper = float(eligible["net"]["frame_interval"][1])
        if (
            xyz.shape != (3,)
            or not np.isfinite(xyz).all()
            or abs(xyz[2] - epoch.full.model.R_BALL) > 1e-5
            or frames.ndim != 1
            or len(np.unique(frames)) < 2
            or not np.isfinite(frames).all()
            or not np.isin(frames, available).all()
            or np.any(frames <= net_upper)
        ):
            raise ValueError("finite native post-net ground target required")
    except (ValueError, TypeError, KeyError) as error:
        return dict(status="unavailable", reason=str(error))
    return dict(
        status="qualified",
        net=deepcopy(eligible["net"]),
        first_ground=deepcopy(event),
        target=deepcopy(target),
        source="existing original-event and camera-bound native ground witness",
        target_is_independent_xyz_truth=False,
        evidence_consumed_by_fit="scene and original heldout native rows; no independent heldout claim",
        frame_policy="original representative epoch inside its unchanged supplied interval",
        net_height_policy="same initial net chart; ground witness does not determine net height",
    )


def fit(context, source, duration, *, enabled=False, maxiter, seconds, **kwargs):
    """Run ordinary and qualified additional starts inside the original budget.

    A fixed half budget reserves work for each family; unused first-family time
    carries forward. The total iteration allowance and wall deadline do not grow.
    ``best`` always carries its own response and the original epoch-fit schema.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit boolean net seed-family option required")
    if not enabled:
        return epoch.fit(context, source, duration, maxiter=maxiter, seconds=seconds, **kwargs)
    admission = qualify(context)
    if kwargs.get("first_ground_seed") is not None:
        admission = dict(
            status="unavailable", reason="explicit ballistic initializer already supplied"
        )
    elif not kwargs.get("free_net_velocity") or kwargs.get("first_contact_toss") is not None:
        admission = dict(status="unavailable", reason="fixed-contact free net response required")
    elif type(maxiter) is not int or maxiter < 2:
        admission = dict(status="unavailable", reason="two iteration allowances required")
    if admission["status"] != "qualified":
        result = epoch.fit(context, source, duration, maxiter=maxiter, seconds=seconds, **kwargs)
        return result | {
            "seed_family": dict(admission=admission, selected="ordinary", additional_runs=0)
        }
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite shared net family budget required")
    started = time.monotonic()
    allowance = ((maxiter + 1) // 2, maxiter // 2)
    candidates = []
    selected = None
    for index, name in enumerate(("ordinary", "observed_first_ground")):
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            candidates.append(
                dict(
                    name=name, status="deadline_before_start", maxiter=allowance[index], seconds=0.0
                )
            )
            continue
        local_seconds = min(remaining, seconds / 2) if index == 0 else remaining
        options = (
            kwargs if index == 0 else kwargs | {"first_ground_seed": deepcopy(admission["target"])}
        )
        try:
            result = epoch.fit(
                context,
                source,
                duration,
                maxiter=allowance[index],
                seconds=local_seconds,
                **options,
            )
            best = result.get("best")
            if not isinstance(best, dict) or not np.isfinite(best.get("cost", np.nan)):
                raise ValueError("no finite visited input-objective candidate")
            candidates.append(
                dict(
                    name=name,
                    status="completed",
                    maxiter=allowance[index],
                    seconds=local_seconds,
                    result=result,
                )
            )
            if selected is None or best["cost"] < candidates[selected]["result"]["best"]["cost"]:
                selected = len(candidates) - 1
        except (ValueError, TimeoutError) as error:
            candidates.append(
                dict(
                    name=name,
                    status="execution_failed",
                    maxiter=allowance[index],
                    seconds=local_seconds,
                    reason=str(error),
                )
            )
    receipt = dict(
        admission=admission,
        candidates=candidates,
        selected=None if selected is None else candidates[selected]["name"],
        original_maxiter=maxiter,
        iteration_allowances=list(allowance),
        original_seconds=seconds,
        elapsed_seconds=time.monotonic() - started,
        selection="finite visited original input objective; stable ordinary tie preference",
        gate_selection=False,
        external_fitted_seed=False,
        original_context_unchanged=True,
    )
    if selected is None:
        raise ValueError(f"no evaluable terminal-net seed family: {receipt}")
    return candidates[selected]["result"] | {"seed_family": receipt}
