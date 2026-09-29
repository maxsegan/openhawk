"""Optional free-toss admission from an explicit versioned packet field.

The caller supplies the invocation's normally verified packet/labels/cameras.
No extra certificate, ambient input, fitted state or runtime note parsing is used.
The observed interval excludes held-ball frames; exact release is not inferred.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import observation_operator

from copy import deepcopy

import numpy as np

from cv.experiments.connected_shooting import labeled_free_toss_support as free_toss
from cv.experiments.connected_shooting import labeled_toss_front as front
from cv.experiments.connected_shooting.labeled_isolated_serve import bind_toss

SCHEMA = "labeled_free_toss_support_v1"


def admit(
    original: dict,
    labels: dict,
    cameras: dict,
    *,
    clip: str,
    original_contact_interval: tuple[float, float],
    exposure_frames: float,
    support: dict | None = None,
    enabled: bool = False,
) -> tuple[dict, dict]:
    """Return fixed original observations and a separate admission receipt.

    `support` comes from this attempt's explicit `free_toss_support` packet field;
    its source binding is the ordinary packet hash in the invocation. Absent or
    disabled support preserves the original object. A malformed supplied field
    raises instead of silently reverting. This first adapter is admission-only:
    it cannot remove any previously eligible incoming observation.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit boolean admission policy required")
    receipt = dict(schema=SCHEMA, enabled=enabled, original_status=original.get("status"))
    if not enabled or support is None:
        return original, receipt | {"status": "disabled" if not enabled else "absent_support"}
    span = observation_operator.support_span(exposure_frames)
    lo, hi = np.asarray(original_contact_interval, float)
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("finite original contact interval and exposure in (0,1] required")
    if (
        support.get("schema") != SCHEMA
        or not support.get("rationale")
        or not support.get("annotation_origin")
    ):
        raise ValueError("explicit versioned native free-motion review required")
    receipt["observed_support"] = deepcopy(support)
    if support.get("status") == "no_extension":
        return original, receipt | {"status": "explicit_no_extension"}
    if support.get("status") != "supported":
        raise ValueError("supported or explicit no_extension status required")
    interval = np.asarray(support.get("frame_interval"), float)
    if (
        interval.shape != (2,)
        or not np.isfinite(interval).all()
        or np.any(interval != np.floor(interval))
        or interval[0] > interval[1]
    ):
        raise ValueError("ordered native integer free-motion interval required")
    first, last = interval
    if last + span >= lo:
        raise ValueError("entire admitted exposure must precede ORIGINAL contact lower bound")
    bound = bind_toss({"toss_observations": original}, labels, cameras, clip)
    old_rows = {
        float(r["frame"]): r for r in bound.get("rows", []) if float(r["frame"]) + span < lo
    }
    if any(not first <= frame <= last for frame in old_rows):
        raise ValueError("admission-only extension may not remove original eligible rows")
    admitted = front.enrich(free_toss.observations(labels, cameras, clip, support), labels, clip)
    for row in admitted["rows"]:
        if float(row["frame"]) in old_rows:
            row["uncertainty_px"] = old_rows[float(row["frame"])]["uncertainty_px"]
    new_frames = {float(row["frame"]) for row in admitted["rows"]}
    if not old_rows.keys() <= new_frames:
        raise ValueError("extension may not lose previously eligible original support")
    added = sorted(new_frames - old_rows.keys())
    receipt |= dict(
        status="extended" if added else "same_observations",
        original_eligible_frames=sorted(old_rows),
        admitted_frames=sorted(new_frames),
        added_frames=added,
        abstained=admitted["abstained"],
        minimum_joint_observations=4,
        membership="fixed before fit using original contact lower bound",
        priors_added=False,
        native_timestamps_changed=False,
        release_inferred=False,
    )
    if not added:
        return original, receipt
    admitted["upstream_observation_receipt"] = {
        k: deepcopy(v) for k, v in original.items() if k != "rows"
    }
    return admitted, receipt
