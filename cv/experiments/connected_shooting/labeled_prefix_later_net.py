"""Joint non-net serve fitting while retaining a later declared net and response.

Explicit opened labeled API, not an automatic or shortened-point path. Call fit
with the source net_response record; it remains active through every root,
full-point residual and exported chain. Replay requires using_response with the
returned retained_later_net_response. Ordinary prefix defaults are unchanged.
"""

from copy import deepcopy

import numpy as np

from cv.experiments.connected_shooting import net_constraints
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
from cv.experiments.connected_shooting.labeled_passive_tape import using_response


def qualify_later_nets(context: dict) -> None:
    """Require original matching scene/event membership, with no first-flight net."""
    scene, check = context["scene"], context["heldout"]
    a, b = scene.net_hit_frames, check.net_hit_frames
    if a is None or b is None or not any(len(g) for g in a):
        raise ValueError("explicit later net membership in both physical scenes required")
    if len(a[0]) or len(b[0]):
        raise ValueError("the first-flight impact chart cannot include a serve net")
    if any(len(group) > 1 for group in (*a, *b)):
        raise ValueError("only one declared net hit per flight is supported")
    if (
        not np.array_equal(scene.contact_frames, check.contact_frames)
        or len(a) != len(b)
        or any(not np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    ):
        raise ValueError("training and withheld net topology must agree")
    supplied = sorted(float(e["frame"]) for e in context["events"] if e["event_type"] == "net_hit")
    modeled = sorted(float(t) for group in a for t in group)
    if supplied != modeled:
        raise ValueError("original events and physical net membership must agree")
    for event in context["events"]:
        if event["event_type"] != "net_hit":
            continue
        low, high = map(float, event["frame_interval"])
        frame = float(event["frame"])
        if not np.isfinite([low, frame, high]).all() or not low <= frame <= high or low >= high:
            raise ValueError("finite nonzero original net interval required")
        if not any(
            x < low and high < y
            for x, y in zip(scene.contact_frames[1:-1], scene.contact_frames[2:], strict=True)
        ):
            raise ValueError("whole net interval must lie inside a later flight")


def net_residuals(scene, queries, flights) -> np.ndarray:
    """Use original collision residuals at declared hits; clearance elsewhere."""
    values = []
    for i, (query, flight) in enumerate(zip(queries, flights, strict=True)):
        declared = scene.net_hit_frames[i]
        if len(declared):
            from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                collision_residuals,
            )

            values.extend(
                collision_residuals(scene, i, flight.get("net_hits", []), float(declared[0]))
            )
        else:
            values.append(net_constraints.penalty(query, flight["positions"], 0.025)[0])
    return np.asarray(values, float)


def fit(context, source, observations, *, net_response: dict, **kwargs):
    """Run the unchanged joint prefix objective with explicit later-net semantics."""
    from cv.experiments.connected_shooting import labeled_prefix_joint_impact as prefix

    if "preserve_later_nets" in kwargs:
        raise ValueError("this explicit response-bound wrapper controls later-net preservation")
    qualify_later_nets(context)
    record = deepcopy(net_response)
    response = response_from_record(record)
    with using_response(response):
        ctx, result = prefix.fit(context, source, observations, preserve_later_nets=True, **kwargs)
    result["retained_later_net_response"] = record
    result["fit_policy"]["later_net_response_fitted"] = False
    result["fit_policy"]["preserve_later_nets"]["declared_net_chart"] = bool(
        kwargs.get("declared_net_chart", False)
    )
    return ctx, result
