"""Activate an explicit reviewed free-toss span from unchanged original inputs.

Experimental labeled-input preparation. The supplied span excludes hand-held
ball pictures; this module does not infer release, player identity or ground XYZ.
It preserves missing/ambiguous labels as abstentions and adds no automatic track.
"""

from __future__ import annotations
import numpy as np
from cv.experiments.connected_shooting import camera_geometry
from cv.experiments.connected_shooting.labeled_isolated_serve import bind_toss


def observations(labels: dict, cameras: dict, clip: str, support: dict) -> dict:
    """Return original free-flight fronts and explicit missing-support receipts."""
    first, last = support["frame_interval"]
    if not support.get("rationale") or not support.get("annotation_origin"):
        raise ValueError("explicit native-reviewed free-toss qualification required")
    if (
        not np.isfinite([first, last]).all()
        or first != int(first)
        or last != int(last)
        or first > last
    ):
        raise ValueError("ordered native integer free-toss support interval required")
    groups = [r for r in labels["ball"]["records"] if r["clip"] == clip]
    if len(groups) != 1:
        raise ValueError("one original clip ball record required")
    pixels = {r["frame"]: r for r in groups[0]["frames"]}
    views = {r["frame"]: r for r in cameras["cameras"]}
    if len(pixels) != len(groups[0]["frames"]) or len(views) != len(cameras["cameras"]):
        raise ValueError("unique original native records required")
    rows, abstained = [], []
    for frame in range(int(first), int(last) + 1):
        point, view = pixels.get(frame), views.get(frame)
        if point is None or point.get("status") != "visible":
            abstained.append(dict(frame=frame, reason="original ball absent or abstained"))
            continue
        if (
            view is None
            or view.get("status") != "supported"
            or view.get("supported", True) is False
        ):
            abstained.append(dict(frame=frame, reason="original camera unsupported"))
            continue
        radius = point.get("uncertainty_radius_px1080")
        rows.append(
            dict(
                frame=frame,
                pixel=[point["x1080"], point["y1080"]],
                camera=view["P"],
                **camera_geometry.radial_fields(view),
                uncertainty_px=max(2.0, 2.0 if radius is None else float(radius)),
                source="original_labeled_front_explicit_free_toss_span",
            )
        )
    camera_geometry.rows_radial(rows)
    result = dict(
        status="supported" if len(rows) >= 4 else "abstained",
        rows=rows,
        clip=clip,
        support=support,
        abstained=abstained,
        labeled_fronts=len(rows),
        automatic_fallbacks=0,
        release_epoch_inferred=False,
        visible_labels_replaced=0,
        human_derived=True,
    )
    return bind_toss({"toss_observations": result}, labels, cameras, clip)
