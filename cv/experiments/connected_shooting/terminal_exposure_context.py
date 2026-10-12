"""Explicit ground-ending motion context for sensor exposures, never scoring bounds.

A terminal bounce or second bounce ends a scoring point, not the ball's physical
motion. The connected simulator can supply a short subsequent passive flight.
Synthetic measurements must use already recorded raw context, never extrapolated
truth. This helper does not create pre-contact toss motion or extra native images.
"""

from dataclasses import replace

import numpy as np

from cv.experiments.connected_shooting import model, oracle_benchmark as oracle


def imaging_scene(scene: model.Scene, frames: float, termination_kind: str) -> model.Scene:
    scene.validate()
    if model.original_contact_boundary(scene):
        # The modeled domain stops at supplied contact k. There is no ground
        # ending to record passive context after, so the domain is never extended.
        raise ValueError(
            "an original-contact right boundary has no passive ground-ending image context"
        )
    if (
        isinstance(frames, (bool, np.bool_))
        or not np.isfinite(frames)
        or not 0 <= frames <= 1
        or termination_kind not in {"terminal_bounce", "second_bounce"}
    ):
        raise ValueError("explicit <=1frame passive ground-ending image context required")
    bounds = scene.contact_frames.copy()
    bounds[-1] += frames
    extended = replace(scene, contact_frames=bounds)
    extended.validate()
    return extended


def recorded_truth(scene: model.Scene, raw_paths: list[dict]) -> list[dict]:
    """Select the explicit imaging interval from immutable recorded truth samples."""
    scene.validate()
    if len(raw_paths) != len(scene.contact_frames) - 1:
        raise ValueError("raw truth flight inventory differs")
    output = []
    for i, path in enumerate(raw_paths):
        times, xyz = np.asarray(path["frames"], float), np.asarray(path["positions"], float)
        lo, hi = scene.contact_frames[i : i + 2]
        if (
            times.ndim != 1
            or len(times) < 2
            or xyz.shape != (len(times), 3)
            or not np.isfinite(times).all()
            or not np.isfinite(xyz).all()
            or np.any(np.diff(times) <= 0)
            or times[0] > lo
            or times[-1] < hi
        ):
            raise ValueError(
                "recorded context must cover complete exposure; no truth extrapolation"
            )
        selected = np.unique(np.r_[lo, times[(times > lo) & (times < hi)], hi])
        output.append({**path, "frames": selected, "positions": oracle.sample(path, selected)})
    if any(
        not np.allclose(a["positions"][-1], b["positions"][0], atol=1e-8, rtol=0)
        for a, b in zip(output, output[1:])
    ):
        raise ValueError("recorded contact context is discontinuous")
    return output


def recorded_point(scene: model.Scene, point: dict) -> list[dict]:
    """Use actual recorded sample coverage, not an obsolete unscored horizon tag."""
    source = point.get("trajectory_truth", {})
    if source.get("schema") != "simulation_trajectory_samples_v1":
        raise ValueError("recorded point trajectory required for exposure context")
    raw = {
        "frames": np.asarray(source["frames"], float),
        "positions": np.asarray(source["positions"], float),
    }
    return recorded_truth(scene, [raw] * (len(scene.contact_frames) - 1))


#: Two independent annotators agree on a physical event epoch to within about
#: half a native frame, so a modelled scoring termination sitting inside that
#: bracket of the last labeled visible front is not evidence of a later ending.
#: This is the whole allowance; it is never a licence to swallow a live flight.
ANNOTATOR_AGREEMENT_FRAMES = 0.5


def passive_context(
    ending_frame: float,
    native_frames,
    *,
    live_contact_frames=(),
    allowance_frames: float = ANNOTATOR_AGREEMENT_FRAMES,
) -> dict:
    """Name the exposures a scoring termination leaves behind, discarding none.

    The scoring point ends at its second bounce, first out bounce, exit or wall.
    Native exposures recorded after that epoch are passive context: they keep
    their original timestamps and their image residual, and they are never a
    live flight, a new contact or a reason to reject a point that already ended.

    ``ending_frame`` is the epoch after which the point has ended -- the ground
    impact's dwell end, not the labeled ending tag. The allowance is the
    annotators' own agreement on that epoch and is capped at one native frame,
    because a longer passive window is indistinguishable from an erased flight.
    A labeled live contact inside the passive span fails closed for the same
    reason; a contact beyond it is reported as the dead-ball touch it is.
    """
    frames = np.asarray(native_frames, float)
    contacts = np.asarray(live_contact_frames, float).reshape(-1)
    if (
        not np.isfinite(ending_frame)
        or not np.isfinite(allowance_frames)
        or not 0 <= allowance_frames <= 1
        or frames.ndim != 1
        or not len(frames)
        or not np.isfinite(frames).all()
        or np.any(np.diff(frames) <= 0)
        or not np.isfinite(contacts).all()
    ):
        raise ValueError("ordered finite exposures and a <=1frame passive allowance required")
    ending = float(ending_frame)
    passive = frames[frames > ending + 1e-9]
    span = float(passive[-1] - ending) if len(passive) else 0.0
    inside = contacts[(contacts > ending + 1e-9) & (contacts <= ending + span + 1e-9)]
    beyond = contacts[contacts > ending + span + 1e-9]
    reason = None
    if len(inside):
        reason = "live_contact_inside_passive_span"
    elif span > float(allowance_frames) + 1e-9:
        reason = "passive_span_beyond_annotator_agreement"
    return {
        "schema": "connected_passive_post_ending_context_v1",
        "ending_frame": ending,
        "allowance_frames": float(allowance_frames),
        "passive_exposure_frames": passive.tolist(),
        "passive_exposure_count": int(len(passive)),
        "span_frames": span,
        "live_contact_frames_in_span": inside.tolist(),
        "dead_ball_contact_frames_after_ending": beyond.tolist(),
        "observations_discarded": 0,
        "creates_flight": False,
        "valid": reason is None,
        "reason": reason,
    }
