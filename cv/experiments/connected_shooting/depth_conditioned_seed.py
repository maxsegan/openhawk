"""Condition a coarse serve seed on its declared depth, preserving later parameter indices."""

from __future__ import annotations
from dataclasses import replace
import numpy as np
from cv.experiments.connected_shooting import initialization


def condition(scene, initial, bounces, depth, *, bounce_ground_targets=None):
    """Use original incoming training rays; never fitted states or acceptance gates."""
    scene.validate()
    initial = np.asarray(initial, float)
    n = len(scene.pixels)
    if (
        scene.parameterization != "single_shooting"
        or initial.shape not in {(3 + 3 * n,), (3 + 6 * n,), (5 + 3 * n,), (5 + 6 * n,)}
        or not np.isfinite(initial).all()
    ):
        raise ValueError("finite single-shooting seed required")
    receipt = dict(
        status="not_applicable",
        depth_m=float(depth),
        uses_withheld_pixels=False,
        source="original first-flight training observations and declared first ground",
        later_parameter_indices_unchanged=True,
        later_trajectories_may_change=True,
        additional_optimizer_calls=0,
    )
    if (
        not bounces
        or bounces[0] is None
        or not len(bounces[0])
        or (scene.net_hit_frames is not None and len(scene.net_hit_frames[0]))
    ):
        return initial.copy(), receipt | {"reason": "first flight lacks a free incoming ground arc"}
    # Preserve observation partition/radial cameras and dynamics metadata. The
    # gravity initializer itself does not simulate impact/tail dynamics.
    first = replace(
        scene,
        contact_frames=scene.contact_frames[:2].copy(),
        observation_frames=(scene.observation_frames[0],),
        cameras=(scene.cameras[0],),
        pixels=(scene.pixels[0],),
        spin_parameters=scene.spin_parameters[:1],
        camera_distortion=None
        if scene.camera_distortion is None
        else (scene.camera_distortion[0],),
        net_hit_frames=None if scene.net_hit_frames is None else (scene.net_hit_frames[0],),
        bounce_regime_override=(
            scene.bounce_regime_override
            if scene.bounce_regime_override is None or scene.bounce_regime_override[0] == 0
            else None
        ),
        terminal_net_tail=scene.terminal_net_tail if n == 1 else None,
        observed_horizon_tail=scene.observed_horizon_tail if n == 1 else None,
        right_boundary_kind=scene.right_boundary_kind if n == 1 else "supplied_end",
    )
    receipt["first_ground_frame"] = float(bounces[0][0])
    receipt["ground_source"] = "current supplied topology, same as baseline initializer"
    try:
        seed, evidence = initialization.image_ballistic_seed(
            first,
            [float(bounces[0][0])],
            ground_anchor=True,
            consensus=True,
            first_contact_y_m=depth,
            bounce_ground_targets=None
            if bounce_ground_targets is None
            else bounce_ground_targets[:1],
        )
        # Reject a clipped inferred state, rather than silently manufacturing the hypothesis.
        if seed[1] != depth or not evidence["flights"][0]["ground_plane_at_known_bounce"]:
            raise ValueError("first depth or declared ground constraint was not retained")
        if evidence["clipped_parameter_indices"]:
            raise ValueError("conditioned state leaves original numerical bounds")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return initial.copy(), receipt | {"status": "refused_seed_retained", "reason": str(error)}
    out = initial.copy()
    out[:6] = seed[:6]
    return out, receipt | {
        "status": "conditioned",
        "initialization": evidence,
        "changed_parameter_indices": np.flatnonzero(out != initial).tolist(),
    }
