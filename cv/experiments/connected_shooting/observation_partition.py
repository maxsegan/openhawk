"""Explicit fitting/check ownership for supplied native observations.

All-native is a fitting policy, not an independent validation claim. Source
witness ownership is declared separately so older reports replay unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

POLICIES = ("fifth_frame_withheld", "all_native")
DEFAULT = POLICIES[0]
LEGACY_WITNESS = "legacy_fit_frames_including_existing_terminal_activation"
NATIVE_WITNESS = "all_native_fit_frames_v1"


def validate(value: str) -> str:
    if value not in POLICIES:
        raise ValueError("explicit supported observation partition required")
    return value


def all_native(scene) -> bool:
    return validate(getattr(scene, "observation_partition", DEFAULT)) == "all_native"


def check_split(scene) -> str:
    return "check_in_sample" if all_native(scene) else "withheld"


def legacy_fit_frames(scene, index: int, terminal_rebound_frames=None):
    frames = np.asarray(scene.observation_frames[index], float)
    if not all_native(scene):
        return frames
    keep = frames % 5 != 0
    if index == len(scene.pixels) - 1 and terminal_rebound_frames is not None:
        keep |= np.isin(frames, terminal_rebound_frames)
    return frames[keep]


def witness_policy(partition: str) -> str:
    """New search policy; old report replay must pass its recorded policy instead."""
    return NATIVE_WITNESS if validate(partition) == "all_native" else LEGACY_WITNESS


def validate_witness_policy(policy: str, partition: str) -> str:
    if policy not in (LEGACY_WITNESS, NATIVE_WITNESS):
        raise ValueError("explicit source bounce witness observation policy required")
    if policy == NATIVE_WITNESS and validate(partition) != "all_native":
        raise ValueError("native bounce witnesses require all-native fitting")
    return policy


def fitted_witness_policy(scene):
    return validate_witness_policy(
        scene.bounce_witness_observation_policy, scene.observation_partition
    )


def witness_frames(scene, index: int, terminal_rebound_frames=None):
    """Original fitted native rows only; never fill a missing camera or ball."""
    policy = validate_witness_policy(
        scene.bounce_witness_observation_policy, scene.observation_partition
    )
    if policy == NATIVE_WITNESS:
        return np.asarray(scene.observation_frames[index], float)
    return legacy_fit_frames(scene, index, terminal_rebound_frames)


def receipt(scene, check_scene) -> dict:
    rows = []
    for index, (fitted, checked) in enumerate(
        zip(scene.observation_frames, check_scene.observation_frames, strict=True)
    ):
        overlap = np.isin(checked, fitted)
        if all_native(scene):
            for check_index, frame in enumerate(checked):
                matches = np.flatnonzero(np.asarray(fitted) == frame)
                if len(matches) != 1:
                    raise ValueError("all-native mode has an unfitted check observation")
                for field in ("pixels", "cameras", "camera_distortion"):
                    left, right = getattr(scene, field), getattr(check_scene, field)
                    if left is None and right is None:
                        continue
                    if (
                        left is None
                        or right is None
                        or not np.array_equal(left[index][matches[0]], right[index][check_index])
                    ):
                        raise ValueError(
                            "all-native check copy differs from the fitted native observation"
                        )
        rows.append(
            {
                "flight_index": index,
                "fitted_pictures": len(fitted),
                "check_pictures": len(checked),
                "check_pictures_used_in_fit": int(np.sum(overlap)),
                "independent_check_pictures": int(np.sum(~overlap)),
                "original_fifth_frame_pictures": int(np.sum(np.asarray(fitted) % 5 == 0)),
                **(
                    {
                        "source_bounce_witness_eligible_native_frames": witness_frames(
                            scene, index
                        ).tolist()
                    }
                    if fitted_witness_policy(scene) == NATIVE_WITNESS
                    else {}
                ),
            }
        )
    if all_native(scene) and any(row["independent_check_pictures"] for row in rows):
        raise ValueError("all-native mode has an unfitted check observation")
    return {
        "policy": getattr(scene, "observation_partition", DEFAULT),
        "check_split": check_split(scene),
        "independent_check_pictures": sum(row["independent_check_pictures"] for row in rows),
        "flights": rows,
        "bounce_witness_observation_policy": fitted_witness_policy(scene),
        "image_gate_population": "legacy_fit_rows_in_sample"
        if all_native(scene)
        else "legacy_fit_rows",
        "expanded_native_residuals": "diagnostic_only" if all_native(scene) else None,
        "numeric_gate_thresholds_changed": False,
        "event_pose_spatial_checks": "shared input/model consistency, not independent validation",
    }


def legacy_evaluation(context, *, exposure_duration=0.25):
    """Rebuild the historical pixel population and axes from original inputs.

    Physical state/contact epochs stay current. Fifth-frame terminal exposures
    already active in the historical procedure remain active. The result is an
    evaluation sampling view, never a claim that these fitted rows are withheld.
    """
    from cv.experiments.connected_shooting import (
        agent_whole_point_search as whole,
        real_exposure_replay as exposure,
    )

    scene = context["scene"]
    if not all_native(scene):
        return scene, context["heldout"], context["axes"]
    masks = [np.asarray(frames) % 5 != 0 for frames in scene.observation_frames]
    fields = {}
    for name in ("observation_frames", "cameras", "pixels", "camera_distortion"):
        groups = getattr(scene, name)
        fields[name] = (
            None
            if groups is None
            else tuple(np.asarray(group)[mask] for group, mask in zip(groups, masks, strict=True))
        )
    evaluated = replace(
        scene,
        **fields,
        observation_partition=DEFAULT,
        bounce_witness_observation_policy=LEGACY_WITNESS,
    )
    evaluated.validate()
    checked = replace(
        context["heldout"],
        observation_partition=DEFAULT,
        bounce_witness_observation_policy=LEGACY_WITNESS,
    )
    reference = {"records": [{"frames": context["attempt"]["owner_ball_labels"]}]}
    axes, _ = whole.training_directions(
        evaluated,
        context["bounces"],
        reference,
        observation_fallback=True,
        fallback_receipt=[],
        exposure_duration=exposure_duration,
    )
    if context.get("terminal_rebound_frames") is not None:
        evaluated, axes, frames, _ = exposure.terminal_rebound_segment(
            evaluated, checked, context["bounces"], axes
        )
        if not np.array_equal(frames, context["terminal_rebound_frames"]):
            raise ValueError("fixed-frame evaluation changed terminal observation ownership")
    return evaluated, checked, axes


def mark_fitted_evaluation(measured, fitted_scene, check_scene):
    """Label legacy evaluation pictures as in-sample under all-native fitting."""
    if not all_native(fitted_scene):
        return measured
    result = dict(measured)
    result["native_projection"] = [
        dict(row, split="check_in_sample") if row["split"] == "withheld" else row
        for row in measured["native_projection"]
    ]
    if measured.get("unscorable_native_projection"):
        result["unscorable_native_projection"] = [
            dict(row, split="check_in_sample") if row["split"] == "withheld" else row
            for row in measured["unscorable_native_projection"]
        ]
    result["rms_px"] = {
        **measured["rms_px"],
        "check_in_sample": measured["rms_px"].get("withheld"),
        "withheld": None,
    }
    result["observation_partition"] = receipt(fitted_scene, check_scene)
    return result
