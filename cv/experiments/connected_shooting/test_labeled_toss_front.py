"""Native paired-front geometry and immutable original observation binding."""

from copy import deepcopy

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_toss_front as front
from cv.experiments.connected_shooting import camera_geometry


def test_ascent_and_descent_fronts_follow_opposite_observed_axes():
    camera = np.array([[1000, 0, 0, 0], [0, 0, -1000, 3000], [0, 1, 0, 10]], float)
    position = np.array([4, 20, 3], float)
    center = camera_geometry.project(camera[None], position[None])
    for velocity_z, axis, sign in [(4.0, [0, -1], -1), (-4.0, [0, 1], 1)]:
        rows = [
            dict(
                frame=100,
                camera=camera.tolist(),
                axis=axis,
                incoming_operator=front.PAIRED_SWEPT_FRONT,
            )
        ]
        result = front.overlay_paired_leading(
            rows, center, position, [0, 0, velocity_z], 100, 25, 0.25
        )
        # At least the declared optical extent plus projected sphere radius;
        # the finite native exposure carries the front farther in either wing.
        assert sign * (result[0, 1] - center[0, 1]) > 2.5
        assert abs(result[0, 0] - center[0, 0]) < 0.1


def test_unresolved_apex_keeps_exact_native_proxy_and_uncertainty():
    row = dict(
        frame=50,
        axis=None,
        incoming_operator=front.UNRESOLVED_AXIS_CENTER_PROXY,
        uncertainty_px=5.0,
    )
    original = deepcopy(row)
    center = np.array([[102.125, 84.875]])
    result = front.overlay_paired_leading([row], center, [1, 2, 3], [0, 0, -3], 60, 25, 0.25)
    np.testing.assert_array_equal(result, center)
    assert row == original
    assert not np.shares_memory(result, center)


def test_original_pair_enrichment_is_immutable_and_checks_front_identity():
    observations = dict(clip="point", rows=[dict(frame=35, pixel=[100.0, 200.0], uncertainty_px=3)])
    labels = dict(
        ball=dict(
            records=[
                dict(
                    clip="point",
                    frames=[
                        dict(
                            frame=35,
                            status="visible",
                            x1080=100.0,
                            y1080=200.0,
                            streak=dict(
                                status="paired",
                                leading=dict(x1080=100.0, y1080=200.0),
                                trailing=dict(x1080=100.0, y1080=208.0),
                            ),
                        )
                    ],
                )
            ]
        )
    )
    before = deepcopy((observations, labels))
    result = front.enrich(observations, labels, "point")
    assert result["rows"][0]["axis"] == [0.0, -8.0]
    assert (observations, labels) == before
    changed = deepcopy(observations)
    changed["rows"][0]["pixel"][0] += 1
    with pytest.raises(ValueError, match="leading coordinates"):
        front.enrich(changed, labels, "point")
    labels["ball"]["records"][0]["frames"][0]["streak"]["status"] = "ambiguous"
    with pytest.raises(ValueError, match="paired support"):
        front.enrich(observations, labels, "point")


def _missing_rows(pixels):
    return [
        dict(
            frame=i,
            pixel=p,
            uncertainty_px=5.0,
            axis=None,
            incoming_operator=front.UNRESOLVED_AXIS_CENTER_PROXY,
        )
        for i, p in enumerate(pixels)
    ]


def test_motion_qualification_is_input_only_preserves_apex_pair_and_rows():
    rows = _missing_rows([[0, 4], [0, 2], [0, 3]])
    rows[0].update(incoming_operator=front.PAIRED_SWEPT_FRONT, axis=[0, -2])
    before = deepcopy(rows)
    qualified = front.qualify_motion_rows(rows)
    assert qualified[0] == rows[0]
    assert qualified[1]["incoming_operator"] == front.UNRESOLVED_AXIS_CENTER_PROXY
    assert qualified[1]["motion_qualification"]["status"] == "unresolved_input_turn"
    assert rows == before
    moving = _missing_rows([[0, 0], [0, 2], [0, 4]])
    qualified = front.qualify_motion_rows(moving)
    assert len(qualified) == 3
    assert all(r["incoming_operator"] == front.CANDIDATE_MOTION_SWEPT_FRONT for r in qualified)
    for old, new in zip(moving, qualified, strict=True):
        for key in ["frame", "pixel", "uncertainty_px", "axis"]:
            assert old[key] == new[key]


def test_candidate_chord_front_matches_known_axis_in_both_wings_and_subpixel_motion():
    camera = np.array([[1000, 0, 0, 0], [0, 0, -1000, 3000], [0, 1, 0, 10]], float)
    position = np.array([4, 20, 3], float)
    row = dict(frame=100, camera=camera.tolist())
    for vz in [4.0, -4.0, 0.10]:
        curve = front.ballistic_curve(position, [0, 0, vz], 100, 25, 100, 0.25)
        uv = camera_geometry.project(np.repeat(camera[None], 2, axis=0), curve[[0, -1]])
        chord = uv[-1] - uv[0]
        if vz == 0.10:
            assert 1e-9 < np.linalg.norm(chord) < 1
        paired = front.paired_leading_front(
            dict(row, axis=chord), position, [0, 0, vz], 100, 25, 0.25
        )
        predicted = front.motion_leading_front(row, position, [0, 0, vz], 100, 25, 0.25)
        np.testing.assert_array_equal(predicted, paired)
    # The exposure straddles the physical apex symmetrically: no silent center fallback.
    with pytest.raises(ValueError, match="zero candidate exposure chord"):
        front.motion_leading_front(row, position, [0, 0, 9.81 * 0.005], 100, 25, 0.25)


def test_motion_overlay_preserves_original_paired_and_unresolved_prediction():
    camera = np.array([[1000, 0, 0, 0], [0, 0, -1000, 3000], [0, 1, 0, 10]], float)
    rows = [
        dict(
            frame=100,
            camera=camera.tolist(),
            axis=[0, 1],
            incoming_operator=front.PAIRED_SWEPT_FRONT,
        ),
        dict(frame=101, axis=None, incoming_operator=front.UNRESOLVED_AXIS_CENTER_PROXY),
    ]
    centers = np.array([[1.0, 2.0], [3.0, 4.0]])
    old = deepcopy(rows)
    paired = front.overlay_paired_leading(rows, centers, [4, 20, 3], [0, 0, -4], 100, 25, 0.25)
    mixed = front.overlay_motion_leading(rows, centers, [4, 20, 3], [0, 0, -4], 100, 25, 0.25)
    np.testing.assert_array_equal(paired, mixed)
    assert rows == old
    assert not np.shares_memory(centers, mixed)
