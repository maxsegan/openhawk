"""Actual center toss rows retain native operators through sparse joint admission."""

from copy import deepcopy

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_prefix_joint_impact as prefix
from cv.experiments.connected_shooting import labeled_toss_front as front
from cv.experiments.connected_shooting import toss_witness as toss


def inputs():
    points = [
        dict(
            frame=f,
            status="visible",
            x1080=100.0 + f,
            y1080=200.0 - f,
            annotation_origin="automatic",
            observation_semantics=toss.AUTOMATIC_CENTER_SEMANTICS,
            support_class="unresolved_or_direct_tracker_observation",
            guide_support=None,
            automatic_sources="tracknet",
            automatic_confidence=None if f == 2 else 0.7,
            automatic_covariance_native_px2=[25.0, 0.0, 16.0],
            automatic_covariance_semantics="innovation_covariance_not_observation_noise",
            uncertainty_radius_px1080=None,
            native_pts_seconds=f / 25.0,
            source_image_sha256="a" * 64,
        )
        for f in range(1, 6)
    ]
    labels = dict(
        attempt=dict(clip="point"), ball=dict(records=[dict(clip="point", frames=points)])
    )
    camera = np.array([[1000, 0, 0, 0], [0, 0, -1000, 3000], [0, 1, 0, 10]], float)
    cameras = dict(
        cameras=[dict(frame=f, status="supported", P=camera.tolist()) for f in range(1, 6)]
    )
    return labels, cameras


def test_centers_keep_native_identity_uncertainty_and_sparse_toss_support():
    labels, cameras = inputs()
    original = deepcopy((labels, cameras))
    observations = toss.precontact_observations(labels, cameras, contact_frame=6.0)
    assert observations["status"] == "abstained"  # Five, below the upstream six-row rule.
    assert observations["labeled_fronts"] == 0
    enriched = front.enrich(observations, labels, "point")
    accepted = prefix._prefix_rows(enriched, 5.5, 0.25, joint_toss_requalification=True)
    assert len(accepted) == 5
    for row, observed in zip(accepted, labels["ball"]["records"][0]["frames"], strict=True):
        assert row["source"] == toss.AUTOMATIC_CENTER_SOURCE
        assert row["incoming_operator"] == front.CENTER
        assert row["axis"] is None
        assert row["uncertainty_px"] == 6.0
        assert row["pixel"] == [observed["x1080"], observed["y1080"]]
        for key in (
            "native_pts_seconds",
            "source_image_sha256",
            "automatic_confidence",
            "automatic_covariance_native_px2",
            "automatic_covariance_semantics",
        ):
            assert row[key] == observed[key]
    # Even monotonic detector motion cannot turn observed centers into fabricated fronts.
    qualified = front.qualify_motion_rows(accepted)
    assert qualified == accepted
    center_predictions = np.array([r["pixel"] for r in accepted])
    args = (qualified, center_predictions, [4, 20, 3], [0, 0, -3], 6.0, 25, 0.25)
    np.testing.assert_array_equal(front.overlay_paired_leading(*args), center_predictions)
    np.testing.assert_array_equal(front.overlay_motion_leading(*args), center_predictions)
    assert (labels, cameras) == original


@pytest.mark.parametrize(
    "changed",
    [
        {"status": "derived_estimate", "support_class": "interpolated_estimate"},
        {"support_class": "guide_observation", "guide_support": {"sources": "tracknet"}},
        {"automatic_sources": "coarse_lock"},
        {"automatic_sources": "interpolated"},
        {"support_class": "unknown"},
        {"automatic_confidence": float("nan")},
        {"observation_semantics": "leading_front"},
    ],
)
def test_unqualified_automatic_rows_cannot_become_toss_witnesses(changed):
    labels, cameras = inputs()
    labels["ball"]["records"][0]["frames"][0].update(changed)
    result = toss.precontact_observations(labels, cameras, contact_frame=6.0)
    assert [row["frame"] for row in result["rows"]] == [2, 3, 4, 5]
    assert result["labeled_fronts"] == 0


def test_sparse_requalification_rechecks_detection_ancestry():
    labels, cameras = inputs()
    enriched = front.enrich(
        toss.precontact_observations(labels, cameras, contact_frame=6), labels, "point"
    )
    enriched["rows"][0]["guide_support"] = {"sources": "interpolated"}
    with pytest.raises(ValueError, match="qualified native detector centers"):
        prefix._prefix_rows(enriched, 5.5, 0.25, joint_toss_requalification=True)


def test_legacy_labeled_precontact_rows_are_exactly_preserved():
    labels, cameras = inputs()
    points = labels["ball"]["records"][0]["frames"]
    for p in points:
        for key in list(p):
            if key not in {"frame", "status", "x1080", "y1080", "uncertainty_radius_px1080"}:
                del p[key]
    rows = toss.precontact_observations(labels, cameras, contact_frame=6)["rows"]
    assert rows == [
        dict(
            frame=p["frame"],
            pixel=[p["x1080"], p["y1080"]],
            uncertainty_px=2.0,
            source="frozen_labeled_front",
            camera=c["P"],
        )
        for p, c in zip(points, cameras["cameras"], strict=True)
    ]


def test_bound_native_guide_detection_survives_but_derived_or_unbound_guide_does_not():
    labels, cameras = inputs()
    point = labels["ball"]["records"][0]["frames"][0]
    point.update(
        support_class="guide_observation",
        automatic_sources="coarse_lock+provenance:coarse",
        guide_support=dict(
            xy=[point["x1080"], point["y1080"]],
            sources="detector_a+detector_b",
            input=dict(path="processed/native_observations.csv", sha256="b" * 64),
        ),
    )
    result = toss.precontact_observations(labels, cameras, contact_frame=6)
    enriched = front.enrich(result, labels, "point")
    accepted = prefix._prefix_rows(enriched, 5.5, 0.25, joint_toss_requalification=True)
    assert [r["frame"] for r in accepted] == [1, 2, 3, 4, 5]
    assert accepted[0]["guide_support"] == point["guide_support"]
    assert accepted[0]["incoming_operator"] == front.CENTER
    original = deepcopy(point)
    for change in (dict(sources="detector_a+interpolated"), dict(xy=[102, 199]), dict(input={})):
        point["guide_support"] = original["guide_support"] | change
        result = toss.precontact_observations(labels, cameras, contact_frame=6)
        assert [r["frame"] for r in result["rows"]] == [2, 3, 4, 5]
