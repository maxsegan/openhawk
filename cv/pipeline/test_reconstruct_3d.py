import json

import numpy as np
import pytest

from cv.pipeline.reconstruct_3d import json_default, joint_refine_enabled, parser


def test_json_default_normalizes_only_numpy_integers() -> None:
    assert json.dumps({"source_frames": [np.int64(10)]}, default=json_default) == (
        '{"source_frames": [10]}'
    )
    with pytest.raises(TypeError, match="ndarray"):
        json.dumps({"unexpected": np.array([1])}, default=json_default)


def test_whole_point_branches_enable_joint_refinement() -> None:
    assert joint_refine_enabled(requested=False, whole_point_branches=True)
    assert not joint_refine_enabled(requested=False, whole_point_branches=False)


def test_event_topology_branches_enable_joint_refinement() -> None:
    assert joint_refine_enabled(
        requested=False,
        whole_point_branches=False,
        event_topology_branches=True,
    )


def test_joint_rally_refinement_is_an_explicit_s6_toggle() -> None:
    args = parser().parse_args(
        [
            "--audit-root",
            "root",
            "--manifest",
            "manifest.json",
            "--event-boundaries",
            "events.json",
            "--output",
            "reconstruction.json",
            "--joint-rally-refinement",
        ]
    )

    assert args.joint_rally_refinement
    assert not args.event_topology_branches
    assert args.math_threads == 1
    assert args.point_timeout_seconds == 600.0


def test_anchor_output_is_an_explicit_path() -> None:
    args = parser().parse_args(
        [
            "--audit-root",
            "root",
            "--manifest",
            "manifest.json",
            "--event-boundaries",
            "events.json",
            "--output",
            "reconstruction.json",
            "--anchors-output-root",
            "anchors",
        ]
    )

    assert str(args.anchors_output_root) == "anchors"


def test_anchor_first_is_default_with_contact_first_escape_hatch() -> None:
    base = [
        "--audit-root",
        "root",
        "--manifest",
        "manifest.json",
        "--event-boundaries",
        "events.json",
        "--output",
        "reconstruction.json",
    ]

    assert parser().parse_args(base).anchor_first
    assert parser().parse_args([*base, "--anchor-first"]).anchor_first
    assert not parser().parse_args([*base, "--contact-first"]).anchor_first
    assert not parser().parse_args(base).legacy_height_on_ray
    assert parser().parse_args([*base, "--legacy-height-on-ray"]).legacy_height_on_ray
    assert parser().parse_args(base).net_point_anchor
    assert not parser().parse_args([*base, "--net-plane-constraint"]).net_point_anchor
    assert parser().parse_args([*base, "--net-anchor-point"]).net_point_anchor
    assert parser().parse_args(base).shared_contact_fit
    assert parser().parse_args([*base, "--shared-contact-fit"]).shared_contact_fit
    assert not parser().parse_args([*base, "--independent-contacts"]).shared_contact_fit
    assert not parser().parse_args(base).match_shared_priors
    assert parser().parse_args([*base, "--match-shared-priors"]).match_shared_priors
    assert not parser().parse_args(base).downweight_contact_adjacent
    assert parser().parse_args([*base, "--downweight-contact-adjacent"]).downweight_contact_adjacent
    assert (
        not parser()
        .parse_args([*base, "--full-contact-adjacent-weight"])
        .downweight_contact_adjacent
    )
