"""Real AO fixture: five declared rebound checks are not contact ownership changes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cv.pipeline import s6_optional_contacts as optional


@pytest.fixture
def source():
    return json.loads(
        (Path(__file__).parent / "testdata/optional_rebound_ownership_source02.json").read_text()
    )


def signature(source):
    return optional.training_observation_signature(
        source["attempt"],
        source["cameras"],
        "fifth_frame_withheld",
        source["rebound_inventory"],
        source["rebound_receipt"],
    )


def test_original_real_source_has_exact_five_approved_rebound_checks(source):
    original = deepcopy(source)
    ordinary = optional.observation_signature(
        source["attempt"], source["cameras"], "fifth_frame_withheld"
    )
    expected = signature(source)
    assert len(ordinary) == 202 and len(expected) == 207
    assert sorted({r[0] for r in expected} - {r[0] for r in ordinary}) == [355, 360, 365, 370, 375]
    assert [r[0] for r in expected] == sorted(source["actual_training_frames"])
    assert source == original


@pytest.mark.parametrize(
    "change",
    [
        "contact_check",
        "unobserved",
        "held_camera",
        "missing_rebound",
        "empty_rebound",
        "wrong_bounce",
        "duplicate",
    ],
)
def test_unapproved_ownership_changes_fail_closed(source, change):
    receipt = source["rebound_receipt"]
    if change == "contact_check":
        receipt["withheld_rows_activated_frames"][0] = 230
    elif change == "unobserved":
        source["attempt"]["owner_ball_labels"] = [
            r for r in source["attempt"]["owner_ball_labels"] if r["frame"] != 355
        ]
    elif change == "held_camera":
        next(r for r in source["cameras"]["cameras"] if r["frame"] == 355)["status"] = "held"
    elif change == "missing_rebound":
        receipt["withheld_rows_activated_frames"].pop()
        receipt["withheld_rows_activated_count"] -= 1
    elif change == "empty_rebound":
        receipt["withheld_rows_activated_frames"] = []
        receipt["withheld_rows_activated_count"] = 0
    elif change == "wrong_bounce":
        receipt["supplied_terminal_bounce_frame"] -= 1
    elif change == "duplicate":
        receipt["withheld_rows_activated_frames"][0] = 360
    with pytest.raises(ValueError):
        signature(source)


@pytest.mark.parametrize("change", ["pixel", "camera", "epoch", "image"])
def test_activated_rows_keep_source_identity_for_cross_branch_comparison(source, change):
    original = signature(source)
    row = next(r for r in source["attempt"]["owner_ball_labels"] if r["frame"] == 355)
    if change == "pixel":
        row["x1080"] += 1
    elif change == "camera":
        next(r for r in source["cameras"]["cameras"] if r["frame"] == 355)["P"][0][0] += 1
    elif change == "epoch":
        row["native_pts_seconds"] += 0.001
    else:
        row["source_image_sha256"] = "different native exposure"
    assert signature(source) != original


def test_ordinary_and_preserved_partitions_remain_unchanged(source):
    old = optional.observation_signature(
        source["attempt"], source["cameras"], "fifth_frame_withheld"
    )
    assert (
        optional.training_observation_signature(
            source["attempt"],
            source["cameras"],
            "fifth_frame_withheld",
            {"mode": "off"},
            {"mode": "off"},
        )
        == old
    )
    source["rebound_receipt"].update(
        original_partition_preserved=True,
        withheld_rows_activated_count=0,
        withheld_rows_activated_frames=[],
    )
    assert signature(source) == old
    all_native = optional.observation_signature(source["attempt"], source["cameras"], "all_native")
    assert (
        optional.training_observation_signature(
            source["attempt"],
            source["cameras"],
            "all_native",
            source["rebound_inventory"],
            source["rebound_receipt"],
        )
        == all_native
    )
