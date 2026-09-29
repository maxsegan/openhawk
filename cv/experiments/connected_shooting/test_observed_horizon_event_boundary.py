"""Actual automatic source4/51 event clipping, through production Scene assembly."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from cv.experiments.connected_shooting import agent_whole_point_search as search


@pytest.fixture(params=[0, 1], ids=["source4", "source51"])
def source(request):
    path = Path(__file__).with_name("fixtures") / "observed_horizon_event_boundary.json"
    return json.loads(path.read_text())["cases"][request.param]


def test_qualified_ground_after_original_net_reaches_actual_scene(source):
    attempt = source["attempt"]
    original = deepcopy(source)
    hypothesis = source["hypothesis"]
    end = source["requested_end_frame"]
    # This is the exact former filter: search claimed a bounce hypothesis but
    # fitted its no-addition event stream. Strict replay correctly rejected it.
    old_events = [row for row in hypothesis["events"] if row["frame"] <= end]
    assert old_events == source["failed_fitted_events"]
    assert old_events != hypothesis["events"]
    events, scope = search.scoped_branch_events(
        hypothesis["events"], end, horizon_contract=attempt["observed_horizon_tail"]
    )
    assert events == hypothesis["events"]
    assert scope["excluded_events"] == []
    assert (
        scope["modeled_branch_end_frame"] == attempt["observed_horizon_tail"]["observation_horizon"]
    )

    def scene(rows):
        return search.prepare_attempt(
            attempt,
            source["cameras"],
            source["surface"],
            rows,
            scope["modeled_branch_end_frame"],
            observation_fallback=True,
            ground_settling=True,
        )

    before, held_before, grounds_before, native_before, _ = scene(old_events)
    after, held_after, grounds_after, native_after, _ = scene(events)
    assert after.observed_horizon_tail["supplied_ground_count"] == 1
    assert before.observed_horizon_tail["supplied_ground_count"] == 0
    assert grounds_before[-1].size == 0
    assert grounds_after[-1].tolist() == [hypothesis["added"][0]["frame"]]
    assert len(before.pixels) == len(after.pixels)
    for left, right in ((before, after), (held_before, held_after)):
        for name in ("pixels", "cameras", "observation_frames"):
            assert all(
                np.array_equal(a, b) for a, b in zip(getattr(left, name), getattr(right, name))
            )
    assert all(np.array_equal(a, b) for a, b in zip(native_before, native_after))
    assert source == original


@pytest.mark.parametrize("ending_kind", ["net_hit", "bounce", "contact"])
def test_physical_and_present_net_event_boundaries_are_unchanged(ending_kind):
    events = [
        dict(event_type="contact", frame=1),
        dict(event_type=ending_kind, frame=10),
        dict(event_type="bounce", frame=14),
    ]
    before = deepcopy(events)
    retained, scope = search.scoped_branch_events(events, 10)
    assert retained == events[:2]
    assert scope == dict(
        modeled_branch_end_frame=10.0,
        excluded_events=[events[-1]],
        rule="event representative at or before declared modeled branch horizon",
        original_event_records_preserved=True,
    )
    assert events == before


def test_unknown_horizon_does_not_admit_later_context_events(source):
    horizon = source["attempt"]["observed_horizon_tail"]
    outside = dict(event_type="bounce", frame=horizon["observation_horizon"] + 1)
    events = [*source["hypothesis"]["events"], outside]
    retained, scope = search.scoped_branch_events(
        events, source["requested_end_frame"], horizon_contract=horizon
    )
    assert retained == source["hypothesis"]["events"]
    assert scope["excluded_events"] == [outside]
