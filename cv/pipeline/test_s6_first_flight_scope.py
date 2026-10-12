"""Source-only partial-flight grammar, barrier ownership and role semantics."""

from copy import deepcopy

import numpy as np
import pytest

from cv.pipeline import s6_first_flight_scope as scope

OP = dict(
    schema="ball_observation_operator_v1", kind="nominal_center", exposure_duration_frames=None
)
SOURCE = dict(
    kind="automatic_point_ledger",
    record=dict(path_base="TENNIS_DATA_ROOT", path="ledger.csv", sha256="a" * 64, bytes=10),
)


def event(kind, frame, **kw):
    return dict(
        event_type=kind,
        frame=round(frame),
        location=dict(frame_subpixel=frame),
        frame_interval=[frame - 0.5, frame + 0.5],
        abstain=False,
        **kw,
    )


def inputs():
    emissions = [event("contact", 4.2), event("bounce", 11.2), event("contact", 18.2)]
    # A piecewise physical ground trajectory projected through a perspective camera.
    rows = []
    P = np.array([[1000.0, 0, 0, 9600], [0, 500, -1000, 5400], [0, 0, 0, 10]])
    for frame in range(1, 25):
        t = (frame - 11.2) / 25
        z = max(0, (-3 if t < 0 else 2) * t - 4.905 * t * t)
        pixel = P @ np.array([2 + t, 12 + 5 * t, z, 1])
        rows.append(
            dict(
                frame=frame, status="visible", x1080=pixel[0] / pixel[2], y1080=pixel[1] / pixel[2]
            )
        )
    return emissions, rows


def test_source_contact_barrier_not_a_fake_ending_or_launch():
    events, rows = inputs()
    result = scope.prepare(events, rows, [1, 24], OP, SOURCE)
    assert result["supported"]
    c = result["contract"]
    assert c["observation_horizon"] == 17
    assert c["physical_ending"] is None and c["first_contact_role"] == "unspecified"
    assert [e["event_type"] for e in result["modeled_events"]] == ["contact", "bounce"]
    assert len(result["original_physical_events"]) == 3
    from cv.experiments.connected_shooting import real_exposure_replay as replay

    labels = {"events": {"records": result["original_physical_events"]}}
    bounded = replay.before_next_physical_event(
        {"postbounce_labeled_frames": list(range(12, 24))}, labels, "pt0001", 11.2, None
    )
    assert bounded["postbounce_labeled_frames"] == list(range(12, 18))
    assert bounded["next_physical_event_boundary"]["original_event"]["frame"] == 18.2


@pytest.mark.parametrize(
    "change",
    ["no_ground", "prior_bounce", "net", "uncertain", "no_rebound", "no_contact", "exposure"],
)
def test_unsupported_physical_prefix_holds(change):
    events, rows = inputs()
    operator = deepcopy(OP)
    if change == "no_ground":
        events.pop(1)
    elif change == "prior_bounce":
        events.insert(0, event("bounce", 2.0))
    elif change == "net":
        events.insert(1, event("net_hit", 8.0))
    elif change == "uncertain":
        events.append(event("contact", 13.0, model_abstain=False) | {"abstain": True})
    elif change == "no_rebound":
        rows = [r for r in rows if r["frame"] <= 11]
    elif change == "no_contact":
        events = events[1:]
    elif change == "exposure":
        operator.update(kind="leading_front", exposure_duration_frames=0.25)
    assert not scope.prepare(events, rows, [1, 24], operator, SOURCE)["supported"]


def test_rejected_toss_catch_is_not_a_physical_barrier_but_remains_in_inventory():
    events, rows = inputs()
    events.insert(0, event("contact", 2.0, model_abstain=True) | {"abstain": True})
    result = scope.prepare(events, rows, [1, 24], OP, SOURCE)
    assert result["supported"] and len(result["original_physical_events"]) == 4
    assert result["original_physical_events"][0]["occurrence_status"] == "absent"


def test_no_toss_or_rally_first_does_not_receive_serve_role():
    events, rows = inputs()
    # There are no precontact ball observations or pose/toss truth inputs.
    rows = [r for r in rows if r["frame"] >= 5]
    result = scope.prepare(events, rows, [1, 24], OP, SOURCE)
    assert result["supported"]
    c = result["contract"]
    attempt = dict(
        events=result["modeled_events"],
        original_physical_events=result["original_physical_events"],
        observation_scope=c,
        ball_observation_operator=OP,
        segmentation_source=SOURCE,
        ending_supplied=False,
        point_end=None,
        owner_end_frame=17,
        owner_end_frame_semantics="observation_horizon_not_physical_event",
    )
    policy = scope.role_policy({"attempts": [attempt]}, "serve")
    assert policy["effective_role"] == "unspecified"
    assert policy["requested_role"] == "serve"
    changed = deepcopy(attempt)
    changed["original_physical_events"][-1]["frame_interval"][0] -= 1
    with pytest.raises(ValueError, match="differs"):
        scope.role_policy({"attempts": [changed]}, "serve")
    assert scope.role_policy({"attempts": [{}]}, "serve") is None
