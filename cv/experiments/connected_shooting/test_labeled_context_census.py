"""Post-ending racket taps must not become evidence for an uninterrupted rebound."""

from copy import deepcopy

from cv.experiments.connected_shooting import labeled_context_census as census


def inventory():
    return dict(
        status="supported",
        postbounce_labeled_frames=list(range(363, 374)),
        postbounce_labeled_frame_count=11,
        last_postbounce_labeled_frame=373,
    )


def test_original_passive_pickup_bounds_context_without_relabeling():
    labels = dict(
        events=dict(
            records=[
                dict(
                    event_type="contact",
                    frame=366.5,
                    frame_interval=[366, 367],
                    phase="post_ending_padding",
                    clip="point",
                    status="labeled",
                )
            ]
        )
    )
    original = deepcopy(labels)
    result = census.before_next_contact(inventory(), labels, "point", 362.5, 0.25)
    assert result["postbounce_labeled_frames"] == [363, 364, 365]
    assert result["next_contact_boundary"]["excluded_frames"] == list(range(366, 374))
    assert result["last_postbounce_labeled_frame"] == 365
    assert labels == original


def test_exposure_crossing_uncertain_contact_is_not_rebound_support():
    labels = dict(
        events=dict(
            records=[
                dict(
                    event_type="contact",
                    frame=364,
                    frame_interval=[363.2, 364.2],
                    status="ambiguous",
                )
            ]
        )
    )
    result = census.before_next_contact(inventory(), labels, "point", 362.5, 0.25)
    assert result["status"] == "no_rebound_exposure_before_next_contact"
    assert result["postbounce_labeled_frames"] == []


def test_other_clip_or_rejected_contact_cannot_truncate_evidence():
    labels = dict(
        events=dict(
            records=[
                dict(event_type="contact", frame=364, clip="other"),
                dict(event_type="contact", frame=364, status="rejected"),
                dict(event_type="contact", frame=341.5),
            ]
        )
    )
    original = inventory()
    assert census.before_next_contact(original, labels, "point", 362.5, 0.25) == original


def test_next_passive_bounce_is_separate_topology_not_first_rebound_evidence():
    labels = dict(
        events=dict(
            records=[
                dict(
                    event_type="bounce",
                    frame=224.5,
                    frame_interval=[224, 225],
                    phase="post_ending_padding",
                    clip="point",
                )
            ]
        )
    )
    original = deepcopy(labels)
    receipt = dict(
        status="supported",
        postbounce_labeled_frames=list(range(219, 228)),
        postbounce_labeled_frame_count=9,
        last_postbounce_labeled_frame=227,
    )
    result = census.before_next_physical_event(receipt, labels, "point", 218.5, 0.25)
    assert result["postbounce_labeled_frames"] == list(range(219, 224))
    assert result["next_physical_event_boundary"]["original_event"]["event_type"] == "bounce"
    assert result["next_physical_event_boundary"]["excluded_frames"] == [224, 225, 226, 227]
    assert labels == original
    assert receipt["postbounce_labeled_frame_count"] == 9


def test_next_net_impact_bounds_context_before_its_original_interval():
    labels = dict(
        events=dict(
            records=[
                dict(event_type="net_hit", frame=986.5, frame_interval=[986, 987], clip="point"),
                dict(event_type="contact", frame=990),
            ]
        )
    )
    receipt = dict(
        status="supported",
        postbounce_labeled_frames=[984, 985, 986, 987],
        postbounce_labeled_frame_count=4,
        last_postbounce_labeled_frame=987,
    )
    result = census.before_next_physical_event(receipt, labels, "point", 983.5, 0.25)
    assert result["postbounce_labeled_frames"] == [984, 985]
    assert result["next_physical_event_boundary"]["original_event"]["event_type"] == "net_hit"


def test_no_later_physical_event_preserves_inventory_exactly():
    original = inventory()
    labels = dict(
        events=dict(
            records=[
                dict(event_type="ending", frame=370),
                dict(event_type="bounce", frame=361),
                dict(event_type="net_hit", frame=369, status="rejected"),
            ]
        )
    )
    assert census.before_next_physical_event(original, labels, "point", 362.5, 0.25) is original
