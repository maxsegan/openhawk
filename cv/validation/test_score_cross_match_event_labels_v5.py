import csv

import pytest

from cv.validation.score_cross_match_event_labels_v5 import (
    load_truth,
    match_events,
    scope_predictions,
)


FIELDS = [
    "clip",
    "event_type",
    "seed_id",
    "seed_frame",
    "labeled_frame",
    "labeled_x540",
    "labeled_y540",
    "verdict",
    "note",
]


def test_truth_attempt_spans_exclude_dead_time(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    rows = [
        ["clip", "contact", "c1", "", "10", "1", "1", "new", ""],
        ["clip", "bounce", "b1", "", "20", "1", "1", "new", ""],
        ["clip", "point_end", "e1", "", "20", "", "", "terminal_bounce", ""],
        ["clip", "contact", "c2", "", "40", "1", "1", "new", ""],
        ["clip", "bounce", "b2", "", "50", "1", "1", "new", ""],
        ["clip", "point_end", "e2", "", "50", "", "", "terminal_bounce", ""],
        ["clip", "coverage", "coverage", "", "", "", "", "complete", ""],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)

    truth, spans, _ = load_truth(path)

    assert len(truth) == 4
    assert spans["clip"] == [(10.0, 20.0), (40.0, 50.0)]


def test_truth_outside_spans_requires_explicit_diagnostic_opt_in(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    rows = [
        ["clip", "contact", "outside", "", "5", "1", "1", "new", ""],
        ["clip", "point_end", "end", "", "2", "", "", "terminal_bounce", ""],
        ["clip", "coverage", "coverage", "", "", "", "", "complete", ""],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)

    with pytest.raises(ValueError, match="truth outside point-ending spans"):
        load_truth(path)
    truth, spans, metadata = load_truth(path, allow_truth_outside_spans=True)

    assert len(truth) == 1
    assert spans["clip"] == []
    assert metadata["truth_outside_point_end_spans"] == [
        {"clip": "clip", "event_type": "contact", "frame": 5.0}
    ]


def test_subframe_matching_uses_declared_tolerance() -> None:
    truth = [{"clip": "clip", "event_type": "contact", "frame": 66.5}]
    predictions = [{"clip": "clip", "event_type": "contact", "frame": 67.0}]

    matches, false_positives, false_negatives = match_events(predictions, truth, 0.5)

    assert len(matches) == 1
    assert not false_positives
    assert not false_negatives


def test_scope_predictions_can_keep_tolerance_around_attempt_boundary() -> None:
    predictions = [
        {"clip": "clip", "event_type": "bounce", "frame": 9.5},
        {"clip": "clip", "event_type": "bounce", "frame": 20.5},
        {"clip": "clip", "event_type": "bounce", "frame": 22.0},
    ]

    scoped = scope_predictions(
        predictions,
        {"clip": [(10.0, 20.0)]},
        boundary_margin_frames=1.0,
    )

    assert [row["frame"] for row in scoped] == [9.5, 20.5]
