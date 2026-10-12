import csv
import math

import numpy as np
import pytest

from cv.pipeline.attempt_observation_scope import structural_horizon
from cv.pipeline.point_ledger import build_scored_ledger, write_ledger
from cv.pipeline.serve_detector import assemble_attempts, write_attempts


def horizon(**changes):
    values = dict(
        following=30.0,
        score_end=25.0,
        maximum_duration=45.0,
        shots=[
            dict(shot_index=4, t_start=0, t_end=10.3),
            dict(shot_index=5, t_start=10.3, t_end=22),
        ],
        play_probabilities={5: 0.99},
        onsets=[10.5],
    )
    return structural_horizon(10.0, **(values | changes))


def test_imminent_main_camera_transition_needs_both_audio_and_view_evidence():
    result = horizon()
    assert result == dict(
        end=22, reason="camera_boundary", camera_transition=10.3, audio_anchor=10.5
    )
    assert horizon(onsets=[])["end"] == 10.3
    assert horizon(play_probabilities={})["end"] == 10.3
    assert horizon(play_probabilities={5: 0.8})["end"] == 10.3
    assert horizon(onsets=[11.5])["end"] == 10.3
    # An old-shot transient cannot suppress an admissible next-shot witness.
    assert horizon(onsets=[10.1, 10.5])["end"] == 22
    assert horizon(onsets=[10.1])["end"] == 10.3
    assert horizon(onsets=[10.1, float("nan"), 22])["end"] == 10.3
    assert horizon(onsets=[10.1, 10.5], play_probabilities={5: 0.8})["end"] == 10.3


def test_context_does_not_bridge_missing_source_or_multiple_cuts():
    assert horizon(shots=[dict(t_start=0, t_end=10.3), dict(t_start=10.4, t_end=22)])["end"] == 10.3
    shots = [
        dict(t_start=0, t_end=10.3),
        dict(t_start=10.3, t_end=10.4),
        dict(t_start=10.4, t_end=22),
    ]
    assert horizon(shots=shots, play_probabilities={1: 1, 2: 1})["end"] == 10.3


@pytest.mark.parametrize(
    "changes,end,reason",
    [
        ({"following": 12}, 12, "next_detected_serve"),
        ({"score_end": 13}, 13, "score_transition"),
        ({"maximum_duration": 4}, 14, "duration_cap"),
    ],
)
def test_context_keeps_explicit_structural_ceilings(changes, end, reason):
    result = horizon(**changes)
    assert result["end"] == end
    assert result["reason"] == reason


def assemble(scope, *, shot_end=20):
    times = np.arange(0, 30.01, 0.2)
    starts = np.zeros(len(times))
    starts[50] = 0.95
    ends = np.zeros(len(times))
    ends[60] = 0.8
    return assemble_attempts(
        times,
        starts,
        ends,
        start_threshold=0.7,
        end_threshold=0.1,
        shots=[dict(t_start=0, t_end=shot_end)],
        score_changes=np.empty(0),
        play_mask=np.ones(len(times), dtype=bool),
        gap_fill_seconds=0,
        observation_scope=scope,
    )


def test_false_early_end_remains_evidence_without_truncating_context():
    original = assemble("predicted")
    expanded = assemble("structural_context")
    assert original == [dict(rally_t_start=10, rally_t_end=12, start_score=0.95, end_score=0.8)]
    assert expanded[0]["rally_t_start"] == original[0]["rally_t_start"]
    assert expanded[0]["predicted_rally_t_end"] == 12
    assert expanded[0]["rally_t_end"] == 20
    assert expanded[0]["observation_context_status"] == "context_only_physical_ending_unresolved"


def test_insufficient_context_stays_in_denominator_without_padding_across_cut():
    assert assemble("predicted", shot_end=10.3) == []
    rows = assemble("structural_context", shot_end=10.3)
    assert len(rows) == 1
    assert rows[0]["rally_t_end"] == 10.3
    assert rows[0]["predicted_rally_t_end"] is None
    assert rows[0]["observation_context_status"] == "insufficient_context"


def test_context_semantics_survive_attempt_and_ledger_exports(tmp_path):
    attempts = assemble("structural_context")
    path = tmp_path / "attempts.csv"
    write_attempts(path, attempts)
    loaded = list(csv.DictReader(path.open()))
    rows = build_scored_ledger(loaded, [], [], peak_z=5)
    output = tmp_path / "ledger.csv"
    write_ledger(output, rows)
    final = list(csv.DictReader(output.open()))
    for key in (
        "observation_scope",
        "predicted_rally_t_end",
        "observation_horizon_reason",
        "observation_context_status",
    ):
        assert final[0][key] == loaded[0][key]
    assert math.isclose(float(final[0]["rally_t_end"]), 20)


def test_start_in_unobserved_camera_gap_does_not_borrow_a_future_shot():
    result = horizon(shots=[dict(t_start=11, t_end=22)])
    assert result["end"] == 10
    assert result["reason"] == "missing_containing_camera"


def test_next_shot_witness_is_order_independent_and_keeps_context_semantics():
    for onsets in ([10.1, 10.7, 10.5], [10.5, 10.1, 10.7]):
        assert horizon(onsets=onsets)["audio_anchor"] == 10.5
    # No view classifier guarantee is invented: an admitted scope remains
    # source context, and its actual view/physical events need downstream evidence.
    assert assemble("structural_context")[0]["observation_context_status"] == (
        "context_only_physical_ending_unresolved"
    )
