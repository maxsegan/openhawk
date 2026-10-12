import numpy as np
import pytest

from cv.pipeline.event_model_v2_features import AutomaticDataset
from cv.pipeline.event_model_v3 import (
    ABSTAIN_FLOORS,
    NMS_RADIUS,
    PER_TYPE_THRESHOLDS,
    decode,
)


def test_decode_applies_iteration2_thresholds_and_four_frame_nms() -> None:
    probabilities = np.asarray(
        [
            [0.01, PER_TYPE_THRESHOLDS["contact"] + 0.01, 0.02, 0.01],
            [0.01, PER_TYPE_THRESHOLDS["contact"] + 0.02, 0.02, 0.01],
            [0.01, 0.02, PER_TYPE_THRESHOLDS["bounce"] - 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    dataset = AutomaticDataset(
        windows=np.zeros((3, 25, 19), dtype=np.float32),
        clips=np.asarray(["m__pt0001"] * 3),
        broadcasts=np.asarray(["m"] * 3),
        frames=np.asarray([10, 10 + NMS_RADIUS, 20]),
    )
    assert [(row["event_type"], row["frame"]) for row in decode(probabilities, dataset)] == [
        ("contact", float(10 + NMS_RADIUS))
    ]


def test_decode_emits_abstentions_between_the_floor_and_the_threshold() -> None:
    probabilities = np.asarray(
        [
            [0.01, PER_TYPE_THRESHOLDS["contact"] + 0.01, 0.02, 0.01],
            [0.01, 0.30, 0.02, 0.01],
            [0.01, 0.01, 0.02, 0.001],
        ],
        dtype=np.float32,
    )
    dataset = AutomaticDataset(
        windows=np.zeros((3, 25, 18), dtype=np.float32),
        clips=np.asarray(["m__pt0001"] * 3),
        broadcasts=np.asarray(["m"] * 3),
        frames=np.asarray([10, 40, 70]),
        court_geometry_missing=np.asarray([False, True, False]),
        court_coordinate_missing=np.asarray([False, True, True]),
    )

    rows = decode(probabilities, dataset, abstain_floors=ABSTAIN_FLOORS)

    assert [(row["frame"], row["abstain"]) for row in rows] == [(10.0, False), (40.0, True)]
    assert rows[1]["court_geometry_missing"] is True
    assert rows[0]["court_coordinate_missing"] is False
    assert rows[1]["abstain_floor"] == ABSTAIN_FLOORS["contact"]
    assert rows[0]["class_probabilities"]["none"] == pytest.approx(0.01)


def test_decode_without_floors_drops_everything_below_the_threshold() -> None:
    probabilities = np.asarray([[0.01, 0.30, 0.02, 0.01]], dtype=np.float32)
    dataset = AutomaticDataset(
        windows=np.zeros((1, 25, 18), dtype=np.float32),
        clips=np.asarray(["m__pt0001"]),
        broadcasts=np.asarray(["m"]),
        frames=np.asarray([10]),
    )

    assert decode(probabilities, dataset) == []
    assert len(decode(probabilities, dataset, thresholds={"contact": 0.0, "bounce": 0.0, "net_hit": 0.0})) == 1
