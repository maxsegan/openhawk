import numpy as np

from cv.pipeline.event_model_v2_features import AutomaticDataset
from cv.pipeline.event_model_v3 import decode_hypotheses


def test_leaky_hypotheses_preserve_multiple_types_without_duplicate_local_peaks() -> None:
    dataset = AutomaticDataset(
        windows=np.zeros((4, 25, 19), dtype=np.float32),
        clips=np.asarray(["match__pt0001"] * 4),
        broadcasts=np.asarray(["match"] * 4),
        frames=np.asarray([10, 12, 20, 30]),
    )
    probabilities = np.asarray(
        [
            [0.1, 0.70, 0.10, 0.10],
            [0.1, 0.60, 0.20, 0.10],
            [0.1, 0.05, 0.80, 0.05],
            [0.1, 0.04, 0.04, 0.30],
        ]
    )

    rows = decode_hypotheses(probabilities, dataset)

    assert [(row["event_type"], row["frame"]) for row in rows] == [
        ("contact", 10.0),
        ("net_hit", 10.0),
        ("bounce", 12.0),
        ("bounce", 20.0),
        ("contact", 20.0),
        ("net_hit", 20.0),
        ("net_hit", 30.0),
    ]
    assert all(row["origin"] == "frozen_event_model_v3_leaky_hypothesis" for row in rows)
