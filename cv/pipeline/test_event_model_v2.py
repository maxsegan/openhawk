import numpy as np

from cv.pipeline.event_model_v2 import PER_TYPE_THRESHOLDS, decode
from cv.pipeline.event_model_v2_features import AutomaticDataset


def test_decode_applies_per_type_thresholds_and_type_nms() -> None:
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
        frames=np.asarray([10, 11, 20]),
    )

    rows = decode(probabilities, dataset)

    assert [(row["event_type"], row["frame"]) for row in rows] == [("contact", 11.0)]
