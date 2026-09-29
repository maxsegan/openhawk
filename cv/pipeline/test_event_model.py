import numpy as np

from cv.pipeline.event_model import add_probability_context


def test_probability_context_uses_neighboring_candidates() -> None:
    rows = [
        {"clip": "m__pt0001", "proposal_frame": 10.0, "source_fps": 25.0},
        {"clip": "m__pt0001", "proposal_frame": 30.0, "source_fps": 25.0},
        {"clip": "m__pt0001", "proposal_frame": 50.0, "source_fps": 25.0},
    ]
    probabilities = np.asarray(
        [[0.1, 0.8, 0.1, 0.0], [0.1, 0.1, 0.8, 0.0], [0.1, 0.8, 0.1, 0.0]]
    )
    add_probability_context(rows, probabilities, probabilities)

    assert rows[1]["context_bounce_bridge"] > 0.0
    assert rows[1]["context_previous_contact_support"] > 0.0
    assert rows[1]["context_next_contact_support"] > 0.0
