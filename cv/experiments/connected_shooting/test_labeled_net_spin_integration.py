import numpy as np

from cv.experiments.connected_shooting import labeled_passive_tape as tape, net_collision
from cv.experiments.connected_shooting.labeled_net_free_response import (
    FreeNetVelocity,
    response_from_record,
)
from cv.experiments.connected_shooting.labeled_net_spin_response import FreeNetState


def test_spin_transition_preserves_incoming_path_and_separates_cached_outgoing_paths():
    args = (
        np.array([5.0, 14.0, 1.2, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0]),
        0.0,
        np.array([0.0, 2.0, 5.5, 10.0]),
        30.0,
        "clay",
    )
    original = net_collision.simulate
    outgoing = (4.0, -8.0, 3.0)
    states = [FreeNetVelocity(outgoing), FreeNetState(outgoing, (100.0, -300.0, 50.0))]
    results = []
    for state in states:
        with tape.using_response(state):
            results.append(net_collision.simulate(*args, net_frame=5.5))
    a, b = results
    assert net_collision.simulate is original
    np.testing.assert_array_equal(a[0][:2], b[0][:2])
    np.testing.assert_array_equal(a[-1][0]["x"], b[-1][0]["x"])
    np.testing.assert_array_equal(a[-1][0]["w_out"], a[-1][0]["w_in"])
    np.testing.assert_array_equal(b[-1][0]["w_out"], states[1].outgoing_spin_rad_s)
    assert np.linalg.norm(a[0][-1] - b[0][-1]) > 1e-4
    restored = response_from_record(b[-1][0]["experimental_response"])
    assert restored.key == states[1].key
    with tape.using_response(restored):
        replay = net_collision.simulate(*args, net_frame=5.5)
    np.testing.assert_array_equal(replay[0], b[0])
    with tape.using_response(states[0]):
        replay = net_collision.simulate(*args, net_frame=5.5)
    np.testing.assert_array_equal(replay[0], a[0])
