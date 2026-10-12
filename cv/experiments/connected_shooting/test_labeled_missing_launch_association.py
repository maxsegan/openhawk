from copy import deepcopy

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_missing_launch_association as model


def fixture():
    contacts = [dict(frame=t, frame_interval=[t - 0.5, t + 0.5]) for t in [70.5, 90.5, 127.5]]
    fitted = {81: np.array([10, 500]), 89: np.array([20, 600]), 127: np.array([30, 100])}
    original = {**fitted, 70: np.array([40, 80])}
    cameras = {f: dict(P=np.eye(3, 4).tolist()) for f in [70, 71, 81, 89, 91, 127, 128]}
    grounds = [dict(frame=80.5)]

    def lookup(frame, pixel):
        return dict(side="near" if pixel[1] > 300 else "far")

    return contacts, fitted, original, cameras, grounds, lookup


def test_original_front_recovers_grammar_without_adding_fit_observation():
    args = fixture()
    before = deepcopy(args[1])
    pixel, r = model.qualify(*args)
    np.testing.assert_array_equal(pixel, [40, 80])
    assert r["implied_first_side"] == "far"
    assert [a["observation_frame"] for a in r["later_associations"]] == [89, 127]
    assert r["trajectory_observations_added"] == 0
    assert 70 not in args[1]
    for f in before:
        np.testing.assert_array_equal(before[f], args[1][f])
    pixel[:] = 0
    np.testing.assert_array_equal(args[2][70], [40, 80])


@pytest.mark.parametrize(
    "failure",
    ["first_camera", "front_camera", "local_front", "no_bounce", "contradiction", "one_contact"],
)
def test_rejects_unsupported_recovery(failure):
    args = list(fixture())
    if failure == "first_camera":
        args[3][71]["supported"] = False
    if failure == "front_camera":
        args[3][70]["supported"] = False
    if failure == "local_front":
        args[1][71] = np.array([40, 80])
    if failure == "no_bounce":
        args[4] = []
    if failure == "contradiction":
        args[1][127] = np.array([30, 600])
    if failure == "one_contact":
        args[0] = args[0][:2]
    with pytest.raises(ValueError):
        model.qualify(*args)
