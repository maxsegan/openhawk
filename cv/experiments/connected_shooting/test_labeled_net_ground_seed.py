import json

import numpy as np
import pytest

from cv.experiments.connected_shooting.labeled_net_ground_seed import ballistic_ground_seed

G = 9.81


def _endpoint(net, velocity, elapsed, gravity=G):
    net = np.asarray(net, float)
    velocity = np.asarray(velocity, float)
    return net + velocity * elapsed + np.array([0.0, 0.0, -0.5 * gravity * elapsed**2])


def test_analytic_endpoint_recovery_is_fps_invariant_for_fixed_duration():
    net = np.array([1.2, 11.885, 1.07])
    velocity = np.array([3.5, -9.0, -2.25])
    duration = 0.4
    ground = _endpoint(net, velocity, duration)
    net_frame = 100.0
    recovered = []
    for fps in (25.0, 50.0, 60.0, 240.0):
        source_net, source_ground = net.copy(), ground.copy()
        receipt = ballistic_ground_seed(
            source_net, net_frame, source_ground, net_frame + duration * fps, fps
        )
        np.testing.assert_array_equal(source_net, net)
        np.testing.assert_array_equal(source_ground, ground)
        assert receipt["status"] == "supported"
        assert receipt["native_timestamps_changed"] is False
        assert "no hard direction or passivity requirement" in receipt["assumptions"]
        assert "original timestamps unchanged" in receipt["assumptions"]
        assert "not a net material law" in " ".join(receipt["assumptions"])
        np.testing.assert_allclose(receipt["elapsed_seconds"], duration)
        np.testing.assert_allclose(receipt["outgoing_velocity_mps"], velocity, atol=1e-12)
        np.testing.assert_allclose(
            _endpoint(net, receipt["outgoing_velocity_mps"], receipt["elapsed_seconds"]),
            ground,
            atol=1e-12,
        )
        assert type(receipt["outgoing_velocity_mps"]) is list
        json.dumps(receipt, allow_nan=False)
        recovered.append(receipt["outgoing_velocity_mps"])
    np.testing.assert_allclose(recovered, np.broadcast_to(velocity, (4, 3)), atol=1e-12)


def test_downward_and_upward_deflection_seeds_are_both_supported():
    net = [0.4, 11.885, 1.0]
    ground = [-1.5, 8.2, 0.033]
    down = ballistic_ground_seed(net, 10.0, ground, 20.0, 50.0)
    up = ballistic_ground_seed(net, 10.0, ground, 60.0, 50.0)
    assert down["status"] == up["status"] == "supported"
    assert down["outgoing_velocity_mps"][2] < 0.0
    assert up["outgoing_velocity_mps"][2] > 0.0
    assert down["outgoing_velocity_mps"][0] < 0.0
    assert down["outgoing_velocity_mps"][1] < 0.0
    np.testing.assert_allclose(_endpoint(net, down["outgoing_velocity_mps"], 0.2), ground)
    np.testing.assert_allclose(_endpoint(net, up["outgoing_velocity_mps"], 1.0), ground)


def test_invalid_interval_nonfinite_and_malformed_inputs_raise():
    net, ground = [0.0, 11.885, 1.0], [1.0, 9.0, 0.03]
    with pytest.raises(ValueError, match="coordinates"):
        ballistic_ground_seed([0.0, 11.885], 10.0, ground, 20.0, 50.0)
    with pytest.raises(ValueError, match="coordinates"):
        ballistic_ground_seed(net, 10.0, [[1.0, 9.0, 0.03]], 20.0, 50.0)
    with pytest.raises(ValueError, match="coordinates"):
        ballistic_ground_seed([0.0, np.nan, 1.0], 10.0, ground, 20.0, 50.0)
    with pytest.raises(ValueError, match="coordinates"):
        ballistic_ground_seed(net, 10.0, [1.0, np.inf, 0.03], 20.0, 50.0)
    with pytest.raises(ValueError, match="fps"):
        ballistic_ground_seed(net, 10.0, ground, 20.0, 0.0)
    with pytest.raises(ValueError, match="gravity"):
        ballistic_ground_seed(net, 10.0, ground, 20.0, 50.0, gravity_mps2=-9.81)
    with pytest.raises(ValueError, match="velocity bound"):
        ballistic_ground_seed(net, 10.0, ground, 20.0, 50.0, maximum_component_mps=0.0)
    with pytest.raises(ValueError, match="duration"):
        ballistic_ground_seed(net, 20.0, ground, 20.0, 50.0)
    with pytest.raises(ValueError, match="duration"):
        ballistic_ground_seed(net, 21.0, ground, 20.0, 50.0)


def test_out_of_bound_velocity_and_net_not_above_ground_are_unsupported():
    fps, elapsed, net_frame = 50.0, 0.5, 10.0
    ground_frame = net_frame + elapsed * fps
    net = np.array([0.0, 11.885, 1.2])
    at_bound = _endpoint(net, [75.0, 0.0, 0.0], elapsed)
    supported = ballistic_ground_seed(net, net_frame, at_bound, ground_frame, fps)
    assert supported["status"] == "supported"
    np.testing.assert_allclose(supported["outgoing_velocity_mps"], [75.0, 0.0, 0.0])
    over = at_bound.copy()
    over[0] += 1e-6
    rejected = ballistic_ground_seed(net, net_frame, over, ground_frame, fps)
    assert rejected["status"] == "unsupported"
    assert rejected["unsupported_reason"] == "velocity_component_exceeds_bound"
    assert "outgoing_velocity_mps" not in rejected
    np.testing.assert_allclose(rejected["elapsed_seconds"], elapsed)
    json.dumps(rejected, allow_nan=False)
    equal_height = ballistic_ground_seed([0.0, 11.885, 0.03], 10.0, [2.0, 9.0, 0.03], 30.0, 50.0)
    below = ballistic_ground_seed([0.0, 11.885, 0.02], 10.0, [2.0, 9.0, 0.04], 30.0, 50.0)
    assert equal_height["status"] == below["status"] == "unsupported"
    assert (
        equal_height["unsupported_reason"]
        == below["unsupported_reason"]
        == ("source_net_not_above_ground")
    )
    assert "outgoing_velocity_mps" not in equal_height
    listed = [0.0, 11.885, 1.2]
    ballistic_ground_seed(listed, net_frame, at_bound.tolist(), ground_frame, fps)
    assert listed == [0.0, 11.885, 1.2]
