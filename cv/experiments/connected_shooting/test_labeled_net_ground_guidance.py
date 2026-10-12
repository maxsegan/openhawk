from types import SimpleNamespace

import numpy as np

from cv.experiments.connected_shooting import net_collision
from cv.experiments.connected_shooting.labeled_net_ground_guidance import apply
from cv.experiments.connected_shooting.labeled_net_height_chart import project_net_height


def test_timing_guidance_is_continuous_when_impact_crosses_scored_horizon():
    theta, _ = project_net_height(
        np.array([5.0, 20.0, 1.0, 0.0, -20.0, -1.0, 0.0, 0.0, 0.0]),
        0.0,
        25.0,
        60.0,
        "hard",
        -0.05,
    )
    result = net_collision.simulate(theta, 0.0, np.array([0.0, 60.0]), 60.0, "hard", net_frame=25.0)
    impact = result[3][0]
    outputs = []
    for horizon in [impact["frame"] - 0.3, impact["frame"] + 0.3]:
        simulated = net_collision.simulate(
            theta, 0.0, np.array([0.0, horizon]), 60.0, "hard", net_frame=25.0
        )
        scene = SimpleNamespace(
            pixels=[[]],
            surface="hard",
            fps=60.0,
            bounce_profile="nominal",
            contact_frames=np.array([0.0, horizon]),
            net_hit_frames=[np.array([25.0])],
        )
        # The integration step can end a fraction beyond its query horizon;
        # choose a .3-frame separation to put the impact on opposite sides.
        flight = dict(start_xyz=theta[:3], end_frame=horizon, bounces=simulated[3])
        expected = impact["frame"] - 5
        evidence = [dict(expected_bounce_frames=[expected])]
        _, r, e = apply(
            scene,
            np.r_[theta, 1.0, 1.0],
            [flight],
            np.array([123.0, 0.0]),
            evidence,
            [expected - 0.5, expected + 0.5],
        )
        outputs.append(r)
        assert abs(e[-1]["numerical_terminal_timing"]["frame"] - impact["frame"]) < 1e-8
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8)
    np.testing.assert_allclose(outputs[0], [12.0, 0.0, 45.0, 0.0], atol=1e-8)


def test_two_impact_guidance_preserves_ordinals_across_final_horizon():
    from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity
    from cv.experiments.connected_shooting.labeled_passive_tape import using_response
    from cv.experiments.connected_shooting.labeled_net_normal_response import NetNormalResponse

    theta = np.array([5.0, 14.0, 1.2, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0])
    params = np.r_[theta, 1.0, 1.0]
    outputs = []
    with using_response(NetNormalResponse(FreeNetVelocity([0.1, 3.0, -3.0]), 0.4)):
        full = net_collision.simulate(
            theta, 0.0, np.array([0.0, 30.0]), 30.0, "grass", net_frame=5.5
        )
        assert len(full[3]) >= 2
        first, second = full[3][:2]
        expected = [first["frame"] - 3, second["frame"] - 4]
        for horizon in [second["frame"] - 0.3, second["frame"] + 0.3]:
            trace = net_collision.simulate(
                theta, 0.0, np.array([0.0, horizon]), 30.0, "grass", net_frame=5.5
            )
            scene = SimpleNamespace(
                pixels=[[]],
                surface="grass",
                fps=30.0,
                bounce_profile="nominal",
                contact_frames=np.array([0.0, horizon]),
                net_hit_frames=[np.array([5.5])],
            )
            native = [dict(start_xyz=theta[:3], end_frame=horizon, bounces=trace[3])]
            evidence = [dict(expected_bounce_frames=expected)]
            _, residual, receipt = apply(
                scene,
                params,
                native,
                np.array([123.0, 456.0, 7.0]),
                evidence,
                [[f - 0.5, f + 0.5] for f in expected],
            )
            outputs.append(residual)
            np.testing.assert_allclose(
                [x["frame"] for x in receipt[-1]["numerical_terminal_timings"]],
                [first["frame"], second["frame"]],
                atol=1e-8,
            )
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8)
    np.testing.assert_allclose(outputs[0], [4.0, 8.0, 7.0, 25.0, 0.0, 35.0, 0.0], atol=1e-8)
