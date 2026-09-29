from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_net_ray_seed as seed


def test_native_seed_repairs_airborne_approach_without_changing_other_parameters(monkeypatch):
    scene = SimpleNamespace(
        pixels=[[]], contact_frames=[0, 50], fps=60, surface="hard", bounce_profile="nominal"
    )
    context = dict(scene=scene, attempt=dict(point_clip="clip"))
    parameters = np.array([5.0, 20.0, 2.0, 1.0, -12.0, -15.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    original = parameters.copy()
    monkeypatch.setattr(seed.model, "chain", lambda *_: [dict(start_xyz=parameters[:3])])
    P = np.array([[1000, 0, 0, 0], [0, 0, 1000, 0], [0, 1, 0, 0]], float)
    xyz = np.array([5.5, seed.NET_Y_M, 0.9])
    front = P @ np.r_[xyz, 1]
    front = front[:2] / front[2]
    labels = dict(
        ball=dict(
            records=[
                dict(
                    clip="clip",
                    frames=[
                        dict(frame=f, status="visible", x1080=front[0], y1080=front[1])
                        for f in [29, 30]
                    ],
                )
            ]
        )
    )
    cameras = dict(cameras=[dict(frame=f, status="supported", P=P.tolist()) for f in [29, 30]])
    event = dict(frame=30, frame_interval=[29, 30])
    result, receipt = seed.seed(context, parameters, labels, cameras, event)
    np.testing.assert_array_equal(parameters, original)
    np.testing.assert_array_equal(
        result[[0, 1, 2, 3, 6, 7, 8, 9, 10]], original[[0, 1, 2, 3, 6, 7, 8, 9, 10]]
    )
    np.testing.assert_allclose(receipt["conditional_mean_xyz_m"], xyz, atol=1e-12)
    assert receipt["projection"]["prior_ground_impacts"] == 0
    assert receipt["projection"]["plane_error_m"] < 1e-6
    assert receipt["projection"]["height_error_m"] < 1e-6
    assert receipt["height_constraint_in_final_fit"] is False
    for row in cameras["cameras"]:
        row["supported"] = False
    with pytest.raises(ValueError, match="no supported original visible front"):
        seed.seed(context, parameters, labels, cameras, event)
