import numpy as np

import anthropometric_contacts as anthro
import rich_ball_physics as rich


def test_ray_at_z_round_trip():
    # A generic finite pinhole camera: the test exercises the plane intersection
    # algebra without relying on any match artifact.
    p = np.array(
        [
            [900.0, 20.0, 480.0, -1600.0],
            [5.0, 880.0, 270.0, -9500.0],
            [0.01, 0.02, 1.0, 8.0],
        ]
    )
    xyz = np.array([4.2, 17.5, 1.37])
    uv = rich.project_one(p, xyz)
    recovered = anthro.ray_at_z(p, uv, xyz[2])
    np.testing.assert_allclose(recovered, xyz, atol=1e-9)


def test_foot_pixel_uses_lower_grounded_ankle():
    points = {
        key: (np.array([0.0, 0.0]), 0.0) for key in anthro.POSE_KEYS
    }
    points["left_ankle"] = (np.array([100.0, 190.0]), 0.9)
    points["right_ankle"] = (np.array([120.0, 160.0]), 0.9)
    pose = anthro.PoseFrame(
        frame=10,
        side="near",
        box=np.array([80.0, 80.0, 140.0, 200.0]),
        conf=0.9,
        points=points,
    )
    foot, method = anthro.foot_pixel(pose)
    np.testing.assert_allclose(foot, [100.0, 190.0])
    assert method == "ankle"
    assert anthro.is_grounded(pose)


def test_player_side_identity_is_complementary():
    near = {"92": "B"}
    names = {"A": "Jannik_Sinner", "B": "Carlos_Alcaraz"}
    assert anthro.player_for_side("pt0092", "near", near, names) == "Carlos_Alcaraz"
    assert anthro.player_for_side("pt0092", "far", near, names) == "Jannik_Sinner"
