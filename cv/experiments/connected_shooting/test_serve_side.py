import numpy as np
import pytest

from cv.experiments.connected_shooting import serve_side


def box(x0, y0, x1, y1, side):
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, side=side)


# fresh source09_case001_short_a2 f253: the near server's ball is nearer the far box centre.
NEAR = box(795.2, 541.1, 886.1, 806.8, "near")
FAR = box(1141.9, 223.4, 1198.5, 340.6, "far")
BALL = np.array([958.0, 406.0])


def centre_distance(row):
    centre = np.array([(row["x0"] + row["x1"]) / 2, (row["y0"] + row["y1"]) / 2])
    return float(np.linalg.norm(centre - BALL))


def test_centre_keeps_the_nearest_box():
    assert serve_side.choose([NEAR, FAR], BALL, 1.0, "centre", centre_distance) is FAR


def test_serve_reach_takes_the_box_whose_reach_holds_the_ball():
    assert serve_side.reach_distance(NEAR, BALL) == 0.0
    assert serve_side.reach_distance(FAR, BALL) > 1.0
    assert serve_side.choose([NEAR, FAR], BALL, 1.0, "serve_reach", centre_distance) is NEAR


def test_both_in_reach_falls_back_to_the_nearest_centre():
    other = box(900.0, 380.0, 960.0, 500.0, "far")
    chosen = serve_side.choose([NEAR, other], BALL, 1.0, "serve_reach", centre_distance)
    assert chosen is other


def test_unknown_association_refuses():
    with pytest.raises(ValueError):
        serve_side.choose([NEAR], BALL, 1.0, "nearest", centre_distance)
