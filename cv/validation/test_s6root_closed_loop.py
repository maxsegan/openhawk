from __future__ import annotations

import numpy as np
import pytest

from cv.validation.s6root_closed_loop import (
    CONTACT_RAY_ARM,
    NET_CONSTRAINT_ARM,
    NET_CONSTRAINT_CONTACT_RAY_ARM,
    PRIMARY_ARM,
    assert_regression_target,
)
from cv.validation.s6root_common import ScaledVerticalCamera, frame_time


class _Camera:
    def p_at(self, frame: float) -> np.ndarray:
        del frame
        return np.arange(12, dtype=float).reshape(3, 4)

    def h_at(self, frame: float) -> np.ndarray:
        del frame
        return np.eye(3)

    def quality_at(self, frame: float) -> dict:
        del frame
        return {"reliable": True}


def test_vertical_scale_preserves_ground_plane_columns() -> None:
    base = _Camera()
    scaled = ScaledVerticalCamera(base, 1.25)
    before = base.p_at(1.0)
    after = scaled.p_at(1.0)
    assert np.array_equal(after[:, [0, 1, 3]], before[:, [0, 1, 3]])
    assert np.array_equal(after[:, 2], 1.25 * before[:, 2])


def test_frame_time_interpolates_half_frames_on_pts() -> None:
    pts = np.asarray([0.0, 0.04, 0.08])
    assert frame_time(pts, 1.0, 25.0) == 0.0
    assert frame_time(pts, 1.5, 25.0) == 0.02


def _closed_loop_payload(
    *,
    accepted: int,
    median_px: float,
    contact_ray_accepted: int = 19,
    contact_ray_median_px: float = 1.95,
) -> dict:
    return {
        "table": [
            {
                "arm": PRIMARY_ARM,
                "accepted": accepted,
                "held_out_median_px_corpus_median": median_px,
            },
            {
                "arm": CONTACT_RAY_ARM,
                "accepted": contact_ray_accepted,
                "held_out_median_px_corpus_median": contact_ray_median_px,
            },
            {
                "arm": NET_CONSTRAINT_ARM,
                "accepted": 19,
                "held_out_median_px_corpus_median": 1.95,
            },
            {
                "arm": NET_CONSTRAINT_CONTACT_RAY_ARM,
                "accepted": 19,
                "held_out_median_px_corpus_median": 1.95,
            },
        ]
    }


def test_closed_loop_regression_target_accepts_19_of_20_below_two_pixels() -> None:
    assert_regression_target(_closed_loop_payload(accepted=19, median_px=1.95))


@pytest.mark.parametrize(
    ("accepted", "median_px"),
    [(18, 1.5), (20, 2.01)],
)
def test_closed_loop_regression_target_rejects_yield_or_error_regression(
    accepted: int,
    median_px: float,
) -> None:
    with pytest.raises(AssertionError, match="closed loop failed"):
        assert_regression_target(_closed_loop_payload(accepted=accepted, median_px=median_px))


@pytest.mark.parametrize(
    ("accepted", "median_px"),
    [(18, 1.5), (20, 2.01)],
)
def test_closed_loop_regression_target_rejects_contact_ray_regression(
    accepted: int,
    median_px: float,
) -> None:
    with pytest.raises(AssertionError, match="contact-ray closed loop failed"):
        assert_regression_target(
            _closed_loop_payload(
                accepted=20,
                median_px=1.5,
                contact_ray_accepted=accepted,
                contact_ray_median_px=median_px,
            )
        )
