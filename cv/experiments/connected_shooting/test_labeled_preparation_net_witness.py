import numpy as np
import pytest

from cv.experiments.connected_shooting.labeled_preparation_net_witness import augment_visible


def test_only_eligible_visible_camera_supported_context_is_added():
    records = [
        dict(frame=f, status="visible", x1080=f, y1080=2, uncertainty_radius_px1080=3)
        for f in range(10, 15)
    ]
    records[3]["status"] = "occluded"
    pixels = {10: np.array([10, 2])}
    result, sigma, added = augment_visible(
        pixels, {10: 3}, records, [10, 11, 12, 13], {10: 0, 11: 0, 13: 0}
    )
    assert set(result) == {10, 11} and added == [11] and sigma[11] == 3
    assert set(pixels) == {10}


def test_existing_pixel_cannot_be_silently_replaced():
    with pytest.raises(ValueError, match="immutable"):
        augment_visible(
            {10: [1, 2]}, None, [dict(frame=10, status="visible", x1080=3, y1080=4)], [10], {10: 0}
        )
