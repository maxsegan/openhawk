"""Shared-contact toss support must reject visible incoming mismatches."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import joint_toss_residual as joint
from cv.experiments.connected_shooting import serve_block_repair_probe as serve


def fixture():
    camera = [[100, 0, 0, 0], [0, 0, 100, 0], [0, 0, 0, 1]]
    rows = []
    for frame in range(90, 100):
        t = (frame - 100) / 50
        rows.append(
            dict(
                frame=frame,
                pixel=[100 * (0.1 + 0.1 * t), 100 * (3 - 3 * t - 4.905 * t * t)],
                uncertainty_px=3,
                camera=camera,
            )
        )
    return dict(status="supported", contact_frame=100, rows=rows), dict(court_xy_m=[0, 0])


def test_native_ballistic_toss_joins_exact_contact_and_passes_original_release_checks():
    obs, feet = fixture()
    _, slack, record = joint.evaluate([0.1, 0, 3], [0.1, 0, -3, 1], obs, feet, 50)
    assert record["image_rms_px"] < 1e-10
    assert np.all(slack >= 0)
    assert record["shared_contact_frame"] == 100
    assert record["shared_contact_xyz_m"] == [0.1, 0, 3]
    assert [r["frame"] for r in record["native_projection"]] == list(range(90, 100))
    assert record["maximum_weighted_rms"] == 2.5


def test_outgoing_contact_that_misses_original_toss_pixels_fails_even_with_plausible_release():
    obs, feet = fixture()
    _, _, record = joint.evaluate([0.4, 0, 3], [0.1, 0, -3, 1], obs, feet, 50)
    assert record["image_rms_px"] == pytest.approx(30)
    assert not record["constraints"]["fronts_within_declared_uncertainty"]
    assert record["constraints"]["release_at_hand_height"]


def test_no_postcontact_toss_picture_or_absent_wing_is_invented():
    obs, feet = fixture()
    obs["rows"][-1]["frame"] = 100
    with pytest.raises(ValueError, match="precede"):
        joint.evaluate([0.1, 0, 3], [0.1, 0, -3, 1], obs, feet, 50)
    obs["status"] = "abstained"
    with pytest.raises(ValueError, match="supported original"):
        joint.evaluate([0.1, 0, 3], [0.1, 0, -3, 1], obs, feet, 50)


def test_raw_toss_failure_blocks_otherwise_accepted_outgoing_gain():
    before = {"accepted_flight_count": 1}
    trial = dict(
        start="reference",
        eligible_improvement=True,
        contact_image_constraints_valid=True,
        all_frozen_gate_constraints_valid=True,
        raw_toss_constraints_valid=False,
        verdict={"accepted_flight_count": 2},
        cost=1,
    )
    result = dict(status="measured", before=before, trials=[trial])
    serve.finalize_selection(result)
    assert result["selected_source"] == "original_reference"
    assert result["selected_verdict"] is before
