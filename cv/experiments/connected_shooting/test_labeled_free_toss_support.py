"""Explicit span does not activate hand-held, abstained or foreign-clip rows."""

import numpy as np
import pytest
from cv.experiments.connected_shooting.labeled_free_toss_support import observations


def test_explicit_interval_preserves_original_fronts_uncertainty_and_abstention():
    points = [
        dict(
            frame=f,
            status="visible",
            x1080=f,
            y1080=20,
            uncertainty_radius_px1080=5 if f == 5 else 3,
        )
        for f in range(1, 10)
    ]
    points[3]["status"] = "ambiguous"
    labels = {
        "ball": {"records": [{"clip": "p", "frames": points}, {"clip": "other", "frames": []}]}
    }
    P = np.eye(3, 4).tolist()
    cams = {"cameras": [dict(frame=f, status="supported", P=P) for f in range(1, 10)]}
    support = dict(
        frame_interval=[3, 8],
        rationale="visible separation from hand before3",
        annotation_origin="test",
    )
    out = observations(labels, cams, "p", support)
    assert [r["frame"] for r in out["rows"]] == [3, 5, 6, 7, 8]
    assert out["rows"][1]["uncertainty_px"] == 5
    assert out["abstained"] == [dict(frame=4, reason="original ball absent or abstained")]
    assert out["rows"][0]["pixel"] == [3, 20]
    assert not out["release_epoch_inferred"]
    with pytest.raises(ValueError, match="qualification"):
        observations(labels, cams, "p", {"frame_interval": [3, 8]})
    cams["cameras"][4]["k1"] = 0.1
    cams["cameras"][4]["dist_center"] = [0, 0]
    with pytest.raises(ValueError, match="radial"):
        observations(labels, cams, "p", support)
