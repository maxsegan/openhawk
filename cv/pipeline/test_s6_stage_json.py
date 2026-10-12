"""``save`` and ``final_state_binding`` encode numpy on the fly: same bytes as the jsonable pre-pass."""

import json

import numpy as np
import pytest

from cv.experiments.connected_shooting.serve_timing_profile import jsonable
from cv.pipeline import s6_labeled_stage as stage


def _document():
    rng = np.random.default_rng(3)
    return {
        "parameters": rng.normal(size=11),
        "measurement": {
            "rms": np.float64(1.0) / 3,
            "rows": np.asfortranarray(rng.normal(size=(3, 2))),
            "count": np.int64(7),
            "small": np.float32(0.1),
            "flag": np.bool_(False),
            "pairs": (np.arange(3), [np.uint8(4), 2.5, None, "x"]),
            "empty": np.zeros((0, 2)),
        },
        "verdict": {"flights": [{"accepted": True, "z": -0.0, "big": 1e300}]},
        "net_response": None,
        "evaluation_context": {"frames": list(range(4)), "tiny": 5e-324},
        "initial_search": {"parameters": [np.float64(2.0)]},
        "selected_by": {"selected_search": "coarse"},
    }


def test_save_bytes_equal_jsonable_prepass(tmp_path, monkeypatch):
    monkeypatch.delenv("TENNIS_S6_LEAN", raising=False)  # lean receipts write compact JSON
    value = _document()
    stage.save(tmp_path / "doc.json", value)
    expected = json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n"
    assert (tmp_path / "doc.json").read_text() == expected


def test_final_state_binding_hashes_unchanged():
    import hashlib

    value = _document()
    got = stage.final_state_binding(value)["sha256"]
    for name in ("parameters", "measurement", "verdict", "evaluation_context"):
        text = json.dumps(jsonable(value[name]), sort_keys=True, separators=(",", ":"))
        assert got[name] == hashlib.sha256(text.encode()).hexdigest()


def test_save_still_refuses_nonfinite_and_unknown(tmp_path):
    with pytest.raises(ValueError):
        stage.save(tmp_path / "a.json", {"x": np.float64("nan")})
    with pytest.raises(TypeError):
        stage.save(tmp_path / "b.json", {"x": object()})
