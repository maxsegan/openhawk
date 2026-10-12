"""Synthetic checks of the origin matcher's one-to-one rule and tolerance."""

from evaluation.origin_matcher import interpolate_native_pts_ms, match_attempt

IMAGES = [{"frame": i, "native_pts_seconds": i / 50.0} for i in range(200)]  # 50 fps


def _ref(rid, lo, hi, confirmed=True, endpoint=None):
    return {
        "id": rid,
        "origin_pts_ms": [lo, hi],
        "endpoint_pts_ms": endpoint,
        "competitive_confirmed": confirmed,
    }


def test_interpolation_refuses_gaps():
    assert interpolate_native_pts_ms(10.5, IMAGES) == (210.0, "interpolated")
    gappy = [img for img in IMAGES if img["frame"] != 11]
    assert interpolate_native_pts_ms(10.5, gappy)[0] is None


def test_two_frame_tolerance_and_one_to_one():
    flights = [
        {"accepted": True, "start_frame": 10.0, "end_frame": 40.0},  # 200 ms
        {"accepted": True, "start_frame": 100.0, "end_frame": 130.0},  # 2000 ms
        {"accepted": False, "start_frame": 150.0, "end_frame": 160.0},
    ]
    refs = [
        _ref("a", 230.0, 240.0, endpoint=[800.0, 800.0]),  # 200 ms is 30 ms early; 40 ms tol
        _ref("b", 2100.0, 2100.0),  # 100 ms off: no match
        _ref("c", 3000.0, 3000.0),  # only a rejected flight nearby: no match
    ]
    out = match_attempt(refs, {"verdict": {"flights": flights}}, IMAGES, 50.0)
    assert out["tolerance_ms"] == 40.0
    assert out["confirmed_origin_gate_matches"] == 1
    assert out["confirmed_full_span_gates"] == 1
    assert out["confirmed_denominator"] == 3


def test_shared_flight_matches_neither_reference():
    flights = [{"accepted": True, "start_frame": 10.0, "end_frame": 40.0}]
    refs = [_ref("a", 200.0, 200.0), _ref("b", 210.0, 210.0)]
    out = match_attempt(refs, {"verdict": {"flights": flights}}, IMAGES, 50.0)
    assert out["confirmed_origin_gate_matches"] == 0


def test_missing_output_keeps_the_denominator():
    out = match_attempt([_ref("a", 0.0, 10.0)], {}, IMAGES, 50.0)
    assert (out["confirmed_origin_gate_matches"], out["confirmed_denominator"]) == (0, 1)
