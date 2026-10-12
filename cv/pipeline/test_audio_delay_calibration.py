import numpy as np

from cv.pipeline import audio_delay_calibration as cal


def test_confident_single_delay():
    rng = np.random.default_rng(1)
    offsets = list(4.0 + rng.normal(0, 0.3, 20)) + [-3.0, 9.0]
    est = cal.estimate(offsets)
    assert est["confident"]
    assert abs(est["delay_frames"] - 4.0) <= 0.3
    assert len(est["modes"]) == 1


def test_uncertain_keeps_a_second_mode():
    offsets = [0.0, 0.1, -0.1, 0.2, 5.0, 5.1, 4.9]
    est = cal.estimate(offsets)
    assert not est["confident"]
    assert sorted(round(m) for m in est["modes"]) == [0, 5]


def test_too_few_anchors_is_the_prior():
    est = cal.estimate([2.0])
    assert est["modes"] == [0.0] and not est["confident"]


def test_calibrate_finds_a_shifted_onset():
    fps, delay = 50.0, 3.0
    trace = np.zeros(4000)
    anchors = []
    for f in range(100, 1500, 60):
        trace[cal.sample_of_frame(f + delay, fps)] = 50.0
        anchors.append({"clip": "c", "frame": float(f), "kind": "confident_contact"})
    trace += np.random.default_rng(0).normal(0, 1, trace.size)
    est = cal.calibrate({"c": trace}, {"c": fps}, anchors)
    assert est["confident"] and abs(est["delay_frames"] - delay) <= 0.6
    assert est["anchors"]["confident_contact"]["with_peak"] == len(anchors)
