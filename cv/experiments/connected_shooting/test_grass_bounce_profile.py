"""The bounce estimator must recover a bounce it was given, and refuse a bad one."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import grass_bounce_profile as profile
from physics import flight


CAMERA = np.asarray(
    [
        [1400.0, 0.0, 960.0, -960.0 * 5.485],
        [0.0, -1400.0, 540.0, 1400.0 * 3.0 + 540.0 * 5.0],
        [0.0, 0.0, 1.0, 5.0],
    ]
)


def synthetic_bounce(restitution, retention, incoming=(0.0, -22.0, -7.0), fps=25.0):
    """Native fronts of one impact with a known restitution and retention."""
    anchor = np.asarray([5.0, 14.0, 0.0325])
    incoming = np.asarray(incoming, float)
    outgoing = np.asarray(
        [incoming[0] * retention, incoming[1] * retention, -incoming[2] * restitution]
    )
    frames, pixels = [], []
    for offset in (-5, -4, -3, -2, 2, 3, 4, 5):
        seconds = offset / fps
        velocity = incoming if offset < 0 else outgoing
        position = profile._sample(anchor, velocity, np.asarray([seconds]))[0]
        sweep = profile._sample(
            anchor,
            velocity,
            seconds + np.linspace(0.0, profile.EXPOSURE_FRAMES, profile.EXPOSURE_SAMPLES) / fps,
        )
        image = np.asarray([CAMERA @ np.r_[point, 1.0] for point in sweep])
        image = image[:, :2] / image[:, 2:]
        axis = image[-1] - image[0]
        tips, _ = profile.swept_exposure.directional_tips(sweep, CAMERA, axis)
        assert np.isfinite(position).all()
        frames.append(100 + offset)
        pixels.append(tips[1])
    return anchor, np.asarray(frames, float), np.asarray(pixels), outgoing


@pytest.mark.parametrize(("restitution", "retention"), [(0.60, 0.63), (0.89, 0.67), (0.45, 0.80)])
def test_a_known_impact_is_recovered_from_its_native_fronts(restitution, retention):
    anchor, frames, pixels, _ = synthetic_bounce(restitution, retention)
    solved = {}
    for name, mask in (("incoming", frames < 100), ("outgoing", frames > 100)):
        answer = profile._solve_arc(
            anchor,
            frames[mask],
            100.0,
            25.0,
            [CAMERA] * int(mask.sum()),
            pixels[mask],
            [None] * int(mask.sum()),
            np.array([0.0, 15.0, 5.0 if name == "outgoing" else -5.0]),
        )
        assert answer is not None
        solved[name] = answer
    incoming, incoming_rms = solved["incoming"]
    outgoing, outgoing_rms = solved["outgoing"]
    assert max(incoming_rms, outgoing_rms) < 0.5
    assert outgoing[2] / -incoming[2] == pytest.approx(restitution, abs=0.02)
    assert float(np.hypot(*outgoing[:2])) / float(np.hypot(*incoming[:2])) == pytest.approx(
        retention, abs=0.02
    )


def test_the_backward_arc_is_not_a_reversed_forward_arc():
    """Drag opposes motion, so time reversal is not velocity negation."""
    anchor = np.asarray([5.0, 14.0, 0.0325])
    velocity = np.asarray([0.0, -30.0, -8.0])
    seconds = np.asarray([-0.2])
    backward = profile._sample(anchor, velocity, seconds)[0]
    negated, _, _ = flight.sample_states(
        anchor, -velocity, np.zeros(3), np.abs(seconds), substeps_per_second=240.0
    )
    assert float(np.linalg.norm(backward - negated[0])) > 0.05


def test_the_wide_prior_returns_the_prior_when_the_measurement_is_thin():
    angles = np.asarray([16.0, 16.5])
    values = np.asarray([0.60, 0.62])
    fitted = profile.shrunk_regression(angles, values, 0.80, 0.0)
    assert 0.60 < fitted["intercept"] < 0.80
    assert fitted["prior_weight"] > 0.3


def test_a_strong_measurement_overrules_the_prior():
    angles = np.linspace(8.0, 34.0, 200)
    values = np.full(200, 0.61)
    fitted = profile.shrunk_regression(angles, values, 0.80, 0.0)
    assert fitted["intercept"] == pytest.approx(0.61, abs=0.02)


def test_the_chart_prior_orders_the_surfaces_the_way_the_chart_does():
    grass = profile.chart_prediction("grass", 16.0)
    hard = profile.chart_prediction("hard", 16.0)
    clay = profile.chart_prediction("clay", 16.0)
    assert grass["restitution"] < hard["restitution"] < clay["restitution"]
