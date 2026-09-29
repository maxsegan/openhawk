from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from cv.experiments.connected_shooting import measured_dynamics as dynamics, model
from cv.validation import s6_point_bench as generator
from physics import bounce_reference, flight


@pytest.mark.parametrize("state_index", [0, 1, 2])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_integrated_position_velocity_or_spin_holds(monkeypatch, state_index, bad_value):
    def broken_step(state, _dt):
        values = list(state)
        values[3 * state_index + 1] = bad_value
        return tuple(values)

    monkeypatch.setattr(dynamics.fast_flight, "rk4_step", broken_step)
    dynamics._TRACE_CACHE.clear()
    theta = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0, 2.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="nonfinite measured forward state"):
        dynamics.simulate(theta, 1.0, np.array([1.0, 2.0]), 25.0, "hard")


def test_measured_control_matches_independent_refined_impact_and_dwell():
    theta = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0, 2.0, 0.0, 0.0])
    spin = dynamics.spin_vector(theta)
    t, x, v, w = generator.integrate_arc(theta[:3], theta[3:6], spin, max_seconds=2)
    # The old generator uses a secant root. Build the expected fractional-step
    # state independently with physics.flight, keeping a strict numerical
    # comparison rather than loosening it to accept the old interpolation.
    j = int(np.flatnonzero(x[:, 2] <= dynamics.R_BALL)[0]) - 1
    delta = brentq(
        lambda dt: flight.rk4_step(x[j], v[j], w[j], dt)[0][2] - dynamics.R_BALL,
        0,
        t[j + 1] - t[j],
        xtol=5e-15,
        rtol=1e-14,
    )
    impact = t[j] + delta
    xb, vb, wb = flight.rk4_step(x[j], v[j], w[j], delta)
    xb[2] = dynamics.R_BALL
    rebound = bounce_reference.court_bounce(vb, wb, "hard")
    ta, xa, _, _ = generator.integrate_arc(xb, rebound.velocity, rebound.spin, max_seconds=0.15)
    dwell_end = impact + bounce_reference.DWELL_SECONDS
    times = np.unique(
        np.r_[
            np.arange(0, impact, 1 / 240),
            impact,
            impact + 0.002,
            dwell_end,
            dwell_end + np.arange(0.01, 0.11, 0.01),
        ]
    )
    expected = np.array(
        [
            generator._interp_rows(ti, t, x)
            if ti < impact
            else xb
            if ti <= dwell_end
            else generator._interp_rows(ti - dwell_end, ta, xa)
            for ti in times
        ]
    )
    actual, velocity, _, bounces = dynamics.simulate(theta, 10.5, 10.5 + times * 25, 25, "hard")
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert len(bounces) == 1
    assert bounces[0]["frame"] == pytest.approx(10.5 + impact * 25)
    dwell = (times > impact) & (times < dwell_end)
    np.testing.assert_array_equal(velocity[dwell], 0)
    assert actual[:, 2].min() >= dynamics.R_BALL - 1e-10


def test_measured_positions_are_invariant_to_query_density_and_connected():
    scene, _ = model.control()
    scene = replace(
        scene,
        dynamics="measured_240hz",
        contact_frames=np.array([1.0, 26.25, 51.5]),
        observation_frames=(np.arange(1.0, 27), np.arange(27.0, 52)),
    )
    parameters = np.array([5, 3, 2, 2, 20, -2, -1, -17, 4], float)
    dense = model.chain(scene, parameters)
    sparse = model.chain(
        scene, parameters, query_frames=tuple(f[::2] for f in scene.observation_frames)
    )
    np.testing.assert_array_equal(dense[0]["end_xyz"], dense[1]["start_xyz"])
    for a, b in zip(dense, sparse, strict=True):
        np.testing.assert_array_equal(a["positions"][::2], b["positions"])
    assert dense[0]["bounces"]


def test_no_future_impact_export_or_silent_surface_fallback():
    theta = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0, 2.0, 0.0, 0.0])
    _, _, _, long = dynamics.simulate(theta, 1.0, np.array([1.0, 26.0]), 25, "hard")
    _, _, _, short = dynamics.simulate(
        theta, 1.0, np.array([1.0, long[0]["frame"] - 0.001]), 25, "hard"
    )
    assert short == []
    # Grass now carries a declared broadcast-measured row; an unmeasured surface
    # is still refused rather than silently taking another surface's profile.
    dynamics.simulate(theta, 1.0, np.array([1.0, 26.0]), 25, "grass")
    with pytest.raises(ValueError, match="measured surface"):
        dynamics.simulate(theta, 1.0, np.array([1.0, 26.0]), 25, "carpet")
    with pytest.raises(ValueError, match="initial state"):
        dynamics.simulate(theta, 1.0, np.array([0.0, 26.0]), 25, "hard")


@pytest.mark.parametrize("profile", list(dynamics.BOUNCE_PROFILES)[1:])
def test_bounce_perturbations_preserve_incoming_flight_spin_and_dwell(profile):
    theta = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0, 2.0, 0.0, 0.0])
    queries = np.arange(1.0, 27.0)
    nominal, _, _, reference = dynamics.simulate(theta, 1.0, queries, 25, "hard")
    changed, _, _, impacts = dynamics.simulate(
        theta, 1.0, queries, 25, "hard", bounce_profile=profile
    )
    a, b = reference[0], impacts[0]
    before = queries < a["frame"]
    np.testing.assert_array_equal(nominal[before], changed[before])
    assert a["frame"] == b["frame"]
    assert a["dwell_seconds"] == b["dwell_seconds"]
    np.testing.assert_array_equal(a["w_out"], b["w_out"])
    vertical, horizontal = dynamics.BOUNCE_PROFILES[profile]
    np.testing.assert_allclose(b["v_out"], a["v_out"] * [horizontal, horizontal, vertical])
    assert not b["coefficient_clipped"]
    assert not np.allclose(nominal[~before], changed[~before])


def test_coefficient_saturation_is_recorded_and_nominal_velocity_is_bit_exact():
    incoming = np.array([1.0, 30.0, -0.01])
    rebound = bounce_reference.court_bounce(incoming, np.zeros(3), "hard")
    nominal, _ = dynamics.rebound_velocity(rebound, "nominal")
    np.testing.assert_array_equal(nominal, rebound.velocity)
    changed, record = dynamics.rebound_velocity(rebound, "restitution_high")
    assert record["coefficient_clipped"]
    assert record["requested_restitution"] > 1
    assert record["applied_restitution"] == 1
    assert changed[2] == abs(incoming[2])
    with pytest.raises(ValueError, match="bounce profile"):
        dynamics.rebound_velocity(rebound, "unknown")


def test_perturbed_dynamics_keeps_shared_contact_structural():
    scene, _ = model.control()
    scene = replace(scene, dynamics="measured_240hz", bounce_profile="retention_low")
    parameters = np.array([5, 3, 1, 2, 20, -8, -1, -17, 4], float)
    fits = model.chain(scene, parameters)
    assert fits[0]["bounces"][0]["bounce_profile"] == "retention_low"
    np.testing.assert_array_equal(fits[0]["end_xyz"], fits[1]["start_xyz"])


@pytest.mark.parametrize("profile", list(dynamics.BOUNCE_PROFILES))
def test_explicit_global_correction_can_represent_reference_law(profile):
    theta = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0, 2.0, 0.0, 0.0])
    frames = np.arange(1.0, 30)
    expected = dynamics.simulate(theta, 1, frames, 25, "hard")
    factors = 1 / np.asarray(dynamics.BOUNCE_PROFILES[profile])
    actual = dynamics.simulate(
        theta, 1, frames, 25, "hard", bounce_profile=profile, rebound_scales=factors
    )
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-10)


@pytest.mark.parametrize("scales", [[0.79, 1], [1, 1.21], [1, float("nan")], [1], [1, 1, 1]])
def test_invalid_rebound_corrections_fail_before_simulation(scales):
    with pytest.raises(ValueError, match="rebound corrections"):
        dynamics.simulate(np.zeros(9), 1, np.array([1.0, 2.0]), 25, "hard", rebound_scales=scales)


def test_image_fit_recovers_shared_rebound_bias_in_known_spin_mechanism_control():
    original, _ = model.control()
    frames = np.arange(1.0, 37.0)
    truth = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0])
    scene = model.Scene(
        np.array([1.0, 36.0]),
        (frames,),
        (np.repeat(original.cameras[0][:1], len(frames), axis=0),),
        (np.zeros((len(frames), 2)),),
        np.array([[2.0, 0.0, 0.0]]),
        25.0,
        "hard",
        "measured_240hz",
    )
    scene = replace(
        scene,
        pixels=(model.image_residual(scene, truth).reshape(-1, 2),),
        bounce_profile="retention_high",
        rebound_mode="point_scales",
    )
    seed = np.r_[truth + [0.02, -0.04, 0.01, 0.1, -0.1, 0.1], 1.0, 1.0]
    result = model.fit(scene, seed, max_nfev=80)
    assert result["final_pixel_rms"] < 0.005
    np.testing.assert_allclose(result["rebound_scale_factors"], [1.0, 1 / 1.05], atol=0.01)
