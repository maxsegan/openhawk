"""Formula-extension controls do not change the default physical selector."""

from dataclasses import asdict

import numpy as np
import pytest

from physics import impact, bounce_reference


@pytest.mark.parametrize("surface", ["hard", "clay", "grass"])
@pytest.mark.parametrize("velocity", [(30.0, 10.0), (50.0, 5.0), (12.0, 18.0)])
@pytest.mark.parametrize("spin", [-300.0, 0.0, 200.0, 600.0])
def test_selected_formula_is_bit_identical_to_default(surface, velocity, spin):
    original = impact.court_bounce(*velocity, spin, surface)
    selected = impact.court_bounce(*velocity, spin, surface, regime_override=original.regime)
    explicit_default = impact.court_bounce(*velocity, spin, surface, regime_override=None)
    assert asdict(original) == asdict(selected) == asdict(explicit_default)


def test_contrary_formula_is_explicitly_not_the_default_bounce():
    original = impact.court_bounce(30, 10, 0)
    contrary = "grip" if original.regime == "slide" else "slide"
    forced = impact.court_bounce(30, 10, 0, regime_override=contrary)
    assert forced.regime == contrary
    assert forced.w2 != original.w2
    with pytest.raises(ValueError, match="surrogate"):
        impact.court_bounce(30, 10, 0, regime_override="automatic")


def test_measured_translation_does_not_change_when_only_spin_formula_is_extended():
    velocity, spin = np.array([30.0, 0.0, -10.0]), np.zeros(3)
    original = bounce_reference.court_bounce(velocity, spin, "hard")
    contrary = "grip" if original.regime == "slide" else "slide"
    surrogate = bounce_reference.court_bounce(velocity, spin, "hard", spin_regime_override=contrary)
    np.testing.assert_array_equal(original.velocity, surrogate.velocity)
    assert not np.array_equal(original.spin, surrogate.spin)
    with pytest.raises(ValueError, match="surrogate"):
        bounce_reference.court_bounce(velocity, spin, "hard", spin_regime_override="invalid")
