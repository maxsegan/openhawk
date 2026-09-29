import pytest

from cv.experiments.connected_shooting.labeled_preparation_recovery_association import (
    implied_first_side,
)


def test_two_independent_later_contacts_imply_serve_side():
    assert implied_first_side([(1, "far"), (2, "near")]) == "near"
    assert implied_first_side([(2, "far"), (3, "near")]) == "far"
    for observations in ([(1, "far")], [(1, "far"), (2, "far")], [(1, "far"), (1, "far")]):
        with pytest.raises(ValueError):
            implied_first_side(observations)
