import copy

import pytest

from cv.experiments.connected_shooting import serve_speed_witness as witness


def candidate(speed: float) -> dict:
    return {
        "measurement": {"fit": {"parameters": [0, 0, 2.7, speed, 0, 0]}},
        "evidence": {
            "checks": {},
            "death_reasons": [],
            "survived": True,
            "input_only_rank_score": 10.0,
        },
    }


def test_soft_speed_score_keeps_known_depth_family() -> None:
    graphic = {
        "abstained": False,
        "speed_mps": 132 / 3.6,
        "visible_refresh_observed": True,
        "previous_stable_value": 175,
    }
    rows = [candidate(value) for value in (36.122, 38.395, 39.603)]
    results = [witness.add_to_candidate(row, graphic) for row in rows]
    assert [row["evidence"]["survived"] for row in results] == [True, True, True]
    penalties = [row["evidence"]["serve_speed_witness"]["selector_penalty"] for row in results]
    assert penalties == sorted(penalties)
    relation = results[0]["evidence"]["serve_speed_witness"]["relation"]
    assert relation["resolved_sigma_mps"] == pytest.approx(1.7755, rel=1e-3)
    assert relation["sigma_formula"].startswith("sqrt(")
    assert not results[0]["evidence"]["serve_speed_witness"]["hard_gate"]


def test_speed_scaled_sigma_is_about_two_to_two_point_five_at_fifty() -> None:
    relation = witness.RadarRelation()
    assert relation.sigma_mps(50.0) == pytest.approx(5**0.5)


def test_abstention_does_not_kill_branch() -> None:
    row = copy.deepcopy(candidate(49.0))
    witness.add_to_candidate(row, {"abstained": True, "abstention_reason": "no_read"})
    assert row["evidence"]["survived"]
    assert row["evidence"]["serve_speed_witness"]["selector_penalty"] == pytest.approx(0)


def test_unobserved_refresh_is_not_a_hard_gate() -> None:
    row = candidate(50.0)
    graphic = {
        "abstained": False,
        "speed_mps": 35.0,
        "visible_refresh_observed": False,
        "previous_stable_value": None,
    }
    witness.add_to_candidate(row, graphic)
    assert row["evidence"]["survived"]
    assert row["evidence"]["serve_speed_witness"]["abstained"]


def test_persistent_previous_value_is_not_a_hard_gate() -> None:
    row = candidate(50.0)
    graphic = {
        "abstained": False,
        "speed_mps": 35.0,
        "visible_refresh_observed": False,
        "previous_stable_value": 175,
    }
    witness.add_to_candidate(row, graphic)
    assert row["evidence"]["survived"]


def test_frozen_label_speed_requires_observed_value_change() -> None:
    document = witness.from_label_document(
        {
            "serve_speed_evidence": {
                "status": "visible",
                "value": 121,
                "unit": "mph",
                "schema": "source_speed_v1",
                "annotation_origin": "agent",
                "contact_event_id": "serve",
                "observations": [
                    {"frame": 10, "role": "preceding_display", "value": 106},
                    {"frame": 20, "role": "current_serve", "value": 121},
                ],
            }
        }
    )
    assert document["reading"]["visible_refresh_observed"]
    assert document["reading"]["speed_mps"] == pytest.approx(121 * 0.44704)


def test_unresolved_previous_digits_do_not_break_speed_abstention() -> None:
    document = witness.from_label_document(
        {
            "serve_speed_evidence": {
                "status": "visible",
                "value": 182,
                "unit": "km/h",
                "observations": [
                    {"frame": 10, "role": "preceding_display", "value": None},
                    {"frame": 20, "role": "current_serve", "value": 182},
                ],
            }
        }
    )
    assert not document["reading"]["visible_refresh_observed"]
