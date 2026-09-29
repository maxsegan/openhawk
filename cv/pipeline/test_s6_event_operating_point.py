"""The default-off consumer-side decoder operating point."""

from __future__ import annotations

from copy import deepcopy

import pytest

from cv.pipeline.s6_event_operating_point import (
    OPERATING_POINT_DEFAULT,
    RESTATEMENT_FIELD,
    event_operating_point,
    restatable,
    restate,
)
from cv.pipeline.s6_labeled_stage import shared_settings


def band_row(marginal: float, **overrides) -> dict:
    row = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "bounce",
        "frame": 40.0,
        "abstain": True,
        "model_abstain": True,
        "acceptance_marginal": marginal,
        "path_marginal": marginal,
        "decision_threshold": 0.9907168846560898,
        "abstention_gap": 1.0 - marginal,
    }
    row.update(overrides)
    return row


def test_off_is_the_default_and_returns_the_document_unchanged() -> None:
    assert event_operating_point(OPERATING_POINT_DEFAULT) is None
    assert event_operating_point(None) is None
    document = [band_row(0.5), band_row(0.99, abstain=False, model_abstain=False)]
    original = deepcopy(document)
    output, census = restate(document, None)
    assert output == original
    assert census["declared_floor"] is None and census["restated_rows"] == 0


def test_a_band_row_at_or_above_the_floor_becomes_an_accepted_emission() -> None:
    output, census = restate([band_row(0.83), band_row(0.41)], 0.8)
    assert census["restatable_rows"] == 2 and census["restated_rows"] == 1
    assert output[0]["abstain"] is False and output[0]["model_abstain"] is False
    assert output[0][RESTATEMENT_FIELD]["acceptance_marginal"] == 0.83
    assert output[0][RESTATEMENT_FIELD]["producer_abstained"] is True
    assert output[1]["abstain"] is True and RESTATEMENT_FIELD not in output[1]
    # Everything but the decision is the producer's row, byte for byte.
    assert {k: v for k, v in output[0].items() if k not in (
        "abstain", "model_abstain", RESTATEMENT_FIELD
    )} == {k: v for k, v in band_row(0.83).items() if k not in ("abstain", "model_abstain")}


def test_a_gate_held_refusal_is_never_restated() -> None:
    """`tracking_arc_abstained` is a gate verdict, not an operating point."""
    held = band_row(0.999, model_abstain=False, gate_held=True,
                    point_gate_failure_reasons=["tracking_arc_abstained"])
    assert restatable(held) is False
    output, census = restate([held], 0.2)
    assert census["restatable_rows"] == 0 and census["restated_rows"] == 0
    assert output[0]["abstain"] is True


def test_an_accepted_row_and_a_point_end_are_untouched() -> None:
    accepted = band_row(0.999, abstain=False, model_abstain=False)
    ending = band_row(0.999, event_type="point_end", abstain=False, model_abstain=False)
    output, census = restate([accepted, ending], 0.2)
    assert census["restatable_rows"] == 0
    assert output == [accepted, ending]


def test_the_operating_point_may_only_be_lowered() -> None:
    with pytest.raises(ValueError, match="only LOWER"):
        restate([band_row(0.5)], 0.999)


def test_an_unusable_declaration_raises() -> None:
    for value in ("on", True, float("nan"), 1.5, -0.1):
        with pytest.raises(ValueError, match="finite probability"):
            event_operating_point(value)


def test_the_key_stays_absent_unless_the_policy_declares_it() -> None:
    policy = {"contact_prefix_scope": "off", "contact_components": "off"}
    assert "automatic_event_operating_point" not in shared_settings(policy)
    settings = shared_settings(policy | {"automatic_event_operating_point": 0.2})
    assert settings["automatic_event_operating_point"] == 0.2
    assert shared_settings(policy | {"automatic_event_operating_point": "off"})[
        "automatic_event_operating_point"
    ] == "off"
    with pytest.raises(ValueError, match="finite probability"):
        shared_settings(policy | {"automatic_event_operating_point": "on"})
