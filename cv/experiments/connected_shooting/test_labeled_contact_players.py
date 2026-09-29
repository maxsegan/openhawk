"""The striker identity comes from the labels, mapped onto the roster, never guessed (3D-009)."""

import pytest

from cv.experiments.connected_shooting import agent_whole_point_search as search


def _labels(rows):
    return {"events": {"records": rows}}


def test_hitter_field_is_read_and_overrides_the_alternating_cycle():
    labels = _labels(
        [
            {"event_type": "contact", "frame": 180.5, "hitter": "Alcaraz"},
            {"event_type": "bounce", "frame": 189.5},
            {"event_type": "contact", "frame": 197.5, "hitter": "Djokovic"},
            {"event_type": "contact", "frame": 220.0, "hitter": "Djokovic"},
        ]
    )
    contacts = [{"frame": 180.5}, {"frame": 197.5}, {"frame": 220.0}]
    # The configured cycle assumed Djokovic served; the labels say Alcaraz did, and that
    # Djokovic hit twice in a row (an overhead after a short ball is legal tennis).
    assert search.labeled_contact_players(labels, contacts, ["Djokovic", "Alcaraz"]) == [
        "Alcaraz",
        "Djokovic",
        "Djokovic",
    ]


def test_full_names_map_onto_roster_surnames():
    labels = _labels(
        [
            {"event_type": "contact", "frame": 1.0, "hitter": "Jasmine Paolini"},
            {"event_type": "contact", "frame": 2.0, "player": "Swiatek"},
        ]
    )
    contacts = [{"frame": 1.0}, {"frame": 2.0}]
    assert search.labeled_contact_players(labels, contacts, ["Swiatek", "Paolini"]) == [
        "Paolini",
        "Swiatek",
    ]


def test_unknown_or_ambiguous_hitter_fails_closed():
    labels = _labels([{"event_type": "contact", "frame": 1.0, "hitter": "Nadal"}])
    with pytest.raises(ValueError):
        search.labeled_contact_players(labels, [{"frame": 1.0}], ["Swiatek", "Paolini"])
    labels = _labels([{"event_type": "contact", "frame": 1.0, "hitter": "Williams"}])
    with pytest.raises(ValueError):
        search.labeled_contact_players(labels, [{"frame": 1.0}], ["S Williams", "V Williams"])


def test_unlabeled_contacts_still_use_the_configured_cycle():
    labels = _labels(
        [{"event_type": "contact", "frame": 1.0}, {"event_type": "contact", "frame": 2.0}]
    )
    contacts = [{"frame": 1.0}, {"frame": 2.0}]
    assert search.labeled_contact_players(labels, contacts, ["A", "B"]) == ["A", "B"]
