import json

from cv.pipeline.reconstruction import load_event_boundaries, load_event_hypotheses


def test_reconstruction_filters_gate_held_event_emissions(tmp_path) -> None:
    path = tmp_path / "emissions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "clip": "match__pt0001",
                    "event_type": "contact",
                    "frame": 10,
                    "confidence": 0.9,
                    "point_gate_verdict": "hold",
                    "point_grammar": {"in_play": True},
                },
                {
                    "clip": "match__pt0002",
                    "event_type": "bounce",
                    "frame": 20,
                    "confidence": 0.8,
                    "point_gate_verdict": "retain",
                    "point_grammar": {"in_play": True},
                },
            ]
        )
    )

    loaded = load_event_boundaries(path)

    assert "match__pt0001" not in loaded
    assert loaded["match__pt0002"][0]["event_type"] == "bounce"


def test_reconstruction_defaults_to_in_play_with_lossless_bypass(tmp_path) -> None:
    path = tmp_path / "emissions.json"
    rows = [
        {
            "clip": "match__pt0001",
            "event_type": "contact",
            "frame": 10,
            "point_gate_verdict": "retain",
            "point_grammar": {"in_play": True},
        },
        {
            "clip": "match__pt0001",
            "event_type": "bounce",
            "frame": 20,
            "point_gate_verdict": "retain",
            "point_grammar": {"in_play": False},
        },
    ]
    path.write_text(json.dumps(rows))

    assert len(load_event_boundaries(path)["match__pt0001"]) == 1
    assert len(load_event_boundaries(path, include_dead_time_emissions=True)["match__pt0001"]) == 2


def test_reconstruction_loads_leaky_hypotheses_separately(tmp_path) -> None:
    path = tmp_path / "hypotheses.json"
    path.write_text(
        json.dumps(
            [
                {
                    "clip": "match__pt0001",
                    "event_type": "contact",
                    "frame": 12,
                    "probability": 0.08,
                    "point_grammar": {"in_play": True},
                }
            ]
        )
    )

    loaded = load_event_hypotheses(path)

    assert loaded["match__pt0001"][0]["probability"] == 0.08
