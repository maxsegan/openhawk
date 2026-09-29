from cv.pipeline.event_inference import apply_production_scope


def test_scope_emits_held_points_with_gate_flags(tmp_path) -> None:
    active = tmp_path / "active.json"
    gate = tmp_path / "gate.json"
    active.write_text('{"m/pt0001":{"fps":25,"event_spans":[[10,20]]}}')
    gate.write_text(
        '{"points":1,"held":1,"retained":0,"rows":'
        '[{"match_id":"m","clip":"pt0001","decision":"hold"}]}'
    )
    rows = [{"clip": "m__pt0001", "proposal_frame": 15.0}]

    apply_production_scope(rows, active, gate)

    assert rows[0]["production_scope"] is True
    assert rows[0]["point_gate_verdict"] == "hold"
    assert rows[0]["point_gate_failure_reasons"] == []
    assert rows[0]["gate_held"] is True
