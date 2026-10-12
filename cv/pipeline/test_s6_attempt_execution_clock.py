"""The seed-failure fallback hands the orchestrator a start on the budget clock."""

import json
import time

from cv.pipeline import s6_attempt_execution as execution


def test_seed_failure_fallback_starts_on_the_budget_clock(tmp_path, monkeypatch) -> None:
    from cv.experiments.connected_shooting import candidate_attempts
    from cv.pipeline import s6_component_orchestration, s6_component_scope
    from cv.pipeline import s6_contact_components as components

    # A work clock far from time.monotonic(), as on a long-up box with work budgets on.
    monkeypatch.setattr(execution.work_clock, "now", lambda site="budget": 5.0)
    monkeypatch.setattr(execution.stage, "validated_row", lambda row: (None, {}))
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"attempts": [{}]}))
    monkeypatch.setattr(execution.stage, "resolve", lambda path: packet)

    class Process:
        pid = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(execution.subprocess, "Popen", Process)
    monkeypatch.setattr(candidate_attempts, "should_fallback_to_components", lambda *a: True)
    monkeypatch.setattr(s6_component_scope, "active", lambda attempt: False)
    monkeypatch.setattr(components, "as_unresolved_source", lambda packet: packet)
    seen = {}

    def run_source(row, output, policy, remaining, **kwargs):
        seen["started"] = kwargs["started"]
        seen["remaining"] = remaining
        return {"key": row["key"], "status": "ok", "wall_seconds": -1.0}

    monkeypatch.setattr(s6_component_orchestration, "run_source", run_source)
    row = {"key": "a1", "surface": "hard", "declared_flights": 2, "packet": "packet.json"}
    before = time.monotonic()
    result = execution.execute(row, tmp_path / "out", {}, 300.0)
    assert seen["started"] == 5.0
    assert seen["remaining"] == 300.0
    assert 0.0 <= result["wall_seconds"] <= time.monotonic() - before
