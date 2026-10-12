"""Work-counted S6 budgets: the deadline binds at the same work whatever the wall clock does."""

import pytest

from cv.experiments.connected_shooting import search_budget
from cv.pipeline import s6_work_clock as work_clock


def _checks_until_deadline(monkeypatch, wall):
    monkeypatch.setattr(work_clock.time, "monotonic", wall)
    budget = search_budget.Budget(10 * work_clock.UNIT_SECONDS, None)
    checks = 0
    with pytest.raises(search_budget.SearchDeadline):
        while True:
            checks += 1
            budget.check()
            work_clock.advance(2.5)  # GPU work between checks
    return checks, budget.receipt()["exhausted"]


def test_work_budget_ignores_wall_time(monkeypatch):
    monkeypatch.setenv(work_clock.SWITCH, "on")
    fast = _checks_until_deadline(monkeypatch, lambda: 0.0)
    slow = iter(range(0, 10**9, 1000))
    loaded = _checks_until_deadline(monkeypatch, lambda: float(next(slow)))
    assert fast == loaded == (4, True)  # check k sees 1 + 3.5 (k - 1) units; >= 10 at k = 4


def test_switch_off_keeps_wall_time(monkeypatch):
    monkeypatch.setenv(work_clock.SWITCH, "off")
    monkeypatch.setattr(work_clock.time, "monotonic", lambda: 42.0)
    before = work_clock.units()
    assert work_clock.now() == 42.0
    work_clock.advance(5)
    assert work_clock.units() == before
    assert work_clock.receipt() is None and work_clock.kill_factor() == 1.0


def test_optional_branches_read_the_budget_clock(monkeypatch):
    from types import SimpleNamespace

    from cv.pipeline import s6_optional_contacts as optional

    monkeypatch.setenv(work_clock.SWITCH, "on")
    monkeypatch.setattr(work_clock.time, "monotonic", lambda: 1.0e6)  # wall clock far ahead
    budget = SimpleNamespace(seconds=30, started=work_clock.now(), exhausted=False)
    proposal = {"name": "candidate", "events": []}
    optional.fit_alternatives(lambda *_: {"coarse": [], "refined": []}, [proposal], 50, budget, 90)
    assert budget.exhausted is False


def test_cpu_refine_reads_are_charged_by_problem_size(monkeypatch):
    import importlib

    from cv.pipeline import s6_work_clock as clock

    monkeypatch.setenv("TENNIS_S6_GPU_FIT", "off")
    monkeypatch.delenv("TENNIS_S6_WORK_UNIT_SECONDS", raising=False)
    cpu = importlib.reload(clock)
    try:
        assert cpu.UNIT_SECONDS == 0.0004
        small, large = cpu.refine_read_units(1, 30), cpu.refine_read_units(8, 680)
        assert (
            small == 1.0 and large > small
        )  # floor of one read; more flights and frames cost more
        assert abs(large * cpu.UNIT_SECONDS - (0.000285 + 8 * 0.0000845 + 680 * 0.00000044)) < 1e-12
        monkeypatch.setenv("TENNIS_S6_GPU_FIT", "on")
        assert (
            importlib.reload(clock).refine_read_units(8, 680) == 1.0
        )  # GPU path keeps its calibration
    finally:
        monkeypatch.setenv("TENNIS_S6_GPU_FIT", "off")
        importlib.reload(clock)
