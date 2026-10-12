"""Fitter tests here simulate the wall clock (they patch ``time.monotonic``) to drive deadlines,
so they run the wall-clock budget path. Work-counted budgets (``TENNIS_S6_WORK_BUDGET``, on by
default) are tested in ``cv/pipeline/test_s6_work_clock.py``."""

import pytest


@pytest.fixture(autouse=True)
def _wall_clock_budgets(monkeypatch):
    monkeypatch.setenv("TENNIS_S6_WORK_BUDGET", "off")
