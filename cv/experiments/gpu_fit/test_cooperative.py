"""Cooperative branches: item-ordered results, one launch per round, first error in item order."""

import contextvars

import pytest

from cv.experiments.gpu_fit import cooperative

MARK = contextvars.ContextVar("mark", default="unset")


def test_rounds_order_context_and_errors():
    launches = []

    def launch(requests):
        launches.append(list(requests))
        return [r * 10 for r in requests]

    trace = []

    def branch(k):
        trace.append(("start", k, MARK.get()))
        a = cooperative.request(k, lambda r: -1)
        b = cooperative.request(k + 100, lambda r: -1)
        trace.append(("end", k))
        return a + b

    token = MARK.set("parent")
    try:
        results, stats, error = cooperative.run(branch, [0, 1, 2], launch)
    finally:
        MARK.reset(token)
    assert error is None
    assert results == [1000, 1020, 1040]
    assert launches == [[0, 1, 2], [100, 101, 102]]
    assert stats == {"rounds": 2, "batched": 6}
    assert [t for t in trace if t[0] == "start"] == [("start", k, "parent") for k in range(3)]

    def failing(k):
        cooperative.request(k, lambda r: -1)
        if k >= 1:
            raise ValueError(f"branch {k}")
        return k

    results, stats, error = cooperative.run(failing, [0, 1, 2], launch)
    assert results == [0] and isinstance(error, ValueError) and str(error) == "branch 1"
    # outside run the request is solved immediately
    assert cooperative.request(5, lambda r: r + 1) == 6


def test_branch_log_is_private_per_branch():
    class Attempts:
        def run(self, metadata, function, *args):
            try:
                return function(*args)
            except ValueError as error:
                self.failures.append({**metadata, "reason": str(error)})
                return None

    log = cooperative.BranchLog(Attempts)

    def branch(k):
        mine = log.begin()
        cooperative.request(k, lambda r: r)
        log.run({"k": k}, lambda: (_ for _ in ()).throw(ValueError(str(k))))
        assert log.failures[-1]["k"] == k
        return mine

    results, _, error = cooperative.run(branch, [0, 1], lambda rs: rs)
    assert error is None and [r[0]["k"] for r in results] == [0, 1]
    assert log.failures == []
    with pytest.raises(AssertionError):
        assert log.failures
