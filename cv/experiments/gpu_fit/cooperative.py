"""Cooperative branches whose GPU solves share one launch (GPU-fit lane, research arm).

Question: the component search runs its depth branches one after another, so every GPU solve
pays the kernel's full latency alone. Can the independent branches run "at the same time" so
that each round of their solves is one batched launch, without changing what any branch
computes?

``run(function, items)`` runs ``function(item)`` for every item in its own thread, but only one
thread executes Python at a time: a thread gives up its turn only while it waits for a GPU solve
(``request``). When every live branch is waiting, all their pending solves go to the GPU in one
launch, and the branches resume one by one in item order. There is therefore no concurrent
Python execution (no data races), turns are deterministic, and each worker runs in a copy of the
caller's ``contextvars`` context. Results come back in item order. If a branch raises, the first
raising branch in item order is re-raised after every branch has stopped, and the results of
later branches are discarded, as a sequential loop would never have produced them (their
wall-clock and log side effects are the caller's to scope, see ``on_discard`` and
``BranchLog``).
Outside ``run`` (or nested inside a worker) ``request`` solves immediately.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

_ACTIVE = threading.local()


@dataclass
class _Worker:
    index: int
    turn: threading.Event = field(default_factory=threading.Event)
    state: str = "ready"  # ready | blocked | done
    request: Any = None
    response: Any = None
    result: Any = None
    error: BaseException | None = None


class _Scheduler:
    def __init__(self, launch: Callable[[list], list]):
        self.launch = launch
        self.back = threading.Event()
        self.rounds = 0
        self.batched = 0

    def request(self, worker: _Worker, payload):
        worker.request = payload
        worker.state = "blocked"
        worker.turn.clear()
        self.back.set()
        worker.turn.wait()
        if isinstance(worker.response, BaseException):
            raise worker.response
        return worker.response

    def give_turn(self, worker: _Worker) -> None:
        self.back.clear()
        worker.turn.set()
        self.back.wait()


def request(payload, solve_now: Callable[[Any], Any]):
    """Inside a cooperative worker: wait for the batched answer; otherwise ``solve_now``."""
    worker = getattr(_ACTIVE, "worker", None)
    scheduler = getattr(_ACTIVE, "scheduler", None)
    if worker is None or scheduler is None:
        return solve_now(payload)
    return scheduler.request(worker, payload)


def active() -> bool:
    return getattr(_ACTIVE, "worker", None) is not None


def run(
    function: Callable[[Any], Any],
    items: list,
    launch: Callable[[list], list],
    *,
    on_discard: Callable[[int], None] | None = None,
) -> tuple[list, dict, BaseException | None]:
    """``[function(item) for item in items]`` with batched GPU rounds.

    Returns ``(results, stats, error)``: on success every result and ``error`` None; otherwise
    the results of the items before the first raising item (in item order) and its exception,
    which the caller re-raises after applying those results.
    """
    if active() or len(items) <= 1:
        results = []
        for item in items:
            try:
                results.append(function(item))
            except BaseException as error:
                return results, {"rounds": None, "batched": 0}, error
        return results, {"rounds": None, "batched": 0}, None
    scheduler = _Scheduler(launch)
    workers = [_Worker(index) for index in range(len(items))]

    def body(worker: _Worker, item, context: contextvars.Context):
        worker.turn.wait()
        _ACTIVE.worker, _ACTIVE.scheduler = worker, scheduler
        try:
            worker.result = context.run(function, item)
        except BaseException as error:  # re-raised in item order by the caller
            worker.error = error
        finally:
            worker.state = "done"
            scheduler.back.set()

    threads = [
        threading.Thread(
            target=body,
            args=(worker, item, contextvars.copy_context()),
            name=f"cooperative-{worker.index}",
            daemon=True,
        )
        for worker, item in zip(workers, items, strict=True)
    ]
    for thread in threads:
        thread.start()
    while True:
        ready = [w for w in workers if w.state == "ready"]
        if ready:
            scheduler.give_turn(ready[0])
            continue
        blocked = [w for w in workers if w.state == "blocked"]
        if not blocked:
            break
        try:
            answers = launch([w.request for w in blocked])
        except BaseException as error:  # every waiting branch sees the launch failure
            answers = [error] * len(blocked)
        scheduler.rounds += 1
        scheduler.batched += len(blocked)
        for worker, answer in zip(blocked, answers, strict=True):
            worker.response = answer
            worker.state = "ready"
    for thread in threads:
        thread.join()
    stats = {"rounds": scheduler.rounds, "batched": scheduler.batched}
    for worker in workers:
        if worker.error is not None:
            if on_discard is not None:
                for later in workers[worker.index + 1 :]:
                    on_discard(later.index)
            stats["error_index"] = worker.index
            return [w.result for w in workers[: worker.index]], stats, worker.error
    return [worker.result for worker in workers], stats, None


class BranchLog:
    """``candidate_attempts.Attempts`` whose ``failures`` list is private to each cooperative
    branch (``begin()``), so a branch that reads ``failures[-1]`` sees its own entry; the caller
    appends each branch's list to the main one in item order."""

    def __init__(self, attempts_class):
        self._main: list = []
        self._local = threading.local()
        self._run = attempts_class.run

    @property
    def failures(self) -> list:
        return getattr(self._local, "failures", self._main)

    def begin(self) -> list:
        self._local.failures = []
        return self._local.failures

    def run(self, metadata, function, *args, **kwargs):
        return self._run(self, metadata, function, *args, **kwargs)
