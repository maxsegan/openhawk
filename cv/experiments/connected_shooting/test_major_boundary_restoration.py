"""Recover numerical closure without promoting a wrong physical branch or suffix."""

import time
from types import SimpleNamespace

import numpy as np

from cv.experiments.connected_shooting import labeled_prefix_boundary as boundary
from cv.experiments.connected_shooting.labeled_prefix_joint_impact import _jacobian


NEAR = np.array([0.6, 0.3602])


def evaluate(q):
    # A connected nonlinear endpoint, with the image optimum along its curve.
    return np.array([q[0] - 0.8]), dict(
        boundary_equality_m=np.array([q[1] - q[0] ** 2]),
        clearance_slacks_m=np.array([1.0]),
    )


def truncated_optimizer(monkeypatch, *, expire=None):
    def minimize(fun, initial, **kwargs):
        # A bounded solve makes useful progress but ends farther from closure.
        # Its last, cheaper iterate must not displace the near-feasible one.
        for q in (NEAR, np.array([0.8, 0.7])):
            fun(q)
            kwargs["callback"](q)
        if expire is not None:
            expire()
        return SimpleNamespace(
            x=np.array([0.8, 0.7]), status=9, message="Iteration limit reached", nit=3, nfev=2
        )

    monkeypatch.setattr(boundary, "minimize", minimize)


def solve(fun=evaluate, **kwargs):
    return boundary.solve(
        fun,
        np.zeros(2),
        np.zeros(2),
        np.ones(2),
        jacobian=_jacobian,
        maxiter=3,
        seconds=10,
        started=time.monotonic(),
        **kwargs,
    )


def test_curved_endpoint_closes_near_major_under_original_tolerance(monkeypatch):
    truncated_optimizer(monkeypatch)
    state, termination = solve()
    q = state["best_q"]
    assert state["best_cost"] < 0.041
    assert abs(q[1] - q[0] ** 2) <= boundary.TOLERANCE_M
    assert np.linalg.norm(q - NEAR) < 0.001
    receipt = termination["major_boundary_restoration"]
    assert receipt["status"] == "closed"
    assert receipt["candidate_limit"] == receipt["candidate_count"] == 1
    assert len(receipt["corrections"]) <= receipt["max_corrections"] == 3
    assert receipt["shares_original_wall_budget"]
    assert receipt["promoted"]
    assert termination["status"] == 9  # Do not misreport optimizer convergence.


def test_unreachable_branch_retains_the_feasible_source(monkeypatch):
    truncated_optimizer(monkeypatch)

    def disconnected(q):
        if q[0] > 0.1 and np.max(np.abs(q - NEAR)) > 2e-6:
            raise ValueError("original impact branch ends before boundary closure")
        return evaluate(q)

    state, termination = solve(disconnected)
    assert state["best_q"][0] < 0.01
    assert boundary.feasible_receipt(state["best_receipt"])
    assert termination["major_boundary_restoration"]["status"] != "closed"
    assert state["invalid"] > 0


def test_restoration_cannot_bypass_full_suffix_certification(monkeypatch):
    truncated_optimizer(monkeypatch)

    def certify(q, residual, receipt):
        return (residual, receipt) if q[0] < 0.1 else None

    state, termination = solve(certify_incumbent=certify)
    assert state["best_q"][0] < 0.1
    assert state["incumbent_certification_refusals"] > 0
    assert termination["major_boundary_restoration"]["status"] == "closed"
    assert not termination["major_boundary_restoration"]["promoted"]


def test_restoration_uses_remaining_original_wall_time(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(boundary.time, "monotonic", lambda: clock[0])
    truncated_optimizer(monkeypatch, expire=lambda: clock.__setitem__(0, 11.0))
    state, termination = solve()
    assert state["best_q"][0] < 0.01
    assert termination["kind"] == "optimizer" and termination["status"] == 9
    assert termination["major_boundary_restoration"]["status"] == "wall_timeout"
    assert termination["major_boundary_restoration"]["evaluations"] == 0


def test_custom_event_admission_prevents_restoration_seed(monkeypatch):
    truncated_optimizer(monkeypatch)

    def events(q):
        residual, receipt = evaluate(q)
        receipt["input_event_slacks_frames"] = np.array([0.1 - q[0]])
        return residual, receipt

    state, termination = solve(events)
    assert state["best_q"][0] < 0.1
    assert termination["major_boundary_restoration"]["candidate_count"] == 0


def test_out_of_bound_major_cannot_reverse_a_correction(monkeypatch):
    def outside(fun, initial, **kwargs):
        kwargs["callback"](np.array([1.0 + 1e-12, 0.7]))
        return SimpleNamespace(status=9, message="Iteration limit reached", nit=3, nfev=1)

    monkeypatch.setattr(boundary, "minimize", outside)
    _, termination = solve()
    assert termination["major_boundary_restoration"]["candidate_count"] == 0


def test_bound_and_fixed_coordinate_leave_other_closure_directions(monkeypatch):
    def fit(fun, initial, **kwargs):
        kwargs["callback"](np.array([1.0, 0.9, 0.0]))
        return SimpleNamespace(status=9, message="Iteration limit reached", nit=3, nfev=1)

    def connected(q):
        return np.array([q[0] - 1.0]), dict(
            boundary_equality_m=np.array([q[0] + q[1] - 2.0]),
            clearance_slacks_m=np.ones(1),
        )

    monkeypatch.setattr(boundary, "minimize", fit)
    state, termination = boundary.solve(
        connected,
        np.zeros(3),
        np.zeros(3),
        np.array([1.0, 1.0, 0.0]),
        jacobian=_jacobian,
        maxiter=3,
        seconds=10,
        started=time.monotonic(),
    )
    assert boundary.feasible_receipt(state["best_receipt"])
    assert state["best_q"][2] == 0.0
    assert state["best_cost"] < 0.01
    assert termination["major_boundary_restoration"]["promoted"]
