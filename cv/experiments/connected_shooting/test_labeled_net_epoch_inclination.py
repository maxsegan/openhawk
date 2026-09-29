from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_net_epoch_fit as fit


def test_inclination_requires_explicit_mesh_and_finite_increasing_bounds():
    for mesh, bounds in [(False, (0, 1)), (True, (1, 0)), (True, (0, np.nan))]:
        with pytest.raises(ValueError, match="inclination bounds"):
            fit.fit(
                {},
                np.zeros(11),
                0.25,
                maxiter=1,
                seconds=1,
                mesh_response=mesh,
                inclination_bounds=bounds,
            )


def test_optional_coordinate_and_actual_response_receipt(monkeypatch):
    scene = SimpleNamespace(
        pixels=(np.zeros((2, 2)),),
        contact_frames=np.array([0.0, 20.0]),
        fps=50.0,
        surface="hard",
        bounce_profile="nominal",
    )
    context = dict(
        scene=scene,
        heldout=None,
        bounces=[[]],
        axes=None,
        events=[dict(event_type="net_hit", frame=10.0, frame_interval=[9.0, 11.0])],
        termination_kind="terminal",
    )
    source = np.array([4.0, 4.0, 1.0, 1.0, 20.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    state = {"angle": 0.0}
    monkeypatch.setattr(fit.full, "merge_scene", lambda *args: (scene, None, []))
    monkeypatch.setattr(fit.full.model, "chain", lambda *args: [dict(start_xyz=source[:3])])
    monkeypatch.setattr(
        fit.chart, "project_net_velocity", lambda theta, *args, **kwargs: (theta, {})
    )
    monkeypatch.setattr(
        fit.interval.block.event_constraints,
        "evaluate",
        lambda *args, **kwargs: ([], np.zeros(1), []),
    )
    monkeypatch.setattr(
        fit.full.exposure, "prediction", lambda *args, **kwargs: np.full((2, 2), 1 - state["angle"])
    )

    @contextmanager
    def using(response):
        old = state["angle"]
        state["angle"] = response.normal_angle
        try:
            yield
        finally:
            state["angle"] = old

    monkeypatch.setattr(fit.tape, "using_response", using)
    widths = []

    def solve(fun, x, **kwargs):
        widths.append(len(x))
        candidate = x.copy()
        if len(x) == 9:
            candidate[-1] = 0.2
        fun(candidate)
        return SimpleNamespace(success=True, message="controlled evaluation", nfev=1, njev=0)

    monkeypatch.setattr(fit, "least_squares", solve)
    default = fit.fit(context, source, 0.25, maxiter=1, seconds=10, mesh_response=True)
    inclined = fit.fit(
        context,
        source,
        0.25,
        maxiter=1,
        seconds=10,
        mesh_response=True,
        inclination_bounds=(0.0, np.pi / 2),
    )
    assert widths == [8, 9]
    assert default["best"]["response"]["normal_angle"] == 0
    assert inclined["best"]["response"]["normal_angle"] == 0.2
    np.testing.assert_array_equal(default["best"]["parameters"], inclined["best"]["parameters"])
    assert state["angle"] == 0

    class ExplicitResponse(fit.tape.TapeResponse):
        pass

    custom = fit.fit(
        context,
        source,
        0.25,
        maxiter=1,
        seconds=10,
        mesh_response=True,
        inclination_bounds=(0.0, np.pi / 2),
        response_type=ExplicitResponse,
    )
    assert custom["policy"]["response_type"].endswith("ExplicitResponse")
    assert custom["best"]["response"]["normal_angle"] == 0.2
