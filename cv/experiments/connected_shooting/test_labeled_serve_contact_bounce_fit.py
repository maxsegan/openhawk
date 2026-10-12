"""Latent contact/bounce epochs stay physical, exported and inside original intervals."""

from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    labeled_interval_block_fit as interval,
    labeled_serve_contact_bounce_fit as fitter,
    model,
)

CAMERA = np.array([[1000.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1000.0, 0.0], [0.0, 1.0, 0.0, 20.0]])
FPS = 25.0
CONTACT = 60.0


def _project(xyz):
    q = CAMERA @ np.r_[np.asarray(xyz, float), 1.0]
    return q[:2] / q[2]


def _scene(contacts, first_offset=2.0):
    frames = tuple(
        np.arange(int(np.ceil(a)) + (first_offset if i == 0 else 0), int(np.floor(b)) + 1, 2.0)
        for i, (a, b) in enumerate(zip(contacts[:-1], contacts[1:]))
    )
    scene = model.Scene(
        contact_frames=np.asarray(contacts, float),
        observation_frames=frames,
        cameras=tuple(np.repeat(CAMERA[None], len(f), axis=0) for f in frames),
        pixels=tuple(np.zeros((len(f), 2)) for f in frames),
        spin_parameters=np.zeros((4, 3)),
        fps=FPS,
        surface="hard",
        dynamics="measured_240hz",
        rebound_mode="point_scales",
    )
    scene.validate()
    return scene


def _pictures(scene, parameters):
    fitted = model.chain(scene, parameters)
    return replace(
        scene,
        pixels=tuple(np.asarray([_project(r) for r in f["positions"]]) for f in fitted),
    )


def fixture():
    """Four connected flights, exact projected pictures, one bounce per flight."""
    start = np.array([5.0, 2.0, 2.9])
    velocities = np.array(
        [[0.4, 19.0, -2.0], [-0.3, -18.0, 1.0], [0.2, 17.0, 2.0], [-0.3, -17.0, 1.5]]
    )
    truth = np.r_[start, velocities.ravel(), np.zeros(12), [1.0, 1.0]]
    probe = _scene([CONTACT, CONTACT + 24, CONTACT + 50, CONTACT + 76, CONTACT + 100])
    terminal = float(model.chain(probe, truth)[3]["bounces"][0]["frame"])
    scene = _pictures(_scene([CONTACT, CONTACT + 24, CONTACT + 50, CONTACT + 76, terminal]), truth)
    held_frames = tuple(f[:1] + 1.0 for f in scene.observation_frames)
    heldout = _pictures(
        replace(
            scene,
            observation_frames=held_frames,
            cameras=tuple(np.repeat(CAMERA[None], len(f), axis=0) for f in held_frames),
            pixels=tuple(np.zeros((len(f), 2)) for f in held_frames),
        ),
        truth,
    )
    fitted = model.chain(scene, truth)
    events = [
        dict(event_type="contact", frame=CONTACT, frame_interval=[CONTACT - 0.5, CONTACT + 0.5])
    ]
    for i, flight in enumerate(fitted):
        events.extend(
            dict(
                event_type="bounce",
                frame=float(b["frame"]),
                frame_interval=[b["frame"] - 0.4, b["frame"] + 0.4],
            )
            for b in flight["bounces"]
        )
        events.append(dict(event_type="contact", frame=float(scene.contact_frames[i + 1])))
    axes = np.zeros((sum(map(len, scene.observation_frames)), 2))
    axes[:, 0] = 1.0
    context = dict(
        scene=scene,
        heldout=heldout,
        bounces=tuple(np.asarray([b["frame"] for b in f["bounces"]]) for f in fitted),
        axes=axes,
        events=events,
        depth_bounds=(1.0, 3.0),
        termination_kind="terminal_bounce",
        players=[
            dict(player=f"p{i}", stature_m=1.9, court_centre_xy_m=f["start_xyz"][:2].tolist())
            for i, f in enumerate(fitted)
        ],
    )
    incoming = np.array([0.0, 0.0, -3.0])
    rows = []
    for frame in np.arange(CONTACT - 20, CONTACT - 1, 2.0):
        t = (frame - CONTACT) / FPS
        xyz = start + t * incoming + 0.5 * t * t * np.array([0.0, 0.0, -9.81])
        rows.append(
            dict(
                frame=float(frame),
                pixel=_project(xyz).tolist(),
                uncertainty_px=3.0,
                camera=CAMERA.tolist(),
            )
        )
    observations = dict(status="supported", contact_frame=CONTACT, rows=rows)
    feet = dict(side="deuce", court_xy_m=start[:2].tolist())
    toss = dict(
        shared_contact_frame=CONTACT,
        incoming_contact_velocity_mps=incoming.tolist(),
        release_seconds_before_contact=1.0,
    )
    return dict(
        context=context,
        truth=truth,
        observations=observations,
        feet=feet,
        toss=toss,
        release=[CONTACT - 30, CONTACT - 20],
    )


def test_first_flight_projector_exports_physical_root_into_original_chain():
    case = fixture()
    scene, truth = case["context"]["scene"], case["truth"]
    first = float(case["context"]["bounces"][0][0])
    projector = fitter.FirstFlightProjector(scene, [first - 0.4, first + 0.4], truth[5])
    for epoch in (first - 0.3, first + 0.25):
        exported, receipt = projector.project(truth, epoch)
        chain = model.chain(scene, exported)
        assert abs(chain[0]["bounces"][0]["frame"] - epoch) < 1e-6
        assert abs(receipt["error_frames"]) <= 1e-7
        mask = np.arange(len(truth)) != fitter.FIRST_VZ
        np.testing.assert_array_equal(exported[mask], truth[mask])
        assert exported[fitter.FIRST_VZ] != truth[5]
    with pytest.raises(interval.ChartError, match="outside original interval"):
        projector.project(truth, first + 1.0)


def _run(case, **kwargs):
    return fitter.fit(
        case["context"],
        case["truth"],
        case["observations"],
        case["feet"],
        case["release"],
        None,
        contact_interval=[CONTACT - 0.5, CONTACT + 0.5],
        bounce_interval=[
            float(case["context"]["bounces"][0][0]) - 0.4,
            float(case["context"]["bounces"][0][0]) + 0.4,
        ],
        maxiter=2,
        seconds=120,
        **kwargs,
    )


def test_fit_keeps_native_frames_and_epochs_inside_original_intervals():
    case = fixture()
    original = case["context"]["scene"]
    first = float(case["context"]["bounces"][0][0])
    perturbed = case["truth"].copy()
    perturbed[5] += 0.6
    ctx, result = _run(case, initial_parameters=perturbed, initial_toss=case["toss"])
    best = result["best"]
    assert CONTACT - 0.5 <= result["contact_epoch"] <= CONTACT + 0.5
    assert first - 0.4 <= result["bounce_epoch"] <= first + 0.4
    assert result["export"]["reproduced"] and result["export"]["mismatch_frames"] <= 1e-6
    assert ctx["scene"].contact_frames[0] == result["contact_epoch"]
    np.testing.assert_array_equal(ctx["scene"].contact_frames[1:], original.contact_frames[1:])
    for fitted, native in zip(
        result["fitting_scene"]["observation_frames"], original.observation_frames
    ):
        assert set(native).issubset(set(np.asarray(fitted).tolist()))
    assert all(r["frame"] < CONTACT - 0.5 for r in best["shared_toss"]["native_projection"])
    assert best["shared_toss"]["shared_contact_frame"] == result["contact_epoch"]
    assert np.isfinite(best["cost"]) and best["rms_px"] < 5.0
    # Fourth launch block (velocity 12:15, spin 24:27) and rebound scalars stay source-exact.
    for block in (slice(12, 15), slice(24, 27), slice(27, 29)):
        np.testing.assert_array_equal(best["parameters"][block], case["truth"][block])
    assert np.any(best["parameters"][3:12] != case["truth"][3:12])


def test_fixed_contact_epoch_control_holds_epoch_and_rejects_foreign_intervals():
    case = fixture()
    ctx, result = _run(case, fixed_contact_epoch=CONTACT + 0.2)
    assert result["contact_epoch"] == CONTACT + 0.2
    assert ctx["scene"].contact_frames[0] == CONTACT + 0.2
    assert result["policy"]["fixed_contact_epoch"] == CONTACT + 0.2
    with pytest.raises(ValueError, match="inside the original label interval"):
        fitter.fit(
            case["context"],
            case["truth"],
            case["observations"],
            case["feet"],
            case["release"],
            None,
            contact_interval=[CONTACT - 2, CONTACT],
            bounce_interval=[80, 81],
            maxiter=1,
        )
