"""Evidence-bound net cord: a pixel set, not a searched velocity. Abstain is tape clip."""

from __future__ import annotations

import numpy as np
import pytest

from cv.experiments.connected_shooting import camera_geometry, net_cord_evidence as evidence
from cv.pipeline import net_cord_response as cord
from cv.pipeline.physics_knot_solver import NET_X_CENTER, net_tape_height


def _camera():
    return np.array([[800.0, 0.0, 0.0, 0.0], [0.0, 0.0, 800.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def _pixel(camera, xyz):
    return camera_geometry.project(camera.reshape(1, 3, 4), np.asarray(xyz, float).reshape(1, 3))[0]


def _samples(contact, t0, velocity, fps, frames):
    camera = _camera()
    outgoing = []
    for frame in frames:
        dt = (frame - t0) / fps
        xyz = np.asarray(contact, float) + dt * np.asarray(velocity, float)
        xyz[2] -= 0.5 * evidence.GRAVITY * dt**2
        outgoing.append(
            {
                "frame": float(frame),
                "pixel": _pixel(camera, xyz),
                "camera": camera,
                "source": "ball_row",
            }
        )
    return camera, outgoing


def test_clear_outgoing_pixels_bound_the_velocity_without_a_search(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("evidence bound must not search v_out")

    monkeypatch.setattr(evidence.np.linalg, "lstsq", evidence.np.linalg.lstsq)
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "least_squares", explode)
    monkeypatch.setattr("cv.pipeline.net_cord_response.least_squares", explode)

    contact = np.array([NET_X_CENTER, evidence.NET_Y, net_tape_height(NET_X_CENTER)])
    incoming = np.array([3.0, -16.0, 1.0])
    true = cord.project_to_box(incoming, np.array([1.0, 5.0, 0.3]))
    fps = 50.0
    t0 = 100.0
    camera, outgoing = _samples(contact, t0, true, fps, [101, 102, 103, 104])
    bound = evidence.derive_outgoing_bound(
        net_frame=t0,
        contact_pixel=_pixel(camera, contact),
        contact_camera=camera,
        outgoing=outgoing,
        fps=fps,
        incoming_velocity=incoming,
        h_tol=cord.H_TOL_WIDE_M,
    )
    assert bound.admitted, bound.reason
    assert bound.reason == "outgoing_pixels"
    assert bound.pixel_rms_px < 5.0
    np.testing.assert_allclose(bound.v_center, true, atol=0.5)
    assert cord.in_pixel_set(bound, true)
    assert cord.in_pixel_set(bound, bound.v_center)

    with cord.using_evidence((bound,), h_tol=cord.H_TOL_WIDE_M):
        kept = cord.constrain_outgoing(incoming, true, t0)
        np.testing.assert_allclose(kept, true, atol=1e-6)
        centre = cord.constrain_outgoing(incoming, np.asarray(bound.v_center), t0)
        np.testing.assert_allclose(centre, bound.v_center, atol=1e-6)
        # A law velocity already inside the pixel set is not collapsed to the centre.
        nudge = np.asarray(bound.v_center, float) + np.array([0.4, 0.0, 0.0])
        assert cord.in_pixel_set(bound, nudge)
        assert cord.in_box(incoming, nudge)
        np.testing.assert_allclose(cord.constrain_outgoing(incoming, nudge, t0), nudge, atol=1e-6)
        far = np.array([40.0, 20.0, -10.0])
        assert not cord.in_pixel_set(bound, far)
        moved = cord.constrain_outgoing(incoming, far, t0)
        assert np.linalg.norm(moved - far) > 1.0
        assert np.linalg.norm(moved - bound.v_center) < np.linalg.norm(far - bound.v_center)


def test_no_outgoing_evidence_abstains_to_the_tape_clip():
    contact = np.array([NET_X_CENTER, evidence.NET_Y, net_tape_height(NET_X_CENTER)])
    camera = _camera()
    incoming = np.array([2.0, -12.0, 0.5])
    bound = evidence.derive_outgoing_bound(
        net_frame=10.0,
        contact_pixel=_pixel(camera, contact),
        contact_camera=camera,
        outgoing=[],
        fps=50.0,
        incoming_velocity=incoming,
        h_tol=cord.H_TOL_WIDE_M,
    )
    assert bound.admitted is False
    assert bound.reason == "no_outgoing_observations"
    law = cord.tape_clip_seed(incoming)
    with cord.using_evidence((bound,), h_tol=cord.H_TOL_WIDE_M):
        assert cord.uses_tape_band(10.0) is False
        np.testing.assert_array_equal(cord.constrain_outgoing(incoming, law, 10.0), law)
    short = evidence.derive_outgoing_bound(
        net_frame=10.0,
        contact_pixel=_pixel(camera, contact),
        contact_camera=camera,
        outgoing=[
            {
                "frame": 11.0,
                "pixel": _pixel(camera, contact + np.array([0.02, 0.1, 0.0])),
                "camera": camera,
                "source": "ball_row",
            },
            {
                "frame": 12.0,
                "pixel": _pixel(camera, contact + np.array([0.04, 0.2, 0.0])),
                "camera": camera,
                "source": "ball_row",
            },
        ],
        fps=50.0,
        incoming_velocity=incoming,
        h_tol=cord.H_TOL_WIDE_M,
    )
    assert short.admitted is False
    assert short.reason == "too_few_observations"


def test_a_flight_whose_tail_has_no_samples_abstains():
    contact = np.array([NET_X_CENTER, evidence.NET_Y, net_tape_height(NET_X_CENTER)])
    camera = _camera()

    class Scene:
        net_hit_frames = (np.array([10.0]),)
        observation_frames = (np.array([10.0]),)
        pixels = (_pixel(camera, contact).reshape(1, 2),)
        cameras = (camera.reshape(1, 3, 4),)
        camera_distortion = None
        contact_frames = np.array([0.0, 30.0])
        fps = 50.0
        rebound_mode = "fixed"

    bounds = evidence.bounds_from_scene(
        Scene(),
        (np.array([11.0]),),
        np.array([contact[0], 2.0, contact[2], 1.0, -14.0, 0.4]),
        {},
        cord.H_TOL_WIDE_M,
    )
    assert len(bounds) == 1
    assert bounds[0].admitted is False
    assert bounds[0].reason == "no_outgoing_observations"


def test_disagreeing_streak_abstains():
    contact = np.array([NET_X_CENTER, evidence.NET_Y, net_tape_height(NET_X_CENTER)])
    incoming = np.array([3.0, -16.0, 1.0])
    true = cord.project_to_box(incoming, np.array([1.0, 5.0, 0.2]))
    other = true + np.array([40.0, 0.0, 0.0])
    fps = 50.0
    t0 = 40.0
    camera, outgoing = _samples(contact, t0, true, fps, [41, 42, 43, 44])
    _, streak = _samples(contact, t0, other, fps, [41, 42, 43, 44])
    for row in streak:
        row["source"] = "streak"
    bound = evidence.derive_outgoing_bound(
        net_frame=t0,
        contact_pixel=_pixel(camera, contact),
        contact_camera=camera,
        outgoing=outgoing + streak,
        fps=fps,
        incoming_velocity=incoming,
        h_tol=cord.H_TOL_WIDE_M,
    )
    assert bound.admitted is False
    assert bound.reason == "sources_disagree"


def test_evidence_receipt_is_not_the_default_and_does_not_search():
    assert cord.receipt(cord.TAPE_CLIP) == {}
    receipt = cord.receipt(cord.EVIDENCE_BOUND)
    assert receipt[cord.FIELD] == cord.EVIDENCE_BOUND
    assert receipt["searches_v_out"] is False
    assert "absent_or_ambiguous" in receipt["abstain"]
    mode = cord.using_mode(cord.ADMISSIBLE_SET, h_tol=0.2)
    with mode:
        assert cord.active_mode() == cord.ADMISSIBLE_SET
    with mode:
        assert cord.active_h_tol() == pytest.approx(0.2)
    assert cord.active_mode() == cord.TAPE_CLIP
