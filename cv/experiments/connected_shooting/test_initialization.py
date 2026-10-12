from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting.model import control
from cv.experiments.connected_shooting.initialization import image_ballistic_seed
from cv.experiments.connected_shooting.initialization import backward_arc, physics_backward_seed
from physics import flight
from cv.experiments.connected_shooting import initialization


def ballistic_scene():
    scene, truth = control()
    p = truth[:3].copy()
    targets = []
    for i, (frames, cameras) in enumerate(
        zip(scene.observation_frames, scene.cameras, strict=True)
    ):
        start, end = scene.contact_frames[i : i + 2]
        v = truth[3 + 3 * i : 6 + 3 * i]
        t = (frames - start) / scene.fps
        xyz = p + t[:, None] * v + 0.5 * t[:, None] ** 2 * np.array([0, 0, -9.81])
        projected = np.einsum("nij,nj->ni", cameras, np.c_[xyz, np.ones(len(xyz))])
        targets.append(projected[:, :2] / projected[:, 2:])
        dt = (end - start) / scene.fps
        p = p + dt * v + 0.5 * dt**2 * np.array([0, 0, -9.81])
    return replace(scene, pixels=tuple(targets)), truth


def test_image_only_initializer_recovers_known_gravity_projection():
    scene, truth = ballistic_scene()
    seed, record = image_ballistic_seed(scene, [None, None])
    np.testing.assert_allclose(seed[:9], truth, atol=1e-9)
    assert not record["uses_truth_xyz_or_velocity"]
    assert record["clipped_parameter_indices"] == []


def test_invalid_backward_penalty_plateau_is_not_a_successful_seed(monkeypatch):
    scene, _ = ballistic_scene()

    def invalid(*args, **kwargs):
        raise ValueError("invalid backward state")

    monkeypatch.setattr(initialization, "backward_arc", invalid)
    with pytest.raises(ValueError, match="no finite image-derived backward seed"):
        physics_backward_seed(scene, [10.0, None])


def test_consensus_rejects_underground_launch_instead_of_clipping_it(monkeypatch):
    scene, _ = ballistic_scene()

    def underground(state, query):
        return tuple(
            np.tile(value, (len(query), 1))
            for value in ([5.0, 8.0, -1.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        )

    monkeypatch.setattr(initialization, "backward_arc", underground)
    seed, evidence = physics_backward_seed(scene, [10.0, None])
    assert seed[2] > 0
    assert 2 in evidence["backward_refinement"][0]["backward_local_clipped_indices"]
    with pytest.raises(ValueError, match="no launch-valid consensus"):
        physics_backward_seed(scene, [10.0, None], consensus=True)


def test_bounce_boundary_excludes_later_images_from_seed():
    scene, _ = ballistic_scene()
    changed = [p.copy() for p in scene.pixels]
    changed[0][scene.observation_frames[0] >= 9] += 10000
    a, _ = image_ballistic_seed(scene, [9.0, None])
    b, _ = image_ballistic_seed(replace(scene, pixels=tuple(changed)), [9.0, None])
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="pre-bounce training pictures"):
        image_ballistic_seed(scene, [3.0, None])


def test_degenerate_image_sequence_fails_instead_of_truth_fallback():
    scene, _ = ballistic_scene()
    with pytest.raises(ValueError, match="unidentified"):
        image_ballistic_seed(
            replace(scene, pixels=tuple(np.ones_like(p) for p in scene.pixels)), [None, None]
        )


def test_known_bounce_plane_is_an_exact_seed_constraint_without_xyz_labels():
    scene, _ = control()
    p, v = np.array([5.0, 3.0, 2.0]), np.array([2.0, 20.0, -2.0])
    frames = np.arange(1.0, 26.0)
    t = (frames - 1) / 25
    xyz = p + t[:, None] * v + 0.5 * t[:, None] ** 2 * np.array([0, 0, -9.81])
    P = np.repeat(scene.cameras[0][:1], len(t), axis=0)
    q = np.einsum("nij,nj->ni", P, np.c_[xyz, np.ones(len(xyz))])
    scene = replace(
        scene,
        contact_frames=np.array([1.0, 26.0]),
        observation_frames=(frames,),
        cameras=(P,),
        pixels=(q[:, :2] / q[:, 2:],),
        spin_parameters=np.zeros((1, 3)),
    )
    bounce = 1 + 25 * (-2 + np.sqrt(4 + 2 * 9.81 * (2 - 0.0325))) / 9.81
    seed, record = image_ballistic_seed(scene, [bounce], ground_anchor=True)
    np.testing.assert_allclose(seed[:6], np.r_[p, v], atol=1e-10)
    assert record["flights"][0]["ground_plane_at_known_bounce"]
    assert record["flights"][0]["rank"] == 5
    # A later, unseen bounce is not allowed to constrain the active interval.
    _, record = image_ballistic_seed(scene, [40.0], ground_anchor=True)
    assert not record["flights"][0]["ground_plane_at_known_bounce"]


def test_backward_drag_magnus_state_round_trips_forward():
    incoming = np.array([3.0, 15.0, 5.0, 25.0, -6.0, 2.0])
    x, v, w = (a[0] for a in backward_arc(incoming, np.array([0.5])))
    for _ in range(120):
        x, v, w = flight.rk4_step(x, v, w, 1 / 240)
    np.testing.assert_allclose(x, [3.0, 15.0, 0.0325], atol=1e-7)
    np.testing.assert_allclose(v, incoming[2:5], atol=1e-7)
    with pytest.raises(ValueError, match="descending"):
        backward_arc(np.array([3.0, 15.0, 5.0, 25.0, 6.0, 2.0]), np.array([0.5]))


@pytest.mark.parametrize("consensus", [False, True])
@pytest.mark.parametrize("radial_camera", [False, True])
@pytest.mark.parametrize("include_bounce_exposure", [False, True])
def test_backward_image_seed_recovers_ground_knot_without_truth_state_inputs(
    consensus, radial_camera, include_bounce_exposure
):
    scene, _ = control()
    incoming = np.array([3.0, 15.0, 5.0, 25.0, -6.0, 2.0])
    frames = np.arange(1.0, 15.0)
    bounce = 14.0
    x, v, w = backward_arc(incoming, (bounce - frames) / 25.0)
    P = np.repeat(scene.cameras[0][:1], len(frames), axis=0)
    q = np.einsum("nij,nj->ni", P, np.c_[x, np.ones(len(x))])
    scene = replace(
        scene,
        contact_frames=np.array([1.0, 14.0]),
        observation_frames=(frames,),
        cameras=(P,),
        pixels=(q[:, :2] / q[:, 2:],),
        spin_parameters=np.array([[2.0, 0.0, 0.0]]),
    )
    if radial_camera:
        from cv.experiments.connected_shooting.camera_geometry import distort

        radial = np.tile([2e-8, 960.0, 540.0], (len(frames), 1))
        scene = replace(
            scene, pixels=(distort(scene.pixels[0], radial),), camera_distortion=(radial,)
        )
    seed, record = physics_backward_seed(
        scene, [bounce], consensus=consensus, include_bounce_exposure=include_bounce_exposure
    )
    np.testing.assert_allclose(seed[:3], x[0], atol=1e-5)
    np.testing.assert_allclose(seed[3:6], v[0], atol=1e-5)
    assert record["backward_refinement"][0]["training_pixel_rms"] < 1e-5
    assert not record["uses_truth_xyz_or_velocity"]
    refinement = record["backward_refinement"][0]
    assert refinement["included_exact_bounce_exposure"] is include_bounce_exposure
    assert (bounce in refinement["training_frames"]) is include_bounce_exposure


def test_gravity_consensus_resists_wrong_object_rows_without_mutating_observations():
    base, _ = control()
    p, v = np.array([5.0, 3.0, 2.0]), np.array([2.0, 20.0, -2.0])
    frames = np.arange(1.0, 13.0)
    t = (frames - 1) / 25
    xyz = p + t[:, None] * v + 0.5 * t[:, None] ** 2 * np.array([0, 0, -9.81])
    P = np.repeat(base.cameras[0][:1], len(frames), axis=0)
    q = np.einsum("nij,nj->ni", P, np.c_[xyz, np.ones(len(xyz))])
    pixels = q[:, :2] / q[:, 2:]
    pixels[3:6] += [-110, 80]
    original = pixels.copy()
    bounce = 1 + 25 * (-2 + np.sqrt(4 + 2 * 9.81 * (2 - 0.0325))) / 9.81
    scene = replace(
        base,
        contact_frames=np.array([1.0, bounce]),
        observation_frames=(frames,),
        cameras=(P,),
        pixels=(pixels,),
        spin_parameters=np.zeros((1, 3)),
    )
    ordinary, _ = image_ballistic_seed(scene, [bounce], ground_anchor=True)
    robust, record = image_ballistic_seed(scene, [bounce], ground_anchor=True, consensus=True)
    np.testing.assert_array_equal(pixels, original)
    assert np.linalg.norm(ordinary[:6] - np.r_[p, v]) > 1
    np.testing.assert_allclose(robust[:6], np.r_[p, v], atol=1e-8)
    assert record["flights"][0]["consensus"]["deleted_observations"] == 0


def anchored_two_picture_scene():
    """One flight whose bounce time is known and whose pictures are two."""
    scene, truth = ballistic_scene()
    start = scene.contact_frames[0]
    gravity = np.array([0.0, 0.0, -9.81])
    velocity = truth[3:6]
    # The bounce epoch where this exact arc reaches the ball-centre plane.
    from cv.experiments.connected_shooting.model import R_BALL

    a, b, c = 0.5 * gravity[2], velocity[2], truth[2] - R_BALL
    seconds = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    bounce = float(start + seconds * scene.fps)
    keep = np.asarray([frame for frame in scene.observation_frames[0] if frame < bounce][:2], float)
    index = [list(scene.observation_frames[0]).index(frame) for frame in keep]
    reduced = replace(
        scene,
        contact_frames=np.asarray([scene.contact_frames[0], bounce], float),
        observation_frames=(keep,),
        cameras=(scene.cameras[0][index],),
        pixels=(scene.pixels[0][index],),
        spin_parameters=scene.spin_parameters[:1],
    )
    target = truth[:3] + seconds * velocity + 0.5 * seconds**2 * gravity
    return reduced, truth, bounce, target[:2]


def test_two_pictures_and_a_measured_impact_identify_a_flight():
    scene, truth, bounce, target = anchored_two_picture_scene()
    with pytest.raises(ValueError, match="three pre-bounce training pictures"):
        image_ballistic_seed(scene, [bounce], ground_anchor=True)
    seed, record = image_ballistic_seed(
        scene, [bounce], ground_anchor=True, bounce_ground_targets=[target]
    )
    np.testing.assert_allclose(seed[:6], truth[:6], atol=1e-6)
    assert record["flights"][0]["measured_bounce_position_rows"] == 2
    np.testing.assert_allclose(record["flights"][0]["bounce_ground_target_xy_m"], target)


def test_a_flight_with_no_measured_impact_keeps_the_old_minimum():
    scene, _, bounce, _ = anchored_two_picture_scene()
    with pytest.raises(ValueError, match="three pre-bounce training pictures"):
        image_ballistic_seed(scene, [bounce], ground_anchor=True, bounce_ground_targets=[None])


def test_one_target_per_flight_is_required():
    scene, _ = ballistic_scene()
    with pytest.raises(ValueError, match="one optional bounce ground target"):
        image_ballistic_seed(scene, [None, None], bounce_ground_targets=[None])


def test_serve_contact_anchor_keeps_the_ray_height_inside_the_stature_band():
    from cv.experiments.connected_shooting import initialization

    # Camera looking down the court from behind the baseline at 8 m.
    camera = np.array(
        [
            [1000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1000.0, 8000.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    truth = np.array([1.5, 20.0, 2.9])
    pixel = camera @ np.r_[truth, 1.0]
    pixel = pixel[:2] / pixel[2]

    xyz, receipt = initialization.serve_contact_anchor(camera, pixel, 1.85, 20.0)

    assert receipt["clipped_to_band"] is False
    assert xyz[1] == 20.0
    assert np.allclose(xyz, truth, atol=1e-6)

    # A 1.60 m player cannot reach 2.9 m: the band decides, the beam keeps Y.
    short, short_receipt = initialization.serve_contact_anchor(camera, pixel, 1.60, 20.0)
    assert short_receipt["clipped_to_band"] is True
    assert short[1] == 20.0
    assert 1.5 * 1.60 <= short[2] <= 1.75 * 1.60


def test_serve_contact_anchor_refuses_a_degenerate_band():
    from cv.experiments.connected_shooting import initialization

    camera = np.array([[1000.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1000.0, 8000.0], [0.0, 1.0, 0.0, 0.0]])
    try:
        initialization.serve_contact_anchor(
            camera, np.array([0.0, 0.0]), 1.85, 20.0, height_band=(1.75, 1.5)
        )
    except ValueError as error:
        assert "stature contact-height band" in str(error)
    else:
        raise AssertionError("a non-increasing stature band must fail closed")


def test_toss_contact_seed_uses_contact_before_outgoing_direction():
    scene, truth = ballistic_scene()
    pinhole = truth.copy()
    pinhole[:6] += [1.0, -4.0, 2.0, 10.0, -8.0, 6.0]
    estimate = {
        "status": "supported",
        "contact_xyz_m": truth[:3].tolist(),
        "contact_sigma_m": [0.2, 0.3, 0.2],
    }

    seed, receipt = initialization.toss_contact_seed(scene, pinhole, estimate)

    np.testing.assert_allclose(seed[:3], truth[:3], atol=1e-9)
    np.testing.assert_allclose(
        seed[3:6] / np.linalg.norm(seed[3:6]),
        truth[3:6] / np.linalg.norm(truth[3:6]),
        atol=1e-9,
    )
    assert np.linalg.norm(seed[3:6]) == pytest.approx(np.linalg.norm(pinhole[3:6]))
    assert receipt["method"] == "toss_contact_plus_postcontact_native_ballistic_direction"
    assert receipt["postcontact_ray_equation_rank"] == 3
    assert not receipt["uses_outgoing_serve_arc_for_contact_position"]


def test_outgoing_contact_seed_uses_drag_fronts_and_adds_no_bound():
    scene, truth = ballistic_scene()
    hypothesis = {
        "contact_xyz_m": truth[:3].tolist(),
        "contact_sigma_m": [0.2, 0.3, 0.2],
        "contact_epoch_frame": float(scene.contact_frames[0]),
    }
    seed, receipt = initialization.outgoing_contact_seed(scene, truth, hypothesis)
    np.testing.assert_allclose(seed[:3], truth[:3], atol=1e-9)
    assert receipt["outgoing_pixel_rms"] < 0.1
    assert receipt["method"] == "drag_magnus_outgoing_fronts_extrapolated_backward_to_contact"
    assert not receipt["optimizer_bound"]
    assert not receipt["optimizer_inequality"]
