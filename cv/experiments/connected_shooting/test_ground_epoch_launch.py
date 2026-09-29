"""A connected walk must not throw a whole point away over a court-level handoff.

Each flight in a connected seed walk starts from the previous flight's arrival.
Every flight that has a next contact also owns an ascending start, because its
next-contact ballistic guess carries ``+4.905 * duration``.  A flight without
that anchor -- the terminal flight, or any flight whose next striker position is
explicitly absent -- keeps only the disconnected pinhole velocity, so a
court-level arrival leaves it with no supported launch and the walk raises
``descending initial ground state has no supported flight``.  Recovery rebuilds
one start from the witness that flight does own: its declared ground epoch.
"""

from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    agent_whole_point_search as search,
    initialization,
    model,
    player_position,
    real_exposure_replay as exposure,
)
from cv.experiments.connected_shooting.model import R_BALL

FPS = 25.0
CAMERA = np.array([[1000, 0, 0, 0], [0, 0, 1000, 0], [0, 1, 0, 0]], float)


def single_flight_scene(frames: np.ndarray) -> model.Scene:
    return model.Scene(
        contact_frames=np.array([0.0, 14.0]),
        observation_frames=(frames,),
        cameras=(np.repeat(CAMERA[None], len(frames), axis=0),),
        pixels=(np.zeros((len(frames), 2)),),
        spin_parameters=np.zeros((1, 3)),
        fps=FPS,
        surface="grass",
        dynamics="measured_240hz",
        rebound_mode="point_scales",
    )


def court_level_terminal_case():
    """A terminal flight launched from the court, with its supplied ground ray."""
    start = np.array([2.0, 25.0, R_BALL])
    target = np.array([4.0, 17.0])
    seconds = 10.0 / FPS
    launch = initialization.ground_epoch_launch(start, seconds, np.zeros(3), target_xy=target)
    frames = np.arange(1.0, 14.0)
    scene = single_flight_scene(frames)
    truth = np.r_[start, launch, np.zeros(3), [1.0, 1.0]]
    axes = np.zeros((len(frames), 2))
    scene = replace(scene, pixels=(exposure.prediction(scene, truth, axes, None),))
    descending = np.r_[start, [5.0, -6.0, -0.1], np.zeros(3), [1.0, 1.0]]
    return scene, axes, descending, start, target


def anchor_kwargs(start, target):
    return dict(
        first_contact_xyz_m=start,
        contact_targets_xyz_m=[None],
        bounce_targets=[[(target, 0.3)]],
        max_nfev=40,
        exposure_duration=None,
    )


def test_launch_lands_on_the_declared_epoch_and_keeps_the_supplied_horizontal():
    start = np.array([2.0, 25.0, 0.02])
    seconds = 0.4
    arc = initialization.ground_epoch_launch(start, seconds, np.array([3.0, -9.0, -4.0]))
    assert arc[0] == 3.0 and arc[1] == -9.0
    assert start[2] + arc[2] * seconds - 4.905 * seconds**2 == pytest.approx(R_BALL)
    ray = initialization.ground_epoch_launch(
        start, seconds, np.array([3.0, -9.0, -4.0]), target_xy=np.array([5.0, 17.0])
    )
    np.testing.assert_allclose(start[:2] + ray[:2] * seconds, [5.0, 17.0])
    assert ray[2] == arc[2]


@pytest.mark.parametrize(
    "seconds,horizontal,target",
    [
        (0.0, np.zeros(3), None),
        (-0.4, np.zeros(3), None),
        (float("nan"), np.zeros(3), None),
        (0.4, np.array([np.nan, 0.0, 0.0]), None),
        (0.4, np.zeros(3), np.array([np.inf, 3.0])),
        (0.4, np.zeros(3), np.array([3.0, 4.0, 5.0])),
        (0.4, np.array([200.0, 0.0, 0.0]), None),
    ],
)
def test_unusable_epoch_target_or_launch_abstains_instead_of_being_clipped(
    seconds, horizontal, target
):
    start = np.array([2.0, 25.0, 0.02])
    assert initialization.ground_epoch_launch(start, seconds, horizontal, target_xy=target) is None


def test_anchor_walk_recovers_a_court_level_terminal_start_from_its_ground_epoch():
    scene, axes, descending, start, target = court_level_terminal_case()
    seed, receipt = initialization.anchor_connected_seed(
        scene, descending, axes, [[10.0]], 25.0, **anchor_kwargs(start, target)
    )
    epoch = receipt["flights"][0]["ground_epoch_launch"]
    assert epoch["status"] == "retried" and epoch["supported"]
    assert epoch["declared_ground_frame"] == 10.0
    assert epoch["seconds_to_declared_ground"] == pytest.approx(10.0 / FPS)
    assert epoch["uses_supplied_ground_ray"]
    assert receipt["flights"][0]["selected_guess_index"] == 1
    assert "descending initial ground state" in str(receipt["flights"][0]["unsupported_candidates"])
    assert receipt["flights"][0]["training_pixel_rms"] < 1.0
    # The recovered walk is an ordinary supported flight with its declared ground.
    flight = model.chain(scene, seed)[0]
    assert len(flight["bounces"]) == 1
    assert flight["bounces"][0]["frame"] == pytest.approx(10.0, abs=0.5)
    # Only the bound clip on a sub-radius start moved the supplied position.
    np.testing.assert_allclose(seed[:3], np.r_[start[:2], R_BALL + 1e-5])


def test_anchor_walk_without_a_declared_ground_epoch_still_refuses():
    scene, axes, descending, start, target = court_level_terminal_case()
    kwargs = anchor_kwargs(start, target)
    kwargs["bounce_targets"] = [[]]
    with pytest.raises(ValueError, match="no finite anchored seed"):
        initialization.anchor_connected_seed(scene, descending, axes, [[]], 25.0, **kwargs)


def test_anchor_walk_that_already_solves_is_untouched():
    scene, axes, descending, start, target = court_level_terminal_case()
    airborne = descending.copy()
    airborne[3:6] = initialization.ground_epoch_launch(
        start, 10.0 / FPS, np.zeros(3), target_xy=target
    )
    seed, receipt = initialization.anchor_connected_seed(
        scene, airborne, axes, [[10.0]], 25.0, **anchor_kwargs(start, target)
    )
    assert "ground_epoch_launch" not in receipt["flights"][0]
    assert receipt["flights"][0]["selected_guess_index"] == 0


def absent_player():
    return {
        "court_centre_xy_m": None,
        "player_position_evidence": player_position.absent_position(),
        "stature_m": 1.85,
    }


def structured_case():
    """Two connected flights whose second striker position is explicitly absent."""
    start = np.array([2.0, 25.0, R_BALL])
    launch = initialization.ground_epoch_launch(
        start, 10.0 / FPS, np.zeros(3), target_xy=np.array([4.0, 17.0])
    )
    frames = np.arange(1.0, 14.0)
    scene = replace(
        single_flight_scene(frames),
        contact_frames=np.array([0.0, 14.0, 28.0]),
        observation_frames=(frames, frames + 14.0),
        cameras=tuple(np.repeat(CAMERA[None], len(frames), axis=0) for _ in range(2)),
        pixels=(np.zeros((len(frames), 2)), np.zeros((len(frames), 2))),
        spin_parameters=np.zeros((2, 3)),
    )
    truth = np.r_[start, launch, [-1.0, -8.0, 6.0], np.zeros(6), [1.0, 1.0]]
    axes = np.zeros((2 * len(frames), 2))
    pixels = np.split(exposure.prediction(scene, truth, axes, None), [len(frames)])
    scene = replace(scene, pixels=(pixels[0], pixels[1]))
    descending = np.r_[start, [5.0, -6.0, -0.1], [-1.0, -8.0, 6.0], np.zeros(6), [1.0, 1.0]]
    return scene, axes, descending, truth


def test_structured_walk_recovers_a_court_level_start_without_a_next_striker():
    scene, axes, descending, _ = structured_case()
    players = [absent_player(), absent_player()]
    seed, receipt = search.structure_aware_connected_seed(
        scene, descending, players, axes, ([10.0], [24.0]), 25.0, exposure_duration=None
    )
    epoch = receipt["flights"][0]["ground_epoch_launch"]
    assert epoch["status"] == "retried"
    assert epoch["declared_ground_frame"] == 10.0
    assert not epoch["uses_supplied_ground_ray"]
    assert "descending initial ground state" in epoch["unsupported_reason"]
    assert receipt["flights"][0]["training_pixel_rms"] < 1.0
    assert "ground_epoch_launch" not in receipt["flights"][1]
    assert len(model.chain(scene, seed)[0]["bounces"]) == 1


def test_structured_walk_that_already_solves_is_untouched():
    scene, axes, _, truth = structured_case()
    players = [absent_player(), absent_player()]
    _, receipt = search.structure_aware_connected_seed(
        scene, truth, players, axes, ([10.0], [24.0]), 25.0, exposure_duration=None
    )
    assert all("ground_epoch_launch" not in row for row in receipt["flights"])
