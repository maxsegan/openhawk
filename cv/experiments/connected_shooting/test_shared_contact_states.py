from dataclasses import replace

import numpy as np

from cv.experiments.connected_shooting import (
    agent_whole_point_search,
    camera_geometry,
    model,
    real_exposure_replay,
)


def test_shared_state_arm_only_changes_the_refined_stage():
    options = {
        "shared_contact_states": True,
        "local_flight_refits": True,
        "directional_inequalities": False,
    }

    coarse = agent_whole_point_search.stage_refine_options(options, "coarse")
    refined = agent_whole_point_search.stage_refine_options(options, "refined")

    assert coarse == {"directional_inequalities": False}
    assert refined == options
    assert refined is not options


def test_second_stage_reference_requires_and_copies_the_promoted_rank_one():
    rank_one = {"depth_hypothesis_m": 24.0, "measurement": {}, "evidence": {}}
    document = {
        "schema": "s6_agent_whole_point_search_v2",
        "attempt_id": "attempt-1",
        "configuration": {
            "coarse_iterations": 150,
            "solver_experimental_arm": {
                "refine_inequalities": "terminal_only",
                "anchor_residuals": "present",
                "seed_restarts": "three",
            },
        },
        "coarse_candidates": [{"depth_hypothesis_m": 24.0}],
        "refined_candidates": [rank_one],
        "selected": rank_one,
    }

    coarse, selected, refined = agent_whole_point_search.second_stage_reference(
        document, "attempt-1"
    )
    coarse[0]["depth_hypothesis_m"] = 12.0
    selected["depth_hypothesis_m"] = 12.0
    refined[0]["depth_hypothesis_m"] = 12.0

    assert document["coarse_candidates"][0]["depth_hypothesis_m"] == 24.0
    assert rank_one["depth_hypothesis_m"] == 24.0


def test_single_shooting_seed_lifts_to_zero_gap_shared_contacts():
    scene, truth = model.control()
    single = np.r_[truth, np.zeros(6)]
    shared_scene = replace(scene, parameterization="shared_contact_states")

    shared = model.shared_contact_seed(shared_scene, single)
    flights = model.chain(shared_scene, shared)

    np.testing.assert_array_equal(model.shared_contact_gaps(flights), np.zeros((2, 3)))
    np.testing.assert_array_equal(flights[0]["shared_end_xyz"], flights[1]["start_xyz"])


def test_shared_contact_flights_start_independently_from_the_same_boundary_variable():
    scene, truth = model.control()
    single = np.r_[truth, np.zeros(6)]
    shared_scene = replace(scene, parameterization="shared_contact_states")
    shared = model.shared_contact_seed(shared_scene, single)
    blocks = model.shared_parameter_slices(shared_scene)
    contacts = shared[blocks["contacts"]].reshape(3, 3)
    contacts[1] += [0.25, -0.5, 0.1]

    flights = model.chain(shared_scene, shared)

    np.testing.assert_array_equal(flights[0]["shared_end_xyz"], contacts[1])
    np.testing.assert_array_equal(flights[1]["start_xyz"], contacts[1])
    assert np.linalg.norm(model.shared_contact_gaps(flights)[0]) > 0.5


def test_shared_fit_exports_the_legacy_acceptance_parameter_shape():
    scene, truth = model.control()
    single = np.r_[truth, np.zeros(6)]
    shared_scene = replace(scene, parameterization="shared_contact_states")
    shared = model.shared_contact_seed(shared_scene, single)

    exported = model.shared_to_single_parameters(shared_scene, shared)

    np.testing.assert_array_equal(exported, single)


def test_shared_refinement_uses_sparse_blocks_and_closes_every_endpoint():
    scene, base = model.control()
    scene = replace(
        scene,
        dynamics="measured_240hz",
        rebound_mode="point_scales",
        parameterization="single_shooting",
    )
    truth = np.r_[base, np.zeros(6), [1.0, 1.0]]
    flights = model.chain(scene, truth)
    scene = replace(
        scene,
        pixels=tuple(
            camera_geometry.project(cameras, flight["positions"])
            for cameras, flight in zip(scene.cameras, flights, strict=True)
        ),
        parameterization="shared_contact_states",
    )
    initial = truth.copy()
    initial[3:9] += [0.05, -0.05, 0.02, -0.04, 0.03, -0.02]

    fitted = real_exposure_replay.refine(
        scene,
        initial,
        (np.empty(0), np.empty(0)),
        scene.observation_frames,
        np.zeros((sum(map(len, scene.observation_frames)), 2)),
        None,
        40,
        inequality_constraints=False,
        shared_contact_states=True,
        local_flight_refits=True,
    )

    evidence = fitted["shared_contact_states"]
    assert fitted["primary_optimizer"]["method"] == "scipy_least_squares_trf"
    assert fitted["primary_optimizer"]["jacobian_nonzero_fraction"] < 0.75
    assert evidence["maximum_endpoint_gap_m"] < 1e-3
    assert evidence["certificate_limit_m"] == 1e-3
    assert len(evidence["local_flight_refits"]) == 2
    assert all(row["nonlocal_parameters_bit_unchanged"] for row in evidence["local_flight_refits"])
