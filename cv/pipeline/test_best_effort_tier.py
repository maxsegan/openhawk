import numpy as np

from cv.pipeline import best_effort_tier as bet

FPS = 50.0


def camera() -> np.ndarray:
    """A broadcast-like pinhole looking down the court from behind the near baseline."""
    centre = np.array([5.5, -14.0, 7.0])
    target = np.array([5.5, 12.0, 0.0])
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    return K @ np.hstack([R, (-R @ centre)[:, None]])


P = camera()


def arc(p0, p1, start, end):
    anchors = [
        {"kind": "contact", "frame": start, "xyz_m": p0},
        {"kind": "contact", "frame": end, "xyz_m": p1},
    ]
    positions, frames = bet.ballistic(anchors, FPS)
    return positions, frames


def dense(p0, p1, start, end):
    positions, _ = arc(p0, p1, start, end)
    return {
        "start_frame": start,
        "end_frame": end,
        "start_xyz": positions[0],
        "end_xyz": positions[-1],
        "positions": positions,
        "bounces": [],
    }


def test_ballistic_passes_through_its_anchors():
    anchors = [
        {"kind": "contact", "frame": 10.0, "xyz_m": [5.0, 0.0, 1.0]},
        {"kind": "bounce", "frame": 50.0, "xyz_m": [5.0, 18.0, 0.0325]},
        {"kind": "contact", "frame": 75.0, "xyz_m": [5.0, 24.0, 1.0]},
    ]
    positions, frames = bet.ballistic(anchors, FPS)
    assert np.allclose(positions[0], anchors[0]["xyz_m"])
    assert np.allclose(positions[-1], anchors[-1]["xyz_m"])
    k = int(np.argmin(np.abs(np.asarray(frames) - 50.0)))
    assert np.allclose(positions[k], anchors[1]["xyz_m"], atol=0.05)
    assert bet.path_checks(positions, frames, FPS) == []


def test_arc_crossing_below_the_tape_is_rejected():
    anchors = [
        {"kind": "contact", "frame": 0.0, "xyz_m": [5.0, 6.0, 0.5]},
        {"kind": "bounce", "frame": 10.0, "xyz_m": [5.0, 16.0, 0.0325]},
    ]
    positions, frames = bet.ballistic(anchors, FPS)
    assert "through_net" in bet.path_checks(positions, frames, FPS)


def test_path_checks_reject_fragments_and_far_launches():
    fragment, frames = arc([5.0, -1.0, 1.0], [5.0, 0.0, 1.0], 10.0, 11.5)
    assert "fragment" in bet.path_checks(fragment, frames, FPS)
    far, frames = arc([5.0, -15.0, 5.0], [5.0, 20.0, 0.5], 10.0, 70.0)
    assert {"outside_envelope", "launch_height"} <= set(bet.path_checks(far, frames, FPS))


def test_reversal_at_a_bounce_is_detected():
    anchors = [
        {"kind": "contact", "xyz_m": [5.0, 2.0, 1.0]},
        {"kind": "bounce", "xyz_m": [5.0, 18.0, 0.03]},
        {"kind": "contact", "xyz_m": [5.0, 12.5, 1.0]},
    ]
    assert bet.reverses_at_bounce(anchors)
    anchors[2]["xyz_m"] = [5.0, 24.0, 1.0]
    assert not bet.reverses_at_bounce(anchors)


def test_ray_at_height_inverts_projection():
    point = np.array([3.0, 8.0, bet.BALL_RADIUS_M])
    found = bet.ray_at_height(P, bet.project(P, point), bet.BALL_RADIUS_M)
    assert np.allclose(found, point, atol=1e-6)


def test_attempt_layer_assigns_one_tier_per_slot():
    # Slot 0 validated, slot 1 failed a gate with a fitted state, slot 2 has no state but
    # fitted neighbours share both contacts, slot 3 lies after an out-bounce call.
    contacts = {
        10.0: [5.0, -1.0, 1.0],
        60.0: [5.0, 24.0, 1.1],
        110.0: [4.0, -0.5, 0.9],
        160.0: [6.0, 23.0, 1.0],
        200.0: [6.0, 26.0, 1.0],
    }
    frames = sorted(contacts)
    states = [dense(contacts[a], contacts[b], a, b) for a, b in zip(frames, frames[1:])]
    verdict_flights = [
        {
            "original_flight_index": 0,
            "flight_index": 0,
            "component_index": 0,
            "accepted": True,
            "start_frame": 10.0,
            "end_frame": 60.0,
            "failures": [],
        },
        {
            "original_flight_index": 1,
            "flight_index": 1,
            "component_index": 0,
            "accepted": False,
            "start_frame": 60.0,
            "end_frame": 110.0,
            "failures": ["bounce_count_timing"],
            "reprojection_rms_px": 4.2,
        },
        {
            "original_flight_index": 2,
            "flight_index": None,
            "component_index": None,
            "accepted": False,
            "start_frame": 110.0,
            "end_frame": 160.0,
            "failures": None,
            "reasons": ["component_execution_did_not_emit"],
        },
        {
            "original_flight_index": 3,
            "flight_index": None,
            "component_index": None,
            "accepted": False,
            "start_frame": 160.0,
            "end_frame": 200.0,
            "failures": None,
            "reasons": ["post_out_bounce_origin"],
        },
    ]
    # The component holds slots 0 and 1; slot 3's neighbour state sits in a second component.
    result = {
        "verdict": {"flights": verdict_flights, "complete_point": False},
        "component_results": [
            {"component_index": 0, "result": {"measurement": {"dense_flights": states[:2]}}},
            {"component_index": 1, "result": {"measurement": {"dense_flights": []}}},
        ],
        "source_plan": {
            "original_slots": [
                {
                    "original_flight_index": i,
                    "source_bounce_frames": [],
                    "known_next_contact_index": i + 1,
                }
                for i in range(4)
            ],
            "original_events": [],
        },
    }
    # The anchored slot's next contact comes from the following slot: give it a state too.
    result["component_results"][1]["result"]["measurement"]["dense_flights"] = [states[3]]
    verdict_flights[3].update(flight_index=0, component_index=1)
    observed = {}
    for state in states:
        for position, frame in zip(
            state["positions"],
            np.linspace(state["start_frame"], state["end_frame"], len(state["positions"])),
        ):
            if abs(frame - round(frame)) < 1e-6:
                observed[int(round(frame))] = bet.project(P, position)
    labels = {
        "ball": {
            "records": [
                {
                    "frames": [
                        {
                            "frame": f,
                            "status": "visible",
                            "x1080": float(xy[0]),
                            "y1080": float(xy[1]),
                        }
                        for f, xy in observed.items()
                    ]
                }
            ]
        }
    }
    cameras = {"cameras": [{"frame": f, "supported": True, "P": P.tolist()} for f in range(1, 260)]}
    accepted = [{"start_frame": 10.0}]
    rows = bet.attempt_layer(result, accepted, cameras, labels, {}, FPS)
    assert [r["tier"] for r in rows] == [
        "validated",
        "fitted_unvalidated",
        "anchored",
        "fitted_unvalidated",
    ]
    fitted = rows[1]
    assert fitted["gates_failed"] == ["bounce_count_timing"]
    assert fitted["state_kind"] == "fitted"
    assert fitted["support"]["median_px"] < 1.0
    anchored = rows[2]
    assert anchored["state_kind"] == "interpolated"
    assert [a["source"] for a in anchored["anchors"]] == ["fitted_neighbour", "fitted_neighbour"]
    assert anchored["sub_tier"] == "fitted_neighbour"
    assert anchored["support"]["median_px"] < 1.0
    assert rows[3]["after_point_end"] is True

    layer = bet.summary([rows], attempts=2, s6_complete=[False])
    assert layer["slots_by_tier"] == {
        "validated": 1,
        "fitted_unvalidated": 2,
        "anchored": 1,
        "none": 0,
    }
    assert layer["yields"]["validated"]["complete_points"] == 0
    assert layer["yields"]["extended"]["complete_points"] == 1
    assert layer["yields"]["extended"]["complete_points_every_slot"] == 1
    assert layer["yields"]["validated"]["complete_points_s6_rule"] == 0

    portal = bet.portal_layer(
        {"match_id": "m", "attempts": [{"key": "m__pt0001", "best_effort_flights": rows}]}
    )
    labels_out = [f["label"] for f in portal["points"]["m__pt0001"]]
    assert labels_out[0].startswith("BEST EFFORT · fitted, failed bounce_count_timing")
    assert labels_out[1].startswith("BEST EFFORT · anchored arc")


def test_slot_without_anchors_abstains():
    result = {
        "verdict": {
            "flights": [
                {
                    "original_flight_index": 0,
                    "start_frame": 10.0,
                    "end_frame": 40.0,
                    "accepted": False,
                    "reasons": ["unsupported_original_flight_coverage"],
                }
            ]
        },
        "source_plan": {"original_slots": [{"original_flight_index": 0}], "original_events": []},
    }
    rows = bet.attempt_layer(result, [], {"cameras": []}, {}, {}, FPS)
    assert rows[0]["tier"] == "none" and rows[0]["reason"] == "too_few_anchors"


def test_switch_defaults_off():
    assert not bet.enabled({})
    assert bet.enabled({bet.SWITCH: "on"})


def test_fitted_state_off_the_ball_is_demoted():
    state = dense([5.0, -1.0, 1.0], [5.0, 24.0, 1.1], 10.0, 60.0)
    result = {
        "verdict": {
            "flights": [
                {
                    "original_flight_index": 0,
                    "flight_index": 0,
                    "component_index": 0,
                    "accepted": False,
                    "start_frame": 10.0,
                    "end_frame": 60.0,
                    "failures": ["bounce_count_timing"],
                },
            ]
        },
        "component_results": [
            {"component_index": 0, "result": {"measurement": {"dense_flights": [state]}}}
        ],
        "source_plan": {"original_slots": [{"original_flight_index": 0}], "original_events": []},
    }
    # The detections sit 80 px to the side of the fitted state's projection.
    labels = {
        "ball": {
            "records": [
                {
                    "frames": [
                        {
                            "frame": f,
                            "status": "visible",
                            "x1080": float(bet.project(P, p)[0]) + 80.0,
                            "y1080": float(bet.project(P, p)[1]),
                        }
                        for p, f in zip(
                            state["positions"], np.linspace(10.0, 60.0, len(state["positions"]))
                        )
                        if abs(f - round(f)) < 1e-6
                        for f in [int(round(f))]
                    ]
                }
            ]
        }
    }
    cameras = {"cameras": [{"frame": f, "supported": True, "P": P.tolist()} for f in range(1, 80)]}
    row = bet.attempt_layer(result, [], cameras, labels, {}, FPS)[0]
    assert row["tier"] == "none"
    assert row["demoted_fitted_state"]["checks_failed"] == ["residual"]


def test_demoted_gate_family_falls_back_to_anchors():
    state = dense([5.0, -1.0, 1.0], [5.0, 24.0, 1.1], 10.0, 60.0)
    result = {
        "verdict": {
            "flights": [
                {
                    "original_flight_index": 0,
                    "flight_index": 0,
                    "component_index": 0,
                    "accepted": False,
                    "start_frame": 10.0,
                    "end_frame": 60.0,
                    "failures": ["bounce_rays_agree"],
                }
            ]
        },
        "component_results": [
            {"component_index": 0, "result": {"measurement": {"dense_flights": [state]}}}
        ],
        "source_plan": {"original_slots": [{"original_flight_index": 0}], "original_events": []},
    }
    frames = np.linspace(10.0, 60.0, len(state["positions"]))
    labels = {
        "ball": {
            "records": [
                {
                    "frames": [
                        {
                            "frame": int(round(f)),
                            "status": "visible",
                            "x1080": float(bet.project(P, p)[0]),
                            "y1080": float(bet.project(P, p)[1]),
                        }
                        for p, f in zip(state["positions"], frames)
                        if abs(f - round(f)) < 1e-6
                    ]
                }
            ]
        }
    }
    cameras = {"cameras": [{"frame": f, "supported": True, "P": P.tolist()} for f in range(1, 80)]}
    row = bet.attempt_layer(result, [], cameras, labels, {}, FPS)[0]
    assert row["tier"] == "none"
    assert row["demoted_fitted_state"]["checks_failed"] == ["gate:bounce_rays_agree"]
