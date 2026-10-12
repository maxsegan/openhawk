from cv.experiments.connected_shooting import agent_whole_point_search as search


def candidate(score: float, *, survived: bool) -> dict:
    return {
        "measurement": {"rms_px": {"training": score**0.5}},
        "evidence": {"input_only_rank_score": score, "survived": survived},
    }


def test_reference_restart_keeps_the_lower_input_only_ranked_branch():
    active = candidate(12.0, survived=False)
    reference = candidate(5.0, survived=True)

    selected = search.select_reference_restart(active, reference)

    assert selected is reference
    assert selected["reference_restart"]["selected"] == "inequalities_in_reference"
    assert selected["reference_restart"]["candidates"]["active_arm"] == {
        "training_rms_px": 12.0**0.5,
        "input_only_rank_score": 12.0,
        "survived": False,
    }


def test_reference_restart_tie_preserves_the_active_arm():
    active = candidate(5.0, survived=True)
    reference = candidate(5.0, survived=True)

    selected = search.select_reference_restart(active, reference)

    assert selected is active
    assert selected["reference_restart"]["selected"] == "active_arm"
    assert selected["reference_restart"]["reference_configuration"] == {
        "refine_inequalities": "in",
        "anchor_residuals": "absent",
        "seed_restarts": "off",
        "coarse_iterations": 100,
    }


def test_contact_family_width_reports_each_exact_boundary():
    def row(offset):
        return {
            "measurement": {
                "dense_flights": [
                    {"start_xyz": [0.0, 1.0, 2.0], "end_xyz": [1.0, 2.0, 3.0]},
                    {
                        "start_xyz": [1.0 + offset, 2.0, 3.0],
                        "end_xyz": [4.0 + offset, 5.0, 0.0325],
                    },
                ]
            }
        }

    widths = search.contact_family_widths([row(0.0), row(0.25)])

    assert widths["candidate_count"] == 2
    assert [contact["maximum_pairwise_width_m"] for contact in widths["contacts"]] == [
        0.0,
        0.25,
        0.25,
    ]
