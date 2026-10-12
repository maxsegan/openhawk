import json

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_player_serve_prior as prior

GOOD = {"contact": "reviewed_label", "identity": "reviewed_label", "ground": "reviewed_label"}


def record(attempt_id, contact, feet=None, end="near", player="A", side="deuce", **overrides):
    row = {
        "attempt_id": attempt_id,
        "player_id": player,
        "service_side": side,
        "server_end": end,
        "contact_xyz_m": list(contact),
        "grounded_feet_xy_m": None if feet is None else list(feet),
        "contact_qualified": True,
        "identity_qualified": True,
        "ground_qualified": feet is not None,
        "sources": dict(GOOD),
    }
    row.update(overrides)
    return row


def near_cohort(count=4, start=0):
    rng = np.random.default_rng(start)
    rows = []
    for index in range(count):
        jitter = rng.normal(0, 0.05, 3)
        contact = np.array([6.3, -0.4, 2.9]) + jitter
        rows.append(record(f"n{start + index}", contact, feet=[6.0, -0.9]))
    return rows


def test_far_end_records_and_queries_convert_to_the_near_canonical_frame():
    near = near_cohort(4)
    far = [
        record("f0", [10.97 - 6.3, 23.77 + 0.4, 2.9], feet=[10.97 - 6.0, 23.77 + 0.9], end="far")
    ]
    fitted = prior.fit_prior(near + far, "A", "deuce")
    assert fitted["absolute"]["source_attempt_ids"] == ["n0", "n1", "n2", "n3", "f0"]
    assert np.allclose(prior.canonical([1.0, 2.0, 3.0], "far"), [9.97, 21.77, 3.0])
    assert np.allclose(prior.canonical([1.0, 2.0], "far"), [9.97, 21.77])
    near_query = prior.evaluate(fitted, [6.3, -0.4, 2.9], "near", [6.0, -0.9])
    far_query = prior.evaluate(fitted, [4.67, 24.17, 2.9], "far", [4.97, 24.67])
    assert far_query["energy"] == pytest.approx(near_query["energy"], abs=1e-9)
    assert far_query["energy"] < 0.05
    assert far_query["branches"]["relative"]["canonical_point_m"] == pytest.approx([0.3, 0.5, 2.9])
    # A rotated-but-wrong far query lands far from the cluster.
    assert prior.evaluate(fitted, [6.3, -0.4, 2.9], "far")["energy"] > 50


def test_target_attempt_is_excluded_and_lone_target_abstains():
    rows = near_cohort(3) + [record("target", [2.0, -3.0, 2.2], feet=[2.0, -3.5])]
    fitted = prior.fit_prior(rows, "A", "deuce", target_attempt_id="target")
    assert fitted["target_excluded"] is True
    assert "target" not in fitted["absolute"]["source_attempt_ids"]
    assert "target" not in fitted["relative"]["source_attempt_ids"]
    assert fitted["absolute"]["clusters"][0]["centre_m"][0] == pytest.approx(6.3, abs=0.2)
    lone = prior.fit_prior(rows[-1:], "A", "deuce", target_attempt_id="target")
    assert lone["absolute"]["status"] == "abstained"
    assert lone["relative"]["status"] == "abstained"
    assert prior.evaluate(lone, [2.0, -3.0, 2.2], "near", [2.0, -3.5])["energy"] == 0
    assert prior.optimization_residuals(lone, [2.0, -3.0, 2.2], "near").shape == (0,)


def test_invalid_root_source_excluded_but_valid_court_contact_allowed():
    rows = near_cohort(2)
    boxed = record("box", [6.2, -0.5, 2.85], feet=[6.0, 2.5])
    boxed["sources"]["ground"] = "automatic_player_box"
    fitted = prior.fit_prior(rows + [boxed], "A", "deuce")
    assert "box" in fitted["absolute"]["source_attempt_ids"]
    assert "box" not in fitted["relative"]["source_attempt_ids"]
    assert fitted["rejected"] == [
        {"attempt_id": "box", "absolute_rejections": [], "relative_rejections": ["ground_source"]}
    ]
    pinned = record("pin", [6.2, 24.5, 2.85], feet=[6.0, -0.9], contact_qualified=False)
    pinned["sources"]["contact"] = "fitted_gate_accepted"
    fitted = prior.fit_prior(rows + [pinned], "A", "deuce")
    assert "pin" not in fitted["absolute"]["source_attempt_ids"]
    assert fitted["rejected"][0]["absolute_rejections"] == [
        "contact_not_qualified",
        "contact_source",
    ]


def test_explicit_selection_duplicates_and_missing_qualification():
    rows = near_cohort(2) + [record("other", [6.3, -0.4, 2.9], player="B")]
    rows += [record("ad", [6.3, -0.4, 2.9], side="ad")]
    fitted = prior.fit_prior(rows, "A", "deuce")
    assert fitted["absolute"]["source_attempt_ids"] == ["n0", "n1"]
    assert prior.fit_prior(rows, "B", "deuce")["absolute"]["source_attempt_ids"] == ["other"]
    with pytest.raises(ValueError):
        prior.fit_prior(rows + [record("n0", [6.0, 0.0, 2.9])], "A", "deuce")
    with pytest.raises(ValueError):
        prior.fit_prior(rows, "A", "left")
    unmarked = record("u", [6.0, 0.0, 2.9])
    del unmarked["ground_qualified"]
    with pytest.raises(ValueError):
        prior.fit_prior([unmarked], "A", "deuce")


def test_unqualified_missing_contact_is_rejected_without_parsing_invalid_geometry():
    missing = record("missing", [0, 0, 0], contact_qualified=False, contact_xyz_m=None)
    fitted = prior.fit_prior([missing], "A", "deuce")
    assert fitted["absolute"]["status"] == "abstained"
    assert fitted["relative"]["status"] == "abstained"
    assert fitted["rejected"][0]["absolute_rejections"] == ["contact_not_qualified"]
    missing["contact_qualified"] = True
    with pytest.raises(ValueError, match="finite court"):
        prior.fit_prior([missing], "A", "deuce")


def test_single_cluster_energy_is_floored_mahalanobis_and_branches_share_weight():
    fitted = prior.fit_prior(near_cohort(3), "A", "deuce")
    cluster = fitted["absolute"]["clusters"][0]
    assert cluster["sigma_m"] == pytest.approx([0.5, 0.75, 0.3])
    centre = np.asarray(cluster["centre_m"])
    query = centre + np.array([0.5, 0.75, 0.3])
    only_absolute = prior.evaluate(fitted, query, "near")
    assert only_absolute["branch_availability"] == {
        "absolute": True,
        "relative": False,
        "ground_supplied": False,
        "note": only_absolute["branch_availability"]["note"],
    }
    assert only_absolute["energy"] == pytest.approx(3.0)
    assert only_absolute["optimization_residuals"] == pytest.approx([np.sqrt(3.0)])
    both = prior.evaluate(fitted, query, "near", [6.0, -0.9], weight=2.0)
    assert set(both["branches"]) == {"absolute", "relative"}
    assert both["branches"]["absolute"]["branch_weight"] == 0.5
    assert both["branches"]["absolute"]["weighted_energy"] == pytest.approx(3.0)
    assert both["selector_penalty"] == pytest.approx(0.5 * both["energy"])
    assert both["hard_gate"] is False and both["automatic_inference_eligible"] is False
    json.dumps(fitted), json.dumps(both)


def test_two_cluster_mixture_is_smooth_and_never_hard_assigned():
    wide = [record(f"w{i}", [8.0 + 0.02 * i, -0.3, 2.9], feet=[7.8, -0.8]) for i in range(4)]
    tee = [record(f"t{i}", [5.2 + 0.02 * i, -0.3, 2.9], feet=[5.0, -0.8]) for i in range(4)]
    fitted = prior.fit_prior(wide + tee, "A", "deuce")
    assert fitted["absolute"]["mode"] == "two_cluster"
    assert sorted(c["count"] for c in fitted["absolute"]["clusters"]) == [4, 4]
    xs = np.linspace(4.5, 8.7, 400)
    energies = np.array([prior.evaluate(fitted, [x, -0.3, 2.9], "near")["energy"] for x in xs])
    assert np.all(energies >= 0)
    assert np.max(np.abs(np.diff(energies))) < 0.2  # no jump at the midpoint hand-over
    midpoint = prior.evaluate(fitted, [6.6, -0.3, 2.9], "near")
    nearest = min(midpoint["branches"]["absolute"]["cluster_mahalanobis2"])
    # Soft mixture: bounded by the nearest cluster plus the weight term, never a hard pick.
    assert nearest <= midpoint["energy"] <= nearest + 2 * np.log(2) + 1e-9
    at_centre = prior.evaluate(fitted, [8.03, -0.3, 2.9], "near")["energy"]
    assert at_centre == pytest.approx(2 * np.log(2), abs=0.05)
    few = prior.fit_prior(wide[:2] + tee[:2], "A", "deuce")
    assert few["absolute"]["mode"] == "single_robust_centre"
