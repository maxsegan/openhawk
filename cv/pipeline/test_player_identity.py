"""Tests for once-per-match player identity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cv.pipeline.player_identity import (
    PointAppearance,
    anchor_from_scoreboard,
    board_name_map,
    changeovers_before,
    changeovers_in_tiebreak,
    chroma_histogram,
    cluster_two_identities,
    combine_descriptor,
    constrain_by_parity,
    ends_parity,
    names_are_safe,
    names_by_clip,
    parity_from_scores,
    remove_side_bias,
    resolve_match,
    roster_from_match_id,
    serving_sides,
    shirt_box,
    standardised_depths,
    torso_box,
    torso_colour_histogram,
    write_identity,
)


def look(seed: int, noise: float = 0.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    base = np.zeros(8, dtype=np.float32)
    base[seed % 8] = 1.0
    if noise:
        base = base + generator.normal(scale=noise, size=8).astype(np.float32)
    return base / np.linalg.norm(base)


@pytest.mark.parametrize(
    ("match_id", "expected", "confidence"),
    [
        ("wim2025f_m_sinner_alcaraz", ("sinner", "alcaraz"), "high"),
        ("ao2019f_w_osaka_kvitova", ("osaka", "kvitova"), "high"),
        (
            "atp_2025_540_f_226_jannik_sinner_carlos_alcaraz",
            ("jannik sinner", "carlos alcaraz"),
            "high",
        ),
        ("rotterdam2025f_m_alcaraz_de_minaur", ("alcaraz", "de minaur"), "low"),
        (
            "wta_2026_580_r128_110_yulia_putintseva_beatriz_haddad_maia",
            ("yulia putintseva", "beatriz haddad maia"),
            "low",
        ),
    ],
)
def test_roster_from_match_id(match_id, expected, confidence):
    roster = roster_from_match_id(match_id)
    assert roster.names == expected
    assert roster.confidence == confidence


def test_roster_gives_up_on_an_unparseable_id():
    assert roster_from_match_id("some_random_clip").names is None


def test_changeovers_follow_the_itf_rule():
    # ends change after games 1, 3, 5, ... of a set
    assert [changeovers_before([], games) for games in range(9)] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    # a set finished on an even total defers the change to after game 1 of the next set
    assert changeovers_before([10], 0) == 5
    assert changeovers_before([10], 1) == 6
    # a set finished on an odd total has already changed
    assert changeovers_before([9], 0) == 5
    assert changeovers_before([9], 1) == 6


def test_ends_parity_is_anchored_on_the_first_point():
    points = [
        {"clip": "pt0001", "set_lengths": [], "games_completed_in_set": 0},
        {"clip": "pt0002", "set_lengths": [], "games_completed_in_set": 1},
        {"clip": "pt0003", "set_lengths": [], "games_completed_in_set": 2},
        {"clip": "pt0004", "set_lengths": [], "games_completed_in_set": 3},
        {"clip": "pt0005", "set_lengths": [6], "games_completed_in_set": 0},
    ]
    assert ends_parity(points) == {
        "pt0001": 0,
        "pt0002": 1,
        "pt0003": 1,
        "pt0004": 0,
        "pt0005": 1,
    }


def test_cluster_follows_a_mid_match_swap():
    a, b = look(1), look(2)
    points = [
        PointAppearance("pt0001", near=a, far=b, near_frames=10, far_frames=10),
        PointAppearance("pt0002", near=a, far=b, near_frames=10, far_frames=10),
        PointAppearance("pt0003", near=b, far=a, near_frames=10, far_frames=10),
    ]
    result = cluster_two_identities(points)
    assert result.orientation["pt0001"] == result.orientation["pt0002"]
    assert result.orientation["pt0003"] != result.orientation["pt0001"]
    assert result.within < result.between


def test_cluster_places_a_one_sided_point():
    a, b = look(1), look(2)
    points = [
        PointAppearance("pt0001", near=a, far=b, near_frames=10, far_frames=10),
        PointAppearance("pt0002", near=b, far=None, near_frames=10),
    ]
    result = cluster_two_identities(points)
    assert result.orientation["pt0002"][0] == result.orientation["pt0001"][1]


def test_resolve_match_names_and_cross_checks_parity():
    a, b = look(1, 0.05), look(2, 0.05)
    points = [
        PointAppearance("pt0001", near=a, far=b, near_frames=90, far_frames=80),
        PointAppearance("pt0002", near=b, far=a, near_frames=70, far_frames=95),
    ]
    games = [
        {"clip": "pt0001", "set_lengths": [], "games_completed_in_set": 0},
        {"clip": "pt0002", "set_lengths": [], "games_completed_in_set": 1},
    ]
    payload = resolve_match("wim2025f_m_sinner_alcaraz", points, games=games)
    assert payload["roster"]["names"] == ["sinner", "alcaraz"]
    assert {identity["name"] for identity in payload["identities"]} == {"sinner", "alcaraz"}
    assert payload["identities"][0]["name_source"] == "roster_order_unanchored"
    assert payload["parity"]["agreement"] == 1.0
    near_names = {row["clip"]: row["near_name"] for row in payload["points"]}
    assert near_names["pt0001"] != near_names["pt0002"]
    assert payload["appearance"]["separation"] > 0


def test_resolve_match_reports_parity_disagreement():
    a, b = look(1), look(2)
    points = [
        PointAppearance("pt0001", near=a, far=b, near_frames=9, far_frames=9),
        PointAppearance("pt0002", near=a, far=b, near_frames=9, far_frames=9),
    ]
    games = [
        {"clip": "pt0001", "set_lengths": [], "games_completed_in_set": 0},
        {"clip": "pt0002", "set_lengths": [], "games_completed_in_set": 1},
    ]
    payload = resolve_match("wim2025f_m_sinner_alcaraz", points, games=games)
    assert payload["parity"]["disagree"] == 1
    assert payload["parity"]["agreement"] == 0.5


def test_explicit_anchor_overrides_roster_order():
    a, b = look(1), look(2)
    points = [PointAppearance("pt0001", near=a, far=b, near_frames=5, far_frames=5)]
    payload = resolve_match(
        "wim2025f_m_sinner_alcaraz", points, anchor={"A": "alcaraz", "B": "sinner"}
    )
    assert payload["identities"][0]["name"] == "alcaraz"
    assert payload["identities"][0]["name_source"] == "explicit_anchor"
    assert payload["roster"]["anchored"] is True


def test_write_and_read_back(tmp_path):
    points = [PointAppearance("pt0001", near=look(1), far=look(2), near_frames=3, far_frames=4)]
    payload = resolve_match(
        "ao2019f_w_osaka_kvitova", points, anchor={"A": "osaka", "B": "kvitova"}
    )
    path = tmp_path / "player_identity_v1.json"
    write_identity(path, payload)
    restored = json.loads(path.read_text())
    assert restored["schema"] == "tennis.player_identity.v1"
    assert set(names_by_clip(restored)["pt0001"]) == {"near", "far"}
    assert names_by_clip(restored)["pt0001"]["near"] in {"osaka", "kvitova"}


def test_unanchored_roster_names_are_not_safe_to_emit():
    points = [PointAppearance("pt0001", near=look(1), far=look(2), near_frames=3, far_frames=4)]
    payload = resolve_match("ao2019f_w_osaka_kvitova", points)

    assert payload["identities"][0]["name_source"] == "roster_order_unanchored"
    assert names_are_safe(payload) is False
    assert names_by_clip(payload) == {}


# --------------------------------------------------------------------------- tiebreak ends


def test_tiebreak_changes_ends_every_six_points():
    assert changeovers_in_tiebreak(0) == 0
    assert changeovers_in_tiebreak(5) == 0
    assert changeovers_in_tiebreak(6) == 1
    assert changeovers_in_tiebreak(11) == 1  # a tiebreak won 7-4
    assert changeovers_in_tiebreak(12) == 2
    assert changeovers_in_tiebreak(18) == 3


def test_a_short_tiebreak_flips_the_parity_of_the_rest_of_the_match():
    # A set won 7-6 is thirteen games, so the game rule alone counts seven changes; the
    # tiebreak adds one more when it ends 7-4 and two when it runs to twelve points or
    # more, so a short tiebreak flips the parity of everything after it.
    short = changeovers_before([13], 0, completed_tiebreak_points=[11])
    long = changeovers_before([13], 0, completed_tiebreak_points=[12])
    assert (short, long) == (8, 9)
    assert short % 2 != long % 2
    assert changeovers_before([13], 0) == long  # twelve points is the assumption


def test_parity_changes_inside_a_tiebreak():
    rows = [
        {
            "clip": "a",
            "set_number": 1,
            "games_1": 6,
            "games_2": 6,
            "points_1": "0",
            "points_2": "0",
            "completed_sets": "",
        },
        {
            "clip": "b",
            "set_number": 1,
            "games_1": 6,
            "games_2": 6,
            "points_1": "4",
            "points_2": "2",
            "completed_sets": "",
        },
        {
            "clip": "c",
            "set_number": 1,
            "games_1": 6,
            "games_2": 6,
            "points_1": "5",
            "points_2": "3",
            "completed_sets": "",
        },
    ]
    assert parity_from_scores(rows) == {"a": 0, "b": 1, "c": 1}


def test_parity_from_scores_reads_completed_sets_and_their_tiebreaks():
    def rows(points_1, points_2):
        return [
            {
                "clip": "a",
                "set_number": 1,
                "games_1": 0,
                "games_2": 0,
                "points_1": "0",
                "points_2": "0",
                "completed_sets": "",
            },
            {
                "clip": "b",
                "set_number": 1,
                "games_1": 6,
                "games_2": 6,
                "points_1": points_1,
                "points_2": points_2,
                "completed_sets": "",
            },
            {
                "clip": "c",
                "set_number": 2,
                "games_1": 0,
                "games_2": 0,
                "points_1": "0",
                "points_2": "0",
                "completed_sets": "7-6",
            },
        ]

    # a tiebreak won 7-4 is eleven points: one change inside it, so the next set starts
    # from the other ends than a tiebreak that ran to thirteen
    assert parity_from_scores(rows("7", "4")) == {"a": 0, "b": 1, "c": 0}
    assert parity_from_scores(rows("7", "6")) == {"a": 0, "b": 0, "c": 1}


# --------------------------------------------------------------------------- descriptor


def test_torso_and_shirt_boxes_sit_inside_the_person_box():
    box = (100.0, 200.0, 200.0, 600.0)
    assert torso_box(box) == (100.0, 200.0, 200.0, 440.0)
    x0, y0, x1, y1 = shirt_box(box)
    assert (x0, y0, x1, y1) == (120.0, 272.0, 180.0, 400.0)


def test_chroma_histogram_travels_across_exposure_better_than_the_full_lab_one():
    pytest.importorskip("cv2")

    def patch(blue, green, red):
        generator = np.random.default_rng(0)
        base = np.zeros((20, 20, 3), dtype=np.float32)
        base[..., 0], base[..., 1], base[..., 2] = blue, green, red
        base = base + generator.normal(scale=8, size=base.shape)
        return np.clip(base, 0, 255).astype(np.uint8)

    bright = patch(60, 180, 60)  # a green shirt in the sun
    shaded = patch(45, 135, 45)  # the same shirt in shade
    other = patch(200, 205, 215)  # a white shirt

    chroma_same = float(np.dot(chroma_histogram(bright), chroma_histogram(shaded)))
    lab_same = float(np.dot(torso_colour_histogram(bright), torso_colour_histogram(shaded)))
    assert chroma_same > lab_same
    assert float(np.dot(chroma_histogram(bright), chroma_histogram(other))) < chroma_same
    assert np.isclose(np.linalg.norm(chroma_histogram(bright)), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(torso_colour_histogram(bright)), 1.0, atol=1e-5)
    assert torso_colour_histogram(np.zeros((0, 0, 3), np.uint8)).sum() == 0


def test_combine_descriptor_normalises_each_part_before_weighting():
    long_part = np.ones(64, dtype=np.float32) * 10.0
    short_part = np.array([0.0, 1.0], dtype=np.float32)
    combined = combine_descriptor({"a": long_part, "b": short_part}, weights={"a": 1.0, "b": 1.0})
    assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-5)
    # equal weight means equal energy, whatever the dimensions
    assert np.isclose(
        float(np.linalg.norm(combined[:64])), float(np.linalg.norm(combined[64:])), atol=1e-5
    )


def test_remove_side_bias_recovers_identity_when_the_side_offset_dominates():
    person = {
        "x": np.array([1.0, 0.0, 0.0], np.float32),
        "y": np.array([0.0, 1.0, 0.0], np.float32),
    }
    bias = np.array([0.0, 0.0, 6.0], np.float32)  # every near crop is big and sharp
    points = [
        PointAppearance("pt0001", near=person["x"] + bias, far=person["y"]),
        PointAppearance("pt0002", near=person["y"] + bias, far=person["x"]),
    ]
    raw = cluster_two_identities(points)
    assert raw.orientation["pt0001"] == raw.orientation["pt0002"]  # split by side, not player
    centred = cluster_two_identities(remove_side_bias(points))
    assert centred.orientation["pt0001"] != centred.orientation["pt0002"]


def test_clustering_per_set_survives_a_kit_change():
    set_one = {"x": look(1), "y": look(2)}
    set_two = {"x": look(3), "y": look(4)}  # both players changed shirts at the break
    points = [
        PointAppearance("pt0001", near=set_one["x"], far=set_one["y"]),
        PointAppearance("pt0002", near=set_one["y"], far=set_one["x"]),
        PointAppearance("pt0003", near=set_two["x"], far=set_two["y"]),
        PointAppearance("pt0004", near=set_two["y"], far=set_two["x"]),
    ]
    groups = {"pt0001": "set1", "pt0002": "set1", "pt0003": "set2", "pt0004": "set2"}
    result = cluster_two_identities(points, groups=groups)
    assert result.orientation["pt0001"] != result.orientation["pt0002"]
    assert result.orientation["pt0003"] != result.orientation["pt0004"]
    assert set(result.group_link_margin) == {"set2"}


# --------------------------------------------------------------------------- witnesses


def test_parity_overrules_a_single_appearance_outlier():
    points = [PointAppearance(f"pt000{i}", near=look(1), far=look(2)) for i in range(1, 5)]
    clusters = cluster_two_identities(points)
    clusters.orientation["pt0003"] = "BA"  # one point the descriptor got wrong
    clusters.margin["pt0003"] = 0.01
    parity = {"pt0001": 0, "pt0002": 0, "pt0003": 0, "pt0004": 0}
    orientation, confidence = constrain_by_parity(clusters, parity)
    assert set(orientation.values()) == {"AB"}
    assert confidence > 0.9


def test_serving_side_is_the_shallower_stance():
    depths = standardised_depths(
        [
            {"clip": "a", "near_depth": 0.0, "far_depth": 4.0},
            {"clip": "b", "near_depth": 4.0, "far_depth": 0.0},
            {"clip": "c", "near_depth": 0.0, "far_depth": 4.0},
            {"clip": "d", "near_depth": 4.0, "far_depth": 0.0},
        ]
    )
    assert serving_sides(depths) == {"a": "near", "b": "far", "c": "near", "d": "far"}


def test_scoreboard_anchor_takes_the_majority_and_reports_its_share():
    rows = [
        {"clip": "a", "near_identity": "A", "server_row": 1, "serving_side": "near"},
        {"clip": "b", "near_identity": "A", "server_row": 2, "serving_side": "far"},
        {"clip": "c", "near_identity": "B", "server_row": 1, "serving_side": "far"},
        {"clip": "d", "near_identity": "A", "server_row": 2, "serving_side": "near"},
    ]
    report = anchor_from_scoreboard(rows, board_names={1: "sinner", 2: "alcaraz"})
    assert report["identity_to_row"] == {"A": 1, "B": 2}
    assert report["names"] == {"A": "sinner", "B": "alcaraz"}
    assert report["points_voting"] == 4
    assert report["confidence"] == 0.75


def test_board_names_are_matched_to_the_roster_spelling():
    roster = roster_from_match_id("atp_2025_540_f_226_jannik_sinner_carlos_alcaraz")
    assert board_name_map({1: "SINNER", 2: "ALCARAZ"}, roster) == {
        1: "jannik sinner",
        2: "carlos alcaraz",
    }
    # an unrecognised board keeps its own text rather than borrowing a roster name
    assert board_name_map({1: "MURRAY"}, roster) == {1: "murray"}


def test_resolve_match_anchors_names_on_the_scoreboard_server():
    scores = [
        {
            "clip": f"pt{index:04d}",
            "server": 1 if index % 2 else 2,
            "set_number": 1,
            "games_1": index - 1,
            "games_2": 0,
            "points_1": "0",
            "points_2": "0",
            "completed_sets": "",
        }
        for index in range(1, 41)
    ]
    parity = parity_from_scores(scores)
    points, depths = [], []
    for row in scores:
        clip = row["clip"]
        flipped = parity[clip] == 1
        # identity A is near from the anchor ends and far after a changeover
        near, far = (look(2), look(1)) if flipped else (look(1), look(2))
        points.append(PointAppearance(clip, near=near, far=far, near_frames=9, far_frames=9))
        serving_near = (row["server"] == 1) != flipped  # row 1 is identity A
        depths.append(
            {
                "clip": clip,
                "near_depth": 0.0 if serving_near else 4.0,
                "far_depth": 4.0 if serving_near else 0.0,
            }
        )
    payload = resolve_match(
        "wim2025f_m_sinner_alcaraz",
        points,
        scores=scores,
        depths=depths,
        board_names={1: "SINNER", 2: "ALCARAZ"},
    )
    assert payload["witnesses"]["agree"] is True
    assert payload["identities"][0]["name_source"] == "scoreboard_server_anchor"
    assert {row["identity"]: row["name"] for row in payload["identities"]} == {
        "A": "sinner",
        "B": "alcaraz",
    }
    assert payload["server_anchor"]["confidence"] == 1.0
    assert payload["parity"]["agreement"] == 1.0
    assert payload["appearance"]["side_bias_removed"] is True
    assert names_by_clip(payload)


def test_three_consistent_audit_points_can_anchor_the_scoreboard_names():
    points = [
        PointAppearance(f"pt000{index}", near=look(1), far=look(2), near_frames=9, far_frames=9)
        for index in range(1, 4)
    ]
    scores = [
        {
            "clip": point.clip,
            "server": 1,
            "set_number": 1,
            "games_1": 0,
            "games_2": 0,
            "points_1": "0",
            "points_2": "0",
            "completed_sets": "",
        }
        for point in points
    ]
    depths = [{"clip": point.clip, "near_depth": 0.0, "far_depth": 4.0} for point in points]

    payload = resolve_match(
        "wim2025f_m_sinner_alcaraz",
        points,
        scores=scores,
        depths=depths,
        board_names={1: "SINNER", 2: "ALCARAZ"},
    )

    assert payload["roster"]["anchored"] is True
    assert payload["server_anchor"]["points_voting"] == 3


def test_serve_start_side_overrides_stance_depth_and_falls_back_when_absent():
    points = [
        PointAppearance(f"pt000{index}", near=look(1), far=look(2), near_frames=9, far_frames=9)
        for index in range(1, 4)
    ]
    scores = [
        {
            "clip": point.clip,
            "server": 1,
            "set_number": 1,
            "games_1": 0,
            "games_2": 0,
            "points_1": "0",
            "points_2": "0",
            "completed_sets": "",
        }
        for point in points
    ]
    # Stance says far for all three; the direct serve-start witness corrects two clips.
    depths = [
        {"clip": point.clip, "near_depth": float(index), "far_depth": 0.0}
        for index, point in enumerate(points, 4)
    ]
    payload = resolve_match(
        "wim2025f_m_sinner_alcaraz",
        points,
        scores=scores,
        depths=depths,
        serve_sides={"pt0001": "near", "pt0002": "near", "bad": "centre"},
        board_names={1: "SINNER", 2: "ALCARAZ"},
    )

    by_clip = {row["clip"]: row for row in payload["points"]}
    assert by_clip["pt0001"]["serving_side"] == "near"
    assert by_clip["pt0001"]["serving_side_source"] == "serve_start"
    assert by_clip["pt0003"]["serving_side"] == "far"
    assert by_clip["pt0003"]["serving_side_source"] == "stance_depth"
    assert payload["server_anchor"]["side_source_counts"] == {
        "serve_start": 2,
        "stance_depth": 1,
    }
    assert payload["server_anchor"]["confidence"] == pytest.approx(2 / 3, abs=1e-4)


def test_names_are_withheld_when_the_witnesses_disagree():
    points, scores = [], []
    for index in range(1, 21):
        clip = f"pt{index:04d}"
        # appearance says the players never swapped; parity says they swap every game
        points.append(PointAppearance(clip, near=look(1), far=look(2), near_frames=9, far_frames=9))
        scores.append(
            {
                "clip": clip,
                "server": 1,
                "set_number": 1,
                "games_1": index - 1,
                "games_2": 0,
                "points_1": "0",
                "points_2": "0",
                "completed_sets": "",
            }
        )
    payload = resolve_match("wim2025f_m_sinner_alcaraz", points, scores=scores)
    assert payload["witnesses"]["agree"] is False
    assert payload["identities"][0]["name_source"] == "withheld_witness_disagreement"
    assert all(row["near_name"] == "" and row["far_name"] == "" for row in payload["points"])
    assert names_are_safe(payload) is False
    assert names_by_clip(payload) == {}
