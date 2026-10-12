import csv
import json

from cv.pipeline import player_names as names

MATCH = "wta_2023_580_f_226_aryna_sabalenka_elena_rybakina"
METADATA = {
    "match_key": MATCH,
    "winner_name": "Aryna Sabalenka",
    "loser_name": "Elena Rybakina",
    "score": "4-6 6-3 6-4",
}
# Second set after a 6-4 first set (10 games: 5 end changes), so play starts swapped.
SET2 = {"set_number": "2", "completed_sets": "6-4"}


def ledger_rows():
    """Row 1 (Rybakina) serves game 1 of set 2, row 2 game 2, and so on; four points each."""
    rows = []
    pt = 0
    for games_1, games_2, server in ((0, 0, 1), (1, 0, 2), (1, 1, 1), (2, 1, 2)):
        for points in ("0", "15", "30", "40"):
            pt += 1
            rows.append(
                {
                    "pt": str(pt),
                    "server": str(server),
                    **SET2,
                    "games_1": str(games_1),
                    "games_2": str(games_2),
                    "points_1": points,
                    "points_2": "0",
                    "score_confidence": "1.0",
                    "point_index": str(pt),
                }
            )
    return rows


def serve_sides(rows, row1_near_at_start=True):
    """Who served from which side, under the real end changes of the match."""
    counts = names.changeover_counts(rows)
    sides = {}
    for row in rows:
        row1_near = row1_near_at_start != bool(counts[int(row["pt"])] % 2)
        sides[int(row["pt"])] = "near" if (row["server"] == "1") == row1_near else "far"
    return sides


def test_display_labels():
    assert names.display_label("Aryna Sabalenka") == "Sab."
    assert names.display_label("Elena Rybakina") == "Rybakina"
    assert names.display_label("Alex de Minaur") == "De Minaur"
    assert names.display_label("Felix Auger-Aliassime") == "Aug-Ali."
    assert names.display_label("Giovanni Mpetshi Perricard") == "Mpetshi"
    assert names.display_label("Alejandro Davidovich Fokina") == "Dav."
    assert names.display_label(None) is None
    assert names.display_labels(("Serena Williams", "Venus Williams")) == (
        "S. Williams",
        "V. Williams",
    )


def test_changeover_counts_follow_itf_rule():
    rows = ledger_rows()
    counts = names.changeover_counts(rows)
    # 6-4 first set = 5 changes; then after game 1 of set 2, then after game 3.
    assert [counts[p] for p in (1, 5, 9, 13)] == [5, 6, 6, 7]
    tiebreak = {
        "pt": "99",
        "server": "1",
        "set_number": "1",
        "games_1": "6",
        "games_2": "6",
        "points_1": "4",
        "points_2": "3",
        "completed_sets": "",
    }
    # Twelve games (6 changes) plus one change after the first six tiebreak points.
    assert names.changeover_counts([tiebreak])[99] == 7


def test_row_order_from_metadata_score():
    rows = [dict(r, set_number="1", completed_sets="") for r in ledger_rows()]
    # Row 1 reaching 2 games and row 2 one game fits either order in set 1 (4-6 / 6-4)...
    assert names.row_order(rows, names.parse_sets("4-6 6-3 6-4"))["order"] is None
    # ...but a completed first set read 6-4 on the board says row 1 lost the match.
    report = names.row_order(ledger_rows(), names.parse_sets("4-6 6-3 6-4"))
    assert report["order"] == "row1_loser"
    assert names.parse_sets("6-3 6-7(5) 6-4") == [(6, 3), (6, 7), (6, 4)]


def test_resolve_names_follow_ends_changes():
    rows = ledger_rows()
    sides = serve_sides(rows)
    del sides[3]
    report = names.resolve(MATCH, rows, sides, METADATA)
    assert report["status"] == "named"
    assert report["row_names"] == {"1": "Elena Rybakina", "2": "Aryna Sabalenka"}
    first, after_change = report["attempts"]["1"], report["attempts"]["5"]
    # Five changes before set 2: Rybakina (row 1, near at the start) is far in game 1.
    assert (first["near"], first["far"], first["server"]) == (
        "Aryna Sabalenka",
        "Elena Rybakina",
        "Elena Rybakina",
    )
    assert (after_change["near"], after_change["far"]) == ("Elena Rybakina", "Aryna Sabalenka")
    assert first["source"] == "direct" and first["near_label"] == "Sab."
    assert report["attempts"]["3"]["source"] == "propagated"
    assert report["counts"]["direct"] == 15 and report["counts"]["propagated"] == 1


def test_disagreement_and_abstentions():
    rows = ledger_rows()
    sides = serve_sides(rows)
    sides[2] = "far" if sides[2] == "near" else "near"
    rows[9] = {**rows[9], "server": "", "games_1": "", "point_index": "x"}
    rows[4] = {**rows[4], "server": "", "games_1": "", "point_index": "y"}
    report = names.resolve(MATCH, rows, sides, METADATA)
    assert report["status"] == "named"
    two = report["attempts"]["2"]
    assert (two["source"], two["disagreement"]) == ("propagated", True)
    assert two["near"] == report["attempts"]["1"]["near"]
    assert two["confidence"] == report["attempts"]["1"]["confidence"] / 2
    assert report["counts"]["evidence_rule_disagreements"] == 1
    # Attempt 10 sits between two scored attempts of one game: rules and server carry over.
    ten = report["attempts"]["10"]
    assert (ten["placed_by"], ten["source"], ten["server"]) == (
        "scored_neighbours",
        "propagated",
        "Elena Rybakina",
    )
    # Attempt 5 opens game 2 after a change: its neighbours' counts differ, so it stays null.
    assert report["attempts"]["5"]["source"] == "unresolved"


def test_unresolved_without_metadata_or_side_votes():
    rows = ledger_rows()
    report = names.resolve(MATCH, rows, serve_sides(rows), None)
    assert report["status"] == "unresolved"
    assert report["counts"]["unresolved"] == len(rows)
    report = names.resolve(MATCH, rows, {}, METADATA)
    assert report["status"] == "unresolved"
    assert "serve sides" in report["failures"][0]


def test_run_writes_identity_into_outputs(tmp_path):
    rows = ledger_rows()
    out = tmp_path / "out"
    ledger = out / "upstream" / MATCH / names.LEDGER
    ledger.parent.mkdir(parents=True)
    with ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = tmp_path / "matches.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METADATA))
        writer.writeheader()
        writer.writerow(METADATA)
    sides = serve_sides(rows)
    data = out / "portal" / "3d" / "data"
    data.mkdir(parents=True)
    for pt in range(1, 17):
        doc = {
            "players": {"names": {"near": None, "far": None}},
            "contacts": [{"phase": "serve", "side": sides[pt]}, {"phase": "rally", "side": "far"}],
        }
        (data / f"automatic_s6_{MATCH}__pt{pt:04d}.json").write_text(json.dumps(doc))
    (data / f"automatic_s6_{MATCH}__pt0001__component_0.json").write_text(
        json.dumps({"players": {"names": {"near": None, "far": None}}})
    )
    flight = {"role": "return", "start_xyz_m": [4.0, 23.0, 1.0]}
    (out / "MATCH_RESULT.json").write_text(
        json.dumps({"attempts": [{"key": f"{MATCH}__pt0001", "accepted_flights": [flight]}]})
    )
    summary = names.run(out, MATCH, metadata, out / "portal" / "3d")
    assert summary["status"] == "named"
    assert summary["documents_stamped"] == 17
    assert summary["counts"]["direct"] == 16
    component = json.loads((data / f"automatic_s6_{MATCH}__pt0001__component_0.json").read_text())
    assert component["players"]["names"] == {"near": "Aryna Sabalenka", "far": "Elena Rybakina"}
    assert component["players"]["labels"] == {"near": "Sab.", "far": "Rybakina"}
    document = json.loads((data / f"automatic_s6_{MATCH}__pt0001.json").read_text())
    assert [c["player"] for c in document["contacts"]] == ["Elena Rybakina", "Elena Rybakina"]
    result = json.loads((out / "MATCH_RESULT.json").read_text())
    attempt = result["attempts"][0]
    assert attempt["player_identity"]["server"] == "Elena Rybakina"
    assert attempt["accepted_flights"][0]["hitter"] == "Elena Rybakina"
    assert result["player_identity"]["counts"]["direct"] == 16
    assert (out / names.FILE).is_file() and (data / "player_names.json").is_file()


def test_metadata_tiebreak_length_sets_the_following_set():
    assert names.tiebreak_lengths("7-6(5) 3-6 7-6(10)") == {1: 12, 3: 22}
    row = {
        "pt": "1",
        "server": "1",
        "set_number": "2",
        "games_1": "0",
        "games_2": "0",
        "points_1": "0",
        "points_2": "0",
        "completed_sets": "7-6",
    }
    seen = {
        **row,
        "pt": "0",
        "set_number": "1",
        "games_1": "6",
        "games_2": "6",
        "points_1": "4",
        "points_2": "5",
        "completed_sets": "",
    }
    # Seen up to 4-5 (9 points: one change inside). The metadata says 7-5: 12 points, still one
    # change inside, because the change after point 12 is the set-end change. A 7-6(10) tiebreak
    # is 22 points: three changes inside.
    assert names.changeover_counts([seen, row])[1] == 7 + 1
    assert names.changeover_counts([seen, row], {1: 12})[1] == 7 + 1
    assert names.changeover_counts([seen, row], {1: 22})[1] == 7 + 3
