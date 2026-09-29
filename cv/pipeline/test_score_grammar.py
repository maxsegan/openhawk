import json
import sys

import pytest

from cv.pipeline.score_grammar import (
    ScoreState,
    _tiebreak_server,
    award_point,
    decode,
    parse_read,
    score_runs_observations,
    state_from_read,
)


def test_game_and_serve_advance_together() -> None:
    state = ScoreState((), (0, 0), (0, 0), 1)
    for _ in range(4):
        state = award_point(state, 0)

    assert state.as_read()["g1"] == "1"
    assert state.as_read()["p1"] == "0"
    assert state.server == 2


def test_points_only_board_rows_are_evidence_when_games_column_is_hidden():
    from cv.pipeline.score_grammar import board_reads_from_raw, board_emission_score

    records = [
        {
            "i": 537,
            "raw": {
                "present": True,
                "serving_row": 1,
                "rows": [
                    {"name": "A", "values": [], "points": "15"},
                    {"name": "B", "values": [], "points": "0"},
                ],
            },
        }
    ]
    assert not board_reads_from_raw(records)[0].usable
    read = board_reads_from_raw(records, points_only_boards=True)[0]
    assert read.usable
    assert read.t == 537
    assert read.rows[0].columns == ()
    correct = ScoreState((), (0, 0), (1, 0), 1)
    wrong = ScoreState((), (0, 0), (0, 1), 1)
    assert board_emission_score(correct, read) > board_emission_score(wrong, read)


def test_empty_named_rows_without_score_tokens_remain_unusable():
    from cv.pipeline.score_grammar import board_reads_from_raw

    records = [
        {
            "raw": {
                "present": True,
                "rows": [
                    {"name": "A", "values": [], "points": None},
                    {"name": "B", "values": [], "points": None},
                ],
            }
        }
    ]
    assert not board_reads_from_raw(records)[0].usable


def test_deuce_needs_two_points() -> None:
    deuce = ScoreState((), (3, 3), (3, 3), 1)

    advantage = award_point(deuce, 0)
    assert advantage.as_read()["p1"] == "AD"
    assert award_point(advantage, 1).points == (3, 3)
    assert award_point(advantage, 0).games == (4, 3)


def test_set_completion_records_the_set_and_resets_games() -> None:
    state = ScoreState((), (5, 3), (3, 0), 1)

    won = award_point(state, 0)

    assert won.completed == ((6, 3),)
    assert won.games == (0, 0)
    assert won.sets_won() == (1, 0)


def test_tiebreak_uses_integer_points_and_alternating_serve() -> None:
    tiebreak = ScoreState(((6, 4),), (6, 6), (6, 5), 1)

    assert tiebreak.as_read()["p1"] == "6"
    assert _tiebreak_server(tiebreak) == 1
    won = award_point(tiebreak, 0)
    assert won.completed == ((6, 4), (7, 6))
    assert won.server == 2


def test_state_from_read_rejects_impossible_points() -> None:
    assert (
        state_from_read({"g1": "2", "g2": "1", "p1": "AD", "p2": "AD", "sets1": "", "sets2": ""}, 1)
        is None
    )
    assert (
        state_from_read({"g1": "2", "g2": "1", "p1": "7", "p2": "0", "sets1": "", "sets2": ""}, 1)
        is None
    )
    assert (
        state_from_read({"g1": "2", "g2": "1", "p1": "30", "p2": "0", "sets1": "", "sets2": ""}, 1)
        is not None
    )


def test_decode_repairs_a_single_misread_second() -> None:
    def read(g1: str, p1: str) -> dict:
        return {"g1": g1, "g2": "0", "p1": p1, "p2": "0", "sets1": "", "sets2": ""}

    observations = []
    moment = 0.0
    for value in ["0"] * 20 + ["15"] * 20:
        observations.append((moment, read("0", value), None))
        moment += 1.0
    observations[25] = (observations[25][0], read("4", "40"), None)  # one bad crop

    path = decode(observations)

    assert path[25].as_read()["g1"] == "0"
    assert path[25].as_read()["p1"] == "15"
    assert path[0].as_read()["p1"] == "0"


def test_score_runs_expand_to_one_observation_per_second() -> None:
    rows = [
        {
            "t_start": "0",
            "t_end": "3",
            "g1": "0",
            "g2": "0",
            "p1": "0",
            "p2": "0",
            "sets1": "",
            "sets2": "",
        },
    ]

    observations = score_runs_observations(rows)

    assert len(observations) == 3
    assert parse_read(rows[0])["g1"] == "0"


def test_a_seed_read_with_an_impossible_completed_set_is_rejected() -> None:
    from cv.pipeline.score_grammar import legal_completed_set, legal_running_games

    assert legal_completed_set((6, 3))
    assert legal_completed_set((7, 6))
    assert legal_completed_set((7, 5))
    assert not legal_completed_set((7, 4))
    assert not legal_completed_set((8, 6))
    read = {"g1": "0", "g2": "1", "p1": "0", "p2": "0", "sets1": "6 7", "sets2": "3 4"}
    assert not legal_running_games((6, 4))
    assert legal_running_games((6, 5))
    assert state_from_read(read, 1) is None
    assert state_from_read({**read, "sets2": "3 5"}, 1) is not None


def test_the_deciding_set_tiebreak_runs_to_ten() -> None:
    from cv.pipeline.score_grammar import MatchRules

    rules = MatchRules(best_of=5)
    fifth = ScoreState(((6, 4), (7, 6), (4, 6), (6, 7)), (6, 6), (6, 4), 1)
    assert fifth.is_deciding_set(rules)
    assert fifth.tiebreak_target(rules) == 10
    # 7-4 does not finish a deciding-set tiebreak
    assert award_point(fifth, 0, rules).completed == fifth.completed
    ten = ScoreState(((6, 4), (7, 6), (4, 6), (6, 7)), (6, 6), (9, 4), 1)
    assert award_point(ten, 0, rules).completed[-1] == (7, 6)
    # an ordinary set tiebreak still ends at seven
    second = ScoreState(((6, 4),), (6, 6), (6, 4), 1)
    assert award_point(second, 0, rules).completed[-1] == (7, 6)


def test_a_finished_match_has_no_successors() -> None:
    from cv.pipeline.score_grammar import MatchRules, legal_set_line, successors

    rules = MatchRules(best_of=3)
    over = ScoreState(((6, 4), (6, 3)), (0, 0), (0, 0), 1)
    assert over.is_finished(rules)
    assert successors(over, rules) == []
    assert not legal_set_line(((6, 4), (6, 3), (6, 2)), rules)
    assert legal_set_line(((6, 4), (3, 6), (6, 2)), rules)


def test_a_hidden_games_column_is_not_evidence_against_a_completed_set() -> None:
    """Wimbledon hides the 0-0 games box, so a two-column board reads as one column."""
    from cv.pipeline.score_grammar import BoardRead, RowRead, board_emission_score

    after_set_one = ScoreState(((4, 6),), (0, 0), (0, 0), 2)
    still_in_set_one = ScoreState((), (5, 5), (0, 0), 2)
    read = BoardRead(
        t=0.0,
        rows=(RowRead(("4",), "0", "SINNER"), RowRead(("6",), "0", "ALCARAZ")),
        serving_row=2,
    )

    assert board_emission_score(after_set_one, read) > board_emission_score(still_in_set_one, read)


def test_a_seed_number_beside_the_name_is_not_a_completed_set() -> None:
    from cv.pipeline.score_grammar import BoardRead, RowRead, board_emission_score

    first_set = ScoreState((), (0, 1), (1, 1), 1)
    read = BoardRead(
        t=0.0,
        rows=(RowRead(("1", "0"), "15", "SINNER"), RowRead(("2", "1"), "15", "ALCARAZ")),
    )

    assert board_emission_score(first_set, read) > 0.0


def test_transposed_rows_are_put_back_by_their_names() -> None:
    from cv.pipeline.score_grammar import BoardRead, RowRead, orient_rows

    names = {1: "SINNER", 2: "ALCARAZ"}
    swapped = BoardRead(
        t=0.0,
        rows=(RowRead(("6",), "40", "ALCARAZ"), RowRead(("4",), "15", "SINNER")),
        serving_row=1,
    )

    fixed = orient_rows(swapped, names)

    assert fixed.rows[0].name == "SINNER"
    assert fixed.rows[0].points == "15"
    assert fixed.serving_row == 2


def test_board_decode_recovers_a_set_boundary_the_slotted_read_loses() -> None:
    from cv.pipeline.score_grammar import (
        board_observations,
        board_row_names,
        decode_board,
        expand_board_reads,
    )

    def board(top: tuple[str, ...], bottom: tuple[str, ...], p1: str, p2: str) -> dict:
        return {
            "present": True,
            "rows": [
                {"name": "SINNER", "values": list(top), "points": p1},
                {"name": "ALCARAZ", "values": list(bottom), "points": p2},
            ],
            "serving_row": 1,
        }

    records = []
    moment = 0
    for points in ("0", "15", "30", "40"):
        records.append({"i": moment, "raw": board(("5",), ("5",), points, "0")})
        moment += 30
    # the set ends 5-7 and the board drops its 0-0 games column for a while
    for points in ("0", "15"):
        records.append({"i": moment, "raw": board(("5",), ("7",), points, "0")})
        moment += 30
    for points in ("30", "40"):
        records.append({"i": moment, "raw": board(("5", "0"), ("7", "0"), points, "0")})
        moment += 30

    reads = expand_board_reads(board_observations(records))
    path = decode_board(reads, names=board_row_names(reads))
    last = [state for state in path if state is not None][-1]

    assert last.completed == ((5, 7),)


def test_confidence_falls_where_the_reads_do_not_support_the_state() -> None:
    from cv.pipeline.score_grammar import (
        board_observations,
        decode_board_with_confidence,
        expand_board_reads,
    )

    def board(p1: str) -> dict:
        return {
            "present": True,
            "rows": [
                {"name": "SINNER", "values": ["0"], "points": p1},
                {"name": "ALCARAZ", "values": ["0"], "points": "0"},
            ],
            "serving_row": 1,
        }

    records = [{"i": 0, "raw": board("0")}, {"i": 40, "raw": board("40")}]
    reads = expand_board_reads(board_observations(records), hold_seconds=5.0)
    result = decode_board_with_confidence(reads, ensemble=4, dropout=0.5, seed=1)

    assert len(result.confidence) == len(reads)
    assert result.stale_seconds[0] == 0.0
    assert max(result.stale_seconds) > 5.0


def _gap_reads() -> list:
    """Four crops showing 0-0, then a blind stretch, then a crop showing 15-0."""
    from cv.pipeline.score_grammar import board_observations, expand_board_reads

    def board(points: str | None) -> dict:
        if points is None:
            return {"present": False, "rows": [], "serving_row": None}
        return {
            "present": True,
            "rows": [
                {"name": "SINNER", "values": ["0"], "points": points},
                {"name": "ALCARAZ", "values": ["0"], "points": "0"},
            ],
            "serving_row": 1,
        }

    records = [{"i": second, "raw": board("0")} for second in (0, 10, 20, 30)]
    records += [{"i": second, "raw": board(None)} for second in (40, 60, 80)]
    records += [{"i": second, "raw": board("15")} for second in (100, 110)]
    return expand_board_reads(board_observations(records), hold_seconds=10.0)


def test_a_change_in_a_blind_stretch_is_placed_where_the_evidence_starts() -> None:
    from cv.pipeline.score_grammar import (
        advance_transitions_into_gaps,
        board_row_names,
        decode_board,
    )

    reads = _gap_reads()
    path = decode_board(reads, names=board_row_names(reads))
    raw_change = next(step for step in range(1, len(path)) if path[step] != path[step - 1])
    moved = advance_transitions_into_gaps(reads, path)
    new_change = next(step for step in range(1, len(moved)) if moved[step] != moved[step - 1])

    assert reads[raw_change].t >= 100.0
    assert 40.0 <= reads[new_change].t <= 60.0


def test_the_pull_back_can_be_capped() -> None:
    from cv.pipeline.score_grammar import (
        advance_transitions_into_gaps,
        board_row_names,
        decode_board,
    )

    reads = _gap_reads()
    path = decode_board(reads, names=board_row_names(reads))
    moved = advance_transitions_into_gaps(reads, path, max_pull_back_seconds=5.0)
    new_change = next(step for step in range(1, len(moved)) if moved[step] != moved[step - 1])

    assert reads[new_change].t >= 95.0


def test_the_match_ending_stops_the_decoder_inventing_another_set() -> None:
    from cv.pipeline.score_grammar import MatchRules, state_from_read

    read = {"g1": "0", "g2": "1", "p1": "0", "p2": "0", "sets1": "6 6", "sets2": "4 3"}
    assert state_from_read(read, 1, MatchRules(best_of=3)) is None
    assert state_from_read(read, 1, MatchRules(best_of=5)) is not None


@pytest.mark.parametrize(
    "kwargs", [{"best_of": 2}, {"tiebreak_target": 0}, {"deciding_tiebreak_target": 12}]
)
def test_unsupported_match_format_is_rejected(kwargs):
    from cv.pipeline.score_grammar import MatchRules

    with pytest.raises(ValueError):
        MatchRules(**kwargs)


def test_slotted_cli_never_opens_ambient_raw_reads_and_records_format(tmp_path, monkeypatch):
    from cv.pipeline import score_grammar

    source = tmp_path / "score_runs.csv"
    source.write_text("t_start,t_end,g1,g2,p1,p2,sets1,sets2\n0,1,0,0,0,0,,\n")
    (tmp_path / "vlm_reads.jsonl").write_text("INVALID JSON MUST NOT BE OPENED\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_grammar",
            "--score-runs",
            str(source),
            "--out",
            str(tmp_path / "result"),
            "--best-of",
            "3",
            "--deciding-tiebreak-target",
            "7",
        ],
    )
    assert score_grammar.main() == 0
    report = json.loads((tmp_path / "result" / "score_state_v1.json").read_text())
    assert report["source"] == str(source)
    assert report["match_rules"] == {
        "best_of": 3,
        "tiebreak_target": 7,
        "deciding_tiebreak_target": 7,
    }


def test_raw_cli_requires_unambiguous_explicit_input(tmp_path, monkeypatch):
    from cv.pipeline import score_grammar

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_grammar",
            "--score-runs",
            "unused.csv",
            "--vlm-reads",
            "unused.jsonl",
            "--out",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        score_grammar.main()
