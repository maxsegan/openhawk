from cv.experiments.connected_shooting import ending_witnesses
from cv.pipeline import point_context


def _context(*, winner="server", ending=None, serve=None):
    winner_likelihoods = {"server": 0.9, "receiver": 0.025, "unknown": 0.075}
    if winner == "receiver":
        winner_likelihoods = {"server": 0.025, "receiver": 0.9, "unknown": 0.075}
    if winner == "unknown":
        winner_likelihoods = {"server": 0.0, "receiver": 0.0, "unknown": 1.0}
    return {
        "schema": point_context.SCHEMA,
        "automatic": True,
        "match_id": "match",
        "point_id": "pt0001",
        "serve_ordinal": serve or {"first": 0.25, "second": 0.25, "fault": 0.25, "let": 0.25},
        "ending_kind": ending
        or {"out": 0.25, "second_bounce": 0.25, "net": 0.25, "fov_exit": 0.25},
        "winner": winner,
        "winner_likelihoods": winner_likelihoods,
        "evidence_receipts": {
            "scoreboard": {},
            "segment_adjacency": {},
            "audio_call": {},
            "shipped_ending": {},
        },
        "abstentions": ["audio_call"],
        "provenance": {"human_derived_inputs": []},
    }


def _events(count):
    return [{"event_type": "contact", "frame": 10.0 + index * 20} for index in range(count)]


def test_score_winner_favors_in_unreturned_when_last_hitter_won():
    witness = ending_witnesses.build(_context(winner="server"), _events(3))
    assert witness["last_hitter_role"] == "server"
    assert witness["last_ball_in_likelihoods"]["in"] > 0.8
    assert (
        witness["ending_likelihoods"]["second_bounce"]
        > witness["ending_likelihoods"]["first_bounce_out"]
    )


def test_score_winner_favors_out_or_net_when_last_hitter_lost():
    witness = ending_witnesses.build(_context(winner="receiver"), _events(3))
    assert witness["last_ball_in_likelihoods"]["out_or_net"] > 0.8
    assert (
        witness["ending_likelihoods"]["first_bounce_out"]
        > witness["ending_likelihoods"]["second_bounce"]
    )


def test_same_side_restart_softly_favors_terminal_serve_error():
    context = _context(
        winner="unknown",
        serve={"first": 0.05, "second": 0.05, "fault": 0.72, "let": 0.18},
    )
    witness = ending_witnesses.build(context, _events(1))
    assert witness["same_side_serve_restart_used"] is True
    assert (
        witness["ending_likelihoods"]["first_bounce_out"]
        > witness["ending_likelihoods"]["second_bounce"]
    )


def test_invalid_or_human_context_abstains_and_never_gates():
    context = _context()
    context["evidence_receipts"]["scoreboard"]["human_derived"] = True
    receipt = ending_witnesses.score(context, _events(1), "terminal_bounce")
    assert receipt["penalty"] == 0.0
    assert receipt["hard_gate"] is False
    assert receipt["witness"]["available"] is False


def test_penalty_is_bounded_and_fit_score_can_override_context():
    context = _context(ending={"out": 0.97, "second_bounce": 0.01, "net": 0.01, "fov_exit": 0.01})
    receipt = ending_witnesses.score(context, _events(3), "second_bounce")
    assert 0.0 < receipt["penalty"] <= ending_witnesses.MAXIMUM_LOG_PENALTY
