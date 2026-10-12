import numpy as np
import pytest

from cv.pipeline import event_grammar_decoder as decoder


def probabilities(rows):
    """Build a (frames, 4) probability table from ``{frame: {type: p}}`` rows."""

    table = np.zeros((len(rows), 4), dtype=np.float32)
    for index, row in enumerate(rows):
        for name, value in row.items():
            table[index, decoder.CLASSES.index(name)] = value
        table[index, 0] = max(1.0 - table[index, 1:].sum(), 1e-6)
    return table


def test_side_of_reads_the_net_line():
    assert decoder.side_of(0.9) == 1
    assert decoder.side_of(0.1) == -1
    assert decoder.side_of(float("nan")) == 0
    assert decoder.side_of(None) == 0


def test_build_nodes_keeps_only_local_maxima_above_the_floor():
    table = probabilities(
        [{"contact": 0.02}, {"contact": 0.60}, {"contact": 0.30}, {"contact": 0.001}]
    )
    frames = np.asarray([10, 11, 12, 13])
    court = np.full(4, np.nan, dtype=np.float32)
    nodes = decoder.build_nodes(table, frames, court)
    assert [node.frame for node in nodes] == [11]
    assert nodes[0].event_type == "contact"
    assert nodes[0].gain > 0


def test_tentative_node_switch_relaxes_local_maximum_only_when_requested():
    table = probabilities([{"contact": 0.8}, {"contact": 0.6}])
    frames = np.asarray([10, 11])
    court = np.full(2, np.nan, dtype=np.float32)

    firm = decoder.build_nodes(table, frames, court)
    tentative = decoder.build_nodes(
        table,
        frames,
        court,
        tentative_probability_floor=0.05,
        relax_local_maximum=True,
    )

    assert [node.frame for node in firm] == [10]
    assert [node.frame for node in tentative] == [10, 11]


def test_build_nodes_gain_is_the_log_odds_against_none():
    table = probabilities([{"bounce": 0.5}])
    nodes = decoder.build_nodes(table, np.asarray([4]), np.asarray([np.nan]))
    assert nodes[0].gain == pytest.approx(np.log(0.5) - np.log(0.5), abs=1e-6)


def test_transition_score_forbids_two_bounces_in_a_row():
    left = decoder.Node(0, 10, "bounce", 1.0, 0)
    right = decoder.Node(1, 20, "bounce", 1.0, 0)
    assert decoder.transition_score(left, right, 0) <= decoder.NEGATIVE_INFINITY / 2


def test_transition_score_enforces_the_minimum_flight_duration():
    left = decoder.Node(0, 10, "contact", 1.0, 0)
    close = decoder.Node(1, 11, "bounce", 1.0, 0)
    far = decoder.Node(2, 20, "bounce", 1.0, 0)
    assert decoder.transition_score(left, close, 0) <= decoder.NEGATIVE_INFINITY / 2
    assert decoder.transition_score(left, far, 0) == pytest.approx(0.0)


def test_transition_score_penalises_a_contact_that_repeats_a_side():
    left = decoder.Node(0, 10, "bounce", 1.0, 0)
    same = decoder.Node(1, 20, "contact", 1.0, 1)
    other = decoder.Node(2, 20, "contact", 1.0, -1)
    assert decoder.transition_score(left, same, 1) < decoder.transition_score(left, other, 1)
    assert decoder.transition_score(left, same, 0) == decoder.transition_score(left, other, 0)


def test_viterbi_on_an_empty_lattice():
    path, marginals, score = decoder.viterbi([])
    assert path == [] and score == 0.0 and marginals.size == 0


def test_viterbi_recovers_a_contact_bounce_contact_rally():
    rows = []
    frames = []
    for frame in range(0, 40):
        row = {}
        if frame == 5:
            row["contact"] = 0.9
        if frame == 15:
            row["bounce"] = 0.9
        if frame == 25:
            row["contact"] = 0.9
        rows.append(row)
        frames.append(frame)
    table = probabilities(rows)
    court = np.where(np.arange(40) < 20, 0.2, 0.8).astype(np.float32)
    events = decoder.decode_clip(table, np.asarray(frames), court, np.arange(40), "clip")
    assert [(event.frame, event.event_type) for event in events] == [
        (5, "contact"),
        (15, "bounce"),
        (25, "contact"),
    ]
    assert all(event.marginal > 0.5 for event in events)


def test_viterbi_drops_the_second_of_two_bounces():
    rows = [{} for _ in range(40)]
    rows[5] = {"contact": 0.9}
    rows[12] = {"bounce": 0.9}
    rows[18] = {"bounce": 0.8}
    rows[30] = {"contact": 0.9}
    table = probabilities(rows)
    court = np.full(40, np.nan, dtype=np.float32)
    events = decoder.decode_clip(table, np.arange(40), court, np.arange(40), "clip")
    bounces = [event for event in events if event.event_type == "bounce"]
    assert len(bounces) == 1
    assert bounces[0].frame == 12


@pytest.mark.parametrize("exact_side", [False, True])
@pytest.mark.parametrize("mode", decoder.EMISSION_MODES)
def test_opt_in_terminal_state_recovers_two_bounces_without_duplicate_observations(
    exact_side, mode
):
    table = probabilities([{"contact": 0.99}, {"bounce": 0.99}, {"bounce": 0.99}])
    config = decoder.GrammarConfig(terminal_second_bounce=True, exact_side_state=exact_side)
    events = decoder.decode_clip(
        table,
        np.array([10, 30, 50]),
        np.full(3, np.nan),
        np.arange(3),
        "clip",
        config=config,
        emission_mode=mode,
    )
    assert [(event.frame, event.event_type) for event in events] == [
        (10, "contact"),
        (30, "bounce"),
        (50, "bounce"),
    ]
    assert events[-1].terminal_kind == "second_bounce"
    assert events[-1].preceding_bounce_frame == 30
    assert 0.0 < events[-1].terminal_marginal <= events[-1].marginal <= 1.0


def test_terminal_state_cannot_start_or_resume_play_after_a_second_bounce():
    config = decoder.GrammarConfig(terminal_second_bounce=True).resolved()
    nodes = [
        decoder.Node(0, 10, "contact", 5.0, 0),
        decoder.Node(1, 30, "bounce", 5.0, 0),
        decoder.Node(2, 50, "bounce", 5.0, 0, terminal=True),
        decoder.Node(3, 70, "contact", 10.0, 0),
    ]
    assert decoder.viterbi([nodes[2]], config)[0] == []
    assert decoder.viterbi_exact_side([nodes[2]], config)[0] == []
    assert decoder.nbest_paths([nodes[2]], config) == []
    assert decoder.viterbi_tables([nodes[2]], config)["path"] == []
    for _score, path in decoder.nbest_paths(nodes, config, count=30):
        if 2 in path:
            assert path[-1] == 2
            assert path[-2] == 1
            assert 3 not in path
    assert decoder.transition_score(nodes[0], nodes[2], 0, config) <= decoder.NEGATIVE_INFINITY / 2


def test_terminal_transition_tables_and_direct_decoder_agree():
    config = decoder.GrammarConfig(terminal_second_bounce=True).resolved()
    nodes = decoder.build_nodes(
        probabilities([{"contact": 0.99}, {"bounce": 0.99}, {"bounce": 0.99}]),
        np.array([10, 30, 50]),
        np.full(3, np.nan),
        config=config,
    )
    direct, _marginal, _score = decoder.viterbi(nodes, config)
    assert decoder.viterbi_tables(nodes, config)["path"] == direct
    assert decoder.viterbi_exact_side(nodes, config)[0] == direct
    assert decoder.nbest_paths(nodes, config)[0][1] == direct
    assert nodes[direct[-1]].terminal


def test_second_bounce_state_is_opt_in_at_the_low_level():
    assert decoder.GrammarConfig().resolved().terminal_second_bounce is False
    assert (
        decoder.GrammarConfig(terminal_second_bounce=True).as_dict()["terminal_second_bounce"]
        is True
    )


def test_terminal_second_bounce_prior_is_explicit_and_configurable():
    config = decoder.GrammarConfig(
        terminal_second_bounce=True, terminal_second_bounce_log_prior=-2.0
    ).resolved()
    previous = decoder.Node(0, 10, "bounce", 5.0, 0)
    terminal = decoder.Node(1, 30, "bounce", 5.0, 0, terminal=True)
    assert decoder.transition_score(previous, terminal, 0, config) == pytest.approx(-2.0)


def test_marginals_fall_when_two_readings_compete():
    rows = [{} for _ in range(40)]
    rows[5] = {"contact": 0.9}
    rows[20] = {"bounce": 0.45}
    rows[21] = {"net_hit": 0.44}
    table = probabilities(rows)
    court = np.full(40, np.nan, dtype=np.float32)
    nodes = decoder.build_nodes(table, np.arange(40), court)
    _path, marginals, _score = decoder.viterbi(nodes)
    contested = [
        marginals[index]
        for index, node in enumerate(nodes)
        if node.event_type in {"bounce", "net_hit"}
    ]
    assert contested
    assert max(contested) < 0.95


def test_a_confident_isolated_event_keeps_a_high_marginal():
    rows = [{} for _ in range(20)]
    rows[9] = {"contact": 0.99}
    table = probabilities(rows)
    nodes = decoder.build_nodes(table, np.arange(20), np.full(20, np.nan, dtype=np.float32))
    _path, marginals, _score = decoder.viterbi(nodes)
    assert marginals.max() > 0.95


def test_decode_splits_points():
    table = probabilities([{"contact": 0.9}, {"contact": 0.9}])
    clips = np.asarray(["a", "b"])
    events = decoder.decode(table, clips, np.asarray([4, 4]), np.asarray([np.nan, np.nan]))
    assert sorted(event.clip for event in events) == ["a", "b"]
    assert sorted(event.row for event in events) == [0, 1]


def test_decode_returns_nothing_when_no_reading_beats_none():
    table = probabilities([{"contact": 0.002}, {"bounce": 0.001}])
    events = decoder.decode(
        table, np.asarray(["a", "a"]), np.asarray([1, 9]), np.asarray([np.nan, np.nan])
    )
    assert events == []


def test_grammar_config_defaults_reproduce_the_module_constants():
    resolved = decoder.GrammarConfig().resolved()
    assert resolved.floors == decoder.TYPE_FLOORS
    assert resolved.type_log_bonus == decoder.TYPE_LOG_BONUS
    assert resolved.transition_log_prior == decoder.TRANSITION_LOG_PRIOR
    assert resolved.minimum_gap == decoder.MINIMUM_GAP
    document = decoder.GrammarConfig().as_dict()
    assert document["transition_log_prior"]["contact->bounce"] == 0.0
    assert document["minimum_gap"]["contact->contact"] == 6


def test_per_type_floor_admits_a_low_probability_node():
    table = probabilities([{"net_hit": 0.004}])
    frames = np.asarray([10])
    court = np.asarray([np.nan])
    assert decoder.build_nodes(table, frames, court) == []
    config = decoder.GrammarConfig(floors={**decoder.TYPE_FLOORS, "net_hit": 0.002})
    nodes = decoder.build_nodes(table, frames, court, config=config)
    assert [node.event_type for node in nodes] == ["net_hit"]


def test_type_log_bonus_raises_the_gain_by_exactly_its_value():
    table = probabilities([{"net_hit": 0.05}])
    plain = decoder.build_nodes(table, np.asarray([4]), np.asarray([np.nan]))
    config = decoder.GrammarConfig(type_log_bonus={**decoder.TYPE_LOG_BONUS, "net_hit": 1.5})
    boosted = decoder.build_nodes(table, np.asarray([4]), np.asarray([np.nan]), config=config)
    assert boosted[0].gain == pytest.approx(plain[0].gain + 1.5)


def test_a_boosted_net_hit_can_win_a_path_it_would_otherwise_lose():
    rows = [{} for _ in range(40)]
    rows[5] = {"contact": 0.9}
    rows[12] = {"net_hit": 0.25, "bounce": 0.22}
    rows[25] = {"contact": 0.9}
    table = probabilities(rows)
    court = np.full(40, np.nan, dtype=np.float32)
    plain = decoder.decode_clip(table, np.arange(40), court, np.arange(40), "c")
    config = decoder.GrammarConfig(
        floors={**decoder.TYPE_FLOORS, "net_hit": 0.002},
        type_log_bonus={**decoder.TYPE_LOG_BONUS, "net_hit": 2.5},
    )
    boosted = decoder.decode_clip(table, np.arange(40), court, np.arange(40), "c", config=config)
    assert "net_hit" not in {event.event_type for event in plain}
    assert "bounce" in {event.event_type for event in plain}
    assert "net_hit" in {event.event_type for event in boosted}


def test_transition_score_honours_a_relaxed_prior():
    left = decoder.Node(0, 10, "contact", 1.0, 0)
    right = decoder.Node(1, 20, "net_hit", 1.0, 0)
    assert decoder.transition_score(left, right, 0) == pytest.approx(-1.5)
    config = decoder.GrammarConfig(
        transition_log_prior={**decoder.TRANSITION_LOG_PRIOR, ("contact", "net_hit"): 0.0}
    )
    assert decoder.transition_score(left, right, 0, config) == pytest.approx(0.0)


def test_endpoint_prior_is_default_off_and_can_prefer_a_bounce_ending():
    nodes = [
        decoder.Node(0, 10, "contact", 4.0, -1),
        decoder.Node(1, 20, "bounce", 1.0, 1),
        decoder.Node(2, 30, "contact", 0.5, 1),
    ]
    plain_path, _marginals, plain_score = decoder.viterbi(nodes)
    config = decoder.GrammarConfig(end_log_prior={"contact": 0.0, "bounce": 2.0, "net_hit": 0.0})
    ending_path, _marginals, ending_score = decoder.viterbi(nodes, config)

    assert plain_path == [0, 1, 2]
    assert ending_path == [0, 1]
    assert ending_score > plain_score
    assert decoder.GrammarConfig().resolved().end_log_prior == decoder.END_LOG_PRIOR


class _Store:
    def __init__(self, frames):
        self._frames = np.asarray(frames)

    def clips(self):
        return np.asarray(["b__pt0001"] * len(self._frames))

    def frames(self):
        return self._frames

    def broadcasts(self):
        return np.asarray(["b"] * len(self._frames))


def _event(row, frame, marginal=0.9, event_type="contact"):
    return decoder.DecodedEvent("b__pt0001", frame, event_type, 0.8, marginal, row, True)


def test_terminal_state_emits_a_confirmable_second_bounce():
    frames = np.array([10, 30, 50])
    table = probabilities([{"contact": 0.99}, {"bounce": 0.99}, {"bounce": 0.99}])
    events = decoder.decode_clip(
        table,
        frames,
        np.full(3, np.nan),
        np.arange(3),
        "b__pt0001",
        config=decoder.GrammarConfig(terminal_second_bounce=True),
    )
    rows = decoder.emission_rows(
        events,
        _Store(frames),
        table,
        np.zeros(3),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        {"b": 25.0},
        0.5,
    )
    assert rows[-1]["terminal_evidence"]["status"] == "candidate"
    assert rows[-1]["terminal_evidence"]["preceding_bounce_candidate_frame"] == 30
    for row in rows:
        row["location"]["court_x_fraction"] = 0.5
        row["location"]["court_y_fraction"] = 0.5
    rows[-1]["terminal_evidence"]["preceding_bounce_in_bounds"] = True
    endings = decoder.point_end_rows(rows)
    assert endings[0]["frame"] == 50.0
    assert endings[0]["point_end"]["termination_kind"] == "second_bounce"


def test_second_bounce_support_cannot_be_promoted_by_its_terminal_suffix():
    frames = np.array([10, 30, 50])
    table = probabilities([{"contact": 0.99}, {"bounce": 0.99}, {"bounce": 0.99}])
    support = decoder.DecodedEvent(
        "b__pt0001",
        30,
        "bounce",
        0.99,
        0.99,
        1,
        True,
        terminal_support_marginal=0.4,
    )
    second = decoder.DecodedEvent(
        "b__pt0001",
        50,
        "bounce",
        0.99,
        0.99,
        2,
        True,
        terminal_kind="second_bounce",
        terminal_marginal=0.99,
        preceding_bounce_frame=30,
        preceding_bounce_row=1,
    )
    rows = decoder.emission_rows(
        [_event(0, 10), support, second],
        _Store(frames),
        table,
        np.zeros(3),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        {"b": 25.0},
        0.5,
    )
    assert rows[1]["abstain"] is True
    assert rows[1]["terminal_support"]["status"] == "withheld"
    assert rows[2]["abstain"] is False
    assert rows[2]["terminal_evidence"]["preceding_bounce_in_bounds"] is None


def test_frame_correction_moves_the_emitted_frame_and_clips_the_shift():
    store = _Store([100, 200])
    rows = decoder.emission_rows(
        [_event(0, 100), _event(1, 200)],
        store,
        np.tile(np.asarray([0.1, 0.6, 0.2, 0.1], dtype=np.float32), (2, 1)),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        {"b": 25.0},
        0.5,
        time_offset=np.asarray([2.2, -9.0], dtype=np.float32),
        correct_frames=True,
    )
    by_frame = {row["frame"]: row for row in rows}
    assert set(by_frame) == {102.0, 197.0}
    assert by_frame[102.0]["candidate_frame"] == 100.0
    assert by_frame[102.0]["location"]["frame_subpixel"] == pytest.approx(102.2)
    assert by_frame[197.0]["location"]["frame_subpixel"] == pytest.approx(
        200 - decoder.MAX_TIME_CORRECTION
    )


def test_without_frame_correction_the_candidate_frame_is_emitted():
    store = _Store([100])
    rows = decoder.emission_rows(
        [_event(0, 100)],
        store,
        np.asarray([[0.1, 0.6, 0.2, 0.1]], dtype=np.float32),
        np.asarray([0.4], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        {"b": 25.0},
        0.5,
    )
    assert rows[0]["frame"] == 100.0
    assert rows[0]["location"]["frame_subpixel"] == pytest.approx(100.4)


@pytest.mark.parametrize("fps", [25.0, 50.0, 60000 / 1001])
def test_emission_seconds_use_the_same_one_based_origin_as_audio_crops(fps):
    rows = decoder.emission_rows(
        [_event(0, 1), _event(1, 100)],
        _Store([1, 100]),
        np.tile([0.1, 0.6, 0.2, 0.1], (2, 1)),
        np.array([0.0, 0.4]),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        {"b": fps},
        0.5,
    )
    assert rows[0]["location"]["time_seconds"] == 0.0
    assert rows[1]["location"]["time_seconds"] == pytest.approx(99.4 / fps)
    assert rows[0]["location"]["time_reference"] == "clip_start"
    assert rows[0]["location"]["frame_index_origin"] == 1


@pytest.mark.parametrize("rates", [{}, {"b": 0.0}, {"b": -25.0}, {"b": float("nan")}])
def test_emission_seconds_do_not_invent_a_frame_rate(rates):
    with pytest.raises(ValueError, match="frame rate"):
        decoder.emission_rows(
            [_event(0, 1)],
            _Store([1]),
            np.array([[0.1, 0.6, 0.2, 0.1]]),
            np.zeros(1),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            rates,
            0.5,
        )


def test_frame_correction_collapses_two_rows_onto_one_frame():
    store = _Store([100, 101])
    rows = decoder.emission_rows(
        [_event(0, 100, marginal=0.4), _event(1, 101, marginal=0.95)],
        store,
        np.tile(np.asarray([0.1, 0.6, 0.2, 0.1], dtype=np.float32), (2, 1)),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        {"b": 25.0},
        0.5,
        time_offset=np.asarray([1.0, 0.0], dtype=np.float32),
        correct_frames=True,
    )
    assert len(rows) == 1
    assert rows[0]["frame"] == 101.0
    assert rows[0]["path_marginal"] == pytest.approx(0.95)


def test_every_emission_declares_that_the_point_grammar_is_absent():
    store = _Store([100])
    rows = decoder.emission_rows(
        [_event(0, 100)],
        store,
        np.asarray([[0.1, 0.6, 0.2, 0.1]], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        {"b": 25.0},
        0.5,
    )
    block = rows[0]["point_grammar"]
    assert block["present"] is False
    assert "point_gate_verdict" in block["absent_fields"]
    assert rows[0]["location"]["court_x_m"] is None


def test_rethreshold_emissions_preserves_paths_and_regenerates_point_end():
    physical = [
        {
            "clip": "b__pt0001",
            "match_id": "b",
            "event_type": "contact",
            "frame": 10.0,
            "confidence": 0.95,
            "probability": 0.8,
            "path_marginal": 0.95,
            "abstain": True,
            "decision_threshold": 0.99,
            "location": {"image_x": 1.0, "image_y": 2.0},
        },
        {
            "clip": "b__pt0001",
            "match_id": "b",
            "event_type": "bounce",
            "frame": 20.0,
            "confidence": 0.85,
            "probability": 0.7,
            "path_marginal": 0.85,
            "abstain": True,
            "decision_threshold": 0.99,
            "location": {"image_x": 3.0, "image_y": 4.0},
        },
    ]
    stale_end = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "point_end",
        "frame": 5.0,
        "abstain": False,
    }

    rows = decoder.rethreshold_emission_rows([*physical, stale_end], 0.9)

    assert [row["event_type"] for row in rows] == ["contact", "bounce"]
    assert rows[0]["abstain"] is False
    assert rows[0]["decision_threshold"] == 0.9
    assert rows[-1]["abstain"] is True
    assert not decoder.point_end_rows(rows)


def test_point_end_requires_an_in_court_first_bounce_before_second_bounce():
    first = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "bounce",
        "frame": 30.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {"court_x_fraction": 0.5, "court_y_fraction": 0.5},
    }
    row = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "bounce",
        "frame": 50.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {"court_x_fraction": 0.5, "court_y_fraction": 0.5},
        "terminal_evidence": {
            "termination_kind": "second_bounce",
            "source": "test_witness",
            "preceding_bounce_candidate_frame": 30.0,
        },
    }
    ends = decoder.point_end_rows([first, row])
    assert ends[0]["frame"] == 50.0
    assert ends[0]["point_end"]["termination_kind"] == "second_bounce"


def test_point_end_keeps_a_withheld_second_bounce_support_as_geometry_evidence():
    row = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "bounce",
        "frame": 50.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {"court_x_fraction": 0.5, "court_y_fraction": 0.5},
        "terminal_evidence": {
            "termination_kind": "second_bounce",
            "source": "test_witness",
            "preceding_bounce_candidate_frame": 30.0,
            "preceding_bounce_in_bounds": True,
        },
    }
    end = decoder.point_end_rows([row])[0]
    assert end["point_end"]["termination_kind"] == "second_bounce"
    assert end["point_end"]["evidence"]["first_bounce_emitted"] is False


def test_point_end_uses_an_out_first_bounce_without_a_terminal_state():
    contact = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "contact",
        "frame": 20.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {},
    }
    row = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "bounce",
        "frame": 50.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {"court_x_fraction": 1.2, "court_y_fraction": 0.5},
    }
    ends = decoder.point_end_rows([contact, row])
    assert ends[0]["frame"] == 50.0
    assert ends[0]["point_end"]["termination_kind"] == "first_bounce_out"
    for change in (
        {"abstain": True},
        {"on_best_path": False},
        {"gate_held": True},
        {"event_type": "contact"},
    ):
        assert decoder.point_end_rows([contact, {**row, **change}]) == []
    later = {**row, "frame": 60.0, "event_type": "contact", "terminal_evidence": None}
    assert decoder.point_end_rows([contact, row, later]) == []


def test_point_end_can_use_a_conservative_track_exit():
    row = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "contact",
        "frame": 50.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "track_exit_after": True,
        "location": {},
    }
    end = decoder.point_end_rows([row])[0]
    assert end["point_end"]["termination_kind"] == "leaving_view"


def test_later_court_plane_exit_does_not_turn_an_in_court_bounce_out():
    contact = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "contact",
        "frame": 20.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {},
    }
    bounce = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "bounce",
        "frame": 50.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "court_exit_after": True,
        "location": {"court_x_fraction": 0.98, "court_y_fraction": 0.5},
    }
    assert decoder.point_end_rows([contact, bounce]) == []
    # A later airborne projection cannot replace a missing impact location.
    assert decoder.point_end_rows([contact, {**bounce, "location": {}}]) == []
    # An independently out bounce retains its original ending evidence.
    outside = {**bounce, "location": {"court_x_fraction": 1.2, "court_y_fraction": 0.5}}
    assert (
        decoder.point_end_rows([contact, outside])[0]["point_end"]["termination_kind"]
        == "first_bounce_out"
    )
    # The first valid bounce must not preempt the actual second-bounce ending.
    second = {
        **outside,
        "frame": 70.0,
        "terminal_evidence": {
            "termination_kind": "second_bounce",
            "preceding_bounce_candidate_frame": 50.0,
        },
    }
    end = decoder.point_end_rows([contact, bounce, second])[0]
    assert end["frame"] == 70.0
    assert end["point_end"]["termination_kind"] == "second_bounce"


def test_point_end_uses_context_softly_between_physical_endings():
    contact = {
        "clip": "b__pt0001",
        "match_id": "b",
        "event_type": "contact",
        "frame": 20.0,
        "confidence": 0.9,
        "probability": 0.9,
        "abstain": False,
        "on_best_path": True,
        "location": {},
    }
    net = {
        **contact,
        "event_type": "net_hit",
        "frame": 30.0,
        "confidence": 0.95,
    }
    first = {
        **contact,
        "event_type": "bounce",
        "frame": 40.0,
        "location": {"court_x_fraction": 0.5, "court_y_fraction": 0.5},
    }
    terminal = {
        **first,
        "frame": 50.0,
        "terminal_evidence": {
            "termination_kind": "second_bounce",
            "preceding_bounce_candidate_frame": 40.0,
        },
    }
    context = {
        "schema": "tennis_point_context_prior_v1",
        "automatic": True,
        "match_id": "b",
        "point_id": "pt0001",
        "serve_ordinal": {"first": 0.7, "second": 0.1, "fault": 0.1, "let": 0.1},
        "ending_kind": {"out": 0.02, "second_bounce": 0.03, "net": 0.94, "fov_exit": 0.01},
        "winner_likelihoods": {"server": 0.05, "receiver": 0.05, "unknown": 0.9},
        "evidence_receipts": {"audio_call": {}},
        "abstentions": [],
        "provenance": {"human_derived_inputs": []},
    }

    plain = decoder.point_end_rows([contact, net, first, terminal])[0]
    contextual = decoder.point_end_rows(
        [contact, net, first, terminal], point_contexts={"b__pt0001": context}
    )[0]

    assert plain["point_end"]["termination_kind"] == "second_bounce"
    assert contextual["point_end"]["termination_kind"] == "ground_after_net_hit"
    witness = contextual["point_end"]["evidence"]["point_context_witness"]
    assert witness["hard_gate"] is False
    assert witness["maximum_context_log_penalty"] == 1.25


def test_rethreshold_emissions_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="between zero and one"):
        decoder.rethreshold_emission_rows([], 1.1)


def test_a_per_point_net_line_resolves_per_clip():
    config = decoder.GrammarConfig(
        net_line={"m__pt0001": 1.3, "m__pt0002": 2.1},
        side_dead_band={"m__pt0001": 0.1},
        net_line_default=1.5,
    )

    assert config.for_clip("m__pt0001").net_line == pytest.approx(1.3)
    assert config.for_clip("m__pt0001").side_dead_band == pytest.approx(0.1)
    assert config.for_clip("m__pt0002").net_line == pytest.approx(2.1)
    # A clip the table does not name falls back, and never to a dead band.
    assert config.for_clip("m__pt0009").net_line == pytest.approx(1.5)
    assert config.for_clip("m__pt0009").side_dead_band == pytest.approx(0.0)
    assert config.for_clip(None).net_line == pytest.approx(1.5)


def test_a_per_point_net_line_is_summarised_in_the_manifest():
    described = decoder.GrammarConfig(
        net_line={"a": 1.0, "b": 2.0, "c": 3.0}, net_line_default=1.5
    ).as_dict()["net_line"]

    assert described == {
        "kind": "per_point",
        "points": 3,
        "min": 1.0,
        "median": 2.0,
        "max": 3.0,
    }


def test_build_nodes_reads_the_net_line_of_the_clip_it_is_given():
    table = probabilities([{"contact": 0.9}])
    frames = np.asarray([10])
    court = np.asarray([1.5], dtype=np.float32)
    config = decoder.GrammarConfig(net_line={"near__pt0001": 2.0, "far__pt0001": 1.0})

    near = decoder.build_nodes(table, frames, court, config=config, clip="near__pt0001")
    far = decoder.build_nodes(table, frames, court, config=config, clip="far__pt0001")

    assert near[0].side == -1
    assert far[0].side == 1


def test_decode_gives_each_point_its_own_net_line():
    table = probabilities([{"contact": 0.9}, {"contact": 0.9}])
    clips = np.asarray(["a__pt0001", "b__pt0001"])
    frames = np.asarray([10, 10])
    court = np.asarray([1.5, 1.5], dtype=np.float32)
    config = decoder.GrammarConfig(net_line={"a__pt0001": 2.0, "b__pt0001": 1.0})

    sides = {
        clip: decoder.build_nodes(
            table[index : index + 1],
            frames[index : index + 1],
            court[index : index + 1],
            config=config,
            clip=clip,
        )[0].side
        for index, clip in enumerate(clips.tolist())
    }
    events = decoder.decode(table, clips, frames, court, config=config)

    assert sides == {"a__pt0001": -1, "b__pt0001": 1}
    assert sorted(event.clip for event in events) == ["a__pt0001", "b__pt0001"]


def test_a_scalar_net_line_still_reproduces_the_shipped_default():
    table = probabilities([{"contact": 0.9}])
    frames = np.asarray([10])
    court = np.asarray([0.9], dtype=np.float32)

    default = decoder.build_nodes(table, frames, court)
    legacy = decoder.build_nodes(
        table, frames, court, config=decoder.GrammarConfig(net_line=decoder.LEGACY_NET_LINE)
    )

    assert default[0].side == legacy[0].side == 1
    assert decoder.GrammarConfig().resolved().net_line == decoder.NET_LINE


def _proposal(clip, frame, kinds, confidence, *, half_window=3, source="physics_departure"):
    return {
        "clip": clip,
        "frame": float(frame),
        "start_frame": float(frame - half_window),
        "end_frame": float(frame + half_window),
        "kinds": list(kinds),
        "source": source,
        "confidence": float(confidence),
    }


def test_proposal_evidence_is_bounded_and_typed():
    clips = np.asarray(["a", "a", "b"])
    frames = np.asarray([10, 20, 10])
    prior, admits = decoder.proposal_evidence(
        clips,
        frames,
        [_proposal("a", 10, ("bounce",), 0.8)],
        weight=2.0,
        cap=1.0,
    )
    assert prior.shape == (3, len(decoder.EVENT_TYPES))
    assert prior[0, decoder.EVENT_TYPES.index("bounce")] == pytest.approx(1.0)
    assert prior[0, decoder.EVENT_TYPES.index("contact")] == 0.0
    # a different clip and a frame outside the proposal's own range get nothing
    assert prior[1].tolist() == [0.0, 0.0, 0.0]
    assert prior[2].tolist() == [0.0, 0.0, 0.0]
    assert not admits.any()


def test_overlapping_proposals_take_the_maximum_not_the_sum():
    clips = np.asarray(["a"])
    frames = np.asarray([10])
    rows = [
        _proposal("a", 10, ("bounce",), 0.4),
        _proposal("a", 11, ("bounce",), 0.6),
        _proposal("a", 9, ("bounce",), 0.5, source="track_corner"),
    ]
    prior, _admits = decoder.proposal_evidence(clips, frames, rows, weight=1.0, cap=5.0)
    assert prior[0, decoder.EVENT_TYPES.index("bounce")] == pytest.approx(0.6)


def test_proposal_evidence_can_be_restricted_to_named_sources():
    prior, _admits = decoder.proposal_evidence(
        np.asarray(["a"]),
        np.asarray([10]),
        [_proposal("a", 10, ("bounce",), 1.0, source="pose_swing")],
        weight=1.0,
        cap=1.0,
        sources=("physics_departure",),
    )
    assert not prior.any()


def test_proposal_evidence_rejects_a_negative_weight_or_cap():
    with pytest.raises(ValueError):
        decoder.proposal_evidence(np.asarray(["a"]), np.asarray([1]), [], weight=-1.0)
    with pytest.raises(ValueError):
        decoder.proposal_evidence(np.asarray(["a"]), np.asarray([1]), [], cap=-1.0)


def test_the_prior_raises_the_node_gain_by_exactly_its_value():
    table = probabilities([{"bounce": 0.5}])
    frames, court = np.asarray([4]), np.asarray([np.nan])
    plain = decoder.build_nodes(table, frames, court)
    prior = np.zeros((1, len(decoder.EVENT_TYPES)))
    prior[0, decoder.EVENT_TYPES.index("bounce")] = 0.75
    raised = decoder.build_nodes(table, frames, court, evidence_log_prior=prior)
    assert raised[0].gain == pytest.approx(plain[0].gain + 0.75)


def test_the_prior_does_not_change_which_rows_become_nodes():
    table = probabilities(
        [{"contact": 0.02}, {"contact": 0.60}, {"contact": 0.30}, {"contact": 0.001}]
    )
    frames = np.asarray([10, 11, 12, 13])
    court = np.full(4, np.nan, dtype=np.float32)
    prior = np.zeros((4, len(decoder.EVENT_TYPES)))
    prior[:, decoder.EVENT_TYPES.index("contact")] = 5.0
    nodes = decoder.build_nodes(table, frames, court, evidence_log_prior=prior)
    assert [node.frame for node in nodes] == [11]


def test_admission_can_add_a_node_the_local_maximum_test_drops():
    table = probabilities(
        [{"contact": 0.02}, {"contact": 0.60}, {"contact": 0.30}, {"contact": 0.001}]
    )
    frames = np.asarray([10, 11, 12, 13])
    court = np.full(4, np.nan, dtype=np.float32)
    admits = np.zeros((4, len(decoder.EVENT_TYPES)), dtype=bool)
    admits[2, decoder.EVENT_TYPES.index("contact")] = True
    nodes = decoder.build_nodes(table, frames, court, evidence_admits=admits)
    assert [node.frame for node in nodes] == [11, 12]
    # admission never overrides the type floor
    admits[3, decoder.EVENT_TYPES.index("contact")] = True
    nodes = decoder.build_nodes(table, frames, court, evidence_admits=admits)
    assert [node.frame for node in nodes] == [11, 12]


def test_admission_is_restricted_to_unobserved_frames():
    _prior, admits = decoder.proposal_evidence(
        np.asarray(["a", "a"]),
        np.asarray([10, 11]),
        [_proposal("a", 10, ("contact",), 1.0)],
        weight=1.0,
        cap=1.0,
        track_observed=np.asarray([True, False]),
        admission_strength=0.5,
    )
    assert not admits[0].any()
    assert admits[1, decoder.EVENT_TYPES.index("contact")]


def test_a_zero_weight_prior_reproduces_the_shipped_decode():
    table = probabilities(
        [
            {"contact": 0.9},
            {"bounce": 0.1},
            {"bounce": 0.8},
            {"contact": 0.2},
            {"contact": 0.85},
        ]
    )
    frames = np.asarray([10, 14, 18, 22, 26])
    clips = np.asarray(["a"] * 5)
    court = np.asarray([0.9, 0.9, 0.2, 0.2, 0.2], dtype=np.float32)
    prior, admits = decoder.proposal_evidence(
        clips, frames, [_proposal("a", 18, ("bounce",), 1.0)], weight=0.0, cap=1.0
    )
    plain = decoder.decode(table, clips, frames, court)
    same = decoder.decode(
        table, clips, frames, court, evidence_log_prior=prior, evidence_admits=admits
    )
    assert [(row.frame, row.event_type, row.marginal) for row in plain] == [
        (row.frame, row.event_type, row.marginal) for row in same
    ]


def test_a_compatible_proposal_can_win_a_node_a_shortfall_kept_off_the_path():
    table = probabilities(
        [
            {"contact": 0.9},
            {"bounce": 0.30},
            {"contact": 0.9},
        ]
    )
    frames = np.asarray([10, 16, 22])
    clips = np.asarray(["a"] * 3)
    court = np.asarray([0.9, 0.5, 0.2], dtype=np.float32)
    grammar = decoder.GrammarConfig(transition_log_prior={**decoder.TRANSITION_LOG_PRIOR})
    plain = decoder.decode(table, clips, frames, court, config=grammar)
    prior, _admits = decoder.proposal_evidence(
        clips, frames, [_proposal("a", 16, ("bounce",), 1.0)], weight=2.0, cap=2.0
    )
    raised = decoder.decode(table, clips, frames, court, config=grammar, evidence_log_prior=prior)
    plain_bounce = next((row for row in plain if row.event_type == "bounce"), None)
    raised_bounce = next((row for row in raised if row.event_type == "bounce"), None)
    assert raised_bounce is not None
    assert plain_bounce is None or raised_bounce.marginal > plain_bounce.marginal


def test_sigma_strength_is_unsaturated_where_confidence_is_not():
    strong = {"confidence": 1.0, "evidence": {"departure_sigma": 60.0, "departure_k": 3.0}}
    weak = {"confidence": 1.0, "evidence": {"departure_sigma": 7.0, "departure_k": 3.0}}
    assert decoder.proposal_strength(strong, decoder.PROPOSAL_STRENGTH_CONFIDENCE, 30.0) == (
        decoder.proposal_strength(weak, decoder.PROPOSAL_STRENGTH_CONFIDENCE, 30.0)
    )
    assert decoder.proposal_strength(
        strong, decoder.PROPOSAL_STRENGTH_SIGMA, 30.0
    ) > decoder.proposal_strength(weak, decoder.PROPOSAL_STRENGTH_SIGMA, 30.0)
    assert 0.0 <= decoder.proposal_strength(weak, decoder.PROPOSAL_STRENGTH_SIGMA, 30.0) <= 1.0
    assert decoder.proposal_strength(strong, decoder.PROPOSAL_STRENGTH_SIGMA, 30.0) == 1.0


def test_sigma_strength_falls_back_to_confidence_without_a_sigma():
    row = {"confidence": 0.4, "evidence": {"swing_cue": "groundstroke"}}
    assert decoder.proposal_strength(row, decoder.PROPOSAL_STRENGTH_SIGMA, 30.0) == pytest.approx(
        0.4
    )


def test_sigma_strength_rejects_a_reference_below_the_arm_threshold():
    row = {"confidence": 1.0, "evidence": {"departure_sigma": 9.0, "departure_k": 3.0}}
    with pytest.raises(ValueError):
        decoder.proposal_strength(row, decoder.PROPOSAL_STRENGTH_SIGMA, 3.0)


def test_the_local_baseline_keeps_only_evidence_that_stands_out():
    clips = np.asarray(["a"] * 5)
    frames = np.asarray([10, 11, 12, 13, 14])
    rows = [_proposal("a", frame, ("bounce",), 0.5, half_window=0) for frame in (10, 11, 13, 14)]
    rows.append(_proposal("a", 12, ("bounce",), 1.0, half_window=0))
    column = decoder.EVENT_TYPES.index("bounce")
    flat, _admits = decoder.proposal_evidence(clips, frames, rows, weight=1.0, cap=1.0)
    assert flat[:, column].tolist() == pytest.approx([0.5, 0.5, 1.0, 0.5, 0.5])
    contrast, _admits = decoder.proposal_evidence(
        clips, frames, rows, weight=1.0, cap=1.0, local_baseline_frames=2
    )
    assert contrast[:, column].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.0])


def _timing_pair(rows, peaks):
    """A (pmf, grid) pair whose row ``i`` peaks on offset ``peaks[i]``."""

    from cv.pipeline import event_time_distribution as timing

    pmf = np.full((rows, timing.TIME_BINS), 0.01, dtype=np.float32)
    for row, peak in enumerate(peaks):
        pmf[row, timing.TIME_RADIUS + peak] += 1.0
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf, np.tile(timing.expected_grid(), (rows, 1))


def _emissions(**extra):
    store = _Store([100, 200])
    return decoder.emission_rows(
        [_event(0, 100), _event(1, 200)],
        store,
        np.tile(np.asarray([0.1, 0.6, 0.2, 0.1], dtype=np.float32), (2, 1)),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        {"b": 25.0},
        0.5,
        time_offset=np.asarray([2.2, -1.0], dtype=np.float32),
        correct_frames=True,
        **extra,
    )


def test_the_timing_distribution_is_additive_and_changes_nothing_else():
    from cv.pipeline import event_time_distribution as timing

    additive = {"time_distribution", "time_neighborhood"}
    without = _emissions()
    with_pmf = _emissions(time_distribution=_timing_pair(2, [2, -1]))
    assert [row.keys() - additive for row in with_pmf] == [row.keys() for row in without]
    for before, after in zip(without, with_pmf, strict=True):
        assert {k: v for k, v in after.items() if k not in additive} == before
        assert after["time_distribution"]["status"] == timing.AVAILABLE


def test_each_emission_carries_its_own_row_of_the_timing_distribution():
    rows = _emissions(time_distribution=_timing_pair(2, [2, -1]))
    by_frame = {row["frame"]: row for row in rows}
    for frame, peak in ((102.0, 2.0), (199.0, -1.0)):
        record = by_frame[frame]["time_distribution"]
        assert record["offset_frames"] == [float(v) for v in range(-8, 9)]
        peaked = record["offset_frames"][
            record["probabilities"].index(max(record["probabilities"]))
        ]
        # The grid is relative to the row's own native crop frame, not to the
        # emitted (corrected) epoch.
        assert peaked == peak
        assert by_frame[frame]["candidate_frame"] + peaked == frame


def test_a_malformed_timing_distribution_is_refused_rather_than_emitted():
    with pytest.raises(ValueError, match="normalised"):
        _emissions(
            time_distribution=(np.zeros((2, 17), dtype=np.float32), _timing_pair(2, [0, 0])[1])
        )
    with pytest.raises(ValueError, match="expected 2"):
        _emissions(time_distribution=_timing_pair(3, [0, 0, 0]))
