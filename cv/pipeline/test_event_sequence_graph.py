from event_sequence_graph import (
    EventNode,
    complete_bridges,
    decode_soft_path,
    sequence_features,
    transition_score,
)


def test_transition_prefers_contact_bounce_contact() -> None:
    contact = EventNode("c1", 10.0, "contact", 0.8, "near")
    bounce = EventNode("b1", 30.0, "bounce", 0.8)
    next_contact = EventNode("c2", 50.0, "contact", 0.8, "far")

    assert transition_score(contact, bounce) > transition_score(contact, next_contact)
    assert transition_score(bounce, next_contact) > 0.0


def test_sequence_features_reward_bridged_bounce() -> None:
    nodes = [
        EventNode("c1", 10.0, "contact", 0.9),
        EventNode("b1", 30.0, "bounce", 0.2),
        EventNode("c2", 50.0, "contact", 0.9),
        EventNode("b2", 90.0, "bounce", 0.2),
    ]

    features = sequence_features(nodes)

    assert features["b1"]["bridge_support"] > features["b2"]["bridge_support"]


def test_decoder_can_skip_close_competing_type() -> None:
    nodes = [
        EventNode("c1", 10.0, "contact", 0.95),
        EventNode("wrong", 10.5, "bounce", 0.8),
        EventNode("b1", 30.0, "bounce", 0.95),
        EventNode("c2", 50.0, "contact", 0.95),
    ]

    selected = decode_soft_path(nodes, event_cost=0.5)

    assert [node.node_id for node in selected] == ["c1", "b1", "c2"]


def test_bridge_completion_uses_seconds_at_native_cadence() -> None:
    anchors = [
        EventNode("b0", 25.0, "bounce", 0.95, fps=25.0),
        EventNode("c2", 75.0, "contact", 0.95, fps=25.0),
    ]
    candidates = [
        EventNode("c1", 50.0, "contact", 0.86, fps=25.0),
        EventNode("wrong", 51.0, "bounce", 0.99, fps=25.0),
    ]

    selected = complete_bridges(anchors, candidates)

    assert [node.node_id for node in selected] == ["b0", "c1", "c2"]
