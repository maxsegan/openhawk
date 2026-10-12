"""The search writes the net-cord response; the replay reads it from verdict rows.

A gate change can be live in the search and absent from the verdict. The first
owner-approved gate wave measured nothing because ``per_flight_rescore`` rebuilt
the witness from ``report['configuration']``, and several writers build that dict
independently. These tests fail if the verdict's witness rows do not carry the
admissible-set response.
"""

from __future__ import annotations

import ast
from pathlib import Path

SEARCH = Path(__file__).with_name("agent_whole_point_search.py")
RESCORE = Path(__file__).with_name("per_flight_rescore.py")
ACCEPTANCE = Path(__file__).with_name("per_flight_acceptance.py")


def _configuration_sites(tree: ast.AST) -> list[ast.Dict | ast.Call]:
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "configuration":
                    sites.append(keyword.value)
    return sites


def test_every_configuration_receipt_carries_the_net_cord_response():
    sites = _configuration_sites(ast.parse(SEARCH.read_text()))
    assert len(sites) >= 3, "expected the search's several configuration receipts"
    for site in sites:
        names = {
            node.id
            for node in ast.walk(site)
            if isinstance(node, ast.Name) and node.id == "net_cord_response_receipt"
        }
        assert names, (
            "a configuration receipt omits net_cord_response_receipt; a report written "
            "from it would make the replay rebuild the tape-clip path"
        )


def test_the_replay_reads_verdict_flight_rows_not_a_configuration_key():
    rescore_text = RESCORE.read_text()
    assert "replay_from_verdict" in rescore_text, (
        "the replay must wrap using_response from the frozen verdict rows"
    )
    assert 'report["verdict"]' in rescore_text or "replay_from_verdict" in rescore_text
    forbidden = (
        'report["configuration"].get("net_cord_response")',
        'report["configuration"].get("net_cord_response_applied")',
    )
    for text in forbidden:
        assert text not in rescore_text, (
            "the replay is keying the net-cord response off a configuration receipt; "
            "that is the failure that voided the first gate wave"
        )


def test_the_witness_is_published_on_verdict_flight_rows():
    search_text = SEARCH.read_text()
    acceptance_text = ACCEPTANCE.read_text()
    assert '"net_cord_response"' in search_text or '"net_cord_response"' in acceptance_text, (
        "the verdict stopped publishing net_cord_response on flight rows"
    )
    assert "v_out_mps" in acceptance_text or "v_out_mps" in search_text


def test_the_receipt_is_absent_when_the_key_is_off():
    source = ast.parse(SEARCH.read_text())
    assignment = None
    for node in ast.walk(source):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "net_cord_response_receipt" for t in node.targets
        ):
            assignment = node
    assert assignment is not None, "net_cord_response_receipt is no longer built in one place"
    text = ast.unparse(assignment)
    assert "receipt" in text
    from cv.pipeline import net_cord_response as cord

    assert cord.receipt(cord.TAPE_CLIP) == {}
    assert cord.receipt(cord.ADMISSIBLE_SET)[cord.FIELD] == cord.ADMISSIBLE_SET


def test_a_verdict_that_dropped_the_search_s_response_is_detected():
    from cv.experiments.connected_shooting.admissible_net_response import attach_flight_witness
    from cv.pipeline import net_cord_response as cord

    frozen = {
        "configuration": {cord.FIELD: cord.ADMISSIBLE_SET},
        "net_response": {"outgoing_velocity_mps": [0.8, 1.2, 0.2]},
        "verdict": {
            "flights": [
                {"flight_index": 0, "modeled_net_hits": [{"frame": 12.0, "x": [5.5, 11.885, 0.94]}]}
            ]
        },
    }
    attached = attach_flight_witness(frozen["verdict"]["flights"], frozen)
    rebuilt = [{"flight_index": 0, "modeled_net_hits": [{"frame": 12.0}]}]
    assert attached[0].get("net_cord_response")
    assert rebuilt[0].get("net_cord_response") is None
    missing = set(attached[0]) - set(rebuilt[0])
    assert "net_cord_response" in missing
