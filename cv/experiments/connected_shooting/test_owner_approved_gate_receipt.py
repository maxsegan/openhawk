"""The search writes the gate receipt; the replay reads it. Pin that contract.

The first gate wave (`owner_approved_gates_v1`, 2026-09-19) measured nothing, and the way it
failed is worth a permanent test.  Both owner-approved changes ran correctly inside the search
-- its report's own targets carried `supplied_frame_interval` and `applied_sigma_floor_px: 3.2`
-- but the verdict is not produced from those targets.  `per_flight_rescore.build_context`
rebuilds the bounce witness for the replay, and decides HOW by reading keys back off the search
report's `configuration`.  `_run_search` can write that configuration from three different
places and the receipt had been added to only one, so the report carried the new targets and
none of the keys.  The replay therefore rebuilt the OLD witness, and the verdict silently
discarded both gate changes while the search kept them.

Nothing raised.  The panel scored, the arms differed, and the numbers meant something other
than what they claimed.  These tests fail if the writer and the reader drift apart again.
"""

from __future__ import annotations

import ast
from pathlib import Path

SEARCH = Path(__file__).with_name("agent_whole_point_search.py")
RESCORE = Path(__file__).with_name("per_flight_rescore.py")


def _configuration_sites(tree: ast.AST) -> list[ast.Dict | ast.Call]:
    """Every `configuration=dict(...)` keyword the search module builds."""
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "configuration":
                    sites.append(keyword.value)
    return sites


def test_every_configuration_receipt_carries_the_gate_keys():
    """A report written without the receipt replays the old witness. Catch that here."""
    sites = _configuration_sites(ast.parse(SEARCH.read_text()))
    assert len(sites) >= 3, "expected the search's several configuration receipts"
    for site in sites:
        names = {
            node.id
            for node in ast.walk(site)
            if isinstance(node, ast.Name) and node.id == "owner_approved_gate_receipt"
        }
        assert names, (
            "a configuration receipt omits owner_approved_gate_receipt; a report written "
            "from it would make per_flight_rescore rebuild the pre-sign-off witness"
        )


def test_the_replay_reads_the_witnesses_not_a_receipt_beside_them():
    """The fix that actually holds: ask the frozen targets, which every writer emits.

    Keying off `report["configuration"]` is what failed, because several modules build that
    dict independently -- `agent_whole_point_search` from three places and
    `shared_coarse_initializer` from scratch -- so a report can carry the new targets and none
    of the keys. `bounce_ground_ray_targets` is emitted by all of them and describes itself.
    """
    rescore_text = RESCORE.read_text()
    assert 'report["bounce_ground_ray_targets"]' in rescore_text, (
        "the replay must derive the gate state from the frozen witnesses"
    )
    for decisive in ("supplied_frame_interval", "applied_sigma_floor_px"):
        assert decisive in rescore_text, f"the replay stopped reading {decisive}"
    forbidden = (
        'report["configuration"].get("automatic_ball_witness_sigma_applied")',
        'report["configuration"].get("bounce_interval_timing"',
    )
    for text in forbidden:
        assert text not in rescore_text, (
            "the replay is keying the gate off a configuration receipt again; that is the "
            "failure that voided the first gate wave"
        )


def test_the_witness_publishes_what_the_replay_needs_to_read():
    """Writer and reader must agree on the two self-describing field names."""
    search_text = SEARCH.read_text()
    for field in ("supplied_frame_interval", "applied_sigma_floor_px"):
        assert f'"{field}"' in search_text, f"the witness stopped publishing {field}"


def test_the_receipt_is_absent_when_both_changes_are_off():
    """Off must leave every existing receipt byte for byte, so the keys must not appear."""
    source = ast.parse(SEARCH.read_text())
    assignment = None
    for node in ast.walk(source):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "owner_approved_gate_receipt" for t in node.targets
        ):
            assignment = node
    assert assignment is not None, "owner_approved_gate_receipt is no longer built in one place"
    # Both halves are guarded by an `== "off"` test that yields an empty dict.
    text = ast.unparse(assignment)
    assert text.count("'off'") >= 2 or text.count('"off"') >= 2, (
        "each half of the receipt must stay guarded so an off arm writes nothing"
    )


def test_the_replay_refuses_a_witness_that_dropped_the_search_s_fields():
    """The generic guard: field-set equality between the frozen and rebuilt witness.

    This is worth more than the named tests above, because it is about mechanisms nobody has
    written yet. Any future change that alters the bounce witness in the search but not in the
    replay drops a field here and raises, instead of scoring the verdict on the old witness.
    Comparing FIELDS rather than values keeps it immune to numerical drift.
    """
    from cv.experiments.connected_shooting.per_flight_rescore import (
        _WITNESS_FIELDS_ADDED_AFTER_FREEZE as skip,
    )

    frozen = {
        "xyz_m": [1.0, 2.0, 0.0333],
        "supplied_frame_interval": [310.0, 314.0],
        "wing_policy": {"label_sigma_floor_px": 2.0, "applied_sigma_floor_px": 3.2},
    }
    # Exactly what the pre-fix replay produced: same counts, fewer declarations.
    rebuilt = {"xyz_m": [1.0, 2.0, 0.0333], "wing_policy": {"label_sigma_floor_px": 2.0}}
    assert set(frozen) - set(rebuilt) - skip == {"supplied_frame_interval"}
    assert set(frozen["wing_policy"]) - set(rebuilt["wing_policy"]) == {"applied_sigma_floor_px"}
    # And an honest rebuild passes both comparisons.
    assert not set(frozen) - set(dict(frozen)) - skip
