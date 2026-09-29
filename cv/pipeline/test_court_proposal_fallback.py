"""The wider proposal beam changes candidate availability, never admission rules."""

import numpy as np
import pytest

from cv.pipeline import court_topology as ct


def proposal(score, identity=1):
    return ct.CourtTopologyHypothesis(np.eye(3) * identity, score)


def test_original_mask_ranking_returns_without_expanding(monkeypatch):
    calls = []
    original = [[], [proposal(0.86, 2)], [proposal(0.91, 3)]]

    def candidates(_image, **kwargs):
        assert "proposal_pool" not in kwargs
        calls.append(kwargs)
        return original[len(calls) - 1]

    monkeypatch.setattr(ct, "court_topology_hypotheses", candidates)
    monkeypatch.setattr(ct, "court_surface_consistency", lambda *_: 0.95)
    result = ct.solve_court_h_topology(np.zeros((32, 64, 3), np.uint8))
    assert len(calls) == 3
    assert result.source == "relaxed_chroma_balanced"
    np.testing.assert_array_equal(result.homography, np.eye(3) * 3)


def test_explicit_wider_pool_preserves_acceptance_witnesses(monkeypatch):
    pools = []

    def candidates(_image, **kwargs):
        pool = kwargs.get("proposal_pool", 48)
        pools.append(pool)
        if pool == 48:
            return [proposal(0.79)]
        return [proposal(0.89)]

    monkeypatch.setattr(ct, "court_topology_hypotheses", candidates)
    monkeypatch.setattr(ct, "court_surface_consistency", lambda *_: 0.95)
    result = ct.solve_court_h_topology(np.zeros((32, 64, 3), np.uint8), proposal_pool=384)
    assert pools == [384]
    assert result.source == "standard"
    assert result.topology_score == 0.89


@pytest.mark.parametrize(
    "score,surface,player",
    [(0.79, 0.99, 0.99), (0.95, 0.49, 0.99), (0.89, 0.81, 0.99), (0.95, 0.99, 0.59)],
)
def test_wider_beam_cannot_bypass_topology_surface_or_player_witness(
    monkeypatch, score, surface, player
):
    pools = []

    def candidates(_image, **kwargs):
        pools.append(kwargs.get("proposal_pool", 48))
        return [] if pools[-1] == 48 else [proposal(score)]

    monkeypatch.setattr(ct, "court_topology_hypotheses", candidates)
    monkeypatch.setattr(ct, "court_surface_consistency", lambda *_: surface)
    monkeypatch.setattr(ct, "player_foot_geometry_score", lambda *_: player)
    with pytest.raises(ValueError, match="no court topology"):
        ct.solve_court_h_topology(
            np.zeros((32, 64, 3), np.uint8), player_feet=np.ones((2, 2)), proposal_pool=384
        )
    assert pools == [384, 384, 384]


def test_expansion_never_preempts_later_original_anchor(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from cv.pipeline import court_topology_runner as runner

    calls = []
    winner = SimpleNamespace(rank=1.0)

    def solve(index, path, *, proposal_pool=48):
        calls.append((index, proposal_pool))
        assert proposal_pool == 48
        if len(calls) == 1:
            return {"frame": path.name, "error": ct.NO_TOPOLOGY_SOLUTION}
        return winner

    monkeypatch.setattr(runner, "_solve_anchor_frame", solve)
    actual, attempts = runner.select_anchor([Path(f"f_{i:04d}.jpg") for i in range(20)])
    assert actual is winner and len(calls) == 2 and len(attempts) == 1


def test_all_original_anchor_rejections_precede_expansion(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from cv.pipeline import court_topology_runner as runner

    frames = [Path(f"f_{i:04d}.jpg") for i in range(20)]
    original_indices = list(
        dict.fromkeys(
            round(19 * q) for q in (*runner.ANCHOR_QUANTILES, *runner.FALLBACK_ANCHOR_QUANTILES)
        )
    )
    calls = []
    winner = SimpleNamespace(rank=1.0)

    def solve(index, path, *, proposal_pool=48):
        calls.append((index, proposal_pool))
        if proposal_pool == 48:
            return {"frame": path.name, "error": ct.NO_TOPOLOGY_SOLUTION}
        return winner

    monkeypatch.setattr(runner, "_solve_anchor_frame", solve)
    actual, attempts = runner.select_anchor(frames)
    assert actual is winner
    assert calls == [(i, 48) for i in original_indices] + [(original_indices[0], 384)]
    assert len(attempts) == len(original_indices)
    assert all(set(row) == {"frame", "error"} for row in attempts)


def test_unreadable_anchor_does_not_trigger_expensive_beam_retry(monkeypatch):
    from pathlib import Path
    from cv.pipeline import court_topology_runner as runner

    calls = []

    def solve(index, path, *, proposal_pool=48, surface_witness_policy=None):
        calls.append(proposal_pool)
        return {"frame": path.name, "error": "unreadable_frame"}

    monkeypatch.setattr(runner, "_solve_anchor_frame", solve)
    actual, _ = runner.select_anchor([Path(f"f_{i:04d}.jpg") for i in range(20)])
    # The expensive 384-proposal beam is still never reached on unreadable frames.  The
    # illumination fallback does add one further pass at the ordinary pool, which is the
    # cost of recovering a sun-split court; it must not escalate the beam.
    assert actual is None and all(pool == 48 for pool in calls)
    # An unreadable frame is not a rejected court, so the illumination pass must not fire.
    assert len(calls) == 10
