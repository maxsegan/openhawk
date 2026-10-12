from ball_track_consensus import Candidate
from ball_track_detour_repair import find_detour_replacements


def _row(frame: int, x: float, y: float) -> dict:
    return {"frame": f"f_{frame:04d}.jpg", "x": x, "y": y}


def test_replaces_short_false_lock_with_supported_bridge() -> None:
    base = {
        1: _row(1, 0, 0),
        4: _row(4, 100, 100),
        5: _row(5, 105, 100),
        6: _row(6, 110, 100),
        9: _row(9, 40, 0),
    }
    candidates = {
        frame: [
            Candidate(
                x=frame * 5.0,
                y=0.0,
                score=0.2,
                rank=2,
                sources=("wasb", "tracknetv2"),
            )
        ]
        for frame in (4, 5, 6)
    }

    replacements = find_detour_replacements(base, candidates, fps=25.0)

    assert set(replacements) == {4, 5, 6}
    assert replacements[5].x == 25.0


def test_preserves_short_run_without_complete_alternate_support() -> None:
    base = {
        1: _row(1, 0, 0),
        4: _row(4, 100, 100),
        5: _row(5, 105, 100),
        6: _row(6, 110, 100),
        9: _row(9, 40, 0),
    }
    candidates = {
        4: [Candidate(20, 0, 0.2, 2, ("wasb", "tracknetv2"))],
        6: [Candidate(30, 0, 0.2, 2, ("wasb", "tracknetv2"))],
    }

    assert find_detour_replacements(base, candidates, fps=25.0) == {}
