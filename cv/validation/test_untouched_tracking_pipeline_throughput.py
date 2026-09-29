from __future__ import annotations

import sys
from pathlib import Path

from cv.pipeline import tracking_composition


def test_tracking_subprocesses_use_modules_from_repository(monkeypatch) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs

    monkeypatch.setattr(tracking_composition.subprocess, "run", fake_run)

    tracking_composition.run(tracking_composition.module_command("ball_neural_batched"))

    assert observed["command"] == [
        sys.executable,
        "-m",
        "cv.pipeline.ball_neural_batched",
    ]
    assert observed["kwargs"]["cwd"] == tracking_composition.ROOT


def test_balanced_lanes_assigns_longest_matches_first(monkeypatch) -> None:
    weights = {"large": 100, "medium": 70, "small": 30}
    monkeypatch.setattr(
        tracking_composition,
        "frame_count",
        lambda out_dir, match: weights[match["id"]],
    )
    matches = [{"id": match_id} for match_id in ("small", "large", "medium")]

    lanes = tracking_composition.balanced_lanes(matches, Path("/unused"), [0, 1])

    assert [[match["id"] for match in lane] for lane in lanes] == [
        ["large"],
        ["medium", "small"],
    ]
