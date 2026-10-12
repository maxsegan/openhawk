"""Consistent legal-family reporting for opened connected-search attempts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def family(
    candidates: list[dict],
    *,
    arm: str,
    eligible: Callable[[dict], bool],
    rank_score: Callable[[dict], float],
    required_checks: tuple[str, ...] = (
        "connected_input_physics",
        "serve_depth_plausible",
        "serve_height_plausible",
    ),
) -> dict:
    survivors = [row for row in candidates if eligible(row)]
    survivors.sort(key=lambda row: (rank_score(row), row["depth_hypothesis_m"]))
    depths = sorted(float(row["depth_hypothesis_m"]) for row in survivors)
    selected = survivors[0] if survivors else None
    legal = [
        bool(
            row["evidence"].get("survived")
            and all(row["evidence"].get("checks", {}).get(name, False) for name in required_checks)
        )
        for row in survivors
    ]
    whole = bool(survivors and all(legal))
    return {
        "schema": "connected_legal_depth_family_v1",
        "arm": arm,
        "count": len(survivors),
        "depth_y_range_m": None if not depths else [min(depths), max(depths)],
        "width_m": None if not depths else max(depths) - min(depths),
        "midpoint_m": None if not depths else float((min(depths) + max(depths)) / 2),
        "member_depths_m": depths,
        "selected_depth_y_m": None if selected is None else float(selected["depth_hypothesis_m"]),
        "selected_contact_xyz_m": None
        if selected is None
        else selected["evidence"]["contact_xyz_m"][0]
        if np.ndim(selected["evidence"]["contact_xyz_m"]) == 2
        else selected["evidence"]["contact_xyz_m"],
        "whole_family_continuous_and_legal": whole,
        "reconstructed_on_family_basis": whole,
        "acceptance_rule": (
            "count reconstructed when every surviving member passes the arm's explicit required "
            "checks; family width is retained as uncertainty"
        ),
        "required_checks": list(required_checks),
    }
