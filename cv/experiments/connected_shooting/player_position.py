"""Optional court root for one associated contact player.

A player state carries ``court_centre_xy_m``: the striking player's court root,
which the athlete arm turns into a *soft* reach hinge and several initializers
turn into an anchor.  The artifact it comes from can say nothing about that
player anywhere near the contact picture -- not a blank row a fallback can reach
across, but no sided row at all inside the bounded search radius.

The ordinary modes fail closed there, which is right: a root that is silently
invented, widened until meaningless, or carried from an arbitrary picture is
worse than no root.  This module adds the third honest answer.  A state may
declare its position explicitly ``absent``, in which case ``court_centre_xy_m``
is ``None`` and every root-dependent soft term abstains with a fixed-shape zero
residual, while independently established facts -- the player's side, supplied
name and stature, the ball/camera/event observations -- are retained untouched.

The declaration is deliberately narrow.  A ``None`` root without it is
malformed; a declaration alongside a purported root is malformed; a root that is
present but non-finite or the wrong shape is malformed.  Hard geometric and
observational gates never see a weaker bound because of an absence: consumers
that cannot abstain honestly must refuse, not relax.  Mirrors the existing
``athlete_priors`` unavailable-evidence contract for stature.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ABSENT = "absent"
ABSENT_REASON = (
    "no automatic player row of the required side within the bounded fallback radius; "
    "court root absent, root-dependent soft terms abstain"
)


def absent_position(reason: str = ABSENT_REASON) -> dict[str, str]:
    """The explicit per-player declaration that no court root is available."""
    return {"status": ABSENT, "reason": reason}


def absent_contact_state(
    frame: int,
    side: str,
    *,
    player_name: str | None,
    stature_m: float | None,
    image_coordinate_scale: float,
    reason: str,
    interpretation: str,
) -> dict:
    """Declare an independently associated actor without inventing a court root.

    The caller must establish the side and exhaust its bounded source lookup.
    A missing soft position does not change contact topology or hard bounds.
    """
    if side not in {"near", "far"}:
        raise ValueError("required player side must be near or far")
    return dict(
        frame=int(frame),
        side=side,
        player=player_name,
        stature_m=stature_m,
        court_centre_xy_m=None,
        player_position_evidence=absent_position(reason),
        court_position_source=None,
        court_position_sigma_m=None,
        image_association_distance_px=None,
        pixel_nearest_side=side,
        pixel_nearest_distance_px=None,
        sided_file_disagrees_with_pixel_nearest=False,
        unscaled_pixel_nearest_side=side,
        coordinate_scale_changes_pixel_nearest_side=False,
        image_coordinate_scale=image_coordinate_scale,
        pose_wrist_witness={
            "status": "abstained",
            "abstention_reason": "no_automatic_pose_row_at_this_contact_picture",
        },
        state_dof=0,
        interpretation=interpretation,
    )


def declared_absent(player: dict[str, Any]) -> bool:
    """Whether this state carries the explicit absent-position declaration."""
    return (player.get("player_position_evidence") or {}).get("status") == ABSENT


def root_available(player: dict[str, Any]) -> bool:
    """Distinguish a supplied court root from an explicitly absent position.

    A finite two-vector root is evidence.  ``None`` is accepted only alongside an
    explicit ``player_position_evidence`` absent declaration; a declaration
    alongside a root, a missing root without one, and a purported root that is
    not a finite two-vector are all malformed and refused.
    """
    root = player.get("court_centre_xy_m")
    declared = declared_absent(player)
    if root is None:
        if not declared:
            raise ValueError(
                "player court root missing without an explicit absent player-position declaration"
            )
        return False
    if declared:
        raise ValueError("a supplied court root cannot also declare the position absent")
    value = np.asarray(root, float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("finite two-component court root required")
    return True


def root_xy(player: dict[str, Any]) -> np.ndarray | None:
    """The validated court root, or ``None`` when explicitly absent."""
    return np.asarray(player["court_centre_xy_m"], float) if root_available(player) else None


def require_root(player: dict[str, Any], term: str) -> np.ndarray:
    """The validated root for a consumer that cannot abstain honestly.

    Used by hard bounds and cuts: an absent root must refuse the term outright
    rather than widen or drop a geometric admission.
    """
    root = root_xy(player)
    if root is None:
        raise ValueError(f"{term} requires a present court root; this player's position is absent")
    return root


def feet_xy(feet: dict) -> np.ndarray | None:
    """Validate the feet proxy, preserving an explicitly absent court position."""
    return root_xy(
        {
            "court_centre_xy_m": feet.get("court_xy_m"),
            "player_position_evidence": feet.get("court_position"),
        }
    )


def roots_available(players: list[dict[str, Any]]) -> np.ndarray:
    """Per-player availability mask, validating every state on the way."""
    return np.asarray([root_available(row) for row in players], bool)


def placeholder_roots(players: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Roots with a masked-away origin placeholder, plus the availability mask.

    The placeholder only ever multiplies or feeds a masked zero; it is never
    reported, exported or read back as a position.
    """
    available = roots_available(players)
    roots = np.asarray(
        [
            row["court_centre_xy_m"] if known else (0.0, 0.0)
            for row, known in zip(players, available)
        ],
        float,
    )
    if roots.shape != (len(players), 2) or not np.isfinite(roots).all():
        raise ValueError("finite court roots required for every present player position")
    return roots, available


def state_receipt(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit soft-evidence record of which contact positions were absent."""
    absent = [index for index, known in enumerate(roots_available(players)) if not known]
    return {
        "schema": "connected_player_position_v1",
        "status": "complete" if not absent else "partial",
        "absent_contact_indices": absent,
        "absent_declarations": [players[index]["player_position_evidence"] for index in absent],
        "abstained_terms_fixed_zero": True,
        "hard_gate": False,
        "interpretation": (
            "absent automatic court roots; root-dependent soft priors, anchors and witnesses "
            "abstain, while side, supplied stature and ball/camera/event evidence are retained"
        ),
    }


def replay_policy(args, configuration: dict) -> str:
    """Rebuild the policy recorded by the search, refusing an explicit mismatch."""
    recorded = configuration.get("missing_player_position", "off")
    if recorded not in ("off", "abstain"):
        raise ValueError("unsupported recorded missing-player-position policy")
    if getattr(args, "missing_player_position", recorded) != recorded:
        raise ValueError("missing-player-position replay policy differs from the search")
    return recorded


def contact_states(players: list[dict], right_contact_player: dict | None = None) -> list[dict]:
    """Original contact order, including the associated prefix closing actor.

    The closing actor keeps index len(players); no missing actor is synthesized.
    This extends the same absence receipt rather than creating a second schema.
    """
    return players if right_contact_player is None else [*players, right_contact_player]


def require_replayed_states(
    configuration: dict, players: list[dict], *, right_contact_player: dict | None = None
) -> None:
    """Source reconstruction must reproduce the recorded absence, not trust it."""
    actual = state_receipt(contact_states(players, right_contact_player))
    if configuration.get("missing_player_position", "off") == "abstain":
        if configuration.get("player_position_receipt") != actual:
            raise ValueError("reconstructed player-position evidence differs from the search")
    elif actual["absent_contact_indices"]:
        raise ValueError("absent player position requires the recorded abstain policy")
