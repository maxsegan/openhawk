"""Soft stature/pose witnesses for evaluation-only connected shooting.

These are deliberately not population-calibrated anatomical limits.  Stature
turns the former one-size-fits-all serve-height and root-reach cutoffs into
reported, scale-aware hinge penalties.  A pose wrist is an additional image
witness when present; missing or low-confidence wrists abstain.

Stature is supplied evidence.  A player state may instead declare its athlete
evidence explicitly ``unavailable`` (no roster identity, no measured stature),
in which case every stature-scaled term for that player abstains with a
fixed-shape zero residual and a null receipt.  A missing stature without that
declaration is malformed and fails closed; nothing here invents a height.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cv.experiments.connected_shooting import player_position

ROOT_REACH_LOSSES = ("quadratic", "cauchy")
ROOT_REACH_CAUCHY_SCALE = 4.0


def root_reach_loss(players: list[dict[str, Any]]) -> str:
    """One invocation-wide loss, retained on every serialized player state."""
    modes = {row.get("athlete_root_reach_loss", "quadratic") for row in players}
    if len(modes) != 1 or not modes <= set(ROOT_REACH_LOSSES):
        raise ValueError("one supported uniform athlete root-reach loss required")
    return modes.pop()


def configure_root_reach(players: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Bind the shared policy once; legacy quadratic states remain byte-compatible."""
    if mode not in ROOT_REACH_LOSSES:
        raise ValueError("supported athlete root-reach loss required")
    for row in players:
        if row.get("athlete_root_reach_loss", mode) != mode:
            raise ValueError("player root-reach loss differs from shared policy")
        if mode != "quadratic":
            row["athlete_root_reach_loss"] = mode
    root_reach_loss(players)
    return players


def root_reach_configuration(mode: str) -> dict[str, Any]:
    if mode not in ROOT_REACH_LOSSES:
        raise ValueError("supported athlete root-reach loss required")
    return {} if mode == "quadratic" else {"athlete_root_reach_loss": mode}


def replay_root_reach(args: Any, report: dict) -> str:
    requested = getattr(args, "athlete_root_reach_loss", "quadratic")
    declared = report.get("configuration", {}).get("athlete_root_reach_loss", "quadratic")
    if declared not in ROOT_REACH_LOSSES or requested != declared:
        raise ValueError("replay athlete root-reach loss must match the cached search")
    return declared


def root_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve independent calibration disagreement without choosing either as truth.

    Only the shared camera transform emits these fields. No uncertainty scale or
    admission threshold is inferred from their disagreement.
    """
    if "s6_root_status" not in row:
        return {}
    original = [row.get("s6_original_court_x"), row.get("s6_original_court_y")]
    derived = [row.get("court_x"), row.get("court_y")]

    def finite_xy(values):
        try:
            result = np.asarray(values, float)
            return result.tolist() if np.isfinite(result).all() else None
        except (ValueError, TypeError):
            return None

    original, derived = finite_xy(original), finite_xy(derived)
    return {
        "root_observation": {
            "source_frame": row.get("frame"),
            "derivation_status": row["s6_root_status"],
            "proxy": row.get("s6_root_proxy"),
            "original_court_xy_m": original,
            "same_camera_court_xy_m": derived,
            "original_derived_disagreement_m": (
                None
                if original is None or derived is None
                else float(np.linalg.norm(np.asarray(original) - derived))
            ),
            "interpretation": (
                "independent original/derived root disagreement; diagnostic uncertainty, "
                "neither root is certified athlete truth; no numerical reweighting"
            ),
        }
    }


@dataclass(frozen=True)
class AthletePriorConfig:
    serve_height_low_statures: float = 1.50
    serve_height_high_statures: float = 1.75
    serve_height_sigma_statures: float = 0.08
    root_reach_limit_statures: float = 1.35
    root_reach_sigma_statures: float = 0.10
    wrist_confidence_minimum: float = 0.25
    wrist_ball_allowance_m: float = 0.80
    wrist_ball_sigma_m: float = 0.10

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), float)
        if (
            not np.isfinite(values).all()
            or np.any(values <= 0)
            or self.serve_height_low_statures >= self.serve_height_high_statures
            or self.wrist_confidence_minimum >= 1
        ):
            raise ValueError("finite positive ordered athlete-prior configuration required")


UNAVAILABLE = "unavailable"
UNAVAILABLE_REASON = "no roster identity or measured stature supplied; stature terms abstain"


def unavailable_evidence(reason: str = UNAVAILABLE_REASON) -> dict[str, str]:
    """The explicit per-player declaration that no athlete evidence exists."""
    return {"status": UNAVAILABLE, "reason": reason}


def stature_available(player: dict[str, Any]) -> bool:
    """Distinguish supplied stature from explicitly unavailable athlete evidence.

    A finite positive stature is evidence.  A ``None`` stature is accepted only
    when the state carries an explicit ``athlete_evidence`` unavailable
    declaration; a declaration alongside a stature, or a missing stature without
    one, is malformed and refused.
    """
    stature = player.get("stature_m")
    declared = (player.get("athlete_evidence") or {}).get("status")
    if stature is None:
        if declared != UNAVAILABLE:
            raise ValueError(
                "player stature missing without an explicit unavailable athlete-evidence "
                "declaration"
            )
        return False
    if declared == UNAVAILABLE:
        raise ValueError("a supplied stature cannot also declare athlete evidence unavailable")
    if not np.isfinite(float(stature)) or float(stature) <= 0:
        raise ValueError("finite positive statures and court roots required")
    return True


def _outside_interval(value: float, low: float, high: float) -> float:
    return float(max(low - value, value - high, 0.0))


def optimization_residuals(
    contacts_xyz_m: np.ndarray,
    players: list[dict[str, Any]],
    config: AthletePriorConfig = AthletePriorConfig(),
) -> np.ndarray:
    """Return scale-free soft hinges that can guide, but never hard-cut, a fit."""
    config.validate()
    contacts = np.asarray(contacts_xyz_m, float)
    if contacts.shape != (len(players), 3) or not np.isfinite(contacts).all():
        raise ValueError("one finite 3D contact per athlete prior required")
    available = np.asarray([stature_available(row) for row in players], bool)
    # Unavailable players carry a unit placeholder scale that only ever multiplies
    # a masked zero; it is never reported or read back as a height.
    statures = np.asarray(
        [row["stature_m"] if known else 1.0 for row, known in zip(players, available)], float
    )
    # An absent court root masks that player's reach term only; serve height is a
    # stature term and is unaffected by whether the root is known.
    roots, rooted = player_position.placeholder_roots(players)
    if not np.isfinite(statures).all() or np.any(statures <= 0):
        raise ValueError("finite positive statures and court roots required")
    serve_low = config.serve_height_low_statures * statures[0]
    serve_high = config.serve_height_high_statures * statures[0]
    serve = _outside_interval(contacts[0, 2], serve_low, serve_high) / (
        config.serve_height_sigma_statures * statures[0]
    )
    from cv.pipeline import s6_first_contact_role as role

    if not available[0] or not role.serve_priors_applicable(players):
        serve = 0.0
    distances = np.linalg.norm(contacts[:, :2] - roots, axis=1)
    reach = np.maximum(distances - config.root_reach_limit_statures * statures, 0.0) / (
        config.root_reach_sigma_statures * statures
    )
    reach = np.where(available & rooted, reach, 0.0)
    if root_reach_loss(players) == "cauchy":
        c = ROOT_REACH_CAUCHY_SCALE
        # Squaring this residual reproduces rho exactly in every existing solver
        # and selector. Only reach is robustified; serve height is unchanged.
        reach = c * np.sqrt(np.log1p((reach / c) ** 2))
    return np.r_[serve, reach]


def evaluate(
    contacts_xyz_m: np.ndarray,
    players: list[dict[str, Any]],
    config: AthletePriorConfig = AthletePriorConfig(),
) -> dict[str, Any]:
    """Report every soft term, including pose-wrist support and sensitivities."""
    contacts = np.asarray(contacts_xyz_m, float)
    residuals = optimization_residuals(contacts, players, config)
    loss = root_reach_loss(players)
    serve_player = players[0]
    from cv.pipeline import s6_first_contact_role as role

    is_rally = not role.serve_priors_applicable(players)
    serve_available = stature_available(serve_player)
    serve_stature = float(serve_player["stature_m"]) if serve_available else None
    low = None if serve_stature is None else config.serve_height_low_statures * serve_stature
    high = None if serve_stature is None else config.serve_height_high_statures * serve_stature
    abstained = [] if serve_available or is_rally else ["serve_height"]
    rows = []
    for index, (contact, player, residual) in enumerate(
        zip(contacts, players, residuals[1:], strict=True)
    ):
        available = stature_available(player)
        rooted = player_position.root_available(player)
        stature = float(player["stature_m"]) if available else None
        if not (available and rooted):
            abstained.append(f"root_reach[{index}]")
        root_distance = (
            float(np.linalg.norm(contact[:2] - player["court_centre_xy_m"])) if rooted else None
        )
        wrist = player.get("pose_wrist_witness", {})
        wrist_supported = wrist.get("status") == "supported"
        wrist_proxy = wrist.get("wrist_to_ball_proxy_m")
        wrist_excess = (
            None
            if not wrist_supported or wrist_proxy is None
            else max(0.0, float(wrist_proxy) - config.wrist_ball_allowance_m)
        )
        rows.append(
            {
                "player": player["player"],
                "stature_m": stature,
                "root_to_contact_m": root_distance,
                "zero_penalty_root_reach_m": (
                    None if stature is None else config.root_reach_limit_statures * stature
                ),
                "root_reach_sigma_m": (
                    None if stature is None else config.root_reach_sigma_statures * stature
                ),
                **({} if available else {"athlete_evidence": player["athlete_evidence"]}),
                **(
                    {}
                    if rooted
                    else {"player_position_evidence": player["player_position_evidence"]}
                ),
                "root_reach_residual": float(residual),
                "root_reach_selector_penalty": float(0.5 * residual**2),
                **(
                    {
                        "root_reach_raw_residual": (
                            0.0
                            if stature is None or root_distance is None
                            else max(
                                root_distance - config.root_reach_limit_statures * stature, 0.0
                            )
                            / (config.root_reach_sigma_statures * stature)
                        ),
                        "root_reach_loss": loss,
                        "root_reach_cauchy_scale": ROOT_REACH_CAUCHY_SCALE,
                    }
                    if loss == "cauchy"
                    else {}
                ),
                **(
                    {"root_observation": player["root_observation"]}
                    if "root_observation" in player
                    else {}
                ),
                "pose_wrist_witness": wrist,
                "wrist_selector_penalty": (
                    0.0
                    if wrist_excess is None
                    else float(0.5 * (wrist_excess / config.wrist_ball_sigma_m) ** 2)
                ),
            }
        )
    serve_residual = float(residuals[0])
    total = float(0.5 * np.sum(residuals**2))
    return {
        "schema": "connected_athlete_soft_priors_v1",
        "hard_gate": False,
        **root_reach_configuration(loss),
        "serve_height": {
            **(
                {
                    "applicable": False,
                    "reason": role.abstention_reason(role.bound_role(players)),
                    "role_evidence": serve_player[role.FIELD],
                }
                if is_rally
                else {}
            ),
            "player": serve_player["player"],
            "stature_m": serve_stature,
            "contact_height_m": float(contacts[0, 2]),
            "zero_penalty_interval_m": None if serve_stature is None or is_rally else [low, high],
            "interval_stature_multipliers": [
                config.serve_height_low_statures,
                config.serve_height_high_statures,
            ],
            "outside_sigma_m": (
                None
                if serve_stature is None
                else config.serve_height_sigma_statures * serve_stature
            ),
            **({} if serve_available else {"athlete_evidence": serve_player["athlete_evidence"]}),
            "residual": serve_residual,
            "selector_penalty": float(0.5 * serve_residual**2),
        },
        "contacts": rows,
        "optimization_residuals": residuals.tolist(),
        "selector_penalty": total,
        "athlete_evidence": {
            "status": (
                "supplied"
                if not abstained
                else ("unavailable" if len(abstained) == 1 + len(players) else "partial")
            ),
            "abstained_terms": abstained,
            "abstained_residuals_fixed_zero": True,
        },
        "player_position": player_position.state_receipt(players),
        "interpretation": (
            "soft stature-scaled serve-height and root-reach hinges; pose wrist is a "
            "reported native-image/racket-distance witness and abstains when unavailable"
        ),
    }
