"""Input-declared native center or swept-front semantics for shared S6.

No coordinates or native epochs are converted. Legacy labeled packets keep the
historical quarter-frame leading-front operator; explicitly typed producers must
agree across their visible rows, attempt declaration and report replay.
"""

from __future__ import annotations

import math

SCHEMA = "ball_observation_operator_v1"
SEMANTICS = {
    "visible_blur_leading_edge_in_travel_direction": "leading_front",
    "visible blur leading edge in travel direction; native pixel uncertainty": "leading_front",
    "detector_heatmap_nominal_centre": "nominal_center",
}


def support_span(duration: float | None) -> float:
    """Forward native support; a nominal center samples only its own epoch."""
    if duration is None:
        return 0.0
    if isinstance(duration, bool) or not math.isfinite(duration) or not 0 < duration <= 1:
        raise ValueError("front duration must be positive finite and at most one native frame")
    return float(duration)


def declaration(kind: str) -> dict:
    if kind not in ("leading_front", "nominal_center"):
        raise ValueError("unknown ball observation operator")
    return dict(
        schema=SCHEMA,
        kind=kind,
        exposure_duration_frames=None if kind == "nominal_center" else 0.25,
    )


def from_packet(packet: dict) -> dict:
    attempts = packet.get("attempts", [])
    if len(attempts) != 1:
        raise ValueError("one attempt required for homogeneous observation operator")
    attempt = attempts[0]
    kinds = []
    for source in (packet, attempt):
        explicit = source.get("ball_observation_operator")
        if explicit is not None:
            if not isinstance(explicit, dict) or explicit != declaration(explicit.get("kind")):
                raise ValueError("invalid typed ball observation operator")
            kinds.append(explicit["kind"])
    convention = attempt.get("ball_convention")
    if convention is not None:
        if convention not in SEMANTICS:
            raise ValueError("unknown ball convention")
        kinds.append(SEMANTICS[convention])
    untyped_visible = False
    for name in ("ball_observations", "owner_ball_labels"):
        for row in attempt.get(name, []):
            semantic = row.get("observation_semantics")
            if semantic is not None:
                if semantic not in SEMANTICS:
                    raise ValueError("unknown ball observation semantics")
                kinds.append(SEMANTICS[semantic])
            elif row.get("status") == "visible":
                untyped_visible = True
    if len(set(kinds)) > 1:
        raise ValueError(
            "mixed center/front observations require a separate explicit mixed operator"
        )
    if "nominal_center" in kinds and untyped_visible:
        raise ValueError("center packets require typed visible observations")
    return declaration(kinds[0] if kinds else "leading_front")


def from_report(report: dict, packet: dict) -> dict:
    expected = from_packet(packet)
    config = report["configuration"]
    declared = config.get("ball_observation_operator")
    if declared is not None and declared != expected:
        raise ValueError("search observation operator differs from source packet")
    if declared is None and expected["kind"] != "leading_front":
        raise ValueError("center search must carry its original typed operator receipt")
    if config["exposure_duration_frames"] != expected["exposure_duration_frames"]:
        raise ValueError("search duration differs from source observation semantics")
    return expected
