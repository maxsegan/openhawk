"""Whole-point interpretation branches for physics reconstruction.

Each branch keeps a discrete event interpretation and stroke family for every flight.
Continuous trajectory parameters are optimized separately inside each branch; alternatives
are compared only after whole-point physics, tennis grammar, event probability, and empirical
spin priors have been scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NET_Y_M = 23.77 / 2.0
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


@dataclass(frozen=True)
class SpinPrior:
    profile: str
    center_rpm: float
    sigma_rpm: float
    evidence_cost: float
    sidespin_center_rpm: float = 0.0
    sidespin_sigma_rpm: float = 1000.0
    rifle_center_rpm: float = 0.0
    rifle_sigma_rpm: float = 650.0
    beam_cost: float | None = None

    @property
    def center_components_rpm(self) -> tuple[float, float, float]:
        return (
            self.center_rpm,
            self.sidespin_center_rpm,
            self.rifle_center_rpm,
        )

    @property
    def sigma_components_rpm(self) -> tuple[float, float, float]:
        return (
            self.sigma_rpm,
            self.sidespin_sigma_rpm,
            self.rifle_sigma_rpm,
        )


def typical_spin_prior(
    *,
    gender: str,
    phase: str,
    speed_kmh: float,
    profile: str,
) -> tuple[float, float]:
    """Return a deliberately broad signed-RPM prior.

    The values are population anchors, not player truth. Serve spin has a speed tradeoff;
    normal groundstroke spin stays broad because measured women's data did not find a
    dependable speed-spin relationship. Negative RPM denotes backspin/slice.
    """
    gender = "women" if gender == "women" else "men"
    if phase == "serve":
        if gender == "women":
            tradeoff = float(np.clip(5000.0 - 20.0 * speed_kmh, 1300.0, 3300.0))
        else:
            tradeoff = float(np.clip(5700.0 - 20.0 * speed_kmh, 1200.0, 3300.0))
        if profile == "high_spin_serve":
            return tradeoff + 650.0, 850.0
        return max(700.0, tradeoff - 500.0), 900.0
    if profile == "slice":
        return (-1500.0, 850.0) if gender == "women" else (-1900.0, 950.0)
    if profile == "lob":
        return (2600.0, 1200.0) if gender == "women" else (3200.0, 1300.0)
    return (1900.0, 850.0) if gender == "women" else (2700.0, 950.0)


def shot_spin_options(
    *,
    gender: str,
    phase: str,
    speed_kmh: float,
    fitted_signed_rpm: float,
    fitted_sidespin_rpm: float = 0.0,
    fitted_rifle_rpm: float = 0.0,
    apex_m: float,
    duration_s: float,
    pose_slice_probability: float | None = None,
) -> list[SpinPrior]:
    """Generate discrete stroke-family interpretations with evidence costs."""
    if phase == "serve":
        serve_total = {
            "flat_serve": float(np.clip(2700.0 - 8.0 * speed_kmh, 900.0, 1800.0)),
            "slice_serve": float(np.clip(4800.0 - 15.0 * speed_kmh, 1700.0, 3000.0)),
            "kick_serve": float(np.clip(5400.0 - 15.0 * speed_kmh, 2400.0, 3800.0)),
        }
        profiles = (
            ("flat_serve", 1.0),
            ("slice_serve", -1.0),
            ("slice_serve", 1.0),
            ("kick_serve", -1.0),
            ("kick_serve", 1.0),
        )
    else:
        profiles = (("drive", 0.0), ("slice", 0.0), ("lob", 0.0))
    options = []
    for profile, side_sign in profiles:
        if phase == "serve":
            total = serve_total[profile]
            if profile == "flat_serve":
                center = total
                sidespin_center = 0.0
                sigma = 900.0
                sidespin_sigma = 1100.0
            elif profile == "slice_serve":
                center = 0.25 * total
                sidespin_center = side_sign * 0.97 * total
                sigma = 800.0
                sidespin_sigma = 850.0
            else:
                center = 0.85 * total
                sidespin_center = side_sign * 0.53 * total
                sigma = 900.0
                sidespin_sigma = 900.0
            rifle_sigma = 550.0
        else:
            center, sigma = typical_spin_prior(
                gender=gender,
                phase=phase,
                speed_kmh=speed_kmh,
                profile=profile,
            )
            sidespin_center = 0.0
            sidespin_sigma = 850.0 if profile == "drive" else 1000.0
            rifle_sigma = 650.0
        trajectory_cost = 0.5 * (
            ((fitted_signed_rpm - center) / sigma) ** 2
            + (
                (fitted_sidespin_rpm - sidespin_center)
                / sidespin_sigma
            )
            ** 2
            + (fitted_rifle_rpm / rifle_sigma) ** 2
        )
        evidence_cost = 0.0
        if profile == "slice" and pose_slice_probability is not None:
            evidence_cost -= math.log(max(pose_slice_probability, 1e-4))
        elif phase != "serve" and pose_slice_probability is not None:
            evidence_cost -= math.log(max(1.0 - pose_slice_probability, 1e-4))
        if profile == "lob":
            lob_evidence = max(
                (apex_m - 3.5) / 1.5,
                (duration_s - 1.15) / 0.55,
            )
            evidence_cost += max(0.0, 1.5 - lob_evidence)
        elif apex_m > 5.0 and phase != "serve":
            evidence_cost += (apex_m - 5.0) / 1.5
        label = (
            f"{profile}_{'left' if side_sign < 0 else 'right'}"
            if phase == "serve" and profile != "flat_serve"
            else profile
        )
        options.append(
            SpinPrior(
                label,
                center,
                sigma,
                float(evidence_cost),
                sidespin_center,
                sidespin_sigma,
                0.0,
                rifle_sigma,
                trajectory_cost + evidence_cost,
            )
        )
    return sorted(
        options,
        key=lambda row: (
            row.beam_cost
            if row.beam_cost is not None
            else row.evidence_cost
        ),
    )[:4]


def bounce_grammar_cost(
    bounce: dict | None,
    *,
    start_side: str,
    end_side: str,
) -> float:
    if bounce is None:
        return 0.0
    if start_side == end_side or start_side not in {"near", "far"}:
        return 8.0
    y = float(np.asarray(bounce["x"], float)[1])
    expected_far = start_side == "near"
    wrong_half = (expected_far and y < NET_Y_M) or (not expected_far and y > NET_Y_M)
    outside = max(0.0, -y) + max(0.0, y - 23.77)
    return (12.0 if wrong_half else 0.0) + 4.0 * outside


def bounce_options(
    candidates: list[dict],
    *,
    start_side: str,
    end_side: str,
    hard_anchor: dict | None = None,
) -> list[dict]:
    """Preserve discrete event alternatives; never average candidate locations."""
    if hard_anchor is not None:
        return [
            {
                "anchor": hard_anchor,
                "event_cost": 0.0,
                "grammar_cost": bounce_grammar_cost(
                    hard_anchor,
                    start_side=start_side,
                    end_side=end_side,
                ),
                "label": "trusted_bounce",
            }
        ]
    output = [
        {
            "anchor": None,
            "event_cost": -math.log(0.35),
            "grammar_cost": 0.0,
            "label": "no_bounce_anchor",
        }
    ]
    for candidate in sorted(
        candidates,
        key=lambda row: float(row.get("probability", 0.0)),
        reverse=True,
    )[:2]:
        probability = float(np.clip(candidate.get("probability", 0.0), 1e-4, 1.0))
        output.append(
            {
                "anchor": candidate,
                "event_cost": -math.log(probability),
                "grammar_cost": bounce_grammar_cost(
                    candidate,
                    start_side=start_side,
                    end_side=end_side,
                ),
                "label": f"bounce_f{float(candidate['frame']):.2f}",
            }
        )
    return sorted(
        output,
        key=lambda row: row["event_cost"] + row["grammar_cost"],
    )[:2]


def build_point_branches(
    shot_options: dict[int, dict],
    *,
    beam_width: int = 6,
) -> list[dict]:
    """Beam-search discrete whole-point interpretations."""

    def profile_family(profile: str) -> str:
        for suffix in ("_left", "_right"):
            if profile.endswith(suffix):
                return profile[: -len(suffix)]
        return profile

    def serve_family(branch: dict) -> str | None:
        for shot in branch["shots"].values():
            family = profile_family(shot["profile"])
            if family.endswith("_serve"):
                return family
        return None

    beam = [{"shots": {}, "prior_cost": 0.0, "beam_cost": 0.0}]
    for shot_index in sorted(shot_options):
        expanded = []
        spin_options = shot_options[shot_index]["spin"]
        event_options = shot_options[shot_index]["bounce"]
        for branch in beam:
            for spin in spin_options:
                for event in event_options:
                    shot_prior_cost = (
                        spin.evidence_cost
                        + float(event["event_cost"])
                        + float(event["grammar_cost"])
                    )
                    shot_beam_cost = (
                        (
                            spin.beam_cost
                            if spin.beam_cost is not None
                            else spin.evidence_cost
                        )
                        + float(event["event_cost"])
                        + float(event["grammar_cost"])
                    )
                    expanded.append(
                        {
                            "shots": {
                                **branch["shots"],
                                shot_index: {
                                    "profile": spin.profile,
                                    "spin_center_rpm": spin.center_rpm,
                                    "spin_sigma_rpm": spin.sigma_rpm,
                                    "spin_center_components_rpm": list(
                                        spin.center_components_rpm
                                    ),
                                    "spin_sigma_components_rpm": list(
                                        spin.sigma_components_rpm
                                    ),
                                    "bounce_anchor": event["anchor"],
                                    "bounce_label": event["label"],
                                    "local_cost": shot_prior_cost,
                                    "start_side": shot_options[shot_index].get(
                                        "start_side",
                                        "unknown",
                                    ),
                                    "end_side": shot_options[shot_index].get(
                                        "end_side",
                                        "unknown",
                                    ),
                                },
                            },
                            "prior_cost": (
                                branch["prior_cost"] + shot_prior_cost
                            ),
                            "beam_cost": (
                                branch["beam_cost"] + shot_beam_cost
                            ),
                        }
                    )
        ranked = sorted(expanded, key=lambda row: row["beam_cost"])
        diverse = []
        available_serve_families = {
            serve_family(branch)
            for branch in ranked
            if serve_family(branch) is not None
        }
        if len(available_serve_families) > 1:
            seen_serve_families = set()
            for branch in ranked:
                family = serve_family(branch)
                if family is None or family in seen_serve_families:
                    continue
                diverse.append(branch)
                seen_serve_families.add(family)
                if len(diverse) == beam_width:
                    break
        selected_ids = {id(branch) for branch in diverse}
        seen_families = set()
        for branch in ranked:
            if id(branch) in selected_ids:
                continue
            family = profile_family(
                branch["shots"][shot_index]["profile"]
            )
            if family in seen_families:
                continue
            diverse.append(branch)
            selected_ids.add(id(branch))
            seen_families.add(family)
            if len(diverse) == beam_width:
                break
        if len(diverse) < beam_width:
            diverse.extend(
                branch
                for branch in ranked
                if id(branch) not in selected_ids
            )
        beam = diverse[:beam_width]
    for index, branch in enumerate(beam):
        branch["branch_id"] = f"interpretation_{index:02d}"
    return beam


def refined_branch_score(
    *,
    fits: dict,
    branch: dict,
    junction_gaps_m: list[float],
) -> dict:
    """Comparable posterior-like cost after independent whole-point optimization."""
    if not fits:
        return {
            "total": math.inf,
            "reprojection": math.inf,
            "junction": math.inf,
            "spin": math.inf,
            "grammar": math.inf,
            "prior": float(branch["prior_cost"]),
        }
    reprojection = float(
        sum(
            float(getattr(fit, "_weighted_rms_px", fit.rms_px)) ** 2
            * max(1, int(getattr(fit, "n_obs", 1)))
            for fit in fits.values()
        )
    )
    junction = float(sum((gap / 0.20) ** 2 for gap in junction_gaps_m))
    spin_cost = 0.0
    grammar_cost = 0.0
    for shot_index, interpretation in branch["shots"].items():
        fit = fits.get(shot_index)
        if fit is None:
            spin_cost += 25.0
            continue
        fitted_rpm = np.zeros(3)
        fitted_rpm[: min(3, max(0, len(fit.theta) - 6))] = (
            np.asarray(fit.theta[6:9], float) * 100.0 * RAD_S_TO_RPM
        )
        centers = np.asarray(
            interpretation.get(
                "spin_center_components_rpm",
                [interpretation["spin_center_rpm"], 0.0, 0.0],
            ),
            float,
        )
        sigmas = np.asarray(
            interpretation.get(
                "spin_sigma_components_rpm",
                [interpretation["spin_sigma_rpm"], 1200.0, 800.0],
            ),
            float,
        )
        spin_cost += 0.5 * float(np.sum(((fitted_rpm - centers) / sigmas) ** 2))
        bounces = list(getattr(fit, "bounces", []))
        if len(bounces) > 1:
            grammar_cost += 16.0 * (len(bounces) - 1)
        for bounce in bounces:
            grammar_cost += bounce_grammar_cost(
                bounce,
                start_side=interpretation.get("start_side", "unknown"),
                end_side=interpretation.get("end_side", "unknown"),
            )
        positions = np.asarray(getattr(fit, "xs_obs", []), float)
        if len(positions) >= 2:
            crossings = []
            for first, second in zip(positions[:-1], positions[1:]):
                delta_first = first[1] - NET_Y_M
                delta_second = second[1] - NET_Y_M
                if delta_first == 0:
                    crossings.append(float(first[2]))
                elif delta_first * delta_second < 0:
                    fraction = abs(delta_first) / (
                        abs(delta_first) + abs(delta_second)
                    )
                    crossings.append(
                        float(first[2] + fraction * (second[2] - first[2]))
                    )
            if crossings:
                net_height = 0.914 + 0.0325
                grammar_cost += sum(
                    (max(0.0, net_height - height) / 0.08) ** 2
                    for height in crossings
                )
            elif (
                interpretation.get("start_side") in {"near", "far"}
                and interpretation.get("end_side") in {"near", "far"}
                and interpretation["start_side"] != interpretation["end_side"]
            ):
                grammar_cost += 20.0
    prior = float(branch["prior_cost"])
    return {
        "total": reprojection + junction + spin_cost + grammar_cost + prior,
        "reprojection": reprojection,
        "junction": junction,
        "spin": float(spin_cost),
        "grammar": float(grammar_cost),
        "prior": prior,
    }


def branch_is_decisive(
    ranked: list[dict],
    *,
    minimum_margin: float = 8.0,
) -> tuple[bool, float | None]:
    if not ranked:
        return False, None
    if len(ranked) == 1:
        return True, math.inf
    margin = float(ranked[1]["score"]["total"] - ranked[0]["score"]["total"])
    return margin >= minimum_margin, margin
