"""Court bounce fitted to measured Hawk-Eye rebounds.

Provenance
----------
``MEASURED`` holds the coefficients fitted by ``cv.validation.hawkeye_bounce_model`` over
consecutive strike pairs of the Hawk-Eye CourtVision corpus at
``data/external/ryurko-hawkeye`` (Roland Garros clay, Australian Open hard,
2019-2021).  Both flights bracketing each impact are fitted with :mod:`physics.flight` on the
production aero constants; the incoming velocity comes from the ``hit``/``peak``/``net``/
``bounce`` landmarks and the outgoing velocity from ``bounce``/``peak``/next ``hit``.

What the corpus can and cannot measure
--------------------------------------
* **Vertical restitution and horizontal retention are measured**, but only *jointly with an
  assumed outgoing spin*.  The corpus has no timestamps, so the rebound arc is identified
  from position landmarks alone, and a change in assumed outgoing topspin trades against
  outgoing speed at essentially zero residual cost.  ``MEASURED`` therefore records the value
  at the Cross-predicted outgoing spin (``*_at_model_spin``) *and* at zero outgoing spin
  (``*_spin_free``).  The truth lies between them; the width of that band is part of the
  model's stated uncertainty, not noise.
* **Outgoing spin is not measured.**  Refitting the rebound with topspin fixed anywhere in
  0-4000 rpm moves the landmark residual by well under a millimetre.  This module therefore
  takes outgoing spin from the Cross slide/grip solution in :mod:`physics.impact` and says so.
* **Dwell time is not measurable here.**  ``DWELL_SECONDS`` is taken from
  ``physics/REFERENCE.md`` (Cross: 4-5 ms) and is used only to offset the outgoing flight's
  time origin; no corpus quantity constrains it.
* **Only clay and hard court come from the corpus.**  Grass is absent from it.  The grass row
  below is measured instead from the labeled grass broadcasts themselves by
  ``cv.experiments.connected_shooting.grass_bounce_profile``: each labeled impact fixes the
  ball's court position through its own observation ray, the arcs on both sides are fitted to
  the native fronts, and the ratio of the fitted velocities is read off directly.  It is a
  thin, human-label-derived calibration -- eleven impacts across eight attempts -- shrunk
  toward the Tennis Industry chart expectation with a wide prior, and it carries a per-impact
  spread of about 0.3.  The same estimator run on the hard and clay attempts, where the corpus
  already knows the answer, returns 0.878 and 0.696 restitution against the corpus's 0.894 and
  0.916 at model spin: close on hard, low on clay, and low on horizontal retention for both.
  Treat the grass row as a directional replacement for fitting grass on the hard profile, not
  as a measurement of the same standing as the corpus rows.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics import impact, surface_model

# Contact duration.  physics/REFERENCE.md, Cross 2020: 4-5 ms; the Hawk-Eye corpus carries no
# timestamps and cannot measure it.
DWELL_SECONDS = 0.0045

# Filled from processed/wk3_synthpoint/bounce_model/bounce_model.json.  Every coefficient is a
# least-squares fit of the quantity against incidence angle, centred on 16 degrees (the corpus
# median groundstroke incidence) so the intercept reads as "a typical rally bounce".
MEASURED_FULL_CORPUS: dict[str, dict[str, float]] = {
    "clay": {
        "incidence_deg_median": 16.8162,
        "n": 1322,
        "restitution_intercept_at_model_spin": 0.916414,
        "restitution_intercept_spin_free": 0.745581,
        "restitution_residual_std_at_model_spin": 0.136023,
        "restitution_residual_std_spin_free": 0.09208,
        "restitution_slope_at_model_spin": -0.006938,
        "restitution_slope_spin_free": 0.000275,
        "retention_intercept_at_model_spin": 0.676912,
        "retention_intercept_spin_free": 0.559693,
        "retention_residual_std_at_model_spin": 0.171067,
        "retention_residual_std_spin_free": 0.132335,
        "retention_slope_at_model_spin": -0.006952,
        "retention_slope_spin_free": -0.001105,
    },
    # Broadcast-label-derived, not from the Hawk-Eye corpus.  See the module note.
    # Artifact: TENNIS_DATA_ROOT/processed/refusals/bounce_profile/bounce_profile.json.
    "grass": {
        "incidence_deg_median": 20.1762,
        "n": 11,
        "restitution_intercept_at_model_spin": 0.59699,
        "restitution_intercept_spin_free": 0.59699,
        "restitution_residual_std_at_model_spin": 0.296584,
        "restitution_residual_std_spin_free": 0.296584,
        "restitution_slope_at_model_spin": -0.007244,
        "restitution_slope_spin_free": -0.007244,
        "retention_intercept_at_model_spin": 0.633372,
        "retention_intercept_spin_free": 0.633372,
        "retention_residual_std_at_model_spin": 0.350348,
        "retention_residual_std_spin_free": 0.350348,
        "retention_slope_at_model_spin": -0.001257,
        "retention_slope_spin_free": -0.001257,
    },
    "hard": {
        "incidence_deg_median": 16.3191,
        "n": 1294,
        "restitution_intercept_at_model_spin": 0.893999,
        "restitution_intercept_spin_free": 0.740133,
        "restitution_residual_std_at_model_spin": 0.113189,
        "restitution_residual_std_spin_free": 0.083971,
        "restitution_slope_at_model_spin": -0.006849,
        "restitution_slope_spin_free": -0.000283,
        "retention_intercept_at_model_spin": 0.668357,
        "retention_intercept_spin_free": 0.561441,
        "retention_residual_std_at_model_spin": 0.14599,
        "retention_residual_std_spin_free": 0.116354,
        "retention_slope_at_model_spin": -0.008808,
        "retention_slope_spin_free": -0.003293,
    },
}

PROVENANCE_FULL_CORPUS: dict[str, object] = {
    "law": "full_corpus",
    "aero_params": {
        "C_drag": 0.55,
        "C_lift": 0.6,
        "C_spin_decay": 0.025,
        "J_ball": 3.155e-05,
        "R_ball": 0.0325,
        "g": 9.81,
        "m_ball": 0.057,
        "rho_air": 1.205,
    },
    "artifact": "data/processed/wk3_synthpoint/bounce_model/bounce_model.json",
    "files_read": 200,
    "incoming_spin_prior_rpm": {"clay": 2184.25, "hard": 1870.06},
    "grass_source": {
        "origin": "labeled grass broadcasts, not the Hawk-Eye corpus",
        "estimator": "cv.experiments.connected_shooting.grass_bounce_profile",
        "impacts": 11,
        "attempts": 8,
        "human_derived": True,
        "prior": "physics.impact chart model at the median incidence, 0.15 intercept sd",
        "control_on_measured_surfaces": {
            "hard": {
                "estimated_restitution": 0.8781,
                "corpus_restitution_at_model_spin": 0.893999,
                "estimated_retention": 0.6047,
                "corpus_retention_at_model_spin": 0.668357,
            },
            "clay": {
                "estimated_restitution": 0.6963,
                "corpus_restitution_at_model_spin": 0.916414,
                "estimated_retention": 0.4083,
                "corpus_retention_at_model_spin": 0.676912,
            },
        },
        "known_bias": (
            "the estimator reads horizontal retention low, by 0.06 on hard and 0.27 on clay, "
            "because one broadcast camera sees along-axis speed poorly; the grass retention "
            "row inherits that bias and is not on the same scale as the corpus rows"
        ),
        "spin_resolved": False,
    },
    "outgoing_spin_identifiable": False,
    "outgoing_spin_rmse_spread_m_median": 0.0002825082407889616,
    "pairs_enumerated": 3200,
    "pairs_measured": 2616,
}

# The production law.  Same estimator, pair selection and aero constants as the full-corpus
# law, but every Hawk-Eye match that is also one of our broadcast evaluation matches (51 matches,
# listed in the sidecar) is dropped before
# fitting, including from the incoming-spin prior, so the Hawk-Eye comparison on those matches
# is out of sample.  Grass is not from the corpus and is carried over unchanged.  Every
# corpus coefficient moves by less than one match-bootstrap sd (clay restitution 0.916 ->
# 0.913); docs/PHYSICS.md.  The fitted and excluded match ids are in
# the sidecar bounce_law_hawkeye_holdout.json.
MEASURED_HAWKEYE_HOLDOUT: dict[str, dict[str, float]] = {
    "clay": {
        "incidence_deg_median": 16.8094,
        "n": 1171,
        "restitution_intercept_at_model_spin": 0.913339,
        "restitution_intercept_spin_free": 0.744743,
        "restitution_residual_std_at_model_spin": 0.134705,
        "restitution_residual_std_spin_free": 0.091833,
        "restitution_slope_at_model_spin": -0.006662,
        "restitution_slope_spin_free": 0.000411,
        "retention_intercept_at_model_spin": 0.674172,
        "retention_intercept_spin_free": 0.558673,
        "retention_residual_std_at_model_spin": 0.168761,
        "retention_residual_std_spin_free": 0.131143,
        "retention_slope_at_model_spin": -0.006733,
        "retention_slope_spin_free": -0.001,
    },
    "grass": MEASURED_FULL_CORPUS["grass"],
    "hard": {
        "incidence_deg_median": 16.3719,
        "n": 1256,
        "restitution_intercept_at_model_spin": 0.892781,
        "restitution_intercept_spin_free": 0.73932,
        "restitution_residual_std_at_model_spin": 0.113402,
        "restitution_residual_std_spin_free": 0.084246,
        "restitution_slope_at_model_spin": -0.006828,
        "restitution_slope_spin_free": -0.000195,
        "retention_intercept_at_model_spin": 0.667445,
        "retention_intercept_spin_free": 0.560797,
        "retention_residual_std_at_model_spin": 0.146162,
        "retention_residual_std_spin_free": 0.11645,
        "retention_slope_at_model_spin": -0.008845,
        "retention_slope_spin_free": -0.003263,
    },
}

_HOLDOUT_MATCHES = json.loads(
    (Path(__file__).with_name("bounce_law_hawkeye_holdout.json")).read_text(encoding="utf-8")
)

PROVENANCE_HAWKEYE_HOLDOUT: dict[str, object] = {
    **PROVENANCE_FULL_CORPUS,
    "law": "hawkeye_holdout",
    "artifact": (
        "data/processed/abstract-20260929/bounce_holdout_clean/bounce_holdout.json"
    ),
    "estimator": _HOLDOUT_MATCHES["estimator"],
    "files_read": 200,
    "files_fitted": len(_HOLDOUT_MATCHES["fitted_hawkeye_match_ids"]),
    "incoming_spin_prior_rpm": {"clay": 2137.12, "hard": 1869.12},
    "pairs_enumerated": 3200,
    "pairs_enumerated_kept": 2973,
    "pairs_measured": 2427,
    "hawkeye_matches_fitted": _HOLDOUT_MATCHES["fitted_hawkeye_match_ids"],
    "hawkeye_matches_excluded": _HOLDOUT_MATCHES["excluded_overlap_hawkeye_match_ids"],
    "hawkeye_matches_excluded_from_fit_sample": _HOLDOUT_MATCHES[
        "excluded_hawkeye_match_ids_in_fit_sample"
    ],
    "fit_matches_sidecar": "physics/bounce_law_hawkeye_holdout.json",
}

#: Law selector.  Unset selects ``DEFAULT_LAW``; any other value must name a key of ``LAWS``.
#: Read once at import, so every process of a run inherits its launcher's choice.
LAW_VARIABLE = "TENNIS_BOUNCE_LAW"
DEFAULT_LAW = "hawkeye_holdout"
LAWS: dict[str, tuple[dict[str, dict[str, float]], dict[str, object]]] = {
    "hawkeye_holdout": (MEASURED_HAWKEYE_HOLDOUT, PROVENANCE_HAWKEYE_HOLDOUT),
    # Rollback: TENNIS_BOUNCE_LAW=full_corpus restores the law fitted on all 200 files.
    "full_corpus": (MEASURED_FULL_CORPUS, PROVENANCE_FULL_CORPUS),
}


def _selected_law() -> str:
    name = os.environ.get(LAW_VARIABLE) or DEFAULT_LAW
    if name not in LAWS:
        raise ValueError(f"{LAW_VARIABLE}={name!r}; known bounce laws are {sorted(LAWS)}")
    return name


LAW_NAME = _selected_law()
MEASURED, PROVENANCE = LAWS[LAW_NAME]


def law_record() -> dict[str, object]:
    """Identity of the selected law for a run's provenance record."""
    return {
        "name": f"physics.bounce_reference:{LAW_NAME}",
        "law": LAW_NAME,
        "selector": LAW_VARIABLE,
        "artifact": PROVENANCE["artifact"],
        "hawkeye_matches_fitted": PROVENANCE.get("hawkeye_matches_fitted"),
        "hawkeye_matches_excluded": PROVENANCE.get("hawkeye_matches_excluded", []),
        "incoming_spin_prior_rpm": PROVENANCE["incoming_spin_prior_rpm"],
    }


@dataclass(frozen=True)
class MeasuredBounceResult:
    velocity: np.ndarray
    spin: np.ndarray
    restitution: float
    horizontal_retention: float
    regime: str
    spin_source: str
    #: The region-aware surface state at this landing point, or ``None`` when the surface
    #: carried no model and the bounce is the plain measured lookup.
    surface_state: surface_model.SurfaceState | None = None


def surfaces() -> tuple[str, ...]:
    return tuple(sorted(MEASURED))


def _coefficients(surface: str) -> dict[str, float]:
    key = str(surface).lower()
    if key not in MEASURED:
        raise ValueError(
            f"no measured bounce coefficients for surface {surface!r}; "
            f"measured surfaces are {sorted(MEASURED)}"
        )
    return MEASURED[key]


def restitution(theta1_deg: float, surface: str, *, spin_free: bool = False) -> float:
    """Measured vertical restitution ``v_z_out / |v_z_in|`` at an incidence angle."""
    row = _coefficients(surface)
    suffix = "spin_free" if spin_free else "at_model_spin"
    value = row[f"restitution_intercept_{suffix}"] + row[f"restitution_slope_{suffix}"] * (
        float(theta1_deg) - 16.0
    )
    return float(np.clip(value, 0.05, 1.0))


def horizontal_retention(theta1_deg: float, surface: str, *, spin_free: bool = False) -> float:
    """Measured horizontal speed retention ``|v_h_out| / |v_h_in|`` at an incidence angle."""
    row = _coefficients(surface)
    suffix = "spin_free" if spin_free else "at_model_spin"
    value = row[f"retention_intercept_{suffix}"] + row[f"retention_slope_{suffix}"] * (
        float(theta1_deg) - 16.0
    )
    return float(np.clip(value, 0.05, 1.0))


def base_coefficients(
    incidence_deg: float, model: surface_model.SurfaceModel, *, spin_free: bool = False
) -> tuple[float, float]:
    """The pair the region model modifies, before any wear, kernel or line term.

    ``reference`` is this module's own measured row.  ``literature`` is the fresh pair of
    ``physics.surface_model``, which is what grass has instead of a corpus row: the Hawk-Eye
    corpus contains no grass at all.  ``hard_reference`` is the exact current grass-substitution
    prior used by the regional-v2 arm.
    """
    if model.base_source == "literature":
        return surface_model.fresh_base(model.surface, incidence_deg)
    if model.base_source == "hard_reference":
        return (
            restitution(incidence_deg, "hard", spin_free=spin_free),
            horizontal_retention(incidence_deg, "hard", spin_free=spin_free),
        )
    return (
        restitution(incidence_deg, model.surface, spin_free=spin_free),
        horizontal_retention(incidence_deg, model.surface, spin_free=spin_free),
    )


def court_bounce(
    velocity: np.ndarray,
    spin: np.ndarray,
    surface: str | surface_model.SurfaceModel,
    *,
    spin_free: bool = False,
    spin_regime_override: str | None = None,
    position: np.ndarray | None = None,
) -> MeasuredBounceResult:
    """Apply the measured bounce to a descending 3D velocity and spin.

    ``velocity`` is the incoming velocity in court coordinates with a negative vertical
    component.  The horizontal direction is preserved: the corpus measures a median lateral
    deflection of a fraction of a degree, which a deterministic model should not invent.

    ``spin_regime_override`` is a research optimization surrogate, not a physical
    regime judgment. The default keeps the existing Cross selector unchanged.

    ``surface`` may be a plain surface name, which behaves exactly as it always has, or a
    ``physics.surface_model`` spec such as ``"grass@w=1.06,a=0.15,base=fresh"``.  A spec that
    is not inert needs ``position``, the landing point in the court frame, because its whole
    content is that the court is not the same everywhere: it carries a per-broadcast wear
    scalar, a fixed spatial wear kernel and a line-skid switch.  A bounce with a model is
    solved through Cross's own equations at the local ``(e_y, mu)``, not through a second copy
    of the impact algebra.
    """
    if spin_regime_override not in {None, "slide", "grip"}:
        raise ValueError("supported explicit spin regime surrogate required")
    model = surface_model.parse(surface)
    velocity = np.asarray(velocity, dtype=float)
    spin = np.asarray(spin, dtype=float)
    horizontal = float(math.hypot(velocity[0], velocity[1]))
    vertical = float(-velocity[2])
    if vertical <= 0.0 or horizontal <= 0.0:
        raise ValueError("court bounce needs a descending, forward-moving incoming velocity")
    incidence = math.degrees(math.atan2(vertical, horizontal))
    base_restitution, base_retention = base_coefficients(incidence, model, spin_free=spin_free)
    state = None
    cross_coefficients = None
    cross_surface = "hard" if model.base_source == "hard_reference" else model.surface
    if model.inert:
        e_y, retention = base_restitution, base_retention
    else:
        if position is None:
            raise ValueError("a region-aware surface model needs the landing point; pass position=")
        point = np.asarray(position, dtype=float)
        e_y, retention, state = model.coefficients(
            float(point[0]), float(point[1]), base_restitution, base_retention
        )
        # The friction the Cross spin solution must see is the one this retention implies,
        # so the outgoing spin and the outgoing speed describe the same impact.
        tangent = math.tan(math.radians(min(max(incidence, 1.0), 80.0)))
        if not (
            model.base_source == "hard_reference"
            and e_y == base_restitution
            and retention == base_retention
        ):
            cross_coefficients = (e_y, (1.0 - retention) / ((1.0 + e_y) * tangent))
    direction = np.array([velocity[0], velocity[1], 0.0]) / horizontal
    outgoing = direction * (horizontal * retention)
    outgoing[2] = vertical * e_y
    # Outgoing spin is unmeasurable from the corpus; take Cross's slide/grip solution on the
    # measured incoming state and label it as such.
    axis = np.array([-direction[1], direction[0], 0.0])
    topspin_in = float(np.dot(spin, axis))
    try:
        cross = impact.court_bounce(
            horizontal,
            vertical,
            topspin_in,
            surface=cross_surface,
            regime_override=spin_regime_override,
            coefficients=cross_coefficients,
        )
        outgoing_spin = axis * float(cross.w2)
        regime = str(cross.regime)
    except (ValueError, ZeroDivisionError):
        outgoing_spin = axis * (horizontal * retention / impact.R_BALL)
        regime = "rolling_fallback"
    # Preserve any rifle (court-normal) spin component; the corpus cannot see it either.
    outgoing_spin[2] = float(spin[2])
    return MeasuredBounceResult(
        velocity=outgoing,
        spin=outgoing_spin,
        restitution=e_y,
        horizontal_retention=retention,
        regime=regime,
        spin_source="physics.impact.court_bounce (Cross); not measurable from the corpus",
        surface_state=state,
    )
