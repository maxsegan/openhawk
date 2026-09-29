"""Impact models: court bounce (Cross 2020) and racket impact (Cross 2005 via Ido's note).

Companions to ``flight.py`` (Tomer's 3D flight integrator). The calibrated scalar court and
racket models remain planar; ``court_bounce_vector`` extends court contact to a coupled
two-axis tangential impulse while preserving the scalar model exactly when lateral contact
slip is zero. Flight handles full 3D spin. Conventions follow the papers:

Court bounce (Cross 2020, Sports Engineering 23:9), deformation via normal-force offset D:
  - sliding regime (eq 9-10):  v_x2/v_x1 = 1 - mu (1+e_y) tan(th1)
  - gripping regime (eq 12-14) in terms of e_x (~0.1) and D
  - regime test: slides throughout iff resulting v_x2 > R w2
  - hard-court representative params: e_y = 0.95 - 0.005 th1_deg, mu = 0.73,
    e_x = 0.1, D = 3e-4 * v1

Racket impact (Cross 2005; Ido's working eqs 5-7, racket frame, tangential/normal):
  v_t2 = a v_t1 + b R w1 ;  w2 = c w1 + d v_t1 / R ;  v_n2 = e_n v_n1
  a=0.648 b=0.3 c=0.4 d=0.583, e_n=0.42 (apparent, hand-held racket).

Per-surface (Tennis Industry chart): COR (=e_y scale) and COF (=mu):
  grass .60/.60, hard .83/.70, clay .85/.80.
Note: the chart's COR is a bulk speed ratio; we keep Cross's angle-dependent e_y form and
scale it per surface relative to hard court.

Used inversely by the CV pipeline: classify trajectory junctions (bounce vs racket hit vs
noise) and bound feasible outgoing states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Must match physics.flight.default_params.  Deriving ALPHA from the selected
# inertia avoids a second, inconsistent rotational calibration.
M_BALL = 0.057
R_BALL = 0.0325       # m
J_BALL = 3.155e-5     # kg m^2
ALPHA = J_BALL / (M_BALL * R_BALL**2)

# racket-impact constants (Cross 2005 fits)
RA, RB, RC, RD = 0.648, 0.3, 0.4, 0.583
E_N_RACKET = 0.42

SURFACES = {
    #            COR   COF(mu)
    "hard":  (0.83, 0.70),
    "clay":  (0.85, 0.80),
    "grass": (0.60, 0.60),
}
HARD_COR = 0.83
E_X_COURT = 0.10


@dataclass
class BounceResult:
    vx2: float      # horizontal speed after bounce (incidence plane, m/s, >=0 forward)
    vy2: float      # vertical speed after bounce (upward, m/s)
    w2: float       # topspin angular velocity after bounce (rad/s)
    regime: str     # 'slide' or 'grip'


@dataclass
class BounceVectorResult:
    velocity: np.ndarray
    spin: np.ndarray
    regime: str
    forward_regime: str
    lateral_regime: str


def court_ey(theta1_deg: float, surface: str = "hard") -> float:
    """Angle-dependent normal COR, Cross 2020 hard-court form scaled per surface."""
    base = 0.95 - 0.005 * theta1_deg
    scale = SURFACES[surface][0] / HARD_COR
    return max(0.1, min(0.95, base * scale))


def court_bounce(vx1: float, vy1: float, w1: float, surface: str = "hard",
                 e_x: float = E_X_COURT, *, regime_override: str | None = None,
                 coefficients: tuple[float, float] | None = None) -> BounceResult:
    """Bounce off the court (Cross 2020). Inputs: incoming horizontal speed vx1 (>0,
    direction of travel), downward speed vy1 (>0), topspin w1 (rad/s, + = topspin).

    An explicit regime override extends one formula for research optimization;
    it is NOT a physically selected bounce. Candidates must be checked again
    with the default selector before use. Default arithmetic is unchanged.

    ``coefficients`` substitutes an explicit ``(e_y, mu)`` pair for the per-surface chart
    lookup, leaving every Cross equation as written. It is how a region- and day-aware
    surface (``physics.surface_model``) states the local court at one landing point without
    a second copy of the impact algebra. Omitting it is the unchanged default.
    """
    if regime_override not in {None, "slide", "grip"}:
        raise ValueError("explicit slide/grip surrogate or default selector required")
    if vx1 <= 0 or vy1 <= 0:
        raise ValueError("vx1, vy1 must be positive (incidence-plane convention)")
    v1 = math.hypot(vx1, vy1)
    th1 = math.degrees(math.atan2(vy1, vx1))
    if coefficients is None:
        e_y = court_ey(th1, surface)
        mu = SURFACES[surface][1]
    else:
        e_y, mu = (float(value) for value in coefficients)
        if not (0.0 < e_y <= 1.0) or not (0.0 <= mu <= 2.0):
            raise ValueError("explicit court coefficients must be a physical (e_y, mu) pair")
    D = 3e-4 * v1
    tan1 = vy1 / vx1

    # sliding solution (eq 9, 10)
    vx2_slide = vx1 * (1 - mu * (1 + e_y) * tan1)
    w2_slide = w1 + (1 + e_y) * (mu - D / R_BALL) * vy1 / (ALPHA * R_BALL)

    if regime_override == "slide" or (
        regime_override is None and vx2_slide > R_BALL * w2_slide
    ):
        return BounceResult(vx2_slide, e_y * vy1, w2_slide, "slide")

    # gripping solution (eq 12, 14) in terms of e_x
    S1 = R_BALL * w1 / vx1
    vx2 = vx1 * ((1 - ALPHA * e_x) / (1 + ALPHA)
                 + ALPHA * (1 + e_x) * S1 / (1 + ALPHA)
                 - (1 + e_y) * D * tan1 / ((1 + ALPHA) * R_BALL))
    Rw2 = ((1 + e_x) / (1 + ALPHA)) * vx1 \
        - (D * (1 + e_y) / (R_BALL * (1 + ALPHA))) * vy1 \
        + ((ALPHA - e_x) / (1 + ALPHA)) * R_BALL * w1
    return BounceResult(vx2, e_y * vy1, Rw2 / R_BALL, "grip")


def court_bounce_vector(
    velocity: np.ndarray,
    spin: np.ndarray,
    surface: str = "hard",
    e_x: float = E_X_COURT,
) -> BounceVectorResult:
    """Three-axis court impact with an isotropic tangential friction budget.

    Normal restitution retains the angle/surface calibration used by the Cross model.
    A single Coulomb budget acts on the complete two-axis contact-slip vector, avoiding
    order-dependent forward/lateral updates. Spin about the court normal is unchanged by
    ideal point contact; horizontal-axis spin can exchange angular and linear momentum
    in both tangent axes.
    """
    velocity = np.asarray(velocity, float)
    spin = np.asarray(spin, float)
    if velocity.shape != (3,) or spin.shape != (3,):
        raise ValueError("velocity and spin must be three-vectors")
    horizontal = velocity[:2]
    horizontal_speed = float(np.linalg.norm(horizontal))
    downward_speed = -float(velocity[2])
    if horizontal_speed <= 0 or downward_speed <= 0:
        raise ValueError("court impact requires horizontal travel and downward velocity")

    forward = np.array([horizontal[0], horizontal[1], 0.0]) / horizontal_speed
    lateral = np.array([-forward[1], forward[0], 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    topspin = float(np.dot(spin, lateral))
    roll_spin = float(np.dot(spin, forward))
    normal_spin = float(np.dot(spin, normal))

    # Unified impulse form of the calibrated Cross model, continuous in roll spin.
    # Contact slip s = (v_f - R*w_top, v_l + R*w_roll) with v_l = 0 by frame choice.
    # The deformation offset D shifts the normal-force line forward: it brakes v_f
    # and torques w_top in BOTH regimes, exactly as in the scalar equations. Regime
    # selection follows the scalar criterion generalized to 2D: slide iff the
    # residual slip after a full-slide impulse has not crossed zero.
    theta1 = math.degrees(math.atan2(downward_speed, horizontal_speed))
    e_y = court_ey(theta1, surface)
    mu = SURFACES[surface][1]
    speed_in = math.hypot(horizontal_speed, downward_speed)
    D = 3e-4 * speed_in
    normal_impulse_per_mass = (1.0 + e_y) * downward_speed
    deformation_brake = normal_impulse_per_mass * D / R_BALL

    slip = np.array([horizontal_speed - R_BALL * topspin, R_BALL * roll_spin])
    slip_norm = float(np.linalg.norm(slip))

    def apply(j_f, j_l):
        """Impulse update. The D torque (about the lateral axis only — the offset
        lies along forward) enters the topspin update in BOTH regimes, exactly as
        in the scalar equations (eqs 10 and 14)."""
        forward_speed_out = horizontal_speed + j_f
        lateral_speed_out = j_l
        topspin_out = topspin + (-j_f - deformation_brake) / (ALPHA * R_BALL)
        roll_spin_out = roll_spin + j_l / (ALPHA * R_BALL)
        return forward_speed_out, lateral_speed_out, topspin_out, roll_spin_out

    # full-slide impulse: Coulomb friction mu*N opposing the slip direction
    slide_dir = slip / slip_norm if slip_norm > 1e-12 else np.zeros(2)
    j_slide = -mu * normal_impulse_per_mass * slide_dir
    v_f_s, v_l_s, w_top_s, w_roll_s = apply(float(j_slide[0]), float(j_slide[1]))
    residual_slip = np.array([v_f_s - R_BALL * w_top_s, v_l_s + R_BALL * w_roll_s])

    if slip_norm > 1e-12 and float(np.dot(residual_slip, slide_dir)) > 0.0:
        # sliding persists through the impact — the scalar criterion
        # (vx2_slide > R*w2_slide) generalized to 2D slip
        forward_speed_out, lateral_speed_out = v_f_s, v_l_s
        topspin_out, roll_spin_out = w_top_s, w_roll_s
        regime = "slide"
    else:
        # grip: zero residual slip with tangential restitution e_x; the forward
        # component carries the additional D brake of scalar eq 12
        j_grip_f = (-ALPHA * (1.0 + e_x) / (1.0 + ALPHA) * float(slip[0])
                    - normal_impulse_per_mass * D / ((1.0 + ALPHA) * R_BALL))
        j_grip_l = -ALPHA * (1.0 + e_x) / (1.0 + ALPHA) * float(slip[1])
        forward_speed_out, lateral_speed_out, topspin_out, roll_spin_out = apply(
            j_grip_f, j_grip_l)
        regime = "grip"

    velocity_out = (
        forward_speed_out * forward
        + lateral_speed_out * lateral
        + e_y * downward_speed * normal
    )
    spin_out = (
        roll_spin_out * forward
        + topspin_out * lateral
        + normal_spin * normal
    )
    return BounceVectorResult(
        velocity=velocity_out,
        spin=spin_out,
        regime=regime,
        forward_regime=regime,
        lateral_regime=regime,
    )


@dataclass
class RacketResult:
    vt2: float     # outgoing tangential speed (racket frame)
    vn2: float     # outgoing normal speed (racket frame)
    w2: float      # outgoing spin (rad/s)


def racket_impact(vt1: float, vn1: float, w1: float) -> RacketResult:
    """Racket-frame impact (Cross 2005 / Ido eqs 5-7). Incoming normal speed vn1 > 0
    toward the strings; tangential vt1 along the face; spin w1."""
    vt2 = RA * vt1 + RB * R_BALL * w1
    w2 = RC * w1 + RD * vt1 / R_BALL
    vn2 = E_N_RACKET * vn1
    return RacketResult(vt2, vn2, w2)


def racket_hit_feasible(v_in, v_out, max_racket_speed: float = 45.0,
                        tol: float = 0.35) -> bool:
    """Inverse feasibility: could a racket hit turn court-frame velocity v_in (2-3 vector)
    into v_out, for SOME racket velocity |V_R| <= max_racket_speed and face orientation?

    Conservative envelope (not exact inversion): energy and speed bounds derived from the
    impact model with |V_R| bounded. Outgoing court-frame speed can't exceed
    e_n-weighted closing speed + racket speed contribution; require also a real momentum
    change (noise junctions have tiny delta-v).
    """
    import numpy as np
    vi = np.asarray(v_in, float)
    vo = np.asarray(v_out, float)
    dv = float(np.linalg.norm(vo - vi))
    if dv < 2.0:                       # no meaningful impulse -> not a hit
        return False
    # max |v_out|: head-on with racket at max speed (racket frame in: |vi|+VR;
    # out: e_n*(|vi|+VR); court frame: + VR) with tolerance for tangential effects
    vmax = E_N_RACKET * (float(np.linalg.norm(vi)) + max_racket_speed) \
        + max_racket_speed
    return float(np.linalg.norm(vo)) <= vmax * (1 + tol)


# ------------------------------- self-tests ----------------------------------------------

def _selftests() -> None:
    # Racket, Cross Fig. f case (a): ball 15 m/s head-on into racket moving 20 m/s,
    # topspin 400 rad/s, face normal to travel: racket frame vn1=35, vt1=0.
    r = racket_impact(vt1=0.0, vn1=35.0, w1=400.0)
    v_out_court = r.vn2 + 20.0            # back to court frame along the normal
    assert abs(v_out_court - 34.9) < 0.5, v_out_court
    assert abs(r.w2 - 160.0) < 1.0, r.w2   # 0.4 * 400
    # Court bounce, Cross 2020 simplification: w1=0, D=0, e_x=0 -> vx2/vx1 = 1/(1+alpha)
    # = 0.645. Emulate D=0 by tiny v1 (D ~ v1) and e_x=0, grip regime forced via w1=0
    # steep incidence.
    b = court_bounce(vx1=1.0, vy1=2.5, w1=0.0, surface="hard", e_x=0.0)
    if b.regime == "grip":
        assert abs(b.vx2 / 1.0 - (1 / (1 + ALPHA))) < 0.08, b
    # Topspin raises rebound spin (Fig 8c qualitative)
    b0 = court_bounce(20.0, 8.0, 100.0, "clay")
    b1 = court_bounce(20.0, 8.0, 400.0, "clay")
    assert b1.w2 > b0.w2
    # Grass bounces lower than clay (per-surface e_y scaling)
    g = court_bounce(20.0, 8.0, 200.0, "grass")
    c = court_bounce(20.0, 8.0, 200.0, "clay")
    assert g.vy2 < c.vy2
    print("impact.py self-tests OK")


if __name__ == "__main__":
    _selftests()
