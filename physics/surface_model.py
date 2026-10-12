"""A region- and day-aware court surface, in a handful of parameters with priors.

``physics.bounce_reference`` stores one vertical restitution and one horizontal retention per
surface.  The owner's diagnosis (2026-09-08) is that this is wrong in a way that matters:
grass wears by tournament day and by landing region, and a ball that hits a painted line
skids.  ``docs/brainstorm/surface_bounce_literature.md`` collects the measurements behind that
and proposes the model implemented here.  It is deliberately small:

    e_y      = e_base(theta)  * w * (1 + a k(x, y))
    friction = mu_base_ratio  * w * (1 + a g k(x, y)) * (mu_line / mu_field  if on a line)
    retention = 1 - (1 - retention_base(theta)) * friction

Three quantities vary per broadcast and everything else is frozen:

``w``   a pace/wear scalar on both restitution and friction, prior ``N(1, 0.08)``.  It absorbs
        moisture, ball choice and the fortnight-average condition of the court.
``a``   the amplitude of a *fixed* spatial kernel ``k``, prior ``N(0.15, 0.08)`` on grass and
        ``N(0, 0.03)`` on clay and hard.  One number, not four region intercepts: the labeled
        broadcasts cannot identify four.
``line`` a switch, not a map.  Inside :data:`LINE_SWITCH_M` of a painted line the field
        friction is replaced by that surface's line friction.

Why a *modifier* over the existing base and not a fresh coefficient table
-------------------------------------------------------------------------
Clay and hard already have a measured base: the Hawk-Eye corpus rows in
``physics.bounce_reference``.  Replacing them with the literature's ``(e_y, mu)`` pair would
move a fitter that already works on 75 of the 89 labeled attempts, for no measurement reason.
So the base pair stays whatever the surface already has, and this module supplies only the
*ratios*: at ``w = 1``, ``a = 0`` and off a line the model is the identity, exactly.

Grass has no corpus base.  The literature arm uses its fresh-grass pair (``e_y(16 deg) = 0.72``,
``mu = 0.55``) through the Cross sliding law, with the worn pair (``0.85`` / ``0.70``) fixing
the kernel shape at its peak.  The regional-v2 arm instead uses ``hard_reference`` so its prior
is exactly today's explicit hard substitution.  The Tennis Industry chart's grass COR of 0.60
is a bulk speed ratio, not a normal restitution; treating it as one caused the earlier
execution-confounded loss recorded in ``EXPERIMENTS.md``.

Friction acts on the retention *deficit* because in the sliding regime -- where most in-play
court impacts stay, per Allen, Haake & Goodwill 2010 -- Cross's own equation 9 reads
``1 - v_x2/v_x1 = mu (1 + e_y) tan(theta)``: the lost fraction of horizontal speed is directly
proportional to ``mu``.  Scaling that deficit is therefore the same statement as scaling
``mu``, and it composes with a base retention that was measured rather than derived.

Nothing here is fitted inside this module.  ``w`` is estimated per broadcast by
``cv.experiments.connected_shooting.surface_wear_fit``; ``a`` and the line coefficients are
priors from the literature note until a corpus or a label set can identify them.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------------------
# Court geometry, pipeline frame: x across (0 .. 10.97), y along (0 .. 23.77), net at 11.885.

COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = COURT_LENGTH_M / 2.0
SERVICE_LINE_OFFSET_M = 6.40
SINGLES_INSET_M = 1.37
CENTRE_X_M = COURT_WIDTH_M / 2.0

# Kernel cell edges, from the literature note.
BASELINE_STRIP_M = 2.5
NEAR_NET_M = 3.0
# The note's line switch width.  Distances below are to a line's centre; a 5 cm line plus a
# 6.7 cm ball already puts the contact patch on the paint at roughly this separation.
LINE_SWITCH_M = 0.08

# The fixed spatial kernel k(x, y): where a modern baseline game wears a grass court.
REGION_KERNEL = {"baseline": 1.0, "backcourt": 0.4, "service_box": 0.0, "near_net": -0.2}

WEAR_PRIOR_MEAN = 1.0
WEAR_PRIOR_SD = 0.08
# A broadcast cannot be twice as fast as its own surface class; the estimator stays inside a
# stated box and reports when it reaches it.
WEAR_BOUNDS = (0.80, 1.25)


@dataclass(frozen=True)
class SurfaceLiterature:
    """Everything frozen about one surface class."""

    fresh_restitution_16deg: float
    fresh_friction: float
    worn_restitution_16deg: float
    worn_friction: float
    line_friction: float
    line_friction_sd: float
    line_restitution_factor: float
    amplitude_prior_mean: float
    amplitude_prior_sd: float
    base: str

    @property
    def friction_coupling(self) -> float:
        """How much faster friction wears than restitution, at the kernel peak.

        ``e_y`` gains ``(worn/fresh - 1)`` at ``k = 1`` when ``a`` is at the amplitude that
        reaches the worn pair; friction gains its own ratio over the same step.  One amplitude
        therefore drives both, with this fixed ratio between them.
        """
        gain_restitution = self.worn_restitution_16deg / self.fresh_restitution_16deg - 1.0
        gain_friction = self.worn_friction / self.fresh_friction - 1.0
        if abs(gain_restitution) < 1e-9:
            return 1.0
        return gain_friction / gain_restitution


LITERATURE: dict[str, SurfaceLiterature] = {
    # Cross 2003/2010 playing-speed grass, fresh to worn; the note's section 2.  The base is
    # the literature pair itself because the Hawk-Eye corpus has no grass at all.
    "grass": SurfaceLiterature(
        fresh_restitution_16deg=0.72,
        fresh_friction=0.55,
        worn_restitution_16deg=0.85,
        worn_friction=0.70,
        line_friction=0.55,
        line_friction_sd=0.10,
        # Wimbledon's titanium-dioxide transfer lines have no published playing-speed COR.
        line_restitution_factor=1.0,
        amplitude_prior_mean=0.15,
        amplitude_prior_sd=0.08,
        base="literature_cross_fresh_grass",
    ),
    # Hard and clay keep their measured Hawk-Eye base; the literature pair below only fixes
    # the friction ratios the switch and the kernel multiply it by.
    "hard": SurfaceLiterature(
        fresh_restitution_16deg=0.89,
        fresh_friction=0.70,
        worn_restitution_16deg=0.89,
        worn_friction=0.70,
        line_friction=0.55,
        line_friction_sd=0.08,
        # No playing-speed acrylic paint measurement was found; a colour coat is smoother than
        # the sand-filled texture coat, so the clay-tape rise is halved and declared.
        line_restitution_factor=1.02,
        amplitude_prior_mean=0.0,
        amplitude_prior_sd=0.03,
        base="hawkeye_corpus",
    ),
    "clay": SurfaceLiterature(
        fresh_restitution_16deg=0.92,
        fresh_friction=0.80,
        worn_restitution_16deg=0.92,
        worn_friction=0.80,
        line_friction=0.50,
        line_friction_sd=0.08,
        # Appl. Sci. 14:5674 (2024): clay 0.7265 against white tape 0.7617, a 4.9% rise.
        line_restitution_factor=1.049,
        amplitude_prior_mean=0.0,
        amplitude_prior_sd=0.03,
        base="hawkeye_corpus",
    ),
}


# ---------------------------------------------------------------------------------------
# The measured alternative to the literature kernel.
#
# ``cv.validation.hawkeye_surface_regions`` measures the Hawk-Eye bounce as a function of
# landing region, line proximity and round, controlling for incidence angle and incoming
# speed.  Over 21,810 impacts (11,135 clay, 10,675 hard) the landing region moves the bounce by
# far more than the surface class does, and it moves it the *other way* from the literature's
# grass-wear kernel: the baseline strip is the LOW-restitution, HIGH-friction cell, not the
# high one.  These offsets are that regression, additive on the base pair and re-centred so a
# sample-weighted random bounce sees zero shift -- which is what keeps the existing measured
# base calibration intact.
#
# ``day_*`` is the per-round slope.  A round is a proxy for tournament day: round r of a slam
# is played around day 2r-1.  It is confounded with court assignment, because later rounds are
# on show courts.
MEASURED_REGION_MAP: dict[str, dict[str, Any]] = {
    "clay": {
        "n": 11135,
        "restitution": {
            "backcourt": 0.0114,
            "baseline": -0.0730,
            "service_box": 0.0245,
            "near_net": -0.0113,
        },
        "restitution_se": {
            "backcourt": 0.0,
            "baseline": 0.0028,
            "service_box": 0.0023,
            "near_net": 0.0142,
        },
        "retention": {
            "backcourt": -0.0043,
            "baseline": -0.1461,
            "service_box": 0.0961,
            "near_net": -0.0794,
        },
        "retention_se": {
            "backcourt": 0.0,
            "baseline": 0.0041,
            "service_box": 0.0034,
            "near_net": 0.0208,
        },
        # Historical execution-confounded estimates: these landmarks had no timestamps, so
        # retention was inferred at assumed spin.  Retained to reproduce that older arm, never
        # interpreted as evidence that a painted line has no effect.
        "line_restitution": 0.0053,
        "line_restitution_se": 0.0052,
        "line_retention": 0.0074,
        "line_retention_se": 0.0075,
        "on_line_n": 444,
        # The day proxy shares the same timing/retention confound and is not physical evidence.
        "day_restitution_per_round": 0.0006,
        "day_restitution_per_round_se": 0.0009,
        "day_retention_per_round": 0.0023,
        "day_retention_per_round_se": 0.0014,
        "reference_round": 1.0,
        "source": "hawkeye_surface_regions, Roland Garros 2019-2021, 21,810-impact run",
    },
    "hard": {
        "n": 10675,
        "restitution": {
            "backcourt": 0.0105,
            "baseline": -0.0724,
            "service_box": 0.0311,
            "near_net": -0.0420,
        },
        "restitution_se": {
            "backcourt": 0.0,
            "baseline": 0.0026,
            "service_box": 0.0023,
            "near_net": 0.0125,
        },
        "retention": {
            "backcourt": -0.0059,
            "baseline": -0.1504,
            "service_box": 0.1143,
            "near_net": -0.0867,
        },
        "retention_se": {
            "backcourt": 0.0,
            "baseline": 0.0038,
            "service_box": 0.0033,
            "near_net": 0.0180,
        },
        "line_restitution": -0.0078,
        "line_restitution_se": 0.0046,
        "line_retention": -0.0133,
        "line_retention_se": 0.0066,
        "on_line_n": 482,
        "day_restitution_per_round": -0.0001,
        "day_restitution_per_round_se": 0.0007,
        "day_retention_per_round": 0.0007,
        "day_retention_per_round_se": 0.0011,
        "reference_round": 1.0,
        "source": "hawkeye_surface_regions, Australian Open 2020-2021, 21,810-impact run",
    },
    # Grass is absent from the corpus.  This sample-weighted pooled shape is retained only to
    # reproduce the older transferred-map arm; regional-v2 does not use it.  The source lacked
    # timestamps and therefore cannot establish a region, line or day null.
    "grass": {
        "n": 0,
        "restitution": {
            "backcourt": 0.0110,
            "baseline": -0.0727,
            "service_box": 0.0277,
            "near_net": -0.0263,
        },
        "restitution_se": {
            "backcourt": 0.0,
            "baseline": 0.0019,
            "service_box": 0.0016,
            "near_net": 0.0094,
        },
        "retention": {
            "backcourt": -0.0051,
            "baseline": -0.1482,
            "service_box": 0.1050,
            "near_net": -0.0830,
        },
        "retention_se": {
            "backcourt": 0.0,
            "baseline": 0.0028,
            "service_box": 0.0024,
            "near_net": 0.0138,
        },
        "line_restitution": 0.0,
        "line_restitution_se": 0.0070,
        "line_retention": 0.0,
        "line_retention_se": 0.0100,
        "on_line_n": 0,
        "day_restitution_per_round": 0.0,
        "day_restitution_per_round_se": 0.0,
        "day_retention_per_round": 0.0,
        "day_retention_per_round_se": 0.0,
        "reference_round": 1.0,
        "source": "transferred pooled clay+hard shape; the corpus has no grass",
    },
}


def surfaces() -> tuple[str, ...]:
    return tuple(sorted(LITERATURE))


def landing_region(x: float, y: float) -> str:
    """Which of the four fixed kernel cells a landing point is in."""
    depth = min(float(y), COURT_LENGTH_M - float(y))
    if depth <= BASELINE_STRIP_M:
        return "baseline"
    if depth <= COURT_LENGTH_M / 2.0 - SERVICE_LINE_OFFSET_M:
        return "backcourt"
    if abs(float(y) - NET_Y_M) >= NEAR_NET_M:
        return "service_box"
    return "near_net"


def region_kernel(x: float, y: float) -> float:
    """``k(x, y)``: the fixed wear shape, +1 on the baseline strip, -0.2 at the net."""
    return REGION_KERNEL[landing_region(x, y)]


def line_distance_m(x: float, y: float) -> tuple[float, str]:
    """Distance to the nearest painted line centre, and which line it is.

    Lines are finite segments: the centre service line stops at the service lines and the
    service lines stop at the singles sidelines, so a ball in the backcourt is not "on the
    centre service line" merely because it is near the middle of the court.
    """
    x, y = float(x), float(y)

    def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        dx, dy = second[0] - first[0], second[1] - first[1]
        fraction = ((x - first[0]) * dx + (y - first[1]) * dy) / (dx * dx + dy * dy)
        fraction = min(max(fraction, 0.0), 1.0)
        nearest_x, nearest_y = first[0] + fraction * dx, first[1] + fraction * dy
        return math.hypot(x - nearest_x, y - nearest_y)

    service_near = NET_Y_M - SERVICE_LINE_OFFSET_M
    service_far = NET_Y_M + SERVICE_LINE_OFFSET_M
    singles_right = COURT_WIDTH_M - SINGLES_INSET_M
    segments = (
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0), "baseline"),
        ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M), "baseline"),
        ((0.0, 0.0), (0.0, COURT_LENGTH_M), "doubles_sideline"),
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M), "doubles_sideline"),
        ((SINGLES_INSET_M, 0.0), (SINGLES_INSET_M, COURT_LENGTH_M), "singles_sideline"),
        ((singles_right, 0.0), (singles_right, COURT_LENGTH_M), "singles_sideline"),
        ((SINGLES_INSET_M, service_near), (singles_right, service_near), "service_line"),
        ((SINGLES_INSET_M, service_far), (singles_right, service_far), "service_line"),
        ((CENTRE_X_M, service_near), (CENTRE_X_M, service_far), "centre_service_line"),
    )
    return min((distance(first, second), name) for first, second, name in segments)


def on_line(x: float, y: float) -> bool:
    return line_distance_m(x, y)[0] <= LINE_SWITCH_M


@dataclass(frozen=True)
class SurfaceState:
    """The surface as it is at one landing point, in modifier form."""

    surface: str
    restitution_factor: float
    friction_factor: float
    region: str
    kernel: float
    on_line: bool
    line_distance_m: float


@dataclass(frozen=True)
class SurfaceModel:
    """One broadcast's surface: a class, a wear scalar and a kernel amplitude.

    ``base_source`` names which pair the modifiers are applied to.  ``reference`` is the
    surface's row in :mod:`physics.bounce_reference` -- the Hawk-Eye corpus on clay and hard,
    and on grass the earlier execution-confounded broadcast row, so an unqualified surface name
    keeps doing exactly what it does today.  ``literature`` is the fresh pair of
    :func:`fresh_base`. ``hard_reference`` is the regional-v2 grass prior: the measured
    hard row, exactly matching today's explicit hard substitution when modifiers are inert.
    """

    surface: str
    wear: float = WEAR_PRIOR_MEAN
    amplitude: float | None = None
    line_switch: bool = True
    base_source: str = "reference"
    #: ``literature`` is the fixed kernel of the note; ``measured`` is the Hawk-Eye
    #: regression in :data:`MEASURED_REGION_MAP`, which has the opposite sign on
    #: restitution and is the only one of the two with a corpus behind it.
    region_map: str = "literature"
    #: Round index of this broadcast, a proxy for tournament day.  ``None`` holds the day
    #: term at the corpus reference round instead of inventing one.
    day: float | None = None
    provenance: str = "prior"

    def __post_init__(self) -> None:
        if self.surface not in LITERATURE:
            raise ValueError(
                f"no surface literature for {self.surface!r}; known surfaces are {surfaces()}"
            )
        if not math.isfinite(self.wear) or not WEAR_BOUNDS[0] <= self.wear <= WEAR_BOUNDS[1]:
            raise ValueError(f"wear scalar must be finite and inside {WEAR_BOUNDS}")
        if self.amplitude is None:
            object.__setattr__(
                self,
                "amplitude",
                0.0
                if self.base_source == "hard_reference"
                else LITERATURE[self.surface].amplitude_prior_mean,
            )
        if not math.isfinite(float(self.amplitude)) or not -0.5 <= float(self.amplitude) <= 0.8:
            raise ValueError("kernel amplitude must be finite and inside [-0.5, 0.8]")
        if self.base_source not in {"reference", "literature", "hard_reference"}:
            raise ValueError("base_source is 'reference', 'literature' or 'hard_reference'")
        if self.base_source == "hard_reference" and self.surface != "grass":
            raise ValueError("the hard-reference prior is only defined for grass")
        if self.region_map not in {"literature", "measured"}:
            raise ValueError("region_map is 'literature' or 'measured'")
        if self.day is not None and not 1.0 <= float(self.day) <= 8.0:
            raise ValueError("a round index is between 1 and 8")

    @property
    def literature(self) -> SurfaceLiterature:
        return LITERATURE[self.surface]

    @property
    def inert(self) -> bool:
        """True when the model is the identity everywhere, lines included.

        An inert model is what a bare surface name resolves to, so the existing fitter is
        arithmetically untouched until a spec asks for the region-aware court.
        """
        return (
            self.base_source in {"reference", "hard_reference"}
            and self.region_map == "literature"
            and self.wear == 1.0
            and float(self.amplitude) == 0.0
            and not self.line_switch
            and self.day is None
        )

    def state(self, x: float, y: float) -> SurfaceState:
        """The restitution and friction multipliers at one landing point."""
        literature = self.literature
        kernel = region_kernel(x, y)
        amplitude = float(self.amplitude)
        restitution = self.wear * (1.0 + amplitude * kernel)
        friction = self.wear * (1.0 + amplitude * literature.friction_coupling * kernel)
        distance, _ = line_distance_m(x, y)
        struck_line = self.line_switch and distance <= LINE_SWITCH_M
        if struck_line and self.region_map == "literature":
            # A switch, not a scaling: the paint is a different material, so the local
            # friction becomes the line's own and the wear kernel no longer applies to it.
            # The factor is expressed against the base pair's field friction because that is
            # what the base retention deficit already carries.
            base_friction = (
                LITERATURE["hard"].fresh_friction
                if self.base_source == "hard_reference"
                else literature.fresh_friction
            )
            friction = literature.line_friction / base_friction
            restitution *= literature.line_restitution_factor
        return SurfaceState(
            surface=self.surface,
            restitution_factor=float(restitution),
            friction_factor=float(friction),
            region=landing_region(x, y),
            kernel=kernel,
            on_line=bool(struck_line),
            line_distance_m=float(distance),
        )

    def coefficients(
        self, x: float, y: float, base_restitution: float, base_retention: float
    ) -> tuple[float, float, SurfaceState]:
        """Apply the model to one impact's base coefficients.

        ``base_restitution`` and ``base_retention`` are whatever the surface's own reference
        supplies at this incidence angle.  The friction factor multiplies the retention
        *deficit*, which is the sliding law's ``mu (1 + e_y) tan(theta)``.
        """
        state = self.state(x, y)
        if self.region_map == "measured":
            measured = MEASURED_REGION_MAP[self.surface]
            day = 0.0 if self.day is None else float(self.day) - measured["reference_round"]
            restitution = (
                float(base_restitution)
                + measured["restitution"][state.region]
                + measured["day_restitution_per_round"] * day
                + (measured["line_restitution"] if state.on_line else 0.0)
            )
            retention = (
                float(base_retention)
                + measured["retention"][state.region]
                + measured["day_retention_per_round"] * day
                + (measured["line_retention"] if state.on_line else 0.0)
            )
            # The wear scalar still multiplies, so a broadcast may still be fast or slow.
            restitution *= self.wear
            retention = 1.0 - (1.0 - retention) * self.wear
        else:
            restitution = float(base_restitution) * state.restitution_factor
            deficit = (1.0 - float(base_retention)) * state.friction_factor
            retention = 1.0 - deficit
        return (
            float(min(max(restitution, 0.05), 1.0)),
            float(min(max(retention, 0.05), 1.0)),
            state,
        )

    def field_friction(self) -> float:
        """The friction the base pair's retention deficit stands for."""
        if self.base_source == "hard_reference":
            return LITERATURE["hard"].fresh_friction
        return self.literature.fresh_friction

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "base_source": self.base_source,
            "region_map": self.region_map,
            "day": None if self.day is None else float(self.day),
            "wear": float(self.wear),
            "amplitude": float(self.amplitude),
            "line_switch": bool(self.line_switch),
            "provenance": self.provenance,
            "base": (
                LITERATURE["hard"].base
                if self.base_source == "hard_reference"
                else self.literature.base
            ),
            "friction_coupling": float(self.literature.friction_coupling),
        }


def fresh_base(surface: str, theta1_deg: float) -> tuple[float, float]:
    """The literature base pair for a surface with no corpus row, at one incidence angle.

    Restitution carries Cross's own angle law (``-0.005`` per degree) about 16 degrees, and
    retention is the sliding law at the fresh friction.  Used for grass, which the Hawk-Eye
    corpus does not contain.
    """
    literature = LITERATURE[surface]
    theta = float(theta1_deg)
    restitution = literature.fresh_restitution_16deg - 0.005 * (theta - 16.0)
    restitution = min(max(restitution, 0.05), 1.0)
    retention = 1.0 - literature.fresh_friction * (1.0 + restitution) * math.tan(
        math.radians(min(max(theta, 1.0), 80.0))
    )
    return restitution, min(max(retention, 0.05), 1.0)


# ---------------------------------------------------------------------------------------
# Selection: a surface name that may carry a fitted broadcast model.

_SPEC = re.compile(r"^(?P<surface>[a-z]+)(?:@(?P<terms>[a-z0-9_.=,+-]+))?$")

#: Arm selector.  A JSON object mapping a bare surface name to a full spec, e.g.
#: ``{"grass": "grass@w=1.06,a=0.15,base=fresh"}``.  The connected fitter names its surface
#: with a bare string that reaches this module through several processes it does not own, so
#: the *arm* -- which court model that name stands for -- is selected here, once, from the
#: environment of the process the sweep launched.  It is explicit configuration: the sweep
#: writes the exact value into its own frozen configuration and its report.
#:
#: Unset, every bare surface name resolves to the fully inert model and nothing changes.
OVERRIDE_VARIABLE = "TENNIS_SURFACE_MODEL"

_OVERRIDE_CACHE: tuple[str | None, dict[str, str]] = (None, {})


def overrides() -> dict[str, str]:
    """The selected arm, read from the environment and cached on its exact text."""
    global _OVERRIDE_CACHE
    raw = os.environ.get(OVERRIDE_VARIABLE)
    if _OVERRIDE_CACHE[0] == raw:
        return _OVERRIDE_CACHE[1]
    if not raw:
        parsed: dict[str, str] = {}
    else:
        parsed = {str(key): str(value) for key, value in json.loads(raw).items()}
        for key, value in parsed.items():
            model = parse(value, resolve_override=False)
            if model.surface != key:
                raise ValueError(
                    f"the {OVERRIDE_VARIABLE} entry for {key!r} names surface {model.surface!r}"
                )
    _OVERRIDE_CACHE = (raw, parsed)
    return parsed


def parse(spec: str | SurfaceModel, *, resolve_override: bool = True) -> SurfaceModel:
    """Read a surface spec.

    ``"clay"`` is the surface with a fully inert model -- byte-identical to not having one,
    line switch included, so an unqualified surface name is the fitter as it stands.
    ``"grass@w=1.06,a=0.15"`` is the same surface with a fitted broadcast wear scalar and
    kernel amplitude, and switches the line skid on.  ``noline`` switches it back off.
    """
    if isinstance(spec, SurfaceModel):
        return spec
    text = str(spec).strip().lower()
    if resolve_override and "@" not in text:
        selected = overrides().get(text)
        if selected is not None:
            return parse(selected, resolve_override=False)
    found = _SPEC.match(text)
    if found is None:
        raise ValueError(f"unreadable surface spec {spec!r}")
    surface = found.group("surface")
    terms = found.group("terms")
    if not terms:
        return SurfaceModel(
            surface=surface, wear=1.0, amplitude=0.0, line_switch=False, provenance="inert"
        )
    wear, amplitude, line_switch = 1.0, LITERATURE[surface].amplitude_prior_mean, True
    base_source, region_map, day = "reference", "literature", None
    for term in terms.split(","):
        if term == "noline":
            line_switch = False
            continue
        if "=" not in term:
            raise ValueError(f"unreadable surface term {term!r}")
        name, value = term.split("=", 1)
        if name == "w":
            wear = float(value)
        elif name == "a":
            amplitude = float(value)
        elif name == "base":
            base_source = {
                "fresh": "literature",
                "literature": "literature",
                "reference": "reference",
                "hard": "hard_reference",
            }[value]
        elif name == "region":
            region_map = {"measured": "measured", "literature": "literature"}[value]
        elif name == "r":
            day = float(value)
        else:
            raise ValueError(f"unknown surface term {name!r}")
    return SurfaceModel(
        surface=surface,
        wear=wear,
        amplitude=amplitude,
        line_switch=line_switch,
        base_source=base_source,
        region_map=region_map,
        day=day,
        provenance="spec",
    )


def format_spec(model: SurfaceModel) -> str:
    """The inverse of :func:`parse`, so a fitted model can travel as one CLI argument."""
    if model.inert:
        return model.surface
    terms = [f"w={model.wear:.4f}", f"a={float(model.amplitude):.4f}"]
    if model.base_source == "literature":
        terms.append("base=fresh")
    elif model.base_source == "hard_reference":
        terms.append("base=hard")
    if model.region_map == "measured":
        terms.append("region=measured")
    if model.day is not None:
        terms.append(f"r={model.day:g}")
    if not model.line_switch:
        terms.append("noline")
    return f"{model.surface}@{','.join(terms)}"


def load_broadcast_models(path: str) -> dict[str, SurfaceModel]:
    """Read a fitted per-broadcast wear artifact into models, keyed by broadcast."""
    payload = json.loads(open(path).read())
    output = {}
    for key, row in payload["broadcasts"].items():
        output[key] = SurfaceModel(
            surface=row["surface"],
            wear=float(row["wear"]),
            amplitude=float(row["amplitude"]),
            base_source=row.get("base_source", "reference"),
            region_map=row.get("region_map", "literature"),
            day=row.get("day"),
            provenance=row.get("provenance", "fitted"),
        )
    return output
