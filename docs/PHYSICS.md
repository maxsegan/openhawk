# Physics

This page states what the production fitter models, what it takes from Rod Cross's work, what it
changes, and what it leaves out. Every claim points at code in this repository. Detailed
derivations and equation-by-equation references are in [physics/REFERENCE.md](../physics/REFERENCE.md).
The papers and the Tennis Industry court chart it cites are not redistributed here.

## References

1. R. Cross, *Calculations of groundstroke trajectories in tennis*, Sports Engineering 23:9 (2020).
2. R. Cross, *Effects of friction between the ball and strings in tennis*, Sports Engineering 3
   (2000), Appendix (Stepanek drag/lift correlations).
3. R. Cross and C. Lindsey, *Measurements of drag and lift on tennis balls in flight* (2014).
4. I. Yavetz, notes on R. Cross's racket return velocity and spin model (2005).
5. Tennis Industry magazine court COR/COF chart.
6. Hawk-Eye CourtVision data (Roland Garros 2019-2021, Australian Open 2020-2021), public
   ball-trajectory landmarks (hit, peak, net, bounce).

## Flight (3D vector form)

`physics/flight.py` integrates gravity, quadratic drag and Magnus lift on the full vectors:

- ball 57 g, radius 32.5 mm, inertia 3.155e-5 kg m²; air 1.205 kg/m³;
- drag coefficient 0.55 (Cross 2020), lift coefficient 0.6·S with spin ratio S = Rω/v
  (Cross 2020), Magnus force along ω × v;
- a slow exponential spin decay (0.025).

Cross's groundstroke model is planar: it lives in the vertical plane of flight with a single
topspin/backspin scalar. Here **spin is a free 3-vector per flight**, parameterised as
(top, side, rifle) relative to the launch velocity (`cv/pipeline/physics_events.py`), each bounded
at ±600 rad/s. Magnus acts on the full vector, so sidespin curves the ball laterally in the air.
The Stepanek and Cross-Lindsey drag/lift curves are implemented as a reference profile; on 280 real
development flights they did not beat the constant coefficients, which remain the default.

## Court bounce (measured law)

Production bounces use `physics.bounce_reference.court_bounce`:

- **Vertical restitution and horizontal retention** are linear in incidence angle (centred at
  16°, the median groundstroke incidence) and were fitted per surface to consecutive strike pairs
  of the Hawk-Eye corpus: both flights around each impact are fitted with `physics.flight` on the
  production aerodynamic constants (`cv/validation/hawkeye_bounce_model.py`).
- The corpus has no timestamps, so restitution is identified only jointly with an assumed
  outgoing spin. The law is recorded at the Cross-predicted outgoing spin and at zero outgoing
  spin; the truth lies between, and that band is part of the stated uncertainty.
- **Outgoing spin** is not measurable from the landmarks; it comes from Cross's slide/grip
  solution (`physics/impact.py`, Cross 2020 eqs. 5-14). Contact duration is Cross's 4-5 ms.
- **Horizontal heading is preserved.** The corpus shows a median lateral deflection of a
  fraction of a degree, so the deterministic law scales the incoming horizontal velocity without
  turning it. Spin about the court normal passes through unchanged. In production, sidespin
  therefore bends the flight but does not turn the bounce.
- A 3D bounce with one isotropic Coulomb friction budget on the full two-axis contact slip
  (`physics.impact.court_bounce_vector`, and `physics/passive_bounce.py`), in which sidespin does
  change the rebound direction, is implemented and tested but is **not** on the production path
  (`passive_bounce_response: off` in `cv/pipeline/product_s6_policy.json`). The one production
  place where a rebound may turn is the first ground impact after a net hit
  (`net_ground_horizontal: heading`).

What the corpus measurement changes relative to Cross's model and the court chart
(full-corpus fit; 2,616 impacts, clay 1,322 and hard 1,294):

- restitution falls with incidence angle at about −0.0069 per degree (Cross: about −0.005);
- horizontal retention falls with incidence angle (0.697 → 0.590 on hard over the observed
  range), where Cross's model is roughly flat;
- the chart's clay/hard friction difference is absent at a typical bounce (retention 0.673 vs
  0.672);
- with spin restored, vertical restitution is 0.89-0.92 against Cross's 0.87-0.89.

**Hold-out.** The production law (`hawkeye_holdout`) was refitted without any Hawk-Eye match that
is also one of the broadcast evaluation matches, including in the incoming-spin prior, so that a later
Hawk-Eye comparison on those matches is not circular: 2,427 impacts from 192 matches. It differs
from the full-corpus law by less than one match-bootstrap standard deviation (restitution at 16°:
clay 0.916 → 0.913, hard 0.894 → 0.893). The fitted and excluded match ids are in
`physics/bounce_law_hawkeye_holdout.json`; every run records the law in its provenance
(`court_bounce_law`). `TENNIS_BOUNCE_LAW=full_corpus` selects the full-corpus law.

**Grass** is absent from the corpus. The grass row is a thin calibration from 11 labelled grass
impacts, shrunk toward the chart prior; treat it as directional.

## Whole-point fitting

S6 (`cv/experiments/connected_shooting/`, driven by `cv/pipeline/s6_*`) fits one physical ball
per rally from monocular video:

- one initial state per point and a free velocity and spin change at each racket contact; racket
  impacts are **not** modelled physically (Cross's racket model in `physics.impact.racket_impact`
  is implemented and tested but not used in fitting);
- contacts and bounces (times from the event stage, positions constrained by the camera rays and
  the court plane) are the anchors between which the flight equations run; the net is a physical
  obstacle and a possible contact, but its height alone does not fix depth;
- the objective is pixel reprojection of the 2D ball track with robust losses, plus physically
  motivated priors (player reach at contacts, serve toss relative to the server);
- flight and point gates decide acceptance; a flight that fails is not emitted, and a point can
  abstain entirely.

Uncertain automatic events are adjudicated by the same physics: each candidate is inserted, the
point is refitted, and the candidate survives only if the refit passes the gates.

## Caveats

- On development matches, fitted sidespin is large (median about 1,000 rpm) and the rifle-spin
  component sits at its bound on nearly half of accepted flights. Rifle spin in real tennis is
  small, so this component is likely absorbing misfit. Fitted spin is a fitting parameter, not a
  measurement, until a constrained refit (rifle fixed at zero) is run.
- A net-cord "admissible set" response (`cv/pipeline/net_cord_response.py`) exists but is not
  promoted; the default net response is a hand-set tape model.
- Not modelled: racket impact inside the fitter, calibrated net-cord physics, identifiable
  outgoing spin, validated grass restitution, line skid.
