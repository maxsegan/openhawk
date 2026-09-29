# Tennis physics reference model

## Scope and source hierarchy

This note transcribes the four supplied references and identifies where the executable reference
model needs evidence outside them. Page numbers below are PDF pages. The court chart is a one-page
image embedded in the DOCX.

1. Rod Cross, *Calculations of groundstroke trajectories in tennis*, Sports Engineering 23:9
   (2020), `Groundstrokes.pdf`.
2. Ido Yavetz, *How to calculate return velocity and spin according to R. Cross (Oct. 2005),
   “Bounce of a Spinning Ball Near Normal Incidence”*, `tennis physics.pdf`.
3. Ido Yavetz, *Variable tension and further corrections due to impact speed and string type*
   (2022), `Variable tension and further corrections due to impact speed and string type.pdf`.
4. Tennis Industry court COR/COF chart,
   `TTVR COR_COF Court chart from Tennis Industry Mag.docx`.

The supplied groundstroke paper does **not** contain a saturated lift law, a Reynolds-dependent drag
fit, or a spin-decay differential equation. It deliberately calculates its figures with constant
drag, linear lift, and constant spin. For the requested richer reference implementation this note
also uses:

- A. Stepanek's wind-tunnel correlations as reproduced in Rod Cross, *Effects of friction between
  the ball and strings in tennis*, Sports Engineering 3 (2000), Appendix, p. 97. These give the
  closed-form, saturating spin-ratio curves.
- Rod Cross and Crawford Lindsey, *Measurements of drag and lift on tennis balls in flight*, Sports
  Engineering 17 (2014), pp. 89–96, especially Figs. 7–8. This gives the free-flight drag result and
  the observed 2% spin loss over 6.4 m.
- S. R. Goodwill, S. B. Chin, and S. J. Haake, *Aerodynamics of spinning and non-spinning tennis
  balls*, JWEIA 92 (2004), pp. 935–958, especially Figs. 6–10. This establishes that wind-tunnel
  coefficients can depend on Reynolds number, spin ratio, and wear, but does not publish a complete
  analytic two-variable fit.

`physics.reference` exposes two profiles. `GROUNDSTROKE_PARAMS` exactly represents the assumptions
used for Cross's 2020 plotted examples. The default `stepanek` profile implements the requested
saturated empirical coefficients and measured flight spin decay.

## Flight equations

Let the ball have mass (m), radius (R), area (A=\pi R^2), air density (\rho), velocity
vector \(\mathbf v\), angular velocity \(\boldsymbol\omega\), and spin ratio

\[
S = \frac{R\lVert\boldsymbol\omega_\perp\rVert}{\lVert\mathbf v\rVert},\qquad
\boldsymbol\omega_\perp = \boldsymbol\omega -
(\boldsymbol\omega\cdot\hat{\mathbf v})\hat{\mathbf v}.
\]

The vector equations implemented by the reference model are

\[
\dot{\mathbf x}=\mathbf v,
\]

\[
\dot{\mathbf v}= -g\hat{\mathbf z}
-\frac{\rho A}{2m}C_D\lVert\mathbf v\rVert\mathbf v
+\frac{\rho A}{2m}C_L\lVert\mathbf v\rVert^2
\frac{\boldsymbol\omega\times\mathbf v}
{\lVert\boldsymbol\omega\times\mathbf v\rVert}.
\]

The planar components are Cross 2020 equations 3–4 (p. 2). The vector form preserves their sign:
topspin gives downward Magnus acceleration. Rifle spin is excluded from (S) and produces no
Magnus force because the cited measurements concern spin perpendicular to flight.

### Drag

| Source/profile | Equation | Dependence | Evidence and limitation |
|---|---|---|---|
| Cross 2020 | (C_D=0.55) | Constant | p. 2, equations 3–4 discussion. The paper says the rough cloth keeps flow turbulent and drag approximately speed-independent. |
| Cross–Lindsey 2014 free flight | mean (C_D=0.507\pm0.024), observed range 0.453–0.567 | No significant speed or spin trend over 14–30 m/s and roughly −2400 to +2500 rpm | Fig. 7. Shot-to-shot scatter is material and unresolved by a deterministic mean model. |
| Goodwill et al. 2004 wind tunnel | plotted (C_D(Re,S,\text{wear})) | Reynolds, spin ratio, and wear | Figs. 6–7, pp. 949–950. New-ball (C_D) rises about 0.65→0.69 with (S) at (Re=105,000); coefficients differ at (Re=210,000). No complete analytic surface is tabulated. |
| Stepanek/Cross reference default | (C_D=0.508+[22.503+4.196S^{-5/2}]^{-2/5}), with (C_D(0)=0.508) | Spin ratio; saturates at 0.7958 | Cross 2000 Appendix p. 97. It assumes Reynolds independence; use only as the explicit empirical reference, not proof that Reynolds effects are absent. |

The Reynolds number is (Re=\rho\lVert\mathbf v\rVert(2R)/\mu_{air}). At 30 and 55 m/s with
the reference constants it is about 132,000 and 243,000. `reference.reynolds_number` reports it,
but the default coefficient does not invent a (Re) interpolation that the sources do not specify.

### Lift/Magnus

| Source/profile | Equation | Saturation |
|---|---|---|
| Cross 2020 | (C_L=0.6S) | None; p. 2. |
| Stepanek/Cross reference default | (C_L=[2.022+0.981/S]^{-1}), (C_L(0)=0) | Yes; (C_L\to1/2.022=0.4946). Cross 2000 Appendix p. 97. |

Cross–Lindsey 2014 Fig. 7 found lift approximately linear in (S), but with nonzero low-spin
side force caused by cloth asymmetry. A deterministic spin-only model cannot reproduce that
knuckleball-like term. Goodwill et al. 2004 Figs. 8–9 show Reynolds and wear interactions,
particularly below (S=0.1); the figures do not uniquely define a continuous production law.

### Spin in flight

Cross 2020 assumes (\dot{\boldsymbol\omega}=0) (p. 2). Cross–Lindsey 2014 reports that spin fell
about 2% over the 6.4 m measurement path. The reference model encodes only that measured aggregate:

\[
\dot{\boldsymbol\omega}=-k_s\lVert\mathbf v\rVert\boldsymbol\omega,
\qquad k_s=-\ln(0.98)/6.4=0.003156673\ \mathrm{m}^{-1}.
\]

Thus spin is exponential in distance traveled and retains exactly 98% over 6.4 m. This is an
empirical closure, not a measured torque law across speed, spin, ball wear, or axis orientation.

### Constants

Cross 2020 uses (R=33) mm and (I=\alpha mR^2), with (\alpha=0.55) for a 33 mm outer radius
and 6 mm wall (p. 2). With (m=0.057) kg this gives (I=3.414015\times10^{-5}\) kg m². The
reference uses (\rho=1.21) kg/m³ and (g=9.81) m/s².

## Court bounce

Cross 2020 defines positive magnitudes (v_{x1},v_{y1}) before impact and (v_{x2},v_{y2}) after,
normal COR (e_y=v_{y2}/v_{y1}), moment (I=\alpha mR^2), sliding friction (\mu), deformation
offset (D), and tangential COR (e_x). Impulse and angular impulse are (p. 3, equations 5–8)

\[
\int Fdt=m(v_{x1}-v_{x2}),\quad
\int Ndt=mv_{y1}(1+e_y),
\]

\[
R\int Fdt-D\int Ndt=I(\omega_2-\omega_1).
\]

### Sliding throughout

For Coulomb sliding (F=\mu N), Cross equations 9–10 (p. 3) are

\[
\frac{v_{x2}}{v_{x1}}=1-\mu(1+e_y)\tan\theta_1,
\]

\[
\omega_2=\omega_1+(1+e_y)\left(\mu-\frac{D}{R}\right)
\frac{v_{y1}}{\alpha R}.
\]

The ball slid throughout if the predicted (v_{x2}>R\omega_2). Incoming spin does not alter
the sliding-law outgoing center speed, but it shifts the regime boundary and adds directly to
outgoing spin.

### Gripping/rolling during impact

Cross defines (p. 3, equation 11)

\[
e_x=\frac{R\omega_2-v_{x2}}{v_{x1}-R\omega_1}.
\]

(e_x=0) is rolling at separation; (e_x\approx0.1) is typical after grip. With
(S_1=R\omega_1/v_{x1}), equations 12 and 14 (p. 4) are

\[
\frac{v_{x2}}{v_{x1}}=
\frac{1-\alpha e_x}{1+\alpha}
+\frac{\alpha(1+e_x)}{1+\alpha}S_1
-\frac{(1+e_y)D\tan\theta_1}{(1+\alpha)R},
\]

\[
R\omega_2=
\frac{1+e_x}{1+\alpha}v_{x1}
-\frac{D(1+e_y)}{R(1+\alpha)}v_{y1}
+\frac{\alpha-e_x}{1+\alpha}R\omega_1.
\]

At (\omega_1=D=e_x=0), (v_{x2}/v_{x1}=1/(1+0.55)=0.645), the paper's explicit numeric
example (p. 4). The paper's plotted hard-court calculation uses
(e_y=0.95-0.005\theta_1) with degrees, (D=3\times10^{-4}v_1) m, (\mu=0.73), and
(e_x=0.1) (p. 4). It says (D) can vary by a factor of two even at fixed speed.

`reference.court_bounce` generalizes the tangential impulse isotropically to the two-dimensional
court plane and retains spin about the court normal. That 3D extension is not derived in the
planar paper; it is selected because it conserves the same impulse/angular-impulse relation and is
symmetric under coordinate reflection.

### Surfaces

| Court | chart COR (speed out / speed in) | bounce height / trajectory height | chart COF |
|---|---:|---:|---:|
| Grass | 0.60 | 36% | 0.60 |
| Hard | 0.83 | 69% | 0.70 |
| Clay | 0.85 | 72% | 0.80 |

The chart COR is a **bulk speed ratio**, while Cross's (e_y) is a normal velocity ratio. They
cannot be equated without contradiction. The reference follows the existing code's explicit
engineering mapping: keep Cross's hard-court angle law and scale it by chart COR / 0.83; use chart
COF directly. No cited source validates that mapping, so surface-specific normal COR is unresolved.

## Racket impact: Cross 2005 near-normal model

In the racket frame, Cross defines normal and tangential COR (Yavetz pp. 1–2, equations 1–4) and
conserves tangential angular momentum. The directly fitted 39-bounce coefficients give (Yavetz
p. 3 and p. 5, equations 5–7)

\[
v_{t2}=0.648v_{t1}+0.300R\omega_1,
\]

\[
\omega_2=0.400\omega_1+0.583\frac{v_{t1}}{R},
\]

\[
v_{n2}=0.420v_{n1}.
\]

The single-(e_t) approximation is (e_t=-0.11), producing
(a,b,c,d=(0.684,0.316,0.426,0.574)), but no one (e_t) exactly matches all four direct fits
(Yavetz p. 3). The reference uses the direct fits.

Yavetz p. 3, Fig. f gives four checks for an incoming 15 m/s ball with 400 rad/s spin and a
20 m/s racket: outgoing `(speed, angle, spin)` is `(34.9 m/s, 6.5°, 160 rad/s)` for normal racket
motion; `(31.8, 13.6°, −16.7)` when the racket moves 30° upward; `(34.8, 1.1°, 106)` with the face
tilted 5°; and `(30.4, 8.6°, −65.8)` with both changes. All four are executable tests.

The paper assumes incidence no more than 45° from the racket normal and coplanar racket normal,
motion, incoming path, and spin. Tangential COR depends on incidence direction; the source says
further theory is needed. Non-coplanar spin-axis reorientation and gyroscopic effects are expressly
unresolved (Yavetz pp. 6–7). The reference does not fabricate a general 3D racket collision.

## Tension, impact speed, and string bite note

For string tension (T) in newtons, Yavetz p. 1 proposes

\[
e_N=0.43-\frac{0.03}{56}(T-224),\qquad
e_T=\frac{60}{18667}T-\frac{250133}{373340},
\]

followed by the four scaled coefficient equations implemented in `racket_coefficients`. At
(T=242.667\) N the note's check is
((e_N,e_T,a,b,c,d)=(0.42,0.11,0.648,0.3,0.4,0.583)). The empirical tension range was only
50–63 lb; continuing below 50 lb versus clamping at 224 N are explicitly competing, unresolved
options. Tension above 75 lb is excluded by the note.

The proposed normal-speed correction (p. 2) is

\[
T_V=pV^2+qV+r,
\quad p=\frac{nT}{1080},\quad q=\frac{11nT}{540},\quad
r=\frac{(27-35n)T}{27},
\]

with speed clamped outside 10–40 m/s. For (T=55) lb and (n=0.1), it returns 55 lb at 28 m/s
and 60.5 lb at 40 m/s. Only (e_N) is recomputed from (T_V). Page 3 defines a speculative
0–1 string-bite slider that quadratically moves `(a,c)` toward zero and `(b,d)` toward one, with
the baseline at 0.5. The note repeatedly requests player testing; these are proposals, not validated
physical laws, so the reference exposes the tension/speed calculation but does not use it in Stage 6.

## Typical measured and modeled values

- Cross 2020 Fig. 6 (p. 5): a 30 m/s shot from 1 m that lands at 22.77 m needs 7.4° launch and
  lands at 15.2° without spin; at 4000 rpm it needs 15.2° and lands at 24.1°. The reference's
  exact Cross-2020 mode reproduces both within plot-reading tolerance.
- Cross 2020 Fig. 5 and text (p. 5): for the fixed 22.77 m depth, landing speed is about 65% of
  launch speed and landing angle is about 7–12° steeper than launch.
- Cross 2020 p. 8: the highest Australian Open groundstrokes are about 130 km/h (36 m/s).
  Lane et al. values summarized on p. 9 put court-incidence speeds at 15–40 m/s and angles at
  10–30°.
- Cross 2020 uses 0–4000 rpm as the modeled groundstroke span in Figs. 4–10. That is a modeling
  range, not a claim that every value is typical.
- Goodwill et al. 2004 p. 955, Fig. 10: the serve example is 50 m/s (113 mph), 4.25° downward,
  contact height 2.7 m, and 2000 rpm ((S=0.11)); it slows to 21 m/s by the baseline. A worn ball
  lands 0.5 m farther for the same launch, or passes the net 35 mm lower when angle is adjusted to
  equalize landing.
- Cross 2020 p. 7: an illustrative post-bounce state (v_x=17) m/s, (v_y=6) m/s travels 3 m in
  0.18 s and initially rises 1.08 m in that interval when gravity/drag are ignored for the mental
  estimate.

## Comparison with current code

| Physical item | Current implementation | Reference | Assessment |
|---|---|---|---|
| Force equations | `flight.py:167–181`, full vector drag and `omega × v` Magnus | Same vector force balance | Matches Cross planar equations and valid 3D direction. |
| Drag | `C_drag=0.55`, constant (`flight.py:14,171`) | Stepanek/Cross (C_D(S)), 0.508→0.7958 | Current matches Cross 2020's chosen simplification, not the saturated empirical curve or wind-tunnel Re/wear variation. |
| Lift | `C_lift=0.6` multiplying (R(\omega\times v)) (`flight.py:15,177–181`) | Saturating (C_L(S)) applied as force magnitude | Current is exactly (C_L=0.6S); it has no saturation. Rifle spin correctly produces zero force through the cross product, though the batch path enables lift based on total spin. |
| Spin decay | `C_spin_decay=0.025`; equivalent distance rate 0.0016733/m (`flight.py:186–216`) | 0.00315667/m from 2% per 6.4 m | Both are exponential in distance. Current retains 98.93% over 6.4 m; reference retains 98.00%. The source only supports an aggregate. |
| Ball geometry | (R=32.5) mm, (I=3.155\times10^{-5}), (\alpha=0.5240) (`flight.py:11–12`; `impact.py:37–40`) | (R=33) mm, (\alpha=0.55), (I=3.414015\times10^{-5}) | Current uses a project-measured set; Cross uses rounded construction values. This is a deliberate calibration difference, not a proven bug. |
| Court normal COR | Cross angle law scaled by chart bulk COR (`impact.py:73–77`) | Same explicit mapping | Matches, including the unresolved bulk-vs-normal assumption. |
| Court tangential law | Isotropic 3D slide/grip impulse (`impact.py:112–208`) | Same Cross equations/generalization | Matches equation-by-equation. Current chart hard COF 0.70 differs from Cross's illustrative 0.73 but correctly matches the supplied chart. |
| Racket impact | Direct Cross coefficients and (e_N=0.42) (`impact.py:218–224`) | Same, with 33 mm paper radius | Matches except radius. Current does not implement the speculative variable tension/speed/bite note. |

## Measured court bounce: `physics/bounce_reference.py`

`physics.bounce_reference` is a second court-bounce profile fitted to measured rebounds rather
than transcribed from Cross. Its source is the Hawk-Eye CourtVision corpus at
`data/external/ryurko-hawkeye` (Roland Garros clay and Australian Open hard,
2019-2021). `cv/validation/hawkeye_bounce_model.py` fits the incoming flight of a strike from
its `hit`/`peak`/`net`/`bounce` landmarks and the outgoing flight of the same impact from
`bounce`/`peak`/next `hit`, both with `physics/flight.py` on the production aero constants, and
reads the velocity change off the two solutions.

**What the corpus can measure.** Vertical restitution and horizontal speed retention, at 2,616
impacts. **What it cannot.** Outgoing spin: refitting the rebound with topspin fixed anywhere in
0-4000 rpm moves the landmark residual by a median of 0.28 mm, while the implied outgoing speed
moves 1.54 m/s. Restitution and retention are therefore only identified jointly with an assumed
rebound spin, and `MEASURED` publishes both ends of that band: `*_at_model_spin` (rebound spin
from Cross's own slide/grip solution) and `*_spin_free` (no rebound spin at all). Dwell time is
not measurable without timestamps; `DWELL_SECONDS = 0.0045` is taken from the 4-5 ms in the
Cross material above. Grass is absent from the corpus, and the module raises rather than
extrapolating the chart COR to it.

| quantity, at the corpus median incidence | clay | hard | Cross (`physics/impact.py`) |
|---|---:|---:|---|
| restitution, rebound spin from Cross | 0.916 | 0.894 | 0.887 / 0.868 |
| restitution, spin-free rebound | 0.746 | 0.740 | same |
| restitution slope, per degree of incidence | -0.0069 | -0.0068 | -0.0051 / -0.0050 |
| horizontal retention, rebound spin from Cross | 0.677 | 0.668 | 0.679 / 0.673 |
| horizontal retention, spin-free rebound | 0.560 | 0.561 | same |
| retention slope, per degree of incidence | -0.0070 | -0.0088 | approximately flat |
| impacts | 1,322 | 1,294 | |

**Where the shipped equations are wrong.**

1. **The angle law is too shallow.** Cross's `e_y = 0.95 - 0.005*theta1` scaled by chart COR
   falls about 0.005 per degree; the measurement falls about 0.0069. Cross therefore
   under-predicts restitution on shallow balls (12-16 degrees: 0.914 measured against 0.878
   predicted on hard) and over-predicts it on steep ones (above 20 degrees: 0.814 against
   0.829).
2. **Horizontal retention has a real angle dependence that Cross does not reproduce.** The
   Cross prediction is nearly flat across incidence; the measurement falls from 0.697 at 12-16
   degrees to 0.590 above 20 degrees on hard. On steep balls Cross over-predicts forward speed
   retention by about 0.09.
3. **The chart's surface friction split does not appear in the data.** The court chart gives
   clay COF 0.80 against hard 0.70, a 14% difference, and clay COR 0.85 against hard 0.83.
   Measured horizontal retention is 0.673 on clay and 0.672 on hard, and measured restitution
   differs by 1.2%. Whatever separates these two surfaces in play, it is not the friction
   coefficient the chart supplies.
4. **The previously reported 0.15 restitution deficit against Cross is an artefact.** The v3
   corpus study (`data/processed/external/hawkeye_flight_fit_v3.json`, experiment C) measured
   `e_y` about 0.15 below Cross by reading the rebound off the post-bounce apex with no
   outgoing spin. A gripped bounce leaves the ball near rolling, at roughly 4,000 rpm. Restore
   that spin and the same impacts measure 0.89-0.92 against Cross's 0.87-0.89. The equations
   were not wrong there; the spin-free rebound assumption was.
5. **Direction preservation holds.** The isotropic 3D extension in `reference.court_bounce`
   assumes the rebound keeps the incoming horizontal heading. Measured lateral deflection has a
   median of -0.3 degrees and a 95th percentile of 3.1-3.8 degrees, so a deterministic model is
   right not to invent a systematic sideways term.

**Limits.** Incoming spin is measured only for the last strike of a point, so rally impacts use
the surface-median measured topspin (2,184 rpm clay, 1,870 rpm hard) as a fixed prior; the v3
study established that per-shot spin is not recoverable from these landmarks. The two
populations disagree: rally impacts, whose outgoing arc is pinned by the next hit, measure
`e_y` 0.875, while last-strike impacts, whose outgoing arc rests on the apex alone, measure
0.928. The 2019-2021 corpus is two tournaments, not a survey of surfaces.

The numerical and held-out comparisons are recorded in `WK3_REPORT.md` and in
`$TENNIS_DATA_ROOT/processed/wk3_physics/`.
