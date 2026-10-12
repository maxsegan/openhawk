# Evaluation

## What is scored

The reported yield counts **flights**. A labelled flight is one ball flight whose origin is an
independently labelled competitive racket contact; its endpoint is the next contact or a
confirmed ending. The rule in [evaluation/origin_matcher.py](../evaluation/origin_matcher.py):

- the start time of an accepted model flight must lie within the labelled origin interval
  widened by **two native frame periods** (40 ms at 50 fps, 80 ms at 25 fps);
- matching is **one-to-one**: a labelled origin with two eligible flights, or a flight eligible
  for two origins, matches nothing;
- abstentions, held attempts, refused broadcasts and failed fits keep every labelled flight in
  the denominator;
- references are built from labels only, never from model output.

`origin_matcher.py` is an exact extraction of the two functions the private evaluation harness
calls (checked identical on 2,000 randomized attempts). The layers around it are not released:
building references from labels, digest-pinned input bindings, aggregation over fitting
components, and the classification of unmatched accepted flights into post-point flights
(emitted but tagged `after_point_end`), origin offsets (a correct flight whose start is offset
from the label) and wrong accepts.

This measures flight **recovery** at the right time, not 3D accuracy; see
[3D accuracy](#3d-accuracy) for what has been measured about the 3D itself.

## Arms and panels

- **AA**: automatic ball track and automatic events (the product). **AL**: automatic ball track
  with labelled events, a ceiling for the event stage.
- **Panels** are sets of broadcasts with labelled points. A **sealed** panel is measured once on a
  frozen commit, with no tuning and no reruns after results are seen; afterwards it counts as
  development data. The pipeline was developed on other, repeatedly inspected broadcasts.

## Production measures (unlabelled broadcasts)

Production broadcasts have no labels, so the production tracker reports output, not accuracy:

- **Flight slots**: the flights an attempt's event sequence implies (one per struck ball).
- **Yield** of a tier = (share of serve attempts with at least one flight of that tier) ×
  (that tier's flights / flight slots of those attempts). **Validated** counts confirmed flights
  only; **extended** adds the best-effort tier.
- **Complete point**: every flight from the serve to the end of the point is in ONE connected
  fit. Separate pieces covering a point do not count. Shares are over serve attempts.
- **Pose coverage**: share of player-frames (one frame, one side) in active play with a player
  box that have at least one fitted 3D joint.

Wrong accepts in production are estimated on **fresh panels**: matches drawn from the production
queue before any of their output was read, processed with the change on and off, and judged
blind by two VLM judges on rendered audits of each sampled flight (3D path over the broadcast
and in 3D). Fresh panels are not sealed: they are measured once per change, but the judging is
by models, the samples are small and the panel then becomes development data.

## 3D accuracy

**Synthetic recovery.** 40 simulated points rendered through exact synthetic cameras, each with
6 noise draws (240 points, 744 flights). Idealised inputs: exact contact and bounce frames, exact
cameras, a physics and bounce model that match the simulator, known shutter and image axes,
and ball observations that are projected ball centres plus 2 px Gaussian noise; the fit starts
cold from the observations only. Tolerance (fixed before the run): bounce, path RMS and ending
within 0.3048 m (1 ft), contact and path peak within 0.9144 m (3 ft), every required anchor
present and no gap. Result: 240/240 points and 744/744 flights within tolerance; path RMS median
3.2 cm, p90 9.3 cm, max 30.2 cm; per-flight peak error median 5.3 cm, p90 16.2 cm. This checks
the fitter, not the detectors: real video adds event, ball and camera errors.

**Bounce law hold-out.** The production bounce law is fitted on 2,427 Hawk-Eye impacts (clay
1,171, hard 1,256) from 192 matches of the public Hawk-Eye corpus, after removing all 51 matches
that are also broadcast evaluation matches (and recomputing the spin prior without them).
Against the fit on all 2,616 impacts, every coefficient moves by at most 0.5 match-bootstrap
standard deviations (largest: the clay restitution intercept). See [PHYSICS.md](PHYSICS.md).

**Against Hawk-Eye on 25 matches.** The pipeline was run on 25 broadcast matches that are also
in the Hawk-Eye corpus (9 of them inspected during development, 16 never inspected), and the
fitted 3D of validated flights was compared with Hawk-Eye landmarks matched in time
(Hawk-Eye is used only for scoring):

| quantity | n | error |
|---|---:|---|
| bounce position | 5,711 | median 0.19 m, p90 0.60 m |
| net-crossing height | 5,828 | median absolute 8.0 cm (signed median +3.5 cm) |
| hit (flight start) height | 5,455 | median signed +11.0 cm, median absolute 14.5 cm, p90 46 cm |

These used the earlier per-point cameras. Production now shares one camera centre per match; on
a 5-match A/B (drawn from the same 25 matches, so development data) that lowered the net-height
error from 9.8 to 6.3 cm and the hit-height bias from +11.9 to +7.9 cm, with bounce error
0.20 to 0.19 m.

**Not yet run:** the full point-by-point comparison against Hawk-Eye on all 51 reserved
matches (the matches excluded from the bounce-law fit).

## Labels

Development labels were produced by frontier vision-language models used as labellers under a
fixed procedure (GPT-6 Astra via Codex for most panels, Claude Opus 5.5 for sealed panel F).
On panel C the two labellers agreed on 97% of flight origins.
Labels are used for training and calibration (event and point models, the grass bounce row) and
for scoring; inference never reads them.

## Available on request (not in this repository)

- per-panel label sets (event times and types, flight origins) and the reference-building code;
- the full panel scoring harness and the per-panel score files behind the README table;
- the development event-model training set definition.
