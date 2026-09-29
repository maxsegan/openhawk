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

This measures flight **recovery** at the right time, not 3D accuracy. An independent 3D check
against Hawk-Eye on the 51 broadcast evaluation matches that also appear in the public
Hawk-Eye corpus is planned; the production bounce law was fitted without those matches.

## Arms and panels

- **AA**: automatic ball track and automatic events (the product). **AL**: automatic ball track
  with labelled events, a ceiling for the event stage.
- **Panels** are sets of broadcasts with labelled points. A **sealed** panel is measured once on a
  frozen commit, with no tuning and no reruns after results are seen; afterwards it counts as
  development data. The pipeline was developed on other, repeatedly inspected broadcasts.

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
