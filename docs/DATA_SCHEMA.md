# Output and data schema

## Court frame

All 3D positions are metres in the court frame of `cv/pipeline/court.py`:

- x across the court (0 to 10.97 m between the doubles sidelines),
- y along the court (0 to 23.77 m between the baselines; the net is at y = 11.885 m),
- z up from the court surface,
- origin at the near court's left doubles corner as seen from the broadcast camera.

Positions outside those ranges are valid (behind the baseline, wide of the sideline, in the air).
Times are native picture indices of the attempt clip (`*_frame`, fractional for sub-frame times);
the pipeline keeps the source's native timestamps alongside and never re-times a broadcast.

## MATCH_RESULT.json (`product_match_result_v1`)

Written by `cv/pipeline/product_runner.py` (`step_assemble`), one per broadcast.

| field | meaning |
|---|---|
| `schema` | `product_match_result_v1` |
| `match_id` | match key |
| `source_video` | file record of the input video: name, sha256, bytes |
| `code` | git commit of the code that ran, and whether the tree was dirty |
| `observation_origin` | always `automatic` |
| `evaluation_labels_read` | always `false`: no label reaches inference |
| `court_bounce_law` | the bounce law used (name, fitted/excluded Hawk-Eye matches, per-step record) |
| `counts` | attempts in the point ledger, fitted, held or failed, attempts with flights, accepted flights, flights tagged after the point ended, event-adjudication mode and actions |
| `cost` | dollars spent on paid model calls (0 by default), per model, number of calls |
| `time` | wall, CPU and GPU seconds, total and per step |
| `yield` | `null`: no reference flights exist for an unlabelled broadcast, so none is claimed |
| `event_beam`, `audio_delay_calibration` | the event beam's switch and counts, and the per-match audio delay it calibrated |
| `flight_validation` | post-validation switch and counts of failing flights by reason |
| `fit_only_bounce_rules` | rules version, switch, and how many flights were cut or demoted by each rule |
| `post_fit_checks` | checks version, switch, the checks that demote, and pass/fail/abstain tallies |
| `best_effort` | best-effort tier switch, flight slots per tier (`validated`, `fitted_unvalidated`, `anchored`, `none`) and both yields |
| `player_identity` | summary of `PLAYER_IDENTITY.json` (below): status, names per scoreboard row, confidence |
| `attempts` | one row per serve attempt / rally (below) |

### `attempts[]`

| field | meaning |
|---|---|
| `key`, `clip` | attempt id (`MATCH__ptNNNN`) and clip name |
| `ledger` | the automatic point ledger row: `attempt_role` (`first_serve` for the first attempt of a point, `continuation` for later attempts such as a second serve after a fault), `point_index`, rally start/end times in seconds of the broadcast (`rally_t_start`, `rally_t_end`), audio `impact_count`, `outcome_hint`, and the **scoreboard reading** when available (`set_number`, `games_1/2`, `points_1/2`, `sets_won_1/2`, `server`, `score_source`, `score_confidence`; empty when abstained) |
| `first_fit_status`, `held_reason` | `completed`, or why the attempt was held or failed before/at fitting |
| `cascade_edited`, `cascade_refit_status`, `final_source` | whether event adjudication edited the attempt and whether the result is the first fit or the refit |
| `result` | file record of the full S6 result for this attempt (large; internal format) |
| `accepted_flights` | the accepted 3D flights (below); rejected flights are not emitted |
| `point_end_call` | automatic point-end call if made: frame, evidence (e.g. `out_bounce:court_reading_out`), court x,y of the deciding bounce, distance out, margin |
| `best_effort_flights` | one row per original flight slot: `tier` (`validated`, `fitted_unvalidated`, `anchored`, `none`), the reason, and for best-effort rows the path and its support against the observed ball (`support`: median / p90 reprojection px); best-effort paths are fitted or interpolated states, never validated |
| `player_identity` | per attempt: who is near and far, who serves, how it was decided (`direct`, `propagated`, `unresolved`) and a confidence |
| `event_beam_contacts` | contacts the event beam added, with their witnesses |

### `accepted_flights[]`

| field | meaning |
|---|---|
| `role` | `serve`, `return`, `interior_contact_flight`, `mid_rally`, `final`, `unknown_origin_flight` (the first flight of an attempt is named `serve` even when the attempt starts mid-rally) |
| `start_frame`, `end_frame` | fractional native clip frames of the flight's start and end events |
| `original_flight_index` | index of the flight in the attempt's event sequence |
| `start_xyz_m`, `end_xyz_m` | fitted 3D start and end positions |
| `bounces_xyz_m` | fitted 3D bounce positions inside the flight |
| `net_hits` | number of net contacts inside the flight |
| `positions_m` | fitted 3D path, 240 Hz integration, every 4th sample kept (60 Hz) |
| `after_point_end`, `point_end_signal` | tagged, not removed, when the flight follows the automatic point-end call (e.g. a dead ball knocked back) |
| `confirmed` | `true` when the flight passed post-validation and, when they are on, the fit-only bounce rules and the post-fit checks: the **validated** tier. Unconfirmed flights stay listed with their reasons |
| `validation`, `post_fit_checks` | per-flight post-validation result (`ok`, `reasons`, `tags`) and post-fit check results |
| `hitter_side`, `hitter` | side (`near`/`far`) of the player who struck the flight, and their name when the identity is resolved |

Every 3D quantity is **fitted**, not observed: it is the state of the physical model that best
explains the 2D ball track, events and camera. Observed 2D positions and event times remain in
the per-attempt result. On one development match, 10 of 251 accepted flights lacked the dense 3D
fields (a key mismatch between the verdict and the dense measurement); consumers should test for
`start_xyz_m` before use.

## Player identity and 3D pose

`PLAYER_IDENTITY.json` (`player_names_v1`) records the match-level decision: which scoreboard
row is which player (from a match metadata table of winner, loser and score checked against the
decoded scoreboard), which row started near (end-change rules plus the observed serve sides),
the votes and disagreements, and per attempt the resulting names with `direct`, `propagated` or
`unresolved` status. Without a metadata table every attempt stays near/far.

`pose/player_pose_3d_v1.npz` holds one row per native frame and side (`near`, `far`) of every
exported attempt window: the 17 COCO 2D keypoints and confidences from the pose detector (model
observations, native pixels), the 17 **fitted** 3D joints in court metres, a per-joint status
(0 missing, 1 lifted from a confident keypoint, 2 lifted from a low-confidence keypoint,
3 interpolated across a short gap; nothing is extrapolated), the court root, a camera
reliability flag, the vertical scale and an active-play flag. Read it with
`cv.pipeline.player_pose_3d.load_pose`. `pose/player_pose_3d_v1.json` is the receipt (inputs
with sha256, model, settings, counts, timing). The 3D viewer's documents carry the same
skeletons as compact per-frame layers (`cv/pipeline/player_pose_portal.py`).

## Released dataset (proposed)

The public dataset will contain processed outputs only: per match the metadata (tour,
tournament, round, surface, players, score), per point the ledger row and scoreboard reading, the
automatic events with times, each validated flight's fitted 3D path, bounces and net hits
(best-effort flights clearly marked as such), the point-end call, player identities, the fitted
3D player skeletons, and provenance (code commit, bounce law, model identities). It will not
contain video, frames, or local file paths. The release format and its exporter are not part of
this repository yet; this page will be updated with the released schema.
