"""Synthetic WHOLE-POINT closed loop for the Stage-6 3D stage.

Evaluation-only.  This module builds complete synthetic tennis points -- every flight of a
rally joined at shared contacts, with the measured court bounce applied at every impact --
projects them through the cohort's own per-frame cameras and frame rates, writes exactly the
artifacts the real chain hands the fitter, runs the unchanged default 3D command through its
published CLI contract, and scores COMPLETE POINTS against the synthetic truth.

The three parts are separate commands so a long run can be resumed:

``generate``
    Build the truth and materialise one audit root per noise rung.
``run``
    Invoke ``cv.pipeline.reconstruct_3d`` on one audit root with one fitter arm.
``audit``
    Score a finished report against the truth: attempted / solved / accepted flights,
    truth-good flights, wrong accepts, and complete points, plus a first-cause attribution
    for every incomplete point using the ``docs/wk1/solve_failures.md`` taxonomy.

BENCHMARK BOUNDARIES AND LIMITATIONS

    A contact-to-contact flight ends at the next shot.  The rally-closing (TERMINAL) flight runs
    from the last contact to a declared boundary:

    * the SECOND bounce, when the first bounce is IN  (``second_bounce``);
    * the FIRST bounce, when that bounce is OUT       (``terminal_bounce``);
    Leaving the field of view is an observation boundary, never a physical ending. V5
    retains the simulated first/second ground impact even when no picture sees it.

    Historical v1-v4 out-of-view boundaries remain readable for frozen comparisons.
    Generation excludes same-half first bounces and intervening out bounces; their correct
    fault/error handling is not implemented here. A benchmark sample is not a full rules engine.

    A wall is the same rule where a wall exists; this bench has no wall geometry, so it
    generates the court-exit form of that ending and says so.  A serve fault into the net ends
    at the net; no net-cord flight is generated here (``reconstruction.py`` deletes any fit whose
    flight holds a ``net_hit``), so that form is not exercised.

    So a terminal flight whose first bounce is IN legitimately holds TWO bounces: the first is an
    ordinary bounce knot fitted with the measured bounce model (``physics/bounce_reference.py``)
    and the SECOND is the termination anchor, on the court plane.

    Everything after the termination -- the roll, the dead ball, the pickup -- is not part of any
    flight.  It stays in the GENERATED track and in the emitted events, because the real chain
    hands it to the fitter and the fitter has to reject it; it is excluded from truth-good
    scoring and from the complete-point definition, and whether a fitter also fitted it is a
    diagnostic column (``fitted_dead_ball``), never a penalty.

    The generator emits a ``point_end`` row at the termination frame, in the runner's event
    format. These are synthetic oracle event inputs, not evidence that the runtime event model
    can confirm the same ending from pixels. No synthetic pass rate establishes real-video yield.

Nothing here edits or imports a private fitter entry point: the reconstruction is run as a
subprocess through the documented command in ``docs/wk1/nightly.md``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.pipeline import contact_frame_refiner  # noqa: E402
from cv.pipeline import camera_cal  # noqa: E402
from cv.pipeline import resolution as res  # noqa: E402
from cv.pipeline.artifact_cache import (  # noqa: E402
    stage_identity,
    stage_receipt_matches,
    write_stage_receipt,
)
from cv.pipeline.reconstruction import PointCamera  # noqa: E402
from cv.pipeline.provenance import file_record, git_record  # noqa: E402
from cv.validation import hawkeye_flight_fit as hff  # noqa: E402
from physics import bounce_reference, flight  # noqa: E402

SEED = 20260903
COHORT_RELATIVE = Path("processed/wk3_nightly/cohort_root_v3")
NOISE_RELATIVE = Path("processed/wk3_s6bench/full/noise_models.json")
CORPUS_ROOT = Path("data/external/ryurko-hawkeye")

TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"
CAMERA_NAME = "camera_P_per_frame_v1.npz"
EVENT_NAME = "event_emissions.json"

COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = 11.885
BALL_RADIUS_M = 0.0325
AERO_PARAMS = dict(flight.default_params)

RUNGS = ("clean", "track_error", "event_timing", "missing_anchors", "realistic", "realistic_v2")

# Bounces inside a terminal (rally-closing) flight on real data, measured in
# docs/wk1/solve_failures.md over the 39 lost terminal flights: 2 spans hold no owner bounce,
# 27 hold one and 10 hold two.  The last shot always bounces here, so the sampled quantity is
# the number of *extra* dead-ball bounces after it: zero for the 29 one-or-fewer spans, one
# for the 10 two-bounce spans.
TERMINAL_BOUNCE_MIX = ((0, 29), (1, 10))
TERMINAL_MIN_SECONDS = 0.55

# --------------------------------------------------------------------------------------- #
# What "right" means.  Two criteria, both computed on every run; `metric_v2` is the default.
# --------------------------------------------------------------------------------------- #
# `pixel_v1` is the criterion the bench shipped with: 10 cm at every scored bounce, 12 px at
# every racket contact, and nothing else.  `docs/wk1/dev_loop.md` measured what it cannot see:
# a rally-closing flight with no bounce witness sat a median of 7.9 m from its own contact in
# the court frame while reprojecting to 1.6 px, and passed.  Both of that subset's "complete"
# points were complete only because of it.  So `pixel_v1` is kept for comparison and is no
# longer the default.
CRITERION_PIXEL_V1 = "pixel_v1"
CRITERION_METRIC_V2 = "metric_v2"
CRITERIA = (CRITERION_METRIC_V2, CRITERION_PIXEL_V1)
DEFAULT_CRITERION = CRITERION_METRIC_V2

# Historical strict thresholds retained for comparability. They are NOT the current
# owner's roughly foot-scale target. A separate versioned criterion must express that target.

# 1x the anchor.  A bounce is the one place the ball's 3D position is directly observable --
# it is on the court plane -- so this historical threshold has no slack. Unchanged from v1.
BOUNCE_TOLERANCE_M = 0.10

# The pixel witness at a racket contact, unchanged from v1.  docs/wk1/gate_audit.md measured
# 10 cm as 3 px far court / 5 px near court, so 12 px is already about 25-40 cm of slack ACROSS
# the ray; it says nothing at all along it, which is why CONTACT_TOLERANCE_M sits beside it.
CONTACT_PIXEL_TOLERANCE_PX = 12.0

# 2.5x the historical anchor. A contact is a fitted shot location, and it
# is fitted, not observed, and it is shared by two flights, so each end carries the junction
# error too.  2.5x is the smallest multiple that does not make the criterion a junction test.
CONTACT_TOLERANCE_M = 0.25

# 2.5x the anchor.  A junction is one contact seen twice, so it gets the contact number: the
# two fitted flights must agree about where the ball was to the same tolerance the contact
# itself is held to.
JUNCTION_TOLERANCE_M = 0.25

# 5x the anchor.  The termination is the far end of the longest unwitnessed stretch in the
# point (the rally-closing flight has no next contact to pin it), so it carries the most
# accumulated error of any scored place.  Half a metre is about a ball's width of court-line
# call: past it, "where the point ended" is a different answer.
TERMINATION_TOLERANCE_M = 0.50

# 3x the anchor, as a root-mean-square over the whole flight rather than at one place.  The
# historical 10 cm reference is per-shot; a curve that is 30 cm RMS away from the truth is still
# recognisably the same flight, and one that is further is not.  This is the condition that
# `pixel_v1` had no analogue for: it is what stops a fit riding the right image ray metres away
# in depth from counting as right.
TRAJECTORY_RMS_TOLERANCE_M = 0.30

# An `out_of_view` ending has no 3D anchor, so it is scored where the truth last had one: the
# last frame the ball is still inside the image.  The fit must be within TERMINATION_TOLERANCE_M
# there AND still be describing a flight -- above the court and moving forward, not a fitted
# span that has run out.  A ball that leaves a broadcast frame is metres up and travelling; a
# fit that is on the ground there, or falling vertically, is not the same event even if it is
# close in metres.
AIRBORNE_MIN_HEIGHT_M = 0.20  # 6x the ball radius: unambiguously not a fit that has landed.
NEAR_VERTICAL_ANGLE_DEG = 75.0  # steeper than any real tennis ball still inside the frame.

# A mid-air stop, reported for every flight and blocking for `out_of_view` endings: the fitted
# trajectory simply ends, above the court, before the flight does.  docs/wk1/dev_loop.md saw
# this on both lob points -- "the trajectory ends at a non-zero height with a near-vertical last
# segment, where the out-of-frame gap starts.  That is the fitter's span ending, not a physical
# flight."
MID_AIR_STOP_HEIGHT_M = 0.30  # a ball this high is not resting on the court.
MID_AIR_STOP_SLACK_FRAMES = 2.0  # a fit may end a frame or two short without that being a stop.

# Written into every audit so a stale artifact cannot be read as if it used the other criterion.
CRITERION_DEFINITION = {
    CRITERION_PIXEL_V1: {
        "bounce": f"every scored bounce within {BOUNCE_TOLERANCE_M} m",
        "contact_px": f"every racket contact within {CONTACT_PIXEL_TOLERANCE_PX} px",
        "known_blind_spot": (
            "a flight with no bounce witness can be metres wrong in depth and still pass; "
            "docs/wk1/dev_loop.md measured a median of 7.9 m at 1.6 px on terminal flights"
        ),
    },
    CRITERION_METRIC_V2: {
        "bounce": f"every scored bounce within {BOUNCE_TOLERANCE_M} m",
        "contact_px": f"every racket contact within {CONTACT_PIXEL_TOLERANCE_PX} px",
        "contact_m": f"every racket contact within {CONTACT_TOLERANCE_M} m in 3D, both ends",
        "termination": (
            f"a bounce termination within {TERMINATION_TOLERANCE_M} m in 3D; an out_of_view "
            f"termination within {TERMINATION_TOLERANCE_M} m at the last frame the truth ball "
            "is still inside the image"
        ),
        "airborne": (
            f"at an out_of_view exit the fit is above {AIRBORNE_MIN_HEIGHT_M} m, has not stopped "
            f"in mid-air, and is not falling more steeply than {NEAR_VERTICAL_ANGLE_DEG} deg"
        ),
        "junction": f"the gap to each adjacent fitted flight within {JUNCTION_TOLERANCE_M} m",
        "trajectory": (
            f"the sampled 3D trajectory within {TRAJECTORY_RMS_TOLERANCE_M} m RMS of the "
            "re-integrated truth curve at every whole frame of the flight's scored span"
        ),
        "anchor": "historical 0.10 m bounce reference; not the current owner's spatial target",
    },
}

# Singles court lines in the pipeline court frame.  A bounce is IN when the ball touches the
# line, so the ball radius is added to every edge -- the same convention Hawk-Eye publishes.
SINGLES_HALF_WIDTH_M = 8.23 / 2.0
SINGLES_X_MIN_M = COURT_WIDTH_M / 2.0 - SINGLES_HALF_WIDTH_M
SINGLES_X_MAX_M = COURT_WIDTH_M / 2.0 + SINGLES_HALF_WIDTH_M

# The terminations the owner labels, in the order this module tests them.
TERMINATION_KINDS = ("terminal_bounce", "second_bounce", "out_of_view")

# Written into every truth file and every audit summary so a stale artifact cannot be misread.
TRUTH_DEFINITION = {
    "flight_ends_at": "the next shot",
    "terminal_flight_ends_at": (
        "the SECOND bounce when the first bounce is in; the FIRST bounce when it is out; "
        "the frame the ball leaves the field of view when it leaves first and does not return"
    ),
    "termination_kinds": list(TERMINATION_KINDS),
    "in_bounds_test": "the bounce spot against the singles lines, ball touching the line",
    "terminal_flight_bounces_when_first_bounce_is_in": 2,
    "dead_ball": (
        "generated and emitted after the termination so the fitter has to reject it; "
        "excluded from truth-good scoring and from complete points; diagnostic only"
    ),
    "walls": "not generated: this bench has no wall geometry",
    "same_half_first_bounce": "excluded: correct fault/error ending not implemented",
    "intermediate_out_bounce": "excluded: a rally cannot continue through an out bounce",
    "serve_fault_into_the_net": "not generated: reconstruction.py deletes any fit holding a net_hit",
}

COMPLETE_TRUTH_DEFINITION = {
    **TRUTH_DEFINITION,
    "terminal_flight_ends_at": "first OUT bounce or second bounce after an IN first bounce",
    "termination_kinds": ["terminal_bounce", "second_bounce"],
    "observation_boundary": "field-of-view exit recorded separately; hidden trajectory retained",
    "net_clearance": "every interpolant crossing before physical ending clears net by ball radius",
    "scope": "ground-ending development controls; not net-impact dynamics or a full rules engine",
}

# A terminal flight whose fit reaches the termination is scored at the fitter's own position
# there, whether it reported a bounce record or simply ended its flight at the termination.
FIT_ENDPOINT_SLACK_FRAMES = 1.5

# Dead ball: what the generator keeps rolling AFTER the termination, so the fitter sees it and
# has to reject it.  At least this long, and at most this long, subject to the host span.
DEAD_BALL_MIN_SECONDS = 0.25
DEAD_BALL_MAX_SECONDS = 1.30
DEAD_BALL_BOUNCE_CAP = 3

# Native frame.  A tracker cannot see a ball outside it, so those rows are absent from the
# generated track (a gap) and marked out-of-frame in the truth.
IMAGE_WIDTH_PX = 1920
IMAGE_HEIGHT_PX = 1080

# Lob skeletons: the last shot of the rally is re-solved to cross the net this high, which sends
# the projected ball out through the top edge of the frame on a broadcast camera.
LOB_NET_CLEARANCE_M = (7.0, 13.0)
LOB_MIN_TOP_EXIT_FRAMES = 3
LOB_MAX_OUT_OF_FRAME_FRACTION = 0.35
OUT_OF_FRAME_FRACTION = 0.06

# Terminal-exit skeletons: the last shot is re-aimed long, over the baseline and high, so the
# ball leaves the field of view and never comes back -- the owner's ``out_of_view`` ending.
EXIT_OVERSHOOT_M = (2.5, 8.0)
EXIT_NET_CLEARANCE_M = (2.5, 6.0)

# Generation modes.
MODE_RALLY = "rally"
MODE_LOB = "lob"
MODE_TERMINAL_EXIT = "terminal_exit"

# docs/wk1/solve_failures.md taxonomy, in pipeline order.
FIRST_CAUSE_ORDER = (
    "camera_abstained",
    "point_held",
    "flight_not_attempted",
    "flight_count_mismatch",
    "net_hit_flight_discarded",
    "flight_not_solved",
    "solved_but_rejected",
    "accepted_but_wrong",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {"n": 0, "median": None, "p90": None, "max": None}
    array = np.asarray(finite)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _in_image(pixel: Sequence[float]) -> bool:
    """Is the projected ball inside the native frame the tracker actually sees?"""
    x, y = float(pixel[0]), float(pixel[1])
    return 0.0 <= x <= IMAGE_WIDTH_PX - 1.0 and 0.0 <= y <= IMAGE_HEIGHT_PX - 1.0


def _longest_run(frames: Sequence[int]) -> int:
    """Longest run of consecutive frames, for "the ball left the frame for some frames"."""
    ordered = sorted(int(frame) for frame in frames)
    best = run = 0
    previous: int | None = None
    for frame in ordered:
        run = run + 1 if previous is not None and frame == previous + 1 else 1
        previous = frame
        best = max(best, run)
    return best


def bounce_in_bounds(xyz: Sequence[float]) -> bool:
    """Is a court impact IN, against the singles lines, with the ball touching the line?

    The generator lands every shot on the Hawk-Eye skeleton's own measured bounce spot, so this
    reads the skeleton's spot against the court lines, which is the owner's rule.
    """
    x, y = float(xyz[0]), float(xyz[1])
    return (
        SINGLES_X_MIN_M - BALL_RADIUS_M <= x <= SINGLES_X_MAX_M + BALL_RADIUS_M
        and -BALL_RADIUS_M <= y <= COURT_LENGTH_M + BALL_RADIUS_M
    )


def bounce_in_receiving_half(hit: Sequence[float], bounce: Sequence[float]) -> bool:
    """Supported skeleton shots must first land across the net, not in the hitter's half.

    This is only a generator guard. It does not judge service-box legality, line calls,
    reaching over the net after backspin, or other rules exceptions in real inference.
    """
    values = (float(hit[1]), float(bounce[1]))
    return all(math.isfinite(value) for value in values) and (
        (values[0] - NET_Y_M) * (values[1] - NET_Y_M) < 0.0
    )


def termination_of(
    bounces: Sequence[dict[str, Any]],
    *,
    end_frame: float,
    out_of_frame_frames: Iterable[int] = (),
) -> tuple[float, str]:
    """Physical bounce ending, or explicitly requested historical visibility boundary.

    ``bounces`` are the truth bounces of the terminal flight in time order, each carrying
    ``frame`` and ``xyz``.  The bounce rule fires first: the SECOND bounce when the first is in,
    the FIRST bounce when it is out.  The court exit overrides it only when the ball is already
    gone at that frame and never came back. This optional override exists only to read/test
    historical v1-v4 semantics; v5 generation never supplies out_of_frame_frames here.
    """
    ordered = sorted(bounces, key=lambda row: float(row["frame"]))
    if not ordered:
        candidate, kind = float(end_frame), "span_end"
    elif not bounce_in_bounds(ordered[0]["xyz"]):
        candidate, kind = float(ordered[0]["frame"]), "terminal_bounce"
    elif len(ordered) > 1:
        candidate, kind = float(ordered[1]["frame"]), "second_bounce"
    else:
        candidate, kind = float(end_frame), "span_end"
    exit_frame = _uninterrupted_exit(out_of_frame_frames, candidate)
    if exit_frame is not None:
        return exit_frame, "out_of_view"
    return candidate, kind


def _uninterrupted_exit(out_of_frame_frames: Iterable[int], candidate: float) -> float | None:
    """The frame the ball left the image for good, if it never returned before ``candidate``."""
    gone = {int(frame) for frame in out_of_frame_frames if float(frame) <= candidate}
    if not gone:
        return None
    last = max(gone)
    if last < math.floor(candidate):
        # It came back: there are visible frames between ``last`` and the bounce.
        return None
    first = last
    while first - 1 in gone:
        first -= 1
    return float(first)


def net_clearances(frames: np.ndarray, positions: np.ndarray) -> list[dict[str, float]]:
    """Audit every net-plane crossing of the recorded piecewise-linear truth.

    The generator models no net impact: an intersecting flight must be resampled,
    not allowed to continue through the net. This includes rebound segments.
    """
    frames, positions = np.asarray(frames, float), np.asarray(positions, float)
    if (
        frames.ndim != 1
        or len(frames) < 2
        or positions.shape != (len(frames), 3)
        or not np.isfinite(frames).all()
        or not np.isfinite(positions).all()
        or np.any(np.diff(frames) <= 0)
    ):
        raise ValueError("finite ordered trajectory samples required")
    rows = []
    offsets = positions[:, 1] - NET_Y_M
    for i in range(len(frames) - 1):
        a, b = offsets[i : i + 2]
        if a == 0 and b == 0:
            fractions = (0.0, 1.0)
        elif a == 0:
            fractions = (0.0,)
        elif b == 0 or a * b < 0:
            fractions = (float(-a / (b - a)),)
        else:
            continue
        for fraction in fractions:
            xyz = positions[i] + fraction * (positions[i + 1] - positions[i])
            if not -0.914 - BALL_RADIUS_M <= xyz[0] <= 11.884 + BALL_RADIUS_M:
                continue
            frame = float(frames[i] + fraction * (frames[i + 1] - frames[i]))
            if rows and abs(rows[-1]["frame"] - frame) < 1e-9:
                continue
            rows.append(
                {
                    "frame": frame,
                    "ball_surface_clearance_m": float(
                        xyz[2] - BALL_RADIUS_M - camera_cal.net_height_at_x(float(xyz[0]))
                    ),
                }
            )
    return rows


def terminal_scored_end(flight: dict[str, Any]) -> float:
    """The last frame of a flight's scored span.

    A contact-to-contact flight's scored span is exactly the generated one, because it ends at
    the next shot.  A terminal flight's ends at its termination.  ``truth.v3`` records the
    termination the generator built; a ``truth.v1``/``v2`` flight is re-derived from its bounces
    so a run made before this definition existed can still be re-scored.
    """
    if not flight.get("terminal"):
        return float(flight["end_frame"])
    recorded = flight.get("termination_frame")
    if recorded is not None:
        return float(recorded)
    frame, _ = termination_of(
        flight.get("bounces") or [],
        end_frame=float(flight["end_frame"]),
    )
    return frame


def terminal_termination_kind(flight: dict[str, Any]) -> str | None:
    """The termination kind of a terminal flight, recorded or re-derived."""
    if not flight.get("terminal"):
        return None
    recorded = flight.get("termination_kind")
    if recorded is not None:
        return str(recorded)
    _, kind = termination_of(
        flight.get("bounces") or [],
        end_frame=float(flight["end_frame"]),
    )
    return kind


def scored_bounces(flight: dict[str, Any]) -> list[dict[str, Any]]:
    """The truth bounces inside the scored span.

    A terminal flight with an IN first bounce keeps TWO: the bounce knot and the termination
    anchor.  One with an OUT first bounce keeps that bounce.  One that ends at the court exit
    keeps whichever bounces happened before the ball left, normally none.
    """
    end = terminal_scored_end(flight)
    return [row for row in flight.get("bounces") or [] if float(row["frame"]) <= end + 1e-9]


# --------------------------------------------------------------------------------------- #
# Host geometry: real cohort points supply the camera, the frame rate and the frame window
# --------------------------------------------------------------------------------------- #
@dataclass
class HostPoint:
    match_id: str
    clip: str
    surface: str
    fps: float
    span: tuple[float, float]
    camera_rows: dict[str, np.ndarray]
    homography: np.ndarray
    frame_homographies: dict[int, np.ndarray]

    @property
    def key(self) -> str:
        return f"{self.match_id}__{self.clip}"


def load_hosts(cohort_root: Path, *, surfaces: Sequence[str] = ("hard", "clay")) -> list[HostPoint]:
    """Every cohort point whose camera the pipeline itself accepts, on a measured surface."""
    manifest = json.loads((cohort_root / "manifest.json").read_text())
    active = json.loads((cohort_root / "active_play_v1.json").read_text())
    validity_path = cohort_root / "point_validity_gate_v1.json"
    validity = {}
    if validity_path.is_file():
        validity = {
            f"{row['match_id']}__{row['clip']}": row
            for row in json.loads(validity_path.read_text())["rows"]
        }
    hosts: list[HostPoint] = []
    for match in manifest["matches"]:
        if str(match["surface"]) not in set(surfaces):
            continue
        match_dir = cohort_root / str(match["id"])
        if not (match_dir / CAMERA_NAME).is_file():
            continue
        projection = np.load(match_dir / CAMERA_NAME, allow_pickle=True)
        point_homography = np.load(match_dir / "court_H_per_point.npz")
        frame_homography_path = match_dir / "court_H_per_frame_v1.npz"
        frame_homography = (
            np.load(frame_homography_path, allow_pickle=True)
            if frame_homography_path.is_file()
            else None
        )
        for point_id in match["point_ids"]:
            clip = f"pt{int(point_id):04d}"
            key = f"{match['id']}__{clip}"
            active_row = active.get(f"{match['id']}/{clip}")
            if active_row is None or not active_row.get("active_spans"):
                continue
            row = validity.get(key)
            if row is not None and row.get("decision") != "retain":
                continue
            try:
                camera = PointCamera(match_dir, clip)
            except (ValueError, IndexError):
                continue
            if not camera.quality_summary(active_row["active_spans"])["accepted"]:
                continue
            selected = projection["clips"] == clip
            camera_rows = {
                field_name: projection[field_name][selected] for field_name in projection.files
            }
            homography_index = list(point_homography["pts"].astype(int)).index(int(point_id))
            frames = camera_rows["frames"].astype(int)
            span = max(active_row["active_spans"], key=lambda pair: pair[1] - pair[0])
            start = max(float(span[0]), float(frames.min()))
            end = min(float(span[1]), float(frames.max()))
            if end - start < 3.0 * float(match["source_fps"]):
                continue
            per_frame: dict[int, np.ndarray] = {}
            if frame_homography is not None and {"clips", "frames", "H"}.issubset(
                frame_homography.files
            ):
                mask = frame_homography["clips"] == clip
                per_frame = dict(
                    zip(
                        frame_homography["frames"][mask].astype(int),
                        frame_homography["H"][mask],
                    )
                )
            hosts.append(
                HostPoint(
                    match_id=str(match["id"]),
                    clip=clip,
                    surface=str(match["surface"]),
                    fps=float(match["source_fps"]),
                    span=(start, end),
                    camera_rows=camera_rows,
                    homography=point_homography["H"][homography_index],
                    frame_homographies=per_frame,
                )
            )
    return hosts


def host_projection(host: HostPoint, frame: float) -> np.ndarray:
    frames = host.camera_rows["frames"].astype(int)
    rounded = int(round(frame))
    index = int(np.argmin(np.abs(frames - rounded)))
    return host.camera_rows["P"][index]


def project(projection: np.ndarray, point: Sequence[float]) -> np.ndarray:
    homogeneous = projection @ np.array([point[0], point[1], point[2], 1.0], dtype=float)
    if abs(homogeneous[2]) < 1e-9:
        return np.array([math.nan, math.nan])
    return homogeneous[:2] / homogeneous[2]


# --------------------------------------------------------------------------------------- #
# Rally skeletons from the Hawk-Eye corpus
# --------------------------------------------------------------------------------------- #
@dataclass
class SkeletonStrike:
    hit: np.ndarray
    net: np.ndarray | None
    bounce: np.ndarray
    apex_height_m: float
    spin_rpm: float | None
    serve_speed_ms: float | None


@dataclass
class Skeleton:
    source: str
    point_id: str
    surface: str
    strikes: list[SkeletonStrike]


def _to_pipeline(point: Sequence[float]) -> np.ndarray:
    """Corpus court frame (x length, y width, net at x=0) -> pipeline frame."""
    return np.array(
        [float(point[1]) + COURT_WIDTH_M / 2.0, float(point[0]) + NET_Y_M, float(point[2])]
    )


def load_skeletons(
    corpus_root: Path, *, max_files: int, min_shots: int = 3, max_shots: int = 6
) -> list[Skeleton]:
    """Real rallies: strike order, hit heights, net clearances, bounce spots, serve speeds."""
    clay = sorted(glob.glob(str(corpus_root / "ball_trajectory" / "*roland_garros*.csv")))
    hard = sorted(glob.glob(str(corpus_root / "ball_trajectory" / "*australian_open*.csv")))
    half = max(max_files // 2, 1)
    files = clay[:half] + hard[:half]
    pbp_dir = str(corpus_root / "play_by_play")
    output: list[Skeleton] = []
    for path in files:
        grouped: dict[tuple[str, str], list[hff.Strike]] = defaultdict(list)
        for strike in hff.load_strikes(path, pbp_dir):
            grouped[(strike.key[1], strike.key[2])].append(strike)
        for (point_id, serve_num), rally in sorted(grouped.items()):
            rally.sort(key=lambda row: row.strike_index)
            if rally[0].strike_index != 1 or len(rally) < min_shots:
                continue
            indices = [row.strike_index for row in rally]
            if indices != list(range(1, len(rally) + 1)):
                continue
            strikes = []
            usable = True
            for strike in rally[:max_shots]:
                if strike.net is None or strike.bounce is None:
                    usable = False
                    break
                strikes.append(
                    SkeletonStrike(
                        hit=_to_pipeline(strike.hit),
                        net=_to_pipeline(strike.net),
                        bounce=_to_pipeline(strike.bounce),
                        apex_height_m=float(strike.z_peak_obs),
                        spin_rpm=strike.spin_rpm,
                        serve_speed_ms=strike.serve_speed_ms,
                    )
                )
            if not usable or len(strikes) < min_shots:
                continue
            output.append(
                Skeleton(
                    source=os.path.basename(path),
                    point_id=f"{point_id}_{serve_num}",
                    surface=rally[0].surface,
                    strikes=strikes,
                )
            )
    return output


# --------------------------------------------------------------------------------------- #
# Whole-point physics
# --------------------------------------------------------------------------------------- #
def _topspin_axis(velocity: Sequence[float]) -> np.ndarray:
    horizontal = np.array([float(velocity[0]), float(velocity[1]), 0.0])
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-9:
        return np.zeros(3)
    horizontal /= norm
    return np.array([-horizontal[1], horizontal[0], 0.0])


INTEGRATION_DT = 1.0 / 240.0


def integrate_arc(
    start: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    *,
    max_seconds: float,
    stop_at_ground: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-step RK4 arc from ``physics.flight``, stopped one step past the court plane."""
    times = [0.0]
    positions = [np.asarray(start, dtype=float)]
    velocities = [np.asarray(velocity, dtype=float)]
    spins = [np.asarray(spin, dtype=float)]
    x, v, w = positions[0].copy(), velocities[0].copy(), spins[0].copy()
    elapsed = 0.0
    while elapsed < max_seconds:
        x, v, w = flight.rk4_step(x, v, w, INTEGRATION_DT, params=AERO_PARAMS)
        elapsed += INTEGRATION_DT
        times.append(elapsed)
        positions.append(x.copy())
        velocities.append(v.copy())
        spins.append(w.copy())
        if stop_at_ground and x[2] <= BALL_RADIUS_M and v[2] < 0.0:
            break
        if abs(x[0]) > 60.0 or abs(x[1]) > 60.0 or x[2] > 30.0:
            break
    return (
        np.asarray(times),
        np.stack(positions),
        np.stack(velocities),
        np.stack(spins),
    )


def _interp_rows(value: float, times: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.array(
        [float(np.interp(value, times, rows[:, axis])) for axis in range(rows.shape[1])]
    )


def _descending_crossing(times: np.ndarray, positions: np.ndarray, height: float) -> float | None:
    offsets = positions[:, 2] - height
    candidates = np.flatnonzero((offsets[:-1] > 0.0) & (offsets[1:] <= 0.0))
    if not len(candidates):
        return None
    index = int(candidates[0])
    fraction = float(offsets[index] / (offsets[index] - offsets[index + 1]))
    return float(times[index] + fraction * (times[index + 1] - times[index]))


def _height_crossings(times: np.ndarray, positions: np.ndarray, height: float) -> list[float]:
    offsets = positions[:, 2] - height
    changes = np.flatnonzero(offsets[:-1] * offsets[1:] < 0.0)
    output = []
    for index in changes:
        index = int(index)
        fraction = float(offsets[index] / (offsets[index] - offsets[index + 1]))
        output.append(float(times[index] + fraction * (times[index + 1] - times[index])))
    return output


def _plane_time(times: np.ndarray, positions: np.ndarray, axis: int, value: float) -> float | None:
    offsets = positions[:, axis] - value
    changes = np.flatnonzero(offsets[:-1] * offsets[1:] < 0.0)
    if not len(changes):
        return None
    index = int(changes[0])
    fraction = float(offsets[index] / (offsets[index] - offsets[index + 1]))
    return float(times[index] + fraction * (times[index + 1] - times[index]))


@dataclass
class Piece:
    """One ballistic segment: a contact-to-bounce or bounce-to-contact arc."""

    t0: float
    t1: float
    x0: np.ndarray
    v0: np.ndarray
    w0: np.ndarray
    times: np.ndarray = field(repr=False, default=None)
    positions: np.ndarray = field(repr=False, default=None)

    def sample(self, seconds: Sequence[float]) -> np.ndarray:
        offsets = np.asarray(seconds, dtype=float) - self.t0
        return np.stack([_interp_rows(value, self.times, self.positions) for value in offsets])


def recorded_trajectory(pieces: Sequence[Piece], start_frame: float, fps: float) -> dict:
    """Persist the exact interpolant used to render observations, including invisible arcs.

    Keep integration samples and clipped piece endpoints. Between court impact
    and rebound the generator holds the same ground position during its declared
    dwell time; those two endpoints preserve that convention without re-fitting.
    """
    frames: list[float] = []
    positions: list[list[float]] = []
    for piece in pieces:
        duration = piece.t1 - piece.t0
        offsets = np.unique(
            np.r_[0.0, piece.times[(piece.times > 0) & (piece.times < duration)], duration]
        )
        for offset in offsets:
            frame = float(start_frame + (piece.t0 + offset) * fps)
            xyz = _interp_rows(float(offset), piece.times, piece.positions).tolist()
            if frames and abs(frame - frames[-1]) <= 1e-9:
                if np.linalg.norm(np.asarray(xyz) - positions[-1]) > 1e-6:
                    raise ValueError("simulation pieces disagree at a shared time")
                continue
            if frames and frame < frames[-1]:
                raise ValueError("simulation pieces are not ordered")
            frames.append(frame)
            positions.append(xyz)
    return {
        "schema": "simulation_trajectory_samples_v1",
        "source": "generator_piece_interpolant",
        "integration_dt_seconds": INTEGRATION_DT,
        "includes_out_of_view": True,
        "frames": frames,
        "positions": positions,
    }


def as_lob(strike: SkeletonStrike, rng: np.random.Generator) -> SkeletonStrike:
    """The same shot, re-aimed over the net at lob height.

    The corpus has no lob label, so a lob is built by keeping a real strike's hit position and
    bounce spot and replacing its measured net clearance with a sampled lob clearance.  The
    apex then rises far enough that a broadcast camera loses the ball out of the top of frame.
    """
    if strike.net is None:
        return strike
    height = float(rng.uniform(*LOB_NET_CLEARANCE_M))
    return SkeletonStrike(
        hit=strike.hit,
        net=np.array([float(strike.net[0]), float(strike.net[1]), height]),
        bounce=strike.bounce,
        apex_height_m=max(float(strike.apex_height_m), height),
        spin_rpm=strike.spin_rpm,
        serve_speed_ms=None,
    )


def as_terminal_exit(strike: SkeletonStrike, rng: np.random.Generator) -> SkeletonStrike:
    """The same shot, hit long and high so the ball leaves the field of view and stays gone.

    The bounce spot is pushed past the baseline the ball is travelling towards and the net
    clearance is raised, which is how a real over-hit leaves a broadcast frame: it goes out
    through the top edge and the point is over before it lands.
    """
    bounce = np.asarray(strike.bounce, dtype=float).copy()
    overshoot = float(rng.uniform(*EXIT_OVERSHOOT_M))
    if float(bounce[1]) >= float(strike.hit[1]):
        bounce[1] = COURT_LENGTH_M + overshoot
    else:
        bounce[1] = -overshoot
    height = float(rng.uniform(*EXIT_NET_CLEARANCE_M))
    net = strike.net
    if net is not None:
        net = np.array([float(net[0]), float(net[1]), height])
    return SkeletonStrike(
        hit=strike.hit,
        net=net,
        bounce=bounce,
        apex_height_m=max(float(strike.apex_height_m), height),
        spin_rpm=strike.spin_rpm,
        serve_speed_ms=None,
    )


def solve_shot(
    start: np.ndarray,
    target_bounce_xy: Sequence[float],
    target_net_z: float | None,
    spin_rpm: float,
    *,
    speed_hint: float | None = None,
    lob: bool = False,
) -> dict[str, Any] | None:
    """Launch velocity that lands on the skeleton's bounce spot at its net clearance."""
    heading = math.atan2(target_bounce_xy[1] - start[1], target_bounce_xy[0] - start[0])
    horizontal = float(math.hypot(target_bounce_xy[0] - start[0], target_bounce_xy[1] - start[1]))
    if horizontal < 3.0:
        return None
    spin_magnitude = float(spin_rpm) * 2.0 * math.pi / 60.0
    guess_speed = float(speed_hint) if speed_hint else max(horizontal / 0.75, 12.0)
    crosses_net = (start[1] - NET_Y_M) * (target_bounce_xy[1] - NET_Y_M) < 0.0

    def state(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed, elevation, yaw = parameters
        direction = heading + yaw
        velocity = np.array(
            [
                speed * math.cos(elevation) * math.cos(direction),
                speed * math.cos(elevation) * math.sin(direction),
                speed * math.sin(elevation),
            ]
        )
        return velocity, spin_magnitude * _topspin_axis(velocity)

    max_seconds = 5.0 if lob else 3.0

    def residual(parameters: np.ndarray) -> list[float]:
        velocity, spin = state(parameters)
        times, positions, _, _ = integrate_arc(start, velocity, spin, max_seconds=max_seconds)
        ground = _descending_crossing(times, positions, BALL_RADIUS_M)
        if ground is None:
            return [30.0] * 3
        landing = _interp_rows(ground, times, positions)
        rows = [
            float(landing[0] - target_bounce_xy[0]),
            float(landing[1] - target_bounce_xy[1]),
        ]
        if target_net_z is None or not crosses_net:
            rows.append(0.0)
            return rows
        crossing = _plane_time(times, positions, 1, NET_Y_M)
        if crossing is None:
            return [30.0] * 3
        rows.append(float(_interp_rows(crossing, times, positions)[2] - target_net_z))
        return rows

    try:
        solution = least_squares(
            residual,
            [guess_speed, math.radians(40.0 if lob else 8.0), 0.0],
            bounds=(
                [6.0, math.radians(-35.0), math.radians(-25.0)],
                [68.0, math.radians(75.0 if lob else 55.0), math.radians(25.0)],
            ),
            xtol=1e-3,
            ftol=1e-3,
            max_nfev=45,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    values = np.asarray(residual(solution.x), dtype=float)
    if float(np.max(np.abs(values))) > 0.30:
        return None
    velocity, spin = state(solution.x)
    return {"velocity": velocity, "spin": spin, "residual_max_m": float(np.max(np.abs(values)))}


def simulate_rally(
    host: HostPoint,
    skeleton: Skeleton,
    shots: int,
    rng: np.random.Generator,
    *,
    mode: str = MODE_RALLY,
) -> dict[str, Any] | str:
    """Simulate exactly ``shots`` joined flights, a termination and a dead ball, or a reason.

    ``mode`` shapes the last shot only.  ``lob`` re-aims it over the net at lob height, so the
    projected ball leaves the frame through the top edge and comes back.  ``terminal_exit``
    re-aims it long and high, so the ball leaves the field of view and never comes back.
    Its physical ground ending is still retained as hidden truth.
    """
    fps = host.fps
    span_start, span_end = host.span
    start_frame = float(math.ceil(span_start)) + 1.0
    budget_seconds = (span_end - start_frame) / fps
    if budget_seconds < 3.0:
        return "span_too_short"

    strikes = list(skeleton.strikes[:shots])
    if mode == MODE_LOB:
        strikes[-1] = as_lob(strikes[-1], rng)
    elif mode == MODE_TERMINAL_EXIT:
        strikes[-1] = as_terminal_exit(strikes[-1], rng)

    pieces: list[Piece] = []
    contacts: list[dict[str, Any]] = []
    bounces: list[dict[str, Any]] = []
    position = strikes[0].hit.copy()
    clock = 0.0
    contacts.append({"time": 0.0, "xyz": position.copy(), "phase": "serve"})
    rebound_state: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

    for index in range(shots):
        strike = strikes[index]
        successor = strikes[index + 1] if index + 1 < shots else None
        lob_shot = mode in {MODE_LOB, MODE_TERMINAL_EXIT} and index == shots - 1
        spin_rpm = strike.spin_rpm if strike.spin_rpm else (2400.0 if index == 0 else 1900.0)
        hint = strike.serve_speed_ms if index == 0 else None
        shot = solve_shot(
            position,
            strike.bounce[:2],
            float(strike.net[2]) if strike.net is not None else None,
            spin_rpm,
            speed_hint=hint,
            lob=lob_shot,
        )
        if shot is None:
            return "shot_unsolvable"
        times, positions, velocities, spins = integrate_arc(
            position, shot["velocity"], shot["spin"], max_seconds=5.0 if lob_shot else 3.0
        )
        ground = _descending_crossing(times, positions, BALL_RADIUS_M)
        if ground is None or ground < 0.20:
            return "no_bounce"
        bounce_xyz = _interp_rows(ground, times, positions)
        if not bounce_in_receiving_half(position, bounce_xyz):
            return "same_half_bounce_unsupported"
        if successor is not None and not bounce_in_bounds(bounce_xyz):
            return "intermediate_out_bounce"
        bounce_velocity = _interp_rows(ground, times, velocities)
        bounce_spin = _interp_rows(ground, times, spins)
        pieces.append(
            Piece(
                clock,
                clock + ground,
                position.copy(),
                shot["velocity"],
                shot["spin"],
                times=times,
                positions=positions,
            )
        )
        bounces.append({"time": clock + ground, "xyz": bounce_xyz.copy()})
        try:
            rebound = bounce_reference.court_bounce(bounce_velocity, bounce_spin, host.surface)
        except ValueError:
            return "bounce_model_refused"
        after_time = clock + ground + bounce_reference.DWELL_SECONDS
        after_position = np.array([bounce_xyz[0], bounce_xyz[1], BALL_RADIUS_M])
        rebound_state = (after_position, rebound.velocity, rebound.spin, after_time)
        if successor is None:
            break
        rise_times, rise_positions, _, _ = integrate_arc(
            after_position, rebound.velocity, rebound.spin, max_seconds=2.5
        )
        target_height = float(successor.hit[2])
        crossings = [
            value
            for value in _height_crossings(rise_times, rise_positions, target_height)
            if value > 0.06
        ]
        if not crossings:
            return "receiver_height_unreachable"
        wanted = float(
            math.hypot(successor.hit[0] - bounce_xyz[0], successor.hit[1] - bounce_xyz[1])
        )

        def travelled(value: float) -> float:
            point = _interp_rows(value, rise_times, rise_positions)
            return float(math.hypot(point[0] - bounce_xyz[0], point[1] - bounce_xyz[1]))

        chosen = min(crossings, key=lambda value: abs(travelled(value) - wanted))
        next_position = _interp_rows(chosen, rise_times, rise_positions)
        if not (-1.0 <= next_position[0] <= COURT_WIDTH_M + 1.0):
            return "receiver_off_court"
        if not (-2.0 <= next_position[1] <= COURT_LENGTH_M + 2.0):
            return "receiver_off_court"
        pieces.append(
            Piece(
                after_time,
                after_time + chosen,
                after_position,
                rebound.velocity,
                rebound.spin,
                times=rise_times,
                positions=rise_positions,
            )
        )
        clock = after_time + chosen
        position = next_position
        contacts.append({"time": clock, "xyz": position.copy(), "phase": "rally"})
        if clock + 1.2 > budget_seconds:
            return "span_exhausted"

    if rebound_state is None or len(contacts) < 2:
        return "too_few_contacts"

    # The termination, and then the dead ball.  The last shot's bounce is already the last row of
    # ``bounces``: it is the terminal flight's FIRST bounce, and whether it is IN decides the
    # ending.  Out -> the point ends there.  In -> the point ends at the SECOND bounce, which the
    # loop below has to reach.  Either way the simulation keeps running afterwards, because the
    # real chain hands the fitter the dead ball and the fitter has to reject it.
    first_bounce_in = bounce_in_bounds(bounces[-1]["xyz"])
    bounce_termination: float | None = None if first_bounce_in else float(bounces[-1]["time"])

    # docs/wk1/solve_failures.md measured 27 of 39 real rally-closing spans holding one owner
    # bounce and 10 holding two; that is the number of EXTRA bounces the dead ball is given.
    weights = np.asarray([count for _, count in TERMINAL_BOUNCE_MIX], dtype=float)
    dead_ball_wanted = int(
        rng.choice([value for value, _ in TERMINAL_BOUNCE_MIX], p=weights / weights.sum())
    )
    dead_ball_seconds = float(rng.uniform(DEAD_BALL_MIN_SECONDS, DEAD_BALL_MAX_SECONDS))

    tail_position, tail_velocity, tail_spin, tail_clock = rebound_state
    dead_ball_bounces = 0
    while True:
        remaining = budget_seconds - tail_clock - 0.15
        if remaining <= 0.1:
            break
        if bounce_termination is not None and tail_clock >= bounce_termination + dead_ball_seconds:
            break
        times, positions, velocities, spins = integrate_arc(
            tail_position, tail_velocity, tail_spin, max_seconds=min(remaining, 2.5)
        )
        ground = _descending_crossing(times, positions, BALL_RADIUS_M)
        stop_bouncing = bounce_termination is not None and (
            dead_ball_bounces >= min(dead_ball_wanted, DEAD_BALL_BOUNCE_CAP)
        )
        if ground is None or stop_bouncing:
            duration = float(times[-1]) if ground is None else max(ground - 0.10, 0.20)
            duration = min(duration, remaining)
            if bounce_termination is not None:
                duration = min(
                    duration, max(bounce_termination + dead_ball_seconds - tail_clock, 0.10)
                )
            if duration <= 0.05:
                break
            pieces.append(
                Piece(
                    tail_clock,
                    tail_clock + duration,
                    tail_position,
                    tail_velocity,
                    tail_spin,
                    times=times,
                    positions=positions,
                )
            )
            tail_clock += duration
            break
        landing = _interp_rows(ground, times, positions)
        velocity = _interp_rows(ground, times, velocities)
        spin = _interp_rows(ground, times, spins)
        pieces.append(
            Piece(
                tail_clock,
                tail_clock + ground,
                tail_position,
                tail_velocity,
                tail_spin,
                times=times,
                positions=positions,
            )
        )
        bounces.append({"time": tail_clock + ground, "xyz": landing.copy()})
        if bounce_termination is None:
            bounce_termination = tail_clock + ground
        else:
            dead_ball_bounces += 1
        try:
            rebound = bounce_reference.court_bounce(velocity, spin, host.surface)
        except ValueError:
            break
        tail_clock += ground + bounce_reference.DWELL_SECONDS
        tail_position = np.array([landing[0], landing[1], BALL_RADIUS_M])
        tail_velocity = rebound.velocity
        tail_spin = rebound.spin

    if bounce_termination is None:
        # An in-bounds first bounce whose second bounce never happened inside the host span.
        return "second_bounce_unreached"
    if tail_clock > budget_seconds:
        return "span_exhausted"

    frames = np.arange(
        int(math.ceil(start_frame)), int(math.floor(start_frame + tail_clock * fps)) + 1, dtype=int
    )
    if len(frames) < 25:
        return "too_few_frames"
    positions_by_frame: dict[int, np.ndarray] = {}
    for piece in pieces:
        selected = [
            int(frame)
            for frame in frames
            if piece.t0 - 1e-9 <= (float(frame) - start_frame) / fps <= piece.t1 + 1e-9
        ]
        if not selected:
            continue
        sampled = piece.sample([(float(frame) - start_frame) / fps for frame in selected])
        for frame, point in zip(selected, sampled, strict=True):
            positions_by_frame[frame] = point
    if len(positions_by_frame) < len(frames) - 2:
        return "trajectory_gap"

    track_pixels = {}
    for frame in sorted(positions_by_frame):
        pixel = project(host_projection(host, float(frame)), positions_by_frame[frame])
        if not np.all(np.isfinite(pixel)):
            return "projection_singular"
        track_pixels[frame] = pixel
    # A tracker cannot see a ball that is not in the image.  Those frames are dropped from the
    # generated track -- that is the gap a lob leaves through the top edge -- and are marked
    # out-of-frame in the truth.
    out_of_frame = [frame for frame, pixel in track_pixels.items() if not _in_image(pixel)]
    top_exit = [frame for frame in out_of_frame if float(track_pixels[frame][1]) < 0.0]

    def frame_of(seconds: float) -> float:
        return float(start_frame + seconds * fps)

    truth_contacts = [
        {
            "frame": frame_of(row["time"]),
            "xyz": row["xyz"].tolist(),
            "phase": row["phase"],
            "pixel": project(host_projection(host, frame_of(row["time"])), row["xyz"]).tolist(),
        }
        for row in contacts
    ]
    truth_bounces = [
        {
            "frame": frame_of(row["time"]),
            "xyz": row["xyz"].tolist(),
            "in_bounds": bool(bounce_in_bounds(row["xyz"])),
            "pixel": project(host_projection(host, frame_of(row["time"])), row["xyz"]).tolist(),
        }
        for row in bounces
    ]
    if not all(_in_image(row["pixel"]) for row in truth_contacts):
        return "contact_out_of_frame"

    # Physical truth cannot end merely because its projection leaves the picture.
    # Preserve that observation boundary separately, including in terminal-exit cases.
    terminal_frame = frame_of(tail_clock)
    terminal_start = truth_contacts[-1]["frame"]
    terminal_bounces = [
        row for row in truth_bounces if terminal_start < row["frame"] < terminal_frame
    ]
    termination_frame, termination_kind = termination_of(
        terminal_bounces,
        end_frame=terminal_frame,
    )
    observation_exit_frame = _uninterrupted_exit(
        [frame for frame in out_of_frame if float(frame) > terminal_start], termination_frame
    )
    if termination_kind == "span_end":
        return "no_termination"
    if termination_frame <= terminal_start:
        return "terminal_flight_empty"
    if (termination_frame - terminal_start) / fps < TERMINAL_MIN_SECONDS:
        return "terminal_flight_too_short"
    if mode == MODE_TERMINAL_EXIT and observation_exit_frame is None:
        return "not_a_terminal_exit"

    observation_end = observation_exit_frame or termination_frame
    before_termination = [frame for frame in out_of_frame if float(frame) < observation_end]
    if mode == MODE_LOB:
        if _longest_run(top_exit) < LOB_MIN_TOP_EXIT_FRAMES:
            return "lob_stayed_in_frame"
        if len(out_of_frame) > LOB_MAX_OUT_OF_FRAME_FRACTION * len(track_pixels):
            return "off_image"
        sideways = [frame for frame in before_termination if frame not in set(top_exit)]
        if len(sideways) > OUT_OF_FRAME_FRACTION * len(track_pixels):
            return "off_image"
    elif len(before_termination) > OUT_OF_FRAME_FRACTION * len(track_pixels):
        return "off_image"

    scored_bounce_rows = [
        row for row in terminal_bounces if row["frame"] <= termination_frame + 1e-9
    ]
    if observation_exit_frame is None and not all(
        _in_image(row["pixel"]) for row in scored_bounce_rows
    ):
        return "scored_bounce_out_of_frame"
    termination_xyz = scored_bounce_rows[-1]["xyz"]
    # This is an oracle projection, possibly outside the image, not a detector observation.
    termination_pixel = list(scored_bounce_rows[-1]["pixel"])

    for frame in out_of_frame:
        del track_pixels[frame]
    if len(track_pixels) < 25:
        return "too_few_frames"
    visible_after = [frame for frame in track_pixels if float(frame) > termination_frame + 1e-9]
    if observation_exit_frame is None and len(visible_after) < 2:
        return "no_dead_ball"

    trajectory = recorded_trajectory(pieces, float(start_frame), fps)
    truth_frames = np.asarray(trajectory["frames"])
    query = np.unique(np.r_[truth_frames[truth_frames < termination_frame], termination_frame])
    truth_positions = np.asarray(trajectory["positions"])
    positions_to_end = np.column_stack(
        [np.interp(query, truth_frames, truth_positions[:, axis]) for axis in range(3)]
    )
    clearance = net_clearances(query, positions_to_end)
    if any(row["ball_surface_clearance_m"] < 0 for row in clearance):
        return "net_intersection_unsupported"

    boundaries = [row["frame"] for row in truth_contacts] + [terminal_frame]
    flights = []
    for index in range(len(boundaries) - 1):
        low, high = boundaries[index], boundaries[index + 1]
        inside = [row for row in truth_bounces if low < row["frame"] < high]
        terminal = index == len(boundaries) - 2
        entry: dict[str, Any] = {
            "flight_index": index,
            "start_frame": low,
            "end_frame": high,
            "terminal": terminal,
            "bounces": inside,
        }
        if terminal:
            entry["termination_frame"] = float(termination_frame)
            entry["termination_kind"] = termination_kind
            entry["termination_xyz"] = list(termination_xyz)
            entry["first_bounce_in_bounds"] = bool(first_bounce_in)
        entry["scored_end_frame"] = terminal_scored_end(entry)
        entry["scored_bounces"] = len(scored_bounces(entry))
        entry["dead_ball_bounces"] = len(inside) - entry["scored_bounces"]
        entry["out_of_frame_frames"] = sum(
            1 for frame in out_of_frame if low <= float(frame) <= entry["scored_end_frame"]
        )
        flights.append(entry)
    return {
        "host": host.key,
        "source_match": host.match_id,
        "source_point": host.key,
        "mode": mode,
        "lob": mode == MODE_LOB,
        "out_of_frame_frames": sorted(int(frame) for frame in out_of_frame),
        "top_exit_frames": sorted(int(frame) for frame in top_exit),
        "surface": host.surface,
        "fps": fps,
        "skeleton": f"{skeleton.source}:{skeleton.point_id}",
        "start_frame": float(start_frame),
        "end_frame": float(terminal_frame),
        "shots": shots,
        "termination_frame": float(termination_frame),
        "termination_kind": termination_kind,
        "observation_exit_frame": observation_exit_frame,
        "physical_truth_audit": {"net_crossings": clearance, "net_clear": True},
        "termination_xyz": list(termination_xyz),
        "termination_pixel": list(termination_pixel),
        "first_bounce_in_bounds": bool(first_bounce_in),
        "terminal_bounces": len(terminal_bounces),
        "dead_ball_bounces": len(terminal_bounces) - len(scored_bounce_rows),
        "terminal_seconds": float((termination_frame - terminal_start) / fps),
        "frames": [int(frame) for frame in sorted(track_pixels)],
        "track_pixels": {int(frame): value.tolist() for frame, value in track_pixels.items()},
        "positions": {
            int(frame): positions_by_frame[frame].tolist() for frame in sorted(track_pixels)
        },
        "trajectory_truth": trajectory,
        "contacts": truth_contacts,
        "bounces": truth_bounces,
        "flights": flights,
    }


def build_point(
    host: HostPoint,
    skeleton: Skeleton,
    rng: np.random.Generator,
    *,
    max_shots: int = 5,
    reasons: dict[str, int] | None = None,
    mode: str = MODE_RALLY,
) -> dict[str, Any] | None:
    """Longest rally from this skeleton that fits the host point's active span."""
    upper = min(max_shots, len(skeleton.strikes))
    for shots in range(upper, 1, -1):
        outcome = simulate_rally(host, skeleton, shots, rng, mode=mode)
        if isinstance(outcome, dict):
            return outcome
        if reasons is not None:
            reasons[f"{shots}:{outcome}"] = reasons.get(f"{shots}:{outcome}", 0) + 1
    return None


# --------------------------------------------------------------------------------------- #
# Audit-root materialisation: exactly what the real chain hands the fitter
# --------------------------------------------------------------------------------------- #
def _frame_name(frame: int) -> str:
    return f"f_{int(frame):04d}.jpg"


# --------------------------------------------------------------------------------------- #
# The striker witness: two moving players, written the way the real tracker writes them
# --------------------------------------------------------------------------------------- #
# `docs/wk1/point_fit4.md` section 5 measured what this bench used to hand the fitter: a
# `court_y` in a CENTRED frame (-10.285 / +10.285) where `reconstruction.py:load_players` and
# the real cohort both use 0..23.77 m, and a striker that never moved -- two `court_y` values
# across every point of every match, while the bench's own truth contacts run -1.92..25.74 m.
# The fitter reads that column as its contact-reach prior (`MAX_CONTACT_REACH_M`) and as the
# depth at which `rich_ball_physics.contact_prior` places the contact on its image ray, so the
# bench could neither reward using the striker nor punish ignoring it.
#
# What follows generates the striker the way the real chain produces one: a body that stands
# where the Hawk-Eye skeleton's hit position puts it (the hit is at RACKET REACH from the body),
# recovers and split-steps between shots, is projected to a native image box, and is turned back
# into `court_x`/`court_y` by the same box-bottom-centre-through-the-homography rule
# `cv/pipeline/player_side_association.py` uses -- so the written column carries the same root
# convention, and the same round-trip bias, that the real column carries.
#
# Every constant below is either measured on the real cohort
# (`processed/wk3_eventthreshold/default_cohort`, 46 broadcasts, 368 clip-sides) or a stated
# judgement about how a player moves; the measured ones name their number.
STRIKER_REACH_RANGE_M = (0.6, 1.2)
# A serve is struck above the body, so the body-to-contact distance in the COURT PLANE is small.
STRIKER_SERVE_REACH_RANGE_M = (0.15, 0.45)
# How much of the reach is lateral (out to the side) against behind (the ball is in front).
STRIKER_REACH_ANGLE_RANGE_DEG = (15.0, 70.0)
STRIKER_HOME_BEHIND_BASELINE_M = 0.90
STRIKER_RECOVERY_SECONDS = 0.40
STRIKER_RECOVERY_FRACTION = 0.45
STRIKER_SPLIT_STEP_SECONDS = 0.30
STRIKER_SPLIT_STEP_FRACTION = 0.25
STRIKER_APPROACH_FRACTION = 0.55
STRIKER_APPROACH_SECONDS = 0.80
# The receiver of the rally-closing flight chases the ball and pulls up about this far short.
STRIKER_CHASE_STOP_M = 1.20
# A player covers ground at a sprint at most; the plan is relaxed onto this before it is
# interpolated, so a recovery step is never a run the body could not make.
STRIKER_MAX_SPEED_MPS = 5.5
STRIKER_RELAX_PASSES = 6
# Body sway about the planned path: a smoothed random walk, not tracker noise.
STRIKER_SWAY_M = 0.04
STRIKER_SWAY_SECONDS = 0.45
# The projected body: two ground points under the feet and a torso box above them, so the image
# box bottom centre back-projects to the body's own court position (the tracker's root rule).
STRIKER_FOOT_HALF_WIDTH_M = 0.12
STRIKER_TORSO_HALF_WIDTH_M = 0.30
STRIKER_TORSO_HALF_DEPTH_M = 0.20
STRIKER_BODY_HEIGHT_M = 1.82
STRIKER_HIP_HEIGHT_M = 0.95

# Tracker noise on the player boxes, as its own rung parameter beside the ball-track,
# event-timing and missing-anchor sources.  Measured on the real cohort by fitting a local
# 9-frame quadratic to each tracked box edge and taking the residual: median absolute residual
# 0.24 / 0.19 / 0.24 / 0.14 px540 on x0/y0/x1/y1, so a robust sigma of about 0.35 px540, with a
# heavy tail (whole-box excursions when the tracker re-locks) that the standard deviation puts at
# 6 px540.  Detection coverage of the active span is a mean 0.979 with 399 gap runs whose median
# is 2 frames and whose p90 is 8.
STRIKER_NOISE_RUNGS = ("track_error", "realistic", "realistic_v2")
# Per edge (x0, y0, x1, y1), the robust sigma 1.4826 x the measured median absolute residual.
STRIKER_BOX_JITTER_PX = (0.35, 0.28, 0.35, 0.21)
STRIKER_BOX_JITTER_TAIL_RATE = 0.015
STRIKER_BOX_JITTER_TAIL_PX = 8.0
STRIKER_MISS_RATE = 0.021
STRIKER_MISS_RUN_MEAN_FRAMES = 3.5

# Pose keypoints, in the schema of the real cohort's `player_pose_tracked_crop_native_v1.csv`.
# Nothing in `NIGHTLY_ARGUMENTS` passes `--pose-artifact-name`, so the shipped default fitter
# does not read this file; it is written so the pose witness CAN be exercised (the `pose_witness`
# arm below does) and so the racket-hand wrist is on disk at the frame of every contact.
POSE_NAME = "player_pose_tracked_crop_native_v1.csv"
POSE_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
# The wrist holds the racket; the ball leaves the strings about this far beyond the hand.
RACKET_LENGTH_M = 0.40


def contact_side_of(contact: dict[str, Any]) -> str:
    """Which end of the court struck this contact, from the truth position alone."""
    return "near" if float(contact["xyz"][1]) < NET_Y_M else "far"


def _striker_home(side: str) -> np.ndarray:
    depth = (
        -STRIKER_HOME_BEHIND_BASELINE_M
        if side == "near"
        else COURT_LENGTH_M + STRIKER_HOME_BEHIND_BASELINE_M
    )
    return np.array([COURT_WIDTH_M / 2.0, depth], dtype=float)


def striker_contact_body(
    contact: dict[str, Any], side: str, rng: np.random.Generator
) -> np.ndarray:
    """Where the body stands for this hit: at racket reach from the contact, inside it.

    The racket reaches away from the body, so the body sits between the ball and the middle of
    the court, and behind it: the ball is struck in front of the player, toward the net.
    """
    xyz = np.asarray(contact["xyz"], dtype=float)
    low, high = (
        STRIKER_SERVE_REACH_RANGE_M if contact.get("phase") == "serve" else STRIKER_REACH_RANGE_M
    )
    reach = float(rng.uniform(low, high))
    angle = math.radians(float(rng.uniform(*STRIKER_REACH_ANGLE_RANGE_DEG)))
    lateral = reach * math.cos(angle)
    behind = reach * math.sin(angle)
    forward = 1.0 if side == "near" else -1.0
    inward = 1.0 if xyz[0] < COURT_WIDTH_M / 2.0 else -1.0
    return np.array([xyz[0] + inward * lateral, xyz[1] - forward * behind], dtype=float)


def relax_striker_keys(
    keys: Sequence[tuple[float, np.ndarray, bool]], fps: float
) -> list[tuple[float, np.ndarray]]:
    """Pull the free keyframes in until no leg of the plan needs a superhuman sprint.

    A contact keyframe is fixed -- it is the witness the whole file exists for -- so only the
    approach, split-step, recovery, chase and home keys move.  Each free key is projected back
    into the disc its two neighbours can reach it in, a few passes, which is enough because the
    plan is a short chain.
    """
    frames = [float(row[0]) for row in keys]
    positions = [np.asarray(row[1], dtype=float).copy() for row in keys]
    fixed = [bool(row[2]) for row in keys]
    for _ in range(STRIKER_RELAX_PASSES):
        moved = False
        for index in range(len(keys)):
            if fixed[index]:
                continue
            for other in (index - 1, index + 1):
                if not 0 <= other < len(keys):
                    continue
                seconds = abs(frames[index] - frames[other]) / max(fps, 1e-6)
                radius = max(0.05, STRIKER_MAX_SPEED_MPS * seconds)
                offset = positions[index] - positions[other]
                distance = float(np.linalg.norm(offset))
                if distance > radius:
                    positions[index] = positions[other] + offset / distance * radius
                    moved = True
        if not moved:
            break
    return list(zip(frames, positions, strict=True))


def _dedupe_keys(keys: Sequence[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(keys, key=lambda row: float(row[0]))
    frames: list[float] = []
    positions: list[np.ndarray] = []
    for frame, position in ordered:
        if frames and float(frame) - frames[-1] < 0.5:
            frames[-1] = float(frame)
            positions[-1] = np.asarray(position, dtype=float)
            continue
        frames.append(float(frame))
        positions.append(np.asarray(position, dtype=float))
    return np.asarray(frames, dtype=float), np.stack(positions)


def striker_keyframes(
    point: dict[str, Any],
    side: str,
    rng: np.random.Generator,
) -> list[tuple[float, np.ndarray, bool]]:
    """The plan: approach, split-step, hit, recover, and chase the last ball down.

    Each entry is ``(frame, court position, fixed)``; a contact keyframe is fixed because it is
    the striker witness itself, and everything else is free for :func:`relax_striker_keys`.
    """
    fps = float(point["fps"])
    frames = point["frames"]
    first, last = float(min(frames)), float(max(frames))
    home = _striker_home(side)
    contacts = [
        (float(row["frame"]), striker_contact_body(row, side, rng))
        for row in point["contacts"]
        if contact_side_of(row) == side
    ]
    keys: list[tuple[float, np.ndarray, bool]] = []
    if not contacts:
        # A side that never strikes still stands somewhere and still drifts.
        keys.append((first, home + np.array([float(rng.uniform(-2.0, 2.0)), 0.0]), False))
        keys.append((last, home + np.array([float(rng.uniform(-2.0, 2.0)), 0.0]), False))
        return keys
    first_frame, first_body = contacts[0]
    approach = first_body + (home - first_body) * STRIKER_APPROACH_FRACTION
    keys.append((min(first, first_frame - STRIKER_APPROACH_SECONDS * fps), approach, False))
    keys.append(
        (
            first_frame - STRIKER_SPLIT_STEP_SECONDS * fps,
            first_body + (approach - first_body) * STRIKER_SPLIT_STEP_FRACTION,
            False,
        )
    )
    for index, (frame, body) in enumerate(contacts):
        keys.append((frame, body, True))
        recovery = body + (home - body) * STRIKER_RECOVERY_FRACTION
        keys.append((frame + STRIKER_RECOVERY_SECONDS * fps, recovery, False))
        if index + 1 < len(contacts):
            next_frame, next_body = contacts[index + 1]
            keys.append(
                (
                    next_frame - STRIKER_SPLIT_STEP_SECONDS * fps,
                    next_body + (recovery - next_body) * STRIKER_SPLIT_STEP_FRACTION,
                    False,
                )
            )
            continue
        # This side's last hit.  If the other end struck the rally-closing shot, this player is
        # its receiver and runs at the ball; otherwise they recover and wait.
        if contact_side_of(point["contacts"][-1]) != side:
            termination = np.asarray(point["termination_xyz"], dtype=float)[:2]
            offset = home - termination
            norm = float(np.linalg.norm(offset))
            chase = (
                termination + offset / norm * STRIKER_CHASE_STOP_M if norm > 1e-6 else termination
            )
            keys.append((float(point["termination_frame"]), chase, False))
        keys.append((last, home, False))
    return keys


def striker_path(
    point: dict[str, Any],
    side: str,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """One court position per generated frame, interpolated through the plan and swayed."""
    from scipy.interpolate import PchipInterpolator

    fps = float(point["fps"])
    frames = np.asarray(sorted(int(frame) for frame in point["frames"]), dtype=float)
    key_frames, key_positions = _dedupe_keys(
        relax_striker_keys(striker_keyframes(point, side, rng), fps)
    )
    if len(key_frames) < 2:
        positions = np.repeat(key_positions[:1], len(frames), axis=0)
    else:
        # PCHIP and not a spline: a monotone interpolant does not overshoot a keyframe, so the
        # player never runs past the position the plan put them at and comes back.
        positions = np.stack(
            [PchipInterpolator(key_frames, key_positions[:, axis])(frames) for axis in (0, 1)],
            axis=1,
        )
    sway = _striker_sway(len(frames), fps, rng)
    positions = positions + sway
    return {int(frame): positions[index] for index, frame in enumerate(frames)}


def _striker_sway(count: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency body sway: white noise smoothed to about half a second."""
    if count == 0:
        return np.zeros((0, 2))
    sigma = max(1.0, STRIKER_SWAY_SECONDS * fps)
    width = int(math.ceil(3.0 * sigma))
    offsets = np.arange(-width, width + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= float(np.sqrt(np.sum(kernel**2)))
    raw = rng.normal(0.0, STRIKER_SWAY_M, size=(count + 2 * width, 2))
    return np.stack(
        [np.convolve(raw[:, axis], kernel, mode="valid")[:count] for axis in (0, 1)], axis=1
    )


def striker_box_native(projection: np.ndarray, body: Sequence[float]) -> np.ndarray | None:
    """The native-pixel bounding box of a standing body at this court position."""
    x, y = float(body[0]), float(body[1])
    points = [
        (x - STRIKER_FOOT_HALF_WIDTH_M, y, 0.0),
        (x + STRIKER_FOOT_HALF_WIDTH_M, y, 0.0),
    ]
    for dx in (-STRIKER_TORSO_HALF_WIDTH_M, STRIKER_TORSO_HALF_WIDTH_M):
        for dy in (-STRIKER_TORSO_HALF_DEPTH_M, STRIKER_TORSO_HALF_DEPTH_M):
            for height in (STRIKER_HIP_HEIGHT_M, STRIKER_BODY_HEIGHT_M):
                points.append((x + dx, y + dy, height))
    corners = [project(projection, value) for value in points]
    corners = [value for value in corners if np.all(np.isfinite(value))]
    if len(corners) < len(points):
        return None
    stacked = np.stack(corners)
    return np.array(
        [
            float(stacked[:, 0].min()),
            float(stacked[:, 1].min()),
            float(stacked[:, 0].max()),
            float(stacked[:, 1].max()),
        ]
    )


def ground_homography(projection: np.ndarray) -> np.ndarray | None:
    """Image pixels back to court metres on the z=0 plane, from the point's own camera."""
    world_to_image = np.asarray(projection, dtype=float)[:, [0, 1, 3]]
    try:
        return np.linalg.inv(world_to_image)
    except np.linalg.LinAlgError:
        return None


def box_root_court(homography: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """The tracker's root rule: the box bottom centre through the frame's homography.

    `cv/pipeline/player_side_association.py` writes `court_x`/`court_y` this way and says why
    (3.53 px540 median against the owner's corrected roots, against 11.98 for the ankle mean),
    so the synthetic column is built the same way and inherits the same convention.
    """
    foot = np.array([0.5 * (float(box[0]) + float(box[2])), float(box[3]), 1.0])
    world = np.asarray(homography, dtype=float) @ foot
    if abs(world[2]) < 1e-9:
        return np.array([math.nan, math.nan])
    return world[:2] / world[2]


def _striker_miss_frames(frames: Sequence[int], rng: np.random.Generator) -> set[int]:
    """Frames where the tracker has no box for this player, in runs, at the measured rate."""
    total = len(frames)
    missing: set[int] = set()
    if total == 0:
        return missing
    runs = int(rng.poisson(STRIKER_MISS_RATE * total / STRIKER_MISS_RUN_MEAN_FRAMES))
    for _ in range(runs):
        length = int(max(1, round(rng.exponential(STRIKER_MISS_RUN_MEAN_FRAMES))))
        start = int(rng.integers(0, total))
        missing.update(int(frames[index]) for index in range(start, min(total, start + length)))
    return missing


def striker_rows(
    host: HostPoint,
    point: dict[str, Any],
    *,
    rung: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Player-box rows, pose rows and a diagnostic record for one synthetic point.

    The RNG is drawn from the point key and never from the rung's own generator, so adding the
    striker leaves every other sampled quantity -- the ball track error, the event timing, the
    anchor pattern -- on exactly the stream it was on before.
    """
    frames = sorted(int(frame) for frame in point["frames"])
    noisy = rung in STRIKER_NOISE_RUNGS
    paths: dict[str, dict[int, np.ndarray]] = {}
    hands: dict[str, str] = {}
    for side in ("near", "far"):
        rng = np.random.default_rng(_mix(seed, "striker", point["point"], side))
        paths[side] = striker_path(point, side, rng)
        hands[side] = "right" if rng.random() < 0.85 else "left"
    noise_rng = np.random.default_rng(_mix(seed, "striker_noise", point["point"], rung))
    missing = {
        side: (_striker_miss_frames(frames, noise_rng) if noisy else set())
        for side in ("near", "far")
    }
    contacts_by_side: dict[str, list[dict[str, Any]]] = {"near": [], "far": []}
    for contact in point["contacts"]:
        contacts_by_side[contact_side_of(contact)].append(contact)
    # ``realistic_v2``: the tracker holds the WRONG BODY at some contacts.  ``apply_rung``
    # sampled which emitted contacts and how far the wrong body sits from the contact ray
    # (``docs/wk1/track_id.md``: 25 of 662, ray reach median 1.69 m, p90 19.78 m); here the
    # striking side's box is moved that far along the court's lateral axis for the contact
    # frame and its two neighbours, which is the window the fitter's striker witness reads.
    wrong_body: dict[int, dict[str, Any]] = {}
    for emitted_frame, reach in (point.get("striker_wrong_body") or {}).items():
        centre = int(emitted_frame)
        side = "near"
        nearest = min(
            point["contacts"],
            key=lambda row: abs(float(row["frame"]) - float(centre)),
            default=None,
        )
        if nearest is not None:
            side = contact_side_of(nearest)
        direction = 1.0 if noise_rng.random() < 0.5 else -1.0
        for offset in (-1, 0, 1):
            wrong_body[centre + offset] = {
                "side": side,
                "offset": np.array([direction * float(reach), 0.0], dtype=float),
            }

    box_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    root_errors: list[float] = []
    for frame in frames:
        projection = host_projection(host, float(frame))
        homography = ground_homography(projection)
        for side in ("near", "far"):
            if frame in missing[side]:
                continue
            body = paths[side].get(frame)
            if body is None:
                continue
            swap = wrong_body.get(frame)
            if swap is not None and swap["side"] == side:
                body = np.asarray(body, dtype=float) + swap["offset"]
            box = striker_box_native(projection, body)
            if box is None or homography is None:
                continue
            if noisy:
                box = box + noise_rng.normal(
                    0.0, 2.0 * np.asarray(STRIKER_BOX_JITTER_PX, dtype=float)
                )
                if noise_rng.random() < STRIKER_BOX_JITTER_TAIL_RATE:
                    box = box + noise_rng.normal(0.0, 2.0 * STRIKER_BOX_JITTER_TAIL_PX, size=4)
            court = box_root_court(homography, box)
            if not np.all(np.isfinite(court)):
                continue
            root_errors.append(float(np.linalg.norm(court - body)))
            box_rows.append(
                {
                    "clip": point["clip"],
                    "frame": _frame_name(frame),
                    "side": side,
                    "x0": round(float(box[0]) / 2.0, 3),
                    "y0": round(float(box[1]) / 2.0, 3),
                    "x1": round(float(box[2]) / 2.0, 3),
                    "y1": round(float(box[3]) / 2.0, 3),
                    "conf": 0.95,
                    "court_x": round(float(court[0]), 3),
                    "court_y": round(float(court[1]), 3),
                    "track_id": 0 if side == "near" else 1,
                }
            )
            pose_rows.append(
                _pose_row(
                    point,
                    side,
                    frame,
                    body=body,
                    box=box,
                    court=court,
                    projection=projection,
                    hand=hands[side],
                    contacts=contacts_by_side[side],
                )
            )
    diagnostic = {
        "point": point["point"],
        "rung": rung,
        "rows": len(box_rows),
        "missing_frames": {side: len(missing[side]) for side in ("near", "far")},
        "wrong_body_frames": len({frame for frame in wrong_body}),
        "root_error_m_median": float(np.median(root_errors)) if root_errors else None,
        "root_error_m_p90": float(np.percentile(root_errors, 90)) if root_errors else None,
        "court_y_range": [
            float(min(row["court_y"] for row in box_rows)),
            float(max(row["court_y"] for row in box_rows)),
        ]
        if box_rows
        else None,
    }
    return box_rows, pose_rows, diagnostic


# Keypoint offsets from the body's court position, in metres: (lateral, depth, height).
# Lateral is positive toward the player's own right as they face the net.
POSE_OFFSETS = {
    "nose": (0.00, 0.06, 1.70),
    "left_eye": (-0.04, 0.09, 1.73),
    "right_eye": (0.04, 0.09, 1.73),
    "left_ear": (-0.09, 0.02, 1.71),
    "right_ear": (0.09, 0.02, 1.71),
    "left_shoulder": (-0.20, 0.00, 1.50),
    "right_shoulder": (0.20, 0.00, 1.50),
    "left_elbow": (-0.30, 0.10, 1.20),
    "right_elbow": (0.30, 0.10, 1.20),
    "left_wrist": (-0.34, 0.20, 1.00),
    "right_wrist": (0.34, 0.20, 1.00),
    "left_hip": (-0.14, 0.00, 0.95),
    "right_hip": (0.14, 0.00, 0.95),
    "left_knee": (-0.15, 0.04, 0.50),
    "right_knee": (0.15, 0.04, 0.50),
    "left_ankle": (-0.13, 0.02, 0.09),
    "right_ankle": (0.13, 0.02, 0.09),
}
POSE_SWING_FRAMES = 4.0


def _pose_world(body: Sequence[float], side: str, name: str) -> np.ndarray:
    lateral, depth, height = POSE_OFFSETS[name]
    forward = 1.0 if side == "near" else -1.0
    return np.array(
        [float(body[0]) + forward * lateral, float(body[1]) + forward * depth, height],
        dtype=float,
    )


def _pose_row(
    point: dict[str, Any],
    side: str,
    frame: int,
    *,
    body: np.ndarray,
    box: np.ndarray,
    court: np.ndarray,
    projection: np.ndarray,
    hand: str,
    contacts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """One pose row in the real cohort's schema, with the racket hand at the contact.

    The racket-hand wrist is the keypoint the fitter's pose witness reads, so it is the one that
    has to be true: within a few frames of a contact it is placed one racket length back along
    the line from the shoulder to the ball, which is where the hand of a player who just hit
    that ball was.
    """
    world = {name: _pose_world(body, side, name) for name in POSE_KEYPOINTS}
    nearest = min(contacts, key=lambda row: abs(float(row["frame"]) - float(frame)), default=None)
    swing = 0.0
    if nearest is not None:
        swing = float(
            max(0.0, 1.0 - abs(float(nearest["frame"]) - float(frame)) / POSE_SWING_FRAMES)
        )
    if swing > 0.0 and nearest is not None:
        ball = np.asarray(nearest["xyz"], dtype=float)
        shoulder = world[f"{hand}_shoulder"]
        direction = ball - shoulder
        norm = float(np.linalg.norm(direction))
        hit = ball - direction / norm * RACKET_LENGTH_M if norm > 1e-6 else ball
        world[f"{hand}_wrist"] = world[f"{hand}_wrist"] * (1.0 - swing) + hit * swing
        elbow = world[f"{hand}_elbow"]
        world[f"{hand}_elbow"] = elbow * (1.0 - 0.5 * swing) + 0.5 * swing * (
            0.5 * (shoulder + world[f"{hand}_wrist"])
        )
    row: dict[str, Any] = {
        "clip": point["clip"],
        "frame": _frame_name(frame),
        "x0": round(float(box[0]), 2),
        "y0": round(float(box[1]), 2),
        "x1": round(float(box[2]), 2),
        "y1": round(float(box[3]), 2),
        "conf": 0.95,
    }
    for name in POSE_KEYPOINTS:
        pixel = project(projection, world[name])
        finite = bool(np.all(np.isfinite(pixel)))
        row[f"{name}_x"] = round(float(pixel[0]), 2) if finite else ""
        row[f"{name}_y"] = round(float(pixel[1]), 2) if finite else ""
        row[f"{name}_confidence"] = (
            (0.90 if not name.endswith("wrist") else 0.85) if finite else 0.0
        )
    row.update(
        {
            "court_x": round(float(court[0]), 3),
            "court_y": round(float(court[1]), 3),
            "side": side,
            "track_id": 0 if side == "near" else 1,
            "n_variants": 1,
            "track_x0": round(float(box[0]), 1),
            "track_y0": round(float(box[1]), 1),
            "track_x1": round(float(box[2]), 1),
            "track_y1": round(float(box[3]), 1),
        }
    )
    return row


# --------------------------------------------------------------------------------------- #
# Emissions: on a frame, at the track's own pixel, the way the real chain emits
# --------------------------------------------------------------------------------------- #
# ``docs/wk1/point_fit6.md`` measured the defect this replaces: 65.5% of this bench's contact
# emissions sat at a FRACTIONAL frame -- the generator's own continuous contact time -- against
# 0.0% of the real cohort's 792 contacts.  The real chain cannot do that.  Its event model emits
# a physical event ON a native frame, and ``cv/pipeline/reconstruction.py`` then hands that frame
# to ``contact_frame_refiner.refine``, which moves the anchor onto the image where the composed
# track turns its sharpest corner and answers with THAT image's own track pixel
# (``docs/wk1/contactframe.md``).  So that is what the bench emits: the refiner's own rule, run
# on the bench's own track, seeded at the frame nearest the true continuous event time.
#
# The true fractional time stays in ``truth.json`` and is what the audit scores against; the
# difference between the two is the bench's own timing error and is written out per rung as
# ``emission_timing_v1.json``.
EMISSION_FRAME_TYPES = ("contact", "bounce", "point_end")
"""Which emissions leave the generator on a native frame.  All three, by default.

The tuple is a diagnostic knob and not a supported bench variant: it exists so this package
could measure the contact half of the change apart from the bounce half.  Anything left out
keeps the pre-``benchframes`` behaviour -- the generator's own continuous event time, with the
true event pixel -- which ``docs/wk1/point_fit6.md`` measured as a defect of the bench.
"""

EMISSION_SNAP_FRAMES = 3
"""How far an emission may move to reach a frame the track actually has a row on."""

EMISSION_TERMINAL_RIDE_FRAMES = 2.0
"""How near the termination a surviving bounce emission must be for the ending to ride it."""

EMISSION_MIN_CONTACT_GAP_FRAMES = 4.0
"""Two contact emissions closer than this are pushed apart, as the old timing rung did."""


# --------------------------------------------------------------------------------------- #
# ``realistic_v2``: the rung whose parameters are the real cohort's own emission statistics.
#
# ``docs/wk1/bench_frames.md`` section 8 lists what the frame-emitting bench still gets wrong
# about a real emission, and ``docs/wk1/point_fit8.md`` section 1 shows the cost of that: the
# ``realistic`` rung and the real cohort disagree about which fitter arm is better.  Every
# parameter below is one measured real statistic
# (``cv.validation.s6_bench.measure_emission_realism``), applied to the ``realistic`` rung's
# own corruption.  ``realistic`` (v1) is untouched and stays available.
# --------------------------------------------------------------------------------------- #
REALISTIC_V2 = "realistic_v2"
REALISTIC_RUNGS = ("realistic", REALISTIC_V2)
EMISSION_REALISM_RELATIVE = Path("processed/wk3_benchreal/real_emission_model_v1.json")

REALISTIC_V2_PARAMETERS = {
    "emission_pixel_head": (
        "The emitted pixel is the event model's x/y head, not the track: a bootstrap of the "
        "measured head-minus-track offset on the emitted frame, per event type and court side "
        "(real contacts near 5.71 px median / 28.50 p90, far 3.62 / 9.32; bounces 2.93 / 6.51 "
        "near and 2.84 / 5.27 far).  The bench emitted the track pixel exactly."
    ),
    "emitted_frame_offset": (
        "The emitted frame minus the true event time is drawn from the measured real "
        "distribution per event type and court side (contacts 0.34 frames absolute mean near "
        "and 0.305 far; bounces 0.305 near and 0.195 far) and applied as a whole-frame "
        "residual on top of the refiner's own choice, instead of adding the pooled contact "
        "timing error to it."
    ),
    "off_track_frame_rate": (
        "An emission may sit on a frame the track has no row on only at the measured real rate "
        "(contacts 0.38%, bounces 0.14%, point endings 0%); otherwise it snaps to the nearest "
        "tracked frame."
    ),
    "contact_track_gap": (
        "The track's interior gaps within five frames of a contact are held at the measured "
        "real rate and length instead of wherever the dropout sampler put them."
    ),
    "missing_bounce_emission": (
        "A flight's bounce emissions are deleted at the measured real per-flight rate (8.07% "
        "of rally flights, 31.89% of terminal spans) instead of at the anchor-report rate."
    ),
    "camera_abstention": (
        "The camera stage abstains on the measured real share of points (3.76%): the rung's "
        "own camera artifact has no rows for them, so the fitter holds them exactly as it "
        "holds a real point whose camera abstained."
    ),
    "striker_wrong_body": (
        "The sided player box the fitter reads at a contact is the wrong body at the measured "
        "real rate (3.78% of contacts with a box), displaced to reproduce the measured "
        "ray-to-striker reach of a wrong body (median 1.69 m, p90 19.78 m)."
    ),
}
"""Every ``realistic_v2`` parameter, and the measured real statistic behind it."""


def realism_emission_choice(
    track: dict[int, Any],
    true_frame: float,
    truth_pixel: Sequence[float],
    *,
    event_type: str,
    side: str,
    realism: Any,
    rng: np.random.Generator,
    fallback_pixels: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """One emission, with the real cohort's own frame error, snapping rule and x/y head."""
    seed = int(round(float(true_frame)))
    refined = contact_frame_refiner.refine(track, seed)
    frame = int(refined.frame)
    target = realism.sample_frame_offset(rng, event_type, side)
    shift = _round_away_from_zero(float(target) - (float(frame) - float(true_frame)))
    frame += shift
    off_track_allowed = realism.allow_off_track_frame(rng, event_type)
    head = realism.sample_pixel_head(rng, event_type, side)
    pixel_source = "track"
    pixel: list[float] | None = None
    if frame in track:
        pixel = [float(value) for value in track[frame]]
    elif off_track_allowed and fallback_pixels and frame in fallback_pixels:
        pixel = [float(value) for value in fallback_pixels[frame]]
        pixel_source = "projection_no_track_row"
    else:
        for delta in range(1, EMISSION_SNAP_FRAMES + 1):
            for candidate in (frame - delta, frame + delta):
                if candidate in track:
                    frame, pixel, pixel_source = (
                        candidate,
                        [float(value) for value in track[candidate]],
                        "track_snapped",
                    )
                    break
            if pixel is not None:
                break
    if pixel is None:
        if fallback_pixels and frame in fallback_pixels:
            pixel = [float(value) for value in fallback_pixels[frame]]
            pixel_source = "projection_no_track_row"
        else:
            pixel = [float(value) for value in truth_pixel]
            pixel_source = "truth_projection"
    pixel = [float(pixel[0]) + float(head[0]), float(pixel[1]) + float(head[1])]
    turn = refined.votes.get("direction_change")
    return {
        "frame": int(frame),
        "pixel": [float(pixel[0]), float(pixel[1])],
        "pixel_source": f"{pixel_source}+xy_head",
        "truth_frame": float(true_frame),
        "offset_frames": float(frame) - float(true_frame),
        "timing_shift_frames": float(target),
        "timing_shift_applied_frames": int(shift),
        "refiner_reason": str(refined.reason),
        "refiner_offset": int(refined.offset),
        "refiner_abstain": bool(refined.abstain),
        "refiner_sigma_px": float(refined.sigma_px),
        "turn_offset": None if turn is None else int(turn),
        "pixel_error_px": float(
            math.hypot(pixel[0] - float(truth_pixel[0]), pixel[1] - float(truth_pixel[1]))
        ),
        "head_offset_px": [float(head[0]), float(head[1])],
        "court_side": str(side),
        "spacing_adjusted": False,
    }


def _realism_contact_gaps(
    frames: Sequence[int],
    dropped: np.ndarray,
    contact_frames: Sequence[float],
    realism: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hold the track's gaps near a contact at the measured real rate and length.

    The dropout sampler places its runs uniformly over a point, so on the ``realistic`` rung
    51.5% of contacts have a missing interior track row within five frames against the real
    cohort's 1.4%.  This restores the dropped frames around each contact and then re-cuts a
    gap of a measured length at the measured rate.
    """
    index = {int(frame): position for position, frame in enumerate(frames)}
    output = np.array(dropped, dtype=bool)
    for contact in contact_frames:
        centre = int(round(float(contact)))
        window = [
            index[centre + offset]
            for offset in range(-CONTACT_GAP_WINDOW_FRAMES, CONTACT_GAP_WINDOW_FRAMES + 1)
            if centre + offset in index
        ]
        if not window:
            continue
        output[window] = False
        length = realism.sample_contact_gap(rng)
        if length <= 0:
            continue
        start = int(rng.integers(0, len(window)))
        for position in window[start : start + int(length)]:
            output[position] = True
    return output


CONTACT_GAP_WINDOW_FRAMES = 5
"""The window the real contact-gap statistic was measured over, in native frames."""


def _round_away_from_zero(value: float) -> int:
    """Round a half-frame timing error to whole frames without deleting the halves.

    The measured real timing errors are multiples of 0.5 frames (they are an integer emitted
    frame minus the owner's half-frame leading-blur label), so rounding halves toward zero would
    silently drop most of the distribution.
    """
    return int(math.floor(abs(float(value)) + 0.5)) * (1 if value >= 0.0 else -1)


def _emission_pixel(
    track: dict[int, Any],
    fallback_pixels: dict[int, Any] | None,
    frame: int,
) -> tuple[int, list[float] | None, str]:
    """The track's own pixel on ``frame``, or the nearest frame the track has a row on."""
    if frame in track:
        return frame, [float(value) for value in track[frame]], "track"
    if fallback_pixels and frame in fallback_pixels:
        # No tracked row on this frame (a dropped frame, or the net-crossing blank).  An emitter
        # still has the image, so its own estimate on the frame it emitted stands in -- which is
        # closer to the event than a tracked row two or three frames away would be.
        return frame, [float(value) for value in fallback_pixels[frame]], "projection_no_track_row"
    for delta in range(1, EMISSION_SNAP_FRAMES + 1):
        for candidate in (frame - delta, frame + delta):
            if candidate in track:
                return candidate, [float(value) for value in track[candidate]], "track_snapped"
    return frame, None, "no_pixel"


def emission_choice(
    track: dict[int, Any],
    true_frame: float,
    truth_pixel: Sequence[float],
    *,
    shift_frames: float = 0.0,
    fallback_pixels: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Which frame and pixel one physical event is emitted on.

    ``true_frame`` is the generator's own continuous event time and never leaves the truth file.
    ``shift_frames`` is the measured real event-timing error, applied on top in whole frames.
    """
    seed = int(round(float(true_frame)))
    refined = contact_frame_refiner.refine(track, seed)
    frame = int(refined.frame)
    shift = _round_away_from_zero(shift_frames)
    frame += shift
    frame, pixel, pixel_source = _emission_pixel(track, fallback_pixels, frame)
    if pixel is None:
        pixel, pixel_source = [float(value) for value in truth_pixel], "truth_projection"
    turn = refined.votes.get("direction_change")
    return {
        "frame": int(frame),
        "pixel": [float(pixel[0]), float(pixel[1])],
        "pixel_source": pixel_source,
        "truth_frame": float(true_frame),
        "offset_frames": float(frame) - float(true_frame),
        "timing_shift_frames": float(shift_frames),
        "timing_shift_applied_frames": int(shift),
        "refiner_reason": str(refined.reason),
        "refiner_offset": int(refined.offset),
        "refiner_abstain": bool(refined.abstain),
        "refiner_sigma_px": float(refined.sigma_px),
        "turn_offset": None if turn is None else int(turn),
        "pixel_error_px": float(
            math.hypot(pixel[0] - float(truth_pixel[0]), pixel[1] - float(truth_pixel[1]))
        ),
        "spacing_adjusted": False,
    }


def legacy_emission_choice(true_frame: float, truth_pixel: Sequence[float]) -> dict[str, Any]:
    """The pre-``benchframes`` emission: the generator's continuous time and the true pixel."""
    return {
        "frame": float(true_frame),
        "pixel": [float(truth_pixel[0]), float(truth_pixel[1])],
        "pixel_source": "truth_projection",
        "truth_frame": float(true_frame),
        "offset_frames": 0.0,
        "timing_shift_frames": 0.0,
        "timing_shift_applied_frames": 0,
        "refiner_reason": "legacy_truth_time",
        "refiner_offset": 0,
        "refiner_abstain": False,
        "refiner_sigma_px": 0.0,
        "turn_offset": None,
        "pixel_error_px": 0.0,
        "spacing_adjusted": False,
    }


def _respace_contacts(
    rows: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    track: dict[int, Any],
    fallback_pixels: dict[int, Any] | None,
) -> None:
    """Keep two contact emissions at least four frames apart, as the timing rung always did."""
    order = sorted(range(len(rows)), key=lambda index: rows[index]["frame"])
    for previous, current in zip(order, order[1:], strict=False):
        gap = float(rows[current]["frame"]) - float(rows[previous]["frame"])
        if gap >= EMISSION_MIN_CONTACT_GAP_FRAMES:
            continue
        frame = int(rows[previous]["frame"] + EMISSION_MIN_CONTACT_GAP_FRAMES)
        choice = choices[current]
        frame, pixel, pixel_source = _emission_pixel(track, fallback_pixels, frame)
        if pixel is None:
            pixel, pixel_source = list(rows[current]["pixel"]), "previous_emission"
        rows[current]["frame"] = float(frame)
        rows[current]["pixel"] = list(pixel)
        choice.update(
            {
                "frame": int(frame),
                "pixel": [float(pixel[0]), float(pixel[1])],
                "pixel_source": pixel_source,
                "offset_frames": float(frame) - float(choice["truth_frame"]),
                "spacing_adjusted": True,
            }
        )


def _net_crossing_frames(point: dict[str, Any]) -> list[float]:
    frames = sorted(point["positions"])
    crossings = []
    for lower, upper in zip(frames, frames[1:], strict=False):
        if int(upper) - int(lower) > 2:
            # An out-of-frame gap: the crossing inside it is not observable, so do not invent it.
            continue
        first = float(point["positions"][lower][1]) - NET_Y_M
        second = float(point["positions"][upper][1]) - NET_Y_M
        if first * second < 0.0:
            fraction = first / (first - second)
            crossings.append(float(lower) + fraction * float(upper - lower))
    return crossings


def apply_rung(
    point: dict[str, Any],
    rung: str,
    samplers: Any,
    rng: np.random.Generator,
    *,
    emission_frame_types: Sequence[str] = EMISSION_FRAME_TYPES,
    realism: Any | None = None,
) -> dict[str, Any]:
    """Corrupt one clean synthetic point with the s6_bench empirical noise model.

    ``realism`` is the measured real-emission model (``s6_bench.EmissionRealismSamplers``).
    It is required by ``realistic_v2`` and ignored by every other rung, so the five original
    rungs draw from exactly the streams they always drew from.
    """
    if rung == REALISTIC_V2 and realism is None:
        raise ValueError("realistic_v2 needs the measured real emission model")
    frames = np.asarray(point["frames"], dtype=int)
    pixels = np.stack([np.asarray(point["track_pixels"][int(frame)]) for frame in frames])
    sides = [
        "near" if float(point["positions"][int(frame)][1]) < NET_Y_M else "far" for frame in frames
    ]
    contact_frames = [float(row["frame"]) for row in point["contacts"]]
    contact_mask = np.asarray(
        [any(abs(float(frame) - value) <= 2.0 for value in contact_frames) for frame in frames],
        dtype=bool,
    )
    contacts = [dict(row) for row in point["contacts"]]
    bounces = [dict(row) for row in point["bounces"]]
    dropped = np.zeros(len(frames), dtype=bool)
    blanked: set[int] = set()

    if rung in {"track_error", *REALISTIC_RUNGS}:
        errors, _ = samplers.sample_track_errors(rng, sides, contact_only=False)
        pixels = pixels + errors
        contact_errors, _ = samplers.sample_track_errors(rng, sides, contact_only=True)
        pixels[contact_mask] = (
            pixels[contact_mask] - errors[contact_mask] + contact_errors[contact_mask]
        )
    if rung in REALISTIC_RUNGS:
        for start, offsets in samplers.sample_wrong_object_arcs(rng, len(frames)):
            stop = min(len(frames), start + len(offsets))
            pixels[start:stop] += offsets[: stop - start]
        dropped = samplers.sample_dropout_mask(rng, len(frames))
        pixels = pixels + samplers.sample_camera_jitter(rng, [False] * len(frames))
    if rung == REALISTIC_V2:
        # The measured real cohort has almost no track gap near a contact; the dropout
        # sampler puts one within five frames of half of them.
        dropped = _realism_contact_gaps(frames, dropped, contact_frames, realism, rng)
    shifts: np.ndarray | None = None
    if rung in {"event_timing", "realistic"}:
        # Drawn here, in this order, so every rung's noise stream is where it always was; the
        # shift is applied to the emitted FRAME below, once the track it is read off exists.
        shifts = samplers.sample_event_timing(rng, len(contacts) + len(bounces))
    if rung in {"missing_anchors", *REALISTIC_RUNGS}:
        crossings = _net_crossing_frames(point)
        for flight in point["flights"]:
            pattern = set(samplers.sample_anchor_pattern(rng))
            drop_bounce = "bounce" not in pattern
            if rung == REALISTIC_V2:
                # The anchor report's 17.4% is a rate over automatic flight ATTEMPTS; the
                # measured rate over the cohort's own emitted flights is 8.07%, and 31.89%
                # over terminal spans, so realistic_v2 uses those instead.
                drop_bounce = realism.drop_bounce_emission(rng, terminal=bool(flight["terminal"]))
            if drop_bounce:
                bounces = [
                    row
                    for row in bounces
                    if not (flight["start_frame"] < float(row["frame"]) < flight["end_frame"])
                ]
            if "net_crossing" not in pattern:
                for crossing in crossings:
                    if flight["start_frame"] < crossing < flight["end_frame"]:
                        blanked.update(
                            int(value)
                            for value in range(
                                int(math.floor(crossing)) - 3, int(math.ceil(crossing)) + 4
                            )
                        )

    track = {
        int(frame): pixel
        for frame, pixel, drop in zip(frames, pixels, dropped, strict=True)
        if not drop and int(frame) not in blanked
    }
    # Every emission now leaves the generator the way the real chain emits one: on a native
    # frame, carrying the track's own pixel on that frame.  Nothing here reads the continuous
    # contact time except as the seed the event model would have been aiming at.
    fallback_pixels = {int(frame): value for frame, value in point["track_pixels"].items()}
    emission_rows: list[dict[str, Any]] = []
    contact_choices: list[dict[str, Any]] = []
    for index, contact in enumerate(contacts):
        if rung == REALISTIC_V2:
            choice = realism_emission_choice(
                track,
                float(contact["frame"]),
                contact["pixel"],
                event_type="contact",
                side=contact_side_of(contact),
                realism=realism,
                rng=rng,
                fallback_pixels=fallback_pixels,
            )
        elif "contact" in emission_frame_types:
            choice = emission_choice(
                track,
                float(contact["frame"]),
                contact["pixel"],
                shift_frames=0.0 if shifts is None else float(shifts[index]),
                fallback_pixels=fallback_pixels,
            )
        else:
            shift = 0.0 if shifts is None else float(shifts[index])
            choice = legacy_emission_choice(float(contact["frame"]) + shift, contact["pixel"])
            choice["timing_shift_frames"] = shift
            choice["offset_frames"] = shift
            choice["truth_frame"] = float(contact["frame"])
        contact["frame"] = float(choice["frame"])
        contact["pixel"] = list(choice["pixel"])
        contact_choices.append({"event_type": "contact", "phase": contact.get("phase"), **choice})
    if "contact" in emission_frame_types:
        _respace_contacts(contacts, contact_choices, track, fallback_pixels)
    else:
        for lower, upper in zip(contacts, contacts[1:], strict=False):
            if upper["frame"] - lower["frame"] < EMISSION_MIN_CONTACT_GAP_FRAMES:
                upper["frame"] = lower["frame"] + EMISSION_MIN_CONTACT_GAP_FRAMES
    emission_rows.extend(contact_choices)
    for index, bounce in enumerate(bounces):
        if rung == REALISTIC_V2:
            choice = realism_emission_choice(
                track,
                float(bounce["frame"]),
                bounce["pixel"],
                event_type="bounce",
                side="near" if float(bounce["xyz"][1]) < NET_Y_M else "far",
                realism=realism,
                rng=rng,
                fallback_pixels=fallback_pixels,
            )
        elif "bounce" in emission_frame_types:
            choice = emission_choice(
                track,
                float(bounce["frame"]),
                bounce["pixel"],
                shift_frames=0.0 if shifts is None else float(shifts[len(contacts) + index]),
                fallback_pixels=fallback_pixels,
            )
        else:
            shift = 0.0 if shifts is None else float(shifts[len(contacts) + index])
            choice = legacy_emission_choice(float(bounce["frame"]) + shift, bounce["pixel"])
            choice["timing_shift_frames"] = shift
            choice["offset_frames"] = shift
            choice["truth_frame"] = float(bounce["frame"])
        bounce["frame"] = float(choice["frame"])
        bounce["pixel"] = list(choice["pixel"])
        emission_rows.append({"event_type": "bounce", "phase": None, **choice})
    contacts.sort(key=lambda row: row["frame"])
    bounces.sort(key=lambda row: row["frame"])

    # The point ending, in the runner's event format, at the termination.  When the termination
    # is a bounce, it rides on that bounce's emission and therefore inherits its frame and pixel.
    # An ``out_of_view`` ending has no event under it, so it is emitted on the frame nearest the
    # exit with the ball's own image position there, clamped to the frame edge.
    termination_frame = (
        int(round(float(point["termination_frame"])))
        if "point_end" in emission_frame_types
        else float(point["termination_frame"])
    )
    ending = {
        "frame": float(termination_frame),
        "pixel": list(point["termination_pixel"]),
        "kind": str(point["termination_kind"]),
        "terminal_event_type": "bounce",
        "source": "s6_point_bench_termination",
    }
    if point["termination_kind"] == "out_of_view":
        ending["terminal_event_type"] = "track_exit"
        ending["source"] = "s6_point_bench_termination_clamped_to_frame_edge"
    else:
        anchor = min(
            bounces,
            key=lambda row: abs(float(row["frame"]) - float(point["termination_frame"])),
            default=None,
        )
        if rung == REALISTIC_V2:
            # The real event model emits the point ending from its own head, so a deleted
            # terminal bounce does not drag the ending to a bounce hundreds of frames away
            # (``docs/wk1/bench_frames.md`` section 8): the real cohort's endings sit a
            # median of 0 and a p90 of 22 frames from the nearest bounce EMISSION while
            # still marking the point's own end.  So the ending rides the terminal bounce
            # only when that bounce survived, and is emitted on its own otherwise.
            if (
                anchor is not None
                and abs(float(anchor["frame"]) - float(point["termination_frame"]))
                > EMISSION_TERMINAL_RIDE_FRAMES
            ):
                anchor = None
            if anchor is None:
                side = "near" if float(point["termination_xyz"][1]) < NET_Y_M else "far"
                choice = realism_emission_choice(
                    track,
                    float(point["termination_frame"]),
                    point["termination_pixel"],
                    event_type="bounce",
                    side=side,
                    realism=realism,
                    rng=rng,
                    fallback_pixels=fallback_pixels,
                )
                ending["frame"] = float(choice["frame"])
                ending["pixel"] = list(choice["pixel"])
                ending["source"] = "s6_point_bench_termination_emitted"
        if anchor is not None:
            ending["frame"] = float(anchor["frame"])
            ending["pixel"] = list(anchor["pixel"])
    emission_rows.append(
        {
            "event_type": "point_end",
            "phase": None,
            "frame": int(ending["frame"]),
            "pixel": list(ending["pixel"]),
            "pixel_source": "termination_anchor",
            "truth_frame": float(point["termination_frame"]),
            "offset_frames": float(ending["frame"]) - float(point["termination_frame"]),
            "timing_shift_frames": 0.0,
            "timing_shift_applied_frames": 0,
            "refiner_reason": "termination",
            "refiner_offset": 0,
            "refiner_abstain": False,
            "refiner_sigma_px": 0.0,
            "turn_offset": None,
            "pixel_error_px": float(
                math.hypot(
                    float(ending["pixel"][0]) - float(point["termination_pixel"][0]),
                    float(ending["pixel"][1]) - float(point["termination_pixel"][1]),
                )
            ),
            "spacing_adjusted": False,
        }
    )
    for row in emission_rows:
        row["on_observed_track_frame"] = bool(int(row["frame"]) in track)

    camera_abstained = False
    wrong_body: dict[int, float] = {}
    if rung == REALISTIC_V2:
        camera_abstained = realism.camera_abstains(rng)
        for contact in contacts:
            reach = realism.striker_wrong_body(rng)
            if reach is not None:
                wrong_body[int(round(float(contact["frame"])))] = float(reach)
    return {
        **point,
        "track": track,
        "emission_contacts": contacts,
        "emission_bounces": bounces,
        "emission_point_end": ending,
        "emission_rows": emission_rows,
        "camera_abstained": bool(camera_abstained),
        "striker_wrong_body": wrong_body,
    }


def _write_camera_without(source: Path, target: Path, clips: set[str]) -> None:
    """Copy a camera artifact with every row of ``clips`` removed."""
    numbers = {int(str(clip).removeprefix("pt")) for clip in clips}
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    if "clips" in arrays:
        keep = np.asarray([str(value) not in clips for value in arrays["clips"]], dtype=bool)
        arrays = {
            name: (value[keep] if value.shape[:1] == keep.shape else value)
            for name, value in arrays.items()
        }
    elif "pts" in arrays:
        keep = np.asarray([int(value) not in numbers for value in arrays["pts"]], dtype=bool)
        arrays = {
            name: (value[keep] if value.shape[:1] == keep.shape else value)
            for name, value in arrays.items()
        }
    np.savez(target, **arrays)


def materialize(
    root: Path,
    hosts: dict[str, HostPoint],
    points: Sequence[dict[str, Any]],
    *,
    camera_root: Path,
    rung: str = "clean",
    seed: int = SEED,
    emission_frame_types: Sequence[str] = EMISSION_FRAME_TYPES,
) -> dict[str, Any]:
    """Write one complete audit root for one rung."""
    root.mkdir(parents=True, exist_ok=True)
    striker_diagnostics: list[dict[str, Any]] = []
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_match[point["match_id"]].append(point)

    matches = []
    active: dict[str, Any] = {}
    tracking_rows = []
    validity_rows = []
    emissions: list[dict[str, Any]] = []
    emission_rows: list[dict[str, Any]] = []
    for match_id in sorted(by_match):
        rows = sorted(by_match[match_id], key=lambda row: row["clip"])
        host = hosts[rows[0]["host"]]
        match_dir = root / match_id
        match_dir.mkdir(parents=True, exist_ok=True)
        matches.append(
            {
                "id": match_id,
                "cohort": "wk3_synthpoint_v1",
                "point_ids": [int(row["clip"].removeprefix("pt")) for row in rows],
                "source_fps": host.fps,
                "surface": host.surface,
            }
        )
        (match_dir / "audit_frames_native_1080.coordinates.json").write_text(
            json.dumps(
                {
                    "artifact_size": {"height": 1080, "width": 1920},
                    "fps": host.fps,
                    "frames_dir": "audit_frames_native_1080",
                    "image_size": {"height": 1080, "width": 1920},
                    "schema": "tennis.coordinate-space.v1",
                    "source": "cv.validation.s6_point_bench synthetic point",
                },
                indent=1,
                sort_keys=True,
            )
        )
        abstained = {row["clip"] for row in rows if row.get("camera_abstained")}
        for name in (CAMERA_NAME, "court_H_per_point.npz", "court_H_per_frame_v1.npz"):
            source = camera_root / match_id / name
            target = match_dir / name
            if not source.is_file():
                continue
            if target.exists() or target.is_symlink():
                target.unlink()
            if abstained:
                # The camera stage abstained on these points, exactly as it abstains on
                # 3.76% of the real cohort's: the rung's own artifact simply has no rows for
                # them, and ``reconstruction.py`` holds the point for
                # ``camera_projection_unavailable``.  The truth camera under ``cameras/`` is
                # untouched, so the audit can still score what the fitter could not fit.
                _write_camera_without(source, target, abstained)
            else:
                target.symlink_to(source)
        track_rows = []
        box_rows = []
        pose_rows = []
        for point in rows:
            point_host = hosts[point["host"]]
            frames = sorted(point["track"])
            active[f"{match_id}/{point['clip']}"] = {
                "active_spans": [[float(min(point["frames"])), float(max(point["frames"]))]],
                "clip": point["clip"],
                "fps": point_host.fps,
                "point_valid": True,
                "gate_held": False,
                "n_frames": len(point["frames"]),
                "reasons": [],
                "shots": point["shots"],
            }
            tracking_rows.append(
                {
                    "match_id": match_id,
                    "clip": point["clip"],
                    "decision": "retain",
                    "frames": len(point["frames"]),
                    "arc_retained_frames": len(frames),
                }
            )
            validity_rows.append(
                {
                    "match_id": match_id,
                    "clip": point["clip"],
                    "point": f"{match_id}/{point['clip']}",
                    "decision": "retain",
                    "reasons": [],
                    "active_play_valid": True,
                    "court_geometry_valid": True,
                    "tracking_decision": "retain",
                }
            )
            for frame in frames:
                pixel = point["track"][frame]
                track_rows.append(
                    {
                        "clip": point["clip"],
                        "frame": _frame_name(frame),
                        "x": round(float(pixel[0]) / 2.0, 5),
                        "y": round(float(pixel[1]) / 2.0, 5),
                        "track_id": 0,
                        "score": 0.9,
                        "rank": 0,
                        "sources": "wasb+tracknetv2",
                        "regime": "ballistic",
                        "regime_probability": 0.95,
                        "innovation_mahalanobis": 0.0,
                        "confidence": 0.9,
                        "detector_support": 2,
                        "homography_source": "court_H_per_frame_v1.npz",
                        "x_native": round(float(pixel[0]), 5),
                        "y_native": round(float(pixel[1]), 5),
                    }
                )
            point_boxes, point_pose, striker_diagnostic = striker_rows(
                point_host, point, rung=rung, seed=seed
            )
            box_rows.extend(point_boxes)
            pose_rows.extend(point_pose)
            striker_diagnostics.append(striker_diagnostic)
            for row in point["emission_contacts"]:
                emissions.append(_emission(match_id, point, "contact", row))
            for row in point["emission_bounces"]:
                emissions.append(_emission(match_id, point, "bounce", row))
            emissions.append(_point_end_emission(match_id, point))
            for row in point.get("emission_rows") or ():
                emission_rows.append({"point": point["point"], **row})
        write_declared_csv(match_dir / TRACK_NAME, track_rows, kind="track", fps=host.fps)
        write_declared_csv(
            match_dir / f"player_boxes_{int(round(host.fps))}_native_sided_v1.csv",
            box_rows,
            kind="boxes",
            fps=host.fps,
        )
        write_declared_csv(match_dir / POSE_NAME, pose_rows, kind="pose", fps=host.fps)

    write_json(
        root / "manifest.json",
        {
            "schema": "wk3_synthpoint_manifest_v1",
            "matches": matches,
            "points_per_match": {row["id"]: len(row["point_ids"]) for row in matches},
        },
    )
    write_json(root / "active_play_v1.json", active)
    write_json(
        root / "untouched_tracking_point_gate_v1.json",
        {
            "schema": "wk3_synthpoint_tracking_gate_v1",
            "labels_loaded": False,
            "rows": tracking_rows,
        },
    )
    write_json(
        root / "point_validity_gate_v1.json",
        {"schema": "wk3_synthpoint_validity_v1", "labels_loaded": False, "rows": validity_rows},
    )
    emissions.sort(key=lambda row: (row["clip"], row["frame"]))
    write_json(root / EVENT_NAME, emissions)
    write_json(
        root / "emission_timing_v1.json",
        {
            "schema": "tennis.s6-point-bench-emission-timing.v1",
            "rung": rung,
            "definition": EMISSION_DEFINITION,
            "emission_frame_types": list(emission_frame_types),
            "summary": emission_timing_summary(emission_rows),
            "rows": emission_rows,
        },
    )
    write_json(
        root / "striker_witness_v1.json",
        {
            "schema": "tennis.s6-point-bench-striker.v1",
            "rung": rung,
            "noise_applied": rung in STRIKER_NOISE_RUNGS,
            "constants": _striker_constants(),
            "points": striker_diagnostics,
        },
    )
    return {
        "matches": len(matches),
        "points": len(points),
        "emissions": len(emissions),
        "striker": _striker_summary(striker_diagnostics),
        "emission_timing": emission_timing_summary(emission_rows),
    }


EMISSION_DEFINITION = {
    "frame": (
        "Integer native frame only.  Seeded at the frame nearest the generator's continuous "
        "event time and then moved by cv/pipeline/contact_frame_refiner.refine -- the shipped "
        "rule: the window image with the largest track direction change, gated at 45 degrees."
    ),
    "pixel": "The rung's own track pixel on the emitted frame; never an interpolated position.",
    "timing_rungs": (
        "event_timing and realistic add the measured real event-timing error on top, rounded "
        "away from zero to whole frames so the emission stays on a frame."
    ),
    "truth": "truth.json keeps the continuous event time and the true contact pixel; the audit "
    "scores against those, not against the emission.",
}


def emission_timing_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The bench's own emission-vs-truth timing distribution, per event type and overall."""
    output: dict[str, Any] = {}
    for label in ("all", "contact", "bounce", "point_end", "contact_serve", "contact_rally"):
        if label == "all":
            selected = list(rows)
        elif label == "contact_serve":
            selected = [row for row in rows if row.get("phase") == "serve"]
        elif label == "contact_rally":
            selected = [
                row
                for row in rows
                if row["event_type"] == "contact" and row.get("phase") != "serve"
            ]
        else:
            selected = [row for row in rows if row["event_type"] == label]
        if not selected:
            continue
        offsets = [float(row["offset_frames"]) for row in selected]
        output[label] = {
            "count": len(selected),
            "fractional_frames": sum(
                1
                for row in selected
                if abs(float(row["frame"]) - round(float(row["frame"]))) > 1e-9
            ),
            "off_observed_track_frame": sum(
                1 for row in selected if not row.get("on_observed_track_frame", True)
            ),
            "signed_offset_frames": {
                **_stats(offsets),
                "mean": float(np.mean(offsets)),
            },
            "absolute_offset_frames": {
                **_stats([abs(value) for value in offsets]),
                "mean": float(np.mean([abs(value) for value in offsets])),
            },
            "pixel_error_px": _stats([float(row["pixel_error_px"]) for row in selected]),
            "pixel_source": _counts(row["pixel_source"] for row in selected),
            "refiner_reason": _counts(row["refiner_reason"] for row in selected),
            "refiner_offset": _counts(row["refiner_offset"] for row in selected),
        }
    return output


def _striker_constants() -> dict[str, Any]:
    return {
        "reach_range_m": list(STRIKER_REACH_RANGE_M),
        "serve_reach_range_m": list(STRIKER_SERVE_REACH_RANGE_M),
        "reach_angle_range_deg": list(STRIKER_REACH_ANGLE_RANGE_DEG),
        "recovery_seconds": STRIKER_RECOVERY_SECONDS,
        "recovery_fraction": STRIKER_RECOVERY_FRACTION,
        "split_step_seconds": STRIKER_SPLIT_STEP_SECONDS,
        "sway_m": STRIKER_SWAY_M,
        "noise_rungs": list(STRIKER_NOISE_RUNGS),
        "box_jitter_px540": list(STRIKER_BOX_JITTER_PX),
        "box_jitter_tail_rate": STRIKER_BOX_JITTER_TAIL_RATE,
        "box_jitter_tail_px540": STRIKER_BOX_JITTER_TAIL_PX,
        "miss_rate": STRIKER_MISS_RATE,
        "miss_run_mean_frames": STRIKER_MISS_RUN_MEAN_FRAMES,
    }


def _striker_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    lows = [row["court_y_range"][0] for row in rows if row.get("court_y_range")]
    highs = [row["court_y_range"][1] for row in rows if row.get("court_y_range")]
    roots = [row["root_error_m_median"] for row in rows if row.get("root_error_m_median")]
    return {
        "rows": sum(int(row["rows"]) for row in rows),
        "missing_rows": sum(int(value) for row in rows for value in row["missing_frames"].values()),
        "court_y_min": min(lows) if lows else None,
        "court_y_max": max(highs) if highs else None,
        "root_error_m_median": float(np.median(roots)) if roots else None,
    }


def _emission(
    match_id: str, point: dict[str, Any], event_type: str, row: dict[str, Any]
) -> dict[str, Any]:
    frame = float(row["frame"])
    return {
        "abstain": False,
        "clip": f"{match_id}__{point['clip']}",
        "match_id": match_id,
        "event_type": event_type,
        "frame": frame,
        "probability": 0.99,
        "score": 0.99,
        "confidence": 0.99,
        "location": {
            "court_x_m": None,
            "court_y_m": None,
            "fps": point["fps"],
            "frame_subpixel": frame,
            "image_coordinate_space": "native_1920x1080",
            "image_x": float(row["pixel"][0]),
            "image_y": float(row["pixel"][1]),
            "source": "s6_point_bench_true_projection",
            "time_seconds": frame / float(point["fps"]),
        },
    }


def _point_end_emission(match_id: str, point: dict[str, Any]) -> dict[str, Any]:
    """One ``point_end`` row at the termination, shaped like ``point_grammar.point_end_emissions``.

    ``reconstruction.py`` keeps only ``contact``/``bounce``/``net_hit`` rows, so this row is
    carried through the runner's event file without reaching the fitter -- which is exactly what
    the real chain does with it.
    """
    ending = point["emission_point_end"]
    row = _emission(match_id, point, "point_end", ending)
    row["location"]["source"] = ending["source"]
    row["point_end"] = {
        "schema": "point_grammar_point_end_v1",
        "source": "s6_point_bench_truth_termination",
        "chain_id": f"{match_id}__{point['clip']}",
        "chain_length": len(point["emission_contacts"]) + len(point["emission_bounces"]),
        "terminal_event_type": ending["terminal_event_type"],
        "terminal_event_frame": float(ending["frame"]),
        "termination_kind": ending["kind"],
        "structure_span_id": f"{match_id}__{point['clip']}__0",
    }
    row["point_grammar"] = {"in_play": True, "verdict": "point_end"}
    return row


def write_declared_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    kind: str,
    fps: float,
) -> None:
    """Write one synthetic host artifact together with its coordinate declaration.

    The host emits half-native ``x``/``y`` beside native mirrors exactly as the shipped
    producers do, so a consumer resolves the scale here the same way it does on a real
    cohort.  Without the sidecar the fitter's loaders fall back to nothing at all.
    """
    _write_csv(path, rows)
    sidecar = res.coordinate_manifest_path(path)
    if kind == "track":
        res.write_native_dual_coordinate_manifest(
            sidecar,
            image_size=res.NATIVE_SIZE,
            legacy_size=res.LEGACY_TRACKING_SIZE,
            source="wk3 synthetic point host",
            native_columns=res.TRACK_NATIVE_POINT_COLUMNS,
            legacy_columns=("x", "y"),
            extra={"fps": fps},
        )
    elif kind == "boxes":
        res.write_native_dual_coordinate_manifest(
            sidecar,
            image_size=res.NATIVE_SIZE,
            legacy_size=res.LEGACY_TRACKING_SIZE,
            source="wk3 synthetic point host",
            native_columns=(),
            legacy_columns=("x0", "y0", "x1", "y1"),
            extra={
                "fps": fps,
                "artifact_identity": res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
            },
        )
    elif kind == "pose":
        res.write_coordinate_manifest(
            sidecar,
            image_size=res.NATIVE_SIZE,
            artifact_size=res.NATIVE_SIZE,
            source="wk3 synthetic point host",
            extra={"fps": fps, "artifact_identity": res.PLAYER_POSE_NATIVE_IDENTITY},
        )
    else:
        raise ValueError(f"unknown synthetic artifact kind {kind!r}")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_cameras(camera_root: Path, hosts: dict[str, HostPoint], points: Sequence[dict]) -> None:
    """One camera artifact per synthetic match, relabelled from the host points' own rows."""
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_match[point["match_id"]].append(point)
    for match_id, rows in by_match.items():
        match_dir = camera_root / match_id
        match_dir.mkdir(parents=True, exist_ok=True)
        fields: dict[str, list[np.ndarray]] = defaultdict(list)
        point_numbers = []
        homographies = []
        frame_clips: list[np.ndarray] = []
        frame_numbers: list[np.ndarray] = []
        frame_matrices: list[np.ndarray] = []
        for point in sorted(rows, key=lambda row: row["clip"]):
            host = hosts[point["host"]]
            count = len(host.camera_rows["frames"])
            for name, values in host.camera_rows.items():
                if name == "clips":
                    fields["clips"].append(np.full(count, point["clip"], dtype="<U8"))
                else:
                    fields[name].append(values)
            point_numbers.append(int(point["clip"].removeprefix("pt")))
            homographies.append(host.homography)
            if host.frame_homographies:
                keys = sorted(host.frame_homographies)
                frame_clips.append(np.full(len(keys), point["clip"], dtype="<U8"))
                frame_numbers.append(np.asarray(keys, dtype=np.int32))
                frame_matrices.append(np.stack([host.frame_homographies[key] for key in keys]))
        np.savez(
            match_dir / CAMERA_NAME,
            **{name: np.concatenate(values) for name, values in fields.items()},
        )
        np.savez(
            match_dir / "court_H_per_point.npz",
            pts=np.asarray(point_numbers, dtype=np.int32),
            H=np.stack(homographies),
            source=np.full(len(point_numbers), "s6_point_bench_host", dtype="<U24"),
            topology_score=np.ones(len(point_numbers)),
            surface_score=np.ones(len(point_numbers)),
        )
        if frame_clips:
            np.savez(
                match_dir / "court_H_per_frame_v1.npz",
                clips=np.concatenate(frame_clips),
                frames=np.concatenate(frame_numbers),
                H=np.concatenate(frame_matrices),
            )


# --------------------------------------------------------------------------------------- #
# Running the unchanged default 3D stage through its published CLI contract
# --------------------------------------------------------------------------------------- #
NIGHTLY_ARGUMENTS = (
    "--include-dead-time-emissions",
    "--max-nfev",
    "20",
    "--math-threads",
    "1",
    "--point-timeout-seconds",
    "600",
    "--terminal-flights",
    "--anchor-bounce-geometry",
    "--whole-point-branches",
    "--branch-width",
    "3",
    "--branch-margin",
    "8",
)

ARMS = {
    "default": (),
    "net_plane_constraint": ("--net-plane-constraint",),
    "shared_contact_fit": ("--shared-contact-fit",),
    # Nothing in NIGHTLY_ARGUMENTS passes a pose artifact, so the shipped default never reads
    # the generated pose file.  This arm turns that witness on without changing the default.
    "pose_witness": ("--pose-artifact-name", POSE_NAME),
    # An emission on a frame is not the contact, so an observation next to it can sit on the
    # wrong side of the racket.  ``docs/wk1/s6_bench.md`` measured the shipped flag that
    # down-weights those observations as harmful on a bench whose contact sat on the true
    # continuous time; with the contact on a frame it is worth re-measuring.
    "contact_adjacent": ("--downweight-contact-adjacent",),
}


def run_reconstruction(
    root: Path,
    output_root: Path,
    *,
    arm: str,
    workers: int,
    points: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Invoke ``cv.pipeline.reconstruct_3d`` exactly as ``docs/wk1/nightly.md`` documents.

    ``points`` restricts the run to named point keys through the CLI's own ``--point`` flag,
    which is what makes the development loop take seconds instead of minutes.  Nothing else
    about the invocation changes, so a subset run and a sweep run are the same fitter.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known arms are {sorted(ARMS)}")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    command = [
        sys.executable,
        "-m",
        "cv.pipeline.reconstruct_3d",
        "--audit-root",
        os.fspath(root),
        "--manifest",
        os.fspath(root / "manifest.json"),
        "--event-boundaries",
        os.fspath(root / EVENT_NAME),
        "--output",
        os.fspath(report_path),
        "--workers",
        str(workers),
        *NIGHTLY_ARGUMENTS,
        *ARMS[arm],
        *[value for key in (points or ()) for value in ("--point", str(key))],
        "--anchors-output-root",
        os.fspath(output_root / "anchors"),
    ]
    started = time.perf_counter()
    with (output_root / "run.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return {"command": command, "wall_seconds": elapsed, "report": os.fspath(report_path)}


# --------------------------------------------------------------------------------------- #
# Scoring against the synthetic truth
# --------------------------------------------------------------------------------------- #
def _nearest(rows: Sequence[dict[str, Any]], frame: float, key: str) -> dict[str, Any] | None:
    """Match a pipeline record to a truth flight by its start frame, not by index.

    A dropped or retimed contact renumbers the pipeline's flights; matching on the boundary
    keeps the truth alignment honest when that happens.
    """
    best = None
    best_gap = math.inf
    for row in rows:
        value = row.get(key)
        if value is None:
            # A record the fitter never placed on the timeline -- a point that hit
            # ``--point-timeout-seconds`` carries one attempt row per flight with every field
            # null.  It matches no truth flight, and the flight is scored as unattempted,
            # which is what a timeout is.
            continue
        gap = abs(float(value) - frame)
        if gap < best_gap:
            best, best_gap = row, gap
    return best if best is not None and best_gap <= 3.0 else None


def _fit_position_at(fit: dict[str, Any], frame: float, fps: float) -> np.ndarray | None:
    """Where the fit puts the ball at ``frame``, if its own span reaches that far.

    The owner's design makes the terminal flight end at the point's physical ending, so a fitter
    that stops fitting AT the first bounce reports no bounce record -- its endpoint is its
    answer, and it is scored on that rather than marked as a missed bounce.
    """
    end = fit.get("end_frame")
    if end is None or float(end) < frame - FIT_ENDPOINT_SLACK_FRAMES:
        return None
    rows = [
        row
        for row in (fit.get("trajectory") or [])
        if row.get("xyz") is not None and row.get("frame") is not None
    ]
    if rows:
        nearest = min(rows, key=lambda row: abs(float(row["frame"]) - frame))
        position = np.asarray(nearest["xyz"], dtype=float)
        velocity = nearest.get("velocity")
        if velocity is not None and fps > 0.0:
            position = position + np.asarray(velocity, dtype=float) * (
                (frame - float(nearest["frame"])) / fps
            )
        return position
    if fit.get("end_xyz") is not None and abs(float(end) - frame) <= FIT_ENDPOINT_SLACK_FRAMES:
        return np.asarray(fit["end_xyz"], dtype=float)
    return None


def _fit_contact_position(
    fit: dict[str, Any],
    frame: float,
    fps: float,
    endpoint: Sequence[float] | None,
) -> tuple[np.ndarray | None, str]:
    """Where the fit puts the ball at the TRUE contact time.

    ``docs/wk1/point_fit6.md`` measured that the real chain emits a contact on a frame while
    the true contact is between two frames; a fit whose flight boundary is that emitted frame
    therefore reports its endpoint half a frame away from the true contact, which is a property
    of the emission and not of the trajectory.  Scoring the fitted trajectory at the true time
    removes that quantisation and leaves the fit's own error.  When the fit has no trajectory
    reaching the true time, its own endpoint is the answer.
    """
    if fit.get("trajectory"):
        sampled = _fit_position_at(fit, frame, fps)
        if sampled is not None:
            return sampled, "trajectory"
    if endpoint is None:
        return None, "missing"
    return np.asarray(endpoint, dtype=float), "endpoint"


def _fitted_dead_ball(fit: dict[str, Any], scored_end: float) -> bool:
    """Diagnostic only: did the fitter keep fitting past the point's physical ending?"""
    end = fit.get("end_frame")
    if end is not None and float(end) > scored_end + 1.0:
        return True
    return any(
        float(row["frame"]) > scored_end + 1.0
        for row in (fit.get("bounces") or [])
        if row.get("frame") is not None
    )


def load_anchor_availability(anchors_root: Path) -> dict[str, dict[float, dict[str, Any]]]:
    """Per-flight anchor availability, keyed by point and flight start frame.

    ``docs/wk1/solve_failures.md`` attributes most real lost flights to a missing net-crossing
    anchor or to two bounce anchors in one flight; the artifact records both directly.
    """
    output: dict[str, dict[float, dict[str, Any]]] = {}
    if not anchors_root.is_dir():
        return output
    for path in sorted(anchors_root.glob("*/*/anchors_v1.json")):
        payload = json.loads(path.read_text())
        output[str(payload["point"])] = {
            float(flight["start_frame"]): {
                "net_anchor_available": bool(flight.get("net_anchor_available")),
                "bounce_anchors": int(flight.get("bounce_anchors", 0)),
            }
            for flight in payload.get("flights", [])
        }
    return output


# --------------------------------------------------------------------------------------- #
# The truth curve: re-integrated between the truth anchors, as docs/wk1/dev_loop.md does
# --------------------------------------------------------------------------------------- #
# The truth file holds anchors (contacts, bounces, the termination) and not the sampled
# trajectory, and the generator's per-shot spin is not in it.  So each anchor-to-anchor segment
# is solved as a two-point boundary problem on the generator's own physics: the launch velocity
# is solved so the arc lands exactly on the next anchor, and a nominal topspin only shapes the
# curvature in between.  dev_loop.md measured that cost: the anchors are hit to 0.000 m and the
# curve projects to a per-point median of 0.86-2.58 px of the clean rung's own true track.
TRUTH_ARC_SPIN_RPM = 2000.0
TRUTH_ARC_SAMPLES = 60


def truth_segment(
    start: np.ndarray, end: np.ndarray, seconds: float
) -> tuple[np.ndarray, float] | None:
    """The arc from ``start`` to ``end`` in ``seconds``, on the generator's own physics.

    Returns the sampled positions and the metric miss at the far end, so a bad segment can be
    seen rather than silently used.
    """
    if seconds <= 1e-4:
        return None
    spin_magnitude = TRUTH_ARC_SPIN_RPM * 2.0 * math.pi / 60.0
    guess = (end - start) / seconds + np.array([0.0, 0.0, 0.5 * 9.81 * seconds])

    def arc(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spin = spin_magnitude * _topspin_axis(velocity)
        times, positions, _, _ = integrate_arc(
            start, velocity, spin, max_seconds=seconds + 1e-3, stop_at_ground=False
        )
        return times, positions

    def residual(velocity: np.ndarray) -> np.ndarray:
        times, positions = arc(np.asarray(velocity, dtype=float))
        landing = np.array(
            [float(np.interp(seconds, times, positions[:, axis])) for axis in range(3)]
        )
        return landing - end

    try:
        solution = least_squares(residual, guess, xtol=1e-8, ftol=1e-8, max_nfev=40)
    except (ValueError, np.linalg.LinAlgError):
        return None
    times, positions = arc(np.asarray(solution.x, dtype=float))
    grid = np.linspace(0.0, seconds, TRUTH_ARC_SAMPLES)
    sampled = np.stack(
        [
            np.array([float(np.interp(value, times, positions[:, axis])) for axis in range(3)])
            for value in grid
        ]
    )
    return sampled, float(np.linalg.norm(residual(solution.x)))


def truth_flight_anchors(
    point: dict[str, Any], entry: dict[str, Any], *, scored_only: bool
) -> list[tuple[float, np.ndarray]]:
    """The 3D anchors of one truth flight, in time order.

    ``scored_only`` is the scoring view: the flight's start contact, the bounces inside its
    scored span, and its termination.  Everything after the termination -- the dead ball -- is
    dropped, because the owner's definition puts it outside the flight.  The drawing view keeps
    the dead-ball bounces so the tail is visible.
    """
    contacts = {round(float(row["frame"]), 6): row for row in point["contacts"]}
    anchors: list[tuple[float, np.ndarray]] = []
    start = float(entry["start_frame"])
    start_contact = contacts.get(round(start, 6))
    if start_contact is not None:
        anchors.append((start, np.asarray(start_contact["xyz"], dtype=float)))
    bounces = scored_bounces(entry) if scored_only else (entry.get("bounces") or [])
    for bounce in bounces:
        anchors.append((float(bounce["frame"]), np.asarray(bounce["xyz"], dtype=float)))
    end = float(entry["end_frame"])
    end_contact = contacts.get(round(end, 6))
    if end_contact is not None:
        anchors.append((end, np.asarray(end_contact["xyz"], dtype=float)))
    if scored_only and entry.get("terminal") and entry.get("termination_xyz") is not None:
        frame = float(entry.get("termination_frame", terminal_scored_end(entry)))
        if all(abs(frame - existing) > 1e-6 for existing, _ in anchors):
            anchors.append((frame, np.asarray(entry["termination_xyz"], dtype=float)))
    anchors.sort(key=lambda row: row[0])
    return anchors


def truth_flight_paths(point: dict[str, Any], *, scored_only: bool = False) -> list[dict[str, Any]]:
    """Use recorded simulation paths; explicitly retain the legacy approximation fallback."""
    if "trajectory_truth" in point:
        recorded = point["trajectory_truth"]
        if recorded.get("schema") != "simulation_trajectory_samples_v1":
            raise ValueError("unsupported exact trajectory truth schema")
        frames = np.asarray(recorded["frames"], dtype=float)
        positions = np.asarray(recorded["positions"], dtype=float)
        if (
            frames.ndim != 1
            or len(frames) < 2
            or positions.shape != (len(frames), 3)
            or not np.all(np.isfinite(frames))
            or not np.all(np.isfinite(positions))
            or not np.all(np.diff(frames) > 0)
        ):
            raise ValueError("invalid exact trajectory truth samples")
        output = []
        for entry in point["flights"]:
            low = float(entry["start_frame"])
            high = terminal_scored_end(entry) if scored_only else float(entry["end_frame"])
            if low < frames[0] - 1e-6 or high > frames[-1] + 1e-6 or high <= low:
                raise ValueError("exact trajectory does not cover the truth flight")
            selected = np.unique(np.r_[low, frames[(frames > low) & (frames < high)], high])
            sampled = np.column_stack(
                [np.interp(selected, frames, positions[:, axis]) for axis in range(3)]
            )
            output.append(
                {
                    "flight_index": int(entry["flight_index"]),
                    "terminal": bool(entry["terminal"]),
                    "frames": selected,
                    "positions": sampled,
                    "anchor_miss_max_m": None,
                    "source": "generator_piece_interpolant",
                }
            )
        return output
    fps = float(point["fps"])
    output = []
    for entry in point["flights"]:
        anchors = truth_flight_anchors(point, entry, scored_only=scored_only)
        frames: list[float] = []
        positions: list[np.ndarray] = []
        misses: list[float] = []
        for (frame_a, point_a), (frame_b, point_b) in zip(anchors, anchors[1:]):
            segment = truth_segment(point_a, point_b, (frame_b - frame_a) / fps)
            if segment is None:
                continue
            sampled, miss = segment
            grid = np.linspace(frame_a, frame_b, TRUTH_ARC_SAMPLES)
            frames.extend(grid.tolist())
            positions.extend(list(sampled))
            misses.append(miss)
        output.append(
            {
                "flight_index": int(entry["flight_index"]),
                "terminal": bool(entry["terminal"]),
                "frames": np.asarray(frames, dtype=float),
                "positions": (np.stack(positions) if positions else np.zeros((0, 3), dtype=float)),
                "anchor_miss_max_m": max(misses, default=None),
            }
        )
    return output


def sample_path(path: dict[str, Any], frame: float) -> np.ndarray | None:
    frames, positions = path["frames"], path["positions"]
    if not len(frames) or frame < frames[0] - 1e-6 or frame > frames[-1] + 1e-6:
        return None
    return np.array([float(np.interp(frame, frames, positions[:, axis])) for axis in range(3)])


def _truth_paths_worker(point: dict[str, Any]) -> tuple[str, list[list[list[float]]]]:
    paths = truth_flight_paths(point, scored_only=True)
    return str(point["point"]), [
        [path["frames"].tolist(), path["positions"].tolist()] for path in paths
    ]


def truth_scored_paths(
    truth: dict[str, Any],
    *,
    workers: int = 1,
    cache_path: Path | None = None,
    keys: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Every truth flight's scored 3D curve, keyed by point, built once and cached.

    New truth stores the generator's actual interpolant. Legacy truth needs an
    approximate re-integration. Cache receipts cover truth content, code and output
    bytes; a filename alone is not evidence of identity.
    ``keys`` restricts the work to a subset of points, which is what the development inner
    loop wants: twelve points, a second, and no cache file to keep in step.
    """
    if keys is not None:
        cache_path = None
    cache_args = None
    if cache_path is not None:
        cache_args = {
            "stage": "synthetic_truth_curves_"
            + hashlib.sha256(cache_path.name.encode()).hexdigest()[:16],
            "command": ["python", "-m", "cv.validation.s6_point_bench"],
            "inputs": [],
            "configuration": {
                "truth_sha256": hashlib.sha256(
                    json.dumps(truth, sort_keys=True, default=_json_default).encode()
                ).hexdigest()
            },
        }
    if cache_path is not None and stage_receipt_matches(
        out_dir=cache_path.parent, outputs=[cache_path], **cache_args
    ):
        with np.load(cache_path, allow_pickle=False) as archive:
            stored = {name: archive[name] for name in archive.files}
        output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for name in sorted(stored):
            key, index, field_name = name.rsplit("|", 2)
            if field_name != "frames":
                continue
            output[key].append(
                {
                    "flight_index": int(index),
                    "frames": stored[name],
                    "positions": stored[f"{key}|{index}|positions"],
                }
            )
        return dict(output)
    before = stage_identity(**cache_args) if cache_args is not None else None
    points = list(truth["points"])
    if keys is not None:
        wanted = set(keys)
        points = [point for point in points if str(point["point"]) in wanted]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_truth_paths_worker, points, chunksize=4))
    else:
        results = [_truth_paths_worker(point) for point in points]
    output = {}
    for key, paths in results:
        output[key] = [
            {
                "flight_index": index,
                "frames": np.asarray(frames, dtype=float),
                "positions": np.asarray(positions, dtype=float).reshape(-1, 3),
            }
            for index, (frames, positions) in enumerate(paths)
        ]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for key, paths in output.items():
            for path in paths:
                stem = f"{key}|{path['flight_index']}"
                payload[f"{stem}|frames"] = path["frames"]
                payload[f"{stem}|positions"] = path["positions"]
        np.savez_compressed(cache_path, **payload)
        write_stage_receipt(
            out_dir=cache_path.parent, outputs=[cache_path], expected_identity=before, **cache_args
        )
    return output


# --------------------------------------------------------------------------------------- #
# The `metric_v2` witnesses: what a flight has to satisfy in 3D, not just on the image plane
# --------------------------------------------------------------------------------------- #
def last_visible_truth_frame(point: dict[str, Any], flight: dict[str, Any]) -> float | None:
    """The last frame of an ``out_of_view`` flight on which the truth ball is still in the image.

    An `out_of_view` termination has no 3D anchor, so this is the latest place the truth still
    says where the ball was AND a tracker could still have seen it.  Scoring there rather than
    at the exit frame keeps the witness inside the observable part of the point.
    """
    gone = {int(value) for value in (point.get("out_of_frame_frames") or [])}
    start = float(flight["start_frame"])
    frame = math.floor(terminal_scored_end(flight))
    while frame > start:
        if frame not in gone:
            return float(frame)
        frame -= 1
    return None


def fit_final_segment(fit: dict[str, Any]) -> dict[str, Any]:
    """The last sampled step of a fitted flight: where it ends and how it is moving there.

    ``docs/wk1/dev_loop.md``: a fit whose span runs out mid-flight ends "at a non-zero height
    with a near-vertical last segment".  Both halves of that are measured here.
    """
    rows = [
        row
        for row in (fit.get("trajectory") or [])
        if row.get("xyz") is not None and row.get("frame") is not None
    ]
    rows.sort(key=lambda row: float(row["frame"]))
    if not rows:
        return {"frame": None, "height_m": None, "angle_deg": None}
    last = np.asarray(rows[-1]["xyz"], dtype=float)
    angle = None
    if len(rows) > 1:
        step = last - np.asarray(rows[-2]["xyz"], dtype=float)
        horizontal = float(np.linalg.norm(step[:2]))
        if horizontal > 1e-9 or abs(step[2]) > 1e-9:
            angle = math.degrees(math.atan2(abs(float(step[2])), horizontal))
    return {
        "frame": float(rows[-1]["frame"]),
        "height_m": float(last[2]),
        "angle_deg": angle,
    }


def fit_direction_angle_deg(fit: dict[str, Any], frame: float) -> float | None:
    """How steep the fitted flight is at ``frame``, in degrees from horizontal.

    Read off the fitted velocity, so it does not depend on the trajectory sampling rate.
    """
    rows = [
        row
        for row in (fit.get("trajectory") or [])
        if row.get("velocity") is not None and row.get("frame") is not None
    ]
    if not rows:
        return None
    nearest = min(rows, key=lambda row: abs(float(row["frame"]) - frame))
    velocity = np.asarray(nearest["velocity"], dtype=float)
    horizontal = float(np.linalg.norm(velocity[:2]))
    if horizontal < 1e-9 and abs(float(velocity[2])) < 1e-9:
        return None
    return math.degrees(math.atan2(abs(float(velocity[2])), horizontal))


def trajectory_rms(
    fit: dict[str, Any],
    path: dict[str, Any] | None,
    *,
    start_frame: float,
    end_frame: float,
    fps: float,
) -> dict[str, Any]:
    """RMS distance between the fitted flight and the re-integrated truth curve.

    Sampled at every whole frame of the flight's scored span, because that is what the product
    delivers: a ball position per frame, not a position at three anchors.  A frame the fit does
    not reach at all is not a small error, it is no answer, so it makes the RMS infinite and is
    counted separately.
    """
    if path is None or not len(path["frames"]):
        return {"rms_m": None, "frames": 0, "uncovered_frames": 0, "max_m": None}
    frames = [
        float(value)
        for value in range(
            int(math.ceil(start_frame - 1e-6)), int(math.floor(end_frame + 1e-6)) + 1
        )
    ]
    errors: list[float] = []
    uncovered = 0
    for frame in frames:
        truth_position = sample_path(path, frame)
        if truth_position is None:
            continue
        fitted = _fit_position_at(fit, frame, fps)
        if fitted is None:
            uncovered += 1
            errors.append(math.inf)
            continue
        errors.append(float(np.linalg.norm(fitted - truth_position)))
    if not errors:
        return {"rms_m": None, "frames": 0, "uncovered_frames": 0, "max_m": None}
    if uncovered:
        return {
            "rms_m": math.inf,
            "frames": len(errors),
            "uncovered_frames": uncovered,
            "max_m": math.inf,
        }
    values = np.asarray(errors, dtype=float)
    return {
        "rms_m": float(np.sqrt(float(np.mean(values**2)))),
        "frames": len(errors),
        "uncovered_frames": 0,
        "max_m": float(values.max()),
    }


def _within(value: float | None, tolerance: float) -> bool:
    """A missing witness cannot convict: only a measured error above ``tolerance`` fails."""
    return value is None or (math.isfinite(value) and value <= tolerance)


def metric_v2_checks(row: dict[str, Any]) -> dict[str, bool]:
    """Every condition `metric_v2` puts on one flight, named, so a failure says which one.

    All of them must hold.  The thresholds and the reason for each are at the top of this file.
    """
    termination_kind = row.get("termination_kind")
    airborne = True
    if row["terminal"] and termination_kind == "out_of_view":
        height = row.get("exit_height_m")
        angle = row.get("exit_angle_deg")
        airborne = (
            height is not None
            and height >= AIRBORNE_MIN_HEIGHT_M
            and not row.get("mid_air_stop", False)
            and (angle is None or angle <= NEAR_VERTICAL_ANGLE_DEG)
        )
    return {
        "bounce": _within(row.get("bounce_error_m"), BOUNCE_TOLERANCE_M),
        "contact_px": _within(row.get("contact_error_px"), CONTACT_PIXEL_TOLERANCE_PX),
        "contact_m": _within(row.get("contact_error_m"), CONTACT_TOLERANCE_M),
        "termination": _within(row.get("termination_error_m"), TERMINATION_TOLERANCE_M)
        and _within(row.get("exit_error_m"), TERMINATION_TOLERANCE_M),
        "airborne": airborne,
        "junction": _within(row.get("junction_gap_prev_m"), JUNCTION_TOLERANCE_M)
        and _within(row.get("junction_gap_next_m"), JUNCTION_TOLERANCE_M),
        "trajectory": _within(row.get("trajectory_rms_m"), TRAJECTORY_RMS_TOLERANCE_M),
    }


def pixel_v1_checks(row: dict[str, Any]) -> dict[str, bool]:
    """The criterion the bench shipped with: metres at scored bounces, pixels at contacts."""
    return {
        "bounce": _within(row.get("bounce_error_m"), BOUNCE_TOLERANCE_M),
        "contact_px": _within(row.get("contact_error_px"), CONTACT_PIXEL_TOLERANCE_PX),
    }


def audit_report(
    truth: dict[str, Any],
    report: dict[str, Any],
    camera_root: Path,
    anchors_root: Path | None = None,
    *,
    criterion: str = DEFAULT_CRITERION,
    truth_paths: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score one finished reconstruction report against the synthetic truth.

    Both criteria are always measured; ``criterion`` only decides which one drives
    ``truth_good``, ``complete_point``, the wrong-accept count and the first-cause waterfall.
    ``truth_paths`` is the re-integrated truth curve per point; it is built here when it is not
    supplied, but every run scored against the same truth file wants the same curves, so a
    sweep should build it once with :func:`truth_scored_paths` and pass it in.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {CRITERIA}")
    truth_points = {row["point"]: row for row in truth["points"]}
    if truth_paths is None:
        truth_paths = truth_scored_paths(truth)
    anchors = load_anchor_availability(anchors_root) if anchors_root else {}
    cameras: dict[str, Any] = {}

    def camera_for(point_key: str) -> Any:
        if point_key not in cameras:
            match_id, clip = point_key.split("__", 1)
            cameras[point_key] = PointCamera(camera_root / match_id, clip)
        return cameras[point_key]

    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for record in report["points_detail"]:
        point_key = str(record["point"])
        expected = truth_points.get(point_key)
        if expected is None:
            continue
        fits = list(record.get("fits", []))
        attempts = list(record.get("flight_attempts", []))
        camera = camera_for(point_key)
        fps = float(expected["fps"])
        point_paths = truth_paths.get(point_key) or []
        flight_rows = []
        for index, truth_flight in enumerate(expected["flights"]):
            path = point_paths[index] if index < len(point_paths) else None
            start = float(truth_flight["start_frame"])
            scored_end = terminal_scored_end(truth_flight)
            truth_scored = scored_bounces(truth_flight)
            termination_kind = terminal_termination_kind(truth_flight)
            attempt = _nearest(attempts, start, "start_frame")
            fit = _nearest(fits, start, "start_frame")
            anchor_rows = anchors.get(point_key, {})
            anchor = (
                anchor_rows.get(
                    min(anchor_rows, key=lambda value: abs(value - start), default=math.nan)
                )
                if anchor_rows
                else None
            )
            row: dict[str, Any] = {
                "point": point_key,
                "flight_index": index,
                "terminal": bool(truth_flight["terminal"]),
                "truth_bounces": len(truth_flight["bounces"]),
                "scored_end_frame": scored_end,
                "scored_bounces": len(truth_scored),
                "dead_ball_bounces": len(truth_flight["bounces"]) - len(truth_scored),
                "termination_kind": termination_kind,
                "out_of_frame_frames": int(truth_flight.get("out_of_frame_frames") or 0),
                "fitted_dead_ball": None,
                "termination_anchor_source": None,
                "termination_error_m": None,
                "attempted": attempt is not None,
                "solved": fit is not None,
                "accepted": bool(fit and fit.get("held_out_accepted")),
                "held_out_median_px": (fit or {}).get("held_out_reprojection_median_px"),
                "held_out_p90_px": (fit or {}).get("held_out_reprojection_p90_px"),
                "net_anchor_available": (
                    (anchor or {}).get("net_anchor_available")
                    if anchor is not None
                    else (fit or {}).get("net_anchor_available")
                ),
                "bounce_anchors": (anchor or {}).get("bounce_anchors"),
                "reasons": list(record.get("reasons", [])),
                "bounce_error_m": None,
                "contact_error_px": None,
                "contact_error_m": None,
                "start_contact_error_px": None,
                "end_contact_error_px": None,
                "start_contact_error_m": None,
                "end_contact_error_m": None,
                "contact_error_at_emitted_frame_m": None,
                "contact_position_source": None,
                "junction_gap_prev_m": None,
                "junction_gap_next_m": None,
                "failure_reason": None,
                "bounce_ok": False,
                "contact_ok": False,
                "truth_good": False,
                "truth_good_pixel_v1": False,
                "truth_good_metric_v2": False,
                "failed_checks": None,
                "exit_frame": None,
                "exit_error_m": None,
                "exit_height_m": None,
                "exit_angle_deg": None,
                "mid_air_stop": None,
                "fit_end_height_m": None,
                "fit_end_angle_deg": None,
                "trajectory_rms_m": None,
                "trajectory_max_m": None,
                "trajectory_frames": 0,
                "trajectory_uncovered_frames": 0,
            }
            if fit is not None:
                bounce_errors = []
                fitted_bounces = list(fit.get("bounces") or [])
                bounce_sources: list[str] = []
                for bounce in truth_scored:
                    frame = float(bounce["frame"])
                    match = _nearest(fitted_bounces, frame, "frame")
                    fitted_bounce = None
                    source = "missing"
                    if match is not None and match.get("x") is not None:
                        fitted_bounce = np.asarray(match["x"], dtype=float)
                        source = "record"
                    elif bool(truth_flight["terminal"]) and abs(frame - scored_end) <= 1.0:
                        fitted_bounce = _fit_position_at(fit, frame, fps)
                        if fitted_bounce is not None:
                            source = "endpoint"
                    bounce_sources.append(source)
                    bounce_errors.append(
                        math.inf
                        if fitted_bounce is None
                        else float(
                            np.linalg.norm(fitted_bounce - np.asarray(bounce["xyz"], dtype=float))
                        )
                    )
                if bool(truth_flight["terminal"]):
                    # The termination anchor is the LAST scored bounce -- the second bounce when
                    # the first is in, the first bounce when it is out.
                    row["termination_anchor_source"] = (
                        bounce_sources[-1] if bounce_sources else "no_bounce"
                    )
                    row["fitted_dead_ball"] = _fitted_dead_ball(fit, scored_end)
                    truth_termination = truth_flight.get("termination_xyz")
                    if truth_termination is not None:
                        fitted_termination = _fit_position_at(fit, scored_end, fps)
                        row["termination_error_m"] = (
                            None
                            if fitted_termination is None
                            else float(
                                np.linalg.norm(
                                    fitted_termination - np.asarray(truth_termination, dtype=float)
                                )
                            )
                        )
                contact_errors: list[float] = []
                contact_errors_m: list[float] = []
                labelled: dict[str, float] = {}
                labelled_m: dict[str, float] = {}
                endpoint_m: dict[str, float | None] = {}
                contact_sources: dict[str, str] = {}
                endpoints = (
                    (float(truth_flight["start_frame"]), fit.get("start_xyz")),
                    (float(truth_flight["end_frame"]), fit.get("end_xyz")),
                )
                for label, (boundary, fitted) in zip(("start", "end"), endpoints, strict=True):
                    truth_contact = next(
                        (
                            item
                            for item in expected["contacts"]
                            if abs(float(item["frame"]) - boundary) < 1e-6
                        ),
                        None,
                    )
                    if truth_contact is None:
                        continue
                    # The contact is scored where the FIT puts the ball at the true contact
                    # TIME, not at the fit's own boundary frame.  The two are the same thing
                    # whenever the emission sits on the true time, and they are not when the
                    # emission sits on a frame, which is what the real chain emits.  The
                    # boundary-frame reading is kept beside it as a diagnostic.
                    sampled, sample_source = _fit_contact_position(fit, boundary, fps, fitted)
                    endpoint_error = (
                        None
                        if fitted is None
                        else float(
                            np.linalg.norm(
                                np.asarray(fitted, dtype=float)
                                - np.asarray(truth_contact["xyz"], dtype=float)
                            )
                        )
                    )
                    endpoint_m[label] = endpoint_error
                    contact_sources[label] = sample_source
                    if sampled is None:
                        contact_errors.append(math.inf)
                        contact_errors_m.append(math.inf)
                        labelled[label] = math.inf
                        labelled_m[label] = math.inf
                        continue
                    fitted_point = np.asarray(sampled, dtype=float)
                    metric_error = float(
                        np.linalg.norm(fitted_point - np.asarray(truth_contact["xyz"], dtype=float))
                    )
                    contact_errors_m.append(metric_error)
                    labelled_m[label] = metric_error
                    pixel_error = float(
                        np.linalg.norm(
                            project(camera.p_at(boundary), fitted_point)
                            - np.asarray(truth_contact["pixel"], dtype=float)
                        )
                    )
                    contact_errors.append(pixel_error)
                    labelled[label] = pixel_error
                row["bounce_error_m"] = max(bounce_errors) if bounce_errors else None
                row["contact_error_px"] = max(contact_errors) if contact_errors else None
                row["contact_error_m"] = max(contact_errors_m) if contact_errors_m else None
                row["start_contact_error_px"] = labelled.get("start")
                row["end_contact_error_px"] = labelled.get("end")
                row["start_contact_error_m"] = labelled_m.get("start")
                row["end_contact_error_m"] = labelled_m.get("end")
                endpoint_values = [value for value in endpoint_m.values() if value is not None]
                row["contact_error_at_emitted_frame_m"] = (
                    max(endpoint_values) if endpoint_values else None
                )
                row["contact_position_source"] = (
                    "endpoint"
                    if "endpoint" in contact_sources.values()
                    else ("trajectory" if contact_sources else None)
                )
                row["bounce_ok"] = _within(row["bounce_error_m"], BOUNCE_TOLERANCE_M)
                row["contact_ok"] = _within(row["contact_error_px"], CONTACT_PIXEL_TOLERANCE_PX)

                # Where the fitted span itself runs out, and how steeply it is moving there.
                final = fit_final_segment(fit)
                row["fit_end_height_m"] = final["height_m"]
                row["fit_end_angle_deg"] = final["angle_deg"]
                row["mid_air_stop"] = bool(
                    final["frame"] is not None
                    and final["height_m"] is not None
                    and final["height_m"] > MID_AIR_STOP_HEIGHT_M
                    and final["frame"] < scored_end - MID_AIR_STOP_SLACK_FRAMES
                )

                # An `out_of_view` ending has no 3D anchor, so it is witnessed at the last frame
                # the truth ball is still inside the image.
                if termination_kind == "out_of_view":
                    exit_frame = last_visible_truth_frame(expected, truth_flight)
                    row["exit_frame"] = exit_frame
                    if exit_frame is not None:
                        truth_exit = sample_path(path, exit_frame) if path is not None else None
                        fitted_exit = _fit_position_at(fit, exit_frame, fps)
                        if truth_exit is not None:
                            row["exit_error_m"] = (
                                math.inf
                                if fitted_exit is None
                                else float(np.linalg.norm(fitted_exit - truth_exit))
                            )
                        if fitted_exit is not None:
                            row["exit_height_m"] = float(fitted_exit[2])
                            row["exit_angle_deg"] = fit_direction_angle_deg(fit, exit_frame)

                sampled = trajectory_rms(
                    fit, path, start_frame=start, end_frame=scored_end, fps=fps
                )
                row["trajectory_rms_m"] = sampled["rms_m"]
                row["trajectory_max_m"] = sampled["max_m"]
                row["trajectory_frames"] = sampled["frames"]
                row["trajectory_uncovered_frames"] = sampled["uncovered_frames"]
            flight_rows.append(row)
            rows.append(row)
        gaps = [float(value) for value in (record.get("junction_gaps_m") or [])]
        for index, row in enumerate(flight_rows):
            row["junction_gap_prev_m"] = gaps[index - 1] if 0 < index <= len(gaps) else None
            row["junction_gap_next_m"] = gaps[index] if index < len(gaps) else None
            # The junction is only known once both neighbours have been placed, so truth-good is
            # decided here rather than inside the per-flight loop above.
            if row["solved"]:
                checks = {
                    CRITERION_PIXEL_V1: pixel_v1_checks(row),
                    CRITERION_METRIC_V2: metric_v2_checks(row),
                }
                row["truth_good_pixel_v1"] = all(checks[CRITERION_PIXEL_V1].values())
                row["truth_good_metric_v2"] = all(checks[CRITERION_METRIC_V2].values())
                row["failed_checks"] = "+".join(
                    name for name, passed in checks[criterion].items() if not passed
                )
            row["truth_good"] = bool(row[f"truth_good_{criterion}"])
            row["failure_reason"] = flight_failure_reason(row, record)
        complete = bool(flight_rows) and all(
            row["accepted"] and row["truth_good"] for row in flight_rows
        )
        accepted_all = bool(flight_rows) and all(row["accepted"] for row in flight_rows)
        point_rows.append(
            {
                "point": point_key,
                "source_match": str(expected.get("source_match") or point_key.split("__", 1)[0]),
                "mode": str(expected.get("mode") or MODE_RALLY),
                "lob": bool(expected.get("lob")),
                "termination_kind": (
                    terminal_termination_kind(expected["flights"][-1])
                    if expected["flights"]
                    else None
                ),
                "surface": expected["surface"],
                "fps": expected["fps"],
                "flights": len(flight_rows),
                "terminal_bounces": expected["terminal_bounces"],
                "attempted": sum(row["attempted"] for row in flight_rows),
                "solved": sum(row["solved"] for row in flight_rows),
                "accepted": sum(row["accepted"] for row in flight_rows),
                "truth_good": sum(row["truth_good"] for row in flight_rows),
                "wrong_accepts": sum(
                    row["accepted"] and not row["truth_good"] for row in flight_rows
                ),
                "accepted_all_flights": accepted_all,
                "fitted_dead_ball_flights": sum(
                    1 for row in flight_rows if row["fitted_dead_ball"]
                ),
                "out_of_frame_frames": len(expected.get("out_of_frame_frames") or []),
                "all_bounces_ok": bool(flight_rows)
                and all(row["bounce_ok"] for row in flight_rows),
                "all_contacts_ok": bool(flight_rows)
                and all(row["contact_ok"] for row in flight_rows),
                "complete_point": complete,
                "complete_point_pixel_v1": bool(flight_rows)
                and all(row["accepted"] and row["truth_good_pixel_v1"] for row in flight_rows),
                "complete_point_metric_v2": bool(flight_rows)
                and all(row["accepted"] and row["truth_good_metric_v2"] for row in flight_rows),
                "decision": record.get("decision"),
                "point_reasons": record.get("reasons", []),
                "first_cause": first_cause(record, flight_rows),
            }
        )
    accepted = sum(row["accepted"] for row in rows)
    truth_good_accepts = sum(row["accepted"] and row["truth_good"] for row in rows)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in point_rows:
        by_source[row["source_match"]].append(row)
    source_rows = [
        {
            "source_match": key,
            "points": len(subset),
            "lob_points": sum(row["lob"] for row in subset),
            "flights": sum(row["flights"] for row in subset),
            "complete_points": sum(row["complete_point"] for row in subset),
            "accepted_all_flight_points": sum(row["accepted_all_flights"] for row in subset),
            "wrong_accepted_flights": sum(row["wrong_accepts"] for row in subset),
        }
        for key, subset in sorted(by_source.items())
    ]
    by_criterion = {
        name: {
            "truth_good_flights": sum(row[f"truth_good_{name}"] for row in rows),
            "truth_good_accepted_flights": sum(
                row["accepted"] and row[f"truth_good_{name}"] for row in rows
            ),
            "wrong_accepted_flights": accepted
            - sum(row["accepted"] and row[f"truth_good_{name}"] for row in rows),
            "complete_points": sum(row[f"complete_point_{name}"] for row in point_rows),
        }
        for name in CRITERIA
    }
    return {
        "truth_definition": truth.get("truth_definition", TRUTH_DEFINITION),
        "criterion": criterion,
        "criterion_definition": CRITERION_DEFINITION,
        "by_criterion": by_criterion,
        "failed_check_counts": _counts(
            row["failed_checks"] for row in rows if row["accepted"] and not row["truth_good"]
        ),
        # Which single condition binds: how many SOLVED flights each `metric_v2` check passes,
        # so the criterion can be read as a frontier instead of one pass/fail number.
        "metric_v2_check_pass_counts": {
            name: sum(1 for row in rows if row["solved"] and metric_v2_checks(row).get(name, False))
            for name in (
                "bounce",
                "contact_px",
                "contact_m",
                "termination",
                "airborne",
                "junction",
                "trajectory",
            )
        },
        "metric_v2_check_pass_counts_accepted": {
            name: sum(
                1 for row in rows if row["accepted"] and metric_v2_checks(row).get(name, False)
            )
            for name in (
                "bounce",
                "contact_px",
                "contact_m",
                "termination",
                "airborne",
                "junction",
                "trajectory",
            )
        },
        "trajectory_rms_m": _stats(row["trajectory_rms_m"] for row in rows),
        "exit_error_m": _stats(row["exit_error_m"] for row in rows),
        "mid_air_stop_flights": sum(1 for row in rows if row["mid_air_stop"]),
        "truth_schema": str(truth.get("schema", "unknown")),
        "trajectory_truth_sources": dict(
            _counts(
                "generator_piece_interpolant"
                if "trajectory_truth" in point
                else "legacy_nominal_spin_anchor_approximation"
                for point in truth["points"]
            )
        ),
        "points": len(point_rows),
        "flights": len(rows),
        "attempted_flights": sum(row["attempted"] for row in rows),
        "solved_flights": sum(row["solved"] for row in rows),
        "accepted_flights": accepted,
        "truth_good_flights": sum(row["truth_good"] for row in rows),
        "truth_good_accepted_flights": truth_good_accepts,
        "wrong_accepted_flights": accepted - truth_good_accepts,
        "wrong_accept_rate": (accepted - truth_good_accepts) / accepted if accepted else None,
        "complete_points": sum(row["complete_point"] for row in point_rows),
        "accepted_all_flight_points": sum(row["accepted_all_flights"] for row in point_rows),
        "complete_points_excluding_terminal": _complete_excluding_terminal(point_rows, rows),
        "lob_points": sum(row["lob"] for row in point_rows),
        "complete_lob_points": sum(row["complete_point"] and row["lob"] for row in point_rows),
        "dead_ball_bounces_excluded": sum(row["dead_ball_bounces"] for row in rows),
        "scored_terminal_bounces": sum(row["scored_bounces"] for row in rows if row["terminal"]),
        "terminal_flights_fitting_dead_ball": sum(1 for row in rows if row["fitted_dead_ball"]),
        "termination_kind_counts": _counts(
            row["termination_kind"] for row in rows if row["terminal"]
        ),
        "termination_anchor_source_counts": _counts(
            row["termination_anchor_source"] for row in rows if row["terminal"] and row["solved"]
        ),
        "termination_error_m": _stats(
            row["termination_error_m"] for row in rows if row["terminal"]
        ),
        "two_bounce_terminal_flights": sum(
            1 for row in rows if row["terminal"] and row["scored_bounces"] > 1
        ),
        "flights_with_out_of_frame_gap": sum(1 for row in rows if row["out_of_frame_frames"]),
        "points_with_out_of_frame_gap": sum(1 for row in point_rows if row["out_of_frame_frames"]),
        "source_matches": len(source_rows),
        "complete_points_by_source_match": {
            row["source_match"]: row["complete_points"] for row in source_rows
        },
        "points_by_source_match": {row["source_match"]: row["points"] for row in source_rows},
        "source_match_rows": source_rows,
        "points_all_bounces_within_10cm": sum(row["all_bounces_ok"] for row in point_rows),
        "points_all_contacts_within_12px": sum(row["all_contacts_ok"] for row in point_rows),
        "start_contact_error_px": _stats(row["start_contact_error_px"] for row in rows),
        "end_contact_error_px": _stats(row["end_contact_error_px"] for row in rows),
        "junction_gap_m": _stats(_junction_gaps(rows, report)),
        "bounce_error_m": _stats(row["bounce_error_m"] for row in rows),
        "contact_error_px": _stats(row["contact_error_px"] for row in rows),
        "contact_error_m": _stats(row["contact_error_m"] for row in rows),
        "held_out_median_px": _stats(row["held_out_median_px"] for row in rows),
        "first_cause_counts": _counts(row["first_cause"] for row in point_rows),
        "unsolved_flight_anchors": _counts(
            f"net={row['net_anchor_available']},bounces={row['bounce_anchors']}"
            for row in rows
            if row["attempted"] and not row["solved"]
        ),
        "unsolved_flight_terminal_share": {
            "terminal": sum(
                row["terminal"] for row in rows if row["attempted"] and not row["solved"]
            ),
            "rally": sum(
                not row["terminal"] for row in rows if row["attempted"] and not row["solved"]
            ),
        },
        "by_flight_kind": {
            kind: {
                "flights": len(subset),
                "solved": sum(row["solved"] for row in subset),
                "accepted": sum(row["accepted"] for row in subset),
                "truth_good": sum(row["truth_good"] for row in subset),
                "contact_error_px": _stats(row["contact_error_px"] for row in subset),
            }
            for kind, subset in (
                ("terminal", [row for row in rows if row["terminal"]]),
                ("rally", [row for row in rows if not row["terminal"]]),
            )
        },
        "point_rows": point_rows,
        "flight_rows": rows,
    }


def flight_failure_reason(row: dict[str, Any], record: dict[str, Any]) -> str:
    """Why this flight is not an accepted, right flight -- first blocking reason only.

    The unsolved reasons come from the anchor artifact the fitter itself wrote, so they name a
    mechanism (``two_bounce_anchors``) rather than restating that the solver returned nothing.
    """
    if not row["attempted"]:
        return "not_attempted"
    if not row["solved"]:
        bounces = row.get("bounce_anchors")
        if bounces is not None and int(bounces) > 1:
            return f"two_bounce_anchors={int(bounces)}"
        if row.get("net_anchor_available") is False:
            return "no_net_anchor"
        return "solver_returned_none"
    if not row["accepted"]:
        reasons = list(
            (record.get("point_gate") or {}).get("reasons") or record.get("reasons") or []
        )
        return "pixel_gate:" + "+".join(reasons[:3]) if reasons else "pixel_gate_rejected"
    if not row["truth_good"]:
        return "wrong_accept:" + (row.get("failed_checks") or "unknown")
    return ""


def _junction_gaps(rows: Sequence[dict[str, Any]], report: dict[str, Any]) -> list[float]:
    """The pipeline's own adjacent-flight gap, as it reports it."""
    output: list[float] = []
    for record in report.get("points_detail", []):
        output.extend(float(value) for value in record.get("junction_gaps_m", []) or [])
    return output


def _complete_excluding_terminal(
    point_rows: Sequence[dict[str, Any]], flight_rows: Sequence[dict[str, Any]]
) -> int:
    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flight_rows:
        by_point[row["point"]].append(row)
    count = 0
    for row in point_rows:
        rally = [item for item in by_point[row["point"]] if not item["terminal"]]
        if rally and all(item["accepted"] and item["truth_good"] for item in rally):
            count += 1
    return count


def _counts(values: Iterable[Any]) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for value in values:
        output[str(value)] += 1
    return dict(sorted(output.items(), key=lambda item: -item[1]))


def first_cause(record: dict[str, Any], flight_rows: Sequence[dict[str, Any]]) -> str:
    """First blocking stage for one point, in pipeline order.

    The taxonomy follows ``docs/wk1/solve_failures.md`` and ``docs/wk1/gate_audit.md``: a later
    cause cannot be fixed while an earlier one still holds.
    """
    reasons = set(record.get("reasons", []))
    if "camera_calibration_unreliable" in reasons or "camera_projection_unavailable" in reasons:
        return "camera_abstained"
    if "upstream_point_validity_gate" in reasons:
        return "point_held"
    if all(row["accepted"] and row["truth_good"] for row in flight_rows):
        return "complete"
    for row in flight_rows:
        if not row["attempted"]:
            return "flight_not_attempted"
    for row in flight_rows:
        if row["attempted"] and not row["solved"]:
            if any("net_collision" in str(value) for value in row["reasons"]):
                return "net_hit_flight_discarded"
            return "flight_not_solved"
    for row in flight_rows:
        if row["solved"] and not row["accepted"]:
            return "solved_but_rejected"
    return "accepted_but_wrong"


# --------------------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------------------- #
_BUILD_HOSTS: list[HostPoint] = []
_BUILD_SKELETONS: list[Skeleton] = []


def _init_build(hosts: list[HostPoint], skeletons: list[Skeleton]) -> None:
    global _BUILD_HOSTS, _BUILD_SKELETONS
    _BUILD_HOSTS = hosts
    _BUILD_SKELETONS = skeletons


def _build_worker(
    task: tuple[int, int, int, str],
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    host_index, seed, attempts, mode = task
    host = _BUILD_HOSTS[host_index]
    rng = np.random.default_rng(seed)
    candidates = [row for row in _BUILD_SKELETONS if row.surface == host.surface]
    if not candidates:
        return None, {"no_skeleton": 1}
    reasons: dict[str, int] = {}
    for _ in range(attempts):
        skeleton = candidates[int(rng.integers(len(candidates)))]
        try:
            point = build_point(host, skeleton, rng, reasons=reasons, mode=mode)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            reasons["exception"] = reasons.get("exception", 0) + 1
            continue
        if point is not None:
            return point, reasons
    return None, reasons


def bounded_ordered_map(pool, function, tasks: Sequence, batch_size: int):
    """Keep deterministic task order without eagerly launching the entire search.

    Stopping after enough valid points wastes at most the remainder of one batch,
    not hundreds of queued simulations. Works on supported Python 3.12+ runtimes.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(tasks), batch_size):
        yield from pool.map(function, tasks[start : start + batch_size], chunksize=1)


def generate(
    *,
    cohort_root: Path,
    corpus_root: Path,
    output_root: Path,
    points: int,
    workers: int,
    seed: int,
    skeleton_files: int,
    rungs: Sequence[str],
    noise_path: Path,
    emission_realism_path: Path | None = None,
    lob_points: int = 26,
    min_lob_points: int = 20,
    exit_points: int = 14,
    min_exit_points: int = 10,
    emission_frame_types: Sequence[str] = EMISSION_FRAME_TYPES,
) -> dict[str, Any]:
    from cv.validation.s6_bench import EmissionRealismSamplers, EmpiricalSamplers

    if points < 1 or workers < 1:
        raise ValueError("points and workers must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            "generation requires a new/empty output directory; preserve frozen runs"
        )
    if not 0 <= min_lob_points <= lob_points or not 0 <= min_exit_points <= exit_points:
        raise ValueError("special-point minima must be nonnegative and no larger than quotas")
    if lob_points + exit_points > points:
        raise ValueError("lob and exit quotas exceed the total point count")
    # Freeze generator, physical law and source selection before any long simulation.
    inputs = {
        Path(__file__),
        Path(hff.__file__),
        Path(flight.__file__),
        Path(bounce_reference.__file__),
        Path(camera_cal.__file__),
        REPO_ROOT / "physics" / "impact.py",
        noise_path,
    }
    inputs.update(cohort_root.glob("*.json"))
    inputs.update(cohort_root.glob("*/*.npz"))
    half = max(skeleton_files // 2, 1)
    for pattern in ("*roland_garros*.csv", "*australian_open*.csv"):
        for path in sorted((corpus_root / "ball_trajectory").glob(pattern))[:half]:
            inputs.add(path)
            pbp = (
                corpus_root / "play_by_play" / path.name.replace("_ball_trajectory.csv", "_pbp.csv")
            )
            if pbp.is_file():
                inputs.add(pbp)
    if REALISTIC_V2 in rungs and emission_realism_path is not None:
        inputs.add(emission_realism_path)
    ordered_inputs = sorted(inputs)
    bindings = [
        {"resolved_path": str(p.resolve()), "record": file_record(p)} for p in ordered_inputs
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    generation_manifest = {
        "schema": "s6_complete_truth_generation_v1",
        "status": "running",
        "code": git_record(REPO_ROOT),
        "input_bindings": bindings,
        "configuration": {
            "seed": seed,
            "points": points,
            "workers": workers,
            "skeleton_files": skeleton_files,
            "lob_points": lob_points,
            "min_lob_points": min_lob_points,
            "exit_points": exit_points,
            "min_exit_points": min_exit_points,
            "rungs": list(rungs),
            "emission_frame_types": list(emission_frame_types),
        },
        "scope": "synthetic evaluation; oracle event inputs including hidden physical endings",
    }
    write_json(output_root / "generation_manifest.json", generation_manifest)
    hosts = load_hosts(cohort_root)
    if not hosts:
        raise RuntimeError(f"no camera-accepted host points under {cohort_root}")
    skeletons = load_skeletons(corpus_root, max_files=skeleton_files)
    # A lob and an over-hit both need the longest arc, so those tasks get the hosts with the
    # longest active spans.
    longest = sorted(
        range(len(hosts)),
        key=lambda index: -(hosts[index].span[1] - hosts[index].span[0]) / hosts[index].fps,
    )
    special = [
        (
            MODE_LOB,
            min(lob_points, points),
            min_lob_points,
            [
                (longest[index % len(longest)], _mix(seed, "lob", index), 8, MODE_LOB)
                for index in range(int(lob_points * 6) + 24)
            ],
        ),
        (
            MODE_TERMINAL_EXIT,
            min(exit_points, points),
            min_exit_points,
            [
                (longest[index % len(longest)], _mix(seed, "exit", index), 8, MODE_TERMINAL_EXIT)
                for index in range(int(exit_points * 8) + 32)
            ],
        ),
    ]
    tasks = [
        (index % len(hosts), _mix(seed, index), 6, MODE_RALLY)
        for index in range(int(points * 1.6) + 16)
    ]
    built: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_build, initargs=(hosts, skeletons)
    ) as pool:
        for mode, wanted, minimum, mode_tasks in special:
            if wanted == 0:
                continue
            made = 0
            for point, row_reasons in bounded_ordered_map(pool, _build_worker, mode_tasks, workers):
                for key, value in row_reasons.items():
                    reasons[f"{mode}:{key}"] += value
                if point is not None:
                    built.append(point)
                    made += 1
                if made >= wanted:
                    break
            if made < minimum:
                raise RuntimeError(f"only generated {made} {mode} points, wanted {minimum}")
        remaining_tasks = tasks if len(built) < points else []
        for point, row_reasons in bounded_ordered_map(
            pool, _build_worker, remaining_tasks, workers
        ):
            for key, value in row_reasons.items():
                reasons[key] += value
            if point is not None:
                built.append(point)
            if len(built) >= points:
                break
    if len(built) < points:
        raise RuntimeError(f"only generated {len(built)} of {points} synthetic points")
    built = built[:points]

    per_match: dict[str, int] = defaultdict(int)
    for point in built:
        match_id = point["host"].split("__", 1)[0]
        per_match[match_id] += 1
        point["match_id"] = match_id
        point["clip"] = f"pt{per_match[match_id]:04d}"
        point["point"] = f"{match_id}__{point['clip']}"
        # The cohort broadcast whose camera this point borrows.  A fitter may share per-match
        # parameters across every point with the same ``source_match``.
        point["source_match"] = match_id
        point.setdefault("source_point", point["host"])

    host_index = {host.key: host for host in hosts}
    camera_root = output_root / "cameras"
    write_cameras(camera_root, host_index, built)

    truth = {
        "schema": "tennis.s6-point-bench-truth.v5",
        "seed": seed,
        "truth_definition": COMPLETE_TRUTH_DEFINITION,
        "generated": len(built),
        "cohort_root": os.fspath(cohort_root),
        "corpus_root": os.fspath(corpus_root),
        "aero_params": AERO_PARAMS,
        "bounce_model": {
            "module": "physics.bounce_reference",
            "measured": bounce_reference.MEASURED,
            "provenance": bounce_reference.PROVENANCE,
            "dwell_seconds": bounce_reference.DWELL_SECONDS,
        },
        "generation_rejections": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        "points": [
            {
                key: value
                for key, value in point.items()
                if key not in {"track_pixels", "positions", "track"}
            }
            for point in built
        ],
    }
    if any(
        file_record(p) != binding["record"]
        for p, binding in zip(ordered_inputs, bindings, strict=True)
    ):
        raise RuntimeError("generation inputs changed during simulation")
    write_json(output_root / "truth.json", truth)

    noise = json.loads(noise_path.read_text())
    realism = None
    if REALISTIC_V2 in rungs:
        if emission_realism_path is None or not emission_realism_path.is_file():
            raise RuntimeError(
                f"rung {REALISTIC_V2} needs the measured real emission model; run "
                "`python -m cv.validation.s6_bench emission-model` first "
                f"(looked for {emission_realism_path})"
            )
        realism = EmissionRealismSamplers(json.loads(emission_realism_path.read_text()))
    summary = {}
    for rung in rungs:
        rng = np.random.default_rng(_mix(seed, rung))
        samplers = EmpiricalSamplers(noise)
        corrupted = [
            apply_rung(
                point,
                rung,
                samplers,
                rng,
                emission_frame_types=emission_frame_types,
                realism=realism,
            )
            for point in built
        ]
        summary[rung] = materialize(
            output_root / "roots" / rung,
            host_index,
            corrupted,
            camera_root=camera_root,
            rung=rung,
            seed=seed,
            emission_frame_types=emission_frame_types,
        )
    write_json(
        output_root / "generation.json",
        {
            "schema": "tennis.s6-point-bench-generation.v1",
            "hosts_available": len(hosts),
            "skeletons_available": len(skeletons),
            "points": len(built),
            "flights": sum(len(point["flights"]) for point in built),
            "mode_histogram": _counts(point.get("mode", MODE_RALLY) for point in built),
            "termination_kind_histogram": _counts(point["termination_kind"] for point in built),
            "truth_schema": truth["schema"],
            "physical_endings_outside_observation": sum(
                point["observation_exit_frame"] is not None for point in built
            ),
            "first_bounce_in_bounds_points": sum(
                1 for point in built if point["first_bounce_in_bounds"]
            ),
            "lob_points": sum(1 for point in built if point.get("lob")),
            "points_with_top_edge_exit": sum(1 for point in built if point.get("top_exit_frames")),
            "points_with_out_of_frame_gap": sum(
                1 for point in built if point.get("out_of_frame_frames")
            ),
            "out_of_frame_frames": sum(
                len(point.get("out_of_frame_frames") or []) for point in built
            ),
            "top_exit_frames": sum(len(point.get("top_exit_frames") or []) for point in built),
            "flights_with_out_of_frame_gap": sum(
                1 for point in built for row in point["flights"] if row.get("out_of_frame_frames")
            ),
            "source_match_histogram": _counts(point["source_match"] for point in built),
            "scored_terminal_bounces": sum(
                row["scored_bounces"]
                for point in built
                for row in point["flights"]
                if row["terminal"]
            ),
            "generated_terminal_bounces": sum(
                len(row["bounces"])
                for point in built
                for row in point["flights"]
                if row["terminal"]
            ),
            "shots_histogram": _counts(point["shots"] for point in built),
            "terminal_bounce_histogram": _counts(point["terminal_bounces"] for point in built),
            "dead_ball_bounce_histogram": _counts(point["dead_ball_bounces"] for point in built),
            "surface_histogram": _counts(point["surface"] for point in built),
            "fps_histogram": _counts(point["fps"] for point in built),
            "rejections": dict(sorted(reasons.items(), key=lambda item: -item[1])),
            "emission_frame_types": list(emission_frame_types),
            "rungs": summary,
            "noise_model": {
                "path": os.fspath(noise_path),
                "sha256": _sha256(noise_path),
            },
            "emission_realism_model": None
            if emission_realism_path is None or not emission_realism_path.is_file()
            else {
                "path": os.fspath(emission_realism_path),
                "sha256": _sha256(emission_realism_path),
                "parameters": REALISTIC_V2_PARAMETERS,
            },
        },
    )
    if any(
        file_record(p) != binding["record"]
        for p, binding in zip(ordered_inputs, bindings, strict=True)
    ):
        raise RuntimeError("generation inputs changed during materialization")
    generation_manifest.update(status="complete", truth=file_record(output_root / "truth.json"))
    write_json(output_root / "generation_manifest.json", generation_manifest)
    return summary


def _mix(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(("\0".join([str(seed), *map(str, parts)])).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def command_generate(args: argparse.Namespace) -> None:
    summary = generate(
        cohort_root=args.cohort_root,
        corpus_root=args.corpus_root,
        output_root=args.output_root,
        points=args.points,
        workers=args.workers,
        seed=args.seed,
        skeleton_files=args.skeleton_files,
        rungs=args.rung or list(RUNGS),
        emission_realism_path=args.emission_realism,
        noise_path=args.noise_model,
        lob_points=args.lob_points,
        min_lob_points=args.min_lob_points,
        exit_points=args.terminal_exit_points,
        min_exit_points=args.min_terminal_exit_points,
        emission_frame_types=tuple(
            value for value in (args.emission_frame_type or EMISSION_FRAME_TYPES) if value != "none"
        ),
    )
    for rung, row in summary.items():
        print(f"{rung}: {row['points']} points, {row['emissions']} emissions")


def command_run(args: argparse.Namespace) -> None:
    """Fit one rung with one arm.

    ``--root`` and ``--run-root`` are separate arguments for the same reason ``audit``'s roots
    are: a bench owned by one package has to be runnable by another without writing into it.
    """
    outcome = run_reconstruction(
        getattr(args, "root", None) or (args.output_root / "roots" / args.rung),
        getattr(args, "run_root", None) or (args.output_root / "runs" / f"{args.arm}__{args.rung}"),
        arm=args.arm,
        workers=args.workers,
    )
    print(f"{args.arm}/{args.rung}: {outcome['wall_seconds']:.1f} s -> {outcome['report']}")


def command_audit(args: argparse.Namespace) -> None:
    """Score every finished run under ``--runs-root`` against ``--truth``.

    The four roots are separate arguments so a bench owned by one package can be re-scored with
    a new scorer without writing into that package's output tree.
    """
    truth_path = args.truth or (args.output_root / "truth.json")
    runs_root = args.runs_root or (args.output_root / "runs")
    audits_root = args.audits_root or (args.output_root / "audits")
    camera_root = args.cameras_root or (args.output_root / "cameras")
    truth = json.loads(truth_path.read_text())
    cache_root = args.truth_cache_root or (audits_root / "truth_paths")
    truth_paths = truth_scored_paths(
        truth,
        workers=max(1, int(getattr(args, "workers", 1) or 1)),
        cache_path=cache_root / f"{_sha256(truth_path)[:16]}.npz",
    )
    table = []
    for directory in sorted(runs_root.iterdir()):
        report_path = directory / "report.json"
        if not report_path.is_file():
            continue
        if args.run and directory.name not in set(args.run):
            continue
        arm, rung = directory.name.split("__", 1)
        scored = audit_report(
            truth,
            json.loads(report_path.read_text()),
            camera_root,
            directory / "anchors",
            criterion=args.criterion,
            truth_paths=truth_paths,
        )
        write_json(audits_root / f"{directory.name}.json", scored)
        _write_csv(audits_root / f"{directory.name}_flights.csv", scored["flight_rows"])
        _write_csv(audits_root / f"{directory.name}_points.csv", scored["point_rows"])
        _write_csv(
            audits_root / f"{directory.name}_source_matches.csv", scored["source_match_rows"]
        )
        table.append(
            {
                "arm": arm,
                "rung": rung,
                "criterion": args.criterion,
                "complete_points_pixel_v1": scored["by_criterion"][CRITERION_PIXEL_V1][
                    "complete_points"
                ],
                "complete_points_metric_v2": scored["by_criterion"][CRITERION_METRIC_V2][
                    "complete_points"
                ],
                "truth_good_accepted_pixel_v1": scored["by_criterion"][CRITERION_PIXEL_V1][
                    "truth_good_accepted_flights"
                ],
                "truth_good_accepted_metric_v2": scored["by_criterion"][CRITERION_METRIC_V2][
                    "truth_good_accepted_flights"
                ],
                "wrong_accepts_pixel_v1": scored["by_criterion"][CRITERION_PIXEL_V1][
                    "wrong_accepted_flights"
                ],
                "wrong_accepts_metric_v2": scored["by_criterion"][CRITERION_METRIC_V2][
                    "wrong_accepted_flights"
                ],
                "trajectory_rms_median_m": scored["trajectory_rms_m"]["median"],
                "mid_air_stop_flights": scored["mid_air_stop_flights"],
                **{
                    key: scored[key]
                    for key in (
                        "points",
                        "flights",
                        "attempted_flights",
                        "solved_flights",
                        "accepted_flights",
                        "truth_good_flights",
                        "truth_good_accepted_flights",
                        "wrong_accepted_flights",
                        "wrong_accept_rate",
                        "complete_points",
                        "accepted_all_flight_points",
                        "complete_points_excluding_terminal",
                        "lob_points",
                        "complete_lob_points",
                        "source_matches",
                        "dead_ball_bounces_excluded",
                        "terminal_flights_fitting_dead_ball",
                        "flights_with_out_of_frame_gap",
                        "two_bounce_terminal_flights",
                    )
                },
                "bounce_error_median_m": scored["bounce_error_m"]["median"],
                "contact_error_median_px": scored["contact_error_px"]["median"],
                "held_out_median_px": scored["held_out_median_px"]["median"],
                "first_cause_counts": scored["first_cause_counts"],
                "termination_kind_counts": scored["termination_kind_counts"],
                "termination_anchor_source_counts": scored["termination_anchor_source_counts"],
                "complete_points_by_source_match": scored["complete_points_by_source_match"],
            }
        )
    write_json(
        audits_root / "summary.json",
        {
            "schema": "tennis.s6-point-bench-audit.v4",
            "truth_definition": truth.get("truth_definition", TRUTH_DEFINITION),
            "criterion": args.criterion,
            "criterion_definition": CRITERION_DEFINITION,
            "truth": os.fspath(truth_path),
            "truth_sha256": _sha256(truth_path),
            "runs_root": os.fspath(runs_root),
            "rows": table,
        },
    )
    _write_csv(
        audits_root / "summary.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "first_cause_counts",
                    "termination_kind_counts",
                    "termination_anchor_source_counts",
                    "complete_points_by_source_match",
                }
            }
            for row in table
        ],
    )
    for row in table:
        print(
            f"{row['arm']:>22} {row['rung']:>16}  accepted {row['accepted_flights']:>4}/"
            f"{row['flights']}  complete pixel_v1 {row['complete_points_pixel_v1']:>3} -> "
            f"metric_v2 {row['complete_points_metric_v2']:>3} / {row['points']}  "
            f"right {row['truth_good_accepted_pixel_v1']:>4} -> "
            f"{row['truth_good_accepted_metric_v2']:>4}  "
            f"wrong {row['wrong_accepts_pixel_v1']:>4} -> {row['wrong_accepts_metric_v2']:>4}"
        )


# --------------------------------------------------------------------------------------- #
# The development inner loop: a frozen 12-point subset, fitted and scored in seconds
# --------------------------------------------------------------------------------------- #
# The 200-point sweep answers "is the arm better"; it takes minutes and it is read as a table.
# This subset answers "what is the fit actually doing", in seconds, on points a person can hold
# in their head and look at.  The quotas name the shapes the fitter has to get right, in the
# order they are scarce in the bench, so the rarest constraint is satisfied first.
DEV_QUOTAS = (
    ("terminal_bounce", 2, "rally-closing flight ends at a first bounce that is OUT"),
    ("out_of_view", 2, "rally-closing flight ends where the ball leaves the field of view"),
    ("lob_gap", 2, "lob whose projected ball leaves the top of the frame, so the track has a hole"),
    ("long_rally", 2, "four or more shots, so at least three junctions are fitted"),
    ("second_bounce", 4, "rally-closing flight ends at the SECOND bounce: two bounce anchors"),
)
DEV_MIN_SOURCE_MATCHES = 6
DEV_POINTS_NAME = "dev_points.json"


def _dev_category(point: dict[str, Any]) -> list[str]:
    """Every quota bucket this truth point can serve, most constrained first."""
    buckets = []
    if point.get("lob") and point.get("out_of_frame_frames"):
        buckets.append("lob_gap")
    if int(point.get("shots") or 0) >= 4:
        buckets.append("long_rally")
    kind = str(point.get("termination_kind"))
    if kind in TERMINATION_KINDS:
        buckets.append(kind)
    return buckets


def select_dev_points(
    truth: dict[str, Any],
    *,
    quotas: Sequence[tuple[str, int, str]] = DEV_QUOTAS,
    min_source_matches: int = DEV_MIN_SOURCE_MATCHES,
) -> list[dict[str, Any]]:
    """Pick the frozen development subset: the quotas, over >=6 matches and both surfaces.

    Deterministic: candidates are ordered by point key, and each pick prefers a source match and
    a surface that the subset does not have yet, so the spread is a consequence of the rule
    rather than of the corpus order.
    """
    points = {row["point"]: row for row in truth["points"]}
    taken: dict[str, dict[str, Any]] = {}
    used_matches: dict[str, int] = defaultdict(int)
    used_surfaces: dict[str, int] = defaultdict(int)
    for name, wanted, why in quotas:
        candidates = sorted(
            (
                row
                for row in points.values()
                if name in _dev_category(row) and row["point"] not in taken
            ),
            key=lambda row: row["point"],
        )
        for _ in range(wanted):
            if not candidates:
                raise RuntimeError(f"no candidate left for development quota {name!r}")
            best = min(
                candidates,
                key=lambda row: (
                    used_matches[row["source_match"]],
                    used_surfaces[row["surface"]],
                    row["point"],
                ),
            )
            candidates.remove(best)
            used_matches[best["source_match"]] += 1
            used_surfaces[best["surface"]] += 1
            taken[best["point"]] = {
                "point": best["point"],
                "category": name,
                "reason": why,
                "source_match": best["source_match"],
                "surface": best["surface"],
                "fps": best["fps"],
                "shots": best["shots"],
                "flights": len(best["flights"]),
                "termination_kind": best["termination_kind"],
                "lob": bool(best.get("lob")),
                "mode": best.get("mode"),
                "terminal_bounces": best["terminal_bounces"],
                "out_of_frame_frames": len(best.get("out_of_frame_frames") or []),
                "top_exit_frames": len(best.get("top_exit_frames") or []),
                "start_frame": best.get("start_frame"),
                "end_frame": best.get("end_frame"),
            }
    rows = list(taken.values())
    matches = {row["source_match"] for row in rows}
    surfaces = {row["surface"] for row in rows}
    if len(matches) < min_source_matches:
        raise RuntimeError(
            f"development subset spans {len(matches)} source matches, wanted {min_source_matches}"
        )
    if len(surfaces) < 2:
        raise RuntimeError(f"development subset covers one surface only: {sorted(surfaces)}")
    return sorted(rows, key=lambda row: (row["category"], row["point"]))


def command_dev_select(args: argparse.Namespace) -> None:
    truth = json.loads((args.truth or (args.output_root / "truth.json")).read_text())
    rows = select_dev_points(truth)
    payload = {
        "schema": "tennis.s6-point-bench-dev-points.v1",
        "frozen": True,
        "note": (
            "The development subset is frozen.  It is looked at by eye before any 200-point "
            "sweep is read; the sweep is what happens after the subset looks right."
        ),
        "truth": os.fspath(args.truth or (args.output_root / "truth.json")),
        "quotas": [
            {"category": name, "points": count, "reason": why} for name, count, why in DEV_QUOTAS
        ],
        "points": rows,
        "source_matches": sorted({row["source_match"] for row in rows}),
        "surfaces": _counts(row["surface"] for row in rows),
        "termination_kinds": _counts(row["termination_kind"] for row in rows),
        "shots": _counts(row["shots"] for row in rows),
    }
    target = args.dev_points or (args.output_root / DEV_POINTS_NAME)
    write_json(target, payload)
    for row in rows:
        print(
            f"{row['point']:<52} {row['category']:<15} {row['surface']:<5} "
            f"{row['shots']} shots  {row['termination_kind']}"
        )
    print(
        f"{len(rows)} points, {len(payload['source_matches'])} source matches, "
        f"surfaces {payload['surfaces']} -> {target}"
    )


def dev_point_keys(args: argparse.Namespace) -> list[str]:
    if args.points:
        return list(dict.fromkeys(args.points))
    path = args.dev_points or (args.output_root / DEV_POINTS_NAME)
    return [str(row["point"]) for row in json.loads(path.read_text())["points"]]


def _cell(value: Any, spec: str = "7.2f") -> str:
    if value is None:
        return f"{'-':>{spec.split('.')[0]}}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{value:>{spec.split('.')[0]}}"
    if not math.isfinite(number):
        return f"{'inf':>{spec.split('.')[0]}}"
    return f"{number:{spec}}"


def print_dev_table(rows: Sequence[dict[str, Any]], point_rows: Sequence[dict[str, Any]]) -> None:
    """One compact table per flight, then one line per point.  Read top to bottom."""
    header = (
        f"{'point':<34} {'fl':>2} {'kind':<15} {'good':>4} {'bounce_m':>8} "
        f"{'c0_px':>7} {'c0_m':>6} {'c1_px':>7} {'c1_m':>6} {'junc_m':>7} {'term_m':>7} "
        f"{'rms_m':>7} {'stop':>4} {'status':<9} {'reason':<34}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        status = (
            "accepted"
            if row["accepted"]
            else "solved"
            if row["solved"]
            else "attempted"
            if row["attempted"]
            else "skipped"
        )
        print(
            f"{row['point'][:34]:<34} "
            f"{row['flight_index']:>2} "
            f"{(row['termination_kind'] or 'rally'):<15} "
            f"{('yes' if row['truth_good'] else 'no'):>4} "
            f"{_cell(row['bounce_error_m'], '8.3f')} "
            f"{_cell(row['start_contact_error_px'], '7.1f')} "
            f"{_cell(row['start_contact_error_m'], '6.2f')} "
            f"{_cell(row['end_contact_error_px'], '7.1f')} "
            f"{_cell(row['end_contact_error_m'], '6.2f')} "
            f"{_cell(row['junction_gap_next_m'], '7.2f')} "
            f"{_cell(row['termination_error_m'] if row['termination_error_m'] is not None else row['exit_error_m'], '7.2f')} "
            f"{_cell(row['trajectory_rms_m'], '7.2f')} "
            f"{('yes' if row['mid_air_stop'] else ('-' if row['mid_air_stop'] is None else 'no')):>4} "
            f"{status:<9} {str(row['failure_reason'] or '')[:34]:<34}"
        )
    print()
    print(f"{'point':<52} {'complete':>8} {'pixel_v1':>9}  first cause")
    for row in point_rows:
        print(
            f"{row['point']:<52} {('yes' if row['complete_point'] else 'no'):>8} "
            f"{('yes' if row['complete_point_pixel_v1'] else 'no'):>9}  "
            f"{row['first_cause']}"
        )
    print()
    print(
        "term_m = termination error (bounce endings) or the error at the last visible frame "
        "(out_of_view);\nrms_m = 3D RMS against the re-integrated truth curve over the "
        "flight's frames; stop = the fit ends in mid-air before the flight does."
    )


def command_dev_run(args: argparse.Namespace) -> None:
    """Fit ONLY the development points, then print the per-flight table.  Seconds, not minutes."""
    keys = dev_point_keys(args)
    root = args.root or (args.output_root / "roots" / args.rung)
    label = args.label or f"{args.arm}__{args.rung}"
    output_root = args.run_root or (args.output_root / "dev_runs" / label)
    truth_path = args.truth or (args.output_root / "truth.json")
    camera_root = args.cameras_root or (args.output_root / "cameras")
    started = time.perf_counter()
    if args.reuse_report:
        # Re-score a report that is already on disk: the criterion changed, the fit did not.
        source = args.reuse_from or output_root
        outcome = {"report": os.fspath(source / "report.json")}
        anchors_root = source / "anchors"
    else:
        outcome = run_reconstruction(
            root, output_root, arm=args.arm, workers=args.workers, points=keys
        )
        anchors_root = output_root / "anchors"
    fit_seconds = time.perf_counter() - started
    truth = json.loads(truth_path.read_text())
    report = json.loads(Path(outcome["report"]).read_text())
    cache_root = args.truth_cache_root or (args.output_root / "truth_paths")
    scored = audit_report(
        truth,
        report,
        camera_root,
        anchors_root,
        criterion=args.criterion,
        truth_paths=truth_scored_paths(
            truth,
            workers=max(1, int(args.workers or 1)),
            cache_path=cache_root / f"{_sha256(truth_path)[:16]}.npz",
            keys=keys,
        ),
    )
    write_json(output_root / "dev_audit.json", scored)
    _write_csv(output_root / "dev_flights.csv", scored["flight_rows"])
    _write_csv(output_root / "dev_points.csv", scored["point_rows"])
    print_dev_table(scored["flight_rows"], scored["point_rows"])
    total = time.perf_counter() - started
    print()
    print(
        f"{len(keys)} points, {scored['flights']} flights, {args.workers} workers: "
        f"fit {fit_seconds:.1f} s, fit+score {total:.1f} s"
    )
    old = scored["by_criterion"][CRITERION_PIXEL_V1]
    new = scored["by_criterion"][CRITERION_METRIC_V2]
    print(
        f"accepted {scored['accepted_flights']}/{scored['flights']}  "
        f"complete pixel_v1 {old['complete_points']} -> metric_v2 "
        f"{new['complete_points']} / {scored['points']}  "
        f"right {old['truth_good_accepted_flights']} -> "
        f"{new['truth_good_accepted_flights']}  "
        f"wrong {old['wrong_accepted_flights']} -> {new['wrong_accepted_flights']}  "
        f"mid-air stops {scored['mid_air_stop_flights']}  -> {output_root}"
    )


# --------------------------------------------------------------------------------------- #
# The same inner loop on REAL cohort points, with the owner's clicks in place of truth
# --------------------------------------------------------------------------------------- #
# A real point has no synthetic truth.  What it has is the owner's own clicks: bounce clicks,
# which fix a court-plane position exactly; contact clicks and positioned frames, which fix an
# image ray and say nothing about depth.  So the real table reports metres at the bounce, pixels
# at the contact, and leaves the metric contact columns empty rather than inventing them.
REAL_ROOT_RELATIVE = Path("processed/wk3_eventthreshold/default_cohort")
REAL_TRUTH_EVENTS_RELATIVE = Path(
    "processed/wk3_oracle/oracle_3d_ceiling_v2/truth_events_event_model_v3.json"
)


def real_dev_rows(
    report_path: Path, camera_root: Path, truth_events: Path, automatic_events: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-flight owner-witness rows and per-point causes, from the frozen gate-audit scorer.

    The witnesses are ``docs/wk1/gate_audit.md``'s: an owner bounce click projected to the court
    plane (metres), an owner contact click at a flight boundary (pixels), and the owner-positioned
    native frames the fitter never sees (pixels).
    """
    from cv.pipeline.flight_ledger import build as build_flight_ledger
    from cv.validation import flight_gate_audit as fga

    report = json.loads(report_path.read_text())
    ledger = build_flight_ledger([report_path])
    rows = fga.build_flight_rows(report, ledger, camera_root, truth_events)
    reasons = {
        (str(row["point"]), int(row["flight_index"])): list(row.get("reasons") or [])
        for row in ledger["rows"]
    }
    for row in rows:
        row["ledger_reasons"] = "+".join(reasons.get((row["point"], int(row["flight_index"])), []))
    causes = fga.build_point_causes(report, ledger, rows, truth_events, automatic_events)
    return rows, causes


def print_real_dev_table(rows: Sequence[dict[str, Any]], causes: Sequence[dict[str, Any]]) -> None:
    header = (
        f"{'point':<34} {'fl':>2} {'kind':<15} {'good':>4} {'bounce_m':>8} "
        f"{'c0_px':>7} {'c0_m':>6} {'c1_px':>7} {'c1_m':>6} {'junc_m':>7} {'term_m':>7} "
        f"{'status':<9} {'reason':<34}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        status = (
            "accepted"
            if row.get("pixel_accepted")
            else "solved"
            if row.get("solved")
            else "attempted"
        )
        reason = ""
        if not row.get("solved"):
            reason = "solver_returned_none"
        elif not row.get("pixel_accepted"):
            reason = "pixel_gate:" + str(row.get("ledger_reasons") or "")
        elif row.get("truth_good") is False:
            reason = "wrong_accept"
        elif row.get("truth_good") is None:
            reason = "no_owner_witness"
        print(
            f"{row['point'][:34]:<34} {row['flight_index']:>2} "
            f"{('terminal' if row.get('terminal_end') else 'rally'):<15} "
            f"{ ({True: 'yes', False: 'no', None: '-'}[row.get('truth_good')]):>4} "
            f"{_cell(row.get('truth_bounce_error_max_m'), '8.3f')} "
            f"{_cell(row.get('truth_contact_start_px'), '7.1f')} "
            f"{'-':>6} "
            f"{_cell(row.get('truth_contact_end_px'), '7.1f')} "
            f"{'-':>6} "
            f"{_cell(row.get('junction_gap_next_m'), '7.2f')} "
            f"{'-':>7} "
            f"{status:<9} {reason[:34]:<34}"
        )
    print()
    print(f"{'point':<52} {'complete':>8}  first cause")
    for row in causes:
        print(
            f"{str(row['point']):<52} {('yes' if row.get('cause') == 'complete' else 'no'):>8}  {row.get('cause')}"
        )


def command_dev_run_real(args: argparse.Namespace) -> None:
    keys = dev_point_keys(args)
    root = args.root or (_data_root() / REAL_ROOT_RELATIVE)
    camera_root = args.cameras_root or root
    truth_events = args.truth_events or (_data_root() / REAL_TRUTH_EVENTS_RELATIVE)
    output_root = args.run_root or (args.output_root / "dev_runs" / (args.label or "real"))
    started = time.perf_counter()
    report_path = output_root / "report.json"
    if args.reuse_report and report_path.is_file():
        outcome = {"report": os.fspath(report_path), "wall_seconds": 0.0}
    else:
        outcome = run_reconstruction(
            root, output_root, arm=args.arm, workers=args.workers, points=keys
        )
    fit_seconds = time.perf_counter() - started
    rows, causes = real_dev_rows(
        Path(outcome["report"]), camera_root, truth_events, root / EVENT_NAME
    )
    write_json(output_root / "dev_real_flights.json", {"rows": rows, "causes": causes})
    _write_csv(output_root / "dev_real_flights.csv", rows)
    print_real_dev_table(rows, causes)
    total = time.perf_counter() - started
    witnessed = [row for row in rows if row.get("truth_good") is not None]
    print()
    print(
        f"{len(keys)} points, {len(rows)} flights, {args.workers} workers: "
        f"fit {fit_seconds:.1f} s, fit+score {total:.1f} s"
    )
    print(
        f"solved {sum(bool(row.get('solved')) for row in rows)}/{len(rows)}  "
        f"pixel-accepted {sum(bool(row.get('pixel_accepted')) for row in rows)}  "
        f"owner-witnessed {len(witnessed)}  "
        f"truth-good {sum(bool(row.get('truth_good')) for row in witnessed)}  -> {output_root}"
    )


def command_dev_run_dispatch(args: argparse.Namespace) -> None:
    if args.real:
        command_dev_run_real(args)
    else:
        command_dev_run(args)


REAL_DEV_POINTS = 6
REAL_DEV_MIN_OWNER_FRAMES = 5


def select_real_dev_points(
    root: Path,
    truth_events: Path,
    *,
    wanted: int = REAL_DEV_POINTS,
    min_owner_frames: int = REAL_DEV_MIN_OWNER_FRAMES,
) -> list[dict[str, Any]]:
    """Real cohort points that carry enough owner truth to judge a fit by.

    The rule is: the point must be in the fitter's own point universe, it must have owner event
    clicks, and it must have owner-positioned native frames -- the witness the fitter never sees.
    Held-out broadcasts are excluded so the development loop cannot be tuned on the sealed split.
    One point per broadcast, richest first.
    """
    from cv.validation import flight_gate_audit as fga

    active = json.loads((root / "active_play_v1.json").read_text())
    universe = {str(key).replace("/", "__", 1) for key in active}
    events = fga.owner_event_index(truth_events)
    positions = fga.owner_position_index()
    candidates = []
    for point, clicks in events.items():
        if point not in universe or fga.scope_of(point) != "other_38":
            continue
        match_id, clip = point.split("__", 1)
        frames = len(positions.get((match_id, clip), {}))
        if frames < min_owner_frames:
            continue
        candidates.append(
            {
                "point": point,
                "source_match": match_id,
                "owner_positioned_frames": frames,
                "owner_contacts": len(clicks["contact"]),
                "owner_bounces": len(clicks["bounce"]),
                "owner_net_hits": len(clicks["net_hit"]),
                "scope": "other_38",
                "reason": (
                    "real point with owner event clicks and "
                    f"{frames} owner-positioned native frames; development scope"
                ),
            }
        )
    candidates.sort(key=lambda row: (-row["owner_positioned_frames"], row["point"]))
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in candidates:
        if row["source_match"] in used:
            continue
        chosen.append(row)
        used.add(row["source_match"])
        if len(chosen) >= wanted:
            break
    if len(chosen) < wanted:
        raise RuntimeError(f"only {len(chosen)} real development points, wanted {wanted}")
    return chosen


def command_dev_select_real(args: argparse.Namespace) -> None:
    root = args.root or (_data_root() / REAL_ROOT_RELATIVE)
    truth_events = args.truth_events or (_data_root() / REAL_TRUTH_EVENTS_RELATIVE)
    rows = select_real_dev_points(root, truth_events)
    target = args.dev_points or (args.output_root / "real_dev_points.json")
    write_json(
        target,
        {
            "schema": "tennis.s6-point-bench-dev-points.v1",
            "frozen": True,
            "mode": "real",
            "root": os.fspath(root),
            "truth_events": os.fspath(truth_events),
            "points": rows,
            "source_matches": sorted({row["source_match"] for row in rows}),
        },
    )
    for row in rows:
        print(
            f"{row['point']:<62} {row['owner_positioned_frames']:>4} positioned frames, "
            f"{row['owner_contacts']} contacts, {row['owner_bounces']} bounces"
        )
    print(f"{len(rows)} real points -> {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--cohort-root", type=Path, default=_data_root() / COHORT_RELATIVE)
    generate_parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    generate_parser.add_argument("--noise-model", type=Path, default=_data_root() / NOISE_RELATIVE)
    generate_parser.add_argument("--points", type=int, default=200)
    generate_parser.add_argument("--workers", type=int, default=8)
    generate_parser.add_argument("--seed", type=int, default=SEED)
    generate_parser.add_argument("--skeleton-files", type=int, default=40)
    generate_parser.add_argument("--lob-points", type=int, default=26)
    generate_parser.add_argument("--min-lob-points", type=int, default=20)
    generate_parser.add_argument("--terminal-exit-points", type=int, default=14)
    generate_parser.add_argument("--min-terminal-exit-points", type=int, default=10)
    generate_parser.add_argument("--rung", action="append", choices=list(RUNGS))
    generate_parser.add_argument(
        "--emission-realism",
        type=Path,
        default=_data_root() / EMISSION_REALISM_RELATIVE,
        help=(
            "the measured real emission model realistic_v2 draws its parameters from "
            "(cv.validation.s6_bench emission-model)"
        ),
    )
    generate_parser.add_argument(
        "--emission-frame-type",
        action="append",
        choices=[*EMISSION_FRAME_TYPES, "none"],
        help=(
            "which emissions leave the generator on a native frame with the track's own pixel; "
            "defaults to all three.  A diagnostic knob: anything omitted keeps the "
            "pre-benchframes continuous event time, which is not what the real chain emits."
        ),
    )
    generate_parser.set_defaults(handler=command_generate)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--rung", required=True, choices=list(RUNGS))
    run_parser.add_argument("--arm", default="default", choices=sorted(ARMS))
    run_parser.add_argument("--workers", type=int, default=8)
    run_parser.add_argument("--root", type=Path, default=None)
    run_parser.add_argument("--run-root", type=Path, default=None)
    run_parser.set_defaults(handler=command_run)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--truth", type=Path, default=None)
    audit_parser.add_argument("--runs-root", type=Path, default=None)
    audit_parser.add_argument("--audits-root", type=Path, default=None)
    audit_parser.add_argument("--cameras-root", type=Path, default=None)
    audit_parser.add_argument("--run", action="append")
    audit_parser.add_argument("--workers", type=int, default=8)
    audit_parser.add_argument(
        "--truth-cache-root",
        type=Path,
        default=None,
        help="where the re-integrated truth curves are cached, keyed by the truth file's hash",
    )
    audit_parser.add_argument(
        "--criterion",
        default=DEFAULT_CRITERION,
        choices=list(CRITERIA),
        help="which criterion decides truth-good; both are measured either way",
    )
    audit_parser.set_defaults(handler=command_audit)

    select_parser = subparsers.add_parser(
        "dev-select", help="freeze the 12-point development subset"
    )
    select_parser.add_argument("--truth", type=Path, default=None)
    select_parser.add_argument("--dev-points", type=Path, default=None)
    select_parser.add_argument("--root", type=Path, default=None)
    select_parser.add_argument("--truth-events", type=Path, default=None)
    select_parser.add_argument(
        "--real", action="store_true", help="pick REAL cohort points with owner truth instead"
    )
    select_parser.set_defaults(
        handler=lambda args: (
            command_dev_select_real(args) if args.real else command_dev_select(args)
        )
    )

    dev_parser = subparsers.add_parser("dev-run", help="fit and score ONLY the development points")
    dev_parser.add_argument("--rung", default="clean", choices=list(RUNGS))
    dev_parser.add_argument("--arm", default="default", choices=sorted(ARMS))
    dev_parser.add_argument("--workers", type=int, default=8)
    dev_parser.add_argument("--points", action="append")
    dev_parser.add_argument("--dev-points", type=Path, default=None)
    dev_parser.add_argument("--truth", type=Path, default=None)
    dev_parser.add_argument("--cameras-root", type=Path, default=None)
    dev_parser.add_argument("--root", type=Path, default=None)
    dev_parser.add_argument("--run-root", type=Path, default=None)
    dev_parser.add_argument("--label", default=None)
    dev_parser.add_argument(
        "--real",
        action="store_true",
        help="fit REAL cohort points and score them against the owner's clicks",
    )
    dev_parser.add_argument("--truth-events", type=Path, default=None)
    dev_parser.add_argument(
        "--reuse-report",
        action="store_true",
        help="score a report the run root already holds instead of re-fitting",
    )
    dev_parser.add_argument(
        "--criterion",
        default=DEFAULT_CRITERION,
        choices=list(CRITERIA),
        help="which criterion decides truth-good; both are measured either way",
    )
    dev_parser.add_argument("--truth-cache-root", type=Path, default=None)
    dev_parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="with --reuse-report, the run directory holding report.json (default: --run-root)",
    )
    dev_parser.set_defaults(handler=command_dev_run_dispatch)
    return parser


def _data_root() -> Path:
    return Path(os.environ.get("TENNIS_DATA_ROOT", "data"))


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    args.handler(args)


if __name__ == "__main__":
    main()
