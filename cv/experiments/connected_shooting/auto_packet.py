"""Build provenance-honest connected-fitter packets from labels or automatic streams.

This is an evaluation adapter, not an automatic inference stage.  The question it
answers is: *what does the promoted S6 reference reconstruct when one upstream
stream at a time is replaced by the pipeline's own automatic output?*

Four streams are selected independently -- ball observations, physical events plus
the ending, sided player localization, and the per-frame camera.  Each keeps its
own origin and file hashes in a stream manifest, so a mixed packet never claims a
single ancestry.  Automatic detector coordinates are declared ball **centres**
and are never emitted as owner/agent labels.

Every pixel column is resolved through its coordinate sidecar with
``cv.pipeline.resolution``; a file name is never coordinate evidence here.

The unchanged research fitter still names its observation field
``owner_ball_labels``.  This adapter fills that field only as a documented legacy
interface alias of ``ball_observations``; every row retains its actual
``annotation_origin`` and ``observation_semantics``.  For an automatic-centre
packet ``run-fitter`` also selects the nominal-centre projection operator, so a
detector centre is never scored against a swept leading-edge prediction.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from cv.experiments.connected_shooting import agent_whole_point_search as search
from cv.pipeline import automatic_ball_track, paths, provenance
from cv.pipeline import resolution as res
from cv.pipeline.camera_artifacts import (
    FALLBACK_POINT_CAMERA_PREFIXES,
    STATIC_SHOT_REGISTRATION_SUFFIXES,
    TRACKED_REGISTRATION_SUFFIXES,
    split_camera_source,
)

# The composed-track reader and its declared-guide tracing are shared with the
# automatic stage in ``cv.pipeline.automatic_ball_track``; these names are the
# adapter's own interface and are re-exported unchanged.  ``automatic_ball_rows``
# stays a module attribute so existing callers and test doubles keep patching it
# here.
native_frame = automatic_ball_track.native_frame
automatic_ball_rows = automatic_ball_track.automatic_ball_rows
_record = automatic_ball_track.file_binding
_guide_support = automatic_ball_track.guide_support


StreamOrigin = Literal["label", "automatic"]
STREAMS = ("ball", "events", "players", "camera")
BALL_SEMANTICS = {
    "label": "visible_blur_leading_edge_in_travel_direction",
    "automatic": automatic_ball_track.BALL_SEMANTICS_AUTOMATIC,
}
PHYSICAL_EVENT_TYPES = ("contact", "bounce", "net_hit")


def uniform_native_scale(path: Path, columns: tuple[str, ...]) -> float:
    """Return the single sidecar-declared multiplier taking ``columns`` to native.

    The fitter's player interface accepts one scalar, so a non-uniform declared
    space must fail closed here rather than be silently squared into one axis.
    """
    scale_x, scale_y = res.coordinate_scale(path, columns=columns)
    if not math.isclose(scale_x, scale_y, rel_tol=1e-9):
        raise ValueError(f"{path} declares a non-uniform native scale {scale_x}/{scale_y}")
    return float(scale_x)


def label_ball_rows(
    labels: dict[str, Any], *, semantics: str = BALL_SEMANTICS["label"]
) -> list[dict[str, Any]]:
    records = [
        row for row in labels["ball"]["records"] if row.get("clip") == labels["attempt"]["clip"]
    ]
    if len(records) != 1:
        raise ValueError("one label ball record required for the selected clip")
    return [
        {
            **row,
            "annotation_origin": labels.get("annotation_origin", "unknown_human"),
            "observation_semantics": semantics,
        }
        for row in records[0]["frames"]
    ]


def label_events(labels: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in labels["events"]["records"] if row.get("status") == "labeled"]
    endings = [row for row in rows if row["event_type"] == "ending"]
    if len(endings) != 1:
        raise ValueError("label event stream requires exactly one ending")
    origin = labels.get("annotation_origin", "unknown_human")
    physical = [
        {
            "event_type": row["event_type"],
            "frame": float(row["frame"]),
            "frame_interval": [float(value) for value in row["frame_interval"]],
            "note": row.get("note", ""),
            "annotation_origin": origin,
            "source_event_id": row.get("id"),
        }
        for row in rows
        if row["event_type"] in PHYSICAL_EVENT_TYPES
    ]
    physical.sort(key=lambda row: (row["frame"], row["event_type"]))
    return physical, {
        "frame": float(endings[0]["frame"]),
        "kind": labels["attempt"].get("ending_kind"),
        "annotation_origin": origin,
        "source_event_id": endings[0].get("id"),
    }


def automatic_event_epoch(row: dict[str, Any]) -> float:
    """The decoder's own sub-frame epoch, falling back to its rounded frame.

    ``frame`` is the emission's rounded comparison field; the model's regression
    head also reports ``location.frame_subpixel``.  The fitter partitions native
    pictures into flights at the event epoch, so a rounded integer makes the
    picture *at* a contact ambiguous between the incoming and outgoing flight.
    Reading the sub-frame value is using more of the automatic evidence, not a
    correction of it: nothing is shifted toward a label and the fallback is the
    emitted integer.
    """
    subpixel = (row.get("location") or {}).get("frame_subpixel")
    if subpixel is None or not math.isfinite(float(subpixel)):
        return float(row["frame"])
    return float(subpixel)


def automatic_event_interval(
    row: dict[str, Any], frame: float, *, prediction_window: tuple[int, int] | None = None
) -> tuple[list[float], str]:
    """Keep producer support, or declare a one-frame prediction search interval.

    This common engineering search policy is not an observed impact bracket or
    a calibrated confidence interval. Only this generated search domain may be
    intersected with the native window; producer uncertainty remains intact.
    It never consults a reference event.
    """
    supplied = row.get("frame_interval")
    if supplied is not None:
        interval = list(map(float, supplied))
        origin = "producer_supplied_interval"
    else:
        interval = [frame - 1.0, frame + 1.0]
        origin = "prediction_search_radius_one_native_frame_v1"
        if prediction_window is not None:
            low, high = prediction_window
            if not np.isfinite([low, high, frame]).all() or not low <= frame <= high or low >= high:
                raise ValueError("predicted event epoch must lie inside its native window")
            bounded = [max(low, interval[0]), min(high, interval[1])]
            if bounded != interval:
                interval = bounded
                origin = "prediction_search_radius_one_native_frame_window_bounded_v1"
    if (
        len(interval) != 2
        or not np.isfinite(interval).all()
        or not interval[0] <= frame <= interval[1]
    ):
        raise ValueError("automatic predicted epoch must lie inside its finite source interval")
    return interval, origin


#: The quantitative record a considered abstention carries: how far the producer's
#: acceptance marginal fell short of the threshold it was judged against.
ABSTENTION_DECISION_FIELDS = ("acceptance_marginal", "decision_threshold", "abstention_gap")


def automatic_abstention(row: dict[str, Any]) -> dict[str, Any] | None:
    """The producer's own CONSIDERED abstention, or ``None``.

    A considered abstention is not a shrug: the producer emitted a typed
    occurrence at a sub-frame epoch, placed it on the court, and then declined
    to claim it because its acceptance marginal fell short of the operating
    threshold it was judged against.  This reads that record back verbatim; it
    never re-decides the verdict and never reconstructs one.

    ``None`` covers both an accepted emission and an abstention with no decision
    record.  That second case is the distinction ``classify_abstention`` draws
    elsewhere in this repository: an abstention that produced no judgement at
    all is mechanical, and reading one as a careful refusal would turn a
    producer failure into evidence.  Such a row carries nothing, so it is
    exactly as retainable as an emission that was never made -- which is to say
    not at all, and not a reason to refuse the attempt around it either.

    Two producers write this stream and both are accepted, because both state
    their own grammar. ``cv.pipeline.point_grammar.annotate_abstentions`` writes
    ``verdict: "abstained"`` with its reasons. The grammar decoder declares
    ``point_grammar.present: false`` with the schema and the replacement fields
    it used instead, and records the refusal in the decision fields alone. A row
    that claims to abstain under a grammar which ran and said something else is
    contradictory, and raises.
    """
    if row.get("abstain") is not True:
        return None
    grammar = row.get("point_grammar") or {}
    verdict = grammar.get("verdict")
    if verdict is None:
        if grammar.get("present") is not False:
            raise ValueError(
                "an abstained automatic emission must carry its abstained verdict, "
                "or its producer's declaration that the point grammar did not run"
            )
    elif verdict != "abstained":
        raise ValueError(f"emission declares abstain beside the {verdict!r} point-grammar verdict")
    decision = {
        name: float(value)
        for name in ABSTENTION_DECISION_FIELDS
        if isinstance(value := row.get(name), (int, float)) and math.isfinite(float(value))
    }
    # An explicit `verdict: "abstained"` IS the judgement: `annotate_abstentions`
    # writes it only for a refusal the grammar considered, so the numbers are a
    # richer record of that verdict, not what makes it one.  The decoder states no
    # verdict at all, so for that producer the decision record is the ONLY evidence
    # the refusal was considered rather than mechanical, and all of it is required.
    if verdict is None and len(decision) != len(ABSTENTION_DECISION_FIELDS):
        return None
    return {
        **{name: decision.get(name) for name in ABSTENTION_DECISION_FIELDS},
        "class_probabilities": row.get("class_probabilities"),
        "producer_verdict": verdict,
        "producer_grammar_schema": grammar.get("schema"),
        **({"reasons": list(grammar["reasons"])} if "reasons" in grammar else {}),
    }


def automatic_physical_event(
    row: dict[str, Any],
    *,
    prediction_window: tuple[int, int] | None = None,
    declare_abstention: bool = False,
) -> dict[str, Any]:
    """Normalize one original physical emission without deciding its membership.

    ``declare_abstention`` asks for the third status this vocabulary needs.  An
    accepted emission is a claim the producer makes (``predicted``); an abstained
    one is a refusal to claim (``ambiguous``), which is the same word the labeled
    inventory uses for an occurrence its source would not confirm.  Timing is
    unaffected: the regression head reported the same sub-frame epoch either way,
    and it is the *occurrence* that is refused, not its epoch.

    It is a request, not a sniff at the row, because the optional-witness families
    already convert abstained and gate-held emissions under their own ``optional``
    occurrence contract.  Only a caller that means to retain the abstention *as*
    an uncertain occurrence asks for it, and it raises on a row that is not one.
    """
    frame = automatic_event_epoch(row)
    abstention = None
    if declare_abstention:
        abstention = automatic_abstention(row)
        if abstention is None:
            raise ValueError("a declared abstention requires an abstained source emission")
    interval, interval_origin = automatic_event_interval(
        row, frame, prediction_window=prediction_window
    )
    return {
        "event_type": row["event_type"],
        "frame": frame,
        "frame_interval": interval,
        "interval_origin": interval_origin,
        **(
            {
                "original_prediction_search_interval": [frame - 1.0, frame + 1.0],
                "prediction_search_native_window": list(prediction_window),
            }
            if interval_origin == "prediction_search_radius_one_native_frame_window_bounded_v1"
            else {}
        ),
        "exact_epoch_observed": False,
        "timing_status": "predicted",
        "status": "ambiguous" if abstention else "predicted",
        "occurrence_status": "ambiguous" if abstention else "predicted",
        "note": (
            "original automatic abstention retained at its own sub-frame epoch"
            if abstention
            else "accepted automatic emission at its own sub-frame epoch"
        ),
        **({"automatic_abstention": abstention} if abstention else {}),
        "annotation_origin": "automatic",
        "emitted_rounded_frame": float(row["frame"]),
        "epoch_source": (
            "location.frame_subpixel"
            if (row.get("location") or {}).get("frame_subpixel") is not None
            else "emitted_rounded_frame"
        ),
        "automatic_confidence": row.get("confidence"),
        "automatic_probability": row.get("probability"),
        "automatic_location": row.get("location"),
        **(
            {
                "automatic_event_identity_support": row["event_identity_support"],
                "automatic_impulse_support": row.get("impulse_support"),
            }
            if "event_identity_support" in row
            else {}
        ),
        # Present only for a row the optional model-accepted admission actually
        # admitted, so an ordinary packet keeps its shape.  The block records the
        # retained tracking hold this observation was admitted despite.
        **(
            {"automatic_model_accepted_admission": row["model_accepted_admission"]}
            if isinstance(row.get("model_accepted_admission"), dict)
            and row["model_accepted_admission"].get("admitted") is True
            else {}
        ),
    }


def automatic_events(
    path: Path,
    match_id: str,
    clip: str,
    window: tuple[int, int],
    *,
    observation_scope: bool = False,
    require_originating_contact: bool = False,
    leading_physical_prefix: bool = False,
    include_abstained: bool = False,
    allow_terminal_net_scope: bool = False,
    allow_observed_horizon_scope: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select accepted automatic emissions inside the disclosed evaluation window.

    The window is the label cohort's selector for *which* attempt is scored.  It
    never moves a native epoch, adds an event, or repairs a missing one.
    """
    physical, ending = automatic_event_inventory(
        json.loads(path.read_text()),
        match_id,
        clip,
        window,
        observation_scope=observation_scope,
        require_originating_contact=require_originating_contact,
        leading_physical_prefix=leading_physical_prefix,
        include_abstained=include_abstained,
    )
    if ending.pop("unqualified_observation_scope", False):
        from cv.experiments.connected_shooting.observation_scope import (
            UnsupportedOriginTopology,
            qualify,
        )

        try:
            ending["observation_scope"] = qualify(
                physical,
                window,
                allow_terminal_net=allow_terminal_net_scope,
                allow_observed_horizon=allow_observed_horizon_scope,
            )
        except UnsupportedOriginTopology as error:
            if not leading_physical_prefix:
                raise
            # The caller owns the declared leading prefix: it holds the explicit
            # policy and the segmentation binding this adapter does not carry.
            ending["leading_prefix_refusal"] = str(error)
    return physical, ending


def automatic_event_inventory(
    document: list[dict],
    match_id: str,
    clip: str,
    window: tuple[int, int],
    *,
    observation_scope: bool = False,
    require_originating_contact: bool = False,
    leading_physical_prefix: bool = False,
    include_abstained: bool = False,
    isolate_invalid_prefix: bool = False,
    retain_orphan_contacts: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert original accepted events before optional topology qualification.

    Preserve original filtering, native epochs and origin checks. This inventory
    is not admission: consumers must qualify its observation scope separately.

    ``leading_physical_prefix`` is the default-off admission of a stream whose
    first accepted physical row precedes its first accepted contact. It retains
    those rows verbatim instead of refusing; it never drops, retimes or invents
    one, and it requires an unresolved observation scope because a resolved
    automatic ending has no contract that can own the leading evidence.

    ``include_abstained`` is the default-off admission of the producer's own
    recorded abstentions as uncertain occurrences. They are read back through
    ``point_grammar.select_consumer_emissions``, the existing lossless bypass
    that owns what an abstention is, and they are retained *only* as physical
    rows inside the span the accepted inventory already spans. They can never
    become the ending, the originating contact or the leading prefix: an
    abstention is recorded evidence, not a claim, so no topology decision may
    rest on one. Everything the accepted inventory does is unchanged.

    ``retain_orphan_contacts`` is the separate default-off route for a window
    whose only contacts are abstained. Those rows are kept, marked
    ``orphan_before_preparation``, so preparation can hand them to the cascade
    instead of refusing the attempt for having no contact. It does not promote
    them to claims.
    """
    if not isinstance(document, list):
        raise ValueError("automatic event artifact must be a JSON list")
    joined = f"{match_id}__{clip}"
    matched = [
        row
        for row in document
        if row.get("match_id") == match_id
        and row.get("clip") in {joined, clip}
        and window[0] <= float(row["frame"]) <= window[1]
    ]
    selected = [row for row in matched if not row.get("abstain", False)]
    abstained: list[dict[str, Any]] = []
    if include_abstained:
        from cv.pipeline.point_grammar import select_consumer_emissions

        abstained = [
            row
            for row in select_consumer_emissions(
                matched, include_abstained=True, include_dead_time=True
            )
            if row.get("abstain") is True
            and row["event_type"] in PHYSICAL_EVENT_TYPES
            # A mechanical abstention carries no decision record, so it carries no
            # evidence either; it is skipped exactly as an unmade emission is.
            and automatic_abstention(row) is not None
        ]
    endings = [row for row in selected if row["event_type"] == "point_end"]
    scoped = observation_scope and not endings
    if len(endings) != 1 and not scoped:
        raise ValueError(
            f"automatic event stream has {len(endings)} accepted point_end rows "
            "in the evaluation window"
        )
    ending = None if scoped else endings[0]
    end_epoch = float(window[1]) if scoped else automatic_event_epoch(ending)
    contacts = [
        row
        for row in selected
        if row["event_type"] == "contact" and automatic_event_epoch(row) < end_epoch
    ]
    orphan_rows: list[dict[str, Any]] = []
    if not contacts and retain_orphan_contacts:
        for row in matched:
            if row.get("event_type") != "contact" or row.get("abstain") is not True:
                continue
            epoch = automatic_event_epoch(row)
            if not epoch < end_epoch:
                continue
            if automatic_abstention(row) is None:
                continue
            orphan_rows.append(row)
    if not contacts and not orphan_rows:
        raise ValueError(
            "automatic event stream has no contact inside observation scope"
            if scoped
            else "automatic event stream has no contact before point_end"
        )
    opening = contacts or orphan_rows
    first = min(automatic_event_epoch(row) for row in opening)
    leading = [
        row
        for row in selected
        if row["event_type"] in PHYSICAL_EVENT_TYPES
        and window[0] <= automatic_event_epoch(row) < first
    ]
    isolated_prefix: list[dict[str, Any]] | None = None
    if leading and isolate_invalid_prefix and not scoped:
        # The prefix is evidence, not a member of the rally. Drop it from the
        # competitive inventory and keep every later original row as it is.
        isolated_prefix = [
            {"event_type": row["event_type"], "frame": automatic_event_epoch(row)}
            for row in leading
        ]
        leading = []
    if leading and leading_physical_prefix:
        if not scoped:
            raise ValueError(
                "declared leading physical evidence requires an unresolved observation scope; "
                "a resolved automatic ending cannot own a preceding physical row"
            )
    elif require_originating_contact and leading:
        raise ValueError(
            "automatic event stream needs an originating contact; "
            "prior accepted physical events cannot be dropped"
        )
    retain_from = window[0] if scoped or (leading and leading_physical_prefix) else first
    physical = []
    for row in selected:
        if row["event_type"] not in PHYSICAL_EVENT_TYPES:
            continue
        frame = automatic_event_epoch(row)
        if not retain_from <= frame <= end_epoch:
            continue
        physical.append(automatic_physical_event(row, prediction_window=window))
    for row in orphan_rows:
        event = automatic_physical_event(row, prediction_window=window, declare_abstention=True)
        event["orphan_before_preparation"] = True
        physical.append(event)
    if abstained and physical:
        # An abstention may make an occurrence inside the retained span uncertain;
        # it may never move the row the topology opens on.  So the earliest bound
        # is the first *accepted* retained row, not ``retain_from``, and a row that
        # would collide with an accepted (epoch, type) is dropped rather than
        # duplicated -- the accepted claim is the stronger record of the two.
        opening = min(row["frame"] for row in physical)
        taken = {(row["frame"], row["event_type"]) for row in physical}
        for row in abstained:
            frame = automatic_event_epoch(row)
            if not opening <= frame <= end_epoch or (frame, row["event_type"]) in taken:
                continue
            taken.add((frame, row["event_type"]))
            physical.append(
                automatic_physical_event(row, prediction_window=window, declare_abstention=True)
            )
    physical.sort(key=lambda row: (row["frame"], row["event_type"], row["status"]))
    if scoped:
        unresolved = {
            "frame": end_epoch,
            "kind": "unresolved",
            "unqualified_observation_scope": True,
        }
        if isolated_prefix:
            unresolved["isolated_invalid_prefix"] = isolated_prefix
        return physical, unresolved
    end_record = ending.get("point_end", {})
    return physical, {
        "frame": end_epoch,
        "emitted_rounded_frame": float(ending["frame"]),
        "kind": end_record.get("termination_kind") or end_record.get("terminal_event_type"),
        "terminal_event_type": end_record.get("terminal_event_type"),
        "annotation_origin": "automatic",
        "automatic_confidence": ending.get("confidence"),
        "automatic_probability": ending.get("probability"),
        "source": end_record.get("source"),
        "frame_interval": automatic_event_interval(ending, end_epoch)[0],
        "interval_origin": automatic_event_interval(ending, end_epoch)[1],
        "exact_epoch_observed": False,
        **({"isolated_invalid_prefix": isolated_prefix} if isolated_prefix else {}),
    }


def _automatic_camera_lens_arrays(
    arrays: np.lib.npyio.NpzFile | dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate per-native-frame lens declarations before selecting a clip.

    Bundle cameras emit k1/dist_center; accept the shared geometry helper's
    equivalent row form too, but never discard an unsupported lens model.
    """
    shapes = {
        "k1": (len(arrays["frames"]),),
        "dist_center": (len(arrays["frames"]), 2),
        "camera_distortion": (len(arrays["frames"]), 3),
        "k2": (len(arrays["frames"]),),
    }
    if {"distortion", "radial_distortion"}.intersection(arrays):
        raise ValueError("ambiguous radial metadata; explicit native k1/dist_center required")
    if ("k1" in arrays) != ("dist_center" in arrays):
        raise ValueError("automatic radial camera requires both k1 and dist_center")
    result = {}
    for name, shape in shapes.items():
        if name in arrays:
            values = np.asarray(arrays[name], float)
            if values.shape != shape:
                raise ValueError(f"automatic camera {name} requires native-frame shape {shape}")
            result[name] = values
    return result


#: Court-registration suffixes a direct point camera already treats as tracked.
#: ``anchor_static_fallback`` and ``registration_gap_abstention`` stay held:
#: those frames have no tracked homography. The net residual is not a veto.
#: A high broadcast can store the far baseline as the net witness, which makes
#: the 8 px net gate measure the wrong line while the ground-homography
#: projection itself stays finite.
ADMITTED_INTRINSIC_REGISTRATION_SUFFIXES = TRACKED_REGISTRATION_SUFFIXES
MAXIMUM_ADMITTED_GROUND_RESIDUAL_PX = 4.0
INTRINSIC_FALLBACK_REGISTRATION_OFF = "off"
INTRINSIC_FALLBACK_REGISTRATION_ADMIT = "admit"
INTRINSIC_FALLBACK_REGISTRATION_MODES = (
    INTRINSIC_FALLBACK_REGISTRATION_OFF,
    INTRINSIC_FALLBACK_REGISTRATION_ADMIT,
)
SHOT_HOMOGRAPHY_PROPAGATION_OFF = "off"
SHOT_HOMOGRAPHY_PROPAGATION_ON = "on"
SHOT_HOMOGRAPHY_PROPAGATION_MODES = (
    SHOT_HOMOGRAPHY_PROPAGATION_OFF,
    SHOT_HOMOGRAPHY_PROPAGATION_ON,
)


def _admit_intrinsic_fallback_registration(
    *,
    mode: str,
    already_supported: bool,
    source: str,
    projection: np.ndarray,
    ground_residual_px: float | None,
) -> bool:
    """Admit a ground-preserving intrinsic fallback on an already tracked frame.

    Production default is ``admit``. A direct camera is unchanged. A static or
    abstained frame stays held. The stored net residual is ignored: on the
    broadcasts that need this, that residual is the far baseline, not the tape.
    Rollback is the explicit mode ``off``.
    """
    if mode not in INTRINSIC_FALLBACK_REGISTRATION_MODES:
        raise ValueError("explicit intrinsic-fallback registration mode required")
    if already_supported or mode != INTRINSIC_FALLBACK_REGISTRATION_ADMIT:
        return False
    prefix, suffix = split_camera_source(source)
    if prefix != "intrinsic_fallback":
        return False
    if suffix not in ADMITTED_INTRINSIC_REGISTRATION_SUFFIXES:
        return False
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        return False
    return (
        ground_residual_px is not None and ground_residual_px <= MAXIMUM_ADMITTED_GROUND_RESIDUAL_PX
    )


def _admit_shot_homography_propagation(
    *,
    mode: str,
    already_supported: bool,
    source: str,
    projection: np.ndarray,
    ground_residual_px: float | None,
    shot_id: int | None,
    registered_shots: set[int],
) -> bool:
    """Admit a fallback point camera on the same shot's stored homography.

    The function default is off, so a caller that does not name the mode
    leaves the document unchanged. Production turns it on in
    ``PIPELINE_COMPONENT_POLICY``. ``on`` keeps a finite ground-preserving P
    when the point camera is a net/intrinsic fallback and the court track
    registered that frame, or stored the registered camera as
    ``anchor_static_fallback`` on another frame of the same shot. A different
    shot, including a close-up, stays held. A registration-gap hold stays
    held. Without shot ids a static frame is not admitted. Rollback is
    deleting that policy line, or setting ``off``.
    """
    if mode not in SHOT_HOMOGRAPHY_PROPAGATION_MODES:
        raise ValueError("explicit shot-homography propagation mode required")
    if already_supported or mode != SHOT_HOMOGRAPHY_PROPAGATION_ON:
        return False
    prefix, suffix = split_camera_source(source)
    if prefix not in FALLBACK_POINT_CAMERA_PREFIXES:
        return False
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        return False
    if ground_residual_px is None or ground_residual_px > MAXIMUM_ADMITTED_GROUND_RESIDUAL_PX:
        return False
    if suffix in TRACKED_REGISTRATION_SUFFIXES:
        return True
    return (
        suffix in STATIC_SHOT_REGISTRATION_SUFFIXES
        and shot_id is not None
        and int(shot_id) in registered_shots
    )


def automatic_camera_document(
    path: Path,
    labels: dict[str, Any],
    window: tuple[int, int],
    *,
    intrinsic_fallback_registration: str = INTRINSIC_FALLBACK_REGISTRATION_ADMIT,
    shot_homography_propagation: str = SHOT_HOMOGRAPHY_PROPAGATION_OFF,
    shot_ids: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Turn the automatic per-frame camera NPZ into the fitter's camera document.

    ``P`` maps the shared court world frame to native 1920x1080 pixels; the
    artifact's own ``frame_scope`` is retained so a per-point static fallback is
    never reported as per-frame registration.

    Production default ``admit`` keeps the projection on frames whose court
    registration already tracked, when the point camera is only an intrinsic
    fallback. Explicit ``off`` is the rollback and leaves the document without
    the admission fields, matching the previous default.

    ``shot_ids`` maps a frame to the camera shot. Static fallback frames are
    admitted only in a shot that itself has a tracked registration. Omitting
    the map does not admit those frames.
    """
    if intrinsic_fallback_registration not in INTRINSIC_FALLBACK_REGISTRATION_MODES:
        raise ValueError("explicit intrinsic-fallback registration mode required")
    if shot_homography_propagation not in SHOT_HOMOGRAPHY_PROPAGATION_MODES:
        raise ValueError("explicit shot-homography propagation mode required")
    required = {"clips", "frames", "P", "reliable", "source", "confidence"}
    optional = {
        "frame_scope",
        "ground_residual_px",
        "net_residual_px",
        "fallback_ancestry",
        "reference_frame",
        "k1",
        "dist_center",
        "camera_distortion",
        "k2",
    }
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        if not required.issubset(available):
            raise ValueError(f"automatic camera artifact lacks {sorted(required - available)}")
        if {"distortion", "radial_distortion"}.intersection(available):
            raise ValueError("ambiguous radial metadata; explicit native k1/dist_center required")
        # NPZ indexing decompresses an entire member on each access. Keep one
        # array per used member for this call; unknown metadata stays unread.
        arrays = {name: archive[name] for name in required | (optional & available)}
    from cv.experiments.connected_shooting import camera_geometry

    lens_arrays = _automatic_camera_lens_arrays(arrays)

    def optional_residual(name: str, index: int) -> float | None:
        if name not in arrays:
            return None
        value = float(arrays[name][index])
        return value if np.isfinite(value) else None

    clip = labels["attempt"]["clip"]
    registered_shots: set[int] = set()
    if shot_ids is not None:
        for candidate, frame, source_name in zip(
            arrays["clips"], arrays["frames"], arrays["source"], strict=True
        ):
            if str(candidate) != clip:
                continue
            _prefix, suffix = split_camera_source(str(source_name))
            shot = shot_ids.get(int(frame))
            if suffix in TRACKED_REGISTRATION_SUFFIXES and shot is not None:
                registered_shots.add(int(shot))
    rows = []
    scopes: set[str] = set()
    admitted_frames = 0
    shot_admitted_frames = 0
    for index, (candidate, frame) in enumerate(zip(arrays["clips"], arrays["frames"], strict=True)):
        frame = int(frame)
        if str(candidate) != clip or not window[0] <= frame <= window[1]:
            continue
        lens = {name: values[index] for name, values in lens_arrays.items()}
        if "k2" in lens and (not np.isfinite(lens["k2"]) or lens["k2"] != 0):
            raise ValueError("automatic S6 camera does not support nonzero/nonfinite k2")
        radial = camera_geometry.radial_row(lens)
        projection = np.asarray(arrays["P"][index], float)
        reliable = bool(arrays["reliable"][index])
        supported = reliable and projection.shape == (3, 4) and np.isfinite(projection).all()
        source_name = str(arrays["source"][index])
        ground_residual_px = optional_residual("ground_residual_px", index)
        if _admit_intrinsic_fallback_registration(
            mode=intrinsic_fallback_registration,
            already_supported=supported,
            source=source_name,
            projection=projection,
            ground_residual_px=ground_residual_px,
        ):
            supported = True
            admitted_frames += 1
        if _admit_shot_homography_propagation(
            mode=shot_homography_propagation,
            already_supported=supported,
            source=source_name,
            projection=projection,
            ground_residual_px=ground_residual_px,
            shot_id=None if shot_ids is None else shot_ids.get(frame),
            registered_shots=registered_shots,
        ):
            supported = True
            shot_admitted_frames += 1
        scope = str(arrays["frame_scope"][index]) if "frame_scope" in arrays else "unknown"
        scopes.add(scope)
        rows.append(
            {
                "frame": frame,
                "status": "supported" if supported else "held",
                "supported": bool(supported),
                "reason": None if supported else "automatic_camera_unreliable_or_nonfinite",
                "P": projection.tolist() if supported else None,
                "source": str(arrays["source"][index]),
                "confidence": float(arrays["confidence"][index]),
                "frame_scope": scope,
                "annotation_origin": "automatic",
                **(
                    {"k1": float(radial[0]), "dist_center": radial[1:].tolist()}
                    if radial is not None
                    else {}
                ),
                "ground_residual_px": optional_residual("ground_residual_px", index),
                "net_residual_px": optional_residual("net_residual_px", index),
                "fallback_ancestry": (
                    str(arrays["fallback_ancestry"][index])
                    if "fallback_ancestry" in arrays
                    else None
                ),
                "reference_frame": (
                    str(arrays["reference_frame"][index]) if "reference_frame" in arrays else None
                ),
            }
        )
    if not rows:
        raise ValueError("automatic camera artifact has no rows in the evaluation window")
    return {
        "schema": "connected_fitter_camera_packet_v1",
        "scope": (
            "Automatic court-registration camera for one attempt window. The height column "
            "is not independently certified as metric (CAM-001); no airborne accuracy is claimed."
        ),
        "human_derived": False,
        "annotation_origin": "automatic",
        "match_id": labels["match_id"],
        "clip": clip,
        "cameras": rows,
        "frame_scope": sorted(scopes),
        "supported": sum(row["status"] == "supported" for row in rows),
        "total": len(rows),
        "automatic_inference_eligible": True,
        **(
            {
                "intrinsic_fallback_registration": intrinsic_fallback_registration,
                "admitted_intrinsic_fallback_frames": admitted_frames,
            }
            if intrinsic_fallback_registration != INTRINSIC_FALLBACK_REGISTRATION_OFF
            else {}
        ),
        **(
            {
                "shot_homography_propagation": shot_homography_propagation,
                "admitted_shot_homography_frames": shot_admitted_frames,
            }
            if shot_homography_propagation != SHOT_HOMOGRAPHY_PROPAGATION_OFF
            else {}
        ),
    }


def load_label_cameras(path: Path, labels: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if (
        document.get("match_id") != labels["match_id"]
        or document.get("clip") != labels["attempt"]["clip"]
    ):
        raise ValueError("label camera artifact does not match attempt")
    return document


def terminal_bounce_count(events: list[dict[str, Any]], ending_frame: float) -> int:
    contacts = sorted(float(row["frame"]) for row in events if row["event_type"] == "contact")
    if not contacts:
        raise ValueError("at least one contact required")
    return sum(
        row["event_type"] == "bounce" and contacts[-1] < float(row["frame"]) <= ending_frame
        for row in events
    )


def mixed_observation_document(
    labels: dict,
    source_rows: list[dict],
    events: list[dict],
    ending: dict,
    *,
    ball_origin: StreamOrigin,
    event_origin: StreamOrigin,
    ball_operator: dict | None = None,
    fixed_stream_origins: dict | None = None,
) -> dict:
    """Construct the actual consumer document, not an unmodified truth side input.

    Only fixed clip/image/court metadata crosses unconditionally. Inactive ball
    and event fields (including free-form notes and duplicate hints) do not.
    """
    from cv.experiments.connected_shooting import observation_operator

    operator = ball_operator or observation_operator.declaration(
        "nominal_center" if ball_origin == "automatic" else "leading_front"
    )
    if operator != observation_operator.declaration(operator.get("kind")):
        raise ValueError("explicit supported observation operator required")
    if ball_origin == "automatic" and operator["kind"] != "nominal_center":
        raise ValueError("automatic observations require native center semantics")
    semantics = (
        BALL_SEMANTICS["automatic"]
        if operator["kind"] == "nominal_center"
        else BALL_SEMANTICS["label"]
    )
    fixed = fixed_stream_origins or {"players": "label", "camera": "label"}
    if set(fixed) != {"players", "camera"} or any(
        value not in ("label", "automatic") for value in fixed.values()
    ):
        raise ValueError("explicit fixed camera/player origins required")
    clip = labels["attempt"]["clip"]
    window = list(labels["attempt"]["native_window"])
    image_fields = {
        "clip",
        "frame",
        "path_base",
        "image_url",
        "source",
        "native_pts_seconds",
        "clip_seconds",
        "source_reel_frame_1based",
        "source_frame_index",
        "source_pts",
        "source_time_base",
    }
    images = [
        {k: deepcopy(v) for k, v in x.items() if k in image_fields}
        for x in labels["source_pack"]["images"]
        if x["clip"] == clip
    ]
    image_map = {int(x["frame"]): x for x in images}
    rows = []
    ball_keys = {
        "frame",
        "status",
        "x1080",
        "y1080",
        "uncertainty_radius_px1080",
        "streak",
        "annotation_origin",
        "observation_semantics",
        "coordinate_columns_read",
        "declared_source_space",
        "automatic_track_id",
        "automatic_sources",
        "automatic_confidence",
        "automatic_covariance_native_px2",
        "automatic_covariance_semantics",
        "uncertainty_policy",
        "support_class",
        "guide_support",
        "streak_centre",
    }
    for original in source_rows:
        frame = int(original["frame"])
        if not window[0] <= frame <= window[1]:
            continue
        if frame not in image_map:
            raise ValueError(f"stream frame {frame} has no original native image/timestamp")
        image = image_map[frame]
        if image.get("native_pts_seconds") is None:
            raise ValueError(f"stream frame {frame} has no native PTS")
        row = {k: deepcopy(v) for k, v in original.items() if k in ball_keys}
        if "streak" in row:
            streak = row["streak"]
            row["streak"] = {"status": streak.get("status")}
            for tip in ("leading", "trailing"):
                if isinstance(streak.get(tip), dict):
                    row["streak"][tip] = {
                        k: deepcopy(v)
                        for k, v in streak[tip].items()
                        if k in {"x1080", "y1080", "status", "uncertainty_radius_px1080"}
                    }
        row.update(
            native_pts_seconds=image["native_pts_seconds"],
            source_image_sha256=image["source"]["sha256"],
        )
        rows.append(row)
    if len({x["frame"] for x in rows}) != len(rows):
        raise ValueError("duplicate native ball rows")
    by_frame = {x["frame"]: x for x in rows}
    for frame, image in image_map.items():
        if window[0] <= frame <= window[1] and frame not in by_frame:
            rows.append(
                dict(
                    frame=frame,
                    status="not_detected",
                    annotation_origin=ball_origin,
                    observation_semantics=semantics,
                    native_pts_seconds=image.get("native_pts_seconds"),
                    source_image_sha256=image["source"]["sha256"],
                )
            )
    rows.sort(key=lambda x: x["frame"])
    scope_contract = ending.get("observation_scope")
    event_rows = (
        deepcopy(labels["events"]["records"])
        if event_origin == "label"
        else [dict(deepcopy(e), clip=clip) for e in events]
        + (
            []
            if scope_contract is not None
            else [
                dict(
                    event_type="ending",
                    frame=ending["frame"],
                    frame_interval=ending["frame_interval"],
                    interval_origin=ending["interval_origin"],
                    status="predicted",
                    clip=clip,
                    annotation_origin="automatic",
                    exact_epoch_observed=False,
                    ending_kind=ending["kind"],
                    note="automatic terminal path prediction",
                )
            ]
        )
    )
    attempt = {
        k: deepcopy(labels["attempt"][k])
        for k in ("clip", "native_window", "surface", "gender", "session_lighting")
        if k in labels["attempt"]
    }
    attempt.update(
        first_contact_frame=min(e["frame"] for e in events if e["event_type"] == "contact"),
        ending_frame=ending["frame"],
        ending_kind=ending["kind"],
    )
    if scope_contract is not None:
        attempt.pop("ending_frame")
        attempt["observation_scope"] = deepcopy(scope_contract)
    if event_origin == "label":
        for k in ("serve_number", "serve_number_status", "net_cord", "lob_leaves_frame"):
            if k in labels["attempt"]:
                attempt[k] = deepcopy(labels["attempt"][k])
    ball_record = dict(
        case_id="supplied_ball_stream",
        clip=clip,
        window_status="complete_native_inventory",
        notes="",
        frames=rows,
    )
    return dict(
        schema="s6_mixed_observation_document_v1",
        benchmark_id=labels["benchmark_id"],
        match_id=labels["match_id"],
        native_size=deepcopy(labels.get("native_size", [1920, 1080])),
        annotation_origin="mixed_explicit_streams",
        ball_convention=semantics,
        stream_origins=dict(ball=ball_origin, events=event_origin, **fixed),
        attempt=attempt,
        source_pack=dict(fps=labels["source_pack"]["fps"], images=images),
        ball=dict(records=[ball_record]),
        events=dict(records=event_rows),
        # Court labels are fixed camera assistance. No ball/event notes or
        # speed/toss/ending hints are copied alongside this declaration.
        court=deepcopy(labels.get("court", {})),
        court_convention=labels.get("court_convention"),
        automatic_inference_eligible=False,
    )


def compose_s6_documents(
    original: dict,
    labels: dict,
    source_rows: list[dict],
    events: list[dict],
    ending: dict,
    *,
    ball_origin: StreamOrigin,
    event_origin: StreamOrigin,
    ball_operator: dict | None = None,
    fixed_stream_origins: dict | None = None,
) -> tuple[dict, dict]:
    """Shared allowlisted stream construction; callers authenticate source identities.

    This remains supplied-input research, including the automatic/automatic arm.
    No inactive source packet fields or fitted states enter the rebuilt documents.
    """
    from cv.experiments.connected_shooting import observation_operator

    attempt = original["attempts"][0]
    clip = attempt["point_clip"]
    operator = ball_operator or (
        observation_operator.from_packet(original)
        if ball_origin == "label"
        else observation_operator.declaration("nominal_center")
    )
    mixed = mixed_observation_document(
        labels,
        source_rows,
        events,
        ending,
        ball_origin=ball_origin,
        event_origin=event_origin,
        ball_operator=operator,
        fixed_stream_origins=fixed_stream_origins,
    )
    all_rows = mixed["ball"]["records"][0]["frames"]
    first = min(e["frame"] for e in events if e["event_type"] == "contact")
    last = float(ending["frame"])
    origins = mixed["stream_origins"]
    # Rebuild from an allowlist, removing source-specific toss/context witnesses
    # and every duplicate of an inactive stream.
    attempt = dict(
        attempt_id=attempt["attempt_id"],
        clip=attempt["clip"],
        point_clip=clip,
        match_id=attempt["match_id"],
        fps=attempt["fps"],
        events=events,
        first_event_frame=first,
        owner_end_frame=last,
        contact_count=sum(e["event_type"] == "contact" for e in events),
        terminal_bounce_count=terminal_bounce_count(events, last),
        owner_ball_labels=[x for x in all_rows if first <= x["frame"] <= last],
        context_native_frames=[
            x["frame"] for x in all_rows if x["frame"] < first or x["frame"] > last
        ],
        agent_event_rows=mixed["events"]["records"],
        ball_convention=mixed["ball_convention"],
        ball_observation_operator=operator,
        stream_origins=origins,
        annotation_origin="mixed_explicit_streams",
        complete_native_label_inventory=False,
        fractional_estimates=[],
    )
    if ending.get("observation_scope") is not None:
        attempt.update(
            observation_scope=deepcopy(ending["observation_scope"]),
            ending_supplied=False,
            owner_end_frame_semantics="observation_horizon_not_physical_event",
        )
        from cv.experiments.connected_shooting.observation_scope import validate
        from cv.pipeline.s6_contact_prefix_scope import UNRESOLVED_INPUT_SCHEMA

        contract = attempt["observation_scope"]
        if contract.get("schema") == UNRESOLVED_INPUT_SCHEMA:
            # This is pending source qualification, not a modeled prefix. The
            # shared runtime must still qualify coverage/cameras before search.
            if (
                contract.get("ending_semantics") != "unresolved"
                or contract.get("physical_ending") is not None
                or ending.get("kind") != "unresolved"
                or contract.get("native_window") != mixed["attempt"]["native_window"]
                or contract.get("observation_horizon") != last
                or last != contract["native_window"][1]
                or contract.get("original_event_observations") != events
                or contract.get("original_inventory", {}).get("event_origin") != event_origin
            ):
                raise ValueError("unresolved input scope differs from its original streams/window")
        else:
            validate(attempt, mixed)
    attempt["ball_observations"] = attempt["owner_ball_labels"]
    attempt["visible_native_frames"] = sum(
        x["status"] == "visible" for x in attempt["owner_ball_labels"]
    )
    attempt["labeled_native_frames"] = (
        len(attempt["owner_ball_labels"]) if ball_origin == "label" else 0
    )
    packet = dict(
        schema="connected_shooting_stream_packet_v1",
        attempts=[attempt],
        stream_origins=origins,
        ball_observation_operator=operator,
        annotation_origin="mixed_explicit_streams",
        human_derived=True,
        automatic_inference_eligible=False,
        assistance=(
            "fixed labeled attempt window, camera, pose and player metadata"
            if fixed_stream_origins is None
            else "fixed supplied attempt window and declared camera/player inputs"
        ),
    )
    observation_operator.from_packet(packet)
    return mixed, packet


def build_s6_packet(
    row: dict,
    *,
    output: Path,
    ball_origin: StreamOrigin,
    event_origin: StreamOrigin,
    automatic_ball_path: Path,
    automatic_events_path: Path,
    automatic_event_candidates_path: Path | None = None,
    observation_scope: bool = False,
) -> dict:
    """Bind a sanitized mixed document to the current full shared S6 interface.

    Fixed cameras/pose remain unchanged. The source packet preserves labeled
    event semantics in AL; automatic-event arms replace every event duplicate.
    The declared reference flight count is reporting metadata, never inferred
    topology supervision. LL should use its original row unchanged.
    """
    from cv.pipeline import s6_labeled_stage as stage

    if ball_origin not in ("label", "automatic") or event_origin not in ("label", "automatic"):
        raise ValueError("explicit supported stream origins required")
    if output.exists():
        raise FileExistsError(output)
    inputs = stage.validate_row(row)
    labels = json.loads(inputs["labels"].read_text())
    original = json.loads(inputs["packet"].read_text())
    packet = deepcopy(original)
    attempt = packet["attempts"][0]
    clip = attempt["point_clip"]
    window = tuple(labels["attempt"]["native_window"])
    source_rows = (
        label_ball_rows(labels, semantics=labels.get("ball_convention", BALL_SEMANTICS["label"]))
        if ball_origin == "label"
        else automatic_ball_rows(automatic_ball_path, clip)
    )
    if ball_origin == "automatic":
        images = {int(x["frame"]): x for x in labels["source_pack"]["images"] if x["clip"] == clip}
        for observation in source_rows:
            frame = observation["frame"]
            if not window[0] <= frame <= window[1]:
                continue
            image = images.get(frame)
            native_path = (
                automatic_ball_path.parent
                / "audit_frames_native_1080"
                / clip
                / f"f_{frame:04d}.jpg"
            )
            if (
                image is None
                or not native_path.is_file()
                or provenance.file_sha256(native_path) != image["source"]["sha256"]
            ):
                raise ValueError(
                    f"automatic ball frame {frame} does not bind the original native image"
                )
    if event_origin == "label":
        events = deepcopy(attempt["events"])
        ending = dict(frame=attempt["owner_end_frame"], kind=labels["attempt"]["ending_kind"])
    else:
        events, ending = automatic_events(
            automatic_events_path,
            labels["match_id"],
            clip,
            window,
            observation_scope=observation_scope,
        )
    mixed, packet = compose_s6_documents(
        original,
        labels,
        source_rows,
        events,
        ending,
        ball_origin=ball_origin,
        event_origin=event_origin,
    )
    attempt = packet["attempts"][0]
    all_rows = mixed["ball"]["records"][0]["frames"]
    origins = mixed["stream_origins"]
    operator = packet["ball_observation_operator"]
    packet["inputs"] = [
        _record(inputs["packet"], role="original prepared input lineage"),
        _record(inputs["labels"], role="original opened-development input lineage"),
    ]
    if automatic_event_candidates_path is not None:
        from cv.pipeline.event_paths import load_evidence_packet

        if event_origin != "automatic":
            raise ValueError("candidate sidecars require the automatic event stream")
        load_evidence_packet(
            automatic_event_candidates_path, automatic_events_path, f"{labels['match_id']}__{clip}"
        )
        packet["automatic_event_candidates"] = {
            "record": provenance.file_record(automatic_event_candidates_path),
            "resolved_path": str(automatic_event_candidates_path.resolve()),
            "semantics": "retained_hypotheses_only_firm_events_and_observations_unchanged",
        }
    output.mkdir(parents=True)
    document_path = output / "mixed_observations.json"
    document_path.write_text(json.dumps(mixed, indent=2, allow_nan=False) + "\n")
    record = provenance.file_record(document_path)
    packet["external_label_binding"] = dict(
        resolved_path=str(document_path),
        record=record,
        semantics="actual sanitized mixed consumer document",
    )
    packet_path = output / "packet.json"
    packet_path.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    result = {**deepcopy(row), "labels": record, "packet": provenance.file_record(packet_path)}
    if event_origin == "automatic":
        result.update(serve_number=1, serve_number_uncertain=True)
    receipt = dict(
        row=result,
        stream_origins=origins,
        original_row=deepcopy(row),
        observation_operator=operator,
        full_shared_s6_required=True,
        ball_inventory_count=len(all_rows),
        derived_estimate_frames=[x["frame"] for x in all_rows if x["status"] == "derived_estimate"],
        automatic_inputs=[
            _record(p, role=role)
            for p, role, enabled in (
                (automatic_ball_path, "automatic ball producer", ball_origin == "automatic"),
                (automatic_events_path, "automatic event producer", event_origin == "automatic"),
            )
            if enabled
        ],
    )
    (output / "stream_manifest.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return receipt


def _write_input_overlay(
    labels: dict[str, Any],
    attempt: dict[str, Any],
    cameras: dict[str, Any],
    player_path: Path,
    output: Path,
) -> str | None:
    """Render one native picture of the streams that actually entered the packet."""
    observations = attempt["ball_observations"]
    event_frames = [float(row["frame"]) for row in attempt["events"]]
    target = (
        event_frames[0] if event_frames else float(observations[len(observations) // 2]["frame"])
    )
    observation = min(observations, key=lambda row: abs(float(row["frame"]) - target))
    frame = int(observation["frame"])
    image_records = [
        row
        for row in labels["source_pack"]["images"]
        if row["clip"] == attempt["point_clip"] and int(row["frame"]) == frame
    ]
    if len(image_records) != 1:
        return None
    source = paths.data_root() / image_records[0]["image_url"]
    if not source.is_file():
        return None
    image = Image.open(source).convert("RGB")
    if image.size != (1920, 1080):
        raise ValueError("native 1920x1080 source image required for adapter overlay")
    draw = ImageDraw.Draw(image)
    x, y = float(observation["x1080"]), float(observation["y1080"])
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#00ff66", width=4)
    with player_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        read_box, _, _ = res.native_box_reader(player_path, reader.fieldnames or ())
        player_rows = [
            (row, read_box(row))
            for row in reader
            if row.get("clip") == attempt["point_clip"] and native_frame(row["frame"]) == frame
        ]
    for row, box in player_rows:
        draw.rectangle(tuple(box), outline="#ffcc00", width=3)
        draw.text((box[0], max(0, box[1] - 18)), str(row.get("side", "player")), fill="#ffcc00")
    camera = next((row for row in cameras["cameras"] if int(row["frame"]) == frame), None)
    nearby = [
        f"{row['event_type']}@{row['frame']:g}"
        for row in attempt["events"]
        if abs(float(row["frame"]) - frame) <= 4
    ]
    lines = [
        f"native f{frame}; green ball = {observation['observation_semantics']}",
        f"ball/events/players/camera = {attempt['stream_origins']}",
        f"yellow = sided player localization; camera {None if camera is None else camera['status']}",
        "events: " + (", ".join(nearby) if nearby else "none within four frames"),
    ]
    y_text = 18
    for line in lines:
        draw.text((18, y_text), line, fill="white", stroke_width=2, stroke_fill="black")
        y_text += 28
    image.save(output, quality=92)
    return str(output)


def render_blocked_overview(
    labels: dict[str, Any],
    *,
    output: Path,
    blocker: str,
    origins: dict[str, str],
    automatic_ball_path: Path | None = None,
    automatic_events_path: Path | None = None,
    player_path: Path | None = None,
) -> str | None:
    """Picture the streams of an attempt whose packet could not be built.

    A blocked arm still has to be looked at, so this draws whatever the adapter
    could read at the label's first contact picture and prints the blocker on the
    frame. It never invents an observation: a stream it cannot read is simply absent.
    """
    clip = labels["attempt"]["clip"]
    contacts = [
        float(row["frame"])
        for row in labels["events"]["records"]
        if row["event_type"] == "contact" and row.get("status") == "labeled"
    ]
    window = [int(value) for value in labels["attempt"]["native_window"]]
    frame = int(math.ceil(contacts[0])) if contacts else window[0]
    image_records = [
        row
        for row in labels["source_pack"]["images"]
        if row["clip"] == clip and int(row["frame"]) == frame
    ]
    if len(image_records) != 1:
        return None
    source = paths.data_root() / image_records[0]["image_url"]
    if not source.is_file():
        return None
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    ball_note = "automatic ball: unreadable"
    if automatic_ball_path is not None and automatic_ball_path.is_file():
        try:
            row = next(
                row
                for row in automatic_ball_rows(automatic_ball_path, clip)
                if int(row["frame"]) == frame
            )
            x, y = float(row["x1080"]), float(row["y1080"])
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#00ff66", width=4)
            ball_note = f"automatic ball centre at f{frame}"
        except (StopIteration, ValueError, OSError):
            ball_note = f"automatic ball: no row at f{frame}"
    if player_path is not None and player_path.is_file():
        with player_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            read_box, _, _ = res.native_box_reader(player_path, reader.fieldnames or ())
            for row in reader:
                if row.get("clip") != clip or native_frame(row["frame"]) != frame:
                    continue
                box = read_box(row)
                draw.rectangle(tuple(box), outline="#ffcc00", width=3)
                draw.text(
                    (box[0], max(0, box[1] - 18)), str(row.get("side", "player")), fill="#ffcc00"
                )
    event_note = "automatic events: unreadable"
    if automatic_events_path is not None and automatic_events_path.is_file():
        try:
            events, ending = automatic_events(
                automatic_events_path, labels["match_id"], clip, (window[0], window[1])
            )
            event_note = f"{len(events)} automatic events, ending {ending['frame']:.2f}"
        except (ValueError, KeyError, OSError) as exc:
            event_note = f"automatic events: {exc}"
    lines = [
        f"BLOCKED at native f{frame}: {blocker}",
        f"selected streams: {origins}",
        ball_note,
        event_note,
        "yellow = automatic sided player localization",
    ]
    y_text = 18
    for line in lines:
        draw.text((18, y_text), line[:190], fill="white", stroke_width=2, stroke_fill="black")
        y_text += 28
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)
    return str(output)


def build_packet(
    *,
    labels_path: Path,
    output: Path,
    ball_origin: StreamOrigin,
    event_origin: StreamOrigin,
    player_origin: StreamOrigin,
    camera_origin: StreamOrigin,
    automatic_ball_path: Path,
    automatic_events_path: Path,
    label_player_path: Path,
    automatic_player_path: Path,
    automatic_camera_path: Path,
    label_camera_path: Path,
    surface: str = "hard",
    intrinsic_fallback_registration: str = INTRINSIC_FALLBACK_REGISTRATION_ADMIT,
    shot_homography_propagation: str = SHOT_HOMOGRAPHY_PROPAGATION_OFF,
) -> dict[str, Any]:
    """Write one mixed-origin fitter packet and return its stream manifest.

    The label attempt window is an evaluation-cohort selector in every arm and is
    always disclosed.  It never changes an automatic row or event epoch.
    """
    if output.exists():
        raise FileExistsError(output)
    labels = json.loads(labels_path.read_text())
    clip = labels["attempt"]["clip"]
    window = tuple(int(value) for value in labels["attempt"]["native_window"])
    origins = {
        "ball": ball_origin,
        "events": event_origin,
        "players": player_origin,
        "camera": camera_origin,
    }
    if set(origins.values()) - {"label", "automatic"}:
        raise ValueError("each stream origin must be label or automatic")
    source_rows = (
        label_ball_rows(labels)
        if ball_origin == "label"
        else automatic_ball_rows(automatic_ball_path, clip)
    )
    events, ending = (
        label_events(labels)
        if event_origin == "label"
        else automatic_events(automatic_events_path, labels["match_id"], clip, window)
    )
    contacts = sorted(float(row["frame"]) for row in events if row["event_type"] == "contact")
    if not contacts or ending["frame"] <= contacts[-1]:
        raise ValueError("ordered contact(s) and a later point_end are required")
    first, last = contacts[0], float(ending["frame"])
    observations = [
        row
        for row in source_rows
        if first <= int(row["frame"]) <= last and row.get("status") == "visible"
    ]
    cameras = (
        load_label_cameras(label_camera_path, labels)
        if camera_origin == "label"
        else automatic_camera_document(
            automatic_camera_path,
            labels,
            window,
            intrinsic_fallback_registration=intrinsic_fallback_registration,
            shot_homography_propagation=shot_homography_propagation,
        )
    )
    supported_camera_frames = {
        int(row["frame"])
        for row in cameras["cameras"]
        if row.get("status") == "supported" and row.get("P") is not None
    }
    before_camera_mask = len(observations)
    observations = [row for row in observations if int(row["frame"]) in supported_camera_frames]
    if not observations:
        raise ValueError("no ball observation inside the point has a supported selected camera")
    player_path = label_player_path if player_origin == "label" else automatic_player_path
    with player_path.open(newline="") as handle:
        player_fields = csv.DictReader(handle).fieldnames or ()
    player_columns = (
        res.PLAYER_NATIVE_BOX_COLUMNS
        if set(res.PLAYER_NATIVE_BOX_COLUMNS).issubset(player_fields)
        else ("x0", "y0", "x1", "y1")
    )
    player_scale = uniform_native_scale(player_path, player_columns)
    terminal_bounces = terminal_bounce_count(events, last)
    available_source_frames = {
        int(row["frame"]) for row in labels["source_pack"]["images"] if row.get("clip") == clip
    }
    context = sorted(frame for frame in available_source_frames if frame < first or frame > last)
    attempt = {
        "attempt_id": labels["benchmark_id"],
        "clip": f"{labels['match_id']}__{clip}",
        "point_clip": clip,
        "match_id": labels["match_id"],
        "fps": float(labels["source_pack"]["fps"]),
        "stream_origins": origins,
        "events": events,
        "point_end": ending,
        "first_event_frame": first,
        "owner_end_frame": last,
        "contact_count": len(contacts),
        "terminal_bounce_count": terminal_bounces,
        "structurally_ground_replayable": terminal_bounces in {1, 2},
        "ball_observations": observations,
        # Compatibility alias only. These rows retain their real origin.
        "owner_ball_labels": observations,
        "ball_observation_compatibility_alias": {
            "field": "owner_ball_labels",
            "canonical_field": "ball_observations",
            "does_not_assert_human_annotation": True,
        },
        "labeled_native_frames": (len(observations) if ball_origin == "label" else 0),
        "visible_native_frames": len(observations),
        "complete_native_label_inventory": ball_origin == "label",
        "ball_convention": BALL_SEMANTICS[ball_origin],
        "context_native_frames": context,
        "net_hit_modeled": any(row["event_type"] == "net_hit" for row in events),
        "annotation_origin": (
            "mixed" if len(set(origins.values())) > 1 else next(iter(set(origins.values())))
        ),
    }
    stream_records: dict[str, dict[str, Any]] = {
        "ball": {
            "origin": ball_origin,
            "semantics": BALL_SEMANTICS[ball_origin],
            "inputs": [
                _record(
                    labels_path if ball_origin == "label" else automatic_ball_path,
                    role="ball observations",
                )
            ],
            "selected_observations": len(observations),
            "observations_removed_for_camera_abstention": before_camera_mask - len(observations),
        },
        "events": {
            "origin": event_origin,
            "inputs": [
                _record(
                    labels_path if event_origin == "label" else automatic_events_path,
                    role="physical events and the ending",
                )
            ],
            "physical_event_count": len(events),
            "contact_count": len(contacts),
            "terminal_bounce_count": terminal_bounces,
            "point_end": ending,
        },
        "camera": {
            "origin": camera_origin,
            "inputs": [
                _record(
                    label_camera_path if camera_origin == "label" else automatic_camera_path,
                    role="per-frame camera",
                )
            ],
            "supported_frames": len(supported_camera_frames),
            "frame_scope": cameras.get("frame_scope"),
            "per_frame_registration_required": camera_origin == "automatic",
        },
        "players": {
            "origin": player_origin,
            "inputs": [_record(player_path, role="sided player boxes/pose")],
            "coordinate_columns_read": list(player_columns),
            "sidecar_resolved_native_scale": player_scale,
            "status": (
                "uniform_automatic_sided_box_localization"
                if player_origin == "automatic"
                else "promoted_reference_per_attempt_automatic_localization"
            ),
            "automatic_localization_dependency_disclosed": True,
            "note": (
                "The frozen S6 labels contain hitter identities but no player boxes, pose or "
                "court positions. Both arms are therefore automatic localization: the reference "
                "arm keeps the promoted configuration's per-attempt file choice, and the "
                "automatic arm uses the uniform sided-box stream for every attempt. Neither is "
                "an all-label player stream."
            ),
        },
    }
    packet = {
        "schema": "connected_shooting_stream_packet_v1",
        "scope": __doc__,
        "human_derived": any(value == "label" for value in origins.values()),
        "annotation_origin": "per_stream_manifest",
        "stream_origins": origins,
        "evaluation_window": {
            "native_window": list(window),
            "source": _record(labels_path, role="opened development cohort/window selector"),
        },
        "stream_manifest": stream_records,
        "inputs": [record for stream in stream_records.values() for record in stream["inputs"]],
        "code": provenance.git_record(paths.REPO_ROOT),
        "attempts": [attempt],
        "automatic_inference_eligible": False,
        "automatic_inference_ineligibility": (
            "opened label-defined evaluation window and explicit serve-number reference; "
            "stream substitutions are validation only"
        ),
    }
    manifest = {
        "schema": "connected_shooting_stream_manifest_v1",
        "status": "ready",
        "blocker": None,
        "attempt_id": attempt["attempt_id"],
        "match_id": labels["match_id"],
        "clip": clip,
        "surface": surface,
        "native_window": list(window),
        "stream_origins": origins,
        "streams": stream_records,
        "player_csv": str(player_path),
        "player_image_scale": player_scale,
        "packet_observation_count": len(observations),
        "events": [{"event_type": row["event_type"], "frame": row["frame"]} for row in events],
        "point_end": ending,
        "automatic_rows_rewritten_as_agent_labels": False,
        "automatic_inference_eligible": False,
    }
    # Exercise the unchanged fit scene contract now, so a topology or camera loss
    # is counted as an adapter blocker instead of a worker crash later.
    try:
        search.prepare_attempt(attempt, cameras, surface, observation_fallback=True)
    except (ValueError, KeyError, IndexError) as exc:
        manifest["status"] = "blocked"
        manifest["blocker"] = f"fitter_input_contract: {exc}"
    output.mkdir(parents=True)
    (output / "packet.json").write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    (output / "cameras.json").write_text(json.dumps(cameras, indent=2, allow_nan=False) + "\n")
    manifest["input_picture"] = _write_input_overlay(
        labels, attempt, cameras, player_path, output / "input_overview.jpg"
    )
    (output / "stream_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def centre_observation_operator() -> dict[str, Any]:
    """Select the nominal-centre projection for detector-centre observations.

    The fitter hard-codes a 0.25-frame swept leading-edge exposure operator that is
    correct for owner clicks and wrong for a detector centre.  Rather than edit the
    frozen search module, this rebinds three module attributes for the life of one
    in-process run and records exactly what it rebound.  The direction axes become
    inert because the centre operator never reads them, and the pre-contact toss
    witness is denied the label ball rows so an automatic-ball arm cannot borrow
    human observations through it.
    """
    original_prediction = search.exposure.prediction
    original_directions = search.training_directions
    original_toss = search.toss_witness.precontact_observations

    def centre_prediction(
        scene, parameters, axes, duration, cache=None, *, termination_kind="terminal_bounce"
    ):
        return original_prediction(
            scene, parameters, axes, None, cache, termination_kind=termination_kind
        )

    def inert_axes(scene, bounces, reference, **_unused):
        frames = np.concatenate(scene.observation_frames)
        axes = np.tile([1.0, 0.0], (len(frames), 1))
        return axes, {
            "frames": frames.tolist(),
            "sources": ["unused_for_nominal_centre_operator"] * len(frames),
            "axes": axes.tolist(),
            "note": "Axes are inert: the nominal-centre operator never reads a streak direction.",
        }

    def automatic_toss(
        labels,
        cameras,
        *,
        contact_frame,
        automatic_track=None,
        automatic_track_scale=1.0,
        config=None,
    ):
        withheld = {
            **labels,
            "ball": {
                **labels["ball"],
                "records": [
                    {
                        **record,
                        "frames": [
                            {**row, "status": "withheld_from_automatic_ball_arm"}
                            for row in record["frames"]
                        ],
                    }
                    for record in labels["ball"]["records"]
                ],
            },
        }
        return original_toss(
            withheld,
            cameras,
            contact_frame=contact_frame,
            automatic_track=automatic_track,
            automatic_track_scale=automatic_track_scale,
            **({} if config is None else {"config": config}),
        )

    search.exposure.prediction = centre_prediction
    search.training_directions = inert_axes
    search.toss_witness.precontact_observations = automatic_toss
    return {
        "operator": "pinhole projection of the modeled nominal ball centre",
        "exposure_extent_applied": False,
        "rebound_for_this_run": [
            "real_exposure_replay.prediction -> duration=None nominal centre",
            "agent_whole_point_search.training_directions -> inert axes",
            "toss_witness.precontact_observations -> label ball rows withheld",
        ],
        "restore": (original_prediction, original_directions, original_toss),
    }


def _run_fitter(args: argparse.Namespace) -> None:
    """Invoke the unchanged fitter with the packet's declared observation operator."""
    packet = json.loads(args.packet.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "ready":
        raise ValueError("a blocked adapter packet must not enter the fitter")
    semantics = packet["stream_manifest"]["ball"]["semantics"]
    automatic_centres = semantics == BALL_SEMANTICS["automatic"]
    operator = centre_observation_operator() if automatic_centres else None
    command = [
        "agent_whole_point_search",
        "--labels",
        str(args.labels),
        "--packet",
        str(args.packet),
        "--cameras",
        str(args.cameras),
        "--pose-csv",
        str(manifest["player_csv"]),
        "--coarse-iterations",
        str(args.coarse_iterations),
        "--refine-iterations",
        str(args.refine_iterations),
        "--serve-number",
        str(args.serve_number),
        "--serve-number-evidence",
        str(args.reference_configuration),
        "--surface",
        args.surface,
        "--bounce-witness-mode",
        "subframe_graded_circle",
        "--athlete-prior-mode",
        args.athlete_prior_mode,
        "--observation-fallback",
        args.observation_fallback,
        "--pose-image-scale",
        str(manifest["player_image_scale"]),
        "--output",
        str(args.output),
    ]
    for value in args.player_stature:
        command.extend(["--player-stature", value])
    if args.point_context is not None:
        command.extend(["--point-context", str(args.point_context)])
    for value in args.player_order:
        command.extend(["--player-order", value])
    if args.dense_labels is not None:
        command.extend(["--dense-labels", str(args.dense_labels)])
    if automatic_centres:
        command.extend(
            [
                "--automatic-track",
                str(args.automatic_track),
                "--automatic-track-scale",
                str(uniform_native_scale(args.automatic_track, ("x", "y"))),
            ]
        )
    prior_argv = sys.argv
    try:
        sys.argv = command
        search.main()
    finally:
        sys.argv = prior_argv
        if operator is not None:
            (
                search.exposure.prediction,
                search.training_directions,
                search.toss_witness.precontact_observations,
            ) = operator.pop("restore")
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text())
    report["adapter"] = {
        "schema": "connected_shooting_stream_adapter_receipt_v1",
        "stream_manifest": provenance.file_record(args.manifest),
        "stream_origins": packet["stream_origins"],
        "ball_observation_semantics": semantics,
        "nominal_centre_operator": automatic_centres,
        "automatic_rows_rewritten_as_agent_labels": False,
        "player_csv": manifest["player_csv"],
        "player_image_scale": manifest["player_image_scale"],
    }
    if automatic_centres:
        report["configuration"]["exposure_duration_frames"] = None
        report["configuration"]["exposure_duration_measured"] = None
        report["adapter"]["observation_operator"] = operator
        report["observation_model"] = {
            "semantics": semantics,
            "operator": operator["operator"],
            "exposure_extent_applied": False,
        }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def _build(args: argparse.Namespace) -> None:
    manifest = build_packet(
        labels_path=args.labels,
        output=args.output,
        ball_origin=args.ball,
        event_origin=args.events,
        player_origin=args.players,
        camera_origin=args.camera,
        automatic_ball_path=args.automatic_ball,
        automatic_events_path=args.automatic_events,
        label_player_path=args.label_players,
        automatic_player_path=args.automatic_players,
        automatic_camera_path=args.automatic_camera,
        label_camera_path=args.label_camera,
        surface=args.surface,
    )
    print(json.dumps({"status": manifest["status"], "blocker": manifest["blocker"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-packet", help="build one mixed label/automatic stream packet"
    )
    for name in (
        "labels",
        "output",
        "automatic-ball",
        "automatic-events",
        "label-players",
        "automatic-players",
        "automatic-camera",
        "label-camera",
    ):
        build.add_argument("--" + name, type=Path, required=True)
    for name in STREAMS:
        build.add_argument("--" + name, choices=("label", "automatic"), required=True)
    build.add_argument("--surface", choices=("hard", "clay", "grass"), required=True)
    run = subparsers.add_parser("run-fitter", help="run the unchanged fitter on a built packet")
    for name in (
        "labels",
        "packet",
        "cameras",
        "manifest",
        "reference-configuration",
        "automatic-track",
        "output",
    ):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--surface", choices=("hard", "clay", "grass"), required=True)
    run.add_argument("--serve-number", type=int, choices=(1, 2), required=True)
    run.add_argument("--coarse-iterations", type=int, default=100)
    run.add_argument("--refine-iterations", type=int, default=200)
    run.add_argument(
        "--athlete-prior-mode",
        choices=search.ATHLETE_PRIOR_MODES,
        default="global_hard_caps",
    )
    run.add_argument("--player-stature", action="append", default=[], metavar="PLAYER=METRES")
    run.add_argument("--player-order", action="append", default=[], metavar="PLAYER")
    run.add_argument("--observation-fallback", choices=("off", "on"), default="off")
    run.add_argument("--dense-labels", type=Path)
    run.add_argument(
        "--point-context",
        type=Path,
        help="explicit automatic point-context receipt for soft ending selection",
    )
    args = parser.parse_args()
    if args.command == "run-fitter":
        _run_fitter(args)
    else:
        _build(args)


if __name__ == "__main__":
    main()
