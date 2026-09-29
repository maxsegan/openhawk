"""Bind original native pictures to one measured ball pixel for ``event_wing_streak``.

``event_wing_streak`` measures a moving component in three native exposures cut at a
shared crop.  It is deliberately ignorant of where those pictures came from.  This
module is the only thing that answers that question, and it answers it fail-closed:

* the clip's ``extraction_receipt.json`` (``player_frame_extraction_v1``) is present,
  is signed by its own source identity, was produced with ``native`` and
  ``preserve_source_fps`` at the cadence everything else declares, names this clip's
  point, and lists each of the three pictures with the exact sha256 the bytes read here
  hash to -- so the pixels measured are the *original* extracted exposures and not a
  re-encode, a crop or a later replacement;
* the three pictures are the exposures ``f_{n-1}``, ``f_n`` and ``f_{n+1}`` of one clip,
  at their declared native 1920x1080 size, cut at one shared crop with no resizing;
* the frame-cadence audit says the clip's extraction is ``timing_usable`` and that none
  of the three indices is a repeated picture, so the three exposures are three real
  exposures rather than a conversion artefact;
* the declared cadence agrees across the extraction receipt, the coordinate sidecar, the
  cadence audit and the consuming event, and the three indices lie inside the extracted
  window;
* every consumed picture and every consulted extraction receipt is bound by its exact
  sha256 into the caller's existing
  :class:`~cv.pipeline.event_impulse_support.TrackSnapshot`, so the ordinary run
  manifest names it and ``assert_unchanged`` re-checks its bytes before publishing.

Nothing here invents a picture or a timestamp.  A *missing* receipt is unavailable
evidence -- the sample abstains and the caller keeps its original hold -- and is never
read as a claim that the pictures are original.

The exposure clock is the *frame-relative* preserved-native-fps clock and is recorded as
such: ``absolute_pts_seconds_claimed`` is false and no absolute source time is emitted.
The receipt's own point window is recorded verbatim beside it as the extraction's
declared source interval, not as a per-exposure timestamp.  The streak measurement only
uses the clock to check that the two neighbours straddle the current exposure and are
contemporaneous with it, which a frame-relative clock answers exactly, and this path
emits no event time of its own.

The returned record is the ``event_wing_streak`` record plus a ``binding`` block naming
the exact source, clip, frame, observed pixel and measurement configuration digest.  A
consumer must re-check that block against what it asked for: support measured at frame
``n`` carries frame ``n`` and can never be read as support for ``n+1``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import artifact_cache, provenance
from cv.pipeline import event_wing_streak as streak
from cv.pipeline.event_impulse_support import TrackSnapshot
from cv.pipeline.provenance import file_record
from cv.pipeline.resolution import NATIVE_SIZE, FrameSize

SCHEMA = "event_native_picture_binding_v1"

FRAMES_DIR = "audit_frames_native_1080"
CADENCE_SIDECAR = f"{FRAMES_DIR}.coordinates.json"
CADENCE_AUDIT = "frame_cadence_audit_v1.json"
CADENCE_AUDIT_SCHEMA = "frame_cadence_audit_v1"
EXTRACTION_RECEIPT = "extraction_receipt.json"
EXTRACTION_SCHEMA = "player_frame_extraction_v1"
RECEIPT_ROLE = "native_extraction_receipt"
# One 64x64 crop, the same crop coordinates in all three exposures, centred on the
# observed pixel.  This does *not* cover the whole of the measurement's admissible ball
# geometry: ``event_wing_streak`` allows a streak up to 130 native px long, and a
# component that reaches this crop's edge is refused as ``component_clipped_by_patch_edge``
# rather than measured short.  Anchored at the centre, the crop can contain a streak of
# up to roughly 58 px whose observed row sits at its middle, and less than that when the
# row sits off-centre along the smear.  That is a coverage cost -- the very fastest
# smears abstain here -- and not a wrong measurement.
PATCH_NATIVE_PX = 64
NEIGHBOUR_OFFSETS = (-1, 1)
PICTURE_ROLE = "native_event_picture"

CONFIG = {
    "patch_native_px": PATCH_NATIVE_PX,
    "neighbour_offsets": list(NEIGHBOUR_OFFSETS),
    "frames_dir": FRAMES_DIR,
    "cadence_sidecar": CADENCE_SIDECAR,
    "cadence_audit": CADENCE_AUDIT,
    "extraction_receipt": EXTRACTION_RECEIPT,
    "required_extraction_schema": EXTRACTION_SCHEMA,
    "required_extraction_configuration": {"native": True, "preserve_source_fps": True},
    "required_cadence_decision": "timing_usable",
    "required_native_size": [NATIVE_SIZE.width, NATIVE_SIZE.height],
    "resampling": "none",
    "channel_order_into_measurement": "rgb",
    "invents_pictures_or_timestamps": False,
    "ambient_fallback_to_other_sources": False,
    # A smear longer than this crop can hold reaches the crop edge and is refused by
    # the measurement, so the longest streaks this path can support are shorter than
    # the measurement's own geometric maximum.
    "streak_longer_than_crop": "abstains_as_component_clipped_by_patch_edge",
    "original_extraction_binding": (
        "every measured support requires the clip's player_frame_extraction_v1 receipt: "
        "its source identity must re-sign, declare native preserved-source-fps extraction "
        "at the declared cadence and name this clip's point, and each consumed picture "
        "must be a member of its frame inventory with the exact sha256 the read bytes hash "
        "to; a missing receipt is unavailable evidence and abstains"
    ),
    "absolute_pts_binding": (
        "no absolute source timestamp is emitted on this path: the receipt declares the "
        "point's source interval and preserve_source_fps extraction, not a per-exposure "
        "PTS, so the exposure clock stays the frame-relative preserved-native-fps clock "
        "and says so; this path emits no event time"
    ),
}

REASONS = (
    "native_picture_support",
    "missing_native_frame_cadence_audit",
    "native_cadence_not_timing_usable",
    "missing_native_picture_coordinate_sidecar",
    "inconsistent_declared_native_cadence",
    "native_frame_outside_extracted_window",
    "repeated_native_pictures_in_extraction",
    "missing_native_picture",
    "missing_original_extraction_receipt",
    "unreadable_original_extraction_receipt",
    "unsupported_original_extraction_receipt",
    "original_extraction_not_native_preserved_source",
    "original_extraction_source_identity_changed",
    "original_extraction_point_identity_mismatch",
    "original_extraction_cadence_differs",
    "invalid_original_extraction_inventory",
    "native_picture_not_in_original_extraction",
    "native_picture_differs_from_original_extraction",
    "unreadable_native_picture",
    "native_picture_not_declared_native_size",
    "duplicate_native_picture_bytes",
    "observed_pixel_not_measured",
    "patch_outside_native_image",
)

ABSENT_EVIDENCE_REASONS = (
    "missing_native_frame_cadence_audit",
    "missing_native_picture_coordinate_sidecar",
    "missing_native_picture",
    "missing_original_extraction_receipt",
)


def _cadence_row_fps(row: dict) -> float:
    try:
        return float(row["fps"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _point_is_this_clip(clip: str, point: dict) -> bool:
    """Does the receipt describe *this* clip's point, on its own declared window?

    The clip directory name is the extraction's own point number, so a receipt copied
    from a neighbouring point -- the one way a valid, correctly signed receipt could be
    read as evidence about the wrong pictures -- is refused here.
    """
    number = clip.removeprefix("pt")
    start, end = _float(point.get("t0")), _float(point.get("t1"))
    return (
        number.isdigit()
        and _float(point.get("pt")) == float(int(number))
        and math.isfinite(start)
        and math.isfinite(end)
        and end > start
    )


def _reportable(value: float) -> float | None:
    """A cadence that can be written to a JSON document, or nothing when it cannot.

    An unusable cadence is still worth naming in the abstention it caused, but it is
    named as absent rather than as a number no reader can compare.
    """
    return float(value) if math.isfinite(value) else None


@dataclass
class NativePictureSnapshot:
    """Original native pictures for a run, bound to an existing track snapshot.

    ``track`` is the run's existing :class:`TrackSnapshot`; every cadence document, every
    consulted extraction receipt and every consumed picture is appended to its
    ``paths``/``records``, so the ordinary manifest binds them and its
    ``assert_unchanged`` covers them.  No second provenance framework is introduced.

    Evidence that was *absent* cannot be bound by a hash, so the exact paths that were
    looked for and were not there are retained in ``expected_absent``.  A later resume
    can then tell "this support was unavailable because nothing was there" from "this
    support is still unavailable", instead of holding on evidence that has since arrived.
    """

    track: TrackSnapshot
    root: Path
    cadence_audit: dict[tuple[str, str], dict]
    sidecars: dict[str, dict]
    supports: dict[str, dict] = field(default_factory=dict)
    bound: dict[Path, dict] = field(default_factory=dict)
    receipts: dict[tuple[str, str], dict] = field(default_factory=dict)
    expected_absent: dict[Path, str] = field(default_factory=dict)

    @property
    def bound_pictures(self) -> dict[Path, dict]:
        """Only the consumed pictures, not the documents that admitted them."""
        return {
            path: record
            for path, record in self.bound.items()
            if record.get("role") == PICTURE_ROLE
        }

    @classmethod
    def bind(cls, root: Path, match_ids: list[str], track: TrackSnapshot) -> NativePictureSnapshot:
        """Bind the cadence/extraction documents this run's pictures come from.

        Ancestry that is *absent* is absent evidence: every sample then abstains and the
        caller keeps its original hold.  Ancestry that is *present but not what it
        claims to be* -- an unknown cadence schema, a declared size that is not native,
        an unusable declared cadence -- is a configuration error and is refused here,
        because measuring in it would be measuring in the wrong pictures.
        """
        root = Path(root)
        audit_path = root / CADENCE_AUDIT
        cadence: dict[tuple[str, str], dict] = {}
        absent: dict[Path, str] = {}
        wanted = sorted(set(match_ids))
        if not audit_path.is_file():
            absent[audit_path.resolve()] = "missing_native_frame_cadence_audit"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            if audit.get("schema") != CADENCE_AUDIT_SCHEMA:
                raise ValueError("unsupported frame cadence audit schema")
            track.paths.append(audit_path)
            track.records.append(file_record(audit_path, role="native_frame_cadence_audit"))
            for row in audit.get("rows", []):
                key = (str(row.get("match_id")), str(row.get("clip")))
                if key[0] not in wanted:
                    continue
                if key in cadence:
                    raise ValueError("duplicate native cadence audit row")
                cadence[key] = row
        sidecars: dict[str, dict] = {}
        for match_id in wanted:
            path = root / match_id / CADENCE_SIDECAR
            if not path.is_file():
                absent[path.resolve()] = "missing_native_picture_coordinate_sidecar"
                continue
            declared = json.loads(path.read_text())
            if (
                any(
                    declared.get(key) != {"width": NATIVE_SIZE.width, "height": NATIVE_SIZE.height}
                    for key in ("image_size", "artifact_size")
                )
                or declared.get("frames_dir") != FRAMES_DIR
            ):
                raise ValueError("native streak support requires declared native pictures")
            fps = float(declared.get("fps", 0))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("native picture extraction declares no usable cadence")
            track.paths.append(path)
            track.records.append(file_record(path, role="native_picture_coordinate_space"))
            sidecars[match_id] = {"fps": fps, "source": declared.get("source")}
        track.assert_unchanged()
        return cls(track, root, cadence, sidecars, expected_absent=absent)

    # -- measurement ---------------------------------------------------------------

    @staticmethod
    def support_key(
        match_id: str,
        clip: str,
        frame: int,
        point,
        digest: str,
        *,
        fps: float | None = None,
        observed_motion_native_px=None,
    ) -> str:
        """Exact identity of one request: full float bits and the full config digest.

        Nothing here is rounded.  Two observed pixels that differ in their last bit are
        two different requests and get two different measurements.  The declared cadence
        and the measured local motion are part of the identity too, because the
        measurement depends on both: the cadence decides which pictures are admissible
        and the motion decides whether the component's axis is checked against it.  Two
        neighbouring wing contexts can therefore never share one another's support.
        """
        x, y = (float(value) for value in np.ravel(point)[:2])
        motion = (
            "none"
            if observed_motion_native_px is None
            else ",".join(float(value).hex() for value in np.ravel(observed_motion_native_px)[:2])
        )
        return "|".join(
            [
                match_id,
                clip,
                str(int(frame)),
                x.hex(),
                y.hex(),
                "none" if fps is None else float(fps).hex(),
                motion,
                digest,
            ]
        )

    def measure(
        self,
        match_id: str,
        clip: str,
        frame: int,
        observed_point_native,
        *,
        fps: float,
        observed_motion_native_px=None,
    ) -> dict:
        """Measure native streak support at one *measured* observed pixel.

        ``observed_point_native`` must be a measured detector/tracker row in native
        coordinates; an interpolated or fitted point has no picture to be measured in
        and must never be passed here.
        """
        point = np.asarray(np.ravel(observed_point_native)[:2], dtype=float)
        digest = streak.configuration_digest()
        key = self.support_key(
            match_id,
            clip,
            frame,
            point,
            digest,
            fps=fps,
            observed_motion_native_px=observed_motion_native_px,
        )
        if key not in self.supports:
            self.supports[key] = self._measure(
                match_id,
                clip,
                int(frame),
                point,
                fps=float(fps),
                digest=digest,
                observed_motion_native_px=observed_motion_native_px,
            )
        return self.supports[key]

    # -- original extraction ---------------------------------------------------------

    def extraction(self, match_id: str, clip: str) -> dict:
        """The clip's verified original extraction receipt, or why there is none.

        Read once per clip and cached: the answer depends on the receipt's own bytes,
        which the run binds and re-checks, and not on the request that asked for it.
        """
        key = (match_id, clip)
        if key not in self.receipts:
            self.receipts[key] = self._read_extraction(match_id, clip)
        return self.receipts[key]

    def _read_extraction(self, match_id: str, clip: str) -> dict:
        path = self.root / match_id / FRAMES_DIR / clip / EXTRACTION_RECEIPT
        if not path.is_file():
            # Unavailable evidence, never a claim: without the receipt this module
            # cannot say these pictures are the original extraction, so it says nothing.
            self.expected_absent[path.resolve()] = "missing_original_extraction_receipt"
            return {"reason": "missing_original_extraction_receipt", "path": path}
        record = file_record(path, role=RECEIPT_ROLE)
        self._bind(path, record)
        held = {"reason": None, "path": path, "record": record}
        try:
            receipt = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {**held, "reason": "unreadable_original_extraction_receipt"}
        if file_record(path, role=RECEIPT_ROLE) != record:
            raise ValueError("native extraction receipt changed during inference")
        source = receipt.get("source_identity") if isinstance(receipt, dict) else None
        identity = receipt.get("identity") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != EXTRACTION_SCHEMA
            or not isinstance(source, dict)
            or not isinstance(identity, dict)
        ):
            return {**held, "reason": "unsupported_original_extraction_receipt"}
        try:
            provenance.assert_automatic_document(source, context="native frame extraction")
        except provenance.ProvenanceError:
            return {**held, "reason": "unsupported_original_extraction_receipt"}
        configuration = source.get("configuration")
        declared_fps = (
            _float(configuration.get("fps")) if isinstance(configuration, dict) else float("nan")
        )
        if (
            not isinstance(configuration, dict)
            or configuration.get("native") is not True
            or configuration.get("preserve_source_fps") is not True
            or not math.isfinite(declared_fps)
            or declared_fps <= 0
        ):
            return {**held, "reason": "original_extraction_not_native_preserved_source"}
        # Re-sign the source identity with the shared digest helper the receipt was
        # written with, so an edited configuration, input or code closure is refused.
        fingerprint = source.get("fingerprint")
        unsigned = {key: value for key, value in source.items() if key != "fingerprint"}
        if (
            not isinstance(fingerprint, str)
            or artifact_cache._digest_json(unsigned) != fingerprint
            or identity.get("source_fingerprint") != fingerprint
        ):
            return {**held, "reason": "original_extraction_source_identity_changed"}
        point = identity.get("point")
        if not isinstance(point, dict) or not _point_is_this_clip(clip, point):
            return {**held, "reason": "original_extraction_point_identity_mismatch"}
        inventory: dict[str, str] = {}
        frames = receipt.get("frames")
        if not isinstance(frames, list):
            return {**held, "reason": "invalid_original_extraction_inventory"}
        for entry in frames:
            name = entry.get("name") if isinstance(entry, dict) else None
            digest = entry.get("sha256") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not isinstance(digest, str) or name in inventory:
                return {**held, "reason": "invalid_original_extraction_inventory"}
            inventory[name] = digest
        if not inventory:
            return {**held, "reason": "invalid_original_extraction_inventory"}
        return {
            **held,
            "inventory": inventory,
            "fps": declared_fps,
            "source_fingerprint": fingerprint,
            "source_inputs": source.get("inputs", []),
            "point": {key: point.get(key) for key in ("pt", "t0", "t1")},
            "frames_declared": len(inventory),
        }

    def _measure(
        self,
        match_id: str,
        clip: str,
        frame: int,
        point: np.ndarray,
        *,
        fps: float,
        digest: str,
        observed_motion_native_px,
    ) -> dict:
        binding = {
            "schema": SCHEMA,
            "match_id": match_id,
            "clip": clip,
            "frame": frame,
            "observed_point_native": [_reportable(value) for value in point[:2]],
            "config_digest": digest,
            "patch_native_px": PATCH_NATIVE_PX,
            "requested_fps": _reportable(fps),
            "observed_motion_native_px": (
                None
                if observed_motion_native_px is None
                else [
                    _reportable(value) for value in np.ravel(observed_motion_native_px)[:2].tolist()
                ]
            ),
        }

        def held(reason: str, **details) -> dict:
            return {
                "schema": SCHEMA,
                "available": False,
                "reason": reason,
                "measurement": None,
                "binding": {**binding, **details},
            }

        if not np.isfinite(point).all():
            return held("observed_pixel_not_measured")
        cadence = self.cadence_audit.get((match_id, clip))
        if cadence is None:
            return held("missing_native_frame_cadence_audit")
        if cadence.get("decision") != CONFIG["required_cadence_decision"]:
            return held(
                "native_cadence_not_timing_usable", cadence_decision=cadence.get("decision")
            )
        sidecar = self.sidecars.get(match_id)
        if sidecar is None:
            return held("missing_native_picture_coordinate_sidecar")
        declared = _cadence_row_fps(cadence)
        if (
            not math.isfinite(fps)
            or fps <= 0
            or not math.isfinite(declared)
            or abs(declared - sidecar["fps"]) > 1e-9
            or abs(declared - fps) > 1e-9
        ):
            return held(
                "inconsistent_declared_native_cadence",
                cadence_audit_fps=_reportable(declared),
                coordinate_sidecar_fps=_reportable(sidecar["fps"]),
            )
        frames = [frame + offset for offset in (0, *NEIGHBOUR_OFFSETS)]
        try:
            count = int(cadence.get("frame_count", 0))
            duplicate_frames = {int(f) for f in cadence.get("duplicate_frames", [])}
        except (TypeError, ValueError, OverflowError):
            return held("inconsistent_declared_native_cadence")
        if min(frames) < 1 or max(frames) > count:
            return held("native_frame_outside_extracted_window", extracted_frame_count=count)
        repeated = sorted(set(frames) & duplicate_frames)
        if repeated:
            return held("repeated_native_pictures_in_extraction", repeated_frames=repeated)
        directory = self.root / match_id / FRAMES_DIR / clip
        paths = [directory / f"f_{index:04d}.jpg" for index in frames]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            for path in missing:
                self.expected_absent[path.resolve()] = "missing_native_picture"
            return held("missing_native_picture")
        # One shared crop, cut at the same coordinates in all three exposures and never
        # resized, clamped or padded: a crop that would leave the picture abstains.
        # Centred on the native pixel the observed row falls in -- ``floor``, the pixel
        # that contains the point, not a rounding rule that depends on the half-pixel.
        origin = (
            math.floor(float(point[0])) - PATCH_NATIVE_PX // 2,
            math.floor(float(point[1])) - PATCH_NATIVE_PX // 2,
        )
        if (
            origin[0] < 0
            or origin[1] < 0
            or origin[0] + PATCH_NATIVE_PX > NATIVE_SIZE.width
            or origin[1] + PATCH_NATIVE_PX > NATIVE_SIZE.height
        ):
            return held("patch_outside_native_image", patch_origin_native=list(origin))
        # The pictures may only be measured as the original extracted exposures if the
        # clip's own extraction receipt says that is what they are.
        extraction = self.extraction(match_id, clip)
        if extraction["reason"] is not None:
            return held(extraction["reason"])
        if abs(extraction["fps"] - declared) > 1e-9:
            return held(
                "original_extraction_cadence_differs",
                original_extraction_fps=_reportable(extraction["fps"]),
                cadence_audit_fps=_reportable(declared),
            )
        outside = [path.name for path in paths if path.name not in extraction["inventory"]]
        if outside:
            return held("native_picture_not_in_original_extraction", pictures=outside)
        records, patches = [], []
        for path in paths:
            # Hash, read, then hash again: the record must describe the bytes these
            # pixels came from, not a version that replaced them between the two.
            record = file_record(path, role=PICTURE_ROLE)
            # A file that caused an abstention is still consumed evidence. Its later
            # repair must invalidate the old hold just as a changed accepted picture does.
            self._bind(path, record)
            if record["sha256"] != extraction["inventory"][path.name]:
                return held("native_picture_differs_from_original_extraction", pictures=[path.name])
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if file_record(path, role=PICTURE_ROLE) != record:
                raise ValueError("native picture bytes changed during inference")
            if image is None:
                return held("unreadable_native_picture")
            height, width = image.shape[:2]
            if FrameSize(width, height) != NATIVE_SIZE or image.ndim != 3:
                return held(
                    "native_picture_not_declared_native_size", measured_size=[width, height]
                )
            records.append(record)
            patches.append(
                image[
                    origin[1] : origin[1] + PATCH_NATIVE_PX,
                    origin[0] : origin[0] + PATCH_NATIVE_PX,
                    ::-1,
                ]
            )
        if len({record["sha256"] for record in records}) != len(records):
            return held("duplicate_native_picture_bytes", pictures=records)
        clock = {
            "kind": "frame_relative_preserved_native_fps",
            "fps": fps,
            "origin_frame_index": 1,
            "absolute_pts_seconds_claimed": False,
            "coordinate_sidecar_source": sidecar["source"],
            "cadence_decision": cadence.get("decision"),
        }
        geometry = streak.PatchGeometry(
            origin_native=origin, image_size=NATIVE_SIZE, native_px_per_patch_px=1.0
        )
        current, *neighbours = [
            streak.NativeFrame(patch, index, (index - 1) / fps)
            for patch, index in zip(patches, frames, strict=True)
        ]
        support = streak.measure_support(
            current,
            neighbours,
            point,
            geometry,
            observed_motion_native_px=observed_motion_native_px,
        )
        for path, record in zip(paths, records, strict=True):
            self._bind(path, record)
        return {
            **support,
            "binding": {
                **binding,
                "patch_origin_native": list(origin),
                "extraction_verified": True,
                "extraction": {
                    "schema": EXTRACTION_SCHEMA,
                    "receipt": extraction["record"],
                    "source_fingerprint": extraction["source_fingerprint"],
                    "source_inputs": extraction["source_inputs"],
                    "point": extraction["point"],
                    "fps": _reportable(extraction["fps"]),
                    "frames_declared": extraction["frames_declared"],
                    "configuration": dict(CONFIG["required_extraction_configuration"]),
                },
                "native_clock": clock,
                "cadence_audit": {
                    "decision": cadence.get("decision"),
                    "fps": declared,
                    "frame_count": count,
                    "effective_unique_fps": cadence.get("effective_unique_fps"),
                },
                "pictures": records,
            },
        }

    def _bind(self, path: Path, record: dict) -> None:
        """Record one actually consulted file on the run's existing snapshot."""
        previous = self.bound.get(path)
        if previous is not None:
            if previous["sha256"] != record["sha256"]:
                raise ValueError("native picture bytes changed during inference")
            return
        self.bound[path] = record
        self.track.paths.append(path)
        self.track.records.append(record)

    # -- publication ---------------------------------------------------------------

    def expected_absent_inputs(self) -> list[dict]:
        """The exact evidence paths this run looked for and did not find.

        A hash cannot bind a file that was not there, so an unavailable support is only
        reproducible while these paths stay absent.  A resume that finds one of them has
        new evidence and must re-run rather than keep the earlier hold.
        """
        return [
            {**provenance.portable_path(path), "reason": reason}
            for path, reason in sorted(self.expected_absent.items())
        ]

    def support_document(self, configuration: dict | None = None) -> dict:
        """Every measurement this run consulted, keyed by its exact request identity.

        The measurements, their diagnostics and the picture hashes live here and not on
        the event rows, and no image bytes are embedded anywhere.
        """
        return {
            "schema": SCHEMA,
            "configuration": {**CONFIG, **(configuration or {})},
            "measurement_configuration": dict(streak.CONFIG),
            "measurement_config_digest": streak.configuration_digest(),
            "pictures_consumed": len(self.bound_pictures),
            "extraction_receipts_consulted": sum(
                record.get("role") == RECEIPT_ROLE for record in self.bound.values()
            ),
            "requests": len(self.supports),
            "available": sum(record.get("available") is True for record in self.supports.values()),
            "refusal_counts": self.refusal_counts(),
            "expected_absent_inputs": self.expected_absent_inputs(),
            "supports": dict(sorted(self.supports.items())),
        }

    def refusal_counts(self) -> dict[str, int]:
        return dict(
            Counter(
                str(record.get("reason", "unknown"))
                for record in self.supports.values()
                if record.get("available") is not True
            )
        )

    def assert_unchanged(self) -> None:
        self.track.assert_unchanged()


__all__ = [
    "ABSENT_EVIDENCE_REASONS",
    "CADENCE_AUDIT",
    "CADENCE_SIDECAR",
    "CONFIG",
    "EXTRACTION_RECEIPT",
    "EXTRACTION_SCHEMA",
    "FRAMES_DIR",
    "NEIGHBOUR_OFFSETS",
    "PATCH_NATIVE_PX",
    "REASONS",
    "RECEIPT_ROLE",
    "SCHEMA",
    "NativePictureSnapshot",
]
