"""One reader for the composed automatic ball track and its declared guide ancestry.

The S6 evaluation adapter has always read a composed track this way: every pixel
column resolved through the coordinate sidecar, and the *first declared CSV
source* of that sidecar followed as the coarse guide, so a row the producer
locked onto an interpolated guide position is reported as ``derived_estimate``
rather than as a detection.  That rule lived in
``cv.experiments.connected_shooting.auto_packet``; it is shared here without
changing it so the S5 event stage can bind the same track under the same
ancestry contract.  ``auto_packet`` re-exports these names.

Two readers are offered over one implementation:

* :func:`automatic_ball_rows` -- one point-local clip, the S6 signature;
* :func:`broadcast_ball_rows` -- every clip of one artifact in a single pass,
  which is what a whole-broadcast stage needs (the per-clip reader re-parses a
  14 MB artifact once per point).

:func:`bind_current_track` is the S5 selection contract on top of them.  It is
deliberately stricter than the S6 reader: the current S4 track states nearly
every row as ``coarse_lock``, so its measured/derived split is *entirely* a
property of the declared guide.  A missing, malformed, foreign or untraceable
guide therefore fails closed instead of promoting the whole artifact to
observations.  Rows whose coarse claim cannot be traced are neither observed nor
silently dropped: they abstain, and the count is reported.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cv.pipeline import provenance
from cv.pipeline import resolution as res

# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
LEGACY_EVENT_TRACK = "legacy"
CURRENT_EVENT_TRACK = "s4_current"
EVENT_TRACKS = (LEGACY_EVENT_TRACK, CURRENT_EVENT_TRACK)

LEGACY_TRACK_NAME = "ball_track_joint_native1080_availability_v1.csv"
CURRENT_TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"

#: The source marker a composed row carries when it took a coarse guide position.
GUIDE_CLAIM = "coarse_lock"
#: The guide's own marker for a position it interpolated rather than measured.
INTERPOLATED = "interpolated"

VISIBLE = "visible"
DERIVED = "derived_estimate"

BALL_SEMANTICS_AUTOMATIC = "detector_heatmap_nominal_centre"


class GuideAncestryError(ValueError):
    """The declared guide ancestry of a selected track cannot be established."""


def validate_event_track(value: str) -> str:
    """Return ``value`` if it names a supported S5 track selection."""

    if value not in EVENT_TRACKS:
        raise ValueError(f"unsupported event_track: {value!r}; choose one of {EVENT_TRACKS}")
    return value


def track_name(event_track: str = LEGACY_EVENT_TRACK) -> str:
    """The artifact name the S5 event stage reads under ``event_track``."""

    validate_event_track(event_track)
    return LEGACY_TRACK_NAME if event_track == LEGACY_EVENT_TRACK else CURRENT_TRACK_NAME


def track_path(match_root: Path, event_track: str = LEGACY_EVENT_TRACK) -> Path:
    return Path(match_root) / track_name(event_track)


# --------------------------------------------------------------------------- #
# the shared S6 reader
# --------------------------------------------------------------------------- #
def native_frame(value: str | int | float) -> int:
    """Parse one point-local native frame without changing its cadence."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a native frame")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or int(value) != value:
            raise ValueError("finite integer native frame required")
        return int(value)
    stem = Path(value).stem
    if stem.startswith("f_") and stem[2:].isdigit():
        return int(stem[2:])
    if stem.isdigit():
        return int(stem)
    raise ValueError(f"unrecognised native frame: {value!r}")


def file_binding(path: Path, *, role: str) -> dict[str, Any]:
    """A hashed input record for one existing file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"role": role, **provenance.file_record(path)}


def declared_guide(path: Path) -> Path | None:
    """The first declared CSV source of ``path``'s sidecar, when it is usable.

    This is the tracing rule the S6 reader has always applied: a file name is
    never evidence, the sidecar's own ``source`` chain is.  ``None`` means the
    artifact declares no followable guide, which the S6 reader treats as "no
    guide support" and :func:`bind_current_track` treats as a hard failure.
    """

    sidecar = res.read_coordinate_manifest(path) or {}
    first = str(sidecar.get("source", "")).split(" + ")[0]
    guide = Path(first)
    if not guide.is_absolute():
        guide = path.parent / guide
    if guide == path or not guide.is_file() or guide.suffix != ".csv":
        return None
    return guide


def _guide_index(path: Path, clips: set[str] | None) -> dict[str, dict[int, dict]]:
    """Follow the declared guide input; a zero detector count is not a gap."""
    guide = declared_guide(path)
    if guide is None:
        return {}
    binding = file_binding(guide, role="automatic guide observations")
    output: dict[str, dict[int, dict]] = {}
    with guide.open(newline="") as handle:
        reader = csv.DictReader(handle)
        read, _, _ = res.native_point_reader(guide, reader.fieldnames or ())
        for row in reader:
            clip = row.get("clip")
            if clip is None or (clips is not None and clip not in clips):
                continue
            output.setdefault(clip, {})[native_frame(row["frame"])] = {
                "xy": read(row),
                "sources": row.get("sources", ""),
                "input": binding,
            }
    return output


def guide_support(path: Path, clip: str) -> dict[int, dict]:
    """The declared guide's rows for one clip, keyed by native frame."""

    return _guide_index(path, {clip}).get(clip, {})


def _read_ball_rows(path: Path, clips: set[str] | None) -> dict[str, list[dict[str, Any]]]:
    """Read one composed automatic track as native-pixel detector centres."""
    guide_index = _guide_index(path, clips)
    output: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        read, columns, source_size = res.native_point_reader(path, reader.fieldnames or ())
        for source in reader:
            clip = source.get("clip")
            if clip is None or (clips is not None and clip not in clips):
                continue
            frame = native_frame(source["frame"])
            if (clip, frame) in seen:
                raise ValueError(f"duplicate automatic ball centre for {clip} frame {frame}")
            seen.add((clip, frame))
            x, y = read(source)
            if not np.isfinite([x, y]).all():
                continue
            covariance = None
            covariance_keys = (
                "innovation_cov_xx_native",
                "innovation_cov_xy_native",
                "innovation_cov_yy_native",
            )
            if all(source.get(key) not in {None, ""} for key in covariance_keys):
                covariance = [float(source[key]) for key in covariance_keys]
            sources = source.get("sources", "")
            guide = guide_index.get(clip, {}).get(frame)
            traced = (
                GUIDE_CLAIM in sources
                and guide is not None
                and np.allclose([x, y], guide["xy"], atol=1e-7, rtol=0)
            )
            derived = INTERPOLATED in sources or (traced and INTERPOLATED in guide["sources"])
            output.setdefault(clip, []).append(
                {
                    "frame": frame,
                    "status": DERIVED if derived else VISIBLE,
                    "x1080": x,
                    "y1080": y,
                    # Innovation spread includes predicted-state uncertainty. It
                    # must not be presented as a measured localization radius.
                    "uncertainty_radius_px1080": None,
                    "uncertainty_policy": "unmeasured; shared nominal observation policy",
                    "support_class": "interpolated_estimate"
                    if derived
                    else (
                        "guide_observation"
                        if traced
                        else "unresolved_or_direct_tracker_observation"
                    ),
                    "guide_support": guide if traced else None,
                    "annotation_origin": "automatic",
                    "observation_semantics": BALL_SEMANTICS_AUTOMATIC,
                    "coordinate_columns_read": list(columns),
                    "declared_source_space": source_size.label,
                    "automatic_track_id": source.get("track_id"),
                    "automatic_sources": source.get("sources"),
                    "automatic_confidence": (
                        float(source["confidence"])
                        if source.get("confidence") not in {None, ""}
                        else float(source["score"])
                        if source.get("score") not in {None, ""}
                        else None
                    ),
                    "automatic_covariance_native_px2": covariance,
                    "automatic_covariance_semantics": "innovation_covariance_not_observation_noise",
                }
            )
    return output


def automatic_ball_rows(path: Path, clip: str) -> list[dict[str, Any]]:
    """Read one composed automatic track as native-pixel detector centres."""
    rows = _read_ball_rows(path, {clip}).get(clip, [])
    if not rows:
        raise ValueError(f"automatic ball track has no rows for clip {clip}")
    return rows


def broadcast_ball_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Every clip of one composed track, read exactly as :func:`automatic_ball_rows`.

    One pass over the artifact and one pass over its guide, which is what a
    whole-broadcast consumer needs; the per-row result is identical.
    """

    return _read_ball_rows(Path(path), None)


# --------------------------------------------------------------------------- #
# the S5 current-track binding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CurrentTrackBinding:
    """One resolved current-track selection, shared by every S5 consumer.

    ``observations`` are the measured rows only, in native 1920x1080 pixels --
    the space :func:`automatic_ball_rows` reports in, whatever columns the
    sidecar declares.  ``derived`` and ``uncertain`` frames are carried as
    diagnostic context: they are *not* observations, so a crop cut on one of
    those frames is positioned by the same interpolation an absent row gets and
    reports ``track_observed=False``.
    """

    event_track: str
    track: Path
    track_sidecar: Path
    guide: Path
    guide_sidecar: Path
    image_size: res.FrameSize
    columns: tuple[str, ...]
    declared_source_space: str
    observations: dict[str, dict[int, tuple[float, float]]]
    derived: dict[str, tuple[int, ...]]
    uncertain: dict[str, tuple[int, ...]]
    inputs: tuple[dict[str, Any], ...]

    @property
    def artifact_size(self) -> res.FrameSize:
        """The frame :attr:`observations` are written in: always native."""

        return res.NATIVE_SIZE

    @property
    def scale_x(self) -> float:
        """Multiplier taking a native observation into this source's image pixels."""

        return self.image_size.width / res.NATIVE_SIZE.width

    @property
    def scale_y(self) -> float:
        return self.image_size.height / res.NATIVE_SIZE.height

    def counts(self) -> dict[str, int]:
        return {
            "observed_rows": sum(len(rows) for rows in self.observations.values()),
            "derived_rows": sum(len(frames) for frames in self.derived.values()),
            "ancestry_uncertain_rows": sum(len(frames) for frames in self.uncertain.values()),
            "clips": len(set(self.observations) | set(self.derived) | set(self.uncertain)),
        }

    def record(self) -> dict[str, Any]:
        """The manifest/receipt block describing this binding."""

        return {
            "schema": "s5_measured_track_binding_v1",
            "event_track": self.event_track,
            "observation_rule": (
                "measured rows of the selected track only; guide-interpolated and "
                "ancestry-uncertain rows position a crop but are never observed"
            ),
            "coordinate_columns_read": list(self.columns),
            "declared_source_space": self.declared_source_space,
            "observation_space": res.NATIVE_SIZE.label,
            "image_size": {"width": self.image_size.width, "height": self.image_size.height},
            "inputs": list(self.inputs),
            **self.counts(),
        }

    def paths(self) -> list[Path]:
        """The files this binding reads, for cache identity."""

        return [self.track, self.track_sidecar, self.guide, self.guide_sidecar]


def current_track_inputs(match_root: Path) -> list[Path]:
    """Selected track, its sidecar, the declared guide and the guide's sidecar.

    Resolved from the sidecars alone so a cache-identity caller does not have to
    parse either artifact.  Missing files are reported by
    :func:`bind_current_track`, not silently dropped here.
    """

    track = track_path(match_root, CURRENT_EVENT_TRACK)
    if not track.is_file():
        return []
    paths = [track]
    sidecar = res.coordinate_manifest_path(track)
    if sidecar.is_file():
        paths.append(sidecar)
    guide = declared_guide(track)
    if guide is not None:
        paths.append(guide)
        guide_sidecar = res.coordinate_manifest_path(guide)
        if guide_sidecar.is_file():
            paths.append(guide_sidecar)
    return paths


def _require_guide(track: Path) -> Path:
    """The declared guide of ``track``, or an explicit ancestry failure."""

    manifest = res.read_coordinate_manifest(track)
    if manifest is None:
        raise res.MissingCoordinateContract(
            f"{res.coordinate_manifest_path(track)} is missing; {track} declares no "
            "coordinate space and no ancestry"
        )
    first = str(manifest.get("source", "")).split(" + ")[0].strip()
    if not first:
        raise GuideAncestryError(f"{track} declares no source ancestry in its coordinate sidecar")
    guide = Path(first)
    if not guide.is_absolute():
        guide = track.parent / guide
    if guide == track:
        raise GuideAncestryError(f"{track} declares itself as its own guide")
    if guide.suffix != ".csv":
        raise GuideAncestryError(
            f"{track} declares a first source that is not a guide CSV: {first}"
        )
    if not guide.is_file():
        raise GuideAncestryError(f"{track} declares a guide that is missing: {guide}")
    if res.read_coordinate_manifest(guide) is None:
        raise res.MissingCoordinateContract(
            f"{res.coordinate_manifest_path(guide)} is missing; the declared guide {guide} "
            "states no coordinate space"
        )
    return guide


def bind_current_track(match_root: Path, clips: Iterable[str] | None = None) -> CurrentTrackBinding:
    """Resolve the current S4 track of one broadcast, with its guide ancestry.

    Fails closed rather than falling back: an absent track, an absent or
    malformed coordinate sidecar, and an absent, malformed, self-referential or
    untraceable declared guide are all errors.  There is no ambient file search
    and no legacy fallback -- a caller that cannot bind the current track has no
    current track.
    """

    match_root = Path(match_root)
    track = track_path(match_root, CURRENT_EVENT_TRACK)
    if not track.is_file():
        raise FileNotFoundError(f"the selected current ball track is missing: {track}")
    guide = _require_guide(track)
    selected = set(clips) if clips is not None else None
    rows = _read_ball_rows(track, selected)

    observations: dict[str, dict[int, tuple[float, float]]] = {}
    derived: dict[str, list[int]] = {}
    uncertain: dict[str, list[int]] = {}
    columns: tuple[str, ...] = ()
    declared_space = ""
    claimed = 0
    traced = 0
    for clip, clip_rows in rows.items():
        for row in clip_rows:
            columns = tuple(row["coordinate_columns_read"])
            declared_space = row["declared_source_space"]
            frame = int(row["frame"])
            claims_guide = GUIDE_CLAIM in (row["automatic_sources"] or "")
            claimed += claims_guide
            traced += row["guide_support"] is not None
            if row["status"] == DERIVED:
                derived.setdefault(clip, []).append(frame)
            elif claims_guide and row["guide_support"] is None:
                # The row states coarse-guide ancestry that this guide cannot
                # confirm, so whether it is a measurement or another hidden
                # interpolation is unknown.  Abstain instead of asserting.
                uncertain.setdefault(clip, []).append(frame)
            else:
                observations.setdefault(clip, {})[frame] = (
                    float(row["x1080"]),
                    float(row["y1080"]),
                )
    if claimed and not traced:
        raise GuideAncestryError(
            f"{track} states {claimed} coarse-guide rows and {guide} traces none of them; "
            "the declared guide is foreign to this track"
        )
    manifest = res.read_coordinate_manifest(track) or {}
    image = manifest["image_size"]
    return CurrentTrackBinding(
        event_track=CURRENT_EVENT_TRACK,
        track=track,
        track_sidecar=res.coordinate_manifest_path(track),
        guide=guide,
        guide_sidecar=res.coordinate_manifest_path(guide),
        image_size=res.FrameSize(int(image["width"]), int(image["height"])),
        columns=columns,
        declared_source_space=declared_space,
        observations=observations,
        derived={clip: tuple(sorted(frames)) for clip, frames in sorted(derived.items())},
        uncertain={clip: tuple(sorted(frames)) for clip, frames in sorted(uncertain.items())},
        inputs=(
            file_binding(track, role="selected_current_ball_track"),
            file_binding(res.coordinate_manifest_path(track), role="selected_track_coordinates"),
            file_binding(guide, role="declared_guide_observations"),
            file_binding(res.coordinate_manifest_path(guide), role="declared_guide_coordinates"),
        ),
    )
