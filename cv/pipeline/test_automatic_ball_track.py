import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import automatic_ball_track as track
from cv.pipeline import resolution as res

NATIVE_SIDECAR = {
    "schema": "tennis.coordinate-space.v1",
    "image_size": {"width": 1920, "height": 1080},
    "artifact_size": {"width": 1920, "height": 1080},
    "legacy_artifact_size": {"width": 960, "height": 540},
    "coordinate_columns": {
        "legacy_960x540": ["x", "y"],
        "native_1920x1080": ["x_native", "y_native"],
    },
}

TRACK_COLUMNS = ("clip", "frame", "x", "y", "x_native", "y_native", "score", "sources", "track_id")
GUIDE_COLUMNS = ("clip", "frame", "x", "y", "x_native", "y_native", "score", "sources")


def write_track(path: Path, rows, *, source: str | None = None, columns=TRACK_COLUMNS) -> Path:
    """Write one composed-track-shaped CSV and its coordinate sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            record = {
                "clip": row.get("clip", "pt0001"),
                "frame": f"f_{int(row['frame']):05d}.jpg",
                "x": "nan" if row["x"] is None else row["x"] / 2.0,
                "y": "nan" if row["y"] is None else row["y"] / 2.0,
                "x_native": "nan" if row["x"] is None else row["x"],
                "y_native": "nan" if row["y"] is None else row["y"],
                "score": row.get("score", 1.0),
                "sources": row.get("sources", ""),
                "track_id": row.get("track_id", "t1"),
            }
            writer.writerow({name: record[name] for name in columns})
    sidecar = dict(NATIVE_SIDECAR)
    if source is not None:
        sidecar["source"] = source
    res.coordinate_manifest_path(path).write_text(json.dumps(sidecar))
    return path


def write_guide(path: Path, rows) -> Path:
    return write_track(path, rows, columns=GUIDE_COLUMNS)


def guided_track(tmp_path: Path, track_rows, guide_rows) -> Path:
    guide = write_guide(tmp_path / "guide.csv", guide_rows)
    return write_track(tmp_path / "composed.csv", track_rows, source=f"{guide} + other.csv")


# --------------------------------------------------------------------------- #
# parity with the reader this module was extracted from
# --------------------------------------------------------------------------- #
def _previous_s6_reader(path: Path, clip: str) -> list[dict]:
    """The pre-extraction ``auto_packet.automatic_ball_rows``, verbatim.

    Kept here so "the shared module changes nothing for S6" is asserted against
    the actual old code rather than against a paraphrase of it.
    """

    def guide_support(path: Path, clip: str) -> dict[int, dict]:
        sidecar = res.read_coordinate_manifest(path) or {}
        first = str(sidecar.get("source", "")).split(" + ")[0]
        guide = Path(first)
        if not guide.is_absolute():
            guide = path.parent / guide
        if guide == path or not guide.is_file() or guide.suffix != ".csv":
            return {}
        binding = {"role": "automatic guide observations"}
        with guide.open(newline="") as handle:
            reader = csv.DictReader(handle)
            read, _, _ = res.native_point_reader(guide, reader.fieldnames or ())
            return {
                track.native_frame(row["frame"]): {
                    "xy": read(row),
                    "sources": row.get("sources", ""),
                    "input": binding,
                }
                for row in reader
                if row.get("clip") == clip
            }

    rows: list[dict] = []
    seen: set[int] = set()
    guide_rows = guide_support(path, clip)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        read, columns, source_size = res.native_point_reader(path, reader.fieldnames or ())
        for source in reader:
            if source.get("clip") != clip:
                continue
            frame = track.native_frame(source["frame"])
            if frame in seen:
                raise ValueError(f"duplicate automatic ball centre for {clip} frame {frame}")
            seen.add(frame)
            x, y = read(source)
            if not np.isfinite([x, y]).all():
                continue
            sources = source.get("sources", "")
            guide = guide_rows.get(frame)
            traced = (
                "coarse_lock" in sources
                and guide is not None
                and np.allclose([x, y], guide["xy"], atol=1e-7, rtol=0)
            )
            derived = "interpolated" in sources or (traced and "interpolated" in guide["sources"])
            rows.append(
                {
                    "frame": frame,
                    "status": "derived_estimate" if derived else "visible",
                    "x1080": x,
                    "y1080": y,
                    "support_class": "interpolated_estimate"
                    if derived
                    else (
                        "guide_observation"
                        if traced
                        else "unresolved_or_direct_tracker_observation"
                    ),
                    "traced": traced,
                    "coordinate_columns_read": list(columns),
                    "declared_source_space": source_size.label,
                    "automatic_sources": source.get("sources"),
                }
            )
    if not rows:
        raise ValueError(f"automatic ball track has no rows for clip {clip}")
    return rows


def _mixed_rows():
    """Every ancestry case the current track actually produces."""

    return (
        # traced to a measured guide row
        [
            {"frame": 1, "x": 100.0, "y": 200.0, "sources": "coarse_lock+provenance:coarse"},
            # traced to a guide row the guide itself interpolated
            {"frame": 2, "x": 110.0, "y": 205.0, "sources": "coarse_lock+provenance:coarse"},
            # the track states its own interpolation
            {"frame": 3, "x": 120.0, "y": 210.0, "sources": "interpolated"},
            # a coarse claim the guide cannot confirm: no guide row there
            {"frame": 4, "x": 130.0, "y": 215.0, "sources": "coarse_lock+provenance:coarse"},
            # a coarse claim whose pixel disagrees with the guide's
            {"frame": 5, "x": 140.0, "y": 220.0, "sources": "coarse_lock+provenance:coarse"},
            # a direct measured row with no coarse claim at all
            {
                "frame": 6,
                "x": 150.0,
                "y": 225.0,
                "sources": "branched_crop_wasb+provenance:crop",
            },
            # a row with no usable pixel
            {"frame": 7, "x": None, "y": None, "sources": "coarse_lock+provenance:coarse"},
            # another clip, which must not leak into this one
            {"frame": 1, "x": 900.0, "y": 500.0, "clip": "pt0002", "sources": "interpolated"},
        ],
        [
            {"frame": 1, "x": 100.0, "y": 200.0, "sources": "sliding_wasb"},
            {"frame": 2, "x": 110.0, "y": 205.0, "sources": "interpolated"},
            {"frame": 3, "x": 120.0, "y": 210.0, "sources": "sliding_wasb"},
            {"frame": 5, "x": 999.0, "y": 999.0, "sources": "sliding_wasb"},
            {"frame": 6, "x": 150.0, "y": 225.0, "sources": "interpolated"},
            {"frame": 1, "x": 900.0, "y": 500.0, "clip": "pt0002", "sources": "sliding_wasb"},
        ],
    )


def test_the_shared_reader_reproduces_the_previous_s6_rows(tmp_path) -> None:
    track_rows, guide_rows = _mixed_rows()
    path = guided_track(tmp_path, track_rows, guide_rows)

    shared = track.automatic_ball_rows(path, "pt0001")
    previous = _previous_s6_reader(path, "pt0001")

    assert len(shared) == len(previous) == 6
    for new, old in zip(shared, previous, strict=True):
        for field in (
            "frame",
            "status",
            "x1080",
            "y1080",
            "support_class",
            "coordinate_columns_read",
            "declared_source_space",
            "automatic_sources",
        ):
            assert new[field] == old[field]
        assert (new["guide_support"] is not None) is old["traced"]
    assert [row["status"] for row in shared] == [
        "visible",
        "derived_estimate",
        "derived_estimate",
        "visible",
        "visible",
        "visible",
    ]


def test_auto_packet_re_exports_the_shared_reader() -> None:
    from cv.experiments.connected_shooting import auto_packet

    assert auto_packet.automatic_ball_rows is track.automatic_ball_rows
    assert auto_packet.native_frame is track.native_frame
    assert auto_packet._record is track.file_binding
    assert auto_packet.BALL_SEMANTICS["automatic"] == "detector_heatmap_nominal_centre"


def test_the_broadcast_reader_matches_the_per_clip_reader(tmp_path) -> None:
    track_rows, guide_rows = _mixed_rows()
    path = guided_track(tmp_path, track_rows, guide_rows)

    every = track.broadcast_ball_rows(path)

    assert sorted(every) == ["pt0001", "pt0002"]
    for clip in every:
        assert every[clip] == track.automatic_ball_rows(path, clip)


def test_a_clip_with_no_rows_is_refused_by_the_per_clip_reader(tmp_path) -> None:
    path = guided_track(tmp_path, [{"frame": 1, "x": 1.0, "y": 2.0}], [])

    with pytest.raises(ValueError, match="no rows for clip pt0009"):
        track.automatic_ball_rows(path, "pt0009")
    assert "pt0009" not in track.broadcast_ball_rows(path)


def test_duplicate_current_centres_are_refused(tmp_path) -> None:
    rows = [
        {"frame": 4, "x": 100.0, "y": 100.0, "score": 0.1, "sources": "sliding_wasb"},
        {"frame": 4, "x": 700.0, "y": 400.0, "score": 0.9, "sources": "sliding_wasb"},
    ]
    match_root = current_root(tmp_path, rows, [])

    # No score-ordered winner is selected: two centres for one frame is an error.
    with pytest.raises(ValueError, match="duplicate automatic ball centre for pt0001 frame 4"):
        track.bind_current_track(match_root)
    with pytest.raises(ValueError, match="duplicate automatic ball centre"):
        track.automatic_ball_rows(match_root / track.CURRENT_TRACK_NAME, "pt0001")


# --------------------------------------------------------------------------- #
# the S5 binding
# --------------------------------------------------------------------------- #
def current_root(tmp_path: Path, track_rows, guide_rows, *, source: str | None = None) -> Path:
    match_root = tmp_path / "bc"
    guide = write_guide(match_root / "guide.csv", guide_rows)
    write_track(
        match_root / track.CURRENT_TRACK_NAME,
        track_rows,
        source=f"{guide} + candidates.csv" if source is None else source,
    )
    return match_root


def test_only_measured_rows_of_the_current_track_become_observations(tmp_path) -> None:
    track_rows, guide_rows = _mixed_rows()
    match_root = current_root(tmp_path, track_rows, guide_rows)

    binding = track.bind_current_track(match_root)

    # Frame 1 is a traced measurement, frame 6 a direct tracker row.
    assert binding.observations["pt0001"] == {1: (100.0, 200.0), 6: (150.0, 225.0)}
    # Frame 2's guide row is interpolated -- the track's own source string hides
    # that -- and frame 3 says so itself.  Neither is an observation.
    assert binding.derived["pt0001"] == (2, 3)
    # Frames 4 and 5 claim coarse ancestry the guide cannot confirm.
    assert binding.uncertain["pt0001"] == (4, 5)
    assert binding.counts() == {
        # pt0002 carries one more derived row, which the broadcast counts hold.
        "observed_rows": 2,
        "derived_rows": 3,
        "ancestry_uncertain_rows": 2,
        "clips": 2,
    }
    assert binding.guide.name == "guide.csv"
    assert [record["role"] for record in binding.inputs] == [
        "selected_current_ball_track",
        "selected_track_coordinates",
        "declared_guide_observations",
        "declared_guide_coordinates",
    ]
    assert binding.artifact_size == res.FrameSize(1920, 1080)
    assert (binding.scale_x, binding.scale_y) == (1.0, 1.0)


def test_a_direct_measurement_replacing_an_interpolated_guide_row_stays_visible(tmp_path) -> None:
    """The guide interpolated the frame; the track measured it elsewhere."""

    guide_rows = [{"frame": 9, "x": 300.0, "y": 300.0, "sources": "interpolated"}]
    match_root = current_root(
        tmp_path,
        [
            {
                "frame": 9,
                "x": 640.0,
                "y": 480.0,
                "sources": "branched_crop_wasb+provenance:crop",
            }
        ],
        guide_rows,
    )

    binding = track.bind_current_track(match_root)

    assert binding.observations["pt0001"] == {9: (640.0, 480.0)}
    assert binding.derived == {}
    assert binding.uncertain == {}


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    (
        ("missing_guide", track.GuideAncestryError, "declares a guide that is missing"),
        ("no_source", track.GuideAncestryError, "declares no source ancestry"),
        ("self_source", track.GuideAncestryError, "declares itself"),
        ("foreign_suffix", track.GuideAncestryError, "not a guide CSV"),
        ("guide_sidecar", res.MissingCoordinateContract, "states no coordinate space"),
        ("track_sidecar", res.MissingCoordinateContract, "declares no coordinate space"),
    ),
)
def test_broken_guide_ancestry_fails_closed(tmp_path, mutate, error, match) -> None:
    track_rows, guide_rows = _mixed_rows()
    match_root = current_root(tmp_path, track_rows, guide_rows)
    selected = match_root / track.CURRENT_TRACK_NAME
    # A legacy track is available beside it; nothing may fall back to it.
    write_track(
        match_root / track.LEGACY_TRACK_NAME,
        [{"frame": 1, "x": 11.0, "y": 12.0, "sources": "sliding_wasb"}],
    )
    sidecar_path = res.coordinate_manifest_path(selected)
    sidecar = json.loads(sidecar_path.read_text())
    if mutate == "missing_guide":
        (match_root / "guide.csv").unlink()
    elif mutate == "no_source":
        del sidecar["source"]
        sidecar_path.write_text(json.dumps(sidecar))
    elif mutate == "self_source":
        sidecar["source"] = str(selected)
        sidecar_path.write_text(json.dumps(sidecar))
    elif mutate == "foreign_suffix":
        sidecar["source"] = str(match_root / "camera_P_per_point.npz")
        sidecar_path.write_text(json.dumps(sidecar))
    elif mutate == "guide_sidecar":
        res.coordinate_manifest_path(match_root / "guide.csv").unlink()
    elif mutate == "track_sidecar":
        sidecar_path.unlink()

    with pytest.raises(error, match=match):
        track.bind_current_track(match_root)


def test_a_foreign_guide_that_traces_nothing_is_refused(tmp_path) -> None:
    """A guide of another broadcast parses, but confirms no coarse row."""

    track_rows, _ = _mixed_rows()
    foreign = [
        {"frame": 400, "x": 10.0, "y": 10.0, "clip": "pt0099", "sources": "sliding_wasb"},
    ]
    match_root = current_root(tmp_path, track_rows, foreign)

    with pytest.raises(track.GuideAncestryError, match="foreign to this track"):
        track.bind_current_track(match_root)


def test_a_missing_current_track_never_falls_back(tmp_path) -> None:
    match_root = tmp_path / "bc"
    write_track(
        match_root / track.LEGACY_TRACK_NAME,
        [{"frame": 1, "x": 11.0, "y": 12.0, "sources": "sliding_wasb"}],
    )

    with pytest.raises(FileNotFoundError, match=track.CURRENT_TRACK_NAME):
        track.bind_current_track(match_root)
    assert track.current_track_inputs(match_root) == []


def test_the_bound_inputs_are_the_selected_track_and_its_declared_guide(tmp_path) -> None:
    track_rows, guide_rows = _mixed_rows()
    match_root = current_root(tmp_path, track_rows, guide_rows)

    inputs = track.current_track_inputs(match_root)

    assert inputs == [
        match_root / track.CURRENT_TRACK_NAME,
        res.coordinate_manifest_path(match_root / track.CURRENT_TRACK_NAME),
        match_root / "guide.csv",
        res.coordinate_manifest_path(match_root / "guide.csv"),
    ]
    assert inputs == track.bind_current_track(match_root).paths()


def test_the_binding_record_declares_the_rows_it_refused(tmp_path) -> None:
    track_rows, guide_rows = _mixed_rows()
    binding = track.bind_current_track(current_root(tmp_path, track_rows, guide_rows))

    record = binding.record()

    assert record["event_track"] == "s4_current"
    assert record["observation_space"] == "1920x1080"
    assert record["coordinate_columns_read"] == ["x_native", "y_native"]
    assert record["ancestry_uncertain_rows"] == 2
    assert record["derived_rows"] == 3
    assert [item["role"] for item in record["inputs"]] == [
        "selected_current_ball_track",
        "selected_track_coordinates",
        "declared_guide_observations",
        "declared_guide_coordinates",
    ]
    assert all(math.isfinite(float(item["bytes"])) for item in record["inputs"])


def test_legacy_columns_are_read_through_the_sidecar(tmp_path) -> None:
    """A current track without native columns is still read in native pixels."""

    guide = write_guide(
        tmp_path / "bc" / "guide.csv",
        [{"frame": 1, "x": 100.0, "y": 200.0, "sources": "sliding_wasb"}],
    )
    legacy_only = ("clip", "frame", "x", "y", "score", "sources", "track_id")
    write_track(
        tmp_path / "bc" / track.CURRENT_TRACK_NAME,
        [{"frame": 1, "x": 100.0, "y": 200.0, "sources": "coarse_lock+provenance:coarse"}],
        source=str(guide),
        columns=legacy_only,
    )

    binding = track.bind_current_track(tmp_path / "bc")

    assert binding.observations["pt0001"] == {1: (100.0, 200.0)}
    assert binding.columns == ("x", "y")
    assert binding.declared_source_space == "960x540"
    assert binding.artifact_size == res.FrameSize(1920, 1080)
