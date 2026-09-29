import json
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import event_crops


@pytest.mark.parametrize("fps", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_event_crops_reject_absent_or_invalid_source_cadence(tmp_path, fps):
    (tmp_path / f"{event_crops.FRAMES_DIR}.coordinates.json").write_text(json.dumps({"fps": fps}))
    with pytest.raises(ValueError, match="frame rate"):
        event_crops._fps(tmp_path)


def test_event_crops_require_a_source_cadence_sidecar(tmp_path):
    with pytest.raises(ValueError, match="frame rate"):
        event_crops._fps(tmp_path)


@pytest.fixture
def runtime_crop_cache(tmp_path, monkeypatch):
    root = tmp_path / "root"
    match = root / "match"
    match.mkdir(parents=True)
    (root / "active_play_v1.json").write_text(json.dumps({"match/pt0001": {}}))
    (match / event_crops.TRACK_NAME).write_text("track")
    frames = match / event_crops.FRAMES_DIR
    frames.mkdir()
    (frames / "f_0001.jpg").write_bytes(b"frame")
    output = tmp_path / "crops"
    calls = []

    def builder(root, output, jobs, *, alternate=False):
        calls.append(alternate)
        output.mkdir(exist_ok=True)
        shard = output / "match.npz"
        np.savez(shard, example=np.array([1]))
        manifest = {
            "schema": event_crops.ALTERNATE_SCHEMA if alternate else event_crops.SCHEMA,
            "shards": [{"path": str(shard)}],
        }
        (output / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(event_crops, "build_dataset", builder)
    monkeypatch.setattr(
        event_crops,
        "build_alternate_dataset",
        lambda *args: builder(*args, alternate=True),
    )
    return root, output, calls


@pytest.mark.parametrize("alternate", [False, True])
def test_runtime_crop_cache_requires_receipt_and_reuses_content(runtime_crop_cache, alternate):
    root, output, calls = runtime_crop_cache
    builder = event_crops.build_alternate_dataset if alternate else event_crops.build_dataset
    builder(root, output, 1)  # Bare historical manifest is not a receipt.
    assert not event_crops.ensure_runtime_dataset(root, output, alternate=alternate)["reused"]
    assert event_crops.ensure_runtime_dataset(root, output, alternate=alternate)["reused"]
    assert calls == [alternate, alternate]
    assert not event_crops.ensure_runtime_dataset(root, output, reuse=False, alternate=alternate)[
        "reused"
    ]


@pytest.mark.parametrize(
    "name",
    [
        event_crops.TRACK_NAME,
        f"{event_crops.TRACK_NAME}.coordinates.json",
        "court_H_per_point.npz",
        f"{event_crops.FRAMES_DIR}/f_0001.jpg",
        f"{event_crops.FRAMES_DIR}.coordinates.json",
        event_crops.CADENCE_NAME,
        event_crops.AUDIO_CACHE_NAME,
        "audit_reel_point_map.csv",
        event_crops.REEL_NAME,
    ],
)
def test_runtime_crop_cache_invalidates_each_source(runtime_crop_cache, name):
    root, output, calls = runtime_crop_cache
    event_crops.ensure_runtime_dataset(root, output)
    path = root / "match" / name
    path.write_bytes(b"changed source")
    assert not event_crops.ensure_runtime_dataset(root, output)["reused"]
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["manifest.json", "match.npz"])
def test_runtime_crop_cache_rebuilds_corrupt_outputs(runtime_crop_cache, name):
    root, output, calls = runtime_crop_cache
    event_crops.ensure_runtime_dataset(root, output)
    (output / name).write_bytes(b"corrupt")
    assert not event_crops.ensure_runtime_dataset(root, output)["reused"]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "name",
    [
        event_crops.ARC_TRACK_NAME,
        *event_crops.FAR_NATIVE_CANDIDATES,
        *event_crops.SLIDING_CANDIDATES,
    ],
)
def test_runtime_alternate_crop_cache_tracks_optional_witnesses(runtime_crop_cache, name):
    root, output, calls = runtime_crop_cache
    event_crops.ensure_runtime_dataset(root, output, alternate=True)
    path = root / "match" / name
    path.write_text("new witness")
    assert not event_crops.ensure_runtime_dataset(root, output, alternate=True)["reused"]
    path.unlink()
    assert not event_crops.ensure_runtime_dataset(root, output, alternate=True)["reused"]
    assert len(calls) == 3


def test_runtime_crop_cache_rejects_input_change_during_build(runtime_crop_cache, monkeypatch):
    root, output, _calls = runtime_crop_cache
    builder = event_crops.build_dataset

    def mutating_builder(*args):
        result = builder(*args)
        (root / "match" / event_crops.CADENCE_NAME).write_text("new optional input")
        return result

    monkeypatch.setattr(event_crops, "build_dataset", mutating_builder)
    with pytest.raises(RuntimeError, match="changed while building"):
        event_crops.ensure_runtime_dataset(root, output)
    assert not event_crops.receipt_path(output, "event_crops").exists()


def test_runtime_crop_cache_tracks_active_scope(runtime_crop_cache):
    root, output, _calls = runtime_crop_cache
    event_crops.ensure_runtime_dataset(root, output)
    (root / "active_play_v1.json").write_text(json.dumps({"match/pt0002": {}}))
    assert not event_crops.ensure_runtime_dataset(root, output)["reused"]


def test_crop_bounds_centres_and_clamps():
    left, top = event_crops.crop_bounds(960.0, 540.0, 192, 108)
    assert (left, top) == (864, 486)
    left, top = event_crops.crop_bounds(5.0, 5.0, 192, 108)
    assert (left, top) == (0, 0)
    left, top = event_crops.crop_bounds(1919.0, 1079.0, 192, 108)
    assert (left, top) == (1920 - 192, 1080 - 108)


def test_cut_pads_outside_the_frame():
    frame = np.full((1080, 1920, 3), 7, dtype=np.uint8)
    patch = event_crops.cut(frame, -10, -5, 192, 108)
    assert patch.shape == (108, 192, 3)
    assert patch[0, 0].tolist() == [0, 0, 0]
    assert patch[10, 20].tolist() == [7, 7, 7]
    assert event_crops.cut(None, 0, 0, 192, 108).max() == 0


def test_unique_exposures_drops_repeats():
    assert event_crops.unique_exposures([], 5) == [1, 2, 3, 4, 5]
    assert event_crops.unique_exposures([2, 4], 5) == [1, 3, 5]


def test_sequence_frames_keeps_the_centre_and_length():
    exposures = list(range(1, 41))
    members = event_crops.sequence_frames(exposures, 20)
    assert len(members) == event_crops.SEQUENCE_LENGTH
    assert members[event_crops.SEQUENCE_RADIUS] == 20
    assert members == list(range(12, 28))


def test_sequence_frames_uses_unique_exposures_only():
    exposures = event_crops.unique_exposures([3, 4, 5], 40)
    members = event_crops.sequence_frames(exposures, 20)
    assert len(members) == event_crops.SEQUENCE_LENGTH
    assert len(set(members)) == event_crops.SEQUENCE_LENGTH
    assert 3 not in members and 4 not in members and 5 not in members


def test_sequence_frames_clamps_at_a_clip_boundary():
    members = event_crops.sequence_frames(list(range(1, 6)), 2)
    assert len(members) == event_crops.SEQUENCE_LENGTH
    assert min(members) == 1 and max(members) == 5


def test_track_centres_interpolates_gaps_and_reports_observations():
    observations = {10: (100.0, 50.0), 14: (140.0, 90.0)}
    frames = [10, 11, 12, 13, 14]
    x, y, observed = event_crops.track_centres(observations, frames, 2.0, 2.0)
    assert x.tolist() == [200.0, 220.0, 240.0, 260.0, 280.0]
    assert y.tolist() == [100.0, 120.0, 140.0, 160.0, 180.0]
    assert observed.tolist() == [True, False, False, False, True]


def test_track_centres_without_observations_falls_back_to_frame_centre():
    x, y, observed = event_crops.track_centres({}, [1, 2], 2.0, 2.0)
    assert x.tolist() == [960.0, 960.0]
    assert y.tolist() == [540.0, 540.0]
    assert not observed.any()


def test_log_mel_shape_and_response():
    rate = event_crops.SAMPLE_RATE
    time = np.arange(int(rate * event_crops.AUDIO_SECONDS)) / rate
    quiet = event_crops.log_mel(np.zeros_like(time, dtype=np.float32))
    tone = event_crops.log_mel(np.sin(2 * np.pi * 3000.0 * time).astype(np.float32))
    assert quiet.shape == (event_crops.MEL_BANDS, event_crops.MEL_FRAMES)
    assert tone.shape == quiet.shape
    assert tone.max() > quiet.max()
    assert int(np.argmax(tone[:, 10])) > event_crops.MEL_BANDS // 3


def test_mel_filterbank_is_normalised_triangles():
    bank = event_crops.mel_filterbank(bands=8, fft=64, sample_rate=1600)
    assert bank.shape == (8, 33)
    assert (bank >= 0).all()
    assert bank.sum() > 0


def test_memmap_member_matches_the_stored_array(tmp_path):
    path = tmp_path / "shard.npz"
    tight = np.random.randint(0, 255, size=(3, 2, 4, 5, 3), dtype=np.uint8)
    mel = np.random.rand(3, 4, 5).astype(np.float16)
    np.savez(path, tight=tight, mel=mel)
    assert np.array_equal(np.asarray(event_crops.memmap_member(path, "tight")), tight)
    assert np.array_equal(np.asarray(event_crops.memmap_member(path, "mel")), mel)


def test_memmap_member_refuses_a_compressed_member(tmp_path):
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, tight=np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="compressed"):
        event_crops.memmap_member(path, "tight")


def _manifest(frames):
    return {
        "rows_index": [
            {
                "broadcast": "b",
                "clip": "b__pt0001",
                "frame": frame,
                "court_x": None,
                "court_y": None,
            }
            for frame in frames
        ]
    }


def test_join_truth_anchors_within_half_a_frame_and_keeps_the_offset():
    manifest = _manifest([100, 101, 102, 103, 110])
    joined = event_crops.join_truth(
        manifest,
        [{"clip": "b__pt0001", "event_type": "contact", "frame": 101.5}],
    )
    assert joined["assigned"] == 1
    assert joined["truth_type"] == ["none", "contact", "none", "none", "none"]
    assert joined["truth_offset"][1] == pytest.approx(0.5)
    assert joined["ignore"] == [True, False, True, True, False]


def test_join_truth_reports_events_outside_the_candidate_set():
    manifest = _manifest([100])
    joined = event_crops.join_truth(
        manifest, [{"clip": "b__pt0009", "event_type": "bounce", "frame": 4.0}]
    )
    assert joined["assigned"] == 0
    assert joined["outside_automatic_event_scope"] == 1


def test_join_truth_rejects_two_types_on_one_row():
    manifest = _manifest([100])
    with pytest.raises(ValueError, match="conflicting truth"):
        event_crops.join_truth(
            manifest,
            [
                {"clip": "b__pt0001", "event_type": "contact", "frame": 100.0},
                {"clip": "b__pt0001", "event_type": "bounce", "frame": 100.2},
            ],
        )


def test_point_windows_reads_the_reel_point_map(tmp_path):
    (tmp_path / "audit_reel_point_map.csv").write_text(
        "pt,rally_t_start,rally_t_end\n1,0.000,10.000\n2,10.000,24.000\n"
    )
    windows = event_crops.point_windows(tmp_path)
    assert windows == {"pt0001": (0.0, 10.0), "pt0002": (10.0, 24.0)}


def test_point_windows_prefers_the_audio_cache(tmp_path):
    metadata = json.dumps({"windows": {"pt0001": [1.0, 2.0]}})
    np.savez(tmp_path / event_crops.AUDIO_CACHE_NAME, metadata=np.asarray(metadata))
    assert event_crops.point_windows(tmp_path) == {"pt0001": (1.0, 2.0)}


def _sidecar(tmp_path, payload):
    path = tmp_path / "track.csv.coordinates.json"
    path.write_text(json.dumps(payload))
    return path


LEGACY_SIDECAR = {
    "artifact_size": {"width": 960, "height": 540},
    "image_size": {"width": 1920, "height": 1080},
}
NATIVE_SIDECAR = {
    "artifact_size": {"width": 1920, "height": 1080},
    "image_size": {"width": 1920, "height": 1080},
    "legacy_artifact_size": {"width": 960, "height": 540},
    "coordinate_columns": {
        "legacy_960x540": ["x", "y"],
        "native_1920x1080": ["x_native", "y_native"],
    },
}


def test_a_legacy_track_sidecar_keeps_the_x_y_columns_and_the_two_times_scale(tmp_path):
    space = event_crops.track_coordinate_space(
        _sidecar(tmp_path, LEGACY_SIDECAR), ["clip", "frame", "x", "y", "score"]
    )
    assert (space["x_column"], space["y_column"]) == ("x", "y")
    assert (space["scale_x"], space["scale_y"]) == (2.0, 2.0)


def test_a_native_track_sidecar_reads_the_native_columns_at_scale_one(tmp_path):
    # The sidecar calls the artifact native while x/y are still 960x540, so
    # trusting artifact_size alone would put every crop at half the true
    # position with no error raised anywhere.
    space = event_crops.track_coordinate_space(
        _sidecar(tmp_path, NATIVE_SIDECAR),
        ["clip", "frame", "x", "y", "score", "x_native", "y_native"],
    )
    assert (space["x_column"], space["y_column"]) == ("x_native", "y_native")
    assert (space["scale_x"], space["scale_y"]) == (1.0, 1.0)
    assert space["space"] == event_crops.NATIVE_COLUMNS_KEY


def test_a_native_sidecar_without_its_native_columns_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="native columns"):
        event_crops.track_coordinate_space(
            _sidecar(tmp_path, NATIVE_SIDECAR), ["clip", "frame", "x", "y"]
        )


def test_declared_legacy_columns_are_scaled_by_the_legacy_artifact_size(tmp_path):
    payload = dict(NATIVE_SIDECAR)
    payload["coordinate_columns"] = {"legacy_960x540": ["x", "y"]}
    space = event_crops.track_coordinate_space(
        _sidecar(tmp_path, payload), ["clip", "frame", "x", "y"]
    )
    assert (space["x_column"], space["y_column"]) == ("x", "y")
    assert (space["scale_x"], space["scale_y"]) == (2.0, 2.0)


def test_read_track_reads_the_named_columns(tmp_path):
    path = tmp_path / "track.csv"
    path.write_text(
        "clip,frame,x,y,score,x_native,y_native\npt0001,f_0007.jpg,100.0,50.0,0.9,200.0,100.0\n"
    )
    legacy = event_crops.read_track(path)
    assert legacy["pt0001"][7] == (100.0, 50.0)
    native = event_crops.read_track(path, "x_native", "y_native")
    assert native["pt0001"][7] == (200.0, 100.0)
    with pytest.raises(ValueError, match="no column"):
        event_crops.read_track(path, "x_missing", "y_native")


def _candidate_sidecar(tmp_path, name, native=True):
    payload = {
        "schema": "tennis.coordinate-space.v1",
        "image_size": {"width": 1920, "height": 1080},
        "artifact_size": {"width": 1920, "height": 1080}
        if native
        else {"width": 960, "height": 540},
    }
    if native:
        payload["coordinate_columns"] = {"native_1920x1080": ["x_native", "y_native"]}
    (tmp_path / f"{name}.coordinates.json").write_text(json.dumps(payload))


def _write_candidates(tmp_path, name, rows, native=True):
    columns = ["clip", "frame", "x", "y", "score"] + (["x_native", "y_native"] if native else [])
    lines = [",".join(columns)]
    for clip, frame, x, y, score in rows:
        values = [clip, f"f_{frame:04d}.jpg", f"{x / 2:.3f}", f"{y / 2:.3f}", f"{score}"]
        if native:
            values += [f"{x:.3f}", f"{y:.3f}"]
        lines.append(",".join(values))
    (tmp_path / name).write_text("\n".join(lines) + "\n")
    _candidate_sidecar(tmp_path, name, native=native)


def test_read_native_rows_scales_a_legacy_candidate_artifact(tmp_path):
    _write_candidates(tmp_path, "candidates.csv", [("pt0001", 5, 800.0, 400.0, 0.9)], native=False)
    rows = event_crops._read_native_rows(tmp_path / "candidates.csv")
    assert rows["pt0001"][5][0][:2] == pytest.approx((800.0, 400.0))


def test_alternate_centres_leave_an_observed_frame_alone():
    frames = [10, 11]
    centre_x = np.asarray([100.0, 200.0], dtype=np.float32)
    centre_y = np.asarray([100.0, 200.0], dtype=np.float32)
    observed = np.asarray([True, False])
    rows = event_crops.alternate_centres(
        Path("."),
        "pt0001",
        frames,
        centre_x,
        centre_y,
        observed,
        arc_track={10: (900.0, 900.0), 11: (900.0, 900.0)},
    )
    assert [row.frame for row in rows] == [11]
    assert rows[0].source == "arc_track"
    assert rows[0].centre_x == pytest.approx(900.0)


def test_an_alternate_that_does_not_move_the_crop_is_not_cut():
    rows = event_crops.alternate_centres(
        Path("."),
        "pt0001",
        [10],
        np.asarray([100.0], dtype=np.float32),
        np.asarray([100.0], dtype=np.float32),
        np.asarray([False]),
        arc_track={10: (100.0 + event_crops.ALTERNATE_MIN_OFFSET_PX / 2, 100.0)},
    )
    assert rows == []


def test_a_corner_is_labelled_but_names_the_same_pixel():
    arc = {9: (100.0, 100.0), 10: (300.0, 100.0), 11: (300.0, 300.0)}
    rows = event_crops.alternate_centres(
        Path("."),
        "pt0001",
        [10],
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        arc_track=arc,
    )
    assert [(row.source, row.centre_x, row.centre_y) for row in rows] == [
        ("track_corner", 300.0, 100.0)
    ]


def test_the_sliding_fallback_speaks_only_when_nothing_else_does():
    common = dict(
        clip="pt0001",
        frames=[10],
        centre_x=np.asarray([0.0], dtype=np.float32),
        centre_y=np.asarray([0.0], dtype=np.float32),
        observed=np.asarray([False]),
    )
    with_arc = event_crops.alternate_centres(
        Path("."),
        common["clip"],
        common["frames"],
        common["centre_x"],
        common["centre_y"],
        common["observed"],
        arc_track={10: (500.0, 500.0)},
        sliding={10: [(900.0, 900.0, 0.4)]},
    )
    assert [row.source for row in with_arc] == ["arc_track"]
    alone = event_crops.alternate_centres(
        Path("."),
        common["clip"],
        common["frames"],
        common["centre_x"],
        common["centre_y"],
        common["observed"],
        sliding={10: [(900.0, 900.0, 0.4), (800.0, 800.0, 0.9)]},
    )
    assert [(row.source, row.centre_x) for row in alone] == [("sliding", 800.0)]


def test_two_alternates_on_one_frame_collapse_when_they_agree():
    rows = event_crops.alternate_centres(
        Path("."),
        "pt0001",
        [10],
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        arc_track={10: (500.0, 500.0)},
        far_native={10: [(505.0, 500.0, 0.8)]},
    )
    assert [row.source for row in rows] == ["arc_track"]


def test_two_alternates_that_disagree_are_both_cut():
    rows = event_crops.alternate_centres(
        Path("."),
        "pt0001",
        [10],
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        arc_track={10: (500.0, 500.0)},
        far_native={10: [(900.0, 500.0, 0.8)]},
    )
    assert [row.source for row in rows] == ["arc_track", "far_native"]


def test_normalise_shard_rows_accepts_pairs_and_explicit_centres():
    rows = event_crops.normalise_shard_rows(
        [("pt0001", 4), event_crops.ShardRow("pt0001", 5, 10.0, 20.0, False, "far_native")]
    )
    assert [row.source for row in rows] == ["track", "far_native"]
    assert np.isnan(rows[0].centre_x)
    assert rows[1].centre_x == 10.0


def test_source_rank_orders_the_track_row_first():
    ranks = [event_crops._source_rank(name) for name in ("track", *event_crops.ALTERNATE_SOURCES)]
    assert ranks == sorted(ranks) and ranks[0] == 0


def test_court_normalised_projects_through_the_point_homography():
    homography = np.asarray([[1.0 / 192.0, 0.0, 0.0], [0.0, 1.0 / 108.0, 0.0], [0.0, 0.0, 1.0]])
    court_x, court_y = event_crops._court_normalised({1: homography}, 1, 192.0, 108.0)
    assert court_x == pytest.approx(1.0 / 10.97)
    assert court_y == pytest.approx(1.0 / 23.77)
    assert event_crops._court_normalised({}, 1, 0.0, 0.0) == (None, None)


def _proposal_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A one-broadcast root with a native track sidecar and one proposal each."""

    match = tmp_path / "bc"
    match.mkdir()
    (match / f"{event_crops.TRACK_NAME}.coordinates.json").write_text(
        json.dumps({"schema": "tennis.coordinate-space.v1", **NATIVE_SIDECAR})
    )
    (match / event_crops.TRACK_NAME).write_text(
        "clip,frame,x,y,x_native,y_native\n"
        + "".join(
            f"pt0001,f_{frame:04d}.jpg,{250 + frame / 2},200,{500 + frame},400\n"
            for frame in (8, 9, 12, 13)
        )
    )
    document = tmp_path / "proposals.json"
    document.write_text(
        json.dumps(
            {
                "schema": "event_proposals_v1",
                "labels_or_reviewed_inputs": [],
                "proposals": [
                    {
                        "clip": "bc__pt0001",
                        "frame": 10.5,
                        "start_frame": 7.5,
                        "end_frame": 13.5,
                        "kinds": ["bounce"],
                        "source": "physics_departure",
                        "confidence": 0.9,
                        "evidence": {"predicted_x": 900.0, "predicted_y": 750.0},
                    },
                    {
                        "clip": "bc__pt0001",
                        "frame": 9.0,
                        "start_frame": 6.0,
                        "end_frame": 12.0,
                        "kinds": ["contact", "bounce", "net_hit"],
                        "source": "track_corner",
                        "confidence": 0.5,
                        "evidence": {},
                    },
                ],
            }
        )
    )
    return tmp_path, document


def test_proposal_crops_use_the_predicted_pixel_inside_a_track_gap(tmp_path):
    root, document = _proposal_fixture(tmp_path)

    rows = event_crops.proposal_crop_rows(root, "bc", document)

    assert [row.source for row in rows] == ["physics_departure"]
    assert (rows[0].frame, rows[0].centre_x, rows[0].centre_y) == (10, 900.0, 750.0)
    # Frame 10 has no detection of its own, so the row is not an observation.
    assert rows[0].track_observed is False
    assert rows[0].proposal_kinds == ("bounce",)


def test_a_proposal_on_the_composed_centre_is_not_cut_twice(tmp_path):
    # The corner proposal names no pixel of its own, so it falls back to the
    # interpolated composed centre the base store already cut.
    root, document = _proposal_fixture(tmp_path)

    rows = event_crops.proposal_crop_rows(root, "bc", document)

    assert all(row.source != "track_corner" for row in rows)
