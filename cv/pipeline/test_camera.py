from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline.camera import (
    PLAY_CLUSTER_SAMPLE_FRAMES,
    _color_coherence,
    _select_play_clusters,
    extract_small,
    pick_play_clusters,
    segments_from_flags,
    smooth,
)


def test_reviewed_play_clusters_require_explicit_acknowledgement(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "cv/pipeline/camera.py",
            "--video",
            "missing.mp4",
            "--out",
            str(tmp_path),
            "--play-clusters",
            "1,2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "human-assisted" in result.stderr


def test_camera_has_no_match_specific_play_overrides() -> None:
    source = Path("cv/pipeline/camera.py").read_text()

    assert "PLAY_OVERRIDES" not in source
    assert '"rg2025f":' not in source
    assert '"uso2025f":' not in source


def test_camera_reuses_timestamp_aligned_source_observations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        image = np.full((540, 960, 3), index * 40, dtype=np.uint8)
        assert cv2.imwrite(str(source / f"s_{index + 1:06d}.jpg"), image)

    frames = extract_small(
        "unused.mp4",
        str(tmp_path / "small"),
        1.0,
        str(source),
    )

    assert len(frames) == 3
    assert cv2.imread(frames[0]).shape[:2] == (180, 320)


def test_color_coherence_prefers_surface_dominated_frames(tmp_path: Path) -> None:
    uniform = tmp_path / "uniform.jpg"
    varied = tmp_path / "varied.jpg"
    assert cv2.imwrite(str(uniform), np.full((180, 320, 3), (50, 100, 180), dtype=np.uint8))
    rng = np.random.default_rng(7)
    assert cv2.imwrite(str(varied), rng.integers(0, 256, (180, 320, 3), dtype=np.uint8))

    assert _color_coherence(str(uniform)) > 0.95
    assert _color_coherence(str(varied)) < 0.10


def test_play_cluster_selection_combines_wide_players_surface_and_lines() -> None:
    scores = {
        0: (3.0, 9.0, 0.30),
        1: (12.0, 60.0, 0.30),
        2: (5.0, 15.0, 0.28),
        3: (8.0, 15.0, 0.14),
    }
    wide = {0: 1.0, 1: 1.0, 2: 0.6, 3: 1.0}

    assert _select_play_clusters(scores, wide) == {0, 2}


def test_play_cluster_selection_recovers_low_contrast_grass_when_primary_is_empty() -> None:
    scores = {
        0: (1.52, 5.0, 0.304),
        1: (8.61, 105.0, 0.082),
        2: (1.48, 6.0, 0.246),
    }
    wide = {0: 0.857, 1: 0.857, 2: 0.571}

    assert _select_play_clusters(scores, wide) == {0}


def test_wide_view_fallback_prefers_surface_dominant_high_line_cluster() -> None:
    scores = {
        3: (35.0, 139.0, 0.252),
        4: (20.0, 53.0, 0.388),
        7: (3.0, 16.0, 0.162),
    }
    wide = {3: 1.0, 4: 1.0, 7: 0.286}

    assert _select_play_clusters(scores, wide) == {4}


def test_no_play_camera_evidence_yields_no_segments() -> None:
    flags = smooth(np.zeros(120, dtype=bool))

    assert not flags.any()
    assert segments_from_flags(flags, fps=1.0) == []


def test_automatic_court_solver_does_not_import_manual_anchors() -> None:
    source = Path("cv/pipeline/court.py").read_text()

    assert "court_manual_diagnostics" not in source
    assert "allow_manual_anchors" not in source


def test_cluster_evidence_is_not_decided_by_seven_frames(tmp_path: Path, monkeypatch) -> None:
    """Every admission witness is a median or mean over the per-cluster sample, so the
    sample size is the resolution of the decision.  Seven frames gave ``wide_view_fraction``
    only the eight values j/7 and let one person detection decide a 1,700-frame cluster."""
    frames = [str(tmp_path / f"f_{i:06d}.jpg") for i in range(400)]
    labels = np.zeros(len(frames), dtype=int)
    labels[200:] = 1
    scored: list[str] = []
    monkeypatch.setattr("cv.pipeline.camera._courtness", lambda path: scored.append(path) or 12)
    monkeypatch.setattr("cv.pipeline.camera._color_coherence", lambda path: 0.30)

    pick_play_clusters(labels, 2, frames)

    assert PLAY_CLUSTER_SAMPLE_FRAMES > 7
    assert len(scored) == 2 * PLAY_CLUSTER_SAMPLE_FRAMES
    assert len(set(scored)) == len(scored)


def test_small_clusters_still_use_every_frame_they_have(tmp_path: Path, monkeypatch) -> None:
    frames = [str(tmp_path / f"f_{i:06d}.jpg") for i in range(12)]
    labels = np.zeros(len(frames), dtype=int)
    scored: list[str] = []
    monkeypatch.setattr("cv.pipeline.camera._courtness", lambda path: scored.append(path) or 12)
    monkeypatch.setattr("cv.pipeline.camera._color_coherence", lambda path: 0.30)

    pick_play_clusters(labels, 1, frames, sample_frames=PLAY_CLUSTER_SAMPLE_FRAMES)

    assert sorted(set(scored)) == sorted(frames)


def test_seven_frames_discarded_a_measured_play_cluster() -> None:
    """Frozen evidence from wta_2024_1051_f_300_elena_rybakina_marta_kostyuk, whose whole
    broadcast produced no play segment.  Cluster 4 (1,732 frames) is the wide court camera;
    its seven-frame wide-player estimate was 2/7 and its 63-frame estimate is 0.71."""
    scores = {
        0: (5.01, 28.0, 0.179),
        3: (0.0, 0.0, 0.911),
        4: (5.92, 18.0, 0.329),
        9: (13.15, 50.0, 0.263),
    }

    assert _select_play_clusters(scores, {0: 0.0, 3: 0.0, 4: 0.286, 9: 0.0}) == set()

    scores[4] = (8.45, 25.0, 0.338)

    assert _select_play_clusters(scores, {0: 0.0, 3: 0.02, 4: 0.714, 9: 0.0}) == {4}


def test_line_ceiling_still_rejects_measured_scenic_venue_clusters() -> None:
    """Frozen 63-frame evidence from wta_2024_2074_sf_298_katie_boulter_emma_navarro.  Its
    four highest-line clusters are rain-delay venue shots: court lines from several outside
    courts plus enough distant spectators for the wide-player witness to reach 0.98.  The
    play camera is cluster 2, at 44 segments, inside the band."""
    scores = {
        0: (13.29, 137.0, 0.097),
        2: (11.88, 44.0, 0.270),
        3: (9.14, 127.0, 0.072),
        5: (6.72, 105.0, 0.064),
        7: (10.29, 139.0, 0.074),
    }
    wide = {0: 0.81, 2: 0.43, 3: 0.57, 5: 0.87, 7: 0.98}

    assert _select_play_clusters(scores, wide) == {2}


def test_zero_yield_pass_recovers_a_measured_play_camera_above_the_line_ceiling() -> None:
    """Frozen evidence from wta_2021_2012_r16_287_belinda_bencic_petra_martic, which produced
    no play segment at all.  Cluster 0 (1,463 frames) is the grass play camera; 59 detected
    segments put it over the ceiling and its wide-player fraction of 0.51 -- the camera often
    holds one player -- keeps it under the established fallback's 0.75 bar."""
    scores = {
        0: (12.27, 59.0, 0.208),
        1: (2.63, 15.0, 0.175),
        3: (7.11, 69.0, 0.103),
        10: (7.65, 75.0, 0.102),
    }
    wide = {0: 0.51, 1: 0.0, 3: 0.11, 10: 0.49}

    assert _select_play_clusters(scores, wide) == {0}


def test_zero_yield_pass_still_refuses_measured_non_broadcast_sources() -> None:
    """Three sources are not match video at all: a webcam watch-along, an animated
    score tracker and a commentary stream.  Their chrome carries huge line counts and, where
    a webcam is in shot, full wide-player support; surface coherence is what refuses them, and
    it must keep doing so once the ceiling is lifted for a broadcast yielding nothing."""
    watch_along = ({0: (36.3, 157.0, 0.163), 1: (25.4, 156.0, 0.163)}, {0: 1.0, 1: 0.98})
    score_tracker = ({1: (11.0, 44.0, 0.251), 3: (11.9, 46.0, 0.259)}, {1: 0.0, 3: 0.0})
    commentary = ({0: (21.6, 148.0, 0.146), 9: (20.9, 143.0, 0.146)}, {0: 0.97, 9: 1.0})

    for scores, wide in (watch_along, score_tracker, commentary):
        assert _select_play_clusters(scores, wide) == set()


def test_zero_yield_pass_cannot_disturb_a_broadcast_that_already_has_a_play_cluster() -> None:
    play = (4.5, 18.0, 0.250)
    scenic = (16.0, 80.0, 0.400)

    assert _select_play_clusters({0: play, 1: scenic}, {0: 0.60, 1: 0.71}) == {0}
    assert _select_play_clusters({1: scenic}, {1: 0.71}) == {1}
