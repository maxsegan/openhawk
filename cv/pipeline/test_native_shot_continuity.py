from pathlib import Path

import cv2
import numpy as np
import pytest

from cv.pipeline.native_shot_continuity import qualify_native_continuity


def sequence(tmp_path: Path, transform=None):
    base = np.random.default_rng(4).integers(0, 256, (180, 320), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (7, 7), 0)
    paths = []
    for frame in range(1, 26):
        offset = frame if transform is None else transform(frame)
        image = np.roll(base, offset, axis=1)
        path = tmp_path / f"f_{frame:04d}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths, np.arange(1, 26)


def test_smooth_translation_requires_positive_registered_native_support(tmp_path):
    paths, frames = sequence(tmp_path)
    result = qualify_native_continuity(paths, frames, 9, 13, registration_inliers=120)
    assert result["continuous"]
    assert result["reason"] == "registered_distributed_native_motion"
    assert result["native_frames"] == list(range(5, 18))


def test_same_camera_temporal_jump_keeps_boundary(tmp_path):
    paths, frames = sequence(tmp_path, lambda f: f + (25 if f >= 11 else 0))
    result = qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)
    assert not result["continuous"]
    assert result["reason"] == "temporally_localized_or_ambiguous_change"


def test_duplicate_native_frame_cannot_prove_motion_continuity(tmp_path):
    paths, frames = sequence(tmp_path, lambda f: 11 if f == 12 else f)
    assert not qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)[
        "continuous"
    ]


def test_missing_native_picture_does_not_bridge_time(tmp_path):
    paths, frames = sequence(tmp_path)
    del paths[10]
    frames = np.delete(frames, 10)
    result = qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)
    assert not result["continuous"]
    assert result["reason"] == "unavailable_native_neighborhood"


@pytest.mark.parametrize("support", [0, 59.9, float("nan")])
def test_low_or_unknown_registration_preserves_cut(tmp_path, support):
    paths, frames = sequence(tmp_path)
    assert not qualify_native_continuity(paths, frames, 9, 13, registration_inliers=support)[
        "continuous"
    ]


def test_three_frame_blend_does_not_fake_continuity(tmp_path):
    paths, frames = sequence(tmp_path)
    other = np.random.default_rng(7).integers(0, 256, (180, 320), dtype=np.uint8)
    for frame in range(11, 26):
        image = cv2.imread(str(paths[frame - 1]), cv2.IMREAD_GRAYSCALE)
        alpha = min(1.0, (frame - 10) / 3)
        mixed = np.clip((1 - alpha) * image + alpha * other, 0, 255).astype(np.uint8)
        assert cv2.imwrite(str(paths[frame - 1]), mixed)
    assert not qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)[
        "continuous"
    ]


def test_flash_remains_unresolved_not_new_flash_repair(tmp_path):
    paths, frames = sequence(tmp_path)
    image = cv2.imread(str(paths[10]), cv2.IMREAD_GRAYSCALE)
    assert cv2.imwrite(str(paths[10]), np.clip(image.astype(float) + 90, 0, 255).astype(np.uint8))
    assert not qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)[
        "continuous"
    ]


def test_refinement_only_sees_existing_boundaries_after_short_run_merge(monkeypatch):
    from cv.pipeline import native_shot_continuity as native, shot_segments as shots

    paths = [Path(f"f_{f:04d}.jpg") for f in range(1, 101)]

    def image(path, *args):
        frame = shots.frame_number(path)
        value = 200 if 60 <= frame < 72 else 100 + frame % 2
        if 25 <= frame <= 27:
            value = 40 if frame % 2 else 180
        return np.full((180, 320), value, dtype=np.uint8)

    monkeypatch.setattr(shots.cv2, "imread", image)
    monkeypatch.setattr(shots, "_features", lambda path: None)
    monkeypatch.setattr(shots, "_registration_inliers", lambda *args: 100.0)
    original = shots.segment_shots(paths, fps=25, stride=1)
    boundaries = {
        int(original.sampled_frames[i]) for i in np.flatnonzero(np.diff(original.shot_id)) + 1
    }
    assert len(boundaries) >= 2
    seen = []

    def qualify(paths, frames, before, after, **kwargs):
        assert after in boundaries, "absorbed raw peak must not become a new boundary"
        seen.append(after)
        return {"continuous": after == min(boundaries)}

    monkeypatch.setattr(native, "qualify_native_continuity", qualify)
    refined = shots.segment_shots(paths, fps=25, stride=1, native_cut_continuity=True)
    remaining = {
        int(refined.sampled_frames[i]) for i in np.flatnonzero(np.diff(refined.shot_id)) + 1
    }
    assert set(seen) == boundaries
    assert remaining == boundaries - {min(boundaries)}


def test_duplicate_frame_ids_are_not_distinct_evidence(tmp_path):
    paths, frames = sequence(tmp_path)
    frames[11] = frames[10]
    assert not qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)[
        "continuous"
    ]


def test_actual_duplicate_picture_bytes_and_evidence_binding(tmp_path):
    paths, frames = sequence(tmp_path)
    receipt = tmp_path / "extraction_receipt.json"
    receipt.write_text('{"fps":25}')
    initial = qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)
    assert initial["continuous"]
    assert len(initial["native_picture_evidence"]) == 13
    assert initial["native_extraction_receipts"][0]["sha256"]
    paths[11].write_bytes(paths[10].read_bytes())
    repeated = qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)
    assert not repeated["continuous"]
    assert repeated["reason"] == "duplicate_or_static_native_steps"
    assert repeated["native_picture_evidence"] != initial["native_picture_evidence"]


def test_gradual_zoom_retains_positive_native_support(tmp_path):
    paths, frames = sequence(tmp_path)
    base = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    for frame, path in zip(frames, paths, strict=True):
        matrix = cv2.getRotationMatrix2D((160, 90), 0, 1 + frame * 0.003)
        image = cv2.warpAffine(base, matrix, (320, 180), borderMode=cv2.BORDER_REFLECT)
        assert cv2.imwrite(str(path), image)
    assert qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)["continuous"]


def test_local_motion_never_upgrades_original_held_camera_rows(monkeypatch):
    from cv.pipeline import native_shot_continuity as native, shot_segments as shots

    paths = [Path(f"f_{f:04d}.jpg") for f in range(1, 101)]

    def image(path, *args):
        frame = shots.frame_number(path)
        value = 200 if 60 <= frame < 72 else 100 + frame % 2
        if 25 <= frame <= 27:
            value = 40 if frame % 2 else 180
        return np.full((180, 320), value, dtype=np.uint8)

    monkeypatch.setattr(shots.cv2, "imread", image)
    monkeypatch.setattr(shots, "_features", lambda path: None)
    monkeypatch.setattr(
        shots,
        "_registration_inliers",
        lambda ref, path: 0.0 if shots.frame_number(path) < 50 else 100.0,
    )
    monkeypatch.setattr(native, "qualify_native_continuity", lambda *a, **k: {"continuous": True})
    original = shots.segment_shots(paths, fps=25, stride=1)
    assert original.is_play_camera.any() and not original.is_play_camera.all()
    refined = shots.segment_shots(paths, fps=25, stride=1, native_cut_continuity=True)
    np.testing.assert_array_equal(original.is_play_camera, refined.is_play_camera)
    np.testing.assert_array_equal(original.registration_inliers, refined.registration_inliers)
    assert any(
        r["reason"] == "original_camera_class_disagreement"
        for r in refined.boundary_continuity
        if not r["continuous"]
    )


@pytest.mark.parametrize("duration", [12, 16])
def test_long_cross_view_dissolve_preserves_boundary(tmp_path, duration):
    paths, frames = sequence(tmp_path)
    other = np.random.default_rng(7).integers(0, 256, (180, 320), dtype=np.uint8)
    for frame, path in zip(frames, paths, strict=True):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        alpha = np.clip((frame - 3) / duration, 0, 1)
        mixed = np.clip((1 - alpha) * image + alpha * other, 0, 255).astype(np.uint8)
        assert cv2.imwrite(str(path), mixed)
    result = qualify_native_continuity(paths, frames, 9, 13, registration_inliers=700)
    assert not result["continuous"]
    assert result["reason"] == "native_appearance_discontinuity"
