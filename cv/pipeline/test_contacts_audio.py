import os

import numpy as np

import contacts_audio
from contacts_audio import load_or_compute_audio_scores, onset_frames, onset_scores


def test_onset_scores_find_sharp_high_frequency_impacts():
    sample_rate = 16_000
    samples = np.zeros(sample_rate, dtype=np.float32)
    rng = np.random.default_rng(4)
    for center in (4_000, 12_000):
        samples[center : center + 80] = rng.choice([-1.0, 1.0], size=80)
    scores = onset_scores(samples, sample_rate)
    frames = onset_frames(scores, 5.0, 50.0, sample_rate, 320, 160, 0.2, 0.0)
    assert len(frames) == 2
    assert [frame for frame, _ in frames] == [13, 39]


def test_audio_delay_is_cadence_invariant():
    scores = np.zeros(101, dtype=np.float32)
    scores[50] = 10.0

    frame_25 = onset_frames(scores, 5.0, 25.0, 16_000, 320, 160, 0.2, 0.08)[0][0]
    frame_50 = onset_frames(scores, 5.0, 50.0, 16_000, 320, 160, 0.2, 0.08)[0][0]

    assert frame_25 == 12
    assert frame_50 == 23


def test_audio_cache_reuses_matching_windows_and_recomputes_changed_window(
    tmp_path, monkeypatch
) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"video identity")
    cache = tmp_path / "scores.npz"
    calls = []

    def fake_read_audio(path, start, duration, sample_rate):
        calls.append((path, start, duration, sample_rate))
        return np.full(1_000, start + duration, dtype=np.float32)

    monkeypatch.setattr(contacts_audio, "read_audio", fake_read_audio)
    monkeypatch.setattr(
        contacts_audio,
        "onset_scores",
        lambda samples, sample_rate=16_000: np.array([samples[0]], dtype=np.float32),
    )

    windows = {"pt0001": (1.0, 2.0), "pt0002": (3.0, 4.0)}
    first, first_report = load_or_compute_audio_scores(
        os.fspath(video), windows, 16_000, os.fspath(cache)
    )
    second, second_report = load_or_compute_audio_scores(
        os.fspath(video), windows, 16_000, os.fspath(cache)
    )
    changed, changed_report = load_or_compute_audio_scores(
        os.fspath(video), {**windows, "pt0002": (3.0, 4.5)}, 16_000, os.fspath(cache)
    )

    assert first_report["cache_misses"] == 2
    assert second_report["cache_hits"] == 2
    assert changed_report == {
        "cache_path": os.fspath(cache),
        "cache_hits": 1,
        "cache_misses": 1,
    }
    assert len(calls) == 3
    np.testing.assert_array_equal(first["pt0001"], second["pt0001"])
    assert changed["pt0002"].item() == 4.5
