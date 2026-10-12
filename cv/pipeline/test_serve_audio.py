import numpy as np

from cv.pipeline import serve_audio


def _fake_reader(total_s: float, hole: tuple[float, float]):
    """Decoder over a ``total_s`` stream with an audio packet hole; the filter fills it."""

    def read(video, start, duration, sample_rate, audio_filter=None):
        end = min(start + duration, total_s)
        seconds = max(0.0, end - start)
        if audio_filter is None:
            overlap = max(0.0, min(end, hole[1]) - max(start, hole[0]))
            seconds -= overlap
        return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)

    return read


def test_audio_hole_is_filled_and_scan_continues(monkeypatch):
    monkeypatch.setattr(serve_audio, "read_audio", _fake_reader(2000.0, (1332.0, 1338.0)))
    filled: list = []
    flux = serve_audio.full_match_flux("v.mp4", None, "fill", filled)
    hop_s = serve_audio.HOP_SAMPLES / serve_audio.SAMPLE_RATE
    assert abs(len(flux) * hop_s - 2000.0) < 0.1
    assert len(filled) == 1 and filled[0][0] > 1199.0 and abs(filled[0][2] - 600.0) < 0.1


def test_stop_policy_keeps_old_truncation(monkeypatch):
    monkeypatch.setattr(serve_audio, "read_audio", _fake_reader(2000.0, (1332.0, 1338.0)))
    flux = serve_audio.full_match_flux("v.mp4", None, "stop")
    hop_s = serve_audio.HOP_SAMPLES / serve_audio.SAMPLE_RATE
    assert abs(len(flux) * hop_s - 1794.0) < 0.1
