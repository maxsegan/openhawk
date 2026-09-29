"""Full-match racket-impact audio peaks -> ``serve_audio_peaks_v1.csv``.

The window-map rebuild (align.py v2, REPORT.md section 3) needs impact evidence at
ABSOLUTE video times, including inside play segments that were never extracted as rally
clips (the per-clip audio caches only cover frozen v1 windows). This stage computes the
same racket-impact-band positive spectral flux as contacts_audio.py (1800-7000 Hz,
20 ms windows / 10 ms hop at 16 kHz), but over the WHOLE broadcast in one pass, with the
robust-z normalization taken against full-match statistics so that a transient in an
otherwise quiet between-points lull cannot self-normalize into a fake peak (the per-clip
normalization would allow that).

Peaks (local maxima above --z-min, minimum separation --min-sep) are written with their
z-scores; consumers apply their own threshold, so re-tuning needs no recompute. Two uses:

- align.py v2 content validation: a candidate rally segment must contain an impact peak
  (serve anchor / ball-strike evidence) or the selector falls through to the next
  overlapping segment (fixes rg pt0329, where the chosen segment shows a ball kid).
- ``broadcast_serve_offcamera`` flagging: a window whose serve is audible before the play
  camera cuts in (rg pt0125 class) has its first peak before the rally segment start.

    .venv/bin/python cv/pipeline/serve_audio.py \
        --video MATCH.mp4 --out data/processed/rg2025f
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contacts_audio import FFT_SAMPLES, HOP_SAMPLES, SAMPLE_RATE, read_audio  # noqa: E402
from run_manifest import StageRun  # noqa: E402

CHUNK_S = 600.0  # streamed decode chunk; overlapped by one FFT window for continuity


def band_flux(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Positive spectral flux in the racket-impact band (contacts_audio band), un-normalized.

    Vectorized equivalent of contacts_audio.onset_scores before its robust-z step.
    """
    if len(samples) < 2 * FFT_SAMPLES:
        return np.empty(0, dtype=np.float32)
    window = np.hanning(FFT_SAMPLES)
    n = (len(samples) - FFT_SAMPLES) // HOP_SAMPLES + 1
    idx = np.arange(FFT_SAMPLES)[None, :] + HOP_SAMPLES * np.arange(n)[:, None]
    spectra = np.abs(np.fft.rfft(samples[idx] * window[None, :], axis=1))
    frequencies = np.fft.rfftfreq(FFT_SAMPLES, 1.0 / sample_rate)
    band = (frequencies >= 1800.0) & (frequencies <= 7000.0)
    flux = np.maximum(np.diff(spectra[:, band], axis=0), 0.0).sum(axis=1)
    return np.r_[0.0, flux].astype(np.float32)


def full_match_flux(video: str, duration: float | None) -> np.ndarray:
    """Stream the match audio in chunks; concatenate hop-aligned flux frames."""
    chunks: list[np.ndarray] = []
    start = 0.0
    while True:
        span = CHUNK_S if duration is None else min(CHUNK_S, duration - start)
        if span <= 0:
            break
        # one extra FFT window of lead-in so the first diff frame of the chunk is real
        lead = FFT_SAMPLES / SAMPLE_RATE if start > 0 else 0.0
        samples = read_audio(video, start - lead, span + lead, SAMPLE_RATE)
        if len(samples) < 2 * FFT_SAMPLES:
            break
        flux = band_flux(samples)
        if start > 0:
            skip = int(round(lead * SAMPLE_RATE / HOP_SAMPLES))
            flux = flux[skip:]
        chunks.append(flux)
        got_s = len(samples) / SAMPLE_RATE - lead
        start += got_s
        if got_s < span - 1.0:  # ffmpeg returned short: end of stream
            break
        if duration is None and got_s < CHUNK_S - 1.0:
            break
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def robust_z(flux: np.ndarray) -> np.ndarray:
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    return (flux - median) / max(1.4826 * mad, 1e-9)


def peaks(z: np.ndarray, z_min: float, min_sep_s: float) -> list[tuple[float, float]]:
    hop_s = HOP_SAMPLES / SAMPLE_RATE
    sep = max(1, int(round(min_sep_s / hop_s)))
    above = np.where(z >= z_min)[0]
    out: list[tuple[float, float]] = []
    # greedy max-first suppression
    order = above[np.argsort(-z[above])]
    taken = np.zeros(len(z), dtype=bool)
    for i in order:
        if taken[max(0, i - sep) : i + sep + 1].any():
            continue
        taken[i] = True
        out.append((i * hop_s, float(z[i])))
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=None,
                    help="limit (seconds); default: full stream")
    ap.add_argument("--z-min", type=float, default=3.0,
                    help="write peaks at or above this full-match robust z")
    ap.add_argument("--min-sep", type=float, default=0.2)
    ap.add_argument("--output", default="serve_audio_peaks_v1.csv")
    ap.add_argument("--flux-cache", default="serve_audio_flux_v1.npz")
    args = ap.parse_args()
    stage = StageRun(args.out, "serve_audio_peaks_v1", config=args)

    cache_path = os.path.join(args.out, args.flux_cache)
    if os.path.exists(cache_path):
        flux = np.load(cache_path)["flux"]
        print(f"reusing cached flux ({len(flux)} frames) from {cache_path}")
    else:
        flux = full_match_flux(args.video, args.duration)
        np.savez_compressed(cache_path, flux=flux,
                            hop_s=HOP_SAMPLES / SAMPLE_RATE, video=os.path.basename(args.video))
        print(f"flux: {len(flux)} frames ({len(flux) * HOP_SAMPLES / SAMPLE_RATE:.0f} s) "
              f"-> {cache_path}")
    z = robust_z(flux)
    found = peaks(z, args.z_min, args.min_sep)
    out_csv = os.path.join(args.out, args.output)
    with open(out_csv, "w", newline="") as f:
        f.write("t,z\n")
        for t, zv in found:
            f.write(f"{t:.2f},{zv:.2f}\n")
    print(f"{len(found)} peaks (z >= {args.z_min}) -> {out_csv}")
    stage.finish(outputs={"frames": int(len(flux)), "peaks": len(found)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
