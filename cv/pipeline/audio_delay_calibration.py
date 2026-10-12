"""Per-match audio/video delay calibration from label-free anchors.

Broadcast audio is not aligned with the pictures by a fixed amount: per-source constants on the
opened panels ran from -1.0 to +5.4 native frames. Anything that times a contact from the
contact audio score (``contact_audio_scores_16k_native_v1.npz``, 100 Hz) needs the match's own
delay first. This estimates it once per match, without labels:

- anchors: the automatic event model's confident contacts (emitted, not held, contact
  probability >= ``CONFIDENT_P``) and, optionally, image-space ball-direction reversals next to
  a player (their hinge break time) that are not near an automatic contact or bounce;
- per anchor: the strongest audio onset peak (robust z >= ``PEAK_Z``) within
  +-``SEARCH_S`` seconds; its offset in native frames (positive: audio late);
- the delay: the kernel-density mode of the offsets; confidence from the share of offsets
  within 1 frame of it and a seeded bootstrap of the mode.

``confident`` estimates give one delay. Uncertain ones keep the mode plus a second supported
mode (>= 2 frames away) so callers can branch on both; with too few anchors the delay is the
prior (0) with a wide spread. Nothing here reads a label.
"""

from __future__ import annotations

import numpy as np

SCHEMA = "audio_delay_calibration_v1"
AUDIO_HZ = 100.0
CONFIDENT_P = 0.9
PEAK_Z = 3.0
SEARCH_S = 0.25
BANDWIDTH = 0.6  # frames
MIN_ANCHORS = 8
MIN_SUPPORT = 0.4
MIN_STABILITY = 0.8
SECOND_MODE_GAP = 2.0
SECOND_MODE_SUPPORT = 0.2
BOOTSTRAP = 200


def robust_z(trace: np.ndarray) -> np.ndarray:
    med = float(np.median(trace))
    mad = float(np.median(np.abs(trace - med))) or 1e-6
    return (trace - med) / (1.4826 * mad)


def frame_of_sample(i: float, fps: float) -> float:
    return i / AUDIO_HZ * fps + 1.0


def sample_of_frame(f: float, fps: float) -> int:
    return int(round((f - 1.0) / fps * AUDIO_HZ))


def anchor_offset(z: np.ndarray, fps: float, frame: float) -> float | None:
    """Frames from the anchor to the strongest onset peak (z >= PEAK_Z) within SEARCH_S."""
    i = sample_of_frame(frame, fps)
    half = int(round(SEARCH_S * AUDIO_HZ))
    lo, hi = max(1, i - half), min(len(z) - 1, i + half + 1)
    if hi - lo < 3:
        return None
    seg = z[lo:hi]
    peaks = [
        k
        for k in range(1, len(seg) - 1)
        if seg[k] >= PEAK_Z and seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1]
    ]
    if not peaks:
        return None
    k = max(peaks, key=lambda p: seg[p])
    return frame_of_sample(lo + k, fps) - frame


def _density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((grid[:, None] - x[None, :]) / BANDWIDTH) ** 2).sum(1)


def _mode(x: np.ndarray, grid: np.ndarray) -> float:
    return float(grid[int(np.argmax(_density(x, grid)))])


def estimate(offsets: list[float], seed: int = 0) -> dict:
    """Delay (frames, audio minus video) with its confidence and the lag modes to try."""
    x = np.asarray([o for o in offsets if o is not None], float)
    n = int(x.size)
    if n < 3:
        return {
            "schema": SCHEMA,
            "delay_frames": 0.0,
            "modes": [0.0],
            "spread": 4.0,
            "n": n,
            "support": 0.0,
            "stability": 0.0,
            "confident": False,
            "reason": "too_few_anchors",
        }
    grid = np.arange(np.floor(x.min()) - 1.0, np.ceil(x.max()) + 1.0 + 1e-9, 0.1)
    mode = _mode(x, grid)
    support = float(np.mean(np.abs(x - mode) <= 1.0))
    rng = np.random.default_rng(seed)
    boots = np.array([_mode(rng.choice(x, n), grid) for _ in range(BOOTSTRAP)])
    stability = float(np.mean(np.abs(boots - mode) <= 1.0))
    near = np.abs(x - mode) <= 2.0
    spread = float(1.4826 * np.median(np.abs(x[near] - mode))) if near.any() else 4.0
    confident = n >= MIN_ANCHORS and support >= MIN_SUPPORT and stability >= MIN_STABILITY
    modes, weights = [mode], [support]
    reason = "confident"
    if not confident:
        reason = "few_anchors" if n < MIN_ANCHORS else "unstable_mode"
        dens = _density(x, grid)
        far = np.abs(grid - mode) >= SECOND_MODE_GAP
        if far.any():
            k = int(np.argmax(np.where(far, dens, -1.0)))
            second = float(grid[k])
            share = float(np.mean(np.abs(x - second) <= 1.0))
            if share >= SECOND_MODE_SUPPORT:
                modes.append(second)
                weights.append(share)
    return {
        "schema": SCHEMA,
        "delay_frames": round(mode, 2),
        "modes": [round(m, 2) for m in modes],
        "weights": [round(v, 3) for v in weights],
        "spread": round(max(spread, 0.3), 2),
        "n": n,
        "support": round(support, 3),
        "stability": round(stability, 3),
        "bootstrap_p10_p90": [
            round(float(np.percentile(boots, 10)), 2),
            round(float(np.percentile(boots, 90)), 2),
        ],
        "confident": bool(confident),
        "reason": reason,
    }


def calibrate(
    traces: dict[str, np.ndarray],
    fps_by_clip: dict[str, float],
    anchors: list[dict],
    seed: int = 0,
) -> dict:
    """One match: ``anchors`` rows ``{clip, frame, kind}`` (native frames, 1-based)."""
    zs = {clip: robust_z(np.asarray(t, float)) for clip, t in traces.items()}
    offsets, kinds = [], {}
    for a in anchors:
        z = zs.get(a["clip"])
        if z is None:
            continue
        o = anchor_offset(z, fps_by_clip[a["clip"]], float(a["frame"]))
        kinds.setdefault(a["kind"], [0, 0])[0] += 1
        if o is None:
            continue
        kinds[a["kind"]][1] += 1
        offsets.append(o)
    out = estimate(offsets, seed)
    out["anchors"] = {k: {"offered": v[0], "with_peak": v[1]} for k, v in sorted(kinds.items())}
    out["clips"] = len(traces)
    out["search_s"] = SEARCH_S
    return out
