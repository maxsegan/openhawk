"""Unit tests for the noise-calibrated one-vs-two-flight model selection (GRIC / MDL family).

The incumbent decided every split/merge with FIXED chi^2 margins (SPLIT_PENALTY=60,
AUDIO/PHYS_SPLIT_PENALTY=8, FORCE_MERGE_FACTOR/MERGE_LOCAL_PX) that assume a single 2 px
observation noise everywhere — the broadcast-fragile tuning the advisor flagged. Those are
replaced by a criterion whose only free scale is a per-region observation-noise sigma
calibrated from clean single-flight arcs; a split earns its extra flight (9 params) + break
time only when the chi^2 improvement under that calibrated model exceeds the added-parameter
penalty. These tests pin the pieces: the noise model, the calibrated chi^2, the calibration,
the GRIC penalty, and the unified split-vs-merge decision.
"""
import math

import numpy as np

import physics_knot_solver as P
from physics_knot_solver import (
    DEFAULT_SIGMA_PX,
    FLIGHT_PARAMS,
    BREAK_PARAMS,
    NoiseModel,
    calibrate_noise,
    gric_penalty,
    region_of,
    seg_chi2,
    split_beats_merge,
)


class _FakeCamera:
    """Identity homography: image (u, v) maps to court (u, v), so v is the court-y used for the
    near/far region split (boundary NET_Y_COURT)."""

    def h_at(self, frame):
        return np.eye(3, dtype=np.float32)


# --------------------------------------------------------------------------- #
# NoiseModel + region_of
# --------------------------------------------------------------------------- #
def test_noise_model_sigma_array_selects_by_region() -> None:
    nm = NoiseModel(2.0, 5.0)
    reg = np.array([0, 1, 1, 0])
    assert list(nm.sigma_array(reg)) == [2.0, 5.0, 5.0, 2.0]


def test_region_of_splits_near_and_far_at_the_net() -> None:
    cam = _FakeCamera()
    # court-y (= v under identity H) below the net -> near (0); at/above -> far (1)
    uv = np.array([[5.0, P.NET_Y_COURT - 3.0], [5.0, P.NET_Y_COURT + 3.0],
                   [5.0, 0.0], [5.0, 23.0]])
    frames = np.array([10, 11, 12, 13])
    reg = region_of(uv, cam, frames)
    assert list(reg) == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
# seg_chi2 — calibrated, robust, with a fixed-sigma fallback
# --------------------------------------------------------------------------- #
def _fit(res_px, regions, ok=True):
    return {"ok": ok, "per_frame_px": np.array(res_px, float),
            "regions": np.array(regions, int)}


def test_seg_chi2_scales_inversely_with_sigma_squared() -> None:
    fit = _fit([2.0, 2.0, 2.0], [0, 0, 0])
    chi_s1 = seg_chi2(fit, NoiseModel(1.0, 1.0))
    chi_s2 = seg_chi2(fit, NoiseModel(2.0, 2.0))
    assert math.isclose(chi_s1, 12.0)         # sum (2/1)^2 * 3
    assert math.isclose(chi_s2, 3.0)          # sum (2/2)^2 * 3
    assert math.isclose(chi_s1 / chi_s2, 4.0)  # (sigma2/sigma1)^2


def test_seg_chi2_uses_per_region_sigma() -> None:
    fit = _fit([4.0, 4.0], [0, 1])            # one near obs, one far obs
    chi = seg_chi2(fit, NoiseModel(2.0, 4.0))
    assert math.isclose(chi, (4 / 2) ** 2 + (4 / 4) ** 2)  # 4 + 1


def test_seg_chi2_robust_clip_bounds_outliers() -> None:
    # A single wild residual contributes at most ROBUST_CLIP_SIGMA^2, not its raw square.
    fit = _fit([1000.0], [0])
    chi = seg_chi2(fit, NoiseModel(1.0, 1.0))
    assert math.isclose(chi, P.ROBUST_CLIP_SIGMA ** 2)


def test_seg_chi2_falls_back_to_cost_without_residual_arrays() -> None:
    # A hand-built fit (no per_frame_px/regions) uses its legacy `cost`.
    assert seg_chi2({"ok": True, "cost": 7.5}, NoiseModel(1.0, 1.0)) == 7.5
    assert seg_chi2({"ok": False}, NoiseModel(1.0, 1.0)) == math.inf


# --------------------------------------------------------------------------- #
# calibrate_noise
# --------------------------------------------------------------------------- #
def _clean_segment(sigma, region, n=80, seed=0):
    """A synthetic clean single-flight segment: 2-D Gaussian residuals of scale `sigma`."""
    rng = np.random.default_rng(seed)
    comp = rng.normal(0.0, sigma, size=(n, 2))
    per_px = np.hypot(comp[:, 0], comp[:, 1])
    return {"ok": True, "per_frame_px": per_px, "regions": np.full(n, region)}


def test_calibrate_recovers_per_region_sigma() -> None:
    # floor_px=0 to test the RAW measurement mechanism (the production floor is tested below).
    segs = [_clean_segment(1.5, 0, seed=1), _clean_segment(1.5, 0, seed=2),
            _clean_segment(4.0, 1, seed=3), _clean_segment(4.0, 1, seed=4)]
    nm = calibrate_noise(segs, floor_px=0.0)
    assert abs(nm.sigma_near - 1.5) < 0.3
    assert abs(nm.sigma_far - 4.0) < 0.6
    assert nm.sigma_far > nm.sigma_near
    assert nm.n_near > P.CALIB_MIN_OBS and nm.n_far > P.CALIB_MIN_OBS


def test_calibrate_floors_at_default() -> None:
    # Clean arcs measuring BELOW the floor are raised to it (clean-arc residual underestimates
    # the decision-relevant noise; never trust the data more than the incumbent's 2 px).
    segs = [_clean_segment(1.0, 0, seed=i) for i in range(3)]     # measured ~1.0 px
    nm = calibrate_noise(segs)                                     # default floor = 2.0 px
    assert nm.sigma_near == DEFAULT_SIGMA_PX
    # ... but a genuinely noisier region is kept above the floor.
    noisy = [_clean_segment(4.0, 1, seed=i) for i in range(3)]
    nm2 = calibrate_noise(noisy)
    assert nm2.sigma_far > DEFAULT_SIGMA_PX


def test_calibrate_falls_back_when_too_few_clean_obs() -> None:
    # A region with < CALIB_MIN_OBS clean residuals keeps the default sigma.
    segs = [_clean_segment(3.0, 1, n=80, seed=5)]   # only far populated
    nm = calibrate_noise(segs, floor_px=0.0)
    assert nm.sigma_near == DEFAULT_SIGMA_PX          # near unseen -> default
    assert abs(nm.sigma_far - 3.0) < 0.5


def test_calibrate_ignores_contaminated_segments() -> None:
    # A segment whose median residual exceeds CALIB_MAX_MED_PX (a mis-fit / multi-flight blob)
    # must not inflate the calibrated sigma.
    good = [_clean_segment(1.5, 0, seed=i) for i in range(3)]
    bad = {"ok": True, "regions": np.zeros(80, int),
           "per_frame_px": np.full(80, P.CALIB_MAX_MED_PX + 20.0)}
    nm = calibrate_noise(good + [bad], floor_px=0.0)
    assert abs(nm.sigma_near - 1.5) < 0.3            # contaminated segment excluded


def test_calibrate_skips_short_segments() -> None:
    short = {"ok": True, "regions": np.zeros(2 * P.MIN_SEG_OBS - 1, int),
             "per_frame_px": np.full(2 * P.MIN_SEG_OBS - 1, 1.5)}
    nm = calibrate_noise([short])
    assert nm.sigma_near == DEFAULT_SIGMA_PX and nm.sigma_far == DEFAULT_SIGMA_PX


# --------------------------------------------------------------------------- #
# gric_penalty — AIC vs BIC forms
# --------------------------------------------------------------------------- #
def test_gric_penalty_aic_is_two_per_parameter() -> None:
    assert gric_penalty(50, n_extra_params=10, lam=2.0) == 20.0


def test_gric_penalty_bic_scales_with_sample_size() -> None:
    # lam<=0 selects the BIC form ln(N)*params; larger N -> larger penalty.
    p_small = gric_penalty(20, n_extra_params=10, lam=0.0)
    p_big = gric_penalty(200, n_extra_params=10, lam=0.0)
    assert math.isclose(p_small, math.log(20) * 10)
    assert p_big > p_small


def test_default_extra_params_is_flight_plus_break() -> None:
    assert FLIGHT_PARAMS + BREAK_PARAMS == 10


# --------------------------------------------------------------------------- #
# split_beats_merge — the unified one-vs-two decision
# --------------------------------------------------------------------------- #
def test_phantom_split_of_a_clean_arc_does_not_beat_merge() -> None:
    # Splitting one clean flight into two frees ~10 params, absorbing only ~a few chi^2 of
    # noise — below the parameter penalty, so the split loses (merge wins).
    chi2_one = 50.0
    chi2_two = 44.0                     # a marginal, noise-level improvement
    margin = split_beats_merge(chi2_one, chi2_two, n_obs=60, lam=2.0)
    assert margin < 0                   # keep one flight


def test_real_break_beats_merge() -> None:
    # A true contact leaves a large one-flight residual; two flights improve chi^2 far beyond
    # the parameter penalty.
    margin = split_beats_merge(chi2_one=500.0, chi2_two=40.0, n_obs=60, lam=2.0)
    assert margin > 0                   # keep the split


def test_evidence_credit_lowers_the_bar_symmetrically() -> None:
    base = split_beats_merge(100.0, 85.0, n_obs=60, lam=2.0)
    with_credit = split_beats_merge(100.0, 85.0, n_obs=60, lam=2.0, credit=10.0)
    assert math.isclose(with_credit - base, 10.0)


def test_infeasible_debit_raises_the_bar() -> None:
    # A negative credit (the force-merge debit for an unsupported break) makes the same
    # improvement fail where it would otherwise pass.
    improvement_ok = split_beats_merge(100.0, 70.0, n_obs=60, lam=2.0)
    with_debit = split_beats_merge(100.0, 70.0, n_obs=60, lam=2.0, credit=-25.0)
    assert improvement_ok > 0 and with_debit < 0
