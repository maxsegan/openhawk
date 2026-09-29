"""Optional robust native-pixel residuals for the labeled prefix joint impact fit.

Question: can a soft-L1 native image block stop a few corrupted prefix pixel rows from
dragging the joint contact/launch fit, while every physical evidence term (toss, nets,
athlete priors, spin, rebound and normal priors) stays quadratic so a wrong event is
never quietly downweighted as an outlier?

Usage: `labeled_prefix_joint_impact.fit(..., pixel_loss="soft_l1")`. The default "off"
returns the caller's residual object unchanged, so the legacy objective is exact.

The transform is the standard least-squares reparameterization of a robust loss: the
plain sum of squares of the transformed residual reproduces the soft-L1 cost that this
package already spells as `event_constraints.mixed_loss` at C=2, i.e. for a native
residual r (pixels),

    0.5 * transform(r)**2 == C**2 * (sqrt(1 + (r / C)**2) - 1)   with C = SCALE_PX = 2.

Because the transform lives on the residual vector rather than in one solver's loss
argument, every consumer of that vector -- initialization ranking, source-incumbent
retention, regime retries, SLSQP, the fast-local incumbent certification and the full
certification replay -- optimizes and ranks the same objective.

The `r * sqrt(2 / (hypot(1, r / C) + 1))` form is algebraically equal to the direct
`sign(r) * C * sqrt(2 * (sqrt(1 + (r / C)**2) - 1))` but is evaluated without a
cancelling subtraction, so it keeps full precision and the correct unit derivative for
the small residuals that dominate a converged fit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MODES = ("off", "soft_l1")
SCALE_PX = 2.0  # C, in native pixels; matches event_constraints.mixed_loss


def validate(mode: str) -> str:
    """Reject anything but an explicit supported mode. Call before any fitting work."""
    if type(mode) is not str or mode not in MODES:
        raise ValueError(f"pixel_loss must be one of {MODES}, got {mode!r}")
    return mode


def transform(residual: Sequence[float] | np.ndarray) -> np.ndarray:
    """Scale native residuals so a plain sum of squares is the soft-L1 cost at C=2."""
    r = np.asarray(residual, dtype=float)
    return r * np.sqrt(2.0 / (np.hypot(1.0, r / SCALE_PX) + 1.0))


def apply(mode: str, residual):
    """Return the native residual block under `mode`; "off" is the identity."""
    if validate(mode) == "off":
        return residual
    return transform(residual)


def cost(mode: str, residual) -> float:
    """Objective contribution of the native block, in the fit's own `r @ r` units."""
    block = np.asarray(apply(mode, residual), dtype=float)
    return float(block @ block)
