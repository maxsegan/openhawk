"""Batched, temporally coupled 2D-to-3D player-pose lifting around the court root.

Why not lift each joint on its full camera ray (``pose_lift``): on broadcast play cameras the
per-frame projection's vertical scale is off by 10-30% (it puts the camera too close), and for
the far player a camera ray is so shallow that a 0.2 m height error moves a joint metres in
depth. Bodies then lean or zigzag in depth although their 2D keypoints are right.

This lifter keeps the court-plane geometry the camera is good at and calibrates the rest:

* Local camera: per row, the 2x3 Jacobian ``J`` of the projection at the player's court root
  (ground point). Its x/y columns come from the court homography; its z column is scaled by one
  factor per player track (``vertical_scale``) so the player's standing nose height matches
  0.935 x stature (robust 75th percentile over the track).
* Variables: metric offsets of the 17 COCO joints from the root ``(root_x, root_y, 0)``.
* Terms: confidence-weighted reprojection of offsets through ``J`` (pixels, sigma scaled by
  the player's box size, so joints may leave their rays by a few pixels), stature-scaled limb
  lengths with left/right symmetry, ankles near the root and not below the court, a weak
  upright-torso prior, a weak forward (toward the net) prior on elbows/wrists/knees that picks
  the plausible side of each depth ambiguity, and second-difference smoothness between
  consecutive native frames of the same track.

Every output joint is FITTED. A joint without a confident 2D keypoint in its own frame is
filled only by the priors and its temporal neighbours; callers decide how to mark it.
All rows of a match are solved together (Adam on GPU when available).
"""

from __future__ import annotations

import numpy as np

from cv.pipeline.pose import KEYPOINT_NAMES

J = {name: k for k, name in enumerate(KEYPOINT_NAMES)}
#: Joint-to-joint distances as fractions of stature (pose_lift.LIMB_PRIORS plus the head).
BONES = (
    ("left_shoulder", "left_elbow", 0.186),
    ("right_shoulder", "right_elbow", 0.186),
    ("left_elbow", "left_wrist", 0.146),
    ("right_elbow", "right_wrist", 0.146),
    ("left_hip", "left_knee", 0.245),
    ("right_hip", "right_knee", 0.245),
    ("left_knee", "left_ankle", 0.246),
    ("right_knee", "right_ankle", 0.246),
    ("left_shoulder", "left_hip", 0.288),
    ("right_shoulder", "right_hip", 0.288),
    ("left_shoulder", "right_shoulder", 0.230),
    ("left_hip", "right_hip", 0.150),
    ("nose", "left_shoulder", 0.180),
    ("nose", "right_shoulder", 0.180),
    ("nose", "left_eye", 0.022),
    ("nose", "right_eye", 0.022),
    ("left_eye", "right_eye", 0.037),
    ("left_eye", "left_ear", 0.055),
    ("right_eye", "right_ear", 0.055),
    ("left_ear", "right_ear", 0.085),
)
NOSE_HEIGHT_FRACTION = 0.935
ANKLE_HEIGHT_M = 0.08
MIN_CONFIDENCE = 0.2
VERTICAL_SCALE_LIMITS = (0.5, 2.0)
REFINE_LOG_LIMIT = 0.3
FORWARD_JOINTS = (
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
)


def local_jacobians(projections: np.ndarray, roots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Root pixel (N,2) and d(pixel)/d(x,y,z) at the root ground point (N,2,3)."""
    n = len(roots)
    point = np.concatenate([roots, np.zeros((n, 1)), np.ones((n, 1))], axis=1)
    h = np.einsum("nij,nj->ni", projections, point)
    pixel = h[:, :2] / h[:, 2:3]
    # d(u)/dX = (P[0,:3] - u P[2,:3]) / w
    jac = (projections[:, :2, :3] - pixel[:, :, None] * projections[:, 2:3, :3]) / h[
        :, 2, None, None
    ]
    return pixel, jac


def vertical_scales(
    track: np.ndarray,
    jac: np.ndarray,
    offsets: np.ndarray,
    confidences: np.ndarray,
    stature_m: float,
) -> dict[int, float]:
    """Per track: factor on the z column making the standing nose 0.935 x stature tall.

    The nose height under the uncorrected camera is the least-squares height of a vertical
    offset reproducing the observed nose pixel offset from the root; the 75th percentile over
    the track stands for its standing posture.
    """
    nose = J["nose"]
    column = jac[:, :, 2]
    norm = np.einsum("ni,ni->n", column, column)
    heights = np.einsum("ni,ni->n", offsets[:, nose], column) / np.maximum(norm, 1e-9)
    usable = (confidences[:, nose] >= 0.5) & np.isfinite(heights) & (heights > 0.3)
    scales = {}
    for value in np.unique(track):
        mask = usable & (track == value)
        if mask.sum() < 5:
            scales[int(value)] = 1.0
            continue
        standing = float(np.percentile(heights[mask], 75))
        scales[int(value)] = standing / (NOSE_HEIGHT_FRACTION * stature_m)
    return scales


def lift(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    boxes: np.ndarray,
    roots: np.ndarray,
    projections: np.ndarray,
    track: np.ndarray,
    frames: np.ndarray,
    forward_y: np.ndarray,
    stature_m: float,
    *,
    fps: float = 50.0,
    iterations: int = 400,
    device: str | None = None,
) -> dict:
    """Lift N rows (all tracks of a match) together.

    keypoints (N,17,2) native px, confidences (N,17), boxes (N,4) native px, roots (N,2) court m,
    projections (N,3,4), track (N,) int id per (clip, side) run source, frames (N,) native frame,
    forward_y (N,) +1 when the net is toward +y from this player (near side), else -1.
    Returns joints (N,17,3) court metres, reprojection_px (N,17), vertical_scale (N,).
    """
    import torch

    n = len(keypoints)
    if n == 0:
        return {
            "joints": np.zeros((0, 17, 3)),
            "reprojection_px": np.zeros((0, 17)),
            "vertical_scale": np.zeros(0),
            "scale_plausible": np.zeros(0, dtype=bool),
            "scales": {},
            "raw_scales": {},
        }
    root_pixel, jac = local_jacobians(projections.astype(float), roots.astype(float))
    offsets = keypoints.astype(float) - root_pixel[:, None, :]
    conf = np.where(np.isfinite(offsets).all(axis=2), confidences, 0.0)
    conf = np.where(conf >= MIN_CONFIDENCE, conf, 0.0)
    offsets = np.nan_to_num(offsets)
    raw_scales = vertical_scales(track, jac, offsets, conf, stature_m)
    scales = {t: float(np.clip(v, *VERTICAL_SCALE_LIMITS)) for t, v in raw_scales.items()}
    row_scale = np.asarray([scales[int(t)] for t in track])
    # Outside the limits the camera's vertical geometry is not plausible for this track.
    plausible = np.asarray(
        [VERTICAL_SCALE_LIMITS[0] <= raw_scales[int(t)] <= VERTICAL_SCALE_LIMITS[1] for t in track]
    )
    jac = jac.copy()
    jac[:, :, 2] *= row_scale[:, None]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
    jac_t, off_t, conf_t = T(jac), T(offsets), T(conf)
    box_h = np.maximum(boxes[:, 3] - boxes[:, 1], 8.0)
    # Pixel sigma: 2.5% of the player's box height (about 4 cm on the body), at least 1.5 px.
    sigma_px = T(np.maximum(0.025 * box_h, 1.5))[:, None]
    forward = T(forward_y)

    # Initial offsets: the planar (no depth) solution of J o = delta with o_y tied to zero.
    init = np.zeros((n, 17, 3))
    a = jac[:, :, [0, 2]]  # x and z columns
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    inv = (
        np.stack(
            [np.stack([a[:, 1, 1], -a[:, 0, 1]], 1), np.stack([-a[:, 1, 0], a[:, 0, 0]], 1)], 1
        )
        / det[:, None, None]
    )
    xz = np.einsum("nij,nkj->nki", inv, offsets)
    init[:, :, 0], init[:, :, 2] = xz[:, :, 0], xz[:, :, 1]
    # Unseen joints start at a neutral standing template.
    template = np.broadcast_to(_template(stature_m), init.shape).copy()
    template[:, :, 0] *= np.asarray(forward_y, dtype=float)[:, None]
    unseen = conf <= 0
    init[unseen] = template[unseen]
    init[:, :, 2] = np.clip(init[:, :, 2], 0.0, 1.2 * stature_m)
    params = torch.nn.Parameter(T(init))
    # Per-track refinement of the vertical scale (log factor on top of the stature estimate),
    # identified by the vertical limbs across the track's frames.
    track_values, track_index = np.unique(np.asarray(track), return_inverse=True)
    free_scale = torch.nn.Parameter(torch.zeros(len(track_values), dtype=dtype, device=device))
    track_t = torch.as_tensor(track_index, device=device)
    track_rows = T(np.bincount(track_index, minlength=len(track_values)))

    def log_scale_of():
        # Bounded refinement: at most a factor exp(+-REFINE_LOG_LIMIT) on the stature estimate.
        return REFINE_LOG_LIMIT * torch.tanh(free_scale)

    # Temporal neighbours: the previous/next row of the same track at frame -/+ 1.
    track_np, frame_np = np.asarray(track), np.asarray(frames)
    has_prev = np.zeros(n, bool)
    has_prev[1:] = (track_np[1:] == track_np[:-1]) & (frame_np[1:] == frame_np[:-1] + 1)
    has_next = np.zeros(n, bool)
    has_next[:-1] = has_prev[1:]
    middle = np.flatnonzero(has_prev & has_next)
    first_diff = np.flatnonzero(has_prev)
    mid_t = torch.as_tensor(middle, device=device)
    fd_t = torch.as_tensor(first_diff, device=device)
    # Smoothness in metres per frame^2, scaled to 50 fps.
    rate = (fps / 50.0) if fps > 0 else 1.0
    sigma_acc = 0.035 / rate**2
    sigma_vel = 0.25 / rate

    bones_a = torch.as_tensor([J[a] for a, _, _ in BONES], device=device)
    bones_b = torch.as_tensor([J[b] for _, b, _ in BONES], device=device)
    bones_l = T([length * stature_m for _, _, length in BONES])
    sigma_bone = 0.03 * stature_m
    ankles = torch.as_tensor([J["left_ankle"], J["right_ankle"]], device=device)
    hips = torch.as_tensor([J["left_hip"], J["right_hip"]], device=device)
    shoulders = torch.as_tensor([J["left_shoulder"], J["right_shoulder"]], device=device)
    fwd = torch.as_tensor([J[name] for name in FORWARD_JOINTS], device=device)
    nose = J["nose"]

    def project(o):
        factor = torch.exp(log_scale_of())[track_t]
        jz = jac_t[:, :, 2] * factor[:, None]
        return (
            torch.einsum("nij,nkj->nki", jac_t[:, :, :2], o[:, :, :2])
            + jz[:, None, :] * o[:, :, 2:3]
        )

    def loss_fn(o):
        projected = project(o)
        reproj = (((projected - off_t) / sigma_px[:, :, None]) ** 2).sum(-1)
        loss = (conf_t * reproj).sum()
        lengths = (o[:, bones_a] - o[:, bones_b]).norm(dim=-1)
        loss = loss + (((lengths - bones_l) / sigma_bone) ** 2).sum()
        # Left/right symmetric limb lengths within each frame.
        pairs = lengths[:, 0:10].reshape(n, 5, 2)
        loss = loss + (((pairs[:, :, 0] - pairs[:, :, 1]) / (0.5 * sigma_bone)) ** 2).sum()
        ankle = o[:, ankles]
        loss = loss + ((ankle[:, :, :2].mean(1) / 0.30) ** 2).sum()
        loss = loss + ((torch.relu(ANKLE_HEIGHT_M - 0.03 - ankle[:, :, 2]) / 0.01) ** 2).sum()
        lowest = ankle[:, :, 2].min(dim=1).values
        loss = loss + (((lowest - ANKLE_HEIGHT_M) / 0.12) ** 2).sum()
        hip_c = o[:, hips].mean(1)
        shoulder_c = o[:, shoulders].mean(1)
        loss = loss + ((hip_c[:, :2] / 0.35) ** 2).sum()
        lean = shoulder_c[:, :2] - hip_c[:, :2]
        loss = loss + ((lean / 0.18) ** 2).sum()
        loss = (
            loss
            + ((torch.relu(hip_c[:, 2] + 0.25 * stature_m - shoulder_c[:, 2]) / 0.03) ** 2).sum()
        )
        loss = loss + ((torch.relu(shoulder_c[:, 2] - o[:, nose, 2]) / 0.03) ** 2).sum()
        loss = loss + ((torch.relu(-o[:, :, 2]) / 0.02) ** 2).sum()
        # Forward prior: elbows, wrists and knees in front of the hip centre (toward the net).
        ahead = (o[:, fwd, 1] - hip_c[:, None, 1]) * forward[:, None]
        loss = loss + ((torch.relu(-ahead) / 0.25) ** 2).sum()
        if len(middle):
            acc = o[mid_t + 1] - 2 * o[mid_t] + o[mid_t - 1]
            loss = loss + ((acc / sigma_acc) ** 2).sum()
        if len(first_diff):
            vel = o[fd_t] - o[fd_t - 1]
            loss = loss + ((vel / sigma_vel) ** 2).sum()
        # Per-row prior on the track's refinement, so a long track is not freer than a short one.
        loss = loss + (track_rows * (log_scale_of() / 0.10) ** 2).sum()
        return loss / n

    optimizer = torch.optim.Adam([params, free_scale], lr=0.03)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=0.002)
    history = []
    for step in range(iterations):
        optimizer.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 50 == 0 or step == iterations - 1:
            history.append(round(float(loss.detach()), 4))
    with torch.no_grad():
        o = params.detach()
        projected = project(o)
        refined = torch.exp(log_scale_of()).cpu().numpy()
        reprojection = (projected - off_t).norm(dim=-1).cpu().numpy()
        local = o.cpu().numpy().astype(float)
    joints = local.copy()
    joints[:, :, 0] += roots[:, 0:1]
    joints[:, :, 1] += roots[:, 1:2]
    reprojection[conf <= 0] = np.nan
    row_scale = row_scale * refined[track_index]
    scales = {int(t): float(scales[int(t)] * refined[k]) for k, t in enumerate(track_values)}
    return {
        "joints": joints,
        "reprojection_px": reprojection,
        "vertical_scale": row_scale,
        "scale_plausible": plausible,
        "scales": scales,
        "raw_scales": {int(t): round(float(v), 3) for t, v in raw_scales.items()},
        "loss_history": history,
        "device": str(device),
    }


def _template(stature_m: float) -> np.ndarray:
    """A neutral standing pose (root-relative metres) for joints never seen."""
    s = stature_m
    t = np.zeros((17, 3))
    rows = {
        "nose": (0.0, 0.03, 0.935),
        "left_eye": (-0.02, 0.02, 0.945),
        "right_eye": (0.02, 0.02, 0.945),
        "left_ear": (-0.045, 0.0, 0.935),
        "right_ear": (0.045, 0.0, 0.935),
        "left_shoulder": (-0.115, 0.0, 0.82),
        "right_shoulder": (0.115, 0.0, 0.82),
        "left_elbow": (-0.15, 0.03, 0.64),
        "right_elbow": (0.15, 0.03, 0.64),
        "left_wrist": (-0.14, 0.08, 0.50),
        "right_wrist": (0.14, 0.08, 0.50),
        "left_hip": (-0.075, 0.0, 0.53),
        "right_hip": (0.075, 0.0, 0.53),
        "left_knee": (-0.08, 0.03, 0.29),
        "right_knee": (0.08, 0.03, 0.29),
        "left_ankle": (-0.09, 0.0, 0.045),
        "right_ankle": (0.09, 0.0, 0.045),
    }
    for name, (x, y, z) in rows.items():
        t[J[name]] = (x * s, y * s, z * s)
    return t
