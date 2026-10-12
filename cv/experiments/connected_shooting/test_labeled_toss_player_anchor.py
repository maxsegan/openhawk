"""Anchor geometry, declared proxies, robust pooling, abstentions and residual shape."""

from copy import deepcopy

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_toss_player_anchor as anchor

SPACE_NATIVE = dict(
    schema="tennis.coordinate-space.v1",
    image_size=dict(width=1920, height=1080),
    artifact_size=dict(width=1920, height=1080),
)
SPACE_HALF = dict(
    schema="tennis.coordinate-space.v1",
    image_size=dict(width=1920, height=1080),
    artifact_size=dict(width=960, height=540),
)
PLAYER = dict(player="server", side="far", stature_m=1.85, court_centre_xy_m=[5.3, 24.2])


def camera(centre=(5.5, -35.0, 12.0), target=(5.5, 24.0, 0.0), focal=4600.0):
    """Broadcast-like pinhole behind the near baseline looking at the far baseline."""
    c, t = np.asarray(centre, float), np.asarray(target, float)
    forward = t - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    K = np.array([[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]])
    return K @ np.hstack([R, -R @ c[:, None]])


def project(P, xyz):
    h = P @ np.r_[np.asarray(xyz, float), 1.0]
    return h[:2] / h[2]


def rows_and_cameras(frames, P=None, window=7):
    P = camera() if P is None else P
    rows = [
        dict(frame=float(f), pixel=[900.0, 300.0], uncertainty_px=3.0, camera=P.tolist())
        for f in frames
    ]
    cameras = dict(
        clip="pt0001",
        cameras=[
            dict(
                frame=int(f),
                status="supported",
                P=P.tolist(),
                local_anchor_frame=window,
                source="registered",
            )
            for f in frames
        ],
    )
    return rows, cameras


def keypoint_row(frame, feet_xy, P, scale=1.0, conf=0.9, ankle_conf=0.95, side="far", z=0.08):
    uv = project(P, [feet_xy[0], feet_xy[1], z]) / scale
    row = dict(clip="pt0001", frame=f"f_{int(frame):04d}.jpg", side=side, conf=str(conf))
    row.update(x0=str(uv[0] - 20), x1=str(uv[0] + 20), y0=str(uv[1] - 150), y1=str(uv[1] + 6))
    for joint in ("left_ankle", "right_ankle"):
        row[f"{joint}_x"], row[f"{joint}_y"] = str(uv[0]), str(uv[1])
        row[f"{joint}_confidence"] = str(ankle_conf)
    return row


def box_row(frame, feet_xy, P, scale=2.0, conf=0.7, side="far"):
    uv = project(P, [feet_xy[0], feet_xy[1], 0.0]) / scale
    return dict(
        clip="pt0001",
        frame=f"f_{int(frame):04d}.jpg",
        side=side,
        conf=str(conf),
        x0=str(uv[0] - 10),
        x1=str(uv[0] + 10),
        y0=str(uv[1] - 60),
        y1=str(uv[1]),
    )


def test_ground_point_inverts_projection_and_rejects_rays_behind_the_camera():
    P = camera()
    for xy, z in (((5.3, 24.2), 0.0), ((2.0, 23.5), 0.08), ((9.0, 25.0), 0.0)):
        uv = project(P, [*xy, z])
        np.testing.assert_allclose(anchor.ground_point(P, uv, z), xy, atol=1e-6)
    J, g = anchor.ground_jacobians(P, project(P, [5.3, 24.2, 0.0]), 0.0)
    assert J.shape == (2, 2) and g.shape == (2,)
    # Far-end depth is the weakly observed axis: more metres per pixel and per proxy metre.
    assert abs(J[1, 1]) > abs(J[0, 0]) and abs(g[1]) > 3.0
    with pytest.raises(ValueError, match="behind"):
        anchor.ground_point(P, project(P, [5.0, -40.0, 0.0]), 0.0)


def test_keypoint_and_box_artifacts_choose_the_declared_proxy_and_sigma():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    P = camera()
    pose = [keypoint_row(f, (5.3, 24.4), P) for f in frames]
    ankles = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert ankles["status"] == "supported"
    (window,) = ankles["windows"]
    assert window["anchor_kind"] == "ankle_midpoint"
    assert window["sigma_px"] == 6.0 and window["proxy_height_m"] == 0.08
    np.testing.assert_allclose(window["anchor_xy_m"], [5.3, 24.4], atol=1e-3)
    pose = [box_row(f, (5.3, 24.4), P, scale=2.0) for f in frames]
    boxes = anchor.prepare(rows, cameras, pose, 2.0, PLAYER, clip="pt0001", pose_space=SPACE_HALF)
    assert boxes["status"] == "supported"
    (window,) = boxes["windows"]
    assert window["anchor_kind"] == "box_bottom_centre"
    assert window["sigma_px"] == 12.0 and window["proxy_height_m"] == 0.0
    np.testing.assert_allclose(window["anchor_xy_m"], [5.3, 24.4], atol=1e-3)
    # A low-confidence ankle in one frame demotes the whole window to the box proxy.
    pose = [keypoint_row(f, (5.3, 24.4), P, ankle_conf=0.3 if f == 91 else 0.95) for f in frames]
    mixed = anchor.prepare(rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    assert mixed["windows"][0]["anchor_kind"] == "box_bottom_centre"
    # The declared floors dominate: sigma axes are at least the floor and depth is wider.
    lateral, axial = sorted(ankles["windows"][0]["sigma_axes_m"])
    assert lateral >= 0.5 and axial > lateral
    assert ankles["dead_zone_m"] == pytest.approx(0.30 * 1.85)


def supported_anchor(frames=(89, 90, 91, 92, 93), feet=(5.3, 24.4)):
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, feet, camera()) for f in frames]
    record = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert record["status"] == "supported"
    return record


def test_remote_branch_pays_more_than_local_and_release_offsets_inside_dead_zone_are_free():
    record = supported_anchor()
    frames = record["frames"]
    centre = np.asarray(record["windows"][0]["anchor_xy_m"])
    local = np.tile(centre + [0.3, 0.4], (len(frames), 1))  # 0.5 m release offset
    assert np.all(anchor.residuals(record, local) == 0.0)
    # A 1 m/s lateral kick-serve drift over five frames stays inside the zone.
    dts = (np.asarray(frames) - 97.9) / 25.0
    kick = centre + np.c_[0.2 + 1.0 * dts, np.full(len(frames), 0.3)]
    assert np.all(anchor.residuals(record, kick) == 0.0)
    remote = np.tile(centre + [0.0, -3.0], (len(frames), 1))
    farther = np.tile(centre + [0.0, -5.0], (len(frames), 1))
    remote_cost = float(np.sum(anchor.residuals(record, remote) ** 2))
    assert remote_cost > 0 and float(np.sum(anchor.residuals(record, farther) ** 2)) > remote_cost
    # Quadratic in Mahalanobis distance outside the zone, never a cutoff; axial is cheaper.
    axial = float(np.sum(anchor.residuals(record, np.tile(centre + [0.0, -2.0], (5, 1))) ** 2))
    lateral = float(np.sum(anchor.residuals(record, np.tile(centre + [2.0, 0.0], (5, 1))) ** 2))
    assert 0 < axial < lateral
    report = anchor.report(record, [5.4, 23.9, 3.1], [0.5, -1.0, 4.0], frames, 97.9, 25.0)
    assert report["enabled"] and len(report["per_row"]) == 5
    assert report["anchor_cost"] == pytest.approx(sum(r["cost"] for r in report["per_row"]))
    np.testing.assert_allclose(report["player_relative_contact_m"][0], [5.4, 23.9] - centre)
    assert anchor.residuals(None, local).shape == (0,) and anchor.report(None, 0, 0, [], 0, 25) == {
        "enabled": False
    }


def test_wrong_scale_missing_sidecar_or_unsupported_camera_abstains():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, (5.3, 24.4), camera()) for f in frames]
    wrong = anchor.prepare(rows, cameras, pose, 2.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    assert wrong["status"] == "abstained" and "pose_image_scale" in wrong["reason"]
    none = anchor.prepare(rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=None)
    assert none["status"] == "abstained" and "sidecar" in none["reason"]
    unsupported = deepcopy(cameras)
    for cam in unsupported["cameras"][:3]:
        cam["status"] = "unsupported"
    short = anchor.prepare(
        rows, unsupported, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert short["status"] == "abstained" and "supported frozen camera record" in short["reason"]
    assert [p["status"] for p in short["per_frame"]][:3] == ["unanchored"] * 3
    sparse = anchor.prepare(
        rows, cameras, pose[:2] + pose[4:], 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert sparse["status"] == "supported"  # three anchored frames still pool
    sparser = anchor.prepare(
        rows, cameras, pose[:2], 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert (
        sparser["status"] == "abstained" and "fewer than the declared minimum" in sparser["reason"]
    )
    radial = deepcopy(rows)
    radial[0]["k1"] = 0.1
    assert (
        "radial"
        in anchor.prepare(
            radial, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
        )["reason"]
    )
    other = deepcopy(cameras)
    other["cameras"][1]["P"][0][0] += 1.0
    changed = anchor.prepare(rows, other, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    assert changed["status"] == "abstained" and "frozen camera record" in changed["reason"]
    unsided = anchor.prepare(
        rows, cameras, pose, 1.0, dict(PLAYER, side=None), clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert unsided["status"] == "abstained"
    receiver = anchor.prepare(
        rows, cameras, pose, 1.0, dict(PLAYER, side="near"), clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert (
        receiver["status"] == "abstained"
        and "0 sided pose rows" in receiver["per_frame"][0]["reason"]
    )


def test_airborne_drift_inflates_spread_without_moving_the_median_anchor():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    P = camera()
    planted = [keypoint_row(f, (5.3, 24.4), P) for f in frames]
    drifted = deepcopy(planted)
    drifted[-1] = keypoint_row(93, (5.3, 25.3), P)  # heels rise before the jump
    a = anchor.prepare(rows, cameras, planted, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    b = anchor.prepare(rows, cameras, drifted, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    # One airborne frame in five: the median ignores it entirely (a mean would move 0.18 m).
    np.testing.assert_allclose(
        a["windows"][0]["anchor_xy_m"], b["windows"][0]["anchor_xy_m"], atol=1e-6
    )
    assert b["per_frame"][-1]["anchor_xy_m"][1] == pytest.approx(25.3, abs=1e-3)
    # Progressive drift over the last three frames: the anchor moves less than the mean
    # would, the robust spread grows, and a large drift raises the declared warning.
    progressive = deepcopy(planted)
    for i, f in enumerate((91, 92, 93)):
        progressive[2 + i] = keypoint_row(f, (5.3, 24.4 + 0.3 * (i + 1)), P)
    c = anchor.prepare(
        rows, cameras, progressive, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    mean_shift = np.mean([0, 0, 0.3, 0.6, 0.9])
    assert 0 < c["windows"][0]["anchor_xy_m"][1] - 24.4 < mean_shift
    assert c["windows"][0]["spread_m"][1] > a["windows"][0]["spread_m"][1]
    assert c["windows"][0]["spread_warning"] is False
    large = deepcopy(planted)
    for i, f in enumerate((91, 92, 93)):
        large[2 + i] = keypoint_row(f, (5.3, 24.4 + 0.9 * (i + 1)), P)
    d = anchor.prepare(rows, cameras, large, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    assert d["windows"][0]["spread_warning"] is True


def test_raw_root_guard_abstains_and_reports_its_calibration_limitation():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, (5.3, 24.4), camera()) for f in frames]
    far_root = dict(PLAYER, court_centre_xy_m=[5.3, 21.0])
    record = anchor.prepare(
        rows, cameras, pose, 1.0, far_root, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert record["status"] == "abstained" and "raw pose root" in record["reason"]
    guard = record["raw_root_guard"]
    assert guard["distance_m"] == pytest.approx(3.4, abs=0.01) and guard["limit_m"] == 2.0
    assert "independently calibrated" in guard["limitation"]
    ok = anchor.prepare(rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    assert "independently calibrated" in ok["raw_root_guard"]["limitation"]


def test_registration_windows_pool_separately_and_never_across_cuts():
    frames = [29, 30, 31, 32, 33, 34]
    P_a, P_b = camera(), camera(centre=(2.0, -33.0, 11.0), focal=4200.0)
    rows = []
    cameras = dict(clip="pt0001", cameras=[])
    pose = []
    for f in frames:
        P, window, feet = (P_a, 28, (5.3, 24.4)) if f < 32 else (P_b, 36, (6.3, 24.9))
        rows.append(
            dict(frame=float(f), pixel=[900.0, 300.0], uncertainty_px=3.0, camera=P.tolist())
        )
        cameras["cameras"].append(
            dict(
                frame=f,
                status="supported",
                P=P.tolist(),
                local_anchor_frame=window,
                source="registered",
            )
        )
        pose.append(keypoint_row(f, feet, P))
    record = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert record["status"] == "supported" and len(record["windows"]) == 2
    np.testing.assert_allclose(record["windows"][0]["anchor_xy_m"], [5.3, 24.4], atol=1e-3)
    np.testing.assert_allclose(record["windows"][1]["anchor_xy_m"], [6.3, 24.9], atol=1e-3)
    assert record["row_window"] == [0, 0, 0, 1, 1, 1]
    xy = np.array([[5.3, 24.4]] * 3 + [[6.3, 24.9]] * 3)
    assert np.all(anchor.residuals(record, xy) == 0.0)  # each row against its own window
    assert np.any(anchor.residuals(record, xy[::-1]) != 0.0)
    cameras["cameras"][-1]["local_anchor_frame"] = 40  # a one-frame window cannot be pooled
    short = anchor.prepare(rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE)
    assert short["status"] == "abstained" and "never pooled across cuts" in short["reason"]


def test_inputs_are_never_mutated_and_rows_must_match_the_fit():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, (5.3, 24.4), camera()) for f in frames]
    before = deepcopy((rows, cameras, pose, PLAYER))
    record = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert (rows, cameras, pose, PLAYER) == before
    anchor.require_rows(record, frames)
    anchor.require_rows(None, [1, 2])
    with pytest.raises(ValueError, match="differ"):
        anchor.require_rows(record, frames[:-1])
    with pytest.raises(ValueError, match="supported player anchor"):
        anchor.require_rows(anchor.abstained("x"), frames)
    with pytest.raises(ValueError, match="one horizontal"):
        anchor.residuals(record, np.zeros((4, 2)))
    diagnostics = anchor.incoming_diagnostics(
        [5.0, 24.0, 3.0], [0.0, 0.0, -4.0], frames, 97.9, 25.0
    )
    assert diagnostics["incoming_vz_sign_at_contact"] == "descending"
    assert diagnostics["apex_native_frame"] == pytest.approx(97.9 - 25.0 * 4.0 / 9.81)
    assert diagnostics["observed_rows_all_rising"] is False


def test_missing_camera_window_cannot_silently_pool_toss_anchors():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, (5.3, 24.4), camera()) for f in frames]
    # An unrelated context camera must not demote all known windows to None.
    cameras["cameras"].append(dict(frame=200, status="unsupported"))
    supported = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert supported["status"] == "supported"
    assert supported["windows"][0]["registration_anchor_frame"] == 7
    # Missing identity on a consumed row must abstain, never create one pooled window.
    cameras["cameras"][2].pop("local_anchor_frame")
    rejected = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert rejected["status"] == "abstained"
    assert "unknown cuts" in rejected["reason"]


def test_camera_distortion_cannot_be_hidden_by_an_undecorated_toss_row():
    frames = [89, 90, 91]
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, (5.3, 24.4), camera()) for f in frames]
    cameras["cameras"][1]["k1"] = 0.1
    rejected = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    assert rejected["status"] == "abstained"
    assert "radial camera requires k1" in rejected["reason"]


def test_derived_root_guard_discloses_dependence_without_numerical_change():
    frames = [89, 90, 91, 92, 93]
    rows, cameras = rows_and_cameras(frames)
    pose = [keypoint_row(f, (5.3, 24.4), camera()) for f in frames]
    original = anchor.prepare(
        rows, cameras, pose, 1.0, PLAYER, clip="pt0001", pose_space=SPACE_NATIVE
    )
    witness = dict(
        original_court_xy_m=[5.3, 15.0],
        same_camera_court_xy_m=PLAYER["court_centre_xy_m"],
        original_derived_disagreement_m=9.2,
    )
    revised = anchor.prepare(
        rows,
        cameras,
        pose,
        1.0,
        PLAYER | {"root_observation": witness},
        clip="pt0001",
        pose_space=SPACE_NATIVE,
    )
    assert original["status"] == revised["status"] == "supported"
    assert original["windows"] == revised["windows"]
    receipt = revised["raw_root_guard"]
    assert receipt["comparison_root_kind"] == "same_camera_derived_player_root"
    assert receipt["original_root_observation"] == witness
    assert receipt["independent_original_root_used_by_guard"] is False
    assert "not independent" in receipt["limitation"]
