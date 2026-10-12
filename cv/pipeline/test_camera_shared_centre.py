import json

import numpy as np

from cv.pipeline import camera_shared_centre as csc

TRUE_CENTRE = np.array([5.4, -28.0, 9.6])


def _look_at(centre, target, focal, size=csc.IMAGE_SIZE):
    forward = target - centre
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    k = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1.0]])
    return k @ np.hstack([rotation, (-rotation @ centre)[:, None]])


def _views(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        target = np.array([5.485 + rng.normal(0, 1.5), 11.9 + rng.normal(0, 2.0), 0.0])
        out.append(_look_at(TRUE_CENTRE, target, rng.uniform(3000, 3800)))
    return np.stack(out)


def _wrong_vertical(projections, factor):
    """Same ground columns, vertical column from a camera at another distance."""
    wrong = TRUE_CENTRE + (TRUE_CENTRE - csc.COURT_CENTRE) * (factor - 1.0)
    return csc.projections_with_centre(projections, wrong)


def _ground(projection, xy):
    h = projection[:, [0, 1, 3]] @ np.c_[xy, np.ones(len(xy))].T
    return (h[:2] / h[2]).T


def test_shared_centre_recovers_camera_and_keeps_ground():
    truth = _views()
    perturbed = np.stack(
        [_wrong_vertical(p[None], f)[0] for p, f in zip(truth, np.linspace(0.6, 1.0, len(truth)))]
    )
    solution = csc.fit_shared_centre(perturbed)
    assert solution.accepted() == (True, "accepted")
    assert np.allclose(solution.centre, TRUE_CENTRE, atol=0.05)
    rebuilt = csc.projections_with_centre(perturbed, solution.centre)
    xy = np.array([[0.0, 0.0], [10.97, 23.77], [5.0, 11.885]])
    for before, after in zip(perturbed, rebuilt):
        assert np.allclose(_ground(before, xy), _ground(after, xy), atol=1e-6)


def _boxes(projections, stature=1.85, rows=600, seed=1):
    rng = np.random.default_rng(seed)
    view = rng.integers(0, len(projections), rows)
    far = rng.random(rows) < 0.5
    roots = np.c_[
        rng.uniform(2, 9, rows), np.where(far, rng.uniform(22, 26, rows), rng.uniform(-2, 2, rows))
    ]
    heights = []
    for v, root in zip(view, roots):
        top = projections[v] @ np.array([*root, stature, 1.0])
        bottom = projections[v] @ np.array([*root, 0.0, 1.0])
        heights.append(bottom[1] / bottom[2] - top[1] / top[2])
    return csc.PlayerBoxes(projections[view], roots, np.asarray(heights), far)


def test_box_balance_is_neutral_at_the_true_camera_and_tilted_when_too_close():
    truth = _views()
    boxes = _boxes(truth)
    assert abs(csc.box_log_ratio(TRUE_CENTRE, boxes)) < 1e-6
    close = TRUE_CENTRE + (TRUE_CENTRE - csc.COURT_CENTRE) * -0.3
    # a camera placed too close exaggerates perspective: near boxes read too tall
    assert abs(csc.box_log_ratio(close, boxes)) > 0.05
    solution = csc.fit_shared_centre(truth, boxes=boxes)
    assert solution.box_rows == len(boxes.far)
    assert np.allclose(solution.centre, TRUE_CENTRE, atol=0.1)


def test_apply_match_rewrites_cameras_and_abstains_on_inconsistent_views(tmp_path):
    truth = _views(12)
    clips = np.array([f"pt{i:04d}" for i in range(12)])
    np.savez_compressed(
        tmp_path / csc.POINT_CAMERA,
        clips=clips,
        frames=np.arange(12, dtype=np.int32),
        P=_wrong_vertical(truth, 0.7),
        reliable=np.ones(12, bool),
    )
    receipt = csc.apply_match(tmp_path)
    assert receipt["status"] == "applied"
    with np.load(tmp_path / csc.CAMERA) as data:
        assert np.allclose(csc.camera_centre(data["P"][3]), TRUE_CENTRE, atol=0.05)
        assert list(data["clips"]) == list(clips)
    assert json.loads((tmp_path / csc.RECEIPT).read_text())["status"] == "applied"

    # half the views from a camera low beside the court: no single centre explains both
    rng = np.random.default_rng(3)
    side = np.array([-8.0, 11.9, 4.0])
    stretched = truth.copy()
    for i in range(0, 12, 2):
        target = np.array([5.485 + rng.normal(0, 1.5), 11.9 + rng.normal(0, 2.0), 0.0])
        stretched[i] = _look_at(side, target, 2500.0)
    np.savez_compressed(
        tmp_path / csc.POINT_CAMERA,
        clips=clips,
        frames=np.arange(12, dtype=np.int32),
        P=stretched,
        reliable=np.ones(12, bool),
    )
    receipt = csc.apply_match(tmp_path)
    assert receipt["status"] == "abstained"
    with np.load(tmp_path / csc.CAMERA) as data:
        assert np.allclose(data["P"], stretched)
