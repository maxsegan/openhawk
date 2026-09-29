"""A camera one band off the paint is refit; a camera on the paint is left alone."""

import cv2
import numpy as np
import pytest

from cv.pipeline import court
from cv.pipeline import court_paint_refinement as cpr
from cv.pipeline.camera_cal import intrinsic_projection_from_ground

WIDTH, HEIGHT = 1920, 1080


def _look_at(eye, target, focal=1900.0) -> np.ndarray:
    """Pinhole camera at ``eye`` looking at ``target`` (court metres, z up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    intrinsics = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
    return intrinsics @ np.hstack([rotation, (-rotation @ eye)[:, None]])


TRUTH = _look_at((court.COURT_W / 2, -15.0, 10.0), (court.COURT_W / 2, 11.0, 0.0))
TRUE_CORNERS = {
    (x, y): tuple(
        (TRUTH @ np.array([x, y, 0.0, 1.0]))[:2] / (TRUTH @ np.array([x, y, 0.0, 1.0]))[2]
    )
    for x in (0.0, court.COURT_W)
    for y in (0.0, court.COURT_L)
}


def _projection(corners: dict) -> np.ndarray:
    court_points = np.float32(list(corners))
    image_points = np.float32(list(corners.values()))
    homography = cv2.getPerspectiveTransform(court_points, image_points)
    solved = intrinsic_projection_from_ground(homography, w=WIDTH, h=HEIGHT)
    assert solved is not None
    return solved[0]


def _painted_frame(projection: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(3)
    image = np.full((HEIGHT, WIDTH, 3), (60, 110, 190), np.uint8)
    image = cv2.add(image, rng.integers(0, 12, image.shape, dtype=np.uint8))
    homography = cpr.ground_homography(projection)
    for a, b in cpr.LINES.values():
        ends = cpr._to_image(homography, np.array([a, b], float))
        cv2.line(image, tuple(np.int32(ends[0])), tuple(np.int32(ends[1])), (235, 235, 235), 4)
    return image


def _one_band_off() -> np.ndarray:
    """The Monte Carlo defect: the model's far service line drawn on the painted far baseline."""
    homography = cv2.getPerspectiveTransform(
        np.float32(list(TRUE_CORNERS)), np.float32(list(TRUE_CORNERS.values()))
    )
    stretched = {}
    for (x, y), _ in TRUE_CORNERS.items():
        # Map model y so that the far service line lands where the far baseline is painted.
        target_y = (
            y
            if y == 0.0
            else court.COURT_L * court.COURT_L / (court.NET_Y + court.SERVICE_FROM_NET)
        )
        point = cv2.perspectiveTransform(np.float32([[[x, target_y]]]), homography)[0, 0]
        stretched[(x, y)] = (float(point[0]), float(point[1]))
    return _projection(stretched)


def test_camera_on_the_paint_is_left_alone():
    projection = _projection(TRUE_CORNERS)
    image = _painted_frame(projection)
    refined, receipt = cpr.refine_projection(image, projection)
    assert refined is None
    assert receipt["status"] == "on_paint"


def test_camera_one_band_off_is_refit_onto_the_paint():
    truth = _projection(TRUE_CORNERS)
    image = _painted_frame(truth)
    wrong = _one_band_off()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert cpr.worst_deciding(cpr.line_scores(gray, cpr.ground_homography(wrong))) < cpr.TRIGGER
    refined, receipt = cpr.refine_projection(image, wrong)
    assert receipt["status"] == "refit", receipt
    for (x, y), (u, v) in TRUE_CORNERS.items():
        point = refined @ np.array([x, y, 0.0, 1.0])
        assert np.hypot(point[0] / point[2] - u, point[1] / point[2] - v) < 8.0


def test_document_refit_is_reversible_and_off_is_identity():
    truth = _projection(TRUE_CORNERS)
    image = _painted_frame(truth)
    wrong = _one_band_off().tolist()
    document = {
        "schema": "test",
        "cameras": [
            {"frame": frame, "P": wrong, "supported": True, "source": "direct+registered"}
            for frame in (10, 11, 12)
        ],
    }
    unchanged, receipt = cpr.refine_camera_document(
        document, {10: image, 11: image, 12: image}, mode="off"
    )
    assert unchanged is document and receipt["changed_frames"] == 0
    refined, receipt = cpr.refine_camera_document(
        document, {10: image, 11: image, 12: image}, mode="on"
    )
    assert receipt["changed_frames"] == 3
    assert all(row["court_paint_geometry"] == "refit" for row in refined["cameras"])
    assert cpr.strip_refinement(refined) == document
    tampered = {
        **refined,
        "cameras": [dict(refined["cameras"][0], frame=99), *refined["cameras"][1:]],
    }
    assert cpr.strip_refinement(tampered) != document


def test_refit_budget_skips_groups_without_counting_them_unread(monkeypatch):
    image = _painted_frame(_projection(TRUE_CORNERS))
    wrong = _one_band_off().tolist()
    document = {
        "cameras": [{"frame": f, "P": wrong, "supported": True} for f in (10, 11, 12)],
    }
    monkeypatch.setattr(cpr, "MAX_REFITS", 0)
    kept, receipt = cpr.refine_camera_document(
        document, {f: image for f in (10, 11, 12)}, mode="on"
    )
    assert kept is document
    assert receipt["refits"] == 0 and receipt["unread_frames"] == 0
    assert receipt["groups_over_budget"] == 1
    held, receipt = cpr.refine_camera_document(
        document, {f: image for f in (10, 11, 12)}, mode="abstain"
    )
    assert receipt["abstained_frames"] == 3 and cpr.strip_refinement(held) == document


def test_mode_is_explicit():
    with pytest.raises(ValueError):
        cpr.refine_camera_document({"cameras": []}, {}, mode="maybe")


def test_refine_once_is_absent_by_default_and_explicit_when_on():
    from cv.pipeline.s6_labeled_stage import shared_settings

    assert "near_baseline_refinement_once" not in shared_settings({})
    assert (
        shared_settings({"near_baseline_refinement_once": "on"})["near_baseline_refinement_once"]
        == "on"
    )
    with pytest.raises(ValueError):
        shared_settings({"near_baseline_refinement_once": "twice"})


def test_abstain_holds_only_the_run_off_the_paint_and_is_reversible():
    """One projection over two runs: the painted run is kept, the run over a graphic is held."""
    truth = _projection(TRUE_CORNERS)
    painted = _painted_frame(truth)
    graphic = np.full((HEIGHT, WIDTH, 3), (60, 110, 190), np.uint8)
    frames = (10, 11, 12, 40, 41, 42)
    document = {
        "schema": "test",
        "cameras": [
            {"frame": frame, "P": truth.tolist(), "supported": True, "source": "direct+registered"}
            for frame in frames
        ],
    }
    images = {frame: painted if frame < 40 else graphic for frame in frames}
    kept, receipt = cpr.refine_camera_document(document, images, mode="on")
    assert kept is document and receipt["abstained_frames"] == 0
    held, receipt = cpr.refine_camera_document(document, images, mode="abstain")
    assert receipt["abstained_frames"] == 3 and receipt["changed_frames"] == 0
    assert [row["supported"] for row in held["cameras"]] == [True] * 3 + [False] * 3
    assert all(row["P"] is None for row in held["cameras"][3:])
    assert held["court_paint_refinement"] == "abstain"
    assert cpr.strip_refinement(held) == document


def test_verification_ignores_faint_sidelines_and_one_marginal_baseline():
    # Measured frames: exact cameras with faint sidelines (fresh source01, D source12) or a
    # partly hidden near baseline (F alt02); Rome pt0026 draws the court off the paint.
    faint_sidelines = {
        "near_baseline": 1.0,
        "far_baseline": 1.0,
        "near_service": 0.98,
        "far_service": 0.98,
        "left_doubles": 0.15,
        "right_doubles": 0.06,
        "left_singles": 0.0,
        "right_singles": 0.38,
        "centre": 0.02,
    }
    hidden_baseline = {
        "near_baseline": 0.38,
        "far_baseline": 0.94,
        "near_service": 1.0,
        "far_service": 1.0,
        "left_doubles": 0.9,
        "right_doubles": 0.98,
        "left_singles": 0.98,
        "right_singles": 0.98,
        "centre": 0.85,
    }
    off_paint = {
        "near_baseline": 0.83,
        "far_baseline": 0.81,
        "near_service": 0.0,
        "far_service": 0.5,
        "left_doubles": 0.1,
        "right_doubles": 0.1,
        "left_singles": 0.1,
        "right_singles": 0.06,
        "centre": 0.0,
    }
    assert cpr.scores_on_paint(faint_sidelines) is True
    assert cpr.scores_on_paint(hidden_baseline) is True
    assert cpr.scores_on_paint(off_paint) is False
    assert cpr.scores_on_paint({**hidden_baseline, "near_baseline": 0.2}) is False
    assert cpr.scores_on_paint({name: None for name in cpr.LINES}) is None


def test_component_jobs_reuse_the_source_paint_refit(tmp_path, monkeypatch):
    from cv.pipeline import s6_labeled_stage as stage

    calls = []
    refined = {"cameras": [{"frame": 1, "P": [[2.0]]}], "court_paint_refinement": "on"}

    def fake(document, *, mode, frame_paths):
        calls.append(mode)
        return refined, {"changed_frames": 1, "abstained_frames": 0, "unread_frames": 0}

    monkeypatch.setattr(cpr, "refine_camera_document", fake)
    source = tmp_path / "case"
    (source / "components/0/cases/key").mkdir(parents=True)
    (source / "component_plan.json").write_text("{}")
    cameras = {"cameras": [{"frame": 1, "P": [[1.0]]}]}
    labels = {"source_pack": {"images": [{"frame": 1, "source": {"path": "f1.jpg"}}]}}
    cache = stage._paint_cache_dir(source / "components/0/cases/key")
    assert cache == source / stage.PAINT_CACHE_DIR
    assert stage.warm_paint_cache(labels, cameras, {"court_paint_refinement": "on"}, cache)
    first, _ = stage._paint_cameras(cameras, labels, "on", cache)
    other, receipt = stage._paint_cameras(cameras, labels, "abstain", cache)
    assert calls == ["on", "abstain"] and first == refined and receipt.get("cache") is None
    again, receipt = stage._paint_cameras(cameras, labels, "on", cache)
    assert again == refined and receipt["cache"] == "hit" and len(calls) == 2
    assert stage._paint_cache_dir(tmp_path / "other/cases/key") is None
