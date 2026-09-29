from __future__ import annotations

import numpy as np

from cv.pipeline.pose_crop_infer import canvas_to_window, make_variant, pose_windows


def test_large_variant_crops_around_player_focus() -> None:
    window = np.zeros((1000, 1000, 3), dtype=np.uint8)
    _, metadata = make_variant(
        window,
        scale=1.0,
        flip=False,
        canvas=640,
        focus_xy=(500.0, 900.0),
    )

    assert metadata["crop_off"] == (180, 360)
    restored = canvas_to_window(np.asarray([[320.0, 540.0]]), metadata)
    assert np.allclose(restored, [[500.0, 900.0]])


def test_default_large_variant_remains_centered() -> None:
    window = np.zeros((1000, 1000, 3), dtype=np.uint8)
    _, metadata = make_variant(window, scale=1.0, flip=False, canvas=640)
    assert metadata["crop_off"] == (180, 180)


def test_half_precision_is_explicitly_forwarded() -> None:
    class EmptyResult:
        boxes = None
        keypoints = None

    class Model:
        def __init__(self) -> None:
            self.kwargs = None

        def predict(self, images, **kwargs):
            self.kwargs = kwargs
            return [EmptyResult() for _ in images]

    model = Model()
    result = pose_windows(
        model,
        [np.zeros((128, 128, 3), dtype=np.uint8)],
        half=True,
    )

    assert result == [None]
    assert model.kwargs["half"] is True


def test_default_precision_uses_model_default_without_deprecated_flag() -> None:
    class EmptyResult:
        boxes = None
        keypoints = None

    class Model:
        def __init__(self) -> None:
            self.kwargs = None

        def predict(self, images, **kwargs):
            self.kwargs = kwargs
            return [EmptyResult() for _ in images]

    model = Model()
    pose_windows(model, [np.zeros((128, 128, 3), dtype=np.uint8)])

    assert "half" not in model.kwargs


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return _Tensor(self.value[index])


class _Boxes:
    def __init__(self, boxes, conf):
        self.xyxy, self.conf = _Tensor(boxes), _Tensor(conf)

    def __len__(self):
        return len(self.conf.value)


def _people_result():
    from types import SimpleNamespace

    # Strong center/background person plus the actual source-box participant.
    return SimpleNamespace(
        boxes=_Boxes([[250, 230, 300, 330], [400, 300, 460, 400]], [0.99, 0.7]),
        keypoints=SimpleNamespace(data=_Tensor([np.full((17, 3), 1), np.full((17, 3), 2)])),
    )


def test_focus_box_binds_participant_instead_of_confident_center_person():
    from cv.pipeline.pose_crop_infer import select_detection

    result = _people_result()
    assert select_detection(result)["conf"] == 0.99  # historical unseeded behavior
    selected = select_detection(result, focus_box=[398, 298, 462, 402])
    assert selected["conf"] == 0.7
    assert np.all(selected["kpts"] == 2)


def test_disjoint_person_is_not_substituted_and_coordinates_are_explicit_pixels():
    from cv.pipeline.pose_crop_infer import select_detection

    assert select_detection(_people_result(), focus_box=[10, 10, 30, 40]) is None
    # Do not silently infer a normalized coordinate system from small values.
    assert select_detection(_people_result(), focus_box=[0.4, 0.3, 0.46, 0.4]) is None


def test_window_box_roundtrips_scale_crop_and_flip_exactly():
    from cv.pipeline.pose_crop_infer import focus_box_to_canvas

    box = np.array([420.0, 700.0, 510.0, 990.0])
    for scale in [1.0, 1.5]:
        for flip in [False, True]:
            _, meta = make_variant(
                np.zeros((1000, 1000, 3), dtype=np.uint8), scale, flip, focus_xy=(465, 845)
            )
            canvas_box = focus_box_to_canvas(box, meta)
            corners = canvas_to_window(canvas_box.reshape(2, 2), meta)
            assert np.allclose(np.concatenate([corners.min(axis=0), corners.max(axis=0)]), box)


def test_each_pose_variant_receives_transformed_observed_box(monkeypatch):
    from cv.pipeline import pose_crop_infer as infer

    calls = []

    class Model:
        def predict(self, images, **kwargs):
            return [None] * len(images)

    def select(result, canvas, focus_box):
        calls.append(focus_box)

    monkeypatch.setattr(infer, "select_detection", select)
    infer.pose_windows(
        Model(),
        [np.zeros((100, 100, 3), dtype=np.uint8)],
        scales=(1.0, 1.5),
        flips=(False, True),
        focus_boxes=[[20, 30, 40, 60]],
    )
    # Native 100px image is centered on640px canvas, or150px at1.5 scale.
    expected = [
        [290, 300, 310, 330],
        [329, 300, 349, 330],
        [275, 290, 305, 335],
        [334, 290, 364, 335],
    ]
    assert np.allclose(calls, expected)
