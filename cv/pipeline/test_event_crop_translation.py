import json

import numpy as np
import pytest
import torch

from cv.pipeline.event_crop_translation import (
    NativeTranslationStore,
    SCHEMA,
    binding,
    legal_shifts,
    margin_origin,
    shifted_xy,
)
from cv.pipeline.event_video_model import CropStore, EventCropDataset


def fixture_store(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    expanded = tmp_path / "expanded"
    child = expanded / "b"
    child.mkdir(parents=True)
    tight = np.arange(16 * 116 * 200 * 3, dtype=np.uint32).astype(np.uint8).reshape(16, 116, 200, 3)
    context = np.arange(224 * 392 * 3, dtype=np.uint32).astype(np.uint8).reshape(224, 392, 3)
    arrays = {
        "tight.npy": tight[None],
        "context.npy": context[None],
        "origins.npy": np.asarray([[600, 400, 504, 346]], dtype=np.int32),
        "margin_origins.npy": np.asarray([[596, 396, 500, 342]], dtype=np.int32),
    }
    for name, value in arrays.items():
        np.save(child / name, value)
    np.savez(
        base / "b.npz",
        tight=tight[None, :, 4:112, 4:196],
        context=context[None, 4:220, 4:388],
        mel=np.zeros((1, 64, 50), np.float16),
    )
    manifest = {
        "shards": [{"broadcast": "b", "rows": 1, "path": str(base / "b.npz")}],
        "rows_index": [{"broadcast": "b", "clip": "b__pt0001", "frame": 7}],
    }
    (base / "manifest.json").write_text(json.dumps(manifest))
    receipt = {
        "schema": SCHEMA,
        "radius_native_px": 4,
        "native_image_size": [1920, 1080],
        "original_manifest": binding(base / "manifest.json"),
        "shards": [
            {
                "broadcast": "b",
                "rows": 1,
                "zero_offset_equal_rows": 1,
                "original_shard": binding(base / "b.npz"),
                "outputs": {name: binding(child / name) for name in arrays},
            }
        ],
    }
    (expanded / "manifest.json").write_text(json.dumps(receipt))
    source = CropStore.open(base)
    return source, NativeTranslationStore(expanded, source)


def test_default_and_zero_offset_are_byte_equal_for_all_dataset_fields(tmp_path):
    base, margin = fixture_store(tmp_path)
    kwargs = dict(
        targets=np.asarray([2]),
        offsets=np.asarray([0.3]),
        xy=np.asarray([[0.2, 0.4]]),
        xy_mask=np.ones(1),
        weights=np.ones(1),
        time_offsets=np.asarray([0.7]),
        time_mask=np.ones(1),
    )
    old = EventCropDataset(base, np.asarray([0]), **kwargs)[0]
    control = EventCropDataset(base, np.asarray([0]), translation_store=margin, **kwargs)[0]
    assert old.keys() == control.keys()
    for key in old:
        assert (
            torch.equal(old[key], control[key])
            if isinstance(old[key], torch.Tensor)
            else old[key] == control[key]
        )


def test_joint_native_shift_and_all_sixteen_exposures(tmp_path):
    _, margin = fixture_store(tmp_path)
    tight, context, (dx, dy) = margin.sample(0, 0, 9, radius=4, seed=37)
    assert (dx, dy) != (0, 0)
    assert np.array_equal(
        tight, margin.member(0, "tight.npy")[0, :, 4 + dy : 112 + dy, 4 + dx : 196 + dx]
    )
    assert np.array_equal(
        context, margin.member(0, "context.npy")[0, 4 + dy : 220 + dy, 4 + dx : 388 + dx]
    )


def test_translation_rng_is_independent_of_global_numpy_and_torch(tmp_path):
    _, margin = fixture_store(tmp_path)
    np.random.seed(91)
    torch.manual_seed(19)
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state().clone()
    first = margin.sample(0, 0, 12, radius=4, seed=37)
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_state, torch.random.get_rng_state())
    np.random.random(200)
    torch.rand(20)
    second = margin.sample(0, 0, 12, radius=4, seed=37)
    assert first[2] == second[2]
    assert np.array_equal(first[0], second[0])


def test_shift_target_before_horizontal_flip_and_preserve_time(tmp_path, monkeypatch):
    base, margin = fixture_store(tmp_path)
    _, _, shift = margin.sample(0, 0, 0, radius=4, seed=37)
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    dataset = EventCropDataset(
        base,
        np.asarray([0]),
        targets=np.asarray([1]),
        offsets=np.asarray([0.2]),
        xy=np.asarray([[0.25, -0.5]], np.float32),
        xy_mask=np.ones(1),
        weights=np.ones(1),
        augment=True,
        time_offsets=np.asarray([0.7]),
        time_mask=np.ones(1),
        translation_store=margin,
        translation_radius=4,
        translation_seed=37,
    )
    item = dataset[0]
    expected = shifted_xy(np.asarray([0.25, -0.5]), shift) * np.asarray([-1, 1])
    np.testing.assert_allclose(item["xy"], expected)
    assert item["target"] == 1 and item["offset"] == 0.2 and item["time_target"] == 0.7
    assert not item["mel"].any()


@pytest.mark.parametrize("origins", [[0, 0, 0, 0], [1728, 972, 1536, 864], [50, 0, 0, 0]])
def test_both_image_boundaries_constrain_offsets_without_dropping_zero(origins):
    shifts = legal_shifts(np.asarray(origins), 4)
    assert (0, 0) in shifts
    assert len(shifts) < 81
    for dx, dy in shifts:
        for channel, (width, height) in enumerate(((192, 108), (384, 216))):
            left, top = origins[2 * channel : 2 * channel + 2]
            mx, my = margin_origin(left, top, width, height)
            assert 0 <= left + dx <= 1920 - width and 0 <= top + dy <= 1080 - height
            assert 0 <= left + dx - mx <= 8 and 0 <= top + dy - my <= 8


def test_cache_mutation_is_refused(tmp_path):
    base, margin = fixture_store(tmp_path)
    path = margin.directory / "b" / "context.npy"
    value = np.load(path)
    value[0, 0, 0, 0] ^= 1
    np.save(path, value)
    with pytest.raises(ValueError, match="output changed"):
        NativeTranslationStore(margin.directory, base)


def test_original_shard_mutation_is_refused(tmp_path):
    base, margin = fixture_store(tmp_path)
    with (base.directory / "b.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="original shard identity"):
        NativeTranslationStore(margin.directory, base)


def test_non_native_or_unbound_translation_refused(tmp_path):
    base, _ = fixture_store(tmp_path)
    with pytest.raises(ValueError, match="bound expanded RGB"):
        EventCropDataset(base, np.asarray([0]), translation_radius=4)
    with pytest.raises(ValueError, match="zero-offset"):
        legal_shifts(np.asarray([-1, 0, 0, 0]), 4)
    with pytest.raises(ValueError, match="zero through four"):
        legal_shifts(np.asarray([0, 0, 0, 0]), 5)


@pytest.mark.parametrize("corrupt_original", [False, True])
def test_native_builder_preserves_exposure_inventory_and_requires_exact_old_pixels(
    tmp_path, corrupt_original
):
    import cv2

    from cv.pipeline.event_crop_translation import _build_shard
    from cv.pipeline.event_crops import sequence_frames, unique_exposures

    base = tmp_path / "base"
    base.mkdir()
    root = tmp_path / "native" / "b"
    pictures = root / "audit_frames_native_1080" / "pt0001"
    pictures.mkdir(parents=True)
    images = {}
    for frame in range(1, 21):
        image = np.full((1080, 1920, 3), frame * 7, np.uint8)
        path = pictures / f"f_{frame:04d}.jpg"
        assert cv2.imwrite(str(path), image)
        images[frame] = cv2.imread(str(path))
    (root / "frame_cadence_v1.json").write_text(
        json.dumps({"rows": [{"clip": "pt0001", "duplicate_frames": [6]}]})
    )
    sequence = sequence_frames(unique_exposures([6], 20), 10)
    tight = np.stack([images[frame][400:508, 600:792] for frame in sequence])[None]
    if corrupt_original:
        tight[0, 0, 0, 0, 0] ^= 1
    np.savez(
        base / "b.npz",
        tight=tight,
        context=images[10][346:562, 504:888][None],
        centres=np.asarray([[696.0, 454.0]]),
        clips=np.asarray(["b__pt0001"]),
        frames=np.asarray([10]),
    )
    manifest = {
        "root": str(root.parent),
        "shards": [{"broadcast": "b", "rows": 1, "fps": 25.0, "path": str(base / "b.npz")}],
        "rows_index": [{"broadcast": "b", "clip": "b__pt0001", "frame": 10}],
    }
    (base / "manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "prepared"
    destination.mkdir()
    if corrupt_original:
        with pytest.raises(ValueError, match="original crop parity failed"):
            _build_shard((str(base), str(destination), 0))
        assert not (destination / "b" / "receipt.json").exists()
    else:
        receipt = _build_shard((str(base), str(destination), 0))
        assert receipt["zero_offset_equal_rows"] == 1
        assert np.load(destination / "b" / "exposures.npy").tolist() == [sequence]
        assert 6 not in sequence and len(set(sequence)) == 16
        sources = json.loads((destination / "b" / "native_sources.json").read_text())
        assert sources["clock"]["original_shard_fps"] == 25.0
        assert (
            sources["timing_artifacts"][0]["sha256"]
            == binding(root / "frame_cadence_v1.json")["sha256"]
        )
