"""GPU court-line masks equal cv2's bytes; the CUDA chamfer equals its PyTorch reference."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

FRAMES = Path(
    os.environ.get("TENNIS_DATA_ROOT", "data"),
    "processed/gpu-court-20261001/frames/nadal/pt0245",
)
needs_gpu = pytest.mark.skipif(
    "TENNIS_COURT_SIFT_DEVICE" not in os.environ or not FRAMES.is_dir(),
    reason="needs an explicit GPU (TENNIS_COURT_SIFT_DEVICE) and the copied pilot frames",
)


@needs_gpu
def test_masks_are_byte_identical_and_distances_close():
    import torch

    from cv.experiments.gpu_court import dt_cuda, torch_distance, torch_masks
    from cv.pipeline import court_topology

    images = [cv2.imread(str(p)) for p in sorted(FRAMES.glob("f_*.jpg"))[::60]]
    support = torch_masks.frame_support(images)
    for image, (masks, distances) in zip(images, support, strict=True):
        reference = court_topology.court_line_masks(image)
        for (name, mask), (ref_name, ref_mask), distance in zip(
            masks, reference, distances, strict=True
        ):
            assert name == ref_name and np.array_equal(mask, ref_mask)
            cv_distance = cv2.distanceTransform((ref_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
            full = distance.tensor.cpu().numpy()
            assert np.abs(np.minimum(cv_distance, 100) - np.minimum(full, 100)).max() < 1e-3
            rows, cols = np.array([5, 500, 900]), np.array([7, 960, 1800])
            assert np.array_equal(distance[rows, cols], full[rows, cols])
    standard = [masks[0][1] for masks, _ in support]
    sources = torch.from_numpy(np.stack([(m == 0).astype(np.uint8) for m in standard])).cuda()
    assert torch.equal(
        dt_cuda.distance_transform_tensor(sources),
        torch_distance.distance_transform_tensor(sources),
    )
