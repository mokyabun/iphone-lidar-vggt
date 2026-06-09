from __future__ import annotations

import numpy as np

from vggt_lidar_scan.segmentation import central_object_mask


def test_central_object_mask_removes_far_background() -> None:
    depth = np.full((12, 12), 3.0, dtype=np.float32)
    depth[4:8, 4:8] = 1.0

    mask = central_object_mask(depth)

    assert mask[5, 5]
    assert not mask[0, 0]
    assert np.count_nonzero(mask) == 16


def test_central_object_mask_removes_dominant_plane() -> None:
    height = 48
    width = 64
    intrinsics = np.array([[55.0, 0.0, width / 2], [0.0, 55.0, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    y, x = np.mgrid[0:height, 0:width]
    depth = 0.7 + (y.astype(np.float32) / height) * 0.08
    depth[18:34, 28:36] -= 0.22

    mask = central_object_mask(depth, intrinsics)

    assert mask[24, 32]
    assert not mask[4, 4]
    assert np.count_nonzero(mask) < depth.size * 0.2
