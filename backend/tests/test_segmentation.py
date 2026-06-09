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
