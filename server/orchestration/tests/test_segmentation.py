from __future__ import annotations

import numpy as np

from orchestration.models import FrameRecord
from orchestration.segmentation import central_object_mask, propagate_object_mask


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


def test_propagated_mask_rejects_inconsistent_background_depth(monkeypatch) -> None:
    monkeypatch.setenv("OBJECT_PROPAGATION_DILATION_PIXELS", "0")
    monkeypatch.setenv("OBJECT_PROPAGATION_STRIDE", "1")
    frame = FrameRecord(
        frame_id="frame",
        timestamp=0.0,
        image_path="images/frame.png",
        depth_path="depth/frame.float32",
        image_width=16,
        image_height=16,
        depth_width=16,
        depth_height=16,
        intrinsics_depth=[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]],
        camera_to_world=np.eye(4).tolist(),
    )
    source_depth = np.full((16, 16), 3.0, dtype=np.float32)
    source_depth[5:11, 5:11] = 1.0
    source_mask = np.zeros((16, 16), dtype=bool)
    source_mask[5:11, 5:11] = True
    target_depth = source_depth.copy()
    target_depth[5:8, 5:8] = 2.0

    propagated = propagate_object_mask(frame, source_depth, source_mask, frame, target_depth)

    assert propagated is not None
    assert propagated[9, 9]
    assert not propagated[5, 5]
