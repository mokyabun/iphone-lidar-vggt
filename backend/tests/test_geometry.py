from __future__ import annotations

import numpy as np

from vggt_lidar_scan.geometry import keyframe_indices, similarity_umeyama, unproject_depth


def test_keyframe_indices_are_evenly_spaced() -> None:
    assert keyframe_indices(5, 10) == [0, 1, 2, 3, 4]
    assert keyframe_indices(10, 4) == [0, 3, 6, 9]


def test_unproject_depth_identity_camera() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsics = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera_to_world = np.eye(4, dtype=np.float32)

    points, pixels = unproject_depth(depth, intrinsics, camera_to_world, stride=1)

    assert points.shape == (4, 3)
    assert pixels.shape == (4, 2)
    assert np.allclose(points[0], [0.0, 0.0, -1.0])
    assert np.allclose(points[-1], [1.0, -1.0, -1.0])


def test_similarity_umeyama_recovers_scale_and_translation() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = source * 2.0 + np.array([1.0, -1.0, 0.5])

    transform = similarity_umeyama(source, target)
    source_h = np.concatenate([source, np.ones((source.shape[0], 1))], axis=1)
    aligned = (transform @ source_h.T).T[:, :3]

    assert np.allclose(aligned, target)
