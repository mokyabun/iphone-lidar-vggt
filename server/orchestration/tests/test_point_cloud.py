from __future__ import annotations

import sys

import numpy as np

from orchestration.point_cloud import clean_point_cloud, temporal_consistency_filter


def test_clean_point_cloud_removes_isolated_object_outlier(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "open3d", None)
    monkeypatch.setenv("OBJECT_POINT_CLOUD_VOXEL_METERS", "0")
    rng = np.random.default_rng(7)
    object_points = rng.normal(0, 0.02, size=(200, 3)).astype(np.float32)
    points = np.vstack([object_points, np.array([[4.0, 4.0, 4.0]], dtype=np.float32)])
    colors = np.full((points.shape[0], 3), 128, dtype=np.uint8)

    cleaned, cleaned_colors, removed = clean_point_cloud(points, colors, object_mode=True)

    assert cleaned.shape[0] == 200
    assert cleaned_colors.shape == (200, 3)
    assert removed == 1


def test_clean_point_cloud_voxel_merges_duplicate_samples(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "open3d", None)
    points = np.array([[0.001, 0, 0], [0.0015, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=np.uint8)

    cleaned, cleaned_colors, removed = clean_point_cloud(
        points,
        colors,
        object_mode=False,
        voxel_size=0.005,
    )

    assert cleaned.shape == (2, 3)
    assert cleaned_colors.shape == (2, 3)
    assert removed == 1


def test_temporal_consistency_filter_removes_single_frame_points(monkeypatch) -> None:
    shared = np.array([[index * 0.0001, 0, 0] for index in range(40)], dtype=np.float32)
    outlier = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    points = np.vstack([shared, outlier])
    colors = np.full((41, 3), 100, dtype=np.uint8)
    frame_ids = np.array([index % 2 for index in range(40)] + [0], dtype=np.int16)

    filtered, filtered_colors = temporal_consistency_filter(points, colors, frame_ids)

    assert filtered.shape == (40, 3)
    assert filtered_colors.shape == (40, 3)
