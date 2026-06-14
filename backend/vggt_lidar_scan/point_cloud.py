from __future__ import annotations

import os

import numpy as np


def temporal_consistency_filter(
    points: np.ndarray,
    colors: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        points.shape[0] < 40
        or frame_ids.shape[0] != points.shape[0]
        or not _env_bool("OBJECT_TEMPORAL_FILTER", True)
    ):
        return points, colors

    voxel_size = _env_float("OBJECT_TEMPORAL_VOXEL_METERS", 0.006)
    minimum_frames = _env_int("OBJECT_TEMPORAL_MIN_FRAMES", 2)
    neighborhood = _env_int("OBJECT_TEMPORAL_NEIGHBOR_CELLS", 1)
    keys = np.floor(points / voxel_size).astype(np.int64)
    observed_frames: dict[tuple[int, int, int], set[int]] = {}
    for key, frame_id in zip(keys, frame_ids, strict=False):
        observed_frames.setdefault(tuple(int(value) for value in key), set()).add(int(frame_id))

    keep = np.zeros(points.shape[0], dtype=bool)
    for index, key in enumerate(keys):
        x, y, z = (int(value) for value in key)
        support: set[int] = set()
        for dx in range(-neighborhood, neighborhood + 1):
            for dy in range(-neighborhood, neighborhood + 1):
                for dz in range(-neighborhood, neighborhood + 1):
                    support.update(observed_frames.get((x + dx, y + dy, z + dz), ()))
                    if len(support) >= minimum_frames:
                        break
                if len(support) >= minimum_frames:
                    break
            if len(support) >= minimum_frames:
                break
        keep[index] = len(support) >= minimum_frames

    minimum_kept = max(40, int(points.shape[0] * _env_float("OBJECT_TEMPORAL_MIN_KEEP_RATIO", 0.55)))
    if int(keep.sum()) < minimum_kept:
        return points, colors
    return points[keep], colors[keep]


def clean_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    object_mode: bool,
    voxel_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    points = np.asarray(points, dtype=np.float32)
    colors_array = _normalized_colors(colors, points.shape[0])
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors_array = colors_array[finite]
    raw_count = int(points.shape[0])
    if raw_count == 0 or not _env_bool("POINT_CLOUD_CLEANUP", True):
        return points, colors_array, 0

    if voxel_size is None:
        voxel_size = _env_float(
            "OBJECT_POINT_CLOUD_VOXEL_METERS" if object_mode else "SCENE_POINT_CLOUD_VOXEL_METERS",
            0.002 if object_mode else 0.005,
        )

    try:
        points, colors_array = _clean_with_open3d(points, colors_array, object_mode, voxel_size)
    except Exception:
        points, colors_array = _clean_with_numpy(points, colors_array, object_mode, voxel_size)

    return points.astype(np.float32), colors_array.astype(np.uint8), raw_count - int(points.shape[0])


def _clean_with_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    object_mode: bool,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d  # type: ignore

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    minimum = _env_int("POINT_CLOUD_CLEANUP_MIN_POINTS", 40)
    if len(pcd.points) >= minimum:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=min(_env_int("POINT_CLOUD_OUTLIER_NEIGHBORS", 20), len(pcd.points) - 1),
            std_ratio=_env_float("POINT_CLOUD_OUTLIER_STD_RATIO", 1.5),
        )

    if object_mode and len(pcd.points) >= minimum:
        distances = np.asarray(pcd.compute_nearest_neighbor_distance())
        finite_distances = distances[np.isfinite(distances) & (distances > 0)]
        if finite_distances.size:
            radius = max(
                voxel_size * 3.5,
                float(np.median(finite_distances)) * _env_float("POINT_CLOUD_RADIUS_FACTOR", 4.0),
            )
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=_env_int("POINT_CLOUD_RADIUS_MIN_NEIGHBORS", 3),
                radius=radius,
            )

    cleaned_points = np.asarray(pcd.points)
    cleaned_colors = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).round().astype(np.uint8)
    return cleaned_points, cleaned_colors


def _clean_with_numpy(
    points: np.ndarray,
    colors: np.ndarray,
    object_mode: bool,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size > 0 and points.shape[0] > 1:
        keys = np.floor(points / voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        point_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
        color_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(point_sums, inverse, points)
        np.add.at(color_sums, inverse, colors)
        points = (point_sums / counts[:, None]).astype(np.float32)
        colors = np.clip(color_sums / counts[:, None], 0, 255).round().astype(np.uint8)

    if object_mode and points.shape[0] >= 40:
        center = np.median(points, axis=0)
        radial = np.linalg.norm(points - center, axis=1)
        median = float(np.median(radial))
        mad = float(np.median(np.abs(radial - median)))
        if mad > 0:
            limit = median + _env_float("POINT_CLOUD_FALLBACK_MAD_FACTOR", 8.0) * 1.4826 * mad
            keep = radial <= limit
            if int(keep.sum()) >= max(20, int(points.shape[0] * 0.6)):
                points = points[keep]
                colors = colors[keep]
    return points, colors


def _normalized_colors(colors: np.ndarray | None, count: int) -> np.ndarray:
    if colors is None or colors.shape[0] != count:
        return np.full((count, 3), 200, dtype=np.uint8)
    values = np.asarray(colors)
    if np.issubdtype(values.dtype, np.floating) and values.size and float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).round().astype(np.uint8)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default
