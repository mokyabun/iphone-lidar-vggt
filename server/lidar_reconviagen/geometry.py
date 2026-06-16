from __future__ import annotations

import numpy as np

from .models import FrameRecord

OPENCV_TO_ARKIT_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def keyframe_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, max_frames, dtype=int).tolist()


def camera_to_world_for_depth(camera_to_world_arkit: np.ndarray) -> np.ndarray:
    dtype = np.asarray(camera_to_world_arkit).dtype
    conversion = OPENCV_TO_ARKIT_CAMERA.astype(dtype if np.issubdtype(dtype, np.floating) else np.float32)
    return np.asarray(camera_to_world_arkit) @ conversion


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    stride: int,
    min_depth: float = 0.15,
    max_depth: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    xs_valid = xs[valid].astype(np.float32)
    ys_valid = ys[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x_cam = (xs_valid - cx) * z_valid / fx
    y_cam = (ys_valid - cy) * z_valid / fy
    points_camera = np.stack([x_cam, y_cam, z_valid, np.ones_like(z_valid)], axis=1)
    camera_to_world_cv = camera_to_world_for_depth(camera_to_world).astype(np.float64)
    points_world = (camera_to_world_cv @ points_camera.T).T[:, :3]
    pixels = np.stack([xs_valid.astype(np.int32), ys_valid.astype(np.int32)], axis=1)
    finite = np.isfinite(points_world).all(axis=1)
    return points_world[finite].astype(np.float32), pixels[finite]


def colors_for_depth_pixels(image_rgb: np.ndarray, frame: FrameRecord, pixels_depth: np.ndarray) -> np.ndarray:
    if pixels_depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    scale_x = frame.image_width / frame.depth_width
    scale_y = frame.image_height / frame.depth_height
    image_x = np.clip((pixels_depth[:, 0] * scale_x).round().astype(np.int32), 0, frame.image_width - 1)
    image_y = np.clip((pixels_depth[:, 1] * scale_y).round().astype(np.int32), 0, frame.image_height - 1)
    return image_rgb[image_y, image_x].astype(np.uint8)


def bounds(points: np.ndarray) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
    values = points[np.isfinite(points).all(axis=1)]
    if values.size == 0:
        return None, None, None
    lower = np.percentile(values, 1, axis=0)
    upper = np.percentile(values, 99, axis=0)
    return lower.round(5).tolist(), upper.round(5).tolist(), (upper - lower).round(5).tolist()
