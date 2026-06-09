from __future__ import annotations

import numpy as np

from .models import FrameRecord


def keyframe_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, max_frames, dtype=int).tolist()


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    stride: int = 4,
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
    points_world = (camera_to_world @ points_camera.T).T[:, :3]
    pixels = np.stack([xs_valid.astype(np.int32), ys_valid.astype(np.int32)], axis=1)
    return points_world.astype(np.float32), pixels


def colors_for_depth_pixels(image_rgb: np.ndarray, frame: FrameRecord, pixels_depth: np.ndarray) -> np.ndarray:
    if pixels_depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    scale_x = frame.image_width / frame.depth_width
    scale_y = frame.image_height / frame.depth_height
    image_x = np.clip((pixels_depth[:, 0] * scale_x).round().astype(np.int32), 0, frame.image_width - 1)
    image_y = np.clip((pixels_depth[:, 1] * scale_y).round().astype(np.int32), 0, frame.image_height - 1)
    return image_rgb[image_y, image_x].astype(np.uint8)


def apply_confidence_mask(points: np.ndarray, colors: np.ndarray, pixels: np.ndarray, confidence: np.ndarray | None, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    if confidence is None or points.size == 0:
        return points, colors
    values = confidence[pixels[:, 1], pixels[:, 0]]
    keep = values >= minimum
    return points[keep], colors[keep]


def similarity_umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError("source and target must be Nx3 arrays with at least three correspondences")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, sign])
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = np.trace(np.diag(singular_values) @ correction) / variance
    translation = mu_target - scale * rotation @ mu_source

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform

