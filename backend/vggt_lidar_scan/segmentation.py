from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import FrameRecord

_SAM_MODEL_CACHE: Any | None = None


def object_mask(root: Path, frame: FrameRecord, depth: np.ndarray, allow_sam: bool = True) -> np.ndarray:
    backend = os.environ.get("OBJECT_MASK_BACKEND", "sam3_depth").lower()
    intrinsics = np.asarray(frame.intrinsics_depth, dtype=np.float32)
    depth_mask = central_object_mask(depth, intrinsics)
    if allow_sam and backend in {"sam3", "sam3_depth"}:
        try:
            sam_mask = _sam3_center_mask(root / frame.image_path, depth.shape)
            if sam_mask is not None and _mask_is_plausible(sam_mask):
                if backend == "sam3_depth":
                    refined = sam_mask & depth_mask
                    return refined if _mask_is_plausible(refined) else depth_mask
                return sam_mask
        except Exception:
            pass
    return depth_mask


def central_object_mask(depth: np.ndarray, intrinsics: np.ndarray | None = None) -> np.ndarray:
    if intrinsics is not None and _env_bool("OBJECT_REMOVE_DOMINANT_PLANE", True):
        plane_mask = _plane_removed_center_mask(depth, intrinsics)
        if _mask_is_plausible(plane_mask):
            return plane_mask
    return _depth_band_center_mask(depth)


def _depth_band_center_mask(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 8.0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool)

    height, width = depth.shape
    center_fraction = _env_float("OBJECT_CENTER_FRACTION", 0.35)
    band = _env_float("OBJECT_DEPTH_BAND_METERS", 0.45)
    y0 = max(0, int(height * (0.5 - center_fraction / 2)))
    y1 = min(height, int(height * (0.5 + center_fraction / 2)))
    x0 = max(0, int(width * (0.5 - center_fraction / 2)))
    x1 = min(width, int(width * (0.5 + center_fraction / 2)))

    center_depth = depth[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    if center_depth.size == 0:
        center_depth = depth[valid]
    target = float(np.median(center_depth))
    candidate = valid & (np.abs(depth - target) <= band)
    return _component_touching_center(candidate, (y0, y1, x0, x1))


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def _component_touching_center(mask: np.ndarray, center_box: tuple[int, int, int, int]) -> np.ndarray:
    try:
        import cv2  # type: ignore

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return mask
        y0, y1, x0, x1 = center_box
        center_labels = labels[y0:y1, x0:x1]
        labels_in_center, counts = np.unique(center_labels[center_labels > 0], return_counts=True)
        if labels_in_center.size:
            label = int(labels_in_center[np.argmax(counts)])
            return labels == label
        areas = np.bincount(labels.reshape(-1))
        areas[0] = 0
        return labels == int(np.argmax(areas))
    except Exception:
        return _component_touching_center_fallback(mask, center_box)


def _component_touching_center_fallback(mask: np.ndarray, center_box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = mask.shape
    y0, y1, x0, x1 = center_box
    seeds = np.argwhere(mask[y0:y1, x0:x1])
    if seeds.size:
        seed_y, seed_x = seeds[len(seeds) // 2]
        seed = (int(seed_y + y0), int(seed_x + x0))
    else:
        all_pixels = np.argwhere(mask)
        if not all_pixels.size:
            return mask
        center = np.array([height / 2, width / 2])
        seed = tuple(all_pixels[np.argmin(np.linalg.norm(all_pixels - center, axis=1))])

    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([seed])
    visited[seed] = True
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return visited


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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


def _plane_removed_center_mask(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 8.0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool)

    height, width = depth.shape
    stride = _env_int("OBJECT_PLANE_SAMPLE_STRIDE", 4)
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    sampled_depth = depth[ys, xs]
    sampled_valid = np.isfinite(sampled_depth) & (sampled_depth > 0.15) & (sampled_depth < 8.0)
    if np.count_nonzero(sampled_valid) < 50:
        return np.zeros(depth.shape, dtype=bool)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return np.zeros(depth.shape, dtype=bool)

    z = sampled_depth[sampled_valid].astype(np.float64)
    x = xs[sampled_valid].astype(np.float64)
    y = ys[sampled_valid].astype(np.float64)
    points = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], axis=1)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 50:
        return np.zeros(depth.shape, dtype=bool)
    plane = _fit_dominant_plane(points)
    if plane is None:
        return np.zeros(depth.shape, dtype=bool)

    normal, offset = plane
    ys_full, xs_full = np.mgrid[0:height, 0:width]
    z_full = np.where(valid, depth, 0).astype(np.float64)
    x_full = (xs_full.astype(np.float64) - cx) * z_full / fx
    y_full = (ys_full.astype(np.float64) - cy) * z_full / fy
    distances = np.abs(x_full * normal[0] + y_full * normal[1] + z_full * normal[2] + offset)
    plane_distance = _env_float("OBJECT_PLANE_DISTANCE_METERS", 0.025)
    candidate = valid & (distances > plane_distance)

    center_fraction = _env_float("OBJECT_CENTER_FRACTION", 0.35)
    y0 = max(0, int(height * (0.5 - center_fraction / 2)))
    y1 = min(height, int(height * (0.5 + center_fraction / 2)))
    x0 = max(0, int(width * (0.5 - center_fraction / 2)))
    x1 = min(width, int(width * (0.5 + center_fraction / 2)))
    return _component_touching_center(candidate, (y0, y1, x0, x1))


def _fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if points.shape[0] < 50:
        return None

    iterations = _env_int("OBJECT_PLANE_RANSAC_ITERATIONS", 160)
    threshold = _env_float("OBJECT_PLANE_DISTANCE_METERS", 0.025)
    rng = np.random.default_rng(7)
    best_plane: tuple[np.ndarray, float] | None = None
    best_count = 0
    for _ in range(iterations):
        indices = rng.choice(points.shape[0], 3, replace=False)
        sample = points[indices]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal = normal / norm
        offset = -float(normal.dot(sample[0]))
        if not np.isfinite(normal).all() or not np.isfinite(offset):
            continue
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            distances = np.abs(points @ normal + offset)
        distances = distances[np.isfinite(distances)]
        if distances.size == 0:
            continue
        count = int(np.count_nonzero(distances < threshold))
        if count > best_count:
            best_count = count
            best_plane = (normal, offset)
    return best_plane


def _sam3_center_mask(image_path: Path, depth_shape: tuple[int, int]) -> np.ndarray | None:
    global _SAM_MODEL_CACHE
    from ultralytics import SAM  # type: ignore

    model_name = os.environ.get("OBJECT_SAM_MODEL", "sam3.pt")
    if _SAM_MODEL_CACHE is None:
        _SAM_MODEL_CACHE = SAM(model_name)
    model = _SAM_MODEL_CACHE
    with Image.open(image_path) as image:
        width, height = image.size
    points = [[width / 2, height / 2]]
    labels = [1]
    results = model(str(image_path), points=points, labels=labels, verbose=False)
    if not results or getattr(results[0], "masks", None) is None:
        return None
    mask_data = results[0].masks.data
    if mask_data is None or len(mask_data) == 0:
        return None
    mask = mask_data[0].detach().cpu().numpy() > 0.5
    return resize_mask(mask, depth_shape[1], depth_shape[0])


def _mask_is_plausible(mask: np.ndarray) -> bool:
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    min_ratio = _env_float("OBJECT_MIN_MASK_RATIO", 0.002)
    max_ratio = _env_float("OBJECT_MAX_MASK_RATIO", 0.65)
    return min_ratio <= area_ratio <= max_ratio
