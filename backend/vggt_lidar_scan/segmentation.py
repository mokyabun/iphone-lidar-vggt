from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import FrameRecord

_SAM_MODEL_CACHE: Any | None = None


def object_mask(root: Path, frame: FrameRecord, depth: np.ndarray) -> np.ndarray:
    backend = os.environ.get("OBJECT_MASK_BACKEND", "sam3_depth").lower()
    depth_mask = central_object_mask(depth)
    if backend in {"sam3", "sam3_depth"}:
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


def central_object_mask(depth: np.ndarray) -> np.ndarray:
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
    min_ratio = _env_float("OBJECT_MIN_MASK_RATIO", 0.01)
    max_ratio = _env_float("OBJECT_MAX_MASK_RATIO", 0.85)
    return min_ratio <= area_ratio <= max_ratio
