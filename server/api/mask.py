from __future__ import annotations

from collections import deque

import numpy as np

from .config import settings


def central_object_mask(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 8.0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool)

    height, width = depth.shape
    cfg = settings()
    center_fraction = cfg.center_fraction
    y0 = max(0, int(height * (0.5 - center_fraction / 2)))
    y1 = min(height, int(height * (0.5 + center_fraction / 2)))
    x0 = max(0, int(width * (0.5 - center_fraction / 2)))
    x1 = min(width, int(width * (0.5 + center_fraction / 2)))

    center_depth = depth[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    if center_depth.size == 0:
        center_depth = depth[valid]
    target = float(np.median(center_depth))
    candidate = valid & (np.abs(depth - target) <= cfg.depth_band_m)
    return _component_touching_center(candidate, (y0, y1, x0, x1))


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def _component_touching_center(mask: np.ndarray, center_box: tuple[int, int, int, int]) -> np.ndarray:
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
