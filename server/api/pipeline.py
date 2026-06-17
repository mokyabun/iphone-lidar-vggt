from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import settings
from .geometry import bounds, colors_for_depth_pixels, keyframe_indices, unproject_depth
from .mask import central_object_mask, resize_mask
from .models import FrameRecord, ReconstructionResult
from .ply import write_point_cloud_ply
from .reconviagen import align_reconviagen_mesh, generate_mesh
from .scan_io import extracted_scan_package, read_confidence, read_depth, read_frames, read_image, write_json


def reconstruct_scan(package_path: Path, output_dir: Path) -> ReconstructionResult:
    started = time.monotonic()
    cfg = settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    _log(f"reconstruction started: package={package_path} output_dir={output_dir}")
    with extracted_scan_package(package_path) as root:
        with _timed("read scan package"):
            frames = read_frames(root)
        if not frames:
            raise RuntimeError("Scan package did not contain frames.")
        selected = [frames[index] for index in keyframe_indices(len(frames), cfg.max_frames)]
        _log(f"frames: total={len(frames)} selected={len(selected)} max_frames={cfg.max_frames}")
        with _timed("build object masks"):
            masks = {frame.frame_id: central_object_mask(read_depth(root, frame)) for frame in selected}
        with _timed("build LiDAR reference cloud"):
            object_points, object_colors, scene_points = _build_lidar_reference(root, selected, masks)
        _log(f"LiDAR points: object={object_points.shape[0]} scene={scene_points.shape[0]}")
        if object_points.shape[0] < 80:
            raise RuntimeError("Not enough LiDAR object points for metric scale alignment.")

        lidar_output = output_dir / "lidar_reference.ply"
        with _timed("write LiDAR reference PLY"):
            write_point_cloud_ply(lidar_output, object_points, object_colors)
        input_dir = output_dir / "reconviagen_input"
        with _timed("prepare ReconViaGen input views"):
            input_views, mask_warning = _prepare_reconviagen_input(root, selected, masks, object_points, input_dir)
        _log(f"ReconViaGen input views: count={len(input_views)} dir={input_dir}")
        if not input_views:
            raise RuntimeError("No usable views were available for ReconViaGen.")
        if mask_warning:
            warnings.append(mask_warning)
            _log(f"warning: {mask_warning}")

        raw_mesh = output_dir / "reconviagen_raw.glb"
        final_ply = output_dir / "reconviagen_metric.ply"
        preview_glb = output_dir / "reconviagen_metric.glb"
        print_stl = output_dir / "reconviagen_metric_print_mm.stl"
        with _timed("generate ReconViaGen mesh"):
            generate_mesh(input_dir, raw_mesh)
        with _timed("align ReconViaGen mesh to LiDAR"):
            mesh_metrics = align_reconviagen_mesh(raw_mesh, final_ply, preview_glb, print_stl, object_points)

    object_min, object_max, object_extent = bounds(object_points)
    scene_min, scene_max, scene_extent = bounds(scene_points)
    metrics: dict[str, object] = {
        "frame_count": len(frames),
        "selected_keyframes": len(selected),
        "input_views": len(input_views),
        "lidar_points": int(object_points.shape[0]),
        "scene_points": int(scene_points.shape[0]),
        "object_bounds_min_m": object_min,
        "object_bounds_max_m": object_max,
        "object_extent_m": object_extent,
        "scene_bounds_min_m": scene_min,
        "scene_bounds_max_m": scene_max,
        "scene_extent_m": scene_extent,
        "final_output_type": "mesh",
        "final_output_source": "reconviagen_lidar_scale",
        "final_output": str(final_ply),
        "preview_glb_output": str(preview_glb),
        "lidar_reference_output": str(lidar_output),
        "warnings": warnings,
        **mesh_metrics,
    }
    with _timed("write metrics"):
        write_json(output_dir / "metrics.json", metrics)
    _log(f"reconstruction complete in {time.monotonic() - started:.1f}s final_output={final_ply}")
    return ReconstructionResult(
        final_output=final_ply,
        preview_glb_output=preview_glb,
        print_stl_output=Path(mesh_metrics["print_stl_output"]) if mesh_metrics.get("print_stl_output") else None,
        lidar_reference_output=lidar_output,
        metrics=metrics,
    )


def _build_lidar_reference(
    root: Path,
    frames: list[FrameRecord],
    masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    scene_chunks: list[np.ndarray] = []
    cfg = settings()
    for frame in frames:
        depth = read_depth(root, frame)
        confidence = read_confidence(root, frame)
        image = np.asarray(read_image(root, frame), dtype=np.uint8)
        points, pixels = unproject_depth(
            depth,
            np.asarray(frame.intrinsics_depth, dtype=np.float32),
            np.asarray(frame.camera_to_world, dtype=np.float32),
            stride=cfg.stride,
        )
        if not points.size:
            continue
        keep = np.ones(points.shape[0], dtype=bool)
        if confidence is not None:
            keep &= confidence[pixels[:, 1], pixels[:, 0]] >= cfg.confidence_minimum
        scene_chunks.append(points[keep])
        mask = masks[frame.frame_id]
        object_keep = keep & mask[pixels[:, 1], pixels[:, 0]]
        if np.any(object_keep):
            point_chunks.append(points[object_keep])
            color_chunks.append(colors_for_depth_pixels(image, frame, pixels[object_keep]))

    if not point_chunks:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.concatenate(scene_chunks, axis=0) if scene_chunks else np.empty((0, 3), dtype=np.float32),
        )
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    points, colors = _clean_object_cloud(points, colors)
    scene_points = np.concatenate(scene_chunks, axis=0) if scene_chunks else points
    return points, colors, scene_points


def _clean_object_cloud(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    voxel = settings().object_voxel_m
    if voxel > 0 and points.shape[0] > 1:
        keys = np.floor(points / voxel).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        point_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
        color_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(point_sums, inverse, points)
        np.add.at(color_sums, inverse, colors)
        points = (point_sums / counts[:, None]).astype(np.float32)
        colors = np.clip(color_sums / counts[:, None], 0, 255).round().astype(np.uint8)
    if points.shape[0] >= 80:
        center = np.median(points, axis=0)
        radial = np.linalg.norm(points - center, axis=1)
        median = float(np.median(radial))
        mad = float(np.median(np.abs(radial - median)))
        if mad > 0:
            keep = radial <= median + 8.0 * 1.4826 * mad
            if int(keep.sum()) >= max(60, int(points.shape[0] * 0.65)):
                points = points[keep]
                colors = colors[keep]
    return points, colors


def _prepare_reconviagen_input(
    root: Path,
    frames: list[FrameRecord],
    masks: dict[str, np.ndarray],
    object_points: np.ndarray,
    input_dir: Path,
) -> tuple[list[Path], str | None]:
    input_dir.mkdir(parents=True, exist_ok=True)
    object_center = np.median(object_points, axis=0)
    candidates: list[dict[str, object]] = []
    for frame in frames:
        image = np.asarray(read_image(root, frame), dtype=np.uint8)
        mask = resize_mask(masks[frame.frame_id], image.shape[1], image.shape[0])
        ys, xs = np.nonzero(mask)
        if xs.size < 64:
            continue
        area_ratio = float(mask.mean())
        if not 0.002 <= area_ratio <= 0.8:
            continue
        camera_center = np.asarray(frame.camera_to_world, dtype=np.float64)[:3, 3]
        direction = camera_center - object_center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction /= norm
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        center = np.array([xs.mean() / image.shape[1], ys.mean() / image.shape[0]])
        center_score = max(0.0, 1.0 - float(np.linalg.norm(center - 0.5)))
        quality = bbox_area * (0.5 + center_score)
        candidates.append({"frame": frame, "image": image, "mask": mask, "direction": direction, "quality": quality})

    mask_warning: str | None = None
    if not candidates:
        # Mask filtering found no usable views — fall back to full-image (no mask).
        candidates = _unmasked_candidates(root, frames, object_center)
        if not candidates:
            return [], None
        mask_warning = (
            "Object mask detection found no usable views. "
            "Falling back to full-image reconstruction (no foreground masking). "
            "Results may include background geometry."
        )

    selected = _select_diverse_views(candidates, settings().reconviagen_max_images)
    selected = _order_views(selected)
    output_paths: list[Path] = []
    size = settings().reconviagen_input_size
    for index, candidate in enumerate(selected):
        rgba = _crop_rgba(np.asarray(candidate["image"]), np.asarray(candidate["mask"]))
        output = input_dir / f"view_{index:02d}.png"
        Image.fromarray(rgba).resize((size, size), Image.Resampling.LANCZOS).save(output)
        output_paths.append(output)
    write_json(input_dir / "manifest.json", {"view_count": len(output_paths), "frame_ids": [candidate["frame"].frame_id for candidate in selected]})
    return output_paths, mask_warning


def _unmasked_candidates(
    root: Path,
    frames: list[FrameRecord],
    object_center: np.ndarray,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for frame in frames:
        image = np.asarray(read_image(root, frame), dtype=np.uint8)
        full_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
        camera_center = np.asarray(frame.camera_to_world, dtype=np.float64)[:3, 3]
        direction = camera_center - object_center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction /= norm
        quality = float(image.shape[0] * image.shape[1])
        candidates.append({"frame": frame, "image": image, "mask": full_mask, "direction": direction, "quality": quality})
    return candidates


def _select_diverse_views(candidates: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    maximum = min(maximum, len(candidates))
    first = max(candidates, key=lambda candidate: float(candidate["quality"]))
    selected = [first]
    remaining = [candidate for candidate in candidates if candidate is not first]
    best_quality = max(float(candidate["quality"]) for candidate in candidates)
    while remaining and len(selected) < maximum:
        def score(candidate: dict[str, object]) -> float:
            direction = np.asarray(candidate["direction"])
            angular_distance = min(1.0 - float(np.clip(direction @ np.asarray(chosen["direction"]), -1.0, 1.0)) for chosen in selected)
            quality = float(candidate["quality"]) / max(best_quality, 1e-6)
            return angular_distance + 0.2 * quality

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _order_views(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(candidates) < 3:
        return candidates
    directions = np.stack([np.asarray(candidate["direction"]) for candidate in candidates])
    _, _, axes = np.linalg.svd(directions - directions.mean(axis=0), full_matrices=False)
    angles = np.arctan2(directions @ axes[1], directions @ axes[0])
    return [candidate for _, candidate in sorted(zip(angles, candidates), key=lambda item: item[0])]


def _crop_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    padding = int(round(side * settings().crop_padding))
    side = max(16, side + padding * 2)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    crop_x0 = center_x - side // 2
    crop_y0 = center_y - side // 2
    crop_x1 = crop_x0 + side
    crop_y1 = crop_y0 + side
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    source_x0 = max(0, crop_x0)
    source_y0 = max(0, crop_y0)
    source_x1 = min(image.shape[1], crop_x1)
    source_y1 = min(image.shape[0], crop_y1)
    target_x0 = source_x0 - crop_x0
    target_y0 = source_y0 - crop_y0
    target_x1 = target_x0 + source_x1 - source_x0
    target_y1 = target_y0 + source_y1 - source_y0
    rgba[target_y0:target_y1, target_x0:target_x1, :3] = image[source_y0:source_y1, source_x0:source_x1]
    rgba[target_y0:target_y1, target_x0:target_x1, 3] = mask[source_y0:source_y1, source_x0:source_x1].astype(np.uint8) * 255
    return rgba


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[pipeline] {timestamp} {message}", flush=True)


@contextmanager
def _timed(label: str):
    _log(f"{label}: start")
    started = time.monotonic()
    try:
        yield
    finally:
        _log(f"{label}: done in {time.monotonic() - started:.1f}s")
