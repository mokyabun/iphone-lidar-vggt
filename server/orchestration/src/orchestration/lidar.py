from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import colors_for_depth_pixels, unproject_depth
from .io import read_confidence, read_depth, read_image
from .models import FrameRecord
from .point_cloud import temporal_consistency_filter
from .settings import env_int


@dataclass(frozen=True)
class FramePointCloud:
    points: np.ndarray
    colors: np.ndarray
    all_points: np.ndarray
    frame_ids: np.ndarray


def build_lidar_point_cloud(
    root: Path,
    frames: list[FrameRecord],
    stride: int,
    confidence_minimum: int,
    preserve_color: bool = True,
    object_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points, colors, _ = build_lidar_point_cloud_with_reference(
        root,
        frames,
        stride,
        confidence_minimum,
        preserve_color,
        object_masks,
    )
    return points, colors


def build_lidar_point_cloud_with_reference(
    root: Path,
    frames: list[FrameRecord],
    stride: int,
    confidence_minimum: int,
    preserve_color: bool = True,
    object_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_clouds = collect_lidar_frame_clouds(root, frames, stride, confidence_minimum, preserve_color, object_masks)
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    frame_id_chunks: list[np.ndarray] = []
    all_point_chunks: list[np.ndarray] = []

    for frame_cloud in frame_clouds:
        if frame_cloud.points.size:
            point_chunks.append(frame_cloud.points)
            color_chunks.append(frame_cloud.colors)
            frame_id_chunks.append(frame_cloud.frame_ids)
        if frame_cloud.all_points.size:
            all_point_chunks.append(frame_cloud.all_points)

    if not point_chunks:
        all_points = np.concatenate(all_point_chunks, axis=0) if all_point_chunks else np.empty((0, 3), dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), all_points

    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if object_masks:
        points, colors = temporal_consistency_filter(points, colors, np.concatenate(frame_id_chunks, axis=0))
    all_points = np.concatenate(all_point_chunks, axis=0) if all_point_chunks else points
    return points, colors, all_points


def collect_lidar_frame_clouds(
    root: Path,
    frames: list[FrameRecord],
    stride: int,
    confidence_minimum: int,
    preserve_color: bool,
    object_masks: dict[str, np.ndarray] | None,
) -> list[FramePointCloud]:
    if not frames:
        return []

    workers = min(len(frames), env_int("SCAN_FRAME_WORKERS", min(4, os.cpu_count() or 1)))
    if workers <= 1:
        return [
            build_lidar_frame_cloud(root, frame, frame_index, stride, confidence_minimum, preserve_color, object_masks)
            for frame_index, frame in enumerate(frames)
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda item: build_lidar_frame_cloud(
                    root,
                    item[1],
                    item[0],
                    stride,
                    confidence_minimum,
                    preserve_color,
                    object_masks,
                ),
                enumerate(frames),
            )
        )


def build_lidar_frame_cloud(
    root: Path,
    frame: FrameRecord,
    frame_index: int,
    stride: int,
    confidence_minimum: int,
    preserve_color: bool,
    object_masks: dict[str, np.ndarray] | None,
) -> FramePointCloud:
    depth = read_depth(root, frame)
    confidence = read_confidence(root, frame)
    image = np.asarray(read_image(root, frame)) if preserve_color else None
    intrinsics = np.asarray(frame.intrinsics_depth, dtype=np.float32)
    camera_to_world = np.asarray(frame.camera_to_world, dtype=np.float32)

    points, pixels = unproject_depth(depth, intrinsics, camera_to_world, stride=stride)
    if points.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        return FramePointCloud(empty_points, empty_colors, empty_points, np.empty((0,), dtype=np.int16))

    colors = colors_for_depth_pixels(image, frame, pixels) if image is not None else np.full((points.shape[0], 3), 220, dtype=np.uint8)
    keep = confidence_keep(pixels, confidence, confidence_minimum)
    all_points = points[keep]

    if object_masks and frame.frame_id in object_masks:
        mask = object_masks[frame.frame_id]
        keep = keep & mask[pixels[:, 1], pixels[:, 0]]

    frame_points = points[keep]
    frame_colors = colors[keep]
    return FramePointCloud(
        frame_points,
        frame_colors,
        all_points,
        np.full(frame_points.shape[0], frame_index, dtype=np.int16),
    )


def confidence_keep(pixels: np.ndarray, confidence: np.ndarray | None, minimum: int) -> np.ndarray:
    if confidence is None or pixels.size == 0:
        return np.ones(pixels.shape[0], dtype=bool)
    values = confidence[pixels[:, 1], pixels[:, 0]]
    return values >= minimum
