from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from .geometry import apply_confidence_mask, colors_for_depth_pixels, keyframe_indices, unproject_depth
from .io import open_scan_package, read_confidence, read_depth, read_frames, read_image, write_json
from .models import FrameRecord, ReconstructionMetrics
from .ply import write_point_cloud_ply
from .segmentation import object_mask
from .vggt_adapter import run_vggt


def reconstruct_scan(
    package_path: Path,
    output_dir: Path,
    max_frames: int = 48,
    stride: int = 4,
    confidence_minimum: int = 1,
    run_vggt_stage: bool = False,
    preserve_color: bool = True,
    extract_object: bool = False,
    reconstruct_mesh: bool = False,
) -> ReconstructionMetrics:
    max_frames = _env_int("SCAN_MAX_FRAMES", max_frames)
    stride = _env_int("SCAN_STRIDE", stride)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    with open_scan_package(Path(package_path)) as root:
        frames = read_frames(root)
        selected = [frames[index] for index in keyframe_indices(len(frames), max_frames)]
        object_masks = build_object_masks(root, selected) if extract_object else None
        points, colors = build_lidar_point_cloud(root, selected, stride, confidence_minimum, preserve_color, object_masks)

        lidar_output = output_dir / "scan_lidar_points.ply"
        write_point_cloud_ply(lidar_output, points, colors)

        run_tsdf = reconstruct_mesh or _env_bool("SCAN_RUN_TSDF", False)
        tsdf_output = try_open3d_tsdf(root, selected, output_dir, warnings, preserve_color, object_masks) if run_tsdf else None
        vggt_output: Path | None = None
        vggt_points = 0
        if run_vggt_stage:
            try:
                vggt_output, vggt_points = run_vggt(root, selected, output_dir, preserve_color=preserve_color, object_masks=object_masks)
            except Exception as exc:  # noqa: BLE001 - VGGT is optional and environment-sensitive.
                warnings.append(f"VGGT stage skipped: {exc}")

        final_output = output_dir / "scan_final.ply"
        final_source = tsdf_output if reconstruct_mesh and tsdf_output else vggt_output or tsdf_output or lidar_output
        shutil.copyfile(final_source, final_output)

    metrics = ReconstructionMetrics(
        frame_count=len(frames),
        selected_keyframes=len(selected),
        lidar_points=int(points.shape[0]),
        vggt_points=vggt_points,
        final_output=str(final_output),
        lidar_output=str(lidar_output),
        tsdf_output=str(tsdf_output) if tsdf_output else None,
        vggt_output=str(vggt_output) if vggt_output else None,
        warnings=warnings,
    )
    write_json(output_dir / "metrics.json", metrics.model_dump())
    return metrics


def build_lidar_point_cloud(
    root: Path,
    frames: list[FrameRecord],
    stride: int,
    confidence_minimum: int,
    preserve_color: bool = True,
    object_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []

    for frame in frames:
        depth = read_depth(root, frame)
        confidence = read_confidence(root, frame)
        image = np.asarray(read_image(root, frame))
        intrinsics = np.asarray(frame.intrinsics_depth, dtype=np.float32)
        camera_to_world = np.asarray(frame.camera_to_world, dtype=np.float32)

        points, pixels = unproject_depth(depth, intrinsics, camera_to_world, stride=stride)
        colors = colors_for_depth_pixels(image, frame, pixels) if preserve_color else np.full((points.shape[0], 3), 220, dtype=np.uint8)
        if object_masks and frame.frame_id in object_masks and points.size:
            mask = object_masks[frame.frame_id]
            keep = mask[pixels[:, 1], pixels[:, 0]]
            points = points[keep]
            colors = colors[keep]
            pixels = pixels[keep]
        points, colors = apply_confidence_mask(points, colors, pixels, confidence, confidence_minimum)
        if points.size:
            point_chunks.append(points)
            color_chunks.append(colors)

    if not point_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(point_chunks, axis=0), np.concatenate(color_chunks, axis=0)


def build_object_masks(root: Path, frames: list[FrameRecord]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for frame in frames:
        masks[frame.frame_id] = object_mask(root, frame, read_depth(root, frame))
    return masks


def try_open3d_tsdf(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
    warnings: list[str],
    preserve_color: bool = True,
    object_masks: dict[str, np.ndarray] | None = None,
) -> Path | None:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        warnings.append("Open3D is not installed; wrote point-cloud baseline only.")
        return None

    try:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.015,
            sdf_trunc=0.06,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for frame in frames:
            depth_np = read_depth(root, frame)
            image_np = np.asarray(read_image(root, frame))
            if object_masks and frame.frame_id in object_masks:
                depth_np = np.where(object_masks[frame.frame_id], depth_np, 0).astype(np.float32)
            if not preserve_color:
                image_np = np.full_like(image_np, 220)
            if image_np.shape[:2] != depth_np.shape:
                image_np = _resize_rgb_to_depth(image_np, frame.depth_width, frame.depth_height)
            color = o3d.geometry.Image(image_np)
            depth = o3d.geometry.Image(depth_np.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1.0,
                depth_trunc=8.0,
                convert_rgb_to_intensity=False,
            )
            k = np.asarray(frame.intrinsics_depth, dtype=np.float64)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.depth_width, frame.depth_height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
            camera_to_world = np.asarray(frame.camera_to_world, dtype=np.float64)
            world_to_camera = np.linalg.inv(camera_to_world)
            volume.integrate(rgbd, intrinsic, world_to_camera)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        output = output_dir / "scan_lidar_tsdf.ply"
        if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=True):
            warnings.append("Open3D TSDF mesh export failed; wrote point-cloud baseline only.")
            return None
    except Exception as exc:  # noqa: BLE001 - TSDF is a best-effort refinement path.
        warnings.append(f"Open3D TSDF skipped: {exc}")
        return None
    return output


def _resize_rgb_to_depth(image_rgb: np.ndarray, depth_width: int, depth_height: int) -> np.ndarray:
    try:
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return np.asarray(Image.fromarray(image_rgb).resize((depth_width, depth_height), resampling)).astype(np.uint8)
    except Exception:
        y_idx = np.linspace(0, image_rgb.shape[0] - 1, depth_height).round().astype(np.int32)
        x_idx = np.linspace(0, image_rgb.shape[1] - 1, depth_width).round().astype(np.int32)
        return image_rgb[y_idx[:, None], x_idx[None, :]].astype(np.uint8)


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
