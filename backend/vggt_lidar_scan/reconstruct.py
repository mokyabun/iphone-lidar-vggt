from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from .geometry import apply_confidence_mask, camera_to_world_for_depth, colors_for_depth_pixels, keyframe_indices, unproject_depth
from .io import open_scan_package, read_confidence, read_depth, read_frames, read_image, write_json
from .models import FrameRecord, ReconstructionMetrics
from .ply import count_ply_elements, write_point_cloud_ply
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
    object_mask_backend = os.environ.get("OBJECT_MASK_BACKEND", "sam3_depth") if extract_object else None

    with open_scan_package(Path(package_path)) as root:
        frames = read_frames(root)
        selected = [frames[index] for index in keyframe_indices(len(frames), max_frames)]
        print(
            f"[reconstruct] frames={len(frames)} selected={len(selected)} "
            f"vggt={run_vggt_stage} object={extract_object} mesh={reconstruct_mesh}",
            flush=True,
        )
        object_masks = build_object_masks(root, selected) if extract_object else None
        points, colors = build_lidar_point_cloud(root, selected, stride, confidence_minimum, preserve_color, object_masks)
        lidar_points_all = build_lidar_point_cloud(root, selected, stride, confidence_minimum, preserve_color, None)[0] if extract_object else points

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

        mesh_vertices = 0
        mesh_faces = 0
        final_output = output_dir / "scan_final.ply"
        final_source = tsdf_output if reconstruct_mesh and tsdf_output else vggt_output or tsdf_output or lidar_output
        if final_source.exists():
            mesh_vertices, mesh_faces = count_ply_elements(final_source)
        shutil.copyfile(final_source, final_output)

    lidar_bounds = _bounds(lidar_points_all)
    object_bounds = _bounds(points) if extract_object else (None, None, None)
    metrics = ReconstructionMetrics(
        frame_count=len(frames),
        selected_keyframes=len(selected),
        lidar_points=int(points.shape[0]),
        vggt_points=vggt_points,
        mesh_vertices=mesh_vertices if mesh_faces else 0,
        mesh_faces=mesh_faces,
        final_output_type="mesh" if mesh_faces else "point_cloud",
        object_mask_backend=object_mask_backend,
        camera_path_m=_camera_path_m(frames),
        camera_extent_m=_camera_extent_m(frames),
        lidar_bounds_min_m=lidar_bounds[0],
        lidar_bounds_max_m=lidar_bounds[1],
        lidar_extent_m=lidar_bounds[2],
        object_bounds_min_m=object_bounds[0],
        object_bounds_max_m=object_bounds[1],
        object_extent_m=object_bounds[2],
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
    backend = os.environ.get("OBJECT_MASK_BACKEND", "sam3_depth").lower()
    sam_limit = _env_int("OBJECT_SAM_MAX_FRAMES", 3)
    for index, frame in enumerate(frames):
        allow_sam = backend in {"sam3", "sam3_depth"} and index < sam_limit
        print(
            f"[reconstruct] object mask {index + 1}/{len(frames)} frame={frame.frame_id} "
            f"backend={backend if allow_sam else 'depth'}",
            flush=True,
        )
        masks[frame.frame_id] = object_mask(root, frame, read_depth(root, frame), allow_sam=allow_sam)
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
        voxel_length = _env_float("OBJECT_TSDF_VOXEL_METERS", 0.008)
        sdf_trunc = _env_float("OBJECT_TSDF_TRUNC_METERS", 0.035)
        depth_trunc = _env_float("OBJECT_TSDF_DEPTH_TRUNC_METERS", 4.0)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
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
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            k = np.asarray(frame.intrinsics_depth, dtype=np.float64)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.depth_width, frame.depth_height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
            camera_to_world = camera_to_world_for_depth(np.asarray(frame.camera_to_world, dtype=np.float64)).astype(np.float64)
            world_to_camera = np.linalg.inv(camera_to_world)
            volume.integrate(rgbd, intrinsic, world_to_camera)

        mesh = volume.extract_triangle_mesh()
        mesh = _postprocess_open3d_mesh(mesh, o3d)
        output = output_dir / "scan_object_mesh.ply"
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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _postprocess_open3d_mesh(mesh, o3d):  # noqa: ANN001, ANN201 - Open3D runtime type.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    triangle_count = len(mesh.triangles)
    if triangle_count:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_np = np.asarray(labels)
        counts_np = np.asarray(counts)
        if counts_np.size:
            keep_label = int(np.argmax(counts_np))
            remove = labels_np != keep_label
            mesh.remove_triangles_by_mask(remove.tolist())
            mesh.remove_unreferenced_vertices()
    smoothing_iterations = _env_int("OBJECT_MESH_SMOOTH_ITERATIONS", 1)
    if smoothing_iterations > 0 and len(mesh.triangles):
        mesh = mesh.filter_smooth_simple(number_of_iterations=smoothing_iterations)
    target_triangles = _env_int("OBJECT_MESH_MAX_TRIANGLES", 120000)
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.compute_vertex_normals()
    return mesh


def _bounds(points: np.ndarray) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
    if points.size == 0:
        return None, None, None
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return None, None, None
    lower = np.percentile(finite, 1, axis=0)
    upper = np.percentile(finite, 99, axis=0)
    extent = upper - lower
    return _round_vector(lower), _round_vector(upper), _round_vector(extent)


def _camera_positions(frames: list[FrameRecord]) -> np.ndarray:
    if not frames:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray([np.asarray(frame.camera_to_world, dtype=np.float32)[:3, 3] for frame in frames], dtype=np.float32)


def _camera_path_m(frames: list[FrameRecord]) -> float | None:
    positions = _camera_positions(frames)
    if positions.shape[0] < 2:
        return 0.0 if positions.shape[0] == 1 else None
    return round(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()), 4)


def _camera_extent_m(frames: list[FrameRecord]) -> list[float] | None:
    positions = _camera_positions(frames)
    if positions.size == 0:
        return None
    return _round_vector(np.ptp(positions, axis=0))


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values.tolist()]
