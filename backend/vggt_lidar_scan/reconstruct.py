from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from .geometry import apply_confidence_mask, camera_to_world_for_depth, colors_for_depth_pixels, keyframe_indices, unproject_depth
from .generative import run_generative_mesh
from .io import open_scan_package, read_confidence, read_depth, read_frames, read_image, write_json
from .models import FrameRecord, ReconstructionMetrics
from .ply import count_ply_elements, write_point_cloud_ply
from .point_cloud import clean_point_cloud, temporal_consistency_filter
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
    generative_mesh: bool = False,
) -> ReconstructionMetrics:
    max_frames = _env_int("SCAN_MAX_FRAMES", max_frames)
    stride = _env_int("SCAN_STRIDE", stride)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    extract_object = extract_object or generative_mesh
    reconstruct_mesh = reconstruct_mesh or generative_mesh
    object_mask_backend = os.environ.get("OBJECT_MASK_BACKEND", "sam3_depth") if extract_object else None

    with open_scan_package(Path(package_path)) as root:
        frames = read_frames(root)
        selected = [frames[index] for index in keyframe_indices(len(frames), max_frames)]
        print(
            f"[reconstruct] frames={len(frames)} selected={len(selected)} "
            f"vggt={run_vggt_stage} object={extract_object} mesh={reconstruct_mesh} pretty={generative_mesh}",
            flush=True,
        )
        object_masks = build_object_masks(root, selected) if extract_object else None
        points, colors = build_lidar_point_cloud(root, selected, stride, confidence_minimum, preserve_color, object_masks)
        lidar_raw_points = int(points.shape[0])
        points, colors, lidar_removed_points = clean_point_cloud(points, colors, object_mode=extract_object)
        lidar_points_all = build_lidar_point_cloud(root, selected, stride, confidence_minimum, preserve_color, None)[0] if extract_object else points

        lidar_output = output_dir / "scan_lidar_points.ply"
        write_point_cloud_ply(lidar_output, points, colors)

        run_mesh = reconstruct_mesh or _env_bool("SCAN_RUN_TSDF", False)
        mesh_method = _mesh_method() if run_mesh else None
        metric_mesh_output, tsdf_output, actual_mesh_method = build_mesh_output(
            root,
            selected,
            output_dir,
            warnings,
            points,
            colors,
            preserve_color,
            object_masks,
            mesh_method,
        ) if run_mesh else (None, None, None)
        mesh_output = metric_mesh_output
        generative_mesh_output: Path | None = None
        generative_mesh_backend: str | None = None
        generative_mesh_used = False
        if generative_mesh:
            try:
                generative_mesh_output, generative_mesh_backend = run_generative_mesh(
                    root,
                    selected,
                    output_dir,
                    points,
                    object_masks,
                    preserve_color,
                )
                mesh_output = generative_mesh_output
                actual_mesh_method = f"{generative_mesh_backend}_metric_aligned"
                generative_mesh_used = True
            except Exception as exc:  # noqa: BLE001 - metric mesh remains a reliable fallback.
                warnings.append(f"Pretty mesh skipped: {exc}")
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
        use_vggt_as_final = _env_bool("VGGT_AS_FINAL", False)
        if reconstruct_mesh and mesh_output:
            final_source = mesh_output
            final_output_source = actual_mesh_method or "mesh"
        elif use_vggt_as_final and vggt_output:
            final_source = vggt_output
            final_output_source = "vggt"
        elif mesh_output:
            final_source = mesh_output
            final_output_source = actual_mesh_method or "mesh"
        else:
            final_source = lidar_output
            final_output_source = "lidar_metric"
        if final_source.exists():
            mesh_vertices, mesh_faces = count_ply_elements(final_source)
        shutil.copyfile(final_source, final_output)

    lidar_bounds = _bounds(lidar_points_all)
    object_bounds = _bounds(points) if extract_object else (None, None, None)
    metrics = ReconstructionMetrics(
        frame_count=len(frames),
        selected_keyframes=len(selected),
        lidar_points=int(points.shape[0]),
        lidar_raw_points=lidar_raw_points,
        lidar_removed_points=lidar_removed_points,
        vggt_points=vggt_points,
        mesh_vertices=mesh_vertices if mesh_faces else 0,
        mesh_faces=mesh_faces,
        mesh_method=actual_mesh_method if mesh_faces else None,
        final_output_type="mesh" if mesh_faces else "point_cloud",
        final_output_source=final_output_source,
        generative_mesh_requested=generative_mesh,
        generative_mesh_used=generative_mesh_used,
        generative_mesh_backend=generative_mesh_backend,
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
        mesh_output=str(mesh_output) if mesh_output else None,
        metric_mesh_output=str(metric_mesh_output) if metric_mesh_output else None,
        generative_mesh_output=str(generative_mesh_output) if generative_mesh_output else None,
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
    frame_id_chunks: list[np.ndarray] = []

    for frame_index, frame in enumerate(frames):
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
            frame_id_chunks.append(np.full(points.shape[0], frame_index, dtype=np.int16))

    if not point_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if object_masks:
        points, colors = temporal_consistency_filter(points, colors, np.concatenate(frame_id_chunks, axis=0))
    return points, colors


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


def build_mesh_output(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
    warnings: list[str],
    points: np.ndarray,
    colors: np.ndarray,
    preserve_color: bool,
    object_masks: dict[str, np.ndarray] | None,
    mesh_method: str | None,
) -> tuple[Path | None, Path | None, str | None]:
    if mesh_method in {"printable", "printable_metric", "printable_hybrid", "printable_alpha", "alpha", "alpha_shape", "poisson"}:
        requested_method = "printable_alpha" if mesh_method in {"printable", "printable_metric"} else mesh_method
        printable, printable_method = try_open3d_printable_mesh(
            points, colors, output_dir, warnings, preserve_color, requested_method
        )
        if printable:
            return printable, None, printable_method
        tsdf = try_open3d_tsdf(root, frames, output_dir, warnings, preserve_color, object_masks)
        return tsdf, tsdf, "object_tsdf" if tsdf else None
    if mesh_method in {"tsdf", "object_tsdf"}:
        tsdf = try_open3d_tsdf(root, frames, output_dir, warnings, preserve_color, object_masks)
        return tsdf, tsdf, "object_tsdf" if tsdf else None

    warnings.append(f"Unknown MESH_METHOD={mesh_method}; falling back to printable_metric.")
    printable, printable_method = try_open3d_printable_mesh(
        points, colors, output_dir, warnings, preserve_color, "printable_alpha"
    )
    if printable:
        return printable, None, printable_method
    tsdf = try_open3d_tsdf(root, frames, output_dir, warnings, preserve_color, object_masks)
    return tsdf, tsdf, "object_tsdf" if tsdf else None


def try_open3d_printable_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    warnings: list[str],
    preserve_color: bool = True,
    requested_method: str = "printable_alpha",
) -> tuple[Path | None, str | None]:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        _append_warning_once(warnings, "Open3D is not installed; wrote point-cloud baseline only.")
        return None, None

    finite = np.isfinite(points).all(axis=1) if points.size else np.empty((0,), dtype=bool)
    points = points[finite]
    colors = colors[finite] if colors.shape[0] == finite.shape[0] else np.empty((0, 3), dtype=np.uint8)
    points, colors = _trim_printable_outliers(points, colors)
    minimum_points = _env_int("OBJECT_PRINTABLE_MIN_POINTS", 80)
    if points.shape[0] < minimum_points:
        warnings.append(f"Printable mesh skipped: only {points.shape[0]} object points.")
        return None, None

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if preserve_color and colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64) / 255.0, 0.0, 1.0))

        voxel = _env_float("OBJECT_PRINTABLE_POINT_VOXEL_METERS", 0.0)
        if voxel > 0:
            pcd = pcd.voxel_down_sample(voxel)
        if len(pcd.points) >= 30:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=_env_int("OBJECT_PRINTABLE_OUTLIER_NEIGHBORS", 12),
                std_ratio=_env_float("OBJECT_PRINTABLE_OUTLIER_STD_RATIO", 2.0),
            )
        if len(pcd.points) < minimum_points:
            warnings.append(f"Printable mesh skipped after outlier removal: only {len(pcd.points)} object points.")
            return None, None

        alpha = _printable_alpha_radius(pcd)
        candidates = [alpha, alpha * 1.25, alpha * 1.5, alpha * 2.0]
        mesh_candidates: list[tuple[str, object]] = []
        if requested_method not in {"poisson"}:
            alpha_mesh = _best_alpha_mesh(o3d, pcd, candidates)
            if alpha_mesh is not None and len(alpha_mesh.triangles):
                mesh_candidates.append(("printable_alpha", alpha_mesh))
        if requested_method not in {"printable_alpha", "alpha", "alpha_shape"}:
            poisson_mesh = _poisson_mesh(o3d, pcd)
            if poisson_mesh is not None and len(poisson_mesh.triangles):
                mesh_candidates.append(("printable_poisson", poisson_mesh))
        if not mesh_candidates:
            warnings.append("Printable mesh skipped: alpha shape and Poisson produced no triangles.")
            return None, None

        method, mesh = min(mesh_candidates, key=lambda candidate: _mesh_fidelity_score(candidate[1], pcd))
        mesh = _postprocess_printable_mesh(mesh, pcd, o3d)
        if preserve_color and len(mesh.vertices) and pcd.has_colors():
            _transfer_nearest_colors(mesh, pcd, o3d)
        output = output_dir / "scan_object_printable_mesh.ply"
        if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=True):
            warnings.append("Printable alpha mesh export failed; wrote point-cloud baseline only.")
            return None, None
        if not mesh.is_watertight():
            warnings.append("Printable alpha mesh is not watertight; it may need repair before 3D printing.")
        return output, method
    except Exception as exc:  # noqa: BLE001 - printable mesh is a best-effort path.
        warnings.append(f"Printable mesh skipped: {exc}")
        return None, None


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
        _append_warning_once(warnings, "Open3D is not installed; wrote point-cloud baseline only.")
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


def _mesh_method() -> str:
    return os.environ.get("MESH_METHOD", "printable_metric").strip().lower()


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


def _printable_alpha_radius(pcd) -> float:  # noqa: ANN001
    explicit = os.environ.get("OBJECT_ALPHA_METERS")
    if explicit:
        try:
            return max(1e-4, float(explicit))
        except ValueError:
            pass

    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    finite = distances[np.isfinite(distances) & (distances > 0)]
    bbox_extent = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    min_extent = float(np.max([np.min(bbox_extent[bbox_extent > 0]) if np.any(bbox_extent > 0) else 0.02, 0.02]))
    if finite.size:
        radius = float(np.median(finite)) * _env_float("OBJECT_ALPHA_NEIGHBOR_FACTOR", 8.0)
    else:
        radius = min_extent * 0.18
    lower = min_extent * _env_float("OBJECT_ALPHA_MIN_EXTENT_FRACTION", 0.12)
    upper = min_extent * _env_float("OBJECT_ALPHA_MAX_EXTENT_FRACTION", 0.5)
    return float(np.clip(radius, lower, upper))


def _trim_printable_outliers(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trim_percent = _env_float("OBJECT_PRINTABLE_TRIM_PERCENT", 1.0)
    if trim_percent <= 0 or trim_percent >= 10 or points.shape[0] < 200:
        return points, colors
    lower = np.percentile(points, trim_percent, axis=0)
    upper = np.percentile(points, 100.0 - trim_percent, axis=0)
    padding = np.maximum(
        (upper - lower) * _env_float("OBJECT_PRINTABLE_TRIM_PADDING_FRACTION", 0.02),
        _env_float("OBJECT_PRINTABLE_TRIM_PADDING_METERS", 0.0015),
    )
    keep = ((points >= lower - padding) & (points <= upper + padding)).all(axis=1)
    minimum_kept = max(80, int(points.shape[0] * 0.8))
    if int(keep.sum()) < minimum_kept:
        return points, colors
    return points[keep], colors[keep]


def _best_alpha_mesh(o3d, pcd, alpha_candidates: list[float]):  # noqa: ANN001, ANN201
    best = None
    best_score = float("inf")
    for alpha in alpha_candidates:
        if alpha <= 0:
            continue
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        except Exception:
            continue
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == 0:
            continue
        try:
            mesh.orient_triangles()
        except Exception:
            pass
        mesh.compute_vertex_normals()
        score = _mesh_fidelity_score(mesh, pcd)
        if score < best_score:
            best = mesh
            best_score = score
    return best


def _poisson_mesh(o3d, pcd):  # noqa: ANN001, ANN201
    points = np.asarray(pcd.points)
    if points.shape[0] < _env_int("OBJECT_POISSON_MIN_POINTS", 250):
        return None
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    finite = distances[np.isfinite(distances) & (distances > 0)]
    radius = float(np.median(finite)) * 5.0 if finite.size else 0.02
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(radius, 0.005),
            max_nn=_env_int("OBJECT_POISSON_NORMAL_NEIGHBORS", 30),
        )
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(_env_int("OBJECT_POISSON_ORIENT_NEIGHBORS", 20))
    except Exception:
        pass
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=_env_int("OBJECT_POISSON_DEPTH", 8),
        scale=_env_float("OBJECT_POISSON_SCALE", 1.05),
        linear_fit=True,
    )
    density_values = np.asarray(densities)
    if density_values.size:
        cutoff = np.quantile(density_values, _env_float("OBJECT_POISSON_DENSITY_TRIM", 0.03))
        mesh.remove_vertices_by_mask((density_values < cutoff).tolist())

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    padding = np.maximum(extent * _env_float("OBJECT_POISSON_CROP_PADDING", 0.04), 0.002)
    crop = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(bbox.min_bound) - padding,
        np.asarray(bbox.max_bound) + padding,
    )
    return mesh.crop(crop)


def _mesh_fidelity_score(mesh, pcd) -> float:  # noqa: ANN001
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        return float("inf")
    sample_count = min(12000, max(1000, len(pcd.points) * 2))
    sampled = mesh.sample_points_uniformly(number_of_points=sample_count)
    mesh_to_cloud = np.asarray(sampled.compute_point_cloud_distance(pcd))
    cloud_to_mesh = np.asarray(pcd.compute_point_cloud_distance(sampled))
    extent = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())
    diagonal = max(float(np.linalg.norm(extent)), 1e-6)
    distances = np.concatenate([mesh_to_cloud, cloud_to_mesh])
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("inf")
    score = (float(np.median(finite)) + 0.25 * float(np.percentile(finite, 90))) / diagonal
    if not mesh.is_watertight():
        score += _env_float("OBJECT_MESH_OPEN_SURFACE_PENALTY", 0.015)
    return score


def _postprocess_printable_mesh(mesh, pcd, o3d):  # noqa: ANN001, ANN201
    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_np = np.asarray(labels)
        counts_np = np.asarray(counts)
        if counts_np.size:
            keep_label = int(np.argmax(counts_np))
            mesh.remove_triangles_by_mask((labels_np != keep_label).tolist())
            mesh.remove_unreferenced_vertices()

    subdivision_iterations = _env_int("OBJECT_PRINTABLE_SUBDIVISION_ITERATIONS", 1)
    if subdivision_iterations > 0 and 0 < len(mesh.triangles) < _env_int("OBJECT_PRINTABLE_SUBDIVIDE_MAX_FACES", 5000):
        mesh = mesh.subdivide_loop(number_of_iterations=subdivision_iterations)

    smoothing_iterations = _env_int("OBJECT_PRINTABLE_SMOOTH_ITERATIONS", 5)
    if smoothing_iterations > 0 and len(mesh.triangles):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iterations)

    target_triangles = _env_int("OBJECT_PRINTABLE_MAX_TRIANGLES", 50000)
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    try:
        mesh.orient_triangles()
    except Exception:
        pass
    mesh.compute_vertex_normals()
    return mesh


def _transfer_nearest_colors(mesh, pcd, o3d) -> None:  # noqa: ANN001
    source_colors = np.asarray(pcd.colors)
    if source_colors.size == 0:
        return
    tree = o3d.geometry.KDTreeFlann(pcd)
    colors = []
    for vertex in np.asarray(mesh.vertices):
        _, indices, _ = tree.search_knn_vector_3d(vertex, 1)
        colors.append(source_colors[int(indices[0])] if indices else np.array([0.8, 0.8, 0.8]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0))


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


def _append_warning_once(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)
