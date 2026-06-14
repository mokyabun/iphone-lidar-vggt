from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_image
from .models import FrameRecord
from .segmentation import resize_mask


@dataclass(frozen=True)
class ReconViaGenResult:
    mesh_path: Path
    preview_glb_path: Path
    print_stl_path: Path | None
    print_mesh_watertight: bool
    alignment_rmse_m: float
    alignment_scale: float


def run_reconviagen(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
    lidar_points: np.ndarray,
    scene_points: np.ndarray,
    object_masks: dict[str, np.ndarray] | None,
    preserve_color: bool,
) -> ReconViaGenResult:
    if lidar_points.shape[0] < 80:
        raise RuntimeError("not enough metric LiDAR points for AI mesh alignment")

    input_dir = output_dir / "reconviagen_input"
    prepare_multiview_input(root, frames, input_dir, lidar_points, object_masks)
    raw_output = output_dir / "reconviagen_raw.glb"
    _generate_mesh(input_dir, raw_output)
    final_output = output_dir / "scan_object_ai_mesh.ply"
    preview_output = output_dir / "scan_object_preview.glb"
    print_output = output_dir / "scan_object_print.stl"
    return _convert_and_align_mesh(
        raw_output,
        final_output,
        preview_output,
        print_output,
        lidar_points,
        scene_points,
        preserve_color,
    )


def prepare_multiview_input(
    root: Path,
    frames: list[FrameRecord],
    input_dir: Path,
    lidar_points: np.ndarray,
    object_masks: dict[str, np.ndarray] | None,
) -> list[Path]:
    from PIL import Image

    input_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    object_center = np.median(lidar_points, axis=0)
    for frame in frames:
        mask = object_masks.get(frame.frame_id) if object_masks else None
        if mask is None:
            continue
        image = np.asarray(read_image(root, frame).convert("RGB"), dtype=np.uint8)
        full_mask = resize_mask(mask, image.shape[1], image.shape[0])
        ys, xs = np.nonzero(full_mask)
        if xs.size < 64:
            continue
        area_ratio = float(full_mask.mean())
        if not 0.002 <= area_ratio <= 0.8:
            continue
        camera_center = np.asarray(frame.camera_to_world, dtype=np.float64)[:3, 3]
        direction = camera_center - object_center
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-6:
            continue
        direction /= direction_norm
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        center = np.array([xs.mean() / image.shape[1], ys.mean() / image.shape[0]])
        center_score = max(0.0, 1.0 - float(np.linalg.norm(center - 0.5)))
        sharpness = _masked_sharpness(image, full_mask)
        quality = bbox_area * (0.5 + center_score) * (1.0 + min(sharpness / 24.0, 2.0))
        candidates.append(
            {
                "frame": frame,
                "image": image,
                "mask": full_mask,
                "direction": direction,
                "quality": quality,
            }
        )

    if not candidates:
        raise RuntimeError("no usable masked views were available for ReconViaGen")

    selected = _select_diverse_views(candidates, _env_int("RECONVIAGEN_MAX_IMAGES", 6))
    selected = _order_views(selected)
    output_paths: list[Path] = []
    size = _env_int("RECONVIAGEN_INPUT_SIZE", 1024)
    resampling = getattr(Image, "Resampling", Image)
    for index, candidate in enumerate(selected):
        rgba = _crop_rgba(candidate["image"], candidate["mask"])
        output = input_dir / f"view_{index:02d}.png"
        Image.fromarray(rgba).resize((size, size), resampling.LANCZOS).save(output)
        output_paths.append(output)

    manifest = {
        "view_count": len(output_paths),
        "frame_ids": [candidate["frame"].frame_id for candidate in selected],
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_paths


def _select_diverse_views(candidates: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    maximum = min(maximum, len(candidates))
    first = max(candidates, key=lambda candidate: float(candidate["quality"]))
    selected = [first]
    remaining = [candidate for candidate in candidates if candidate is not first]
    best_quality = max(float(candidate["quality"]) for candidate in candidates)
    while remaining and len(selected) < maximum:
        def score(candidate: dict[str, object]) -> float:
            direction = np.asarray(candidate["direction"])
            angular_distance = min(
                1.0 - float(np.clip(direction @ np.asarray(chosen["direction"]), -1.0, 1.0))
                for chosen in selected
            )
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
    x_axis, y_axis = axes[:2]
    angles = np.arctan2(directions @ y_axis, directions @ x_axis)
    return [candidate for _, candidate in sorted(zip(angles, candidates), key=lambda item: item[0])]


def _crop_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    padding = int(round(side * _env_float("RECONVIAGEN_CROP_PADDING", 0.2)))
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
    rgba[target_y0:target_y1, target_x0:target_x1, 3] = (
        mask[source_y0:source_y1, source_x0:source_x1].astype(np.uint8) * 255
    )
    return rgba


def _generate_mesh(input_dir: Path, output_path: Path) -> None:
    worker_url = os.environ.get("RECONVIAGEN_WORKER_URL")
    if worker_url:
        body = json.dumps({"input_dir": str(input_dir), "output_path": str(output_path)}).encode()
        timeout = _env_int("RECONVIAGEN_TIMEOUT_SECONDS", 1800)
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while True:
            error_path = os.environ.get("RECONVIAGEN_WORKER_ERROR")
            if error_path and Path(error_path).exists():
                message = Path(error_path).read_text().strip()
                raise RuntimeError(f"ReconViaGen worker failed to start: {message}")
            request = urllib.request.Request(
                worker_url.rstrip("/") + "/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=max(1, int(deadline - time.monotonic()))) as response:
                    result = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                try:
                    payload = json.loads(exc.read())
                    message = payload.get("error", str(exc))
                except Exception:
                    message = str(exc)
                raise RuntimeError(f"ReconViaGen generation failed: {message}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"ReconViaGen worker request failed: {last_error}") from last_error
                time.sleep(5)
        if result.get("status") != "ok" or not output_path.exists():
            raise RuntimeError(result.get("error") or "ReconViaGen worker did not write a mesh")
        return

    repo_dir = Path(os.environ.get("RECONVIAGEN_REPO_DIR", "/workspace/cache/ReconViaGen")).expanduser()
    python_bin = os.environ.get("RECONVIAGEN_PYTHON", sys.executable)
    runner = Path(__file__).with_name("reconviagen_worker.py")
    if not (repo_dir / "trellis" / "pipelines" / "trellis_hybrid_pipeline.py").exists():
        raise RuntimeError("ReconViaGen v0.5 is not prepared")
    subprocess.run(
        [
            python_bin,
            str(runner),
            "--once",
            "--input-dir",
            str(input_dir),
            "--output-path",
            str(output_path),
        ],
        check=True,
        timeout=_env_int("RECONVIAGEN_TIMEOUT_SECONDS", 1800),
        cwd=repo_dir,
        env=os.environ.copy(),
    )
    if not output_path.exists():
        raise RuntimeError("ReconViaGen completed without writing a mesh")


def _convert_and_align_mesh(
    source_path: Path,
    output_path: Path,
    preview_glb_path: Path,
    print_stl_path: Path,
    lidar_points: np.ndarray,
    scene_points: np.ndarray,
    preserve_color: bool,
) -> ReconViaGenResult:
    try:
        import open3d as o3d  # type: ignore
        import trimesh
    except Exception as exc:
        raise RuntimeError(f"AI mesh conversion dependencies are unavailable: {exc}") from exc

    loaded = trimesh.load(source_path, force="scene")
    mesh = _scene_to_mesh(loaded, trimesh)
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        raise RuntimeError("ReconViaGen output mesh was empty")

    sample_count = min(_env_int("AI_ALIGNMENT_SAMPLES", 2500), max(500, mesh.faces.shape[0] * 2))
    samples, _ = trimesh.sample.sample_surface(mesh, sample_count)
    transform = _best_metric_transform(np.asarray(samples), lidar_points)
    aligned_samples = np.asarray(samples) @ transform[:3, :3] + transform[:3, 3]
    transform, aligned_samples = _refine_metric_transform_icp(transform, aligned_samples, lidar_points)
    alignment_rmse = _alignment_rmse(aligned_samples, lidar_points)
    alignment_scale = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
    vertices = np.asarray(mesh.vertices) @ transform[:3, :3] + transform[:3, 3]
    normalization = _object_asset_normalization(vertices, lidar_points, scene_points)
    vertices = vertices + normalization
    mesh.vertices = vertices
    _clean_trimesh(mesh)
    mesh.fix_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not preserve_color:
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.tile(np.array([[220, 220, 220, 255]], dtype=np.uint8), (vertices.shape[0], 1)),
        )

    preview_glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(preview_glb_path, file_type="glb")

    output_mesh = o3d.geometry.TriangleMesh()
    output_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    output_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    if preserve_color:
        colors = _vertex_colors(mesh)
        if colors is not None and colors.shape[0] == vertices.shape[0]:
            output_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    output_mesh.remove_degenerate_triangles()
    output_mesh.remove_duplicated_triangles()
    output_mesh.remove_duplicated_vertices()
    output_mesh.remove_unreferenced_vertices()
    output_mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(output_path), output_mesh, write_ascii=True):
        raise RuntimeError("failed to export the aligned ReconViaGen mesh")

    print_mesh, watertight = _prepare_print_mesh(mesh, trimesh)
    actual_print_path: Path | None = None
    if print_mesh is not None:
        print_mesh.export(print_stl_path, file_type="stl")
        actual_print_path = print_stl_path
    return ReconViaGenResult(
        mesh_path=output_path,
        preview_glb_path=preview_glb_path,
        print_stl_path=actual_print_path,
        print_mesh_watertight=watertight,
        alignment_rmse_m=round(alignment_rmse, 5),
        alignment_scale=round(alignment_scale, 6),
    )


def _scene_to_mesh(loaded, trimesh):  # noqa: ANN001
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    if not isinstance(loaded, trimesh.Scene):
        raise RuntimeError("ReconViaGen output format is unsupported")
    try:
        mesh = loaded.to_geometry()
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
    except Exception:
        pass
    geometries = loaded.dump(concatenate=False)
    meshes = [geometry for geometry in geometries if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError("ReconViaGen output did not contain triangle geometry")
    return trimesh.util.concatenate(meshes)


def _refine_metric_transform_icp(
    transform: np.ndarray,
    aligned_source: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = _finite_subsample(aligned_source, _env_int("AI_ICP_SAMPLES", 3500))
    target = _finite_subsample(target_points, _env_int("AI_ICP_TARGET_SAMPLES", 5000))
    if source.shape[0] < 20 or target.shape[0] < 20:
        return transform, aligned_source

    target_diagonal = max(float(np.linalg.norm(_robust_extent(target))), 1e-4)
    maximum_distance = _env_float("AI_ICP_MAX_DISTANCE_METERS", max(0.008, target_diagonal * 0.08))
    iterations = _env_int("AI_ICP_ITERATIONS", 20)
    current = source.copy()
    refined = transform.copy()
    best_score = _symmetric_distance_score(current, target)
    for _ in range(iterations):
        nearest_indices, distances = _nearest_matches(current, target)
        finite = np.isfinite(distances)
        if not np.any(finite):
            break
        adaptive = min(maximum_distance, float(np.percentile(distances[finite], 75)) * 1.5)
        keep = finite & (distances <= max(adaptive, 0.002))
        if np.count_nonzero(keep) < 12:
            break
        delta_rotation, delta_translation = _rigid_transform(current[keep], target[nearest_indices[keep]])
        candidate = current @ delta_rotation + delta_translation
        candidate_score = _symmetric_distance_score(candidate, target)
        if candidate_score > best_score * 1.002:
            break
        current = candidate
        refined[:3, :3] = refined[:3, :3] @ delta_rotation
        refined[:3, 3] = refined[:3, 3] @ delta_rotation + delta_translation
        if abs(best_score - candidate_score) < 1e-5:
            best_score = candidate_score
            break
        best_score = candidate_score

    all_aligned = aligned_source
    delta_rotation = np.linalg.solve(transform[:3, :3], refined[:3, :3])
    delta_translation = refined[:3, 3] - transform[:3, 3] @ delta_rotation
    all_aligned = all_aligned @ delta_rotation + delta_translation
    return refined, all_aligned


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def _nearest_matches(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for start in range(0, query.shape[0], 256):
        values = query[start : start + 256]
        squared = ((values[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(squared, axis=1)
        indices.append(nearest)
        distances.append(np.sqrt(squared[np.arange(nearest.shape[0]), nearest]))
    return np.concatenate(indices), np.concatenate(distances)


def _alignment_rmse(source: np.ndarray, target: np.ndarray) -> float:
    distances = _nearest_distances(_finite_subsample(source, 4000), _finite_subsample(target, 5000))
    finite = distances[np.isfinite(distances)]
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else float("nan")


def _object_asset_normalization(
    vertices: np.ndarray,
    object_points: np.ndarray,
    scene_points: np.ndarray,
) -> np.ndarray:
    object_values = _finite_subsample(object_points, 8000)
    scene_values = _finite_subsample(scene_points, 12000)
    center = np.median(object_values, axis=0)
    mesh_bottom = float(np.percentile(vertices[:, 1], 1))
    support_y = _support_plane_height(object_values, scene_values)
    floor_y = support_y if support_y is not None and abs(support_y - mesh_bottom) < 0.01 else mesh_bottom
    return np.array([-center[0], -floor_y, -center[2]], dtype=np.float64)


def _support_plane_height(object_points: np.ndarray, scene_points: np.ndarray) -> float | None:
    if object_points.shape[0] < 20 or scene_points.shape[0] < 50:
        return None
    lower = np.percentile(object_points, 2, axis=0)
    upper = np.percentile(object_points, 98, axis=0)
    xz_padding = np.maximum((upper[[0, 2]] - lower[[0, 2]]) * 0.2, 0.03)
    object_bottom = float(np.percentile(object_points[:, 1], 2))
    within_xz = (
        (scene_points[:, 0] >= lower[0] - xz_padding[0])
        & (scene_points[:, 0] <= upper[0] + xz_padding[0])
        & (scene_points[:, 2] >= lower[2] - xz_padding[1])
        & (scene_points[:, 2] <= upper[2] + xz_padding[1])
    )
    vertical_band = np.abs(scene_points[:, 1] - object_bottom) <= _env_float("AI_SUPPORT_BAND_METERS", 0.06)
    candidates = scene_points[within_xz & vertical_band, 1]
    if candidates.size < 20:
        return None
    return float(np.median(candidates))


def _prepare_print_mesh(mesh, trimesh):  # noqa: ANN001
    printable = mesh.copy()
    printable.visual = trimesh.visual.ColorVisuals(mesh=printable)
    components = printable.split(only_watertight=False)
    if components:
        printable = max(components, key=lambda component: float(component.area))
    _clean_trimesh(printable)
    try:
        trimesh.repair.fill_holes(printable)
        trimesh.repair.fix_normals(printable)
    except Exception:
        pass
    if printable.is_watertight:
        return printable, True

    if not _env_bool("AI_PRINT_VOXEL_REPAIR", True):
        return printable, False
    extent = np.asarray(printable.extents, dtype=np.float64)
    minimum_extent = float(np.min(extent[extent > 0])) if np.any(extent > 0) else 0.1
    pitch = _env_float("AI_PRINT_VOXEL_METERS", max(0.0015, minimum_extent / 180.0))
    try:
        repaired = printable.voxelized(pitch=pitch).fill().marching_cubes
        repaired.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(repaired)
        return repaired, bool(repaired.is_watertight)
    except Exception:
        return printable, False


def _clean_trimesh(mesh) -> None:  # noqa: ANN001
    try:
        valid_faces = np.asarray(mesh.unique_faces()) & np.asarray(mesh.nondegenerate_faces())
        mesh.update_faces(valid_faces)
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()


def _best_metric_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source = _finite_subsample(source_points, 1800)
    target = _finite_subsample(target_points, 1800)
    if source.shape[0] < 3 or target.shape[0] < 3:
        raise RuntimeError("not enough points for metric alignment")
    source_center, source_axes = _pca_frame(source)
    target_center, target_axes = _pca_frame(target)
    source_local = (source - source_center) @ source_axes
    target_extent = _robust_extent((target - target_center) @ target_axes)

    best_score = float("inf")
    best_transform: np.ndarray | None = None
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            local_rotation = permutation_matrix @ np.diag(signs)
            if np.linalg.det(local_rotation) < 0.5:
                continue
            candidate_local = source_local @ local_rotation
            source_extent = _robust_extent(candidate_local)
            valid = source_extent > 1e-6
            if not np.any(valid):
                continue
            scale = float(np.median(target_extent[valid] / source_extent[valid]))
            candidate = candidate_local * scale @ target_axes.T + target_center
            score = _symmetric_distance_score(candidate, target)
            if score < best_score:
                rotation = source_axes @ local_rotation @ target_axes.T
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotation * scale
                transform[:3, 3] = target_center - source_center @ transform[:3, :3]
                best_transform = transform
                best_score = score
    if best_transform is None:
        raise RuntimeError("could not align ReconViaGen mesh to LiDAR points")
    return best_transform


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, axes = np.linalg.eigh(np.cov(points - center, rowvar=False))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def _robust_extent(points: np.ndarray) -> np.ndarray:
    return np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)


def _symmetric_distance_score(source: np.ndarray, target: np.ndarray) -> float:
    distances = np.concatenate([_nearest_distances(source, target), _nearest_distances(target, source)])
    diagonal = max(float(np.linalg.norm(_robust_extent(target))), 1e-6)
    return (float(np.median(distances)) + 0.25 * float(np.percentile(distances, 90))) / diagonal


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, query.shape[0], 256):
        values = query[start : start + 256]
        squared = ((values[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(chunks)


def _finite_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] <= maximum:
        return values
    indices = np.linspace(0, values.shape[0] - 1, maximum).round().astype(np.int64)
    return values[indices]


def _vertex_colors(mesh) -> np.ndarray | None:  # noqa: ANN001
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors)
    except Exception:
        return None
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None
    return colors[:, :3].astype(np.uint8)


def _masked_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    gray = image.astype(np.float32).mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    values = np.concatenate([horizontal[mask[:, 1:]], vertical[mask[1:, :]]])
    return float(np.mean(values)) if values.size else 0.0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return max(1, int(value)) if value else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}
