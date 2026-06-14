from __future__ import annotations

import itertools
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from .io import read_image
from .models import FrameRecord
from .segmentation import resize_mask


def run_generative_mesh(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
    lidar_points: np.ndarray,
    object_masks: dict[str, np.ndarray] | None,
    preserve_color: bool,
) -> tuple[Path, str]:
    if lidar_points.shape[0] < 80:
        raise RuntimeError("not enough metric LiDAR points for generative mesh alignment")

    input_path = prepare_generative_input(root, frames, output_dir, object_masks)
    raw_mesh_path, backend = _run_mesh_generator(input_path, output_dir)
    output_path = output_dir / "scan_object_pretty_mesh.ply"
    _convert_and_align_mesh(raw_mesh_path, output_path, lidar_points, preserve_color)
    return output_path, backend


def prepare_generative_input(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
    object_masks: dict[str, np.ndarray] | None,
) -> Path:
    from PIL import Image

    candidates: list[tuple[float, FrameRecord, np.ndarray, np.ndarray]] = []
    for frame in frames:
        image = np.asarray(read_image(root, frame).convert("RGB"), dtype=np.uint8)
        mask = object_masks.get(frame.frame_id) if object_masks else None
        if mask is None:
            continue
        full_mask = resize_mask(mask, image.shape[1], image.shape[0])
        ys, xs = np.nonzero(full_mask)
        if xs.size < 64:
            continue
        area_ratio = float(full_mask.mean())
        if area_ratio <= 0 or area_ratio > 0.8:
            continue
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        center = np.array([float(xs.mean()) / image.shape[1], float(ys.mean()) / image.shape[0]])
        center_score = max(0.0, 1.0 - float(np.linalg.norm(center - 0.5)))
        sharpness = _masked_sharpness(image, full_mask)
        score = bbox_area * (0.5 + center_score) * (1.0 + min(sharpness / 24.0, 2.0))
        candidates.append((score, frame, image, full_mask))

    if not candidates:
        raise RuntimeError("no usable object mask was available for generative reconstruction")

    _, _, image, mask = max(candidates, key=lambda candidate: candidate[0])
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    padding = int(round(side * _env_float("GENERATIVE_CROP_PADDING", 0.25)))
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    side = max(16, side + padding * 2)

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
    target_x1 = target_x0 + (source_x1 - source_x0)
    target_y1 = target_y0 + (source_y1 - source_y0)
    rgba[target_y0:target_y1, target_x0:target_x1, :3] = image[source_y0:source_y1, source_x0:source_x1]
    rgba[target_y0:target_y1, target_x0:target_x1, 3] = (
        mask[source_y0:source_y1, source_x0:source_x1].astype(np.uint8) * 255
    )

    size = _env_int("GENERATIVE_INPUT_SIZE", 512)
    resampling = getattr(Image, "Resampling", Image)
    output = output_dir / "scan_generative_input.png"
    Image.fromarray(rgba, mode="RGBA").resize((size, size), resampling.LANCZOS).save(output)
    return output


def _run_mesh_generator(input_path: Path, output_dir: Path) -> tuple[Path, str]:
    generated_dir = output_dir / "generative_raw"
    generated_dir.mkdir(parents=True, exist_ok=True)
    timeout = _env_int("GENERATIVE_MESH_TIMEOUT_SECONDS", 900)
    custom_runner = os.environ.get("GENERATIVE_MESH_RUNNER")
    if custom_runner:
        command = shlex.split(custom_runner) + [
            "--image",
            str(input_path),
            "--output-dir",
            str(generated_dir),
        ]
        subprocess.run(command, check=True, timeout=timeout)
        return _find_generated_mesh(generated_dir), "external"

    repo_dir = Path(os.environ.get("SPAR3D_REPO_DIR", "/workspace/cache/spar3d")).expanduser()
    if not (repo_dir / "spar3d").is_dir():
        raise RuntimeError(
            "SPAR3D is not prepared; set HF_TOKEN and APP_PREPARE_SPAR3D=1, "
            "or configure GENERATIVE_MESH_RUNNER"
        )
    python_bin = os.environ.get("SPAR3D_PYTHON", sys.executable)
    command = [
        python_bin,
        str(Path(__file__).with_name("spar3d_runner.py")),
        "--image",
        str(input_path),
        "--output-dir",
        str(generated_dir),
    ]
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{repo_dir}{os.pathsep}{python_path}" if python_path else str(repo_dir)
    print("[generative] SPAR3D inference started", flush=True)
    subprocess.run(command, check=True, timeout=timeout, cwd=repo_dir, env=environment)
    print("[generative] SPAR3D inference completed; aligning to LiDAR scale", flush=True)
    return _find_generated_mesh(generated_dir), "spar3d"


def _find_generated_mesh(output_dir: Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("**/mesh.glb", "**/mesh.ply", "**/mesh.obj", "**/*.glb", "**/*.ply", "**/*.obj"):
        candidates.extend(output_dir.glob(pattern))
    candidates = [path for path in candidates if path.is_file() and path.name != "points.ply"]
    if not candidates:
        raise RuntimeError(f"generative runner completed without a mesh in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _convert_and_align_mesh(
    source_path: Path,
    output_path: Path,
    lidar_points: np.ndarray,
    preserve_color: bool,
) -> None:
    try:
        import open3d as o3d  # type: ignore
        import trimesh
    except Exception as exc:
        raise RuntimeError(f"generative mesh conversion dependencies are unavailable: {exc}") from exc

    loaded = trimesh.load(source_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("generated asset did not contain triangle geometry")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError("generated asset format is unsupported")
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        raise RuntimeError("generated mesh was empty")

    sample_count = min(_env_int("GENERATIVE_ALIGNMENT_SAMPLES", 2500), max(500, mesh.faces.shape[0] * 2))
    samples, _ = trimesh.sample.sample_surface(mesh, sample_count)
    transform = _best_metric_transform(np.asarray(samples), lidar_points)
    vertices = _apply_transform(np.asarray(mesh.vertices), transform)

    output_mesh = o3d.geometry.TriangleMesh()
    output_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    output_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    if preserve_color:
        colors = _trimesh_vertex_colors(mesh)
        if colors is not None and colors.shape[0] == vertices.shape[0]:
            output_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    output_mesh.remove_degenerate_triangles()
    output_mesh.remove_duplicated_triangles()
    output_mesh.remove_duplicated_vertices()
    output_mesh.remove_unreferenced_vertices()
    output_mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(output_path), output_mesh, write_ascii=True):
        raise RuntimeError("failed to export aligned generative mesh")


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
                translation = target_center - source_center @ (rotation * scale)
                best_transform = np.eye(4, dtype=np.float64)
                best_transform[:3, :3] = rotation * scale
                best_transform[:3, 3] = translation
                best_score = score

    if best_transform is None:
        raise RuntimeError("could not align generated mesh to metric LiDAR points")
    return best_transform


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3] + transform[:3, 3]


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def _robust_extent(points: np.ndarray) -> np.ndarray:
    return np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)


def _symmetric_distance_score(source: np.ndarray, target: np.ndarray) -> float:
    source_distances = _nearest_distances(source, target)
    target_distances = _nearest_distances(target, source)
    combined = np.concatenate([source_distances, target_distances])
    diagonal = max(float(np.linalg.norm(_robust_extent(target))), 1e-6)
    return (float(np.median(combined)) + 0.25 * float(np.percentile(combined, 90))) / diagonal


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, query.shape[0], 256):
        values = query[start : start + 256]
        squared = ((values[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(chunks)


def _finite_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if finite.shape[0] <= maximum:
        return finite
    indices = np.linspace(0, finite.shape[0] - 1, maximum).round().astype(np.int64)
    return finite[indices]


def _trimesh_vertex_colors(mesh) -> np.ndarray | None:  # noqa: ANN001
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
