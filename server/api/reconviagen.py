from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .config import env_bool, settings


def generate_mesh(input_dir: Path, output_path: Path) -> None:
    cfg = settings()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if env_bool("RECONVIAGEN_MOCK", False):
        _log(f"mock mesh generation: output_path={output_path}")
        _write_mock_mesh(output_path)
        return
    if cfg.reconviagen_worker_url:
        _log(f"using ReconViaGen worker: url={cfg.reconviagen_worker_url} input_dir={input_dir} output_path={output_path}")
        _generate_with_worker(cfg.reconviagen_worker_url, input_dir, output_path)
        return
    if not cfg.reconviagen_command:
        raise RuntimeError(
            "ReconViaGen is not configured. Start the server with ./run.sh for the managed worker, or set "
            "RECONVIAGEN_COMMAND='python /path/to/reconviagen_runner.py --input-dir {input_dir} --output-path {output_path}'"
        )
    command = cfg.reconviagen_command.format(
        input_dir=str(input_dir),
        output_path=str(output_path),
    )
    _log(f"running ReconViaGen command: {command}")
    subprocess.run(
        shlex.split(command),
        check=True,
        timeout=cfg.reconviagen_timeout_seconds,
        env=os.environ.copy(),
    )
    if not output_path.exists():
        raise RuntimeError("ReconViaGen finished without writing the configured output mesh.")


def align_reconviagen_mesh(
    raw_mesh_path: Path,
    output_ply: Path,
    preview_glb: Path,
    print_stl: Path,
    lidar_points: np.ndarray,
) -> dict[str, object]:
    loaded = trimesh.load(raw_mesh_path, force="scene")
    mesh = _scene_to_mesh(loaded)
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        raise RuntimeError("ReconViaGen output mesh was empty.")

    sample_count = min(settings().alignment_samples, max(600, mesh.faces.shape[0] * 2))
    samples, _ = trimesh.sample.sample_surface(mesh, sample_count)
    transform = _best_metric_transform(np.asarray(samples), lidar_points)
    aligned_samples = _apply_transform(np.asarray(samples), transform)
    transform, aligned_samples = _refine_icp(transform, aligned_samples, lidar_points)
    rmse = _alignment_rmse(aligned_samples, lidar_points)
    scale = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))

    vertices = _apply_transform(np.asarray(mesh.vertices), transform)
    vertices += _center_on_reference(vertices, lidar_points)
    mesh.vertices = vertices
    _clean_mesh(mesh)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_ply, file_type="ply")
    mesh.export(preview_glb, file_type="glb")
    print_path: str | None = None
    watertight = bool(mesh.is_watertight)
    if settings().print_stl:
        printable = mesh.copy()
        printable.apply_scale(1000.0)
        printable.export(print_stl, file_type="stl")
        print_path = str(print_stl)

    return {
        "alignment_rmse_m": round(float(rmse), 5),
        "alignment_scale": round(scale, 6),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "aligned_object_extent_m": _robust_extent(np.asarray(mesh.vertices)).round(5).tolist(),
        "print_mesh_watertight": watertight,
        "print_stl_output": print_path,
    }


def export_raw_mesh(raw_mesh_path: Path, output_ply: Path) -> dict[str, object]:
    loaded = trimesh.load(raw_mesh_path, force="scene")
    mesh = _scene_to_mesh(loaded)
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        raise RuntimeError("ReconViaGen output mesh was empty.")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_ply, file_type="ply")
    return {
        "raw_mesh_vertices": int(mesh.vertices.shape[0]),
        "raw_mesh_faces": int(mesh.faces.shape[0]),
        "raw_object_extent_m": _robust_extent(np.asarray(mesh.vertices)).round(5).tolist(),
    }


def _generate_with_worker(worker_url: str, input_dir: Path, output_path: Path) -> None:
    body = json.dumps({"input_dir": str(input_dir), "output_path": str(output_path)}).encode()
    deadline = time.monotonic() + settings().reconviagen_timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        request = urllib.request.Request(
            worker_url.rstrip("/") + "/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        remaining = max(1, int(deadline - time.monotonic()))
        _log(f"worker request attempt={attempt} timeout_seconds={remaining}")
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=remaining) as response:
                result = json.loads(response.read())
            _log(f"worker response received in {time.monotonic() - started:.1f}s result={result}")
            if result.get("status") != "ok" or not output_path.exists():
                raise RuntimeError(result.get("error") or "ReconViaGen worker did not write a mesh.")
            return
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode(errors="ignore")
            _log(f"worker HTTP failure: status={exc.code} payload={payload}")
            raise RuntimeError(f"ReconViaGen worker failed: {payload}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"ReconViaGen worker request timed out: {exc}") from exc
            _log(f"worker unavailable, retrying in 5s: {exc}")
            time.sleep(5)


def _scene_to_mesh(loaded: object) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    if not isinstance(loaded, trimesh.Scene):
        raise RuntimeError("ReconViaGen output format is unsupported.")
    geometries = loaded.dump(concatenate=False)
    meshes = [geometry for geometry in geometries if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError("ReconViaGen output did not contain triangle geometry.")
    return trimesh.util.concatenate(meshes)


def _best_metric_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source = _finite_subsample(source_points, 1800)
    target = _finite_subsample(target_points, 1800)
    if source.shape[0] < 3 or target.shape[0] < 3:
        raise RuntimeError("Not enough points for metric alignment.")
    source_center, source_axes = _pca_frame(source)
    target_center, target_axes = _pca_frame(target)
    source_local = (source - source_center) @ source_axes
    target_extent = _robust_extent((target - target_center) @ target_axes)

    import itertools

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
        raise RuntimeError("Could not align ReconViaGen mesh to LiDAR points.")
    return best_transform


def _refine_icp(
    transform: np.ndarray,
    aligned_source: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = _finite_subsample(aligned_source, 3000)
    target = _finite_subsample(target_points, 5000)
    if source.shape[0] < 20 or target.shape[0] < 20:
        return transform, aligned_source
    tree = cKDTree(target)
    current = source.copy()
    refined = transform.copy()
    max_distance = max(0.008, float(np.linalg.norm(_robust_extent(target))) * 0.08)
    best_score = _symmetric_distance_score(current, target)
    for _ in range(settings().icp_iterations):
        distances, indices = tree.query(current, k=1)
        keep = distances <= min(max_distance, max(float(np.percentile(distances, 75)) * 1.5, 0.002))
        if int(keep.sum()) < 12:
            break
        rotation, translation = _rigid_transform(current[keep], target[indices[keep]])
        candidate = current @ rotation + translation
        score = _symmetric_distance_score(candidate, target)
        if score > best_score * 1.002:
            break
        current = candidate
        refined[:3, :3] = refined[:3, :3] @ rotation
        refined[:3, 3] = refined[:3, 3] @ rotation + translation
        best_score = score
    delta_rotation = np.linalg.solve(transform[:3, :3], refined[:3, :3])
    delta_translation = refined[:3, 3] - transform[:3, 3] @ delta_rotation
    return refined, aligned_source @ delta_rotation + delta_translation


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


def _alignment_rmse(source: np.ndarray, target: np.ndarray) -> float:
    distances = cKDTree(_finite_subsample(target, 5000)).query(_finite_subsample(source, 4000), k=1)[0]
    return float(np.sqrt(np.mean(np.square(distances)))) if distances.size else float("nan")


def _symmetric_distance_score(source: np.ndarray, target: np.ndarray) -> float:
    source = _finite_subsample(source, 2000)
    target = _finite_subsample(target, 2000)
    source_to_target = cKDTree(target).query(source, k=1)[0]
    target_to_source = cKDTree(source).query(target, k=1)[0]
    distances = np.concatenate([source_to_target, target_to_source])
    diagonal = max(float(np.linalg.norm(_robust_extent(target))), 1e-6)
    return (float(np.median(distances)) + 0.25 * float(np.percentile(distances, 90))) / diagonal


def _center_on_reference(vertices: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = np.median(_finite_subsample(reference, 8000), axis=0)
    bottom = float(np.percentile(vertices[:, 1], 1))
    return np.array([-center[0], -bottom, -center[2]], dtype=np.float64)


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3] + transform[:3, 3]


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, axes = np.linalg.eigh(np.cov(points - center, rowvar=False))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def _robust_extent(points: np.ndarray) -> np.ndarray:
    return np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)


def _finite_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] <= maximum:
        return values
    indices = np.linspace(0, values.shape[0] - 1, maximum).round().astype(np.int64)
    return values[indices]


def _clean_mesh(mesh: trimesh.Trimesh) -> None:
    try:
        valid_faces = np.asarray(mesh.unique_faces()) & np.asarray(mesh.nondegenerate_faces())
        mesh.update_faces(valid_faces)
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fix_normals()
    except Exception:
        pass


def _write_mock_mesh(output_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    mesh.export(output_path, file_type="glb")


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[reconviagen-client] {timestamp} {message}", flush=True)
