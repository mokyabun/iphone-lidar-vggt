from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

from vggt_lidar_scan.reconstruct import reconstruct_scan


def test_reconstruct_scan_writes_final_ply(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "images").mkdir(parents=True)
    (package / "depth").mkdir()
    (package / "confidence").mkdir()

    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(package / "images" / "frame_000001.jpg")
    np.ones((2, 2), dtype=np.float32).tofile(package / "depth" / "frame_000001.float32")
    np.full((2, 2), 2, dtype=np.uint8).tofile(package / "confidence" / "frame_000001.uint8")

    frame = {
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "image_path": "images/frame_000001.jpg",
        "depth_path": "depth/frame_000001.float32",
        "confidence_path": "confidence/frame_000001.uint8",
        "image_width": 4,
        "image_height": 4,
        "depth_width": 2,
        "depth_height": 2,
        "intrinsics_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "orientation": "landscapeRight",
    }
    (package / "frames.jsonl").write_text(json.dumps(frame) + "\n")

    metrics = reconstruct_scan(package, tmp_path / "out", stride=1)

    assert metrics.lidar_points == 4
    assert (tmp_path / "out" / "scan_final.ply").exists()


def test_reconstruct_scan_falls_back_when_open3d_tsdf_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SCAN_RUN_TSDF", "1")
    package = tmp_path / "package"
    (package / "images").mkdir(parents=True)
    (package / "depth").mkdir()

    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(package / "images" / "frame_000001.jpg")
    np.ones((2, 2), dtype=np.float32).tofile(package / "depth" / "frame_000001.float32")

    frame = {
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "image_path": "images/frame_000001.jpg",
        "depth_path": "depth/frame_000001.float32",
        "confidence_path": None,
        "image_width": 4,
        "image_height": 4,
        "depth_width": 2,
        "depth_height": 2,
        "intrinsics_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "orientation": "landscapeRight",
    }
    (package / "frames.jsonl").write_text(json.dumps(frame) + "\n")

    class FailingVolume:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("synthetic TSDF failure")

    fake_open3d = types.SimpleNamespace(
        pipelines=types.SimpleNamespace(
            integration=types.SimpleNamespace(
                ScalableTSDFVolume=FailingVolume,
                TSDFVolumeColorType=types.SimpleNamespace(RGB8="RGB8"),
            )
        )
    )
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)

    metrics = reconstruct_scan(package, tmp_path / "out", stride=1)

    assert metrics.lidar_points == 4
    assert metrics.tsdf_output is None
    assert any("Open3D TSDF skipped" in warning for warning in metrics.warnings)
    assert (tmp_path / "out" / "scan_final.ply").exists()


def test_reconstruct_scan_prefers_mesh_when_requested(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "package"
    (package / "images").mkdir(parents=True)
    (package / "depth").mkdir()

    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(package / "images" / "frame_000001.jpg")
    np.ones((2, 2), dtype=np.float32).tofile(package / "depth" / "frame_000001.float32")

    frame = {
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "image_path": "images/frame_000001.jpg",
        "depth_path": "depth/frame_000001.float32",
        "confidence_path": None,
        "image_width": 4,
        "image_height": 4,
        "depth_width": 2,
        "depth_height": 2,
        "intrinsics_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "orientation": "landscapeRight",
    }
    (package / "frames.jsonl").write_text(json.dumps(frame) + "\n")

    def fake_tsdf(root, frames, output_dir, warnings, preserve_color=True, object_masks=None):
        path = output_dir / "scan_object_mesh.ply"
        path.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                    "element face 1",
                    "property list uchar int vertex_indices",
                    "end_header",
                    "0 0 0 255 0 0",
                    "1 0 0 0 255 0",
                    "0 1 0 0 0 255",
                    "3 0 1 2",
                ]
            )
            + "\n"
        )
        return path

    monkeypatch.setattr("vggt_lidar_scan.reconstruct.try_open3d_tsdf", fake_tsdf)

    metrics = reconstruct_scan(package, tmp_path / "out", stride=1, reconstruct_mesh=True)

    assert metrics.final_output_type == "mesh"
    assert metrics.mesh_vertices == 3
    assert metrics.mesh_faces == 1
    assert "element face 1" in (tmp_path / "out" / "scan_final.ply").read_text()


def test_reconstruct_scan_falls_back_when_ai_mesh_fails(tmp_path: Path, monkeypatch) -> None:
    package = _write_test_package(tmp_path)
    metric_mesh = _write_test_mesh(tmp_path / "metric.ply")
    monkeypatch.setattr(
        "vggt_lidar_scan.reconstruct.build_object_masks",
        lambda root, frames: {"frame_000001": np.ones((2, 2), dtype=bool)},
    )
    monkeypatch.setattr(
        "vggt_lidar_scan.reconstruct.build_mesh_output",
        lambda *args, **kwargs: (metric_mesh, None, "printable_alpha"),
    )
    monkeypatch.setattr(
        "vggt_lidar_scan.reconstruct.run_reconviagen",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic AI failure")),
    )

    metrics = reconstruct_scan(package, tmp_path / "out", stride=1, ai_mesh=True)

    assert metrics.ai_mesh_requested is True
    assert metrics.ai_mesh_used is False
    assert metrics.final_output_source == "printable_alpha"
    assert any("AI mesh skipped: synthetic AI failure" in warning for warning in metrics.warnings)


def test_reconstruct_scan_uses_reconviagen_mesh(tmp_path: Path, monkeypatch) -> None:
    package = _write_test_package(tmp_path)
    metric_mesh = _write_test_mesh(tmp_path / "metric.ply")
    ai_mesh = _write_test_mesh(tmp_path / "ai.ply")
    monkeypatch.setattr(
        "vggt_lidar_scan.reconstruct.build_object_masks",
        lambda root, frames: {"frame_000001": np.ones((2, 2), dtype=bool)},
    )
    monkeypatch.setattr(
        "vggt_lidar_scan.reconstruct.build_mesh_output",
        lambda *args, **kwargs: (metric_mesh, None, "printable_alpha"),
    )
    monkeypatch.setattr("vggt_lidar_scan.reconstruct.run_reconviagen", lambda *args, **kwargs: ai_mesh)

    metrics = reconstruct_scan(package, tmp_path / "out", stride=1, ai_mesh=True)

    assert metrics.ai_mesh_used is True
    assert metrics.final_output_source == "reconviagen_v05_metric_aligned"
    assert metrics.mesh_faces == 1
    assert metrics.ai_mesh_output == str(ai_mesh)


def _write_test_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    (package / "images").mkdir(parents=True)
    (package / "depth").mkdir()
    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(package / "images" / "frame_000001.jpg")
    np.ones((2, 2), dtype=np.float32).tofile(package / "depth" / "frame_000001.float32")
    frame = {
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "image_path": "images/frame_000001.jpg",
        "depth_path": "depth/frame_000001.float32",
        "confidence_path": None,
        "image_width": 4,
        "image_height": 4,
        "depth_width": 2,
        "depth_height": 2,
        "intrinsics_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "orientation": "landscapeRight",
    }
    (package / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    return package


def _write_test_mesh(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        )
        + "\n"
    )
    return path
