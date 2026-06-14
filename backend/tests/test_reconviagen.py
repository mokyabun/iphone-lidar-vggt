from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vggt_lidar_scan.models import FrameRecord
from vggt_lidar_scan.reconviagen import (
    _alignment_rmse,
    _best_metric_transform,
    _export_print_stl,
    _nearest_distances,
    _object_asset_normalization,
    _prepare_print_mesh,
    _refine_metric_transform_icp,
    prepare_multiview_input,
)


def test_metric_alignment_recovers_uniform_scale_and_axis_orientation() -> None:
    rng = np.random.default_rng(21)
    target = rng.normal(size=(150, 3)) * np.array([0.035, 0.075, 0.022])
    target[:35] += np.array([0.02, -0.04, 0.015])
    rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    source = target @ rotation * 3.8 + np.array([0.8, -1.1, 0.45])

    transform = _best_metric_transform(source, target)
    aligned = source @ transform[:3, :3] + transform[:3, 3]

    assert np.median(_nearest_distances(aligned, target)) < 1e-5


def test_prepare_multiview_input_writes_diverse_rgba_views(tmp_path: Path) -> None:
    root = tmp_path / "scan"
    (root / "images").mkdir(parents=True)
    frames: list[FrameRecord] = []
    masks: dict[str, np.ndarray] = {}
    camera_positions = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    for index, position in enumerate(camera_positions):
        frame_id = f"frame_{index:06d}"
        Image.new("RGB", (64, 64), color=(80 + index * 20, 120, 60)).save(root / "images" / f"{frame_id}.png")
        transform = np.eye(4)
        transform[:3, 3] = position
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                timestamp=float(index),
                image_path=f"images/{frame_id}.png",
                depth_path=f"depth/{frame_id}.float32",
                image_width=64,
                image_height=64,
                depth_width=16,
                depth_height=16,
                intrinsics_depth=np.eye(3).tolist(),
                camera_to_world=transform.tolist(),
            )
        )
        mask = np.zeros((16, 16), dtype=bool)
        mask[3:13, 4:12] = True
        masks[frame_id] = mask

    lidar_points = np.random.default_rng(0).normal(size=(100, 3)) * 0.03
    outputs = prepare_multiview_input(root, frames, tmp_path / "input", lidar_points, masks)

    assert len(outputs) == 4
    assert all(Image.open(path).mode == "RGBA" for path in outputs)
    assert all(np.asarray(Image.open(path))[:, :, 3].max() == 255 for path in outputs)


def test_icp_refinement_reduces_metric_alignment_error() -> None:
    rng = np.random.default_rng(9)
    target = rng.normal(size=(500, 3)) * np.array([0.08, 0.12, 0.05])
    angle = np.deg2rad(5.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    source = target @ rotation + np.array([0.012, -0.006, 0.009])
    initial = np.eye(4)

    refined, aligned = _refine_metric_transform_icp(initial, source, target)

    assert _alignment_rmse(aligned, target) < _alignment_rmse(source, target)
    assert not np.allclose(refined, initial)


def test_object_asset_normalization_centers_object_and_places_support_at_zero() -> None:
    rng = np.random.default_rng(4)
    object_points = rng.uniform([-0.1, 0.32, -0.08], [0.1, 0.55, 0.08], size=(400, 3))
    support = rng.uniform([-0.13, 0.318, -0.11], [0.13, 0.322, 0.11], size=(300, 3))
    vertices = object_points.copy()

    offset = _object_asset_normalization(vertices, object_points, np.concatenate([object_points, support]))
    normalized = vertices + offset

    assert abs(np.median(normalized[:, 0])) < 0.01
    assert abs(np.median(normalized[:, 2])) < 0.01
    assert abs(0.32 + offset[1]) < 0.01


def test_voxel_print_repair_preserves_metric_transform_and_exports_mm(tmp_path: Path, monkeypatch) -> None:
    import trimesh

    source = trimesh.creation.box(extents=[0.12, 0.08, 0.05])
    source.update_faces(np.arange(len(source.faces) - 4))
    monkeypatch.setattr(trimesh.repair, "fill_holes", lambda mesh: False)
    monkeypatch.setenv("AI_PRINT_VOXEL_METERS", "0.005")

    repaired, watertight = _prepare_print_mesh(source, trimesh)
    output = tmp_path / "print.stl"
    _export_print_stl(repaired, output)
    exported = trimesh.load(output)

    assert watertight is True
    assert np.allclose(repaired.extents, source.extents, atol=0.012)
    assert np.allclose(exported.extents, repaired.extents * 1000.0, atol=0.1)
