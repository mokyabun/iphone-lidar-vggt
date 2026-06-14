from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vggt_lidar_scan.models import FrameRecord
from vggt_lidar_scan.reconviagen import (
    _best_metric_transform,
    _nearest_distances,
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
