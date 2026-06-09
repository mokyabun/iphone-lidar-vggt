from __future__ import annotations

import json
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

