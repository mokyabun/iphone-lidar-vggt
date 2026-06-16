from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orchestration.io import read_depth, read_frames


def test_read_frames_and_depth(tmp_path: Path) -> None:
    (tmp_path / "depth").mkdir()
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    depth.tofile(tmp_path / "depth" / "frame_000001.float32")
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
    (tmp_path / "frames.jsonl").write_text(json.dumps(frame) + "\n")

    frames = read_frames(tmp_path)
    loaded = read_depth(tmp_path, frames[0])

    assert len(frames) == 1
    assert np.allclose(loaded, depth)

