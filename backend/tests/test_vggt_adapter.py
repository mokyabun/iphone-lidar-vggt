from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image

from vggt_lidar_scan.models import FrameRecord
from vggt_lidar_scan.vggt_adapter import _as_numpy, _limit_vggt_frames, _vggt_image_paths


def test_as_numpy_accepts_numpy_array() -> None:
    value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    converted = _as_numpy(value)

    assert converted is value


def test_as_numpy_accepts_tensor_like_object() -> None:
    class TensorLike:
        def __init__(self) -> None:
            self.value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        def detach(self) -> TensorLike:
            return self

        def float(self) -> TensorLike:
            return self

        def cpu(self) -> TensorLike:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    converted = _as_numpy(TensorLike())

    assert np.allclose(converted, [[1.0, 2.0, 3.0]])


def test_limit_vggt_frames_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("VGGT_MAX_IMAGES", "3")
    frames = [
        FrameRecord(
            frame_id=f"frame_{index}",
            timestamp=float(index),
            image_path="images/frame.jpg",
            depth_path="depth/frame.float32",
            image_width=4,
            image_height=4,
            depth_width=2,
            depth_height=2,
            intrinsics_depth=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            camera_to_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        for index in range(8)
    ]

    limited = _limit_vggt_frames(frames)

    assert len(limited) == 3
    assert [frame.frame_id for frame in limited] == ["frame_0", "frame_3", "frame_7"]


def test_vggt_image_paths_reuses_existing_images_for_direct_adapter(tmp_path: Path) -> None:
    root = tmp_path / "package"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(root / "images" / "frame.jpg")
    frame = _frame_record("images/frame.jpg")

    paths = _vggt_image_paths(root, [frame], tmp_path / "out", materialize=False)

    assert paths == [root / "images" / "frame.jpg"]
    assert not (tmp_path / "out" / "vggt_scene").exists()


def test_vggt_image_paths_materializes_for_external_runner(tmp_path: Path) -> None:
    root = tmp_path / "package"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(120, 80, 40)).save(root / "images" / "frame.jpg")
    frame = _frame_record("images/frame.jpg")

    paths = _vggt_image_paths(root, [frame], tmp_path / "out", materialize=True)

    assert paths == [tmp_path / "out" / "vggt_scene" / "images" / "000000.jpg"]
    assert paths[0].exists()


def _frame_record(image_path: str) -> FrameRecord:
    return FrameRecord(
        frame_id="frame_0",
        timestamp=0.0,
        image_path=image_path,
        depth_path="depth/frame.float32",
        image_width=4,
        image_height=4,
        depth_width=2,
        depth_height=2,
        intrinsics_depth=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        camera_to_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )
