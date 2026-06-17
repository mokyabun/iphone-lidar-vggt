from __future__ import annotations

import json
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .models import FrameRecord


@contextmanager
def extracted_scan_package(package_path: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="scan-package-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(package_path) as archive:
            archive.extractall(root)
        nested = root / "ScanPackage"
        yield nested if nested.exists() else root


def read_frames(root: Path) -> list[FrameRecord]:
    frames_path = root / "frames.jsonl"
    frames: list[FrameRecord] = []
    for line in frames_path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        frames.append(
            FrameRecord(
                frame_id=raw["frame_id"],
                image_path=raw["image_path"],
                depth_path=raw["depth_path"],
                confidence_path=raw.get("confidence_path"),
                image_width=int(raw["image_width"]),
                image_height=int(raw["image_height"]),
                depth_width=int(raw["depth_width"]),
                depth_height=int(raw["depth_height"]),
                intrinsics_depth=raw["intrinsics_depth"],
                camera_to_world=raw["camera_to_world"],
            )
        )
    return frames


def read_depth(root: Path, frame: FrameRecord) -> np.ndarray:
    data = np.fromfile(root / frame.depth_path, dtype=np.float32)
    return data.reshape((frame.depth_height, frame.depth_width))


def read_confidence(root: Path, frame: FrameRecord) -> np.ndarray | None:
    if not frame.confidence_path:
        return None
    path = root / frame.confidence_path
    if not path.exists():
        return None
    return np.fromfile(path, dtype=np.uint8).reshape((frame.depth_height, frame.depth_width))


def read_image(root: Path, frame: FrameRecord) -> Image.Image:
    return Image.open(root / frame.image_path).convert("RGB")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
