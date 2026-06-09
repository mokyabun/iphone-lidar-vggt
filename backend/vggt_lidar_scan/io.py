from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .models import FrameRecord, ScanMetadata


class ScanPackageError(RuntimeError):
    """Raised when a scan package is missing required files or metadata."""


@contextmanager
def open_scan_package(path: Path) -> Iterator[Path]:
    path = Path(path)
    if path.is_dir():
        yield path
        return

    if not path.exists():
        raise ScanPackageError(f"Scan package does not exist: {path}")
    if path.suffix.lower() != ".zip":
        raise ScanPackageError(f"Expected a directory or .zip package: {path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="scan_package_"))
    try:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temp_dir)
        root = _find_package_root(temp_dir)
        yield root
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _find_package_root(path: Path) -> Path:
    if (path / "frames.jsonl").exists():
        return path
    candidates = [candidate for candidate in path.iterdir() if candidate.is_dir() and (candidate / "frames.jsonl").exists()]
    if len(candidates) == 1:
        return candidates[0]
    raise ScanPackageError("Could not find frames.jsonl in extracted scan package")


def read_metadata(root: Path) -> ScanMetadata:
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return ScanMetadata()
    return ScanMetadata.model_validate_json(metadata_path.read_text())


def read_frames(root: Path) -> list[FrameRecord]:
    frames_path = root / "frames.jsonl"
    if not frames_path.exists():
        raise ScanPackageError(f"Missing frames.jsonl: {frames_path}")

    frames: list[FrameRecord] = []
    for line_number, line in enumerate(frames_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            frames.append(FrameRecord.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - preserve malformed line context.
            raise ScanPackageError(f"Invalid frame record at line {line_number}: {exc}") from exc

    if not frames:
        raise ScanPackageError("Scan package contains no frames")
    return frames


def read_depth(root: Path, frame: FrameRecord) -> np.ndarray:
    depth_path = root / frame.depth_path
    if not depth_path.exists():
        raise ScanPackageError(f"Missing depth file: {depth_path}")
    depth = np.fromfile(depth_path, dtype=np.float32)
    expected = frame.depth_width * frame.depth_height
    if depth.size != expected:
        raise ScanPackageError(f"Depth file {depth_path} has {depth.size} values, expected {expected}")
    return depth.reshape((frame.depth_height, frame.depth_width))


def read_confidence(root: Path, frame: FrameRecord) -> np.ndarray | None:
    if frame.confidence_path is None:
        return None
    confidence_path = root / frame.confidence_path
    if not confidence_path.exists():
        return None
    confidence = np.fromfile(confidence_path, dtype=np.uint8)
    expected = frame.depth_width * frame.depth_height
    if confidence.size != expected:
        return None
    return confidence.reshape((frame.depth_height, frame.depth_width))


def read_image(root: Path, frame: FrameRecord) -> Image.Image:
    image_path = root / frame.image_path
    if not image_path.exists():
        raise ScanPackageError(f"Missing image file: {image_path}")
    return Image.open(image_path).convert("RGB")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

