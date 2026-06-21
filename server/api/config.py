from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return int(value) if value else default
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    run_root: Path = Path(os.environ.get("APP_RUN_ROOT", "runs/api"))
    max_frames: int = env_int("SCAN_MAX_FRAMES", 24)
    stride: int = env_int("SCAN_STRIDE", 4)
    confidence_minimum: int = env_int("SCAN_CONFIDENCE_MINIMUM", 1)
    reconviagen_command: str = os.environ.get("RECONVIAGEN_COMMAND", "")
    reconviagen_worker_url: str = os.environ.get("RECONVIAGEN_WORKER_URL", "")
    reconviagen_timeout_seconds: int = env_int("RECONVIAGEN_TIMEOUT_SECONDS", 2400)
    sam3_worker_url: str = os.environ.get("SAM3_WORKER_URL", "")
    sam3_timeout_seconds: int = env_int("SAM3_TIMEOUT_SECONDS", 900)
    sam3_center_box_fraction: float = env_float("SAM3_CENTER_BOX_FRACTION", 0.55)
    reconviagen_max_images: int = env_int("RECONVIAGEN_MAX_IMAGES", 6)
    reconviagen_input_size: int = env_int("RECONVIAGEN_INPUT_SIZE", 512)
    crop_padding: float = env_float("RECONVIAGEN_CROP_PADDING", 0.18)
    depth_band_m: float = env_float("OBJECT_DEPTH_BAND_METERS", 0.45)
    center_fraction: float = env_float("OBJECT_CENTER_FRACTION", 0.35)
    object_voxel_m: float = env_float("OBJECT_POINT_VOXEL_METERS", 0.006)
    alignment_samples: int = env_int("ALIGNMENT_SAMPLES", 6000)
    icp_iterations: int = env_int("ALIGNMENT_ICP_ITERATIONS", 8)
    print_stl: bool = env_bool("EXPORT_PRINT_STL", True)


def settings() -> Settings:
    return Settings()
