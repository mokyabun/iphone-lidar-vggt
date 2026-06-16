from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


class ApiSettings(_EnvSettings):
    vggt_preload: bool = Field(False, validation_alias="VGGT_PRELOAD")
    vggt_runner: str | None = Field(None, validation_alias="VGGT_RUNNER")
    reconviagen_worker_url: str | None = Field(None, validation_alias="RECONVIAGEN_WORKER_URL")
    reconviagen_worker_error: Path = Field(
        Path("/workspace/cache/reconviagen-worker.error"),
        validation_alias="RECONVIAGEN_WORKER_ERROR",
    )


class ScanSettings(_EnvSettings):
    max_frames: int = Field(24, validation_alias="SCAN_MAX_FRAMES")
    stride: int = Field(4, validation_alias="SCAN_STRIDE")
    frame_workers: int = Field(4, validation_alias="SCAN_FRAME_WORKERS")
    run_tsdf: bool = Field(False, validation_alias="SCAN_RUN_TSDF")
    mesh_method: str = Field("printable_metric", validation_alias="MESH_METHOD")


class ObjectSettings(_EnvSettings):
    mask_backend: str = Field("sam3_depth", validation_alias="OBJECT_MASK_BACKEND")
    sam_model: str = Field("sam3.pt", validation_alias="OBJECT_SAM_MODEL")
    sam_max_frames: int = Field(3, validation_alias="OBJECT_SAM_MAX_FRAMES")
    mask_propagation: bool = Field(True, validation_alias="OBJECT_MASK_PROPAGATION")
    propagation_anchors: int = Field(2, validation_alias="OBJECT_PROPAGATION_ANCHORS")
    propagation_stride: int = Field(2, validation_alias="OBJECT_PROPAGATION_STRIDE")
    propagation_depth_tolerance_m: float = Field(0.08, validation_alias="OBJECT_PROPAGATION_DEPTH_TOLERANCE_METERS")
    propagation_dilation_px: int = Field(3, validation_alias="OBJECT_PROPAGATION_DILATION_PIXELS")
    center_fraction: float = Field(0.35, validation_alias="OBJECT_CENTER_FRACTION")
    depth_band_m: float = Field(0.45, validation_alias="OBJECT_DEPTH_BAND_METERS")
    min_mask_ratio: float = Field(0.002, validation_alias="OBJECT_MIN_MASK_RATIO")
    max_mask_ratio: float = Field(0.65, validation_alias="OBJECT_MAX_MASK_RATIO")
    remove_dominant_plane: bool = Field(True, validation_alias="OBJECT_REMOVE_DOMINANT_PLANE")
    plane_distance_m: float = Field(0.025, validation_alias="OBJECT_PLANE_DISTANCE_METERS")
    plane_sample_stride: int = Field(4, validation_alias="OBJECT_PLANE_SAMPLE_STRIDE")
    plane_ransac_iterations: int = Field(160, validation_alias="OBJECT_PLANE_RANSAC_ITERATIONS")


def api_settings() -> ApiSettings:
    return ApiSettings()


def scan_settings() -> ScanSettings:
    return ScanSettings()


def object_settings() -> ObjectSettings:
    return ObjectSettings()


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return max(1, int(value)) if value else default
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


def env_nonnegative_float(name: str, default: float) -> float:
    return max(0.0, env_float(name, default))


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default
