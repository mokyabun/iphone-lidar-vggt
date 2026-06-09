from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Matrix3 = list[list[float]]
Matrix4 = list[list[float]]


class ScanMetadata(BaseModel):
    app_version: str = "0.1.0"
    package_version: int = 1
    device_model: str | None = None
    os_version: str | None = None
    lidar_supported: bool = True
    scan_mode: Literal["object", "space"] = "object"
    notes: str | None = None


class FrameRecord(BaseModel):
    frame_id: str
    timestamp: float
    image_path: str
    depth_path: str
    confidence_path: str | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    depth_width: int = Field(gt=0)
    depth_height: int = Field(gt=0)
    intrinsics_depth: Matrix3
    camera_to_world: Matrix4
    orientation: str = "landscapeRight"

    @property
    def image_file(self) -> Path:
        return Path(self.image_path)

    @property
    def depth_file(self) -> Path:
        return Path(self.depth_path)


class ReconstructionMetrics(BaseModel):
    frame_count: int
    selected_keyframes: int
    lidar_points: int
    vggt_points: int = 0
    mesh_vertices: int = 0
    mesh_faces: int = 0
    mesh_method: str | None = None
    final_output_type: Literal["point_cloud", "mesh"] = "point_cloud"
    object_mask_backend: str | None = None
    camera_path_m: float | None = None
    camera_extent_m: list[float] | None = None
    lidar_bounds_min_m: list[float] | None = None
    lidar_bounds_max_m: list[float] | None = None
    lidar_extent_m: list[float] | None = None
    object_bounds_min_m: list[float] | None = None
    object_bounds_max_m: list[float] | None = None
    object_extent_m: list[float] | None = None
    final_output: str
    lidar_output: str | None = None
    mesh_output: str | None = None
    tsdf_output: str | None = None
    vggt_output: str | None = None
    warnings: list[str] = Field(default_factory=list)
