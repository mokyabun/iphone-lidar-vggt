from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    image_path: str
    depth_path: str
    confidence_path: str | None
    image_width: int
    image_height: int
    depth_width: int
    depth_height: int
    intrinsics_depth: list[list[float]]
    camera_to_world: list[list[float]]


@dataclass(frozen=True)
class ReconstructionResult:
    final_output: Path
    preview_glb_output: Path | None
    print_stl_output: Path | None
    lidar_reference_output: Path
    metrics: dict[str, object]


@dataclass(frozen=True)
class ReconstructionOptions:
    enable_sam3_object_masking: bool = False
    enable_lidar_scale_alignment: bool = True
    sam3_text_prompt: str = ""
