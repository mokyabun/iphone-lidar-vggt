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
    sam3_focus_box_fraction: float = env_float("SAM3_FOCUS_BOX_FRACTION", 0.10)
    sam3_negative_prompts: bool = env_bool("SAM3_NEGATIVE_PROMPTS", True)
    sam3_bottom_negative_fraction: float = env_float("SAM3_BOTTOM_NEGATIVE_FRACTION", 0.30)
    sam3_side_negative_fraction: float = env_float("SAM3_SIDE_NEGATIVE_FRACTION", 0.12)
    reconviagen_max_images: int = env_int("RECONVIAGEN_MAX_IMAGES", 6)
    reconviagen_input_size: int = env_int("RECONVIAGEN_INPUT_SIZE", 512)
    crop_padding: float = env_float("RECONVIAGEN_CROP_PADDING", 0.18)
    depth_band_m: float = env_float("OBJECT_DEPTH_BAND_METERS", 0.45)
    center_fraction: float = env_float("OBJECT_CENTER_FRACTION", 0.35)
    object_voxel_m: float = env_float("OBJECT_POINT_VOXEL_METERS", 0.006)
    alignment_samples: int = env_int("ALIGNMENT_SAMPLES", 6000)
    icp_iterations: int = env_int("ALIGNMENT_ICP_ITERATIONS", 8)
    print_stl: bool = env_bool("EXPORT_PRINT_STL", True)
    mesh_cleanup_min_component_faces: int = env_int("MESH_CLEANUP_MIN_COMPONENT_FACES", 8)
    mesh_cleanup_min_component_face_ratio: float = env_float("MESH_CLEANUP_MIN_COMPONENT_FACE_RATIO", 0.0)
    mesh_cleanup_max_components: int = env_int("MESH_CLEANUP_MAX_COMPONENTS", 0)
    mesh_post_floor_cleanup_min_component_faces: int = env_int("MESH_POST_FLOOR_CLEANUP_MIN_COMPONENT_FACES", 8)
    mesh_post_floor_cleanup_bottom_fraction: float = env_float("MESH_POST_FLOOR_CLEANUP_BOTTOM_FRACTION", 0.12)
    mesh_post_floor_cleanup_max_thickness_fraction: float = env_float(
        "MESH_POST_FLOOR_CLEANUP_MAX_THICKNESS_FRACTION", 0.07
    )
    mesh_post_floor_cleanup_min_normal_y: float = env_float("MESH_POST_FLOOR_CLEANUP_MIN_NORMAL_Y", 0.75)
    mesh_post_floor_cleanup_max_remove_face_ratio: float = env_float(
        "MESH_POST_FLOOR_CLEANUP_MAX_REMOVE_FACE_RATIO", 0.3
    )
    mesh_floor_trim_min_faces: int = env_int("MESH_FLOOR_TRIM_MIN_FACES", 700)
    mesh_floor_trim_bottom_fraction: float = env_float("MESH_FLOOR_TRIM_BOTTOM_FRACTION", 0.35)
    mesh_floor_trim_top_fraction: float = env_float("MESH_FLOOR_TRIM_TOP_FRACTION", 0.65)
    mesh_floor_trim_max_thickness_fraction: float = env_float("MESH_FLOOR_TRIM_MAX_THICKNESS_FRACTION", 0.35)
    mesh_floor_trim_min_normal_y: float = env_float("MESH_FLOOR_TRIM_MIN_NORMAL_Y", 0.82)
    mesh_floor_trim_min_footprint_ratio: float = env_float("MESH_FLOOR_TRIM_MIN_FOOTPRINT_RATIO", 0.04)
    mesh_floor_trim_max_remove_face_ratio: float = env_float("MESH_FLOOR_TRIM_MAX_REMOVE_FACE_RATIO", 0.35)


def settings() -> Settings:
    return Settings()
