from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReconViaGenSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    worker_url: str | None = Field(None, validation_alias="RECONVIAGEN_WORKER_URL")
    worker_error: Path | None = Field(None, validation_alias="RECONVIAGEN_WORKER_ERROR")
    repo_dir: Path = Field(Path("/workspace/cache/ReconViaGen"), validation_alias="RECONVIAGEN_REPO_DIR")
    python_bin: str | None = Field(None, validation_alias="RECONVIAGEN_PYTHON")
    timeout_seconds: int = Field(1800, validation_alias="RECONVIAGEN_TIMEOUT_SECONDS")
    max_images: int = Field(6, validation_alias="RECONVIAGEN_MAX_IMAGES")
    input_size: int = Field(1024, validation_alias="RECONVIAGEN_INPUT_SIZE")
    crop_padding: float = Field(0.2, validation_alias="RECONVIAGEN_CROP_PADDING")
    low_vram: bool = Field(True, validation_alias="RECONVIAGEN_LOW_VRAM")
    ss_model: str = Field("Stable-X/trellis-vggt-v0-2", validation_alias="RECONVIAGEN_SS_MODEL")
    trellis_model: str = Field("microsoft/TRELLIS.2-4B", validation_alias="RECONVIAGEN_TRELLIS_MODEL")
    seed: int = Field(0, validation_alias="RECONVIAGEN_SEED")
    pipeline_type: str = Field("1024_cascade", validation_alias="RECONVIAGEN_PIPELINE_TYPE")
    ss_source: str = Field("mesh", validation_alias="RECONVIAGEN_SS_SOURCE")
    decimation_target: int = Field(500000, validation_alias="RECONVIAGEN_DECIMATION_TARGET")
    texture_size: int = Field(2048, validation_alias="RECONVIAGEN_TEXTURE_SIZE")
    ss_steps: int = Field(12, validation_alias="RECONVIAGEN_SS_STEPS")
    ss_guidance: float = Field(7.5, validation_alias="RECONVIAGEN_SS_GUIDANCE")
    shape_steps: int = Field(12, validation_alias="RECONVIAGEN_SHAPE_STEPS")
    texture_steps: int = Field(12, validation_alias="RECONVIAGEN_TEXTURE_STEPS")
    alignment_samples: int = Field(2500, validation_alias="AI_ALIGNMENT_SAMPLES")
    icp_samples: int = Field(3500, validation_alias="AI_ICP_SAMPLES")
    icp_target_samples: int = Field(5000, validation_alias="AI_ICP_TARGET_SAMPLES")
    icp_iterations: int = Field(20, validation_alias="AI_ICP_ITERATIONS")
    icp_max_distance_m: float | None = Field(None, validation_alias="AI_ICP_MAX_DISTANCE_METERS")
    support_band_m: float = Field(0.06, validation_alias="AI_SUPPORT_BAND_METERS")
    print_voxel_repair: bool = Field(True, validation_alias="AI_PRINT_VOXEL_REPAIR")
    print_voxel_m: float | None = Field(None, validation_alias="AI_PRINT_VOXEL_METERS")


def reconviagen_settings() -> ReconViaGenSettings:
    return ReconViaGenSettings()
