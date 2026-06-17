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

    repo_dir: Path = Field(Path("/workspace/cache/ReconViaGen"), validation_alias="RECONVIAGEN_REPO_DIR")
    low_vram: bool = Field(False, validation_alias="RECONVIAGEN_LOW_VRAM")
    require_cuda: bool = Field(True, validation_alias="RECONVIAGEN_REQUIRE_CUDA")
    torch_num_threads: int = Field(4, validation_alias="RECONVIAGEN_TORCH_NUM_THREADS")
    ss_model: str = Field("Stable-X/trellis-vggt-v0-2", validation_alias="RECONVIAGEN_SS_MODEL")
    trellis_model: str = Field("microsoft/TRELLIS.2-4B", validation_alias="RECONVIAGEN_TRELLIS_MODEL")
    seed: int = Field(0, validation_alias="RECONVIAGEN_SEED")
    pipeline_type: str = Field("512", validation_alias="RECONVIAGEN_PIPELINE_TYPE")
    ss_source: str = Field("direct", validation_alias="RECONVIAGEN_SS_SOURCE")
    preprocess_image: bool = Field(False, validation_alias="RECONVIAGEN_PREPROCESS_IMAGE")
    max_num_tokens: int = Field(49152, validation_alias="RECONVIAGEN_MAX_NUM_TOKENS")
    decimation_target: int = Field(200000, validation_alias="RECONVIAGEN_DECIMATION_TARGET")
    texture_size: int = Field(1024, validation_alias="RECONVIAGEN_TEXTURE_SIZE")
    ss_steps: int = Field(8, validation_alias="RECONVIAGEN_SS_STEPS")
    ss_guidance: float = Field(7.5, validation_alias="RECONVIAGEN_SS_GUIDANCE")
    slat_steps: int = Field(8, validation_alias="RECONVIAGEN_SLAT_STEPS")
    slat_guidance: float = Field(7.5, validation_alias="RECONVIAGEN_SLAT_GUIDANCE")
    shape_steps: int = Field(8, validation_alias="RECONVIAGEN_SHAPE_STEPS")
    shape_guidance: float = Field(7.5, validation_alias="RECONVIAGEN_SHAPE_GUIDANCE")
    texture_steps: int = Field(8, validation_alias="RECONVIAGEN_TEXTURE_STEPS")
    texture_guidance: float = Field(1.0, validation_alias="RECONVIAGEN_TEXTURE_GUIDANCE")


def reconviagen_settings() -> ReconViaGenSettings:
    return ReconViaGenSettings()
