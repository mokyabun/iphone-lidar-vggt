from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VGGTSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    runner: str | None = Field(None, validation_alias="VGGT_RUNNER")
    max_images: int = Field(12, validation_alias="VGGT_MAX_IMAGES")
    allow_cpu: bool = Field(False, validation_alias="VGGT_ALLOW_CPU")
    empty_cache_after_run: bool = Field(False, validation_alias="VGGT_EMPTY_CACHE_AFTER_RUN")
    cache_root: Path = Field(Path.home() / ".cache" / "vggt-lidar", validation_alias="VGGT_CACHE_ROOT")
    repo_dir: Path | None = Field(None, validation_alias="VGGT_REPO_DIR")
    repo_url: str = Field("https://github.com/facebookresearch/vggt.git", validation_alias="VGGT_REPO_URL")
    auto_download: bool = Field(True, validation_alias="VGGT_AUTO_DOWNLOAD")
    install_repo: bool = Field(True, validation_alias="VGGT_INSTALL_REPO")
    download_weights: bool = Field(True, validation_alias="VGGT_DOWNLOAD_WEIGHTS")
    hf_home: Path | None = Field(None, validation_alias="HF_HOME")
    torch_num_threads: int = Field(4, validation_alias="TORCH_NUM_THREADS")
    torch_num_interop_threads: int = Field(1, validation_alias="TORCH_NUM_INTEROP_THREADS")
    torch_cudnn_benchmark: bool = Field(True, validation_alias="TORCH_CUDNN_BENCHMARK")
    torch_float32_matmul_precision: str = Field("high", validation_alias="TORCH_FLOAT32_MATMUL_PRECISION")


def vggt_settings() -> VGGTSettings:
    return VGGTSettings()
