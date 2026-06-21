from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


@dataclass(frozen=True)
class SAM3Settings:
    repo_dir: Path = Path(os.environ.get("SAM3_REPO_DIR", "/workspace/cache/sam3"))
    confidence_threshold: float = _env_float("SAM3_CONFIDENCE_THRESHOLD", 0.35)
    require_cuda: bool = _env_bool("SAM3_REQUIRE_CUDA", True)
    mock: bool = _env_bool("SAM3_MOCK", False)


def sam3_settings() -> SAM3Settings:
    return SAM3Settings()
