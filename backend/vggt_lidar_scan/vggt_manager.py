from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/facebookresearch/vggt.git"
DEFAULT_MODEL_ID = "facebook/VGGT-1B"


def default_cache_root() -> Path:
    return Path(os.environ.get("VGGT_CACHE_ROOT", Path.home() / ".cache" / "vggt-lidar"))


def configure_huggingface_cache() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", default_cache_root() / "huggingface")).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    return hf_home


def ensure_vggt_repo(auto_download: bool | None = None) -> Path | None:
    if importlib.util.find_spec("vggt") is not None:
        return None

    repo_dir = Path(os.environ.get("VGGT_REPO_DIR", default_cache_root() / "vggt")).expanduser()
    if repo_dir.exists():
        _prepend_python_path(repo_dir)
        return repo_dir

    if auto_download is None:
        auto_download = os.environ.get("VGGT_AUTO_DOWNLOAD", "1") not in {"0", "false", "False"}
    if not auto_download:
        return None

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = os.environ.get("VGGT_REPO_URL", DEFAULT_REPO_URL)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    _prepend_python_path(repo_dir)
    return repo_dir


def ensure_vggt_weights(model_id: str = DEFAULT_MODEL_ID) -> Path:
    hf_home = configure_huggingface_cache()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency.
        raise RuntimeError("Install huggingface-hub to auto-download VGGT weights") from exc

    return Path(snapshot_download(repo_id=model_id, cache_dir=str(hf_home / "hub")))


def prepare_vggt(download_weights: bool = True) -> dict[str, str | None]:
    repo = ensure_vggt_repo(auto_download=True)
    weights = ensure_vggt_weights() if download_weights else None
    return {
        "repo": str(repo) if repo else None,
        "weights": str(weights) if weights else None,
    }


def _prepend_python_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def main() -> None:
    prepared = prepare_vggt(download_weights=True)
    print(f"VGGT repo: {prepared['repo']}")
    print(f"VGGT weights: {prepared['weights']}")
