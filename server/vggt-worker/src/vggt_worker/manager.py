from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from .settings import vggt_settings


DEFAULT_REPO_URL = "https://github.com/facebookresearch/vggt.git"
DEFAULT_MODEL_ID = "facebook/VGGT-1B"


def default_cache_root() -> Path:
    return vggt_settings().cache_root.expanduser()


def configure_huggingface_cache() -> Path:
    settings = vggt_settings()
    hf_home = (settings.hf_home or (settings.cache_root / "huggingface")).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    return hf_home


def ensure_vggt_repo(auto_download: bool | None = None) -> Path | None:
    settings = vggt_settings()
    repo_dir = (settings.repo_dir or (settings.cache_root / "vggt")).expanduser()
    if repo_dir.exists():
        sync_vggt_repo(repo_dir)
        ensure_vggt_package_installed(repo_dir)
        _prepend_python_path(repo_dir)
        return repo_dir

    if importlib.util.find_spec("vggt") is not None:
        return None

    if auto_download is None:
        auto_download = settings.auto_download
    if not auto_download:
        return None

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", settings.repo_ref, settings.repo_url, str(repo_dir)],
        check=True,
    )
    ensure_vggt_package_installed(repo_dir)
    _prepend_python_path(repo_dir)
    return repo_dir


def sync_vggt_repo(repo_dir: Path) -> None:
    settings = vggt_settings()
    if not settings.repo_update or not (repo_dir / ".git").exists():
        return
    subprocess.run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", settings.repo_url], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", settings.repo_ref], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "FETCH_HEAD"], check=True)


def vggt_repo_revision(repo_dir: Path) -> str:
    if not (repo_dir / ".git").exists():
        return "local"
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def ensure_vggt_package_installed(repo_dir: Path) -> None:
    if not vggt_settings().install_repo:
        return
    marker = repo_dir / ".vggt_lidar_installed"
    pyproject = repo_dir / "pyproject.toml"
    revision = vggt_repo_revision(repo_dir)
    if marker.exists() and marker.read_text().strip() == revision:
        return
    if not pyproject.exists():
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)], check=True)
    marker.write_text(f"{revision}\n")


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
    download_weights = vggt_settings().download_weights
    prepared = prepare_vggt(download_weights=download_weights)
    print(f"VGGT repo: {prepared['repo']}")
    print(f"VGGT weights: {prepared['weights']}")
