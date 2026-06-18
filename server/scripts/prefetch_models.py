"""Prefetch ReconViaGen model artifacts into the shared Hugging Face / torch cache.

Run inside the reconviagen env. Model repos are taken from RECONVIAGEN_* env vars
set by env_reconviagen.sh.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError


def log(message: str) -> None:
    print(f"[prefetch-models] {message}", flush=True)


def summarize(path: Path) -> str:
    try:
        files = [item for item in path.rglob("*") if item.is_file()]
    except OSError as exc:
        return f"path={path} summary_error={exc}"
    total = 0
    for item in files:
        try:
            total += item.stat().st_size
        except OSError:
            pass
    return f"path={path} files={len(files)} size={total / (1024 ** 3):.2f}GiB"


def resolve_snapshot(label: str, model: str) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        log(f"{label}: using local path {summarize(path)}")
        return path

    log(f"{label}: snapshot_download start repo={model}")
    started = time.perf_counter()
    snapshot = Path(snapshot_download(model))
    log(f"{label}: snapshot_download done in {time.perf_counter() - started:.1f}s {summarize(snapshot)}")
    return snapshot


def walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(walk_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(walk_strings(item))
        return strings
    return []


def looks_like_hf_repo(value: str) -> bool:
    if value.startswith((".", "/", "http://", "https://")):
        return False
    if value.startswith(("ckpts/", "checkpoints/", "models/", "weights/")):
        return False
    if "/" not in value:
        return False
    if value.count("/") != 1:
        return False
    if value.endswith((".json", ".safetensors", ".pth", ".pt", ".ckpt")):
        return False
    return True


def prefetch_nested_repos(snapshot: Path) -> None:
    for config_path in snapshot.rglob("pipeline.json"):
        try:
            config = json.loads(config_path.read_text())
        except Exception as exc:
            log(f"nested repo scan skipped for {config_path}: {exc}")
            continue
        repos = sorted({value for value in walk_strings(config) if looks_like_hf_repo(value)})
        for repo in repos:
            try:
                resolve_snapshot(f"nested model from {config_path.name}", repo)
            except RepositoryNotFoundError:
                log(f"nested model from {config_path.name}: skipping unresolved repo={repo}")


def prefetch_dinov2() -> None:
    if os.environ.get("RECONVIAGEN_PREFETCH_DINOV2", "1").lower() in {"0", "false", "no", "off"}:
        log("DINOv2 torch hub prefetch skipped.")
        return

    import torch

    model_name = os.environ.get("RECONVIAGEN_DINOV2_MODEL", "dinov2_vitl14_reg")
    log(f"DINOv2 torch hub prefetch start model={model_name} torch_hub_dir={torch.hub.get_dir()}")
    started = time.perf_counter()
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, trust_repo=True)
    except TypeError:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.cpu()
    del model
    gc.collect()
    log(f"DINOv2 torch hub prefetch done in {time.perf_counter() - started:.1f}s")


def main() -> None:
    log(
        "cache roots: "
        f"HF_HOME={os.environ.get('HF_HOME')} "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')} "
        f"TORCH_HOME={os.environ.get('TORCH_HOME')}"
    )

    ss_snapshot = resolve_snapshot("sparse-structure model", os.environ["RECONVIAGEN_SS_MODEL"])
    trellis_snapshot = resolve_snapshot("TRELLIS.2 model", os.environ["RECONVIAGEN_TRELLIS_MODEL"])
    resolve_snapshot("VGGT model", os.environ["RECONVIAGEN_VGGT_MODEL"])
    resolve_snapshot("BiRefNet model", os.environ["RECONVIAGEN_BIREFNET_MODEL"])
    prefetch_nested_repos(ss_snapshot)
    prefetch_nested_repos(trellis_snapshot)
    prefetch_dinov2()


if __name__ == "__main__":
    main()
