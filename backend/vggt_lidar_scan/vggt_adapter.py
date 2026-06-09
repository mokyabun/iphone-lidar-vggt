from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import keyframe_indices
from .io import read_image
from .models import FrameRecord
from .ply import write_point_cloud_ply
from .segmentation import resize_mask
from .vggt_manager import configure_huggingface_cache, ensure_vggt_repo


@dataclass
class _VGGTRuntime:
    torch: Any
    model: Any
    load_and_preprocess_images: Any
    pose_encoding_to_extri_intri: Any
    unproject_depth_map_to_point_map: Any
    device: str
    dtype: Any


_RUNTIME_CACHE: _VGGTRuntime | None = None


def preload_vggt() -> str:
    runtime = _load_vggt_runtime()
    return f"VGGT loaded on {runtime.device} with dtype {runtime.dtype}"


def run_vggt(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
    preserve_color: bool = True,
    object_masks: dict[str, np.ndarray] | None = None,
) -> tuple[Path, int]:
    frames = _limit_vggt_frames(frames)
    image_dir = output_dir / "vggt_scene" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    for index, frame in enumerate(frames):
        source = root / frame.image_path
        target = image_dir / f"{index:06d}.jpg"
        if source.exists():
            shutil.copyfile(source, target)
        else:
            read_image(root, frame).save(target, quality=94)
        image_paths.append(target)

    runner = os.environ.get("VGGT_RUNNER")
    if runner:
        return _run_external_vggt(runner, image_dir, output_dir)
    return _run_python_vggt(image_paths, output_dir, frames, preserve_color, object_masks)


def _run_external_vggt(runner: str, image_dir: Path, output_dir: Path) -> tuple[Path, int]:
    vggt_output_dir = output_dir / "vggt_external"
    vggt_output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [runner, "--image-dir", str(image_dir), "--output-dir", str(vggt_output_dir)],
        check=True,
    )
    candidate = vggt_output_dir / "scan_vggt_points.ply"
    if not candidate.exists():
        raise RuntimeError(f"VGGT runner completed but did not create {candidate}")
    return candidate, _count_ply_vertices(candidate)


def _run_python_vggt(
    image_paths: list[Path],
    output_dir: Path,
    frames: list[FrameRecord],
    preserve_color: bool,
    object_masks: dict[str, np.ndarray] | None,
) -> tuple[Path, int]:
    runtime = _load_vggt_runtime()
    torch = runtime.torch

    if not image_paths:
        raise RuntimeError("No keyframe images available for VGGT")

    images = runtime.load_and_preprocess_images([str(path) for path in image_paths]).to(runtime.device, non_blocking=True)[None]

    with torch.inference_mode():
        with torch.amp.autocast(runtime.device, dtype=runtime.dtype, enabled=runtime.device == "cuda"):
            aggregated_tokens_list, ps_idx = runtime.model.aggregator(images)
            pose_enc = runtime.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = runtime.pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = runtime.model.depth_head(aggregated_tokens_list, images, ps_idx)
            point_map = runtime.unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

    point_map_np = _as_numpy(point_map)
    if point_map_np.ndim != 4 or point_map_np.shape[-1] != 3:
        raise RuntimeError(f"Unexpected VGGT point map shape: {point_map_np.shape}")
    points = point_map_np.reshape(-1, 3)
    confidence = _as_numpy(depth_conf).reshape(-1)
    keep = np.isfinite(points).all(axis=1)
    colors = _colors_for_vggt_points(image_paths, point_map_np.shape[2], point_map_np.shape[1]) if preserve_color else None
    if object_masks:
        object_keep = _vggt_object_keep(frames, object_masks, point_map_np.shape[2], point_map_np.shape[1])
        if object_keep.size == keep.size:
            keep &= object_keep
    if confidence.size == points.shape[0]:
        finite_confidence = confidence[np.isfinite(confidence)]
        if finite_confidence.size:
            keep &= confidence > np.percentile(finite_confidence, 60)
    points = points[keep].astype(np.float32)
    colors = colors[keep] if colors is not None and colors.shape[0] == keep.size else None

    output = output_dir / "scan_vggt_points.ply"
    write_point_cloud_ply(output, points, colors)
    return output, int(points.shape[0])


def _load_vggt_runtime() -> _VGGTRuntime:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    configure_huggingface_cache()
    ensure_vggt_repo()
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as exc:  # noqa: BLE001 - report missing optional stack clearly.
        raise RuntimeError(f"VGGT Python imports failed: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        allow_cpu = os.environ.get("VGGT_ALLOW_CPU", "0") in {"1", "true", "True"}
        if not allow_cpu:
            raise RuntimeError("VGGT direct adapter requires CUDA; set VGGT_ALLOW_CPU=1 only for slow development tests")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = torch.float32
    if device == "cuda":
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
        dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    _RUNTIME_CACHE = _VGGTRuntime(
        torch=torch,
        model=model,
        load_and_preprocess_images=load_and_preprocess_images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
        device=device,
        dtype=dtype,
    )
    return _RUNTIME_CACHE


def _limit_vggt_frames(frames: list[FrameRecord]) -> list[FrameRecord]:
    max_images = _env_int("VGGT_MAX_IMAGES", 12)
    if len(frames) <= max_images:
        return frames
    return [frames[index] for index in keyframe_indices(len(frames), max_images)]


def _count_ply_vertices(path: Path) -> int:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    return 0


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _colors_for_vggt_points(image_paths: list[Path], width: int, height: int) -> np.ndarray:
    from PIL import Image

    chunks: list[np.ndarray] = []
    for path in image_paths:
        rgb = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        chunks.append(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.uint8)


def _vggt_object_keep(frames: list[FrameRecord], object_masks: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for frame in frames:
        mask = object_masks.get(frame.frame_id)
        if mask is None:
            chunks.append(np.ones(width * height, dtype=bool))
        else:
            chunks.append(resize_mask(mask, width, height).reshape(-1))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=bool)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default
