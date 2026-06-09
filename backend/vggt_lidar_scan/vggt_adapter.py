from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .io import read_image
from .models import FrameRecord
from .ply import write_point_cloud_ply
from .vggt_manager import configure_huggingface_cache, ensure_vggt_repo


def run_vggt(root: Path, frames: list[FrameRecord], output_dir: Path) -> tuple[Path, int]:
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
    return _run_python_vggt(image_paths, output_dir)


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


def _run_python_vggt(image_paths: list[Path], output_dir: Path) -> tuple[Path, int]:
    configure_huggingface_cache()
    ensure_vggt_repo()
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as exc:  # noqa: BLE001 - report missing optional stack clearly.
        raise RuntimeError("Install the VGGT package or set VGGT_RUNNER for GPU inference") from exc

    if not image_paths:
        raise RuntimeError("No keyframe images available for VGGT")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        allow_cpu = os.environ.get("VGGT_ALLOW_CPU", "0") in {"1", "true", "True"}
        if not allow_cpu:
            raise RuntimeError("VGGT direct adapter requires CUDA; set VGGT_ALLOW_CPU=1 only for slow development tests")

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)[None]

    with torch.no_grad():
        autocast_device = "cuda" if device == "cuda" else "cpu"
        with torch.amp.autocast(autocast_device, dtype=dtype, enabled=device == "cuda"):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

    points = np.asarray(point_map.detach().cpu()).reshape(-1, 3)
    confidence = np.asarray(depth_conf.detach().cpu()).reshape(-1)
    keep = np.isfinite(points).all(axis=1) & (confidence > np.percentile(confidence, 60))
    points = points[keep].astype(np.float32)

    output = output_dir / "scan_vggt_points.ply"
    write_point_cloud_ply(output, points)
    return output, int(points.shape[0])


def _count_ply_vertices(path: Path) -> int:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    return 0
