from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from spar3d.system import SPAR3D
    from spar3d.utils import get_device

    device = "cpu" if _env_bool("SPAR3D_USE_CPU", False) else get_device()
    low_vram = _env_bool("SPAR3D_LOW_VRAM", False)
    print(f"[spar3d] loading model on {device} low_vram={low_vram}", flush=True)
    model = SPAR3D.from_pretrained(
        os.environ.get("SPAR3D_MODEL_ID", "stabilityai/stable-point-aware-3d"),
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=low_vram,
    )
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGBA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[spar3d] reconstructing textured mesh", flush=True)
    with torch.no_grad():
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if "cuda" in device
            else nullcontext()
        ):
            mesh, _ = model.run_image(
                [image],
                bake_resolution=_env_int("SPAR3D_TEXTURE_RESOLUTION", 1024),
                remesh="none",
                vertex_count=-1,
                return_points=False,
            )
    mesh.export(output_dir / "mesh.glb", include_normals=True)
    print(f"[spar3d] wrote {output_dir / 'mesh.glb'}", flush=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


if __name__ == "__main__":
    main()
