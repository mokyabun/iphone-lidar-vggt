from __future__ import annotations

import gc
import sys
from pathlib import Path

from .settings import reconviagen_settings


class ReconViaGenService:
    def __init__(self) -> None:
        settings = reconviagen_settings()
        repo_dir = settings.repo_dir.expanduser()
        vendor_dirs = [
            repo_dir,
            repo_dir / "wheels" / "TRELLIS.2",
            repo_dir / "wheels" / "vggt",
            repo_dir / "wheels" / "dust3r",
            repo_dir / "wheels" / "mast3r",
        ]
        for vendor_dir in reversed(vendor_dirs):
            sys.path.insert(0, str(vendor_dir))

        import torch
        from trellis.pipelines import TrellisVGGTTo3DPipeline
        from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.torch = torch
        low_vram = settings.low_vram
        print("[reconviagen] loading sparse-structure pipeline", flush=True)
        vggt_pipeline = TrellisVGGTTo3DPipeline.from_pretrained(settings.ss_model)
        vggt_pipeline.cuda()
        vggt_pipeline.VGGT_model.cuda()
        vggt_pipeline.birefnet_model.cuda()
        if "slat_decoder_gs" in vggt_pipeline.models:
            del vggt_pipeline.models["slat_decoder_gs"]
        if low_vram:
            vggt_pipeline.VGGT_model.cpu()
            for model in vggt_pipeline.models.values():
                model.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        print("[reconviagen] loading TRELLIS.2 pipeline", flush=True)
        trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(settings.trellis_model)
        trellis2_pipeline.cuda()
        trellis2_pipeline.low_vram = low_vram
        self.pipeline = TrellisHybridPipeline(vggt_pipeline, trellis2_pipeline, low_vram=low_vram)
        print("[reconviagen] worker ready", flush=True)

    def generate(self, input_dir: Path, output_path: Path) -> None:
        from PIL import Image

        image_paths = sorted(input_dir.glob("view_*.png"))
        if not image_paths:
            raise RuntimeError("ReconViaGen input directory did not contain any views")
        images = [Image.open(path).convert("RGBA") for path in image_paths]
        params = _sampler_params()
        print(f"[reconviagen] reconstructing {len(images)} views", flush=True)
        meshes, latents = self.pipeline.run_multi_image(
            images,
            strategy="adaptive_guidance_weight",
            seed=reconviagen_settings().seed,
            ss_sampler_params=params["ss"],
            slat_sampler_params=params["slat"],
            shape_slat_sampler_params=params["shape"],
            tex_slat_sampler_params=params["texture"],
            pipeline_type=reconviagen_settings().pipeline_type,
            preprocess_image=True,
            return_latent=True,
            ss_source=reconviagen_settings().ss_source,
        )
        mesh = meshes[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import o_voxel

            resolution = latents[2]
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=self.pipeline.pbr_attr_layout,
                grid_size=resolution,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=reconviagen_settings().decimation_target,
                texture_size=reconviagen_settings().texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )
            glb.export(output_path, extension_webp=True)
        except Exception as exc:
            print(f"[reconviagen] textured postprocess failed: {exc}", flush=True)
            print("[reconviagen] exporting raw geometry fallback", flush=True)
            _export_raw_mesh(mesh, output_path)
        self.torch.cuda.empty_cache()
        print(f"[reconviagen] wrote {output_path}", flush=True)


def _sampler_params() -> dict[str, dict[str, object]]:
    return {
        "ss": {
            "steps": reconviagen_settings().ss_steps,
            "cfg_strength": reconviagen_settings().ss_guidance,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        "slat": {
            "steps": reconviagen_settings().slat_steps,
            "cfg_strength": reconviagen_settings().slat_guidance,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "shape": {
            "steps": reconviagen_settings().shape_steps,
            "guidance_strength": reconviagen_settings().shape_guidance,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "texture": {
            "steps": reconviagen_settings().texture_steps,
            "guidance_strength": reconviagen_settings().texture_guidance,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
    }


def _export_raw_mesh(mesh: object, output_path: Path) -> None:
    import numpy as np
    import trimesh

    vertices = _as_numpy(mesh.vertices)
    faces = _as_numpy(mesh.faces).astype(np.int64, copy=False)
    raw_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    raw_mesh.export(output_path, file_type="glb")


def _as_numpy(value: object):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return value
