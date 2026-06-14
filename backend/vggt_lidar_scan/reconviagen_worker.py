from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReconViaGenRuntime:
    def __init__(self) -> None:
        repo_dir = Path(os.environ.get("RECONVIAGEN_REPO_DIR", "/workspace/cache/ReconViaGen"))
        trellis2_dir = repo_dir / "wheels" / "TRELLIS.2"
        sys.path.insert(0, str(repo_dir))
        sys.path.insert(0, str(trellis2_dir))
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("XFORMERS_DISABLED", "1")

        import torch
        from trellis.pipelines import TrellisVGGTTo3DPipeline
        from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.torch = torch
        low_vram = _env_bool("RECONVIAGEN_LOW_VRAM", True)
        print("[reconviagen] loading sparse-structure pipeline", flush=True)
        vggt_pipeline = TrellisVGGTTo3DPipeline.from_pretrained(
            os.environ.get("RECONVIAGEN_SS_MODEL", "Stable-X/trellis-vggt-v0-2")
        )
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
        trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            os.environ.get("RECONVIAGEN_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
        )
        trellis2_pipeline.cuda()
        trellis2_pipeline.low_vram = low_vram
        self.pipeline = TrellisHybridPipeline(vggt_pipeline, trellis2_pipeline, low_vram=low_vram)
        print("[reconviagen] worker ready", flush=True)

    def generate(self, input_dir: Path, output_path: Path) -> None:
        from PIL import Image
        import o_voxel

        image_paths = sorted(input_dir.glob("view_*.png"))
        if not image_paths:
            raise RuntimeError("ReconViaGen input directory did not contain any views")
        images = [Image.open(path).convert("RGBA") for path in image_paths]
        params = _sampler_params()
        print(f"[reconviagen] reconstructing {len(images)} views", flush=True)
        meshes, latents = self.pipeline.run_multi_image(
            images,
            strategy="adaptive_guidance_weight",
            seed=_env_int("RECONVIAGEN_SEED", 0),
            ss_sampler_params=params["ss"],
            slat_sampler_params=params["slat"],
            shape_slat_sampler_params=params["shape"],
            tex_slat_sampler_params=params["texture"],
            pipeline_type=os.environ.get("RECONVIAGEN_PIPELINE_TYPE", "1024_cascade"),
            preprocess_image=True,
            return_latent=True,
            ss_source=os.environ.get("RECONVIAGEN_SS_SOURCE", "mesh"),
        )
        mesh = meshes[0]
        resolution = latents[2]
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=self.pipeline.pbr_attr_layout,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=_env_int("RECONVIAGEN_DECIMATION_TARGET", 500000),
            texture_size=_env_int("RECONVIAGEN_TEXTURE_SIZE", 2048),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        glb.export(output_path, extension_webp=True)
        self.torch.cuda.empty_cache()
        print(f"[reconviagen] wrote {output_path}", flush=True)


def _sampler_params() -> dict[str, dict[str, object]]:
    return {
        "ss": {
            "steps": _env_int("RECONVIAGEN_SS_STEPS", 12),
            "cfg_strength": _env_float("RECONVIAGEN_SS_GUIDANCE", 7.5),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        "slat": {
            "steps": 12,
            "cfg_strength": 7.5,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "shape": {
            "steps": _env_int("RECONVIAGEN_SHAPE_STEPS", 12),
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "texture": {
            "steps": _env_int("RECONVIAGEN_TEXTURE_STEPS", 12),
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-path")
    args = parser.parse_args()

    runtime = ReconViaGenRuntime()
    if args.once:
        if not args.input_dir or not args.output_path:
            parser.error("--once requires --input-dir and --output-path")
        runtime.generate(Path(args.input_dir), Path(args.output_path))
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            self._json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                runtime.generate(Path(payload["input_dir"]), Path(payload["output_path"]))
                self._json(200, {"status": "ok"})
            except Exception as exc:  # noqa: BLE001
                print(f"[reconviagen] generation failed: {exc}", flush=True)
                self._json(500, {"status": "error", "error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            print(f"[reconviagen] {format % args}", flush=True)

        def _json(self, status: int, payload: dict[str, str]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return max(1, int(value)) if value else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


if __name__ == "__main__":
    main()
