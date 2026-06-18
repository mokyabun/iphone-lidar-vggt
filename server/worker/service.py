from __future__ import annotations

import gc
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    from .settings import ReconViaGenSettings, reconviagen_settings
except ImportError:
    from settings import ReconViaGenSettings, reconviagen_settings


class ReconViaGenService:
    def __init__(self) -> None:
        started = time.perf_counter()
        _configure_thread_env()
        settings = reconviagen_settings()
        _log("initializing ReconViaGen worker")
        _log(f"settings: {_settings_summary(settings)}")
        _log_thread_diagnostics()
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
        missing_vendor_dirs = [str(path) for path in vendor_dirs if not path.exists()]
        if missing_vendor_dirs:
            _log(f"warning: missing ReconViaGen vendor dirs: {missing_vendor_dirs}")

        import torch

        if settings.torch_num_threads > 0:
            torch.set_num_threads(settings.torch_num_threads)
            try:
                torch.set_num_interop_threads(max(1, min(settings.torch_num_threads, 4)))
            except RuntimeError:
                pass
            _log(f"torch threads: intra_op={torch.get_num_threads()} inter_op={torch.get_num_interop_threads()}")
        _log_cuda_diagnostics(torch)
        if settings.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for ReconViaGen, but PyTorch cannot use it. "
                "Check the worker log diagnostics above. Common causes: CPU-only torch in the reused venv, "
                "no NVIDIA device exposed to the container, incompatible driver, or CUDA_VISIBLE_DEVICES hiding the GPU. "
                "Set APP_UPDATE_ENVS=1 to reinstall dependencies if the venv was created with the wrong torch build."
            )

        from trellis.pipelines import TrellisVGGTTo3DPipeline
        from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.torch = torch
        low_vram = settings.low_vram
        with _timed("resolve sparse-structure model snapshot"):
            ss_model = _snapshot_path(settings.ss_model)
        _log(f"sparse-structure model path: {ss_model}")
        with _timed("load sparse-structure pipeline"):
            vggt_pipeline = TrellisVGGTTo3DPipeline.from_pretrained(str(ss_model))
        with _timed("move sparse-structure pipeline to cuda"):
            vggt_pipeline.cuda()
            vggt_pipeline.VGGT_model.cuda()
            vggt_pipeline.birefnet_model.cuda()
        _log_cuda(torch, "after sparse-structure cuda")
        if "slat_decoder_gs" in vggt_pipeline.models:
            del vggt_pipeline.models["slat_decoder_gs"]
            _log("removed unused slat_decoder_gs model")
        if low_vram:
            with _timed("offload sparse-structure pipeline for low_vram"):
                vggt_pipeline.VGGT_model.cpu()
                for model in vggt_pipeline.models.values():
                    model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            _log_cuda(torch, "after sparse-structure offload")

        with _timed("resolve TRELLIS.2 model snapshot"):
            trellis_model = _snapshot_path(settings.trellis_model)
        _log(f"TRELLIS.2 model path: {trellis_model}")
        with _timed("load TRELLIS.2 pipeline"):
            trellis2_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(trellis_model))
        trellis2_pipeline.low_vram = low_vram
        with _timed("move TRELLIS.2 pipeline to cuda"):
            trellis2_pipeline.cuda()
        _log_cuda(torch, "after TRELLIS.2 cuda")
        self.pipeline = TrellisHybridPipeline(vggt_pipeline, trellis2_pipeline, low_vram=low_vram)
        _log(f"worker ready in {time.perf_counter() - started:.1f}s")

    def generate(self, input_dir: Path, output_path: Path) -> None:
        from PIL import Image

        started = time.perf_counter()
        settings = reconviagen_settings()
        image_paths = sorted(input_dir.glob("view_*.png"))
        if not image_paths:
            raise RuntimeError("ReconViaGen input directory did not contain any views")
        _log(f"generation requested: input_dir={input_dir} output_path={output_path}")
        _log(
            "run config: "
            f"views={len(image_paths)} pipeline_type={settings.pipeline_type} ss_source={settings.ss_source} "
            f"preprocess_image={settings.preprocess_image} max_num_tokens={settings.max_num_tokens}"
        )
        with _timed("load input views"):
            images = [Image.open(path).convert("RGBA") for path in image_paths]
        _log(f"input view sizes: {[_image_summary(path, image) for path, image in zip(image_paths, images)]}")
        params = _sampler_params(settings)
        _log(f"sampler params: {params}")
        _log_cuda(self.torch, "before reconstruction")
        with _timed("run ReconViaGen hybrid pipeline"):
            meshes, latents = self.pipeline.run_multi_image(
                images,
                strategy="adaptive_guidance_weight",
                seed=settings.seed,
                ss_sampler_params=params["ss"],
                slat_sampler_params=params["slat"],
                shape_slat_sampler_params=params["shape"],
                tex_slat_sampler_params=params["texture"],
                pipeline_type=settings.pipeline_type,
                preprocess_image=settings.preprocess_image,
                return_latent=True,
                ss_source=settings.ss_source,
                max_num_tokens=settings.max_num_tokens,
            )
        _log_cuda(self.torch, "after reconstruction")
        mesh = meshes[0]
        _log(f"mesh generated: {_mesh_summary(mesh)} latent_resolution={latents[2]}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import o_voxel

            resolution = latents[2]
            _log(
                "postprocess to textured GLB: "
                f"resolution={resolution} decimation_target={settings.decimation_target} "
                f"texture_size={settings.texture_size}"
            )
            with _timed("o_voxel textured postprocess"):
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=self.pipeline.pbr_attr_layout,
                    grid_size=resolution,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=settings.decimation_target,
                    texture_size=settings.texture_size,
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    use_tqdm=True,
                )
            with _timed("export textured GLB"):
                glb.export(output_path, extension_webp=True)
        except Exception as exc:
            _log(f"textured postprocess failed: {exc}")
            _log("exporting raw geometry fallback")
            with _timed("export raw geometry GLB"):
                _export_raw_mesh(mesh, output_path)
        self.torch.cuda.empty_cache()
        _log_cuda(self.torch, "after cuda cache clear")
        size = output_path.stat().st_size if output_path.exists() else 0
        _log(f"wrote {output_path} ({size} bytes) in {time.perf_counter() - started:.1f}s")


def _sampler_params(settings: ReconViaGenSettings) -> dict[str, dict[str, object]]:
    return {
        "ss": {
            "steps": settings.ss_steps,
            "cfg_strength": settings.ss_guidance,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        "slat": {
            "steps": settings.slat_steps,
            "cfg_strength": settings.slat_guidance,
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "shape": {
            "steps": settings.shape_steps,
            "guidance_strength": settings.shape_guidance,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "texture": {
            "steps": settings.texture_steps,
            "guidance_strength": settings.texture_guidance,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
    }


def _snapshot_path(model: str) -> Path | str:
    path = Path(model).expanduser()
    if path.exists():
        _log(f"model path exists locally: {path}")
        _log_directory_summary(path)
        return path

    from huggingface_hub import snapshot_download

    _log(f"resolving Hugging Face model snapshot: repo={model}")
    started = time.perf_counter()
    snapshot = Path(snapshot_download(model))
    _log(f"resolved Hugging Face snapshot in {time.perf_counter() - started:.1f}s: {snapshot}")
    _log_directory_summary(snapshot)
    return snapshot


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[reconviagen] {timestamp} {message}", flush=True)


def _configure_thread_env() -> None:
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def _log_thread_diagnostics() -> None:
    try:
        nproc_soft, nproc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except Exception:
        nproc_soft, nproc_hard = "unknown", "unknown"
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except Exception:
        cpu_count = os.cpu_count()
    env = {
        name: os.environ.get(name, "<unset>")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        )
    }
    _log(f"thread limits: cpus={cpu_count} RLIMIT_NPROC=({nproc_soft}, {nproc_hard}) env={env}")


@contextmanager
def _timed(label: str):
    _log(f"{label}: start")
    started = time.perf_counter()
    stop_heartbeat = threading.Event()
    heartbeat = _start_heartbeat(label, started, stop_heartbeat)
    try:
        yield
    finally:
        stop_heartbeat.set()
        if heartbeat is not None:
            heartbeat.join(timeout=1)
        _log(f"{label}: done in {time.perf_counter() - started:.1f}s")


def _start_heartbeat(label: str, started: float, stop_event: threading.Event) -> threading.Thread | None:
    settings = reconviagen_settings()
    interval = settings.heartbeat_seconds
    if interval <= 0:
        return None

    def run() -> None:
        while not stop_event.wait(interval):
            elapsed = time.perf_counter() - started
            _log(f"{label}: still running after {elapsed:.1f}s")
            _dump_thread_stacks(label)

    thread = threading.Thread(target=run, name=f"heartbeat:{label}", daemon=True)
    thread.start()
    return thread


def _dump_thread_stacks(label: str) -> None:
    current_ident = threading.get_ident()
    frames = sys._current_frames()
    for thread in threading.enumerate():
        # Skip the heartbeat thread itself (it is just running this dump) and
        # tqdm's monitor daemon, which is always idle and only adds noise.
        if thread.ident == current_ident or _is_noise_thread(thread):
            continue
        frame = frames.get(thread.ident)
        if frame is None:
            continue
        stack = "".join(traceback.format_stack(frame, limit=12)).rstrip()
        _log(f"{label}: stack thread={thread.name} ident={thread.ident}\n{stack}")


def _is_noise_thread(thread: threading.Thread) -> bool:
    name = thread.name or ""
    return name == "tqdm_monitor" or name.startswith("heartbeat:")


def _settings_summary(settings: ReconViaGenSettings) -> str:
    return (
        f"repo_dir={settings.repo_dir} low_vram={settings.low_vram} require_cuda={settings.require_cuda} "
        f"torch_num_threads={settings.torch_num_threads} "
        f"ss_model={settings.ss_model} trellis_model={settings.trellis_model} "
        f"pipeline_type={settings.pipeline_type} ss_source={settings.ss_source} "
        f"preprocess_image={settings.preprocess_image}"
    )


def _log_cuda(torch, label: str) -> None:
    if not torch.cuda.is_available():
        _log(f"{label}: cuda unavailable")
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    _log(f"{label}: cuda allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB")


def _log_cuda_diagnostics(torch) -> None:
    _log(
        "torch: "
        f"version={torch.__version__} cuda_build={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    if torch.version.cuda is None:
        _log("torch diagnostic: installed PyTorch build is CPU-only.")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = ".".join(str(part) for part in torch.cuda.get_device_capability(index))
            _log(f"cuda device {index}: name={torch.cuda.get_device_name(index)} capability={capability}")
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        _log("nvidia-smi: not found in PATH")
        return
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index,name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        _log(f"nvidia-smi: failed to run: {exc}")
        return
    output = (result.stdout or result.stderr).strip()
    _log(f"nvidia-smi: returncode={result.returncode} output={output or '<empty>'}")


def _log_directory_summary(path: Path) -> None:
    try:
        files = [item for item in path.rglob("*") if item.is_file()]
    except Exception as exc:
        _log(f"model directory summary failed for {path}: {exc}")
        return
    total_bytes = 0
    missing_size = 0
    for item in files:
        try:
            total_bytes += item.stat().st_size
        except OSError:
            missing_size += 1
    largest_items: list[tuple[int, str]] = []
    for item in files:
        try:
            largest_items.append((item.stat().st_size, item.name))
        except OSError:
            continue
    largest = sorted(largest_items, reverse=True)[:5]
    largest_summary = ", ".join(f"{name}:{size / (1024**3):.2f}GiB" for size, name in largest)
    _log(
        f"model snapshot summary: path={path} files={len(files)} "
        f"size={total_bytes / (1024**3):.2f}GiB unreadable_files={missing_size} "
        f"largest=[{largest_summary}]"
    )


def _image_summary(path: Path, image: object) -> str:
    return f"{path.name}:{image.width}x{image.height}/{image.mode}"


def _mesh_summary(mesh: object) -> str:
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    coords = getattr(mesh, "coords", None)
    attrs = getattr(mesh, "attrs", None)
    return (
        f"vertices={_shape(vertices)} faces={_shape(faces)} "
        f"coords={_shape(coords)} attrs={_shape(attrs)}"
    )


def _shape(value: object) -> object:
    return getattr(value, "shape", None)


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
