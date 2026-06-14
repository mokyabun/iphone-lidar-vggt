from __future__ import annotations

import importlib.util
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .reconstruct import reconstruct_scan
from .vggt_adapter import preload_vggt

RUN_ROOT = Path("runs/api")
_RECONSTRUCTION_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _env_bool("VGGT_PRELOAD", False):
        thread = threading.Thread(target=_preload_vggt_background, name="vggt-preload", daemon=True)
        thread.start()
    yield


app = FastAPI(title="VGGT iPhone LiDAR Scanner API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    vggt_available = bool(os.environ.get("VGGT_RUNNER")) or importlib.util.find_spec("vggt") is not None
    ai_state, ai_reason = _reconviagen_state()
    return {
        "pipelines": {
            "metric": {
                "state": "available",
                "options": ["color", "object", "mesh"],
            },
            "vggt": {
                "state": "available" if vggt_available else "unavailable",
                "reason": None if vggt_available else "VGGT runtime is not installed.",
                "options": ["color", "object"],
            },
            "ai_mesh": {
                "state": ai_state,
                "reason": ai_reason,
                "options": ["color", "object", "mesh", "preview_glb", "print_stl"],
                "required_options": ["object", "mesh"],
            },
        }
    }


@app.post("/jobs")
def create_job(
    scan_package: UploadFile = File(...),
    run_vggt: bool = False,
    preserve_color: bool = True,
    extract_object: bool = False,
    reconstruct_mesh: bool = False,
    ai_mesh: bool = False,
) -> dict[str, object]:
    job_id = uuid.uuid4().hex
    job_dir = RUN_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    package_path = job_dir / "ScanPackage.zip"

    with package_path.open("wb") as handle:
        shutil.copyfileobj(scan_package.file, handle)

    metrics = _run_reconstruction(
        package_path,
        job_dir / "output",
        job_dir / "error.txt",
        run_vggt=run_vggt,
        preserve_color=preserve_color,
        extract_object=extract_object,
        reconstruct_mesh=reconstruct_mesh,
        ai_mesh=ai_mesh,
    )

    return {"job_id": job_id, "metrics": metrics.model_dump()}


@app.post("/reconstruct")
def reconstruct_now(
    scan_package: UploadFile = File(...),
    run_vggt: bool = False,
    preserve_color: bool = True,
    extract_object: bool = False,
    reconstruct_mesh: bool = False,
    ai_mesh: bool = False,
) -> FileResponse:
    job_id = uuid.uuid4().hex
    job_dir = RUN_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    package_path = job_dir / "ScanPackage.zip"

    with package_path.open("wb") as handle:
        shutil.copyfileobj(scan_package.file, handle)

    metrics = _run_reconstruction(
        package_path,
        job_dir / "output",
        job_dir / "error.txt",
        run_vggt=run_vggt,
        preserve_color=preserve_color,
        extract_object=extract_object,
        reconstruct_mesh=reconstruct_mesh,
        ai_mesh=ai_mesh,
    )

    result_path = Path(metrics.final_output)
    return FileResponse(
        result_path,
        filename="scan_final.ply",
        media_type="application/octet-stream",
        headers={
            "X-Job-ID": job_id,
            "X-Reconstruction-Metrics": json.dumps(metrics.model_dump()),
        },
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job_dir = RUN_ROOT / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    metrics_path = job_dir / "output" / "metrics.json"
    if metrics_path.exists():
        return {"job_id": job_id, "status": "complete", "metrics_path": str(metrics_path)}
    error_path = job_dir / "error.txt"
    if error_path.exists():
        return {"job_id": job_id, "status": "failed", "error": error_path.read_text()}
    return {"job_id": job_id, "status": "processing"}


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    result = RUN_ROOT / job_id / "output" / "scan_final.ply"
    if not result.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result, filename="scan_final.ply")


@app.get("/jobs/{job_id}/preview")
def get_preview_asset(job_id: str) -> FileResponse:
    result = RUN_ROOT / job_id / "output" / "scan_object_preview.glb"
    if not result.exists():
        raise HTTPException(status_code=404, detail="Preview GLB not found")
    return FileResponse(result, filename="scan_object_preview.glb", media_type="model/gltf-binary")


@app.get("/jobs/{job_id}/print")
def get_print_asset(job_id: str) -> FileResponse:
    result = RUN_ROOT / job_id / "output" / "scan_object_print.stl"
    if not result.exists():
        raise HTTPException(status_code=404, detail="Print STL not found")
    return FileResponse(result, filename="scan_object_print.stl", media_type="model/stl")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}


def _reconviagen_state() -> tuple[str, str | None]:
    error_path = Path(os.environ.get("RECONVIAGEN_WORKER_ERROR", "/workspace/cache/reconviagen-worker.error"))
    worker_url = os.environ.get("RECONVIAGEN_WORKER_URL")
    if worker_url:
        try:
            with urllib.request.urlopen(worker_url.rstrip("/") + "/health", timeout=1) as response:
                if response.status == 200:
                    return "available", None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
    if error_path.exists():
        message = error_path.read_text().strip()
        return "unavailable", message or "ReconViaGen worker failed to start."
    if not worker_url:
        return "unavailable", "ReconViaGen worker is not configured."
    return "loading", "ReconViaGen worker is loading."


def _run_reconstruction(
    package_path: Path,
    output_dir: Path,
    error_path: Path,
    *,
    run_vggt: bool,
    preserve_color: bool,
    extract_object: bool,
    reconstruct_mesh: bool,
    ai_mesh: bool,
):
    if not _RECONSTRUCTION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another reconstruction is already running. Wait for it to finish and retry.")

    started = time.monotonic()
    print(
        f"[api] reconstruction start package={package_path} "
        f"vggt={run_vggt} color={preserve_color} object={extract_object} "
        f"mesh={reconstruct_mesh} ai_mesh={ai_mesh}",
        flush=True,
    )
    try:
        metrics = reconstruct_scan(
            package_path,
            output_dir,
            run_vggt_stage=run_vggt,
            preserve_color=preserve_color,
            extract_object=extract_object,
            reconstruct_mesh=reconstruct_mesh,
            ai_mesh=ai_mesh,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - direct demo endpoint should return a clear API error.
        error_path.write_text(str(exc))
        print(f"[api] reconstruction failed after {time.monotonic() - started:.1f}s: {exc}", flush=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        _RECONSTRUCTION_LOCK.release()

    print(
        f"[api] reconstruction complete after {time.monotonic() - started:.1f}s "
        f"output={metrics.final_output_type} mesh_faces={metrics.mesh_faces} vggt_points={metrics.vggt_points}",
        flush=True,
    )
    return metrics


def _preload_vggt_background() -> None:
    try:
        print("[startup] VGGT preload started in background", flush=True)
        print(f"[startup] {preload_vggt()}", flush=True)
    except Exception as exc:  # noqa: BLE001 - keep API up so LiDAR-only still works.
        print(f"[startup] VGGT preload failed: {exc}", flush=True)
