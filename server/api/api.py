from __future__ import annotations

import json
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import env_bool, settings
from .models import ReconstructionOptions
from .pipeline import reconstruct_scan

app = FastAPI(title="ReconViaGen LiDAR Scale API")
_LOCK = threading.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    cfg = settings()
    configured = bool(cfg.reconviagen_command or cfg.reconviagen_worker_url or env_bool("RECONVIAGEN_MOCK", False))
    sam3_state, sam3_reason = _worker_capability(cfg.sam3_worker_url, "SAM3 worker")
    return {
        "pipeline": "reconviagen_lidar_scale",
        "state": "available" if configured else "unavailable",
        "reason": None if configured else "Start with ./run.sh or set RECONVIAGEN_COMMAND/RECONVIAGEN_WORKER_URL.",
        "features": {
            "sam3_object_masking": {
                "state": sam3_state,
                "reason": sam3_reason,
                "default_enabled": False,
            },
            "lidar_scale_alignment": {
                "state": "available",
                "reason": None,
                "default_enabled": True,
            },
        },
    }


@app.post("/jobs", status_code=202)
def create_job(
    scan_package: UploadFile = File(...),
    enable_sam3_object_masking: bool = Form(False),
    enable_lidar_scale_alignment: bool = Form(True),
    enable_mesh_fragment_cleanup: bool = Form(True),
    sam3_text_prompt: str = Form(""),
) -> dict[str, object]:
    options = ReconstructionOptions(
        enable_sam3_object_masking=enable_sam3_object_masking,
        enable_lidar_scale_alignment=enable_lidar_scale_alignment,
        enable_mesh_fragment_cleanup=enable_mesh_fragment_cleanup,
        sam3_text_prompt=sam3_text_prompt.strip(),
    )
    job_id, job_dir, package_path = _store_upload(scan_package)
    _log(f"job {job_id}: queued package={package_path} options={options}")
    _write_status(job_dir, "queued")
    thread = threading.Thread(target=_run_job, args=(job_id, package_path, job_dir, options), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/reconstruct")
def reconstruct_now(
    scan_package: UploadFile = File(...),
    enable_sam3_object_masking: bool = Form(False),
    enable_lidar_scale_alignment: bool = Form(True),
    enable_mesh_fragment_cleanup: bool = Form(True),
    sam3_text_prompt: str = Form(""),
) -> FileResponse:
    options = ReconstructionOptions(
        enable_sam3_object_masking=enable_sam3_object_masking,
        enable_lidar_scale_alignment=enable_lidar_scale_alignment,
        enable_mesh_fragment_cleanup=enable_mesh_fragment_cleanup,
        sam3_text_prompt=sam3_text_prompt.strip(),
    )
    job_id, job_dir, package_path = _store_upload(scan_package)
    _log(f"job {job_id}: synchronous reconstruction requested package={package_path} options={options}")
    with _LOCK:
        _log(f"job {job_id}: synchronous reconstruction started")
        result = reconstruct_scan(package_path, job_dir / "output", options=options)
    _log(f"job {job_id}: synchronous reconstruction complete output={result.final_output}")
    return FileResponse(
        result.final_output,
        filename="reconviagen_metric.ply",
        media_type="application/octet-stream",
        headers={"X-Job-ID": job_id, "X-Reconstruction-Metrics": json.dumps(result.metrics)},
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job_dir = settings().run_root / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    metrics_path = job_dir / "output" / "metrics.json"
    if metrics_path.exists():
        return {"job_id": job_id, "status": "complete", "metrics": json.loads(metrics_path.read_text())}
    error_path = job_dir / "error.txt"
    if error_path.exists():
        return {"job_id": job_id, "status": "failed", "error": error_path.read_text()}
    status_path = job_dir / "status.json"
    if status_path.exists():
        return {"job_id": job_id, **json.loads(status_path.read_text())}
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    return _file_response(job_id, "reconviagen_metric.ply", "reconviagen_metric.ply", "application/octet-stream")


@app.get("/jobs/{job_id}/preview")
def get_preview(job_id: str) -> FileResponse:
    return _file_response(job_id, "reconviagen_metric.glb", "reconviagen_metric.glb", "model/gltf-binary")


@app.get("/jobs/{job_id}/raw")
def get_raw_preview(job_id: str) -> FileResponse:
    return _file_response(job_id, "reconviagen_raw.glb", "reconviagen_raw.glb", "model/gltf-binary")


@app.get("/jobs/{job_id}/raw-ply")
def get_raw_ply(job_id: str) -> FileResponse:
    return _file_response(job_id, "reconviagen_raw.ply", "reconviagen_raw.ply", "application/octet-stream")


@app.get("/jobs/{job_id}/raw-stl")
def get_raw_stl(job_id: str) -> FileResponse:
    return _file_response(job_id, "reconviagen_raw.stl", "reconviagen_raw.stl", "model/stl")


@app.get("/jobs/{job_id}/print")
def get_print(job_id: str) -> FileResponse:
    return _file_response(job_id, "reconviagen_metric_print_mm.stl", "reconviagen_metric_print_mm.stl", "model/stl")


@app.get("/jobs/{job_id}/lidar")
def get_lidar_reference(job_id: str) -> FileResponse:
    return _file_response(job_id, "lidar_reference.ply", "lidar_reference.ply", "application/octet-stream")


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def _store_upload(scan_package: UploadFile) -> tuple[str, Path, Path]:
    job_id = uuid.uuid4().hex
    job_dir = settings().run_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    package_path = job_dir / "ScanPackage.zip"
    with package_path.open("wb") as handle:
        shutil.copyfileobj(scan_package.file, handle)
    size = package_path.stat().st_size if package_path.exists() else 0
    _log(f"job {job_id}: stored upload filename={scan_package.filename!r} bytes={size}")
    return job_id, job_dir, package_path


def _run_job(job_id: str, package_path: Path, job_dir: Path, options: ReconstructionOptions) -> None:
    started = time.monotonic()
    _log(f"job {job_id}: waiting for reconstruction lock")
    with _LOCK:
        _log(f"job {job_id}: processing started")
        _write_status(job_dir, "processing")
        try:
            reconstruct_scan(package_path, job_dir / "output", options=options)
        except Exception as exc:
            _log(f"job {job_id}: failed after {time.monotonic() - started:.1f}s: {exc}")
            (job_dir / "error.txt").write_text(str(exc))
            _write_status(job_dir, "failed")
            return
        elapsed = round(time.monotonic() - started, 1)
        _write_status(job_dir, "complete", elapsed_seconds=elapsed)
        _log(f"job {job_id}: complete in {elapsed:.1f}s")


def _write_status(job_dir: Path, status: str, **extra: object) -> None:
    payload = {"status": status, **extra}
    path = job_dir / "status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _file_response(job_id: str, filename: str, download_name: str, media_type: str) -> FileResponse:
    path = settings().run_root / job_id / "output" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{download_name} not found")
    return FileResponse(path, filename=download_name, media_type=media_type)


def _worker_capability(worker_url: str, name: str) -> tuple[str, str | None]:
    if not worker_url:
        return "unavailable", f"{name} is not configured."
    request = urllib.request.Request(worker_url.rstrip("/") + "/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        state = payload.get("status", "available")
        if state == "ok":
            state = "available"
        if state not in {"available", "loading", "unavailable"}:
            state = "available"
        return state, payload.get("reason")
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
        return "loading", f"{name} health check failed: {exc}"


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[api] {timestamp} {message}", flush=True)
