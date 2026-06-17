from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import env_bool, settings
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
    return {
        "pipeline": "reconviagen_lidar_scale",
        "state": "available" if configured else "unavailable",
        "reason": None if configured else "Start with ./run.sh or set RECONVIAGEN_COMMAND/RECONVIAGEN_WORKER_URL.",
    }


@app.post("/jobs", status_code=202)
def create_job(scan_package: UploadFile = File(...)) -> dict[str, object]:
    job_id, job_dir, package_path = _store_upload(scan_package)
    _write_status(job_dir, "queued")
    thread = threading.Thread(target=_run_job, args=(job_id, package_path, job_dir), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/reconstruct")
def reconstruct_now(scan_package: UploadFile = File(...)) -> FileResponse:
    job_id, job_dir, package_path = _store_upload(scan_package)
    with _LOCK:
        result = reconstruct_scan(package_path, job_dir / "output")
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
    return job_id, job_dir, package_path


def _run_job(job_id: str, package_path: Path, job_dir: Path) -> None:
    del job_id
    started = time.monotonic()
    with _LOCK:
        _write_status(job_dir, "processing")
        try:
            reconstruct_scan(package_path, job_dir / "output")
        except Exception as exc:
            (job_dir / "error.txt").write_text(str(exc))
            _write_status(job_dir, "failed")
            return
        _write_status(job_dir, "complete", elapsed_seconds=round(time.monotonic() - started, 1))


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
