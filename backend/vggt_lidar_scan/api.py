from __future__ import annotations

import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .reconstruct import reconstruct_scan
from .vggt_adapter import preload_vggt

RUN_ROOT = Path("runs/api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _env_bool("VGGT_PRELOAD", False):
        try:
            print(f"[startup] {preload_vggt()}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep API up so LiDAR-only still works.
            print(f"[startup] VGGT preload failed: {exc}", flush=True)
    yield


app = FastAPI(title="VGGT iPhone LiDAR Scanner API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
def create_job(scan_package: UploadFile = File(...), run_vggt: bool = False) -> dict[str, object]:
    job_id = uuid.uuid4().hex
    job_dir = RUN_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    package_path = job_dir / "ScanPackage.zip"

    with package_path.open("wb") as handle:
        shutil.copyfileobj(scan_package.file, handle)

    try:
        metrics = reconstruct_scan(package_path, job_dir / "output", run_vggt_stage=run_vggt)
    except Exception as exc:  # noqa: BLE001 - expose job failure in MVP API.
        (job_dir / "error.txt").write_text(str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"job_id": job_id, "metrics": metrics.model_dump()}


@app.post("/reconstruct")
def reconstruct_now(scan_package: UploadFile = File(...), run_vggt: bool = False) -> FileResponse:
    job_id = uuid.uuid4().hex
    job_dir = RUN_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    package_path = job_dir / "ScanPackage.zip"

    with package_path.open("wb") as handle:
        shutil.copyfileobj(scan_package.file, handle)

    try:
        metrics = reconstruct_scan(package_path, job_dir / "output", run_vggt_stage=run_vggt)
    except Exception as exc:  # noqa: BLE001 - direct demo endpoint should return a clear API error.
        (job_dir / "error.txt").write_text(str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}
