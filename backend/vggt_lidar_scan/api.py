from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .reconstruct import reconstruct_scan

RUN_ROOT = Path("runs/api")

app = FastAPI(title="VGGT iPhone LiDAR Scanner API")


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
