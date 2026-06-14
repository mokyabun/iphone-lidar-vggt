from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from vggt_lidar_scan import api
from vggt_lidar_scan.api import app


def test_reconstruct_endpoint_returns_ply(tmp_path: Path) -> None:
    package_dir = tmp_path / "ScanPackage"
    (package_dir / "images").mkdir(parents=True)
    (package_dir / "depth").mkdir()

    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(package_dir / "images" / "frame_000001.jpg")
    np.ones((2, 2), dtype=np.float32).tofile(package_dir / "depth" / "frame_000001.float32")
    frame = {
        "frame_id": "frame_000001",
        "timestamp": 1.0,
        "image_path": "images/frame_000001.jpg",
        "depth_path": "depth/frame_000001.float32",
        "confidence_path": None,
        "image_width": 4,
        "image_height": 4,
        "depth_width": 2,
        "depth_height": 2,
        "intrinsics_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "orientation": "landscapeRight",
    }
    (package_dir / "frames.jsonl").write_text(json.dumps(frame) + "\n")

    archive = tmp_path / "ScanPackage.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(package_dir))

    client = TestClient(app)
    with archive.open("rb") as handle:
        response = client.post("/reconstruct", files={"scan_package": ("ScanPackage.zip", handle, "application/zip")})

    assert response.status_code == 200
    assert response.content.startswith(b"ply\n")
    assert response.headers["x-job-id"]
    metrics = json.loads(response.headers["x-reconstruction-metrics"])
    assert metrics["final_output_type"] == "point_cloud"
    assert metrics["mesh_faces"] == 0
    assert metrics["camera_path_m"] == 0.0
    assert metrics["lidar_extent_m"] is not None


def test_generated_asset_endpoints_return_glb_and_stl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api, "RUN_ROOT", tmp_path)
    output = tmp_path / "job" / "output"
    output.mkdir(parents=True)
    (output / "scan_object_preview.glb").write_bytes(b"glTF")
    (output / "scan_object_print.stl").write_bytes(b"solid scan")
    client = TestClient(app)

    preview = client.get("/jobs/job/preview")
    printable = client.get("/jobs/job/print")

    assert preview.status_code == 200
    assert preview.headers["content-type"] == "model/gltf-binary"
    assert printable.status_code == 200
    assert printable.headers["content-type"] == "model/stl"


def test_capabilities_reports_pipeline_option_support(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RECONVIAGEN_WORKER_ERROR", str(tmp_path / "missing-error"))
    monkeypatch.delenv("RECONVIAGEN_WORKER_URL", raising=False)
    monkeypatch.setattr(api.importlib.util, "find_spec", lambda name: None)
    client = TestClient(app)

    response = client.get("/capabilities")

    assert response.status_code == 200
    pipelines = response.json()["pipelines"]
    assert pipelines["metric"]["state"] == "available"
    assert pipelines["metric"]["options"] == ["color", "object", "mesh"]
    assert pipelines["vggt"]["state"] == "unavailable"
    assert pipelines["ai_mesh"]["state"] == "unavailable"


def test_capabilities_prefers_live_worker_over_stale_error(tmp_path: Path, monkeypatch) -> None:
    error_path = tmp_path / "worker-error"
    error_path.write_text("old startup failure")
    monkeypatch.setenv("RECONVIAGEN_WORKER_ERROR", str(error_path))
    monkeypatch.setenv("RECONVIAGEN_WORKER_URL", "http://worker")

    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *args, **kwargs: HealthyResponse())

    response = TestClient(app).get("/capabilities")

    assert response.json()["pipelines"]["ai_mesh"]["state"] == "available"
