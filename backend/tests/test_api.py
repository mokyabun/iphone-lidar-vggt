from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

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
