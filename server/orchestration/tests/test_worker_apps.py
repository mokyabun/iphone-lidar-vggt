from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from reconviagen_worker.main import create_app as create_reconviagen_app
from vggt_worker.main import create_app as create_vggt_app


class _ReconViaGenService:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def generate(self, input_dir: Path, output_path: Path) -> None:
        self.calls.append((input_dir, output_path))


class _VGGTService:
    def preload(self) -> str:
        return "loaded"

    def generate_from_image_dir(self, image_dir: Path, output_dir: Path, preserve_color: bool = True) -> tuple[Path, int]:
        return output_dir / "scan_vggt_points.ply", 42


def test_reconviagen_worker_app_routes_generate_request() -> None:
    service = _ReconViaGenService()
    client = TestClient(create_reconviagen_app(service))

    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/generate", json={"input_dir": "/tmp/in", "output_path": "/tmp/out.glb"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service.calls == [(Path("/tmp/in"), Path("/tmp/out.glb"))]


def test_vggt_worker_app_routes_preload_and_generate_request() -> None:
    client = TestClient(create_vggt_app(_VGGTService()))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/preload").json() == {"status": "ok", "message": "loaded"}
    response = client.post(
        "/generate",
        json={"image_dir": "/tmp/images", "output_dir": "/tmp/out", "preserve_color": False},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "output_path": "/tmp/out/scan_vggt_points.ply",
        "point_count": 42,
    }
