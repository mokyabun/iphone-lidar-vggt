from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .service import VGGTService
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vggt_worker.service import VGGTService


class GenerateRequest(BaseModel):
    image_dir: str
    output_dir: str
    preserve_color: bool = True


class GenerateResponse(BaseModel):
    status: str
    output_path: str
    point_count: int


class PreloadResponse(BaseModel):
    status: str
    message: str


def create_app(service: VGGTService | None = None) -> FastAPI:
    app = FastAPI(title="VGGT Worker")
    worker = service or VGGTService()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/preload", response_model=PreloadResponse)
    def preload() -> PreloadResponse:
        try:
            message = worker.preload()
        except Exception as exc:  # noqa: BLE001
            print(f"[vggt] preload failed: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PreloadResponse(status="ok", message=message)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            output_path, point_count = worker.generate_from_image_dir(
                Path(request.image_dir),
                Path(request.output_dir),
                preserve_color=request.preserve_color,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[vggt] generation failed: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return GenerateResponse(status="ok", output_path=str(output_path), point_count=point_count)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--image-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-preserve-color", action="store_true")
    args = parser.parse_args()

    if args.once:
        if not args.image_dir or not args.output_dir:
            parser.error("--once requires --image-dir and --output-dir")
        VGGTService().generate_from_image_dir(
            Path(args.image_dir),
            Path(args.output_dir),
            preserve_color=not args.no_preserve_color,
        )
        return

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
