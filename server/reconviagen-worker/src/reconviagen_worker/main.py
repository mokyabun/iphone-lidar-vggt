from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .service import ReconViaGenService
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconviagen_worker.service import ReconViaGenService


class GenerateRequest(BaseModel):
    input_dir: str
    output_path: str


class GenerateResponse(BaseModel):
    status: str


_SERVICE_LOCK = threading.Lock()


def create_app(service: ReconViaGenService | None = None) -> FastAPI:
    app = FastAPI(title="ReconViaGen Worker")
    state = {"service": service}

    def get_service() -> ReconViaGenService:
        if state["service"] is None:
            with _SERVICE_LOCK:
                if state["service"] is None:
                    state["service"] = ReconViaGenService()
        return state["service"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            get_service().generate(Path(request.input_dir), Path(request.output_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[reconviagen] generation failed: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return GenerateResponse(status="ok")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-path")
    args = parser.parse_args()

    if args.once:
        if not args.input_dir or not args.output_path:
            parser.error("--once requires --input-dir and --output-path")
        ReconViaGenService().generate(Path(args.input_dir), Path(args.output_path))
        return

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
