from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .service import ReconViaGenService
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from service import ReconViaGenService


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
        started = time.perf_counter()
        print(
            f"[reconviagen] generate request: input_dir={request.input_dir} output_path={request.output_path}",
            flush=True,
        )
        try:
            get_service().generate(Path(request.input_dir), Path(request.output_path))
        except Exception as exc:
            # Surface the full traceback: the failing frame is usually deep inside the
            # vendored ReconViaGen/TRELLIS pipeline, and str(exc) alone (e.g. a bare
            # "unsupported operand type(s) for /: 'NoneType' and 'float'") gives no hint
            # of where it came from. Log it here and forward it in the 500 detail so it
            # lands in the API/pipeline logs too.
            tb = traceback.format_exc()
            print(
                f"[reconviagen] generation failed after {time.perf_counter() - started:.1f}s: {exc}\n{tb}",
                flush=True,
            )
            detail = f"{type(exc).__name__}: {exc}\n{tb}"
            raise HTTPException(status_code=500, detail=detail) from exc
        print(f"[reconviagen] generation succeeded in {time.perf_counter() - started:.1f}s", flush=True)
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
