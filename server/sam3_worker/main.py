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
    from .service import SAM3Service
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from service import SAM3Service


class SegmentFrame(BaseModel):
    frame_id: str
    image_path: str
    box_xyxy: list[float]


class SegmentRequest(BaseModel):
    frames: list[SegmentFrame]
    output_dir: str


_SERVICE_LOCK = threading.Lock()


def create_app(service: SAM3Service | None = None) -> FastAPI:
    app = FastAPI(title="SAM3 Segmentation Worker")
    state: dict[str, object] = {"service": service, "error": None}

    def get_service() -> SAM3Service:
        if state["service"] is None:
            with _SERVICE_LOCK:
                if state["service"] is None:
                    try:
                        state["service"] = SAM3Service()
                        state["error"] = None
                    except Exception as exc:
                        state["error"] = f"{type(exc).__name__}: {exc}"
                        raise
        return state["service"]  # type: ignore[return-value]

    @app.get("/health")
    def health() -> dict[str, str | None]:
        try:
            get_service()
        except Exception as exc:
            return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        return {"status": "available", "reason": None}

    @app.post("/segment")
    def segment(request: SegmentRequest) -> dict[str, object]:
        started = time.perf_counter()
        print(f"[sam3] segment request: frames={len(request.frames)} output_dir={request.output_dir}", flush=True)
        try:
            payload = get_service().segment(
                [frame.model_dump() for frame in request.frames],
                Path(request.output_dir),
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[sam3] segmentation failed after {time.perf_counter() - started:.1f}s: {exc}\n{tb}", flush=True)
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{tb}") from exc
        print(f"[sam3] segmentation succeeded in {time.perf_counter() - started:.1f}s", flush=True)
        return payload

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
