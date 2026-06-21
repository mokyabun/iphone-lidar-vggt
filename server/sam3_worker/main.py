from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

try:
    from .service import SAM3Service
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from service import SAM3Service


BoxXYXY = Annotated[list[float], Field(min_length=4, max_length=4)]


class SegmentFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    image_path: str
    box_xyxy: BoxXYXY
    positive_boxes_xyxy: list[BoxXYXY] | None = None
    negative_boxes_xyxy: list[BoxXYXY] | None = None
    text_prompt: str | None = None


class SegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        print(
            "[sam3] segment request: "
            f"frames={len(request.frames)} "
            f"output_dir={request.output_dir} "
            f"{_prompt_summary(request.frames)}",
            flush=True,
        )
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


def _prompt_summary(frames: list[SegmentFrame]) -> str:
    text_frames = sum(1 for frame in frames if (frame.text_prompt or "").strip())
    positive_boxes = sum(len(frame.positive_boxes_xyxy or []) for frame in frames)
    negative_boxes = sum(len(frame.negative_boxes_xyxy or []) for frame in frames)
    return (
        f"text_frames={text_frames} "
        f"positive_boxes={positive_boxes} "
        f"negative_boxes={negative_boxes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
