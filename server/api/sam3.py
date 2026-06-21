from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import settings
from .models import FrameRecord
from .scan_io import read_image


def segment_with_sam3(
    root: Path,
    frames: list[FrameRecord],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    cfg = settings()
    if not cfg.sam3_worker_url:
        raise RuntimeError("SAM3 worker is not configured.")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_frames = []
    for frame in frames:
        image_path = root / frame.image_path
        box_xyxy = _central_box_xyxy(frame.image_width, frame.image_height)
        request_frames.append(
            {
                "frame_id": frame.frame_id,
                "image_path": str(image_path),
                "box_xyxy": box_xyxy,
            }
        )
    response = _request_masks(cfg.sam3_worker_url, request_frames, output_dir)
    masks: dict[str, np.ndarray] = {}
    for frame in frames:
        path_value = response.get("masks", {}).get(frame.frame_id)
        if not path_value:
            continue
        mask_path = Path(path_value)
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != (frame.image_height, frame.image_width):
            image = read_image(root, frame)
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
            ) > 0
        masks[frame.frame_id] = mask
    return masks


def _central_box_xyxy(width: int, height: int) -> list[float]:
    fraction = min(max(settings().sam3_center_box_fraction, 0.05), 0.95)
    box_width = width * fraction
    box_height = height * fraction
    x0 = (width - box_width) / 2
    y0 = (height - box_height) / 2
    return [round(x0, 3), round(y0, 3), round(x0 + box_width, 3), round(y0 + box_height, 3)]


def _request_masks(worker_url: str, frames: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    body = json.dumps({"frames": frames, "output_dir": str(output_dir)}).encode()
    deadline = time.monotonic() + settings().sam3_timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        request = urllib.request.Request(
            worker_url.rstrip("/") + "/segment",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        remaining = max(1, int(deadline - time.monotonic()))
        _log(f"SAM3 request attempt={attempt} frames={len(frames)} timeout_seconds={remaining}")
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=remaining) as response:
                result = json.loads(response.read())
            _log(f"SAM3 response received in {time.monotonic() - started:.1f}s result_keys={sorted(result)}")
            if result.get("status") != "ok":
                raise RuntimeError(result.get("error") or "SAM3 worker failed.")
            return result
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode(errors="ignore")
            _log(f"SAM3 HTTP failure: status={exc.code} payload={payload}")
            raise RuntimeError(f"SAM3 worker failed: {payload}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"SAM3 worker request timed out: {exc}") from exc
            _log(f"SAM3 worker unavailable, retrying in 3s: {exc}")
            time.sleep(3)


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[sam3-client] {timestamp} {message}", flush=True)
