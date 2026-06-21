from __future__ import annotations

import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .settings import SAM3Settings, sam3_settings
except ImportError:
    from settings import SAM3Settings, sam3_settings


class SAM3Service:
    def __init__(self) -> None:
        started = time.perf_counter()
        self.settings = sam3_settings()
        _log(f"initializing SAM3 worker settings={self.settings}")
        self.mock = self.settings.mock
        self.model = None
        self.processor = None
        self.torch = None
        if self.mock:
            _log("SAM3_MOCK=1; using central-box mask fallback")
            return

        repo_dir = self.settings.repo_dir.expanduser()
        if repo_dir.exists():
            sys.path.insert(0, str(repo_dir))

        import torch
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.settings.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM3, but PyTorch cannot use it.")

        sam3_root = Path(sam3.__file__).resolve().parent.parent
        bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            bpe_path = repo_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        _log(f"loading SAM3 image model bpe_path={bpe_path}")
        model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.torch = torch
        self.model = model
        self.processor = Sam3Processor(
            model,
            confidence_threshold=self.settings.confidence_threshold,
        )
        _log(f"SAM3 worker ready in {time.perf_counter() - started:.1f}s")

    def segment(self, frames: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        masks: dict[str, str] = {}
        stats: dict[str, object] = {"frames": len(frames), "written": 0}
        for frame in frames:
            frame_id = str(frame["frame_id"])
            image_path = Path(str(frame["image_path"]))
            box_xyxy = [float(value) for value in frame["box_xyxy"]]
            image = Image.open(image_path).convert("RGB")
            mask = self._segment_image(image, box_xyxy)
            if mask is None or not np.any(mask):
                _log(f"no usable mask for frame_id={frame_id}")
                continue
            mask_path = output_dir / f"{_safe_name(frame_id)}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            masks[frame_id] = str(mask_path)
            stats["written"] = int(stats["written"]) + 1
        return {"status": "ok", "masks": masks, "stats": stats}

    def _segment_image(self, image: Image.Image, box_xyxy: list[float]) -> np.ndarray | None:
        if self.mock:
            return _box_mask(image.size, box_xyxy)
        assert self.processor is not None
        assert self.torch is not None
        width, height = image.size
        box = _xyxy_to_normalized_cxcywh(box_xyxy, width, height)
        autocast = (
            self.torch.autocast("cuda", dtype=self.torch.bfloat16)
            if self.torch.cuda.is_available()
            else nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            state = self.processor.set_image(image)
            state = self.processor.add_geometric_prompt(box=box, label=True, state=state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or len(masks) == 0:
            return None
        if scores is not None and len(scores) > 0:
            index = int(scores.argmax().item())
        else:
            index = 0
        mask = masks[index]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        mask = np.squeeze(mask).astype(bool)
        if mask.shape != (height, width):
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.asarray(mask_image.resize((width, height), Image.Resampling.NEAREST)) > 0
        return mask


def _xyxy_to_normalized_cxcywh(box_xyxy: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box_xyxy
    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    return [
        ((x0 + x1) * 0.5) / max(width, 1),
        ((y0 + y1) * 0.5) / max(height, 1),
        max(1.0, x1 - x0) / max(width, 1),
        max(1.0, y1 - y0) / max(height, 1),
    ]


def _box_mask(size: tuple[int, int], box_xyxy: list[float]) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = [int(round(value)) for value in box_xyxy]
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
    return mask


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "frame"


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[sam3] {timestamp} {message}", flush=True)
