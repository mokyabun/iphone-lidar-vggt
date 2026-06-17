# ReconViaGen + LiDAR Scale

This repo was simplified from the previous VGGT/LiDAR experiment. The old project is preserved under `_legacy/`.

The new project has one job:

```text
iPhone LiDAR ScanPackage.zip
  -> depth-only object mask
  -> metric LiDAR reference cloud
  -> ReconViaGen multi-view RGBA crops
  -> ReconViaGen raw mesh
  -> LiDAR scale + PCA/ICP alignment
  -> metric PLY/GLB/STL outputs
```

## What Was Removed

The active project no longer uses:

- `uv`, `uv.lock`, or the uv workspace layout
- standalone `vggt-worker`
- standalone VGGT point-cloud pipeline
- Open3D TSDF reconstruction
- Open3D printable mesh fallback
- SAM/Ultralytics/TIMM segmentation path
- iOS pipeline selector and `Color/Object/Mesh` toggles

The active server dependency set is intentionally small:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `numpy`
- `pillow`
- `scipy`
- `trimesh`

ReconViaGen runs in a separate micromamba worker env so the API/orchestrator env stays small. The orchestrator owns scan parsing, crop preparation, worker routing, and LiDAR metric alignment.

## Server

Install/run with micromamba:

```bash
cd server
chmod +x run.sh
./run.sh
```

By default `run.sh` prepares ReconViaGen `v0.5` under `/workspace/cache/ReconViaGen`, creates/reuses the `api` and `worker-reconviagen` micromamba envs, starts a local worker on `127.0.0.1:8011`, then starts the API on `0.0.0.0:8000`.

RunPod entrypoint:

```bash
bash -lc 'cd /workspace && if [ ! -d iphone-lidar-vggt/.git ]; then git clone https://github.com/mokyabun/iphone-lidar-vggt.git; fi && cd iphone-lidar-vggt && git fetch origin && git reset --hard origin/main && cd server && ./run.sh'
```

Useful switches:

```bash
export APP_PREPARE_RECONVIAGEN=0   # skip ReconViaGen install/update
export APP_START_RECONVIAGEN=0     # start only the API/orchestrator
export APP_UPDATE_ENVS=1           # update existing micromamba envs from yml files
export RECONVIAGEN_REFRESH=1       # force ReconViaGen setup.sh again
export HF_TOKEN=...                # required if gated Hugging Face weights need auth
```

Repeated runs reuse existing envs by default and keep build/download caches under `/workspace/cache`: pip downloads, Hugging Face models, PyTorch extension builds, and ccache compiler objects.

You can still override the generator with a command:

```bash
export RECONVIAGEN_COMMAND='python /path/to/reconviagen_runner.py --input-dir {input_dir} --output-path {output_path}'
```

or an external worker URL:

```bash
export RECONVIAGEN_WORKER_URL='http://127.0.0.1:8011'
```

For a smoke test without ReconViaGen:

```bash
cd server
RECONVIAGEN_MOCK=1 ./run.sh
```

The mock writes a synthetic mesh so you can test upload/job/result plumbing before fighting CUDA/model installs.

## API

- `GET /health`
- `GET /capabilities`
- `POST /jobs` with multipart field `scan_package`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result` -> `reconviagen_metric.ply`
- `GET /jobs/{job_id}/preview` -> `reconviagen_metric.glb`
- `GET /jobs/{job_id}/print` -> `reconviagen_metric_print_mm.stl`
- `GET /jobs/{job_id}/lidar` -> `lidar_reference.ply`

## iOS

Open:

```bash
open ios/VGGTLiDARScanApp.xcodeproj
```

The app now exposes one flow: scan, process, preview/export. Backend settings still let you change the server URL.
