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

ReconViaGen itself is treated as an external generator. The small server owns scan parsing, crop preparation, and LiDAR metric alignment.

## Server

Install/run with micromamba:

```bash
cd server
chmod +x run.sh
./run.sh
```

Configure the actual ReconViaGen runner with either a command:

```bash
export RECONVIAGEN_COMMAND='python /path/to/reconviagen_runner.py --input-dir {input_dir} --output-path {output_path}'
```

or a worker URL:

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
