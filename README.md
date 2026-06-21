# iPhone LiDAR + ReconViaGen

Capture RGB-D object scans on an iPhone with LiDAR, send the scan package to a
FastAPI backend, run [ReconViaGen](https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen)
on multi-view crops, then align the generated mesh back to real-world LiDAR scale.

```text
iPhone LiDAR ScanPackage.zip
  -> RGB images + depth + confidence + camera poses
  -> depth-based central object mask
  -> metric LiDAR reference point cloud
  -> ReconViaGen multi-view RGBA crops
  -> ReconViaGen raw mesh
  -> LiDAR scale + PCA/ICP alignment
  -> metric PLY / GLB / STL outputs
```

![iOS app screen](docs/app-screen.png)

![Point cloud result](docs/point-cloud-result.png)

See [SCHEMA.md](SCHEMA.md) for the full pipeline diagram and stage-by-stage flow.

## Runtime design

The backend runs as **two isolated processes**, each in its own
[micromamba](https://mamba.readthedocs.io/) environment. micromamba (not uv / not
system pip) is used because ReconViaGen's `setup.sh` is conda-native and its
TRELLIS.2 postprocessor wheels are cp310-only — micromamba gives a reproducible
`python 3.10 / torch 2.4.0 / cu121` env regardless of the base image's Python.

| Process | Env | Python | Role |
| --- | --- | --- | --- |
| API orchestrator (`:8000`) | `api` | 3.11 | scan parsing, masking/crops, LiDAR cloud, metric alignment, asset export, job queue, static tester |
| ReconViaGen worker (`:8011`) | `reconviagen` | 3.10 | runs `setup.sh` + TRELLIS.2 hybrid pipeline (`run_multi_image`) |

`server/run.sh` bootstraps micromamba, builds or reuses both envs, starts the
worker, waits for its `/health`, then starts the API. ReconViaGen itself is
cloned at a pinned ref (`v0.5`) into the cache at build time.

## Repository layout

- `ios/VGGTLiDARScanApp` — SwiftUI iPhone app (capture, export, upload, preview).
- `server/api` — FastAPI orchestrator.
- `server/worker` — thin ReconViaGen worker (imports `trellis`/`trellis2` from the cloned repo).
- `server/scripts` — micromamba bootstrap + env builders + model prefetch.
- `server/run.sh` — one-command entrypoint.
- `legacy/` — previous uv-based implementation, kept for reference only.
- `docs` — README images.

## RunPod entrypoint

Main runtime is a RunPod **pytorch** image (any py3.10/py3.11 variant — micromamba
makes the worker env independent of it):

```bash
bash -lc 'cd /workspace && if [ ! -d iphone-lidar-vggt/.git ]; then git clone https://github.com/mokyabun/iphone-lidar-vggt.git; fi && cd iphone-lidar-vggt && git fetch origin && git reset --hard origin/main && cd server && ./run.sh'
```

Gated Hugging Face weights:

```bash
export HF_TOKEN=...   # run.sh maps this to HUGGINGFACE_HUB_TOKEN if unset
```

## Environment lifecycle

By default both envs are reused when their spec hash matches. The ReconViaGen
CUDA build (flash-attn / spconv / kaolin / nvdiffrast) takes 20–40 min, so repeat
boots should only rebuild when requirements, Python versions, CUDA flags, wheel
URLs, or the ReconViaGen ref change.

```bash
export APP_FORCE_REBUILD=1  # force a rebuild even when the spec hash matches
export APP_REUSE_ENV=0      # opt out of env reuse and rebuild every run
```

The spec hash covers the python version, requirements, `setup.sh` CUDA flags,
postprocessor wheel URLs, and ReconViaGen ref — change any of them and the env
rebuilds automatically.

## Useful environment variables

Server / runtime:

```bash
export APP_HOST=0.0.0.0
export APP_PORT=8000
export APP_LOCAL_ROOT=/opt/iphone-lidar-vggt   # envs + micromamba live here (local disk)
export APP_CACHE_ROOT=/workspace/cache         # HF/torch/pip caches + ReconViaGen clone
export APP_PYTHON_VERSION=3.11                 # api env
export APP_PREPARE_RECONVIAGEN=1
export APP_START_RECONVIAGEN=1
```

ReconViaGen env build:

```bash
export RECONVIAGEN_PYTHON_VERSION=3.10
export RECONVIAGEN_REPO_REF=v0.5
export RECONVIAGEN_CUDA_FLAGS='--xformers --flash-attn --spconv --kaolin --nvdiffrast'
export RECONVIAGEN_INSTALL_POSTPROCESSORS=1
export RECONVIAGEN_PREFETCH_MODELS=1
```

Scan processing and alignment:

```bash
export SCAN_MAX_FRAMES=24
export SCAN_STRIDE=4
export RECONVIAGEN_MAX_IMAGES=6
export RECONVIAGEN_INPUT_SIZE=512
export RECONVIAGEN_CROP_PADDING=0.18
export OBJECT_DEPTH_BAND_METERS=0.45
export OBJECT_CENTER_FRACTION=0.35
export ALIGNMENT_SAMPLES=6000
export ALIGNMENT_ICP_ITERATIONS=8
export EXPORT_PRINT_STL=1
```

External generator overrides (skip the managed worker):

```bash
export RECONVIAGEN_WORKER_URL='http://127.0.0.1:8011'
export RECONVIAGEN_COMMAND='python /path/to/runner.py --input-dir {input_dir} --output-path {output_path}'
```

## API

```text
GET  /                       static browser upload tester
GET  /health
GET  /capabilities
POST /jobs                   multipart field: scan_package
GET  /jobs/{job_id}
GET  /jobs/{job_id}/result   reconviagen_metric.ply
GET  /jobs/{job_id}/preview  reconviagen_metric.glb
GET  /jobs/{job_id}/print    reconviagen_metric_print_mm.stl
GET  /jobs/{job_id}/lidar    lidar_reference.ply
POST /reconstruct            synchronous development endpoint
```

The iOS app uses the async job flow: `POST /jobs` queues a job,
`GET /jobs/{job_id}` reports `queued|processing|complete|failed`, `/result`
returns the metric PLY. The static tester at `http://127.0.0.1:8000/` can upload
`test/ScanPackage-*.zip` from a browser.

## Smoke test (no GPU)

```bash
cd server
RECONVIAGEN_MOCK=1 ./run.sh
```

Mock mode skips the ReconViaGen env/worker entirely and writes a synthetic mesh,
exercising upload, job polling, scan parsing, alignment, and asset export. On
macOS the ReconViaGen CUDA env build is skipped automatically (logged).
