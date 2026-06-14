# VGGT iPhone LiDAR Scanner

MVP scaffold for a course project that uses an iPhone/iPad LiDAR device as the capture frontend and a Python backend as the reconstruction/VGGT post-processing stage.

The project is split around one stable contract: a `ScanPackage.zip` containing RGB frames, LiDAR depth, confidence maps, camera intrinsics, and ARKit camera poses. The iOS app creates that package; the backend reconstructs a metric point cloud/mesh baseline and can run VGGT when a GPU environment is available.

## Repository Layout

```text
ios/
  VGGTLiDARScanApp.xcodeproj/     Minimal Xcode project
  VGGTLiDARScanApp/               SwiftUI + ARKit capture app

backend/
  vggt_lidar_scan/                Python package, CLI, FastAPI backend
  tests/                          Unit tests for package IO and geometry helpers

pyproject.toml                    Python dependency/test configuration
```

## MVP Pipeline

1. Scan with a LiDAR-capable iPad/iPhone.
2. Export `ScanPackage.zip` from the app.
3. Run the LiDAR baseline:

```bash
uv run scan-reconstruct path/to/ScanPackage.zip --output runs/demo
```

4. Optional VGGT stage on a GPU host:

```bash
uv run scan-reconstruct path/to/ScanPackage.zip --output runs/demo --run-vggt
```

`--run-vggt` currently supports two paths:

- If `VGGT_RUNNER` is set, the backend calls that command with `--image-dir` and `--output-dir`.
- If the local `vggt` Python package is installed, the backend runs a direct point-cloud inference adapter.

## iOS App

Open `ios/VGGTLiDARScanApp.xcodeproj` in Xcode and run on a LiDAR-capable device. The app intentionally targets device capture first; simulator builds are useful only for compilation checks.

The exported package format is:

```text
ScanPackage/
  metadata.json
  frames.jsonl
  images/frame_000001.jpg
  depth/frame_000001.float32
  confidence/frame_000001.uint8
```

Frame records include RGB/depth dimensions, scaled intrinsics for the depth map, and ARKit camera-to-world transform.

## Backend API

Start the API:

```bash
uv run uvicorn vggt_lidar_scan.api:app --reload
```

Endpoints:

- `POST /reconstruct` multipart upload field: `scan_package`; returns `scan_final.ply` directly for the app demo path
- `POST /jobs` multipart upload field: `scan_package`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

Results include:

- `scan_lidar_points.ply`
- `scan_lidar_tsdf.ply` when Open3D TSDF succeeds
- `scan_object_ai_mesh.ply` when ReconViaGen succeeds
- `scan_vggt_points.ply` when VGGT succeeds
- `scan_final.ply`
- `metrics.json`

## iOS Network Demo Flow

For the app demo, run the backend on a machine reachable from the iPad:

```bash
uv run uvicorn vggt_lidar_scan.api:app --host 0.0.0.0 --port 8000
```

In the app, set the backend URL to the host IP, for example `http://192.168.0.20:8000`. After scanning, tap `Backend`; the app uploads `ScanPackage.zip`, receives `scan_final.ply`, and opens a point-cloud preview.

## Practical Scope

The first target is medium-size object and furniture scanning, not CAD-grade measurement. Good early test objects are textured boxes, chairs, bags, and desks. Transparent, reflective, very thin, dark, or textureless surfaces are expected failure cases.

## VGGT Auto Download

The backend can prepare VGGT automatically:

```bash
uv run --extra vggt vggt-prepare
```

Runtime environment variables:

- `VGGT_AUTO_DOWNLOAD=1` clones `facebookresearch/vggt` into `/workspace/cache/vggt-lidar/vggt` by default if the Python package is not already available.
- `VGGT_REPO_DIR=/path/to/vggt` points the backend at a manually cloned VGGT repo.
- `VGGT_CACHE_ROOT=/workspace/cache/vggt-lidar` changes the repo/weight cache root.
- `VGGT_ALLOW_CPU=1` allows very slow CPU-only VGGT experiments; GPU is expected for real use.

On the first direct VGGT run, `VGGT.from_pretrained("facebook/VGGT-1B")` downloads the Hugging Face checkpoint into the configured cache.

## RunPod Without Rebuilding Images

Use a stock RunPod PyTorch image and run the repository bootstrap script at container start.

Recommended RunPod template values:

- Container image: `runpod/pytorch:cuda12`
- Exposed HTTP port: `8000`
- Container start command:

```bash
bash -lc 'curl -fsSL https://raw.githubusercontent.com/mokyabun/iphone-lidar-vggt/main/run.sh | bash'
```

`run.sh` automatically installs the small system dependencies, clones or updates this repository under `/workspace/iphone-lidar-vggt`, installs the Python package, and starts FastAPI on `0.0.0.0:8000`.

RunPod's persistent network volume is mounted at `/workspace`. The bootstrap
keeps the replaceable Git checkout in `/workspace/iphone-lidar-vggt` and stores
all large reusable state outside that checkout:

- `/workspace/cache/vggt-lidar/huggingface`: Hugging Face model snapshots
- `/workspace/cache/vggt-lidar/torch`: Torch Hub models, including DINOv2
- `/workspace/cache/vggt-lidar/ultralytics`: SAM weights
- `/workspace/cache/vggt-lidar/rembg`: background-removal weights
- `/workspace/cache/ReconViaGen`: ReconViaGen and vendored source repositories
- `/workspace/cache/reconviagen-v05-env`: isolated Python/CUDA environment
- `/workspace/cache/{torch-extensions,triton,cuda}`: compiled runtime caches

The code checkout is reset to the selected Git revision on each launch, while
these model and environment directories remain intact across Pod replacement.
Set `APP_PERSIST_ROOT` only when the persistent volume is mounted somewhere
other than `/workspace`.

Useful overrides:

```bash
APP_REPO_URL=https://github.com/mokyabun/iphone-lidar-vggt.git
APP_REPO_REF=main
APP_DIR=/workspace/iphone-lidar-vggt
APP_UPDATE_MODE=reset
APP_PREPARE_VGGT=0
APP_PREFETCH_VGGT=0
APP_INSTALL_EXTRAS=reconstruction,vggt,segmentation
APP_PREPARE_RECONVIAGEN=1
APP_PREFETCH_RECONVIAGEN=1
RECONVIAGEN_PRELOAD=1
HF_TOKEN=hf_your_read_token
VGGT_MAX_IMAGES=12
VGGT_PRELOAD=0
SCAN_MAX_FRAMES=24
SCAN_RUN_TSDF=0
MESH_METHOD=object_tsdf
OBJECT_MASK_BACKEND=sam3_depth
OBJECT_SAM_MODEL=sam3.pt
```

`APP_PREPARE_VGGT=1` clones `facebookresearch/vggt` and installs it as an editable Python package. `APP_PREFETCH_VGGT=1` also downloads the model checkpoint before serving.

Performance knobs:

- `VGGT_MAX_IMAGES=12` limits VGGT input keyframes. Increase for quality, decrease for speed.
- `VGGT_PRELOAD=0` avoids keeping the standalone VGGT model beside the larger ReconViaGen worker.
- `SCAN_MAX_FRAMES=24` limits LiDAR baseline frames.
- `SCAN_RUN_TSDF=0` skips the slower Open3D TSDF mesh stage. Set to `1` only when you specifically want TSDF output.
- `OBJECT_MASK_BACKEND=sam3_depth` uses SAM 3 center-point segmentation refined by LiDAR depth, with depth-only fallback.

To force the faster depth-only object mask:

```bash
OBJECT_MASK_BACKEND=depth
```

For a private fork or different account, set `APP_REPO_URL` in the RunPod template environment variables and keep the start command pointed at that repo's raw `run.sh`.

To skip VGGT checkpoint predownload for faster pod startup, set:

```bash
APP_PREFETCH_VGGT=0
```

## ReconViaGen AI Mesh

The app's `ReconViaGen` option runs the public ReconViaGen v0.5 hybrid pipeline:

1. Select up to six sharp, angularly diverse masked RGB views.
2. Run ReconViaGen's VGGT-based sparse structure estimation.
3. Generate 1024-cascade geometry and PBR appearance with TRELLIS.2.
4. Remesh and texture the generated asset.
5. Align its rotation, translation, and uniform scale to the metric LiDAR object points.
6. Return an ASCII PLY triangle mesh that the iOS preview can render.

The implementation fixes ReconViaGen's official recommended quality settings:
`adaptive_guidance_weight`, `1024_cascade`, and `ss_source=mesh`. If generation
fails, the existing printable metric LiDAR mesh remains the final result.

The RunPod bootstrap installs ReconViaGen in a separate Python 3.10,
PyTorch 2.4, CUDA 12.1 micromamba environment. The first startup downloads
and compiles several CUDA extensions and can take a while. Later starts reuse
the repository, environment, and model caches. The model worker is loaded once
in the background and reused by reconstruction requests.

TRELLIS.2 uses Meta's gated
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
checkpoint. Accept the model license with the same Hugging Face account as the
token, then add `HF_TOKEN` as a RunPod template or Pod environment variable.
`run.sh` checks this access before starting the worker and reports an immediate,
actionable backend error when access is missing.

Use a dedicated fine-grained read token restricted to that gated model when
possible. A regular account-wide `read` token also works, but grants the Pod
read access to every repository that account can read. Write permission is not
required.

The ReconViaGen path downloads about 21 GB of model checkpoints. The isolated
CUDA/Python environment, source trees, compiled extensions, temporary download
headroom, and generated scans require substantially more space. A **75 GB**
network volume is the practical minimum; **100 GB is recommended**. Use
**150 GB** if standalone VGGT weights, multiple reconstruction models, or many
scan results will be retained.

Recommended A40 settings:

```bash
APP_PREPARE_RECONVIAGEN=1
APP_PREFETCH_RECONVIAGEN=1
RECONVIAGEN_PRELOAD=1
HF_TOKEN=hf_your_read_token
RECONVIAGEN_MAX_IMAGES=6
RECONVIAGEN_PIPELINE_TYPE=1024_cascade
RECONVIAGEN_SS_SOURCE=mesh
RECONVIAGEN_LOW_VRAM=1
RECONVIAGEN_TEXTURE_SIZE=2048
RECONVIAGEN_DECIMATION_TARGET=500000
```

Official project: `https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen/tree/v0.5`

## Docker

RunPod GPU image:

```bash
docker build --platform=linux/amd64 -f Dockerfile.runpod -t vggt-lidar-runpod .
docker run --gpus all -p 8000:8000 -v "$PWD/runs:/app/runs" vggt-lidar-runpod
```

To bake the VGGT repo/checkpoint into the image:

```bash
docker build --platform=linux/amd64 -f Dockerfile.runpod --build-arg PREFETCH_VGGT=1 -t vggt-lidar-runpod .
```

To override the RunPod PyTorch base tag:

```bash
docker build --platform=linux/amd64 -f Dockerfile.runpod \
  --build-arg RUNPOD_PYTORCH_IMAGE=runpod/pytorch:1.0.5-dev-fix-image-vulnerabilities-cu1290-torch290-ubuntu2204 \
  -t vggt-lidar-runpod .
```

Mac mini M4 Pro development image:

```bash
docker build -f Dockerfile.mac -t vggt-lidar-mac-dev .
docker run -p 8000:8000 -v "$PWD/runs:/app/runs" vggt-lidar-mac-dev
```

The Mac Dockerfile is for FastAPI/package development. It does not provide CUDA; use RunPod for real VGGT inference.
