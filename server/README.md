# Server Notes

This server keeps the app layer thin while still managing a local ReconViaGen worker:

1. Parse `ScanPackage.zip`.
2. Build a metric LiDAR object reference cloud.
3. Produce square RGBA crops for ReconViaGen.
4. Call the managed ReconViaGen worker.
5. Align the generated mesh back to LiDAR scale.

## Environment

```bash
./run.sh
```

`run.sh` manages two envs:

- `api`: FastAPI orchestrator for scan parsing, job routing, LiDAR alignment, and asset export; prepared from `api/requirements.txt` in a uv venv.
- `worker-reconviagen`: ReconViaGen AI worker; prepared from `workers/reconviagen/requirements.txt` in a separate uv venv.

Existing envs under `${APP_CACHE_ROOT}/venvs` are reused by default to avoid repeated dependency solving. Set `APP_UPDATE_ENVS=1` when you want to apply dependency file changes.

If the base image already has compatible PyTorch 2.4.x packages, set `RECONVIAGEN_USE_SYSTEM_TORCH=1` before the worker env is first created. The uv env will use system site-packages and skip the pinned `torch`, `torchvision`, and `torchaudio` entries from `workers/reconviagen/requirements.txt`.

`./run.sh` streams API and worker logs to the terminal. The managed worker log is also written to `${RECONVIAGEN_WORKER_LOG:-${APP_CACHE_ROOT}/worker-reconviagen.log}`. CUDA is required by default; if PyTorch cannot see a GPU, prepare/worker startup prints the torch build, `CUDA_VISIBLE_DEVICES`, device count, and `nvidia-smi` output before failing. If a reused env has a CPU-only torch build, rerun with `APP_UPDATE_ENVS=1`.

The TRELLIS.2 textured GLB postprocessor packages default to prebuilt CPython 3.10 Linux wheels from the Microsoft TRELLIS.2 Hugging Face Space. Those CuMesh/o-voxel wheels require a recent `libstdc++` with `GLIBCXX_3.4.32`; prepare will try to install a newer `libstdc++6` on apt-based root containers. Set `RECONVIAGEN_FIX_LIBSTDCXX=0` to disable that system package change. Override `RECONVIAGEN_CUMESH_URL`, `RECONVIAGEN_FLEX_GEMM_URL`, or `RECONVIAGEN_O_VOXEL_URL` only when you need to build from another source.

## ReconViaGen Worker

Default behavior:

```bash
APP_PREPARE_RECONVIAGEN=1 APP_START_RECONVIAGEN=1 ./run.sh
```

One-shot worker runner:

```bash
VIRTUAL_ENV="${APP_CACHE_ROOT:-/workspace/cache}/venvs/worker-reconviagen" \
PATH="${APP_CACHE_ROOT:-/workspace/cache}/venvs/worker-reconviagen/bin:$PATH" \
PYTHONPATH="$PWD" python -m workers.reconviagen.main \
  --once \
  --input-dir /path/to/reconviagen_views \
  --output-path /path/to/raw_reconviagen.glb
```

External override modes are still available:

```bash
export RECONVIAGEN_COMMAND='python /path/to/reconviagen_runner.py --input-dir {input_dir} --output-path {output_path}'
export RECONVIAGEN_WORKER_URL='http://127.0.0.1:8011'
```

The command/worker must write a GLB or other `trimesh`-readable mesh to `{output_path}`.

## Script Layout

- `run.sh`: top-level orchestrator.
- `api/*.py`: orchestrator code.
- `api/requirements.txt`: API uv env dependencies.
- `scripts/install/uv.sh`: uv bootstrap.
- `scripts/env/api.sh`: API env create/update.
- `scripts/env/build_cache.sh`: persistent pip, torch extension, Hugging Face, and ccache paths.
- `workers/reconviagen/requirements.txt`: ReconViaGen worker uv env dependencies.
- `workers/reconviagen/*.py`: ReconViaGen AI worker code.
- `workers/reconviagen/scripts/prepare.sh`: ReconViaGen worker setup.
- `workers/reconviagen/scripts/*.sh`: split repo/env/package/CUDA-extension steps.

`scripts/prepare_reconviagen.sh` remains as a compatibility wrapper.

## Smoke Test Mode

```bash
RECONVIAGEN_MOCK=1 ./run.sh
```

This bypasses the generator and writes a synthetic GLB so API, scan parsing, and alignment code can be tested without GPU dependencies.

Open `http://127.0.0.1:8000/` while the API is running to use the browser upload tester. It accepts local `ScanPackage.zip` files, including packages under the repository `test/` directory, and polls the async `/jobs` flow.
