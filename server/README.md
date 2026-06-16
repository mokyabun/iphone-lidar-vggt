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

- `lidar-reconviagen`: small FastAPI/LiDAR alignment server.
- `reconviagen-v05`: heavy ReconViaGen worker, prepared from `reconviagen-environment.yml`.

## ReconViaGen Worker

Default behavior:

```bash
APP_PREPARE_RECONVIAGEN=1 APP_START_RECONVIAGEN=1 ./run.sh
```

One-shot worker runner:

```bash
micromamba run -n reconviagen-v05 python -m reconviagen_worker.main \
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

## Smoke Test Mode

```bash
RECONVIAGEN_MOCK=1 ./run.sh
```

This bypasses the generator and writes a synthetic GLB so API, scan parsing, and alignment code can be tested without GPU dependencies.
