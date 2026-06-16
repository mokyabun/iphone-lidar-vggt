# Server Notes

This server deliberately does not install ReconViaGen. It is the thin layer around ReconViaGen:

1. Parse `ScanPackage.zip`.
2. Build a metric LiDAR object reference cloud.
3. Produce square RGBA crops for ReconViaGen.
4. Call an external ReconViaGen command or worker.
5. Align the generated mesh back to LiDAR scale.

## Environment

```bash
micromamba create -y -f environment.yml
micromamba run -n lidar-reconviagen python -m uvicorn lidar_reconviagen.api:app --host 0.0.0.0 --port 8000
```

or:

```bash
./run.sh
```

## ReconViaGen Hook

Command mode:

```bash
export RECONVIAGEN_COMMAND='python /path/to/reconviagen_runner.py --input-dir {input_dir} --output-path {output_path}'
```

Worker mode:

```bash
export RECONVIAGEN_WORKER_URL='http://127.0.0.1:8011'
```

The runner/worker must write a GLB or other `trimesh`-readable mesh to `{output_path}`.

## Smoke Test Mode

```bash
RECONVIAGEN_MOCK=1 ./run.sh
```

This bypasses the generator and writes a synthetic GLB so API, scan parsing, and alignment code can be tested without GPU dependencies.
