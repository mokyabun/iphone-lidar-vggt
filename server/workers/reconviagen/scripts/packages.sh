pip_install_if_missing() {
  local import_name="$1"
  shift
  if micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -c "import ${import_name}" >/dev/null 2>&1; then
    return 0
  fi
  LOG_PREFIX="prepare-reconviagen" log "Installing ${import_name}."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install "$@"
}

optional_pip_install_if_missing() {
  local import_name="$1"
  shift
  if ! pip_install_if_missing "${import_name}" "$@"; then
    LOG_PREFIX="prepare-reconviagen" log "Optional ${import_name} install failed; continuing with raw mesh export fallback."
  fi
}

install_reconviagen_base_packages() {
  LOG_PREFIX="prepare-reconviagen" log "Installing ReconViaGen base Python packages."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install --quiet \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
    rembg onnxruntime open3d xatlas pyvista pymeshfix igraph lpips \
    "kornia==0.8.2" "huggingface_hub==0.36.2" "transformers==4.57.1"
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install --quiet \
    zstandard pillow-simd rtree fast-simplification
  pip_install_if_missing utils3d \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
}

verify_reconviagen_base_package_pins() {
  LOG_PREFIX="prepare-reconviagen" log "Verifying ReconViaGen Python package pins."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python - <<'PY'
from importlib.metadata import version

expected = {
    "huggingface-hub": "0.36.2",
    "transformers": "4.57.1",
    "kornia": "0.8.2",
}
for package, expected_version in expected.items():
    installed = version(package)
    print(f"[prepare-reconviagen] {package}={installed}")
    if installed != expected_version:
        raise SystemExit(f"[prepare-reconviagen] Expected {package}=={expected_version}, got {installed}.")
PY
}

patch_flex_gemm_triton_autotuner() {
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python - <<'PY'
from pathlib import Path
import sysconfig

path = Path(sysconfig.get_paths()["purelib"]) / "flex_gemm" / "utils" / "autotuner.py"
if not path.exists():
    raise SystemExit(0)

text = path.read_text()
if "autotuner_kwargs = dict(" in text:
    raise SystemExit(0)

old = """        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
"""
new = """        autotuner_kwargs = dict(
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
        )
        if "do_bench" in inspect.signature(triton.runtime.Autotuner.__init__).parameters:
            autotuner_kwargs["do_bench"] = do_bench
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            **autotuner_kwargs,
        )
"""
if old not in text:
    raise SystemExit("FlexGEMM autotuner layout changed; cannot patch Triton compatibility.")
path.write_text(text.replace(old, new))
PY
}
