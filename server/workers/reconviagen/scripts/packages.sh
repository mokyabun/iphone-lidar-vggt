pip_install_if_missing() {
  local import_name="$1"
  shift
  if venv_run "${RECONVIAGEN_ENV_DIR}" python -c "import ${import_name}" >/dev/null 2>&1; then
    return 0
  fi
  LOG_PREFIX="prepare-reconviagen" log "Installing ${import_name}."
  uv pip install --python "$(venv_python "${RECONVIAGEN_ENV_DIR}")" "$@"
}

optional_pip_install_if_missing() {
  local import_name="$1"
  shift
  if ! pip_install_if_missing "${import_name}" "$@"; then
    LOG_PREFIX="prepare-reconviagen" log "Optional ${import_name} install failed; continuing with raw mesh export fallback."
  fi
}

install_reconviagen_base_packages() {
  LOG_PREFIX="prepare-reconviagen" log "ReconViaGen base Python packages are managed by workers/reconviagen/requirements.txt."
  pip_install_if_missing utils3d \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
}

verify_reconviagen_base_package_pins() {
  LOG_PREFIX="prepare-reconviagen" log "Verifying ReconViaGen Python package pins."
  venv_run "${RECONVIAGEN_ENV_DIR}" python - <<'PY'
from importlib.metadata import version

from packaging.version import Version

exact_expected = {
    "huggingface-hub": "0.33.4",
    "kornia": "0.8.2",
}
minimum_expected = {
    "transformers": "4.56.0",
}
for package, expected_version in exact_expected.items():
    installed = version(package)
    print(f"[prepare-reconviagen] {package}={installed}")
    if installed != expected_version:
        raise SystemExit(f"[prepare-reconviagen] Expected {package}=={expected_version}, got {installed}.")
for package, minimum_version in minimum_expected.items():
    installed = version(package)
    print(f"[prepare-reconviagen] {package}={installed}")
    if Version(installed) < Version(minimum_version):
        raise SystemExit(f"[prepare-reconviagen] Expected {package}>={minimum_version}, got {installed}.")
PY
}

patch_flex_gemm_triton_autotuner() {
  venv_run "${RECONVIAGEN_ENV_DIR}" python - <<'PY'
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
