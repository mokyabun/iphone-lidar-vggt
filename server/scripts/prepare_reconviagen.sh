#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
RECONVIAGEN_REPO_URL="${RECONVIAGEN_REPO_URL:-https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git}"
RECONVIAGEN_REPO_REF="${RECONVIAGEN_REPO_REF:-v0.5}"
RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
RECONVIAGEN_ENV_NAME="${RECONVIAGEN_ENV_NAME:-reconviagen-v05}"
# CUDA extension flags only — --basic is always run separately.
RECONVIAGEN_CUDA_FLAGS="${RECONVIAGEN_CUDA_FLAGS:---xformers --flash-attn --nvdiffrec --spconv --kaolin --nvdiffrast}"
RECONVIAGEN_CUMESH_URL="${RECONVIAGEN_CUMESH_URL:-git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f}"
RECONVIAGEN_FLEX_GEMM_URL="${RECONVIAGEN_FLEX_GEMM_URL:-git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e}"
RECONVIAGEN_INSTALL_POSTPROCESSORS="${RECONVIAGEN_INSTALL_POSTPROCESSORS:-1}"
RECONVIAGEN_REFRESH="${RECONVIAGEN_REFRESH:-0}"

log() {
  printf '[prepare_reconviagen] %s\n' "$*"
}

env_exists() {
  micromamba env list | awk '{print $1}' | grep -qx "${RECONVIAGEN_ENV_NAME}"
}

sync_repo() {
  mkdir -p "$(dirname "${RECONVIAGEN_REPO_DIR}")"
  if [ -d "${RECONVIAGEN_REPO_DIR}/.git" ]; then
    log "Updating ReconViaGen ${RECONVIAGEN_REPO_REF}."
    git -C "${RECONVIAGEN_REPO_DIR}" fetch --depth 1 origin "${RECONVIAGEN_REPO_REF}"
    git -C "${RECONVIAGEN_REPO_DIR}" reset --hard FETCH_HEAD
    git -C "${RECONVIAGEN_REPO_DIR}" submodule update --init --recursive
  else
    log "Cloning ReconViaGen ${RECONVIAGEN_REPO_REF}."
    rm -rf "${RECONVIAGEN_REPO_DIR}"
    git clone --recursive --depth 1 --branch "${RECONVIAGEN_REPO_REF}" \
      "${RECONVIAGEN_REPO_URL}" "${RECONVIAGEN_REPO_DIR}"
  fi
}

repo_revision() {
  git -C "${RECONVIAGEN_REPO_DIR}" rev-parse HEAD
}

# Stamp file lives next to the conda env prefix so it is wiped if the env
# is recreated, which is the right time to force a rebuild.
cuda_stamp_file() {
  echo "${MAMBA_ROOT_PREFIX}/envs/${RECONVIAGEN_ENV_NAME}/.reconviagen-cuda-stamp"
}

cuda_runtime_stamp() {
  # Only inputs that actually affect CUDA extension compilation.
  {
    printf 'repo=%s\n' "$(repo_revision)"
    printf 'cuda_flags=%s\n' "${RECONVIAGEN_CUDA_FLAGS}"
    printf 'postprocessors=%s\n' "${RECONVIAGEN_INSTALL_POSTPROCESSORS}"
  } | cksum | awk '{print $1}'
}

ensure_env() {
  if env_exists; then
    log "Updating micromamba env ${RECONVIAGEN_ENV_NAME}."
    micromamba env update -y -n "${RECONVIAGEN_ENV_NAME}" -f "${SERVER_DIR}/reconviagen-environment.yml"
  else
    log "Creating micromamba env ${RECONVIAGEN_ENV_NAME}."
    micromamba create -y -n "${RECONVIAGEN_ENV_NAME}" -f "${SERVER_DIR}/reconviagen-environment.yml"
  fi
}

verify_torch() {
  log "Checking PyTorch in ${RECONVIAGEN_ENV_NAME}."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python - <<'PY'
import torch

print(f"[prepare_reconviagen] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.__version__.startswith("2.4."):
    raise SystemExit("[prepare_reconviagen] ReconViaGen v0.5 CUDA extensions require PyTorch 2.4.x.")
PY
}

install_base_packages() {
  # Keep these aligned with ReconViaGen v0.5 setup.sh/setup_update.sh.
  log "Installing ReconViaGen base Python packages."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install --quiet \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
    rembg onnxruntime open3d xatlas pyvista pymeshfix igraph lpips \
    "kornia==0.8.2" "huggingface_hub==0.36.2" "transformers==4.57.1"
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install --quiet \
    zstandard pillow-simd rtree fast-simplification
  pip_install_if_missing utils3d \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
}

verify_base_package_pins() {
  log "Verifying ReconViaGen Python package pins."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python - <<'PY'
from importlib.metadata import version

expected = {
    "huggingface-hub": "0.36.2",
    "transformers": "4.57.1",
    "kornia": "0.8.2",
}
for package, expected_version in expected.items():
    installed = version(package)
    print(f"[prepare_reconviagen] {package}={installed}")
    if installed != expected_version:
        raise SystemExit(f"[prepare_reconviagen] Expected {package}=={expected_version}, got {installed}.")
PY
}

pip_install_if_missing() {
  local import_name="$1"
  shift
  if micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -c "import ${import_name}" >/dev/null 2>&1; then
    return 0
  fi
  log "Installing ${import_name}."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install "$@"
}

optional_pip_install_if_missing() {
  local import_name="$1"
  shift
  if ! pip_install_if_missing "${import_name}" "$@"; then
    log "Optional ${import_name} install failed; continuing with raw mesh export fallback."
  fi
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

build_cuda_extensions() {
  local stamp_file expected_stamp installed_stamp
  stamp_file="$(cuda_stamp_file)"
  expected_stamp="$(cuda_runtime_stamp)"
  installed_stamp=""
  if [ -f "${stamp_file}" ]; then
    installed_stamp="$(cat "${stamp_file}")"
  fi
  if [ "${RECONVIAGEN_REFRESH}" != "1" ] && [ "${installed_stamp}" = "${expected_stamp}" ]; then
    log "ReconViaGen CUDA extensions are already current."
    return 0
  fi

  log "Building ReconViaGen CUDA extensions. This can take a while on a fresh pod."
  rm -rf /tmp/extensions
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" bash -lc \
    "cd '${RECONVIAGEN_REPO_DIR}' && . ./setup.sh ${RECONVIAGEN_CUDA_FLAGS}"

  if [ "${RECONVIAGEN_INSTALL_POSTPROCESSORS}" = "1" ]; then
    optional_pip_install_if_missing cumesh "${RECONVIAGEN_CUMESH_URL}" --no-build-isolation --no-deps
    optional_pip_install_if_missing flex_gemm "${RECONVIAGEN_FLEX_GEMM_URL}" --no-build-isolation --no-deps
    if [ -d "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" ]; then
      optional_pip_install_if_missing o_voxel "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation --no-deps
    fi
    patch_flex_gemm_triton_autotuner
  else
    log "Skipping optional postprocessors. Set RECONVIAGEN_INSTALL_POSTPROCESSORS=1 to try CuMesh/FlexGEMM/o-voxel."
  fi

  mkdir -p "$(dirname "${stamp_file}")"
  printf '%s\n' "${expected_stamp}" > "${stamp_file}"
}

if ! command -v micromamba >/dev/null 2>&1; then
  log "micromamba is required before preparing ReconViaGen."
  exit 2
fi

if [ "$(uname -s)" != "Linux" ]; then
  log "Skipping ReconViaGen CUDA setup on $(uname -s)."
  exit 0
fi

sync_repo
ensure_env
verify_torch
install_base_packages
verify_base_package_pins
build_cuda_extensions
log "ReconViaGen is ready in micromamba env ${RECONVIAGEN_ENV_NAME}."
