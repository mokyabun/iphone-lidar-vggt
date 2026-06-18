#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/lib.sh"

LOG_PREFIX="env-reconviagen"

# --- configuration -----------------------------------------------------------
export APP_LOCAL_ROOT="${APP_LOCAL_ROOT:-/opt/iphone-lidar-vggt}"
export APP_BIN_DIR="${APP_BIN_DIR:-${APP_LOCAL_ROOT}/bin}"
export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"

export RECONVIAGEN_PYTHON_VERSION="${RECONVIAGEN_PYTHON_VERSION:-3.10}"
export RECONVIAGEN_ENV_DIR="${RECONVIAGEN_ENV_DIR:-${APP_LOCAL_ROOT}/envs/reconviagen}"
export RECONVIAGEN_REPO_URL="${RECONVIAGEN_REPO_URL:-https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git}"
export RECONVIAGEN_REPO_REF="${RECONVIAGEN_REPO_REF:-v0.5}"
export RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
export RECONVIAGEN_CUDA_FLAGS="${RECONVIAGEN_CUDA_FLAGS:---xformers --flash-attn --spconv --kaolin --nvdiffrast}"
export RECONVIAGEN_INSTALL_POSTPROCESSORS="${RECONVIAGEN_INSTALL_POSTPROCESSORS:-1}"
export RECONVIAGEN_PREFETCH_MODELS="${RECONVIAGEN_PREFETCH_MODELS:-1}"

# Keep TRELLIS.2 postprocessor wheels on a Python 3.10 compatible snapshot.
export RECONVIAGEN_TRELLIS2_WHEEL_REF="${RECONVIAGEN_TRELLIS2_WHEEL_REF:-90d6619f8152991009e68a6bdf6217a8cb7d8bb3}"
export RECONVIAGEN_TRELLIS2_WHEEL_BASE="${RECONVIAGEN_TRELLIS2_WHEEL_BASE:-https://huggingface.co/spaces/microsoft/TRELLIS.2/resolve/${RECONVIAGEN_TRELLIS2_WHEEL_REF}/wheels}"
export RECONVIAGEN_CUMESH_URL="${RECONVIAGEN_CUMESH_URL:-${RECONVIAGEN_TRELLIS2_WHEEL_BASE}/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl}"
export RECONVIAGEN_FLEX_GEMM_URL="${RECONVIAGEN_FLEX_GEMM_URL:-${RECONVIAGEN_TRELLIS2_WHEEL_BASE}/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl}"
export RECONVIAGEN_O_VOXEL_URL="${RECONVIAGEN_O_VOXEL_URL:-${RECONVIAGEN_TRELLIS2_WHEEL_BASE}/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl}"

requirements_file="${SERVER_DIR}/worker/requirements.txt"
env_python_bin="$(env_python "${RECONVIAGEN_ENV_DIR}")"

if [ "$(uname -s)" != "Linux" ]; then
  log "Skipping ReconViaGen CUDA env build on $(uname -s); only the API runs off-Linux."
  exit 0
fi

# --- ReconViaGen source (pinned runtime clone) -------------------------------
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
  git -C "${RECONVIAGEN_REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown
}

# --- spec hash / reuse -------------------------------------------------------
spec_stamp() {
  {
    printf 'py=%s\n' "${RECONVIAGEN_PYTHON_VERSION}"
    printf 'cuda_flags=%s\n' "${RECONVIAGEN_CUDA_FLAGS}"
    printf 'repo_ref=%s\n' "${RECONVIAGEN_REPO_REF}"
    printf 'postprocessors=%s\n' "${RECONVIAGEN_INSTALL_POSTPROCESSORS}"
    printf 'cumesh=%s\n' "${RECONVIAGEN_CUMESH_URL}"
    printf 'flex=%s\n' "${RECONVIAGEN_FLEX_GEMM_URL}"
    printf 'ovoxel=%s\n' "${RECONVIAGEN_O_VOXEL_URL}"
    cat "${requirements_file}"
  } | cksum | awk '{print $1}'
}

stamp_file="${RECONVIAGEN_ENV_DIR}/.spec-stamp"
expected_stamp="$(spec_stamp)"

# --- package install helpers (env pip, no uv shim) ---------------------------
pip_install_if_missing() {
  local import_name="$1"
  shift
  if mm_run "${RECONVIAGEN_ENV_DIR}" python -c "import ${import_name}" >/dev/null 2>&1; then
    return 0
  fi
  log "Installing ${import_name}."
  mm_pip "${RECONVIAGEN_ENV_DIR}" install "$@"
}

optional_pip_install_if_missing() {
  local import_name="$1"
  shift
  if ! pip_install_if_missing "${import_name}" "$@"; then
    log "Optional ${import_name} install failed; continuing with raw mesh export fallback."
  fi
}

install_postprocessors() {
  [ "${RECONVIAGEN_INSTALL_POSTPROCESSORS}" = "1" ] || {
    log "Skipping optional postprocessors (RECONVIAGEN_INSTALL_POSTPROCESSORS=0)."
    return 0
  }
  optional_pip_install_if_missing cumesh "${RECONVIAGEN_CUMESH_URL}" --no-build-isolation --no-deps
  optional_pip_install_if_missing flex_gemm "${RECONVIAGEN_FLEX_GEMM_URL}" --no-build-isolation --no-deps
  if [ -n "${RECONVIAGEN_O_VOXEL_URL}" ]; then
    optional_pip_install_if_missing o_voxel "${RECONVIAGEN_O_VOXEL_URL}" --no-build-isolation --no-deps
  elif [ -d "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" ]; then
    optional_pip_install_if_missing o_voxel "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation --no-deps
  fi
  patch_flex_gemm_triton_autotuner
  verify_flex_gemm_triton_autotuner_patch
}

patch_flex_gemm_triton_autotuner() {
  mm_run "${RECONVIAGEN_ENV_DIR}" python - <<'PY'
from pathlib import Path
import sysconfig

path = Path(sysconfig.get_paths()["purelib"]) / "flex_gemm" / "utils" / "autotuner.py"
if not path.exists():
    raise SystemExit(0)

text = path.read_text()
if "self.keys = key" in text:
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
        self.keys = key
"""
if old in text:
    path.write_text(text.replace(old, new))
    raise SystemExit(0)

patched_super = """        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            **autotuner_kwargs,
        )
"""
if patched_super in text:
    path.write_text(text.replace(patched_super, patched_super + "        self.keys = key\n"))
    raise SystemExit(0)

if old not in text:
    raise SystemExit("FlexGEMM autotuner layout changed; cannot patch Triton compatibility.")
PY
}

verify_flex_gemm_triton_autotuner_patch() {
  [ "${RECONVIAGEN_INSTALL_POSTPROCESSORS}" = "1" ] || return 0
  mm_run "${RECONVIAGEN_ENV_DIR}" python - <<'PY'
from pathlib import Path
import sysconfig

path = Path(sysconfig.get_paths()["purelib"]) / "flex_gemm" / "utils" / "autotuner.py"
if not path.exists():
    raise SystemExit(0)
text = path.read_text()
if "self.keys = key" not in text:
    raise SystemExit("[env-reconviagen] FlexGEMM Triton autotuner patch was not applied.")
print(f"[env-reconviagen] FlexGEMM Triton autotuner patch verified: {path}")
PY
}

prefetch_models() {
  is_enabled "${RECONVIAGEN_PREFETCH_MODELS}" || {
    log "Skipping model prefetch (RECONVIAGEN_PREFETCH_MODELS=0)."
    return 0
  }
  log "Prefetching ReconViaGen model artifacts."
  mm_run "${RECONVIAGEN_ENV_DIR}" env \
    RECONVIAGEN_SS_MODEL="${RECONVIAGEN_SS_MODEL:-Stable-X/trellis-vggt-v0-2}" \
    RECONVIAGEN_TRELLIS_MODEL="${RECONVIAGEN_TRELLIS_MODEL:-microsoft/TRELLIS.2-4B}" \
    RECONVIAGEN_VGGT_MODEL="${RECONVIAGEN_VGGT_MODEL:-Stable-X/vggt-object-v0-1}" \
    RECONVIAGEN_BIREFNET_MODEL="${RECONVIAGEN_BIREFNET_MODEL:-ZhengPeng7/BiRefNet}" \
    RECONVIAGEN_PREFETCH_DINOV2="${RECONVIAGEN_PREFETCH_DINOV2:-1}" \
    RECONVIAGEN_DINOV2_MODEL="${RECONVIAGEN_DINOV2_MODEL:-dinov2_vitl14_reg}" \
    python "${SCRIPT_DIR}/prefetch_models.py"
}

# --- main --------------------------------------------------------------------
sync_repo

if ! env_should_rebuild "${RECONVIAGEN_ENV_DIR}" "${stamp_file}" "${expected_stamp}"; then
  log "Reusing cached reconviagen env at ${RECONVIAGEN_ENV_DIR} (spec ${expected_stamp})."
  install_postprocessors
  exit 0
fi

log "Creating fresh reconviagen env (python ${RECONVIAGEN_PYTHON_VERSION}) at ${RECONVIAGEN_ENV_DIR}."
# libstdcxx-ng/libgcc-ng provide a modern GLIBCXX in-env for the CuMesh/o-voxel
# wheels, replacing the legacy apt/ppa hack. The env lib dir is put on the
# loader path here and again when the worker starts (see run.sh).
create_env "${RECONVIAGEN_ENV_DIR}" "${RECONVIAGEN_PYTHON_VERSION}" pip libstdcxx-ng libgcc-ng
export LD_LIBRARY_PATH="${RECONVIAGEN_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

if is_enabled "${RECONVIAGEN_USE_SYSTEM_TORCH:-0}"; then
  log "RECONVIAGEN_USE_SYSTEM_TORCH is ignored under micromamba; installing pinned torch into the isolated env."
fi

mm_pip "${RECONVIAGEN_ENV_DIR}" install --upgrade pip
log "Installing ReconViaGen base packages from ${requirements_file}."
mm_pip "${RECONVIAGEN_ENV_DIR}" install -r "${requirements_file}"

log "Running ReconViaGen setup.sh (${RECONVIAGEN_CUDA_FLAGS}) in ${RECONVIAGEN_REPO_DIR}."
mm_run "${RECONVIAGEN_ENV_DIR}" bash -lc "cd '${RECONVIAGEN_REPO_DIR}' && . ./setup.sh ${RECONVIAGEN_CUDA_FLAGS}"

install_postprocessors
prefetch_models

write_stamp "${stamp_file}" "${expected_stamp}"
log "reconviagen env ready at ${RECONVIAGEN_ENV_DIR} (repo ${RECONVIAGEN_REPO_REF}@$(repo_revision))."
