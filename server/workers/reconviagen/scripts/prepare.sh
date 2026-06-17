#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
APP_DIR="$(cd "${SERVER_DIR}/.." && pwd)"

export SERVER_DIR
export APP_DIR
export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
export APP_BIN_DIR="${APP_BIN_DIR:-${APP_CACHE_ROOT}/bin}"
export PATH="${APP_BIN_DIR}:${PATH}"
export APP_UPDATE_ENVS="${APP_UPDATE_ENVS:-0}"
export RECONVIAGEN_REPO_URL="${RECONVIAGEN_REPO_URL:-https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git}"
export RECONVIAGEN_REPO_REF="${RECONVIAGEN_REPO_REF:-v0.5}"
export RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
export RECONVIAGEN_ENV_NAME="${RECONVIAGEN_ENV_NAME:-worker-reconviagen}"
export RECONVIAGEN_PYTHON_VERSION="${RECONVIAGEN_PYTHON_VERSION:-3.10}"
export RECONVIAGEN_ENV_DIR="${RECONVIAGEN_ENV_DIR:-${APP_CACHE_ROOT}/venvs/${RECONVIAGEN_ENV_NAME}}"
export RECONVIAGEN_CUDA_FLAGS="${RECONVIAGEN_CUDA_FLAGS:---xformers --flash-attn --spconv --kaolin --nvdiffrast}"
export RECONVIAGEN_TRELLIS2_WHEEL_REF="${RECONVIAGEN_TRELLIS2_WHEEL_REF:-90d6619f8152991009e68a6bdf6217a8cb7d8bb3}"
export RECONVIAGEN_TRELLIS2_WHEEL_BASE="${RECONVIAGEN_TRELLIS2_WHEEL_BASE:-https://huggingface.co/spaces/microsoft/TRELLIS.2/resolve/${RECONVIAGEN_TRELLIS2_WHEEL_REF}/wheels}"
export RECONVIAGEN_CUMESH_URL="${RECONVIAGEN_CUMESH_URL:-${RECONVIAGEN_TRELLIS2_WHEEL_BASE}/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl}"
export RECONVIAGEN_FLEX_GEMM_URL="${RECONVIAGEN_FLEX_GEMM_URL:-${RECONVIAGEN_TRELLIS2_WHEEL_BASE}/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl}"
export RECONVIAGEN_O_VOXEL_URL="${RECONVIAGEN_O_VOXEL_URL:-${RECONVIAGEN_TRELLIS2_WHEEL_BASE}/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl}"
export RECONVIAGEN_INSTALL_POSTPROCESSORS="${RECONVIAGEN_INSTALL_POSTPROCESSORS:-1}"
export RECONVIAGEN_FIX_LIBSTDCXX="${RECONVIAGEN_FIX_LIBSTDCXX:-1}"
export RECONVIAGEN_REFRESH="${RECONVIAGEN_REFRESH:-0}"

. "${SERVER_DIR}/scripts/lib/common.sh"
. "${SERVER_DIR}/scripts/env/build_cache.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/repo.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/env.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/packages.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/extensions.sh"

"${SERVER_DIR}/scripts/install/uv.sh"

if [ "$(uname -s)" != "Linux" ]; then
  LOG_PREFIX="prepare-reconviagen" log "Skipping ReconViaGen CUDA setup on $(uname -s)."
  exit 0
fi

configure_build_cache
sync_reconviagen_repo
ensure_reconviagen_env
verify_reconviagen_torch
install_reconviagen_base_packages
verify_reconviagen_base_package_pins
ensure_reconviagen_libstdcxx
build_reconviagen_cuda_extensions
LOG_PREFIX="prepare-reconviagen" log "ReconViaGen is ready in uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
