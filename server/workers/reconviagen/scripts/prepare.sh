#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
APP_DIR="$(cd "${SERVER_DIR}/.." && pwd)"

export SERVER_DIR
export APP_DIR
export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export APP_UPDATE_ENVS="${APP_UPDATE_ENVS:-0}"
export RECONVIAGEN_REPO_URL="${RECONVIAGEN_REPO_URL:-https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git}"
export RECONVIAGEN_REPO_REF="${RECONVIAGEN_REPO_REF:-v0.5}"
export RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
export RECONVIAGEN_ENV_NAME="${RECONVIAGEN_ENV_NAME:-worker-reconviagen}"
export RECONVIAGEN_CUDA_FLAGS="${RECONVIAGEN_CUDA_FLAGS:---xformers --flash-attn --nvdiffrec --spconv --kaolin --nvdiffrast}"
export RECONVIAGEN_CUMESH_URL="${RECONVIAGEN_CUMESH_URL:-git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f}"
export RECONVIAGEN_FLEX_GEMM_URL="${RECONVIAGEN_FLEX_GEMM_URL:-git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e}"
export RECONVIAGEN_INSTALL_POSTPROCESSORS="${RECONVIAGEN_INSTALL_POSTPROCESSORS:-1}"
export RECONVIAGEN_REFRESH="${RECONVIAGEN_REFRESH:-0}"

. "${SERVER_DIR}/scripts/lib/common.sh"
. "${SERVER_DIR}/scripts/env/build_cache.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/repo.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/env.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/packages.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/extensions.sh"

if ! command -v micromamba >/dev/null 2>&1; then
  LOG_PREFIX="prepare-reconviagen" log "micromamba is required before preparing ReconViaGen."
  exit 2
fi

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
build_reconviagen_cuda_extensions
LOG_PREFIX="prepare-reconviagen" log "ReconViaGen is ready in micromamba env ${RECONVIAGEN_ENV_NAME}."
