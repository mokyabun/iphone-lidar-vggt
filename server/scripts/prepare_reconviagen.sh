#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
RECONVIAGEN_REPO_URL="${RECONVIAGEN_REPO_URL:-https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git}"
RECONVIAGEN_REPO_REF="${RECONVIAGEN_REPO_REF:-v0.5}"
RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
RECONVIAGEN_ENV_NAME="${RECONVIAGEN_ENV_NAME:-reconviagen-v05}"
RECONVIAGEN_SETUP_FLAGS="${RECONVIAGEN_SETUP_FLAGS:---xformers --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrec --spconv --kaolin --nvdiffrast}"
RECONVIAGEN_CUMESH_URL="${RECONVIAGEN_CUMESH_URL:-git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f}"
RECONVIAGEN_FLEX_GEMM_URL="${RECONVIAGEN_FLEX_GEMM_URL:-git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e}"
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

setup_stamp() {
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python - <<'PY'
from pathlib import Path
import sys
print(Path(sys.prefix) / ".reconviagen-setup-stamp")
PY
}

runtime_stamp() {
  {
    printf 'repo=%s\n' "$(repo_revision)"
    printf 'flags=%s\n' "${RECONVIAGEN_SETUP_FLAGS}"
    cksum "${SERVER_DIR}/reconviagen-environment.yml"
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

pip_install_if_missing() {
  local import_name="$1"
  shift
  if micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -c "import ${import_name}" >/dev/null 2>&1; then
    return 0
  fi
  log "Installing ${import_name}."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m pip install "$@"
}

run_reconviagen_setup() {
  local stamp_file expected_stamp installed_stamp
  stamp_file="$(setup_stamp)"
  expected_stamp="$(runtime_stamp)"
  installed_stamp=""
  if [ -f "${stamp_file}" ]; then
    installed_stamp="$(cat "${stamp_file}")"
  fi
  if [ "${RECONVIAGEN_REFRESH}" != "1" ] && [ "${installed_stamp}" = "${expected_stamp}" ]; then
    log "ReconViaGen setup is already current."
    return 0
  fi

  log "Running ReconViaGen setup.sh. This can take a while on a fresh pod."
  rm -rf /tmp/extensions
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" bash -lc \
    "cd '${RECONVIAGEN_REPO_DIR}' && . ./setup.sh ${RECONVIAGEN_SETUP_FLAGS}"

  pip_install_if_missing cumesh "${RECONVIAGEN_CUMESH_URL}" --no-build-isolation --no-deps
  pip_install_if_missing flex_gemm "${RECONVIAGEN_FLEX_GEMM_URL}" --no-build-isolation --no-deps
  if [ -d "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" ]; then
    pip_install_if_missing o_voxel "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation --no-deps
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
run_reconviagen_setup
log "ReconViaGen is ready in micromamba env ${RECONVIAGEN_ENV_NAME}."

