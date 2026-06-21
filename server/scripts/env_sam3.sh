#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/lib.sh"

LOG_PREFIX="env-sam3"

export APP_LOCAL_ROOT="${APP_LOCAL_ROOT:-/opt/iphone-lidar-vggt}"
export APP_BIN_DIR="${APP_BIN_DIR:-${APP_LOCAL_ROOT}/bin}"
export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"

export SAM3_PYTHON_VERSION="${SAM3_PYTHON_VERSION:-3.12}"
export SAM3_ENV_DIR="${SAM3_ENV_DIR:-${APP_LOCAL_ROOT}/envs/sam3}"
export SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"
export SAM3_REPO_REF="${SAM3_REPO_REF:-main}"
export SAM3_REPO_DIR="${SAM3_REPO_DIR:-${APP_CACHE_ROOT}/sam3}"
export SAM3_TORCH_INDEX_URL="${SAM3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export SAM3_TORCH_SPEC="${SAM3_TORCH_SPEC:-torch==2.10.0 torchvision}"
export SAM3_EXTRAS="${SAM3_EXTRAS:-train,dev}"
export SAM3_INSTALL_LIGHT_INFERENCE_DEPS="${SAM3_INSTALL_LIGHT_INFERENCE_DEPS:-1}"
export SAM3_INSTALL_FAST_DEPS="${SAM3_INSTALL_FAST_DEPS:-0}"

requirements_file="${SERVER_DIR}/sam3_worker/requirements.txt"

if [ "$(uname -s)" != "Linux" ]; then
  log "Skipping SAM3 CUDA env build on $(uname -s); only the API runs off-Linux."
  exit 0
fi

sync_repo() {
  mkdir -p "$(dirname "${SAM3_REPO_DIR}")"
  if [ -d "${SAM3_REPO_DIR}/.git" ]; then
    log "Updating SAM3 ${SAM3_REPO_REF}."
    git -C "${SAM3_REPO_DIR}" fetch --depth 1 origin "${SAM3_REPO_REF}"
    git -C "${SAM3_REPO_DIR}" reset --hard FETCH_HEAD
  else
    log "Cloning SAM3 ${SAM3_REPO_REF}."
    rm -rf "${SAM3_REPO_DIR}"
    git clone --depth 1 --branch "${SAM3_REPO_REF}" \
      "${SAM3_REPO_URL}" "${SAM3_REPO_DIR}"
  fi
}

repo_revision() {
  git -C "${SAM3_REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown
}

spec_stamp() {
  {
    printf 'py=%s\n' "${SAM3_PYTHON_VERSION}"
    printf 'repo_ref=%s\n' "${SAM3_REPO_REF}"
    printf 'torch_index=%s\n' "${SAM3_TORCH_INDEX_URL}"
    printf 'torch_spec=%s\n' "${SAM3_TORCH_SPEC}"
    printf 'extras=%s\n' "${SAM3_EXTRAS}"
    printf 'light_deps=%s\n' "${SAM3_INSTALL_LIGHT_INFERENCE_DEPS}"
    printf 'fast_deps=%s\n' "${SAM3_INSTALL_FAST_DEPS}"
    cat "${requirements_file}"
  } | cksum | awk '{print $1}'
}

stamp_file="${SAM3_ENV_DIR}/.spec-stamp"
expected_stamp="$(spec_stamp)"

sync_repo

if ! env_should_rebuild "${SAM3_ENV_DIR}" "${stamp_file}" "${expected_stamp}"; then
  log "Reusing cached sam3 env at ${SAM3_ENV_DIR} (spec ${expected_stamp})."
  exit 0
fi

log "Creating fresh sam3 env (python ${SAM3_PYTHON_VERSION}) at ${SAM3_ENV_DIR}."
create_env "${SAM3_ENV_DIR}" "${SAM3_PYTHON_VERSION}" pip

mm_pip "${SAM3_ENV_DIR}" install --upgrade pip
log "Installing SAM3 worker base packages from ${requirements_file}."
mm_pip "${SAM3_ENV_DIR}" install -r "${requirements_file}"
log "Installing SAM3 torch packages from ${SAM3_TORCH_INDEX_URL}: ${SAM3_TORCH_SPEC}."
mm_pip "${SAM3_ENV_DIR}" install ${SAM3_TORCH_SPEC} --index-url "${SAM3_TORCH_INDEX_URL}"
if [ -n "${SAM3_EXTRAS}" ]; then
  log "Installing SAM3 package from ${SAM3_REPO_DIR} with extras [${SAM3_EXTRAS}]."
  mm_pip "${SAM3_ENV_DIR}" install -e "${SAM3_REPO_DIR}[${SAM3_EXTRAS}]"
else
  log "Installing SAM3 package from ${SAM3_REPO_DIR}."
  mm_pip "${SAM3_ENV_DIR}" install -e "${SAM3_REPO_DIR}"
fi

if is_enabled "${SAM3_INSTALL_LIGHT_INFERENCE_DEPS}"; then
  log "Installing SAM3 lightweight inference dependencies."
  mm_pip "${SAM3_ENV_DIR}" install einops ninja psutil
fi

if is_enabled "${SAM3_INSTALL_FAST_DEPS}"; then
  log "Installing optional SAM3 fast inference dependencies."
  mm_pip "${SAM3_ENV_DIR}" install flash-attn-3 --no-deps --index-url "${SAM3_TORCH_INDEX_URL}"
  mm_pip "${SAM3_ENV_DIR}" install git+https://github.com/ronghanghu/cc_torch.git
fi

write_stamp "${stamp_file}" "${expected_stamp}"
log "sam3 env ready at ${SAM3_ENV_DIR} (repo ${SAM3_REPO_REF}@$(repo_revision))."
