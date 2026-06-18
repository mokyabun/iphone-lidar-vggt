#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SERVER_DIR}/scripts"
. "${SCRIPT_DIR}/lib.sh"

# --- configuration -----------------------------------------------------------
export SERVER_DIR
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8000}"
export APP_LOCAL_ROOT="${APP_LOCAL_ROOT:-/opt/iphone-lidar-vggt}"
export APP_BIN_DIR="${APP_BIN_DIR:-${APP_LOCAL_ROOT}/bin}"
export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${APP_LOCAL_ROOT}/mamba}"
export PATH="${APP_BIN_DIR}:${PATH}"

export APP_PYTHON_VERSION="${APP_PYTHON_VERSION:-3.11}"
export API_ENV_DIR="${API_ENV_DIR:-${APP_LOCAL_ROOT}/envs/api}"
export RECONVIAGEN_PYTHON_VERSION="${RECONVIAGEN_PYTHON_VERSION:-3.10}"
export RECONVIAGEN_ENV_DIR="${RECONVIAGEN_ENV_DIR:-${APP_LOCAL_ROOT}/envs/reconviagen}"
export RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
export RECONVIAGEN_WORKER_HOST="${RECONVIAGEN_WORKER_HOST:-127.0.0.1}"
export RECONVIAGEN_WORKER_PORT="${RECONVIAGEN_WORKER_PORT:-8011}"
export RECONVIAGEN_WORKER_LOG="${RECONVIAGEN_WORKER_LOG:-${APP_CACHE_ROOT}/worker-reconviagen.log}"

export APP_PREPARE_RECONVIAGEN="${APP_PREPARE_RECONVIAGEN:-1}"
export APP_START_RECONVIAGEN="${APP_START_RECONVIAGEN:-1}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

WORKER_PID=""

configure_cache() {
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${APP_CACHE_ROOT}/xdg}"
  export HF_HOME="${HF_HOME:-${APP_CACHE_ROOT}/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export TORCH_HOME="${TORCH_HOME:-${APP_CACHE_ROOT}/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${APP_CACHE_ROOT}/pip}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${APP_CACHE_ROOT}/torch_extensions}"
  mkdir -p "${XDG_CACHE_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" \
    "${PIP_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
}

# The local worker is only managed when ReconViaGen is not mocked or delegated
# to an external command/URL.
should_manage_worker() {
  if is_enabled "${RECONVIAGEN_MOCK:-0}"; then
    return 1
  fi
  if [ -n "${RECONVIAGEN_COMMAND:-}" ] || [ -n "${RECONVIAGEN_WORKER_URL:-}" ]; then
    return 1
  fi
  return 0
}

start_worker() {
  should_manage_worker || return 0
  is_enabled "${APP_START_RECONVIAGEN}" || return 0
  if ! env_exists "${RECONVIAGEN_ENV_DIR}"; then
    LOG_PREFIX="run" log "reconviagen env missing at ${RECONVIAGEN_ENV_DIR}; API starts without a local worker."
    return 0
  fi

  export RECONVIAGEN_WORKER_URL="http://${RECONVIAGEN_WORKER_HOST}:${RECONVIAGEN_WORKER_PORT}"
  export RECONVIAGEN_REPO_DIR
  mkdir -p "$(dirname "${RECONVIAGEN_WORKER_LOG}")"
  LOG_PREFIX="run" log "Starting ReconViaGen worker on ${RECONVIAGEN_WORKER_URL}; logging to ${RECONVIAGEN_WORKER_LOG}."
  (
    cd "${SERVER_DIR}"
    exec "$(micromamba_bin)" run -p "${RECONVIAGEN_ENV_DIR}" env \
      PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
      PYTHONPATH="${SERVER_DIR}:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="${RECONVIAGEN_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}" \
      RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR}" \
      python -u -m worker.main \
      --host "${RECONVIAGEN_WORKER_HOST}" \
      --port "${RECONVIAGEN_WORKER_PORT}"
  ) > >(sed -u 's/^/[worker-reconviagen] /' | tee -a "${RECONVIAGEN_WORKER_LOG}") 2>&1 &
  WORKER_PID="$!"

  local timeout="${RECONVIAGEN_WORKER_BOOT_TIMEOUT:-180}"
  for _ in $(seq 1 "${timeout}"); do
    if curl -fsS "${RECONVIAGEN_WORKER_URL}/health" >/dev/null 2>&1; then
      LOG_PREFIX="run" log "ReconViaGen worker is healthy."
      return 0
    fi
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
      LOG_PREFIX="run" log "ReconViaGen worker exited during startup; see ${RECONVIAGEN_WORKER_LOG}."
      WORKER_PID=""
      return 0
    fi
    sleep 1
  done
  LOG_PREFIX="run" log "ReconViaGen worker did not report healthy within ${timeout}s; continuing."
}

stop_worker() {
  if [ -n "${WORKER_PID}" ]; then
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    wait "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
}

# --- bring up the stack ------------------------------------------------------
configure_cache
"${SCRIPT_DIR}/bootstrap_micromamba.sh"
"${SCRIPT_DIR}/env_api.sh"

if should_manage_worker && is_enabled "${APP_PREPARE_RECONVIAGEN}"; then
  "${SCRIPT_DIR}/env_reconviagen.sh"
fi

start_worker
trap stop_worker EXIT INT TERM

LOG_PREFIX="run" log "Starting API on ${APP_HOST}:${APP_PORT}."
cd "${SERVER_DIR}"
# No exec: keep this shell alive so the stop_worker trap runs on exit.
"$(micromamba_bin)" run -p "${API_ENV_DIR}" env \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
  PYTHONPATH="${SERVER_DIR}:${PYTHONPATH:-}" \
  python -u -m uvicorn api.api:app --host "${APP_HOST}" --port "${APP_PORT}" --log-level info
