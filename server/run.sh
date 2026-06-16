#!/usr/bin/env bash
set -Eeuo pipefail

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
ENV_NAME="${APP_ENV_NAME:-lidar-reconviagen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"

APP_PREPARE_RECONVIAGEN="${APP_PREPARE_RECONVIAGEN:-1}"
APP_START_RECONVIAGEN="${APP_START_RECONVIAGEN:-1}"
RECONVIAGEN_ENV_NAME="${RECONVIAGEN_ENV_NAME:-reconviagen-v05}"
RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
RECONVIAGEN_WORKER_HOST="${RECONVIAGEN_WORKER_HOST:-127.0.0.1}"
RECONVIAGEN_WORKER_PORT="${RECONVIAGEN_WORKER_PORT:-8011}"
RECONVIAGEN_WORKER_LOG="${RECONVIAGEN_WORKER_LOG:-${APP_CACHE_ROOT}/reconviagen-worker.log}"
RECONVIAGEN_WORKER_PID=""

log() {
  printf '[run.sh] %s\n' "$*"
}

is_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    return 0
  fi
  log "Installing micromamba."
  mkdir -p /usr/local/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
}

env_exists() {
  micromamba env list | awk '{print $1}' | grep -qx "$1"
}

ensure_app_env() {
  if ! env_exists "${ENV_NAME}"; then
    log "Creating micromamba env ${ENV_NAME}."
    micromamba create -y -f "${SCRIPT_DIR}/environment.yml"
  else
    log "Updating micromamba env ${ENV_NAME}."
    micromamba env update -y -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
  fi
}

should_manage_reconviagen() {
  if is_enabled "${RECONVIAGEN_MOCK:-0}"; then
    return 1
  fi
  if [ -n "${RECONVIAGEN_COMMAND:-}" ] || [ -n "${RECONVIAGEN_WORKER_URL:-}" ]; then
    return 1
  fi
  return 0
}

prepare_reconviagen() {
  if ! should_manage_reconviagen || ! is_enabled "${APP_PREPARE_RECONVIAGEN}"; then
    return 0
  fi
  "${SCRIPT_DIR}/scripts/prepare_reconviagen.sh"
}

start_reconviagen_worker() {
  if ! should_manage_reconviagen || ! is_enabled "${APP_START_RECONVIAGEN}"; then
    return 0
  fi
  if ! env_exists "${RECONVIAGEN_ENV_NAME}"; then
    log "ReconViaGen env ${RECONVIAGEN_ENV_NAME} is missing; API will start without a local worker."
    return 0
  fi

  export RECONVIAGEN_WORKER_URL="http://${RECONVIAGEN_WORKER_HOST}:${RECONVIAGEN_WORKER_PORT}"
  export RECONVIAGEN_REPO_DIR
  mkdir -p "$(dirname "${RECONVIAGEN_WORKER_LOG}")"
  log "Starting ReconViaGen worker on ${RECONVIAGEN_WORKER_URL}."
  (
    cd "${SCRIPT_DIR}"
    exec micromamba run -n "${RECONVIAGEN_ENV_NAME}" python -m reconviagen_worker.main \
      --host "${RECONVIAGEN_WORKER_HOST}" \
      --port "${RECONVIAGEN_WORKER_PORT}"
  ) >>"${RECONVIAGEN_WORKER_LOG}" 2>&1 &
  RECONVIAGEN_WORKER_PID="$!"
}

stop_reconviagen_worker() {
  if [ -n "${RECONVIAGEN_WORKER_PID}" ]; then
    kill "${RECONVIAGEN_WORKER_PID}" >/dev/null 2>&1 || true
    wait "${RECONVIAGEN_WORKER_PID}" >/dev/null 2>&1 || true
  fi
}

ensure_micromamba
ensure_app_env
prepare_reconviagen
start_reconviagen_worker
trap stop_reconviagen_worker EXIT INT TERM

log "Starting API on ${APP_HOST}:${APP_PORT}."
cd "${SCRIPT_DIR}"
micromamba run -n "${ENV_NAME}" python -m uvicorn lidar_reconviagen.api:app --host "${APP_HOST}" --port "${APP_PORT}"
