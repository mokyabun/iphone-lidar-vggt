#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SERVER_DIR}/.." && pwd)"

export SERVER_DIR
export APP_DIR
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8000}"
export APP_ENV_NAME="${APP_ENV_NAME:-api}"
export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/micromamba}"
export APP_UPDATE_ENVS="${APP_UPDATE_ENVS:-0}"
export APP_PREPARE_RECONVIAGEN="${APP_PREPARE_RECONVIAGEN:-1}"
export APP_START_RECONVIAGEN="${APP_START_RECONVIAGEN:-1}"
export RECONVIAGEN_ENV_NAME="${RECONVIAGEN_ENV_NAME:-worker-reconviagen}"
export RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-${APP_CACHE_ROOT}/ReconViaGen}"
export RECONVIAGEN_WORKER_HOST="${RECONVIAGEN_WORKER_HOST:-127.0.0.1}"
export RECONVIAGEN_WORKER_PORT="${RECONVIAGEN_WORKER_PORT:-8011}"
export RECONVIAGEN_WORKER_LOG="${RECONVIAGEN_WORKER_LOG:-${APP_CACHE_ROOT}/worker-reconviagen.log}"
RECONVIAGEN_WORKER_PID=""

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

. "${SERVER_DIR}/scripts/lib/common.sh"
. "${SERVER_DIR}/scripts/env/build_cache.sh"
. "${SERVER_DIR}/scripts/env/api.sh"
. "${SERVER_DIR}/workers/reconviagen/scripts/runtime.sh"

"${SERVER_DIR}/scripts/install/micromamba.sh"
configure_build_cache
ensure_api_env
prepare_reconviagen_worker
start_reconviagen_worker
trap stop_reconviagen_worker EXIT INT TERM

LOG_PREFIX="run.sh" log "Starting API on ${APP_HOST}:${APP_PORT}."
cd "${SERVER_DIR}"
micromamba run -n "${APP_ENV_NAME}" env \
  PYTHONPATH="${SERVER_DIR}:${PYTHONPATH:-}" \
  python -m uvicorn api.api:app --host "${APP_HOST}" --port "${APP_PORT}"
