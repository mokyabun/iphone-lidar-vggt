#!/usr/bin/env bash
set -Eeuo pipefail

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
ENV_NAME="${APP_ENV_NAME:-lidar-reconviagen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "[run.sh] micromamba is required. Install it first, then rerun this script." >&2
  exit 2
fi

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[run.sh] Creating micromamba env ${ENV_NAME}"
  micromamba create -y -f "${SCRIPT_DIR}/environment.yml"
else
  echo "[run.sh] Updating micromamba env ${ENV_NAME}"
  micromamba env update -y -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
fi

echo "[run.sh] Starting API on ${APP_HOST}:${APP_PORT}"
cd "${SCRIPT_DIR}"
exec micromamba run -n "${ENV_NAME}" python -m uvicorn lidar_reconviagen.api:app --host "${APP_HOST}" --port "${APP_PORT}"
