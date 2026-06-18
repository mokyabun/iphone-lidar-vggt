#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/lib.sh"

export APP_LOCAL_ROOT="${APP_LOCAL_ROOT:-/opt/iphone-lidar-vggt}"
export APP_BIN_DIR="${APP_BIN_DIR:-${APP_LOCAL_ROOT}/bin}"
export APP_PYTHON_VERSION="${APP_PYTHON_VERSION:-3.11}"
export API_ENV_DIR="${API_ENV_DIR:-${APP_LOCAL_ROOT}/envs/api}"

requirements_file="${SERVER_DIR}/api/requirements.txt"
stamp_file="${API_ENV_DIR}/.spec-stamp"
expected_stamp="$(
  {
    printf 'py=%s\n' "${APP_PYTHON_VERSION}"
    cat "${requirements_file}"
  } | cksum | awk '{print $1}'
)"

if ! env_should_rebuild "${API_ENV_DIR}" "${stamp_file}" "${expected_stamp}"; then
  LOG_PREFIX="env-api" log "Reusing cached api env at ${API_ENV_DIR} (spec ${expected_stamp})."
  exit 0
fi

LOG_PREFIX="env-api" log "Creating fresh api env (python ${APP_PYTHON_VERSION}) at ${API_ENV_DIR}."
create_env "${API_ENV_DIR}" "${APP_PYTHON_VERSION}" pip
mm_pip "${API_ENV_DIR}" install --upgrade pip
mm_pip "${API_ENV_DIR}" install -r "${requirements_file}"
write_stamp "${stamp_file}" "${expected_stamp}"
LOG_PREFIX="env-api" log "api env ready at ${API_ENV_DIR}."
