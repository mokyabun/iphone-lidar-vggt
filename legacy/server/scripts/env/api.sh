ensure_api_env() {
  local requirements_file="${SERVER_DIR}/api/requirements.txt"
  if ! env_exists "${APP_ENV_DIR}"; then
    LOG_PREFIX="api-env" log "Creating uv env ${APP_ENV_NAME} at ${APP_ENV_DIR}."
    mkdir -p "$(dirname "${APP_ENV_DIR}")"
    uv venv --python "${APP_PYTHON_VERSION}" "${APP_ENV_DIR}"
    uv pip install --python "$(venv_python "${APP_ENV_DIR}")" -r "${requirements_file}"
    return 0
  fi

  if should_update_envs; then
    LOG_PREFIX="api-env" log "Updating uv env ${APP_ENV_NAME} at ${APP_ENV_DIR}."
    uv pip install --python "$(venv_python "${APP_ENV_DIR}")" -r "${requirements_file}"
  else
    LOG_PREFIX="api-env" log "Using existing uv env ${APP_ENV_NAME}. Set APP_UPDATE_ENVS=1 to update dependencies."
  fi
}
