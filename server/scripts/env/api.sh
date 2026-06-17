ensure_api_env() {
  local env_file="${SERVER_DIR}/api/environment.yml"
  if ! env_exists "${APP_ENV_NAME}"; then
    LOG_PREFIX="api-env" log "Creating micromamba env ${APP_ENV_NAME}."
    micromamba create -y -n "${APP_ENV_NAME}" -f "${env_file}"
    return 0
  fi

  if should_update_envs; then
    LOG_PREFIX="api-env" log "Updating micromamba env ${APP_ENV_NAME}."
    micromamba env update -y -n "${APP_ENV_NAME}" -f "${env_file}"
  else
    LOG_PREFIX="api-env" log "Using existing micromamba env ${APP_ENV_NAME}. Set APP_UPDATE_ENVS=1 to update dependencies."
  fi
}
