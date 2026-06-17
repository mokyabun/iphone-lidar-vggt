ensure_reconviagen_env() {
  local requirements_file="${SERVER_DIR}/workers/reconviagen/requirements.txt"
  if ! env_exists "${RECONVIAGEN_ENV_DIR}"; then
    LOG_PREFIX="worker-reconviagen-env" log "Creating uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
    uv venv --python "${RECONVIAGEN_PYTHON_VERSION}" "${RECONVIAGEN_ENV_DIR}"
    uv pip install --python "$(venv_python "${RECONVIAGEN_ENV_DIR}")" -r "${requirements_file}"
    configure_ccache
    return 0
  fi

  if should_update_envs; then
    LOG_PREFIX="worker-reconviagen-env" log "Updating uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
    uv pip install --python "$(venv_python "${RECONVIAGEN_ENV_DIR}")" -r "${requirements_file}"
  else
    LOG_PREFIX="worker-reconviagen-env" log "Using existing uv env ${RECONVIAGEN_ENV_NAME}. Set APP_UPDATE_ENVS=1 to update dependencies."
  fi
  configure_ccache
}

verify_reconviagen_torch() {
  LOG_PREFIX="prepare-reconviagen" log "Checking PyTorch in ${RECONVIAGEN_ENV_NAME}."
  venv_run "${RECONVIAGEN_ENV_DIR}" python - <<'PY'
import torch

print(f"[prepare-reconviagen] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.__version__.startswith("2.4."):
    raise SystemExit("[prepare-reconviagen] ReconViaGen v0.5 CUDA extensions require PyTorch 2.4.x.")
PY
}
