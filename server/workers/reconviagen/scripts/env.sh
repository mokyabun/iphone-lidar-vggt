ensure_reconviagen_env() {
  local env_file="${SERVER_DIR}/workers/reconviagen/environment.yml"
  if ! env_exists "${RECONVIAGEN_ENV_NAME}"; then
    LOG_PREFIX="worker-reconviagen-env" log "Creating micromamba env ${RECONVIAGEN_ENV_NAME}."
    micromamba create -y -n "${RECONVIAGEN_ENV_NAME}" -f "${env_file}"
    configure_ccache_for_env "${RECONVIAGEN_ENV_NAME}"
    return 0
  fi

  if should_update_envs; then
    LOG_PREFIX="worker-reconviagen-env" log "Updating micromamba env ${RECONVIAGEN_ENV_NAME}."
    micromamba env update -y -n "${RECONVIAGEN_ENV_NAME}" -f "${env_file}"
  else
    LOG_PREFIX="worker-reconviagen-env" log "Using existing micromamba env ${RECONVIAGEN_ENV_NAME}. Set APP_UPDATE_ENVS=1 to update dependencies."
  fi
  configure_ccache_for_env "${RECONVIAGEN_ENV_NAME}"
}

verify_reconviagen_torch() {
  LOG_PREFIX="prepare-reconviagen" log "Checking PyTorch in ${RECONVIAGEN_ENV_NAME}."
  micromamba run -n "${RECONVIAGEN_ENV_NAME}" python - <<'PY'
import torch

print(f"[prepare-reconviagen] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.__version__.startswith("2.4."):
    raise SystemExit("[prepare-reconviagen] ReconViaGen v0.5 CUDA extensions require PyTorch 2.4.x.")
PY
}
