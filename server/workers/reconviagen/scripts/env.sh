ensure_reconviagen_env() {
  local requirements_file="${SERVER_DIR}/workers/reconviagen/requirements.txt"
  local venv_args=()
  if is_enabled "${RECONVIAGEN_USE_SYSTEM_TORCH:-0}"; then
    venv_args+=(--system-site-packages)
  fi

  if ! env_exists "${RECONVIAGEN_ENV_DIR}"; then
    LOG_PREFIX="worker-reconviagen-env" log "Creating uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
    uv venv --python "${RECONVIAGEN_PYTHON_VERSION}" "${venv_args[@]}" "${RECONVIAGEN_ENV_DIR}"
    install_reconviagen_requirements "${requirements_file}"
    configure_ccache
    return 0
  fi

  if should_update_envs; then
    LOG_PREFIX="worker-reconviagen-env" log "Updating uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
    install_reconviagen_requirements "${requirements_file}"
  else
    LOG_PREFIX="worker-reconviagen-env" log "Using existing uv env ${RECONVIAGEN_ENV_NAME}. Set APP_UPDATE_ENVS=1 to update dependencies."
  fi
  configure_ccache
}

install_reconviagen_requirements() {
  local requirements_file="$1"
  local python_bin
  python_bin="$(venv_python "${RECONVIAGEN_ENV_DIR}")"

  if is_enabled "${RECONVIAGEN_USE_SYSTEM_TORCH:-0}"; then
    LOG_PREFIX="worker-reconviagen-env" log "Using image-provided torch packages via system site-packages."
    uv pip install --python "${python_bin}" -r <(
      grep -Ev '^(--extra-index-url[[:space:]]+https://download\.pytorch\.org/whl/|torch==|torchvision==|torchaudio==)' "${requirements_file}"
    )
  else
    uv pip install --python "${python_bin}" -r "${requirements_file}"
  fi
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
