should_manage_reconviagen() {
  if is_enabled "${RECONVIAGEN_MOCK:-0}"; then
    return 1
  fi
  if [ -n "${RECONVIAGEN_COMMAND:-}" ] || [ -n "${RECONVIAGEN_WORKER_URL:-}" ]; then
    return 1
  fi
  return 0
}

prepare_reconviagen_worker() {
  if ! should_manage_reconviagen || ! is_enabled "${APP_PREPARE_RECONVIAGEN}"; then
    return 0
  fi
  "${SERVER_DIR}/workers/reconviagen/scripts/prepare.sh"
}

start_reconviagen_worker() {
  if ! should_manage_reconviagen || ! is_enabled "${APP_START_RECONVIAGEN}"; then
    return 0
  fi
  if ! env_exists "${RECONVIAGEN_ENV_NAME}"; then
    LOG_PREFIX="run.sh" log "ReconViaGen env ${RECONVIAGEN_ENV_NAME} is missing; API will start without a local worker."
    return 0
  fi

  export RECONVIAGEN_WORKER_URL="http://${RECONVIAGEN_WORKER_HOST}:${RECONVIAGEN_WORKER_PORT}"
  export RECONVIAGEN_REPO_DIR
  mkdir -p "$(dirname "${RECONVIAGEN_WORKER_LOG}")"
  LOG_PREFIX="run.sh" log "Starting ReconViaGen worker on ${RECONVIAGEN_WORKER_URL}; logging to ${RECONVIAGEN_WORKER_LOG}."
  (
    cd "${SERVER_DIR}"
    exec micromamba run -n "${RECONVIAGEN_ENV_NAME}" env \
      PYTHONPATH="${SERVER_DIR}:${PYTHONPATH:-}" \
      python -m workers.reconviagen.main \
      --host "${RECONVIAGEN_WORKER_HOST}" \
      --port "${RECONVIAGEN_WORKER_PORT}"
  ) > >(sed -u 's/^/[worker-reconviagen] /' | tee -a "${RECONVIAGEN_WORKER_LOG}") 2>&1 &
  RECONVIAGEN_WORKER_PID="$!"
}

stop_reconviagen_worker() {
  if [ -n "${RECONVIAGEN_WORKER_PID:-}" ]; then
    kill "${RECONVIAGEN_WORKER_PID}" >/dev/null 2>&1 || true
    wait "${RECONVIAGEN_WORKER_PID}" >/dev/null 2>&1 || true
  fi
}
