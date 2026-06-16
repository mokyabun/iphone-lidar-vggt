#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
APP_PERSIST_ROOT="${APP_PERSIST_ROOT:-/workspace}"
APP_CACHE_ROOT="${APP_CACHE_ROOT:-${APP_PERSIST_ROOT}/cache}"
APP_STATE_ROOT="${APP_STATE_ROOT:-${APP_CACHE_ROOT}/state}"
APP_ENV_ROOT="${APP_ENV_ROOT:-${APP_CACHE_ROOT}/envs}"
APP_RUNTIME_DIR="${APP_RUNTIME_DIR:-${APP_CACHE_ROOT}/run}"
APP_VENV_REAL_DIR="${APP_VENV_REAL_DIR:-${APP_ENV_ROOT}/iphone-lidar-vggt}"
APP_RUNS_DIR="${APP_RUNS_DIR:-${APP_PERSIST_ROOT}/runs}"
APP_ENV_FILE="${APP_ENV_FILE:-${SCRIPT_DIR}/.env}"
APP_ENV_LOCAL_FILE="${APP_ENV_LOCAL_FILE:-${SCRIPT_DIR}/.env.local}"
APP_MANAGER_PID_FILE="${APP_MANAGER_PID_FILE:-${APP_RUNTIME_DIR}/run-sh.pid}"
APP_UVICORN_PID_FILE="${APP_UVICORN_PID_FILE:-${APP_RUNTIME_DIR}/uvicorn.pid}"
APP_RECONVIAGEN_WORKER_PID_FILE="${APP_RECONVIAGEN_WORKER_PID_FILE:-${APP_RUNTIME_DIR}/reconviagen-worker.pid}"
APP_LOG_FILE="${APP_LOG_FILE:-${APP_RUNTIME_DIR}/uvicorn.log}"
RECONVIAGEN_WORKER_LOG="${RECONVIAGEN_WORKER_LOG:-${APP_CACHE_ROOT}/reconviagen-worker.log}"
export APP_DIR

log() {
  printf '[manage.sh] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

usage() {
  cat <<'USAGE'
Usage:
  server/manage.sh status
  server/manage.sh list
  server/manage.sh set KEY VALUE
  server/manage.sh unset KEY
  server/manage.sh restart
  server/manage.sh start
  server/manage.sh stop
  server/manage.sh logs [app|worker]

Notes:
  - set/unset writes server/.env.local so APP_UPDATE_MODE=reset will not erase live RunPod tweaks.
  - server/.env is the committed, non-secret baseline.
  - secrets such as HF_TOKEN must be set in RunPod environment variables, not env files.
USAGE
}

is_secret_env_name() {
  case "$1" in
    HF_TOKEN | *_TOKEN | *_SECRET | *_PASSWORD | *_API_KEY | AWS_ACCESS_KEY_ID | AWS_SECRET_ACCESS_KEY)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_key() {
  [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid env key: $1"
  if is_secret_env_name "$1"; then
    die "$1 looks like a secret. Set it in the RunPod template or Pod environment variables."
  fi
}

quote_value() {
  local value="$1"
  if [[ "${value}" =~ ^[A-Za-z0-9_./:@,+-]+$ ]]; then
    printf '%s' "${value}"
  else
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '"%s"' "${value}"
  fi
}

is_alive() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1
}

pid_from_file() {
  local file="$1"
  [ -f "${file}" ] && cat "${file}" 2>/dev/null || true
}

managed_pid() {
  pid_from_file "${APP_MANAGER_PID_FILE}"
}

reload_managed() {
  local pid
  pid="$(managed_pid)"
  if is_alive "${pid}"; then
    kill -HUP "${pid}"
    log "Reload signal sent to run.sh pid ${pid}."
    return 0
  fi
  return 1
}

write_env_value() {
  local key="$1"
  local value="$2"
  local quoted tmp

  validate_key "${key}"
  mkdir -p "$(dirname "${APP_ENV_LOCAL_FILE}")"
  touch "${APP_ENV_LOCAL_FILE}"
  quoted="$(quote_value "${value}")"
  tmp="$(mktemp)"
  awk -v key="${key}" -v line="${key}=${quoted}" '
    BEGIN { found = 0 }
    $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=" {
      print line
      found = 1
      next
    }
    { print }
    END {
      if (!found) {
        print line
      }
    }
  ' "${APP_ENV_LOCAL_FILE}" > "${tmp}"
  mv "${tmp}" "${APP_ENV_LOCAL_FILE}"
  log "Set ${key} in ${APP_ENV_LOCAL_FILE}."
}

unset_env_value() {
  local key="$1"
  local tmp

  validate_key "${key}"
  [ -f "${APP_ENV_LOCAL_FILE}" ] || return 0
  tmp="$(mktemp)"
  awk -v key="${key}" '
    $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=" { next }
    { print }
  ' "${APP_ENV_LOCAL_FILE}" > "${tmp}"
  mv "${tmp}" "${APP_ENV_LOCAL_FILE}"
  log "Removed ${key} from ${APP_ENV_LOCAL_FILE}."
}

show_status() {
  local manager app worker
  manager="$(managed_pid)"
  app="$(pid_from_file "${APP_UVICORN_PID_FILE}")"
  worker="$(pid_from_file "${APP_RECONVIAGEN_WORKER_PID_FILE}")"

  printf 'env base:   %s\n' "${APP_ENV_FILE}"
  printf 'env local:  %s\n' "${APP_ENV_LOCAL_FILE}"
  printf 'cache:      %s\n' "${APP_CACHE_ROOT}"
  printf 'state:      %s\n' "${APP_STATE_ROOT}"
  printf 'venv real:  %s\n' "${APP_VENV_REAL_DIR}"
  printf 'runs real:  %s\n' "${APP_RUNS_DIR}"
  printf 'runtime:    %s\n' "${APP_RUNTIME_DIR}"
  printf 'run.sh:     %s\n' "$(is_alive "${manager}" && printf 'running pid %s' "${manager}" || printf 'not managed')"
  printf 'uvicorn:    %s\n' "$(is_alive "${app}" && printf 'running pid %s' "${app}" || printf 'not running')"
  printf 'worker:     %s\n' "$(is_alive "${worker}" && printf 'running supervisor pid %s' "${worker}" || printf 'not running')"
}

list_env_files() {
  if [ -f "${APP_ENV_FILE}" ]; then
    printf '# %s\n' "${APP_ENV_FILE}"
    sed -n '/^[[:space:]]*#/d;/^[[:space:]]*$/d;p' "${APP_ENV_FILE}"
  fi
  if [ -f "${APP_ENV_LOCAL_FILE}" ]; then
    printf '\n# %s\n' "${APP_ENV_LOCAL_FILE}"
    sed -n '/^[[:space:]]*#/d;/^[[:space:]]*$/d;p' "${APP_ENV_LOCAL_FILE}"
  fi
}

start_managed() {
  local pid
  pid="$(managed_pid)"
  if is_alive "${pid}"; then
    log "run.sh is already running as pid ${pid}."
    return 0
  fi
  mkdir -p "${APP_RUNTIME_DIR}"
  nohup "${SCRIPT_DIR}/run.sh" >> "${APP_RUNTIME_DIR}/run-sh.log" 2>&1 &
  log "Started managed run.sh as pid $!. Logs: ${APP_RUNTIME_DIR}/run-sh.log"
}

stop_managed() {
  local pid
  pid="$(managed_pid)"
  if is_alive "${pid}"; then
    kill -TERM "${pid}"
    log "Stop signal sent to run.sh pid ${pid}."
  else
    log "No managed run.sh pid found."
  fi
}

tail_logs() {
  local target="${1:-app}"
  case "${target}" in
    app)
      tail -n 120 -f "${APP_LOG_FILE}"
      ;;
    worker)
      tail -n 120 -f "${RECONVIAGEN_WORKER_LOG}"
      ;;
    *)
      die "Unknown log target: ${target}; use app or worker."
      ;;
  esac
}

main() {
  local command="${1:-}"
  case "${command}" in
    status)
      show_status
      ;;
    list)
      list_env_files
      ;;
    set)
      [ "$#" -eq 3 ] || die "Usage: server/manage.sh set KEY VALUE"
      write_env_value "$2" "$3"
      reload_managed || log "run.sh is not managed right now; run server/manage.sh start or restart manually."
      ;;
    unset)
      [ "$#" -eq 2 ] || die "Usage: server/manage.sh unset KEY"
      unset_env_value "$2"
      reload_managed || log "run.sh is not managed right now; run server/manage.sh start or restart manually."
      ;;
    restart)
      reload_managed || start_managed
      ;;
    start)
      start_managed
      ;;
    stop)
      stop_managed
      ;;
    logs)
      tail_logs "${2:-app}"
      ;;
    "" | -h | --help | help)
      usage
      ;;
    *)
      usage
      die "Unknown command: ${command}"
      ;;
  esac
}

main "$@"
