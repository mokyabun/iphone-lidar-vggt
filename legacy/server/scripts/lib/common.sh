is_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  local prefix="${LOG_PREFIX:-server}"
  printf '[%s] %s\n' "${prefix}" "$*"
}

env_exists() {
  local env_dir="$1"
  [ -x "${env_dir}/bin/python" ]
}

venv_python() {
  local env_dir="$1"
  printf '%s/bin/python\n' "${env_dir}"
}

venv_run() {
  local env_dir="$1"
  shift
  env \
    VIRTUAL_ENV="${env_dir}" \
    PATH="${env_dir}/bin:${PATH}" \
    "$@"
}

should_update_envs() {
  is_enabled "${APP_UPDATE_ENVS:-0}"
}
