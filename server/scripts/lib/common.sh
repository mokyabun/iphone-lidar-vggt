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
  local env_name="$1"
  micromamba env list | awk '{print $1}' | grep -qx "${env_name}"
}

should_update_envs() {
  is_enabled "${APP_UPDATE_ENVS:-0}"
}
