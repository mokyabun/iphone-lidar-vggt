# Shared helpers for the micromamba-based server scripts.
# Source this file; it does not run anything on its own.

is_enabled() {
  case "${1:-}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  local prefix="${LOG_PREFIX:-server}"
  printf '[%s] %s\n' "${prefix}" "$*"
}

# A conda/micromamba env prefix is ready when it has a python interpreter.
env_exists() {
  [ -x "${1}/bin/python" ]
}

env_python() {
  printf '%s/bin/python\n' "${1}"
}

micromamba_bin() {
  printf '%s/micromamba\n' "${APP_BIN_DIR}"
}

# Run a command inside an env prefix without needing shell activation.
mm_run() {
  local prefix="$1"
  shift
  "$(micromamba_bin)" run -p "${prefix}" "$@"
}

# pip inside an env prefix, with the shared cache.
mm_pip() {
  local prefix="$1"
  shift
  mm_run "${prefix}" python -m pip "$@"
}

# Decide whether an env must be (re)built.
# Default policy is "always fresh"; APP_REUSE_ENV=1 reuses a cached env whose
# stamp matches the expected spec hash. APP_FORCE_REBUILD=1 always rebuilds.
env_should_rebuild() {
  local prefix="$1" stamp_file="$2" expected="$3"
  if is_enabled "${APP_FORCE_REBUILD:-0}"; then
    return 0
  fi
  if ! env_exists "${prefix}"; then
    return 0
  fi
  if is_enabled "${APP_REUSE_ENV:-0}"; then
    if [ -f "${stamp_file}" ] && [ "$(cat "${stamp_file}")" = "${expected}" ]; then
      return 1
    fi
    return 0
  fi
  return 0
}

# Create a fresh env prefix at a pinned python version.
create_env() {
  local prefix="$1" python_version="$2"
  shift 2
  rm -rf "${prefix}"
  mkdir -p "$(dirname "${prefix}")"
  "$(micromamba_bin)" create -y -p "${prefix}" \
    -c conda-forge "python=${python_version}" "$@"
}

write_stamp() {
  local stamp_file="$1" value="$2"
  mkdir -p "$(dirname "${stamp_file}")"
  printf '%s\n' "${value}" > "${stamp_file}"
}
