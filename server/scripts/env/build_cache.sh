# Source this file from run/prepare scripts so the exports stay in the caller.

configure_build_cache() {
  export APP_CACHE_ROOT="${APP_CACHE_ROOT:-/workspace/cache}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${APP_CACHE_ROOT}/xdg}"
  export HF_HOME="${HF_HOME:-${APP_CACHE_ROOT}/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export TORCH_HOME="${TORCH_HOME:-${APP_CACHE_ROOT}/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${APP_CACHE_ROOT}/pip}"
  export CCACHE_DIR="${CCACHE_DIR:-${APP_CACHE_ROOT}/ccache}"
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
  export CCACHE_COMPRESS="${CCACHE_COMPRESS:-1}"
  export CCACHE_BASEDIR="${CCACHE_BASEDIR:-${APP_DIR:-${SERVER_DIR:-$(pwd)}}}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${APP_CACHE_ROOT}/torch_extensions/${RECONVIAGEN_ENV_NAME:-worker-reconviagen}}"
  export CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-ccache}"
  export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
  export CMAKE_CUDA_COMPILER_LAUNCHER="${CMAKE_CUDA_COMPILER_LAUNCHER:-ccache}"

  mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${HF_HUB_CACHE}" \
    "${TORCH_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${CCACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}"
}

configure_ccache_for_env() {
  local env_name="$1"
  if micromamba run -n "${env_name}" bash -lc 'command -v ccache >/dev/null 2>&1'; then
    micromamba run -n "${env_name}" ccache --set-config "cache_dir=${CCACHE_DIR}" >/dev/null 2>&1 || true
    micromamba run -n "${env_name}" ccache --max-size="${CCACHE_MAXSIZE}" >/dev/null 2>&1 || true
  fi
}
