#!/usr/bin/env bash
set -Eeuo pipefail

RECONVIAGEN_STARTUP_BLOCKED=0
ORIGINAL_ENV_NAMES=$'\n'"$(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p')"$'\n'
APP_RELOAD_REQUESTED=0
APP_EXIT_REQUESTED=0
LOADED_ENV_NAMES=""

log() {
  printf '[run.sh] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

set_default() {
  local name="$1"
  local value="$2"
  if [ -z "${!name+x}" ]; then
    printf -v "${name}" '%s' "${value}"
  fi
  export "${name}"
}

is_enabled() {
  case "${1:-}" in
    1 | true | True | TRUE | yes | Yes | YES | on | On | ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

path_is_empty() {
  [ ! -e "$1" ] || [ -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]
}

stamp_hash() {
  cksum | awk '{print $1 ":" $2}'
}

is_original_env_name() {
  case "${ORIGINAL_ENV_NAMES}" in
    *$'\n'"$1"$'\n'*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
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

trim_value() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

track_loaded_env_name() {
  case " ${LOADED_ENV_NAMES} " in
    *" $1 "*)
      ;;
    *)
      LOADED_ENV_NAMES="${LOADED_ENV_NAMES} $1"
      ;;
  esac
}

reset_loaded_env_names() {
  local name
  for name in ${LOADED_ENV_NAMES}; do
    if ! is_original_env_name "${name}"; then
      unset "${name}"
    fi
  done
  LOADED_ENV_NAMES=""
}

load_env_file() {
  local file="$1"
  local line name value

  if [ ! -f "${file}" ]; then
    return 0
  fi

  log "Loading environment defaults from ${file}."
  while IFS= read -r line || [ -n "${line}" ]; do
    line="${line%$'\r'}"
    line="$(trim_value "${line}")"
    case "${line}" in
      "" | \#*)
        continue
        ;;
      export\ *)
        line="${line#export }"
        ;;
    esac

    name="${line%%=*}"
    value="${line#*=}"
    name="$(trim_value "${name}")"
    value="$(trim_value "${value}")"

    if [ "${line}" = "${name}" ] || [[ ! "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      log "Skipping invalid env line in ${file}: ${line}"
      continue
    fi
    if is_secret_env_name "${name}"; then
      log "Skipping secret-like ${name} in ${file}; set it in RunPod environment variables."
      continue
    fi
    if is_original_env_name "${name}"; then
      continue
    fi

    if [[ "${value}" == \"*\" && "${value}" == *\" && "${#value}" -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' && "${#value}" -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    fi

    printf -v "${name}" '%s' "${value}"
    export "${name}"
    track_loaded_env_name "${name}"
  done < "${file}"
}

load_env_files() {
  load_env_file "${APP_ENV_FILE:-${APP_DIR}/server/.env}"
  load_env_file "${APP_ENV_LOCAL_FILE:-${APP_DIR}/server/.env.local}"
}

default_reconviagen_prepare() {
  case "$(uname -s)" in
    Linux) printf '1' ;;
    *) printf '0' ;;
  esac
}

init_defaults() {
  set_default APP_PERSIST_ROOT "/workspace"
  set_default APP_CACHE_ROOT "${APP_PERSIST_ROOT}/cache"
  set_default APP_STATE_ROOT "${APP_CACHE_ROOT}/state"
  set_default APP_ENV_ROOT "${APP_CACHE_ROOT}/envs"
  set_default APP_MODEL_ROOT "${APP_CACHE_ROOT}/vggt-lidar"
  set_default APP_REPO_URL "https://github.com/mokyabun/iphone-lidar-vggt.git"
  set_default APP_REPO_REF "main"
  set_default APP_DIR "${APP_PERSIST_ROOT}/iphone-lidar-vggt"
  set_default APP_RUNS_DIR "${APP_PERSIST_ROOT}/runs"
  set_default APP_HOST "0.0.0.0"
  set_default APP_PORT "8000"
  set_default APP_ENV_FILE "${APP_DIR}/server/.env"
  set_default APP_ENV_LOCAL_FILE "${APP_DIR}/server/.env.local"
  set_default APP_MANAGED_SERVICES "1"
  set_default APP_UPDATE_MODE "reset"
  set_default APP_INSTALL_EXTRAS "reconstruction,vggt,segmentation"
  set_default APP_INSTALL_APT "1"
  set_default APP_PREPARE_VGGT "0"
  set_default APP_PREFETCH_VGGT "0"
  set_default APP_PREPARE_RECONVIAGEN "$(default_reconviagen_prepare)"
  set_default APP_PREFETCH_RECONVIAGEN "1"
  set_default APP_REFRESH_MODEL_CACHE "0"
  set_default PYTHON_BIN ""
  set_default APP_VENV_DIR "${APP_DIR}/server/.venv"
  set_default APP_VENV_REAL_DIR "${APP_ENV_ROOT}/iphone-lidar-vggt"
  set_default APP_PYTHON_BIN "${APP_VENV_DIR}/bin/python"
  set_default APP_UVICORN_BIN "${APP_VENV_DIR}/bin/uvicorn"
  set_default APP_VGGT_PREPARE_BIN "${APP_VENV_DIR}/bin/vggt-prepare"
  set_default APP_USE_SYSTEM_TORCH "1"

  set_default PYTHONUNBUFFERED "1"
  set_default PIP_NO_CACHE_DIR "1"
  set_default UV_CACHE_DIR "${APP_CACHE_ROOT}/uv"
  set_default UV_LINK_MODE "hardlink"
  set_default APP_RUNTIME_DIR "${APP_CACHE_ROOT}/run"
  set_default APP_LOG_FILE "${APP_RUNTIME_DIR}/uvicorn.log"
  set_default APP_MANAGER_PID_FILE "${APP_RUNTIME_DIR}/run-sh.pid"
  set_default APP_UVICORN_PID_FILE "${APP_RUNTIME_DIR}/uvicorn.pid"
  set_default APP_RECONVIAGEN_WORKER_PID_FILE "${APP_RUNTIME_DIR}/reconviagen-worker.pid"

  set_default OMP_NUM_THREADS "4"
  set_default MKL_NUM_THREADS "4"
  set_default OPENBLAS_NUM_THREADS "4"
  set_default NUMEXPR_NUM_THREADS "4"
  set_default TORCH_NUM_THREADS "4"
  set_default TORCH_NUM_INTEROP_THREADS "1"
  set_default TORCH_CUDNN_BENCHMARK "1"
  set_default TORCH_FLOAT32_MATMUL_PRECISION "high"

  set_default XDG_CACHE_HOME "${APP_CACHE_ROOT}/xdg"
  set_default XDG_CONFIG_HOME "${APP_CACHE_ROOT}/config"
  set_default TORCH_HOME "${APP_MODEL_ROOT}/torch"
  set_default TORCH_EXTENSIONS_DIR "${APP_CACHE_ROOT}/torch-extensions"
  set_default TRITON_CACHE_DIR "${APP_CACHE_ROOT}/triton"
  set_default CUDA_CACHE_PATH "${APP_CACHE_ROOT}/cuda"
  set_default NUMBA_CACHE_DIR "${APP_CACHE_ROOT}/numba"
  set_default MPLCONFIGDIR "${APP_CACHE_ROOT}/matplotlib"
  set_default U2NET_HOME "${APP_MODEL_ROOT}/rembg"

  set_default VGGT_AUTO_DOWNLOAD "1"
  set_default VGGT_CACHE_ROOT "${APP_MODEL_ROOT}"
  set_default VGGT_REPO_REF "main"
  set_default VGGT_REPO_UPDATE "1"
  set_default HF_HOME "${APP_MODEL_ROOT}/huggingface"
  set_default HF_HUB_CACHE "${HF_HOME}/hub"
  set_default HF_XET_CACHE "${HF_HOME}/xet"
  set_default HF_ASSETS_CACHE "${HF_HOME}/assets"
  set_default HF_TOKEN_PATH "${HF_HOME}/token"

  set_default RECONVIAGEN_PRELOAD "1"
  set_default RECONVIAGEN_REPO_URL "https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git"
  set_default RECONVIAGEN_REPO_REF "v0.5"
  set_default RECONVIAGEN_REPO_DIR "${APP_CACHE_ROOT}/ReconViaGen"
  set_default RECONVIAGEN_ENV "${APP_CACHE_ROOT}/reconviagen-v05-env"
  set_default RECONVIAGEN_PYTHON "${RECONVIAGEN_ENV}/bin/python"
  set_default RECONVIAGEN_WORKER_PORT "8011"
  set_default RECONVIAGEN_WORKER_ERROR "${APP_CACHE_ROOT}/reconviagen-worker.error"
  set_default RECONVIAGEN_WORKER_LOG "${APP_CACHE_ROOT}/reconviagen-worker.log"
  set_default RECONVIAGEN_WORKER_RESTART "1"
  set_default RECONVIAGEN_WORKER_RESTART_DELAY_SECONDS "20"
  set_default RECONVIAGEN_DINO_MODEL "facebook/dinov3-vitl16-pretrain-lvd1689m"
  set_default SPCONV_ALGO "native"
  set_default OPENCV_IO_ENABLE_OPENEXR "1"
  set_default PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"
  set_default XFORMERS_DISABLED "1"

  set_default YOLO_CONFIG_DIR "${APP_CACHE_ROOT}/ultralytics"
  set_default ULTRALYTICS_SETTINGS "${YOLO_CONFIG_DIR}/settings.json"
  set_default ULTRALYTICS_WEIGHTS_DIR "${APP_MODEL_ROOT}/ultralytics"
  set_default ULTRALYTICS_RUNS_DIR "${APP_CACHE_ROOT}/ultralytics/runs"
}

run_as_root() {
  if [ "$(id -u)" = "0" ]; then
    "$@"
  elif command_exists sudo; then
    sudo "$@"
  else
    log "Skipping root command because neither root nor sudo is available: $*"
    return 1
  fi
}

install_system_packages() {
  if ! is_enabled "${APP_INSTALL_APT}"; then
    log "Skipping apt package install."
    return 0
  fi
  if ! command_exists apt-get; then
    log "apt-get not found; skipping system package install."
    return 0
  fi
  if [ "$(id -u)" != "0" ] && ! command_exists sudo; then
    log "No root or sudo access; skipping system package install."
    return 0
  fi

  local packages=(
    ca-certificates
    curl
    git
    build-essential
    ffmpeg
    libgl1
    libglib2.0-0
    libjpeg-dev
    ninja-build
  )

  log "Installing system packages."
  run_as_root apt-get update
  run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
  run_as_root rm -rf /var/lib/apt/lists/*
}

prepare_cache_dirs() {
  local dirs=(
    "${APP_CACHE_ROOT}"
    "${APP_STATE_ROOT}"
    "${APP_ENV_ROOT}"
    "${APP_MODEL_ROOT}"
    "${VGGT_CACHE_ROOT}"
    "${HF_HOME}"
    "${HF_HUB_CACHE}"
    "${HF_XET_CACHE}"
    "${HF_ASSETS_CACHE}"
    "${TORCH_HOME}"
    "${TORCH_EXTENSIONS_DIR}"
    "${TRITON_CACHE_DIR}"
    "${CUDA_CACHE_PATH}"
    "${NUMBA_CACHE_DIR}"
    "${MPLCONFIGDIR}"
    "${U2NET_HOME}"
    "${RECONVIAGEN_REPO_DIR}"
    "${RECONVIAGEN_ENV}"
    "${APP_VENV_REAL_DIR}"
    "${UV_CACHE_DIR}"
    "${APP_RUNTIME_DIR}"
    "${APP_RUNS_DIR}"
    "${YOLO_CONFIG_DIR}"
    "${YOLO_CONFIG_DIR}/Ultralytics"
    "$(dirname "${ULTRALYTICS_SETTINGS}")"
    "${ULTRALYTICS_WEIGHTS_DIR}"
    "${ULTRALYTICS_RUNS_DIR}"
  )

  log "Preparing cache directories."
  mkdir -p "${dirs[@]}"
  chmod -R a+rwX "${APP_CACHE_ROOT}" "${APP_MODEL_ROOT}" || true

  if [ -d /root/.cache/torch ] && [ -z "$(find "${TORCH_HOME}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    log "Migrating the legacy Torch cache into persistent storage."
    cp -a /root/.cache/torch/. "${TORCH_HOME}/"
  fi
}

ensure_persistent_link() {
  local target="$1"
  local link="$2"
  local label="$3"
  local current backup_dir

  if [ "${target}" = "${link}" ]; then
    return 0
  fi

  mkdir -p "${target}" "$(dirname "${link}")"
  if [ -L "${link}" ]; then
    current="$(readlink "${link}")"
    if [ "${current}" = "${target}" ]; then
      return 0
    fi
    rm -f "${link}"
  elif [ -e "${link}" ]; then
    if [ -d "${link}" ] && path_is_empty "${target}"; then
      log "Moving existing ${label} into persistent storage at ${target}."
      rmdir "${target}" 2>/dev/null || true
      mv "${link}" "${target}"
    elif [ -d "${link}" ] && path_is_empty "${link}"; then
      rmdir "${link}"
    else
      backup_dir="${link}.local.$(date +%Y%m%d%H%M%S)"
      log "Preserving existing ${label} at ${backup_dir} before creating persistent symlink."
      mv "${link}" "${backup_dir}"
    fi
  fi

  ln -s "${target}" "${link}"
}

prepare_persistent_links() {
  ensure_persistent_link "${APP_VENV_REAL_DIR}" "${APP_VENV_DIR}" "app virtualenv"
  ensure_persistent_link "${APP_RUNS_DIR}" "${APP_DIR}/server/runs" "run outputs"
}

backup_non_git_app_dir() {
  if [ -e "${APP_DIR}" ] && [ ! -d "${APP_DIR}/.git" ]; then
    local backup_dir="${APP_DIR}.backup.$(date +%Y%m%d%H%M%S)"
    log "Found non-git APP_DIR. Moving it to ${backup_dir}."
    mv "${APP_DIR}" "${backup_dir}"
  fi
}

git_safe_directory() {
  git -C "$1" config --global --add safe.directory "$1" >/dev/null 2>&1 || true
}

sync_existing_repo() {
  log "Updating repository in ${APP_DIR}."
  git_safe_directory "${APP_DIR}"
  git -C "${APP_DIR}" remote set-url origin "${APP_REPO_URL}"

  case "${APP_UPDATE_MODE}" in
    none)
      log "APP_UPDATE_MODE=none; leaving repository checkout unchanged."
      ;;
    reset)
      git -C "${APP_DIR}" fetch --depth 1 --prune origin "${APP_REPO_REF}"
      git -C "${APP_DIR}" reset --hard FETCH_HEAD
      git -C "${APP_DIR}" clean -fd -e runs/ -e .venv/
      ;;
    pull)
      git -C "${APP_DIR}" fetch --prune origin "${APP_REPO_REF}"
      git -C "${APP_DIR}" checkout "${APP_REPO_REF}" 2>/dev/null \
        || git -C "${APP_DIR}" checkout -B "${APP_REPO_REF}" FETCH_HEAD
      git -C "${APP_DIR}" pull --ff-only origin "${APP_REPO_REF}"
      ;;
    *)
      die "Unknown APP_UPDATE_MODE=${APP_UPDATE_MODE}; use reset, pull, or none."
      ;;
  esac
}

clone_repo() {
  log "Cloning ${APP_REPO_URL}#${APP_REPO_REF} into ${APP_DIR}."
  mkdir -p "$(dirname "${APP_DIR}")"
  git clone --depth 1 --branch "${APP_REPO_REF}" "${APP_REPO_URL}" "${APP_DIR}"
}

sync_repo() {
  backup_non_git_app_dir
  if [ -d "${APP_DIR}/.git" ]; then
    sync_existing_repo
  else
    clone_repo
  fi
}

resolve_python_bin() {
  if [ -n "${PYTHON_BIN}" ]; then
    return 0
  fi
  if command_exists python; then
    PYTHON_BIN=python
  elif command_exists python3; then
    PYTHON_BIN=python3
  else
    die "Python is not installed."
  fi
  export PYTHON_BIN
}

ensure_uv() {
  if command_exists uv; then
    return 0
  fi

  log "Installing uv."
  resolve_python_bin
  "${PYTHON_BIN}" -m pip install --upgrade uv
}

activate_app_venv() {
  case ":${PATH}:" in
    *":${APP_VENV_DIR}/bin:"*)
      ;;
    *)
      export PATH="${APP_VENV_DIR}/bin:${PATH}"
      ;;
  esac
}

ensure_app_venv() {
  resolve_python_bin

  if is_enabled "${APP_USE_SYSTEM_TORCH}"; then
    if [ -f "${APP_VENV_DIR}/pyvenv.cfg" ] \
      && ! grep -Eq '^include-system-site-packages = true$' "${APP_VENV_DIR}/pyvenv.cfg"; then
      log "Recreating app venv with system site packages enabled."
      rm -rf "${APP_VENV_REAL_DIR}"
      mkdir -p "${APP_VENV_REAL_DIR}"
      prepare_persistent_links
    fi
    uv venv --python "${PYTHON_BIN}" --system-site-packages --allow-existing "${APP_VENV_DIR}"
  else
    uv venv --python "${PYTHON_BIN}" --allow-existing "${APP_VENV_DIR}"
  fi
}

app_sync_stamp() {
  local python_version
  local files=(
    "${APP_DIR}/server/pyproject.toml"
    "${APP_DIR}/server/uv.lock"
    "${APP_DIR}/server/orchestration/pyproject.toml"
    "${APP_DIR}/server/reconviagen-worker/pyproject.toml"
    "${APP_DIR}/server/vggt-worker/pyproject.toml"
  )
  local file

  python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  {
    printf 'python=%s\n' "${python_version}"
    printf 'uv=%s\n' "$(uv --version 2>/dev/null || true)"
    printf 'extras=%s\n' "${APP_INSTALL_EXTRAS}"
    printf 'use_system_torch=%s\n' "${APP_USE_SYSTEM_TORCH}"
    printf 'uv_link_mode=%s\n' "${UV_LINK_MODE}"
    for file in "${files[@]}"; do
      if [ -f "${file}" ]; then
        cksum "${file}"
      else
        printf 'missing  %s\n' "${file#${APP_DIR}/}"
      fi
    done
  } | stamp_hash
}

install_python_packages() {
  log "Syncing Python workspace dependencies."
  ensure_uv
  ensure_app_venv

  cd "${APP_DIR}/server"
  local sync_args=()
  local stamp_file="${APP_STATE_ROOT}/app-uv-sync.stamp"
  local current_stamp
  local extra
  IFS=',' read -ra requested_extras <<< "${APP_INSTALL_EXTRAS}"
  for extra in "${requested_extras[@]}"; do
    extra="${extra//[[:space:]]/}"
    case "${extra}" in
      "")
        ;;
      reconstruction | vggt | segmentation | dev | dev-mesh)
        sync_args+=("--extra" "${extra}")
        ;;
      *)
        die "Unknown APP_INSTALL_EXTRAS entry: ${extra}"
        ;;
    esac
  done

  if is_enabled "${APP_USE_SYSTEM_TORCH}"; then
    sync_args+=("--no-install-package" "torch" "--no-install-package" "torchvision")
  fi

  current_stamp="$(app_sync_stamp)"
  if [ -x "${APP_UVICORN_BIN}" ] \
    && [ -f "${stamp_file}" ] \
    && [ "$(cat "${stamp_file}")" = "${current_stamp}" ]; then
    log "Python workspace environment is already current."
    activate_app_venv
    return 0
  fi

  UV_PROJECT_ENVIRONMENT="${APP_VENV_DIR}" uv sync "${sync_args[@]}"
  mkdir -p "$(dirname "${stamp_file}")"
  printf '%s\n' "${current_stamp}" > "${stamp_file}"
  activate_app_venv
}

configure_ultralytics() {
  "${APP_PYTHON_BIN}" -c \
    "from ultralytics import settings; settings.update({'weights_dir': '${ULTRALYTICS_WEIGHTS_DIR}', 'runs_dir': '${ULTRALYTICS_RUNS_DIR}'})" \
    >/dev/null 2>&1 || true
}

prepare_vggt() {
  if is_enabled "${APP_PREPARE_VGGT}"; then
    log "Preparing VGGT repo and Python package."
    VGGT_DOWNLOAD_WEIGHTS="${APP_PREFETCH_VGGT}" "${APP_VGGT_PREPARE_BIN}"
  elif is_enabled "${APP_PREFETCH_VGGT}"; then
    die "APP_PREFETCH_VGGT=1 requires APP_PREPARE_VGGT=1."
  else
    log "Skipping VGGT preparation. Set APP_PREPARE_VGGT=1 to clone and install VGGT before serving."
  fi
}

sync_reconviagen_repo() {
  if [ -d "${RECONVIAGEN_REPO_DIR}/.git" ]; then
    log "Updating ReconViaGen ${RECONVIAGEN_REPO_REF}."
    git_safe_directory "${RECONVIAGEN_REPO_DIR}"
    git -C "${RECONVIAGEN_REPO_DIR}" fetch --depth 1 origin "${RECONVIAGEN_REPO_REF}"
    git -C "${RECONVIAGEN_REPO_DIR}" reset --hard FETCH_HEAD
    git -C "${RECONVIAGEN_REPO_DIR}" submodule update --init --recursive
  else
    log "Cloning ReconViaGen ${RECONVIAGEN_REPO_REF}."
    rm -rf "${RECONVIAGEN_REPO_DIR}"
    git clone --recursive --depth 1 --branch "${RECONVIAGEN_REPO_REF}" \
      "${RECONVIAGEN_REPO_URL}" "${RECONVIAGEN_REPO_DIR}"
  fi
}

reconviagen_revision() {
  git -C "${RECONVIAGEN_REPO_DIR}" rev-parse HEAD
}

installed_reconviagen_revision() {
  if [ -f "${RECONVIAGEN_ENV}/.reconviagen-revision" ]; then
    cat "${RECONVIAGEN_ENV}/.reconviagen-revision"
  fi
}

create_reconviagen_env() {
  local revision="$1"

  log "Creating the isolated ReconViaGen uv environment."
  rm -rf "${RECONVIAGEN_ENV}"
  sync_reconviagen_runtime_environment
  rm -rf /tmp/extensions
  env PATH="${RECONVIAGEN_ENV}/bin:${PATH}" bash -c \
    "cd '${RECONVIAGEN_REPO_DIR}' && . ./setup.sh --xformers --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrec --spconv --kaolin --nvdiffrast"
  printf '%s\n' "${revision}" > "${RECONVIAGEN_ENV}/.reconviagen-revision"
}

ensure_reconviagen_env() {
  local revision
  revision="$(reconviagen_revision)"

  if [ ! -x "${RECONVIAGEN_PYTHON}" ] || [ "$(installed_reconviagen_revision)" != "${revision}" ]; then
    create_reconviagen_env "${revision}"
  else
    log "ReconViaGen environment is already current."
  fi
}

ensure_reconviagen_runtime_dependency() {
  local import_check="$1"
  shift
  local message="$1"
  shift

  if ! "${RECONVIAGEN_PYTHON}" -c "${import_check}" >/dev/null 2>&1; then
    log "${message}"
    uv pip install --python "${RECONVIAGEN_PYTHON}" "$@"
  fi
}

reconviagen_runtime_stamp() {
  local repo_revision="${1:-}"
  local files=(
    "${APP_DIR}/server/reconviagen-runtime/pyproject.toml"
    "${APP_DIR}/server/reconviagen-worker/pyproject.toml"
    "${APP_DIR}/server/uv.lock"
  )
  local file

  {
    printf 'repo_revision=%s\n' "${repo_revision}"
    printf 'uv=%s\n' "$(uv --version 2>/dev/null || true)"
    for file in "${files[@]}"; do
      if [ -f "${file}" ]; then
        cksum "${file}"
      else
        printf 'missing  %s\n' "${file#${APP_DIR}/}"
      fi
    done
  } | stamp_hash
}

sync_reconviagen_runtime_environment() {
  local repo_revision current_stamp stamp_file

  repo_revision="$(reconviagen_revision 2>/dev/null || true)"
  current_stamp="$(reconviagen_runtime_stamp "${repo_revision}")"
  stamp_file="${RECONVIAGEN_ENV}/.runtime-sync-stamp"
  if [ -x "${RECONVIAGEN_PYTHON}" ] \
    && [ -f "${stamp_file}" ] \
    && [ "$(cat "${stamp_file}")" = "${current_stamp}" ]; then
    log "ReconViaGen runtime environment is already synced."
    return 0
  fi

  log "Syncing the isolated ReconViaGen runtime environment."
  UV_PROJECT_ENVIRONMENT="${RECONVIAGEN_ENV}" uv sync \
    --project "${APP_DIR}/server/reconviagen-runtime" \
    --python 3.10
  printf '%s\n' "${current_stamp}" > "${stamp_file}"
}

ensure_reconviagen_runtime_patches() {
  sync_reconviagen_runtime_environment

  ensure_reconviagen_runtime_dependency \
    "import o_voxel" \
    "Installing the nested ReconViaGen o-voxel extension." \
    "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation
}

reconviagen_pythonpath() {
  local paths="${APP_DIR}/server/reconviagen-worker/src:${APP_DIR}/server/orchestration/src"
  if [ -n "${PYTHONPATH:-}" ]; then
    paths="${paths}:${PYTHONPATH}"
  fi
  printf '%s' "${paths}"
}

verify_reconviagen_hf_access() {
  if "${RECONVIAGEN_PYTHON}" -c \
    "from huggingface_hub import hf_hub_download; hf_hub_download('${RECONVIAGEN_DINO_MODEL}', 'config.json')" \
    >/dev/null 2>&1; then
    return 0
  fi

  local access_error
  access_error="ReconViaGen requires access to the gated Hugging Face model ${RECONVIAGEN_DINO_MODEL}. Accept its license at https://huggingface.co/${RECONVIAGEN_DINO_MODEL}, then set HF_TOKEN to a Hugging Face read token and restart run.sh."
  printf '%s\n' "${access_error}" > "${RECONVIAGEN_WORKER_ERROR}"
  RECONVIAGEN_STARTUP_BLOCKED=1
  log "${access_error}"
  return 1
}

prefetch_reconviagen_weights() {
  local stamp_file="${APP_STATE_ROOT}/reconviagen-models-prefetched.stamp"
  local model_stamp

  if ! is_enabled "${APP_PREFETCH_RECONVIAGEN}"; then
    return 0
  fi

  model_stamp="$(printf '%s\n%s\n%s\n' \
    "Stable-X/trellis-vggt-v0-2" \
    "microsoft/TRELLIS.2-4B" \
    "${RECONVIAGEN_DINO_MODEL}" | stamp_hash)"
  if ! is_enabled "${APP_REFRESH_MODEL_CACHE}" \
    && [ -f "${stamp_file}" ] \
    && [ "$(cat "${stamp_file}")" = "${model_stamp}" ]; then
    log "ReconViaGen model cache was already prefetched."
    return 0
  fi

  log "Prefetching ReconViaGen, TRELLIS.2, and DINOv3 weights."
  "${RECONVIAGEN_PYTHON}" -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Stable-X/trellis-vggt-v0-2'); snapshot_download('microsoft/TRELLIS.2-4B'); snapshot_download('${RECONVIAGEN_DINO_MODEL}')"
  mkdir -p "$(dirname "${stamp_file}")"
  printf '%s\n' "${model_stamp}" > "${stamp_file}"
}

prepare_reconviagen() {
  if ! is_enabled "${APP_PREPARE_RECONVIAGEN}"; then
    log "Skipping ReconViaGen preparation."
    return 0
  fi

  if [ "$(uname -s)" != "Linux" ]; then
    log "Skipping ReconViaGen preparation on $(uname -s); CUDA setup is only supported by this script on Linux."
    return 0
  fi

  sync_reconviagen_repo
  ensure_reconviagen_env
  ensure_reconviagen_runtime_patches
  verify_reconviagen_hf_access || return 0
  prefetch_reconviagen_weights
}

start_reconviagen_worker() {
  if ! is_enabled "${RECONVIAGEN_PRELOAD}" || [ ! -x "${RECONVIAGEN_PYTHON}" ]; then
    rm -f "${APP_RECONVIAGEN_WORKER_PID_FILE}"
    return 0
  fi

  export RECONVIAGEN_WORKER_URL="http://127.0.0.1:${RECONVIAGEN_WORKER_PORT}"
  if [ "${RECONVIAGEN_STARTUP_BLOCKED}" = "1" ]; then
    log "ReconViaGen worker was not started; see ${RECONVIAGEN_WORKER_ERROR}."
    return 0
  fi

  rm -f "${RECONVIAGEN_WORKER_ERROR}"
  log "Starting ReconViaGen worker on ${RECONVIAGEN_WORKER_URL}."
  (
    set +e
    while true; do
      env PATH="${RECONVIAGEN_ENV}/bin:${PATH}" \
        PYTHONPATH="$(reconviagen_pythonpath)" \
        "${RECONVIAGEN_PYTHON}" "${APP_DIR}/server/reconviagen-worker/src/reconviagen_worker/main.py" \
        --host 127.0.0.1 --port "${RECONVIAGEN_WORKER_PORT}" 2>&1 \
        | tee -a "${RECONVIAGEN_WORKER_LOG}"
      worker_status="${PIPESTATUS[0]}"
      if [ "${worker_status}" -eq 0 ]; then
        break
      fi
      {
        printf 'worker exited with status %s\n' "${worker_status}"
        tail -n 60 "${RECONVIAGEN_WORKER_LOG}"
      } > "${RECONVIAGEN_WORKER_ERROR}"
      if ! is_enabled "${RECONVIAGEN_WORKER_RESTART}"; then
        break
      fi
      log "ReconViaGen worker exited with status ${worker_status}; restarting in ${RECONVIAGEN_WORKER_RESTART_DELAY_SECONDS}s."
      sleep "${RECONVIAGEN_WORKER_RESTART_DELAY_SECONDS}"
      rm -f "${RECONVIAGEN_WORKER_ERROR}"
    done
  ) &
  RECONVIAGEN_WORKER_SUPERVISOR_PID=$!
  printf '%s\n' "${RECONVIAGEN_WORKER_SUPERVISOR_PID}" > "${APP_RECONVIAGEN_WORKER_PID_FILE}"
}

stop_reconviagen_worker() {
  local pid

  if [ -n "${RECONVIAGEN_WORKER_SUPERVISOR_PID:-}" ]; then
    kill "${RECONVIAGEN_WORKER_SUPERVISOR_PID}" >/dev/null 2>&1 || true
    wait "${RECONVIAGEN_WORKER_SUPERVISOR_PID}" >/dev/null 2>&1 || true
    RECONVIAGEN_WORKER_SUPERVISOR_PID=""
  elif [ -f "${APP_RECONVIAGEN_WORKER_PID_FILE:-}" ]; then
    pid="$(cat "${APP_RECONVIAGEN_WORKER_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${pid}" ]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  fi

  if command_exists pgrep; then
    pgrep -f "${APP_DIR}/server/reconviagen-worker/src/reconviagen_worker/main.py" \
      | while IFS= read -r pid; do
          [ "${pid}" = "$$" ] || kill "${pid}" >/dev/null 2>&1 || true
        done
  fi

  rm -f "${APP_RECONVIAGEN_WORKER_PID_FILE}"
}

start_uvicorn() {
  log "Starting FastAPI on ${APP_HOST}:${APP_PORT}. Logs: ${APP_LOG_FILE}"
  mkdir -p "$(dirname "${APP_LOG_FILE}")"
  "${APP_UVICORN_BIN}" orchestration.api:app --host "${APP_HOST}" --port "${APP_PORT}" >> "${APP_LOG_FILE}" 2>&1 &
  APP_UVICORN_PID=$!
  printf '%s\n' "${APP_UVICORN_PID}" > "${APP_UVICORN_PID_FILE}"
}

stop_uvicorn() {
  local pid

  if [ -n "${APP_UVICORN_PID:-}" ]; then
    kill "${APP_UVICORN_PID}" >/dev/null 2>&1 || true
    wait "${APP_UVICORN_PID}" >/dev/null 2>&1 || true
    APP_UVICORN_PID=""
  elif [ -f "${APP_UVICORN_PID_FILE:-}" ]; then
    pid="$(cat "${APP_UVICORN_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${pid}" ]; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "${APP_UVICORN_PID_FILE}"
}

stop_services() {
  stop_uvicorn
  stop_reconviagen_worker
}

request_reload() {
  APP_RELOAD_REQUESTED=1
  log "Reload requested; stopping managed services."
  stop_services
}

request_exit() {
  APP_EXIT_REQUESTED=1
  log "Shutdown requested; stopping managed services."
  stop_services
}

start_managed_services() {
  mkdir -p "${APP_RUNTIME_DIR}"
  printf '%s\n' "$$" > "${APP_MANAGER_PID_FILE}"
  trap request_reload HUP
  trap request_exit INT TERM

  while true; do
    APP_RELOAD_REQUESTED=0
    reset_loaded_env_names
    init_defaults
    load_env_files
    init_defaults
    prepare_cache_dirs
    prepare_persistent_links
    configure_ultralytics
    start_reconviagen_worker
    start_uvicorn

    wait "${APP_UVICORN_PID}" || true
    stop_services

    if [ "${APP_EXIT_REQUESTED}" = "1" ]; then
      rm -f "${APP_MANAGER_PID_FILE}"
      exit 0
    fi
    if [ "${APP_RELOAD_REQUESTED}" = "1" ]; then
      log "Restarting managed services with the current env files."
      continue
    fi

    rm -f "${APP_MANAGER_PID_FILE}"
    return 1
  done
}

start_app() {
  cd "${APP_DIR}/server"

  if [ "$#" -gt 0 ]; then
    log "Starting custom command: $*"
    exec "$@"
  fi

  if [ -n "${APP_START_COMMAND:-}" ]; then
    log "Starting APP_START_COMMAND: ${APP_START_COMMAND}"
    exec bash -lc "${APP_START_COMMAND}"
  fi

  if is_enabled "${APP_MANAGED_SERVICES}"; then
    start_managed_services
  else
    start_reconviagen_worker
    log "Starting FastAPI on ${APP_HOST}:${APP_PORT}."
    exec "${APP_UVICORN_BIN}" orchestration.api:app --host "${APP_HOST}" --port "${APP_PORT}"
  fi
}

main() {
  init_defaults
  load_env_files
  init_defaults
  install_system_packages
  sync_repo
  load_env_files
  init_defaults
  prepare_cache_dirs
  prepare_persistent_links
  install_python_packages
  configure_ultralytics
  prepare_vggt
  prepare_reconviagen
  start_app "$@"
}

main "$@"
