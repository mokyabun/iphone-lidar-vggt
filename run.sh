#!/usr/bin/env bash
set -Eeuo pipefail

APP_REPO_URL="${APP_REPO_URL:-https://github.com/mokyabun/iphone-lidar-vggt.git}"
APP_REPO_REF="${APP_REPO_REF:-main}"
APP_DIR="${APP_DIR:-/workspace/iphone-lidar-vggt}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_UPDATE_MODE="${APP_UPDATE_MODE:-reset}"
APP_INSTALL_EXTRAS="${APP_INSTALL_EXTRAS:-reconstruction,vggt,segmentation}"
APP_INSTALL_APT="${APP_INSTALL_APT:-1}"
APP_PREPARE_VGGT="${APP_PREPARE_VGGT:-1}"
APP_PREFETCH_VGGT="${APP_PREFETCH_VGGT:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
export VGGT_AUTO_DOWNLOAD="${VGGT_AUTO_DOWNLOAD:-1}"
export VGGT_CACHE_ROOT="${VGGT_CACHE_ROOT:-/workspace/cache/vggt-lidar}"
export HF_HOME="${HF_HOME:-/workspace/cache/vggt-lidar/huggingface}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/workspace/cache/ultralytics}"
export ULTRALYTICS_SETTINGS="${ULTRALYTICS_SETTINGS:-/workspace/cache/ultralytics/settings.json}"
export SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-24}"
export SCAN_STRIDE="${SCAN_STRIDE:-4}"
export SCAN_RUN_TSDF="${SCAN_RUN_TSDF:-0}"
export MESH_METHOD="${MESH_METHOD:-object_tsdf}"
export VGGT_MAX_IMAGES="${VGGT_MAX_IMAGES:-12}"
export VGGT_PRELOAD="${VGGT_PRELOAD:-1}"
export OBJECT_MASK_BACKEND="${OBJECT_MASK_BACKEND:-sam3_depth}"
export OBJECT_SAM_MODEL="${OBJECT_SAM_MODEL:-sam3.pt}"
export OBJECT_SAM_MAX_FRAMES="${OBJECT_SAM_MAX_FRAMES:-3}"
export OBJECT_CENTER_FRACTION="${OBJECT_CENTER_FRACTION:-0.35}"
export OBJECT_DEPTH_BAND_METERS="${OBJECT_DEPTH_BAND_METERS:-0.35}"
export OBJECT_TSDF_VOXEL_METERS="${OBJECT_TSDF_VOXEL_METERS:-0.008}"
export OBJECT_TSDF_TRUNC_METERS="${OBJECT_TSDF_TRUNC_METERS:-0.035}"
export OBJECT_TSDF_DEPTH_TRUNC_METERS="${OBJECT_TSDF_DEPTH_TRUNC_METERS:-4.0}"

log() {
  printf '[run.sh] %s\n' "$*"
}

run_as_root() {
  if [ "$(id -u)" = "0" ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    log "Skipping root command because neither root nor sudo is available: $*"
    return 1
  fi
}

install_system_packages() {
  if [ "${APP_INSTALL_APT}" != "1" ]; then
    log "Skipping apt package install."
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; skipping system package install."
    return 0
  fi

  if [ "$(id -u)" != "0" ] && ! command -v sudo >/dev/null 2>&1; then
    log "No root or sudo access; skipping system package install."
    return 0
  fi

  log "Installing system packages."
  run_as_root apt-get update
  run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0
  run_as_root rm -rf /var/lib/apt/lists/*
}

prepare_cache_dirs() {
  log "Preparing cache directories."
  mkdir -p \
    "${VGGT_CACHE_ROOT}" \
    "${HF_HOME}" \
    "${YOLO_CONFIG_DIR}" \
    "${YOLO_CONFIG_DIR}/Ultralytics" \
    "$(dirname "${ULTRALYTICS_SETTINGS}")"
  chmod -R a+rwX "${VGGT_CACHE_ROOT}" "${YOLO_CONFIG_DIR}" || true
}

backup_non_git_dir() {
  if [ -e "${APP_DIR}" ] && [ ! -d "${APP_DIR}/.git" ]; then
    local backup_dir
    backup_dir="${APP_DIR}.backup.$(date +%Y%m%d%H%M%S)"
    log "Found non-git APP_DIR. Moving it to ${backup_dir}."
    mv "${APP_DIR}" "${backup_dir}"
  fi
}

sync_repo() {
  backup_non_git_dir

  if [ -d "${APP_DIR}/.git" ]; then
    log "Updating repository in ${APP_DIR}."
    git -C "${APP_DIR}" config --global --add safe.directory "${APP_DIR}" >/dev/null 2>&1 || true
    git -C "${APP_DIR}" remote set-url origin "${APP_REPO_URL}"
    git -C "${APP_DIR}" fetch --prune origin "${APP_REPO_REF}"

    case "${APP_UPDATE_MODE}" in
      reset)
        git -C "${APP_DIR}" reset --hard "origin/${APP_REPO_REF}"
        git -C "${APP_DIR}" clean -fd -e runs/ -e .venv/
        ;;
      pull)
        git -C "${APP_DIR}" checkout "${APP_REPO_REF}" 2>/dev/null || git -C "${APP_DIR}" checkout -B "${APP_REPO_REF}" "origin/${APP_REPO_REF}"
        git -C "${APP_DIR}" pull --ff-only origin "${APP_REPO_REF}"
        ;;
      *)
        log "Unknown APP_UPDATE_MODE=${APP_UPDATE_MODE}; use reset or pull."
        exit 2
        ;;
    esac
  else
    log "Cloning ${APP_REPO_URL}#${APP_REPO_REF} into ${APP_DIR}."
    mkdir -p "$(dirname "${APP_DIR}")"
    git clone --depth 1 --branch "${APP_REPO_REF}" "${APP_REPO_URL}" "${APP_DIR}"
  fi
}

install_python_packages() {
  log "Installing Python dependencies."
  if [ -z "${PYTHON_BIN}" ]; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN=python
    else
      PYTHON_BIN=python3
    fi
  fi
  "${PYTHON_BIN}" -m pip install --upgrade pip uv

  cd "${APP_DIR}"
  if [ -n "${APP_INSTALL_EXTRAS}" ]; then
    uv pip install --system -e ".[${APP_INSTALL_EXTRAS}]"
  else
    uv pip install --system -e .
  fi
}

prepare_vggt() {
  if [ "${APP_PREPARE_VGGT}" = "1" ]; then
    log "Preparing VGGT repo and Python package."
    VGGT_DOWNLOAD_WEIGHTS="${APP_PREFETCH_VGGT}" vggt-prepare
  elif [ "${APP_PREFETCH_VGGT}" = "1" ]; then
    log "APP_PREFETCH_VGGT=1 requires APP_PREPARE_VGGT=1."
    exit 2
  else
    log "Skipping VGGT preparation. Set APP_PREPARE_VGGT=1 to clone and install VGGT before serving."
  fi
}

start_app() {
  cd "${APP_DIR}"

  if [ "$#" -gt 0 ]; then
    log "Starting custom command: $*"
    exec "$@"
  fi

  if [ -n "${APP_START_COMMAND:-}" ]; then
    log "Starting APP_START_COMMAND: ${APP_START_COMMAND}"
    exec bash -lc "${APP_START_COMMAND}"
  fi

  log "Starting FastAPI on ${APP_HOST}:${APP_PORT}."
  exec uvicorn vggt_lidar_scan.api:app --host "${APP_HOST}" --port "${APP_PORT}"
}

main() {
  install_system_packages
  prepare_cache_dirs
  sync_repo
  install_python_packages
  prepare_vggt
  start_app "$@"
}

main "$@"
