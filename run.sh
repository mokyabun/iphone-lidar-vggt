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
APP_PREPARE_SPAR3D="${APP_PREPARE_SPAR3D:-auto}"
APP_PREFETCH_SPAR3D="${APP_PREFETCH_SPAR3D:-1}"
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
export SPAR3D_REPO_URL="${SPAR3D_REPO_URL:-https://github.com/Stability-AI/stable-point-aware-3d.git}"
export SPAR3D_REPO_DIR="${SPAR3D_REPO_DIR:-/workspace/cache/spar3d}"
export SPAR3D_VENV="${SPAR3D_VENV:-/workspace/cache/spar3d-venv}"
export SPAR3D_PYTHON="${SPAR3D_PYTHON:-${SPAR3D_VENV}/bin/python}"
export SPAR3D_MODEL_ID="${SPAR3D_MODEL_ID:-stabilityai/stable-point-aware-3d}"
export SPAR3D_LOW_VRAM="${SPAR3D_LOW_VRAM:-0}"
export SPAR3D_TEXTURE_RESOLUTION="${SPAR3D_TEXTURE_RESOLUTION:-1024}"
export GENERATIVE_MESH_TIMEOUT_SECONDS="${GENERATIVE_MESH_TIMEOUT_SECONDS:-1200}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/workspace/cache/ultralytics}"
export ULTRALYTICS_SETTINGS="${ULTRALYTICS_SETTINGS:-/workspace/cache/ultralytics/settings.json}"
export SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-24}"
export SCAN_STRIDE="${SCAN_STRIDE:-4}"
export SCAN_RUN_TSDF="${SCAN_RUN_TSDF:-0}"
export MESH_METHOD="${MESH_METHOD:-printable_metric}"
export VGGT_MAX_IMAGES="${VGGT_MAX_IMAGES:-12}"
export VGGT_PRELOAD="${VGGT_PRELOAD:-1}"
export VGGT_AS_FINAL="${VGGT_AS_FINAL:-0}"
export OBJECT_MASK_BACKEND="${OBJECT_MASK_BACKEND:-sam3_depth}"
export OBJECT_SAM_MODEL="${OBJECT_SAM_MODEL:-sam3.pt}"
export OBJECT_SAM_MAX_FRAMES="${OBJECT_SAM_MAX_FRAMES:-3}"
export OBJECT_CENTER_FRACTION="${OBJECT_CENTER_FRACTION:-0.35}"
export OBJECT_DEPTH_BAND_METERS="${OBJECT_DEPTH_BAND_METERS:-0.35}"
export OBJECT_MIN_MASK_RATIO="${OBJECT_MIN_MASK_RATIO:-0.002}"
export OBJECT_MAX_MASK_RATIO="${OBJECT_MAX_MASK_RATIO:-0.65}"
export OBJECT_REMOVE_DOMINANT_PLANE="${OBJECT_REMOVE_DOMINANT_PLANE:-1}"
export OBJECT_PLANE_DISTANCE_METERS="${OBJECT_PLANE_DISTANCE_METERS:-0.025}"
export OBJECT_PLANE_SAMPLE_STRIDE="${OBJECT_PLANE_SAMPLE_STRIDE:-4}"
export OBJECT_TSDF_VOXEL_METERS="${OBJECT_TSDF_VOXEL_METERS:-0.008}"
export OBJECT_TSDF_TRUNC_METERS="${OBJECT_TSDF_TRUNC_METERS:-0.035}"
export OBJECT_TSDF_DEPTH_TRUNC_METERS="${OBJECT_TSDF_DEPTH_TRUNC_METERS:-4.0}"
export OBJECT_ALPHA_NEIGHBOR_FACTOR="${OBJECT_ALPHA_NEIGHBOR_FACTOR:-8.0}"
export OBJECT_ALPHA_MIN_EXTENT_FRACTION="${OBJECT_ALPHA_MIN_EXTENT_FRACTION:-0.12}"
export OBJECT_ALPHA_MAX_EXTENT_FRACTION="${OBJECT_ALPHA_MAX_EXTENT_FRACTION:-0.5}"
export OBJECT_PRINTABLE_SMOOTH_ITERATIONS="${OBJECT_PRINTABLE_SMOOTH_ITERATIONS:-2}"
export OBJECT_PRINTABLE_SUBDIVISION_ITERATIONS="${OBJECT_PRINTABLE_SUBDIVISION_ITERATIONS:-1}"
export OBJECT_PRINTABLE_TRIM_PERCENT="${OBJECT_PRINTABLE_TRIM_PERCENT:-1.0}"
export OBJECT_TEMPORAL_FILTER="${OBJECT_TEMPORAL_FILTER:-1}"
export OBJECT_TEMPORAL_VOXEL_METERS="${OBJECT_TEMPORAL_VOXEL_METERS:-0.006}"
export OBJECT_TEMPORAL_MIN_FRAMES="${OBJECT_TEMPORAL_MIN_FRAMES:-2}"
export POINT_CLOUD_CLEANUP="${POINT_CLOUD_CLEANUP:-1}"
export OBJECT_POINT_CLOUD_VOXEL_METERS="${OBJECT_POINT_CLOUD_VOXEL_METERS:-0.002}"
export SCENE_POINT_CLOUD_VOXEL_METERS="${SCENE_POINT_CLOUD_VOXEL_METERS:-0.005}"
export POINT_CLOUD_OUTLIER_NEIGHBORS="${POINT_CLOUD_OUTLIER_NEIGHBORS:-20}"
export POINT_CLOUD_OUTLIER_STD_RATIO="${POINT_CLOUD_OUTLIER_STD_RATIO:-1.5}"
export POINT_CLOUD_RADIUS_FACTOR="${POINT_CLOUD_RADIUS_FACTOR:-4.0}"
export POINT_CLOUD_RADIUS_MIN_NEIGHBORS="${POINT_CLOUD_RADIUS_MIN_NEIGHBORS:-3}"
export OBJECT_POISSON_DEPTH="${OBJECT_POISSON_DEPTH:-8}"
export OBJECT_POISSON_DENSITY_TRIM="${OBJECT_POISSON_DENSITY_TRIM:-0.03}"

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
    "${SPAR3D_REPO_DIR}" \
    "${SPAR3D_VENV}" \
    "${YOLO_CONFIG_DIR}" \
    "${YOLO_CONFIG_DIR}/Ultralytics" \
    "$(dirname "${ULTRALYTICS_SETTINGS}")"
  chmod -R a+rwX "${VGGT_CACHE_ROOT}" "${SPAR3D_REPO_DIR}" "${SPAR3D_VENV}" "${YOLO_CONFIG_DIR}" || true
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

prepare_spar3d() {
  local prepare="${APP_PREPARE_SPAR3D}"
  local revision
  local installed_revision=""
  if [ "${prepare}" = "auto" ]; then
    if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
      prepare=1
    elif [ -x "${SPAR3D_PYTHON}" ] && [ -d "${SPAR3D_REPO_DIR}/.git" ]; then
      log "Using cached SPAR3D installation."
      return 0
    else
      prepare=0
    fi
  fi
  if [ "${prepare}" != "1" ]; then
    log "Skipping SPAR3D preparation. Set HF_TOKEN and APP_PREPARE_SPAR3D=1 to enable Pretty Mesh."
    return 0
  fi

  if [ -d "${SPAR3D_REPO_DIR}/.git" ]; then
    log "Updating SPAR3D repository."
    git -C "${SPAR3D_REPO_DIR}" fetch --depth 1 origin main
    git -C "${SPAR3D_REPO_DIR}" reset --hard origin/main
  else
    log "Cloning SPAR3D repository."
    rm -rf "${SPAR3D_REPO_DIR}"
    git clone --depth 1 "${SPAR3D_REPO_URL}" "${SPAR3D_REPO_DIR}"
  fi

  revision="$(git -C "${SPAR3D_REPO_DIR}" rev-parse HEAD)"
  if [ -f "${SPAR3D_VENV}/.spar3d-revision" ]; then
    installed_revision="$(<"${SPAR3D_VENV}/.spar3d-revision")"
  fi
  if [ ! -x "${SPAR3D_PYTHON}" ] || [ "${installed_revision}" != "${revision}" ]; then
    log "Preparing isolated SPAR3D Python environment."
    uv venv --clear --python "${PYTHON_BIN}" --system-site-packages "${SPAR3D_VENV}"
    uv pip install --python "${SPAR3D_PYTHON}" "setuptools==69.5.1" wheel
    (
      cd "${SPAR3D_REPO_DIR}"
      uv pip install --python "${SPAR3D_PYTHON}" -r requirements.txt
    )
    printf '%s\n' "${revision}" > "${SPAR3D_VENV}/.spar3d-revision"
  else
    log "SPAR3D dependencies are already current."
  fi

  if [ "${APP_PREFETCH_SPAR3D}" = "1" ]; then
    log "Prefetching gated SPAR3D weights."
    "${SPAR3D_PYTHON}" -c \
      "from huggingface_hub import snapshot_download; snapshot_download('${SPAR3D_MODEL_ID}')"
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
  prepare_spar3d
  start_app "$@"
}

main "$@"
