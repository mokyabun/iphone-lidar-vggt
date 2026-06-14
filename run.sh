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
APP_PREPARE_VGGT="${APP_PREPARE_VGGT:-0}"
APP_PREFETCH_VGGT="${APP_PREFETCH_VGGT:-0}"
APP_PREPARE_RECONVIAGEN="${APP_PREPARE_RECONVIAGEN:-1}"
APP_PREFETCH_RECONVIAGEN="${APP_PREFETCH_RECONVIAGEN:-1}"
RECONVIAGEN_PRELOAD="${RECONVIAGEN_PRELOAD:-1}"
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
export RECONVIAGEN_REPO_URL="${RECONVIAGEN_REPO_URL:-https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen.git}"
export RECONVIAGEN_REPO_REF="${RECONVIAGEN_REPO_REF:-v0.5}"
export RECONVIAGEN_REPO_DIR="${RECONVIAGEN_REPO_DIR:-/workspace/cache/ReconViaGen}"
export RECONVIAGEN_ENV="${RECONVIAGEN_ENV:-/workspace/cache/reconviagen-v05-env}"
export RECONVIAGEN_PYTHON="${RECONVIAGEN_PYTHON:-${RECONVIAGEN_ENV}/bin/python}"
export RECONVIAGEN_WORKER_PORT="${RECONVIAGEN_WORKER_PORT:-8011}"
export RECONVIAGEN_MAX_IMAGES="${RECONVIAGEN_MAX_IMAGES:-6}"
export RECONVIAGEN_PIPELINE_TYPE="${RECONVIAGEN_PIPELINE_TYPE:-1024_cascade}"
export RECONVIAGEN_SS_SOURCE="${RECONVIAGEN_SS_SOURCE:-mesh}"
export RECONVIAGEN_LOW_VRAM="${RECONVIAGEN_LOW_VRAM:-1}"
export RECONVIAGEN_DECIMATION_TARGET="${RECONVIAGEN_DECIMATION_TARGET:-500000}"
export RECONVIAGEN_TEXTURE_SIZE="${RECONVIAGEN_TEXTURE_SIZE:-2048}"
export RECONVIAGEN_TIMEOUT_SECONDS="${RECONVIAGEN_TIMEOUT_SECONDS:-2400}"
export RECONVIAGEN_WORKER_ERROR="${RECONVIAGEN_WORKER_ERROR:-/workspace/cache/reconviagen-worker.error}"
export MICROMAMBA_BIN="${MICROMAMBA_BIN:-/workspace/cache/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace/cache/micromamba}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/workspace/cache/ultralytics}"
export ULTRALYTICS_SETTINGS="${ULTRALYTICS_SETTINGS:-/workspace/cache/ultralytics/settings.json}"
export SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-24}"
export SCAN_STRIDE="${SCAN_STRIDE:-4}"
export SCAN_RUN_TSDF="${SCAN_RUN_TSDF:-0}"
export MESH_METHOD="${MESH_METHOD:-printable_metric}"
export VGGT_MAX_IMAGES="${VGGT_MAX_IMAGES:-12}"
export VGGT_PRELOAD="${VGGT_PRELOAD:-0}"
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
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    ninja-build
  run_as_root rm -rf /var/lib/apt/lists/*
}

prepare_cache_dirs() {
  log "Preparing cache directories."
  mkdir -p \
    "${VGGT_CACHE_ROOT}" \
    "${HF_HOME}" \
    "${RECONVIAGEN_REPO_DIR}" \
    "${RECONVIAGEN_ENV}" \
    "$(dirname "${MICROMAMBA_BIN}")" \
    "${MAMBA_ROOT_PREFIX}" \
    "${YOLO_CONFIG_DIR}" \
    "${YOLO_CONFIG_DIR}/Ultralytics" \
    "$(dirname "${ULTRALYTICS_SETTINGS}")"
  chmod -R a+rwX "${VGGT_CACHE_ROOT}" "${RECONVIAGEN_REPO_DIR}" "${RECONVIAGEN_ENV}" \
    "${MAMBA_ROOT_PREFIX}" "${YOLO_CONFIG_DIR}" || true
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

prepare_reconviagen() {
  if [ "${APP_PREPARE_RECONVIAGEN}" != "1" ]; then
    log "Skipping ReconViaGen preparation."
    return 0
  fi

  if [ ! -x "${MICROMAMBA_BIN}" ]; then
    log "Installing micromamba."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$(dirname "${MICROMAMBA_BIN}")" --strip-components=1 bin/micromamba
    chmod +x "${MICROMAMBA_BIN}"
  fi

  if [ -d "${RECONVIAGEN_REPO_DIR}/.git" ]; then
    log "Updating ReconViaGen ${RECONVIAGEN_REPO_REF}."
    git -C "${RECONVIAGEN_REPO_DIR}" fetch --depth 1 origin "${RECONVIAGEN_REPO_REF}"
    git -C "${RECONVIAGEN_REPO_DIR}" reset --hard FETCH_HEAD
    git -C "${RECONVIAGEN_REPO_DIR}" submodule update --init --recursive
  else
    log "Cloning ReconViaGen ${RECONVIAGEN_REPO_REF}."
    rm -rf "${RECONVIAGEN_REPO_DIR}"
    git clone --recursive --depth 1 --branch "${RECONVIAGEN_REPO_REF}" \
      "${RECONVIAGEN_REPO_URL}" "${RECONVIAGEN_REPO_DIR}"
  fi

  local revision
  local installed_revision=""
  revision="$(git -C "${RECONVIAGEN_REPO_DIR}" rev-parse HEAD)"
  if [ -f "${RECONVIAGEN_ENV}/.reconviagen-revision" ]; then
    installed_revision="$(<"${RECONVIAGEN_ENV}/.reconviagen-revision")"
  fi
  if [ ! -x "${RECONVIAGEN_PYTHON}" ] || [ "${installed_revision}" != "${revision}" ]; then
    log "Creating the isolated ReconViaGen environment."
    rm -rf "${RECONVIAGEN_ENV}"
    "${MICROMAMBA_BIN}" create -y -p "${RECONVIAGEN_ENV}" \
      python=3.10 pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 \
      -c pytorch -c nvidia -c conda-forge
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" pip install \
      pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
      rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph lpips \
      kornia==0.8.2 huggingface_hub==0.36.2 transformers==4.57.1 \
      zstandard rtree fast-simplification
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" pip install \
      git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    rm -rf /tmp/extensions
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" bash -c \
      "cd '${RECONVIAGEN_REPO_DIR}' && . ./setup.sh --xformers --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrec --spconv --kaolin --nvdiffrast"
    printf '%s\n' "${revision}" > "${RECONVIAGEN_ENV}/.reconviagen-revision"
  else
    log "ReconViaGen environment is already current."
  fi

  if ! "${RECONVIAGEN_PYTHON}" -c "import o_voxel" >/dev/null 2>&1; then
    log "Installing the nested ReconViaGen o-voxel extension."
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" pip install \
      "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation
  fi

  if [ "${APP_PREFETCH_RECONVIAGEN}" = "1" ]; then
    log "Prefetching ReconViaGen and TRELLIS.2 weights."
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" python -c \
      "from huggingface_hub import snapshot_download; snapshot_download('Stable-X/trellis-vggt-v0-2'); snapshot_download('microsoft/TRELLIS.2-4B')"
  fi
}

start_reconviagen_worker() {
  if [ "${RECONVIAGEN_PRELOAD}" != "1" ] || [ ! -x "${RECONVIAGEN_PYTHON}" ]; then
    return 0
  fi
  export RECONVIAGEN_WORKER_URL="http://127.0.0.1:${RECONVIAGEN_WORKER_PORT}"
  rm -f "${RECONVIAGEN_WORKER_ERROR}"
  log "Starting ReconViaGen worker on ${RECONVIAGEN_WORKER_URL}."
  (
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" \
      "${RECONVIAGEN_PYTHON}" "${APP_DIR}/backend/vggt_lidar_scan/reconviagen_worker.py" \
      --host 127.0.0.1 --port "${RECONVIAGEN_WORKER_PORT}" \
      || printf 'worker exited with status %s\n' "$?" > "${RECONVIAGEN_WORKER_ERROR}"
  ) &
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
  prepare_reconviagen
  start_reconviagen_worker
  start_app "$@"
}

main "$@"
