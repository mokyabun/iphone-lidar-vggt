#!/usr/bin/env bash
set -Eeuo pipefail

RECONVIAGEN_STARTUP_BLOCKED=0

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

default_reconviagen_prepare() {
  case "$(uname -s)" in
    Linux) printf '1' ;;
    *) printf '0' ;;
  esac
}

micromamba_platform() {
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) printf 'linux-64' ;;
    Linux-aarch64 | Linux-arm64) printf 'linux-aarch64' ;;
    Darwin-x86_64) printf 'osx-64' ;;
    Darwin-arm64) printf 'osx-arm64' ;;
    *) die "Unsupported micromamba platform: $(uname -s)-$(uname -m)" ;;
  esac
}

init_defaults() {
  set_default APP_PERSIST_ROOT "/workspace"
  set_default APP_CACHE_ROOT "${APP_PERSIST_ROOT}/cache"
  set_default APP_MODEL_ROOT "${APP_CACHE_ROOT}/vggt-lidar"
  set_default APP_REPO_URL "https://github.com/mokyabun/iphone-lidar-vggt.git"
  set_default APP_REPO_REF "main"
  set_default APP_DIR "${APP_PERSIST_ROOT}/iphone-lidar-vggt"
  set_default APP_HOST "0.0.0.0"
  set_default APP_PORT "8000"
  set_default APP_UPDATE_MODE "reset"
  set_default APP_INSTALL_EXTRAS "reconstruction,vggt,segmentation"
  set_default APP_INSTALL_APT "1"
  set_default APP_PREPARE_VGGT "0"
  set_default APP_PREFETCH_VGGT "0"
  set_default APP_PREPARE_RECONVIAGEN "$(default_reconviagen_prepare)"
  set_default APP_PREFETCH_RECONVIAGEN "1"
  set_default PYTHON_BIN ""

  set_default PYTHONUNBUFFERED "1"
  set_default PIP_NO_CACHE_DIR "1"
  set_default UV_LINK_MODE "copy"

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
  set_default RECONVIAGEN_MAX_IMAGES "6"
  set_default RECONVIAGEN_PIPELINE_TYPE "1024_cascade"
  set_default RECONVIAGEN_SS_SOURCE "mesh"
  set_default RECONVIAGEN_LOW_VRAM "1"
  set_default RECONVIAGEN_DECIMATION_TARGET "500000"
  set_default RECONVIAGEN_TEXTURE_SIZE "2048"
  set_default RECONVIAGEN_TIMEOUT_SECONDS "2400"
  set_default RECONVIAGEN_WORKER_ERROR "${APP_CACHE_ROOT}/reconviagen-worker.error"
  set_default RECONVIAGEN_WORKER_LOG "${APP_CACHE_ROOT}/reconviagen-worker.log"
  set_default RECONVIAGEN_WORKER_RESTART "1"
  set_default RECONVIAGEN_WORKER_RESTART_DELAY_SECONDS "20"
  set_default RECONVIAGEN_DINO_MODEL "facebook/dinov3-vitl16-pretrain-lvd1689m"

  set_default MICROMAMBA_BIN "${APP_CACHE_ROOT}/bin/micromamba"
  if [ -z "${MICROMAMBA_PLATFORM+x}" ]; then
    MICROMAMBA_PLATFORM="$(micromamba_platform)"
  fi
  export MICROMAMBA_PLATFORM
  set_default MAMBA_ROOT_PREFIX "${APP_CACHE_ROOT}/micromamba"

  set_default YOLO_CONFIG_DIR "${APP_CACHE_ROOT}/ultralytics"
  set_default ULTRALYTICS_SETTINGS "${YOLO_CONFIG_DIR}/settings.json"
  set_default ULTRALYTICS_WEIGHTS_DIR "${APP_MODEL_ROOT}/ultralytics"
  set_default ULTRALYTICS_RUNS_DIR "${APP_CACHE_ROOT}/ultralytics/runs"

  set_default SCAN_MAX_FRAMES "24"
  set_default SCAN_STRIDE "4"
  set_default SCAN_FRAME_WORKERS "4"
  set_default SCAN_RUN_TSDF "0"
  set_default MESH_METHOD "printable_metric"

  set_default VGGT_MAX_IMAGES "12"
  set_default VGGT_PRELOAD "0"
  set_default VGGT_AS_FINAL "0"
  set_default VGGT_EMPTY_CACHE_AFTER_RUN "0"

  set_default OBJECT_MASK_BACKEND "sam3_depth"
  set_default OBJECT_SAM_MODEL "sam3.pt"
  set_default OBJECT_SAM_MAX_FRAMES "3"
  set_default OBJECT_MASK_PROPAGATION "1"
  set_default OBJECT_PROPAGATION_ANCHORS "2"
  set_default OBJECT_PROPAGATION_DEPTH_TOLERANCE_METERS "0.08"
  set_default OBJECT_CENTER_FRACTION "0.35"
  set_default OBJECT_DEPTH_BAND_METERS "0.35"
  set_default OBJECT_MIN_MASK_RATIO "0.002"
  set_default OBJECT_MAX_MASK_RATIO "0.65"
  set_default OBJECT_REMOVE_DOMINANT_PLANE "1"
  set_default OBJECT_PLANE_DISTANCE_METERS "0.025"
  set_default OBJECT_PLANE_SAMPLE_STRIDE "4"

  set_default OBJECT_TSDF_VOXEL_METERS "0.004"
  set_default OBJECT_TSDF_TRUNC_METERS "0.02"
  set_default OBJECT_TSDF_DEPTH_TRUNC_METERS "4.0"
  set_default OBJECT_MESH_SMOOTH_ITERATIONS "3"
  set_default OBJECT_MESH_SMOOTH_METHOD "taubin"

  set_default OBJECT_ALPHA_NEIGHBOR_FACTOR "8.0"
  set_default OBJECT_ALPHA_MIN_EXTENT_FRACTION "0.12"
  set_default OBJECT_ALPHA_MAX_EXTENT_FRACTION "0.5"
  set_default OBJECT_PRINTABLE_SMOOTH_ITERATIONS "2"
  set_default OBJECT_PRINTABLE_SUBDIVISION_ITERATIONS "1"
  set_default OBJECT_PRINTABLE_TRIM_PERCENT "1.0"

  set_default OBJECT_TEMPORAL_FILTER "1"
  set_default OBJECT_TEMPORAL_VOXEL_METERS "0.006"
  set_default OBJECT_TEMPORAL_MIN_FRAMES "2"

  set_default POINT_CLOUD_CLEANUP "1"
  set_default OBJECT_POINT_CLOUD_VOXEL_METERS "0"
  set_default SCENE_POINT_CLOUD_VOXEL_METERS "0"
  set_default POINT_CLOUD_OUTLIER_NEIGHBORS "20"
  set_default POINT_CLOUD_OUTLIER_STD_RATIO "1.5"
  set_default POINT_CLOUD_RADIUS_FACTOR "4.0"
  set_default POINT_CLOUD_RADIUS_MIN_NEIGHBORS "3"

  set_default OBJECT_POISSON_DEPTH "8"
  set_default OBJECT_POISSON_DENSITY_TRIM "0.03"
  set_default AI_ICP_ITERATIONS "20"
  set_default AI_ICP_MAX_DISTANCE_METERS "0.03"
  set_default AI_PRINT_VOXEL_REPAIR "1"
  set_default AI_PRINT_VOXEL_METERS "0.0015"
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
    "$(dirname "${MICROMAMBA_BIN}")"
    "${MAMBA_ROOT_PREFIX}"
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

install_python_packages() {
  log "Installing Python dependencies."
  resolve_python_bin
  "${PYTHON_BIN}" -m pip install --upgrade pip uv

  cd "${APP_DIR}"
  if [ -n "${APP_INSTALL_EXTRAS}" ]; then
    uv pip install --system -e ".[${APP_INSTALL_EXTRAS}]"
  else
    uv pip install --system -e .
  fi
}

configure_ultralytics() {
  "${PYTHON_BIN}" -c \
    "from ultralytics import settings; settings.update({'weights_dir': '${ULTRALYTICS_WEIGHTS_DIR}', 'runs_dir': '${ULTRALYTICS_RUNS_DIR}'})" \
    >/dev/null 2>&1 || true
}

prepare_vggt() {
  if is_enabled "${APP_PREPARE_VGGT}"; then
    log "Preparing VGGT repo and Python package."
    VGGT_DOWNLOAD_WEIGHTS="${APP_PREFETCH_VGGT}" vggt-prepare
  elif is_enabled "${APP_PREFETCH_VGGT}"; then
    die "APP_PREFETCH_VGGT=1 requires APP_PREPARE_VGGT=1."
  else
    log "Skipping VGGT preparation. Set APP_PREPARE_VGGT=1 to clone and install VGGT before serving."
  fi
}

install_micromamba() {
  if [ -x "${MICROMAMBA_BIN}" ]; then
    return 0
  fi

  log "Installing micromamba (${MICROMAMBA_PLATFORM})."
  curl -Ls "https://micro.mamba.pm/api/micromamba/${MICROMAMBA_PLATFORM}/latest" \
    | tar -xj -C "$(dirname "${MICROMAMBA_BIN}")" --strip-components=1 bin/micromamba
  chmod +x "${MICROMAMBA_BIN}"
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

  log "Creating the isolated ReconViaGen environment."
  rm -rf "${RECONVIAGEN_ENV}"
  "${MICROMAMBA_BIN}" create -y -p "${RECONVIAGEN_ENV}" \
    python=3.10 pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -c conda-forge
  "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" pip install \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
    scikit-image rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph lpips \
    kornia==0.8.2 timm==1.0.22 huggingface_hub==0.36.2 transformers==4.57.1 \
    zstandard rtree fast-simplification
  "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" pip install \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
  rm -rf /tmp/extensions
  "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" bash -c \
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
    "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" pip install "$@"
  fi
}

ensure_reconviagen_runtime_patches() {
  ensure_reconviagen_runtime_dependency \
    "import o_voxel" \
    "Installing the nested ReconViaGen o-voxel extension." \
    "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation

  ensure_reconviagen_runtime_dependency \
    "import timm" \
    "Installing the ReconViaGen timm runtime dependency." \
    "timm==1.0.22"

  ensure_reconviagen_runtime_dependency \
    "import scipy, skimage" \
    "Installing ReconViaGen print-mesh repair dependencies." \
    "scipy>=1.13" "scikit-image>=0.24"
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
  if ! is_enabled "${APP_PREFETCH_RECONVIAGEN}"; then
    return 0
  fi

  log "Prefetching ReconViaGen, TRELLIS.2, and DINOv3 weights."
  "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Stable-X/trellis-vggt-v0-2'); snapshot_download('microsoft/TRELLIS.2-4B'); snapshot_download('${RECONVIAGEN_DINO_MODEL}')"
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

  install_micromamba
  sync_reconviagen_repo
  ensure_reconviagen_env
  ensure_reconviagen_runtime_patches
  verify_reconviagen_hf_access || return 0
  prefetch_reconviagen_weights
}

start_reconviagen_worker() {
  if ! is_enabled "${RECONVIAGEN_PRELOAD}" || [ ! -x "${RECONVIAGEN_PYTHON}" ]; then
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
      "${MICROMAMBA_BIN}" run -p "${RECONVIAGEN_ENV}" \
        "${RECONVIAGEN_PYTHON}" "${APP_DIR}/backend/vggt_lidar_scan/reconviagen_worker.py" \
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
  init_defaults
  install_system_packages
  prepare_cache_dirs
  sync_repo
  install_python_packages
  configure_ultralytics
  prepare_vggt
  prepare_reconviagen
  start_reconviagen_worker
  start_app "$@"
}

main "$@"
