ensure_reconviagen_env() {
  local requirements_file="${SERVER_DIR}/workers/reconviagen/requirements.txt"
  local venv_args=()
  if is_enabled "${RECONVIAGEN_USE_SYSTEM_TORCH:-0}"; then
    venv_args+=(--system-site-packages)
  fi

  if ! env_exists "${RECONVIAGEN_ENV_DIR}"; then
    LOG_PREFIX="worker-reconviagen-env" log "Creating uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
    mkdir -p "$(dirname "${RECONVIAGEN_ENV_DIR}")"
    uv venv --python "${RECONVIAGEN_PYTHON_VERSION}" "${venv_args[@]}" "${RECONVIAGEN_ENV_DIR}"
    install_reconviagen_requirements "${requirements_file}"
    configure_ccache
    return 0
  fi

  if should_update_envs; then
    LOG_PREFIX="worker-reconviagen-env" log "Updating uv env ${RECONVIAGEN_ENV_NAME} at ${RECONVIAGEN_ENV_DIR}."
    install_reconviagen_requirements "${requirements_file}"
  else
    LOG_PREFIX="worker-reconviagen-env" log "Using existing uv env ${RECONVIAGEN_ENV_NAME}. Set APP_UPDATE_ENVS=1 to update dependencies."
  fi
  configure_ccache
}

install_reconviagen_requirements() {
  local requirements_file="$1"
  local python_bin
  python_bin="$(venv_python "${RECONVIAGEN_ENV_DIR}")"

  if is_enabled "${RECONVIAGEN_USE_SYSTEM_TORCH:-0}"; then
    LOG_PREFIX="worker-reconviagen-env" log "Using image-provided torch packages via system site-packages."
    uv pip install --python "${python_bin}" \
      --excludes <(printf 'torch\ntorchvision\ntorchaudio\n') \
      -r <(
      grep -Ev '^(--extra-index-url[[:space:]]+https://download\.pytorch\.org/whl/|torch==|torchvision==|torchaudio==)' "${requirements_file}"
    )
  else
    uv pip install --python "${python_bin}" -r "${requirements_file}"
    if should_update_envs; then
      LOG_PREFIX="worker-reconviagen-env" log "Force-reinstalling PyTorch CUDA packages for ${RECONVIAGEN_ENV_NAME}."
      uv pip install --python "${python_bin}" \
        --reinstall-package torch \
        --reinstall-package torchvision \
        --reinstall-package torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
    fi
  fi
}

verify_reconviagen_torch() {
  LOG_PREFIX="prepare-reconviagen" log "Checking PyTorch in ${RECONVIAGEN_ENV_NAME}."
  venv_run "${RECONVIAGEN_ENV_DIR}" env RECONVIAGEN_REQUIRE_CUDA="${RECONVIAGEN_REQUIRE_CUDA:-1}" python - <<'PY'
import os
import shutil
import subprocess

import torch

require_cuda = os.environ.get("RECONVIAGEN_REQUIRE_CUDA", "1").lower() not in {"0", "false", "no", "off"}
print(
    "[prepare-reconviagen] "
    f"torch={torch.__version__} cuda_build={torch.version.cuda} available={torch.cuda.is_available()} "
    f"device_count={torch.cuda.device_count()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
)
if not torch.__version__.startswith("2.4."):
    raise SystemExit("[prepare-reconviagen] ReconViaGen v0.5 CUDA extensions require PyTorch 2.4.x.")
if require_cuda and not torch.cuda.is_available():
    if torch.version.cuda is None:
        print("[prepare-reconviagen] diagnostic: installed torch build is CPU-only.")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index,name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(f"[prepare-reconviagen] nvidia-smi returncode={result.returncode} output={(result.stdout or result.stderr).strip() or '<empty>'}")
    else:
        print("[prepare-reconviagen] nvidia-smi not found in PATH.")
    raise SystemExit(
        "[prepare-reconviagen] CUDA is required but unavailable. "
        "If this venv was created with CPU-only torch, rerun with APP_UPDATE_ENVS=1; "
        "otherwise check GPU/container driver visibility."
    )
PY
}
