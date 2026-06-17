reconviagen_cuda_stamp_file() {
  echo "${RECONVIAGEN_ENV_DIR}/.reconviagen-cuda-stamp"
}

reconviagen_cuda_runtime_stamp() {
  {
    printf 'repo=%s\n' "$(reconviagen_repo_revision)"
    printf 'cuda_flags=%s\n' "${RECONVIAGEN_CUDA_FLAGS}"
    printf 'postprocessors=%s\n' "${RECONVIAGEN_INSTALL_POSTPROCESSORS}"
  } | cksum | awk '{print $1}'
}

show_reconviagen_ccache_stats() {
  if command -v ccache >/dev/null 2>&1; then
    LOG_PREFIX="prepare-reconviagen" log "ccache stats:"
    ccache --show-stats || true
  fi
}

build_reconviagen_cuda_extensions() {
  local stamp_file expected_stamp installed_stamp
  stamp_file="$(reconviagen_cuda_stamp_file)"
  expected_stamp="$(reconviagen_cuda_runtime_stamp)"
  installed_stamp=""
  if [ -f "${stamp_file}" ]; then
    installed_stamp="$(cat "${stamp_file}")"
  fi
  if [ "${RECONVIAGEN_REFRESH}" != "1" ] && [ "${installed_stamp}" = "${expected_stamp}" ]; then
    LOG_PREFIX="prepare-reconviagen" log "ReconViaGen CUDA extensions are already current."
    show_reconviagen_ccache_stats
    return 0
  fi

  LOG_PREFIX="prepare-reconviagen" log "Building ReconViaGen CUDA extensions. This can take a while on a fresh pod."
  rm -rf /tmp/extensions
  RECONVIAGEN_ENV_PYTHON="$(venv_python "${RECONVIAGEN_ENV_DIR}")" \
    venv_run "${RECONVIAGEN_ENV_DIR}" bash -lc '
      pip() {
        if [ "${1:-}" = "install" ]; then
          shift
          uv pip install --python "${RECONVIAGEN_ENV_PYTHON}" "$@"
        else
          command pip "$@"
        fi
      }
      cd "${RECONVIAGEN_REPO_DIR}" && . ./setup.sh ${RECONVIAGEN_CUDA_FLAGS}
    '

  if [ "${RECONVIAGEN_INSTALL_POSTPROCESSORS}" = "1" ]; then
    optional_pip_install_if_missing cumesh "${RECONVIAGEN_CUMESH_URL}" --no-build-isolation --no-deps
    optional_pip_install_if_missing flex_gemm "${RECONVIAGEN_FLEX_GEMM_URL}" --no-build-isolation --no-deps
    if [ -d "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" ]; then
      optional_pip_install_if_missing o_voxel "${RECONVIAGEN_REPO_DIR}/wheels/TRELLIS.2/o-voxel" --no-build-isolation --no-deps
    fi
    patch_flex_gemm_triton_autotuner
  else
    LOG_PREFIX="prepare-reconviagen" log "Skipping optional postprocessors. Set RECONVIAGEN_INSTALL_POSTPROCESSORS=1 to try CuMesh/FlexGEMM/o-voxel."
  fi

  mkdir -p "$(dirname "${stamp_file}")"
  printf '%s\n' "${expected_stamp}" > "${stamp_file}"
  show_reconviagen_ccache_stats
}
