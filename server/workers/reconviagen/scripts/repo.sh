sync_reconviagen_repo() {
  mkdir -p "$(dirname "${RECONVIAGEN_REPO_DIR}")"
  if [ -d "${RECONVIAGEN_REPO_DIR}/.git" ]; then
    LOG_PREFIX="prepare-reconviagen" log "Updating ReconViaGen ${RECONVIAGEN_REPO_REF}."
    git -C "${RECONVIAGEN_REPO_DIR}" fetch --depth 1 origin "${RECONVIAGEN_REPO_REF}"
    git -C "${RECONVIAGEN_REPO_DIR}" reset --hard FETCH_HEAD
    git -C "${RECONVIAGEN_REPO_DIR}" submodule update --init --recursive
  else
    LOG_PREFIX="prepare-reconviagen" log "Cloning ReconViaGen ${RECONVIAGEN_REPO_REF}."
    rm -rf "${RECONVIAGEN_REPO_DIR}"
    git clone --recursive --depth 1 --branch "${RECONVIAGEN_REPO_REF}" \
      "${RECONVIAGEN_REPO_URL}" "${RECONVIAGEN_REPO_DIR}"
  fi
}

reconviagen_repo_revision() {
  git -C "${RECONVIAGEN_REPO_DIR}" rev-parse HEAD
}
