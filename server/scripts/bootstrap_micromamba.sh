#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/lib.sh"

export APP_LOCAL_ROOT="${APP_LOCAL_ROOT:-/opt/iphone-lidar-vggt}"
export APP_BIN_DIR="${APP_BIN_DIR:-${APP_LOCAL_ROOT}/bin}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${APP_LOCAL_ROOT}/mamba}"

bin="$(micromamba_bin)"
if [ -x "${bin}" ]; then
  LOG_PREFIX="bootstrap-micromamba" log "micromamba already present at ${bin} ($("${bin}" --version 2>/dev/null || echo unknown))."
  exit 0
fi

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) platform="linux-64" ;;
  Linux/aarch64 | Linux/arm64) platform="linux-aarch64" ;;
  Darwin/arm64) platform="osx-arm64" ;;
  Darwin/x86_64) platform="osx-64" ;;
  *)
    LOG_PREFIX="bootstrap-micromamba" log "Unsupported platform $(uname -s)/$(uname -m)."
    exit 1
    ;;
esac

LOG_PREFIX="bootstrap-micromamba" log "Installing micromamba for ${platform} into ${APP_BIN_DIR}."
mkdir -p "${APP_BIN_DIR}" "${MAMBA_ROOT_PREFIX}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
# The release tarball stores the binary at bin/micromamba.
curl -Ls "https://micro.mamba.pm/api/micromamba/${platform}/latest" \
  | tar -xj -C "${tmp_dir}" bin/micromamba
mv "${tmp_dir}/bin/micromamba" "${bin}"
chmod +x "${bin}"
LOG_PREFIX="bootstrap-micromamba" log "Installed micromamba: $("${bin}" --version)."
