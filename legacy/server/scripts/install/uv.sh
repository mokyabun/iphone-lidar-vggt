#!/usr/bin/env bash
set -Eeuo pipefail

if command -v uv >/dev/null 2>&1; then
  exit 0
fi

printf '[install-uv] Installing uv.\n'
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-${APP_BIN_DIR:-${APP_CACHE_ROOT:-/workspace/cache}/bin}}"
mkdir -p "${UV_INSTALL_DIR}"
curl -LsSf https://astral.sh/uv/install.sh | sh
