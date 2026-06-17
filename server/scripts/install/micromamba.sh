#!/usr/bin/env bash
set -Eeuo pipefail

if command -v micromamba >/dev/null 2>&1; then
  exit 0
fi

printf '[install-micromamba] Installing micromamba.\n'
mkdir -p /usr/local/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
