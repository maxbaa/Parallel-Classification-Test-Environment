#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config-yaml>" >&2
  exit 1
fi

CONFIG="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT_DIR/scripts/run_runtime_env.sh" core "$CONFIG"
bash "$ROOT_DIR/scripts/run_runtime_env.sh" rapids "$CONFIG"
bash "$ROOT_DIR/scripts/run_runtime_env.sh" thunder "$CONFIG"
