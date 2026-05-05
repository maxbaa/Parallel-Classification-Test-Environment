#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config-yaml>" >&2
  exit 1
fi

CONFIG="$1"

docker compose run --rm --gpus all core \
  python scripts/run_runtime_suite.py --runtime core --config "$CONFIG"

docker compose run --rm --gpus all rapids \
  python scripts/run_runtime_suite.py --runtime rapids --config "$CONFIG"

docker compose run --rm --gpus all thunder \
  python scripts/run_runtime_suite.py --runtime thunder --config "$CONFIG"
