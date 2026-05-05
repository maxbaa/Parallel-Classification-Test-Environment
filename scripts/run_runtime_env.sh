#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <core|rapids|thunder> <config-yaml>" >&2
  exit 1
fi

RUNTIME="$1"
CONFIG="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${VENV_ROOT:-$ROOT_DIR/.venvs}"
PYTHON_BIN="$VENV_ROOT/$RUNTIME/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Runtime environment not found: $PYTHON_BIN" >&2
  echo "Run: bash scripts/setup_runtime_venvs.sh" >&2
  exit 1
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" scripts/run_runtime_suite.py --runtime "$RUNTIME" --config "$CONFIG" --python "$PYTHON_BIN"
