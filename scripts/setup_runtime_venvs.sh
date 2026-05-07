#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_ROOT="${VENV_ROOT:-$ROOT_DIR/.venvs}"

mkdir -p "$VENV_ROOT"

create_venv() {
  local name="$1"
  local env_dir="$VENV_ROOT/$name"
  if [[ ! -d "$env_dir" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$env_dir"
  fi
  "$env_dir/bin/python" -m pip install --upgrade pip setuptools wheel
  "$env_dir/bin/python" -m pip install -e "$ROOT_DIR"
}

create_venv core
create_venv rapids
create_venv thunder

"$VENV_ROOT/core/bin/python" -m pip install torchgpipe
"$VENV_ROOT/thunder/bin/python" -m pip install -e "$ROOT_DIR"

cat <<EOF
Runtime environments created in:
  $VENV_ROOT/core
  $VENV_ROOT/rapids
  $VENV_ROOT/thunder
EOF
