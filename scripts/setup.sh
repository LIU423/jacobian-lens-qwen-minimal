#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${JLENS_VENV_DIR:-${PROJECT_DIR}/.venv}"

# 固定主依赖版本；模型权重本身由 configs/minimal.json 中的 commit hash 固定。
"${PYTHON_BIN}" -c 'import sys; assert (3, 9) <= sys.version_info[:2] < (3, 13), "Python 3.9-3.12 is required"'
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade \
  pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
"${VENV_DIR}/bin/python" -m pip install --requirement "${PROJECT_DIR}/requirements.lock"
"${VENV_DIR}/bin/python" -m pip install --no-deps --no-build-isolation --editable "${PROJECT_DIR}"

"${VENV_DIR}/bin/python" -m pytest "${PROJECT_DIR}/tests"
echo "Setup complete. Run: ${PROJECT_DIR}/scripts/run_minimal.sh"
