#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${JLENS_VENV_DIR:-${PROJECT_DIR}/.venv}"
CONFIG_PATH="${JLENS_CONFIG:-${PROJECT_DIR}/configs/minimal.json}"
OUTPUT_DIR="${JLENS_OUTPUT_DIR:-${PROJECT_DIR}/outputs/minimal}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing virtual environment. Run ${PROJECT_DIR}/scripts/setup.sh first." >&2
  exit 2
fi

# 一个命令完成预检、拟合、评测和哈希清单；重复运行会原子地更新同名产物。
"${VENV_DIR}/bin/python" -m jlens_qwen reproduce \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${JLENS_DEVICE:-auto}"

echo "Result: ${OUTPUT_DIR}/evaluation.md"
