#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ ! -x .coreml-venv/bin/python ]]; then
  uv venv --python 3.12 .coreml-venv
fi
REQUIREMENTS=requirements-runtime.txt
if [[ "${1:-}" == "--conversion" ]]; then
  REQUIREMENTS=requirements-conversion.txt
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--conversion]" >&2
  exit 1
fi
uv pip install --python .coreml-venv/bin/python -r "$REQUIREMENTS"
