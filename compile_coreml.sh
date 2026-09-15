#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX="${ONNX:-$SCRIPT_DIR/models/big_driving_supercombo.onnx}"
METADATA="${METADATA:-$SCRIPT_DIR/models/big_driving_supercombo_metadata.pkl}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/artifacts/big_driving.mlpackage}"
if [[ ! -x "$SCRIPT_DIR/.coreml-venv/bin/python" ]]; then
  echo "Run ./setup_coreml_env.sh --conversion first." >&2
  exit 1
fi
if [[ ! -f "$ONNX" || ! -f "$METADATA" ]]; then
  echo "Import a trusted source ONNX and matching metadata first; see README.md." >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists; choose a fresh OUTPUT path to preserve it." >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT")"
exec "$SCRIPT_DIR/.coreml-venv/bin/python" "$SCRIPT_DIR/convert_coreml.py" \
  --onnx "$ONNX" --promote-fp32 --output "$OUTPUT"
