#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This build helper targets Apple Silicon macOS only.' >&2
  exit 1
fi
mkdir -p "$SCRIPT_DIR/artifacts"
SOURCE_HASH="$(shasum -a 256 "$SCRIPT_DIR/native/gather.c" | awk '{print $1}')"
xcrun clang -arch arm64 -O3 -std=c11 -Wall -Wextra -Werror \
  -mmacosx-version-min=15.0 -dynamiclib -fvisibility=hidden \
  -DSOURCE_SHA256="\"$SOURCE_HASH\"" \
  -Wl,-exported_symbol,_mac_gather_bytes -Wl,-exported_symbol,_mac_gather_build_id \
  "$SCRIPT_DIR/native/gather.c" -o "$SCRIPT_DIR/artifacts/libcamera_gather.dylib"
printf 'Built source-bound native gather: %s\n' "$SOURCE_HASH"
