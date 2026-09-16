# Mac Accelerator

Independent Apple Silicon / Core ML inference benchmark app and server.
Extracted from the sunnypilot M2 experiment; **no sunnypilot checkout, tinygrad,
device firmware or vehicle-control code is required to build or run this project.**

macOS 15+ · arm64 · M2 tested · Research preview 0.3.0

The native app launches a Core ML CPU + Neural Engine server with authenticated
transport and a synthetic client. This is **bench research only**, not a road-use
accelerator, native external GPU driver, or generic arbitrary-model host. Its model
contract remains the Big Model driving-policy protocol from the original experiment.
There is no live-camera hook, device parameter writer or control-output publisher.

## Install

Download the [Mac app preview](https://github.com/Akarow77/mac-accelerator/releases/tag/v0.3.0),
or build from source below. The app is ad-hoc signed, not Apple-notarized. It needs
this project folder, a Python environment and separately supplied model artifacts;
the app ZIP is not a standalone runtime/model installer. Do not disable Gatekeeper
globally; inspect and build the source if preferred.

```bash
git clone https://github.com/Akarow77/mac-accelerator.git
cd mac-accelerator
./setup_coreml_env.sh
```

The setup script uses `uv` to create a separate Python 3.12 `.coreml-venv` and install
the pinned runtime requirements. Model files, environments and logs are gitignored.

## Model preparation

Supply the trusted original Big Model ONNX and its **matching** metadata. These
are not redistributed or automatically downloaded. If you already prepared them
in the original research checkout, import them once (substitute your local paths):

```bash
.coreml-venv/bin/python import_model.py \
  --onnx /path/to/big_driving_supercombo.onnx \
  --metadata /path/to/big_driving_supercombo_metadata.pkl \
  --coreml /path/to/big_driving.mlpackage
```

This makes independent copies, never symlinks, and refuses to overwrite existing
artifacts. It records the source ONNX SHA-256, not proof of conversion fidelity.
Metadata is a Python pickle and can execute code at runtime: **only import trusted
metadata, never files supplied by an unknown server or model download.** An ONNX
Git LFS pointer is not a model. Models retain their own applicable terms.

If no converted package exists, omit `--coreml`, then run:

```bash
./setup_coreml_env.sh --conversion
./compile_coreml.sh
```

Conversion uses only this repository and installed Python packages; it does not
call sunnypilot or tinygrad. It can take significant time and memory. A matching
metadata file is required; this project does not generate it from an arbitrary ONNX.
Converting successfully is not an accuracy qualification.

Expected local layout:

```text
mac-accelerator/
  .coreml-venv/bin/python
  models/big_driving_supercombo.onnx
  models/big_driving_supercombo_metadata.pkl
  artifacts/big_driving.mlpackage/
  run_coreml_server.sh
```

## Build and run the app

With Apple Command Line Tools installed:

```bash
./build_macos_app.sh
open "dist/Mac Accelerator.app"
```

Select this project folder. **Mac-only test** is on by default. Start launches the
server; Ready / Client connected describe service state, not driving readiness.
The app uses its own identifier `dev.akarow.mac-accelerator`, preferences and
`~/Library/Application Support/Mac Accelerator/auth.key` (directory 0700, key 0600).
It does not reuse or delete the old app's settings/key. Existing clients must use
the new key. Quit the old server first to avoid a port 8066 conflict.

With the localhost server running, a synthetic test from another terminal is:

```bash
.coreml-venv/bin/python synthetic_client.py ::1 \
  --auth-key-file "$HOME/Library/Application Support/Mac Accelerator/auth.key" \
  --expected-model-sha256 <SHA256-of-your-source-ONNX> --frames 100
```

Use `shasum -a 256 models/big_driving_supercombo.onnx` for the SHA. This is a
50ms fail-closed diagnostic; it may fail timing and does not test driving accuracy.
For Mac-only inference timing, stop the app server first and run:

```bash
DURATION=30 ./run_coreml_qualification.sh
```

That benchmark excludes USB/camera capture. Longer runs heat a fanless MacBook Air.

Optional USB mode uses ADB and a powered, off-road comma 3X. Unchecking Mac-only
explicitly enables transient USB-NCM interface setup and scoped-link supervision.
It does not deploy code or set device parameters. No device work is required for
normal Mac-only operation. The legacy wire identity is retained for protocol
compatibility; app/source ownership is independent of sunnypilot.

## Verification and research status

### Preloaded camera preprocessing (bench only)

An independent owned-NV12 preprocessing path can now be included in replay.
It does not receive live 3X camera buffers. Optional tinygrad is used to build
Metal-reference lookup maps at setup and compare CPU versus GPU per-frame work:

```bash
uv pip install --python .coreml-venv/bin/python -r requirements-preprocess.txt
./build_native_preprocess.sh  # optional bounded CPU gather; requires Apple clang
.coreml-venv/bin/python profile_preprocess.py --include-metal --runs 120 \
  --output artifacts/preprocess-this-mac.json
.coreml-venv/bin/python smoke_local.py --nv12 --frames 600 \
  --preprocess-profile artifacts/preprocess-this-mac.json \
  --shadow-log artifacts/shadow-nv12-001.jsonl
```

Choose fresh output paths. Profiles from another chip/OS or code version are
rejected; a failed timing/byte-comparison candidate is not selected. A NumPy-only
coordinate map is **not accepted** as an upstream-equivalent profile because
rounding-boundary differences were found. The latest M2 selection was native CPU gather
using precomputed tinygrad/Metal lookup maps, with Core ML CPU + NE inference.
Calibration changes stop a warmed observer instead of compiling in the frame loop.
The optional native library is built locally, checks buffer/index bounds, and is
bound to its source hash. Rebuild and reprofile after preprocessing changes. This
is a replay capability, not a new live-camera mode in the packaged Mac app.

See [Chestnut analysis, measurements, ownership constraints and M1 feasibility](docs/CAMERA_PIPELINE.md).
This adds no firmware changes or camera consumer, and does not resolve the earlier
unknown physical reboot. Python/runtime/model preparation is still required.

### Observation-only shadow replay

This is Mac-side replay, **not live 3X camera capture**. The easiest synthetic run
starts and stops its own localhost server and retains a private log:

```bash
.coreml-venv/bin/python smoke_local.py --frames 120 \
  --shadow-log artifacts/shadow-test-001.jsonl
```

Use a new log filename on every run. Disconnect injection (expected failure):

```bash
.coreml-venv/bin/python smoke_local.py --frames 120 \
  --shadow-log artifacts/shadow-disconnect-001.jsonl --disconnect-after 1.5
```

For real recorded inputs, run the app's localhost server and use:

```bash
.coreml-venv/bin/python shadow_replay.py \
  --warps /path/warps.npy --policy /path/policy.npy --frames 120 \
  --auth-key-file "$HOME/Library/Application Support/Mac Accelerator/auth.key" \
  --expected-model-sha256 <SHA256-of-your-source-ONNX> \
  --log artifacts/shadow-recorded-001.jsonl
```

Warps must be uint8 `[frames,2,6,128,256]`; policy must be matching float32
`[frames,12]` in protocol order. Pickled NumPy arrays are rejected. The recording
must contain every requested frame; it is never looped. No raw-video conversion
or captured policy fabrication is done by this tool. This run measures local
scheduled-playback-to-output age, not original camera EOF latency.

After 66 contiguous frames, per-frame timing misses against 50ms are counted.
150ms is a stale-stop limit, **not** a passing real-time budget. All results remain
observation-only. See [boundaries and the live capture gate](docs/SHADOW_DESIGN.md).

```bash
.coreml-venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

After model setup, `.coreml-venv/bin/python smoke_local.py` starts a separate
localhost server on a temporary port, runs 20 synthetic frames and stops its own
server. It uses a 500ms diagnostic deadline, **not** 50ms timing qualification.
No USB connection is needed. See the [0.3.0 verification record](docs/RELEASE_0.3.0.md).

Original M2 measurements: Mac-only inference mean 25.69ms / max 39.53ms over 6,000
paced runs; actual camera-to-transport integration did not qualify. Those are
historical results, not a fresh performance claim for this extraction.
See [historical progress](docs/PROJECT_STATUS.md), [stability review](docs/REVIEW_2026-09-16.md),
[protocol](docs/PROTOCOL.md) and [aggregate data](results/m2-coreml-5min.json).

This extraction changes packaging, paths, identity and dependency boundaries;
it does not solve the outstanding camera-copy/latency problem. Future work stays
bench-only until isolation, recurrent accuracy and sustained timing are demonstrated.

MIT code license; [source attribution and third-party notices](NOTICE.md).
Independent experiment; not an official Apple, comma.ai or sunnypilot product.
