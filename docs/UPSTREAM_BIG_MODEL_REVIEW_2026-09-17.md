# Official Big Model / Chestnut review — 2026-09-17

Read-only upstream review. No comma device, system network, firmware or app
installation was changed. Experimental acceleration remains observation-only.

## Which “Big Model”?

- The [2022 openpilot 0.8.14 article](https://blog.comma.ai/0814release/) calls its
  then-new dual-camera network “Big Model.” It is not the 2026 Chestnut model. Its
  50ms discussion covers the 20Hz GPU-service loop, not a published Chestnut
  camera-to-result latency distribution or permission to use 55ms control results.
- The [2026 Chestnut introduction](https://blog.comma.ai/chestnut/) describes a
  roughly 1B-parameter model and a comma four external-GPU product. The
  [0.11.2 release notes](https://github.com/commaai/openpilot/blob/6080cc6168023229437b0ff06cf35bead00a5d0d/RELEASES.md)
  specify 880M parameters. Neither establishes support for a Mac/3X combination.
- The [models README](https://github.com/commaai/openpilot/blob/6080cc6168023229437b0ff06cf35bead00a5d0d/openpilot/selfdrive/modeld/models/README.md)
  points to Netron for examining ONNX architecture. It is not a complete timing,
  recurrent-state, model conversion or accelerator integration specification.

Moving `master` web caches differed from GitHub API content during this review.
Implementation statements below use official commit
`6080cc6168023229437b0ff06cf35bead00a5d0d`, read through GitHub's contents API.

## Execution and memory path

In [official modeld.py](https://github.com/commaai/openpilot/blob/6080cc6168023229437b0ff06cf35bead00a5d0d/openpilot/selfdrive/modeld/modeld.py),
`ModelState.pack_inputs` builds one host allocation containing transforms, scalar
inputs and two padded raw NV12 frames. `run` copies camera bytes into it, uploads
the combined allocation, then runs warp and inference on the model device.
Recurrent outputs alias persistent device-state buffers; final prediction is read
back for parsing. It is therefore not “no copies anywhere.” It avoids repeatedly
round-tripping recurrent state through Python.

Our former 3X experiment instead transmitted two already-warped `[6,128,256]`
images. Equal physical connector type does not make these data paths equivalent.
Raw NV12 on the external GPU is an upstream design choice, not proof that raw
NV12 over the Mac's TCP/NCM path will meet the same timing.

The [official build](https://github.com/commaai/openpilot/blob/6080cc6168023229437b0ff06cf35bead00a5d0d/openpilot/selfdrive/modeld/SConscript)
uses tinygrad's USB+AMD path for Chestnut warp, not an IP inference service. It
also explicitly sets `DEV=METAL JIT=2` for Darwin with a warning about graph
batching when buffers change. That warning is relevant to future tinygrad work;
it does not by itself prove this project's separately tested gather kernel fails.

Useful design lessons: preallocate, preserve state on-device, batch transfers,
avoid blocking local modeld on readback, test changing buffers and preserve a
known-good local path. There is no Apple GPU/ANE frame-pipeline recipe or measured
Mac latency guarantee in the official materials reviewed.

## Local artifact identity caveat

The current local source is `models/big_driving_supercombo.onnx`, 765,950,064 bytes,
SHA-256 `1791d5940b2c048d0639813426dd2cf1d6f2a6727ed51e17c8bcea8bbe754123`.
Its metadata checkpoint is
`b9facbcc-4d47-410e-b3ce-dfcbad12ba92/56320/23e6a04e-e6e5-462b-a0bb-e4088275ee43/12864`.
The inspected current official tree distributes a compiled Big Model artifact.
This review has **not** established that the local ONNX/Core ML package is that
release's identical checkpoint. File size is not a parameter-count proof, and the
server's source hash is not a cryptographic conversion-provenance chain.

## Temporal contract correction

Pinned [sunnypilot compile_modeld.py](https://github.com/sunnypilot/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/sunnypilot/modeld_v2/compile_modeld.py)
derives stride 4 for a `features_buffer` with 32 context steps. The Mac server had
defaulted to stride 2. It now defaults to 4; the shadow identity check requires 4,
and the conservative full-context count changes from 66 to 132 frames. The
validation runner uses the same constant. Queue indexing and rejection of old
stride-2 shadow identities have regression coverage.

Previous stride-2 replay logs are preserved as historical diagnostics, **not**
upstream temporal-correctness validation. The earlier raw graph benchmarks update
feature history each prediction; their durations still describe those executed
calls, but not a faithful 20Hz/5Hz streaming state contract. New frame-pipeline
experiments use the corrected session and at least 132 warmup frames. This is a
meaningful correction, not proof of complete numerical/driving equivalence.

## IPv4 and a Mac DHCP server

Feasible as a connection-management option, not an inference optimization.
[DHCP](https://www.rfc-editor.org/info/rfc2131/) allocates addresses and configuration;
it does not handle every inference packet. The current listener is IPv6-specific,
and the USB supervisor validates scoped IPv6 link-local addresses, so changing
only the configured address would not implement IPv4 support.

Proposed order, not activated:

1. On a detached bench, choose a collision-free isolated USB IPv4 subnet and test
   static endpoints against IPv6 using identical payloads and percentile metrics.
2. Add IPv4 listener and supervisor support while retaining exact USB-interface
   identity checks and refusing Wi-Fi/LAN fallback.
3. If worthwhile, enable an explicit interface-only DHCP server and a matching
   3X client, with a stable lease and reconnect tests. Avoid gateway/DNS changes,
   forwarding or Internet Sharing unless internet access is separately requested.

Do not start a DHCP server on all interfaces or use global Internet Sharing just
to exchange inference data. No DHCP service or system network setting was changed.
