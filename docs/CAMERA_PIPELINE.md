# Camera preprocessing and Chestnut findings

## What was actually implemented

This repository can now turn an **independently owned NV12 camera pair** into the
393,216-byte model input, then feed the observation-only replay client. This work
adds no live VisionIPC consumer, raw-camera network protocol, device installer or
modeld hook. The 3X was disconnected during testing.

The warm-start path is:

1. Validate frame layout, calibration transforms and selected hardware profile.
2. Use tinygrad/Metal once to calculate the actual upstream-style source-index maps.
3. Keep those two maps and the output buffer allocated. Exercise the selected
   backend before accepting any frames.
4. Per frame, gather directly from owned NV12 bytes into six model planes, without
   full-frame U/V deinterleaving or a QCOM GPU readback. The chosen M2 implementation
   performs this gather on CPU. The result is copied into immutable bytes to avoid
   reuse/ownership bugs.
5. Send the small prepared request to the persistent Core ML CPU + Neural Engine
   worker. Maintain single-request ordering and the existing stale/failure latch.

Calibration changes after warmup stop the observer and require an explicit fresh
plan/run. They do not unexpectedly trigger GPU map compilation in the live loop.
The profile and all warmup costs are outside measured replay time. Model resources
already remain loaded for the life of the server; future images cannot be precomputed.

## Why the first CPU map was rejected

Straight NumPy FP32 coordinate calculation was fast but did **not** exactly match
upstream Metal near nearest-neighbor rounding boundaries. On four locally decoded
real-image frames, three test transforms produced respectively 0, 10 and 19 differing
bytes (maximum channel-value error 4). That implementation is not accepted as an
upstream-equivalent runtime profile.

The revised map computes source indices using tinygrad's own FP32 warp operations
at setup, then reuses those indices in a CPU gather. Repeating the same comparisons
gave **zero differing bytes for all 12 frame/transform comparisons**. This is a
limited Metal-reference test, not proof of QCOM equivalence, live calibration,
cross-chip pixel equivalence or driving-model accuracy.

Development comparison tool: `tools/compare_upstream_preprocess.py`. It optionally
uses the original checkout and local video solely as a validation reference. The
runtime does not import openpilot/sunnypilot. The tinygrad package is an optional
pinned dependency for map construction and GPU comparison.

## CPU versus GPU: measure completed work

One 200-sample M2 run, using the revised reference maps:

| Two-camera preprocessing | Mean | p99 | Maximum |
| --- | ---: | ---: | ---: |
| Cached map + CPU gather | 0.873ms | 1.207ms | 1.214ms |
| tinygrad Metal gather, including upload/readback | 5.860ms | 10.151ms | 10.367ms |

A later 120-sample profile before the native backend measured CPU mean 1.086ms / max
2.630ms and Metal mean 4.917ms / max 8.592ms. Both selected CPU. Variation is why
one best mean is not a hard bound. That reference-map build cost
about 598ms, moved to setup rather than repeated for each frame.

The new optional native backend performs only the integer gather, with no changed
sampling or color math. It retains immutable source ownership, validates maps and
output arrays, and checks each source offset in C. It is built with Apple clang;
the embedded source hash detects stale builds (not a security attestation). The
profile also binds the binary hash. It is not bundled into the published app yet.

The subsequent 200-sample same-session comparison measured:

| Two-camera preprocessing | Mean | p99 | Maximum |
| --- | ---: | ---: | ---: |
| NumPy cached gather | 0.837ms | 1.149ms | 1.167ms |
| Native CPU gather | 0.196ms | 0.268ms | 0.305ms |
| tinygrad Metal upload/gather/readback | 2.546ms | 3.267ms | 3.560ms |

All measured outputs matched the same Metal-reference lookup-map result exactly.
The selected backend was native CPU; setup cost was 588ms. These hot-loop results
exclude frame acquisition, scheduling, input transport and model inference. Do not
substitute them for the complete 20Hz replay measurements below.

Input is 7,471,104 bytes per padded pair; output is 393,216 bytes. GPU math can be
fast while uploading the full pair and synchronously reading a small result costs
more overall. The model itself remains on CPU + Neural Engine, not CPU alone.
The CPU gather measurement is on M2, **not the Snapdragon CPU in the 3X**.

## Complete Mac-local NV12 replay, including unsuccessful runs

Each run used 600 frames at 20Hz (30 seconds), owned synthetic NV12 pairs, a
localhost TCP server and Core ML CPU + Neural Engine Big Model inference. The
clock begins at scheduled local playback, not a real camera exposure timestamp.
There are 535 context-ready samples after building temporal history. No real USB,
camera acquisition or vehicle model runs concurrently in these measurements.

| Run | Whole-run mean | p99 | Maximum | Context-ready >50ms |
| --- | ---: | ---: | ---: | ---: |
| Initial cached CPU | 32.694ms | not summarized | 61.347ms | 2 / 535 |
| Cached CPU + source QoS/GC changes | 35.740ms | 39.561ms | 54.370ms | 1 / 535 |
| Native gather + source QoS/GC changes | 35.492ms | 40.041ms | 42.134ms | 0 / 535 |

The final run also had zero misses across all 600 frames. Its preprocessing mean
was 3.439ms / p99 5.160ms / max 7.763ms, and inference mean 26.718ms / max
30.177ms. Wake lateness averaged 2.791ms and reached 5.031ms. The real paced
preprocessing cost is materially higher than the hot-loop microbenchmark; do not
claim the complete pipeline gets a 0.196ms preprocessing bound.

These were sequential diagnostic runs, not controlled causal A/B experiments.
One short passing run does not prove sustained 50ms latency or crash resolution.
`roadReady` and `sustainedQualified` remain false. All unsuccessful results are
retained locally, not discarded in favor of the passing sample.

Native output also matched upstream Metal exactly on four decoded real-image
frames × three test transforms (12 comparisons, zero differing bytes). The tests
are not route-calibrated output/model validation and do not establish QCOM parity.
The optional native tests cover malformed buffers, index bounds, stale source
builds, warmup and immutable output ownership; the full suite has 70 passing tests.
A separate native-NV12 run terminated only its test-owned server after eight
seconds: the observer recorded peer closure, latched failure and exited without
automatic restart. The harness confirmed that the test server had stopped.

## What Chestnut actually teaches

The same connector is not the same transport architecture. Reviewed sources:

- [tinygrad USB support at f6fc4e3](https://github.com/sunnypilot/tinygrad/blob/f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae/tinygrad/runtime/support/usb.py)
  uses asynchronous USB bulk/control requests and an ASM24 USB–PCIe bridge; GPU
  memory/command transfer uses dedicated buffers and completion signals.
- [ASM2464PD DMA documentation](https://github.com/tinygrad/asm2464pd-firmware/blob/master/app/README.md)
  describes USB bulk ↔ bridge SRAM ↔ GPU DMA. This is not a Mac TCP/Core ML server.
- [sunnypilot modeld](https://github.com/sunnypilot/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/sunnypilot/modeld_v2/modeld.py)
  selects AMD for Chestnut, retains GPU-side execution/queues and reads the final
  output. It supports combined and separated warp/policy artifacts; do not assume
  every artifact has the same copy path.
- [upstream preprocessing](https://github.com/sunnypilot/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/selfdrive/modeld/compile_modeld.py)
  defines the NV12 layout, projective sampling and six-plane order used as reference.

Reusable principles are warm persistent resources, batching metadata/input work,
bounded owned buffers, explicit completion and no allocation/compilation surprises
in the per-frame path. A Mac cannot inherit the bridge's GPU DMA interface merely
by using the same USB-C port; this project does not claim zero-copy across USB.

## Why a live receiver is still gated

The locally reviewed VisionIPC client imports shared buffers and returns a buffer
after synchronization; the server reuses a ring of buffers and publishes indices.
These calls do not provide this new observer with an indefinitely held immutable
frame. Retaining a NumPy view or pointer while waiting on the Mac can therefore
outlive the buffer contents. The API here rejects borrowed views and requires
owned bytes. That type check does not solve acquisition-time races: a real adapter
still needs a proven coherent-copy/lifetime strategy.

Candidate deployment to test on a **disconnected, powered bench device** is a
separate normal-priority capture/prepare worker, never a callback in modeld. If 3X
CPU gathering is fast enough, it can retain the small USB payload. If not, raw-frame
transfer and Mac preprocessing need a separately bounded protocol; the current
protocol's 4MiB message cap cannot carry an entire padded pair in one request.
Neither option has been deployed or measured on 3X in this work.

## Priorities, memory and other M chips

The inference server and shadow source thread request macOS user-interactive QoS.
This is not FIFO/realtime scheduling or a global system-priority change. Cyclic GC
is postponed in the bounded replay loop, then restored on exit. Source wake lateness,
preprocessing and inference are recorded separately. No UI status or priority
request is a hard latency guarantee; OS background work and thermals still exist.

Profiles bind to chip, RAM, macOS, Python, NumPy, tinygrad revision/settings and
preprocessing code hash. They check completed outputs and maximum/p99 latency,
not GPU naming or advertised performance. A profile from another chip/OS is refused.
This selects the preprocessing backend only; it does not certify or automatically
change Core ML compute units or model precision. Tinygrad is not assumed faster.

**M1 feasibility only:** M1 has a Neural Engine and is a plausible Core ML target.
Apple lists 11 trillion operations/s for M1, while the M2 announcement describes a
40% faster Neural Engine. Those figures cannot be scaled into this model's actual
latency: partitioning, memory traffic, CPU fallback, OS and heat matter. M2 compute
headroom makes M1 worth testing, but stable 20Hz including transport is unproven.
No M1, M1 Pro/Max or higher-M-series timing/compatibility claim is made here.

Sources: [Apple M1](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/),
[Apple M2](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/),
[Apple ANE deployment guidance](https://machinelearning.apple.com/research/neural-engine-transformers).
