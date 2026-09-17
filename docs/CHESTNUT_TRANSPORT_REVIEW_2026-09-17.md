# Chestnut transport techniques and Mac-side changes

## Verified upstream mechanism

Official openpilot commit `6080cc6168023229437b0ff06cf35bead00a5d0d` pins
commaai/tinygrad commit `d5e17c935daf11f6318e45aade9528f71b8fbdcc`. The latter is
newer/different than this project's installed tinygrad. Sources were inspected
read-only; no dependency or firmware upgrade was performed.

[The pinned USB implementation](https://github.com/commaai/tinygrad/blob/d5e17c935daf11f6318e45aade9528f71b8fbdcc/tinygrad/runtime/support/usb.py)
has several concrete optimizations:

- Two 256KiB SRAM halves, with payload space reserved around sentinel markers.
  Host USB uploads and GPU copies can proceed through alternating halves.
- Host transfer buffers are reused only after their USB transfer completes;
  bridge SRAM is reused only after GPU completion fencing. USB completion alone
  does not prove the GPU has consumed the data.
- Consecutive copies in the same direction are grouped, chunked and emitted as
  compiled host/GPU operations. Per-half bulk transfers are submitted
  asynchronously. Readback is armed before allowing the GPU to fill its buffer.
- Unchanged kernel argument values can avoid repeated control writes. Contiguous
  operations can become streaming transfers instead of many small transactions.
- There are still staging `memcpy` operations, fence waits and USB transactions.
  The design is not universal zero-copy and does not remove completion checks.

[Official modeld](https://github.com/commaai/openpilot/blob/6080cc6168023229437b0ff06cf35bead00a5d0d/openpilot/selfdrive/modeld/modeld.py)
separately packs camera frames and small inputs into one upload and keeps recurrent
state on the device. This architecture, not merely a faster cable, reduces
round trips and per-transfer overhead. Its GPU/USB firmware interfaces cannot be
dropped into a Mac TCP/NCM receiver unchanged.

## What already existed here

- Long-lived TCP connection, `TCP_NODELAY`, one authenticated frame request.
- Scatter/gather `sendmsg` for header/image/policy/tag without concatenating the
  large uncompressed camera request. CRC is calculated across the existing views.
- `recv_into` while receiving a complete frame, followed by an owned immutable
  result. Client outputs do not alias storage reused by a later frame.
- Mac-side recurrent state; the request contains current warped images and small
  policy values, not the entire recurrent feature history.
- Absolute socket deadlines and latched failures; these remain enabled.

Consequently, “add batching/TCP_NODELAY” is not a new fix. The receive/HMAC path did
still rebuild large byte strings unnecessarily.

## Implemented changes

1. Authentication verification hashes a view of the received payload incrementally.
   It no longer rebuilds a full payload-plus-tag merely to take its last 32 bytes.
   It returns one owned immutable `bytes` object; no reused-buffer lifetime change.
   Contiguous signing also avoids a prefix-plus-image temporary. Wire format,
   authentication, CRC, frame identity and deadline checks are unchanged.
2. Shadow logs retain client preparation, send, receive-wait and validation times.
   They also retain server receive-to-inference preparation duration and the
   RTT-minus-inference residual. All are durations within one host's clock.
   **Receive-wait includes inference. The residual is not pure network time.**
3. `tools/profile_wire_overhead.py` isolates authenticated TCP loopback with the
   same 393,296-byte request payload and 36,960-byte response payload (including
   HMAC, excluding framing header). Its `TRANSPORT_PROBE` identity cannot be
   mistaken for the normal Core ML backend. It loads no model and contacts no 3X.

## Measurements on this Mac

Report: `artifacts/wire-overhead-20260917-01.json`. Uncompressed protocol-sized random
payloads; ephemeral random authentication key. Four 500-call verification blocks
in old/new/new/old order, preceded by warmup, then 500 completed loopback requests
per address family after 50 warmups. IPv4 and IPv6 were sequential, not a controlled
network superiority comparison.

| Measurement | Mean ms | p99 ms | Max ms |
| --- | ---: | ---: | ---: |
| Legacy HMAC verification, first block | 0.1666 | 0.2005 | 0.2115 |
| Current HMAC verification, first block | 0.1552 | 0.1862 | 0.2051 |
| Current HMAC verification, second block | 0.1551 | 0.1838 | 0.2283 |
| Legacy HMAC verification, second block | 0.1676 | 0.2048 | 0.2338 |
| IPv4 complete loopback request/result | 0.7384 | 0.8521 | 1.1463 |
| IPv6 complete loopback request/result | 0.7844 | 0.9721 | 1.2818 |

The copy optimization saves about 0.012ms on this Mac. It is worthwhile cleanup,
**not a fix for tens of milliseconds of historical live latency**. The sub-ms
loopback result includes software authentication/framing/client conversion but
excludes physical USB, NCM, device CPU contention and camera/GPU readback. It
cannot establish an actual 3X link RTT or a DHCP/IPv4 performance benefit.

## Highest-value next device measurement

The old live result (~73.54ms camera EOF to response, ~67.46ms copy start to
response) and Mac-only ~20–30ms inference were collected under different
conditions. Their difference is not an exact measurement of network time.
Historical QCOM readback stalls (roughly 10–17ms) are a known component, not
evidence that every missing millisecond is a cable problem.

On a detached bench, distinguish: camera readiness/readback, input preparation,
authenticated request upload, Mac preparation/inference, result download and
client validation. Establish the negotiated USB mode and model-free physical
link RTT first. Do not reinsert synchronous readback into the local modeld just
to obtain a timestamp. A separate bounded capture path needs its own ownership
and interference checks.

The current synthetic NV12 layout is 7,471,104 bytes for a camera pair versus
393,216 warped-image bytes, approximately 19 times larger. Raw-host capture may
avoid a GPU readback but increase link time; neither transport shape should be
selected without measuring the full path. Cable labeling alone does not establish
negotiated speed or payload throughput.

No 3X changes, DHCP activation, control outputs or real-drive qualification.

Verification after the transport change: 78 unit tests and Ruff passed. Legacy
HMAC byte compatibility, metadata tampering and truncated tags have regression
coverage. An authenticated 200-frame Core ML localhost replay completed with 69
full-context samples and zero 50ms misses in that window; the new per-stage fields
were recorded in `artifacts/shadow-transport-stages-20260917-01.jsonl`. This is a
short synthetic local check, not physical-link or sustained qualification.
